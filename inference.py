from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any

import requests

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
MAX_STEPS = 10
TEMPERATURE = 0.1

SYSTEM_PROMPT = """You are a financial analyst AI agent operating in the FinSight
 evaluation environment. You will receive financial data and must take actions using
 this exact JSON format:
{"action_type": "submit_answer"|"request_data"|"noop", "payload": {}, "reasoning": "..."}

For submit_answer, fill payload with your structured answer.
For request_data, set payload to {"data_type": "next_prices"|"financials"|"macro"}.
For noop, set payload to {}.
Always include a reasoning field explaining your decision.
Respond with ONLY valid JSON. No markdown. No explanation outside the JSON."""


def _build_openai_client() -> tuple[Any | None, str | None]:
    if OpenAI is None:
        return None, "openai package import failed"

    if not HF_TOKEN:
        return None, "missing HF_TOKEN/API_KEY"

    try:
        return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN), None
    except Exception as exc:
        return None, f"OpenAI init error: {exc}"


def _safe_post(url: str, **kwargs):
    try:
        response = requests.post(url, timeout=30, **kwargs)
        return response.status_code, response.json() if response.content else {}
    except Exception as exc:
        return 500, {"error": str(exc)}


def run_task(task_id: str) -> dict:
    session_id = str(uuid.uuid4())
    client, client_error = _build_openai_client()
    if client_error:
        print(f"  [{task_id}] warning: {client_error}. Falling back to noop actions.")

    code, obs = _safe_post(
        f"{ENV_BASE_URL}/reset",
        json={"task_id": task_id, "session_id": session_id},
    )
    if code >= 400 or not isinstance(obs, dict):
        return {
            "task_id": task_id,
            "final_score": 0.0,
            "steps_used": 0,
            "data_source": "unknown",
            "error": f"reset failed: {obs}",
        }

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    final_score = 0.0
    steps_used = 0
    data_source = obs.get("data_source", "unknown")

    for step in range(MAX_STEPS):
        user_content = (
            f"Task: {obs.get('task_description', '')}\n"
            f"Step: {obs.get('step_num', 0)} / {obs.get('max_steps', MAX_STEPS)}\n"
            f"Data: {json.dumps(obs.get('data_payload', {}), indent=2)}\n"
            f"Instructions: {obs.get('context', '')}\n"
        )
        messages.append({"role": "user", "content": user_content})

        if client is None:
            action = {
                "action_type": "noop",
                "payload": {},
                "reasoning": "fallback due to missing/unavailable model client",
            }
        else:
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=1000,
                )
                raw = (completion.choices[0].message.content or "").strip()
                action = json.loads(raw)
            except Exception as exc:
                print(f"  [Step {step+1}] LLM/parsing error: {exc}. Using noop.")
                action = {
                    "action_type": "noop",
                    "payload": {},
                    "reasoning": "fallback due to model/parsing failure",
                }

        messages.append({"role": "assistant", "content": json.dumps(action)})

        code, result = _safe_post(
            f"{ENV_BASE_URL}/step",
            json=action,
            headers={"X-Session-ID": session_id},
        )
        if code >= 400:
            print(f"  [Step {step+1}] step failed: {result}")
            break

        reward = result.get("reward", {})
        done = bool(result.get("done", False))
        obs = result.get("observation", obs)
        data_source = obs.get("data_source", data_source)
        steps_used = step + 1

        print(
            f"  [Step {steps_used}] action={action.get('action_type', 'unknown')} "
            f"step_reward={float(reward.get('step_reward', 0.0)):.3f} "
            f"cumulative={float(reward.get('cumulative_reward', 0.0)):.3f}"
        )

        if done:
            final_score = float(reward.get("score", 0.0))
            break

    return {
        "task_id": task_id,
        "final_score": final_score,
        "steps_used": steps_used,
        "data_source": data_source,
    }


def main() -> None:
    print("=== FinSight-Env Baseline Inference ===\n")
    results = []

    for task_id in ["easy", "medium", "hard"]:
        print(f"Running task: {task_id}")
        start = time.time()
        try:
            result = run_task(task_id)
        except Exception as exc:
            result = {
                "task_id": task_id,
                "final_score": 0.0,
                "steps_used": 0,
                "data_source": "unknown",
                "error": str(exc),
            }
        elapsed = time.time() - start
        result["time_seconds"] = round(elapsed, 1)
        results.append(result)
        print()

    total = sum(float(r.get("final_score", 0.0)) for r in results) / max(len(results), 1)

    try:
        _safe_post(
            f"{ENV_BASE_URL}/leaderboard/submit",
            json={
                "agent_name": MODEL_NAME,
                "scores": {r["task_id"]: float(r.get("final_score", 0.0)) for r in results},
                "total_score": total,
            },
        )
    except Exception as exc:
        print(f"Leaderboard submit failed: {exc}")

    print("=" * 65)
    print(f"{'Task':<10} {'Score':>8} {'Steps':>7} {'Time(s)':>9} {'Data Source':>16}")
    print("-" * 65)
    for r in results:
        print(
            f"{r['task_id']:<10} {float(r.get('final_score', 0.0)):>8.3f} "
            f"{int(r.get('steps_used', 0)):>7} {float(r.get('time_seconds', 0.0)):>9} "
            f"{str(r.get('data_source', 'unknown')):>16}"
        )
    print("-" * 65)
    print(f"{'AVERAGE':<10} {total:>8.3f}")
    print("=" * 65)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Fatal inference error handled: {exc}")
