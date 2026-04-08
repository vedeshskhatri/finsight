from __future__ import annotations

import json
import os
import uuid
from typing import Any

import requests

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
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


def _emit_log(event: str, payload: dict[str, Any]) -> None:
    def _format_value(value: Any) -> str:
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, bool):
            return "true" if value else "false"
        if value is None:
            return "null"
        if isinstance(value, str):
            # Quote string values to keep parsing unambiguous.
            return json.dumps(value, ensure_ascii=True)
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"))

    fields = " ".join(f"{k}={_format_value(v)}" for k, v in payload.items())
    if fields:
        print(f"[{event}] {fields}", flush=True)
    else:
        print(f"[{event}]", flush=True)


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
    _emit_log(
        "START",
        {
            "task_id": task_id,
            "session_id": session_id,
            "api_base_url": API_BASE_URL,
            "model_name": MODEL_NAME,
            "local_image_name": LOCAL_IMAGE_NAME,
        },
    )

    client, client_error = _build_openai_client()

    code, obs = _safe_post(
        f"{ENV_BASE_URL}/reset",
        json={"task_id": task_id, "session_id": session_id},
    )
    if code >= 400 or not isinstance(obs, dict):
        result = {
            "task_id": task_id,
            "final_score": 0.0,
            "steps_used": 0,
            "data_source": "unknown",
            "status": "reset_failed",
            "error": f"reset failed: {obs}",
        }
        _emit_log(
            "END",
            {
                "task_id": task_id,
                "session_id": session_id,
                "status": result["status"],
                "final_score": result["final_score"],
                "steps_used": result["steps_used"],
                "data_source": result["data_source"],
                "error": result["error"],
            },
        )
        return result

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    final_score = 0.0
    steps_used = 0
    data_source = obs.get("data_source", "unknown")
    status = "max_steps"
    error_message = client_error

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
                error_message = f"llm_or_parse_error: {exc}"
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
            status = "step_failed"
            error_message = f"step failed: {result}"
            break

        reward = result.get("reward", {})
        done = bool(result.get("done", False))
        obs = result.get("observation", obs)
        data_source = obs.get("data_source", data_source)
        steps_used = step + 1

        _emit_log(
            "STEP",
            {
                "task_id": task_id,
                "session_id": session_id,
                "step": steps_used,
                "action_type": action.get("action_type", "unknown"),
                "step_reward": round(float(reward.get("step_reward", 0.0)), 6),
                "cumulative_reward": round(float(reward.get("cumulative_reward", 0.0)), 6),
                "done": done,
            },
        )

        if done:
            final_score = float(reward.get("score", 0.0))
            status = "completed"
            break

    result = {
        "task_id": task_id,
        "final_score": final_score,
        "steps_used": steps_used,
        "data_source": data_source,
        "status": status,
    }
    if error_message:
        result["error"] = error_message

    _emit_log(
        "END",
        {
            "task_id": task_id,
            "session_id": session_id,
            "status": status,
            "final_score": final_score,
            "steps_used": steps_used,
            "data_source": data_source,
            "error": error_message,
        },
    )
    return result


def main() -> None:
    results = []

    for task_id in ["easy", "medium", "hard"]:
        try:
            result = run_task(task_id)
        except Exception as exc:
            result = {
                "task_id": task_id,
                "final_score": 0.0,
                "steps_used": 0,
                "data_source": "unknown",
                "status": "task_exception",
                "error": str(exc),
            }
            _emit_log(
                "END",
                {
                    "task_id": task_id,
                    "status": "task_exception",
                    "final_score": 0.0,
                    "steps_used": 0,
                    "data_source": "unknown",
                    "error": str(exc),
                },
            )
        results.append(result)

    total = sum(float(r.get("final_score", 0.0)) for r in results) / max(len(results), 1)

    submit_code, submit_response = _safe_post(
        f"{ENV_BASE_URL}/leaderboard/submit",
        json={
            "agent_name": MODEL_NAME,
            "scores": {r["task_id"]: float(r.get("final_score", 0.0)) for r in results},
            "total_score": total,
        },
    )
    _emit_log(
        "END",
        {
            "task_id": "all",
            "status": "summary",
            "average_score": total,
            "leaderboard_submit_code": submit_code,
            "leaderboard_submit_response": submit_response,
        },
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        _emit_log("END", {"task_id": "all", "status": "fatal", "error": str(exc)})
