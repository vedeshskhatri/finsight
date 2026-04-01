from __future__ import annotations

import re
import uuid
from typing import Any

import requests
import streamlit as st

BASE_URL = "http://localhost:7860"
TIMEOUT_SECONDS = 20


def api_get(path: str) -> tuple[bool, dict | list | str]:
    try:
        response = requests.get(f"{BASE_URL}{path}", timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        return True, response.json()
    except Exception as exc:
        return False, str(exc)


def api_post(path: str, payload: dict, headers: dict[str, str] | None = None) -> tuple[bool, dict | list | str]:
    try:
        response = requests.post(
            f"{BASE_URL}{path}",
            json=payload,
            headers=headers or {},
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return True, response.json()
    except Exception as exc:
        return False, str(exc)


def parse_easy_action(observation: dict[str, Any]) -> dict[str, Any]:
    narrative = observation.get("data_payload", {}).get("earnings_narrative", "")
    numbers = re.findall(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?", narrative)

    def to_float(index: int, default: float) -> float:
        if index >= len(numbers):
            return default
        return float(numbers[index].replace(",", ""))

    # Simple extraction baseline for the 5 required fields.
    eps = to_float(0, 0.0)
    revenue = to_float(1, 0.0)
    yoy_percent = to_float(2, 0.0)
    margin_percent = to_float(3, 0.0)
    guidance_eps = to_float(4, 0.0)

    payload = {
        "eps": eps,
        "revenue": revenue,
        "yoy_revenue_growth": yoy_percent / 100.0,
        "operating_margin": margin_percent / 100.0,
        "guidance_eps": guidance_eps,
    }

    return {
        "action_type": "submit_answer",
        "payload": payload,
        "reasoning": "Parsed visible numbers from the narrative as a baseline extraction strategy.",
    }


def parse_medium_action(observation: dict[str, Any]) -> dict[str, Any]:
    tickers_data = observation.get("data_payload", {}).get("tickers_data", {})
    if not tickers_data:
        return {
            "action_type": "submit_answer",
            "payload": {
                "anomalous_ticker": "AAPL",
                "classification": "noise",
                "action_recommendation": "Hold",
                "volatility_rank": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
            },
            "reasoning": "Fallback payload due to missing ticker metrics.",
        }

    ranking = sorted(
        tickers_data.keys(),
        key=lambda t: float(tickers_data[t].get("intraday_volatility", 0.0)),
        reverse=True,
    )
    top_ticker = ranking[0]
    z_score = float(tickers_data[top_ticker].get("z_score_latest", 0.0))
    is_genuine = z_score > 2.0

    classification = "genuine_anomaly" if is_genuine else "noise"
    action_rec = "Escalate" if is_genuine else "Hold"

    return {
        "action_type": "submit_answer",
        "payload": {
            "anomalous_ticker": top_ticker,
            "classification": classification,
            "action_recommendation": action_rec,
            "volatility_rank": ranking,
        },
        "reasoning": "Selected highest volatility ticker and classified by z_score threshold.",
    }


def parse_hard_action(observation: dict[str, Any]) -> dict[str, Any]:
    data_payload = observation.get("data_payload", {})
    portfolio = data_payload.get("portfolio", {})
    post_alloc = portfolio.get("post_shock_allocation", {})

    if not post_alloc:
        trades: list[dict[str, Any]] = []
    else:
        target = 0.20
        total_value = float(portfolio.get("portfolio_value", 100000.0))
        deviations = {t: float(w) - target for t, w in post_alloc.items()}
        over = sorted([t for t in deviations if deviations[t] > 0], key=lambda t: deviations[t], reverse=True)
        under = sorted([t for t in deviations if deviations[t] < 0], key=lambda t: deviations[t])

        trades = []
        for sell_ticker, buy_ticker in zip(over[:2], under[:2]):
            amount = min(deviations[sell_ticker], -deviations[buy_ticker]) * total_value
            amount = max(0.0, round(amount, 2))
            if amount > 0:
                trades.append({"action": "SELL", "ticker": sell_ticker, "amount_usd": amount})
                trades.append({"action": "BUY", "ticker": buy_ticker, "amount_usd": amount})

        trades = trades[:3]

    return {
        "action_type": "submit_answer",
        "payload": {"trades": trades},
        "reasoning": "Moved capital from overweight assets to underweight assets with <=3 trades.",
    }


def build_action(task_id: str, observation: dict[str, Any]) -> dict[str, Any]:
    if task_id == "easy":
        return parse_easy_action(observation)
    if task_id == "medium":
        return parse_medium_action(observation)
    return parse_hard_action(observation)


def run_task(task_id: str) -> dict[str, Any]:
    session_id = str(uuid.uuid4())
    ok_reset, reset_data = api_post("/reset", {"task_id": task_id, "session_id": session_id})
    if not ok_reset:
        return {"ok": False, "error": f"Reset failed: {reset_data}"}

    observation = reset_data if isinstance(reset_data, dict) else {}
    action = build_action(task_id, observation)

    ok_step, step_data = api_post(
        "/step",
        action,
        headers={"X-Session-ID": session_id},
    )
    if not ok_step:
        return {"ok": False, "error": f"Step failed: {step_data}", "observation": observation}

    return {
        "ok": True,
        "session_id": session_id,
        "reset_observation": observation,
        "step_result": step_data,
    }


def render_status() -> None:
    st.subheader("System Status")
    ok, health = api_get("/health")
    if not ok:
        st.error(f"Health check failed: {health}")
        return

    status = str((health or {}).get("status", "unknown"))
    if status == "ok":
        st.success("Backend status is healthy (ok)")
    else:
        st.error(f"Backend status is not healthy: {status}")

    st.metric("Backend Status", status)


def render_task_buttons() -> str | None:
    st.subheader("Run Task")
    c1, c2, c3 = st.columns(3)

    selected_task: str | None = None
    with c1:
        if st.button("Run Easy Task", use_container_width=True):
            selected_task = "easy"
    with c2:
        if st.button("Run Medium Task", use_container_width=True):
            selected_task = "medium"
    with c3:
        if st.button("Run Hard Task", use_container_width=True):
            selected_task = "hard"

    return selected_task


def render_task_output(result: dict[str, Any] | None) -> None:
    st.subheader("Task Output")
    if result is None:
        st.info("Click a task button to run a session.")
        return

    if not result.get("ok"):
        st.error(result.get("error", "Unknown task execution error."))
        if result.get("observation"):
            st.json(result["observation"])
        return

    reset_obs = result.get("reset_observation", {})
    step_result = result.get("step_result", {})
    reward = step_result.get("reward", {})
    observation_after = step_result.get("observation", {})

    st.metric("Session ID", result.get("session_id", "unknown"))
    st.metric("Final Score", f"{float(reward.get('score', 0.0)):.3f}")
    st.metric("Data Source", observation_after.get("data_source", reset_obs.get("data_source", "unknown")))

    st.write("Task Description")
    st.write(reset_obs.get("task_description", "N/A"))

    st.write("Observation Data")
    st.json(reset_obs.get("data_payload", {}))

    st.write("Feedback")
    st.write(reward.get("feedback", "No feedback returned."))

    st.write("Step Result")
    st.json(step_result)


def render_leaderboard() -> None:
    st.subheader("Leaderboard")
    ok, data = api_get("/leaderboard")
    if not ok:
        st.error(f"Failed to load leaderboard: {data}")
        return

    rows = []
    leaderboard = data if isinstance(data, list) else []
    for idx, entry in enumerate(leaderboard, start=1):
        scores = entry.get("scores", {}) if isinstance(entry, dict) else {}
        rows.append(
            {
                "Rank": idx,
                "Agent Name": entry.get("agent_name", "unknown"),
                "Easy Score": round(float(scores.get("easy", 0.0)), 3),
                "Medium Score": round(float(scores.get("medium", 0.0)), 3),
                "Hard Score": round(float(scores.get("hard", 0.0)), 3),
                "Total Score": round(float(entry.get("total_score", 0.0)), 3),
            }
        )

    if not rows:
        st.table([])
    else:
        st.table(rows)


def main() -> None:
    st.title("FinSight-Env Dashboard")
    render_status()

    task_choice = render_task_buttons()

    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None

    if task_choice:
        st.session_state["last_result"] = run_task(task_choice)

    render_task_output(st.session_state.get("last_result"))
    render_leaderboard()


if __name__ == "__main__":
    main()
