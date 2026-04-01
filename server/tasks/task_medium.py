from __future__ import annotations

from server.data.fetcher import LiveDataFetcher
from server.graders.grader_medium import AnomalyGrader
from server.models import TaskInfo


class AnomalyTriageTask:
    task_id = "medium"
    name = "Market Anomaly Triage"
    difficulty = "medium"
    max_steps = 10
    description = "Identify anomalous ticker in a 5-stock price stream and classify severity."

    def __init__(self) -> None:
        self.grader = AnomalyGrader()

    def task_info(self) -> TaskInfo:
        return TaskInfo(
            task_id=self.task_id,
            name=self.name,
            difficulty=self.difficulty,
            description=self.description,
            max_steps=self.max_steps,
            observation_space="30-point intraday streams with volatility and anomaly metrics.",
            action_space=(
                "submit_answer with anomalous ticker/classification/action/ranking, "
                "request_data(macro), noop."
            ),
        )

    def initialize(self, fetcher: LiveDataFetcher, session_id: str, episode_num: int = 0) -> dict:
        stream = fetcher.fetch_price_stream(LiveDataFetcher.TICKERS)

        payload = {
            "tickers_data": stream["tickers_data"],
            "instructions": (
                "Identify the highest-volatility ticker, classify if this is genuine anomaly "
                "(z_score > 2) or noise, propose action, and rank all tickers by volatility."
            ),
        }

        return {
            "task_description": self.description,
            "data_payload": payload,
            "ground_truth": stream["ground_truth"],
            "data_source": stream["data_source"],
            "task_state": {"tickers": list(stream["tickers_data"].keys())},
        }

    def context(self) -> str:
        return (
            "Evaluate intraday anomalies using provided metrics only. Submit your answer as: "
            "{'action_type': 'submit_answer', 'payload': {'anomalous_ticker': 'TSLA', "
            "'classification': 'genuine_anomaly', 'action_recommendation': 'Escalate', "
            "'volatility_rank': ['TSLA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL']}, "
            "'reasoning': 'TSLA has highest intraday volatility and z-score above 2.'}."
        )

    def request_data(self, fetcher: LiveDataFetcher, session_state: dict, data_type: str) -> dict:
        if data_type == "macro":
            return fetcher.fetch_macro_context()
        return {"message": "Only macro context is available for medium task request_data."}

    def grade(self, submitted: dict, ground_truth: dict, cumulative_before: float):
        return self.grader.grade(submitted=submitted, ground_truth=ground_truth, cumulative_before=cumulative_before)
