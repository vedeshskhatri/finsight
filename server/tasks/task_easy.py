from __future__ import annotations

from server.graders.grader_easy import EarningsGrader
from server.models import TaskInfo


class EarningsExtractionTask:
    task_id = "easy"
    name = "Earnings Metric Extraction"
    difficulty = "easy"
    max_steps = 10
    description = "Extract 5 financial metrics from a narrative earnings summary."

    _ticker_rotation = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "META", "NVDA", "JPM"]

    def __init__(self) -> None:
        self.grader = EarningsGrader()

    def task_info(self) -> TaskInfo:
        return TaskInfo(
            task_id=self.task_id,
            name=self.name,
            difficulty=self.difficulty,
            description=self.description,
            max_steps=self.max_steps,
            observation_space="Narrative earnings text and optional context slices.",
            action_space="submit_answer with 5 metrics, request_data(financials|macro|next_prices), noop.",
        )

    def initialize(self, fetcher, session_id: str, episode_num: int = 0) -> dict:
        ticker = self._ticker_rotation[episode_num % len(self._ticker_rotation)]
        snapshot = fetcher.fetch_earnings_snapshot(ticker)

        payload = {
            "ticker": ticker,
            "earnings_narrative": snapshot["narrative"],
            "required_fields": [
                "eps",
                "revenue",
                "yoy_revenue_growth",
                "operating_margin",
                "guidance_eps",
            ],
        }

        return {
            "task_description": self.description,
            "data_payload": payload,
            "ground_truth": snapshot["ground_truth"],
            "data_source": snapshot["data_source"],
            "task_state": {"ticker": ticker},
        }

    def context(self) -> str:
        return (
            "You are evaluating an earnings release narrative. Extract numeric values exactly and "
            "submit structured JSON with decimal percentages (e.g., 0.298 not 29.8). "
            "Submit your answer as: {'action_type': 'submit_answer', 'payload': {'eps': 6.43, "
            "'revenue': 394300000000.0, 'yoy_revenue_growth': 0.021, 'operating_margin': 0.298, "
            "'guidance_eps': 7.12}, 'reasoning': 'I extracted EPS and other fields from the narrative.'}."
        )

    def request_data(self, fetcher, session_state: dict, data_type: str) -> dict:
        ticker = session_state.get("task_state", {}).get("ticker", "AAPL")
        if data_type == "financials":
            return fetcher.fetch_additional_financials(ticker)
        if data_type == "macro":
            return fetcher.fetch_macro_context()
        return {"message": "No additional data for requested type in easy task."}

    def grade(self, submitted: dict, ground_truth: dict, cumulative_before: float):
        return self.grader.grade(submitted=submitted, ground_truth=ground_truth, cumulative_before=cumulative_before)
