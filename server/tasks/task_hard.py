from __future__ import annotations

from server.data.fetcher import LiveDataFetcher
from server.graders.grader_hard import PortfolioGrader
from server.models import TaskInfo


class PortfolioRebalanceUnderShockTask:
    task_id = "hard"
    name = "Portfolio Rebalancing Under Market Shock"
    difficulty = "hard"
    max_steps = 10
    description = "Rebalance a shocked portfolio to equal weights in <=3 trades under constraints."

    def __init__(self) -> None:
        self.grader = PortfolioGrader()

    def task_info(self) -> TaskInfo:
        return TaskInfo(
            task_id=self.task_id,
            name=self.name,
            difficulty=self.difficulty,
            description=self.description,
            max_steps=self.max_steps,
            observation_space="Pre/post shock portfolio allocations, prices, and constraints.",
            action_space="submit_answer with trades list, request_data(next_prices|macro), noop.",
        )

    def initialize(self, fetcher: LiveDataFetcher, session_id: str, episode_num: int = 0) -> dict:
        snap = fetcher.fetch_portfolio_snapshot(LiveDataFetcher.TICKERS)
        gt = snap["ground_truth"]

        payload = {
            "portfolio": snap["portfolio"],
            "constraints": snap["constraints"],
            "shock_event": {
                "shocked_ticker": gt["shocked_ticker"],
                "shock_magnitude": gt["shock_magnitude"],
                "narrative": (
                    f"Market shock alert: {gt['shocked_ticker']} dropped by "
                    f"{gt['shock_magnitude'] * 100:.2f}% and distorted portfolio weights."
                ),
            },
        }

        return {
            "task_description": self.description,
            "data_payload": payload,
            "ground_truth": gt,
            "data_source": snap["data_source"],
            "task_state": {
                "tickers": list(snap["portfolio"]["post_shock_allocation"].keys()),
                "latest_prices": dict(snap["portfolio"]["shocked_prices"]),
            },
        }

    def context(self) -> str:
        return (
            "You must rebalance the POST-SHOCK portfolio to equal 20% weights using at most 3 trades, "
            "while satisfying max 40% per asset, min 5% floor, and 0.1% transaction cost per trade. "
            "Submit as: {'action_type': 'submit_answer', 'payload': {'trades': ["
            "{'action': 'SELL', 'ticker': 'AAPL', 'amount_usd': 4500.0}, "
            "{'action': 'BUY', 'ticker': 'TSLA', 'amount_usd': 4500.0}]}, "
            "'reasoning': 'Sell overweight names and buy shocked underweight name.'}."
        )

    def request_data(self, fetcher: LiveDataFetcher, session_state: dict, data_type: str) -> dict:
        tickers = session_state.get("task_state", {}).get("tickers", LiveDataFetcher.TICKERS)
        latest_prices = session_state.get("task_state", {}).get("latest_prices", {})

        if data_type == "next_prices":
            update = fetcher.fetch_next_prices(tickers=tickers, current_prices=latest_prices)
            session_state.setdefault("task_state", {})["latest_prices"] = dict(update["next_prices"])
            return update

        if data_type == "macro":
            return fetcher.fetch_macro_context()

        return {"message": "Only next_prices and macro are available for hard task request_data."}

    def grade(self, submitted: dict, ground_truth: dict, cumulative_before: float):
        return self.grader.grade(submitted=submitted, ground_truth=ground_truth, cumulative_before=cumulative_before)
