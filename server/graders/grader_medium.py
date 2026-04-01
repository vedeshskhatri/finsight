from __future__ import annotations

from server.models import FinancialReward


class AnomalyGrader:
    def _kendall_tau(self, submitted_rank: list[str], truth_rank: list[str]) -> float:
        n = len(truth_rank)
        if n <= 1:
            return 1.0

        pos_sub = {ticker: i for i, ticker in enumerate(submitted_rank)}
        pos_truth = {ticker: i for i, ticker in enumerate(truth_rank)}

        discordant = 0
        concordant = 0
        tickers = truth_rank
        for i in range(n):
            for j in range(i + 1, n):
                a = tickers[i]
                b = tickers[j]
                if a not in pos_sub or b not in pos_sub:
                    continue
                truth_order = pos_truth[a] - pos_truth[b]
                sub_order = pos_sub[a] - pos_sub[b]
                if truth_order * sub_order > 0:
                    concordant += 1
                elif truth_order * sub_order < 0:
                    discordant += 1

        total_pairs = n * (n - 1) / 2
        if total_pairs == 0:
            return 1.0
        return (concordant - discordant) / total_pairs

    def grade(self, submitted: dict, ground_truth: dict, cumulative_before: float = 0.0) -> FinancialReward:
        submitted = submitted or {}

        highest_ticker = ground_truth.get("highest_volatility_ticker")
        is_genuine = bool(ground_truth.get("is_genuine_anomaly", False))
        truth_ranking = ground_truth.get("volatility_ranking", [])

        ticker_correct = float(submitted.get("anomalous_ticker") == highest_ticker)
        correct_ticker = 0.35 * ticker_correct

        expected_class = "genuine_anomaly" if is_genuine else "noise"
        class_correct = float(submitted.get("classification") == expected_class)
        correct_classification = 0.30 * class_correct

        submitted_action = submitted.get("action_recommendation")
        if is_genuine:
            action_base = 1.0 if submitted_action == "Escalate" else 0.5 if submitted_action == "Investigate" else 0.0
        else:
            action_base = 1.0 if submitted_action == "Hold" else 0.5 if submitted_action == "Investigate" else 0.0
        correct_action = 0.15 * action_base

        submitted_rank = submitted.get("volatility_rank") or []
        if not isinstance(submitted_rank, list):
            submitted_rank = []
        tau = self._kendall_tau(submitted_rank, truth_ranking)
        volatility_rank_accuracy = 0.20 * ((tau + 1.0) / 2.0)

        score = max(
            0.0,
            min(1.0, correct_ticker + correct_classification + correct_action + volatility_rank_accuracy),
        )

        feedback = (
            f"Expected ticker={highest_ticker}, submitted={submitted.get('anomalous_ticker')}. "
            f"Expected classification={expected_class}, submitted={submitted.get('classification')}. "
            f"Action submitted={submitted_action}. Kendall tau={tau:.3f}."
        )

        return FinancialReward(
            score=score,
            partial_scores={
                "correct_ticker": correct_ticker,
                "correct_classification": correct_classification,
                "correct_action": correct_action,
                "volatility_rank_accuracy": volatility_rank_accuracy,
            },
            feedback=feedback,
            done=True,
            step_reward=score,
            cumulative_reward=cumulative_before + score,
        )
