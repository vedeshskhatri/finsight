from __future__ import annotations

from typing import Any

from server.models import FinancialReward


class EarningsGrader:
    REQUIRED_FIELDS = [
        "eps",
        "revenue",
        "yoy_revenue_growth",
        "operating_margin",
        "guidance_eps",
    ]

    @staticmethod
    def _is_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and value is not None

    @staticmethod
    def _to_float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def grade(self, submitted: dict, ground_truth: dict, cumulative_before: float = 0.0) -> FinancialReward:
        submitted = submitted or {}
        present_fields = [
            field for field in self.REQUIRED_FIELDS if self._is_number(submitted.get(field))
        ]

        field_presence = (len(present_fields) / len(self.REQUIRED_FIELDS)) * 0.30

        per_field_scores: list[float] = []
        field_feedback: list[str] = []

        for field in self.REQUIRED_FIELDS:
            if field not in submitted or not self._is_number(submitted.get(field)):
                per_field_scores.append(0.0)
                field_feedback.append(f"{field}: missing or non-numeric.")
                continue

            submitted_val = float(submitted[field])
            truth_val = float(ground_truth.get(field, 0.0))

            if truth_val == 0:
                rel_error = abs(submitted_val - truth_val)
            else:
                rel_error = abs(submitted_val - truth_val) / abs(truth_val)

            if rel_error <= 0.02:
                score = 1.0
            elif rel_error <= 0.05:
                score = 0.7
            elif rel_error <= 0.10:
                score = 0.4
            elif rel_error <= 0.20:
                score = 0.1
            else:
                score = 0.0

            per_field_scores.append(score)
            field_feedback.append(
                f"{field}: submitted={submitted_val:.6g}, truth={truth_val:.6g}, relative_error={rel_error:.4f}."
            )

        numerical_accuracy = (sum(per_field_scores) / len(self.REQUIRED_FIELDS)) * 0.50

        values_are_numeric = all(self._is_number(submitted.get(f)) for f in self.REQUIRED_FIELDS)
        percentages_decimal = True
        for p in ["yoy_revenue_growth", "operating_margin"]:
            parsed = self._to_float(submitted.get(p))
            if parsed is None or abs(parsed) > 1.0:
                percentages_decimal = False
                break

        if values_are_numeric and percentages_decimal:
            format_validity = 0.20
        elif values_are_numeric or percentages_decimal:
            format_validity = 0.10
        else:
            format_validity = 0.0

        score = max(0.0, min(1.0, field_presence + numerical_accuracy + format_validity))
        feedback = (
            f"Field presence score={field_presence:.3f}; numerical accuracy score={numerical_accuracy:.3f}; "
            f"format validity score={format_validity:.3f}. "
            + " ".join(field_feedback)
        )

        return FinancialReward(
            score=score,
            partial_scores={
                "field_presence": field_presence,
                "numerical_accuracy": numerical_accuracy,
                "format_validity": format_validity,
            },
            feedback=feedback,
            done=True,
            step_reward=score,
            cumulative_reward=cumulative_before + score,
        )
