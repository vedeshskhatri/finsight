from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.graders.grader_easy import EarningsGrader
from server.graders.grader_hard import PortfolioGrader
from server.graders.grader_medium import AnomalyGrader


def test_easy_all_fields_correct_close_to_one():
    grader = EarningsGrader()
    truth = {
        "eps": 6.43,
        "revenue": 394_300_000_000.0,
        "yoy_revenue_growth": 0.021,
        "operating_margin": 0.298,
        "guidance_eps": 7.12,
    }
    submitted = dict(truth)
    reward = grader.grade(submitted, truth)
    assert reward.score == pytest.approx(1.0, abs=1e-6)
    assert reward.partial_scores["field_presence"] == pytest.approx(0.30, abs=1e-6)
    assert reward.partial_scores["format_validity"] == pytest.approx(0.20, abs=1e-6)


def test_easy_all_fields_missing_zero_score():
    grader = EarningsGrader()
    truth = {
        "eps": 4.0,
        "revenue": 100.0,
        "yoy_revenue_growth": 0.1,
        "operating_margin": 0.2,
        "guidance_eps": 5.0,
    }
    reward = grader.grade({}, truth)
    assert reward.score == pytest.approx(0.0, abs=1e-6)


def test_easy_partial_fields_between_zero_and_one():
    grader = EarningsGrader()
    truth = {
        "eps": 10.0,
        "revenue": 200.0,
        "yoy_revenue_growth": 0.10,
        "operating_margin": 0.15,
        "guidance_eps": 9.0,
    }
    submitted = {
        "eps": 10.1,
        "revenue": 210.0,
    }
    reward = grader.grade(submitted, truth)
    assert 0.0 < reward.score < 1.0
    assert reward.partial_scores["field_presence"] == pytest.approx(0.12, abs=1e-6)


def test_easy_wrong_types_penalized_format_validity():
    grader = EarningsGrader()
    truth = {
        "eps": 10.0,
        "revenue": 200.0,
        "yoy_revenue_growth": 0.10,
        "operating_margin": 0.15,
        "guidance_eps": 9.0,
    }
    submitted = {
        "eps": "10.0",
        "revenue": "200.0",
        "yoy_revenue_growth": "0.1",
        "operating_margin": "0.15",
        "guidance_eps": "9.0",
    }
    reward = grader.grade(submitted, truth)
    assert reward.partial_scores["format_validity"] == pytest.approx(0.10, abs=1e-6)
    assert reward.score < 0.5


def test_medium_correct_submission_near_one():
    grader = AnomalyGrader()
    truth = {
        "highest_volatility_ticker": "TSLA",
        "z_score": 2.3,
        "is_genuine_anomaly": True,
        "volatility_ranking": ["TSLA", "AAPL", "MSFT", "AMZN", "GOOGL"],
    }
    submitted = {
        "anomalous_ticker": "TSLA",
        "classification": "genuine_anomaly",
        "action_recommendation": "Escalate",
        "volatility_rank": ["TSLA", "AAPL", "MSFT", "AMZN", "GOOGL"],
    }
    reward = grader.grade(submitted, truth)
    assert reward.score == pytest.approx(1.0, abs=1e-6)


def test_hard_four_trades_trade_count_invalid():
    grader = PortfolioGrader()
    truth = {
        "post_shock_values": {
            "AAPL": 20_000.0,
            "MSFT": 20_000.0,
            "GOOGL": 20_000.0,
            "AMZN": 20_000.0,
            "TSLA": 20_000.0,
        },
        "post_shock_total": 100_000.0,
        "constraints": {"transaction_cost_rate": 0.001},
    }
    submitted = {
        "trades": [
            {"action": "BUY", "ticker": "AAPL", "amount_usd": 1000.0},
            {"action": "SELL", "ticker": "MSFT", "amount_usd": 1000.0},
            {"action": "BUY", "ticker": "GOOGL", "amount_usd": 1000.0},
            {"action": "SELL", "ticker": "AMZN", "amount_usd": 1000.0},
        ]
    }
    reward = grader.grade(submitted, truth)
    assert reward.partial_scores["trade_count_valid"] == pytest.approx(0.0, abs=1e-6)


def test_hard_single_violation_over_40pct_scores_half_component():
    grader = PortfolioGrader()
    truth = {
        "post_shock_values": {
            "AAPL": 35_000.0,
            "MSFT": 15_000.0,
            "GOOGL": 15_000.0,
            "AMZN": 15_000.0,
            "TSLA": 20_000.0,
        },
        "post_shock_total": 100_000.0,
        "constraints": {"transaction_cost_rate": 0.001},
    }
    submitted = {
        "trades": [
            {"action": "BUY", "ticker": "AAPL", "amount_usd": 10000.0},
        ]
    }
    reward = grader.grade(submitted, truth)
    assert reward.partial_scores["constraint_no_single_over_40pct"] == pytest.approx(0.125, abs=1e-6)
    assert 0.0 <= reward.score <= 1.0
