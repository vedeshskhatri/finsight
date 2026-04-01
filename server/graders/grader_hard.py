from __future__ import annotations

from server.models import FinancialReward


class PortfolioGrader:
    def simulate_trades(self, ground_truth: dict, trades: list[dict]) -> dict:
        post_shock_values = {
            t: float(v) for t, v in (ground_truth.get("post_shock_values") or {}).items()
        }
        portfolio_value = float(ground_truth.get("post_shock_total", sum(post_shock_values.values())))

        cash = 0.0
        cost_rate = float(
            (ground_truth.get("constraints") or {}).get("transaction_cost_rate", 0.001)
        )

        trace: list[str] = []
        for idx, trade in enumerate(trades):
            action = str(trade.get("action", "")).upper()
            ticker = trade.get("ticker")
            amount = float(trade.get("amount_usd", 0.0) or 0.0)
            if ticker not in post_shock_values or amount <= 0:
                trace.append(f"trade_{idx+1}: ignored invalid trade {trade}")
                continue

            if action == "BUY":
                post_shock_values[ticker] += amount
                cash -= amount * (1.0 + cost_rate)
                trace.append(
                    f"trade_{idx+1}: BUY {ticker} ${amount:.2f}, cash impact=-${amount * (1.0 + cost_rate):.2f}"
                )
            elif action == "SELL":
                sell_amount = min(amount, post_shock_values[ticker])
                post_shock_values[ticker] -= sell_amount
                cash += sell_amount
                post_shock_values[ticker] -= sell_amount * cost_rate
                trace.append(
                    f"trade_{idx+1}: SELL {ticker} ${sell_amount:.2f}, cost=${sell_amount * cost_rate:.2f}"
                )
            else:
                trace.append(f"trade_{idx+1}: ignored unknown action {action}")

        final_total = sum(post_shock_values.values()) + cash
        if final_total <= 0:
            final_total = 1.0

        final_weights = {t: float(v / final_total) for t, v in post_shock_values.items()}
        return {
            "final_weights": final_weights,
            "final_total": final_total,
            "cash": cash,
            "trace": trace,
        }

    def grade(self, submitted: dict, ground_truth: dict, cumulative_before: float = 0.0) -> FinancialReward:
        submitted = submitted or {}
        trades = submitted.get("trades", [])
        if not isinstance(trades, list):
            trades = []

        sim = self.simulate_trades(ground_truth, trades)
        final_weights = sim["final_weights"]
        tickers = list(final_weights.keys())

        trade_count_valid_base = 1.0 if len(trades) <= 3 else 0.0
        trade_count_valid = 0.10 * trade_count_valid_base

        over_40_violations = sum(1 for w in final_weights.values() if w > 0.40)
        if over_40_violations == 0:
            over_40_base = 1.0
        elif over_40_violations == 1:
            over_40_base = 0.5
        else:
            over_40_base = 0.0
        constraint_no_single_over_40pct = 0.25 * over_40_base

        floor_pass = sum(1 for w in final_weights.values() if w >= 0.05)
        floor_base = floor_pass / len(tickers) if tickers else 0.0
        constraint_min_floor_5pct = 0.15 * floor_base

        target = 0.20
        if tickers:
            mae = sum(abs(final_weights[t] - target) for t in tickers) / len(tickers)
        else:
            mae = 1.0
        target_base = max(0.0, 1.0 - (mae / 0.20))
        target_proximity = 0.35 * target_base

        total_traded_usd = sum(abs(float((tr or {}).get("amount_usd", 0.0) or 0.0)) for tr in trades)
        cost = total_traded_usd * 0.001
        max_expected_cost = 100_000 * 0.05
        cost_base = max(0.0, 1.0 - (cost / max_expected_cost))
        transaction_cost_efficiency = 0.15 * cost_base

        score = max(
            0.0,
            min(
                1.0,
                trade_count_valid
                + constraint_no_single_over_40pct
                + constraint_min_floor_5pct
                + target_proximity
                + transaction_cost_efficiency,
            ),
        )

        feedback = (
            f"Trade count={len(trades)}. Over-40% violations={over_40_violations}. "
            f"Floor passes={floor_pass}/{len(tickers)}. MAE to target={mae:.4f}. "
            f"Transaction cost=${cost:.2f}. Trace: {' | '.join(sim['trace'])}"
        )

        return FinancialReward(
            score=score,
            partial_scores={
                "trade_count_valid": trade_count_valid,
                "constraint_no_single_over_40pct": constraint_no_single_over_40pct,
                "constraint_min_floor_5pct": constraint_min_floor_5pct,
                "target_proximity": target_proximity,
                "transaction_cost_efficiency": transaction_cost_efficiency,
            },
            feedback=feedback,
            done=True,
            step_reward=score,
            cumulative_reward=cumulative_before + score,
        )
