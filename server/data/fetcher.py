from __future__ import annotations

import hashlib
import math
import time
from typing import Any

import numpy as np
import requests
import yfinance as yf


class LiveDataFetcher:
    TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

    def __init__(self) -> None:
        pass

    @property
    def _hour_seed(self) -> int:
        return int(time.time()) // 3600

    def _stable_ticker_seed(self, ticker: str, extra: int = 0) -> int:
        digest = hashlib.sha256(ticker.encode("utf-8")).hexdigest()
        ticker_hash = int(digest[:8], 16) % 10000
        return self._hour_seed + ticker_hash + extra

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return float(default)
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _format_percent(self, value: float) -> str:
        return f"{value * 100:.2f}%"

    def _format_usd(self, value: float) -> str:
        if abs(value) >= 1e9:
            return f"${value / 1e9:.2f} billion"
        if abs(value) >= 1e6:
            return f"${value / 1e6:.2f} million"
        return f"${value:,.2f}"

    def _coingecko_spot(self) -> dict[str, float]:
        try:
            url = (
                "https://api.coingecko.com/api/v3/simple/price"
                "?ids=bitcoin,ethereum&vs_currencies=usd"
            )
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            payload = response.json()
            return {
                "bitcoin_usd": self._safe_float(payload.get("bitcoin", {}).get("usd"), 0.0),
                "ethereum_usd": self._safe_float(payload.get("ethereum", {}).get("usd"), 0.0),
            }
        except Exception:
            return {"bitcoin_usd": 0.0, "ethereum_usd": 0.0}

    def fetch_earnings_snapshot(self, ticker: str) -> dict:
        data_source = "live"
        info = {}
        try:
            info = yf.Ticker(ticker).info
        except Exception:
            info = {}

        required = {
            "eps": info.get("trailingEps"),
            "revenue": info.get("totalRevenue"),
            "yoy_revenue_growth": info.get("revenueGrowth"),
            "operating_margin": info.get("operatingMargins"),
            "guidance_eps": info.get("forwardEps"),
        }

        if any(v is None for v in required.values()):
            data_source = "synthetic_seeded"
            rng = np.random.default_rng(self._stable_ticker_seed(ticker))
            eps = float(rng.uniform(2.0, 15.0))
            revenue = float(rng.uniform(10e9, 500e9))
            growth = float(rng.uniform(-0.05, 0.25))
            margin = float(rng.uniform(0.10, 0.40))
            guidance = float(eps * rng.uniform(0.95, 1.15))
            ground_truth = {
                "eps": eps,
                "revenue": revenue,
                "yoy_revenue_growth": growth,
                "operating_margin": margin,
                "guidance_eps": guidance,
            }
        else:
            ground_truth = {
                "eps": self._safe_float(required["eps"]),
                "revenue": self._safe_float(required["revenue"]),
                "yoy_revenue_growth": self._safe_float(required["yoy_revenue_growth"]),
                "operating_margin": self._safe_float(required["operating_margin"]),
                "guidance_eps": self._safe_float(required["guidance_eps"]),
            }

        narrative = (
            f"{ticker} reported trailing twelve-month earnings per share of "
            f"{self._format_usd(ground_truth['eps'])}, with total revenues reaching "
            f"{self._format_usd(ground_truth['revenue'])}. Year-over-year revenue growth "
            f"stood at {self._format_percent(ground_truth['yoy_revenue_growth'])}. "
            f"Operating margins came in at {self._format_percent(ground_truth['operating_margin'])}, "
            f"while forward guidance EPS is projected at {self._format_usd(ground_truth['guidance_eps'])}."
        )

        return {
            "narrative": narrative,
            "ground_truth": ground_truth,
            "data_source": data_source,
            "ticker": ticker,
        }

    def fetch_price_stream(self, tickers: list[str]) -> dict:
        data_source = "live"
        tickers_data: dict[str, dict] = {}

        try:
            df = yf.download(
                tickers=tickers,
                period="1d",
                interval="5m",
                auto_adjust=False,
                progress=False,
                group_by="ticker",
                threads=False,
            )

            if df is None or df.empty:
                raise ValueError("No intraday data returned")

            for ticker in tickers:
                if ticker not in df.columns.get_level_values(0):
                    raise ValueError(f"Missing ticker in downloaded frame: {ticker}")

                sub = df[ticker].tail(30)
                closes = sub["Close"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
                volumes = sub["Volume"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()

                if len(closes) < 10 or len(volumes) < 10:
                    raise ValueError(f"Insufficient data points for {ticker}")

                returns = closes.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
                volatility = float(np.std(returns)) if len(returns) > 0 else 0.0

                vol_std = float(np.std(volumes))
                volume_anomaly_score = 0.0 if vol_std == 0 else float((volumes.iloc[-1] - np.mean(volumes)) / vol_std)

                close_std = float(np.std(closes))
                z_score_latest = 0.0 if close_std == 0 else float((closes.iloc[-1] - np.mean(closes)) / close_std)

                price_direction = "flat"
                if closes.iloc[-1] > closes.iloc[0]:
                    price_direction = "up"
                elif closes.iloc[-1] < closes.iloc[0]:
                    price_direction = "down"

                tickers_data[ticker] = {
                    "close_prices": [float(v) for v in closes.tolist()],
                    "volumes": [float(v) for v in volumes.tolist()],
                    "intraday_volatility": volatility,
                    "volume_anomaly_score": volume_anomaly_score,
                    "z_score_latest": z_score_latest,
                    "price_direction": price_direction,
                }
        except Exception:
            data_source = "synthetic_seeded"
            shock_rng = np.random.default_rng(self._hour_seed + 777)
            shocked_ticker = tickers[int(shock_rng.integers(0, len(tickers)))]

            for ticker in tickers:
                rng = np.random.default_rng(self._stable_ticker_seed(ticker))
                steps = 30
                dt = 1 / 288
                mu = 0.0001
                sigma = 0.06 if ticker == shocked_ticker else 0.02
                base_price = float(rng.uniform(80.0, 420.0))

                prices = [base_price]
                for _ in range(steps - 1):
                    z = float(rng.normal(0, 1))
                    drift = (mu - 0.5 * sigma * sigma) * dt
                    diffusion = sigma * math.sqrt(dt) * z
                    next_price = prices[-1] * math.exp(drift + diffusion)
                    prices.append(float(max(1.0, next_price)))

                volumes = [float(max(100000, rng.normal(3_000_000, 900_000))) for _ in range(steps)]
                returns = np.diff(prices) / np.maximum(np.array(prices[:-1]), 1e-9)
                volatility = float(np.std(returns)) if len(returns) > 0 else 0.0

                vol_std = float(np.std(volumes))
                volume_anomaly_score = 0.0 if vol_std == 0 else float((volumes[-1] - np.mean(volumes)) / vol_std)

                close_std = float(np.std(prices))
                z_score_latest = 0.0 if close_std == 0 else float((prices[-1] - np.mean(prices)) / close_std)

                price_direction = "flat"
                if prices[-1] > prices[0]:
                    price_direction = "up"
                elif prices[-1] < prices[0]:
                    price_direction = "down"

                tickers_data[ticker] = {
                    "close_prices": [float(v) for v in prices],
                    "volumes": [float(v) for v in volumes],
                    "intraday_volatility": volatility,
                    "volume_anomaly_score": volume_anomaly_score,
                    "z_score_latest": z_score_latest,
                    "price_direction": price_direction,
                }

        highest = max(tickers_data.items(), key=lambda kv: kv[1]["intraday_volatility"])
        highest_ticker = highest[0]
        z_score = float(tickers_data[highest_ticker]["z_score_latest"])
        ranking = sorted(tickers, key=lambda t: tickers_data[t]["intraday_volatility"], reverse=True)

        ground_truth = {
            "highest_volatility_ticker": highest_ticker,
            "z_score": z_score,
            "is_genuine_anomaly": bool(z_score > 2.0),
            "volatility_ranking": ranking,
        }

        return {
            "tickers_data": tickers_data,
            "ground_truth": ground_truth,
            "data_source": data_source,
        }

    def fetch_portfolio_snapshot(self, tickers: list[str]) -> dict:
        data_source = "live"
        portfolio_value = 100_000.0
        target_weights = {t: 0.20 for t in tickers}

        prices: dict[str, float] = {}
        fundamentals: dict[str, dict] = {}

        try:
            for ticker in tickers:
                info = yf.Ticker(ticker).info
                cp = info.get("currentPrice")
                beta = info.get("beta")
                hi = info.get("fiftyTwoWeekHigh")
                lo = info.get("fiftyTwoWeekLow")
                if cp is None or beta is None or hi is None or lo is None:
                    raise ValueError(f"Missing fields for {ticker}")
                prices[ticker] = self._safe_float(cp)
                fundamentals[ticker] = {
                    "beta": self._safe_float(beta),
                    "fiftyTwoWeekHigh": self._safe_float(hi),
                    "fiftyTwoWeekLow": self._safe_float(lo),
                }
        except Exception:
            data_source = "synthetic_seeded"
            for ticker in tickers:
                rng = np.random.default_rng(self._stable_ticker_seed(ticker, extra=99))
                cp = float(rng.uniform(50.0, 450.0))
                hi = cp * float(rng.uniform(1.05, 1.40))
                lo = cp * float(rng.uniform(0.60, 0.95))
                prices[ticker] = cp
                fundamentals[ticker] = {
                    "beta": float(rng.uniform(0.7, 1.9)),
                    "fiftyTwoWeekHigh": hi,
                    "fiftyTwoWeekLow": lo,
                }

        weight_rng = np.random.default_rng(self._hour_seed)
        raw_weights = weight_rng.random(len(tickers))
        raw_weights = raw_weights / np.sum(raw_weights)
        pre_shock_allocation = {ticker: float(w) for ticker, w in zip(tickers, raw_weights.tolist())}
        pre_shock_values = {ticker: float(pre_shock_allocation[ticker] * portfolio_value) for ticker in tickers}

        shock_rng = np.random.default_rng(self._hour_seed + 42)
        shocked_ticker = tickers[int(shock_rng.integers(0, len(tickers)))]
        shock_multiplier = float(shock_rng.uniform(0.82, 0.91))

        shocked_prices = dict(prices)
        shocked_prices[shocked_ticker] = prices[shocked_ticker] * shock_multiplier

        post_shock_values = dict(pre_shock_values)
        post_shock_values[shocked_ticker] = pre_shock_values[shocked_ticker] * shock_multiplier
        post_total = sum(post_shock_values.values())
        post_shock_allocation = {ticker: float(v / post_total) for ticker, v in post_shock_values.items()}

        constraints = {
            "max_single_asset": 0.40,
            "max_trades": 3,
            "transaction_cost_rate": 0.001,
            "min_allocation_floor": 0.05,
        }

        ground_truth = {
            "target_weights": target_weights,
            "constraints": constraints,
            "shocked_ticker": shocked_ticker,
            "shock_magnitude": float(1.0 - shock_multiplier),
            "portfolio_value": portfolio_value,
            "tickers": tickers,
            "post_shock_values": post_shock_values,
            "post_shock_total": post_total,
        }

        portfolio = {
            "portfolio_value": portfolio_value,
            "prices": prices,
            "shocked_prices": shocked_prices,
            "fundamentals": fundamentals,
            "target_allocation": target_weights,
            "pre_shock_allocation": pre_shock_allocation,
            "post_shock_allocation": post_shock_allocation,
            "pre_shock_values": pre_shock_values,
            "post_shock_values": post_shock_values,
        }

        return {
            "portfolio": portfolio,
            "constraints": constraints,
            "ground_truth": ground_truth,
            "data_source": data_source,
        }

    def fetch_additional_financials(self, ticker: str) -> dict:
        rng = np.random.default_rng(self._stable_ticker_seed(ticker, extra=7))
        return {
            "sector_comparison": {
                "ticker": ticker,
                "sector": "Technology",
                "sector_avg_operating_margin": float(rng.uniform(0.12, 0.33)),
            },
            "historical_eps_trend": [float(rng.uniform(2.0, 12.0)) for _ in range(4)],
            "analyst_estimates": {
                "consensus_eps_next_q": float(rng.uniform(2.0, 12.0)),
                "buy_rating_ratio": float(rng.uniform(0.45, 0.90)),
            },
        }

    def fetch_macro_context(self) -> dict:
        rng = np.random.default_rng(self._hour_seed + 2048)
        cg = self._coingecko_spot()
        return {
            "vix_equivalent": float(rng.uniform(14.0, 38.0)),
            "sector_performance": {
                "technology": float(rng.uniform(-0.03, 0.03)),
                "consumer_discretionary": float(rng.uniform(-0.03, 0.03)),
                "financials": float(rng.uniform(-0.03, 0.03)),
            },
            "crypto_context": cg,
        }

    def fetch_next_prices(self, tickers: list[str], current_prices: dict[str, float]) -> dict:
        updates: dict[str, float] = {}
        for ticker in tickers:
            rng = np.random.default_rng(self._stable_ticker_seed(ticker, extra=500))
            mu = 0.0001
            sigma = 0.02
            z = float(rng.normal(0, 1))
            dt = 1 / 288
            factor = math.exp((mu - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt) * z)
            updates[ticker] = float(max(1.0, current_prices[ticker] * factor))

        return {
            "next_prices": updates,
            "timestamp_hour_seed": self._hour_seed,
        }
