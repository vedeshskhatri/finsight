---
title: FinSight-Env
emoji: "💹"
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# FinSight-Env

## Why This Environment Exists
FinSight-Env is a live financial operations environment for training and evaluating AI agents on tasks that resemble real analyst workflows instead of toy workflows. Most agent benchmarks under-represent finance-specific pressure: extracting critical numbers from narrative disclosures, triaging intraday anomalies with noisy streams, and rebalancing a portfolio under sudden market stress with strict constraints.

FinSight-Env closes that gap with fresh market data on every reset, deterministic grading, and a realistic market shock scenario in the hard task.

## Environment Overview

```text
+--------------------+        +----------------------+        +-------------------+
| Agent / LLM Policy | <----> | FastAPI Env Endpoints| <----> | FinSightEnv Core  |
+--------------------+        +----------------------+        +-------------------+
                                         |                               |
                                         v                               v
                              +--------------------+            +-------------------+
                              | Task Controllers   |            | Session Store     |
                              | easy/medium/hard   |            | locked groundtruth|
                              +--------------------+            +-------------------+
                                         |
                                         v
                              +--------------------+
                              | LiveDataFetcher    |
                              | yfinance primary   |
                              | CoinGecko enrich   |
                              | GBM fallback       |
                              +--------------------+
                                         |
                                         v
                              +--------------------+
                              | Deterministic      |
                              | Graders            |
                              +--------------------+
```

## Observation Space

| Field | Type | Description |
|---|---|---|
| session_id | str | Stable episode/session identifier |
| task_id | easy\|medium\|hard | Current task |
| task_description | str | Human-readable task summary |
| step_num | int | Current step index |
| max_steps | int | Max steps before forced termination |
| data_payload | dict | Financial data visible to the agent |
| context | str | Action format instructions with concrete examples |
| data_source | live\|synthetic_seeded | Source tag for reproducibility |
| available_actions | list[str] | Always: submit_answer, request_data, noop |

## Action Space

| action_type | payload schema | When to use |
|---|---|---|
| submit_answer | Task-specific structured JSON | Final answer submission |
| request_data | {"data_type": "next_prices"\|"financials"\|"macro"} | Pull additional context |
| noop | {} | Skip step (penalized) |

## Tasks

### Easy: Earnings Metric Extraction
- Description: Parse an earnings narrative and extract 5 numerical metrics.
- What the agent sees: Natural-language paragraph containing EPS, revenue, growth, margin, and guidance.
- What the agent must submit: {"eps", "revenue", "yoy_revenue_growth", "operating_margin", "guidance_eps"}.
- Grading breakdown:

| Component | Weight | Rule |
|---|---|---|
| field_presence | 0.30 | Required keys present with numeric values |
| numerical_accuracy | 0.50 | Relative error tiers per metric |
| format_validity | 0.20 | Numeric types and decimal-form percentages |

### Medium: Market Anomaly Triage
- Description: Analyze 30-point intraday streams for 5 tickers and identify anomalies.
- What the agent sees: Price/volume streams plus computed volatility, z-score, and volume anomaly metrics.
- What the agent must submit: anomalous_ticker, classification, action_recommendation, volatility_rank.
- Grading breakdown:

| Component | Weight | Rule |
|---|---|---|
| correct_ticker | 0.35 | Exact match to pre-locked highest-volatility ticker |
| correct_classification | 0.30 | Match genuine_anomaly/noise from locked z-score rule |
| correct_action | 0.15 | Deterministic mapping from classification to action |
| volatility_rank_accuracy | 0.20 | Kendall tau normalized to [0,1] |

### Hard: Portfolio Rebalancing Under Market Shock
- Description: One asset experiences an 8-18% shock and distorts allocations. Agent must rebalance under strict constraints.
- What the agent sees: Pre-shock and post-shock allocations, current prices, constraints, target weights, and explicit shocked ticker plus magnitude.
- What the agent must submit: Trade sequence of <=3 BUY/SELL instructions with USD amounts.
- Constraint rules:
  - No allocation above 40% after simulation.
  - Cannot drop any asset below 5%.
  - Max 3 trades.
  - Transaction cost = 0.1% of trade value.
- Grading breakdown:

| Component | Weight | Rule |
|---|---|---|
| trade_count_valid | 0.10 | 1.0 if <=3 trades else 0.0 |
| constraint_no_single_over_40pct | 0.25 | 1.0 no violations, 0.5 one, 0.0 two+ |
| constraint_min_floor_5pct | 0.15 | Proportional pass ratio |
| target_proximity | 0.35 | Normalized MAE to 20% target |
| transaction_cost_efficiency | 0.15 | Lower total traded notional gets higher score |

## Reward Function Design
- Step rewards:
  - request_data: +0.05 (easy/medium), +0.08 (hard)
  - noop: -0.05
  - submit_answer: immediate final score for that episode
- Cumulative reward: tracked as the running sum of step rewards.
- Termination:
  - submit_answer always ends episode.
  - hitting max_steps without submission ends with score 0.0.

## Data Sources
- Primary: Yahoo Finance via yfinance for earnings and intraday streams.
- Enrichment/Fallback assist: CoinGecko public API for crypto context.
- Synthetic fallback: GBM-style seeded generation with hour-based seeds for reproducibility.
- Every observation includes data_source: live or synthetic_seeded.

## Setup and Running

### Docker
```bash
docker build -t finsight-env .
docker run -p 7860:7860 finsight-env
```

### Local
```bash
pip install -r requirements.txt
uvicorn server.main:app --port 7860
```

### Inference
```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
export HF_TOKEN=your_token_here
python inference.py
```

## Baseline Scores

| Task | Score | Steps Used | Data Source |
|---|---:|---:|---|
| easy | 0.000-1.000 | 1-10 | live/synthetic_seeded |
| medium | 0.000-1.000 | 1-10 | live/synthetic_seeded |
| hard | 0.000-1.000 | 1-10 | live/synthetic_seeded |

## Leaderboard
FinSight-Env exposes a persistent leaderboard:
- POST /leaderboard/submit: append a score entry, sort by total_score descending, keep top 10.
- GET /leaderboard: return current top 10.

The leaderboard persists to leaderboard.json and uses a threading lock to avoid race conditions on concurrent writes.
