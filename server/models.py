from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class FinancialObservation(BaseModel):
    session_id: str
    task_id: Literal["easy", "medium", "hard"]
    task_description: str
    step_num: int
    max_steps: int
    data_payload: dict
    context: str
    data_source: Literal["live", "synthetic_seeded"]
    available_actions: list[str] = Field(
        default_factory=lambda: ["submit_answer", "request_data", "noop"]
    )


class FinancialAction(BaseModel):
    action_type: Literal["submit_answer", "request_data", "noop"]
    payload: dict
    reasoning: str


class FinancialReward(BaseModel):
    score: float
    partial_scores: dict[str, float]
    feedback: str
    done: bool
    step_reward: float
    cumulative_reward: float


class TaskInfo(BaseModel):
    task_id: str
    name: str
    difficulty: Literal["easy", "medium", "hard"]
    description: str
    max_steps: int
    observation_space: str
    action_space: str
