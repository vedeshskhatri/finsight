from __future__ import annotations

from server.data.fetcher import LiveDataFetcher
from server.models import FinancialAction, FinancialObservation, FinancialReward
from server.session_store import SessionStore
from server.tasks.task_easy import EarningsExtractionTask
from server.tasks.task_hard import PortfolioRebalanceUnderShockTask
from server.tasks.task_medium import AnomalyTriageTask


class FinSightEnv:
    def __init__(self) -> None:
        self.fetcher = LiveDataFetcher()
        self.sessions = SessionStore()
        self._episode_counter = 0
        self.tasks = {
            "easy": EarningsExtractionTask(),
            "medium": AnomalyTriageTask(),
            "hard": PortfolioRebalanceUnderShockTask(),
        }
        self.request_step_reward = {"easy": 0.05, "medium": 0.05, "hard": 0.08}
        self.valid_request_types = {
            "easy": {"financials", "macro"},
            "medium": {"macro"},
            "hard": {"next_prices", "macro"},
        }

    def _build_observation(self, session_state: dict) -> FinancialObservation:
        return FinancialObservation(
            session_id=session_state["session_id"],
            task_id=session_state["task_id"],
            task_description=session_state["task_description"],
            step_num=session_state["step_num"],
            max_steps=session_state["max_steps"],
            data_payload=session_state["data_payload"],
            context=session_state["context"],
            data_source=session_state["data_source"],
            available_actions=["submit_answer", "request_data", "noop"],
        )

    def reset(self, task_id: str, session_id: str) -> FinancialObservation:
        if task_id not in self.tasks:
            raise ValueError(f"Unknown task_id: {task_id}")

        task = self.tasks[task_id]
        self._episode_counter += 1
        initial = task.initialize(
            fetcher=self.fetcher,
            session_id=session_id,
            episode_num=self._episode_counter,
        )

        # Ground truth is locked at reset and never recomputed.
        session_state = {
            "session_id": session_id,
            "task_id": task_id,
            "task_description": initial["task_description"],
            "step_num": 0,
            "max_steps": task.max_steps,
            "data_payload": initial["data_payload"],
            "context": task.context(),
            "data_source": initial["data_source"],
            "available_actions": ["submit_answer", "request_data", "noop"],
            "ground_truth": initial["ground_truth"],
            "cumulative_reward": 0.0,
            "done": False,
            "task_state": initial.get("task_state", {}),
        }

        self.sessions.upsert(session_id, session_state)
        return self._build_observation(session_state)

    def step(
        self,
        action: FinancialAction,
        session_id: str,
    ) -> tuple[FinancialObservation, FinancialReward, bool, dict]:
        session_state = self.sessions.get(session_id)
        if session_state is None:
            raise KeyError(f"Session not found: {session_id}")

        if session_state.get("done"):
            obs = self._build_observation(session_state)
            reward = FinancialReward(
                score=0.0,
                partial_scores={"episode_done": 1.0},
                feedback="Episode is already complete.",
                done=True,
                step_reward=0.0,
                cumulative_reward=float(session_state.get("cumulative_reward", 0.0)),
            )
            info = {
                "session_id": session_id,
                "step_num": session_state["step_num"],
                "data_source": session_state["data_source"],
                "task_id": session_state["task_id"],
            }
            return obs, reward, True, info

        task_id = session_state["task_id"]
        task = self.tasks[task_id]

        session_state["step_num"] += 1
        step_reward = 0.0
        done = False
        score = 0.0
        partial_scores: dict[str, float] = {}
        feedback = ""

        if action.action_type == "noop":
            step_reward = -0.05
            feedback = "No-op action taken. Penalty applied for wasting a step."
            partial_scores = {"noop_penalty": -0.05}

        elif action.action_type == "request_data":
            data_type = str(action.payload.get("data_type", ""))
            if data_type in self.valid_request_types.get(task_id, set()):
                additional = task.request_data(
                    fetcher=self.fetcher,
                    session_state=session_state,
                    data_type=data_type,
                )
                session_state["data_payload"] = {
                    **session_state.get("data_payload", {}),
                    "additional_data": additional,
                }
                step_reward = self.request_step_reward[task_id]
                feedback = f"Additional data fetched for data_type={data_type}."
                partial_scores = {"request_data_bonus": step_reward}
            else:
                step_reward = 0.0
                feedback = f"Invalid data_type={data_type} for task={task_id}."
                partial_scores = {"invalid_request_data": 1.0}

        elif action.action_type == "submit_answer":
            graded = task.grade(
                submitted=action.payload,
                ground_truth=session_state["ground_truth"],
                cumulative_before=float(session_state.get("cumulative_reward", 0.0)),
            )
            score = graded.score
            step_reward = graded.step_reward
            partial_scores = graded.partial_scores
            feedback = graded.feedback
            done = True

        if session_state["step_num"] >= session_state["max_steps"] and not done:
            done = True
            step_reward = 0.0
            score = 0.0
            partial_scores = {"max_steps_reached": 1.0}
            feedback = "Maximum steps reached without valid submission."

        session_state["cumulative_reward"] = float(session_state.get("cumulative_reward", 0.0)) + step_reward
        session_state["done"] = done
        self.sessions.upsert(session_id, session_state)

        reward = FinancialReward(
            score=float(score if done and action.action_type == "submit_answer" else 0.0),
            partial_scores=partial_scores,
            feedback=feedback,
            done=done,
            step_reward=float(step_reward),
            cumulative_reward=float(session_state["cumulative_reward"]),
        )

        obs = self._build_observation(session_state)
        info = {
            "session_id": session_id,
            "step_num": session_state["step_num"],
            "data_source": session_state["data_source"],
            "task_id": task_id,
        }
        return obs, reward, done, info

    def state(self, session_id: str) -> dict:
        session_state = self.sessions.get(session_id)
        if session_state is None:
            raise KeyError(f"Session not found: {session_id}")
        return dict(session_state)
