from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel

from server.env import FinSightEnv
from server.models import FinancialAction, TaskInfo


class ResetRequest(BaseModel):
	task_id: str | None = "easy"
	session_id: str | None = None


class LeaderboardSubmitRequest(BaseModel):
	session_id: str | None = None
	agent_name: str
	scores: dict[str, float]
	total_score: float | None = None


class LeaderboardEntry(BaseModel):
	session_id: str
	agent_name: str
	scores: dict[str, float]
	total_score: float


app = FastAPI(title="FinSight-Env", version="1.0.0")
env = FinSightEnv()
leaderboard_lock = threading.Lock()
leaderboard_path = Path(__file__).resolve().parent.parent / "leaderboard.json"


@app.on_event("startup")
def startup_event() -> None:
	if not leaderboard_path.exists():
		leaderboard_path.write_text("[]", encoding="utf-8")


def _read_leaderboard() -> list[dict]:
	if not leaderboard_path.exists():
		return []
	try:
		return json.loads(leaderboard_path.read_text(encoding="utf-8"))
	except Exception:
		return []


def _write_leaderboard(data: list[dict]) -> None:
	leaderboard_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


@app.post("/reset")
async def reset(request: Request):
	task_id = "easy"
	session_id: str | None = None

	try:
		payload = await request.json()
	except Exception:
		payload = None

	if isinstance(payload, dict):
		task_id = str(payload.get("task_id") or "easy")
		raw_session_id = payload.get("session_id")
		if raw_session_id is not None:
			session_id = str(raw_session_id)

	session_id = session_id or str(uuid.uuid4())
	try:
		return env.reset(task_id=task_id, session_id=session_id)
	except ValueError as exc:
		raise HTTPException(status_code=422, detail=str(exc)) from exc
	except Exception as exc:
		raise HTTPException(status_code=500, detail=f"reset failed: {exc}") from exc


@app.post("/step")
def step(action: FinancialAction, x_session_id: str = Header(..., alias="X-Session-ID")):
	try:
		observation, reward, done, info = env.step(action=action, session_id=x_session_id)
		return {
			"observation": observation,
			"reward": reward,
			"done": done,
			"info": info,
		}
	except KeyError as exc:
		raise HTTPException(status_code=404, detail=str(exc)) from exc
	except Exception as exc:
		raise HTTPException(status_code=500, detail=f"step failed: {exc}") from exc


@app.get("/state")
def state(x_session_id: str = Header(..., alias="X-Session-ID")):
	try:
		return env.state(session_id=x_session_id)
	except KeyError as exc:
		raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/tasks", response_model=list[TaskInfo])
def tasks():
	return [env.tasks[k].task_info() for k in ["easy", "medium", "hard"]]


@app.get("/health")
def health():
	return {"status": "ok", "version": "1.0.0", "tasks_available": ["easy", "medium", "hard"]}


@app.get("/")
def root():
	return {
		"name": "FinSight-Env",
		"status": "running",
		"message": "Use /docs for interactive API docs, /health for status, and /tasks for task metadata.",
		"endpoints": ["/health", "/tasks", "/reset", "/step", "/leaderboard", "/docs", "/web"],
	}


@app.get("/web")
def web():
	return {
		"title": "FinSight-Env",
		"description": "Live Financial Operations Environment for AI agent training and evaluation.",
		"quickstart": {
			"reset": {"method": "POST", "path": "/reset", "body": {"task_id": "easy", "session_id": "optional"}},
			"step": {
				"method": "POST",
				"path": "/step",
				"headers": {"X-Session-ID": "<session_id_from_reset>"},
				"body": {"action_type": "noop", "payload": {}, "reasoning": "example"},
			},
		},
	}


@app.post("/leaderboard/submit")
def leaderboard_submit(payload: LeaderboardSubmitRequest):
	total_score = payload.total_score
	if total_score is None:
		if payload.scores:
			total_score = sum(payload.scores.values()) / len(payload.scores)
		else:
			total_score = 0.0

	entry = LeaderboardEntry(
		session_id=payload.session_id or str(uuid.uuid4()),
		agent_name=payload.agent_name,
		scores=payload.scores,
		total_score=float(total_score),
	)

	with leaderboard_lock:
		data = _read_leaderboard()
		data.append(entry.model_dump())
		data = sorted(data, key=lambda x: float(x.get("total_score", 0.0)), reverse=True)[:10]
		_write_leaderboard(data)

	rank = next((i + 1 for i, row in enumerate(data) if row["session_id"] == entry.session_id), len(data))
	return {"rank": rank, "leaderboard": data}


@app.get("/leaderboard")
def leaderboard():
	with leaderboard_lock:
		data = _read_leaderboard()
	return data


def main():
	import uvicorn

	uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
	main()
