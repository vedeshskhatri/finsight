from __future__ import annotations

from threading import Lock


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, dict] = {}
        self._lock = Lock()

    def upsert(self, session_id: str, state: dict) -> None:
        with self._lock:
            self._sessions[session_id] = state

    def get(self, session_id: str) -> dict | None:
        with self._lock:
            return self._sessions.get(session_id)

    def exists(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._sessions

    def all(self) -> dict[str, dict]:
        with self._lock:
            return dict(self._sessions)
