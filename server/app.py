from __future__ import annotations

# OpenEnv validator compatibility: expose app at server/app.py
from server.main import app

__all__ = ["app"]
