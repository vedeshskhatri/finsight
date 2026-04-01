from __future__ import annotations

import uvicorn


def main() -> None:
    uvicorn.run("server.main:app", host="0.0.0.0", port=7860, workers=1)
