"""Run the SmartDoc Assistant LINE webhook with Uvicorn."""

from __future__ import annotations

from assistant import SmartDocAssistant
from config import settings
from line_webhook import create_app


app = create_app(SmartDocAssistant(settings), settings)

