# tools/llm_tool_router.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI


@dataclass(frozen=True)
class LLMToolDecision:
    tool: str                   # "none" | "time" | "web" | "weather"
    query: Optional[str] = None # for web search
    timezone: Optional[str] = None # for time tool (optional)


def decide_tool(api_key: str, user_text: str) -> LLMToolDecision:
    """
    Uses the LLM to decide whether to call a tool.
    Returns a small JSON decision.
    """
    client = OpenAI(api_key=api_key)

    system = (
        "You are a tool router for a voice assistant.\n"
        "Choose ONE tool that best answers the user.\n"
        "Tools:\n"
        "- none: answer normally (no tool)\n"
        "- time: for current time questions\n"
        "- weather: for weather/forecast questions\n"
        "- web: for facts/prices/news/research requiring the internet\n\n"
        "Return ONLY valid JSON with keys: tool, query, timezone.\n"
        "Rules:\n"
        "- tool must be one of: none, time, weather, web\n"
        "- If tool=web, produce a CLEAN search query (remove 'google', 'look up', etc.)\n"
        "- If tool=time, timezone may be null unless user specifies a city/timezone\n"
    )

    resp = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ],
    )

    raw = (resp.output_text or "").strip()

    try:
        data = json.loads(raw)
        tool = str(data.get("tool", "none")).strip().lower()
        query = data.get("query")
        timezone = data.get("timezone")
        if tool not in {"none", "time", "weather", "web"}:
            tool = "none"
        if tool != "web":
            query = None
        if tool != "time":
            timezone = None
        return LLMToolDecision(tool=tool, query=query, timezone=timezone)
    except Exception:
        return LLMToolDecision(tool="none")