# tools/planner.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional, List

from openai import OpenAI


@dataclass(frozen=True)
class Plan:
    needs_tools: bool
    tools: List[str]              # subset of ["time", "weather", "web"]
    web_query: Optional[str] = None
    timezone: Optional[str] = None


def decide_plan(api_key: str, user_text: str) -> Plan:
    """
    One-call intent planner:
    - Prefer NO tools when reasoning is enough (math, conversions, time differences, etc.)
    - Use tools only when needed
    """
    client = OpenAI(api_key=api_key)

    system = (
        "You are an intent planner for a voice assistant.\n"
        "Decide whether ANY external tool is needed to answer the user.\n\n"
        "Available tools:\n"
        "- time: ONLY to get the current time right now\n"
        "- weather: ONLY for weather/forecast\n"
        "- web: ONLY for internet research, current prices/news, or facts not known from the user message\n\n"
        "CRITICAL RULES:\n"
        "- If the user is asking to CALCULATE anything (math, conversions, per-year/per-month, minutes until, time differences), DO NOT use tools.\n"
        "- If the user asks something answerable by general knowledge or reasoning, DO NOT use tools.\n"
        "- Use time ONLY for 'what time is it right now' (not durations).\n"
        "- Use web ONLY when you truly need live or specific factual lookup.\n\n"
        "Return ONLY valid JSON with these keys:\n"
        "- needs_tools: boolean\n"
        "- tools: array (each item is one of: time, weather, web)\n"
        "- web_query: string or null (only if web is included)\n"
        "- timezone: string or null (only if time is included)\n\n"
        "If tools is empty, needs_tools must be false."
    )

    resp = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": (user_text or "").strip()},
        ],
    )

    raw = (resp.output_text or "").strip()

    try:
        data = json.loads(raw)

        tools = data.get("tools") or []
        if not isinstance(tools, list):
            tools = []

        tools_norm: List[str] = []
        for t in tools:
            t = str(t).strip().lower()
            if t in {"time", "weather", "web"} and t not in tools_norm:
                tools_norm.append(t)

        needs_tools = bool(data.get("needs_tools", False) and bool(tools_norm))

        web_query = data.get("web_query") if "web" in tools_norm else None
        timezone = data.get("timezone") if "time" in tools_norm else None

        web_query = (str(web_query).strip() if isinstance(web_query, str) and web_query.strip() else None)
        timezone = (str(timezone).strip() if isinstance(timezone, str) and timezone.strip() else None)

        return Plan(needs_tools=needs_tools, tools=tools_norm, web_query=web_query, timezone=timezone)

    except Exception:
        # Safe fallback: no tools.
        return Plan(needs_tools=False, tools=[])