# tools/router.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from tools.time_tool import get_time_string
from tools.web_tool import web_search


@dataclass(frozen=True)
class ToolResult:
    handled: bool
    text: str = ""
    tool_name: Optional[str] = None


_TIME_PATTERNS = [
    r"\bwhat\s*time\s+is\s+it\b",
    r"\btime\s+is\s+it\b",
    r"\bcurrent\s+time\b",
    r"\bwhat'?s\s+the\s+time\b",
]

_WEATHER_PATTERNS = [
    r"\bwhat'?s\s+the\s+weather\b",
    r"\bweather\b",
    r"\bforecast\b",
    r"\btemperature\b",
]

_RESEARCH_PATTERNS = [
    r"\bresearch\b",
    r"\blook\s+up\b",
    r"\bsearch\b",
    r"\bgoogle\b",
    r"\bfind\s+me\b",
]


def _matches_any(text: str, patterns: list[str]) -> bool:
    t = text.lower().strip()
    return any(re.search(p, t) for p in patterns)


def route(user_text: str) -> ToolResult:
    t = (user_text or "").strip()

    if not t:
        return ToolResult(False)

    # Time tool
    if _matches_any(t, _TIME_PATTERNS):
        return ToolResult(
            True,
            get_time_string(),
            "time_tool",
        )

    # Weather placeholder
    if _matches_any(t, _WEATHER_PATTERNS):
        return ToolResult(
            True,
            "Weather tool not installed yet, sir.",
            "weather_tool",
        )

    # Web search
    if _matches_any(t, _RESEARCH_PATTERNS):
        return ToolResult(
            True,
            web_search(user_text),
            "web_tool",
        )

    return ToolResult(False)