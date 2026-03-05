# tools/time_tool.py
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional


def get_time_string(timezone: Optional[str] = None) -> str:
    """
    Returns the current time formatted in a Josh-style response.
    If timezone is provided (IANA string), it will use it.
    Otherwise, it uses local system time.
    """

    try:
        if timezone:
            now = datetime.now(ZoneInfo(timezone))
        else:
            now = datetime.now()
    except Exception:
        # Fallback to local time if timezone fails
        now = datetime.now()

    time_str = now.strftime("%I:%M %p").lstrip("0")
    date_str = now.strftime("%A, %B %d")

    return f"It is {time_str}, sir."