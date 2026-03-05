# core/brain.py
from __future__ import annotations
from typing import List, Tuple, Iterable
from openai import OpenAI
from core.memory import LongTermMemory


class Brain:
    def __init__(self, api_key: str, system_persona: str, max_turns_in_context: int = 8):
        self.client = OpenAI(api_key=api_key)
        self.system_persona = system_persona
        self.max_turns = max_turns_in_context

    # -------------------------
    # Prompt Builder
    # -------------------------
    def _build_prompt(
        self,
        user_text: str,
        short_history: List[Tuple[str, str]],
        long_mem: LongTermMemory,
    ) -> str:

        long_mem_text = "\n".join(f"- {x}" for x in long_mem.facts)

        hist_text = "\n".join(
            [f"User: {u}\nJ.O.S.H.: {a}" for (u, a) in short_history[-self.max_turns:]]
        )

        return f"""{self.system_persona}

Long-term memory (facts):
{long_mem_text if long_mem_text else "- (none yet)"}

Recent conversation:
{hist_text if hist_text else "(none yet)"}

User:
{user_text}

J.O.S.H.:
""".strip()

    # -------------------------
    # Normal (Non-Streaming) Response
    # -------------------------
    def think(
        self,
        user_text: str,
        short_history: List[Tuple[str, str]],
        long_mem: LongTermMemory,
    ) -> str:

        prompt = self._build_prompt(user_text, short_history, long_mem)

        response = self.client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
        )

        return (response.output_text or "").strip()

    # -------------------------
    # Streaming Response
    # -------------------------
    def think_stream(
        self,
        user_text: str,
        short_history: List[Tuple[str, str]],
        long_mem: LongTermMemory,
    ) -> Iterable[str]:
        """
        Yields text chunks as they arrive from OpenAI.
        """

        prompt = self._build_prompt(user_text, short_history, long_mem)

        stream = self.client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            stream=True,
        )

        for event in stream:
            # Streaming delta event
            if getattr(event, "type", None) == "response.output_text.delta":
                delta = getattr(event, "delta", None)
                if delta:
                    yield delta