from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class LongTermMemory:
    facts: List[str]

    @classmethod
    def load(cls, path: Path) -> "LongTermMemory":
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                facts = list(data.get("facts", []))
                return cls(facts=facts)
            except Exception:
                pass
        return cls(facts=[])

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"facts": self.facts}, indent=2), encoding="utf-8")

    def add_fact(self, fact: str) -> None:
        fact = fact.strip()
        if not fact:
            return
        if fact not in self.facts:
            self.facts.append(fact)
            self.facts = self.facts[-50:]


def maybe_store_memory(user_text: str, mem: LongTermMemory) -> None:
    triggers = [
        "remember that",
        "from now on",
        "my preference is",
        "i don't like",
        "i do not like",
        "please remember",
    ]
    lower = user_text.lower()
    if any(t in lower for t in triggers):
        mem.add_fact(user_text)