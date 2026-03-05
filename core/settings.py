from __future__ import annotations
import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

BASE_DIR = Path(__file__).resolve().parents[1]  # .../josh/


@dataclass(frozen=True)
class Settings:
    # Files
    config_file: Path
    keyword_path: Path
    memory_file: Path

    # API
    openai_api_key: str

    # Wake word
    picovoice_access_key: str
    pre_roll_seconds: float

    # Audio recording
    in_rate: int
    channels: int
    silence_threshold: float
    silence_seconds: float
    max_record_seconds: float
    min_record_seconds: float
    block_dur: float
    start_grace_seconds: float
    speech_start_threshold: float

    # TTS
    piper_exe: Path
    piper_voice: Path

    # Memory
    max_turns_in_context: int
    idle_timeout_seconds: int

    # Persona
    system_persona: str


def load_settings() -> Settings:
    config_file = BASE_DIR / "josh_config.json"
    if not config_file.exists():
        raise RuntimeError(f"Missing config file: {config_file}")

    config = json.loads(config_file.read_text(encoding="utf-8"))
    api_key = (config.get("api_key") or "").strip()
    if not api_key:
        raise RuntimeError("api_key is missing/blank in josh_config.json")

    # ✅ Move Picovoice key out of code if you want:
    # Put "picovoice_access_key" in josh_config.json and use it here.
    picovoice_key = (config.get("picovoice_access_key") or "").strip()
    if not picovoice_key:
        # fallback to old behavior (hardcoded key in v3); but better to put in config
        picovoice_key = "gE5cH243GhDjxY4g7SdIddNLp1s54sclQPSKqOC4b7FY9ennW49NjQ=="

    system_persona = """
You are J.O.S.H. (Just One Smart Helper), also known as Josh.

Core Personality:
- Dry, intelligent sarcasm.
- Respectful at all times.
- Always address the user as "sir".
- Call out mistakes directly but never insult.
- Tone is composed, professional, subtly amused.
- Keep responses concise and useful.

Hard Output Rules (must follow):
1) Do NOT ask generic closing/offer-to-help questions.
   Forbidden closers include (and anything similar):
   - "How can I help you?"
   - "How may I assist you?"
   - "Let me know if you need anything else."
   - "Anything else?"
   - "Is there anything else I can do?"
   - "Feel free to ask..."
   - "I'm here if you need..."
   If you feel tempted to add a closer, STOP. End the response after the useful content with a period.

2) Respond in 1–3 sentences unless asked for detail.

3) If unsure, ask ONE short clarifying question (and only if necessary to proceed).

4) If the user asks to do something risky (send email, delete files, spend money), ask for confirmation.

Examples (follow these patterns):
User: "Thanks"
Josh: "You're welcome, sir."

User: "It runs."
Josh: "Good. Now it’s worth polishing, sir."

User: "Can you fix this bug?"
Josh: "Yes, sir. Paste the error output and the function where it occurs."
""".strip()

    return Settings(
        config_file=config_file,
        keyword_path=BASE_DIR / "josh_wake_up.ppn",
        memory_file=BASE_DIR / "josh_memory.json",
        openai_api_key=api_key,
        picovoice_access_key=picovoice_key,
        pre_roll_seconds=0.8,
        in_rate=16000,
        channels=1,
        silence_threshold=0.010,
        silence_seconds=1.2,
        max_record_seconds=15.0,
        min_record_seconds=0.40,
        block_dur=0.03,
        start_grace_seconds=1.5,
        speech_start_threshold=0.02,
        piper_exe=BASE_DIR / "piper" / "piper.exe",
        piper_voice=BASE_DIR / "piper" / "voices" / "en_US-ryan-medium.onnx",
        max_turns_in_context=8,
        idle_timeout_seconds=300,
        system_persona=system_persona,
    )