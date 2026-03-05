from __future__ import annotations
from faster_whisper import WhisperModel


class WhisperSTT:
    def __init__(self, model_size: str = "base"):
        # cpu + int8 = your current fast setup
        self._whisper = WhisperModel(model_size, device="cpu", compute_type="int8")

    def transcribe(self, wav_path: str) -> str:
        segments, _info = self._whisper.transcribe(wav_path, vad_filter=True, language="en")
        return " ".join(seg.text.strip() for seg in segments).strip()