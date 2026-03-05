# audio/tts_piper.py
from __future__ import annotations

import os
import re
import subprocess
import tempfile
import threading
from pathlib import Path
from queue import Queue
from typing import Optional

import sounddevice as sd
import soundfile as sf


# ----------------------------
# Global interrupt state
# ----------------------------
_tts_stop_event = threading.Event()
_tts_lock = threading.Lock()
_tts_is_speaking = False


def stop_speaking() -> None:
    """
    Immediately request TTS to stop (barge-in).
    Safe to call from any thread.
    """
    _tts_stop_event.set()


def _reset_stop_flag() -> None:
    _tts_stop_event.clear()


def is_speaking() -> bool:
    return _tts_is_speaking


# ----------------------------
# Non-streaming: speak once (interruptible)
# ----------------------------
def speak_piper(text: str, *, piper_exe: Path, piper_voice: Path) -> None:
    """
    Offline TTS via Piper. Generates a WAV, plays it in small chunks so we can interrupt, then deletes it.
    """
    text = (text or "").strip()
    if not text:
        return

    global _tts_is_speaking

    # Prevent overlapping speech threads
    if _tts_is_speaking:
        _tts_stop_event.set()
        while _tts_is_speaking:
            threading.Event().wait(0.01)

    with _tts_lock:
        _reset_stop_flag()

    fd, out_wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    _tts_is_speaking = True

    try:
        cmd = [str(piper_exe), "-m", str(piper_voice), "-f", out_wav, "-q"]
        subprocess.run(cmd, input=text.encode("utf-8"), check=True)

        audio, sr = sf.read(out_wav, dtype="int16")
        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)

        # Play in chunks so we can stop mid-utterance
        block = 2048
        with sd.OutputStream(samplerate=sr, channels=1, dtype="int16", blocksize=block) as out:
            n = audio.shape[0]
            i = 0
            while i < n:
                if _tts_stop_event.is_set():
                    break
                j = min(i + block, n)
                out.write(audio[i:j])
                i = j

    finally:
        _tts_is_speaking = False
        try:
            os.remove(out_wav)
        except OSError:
            pass
        # keep stop flag as-is (caller may want it set). next speak resets.


def say(text: str, settings) -> None:
    speak_piper(text, piper_exe=settings.piper_exe, piper_voice=settings.piper_voice)


# ----------------------------
# Streaming: speak as text arrives (interruptible)
# ----------------------------
_SENT_END_RE = re.compile(r"([.!?]+)(\s+|$)")


def _tts_worker(
    q: Queue[Optional[str]],
    *,
    piper_exe: Path,
    piper_voice: Path,
):
    buf = ""

    def flush_sentence(sentence: str):
        s = sentence.strip()
        if s:
            speak_piper(s, piper_exe=piper_exe, piper_voice=piper_voice)

    while True:
        item = q.get()

        if item is None:
            if _tts_stop_event.is_set():
                return
            if buf.strip():
                flush_sentence(buf)
            return

        if _tts_stop_event.is_set():
            return

        buf += item

        while True:
            if _tts_stop_event.is_set():
                return

            m = _SENT_END_RE.search(buf)
            if not m:
                break

            end_idx = m.end()
            sentence = buf[:end_idx]
            buf = buf[end_idx:]

            if len(sentence.strip()) >= 2:
                flush_sentence(sentence)


def stream_to_piper(
    brain,
    user_text: str,
    short_history,
    long_mem,
    *,
    piper_exe: Path,
    piper_voice: Path,
) -> str:
    q: Queue[Optional[str]] = Queue()
    worker = threading.Thread(
        target=_tts_worker,
        args=(q,),
        kwargs={"piper_exe": piper_exe, "piper_voice": piper_voice},
        daemon=True,
    )
    worker.start()

    full = ""

    try:
        for delta in brain.think_stream(user_text, short_history, long_mem):
            if not delta:
                continue
            if _tts_stop_event.is_set():
                break

            print(delta, end="", flush=True)
            full += delta
            q.put(delta)

    finally:
        q.put(None)
        worker.join()

    return full.strip()