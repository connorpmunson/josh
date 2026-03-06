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


def speak_piper(text: str, *, piper_exe: Path, piper_voice: Path) -> None:
    text = (text or "").strip()
    if not text:
        return

    fd, out_wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    try:
        cmd = [str(piper_exe), "-m", str(piper_voice), "-f", out_wav, "-q"]
        subprocess.run(cmd, input=text.encode("utf-8"), check=True)

        audio, sr = sf.read(out_wav, dtype="int16")
        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)

        with sd.OutputStream(samplerate=sr, channels=1, dtype="int16", blocksize=8192) as out:
            out.write(audio)
    finally:
        try:
            os.remove(out_wav)
        except OSError:
            pass


def say(text: str, settings) -> None:
    speak_piper(text, piper_exe=settings.piper_exe, piper_voice=settings.piper_voice)


_SENT_END_RE = re.compile(r"([.!?]+)(\s+|$)")


def _tts_worker(q: Queue[Optional[str]], *, piper_exe: Path, piper_voice: Path):
    buf = ""

    def flush_sentence(sentence: str):
        s = sentence.strip()
        if s:
            speak_piper(s, piper_exe=piper_exe, piper_voice=piper_voice)

    while True:
        item = q.get()
        if item is None:
            if buf.strip():
                flush_sentence(buf)
            return

        buf += item

        while True:
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
            print(delta, end="", flush=True)
            full += delta
            q.put(delta)
    finally:
        q.put(None)
        worker.join()

    final = full.strip()

    # Failsafe: if streaming yielded nothing, do one normal call and speak it.
    if not final:
        final = (brain.think(user_text, short_history, long_mem) or "").strip()
        if final:
            speak_piper(final, piper_exe=piper_exe, piper_voice=piper_voice)

    return final