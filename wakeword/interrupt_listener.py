# wakeword/interrupt_listener.py
from __future__ import annotations

from pathlib import Path
import sounddevice as sd
import pvporcupine


def wait_for_wake_word_interrupt(access_key: str, keyword_path: Path, stop_event) -> None:
    """
    Listen for the wake word while TTS is playing.
    When detected, sets stop_event and returns.
    stop_event is also used to cleanly stop the listener when speech finishes.
    """
    porcupine = pvporcupine.create(
        access_key=access_key,
        keyword_paths=[str(keyword_path)],
    )

    sample_rate = porcupine.sample_rate
    frame_length = porcupine.frame_length

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocksize=frame_length,
        ) as stream:
            while not stop_event.is_set():
                pcm, _ = stream.read(frame_length)

                pcm2d = pcm.reshape(-1, 1)
                pcm1d = pcm2d[:, 0]

                if porcupine.process(pcm1d) >= 0:
                    stop_event.set()
                    return
    finally:
        porcupine.delete()