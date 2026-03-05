# wakeword/porcupine_listener.py
from __future__ import annotations
from pathlib import Path
import sounddevice as sd
import pvporcupine


def wait_for_wake_word(access_key: str, keyword_path: Path, pre_roll_seconds: float):
    """
    Always-on local wake word detection.
    Returns a list of int16 blocks captured AFTER detection (post-roll).
    """
    porcupine = pvporcupine.create(
        access_key=access_key,
        keyword_paths=[str(keyword_path)],
    )

    sample_rate = porcupine.sample_rate
    frame_length = porcupine.frame_length
    post_roll_blocks = max(1, int(pre_roll_seconds * sample_rate / frame_length))

    print("J.O.S.H.: Sleeping. Say 'Josh wake up' to begin, sir.")

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocksize=frame_length,
        ) as stream:
            while True:
                pcm, _ = stream.read(frame_length)

                pcm2d = pcm.reshape(-1, 1)
                pcm1d = pcm2d[:, 0]

                if porcupine.process(pcm1d) >= 0:
                    post = []
                    for _ in range(post_roll_blocks):
                        p, _ = stream.read(frame_length)
                        post.append(p.reshape(-1, 1).copy())
                    return post
    finally:
        porcupine.delete()