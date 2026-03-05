from __future__ import annotations
import time
import numpy as np
import sounddevice as sd
import soundfile as sf


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))) + 1e-12)


def record_wav(
    path: str,
    *,
    in_rate: int,
    channels: int,
    block_dur: float,
    silence_threshold: float,
    silence_seconds: float,
    max_record_seconds: float,
    min_record_seconds: float,
    start_grace_seconds: float,
    speech_start_threshold: float,
    pre_roll_frames=None,
) -> str:
    """
    Records until silence stop, with grace + latch.
    Writes 16-bit PCM WAV at in_rate.
    """
    print("\n[REC] Speak now, sir... (auto-stops on silence)")

    blocksize = int(in_rate * block_dur)
    max_frames = int(in_rate * max_record_seconds)
    min_frames = int(in_rate * min_record_seconds)
    grace_frames = int(in_rate * start_grace_seconds)

    frames = list(pre_roll_frames) if pre_roll_frames else []
    silent_blocks = 0
    silent_blocks_needed = int(silence_seconds / block_dur)

    total = sum(len(b) for b in frames)
    speech_started = False
    start = time.time()

    with sd.InputStream(
        samplerate=in_rate,
        channels=channels,
        dtype="float32",
        blocksize=blocksize,
    ) as stream:
        while True:
            block, _ = stream.read(blocksize)
            total += len(block)

            block_i16 = np.clip(block * 32767, -32768, 32767).astype(np.int16)
            frames.append(block_i16)

            level = _rms(block)

            if not speech_started and level >= speech_start_threshold:
                speech_started = True

            if total < grace_frames:
                silent_blocks = 0
            else:
                if not speech_started:
                    silent_blocks = 0
                else:
                    if level < silence_threshold:
                        silent_blocks += 1
                    else:
                        silent_blocks = 0

            if total >= max_frames:
                break
            if total >= min_frames and speech_started and silent_blocks >= silent_blocks_needed:
                break

    # normalize all frames to (N,1)
    fixed = []
    for f in frames:
        if f.ndim == 1:
            f = f.reshape(-1, 1)
        fixed.append(f)

    audio = np.concatenate(fixed, axis=0) if fixed else np.zeros((0, 1), dtype=np.int16)
    sf.write(path, audio, in_rate, subtype="PCM_16")

    dur = time.time() - start
    print(f"[REC] captured ~{dur:.2f}s (speech_started={speech_started})")
    return path