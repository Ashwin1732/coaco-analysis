"""Convert raw voltage arrays to playable WAV files."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.io import wavfile

logger = logging.getLogger(__name__)

INT16_HEADROOM = 32000


def normalize_for_playback(voltage: np.ndarray) -> np.ndarray:
    """Remove DC offset and scale peak to int16 headroom."""
    x = voltage.astype(np.float64)
    x = x - np.mean(x)
    peak = float(np.max(np.abs(x)))
    if peak <= 0:
        return np.zeros(len(x), dtype=np.int16)
    scaled = x / peak * INT16_HEADROOM
    clipped = np.clip(scaled, -32767, 32767)
    return clipped.astype(np.int16)


def write_wav(voltage: np.ndarray, sample_rate_hz: float, output_path) -> object:
    """Write a mono 16-bit PCM WAV at the true derived sample rate."""
    if hasattr(output_path, "parent"):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
    pcm = normalize_for_playback(voltage)
    rate = int(round(sample_rate_hz))
    if rate <= 0:
        raise ValueError(f"Invalid sample rate: {sample_rate_hz}")

    if rate < 2000 or rate > 96000:
        logger.warning("Unusual sample rate %.2f Hz — writing WAV without resampling", rate)

    # wavfile.write accepts strings, Path objects, and file-like objects
    out = str(output_path) if hasattr(output_path, "parent") else output_path
    wavfile.write(out, rate, pcm)
    
    name = getattr(output_path, "name", "file-like object")
    logger.info("Wrote WAV %s (%d samples @ %d Hz)", name, len(pcm), rate)
    return output_path
