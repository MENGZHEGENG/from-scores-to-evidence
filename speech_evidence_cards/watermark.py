from __future__ import annotations

import numpy as np


def watermark_key(length: int, *, seed: int = 2027) -> np.ndarray:
    rng = np.random.default_rng(seed)
    key = rng.choice([-1.0, 1.0], size=length)
    return key / np.sqrt(length)


def embed_spread_spectrum(signal, *, strength: float = 0.002, seed: int = 2027) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    return signal + strength * watermark_key(signal.size, seed=seed)


def watermark_score(signal, *, seed: int = 2027) -> float:
    signal = np.asarray(signal, dtype=float)
    key = watermark_key(signal.size, seed=seed)
    centered = signal - float(np.mean(signal))
    scale = float(np.linalg.norm(centered))
    if scale == 0.0:
        return 0.5
    correlation = float(np.dot(centered / scale, key))
    return float(1.0 / (1.0 + np.exp(-20.0 * correlation)))
