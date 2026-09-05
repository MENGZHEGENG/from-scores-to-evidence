from __future__ import annotations

from itertools import pairwise

import numpy as np


def _validate(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if labels.shape != scores.shape:
        raise ValueError(f"labels and scores must have the same shape, got {labels.shape} and {scores.shape}")
    unique = set(np.unique(labels).tolist())
    if not unique.issubset({0, 1}) or len(unique) != 2:
        raise ValueError("labels must contain both binary classes 0 and 1")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must be finite")
    return labels, scores


def eer(labels, scores) -> float:
    """Equal error rate for scores where larger values indicate class 1."""
    labels, scores = _validate(labels, scores)
    order = np.argsort(scores)[::-1]
    sorted_labels = labels[order]
    positives = np.sum(sorted_labels == 1)
    negatives = np.sum(sorted_labels == 0)
    false_rejects = np.r_[positives, positives - np.cumsum(sorted_labels == 1)] / positives
    false_accepts = np.r_[0, np.cumsum(sorted_labels == 0)] / negatives
    idx = int(np.argmin(np.abs(false_rejects - false_accepts)))
    return float((false_rejects[idx] + false_accepts[idx]) / 2.0)


def operating_threshold(labels, scores) -> float:
    """Threshold closest to the equal-error operating point."""
    labels, scores = _validate(labels, scores)
    best_key: tuple[float, float] | None = None
    best_threshold = 0.5
    for threshold in sorted({float(score) for score in scores}):
        predictions = scores >= threshold
        false_positive = np.sum((predictions == 1) & (labels == 0))
        true_negative = np.sum((predictions == 0) & (labels == 0))
        false_negative = np.sum((predictions == 0) & (labels == 1))
        true_positive = np.sum((predictions == 1) & (labels == 1))
        fpr = false_positive / (false_positive + true_negative) if false_positive + true_negative else 0.0
        fnr = false_negative / (false_negative + true_positive) if false_negative + true_positive else 0.0
        key = (abs(fpr - fnr), (fpr + fnr) / 2.0)
        if best_key is None or key < best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def min_dcf(labels, scores, p_target: float = 0.05, c_miss: float = 1.0, c_fa: float = 10.0) -> float:
    labels, scores = _validate(labels, scores)
    order = np.argsort(scores)[::-1]
    sorted_labels = labels[order]
    positives = np.sum(sorted_labels == 1)
    negatives = np.sum(sorted_labels == 0)
    p_miss = np.r_[positives, positives - np.cumsum(sorted_labels == 1)] / positives
    p_fa = np.r_[0, np.cumsum(sorted_labels == 0)] / negatives
    costs = c_miss * p_target * p_miss + c_fa * (1.0 - p_target) * p_fa
    default = min(c_miss * p_target, c_fa * (1.0 - p_target))
    return float(np.min(costs) / default)


def expected_calibration_error(labels, probabilities, n_bins: int = 15) -> float:
    labels, probabilities = _validate(labels, probabilities)
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must be in [0, 1]")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for left, right in pairwise(edges):
        mask = (probabilities >= left) & (probabilities < right)
        if right == 1.0:
            mask |= probabilities == 1.0
        if np.any(mask):
            total += float(np.mean(mask)) * abs(float(np.mean(probabilities[mask])) - float(np.mean(labels[mask])))
    return float(total)


def metric_row(name: str, labels, scores) -> dict[str, float | int | str]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    return {
        "method": name,
        "n": int(labels.size),
        "bonafide": int(np.sum(labels == 1)),
        "spoof": int(np.sum(labels == 0)),
        "eer_percent": 100.0 * eer(labels, scores),
        "min_dcf": min_dcf(labels, scores),
        "ece": expected_calibration_error(labels, np.clip(scores, 0.0, 1.0)),
    }


def paired_bootstrap_delta(labels, first_scores, second_scores, *, n_resamples: int = 1000, seed: int = 2027):
    labels = np.asarray(labels, dtype=int)
    first_scores = np.asarray(first_scores, dtype=float)
    second_scores = np.asarray(second_scores, dtype=float)
    if not (labels.shape == first_scores.shape == second_scores.shape):
        raise ValueError("labels and scores must have matching shapes")
    rng = np.random.default_rng(seed)
    indices = np.arange(labels.size)
    deltas = []
    for _ in range(n_resamples):
        sample = rng.choice(indices, size=indices.size, replace=True)
        if len(np.unique(labels[sample])) < 2:
            continue
        deltas.append(100.0 * (eer(labels[sample], second_scores[sample]) - eer(labels[sample], first_scores[sample])))
    if not deltas:
        raise ValueError("bootstrap could not draw a two-class sample")
    arr = np.asarray(deltas, dtype=float)
    return {
        "delta_eer_percent": float(100.0 * (eer(labels, second_scores) - eer(labels, first_scores))),
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
        "n_resamples": int(arr.size),
    }
