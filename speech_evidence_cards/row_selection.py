from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import eer


DEFAULT_SCORE_COLUMN = "retrieval_score"


def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def _label_mask(frame: pd.DataFrame, label_value: int) -> pd.Series:
    return pd.to_numeric(frame["label"], errors="coerce").astype("Int64").eq(label_value)


def _ks_statistic(left: np.ndarray, right: np.ndarray) -> float:
    left = np.sort(np.asarray(left, dtype=float))
    right = np.sort(np.asarray(right, dtype=float))
    if left.size == 0 or right.size == 0:
        return float("nan")
    values = np.sort(np.concatenate([left, right]))
    left_cdf = np.searchsorted(left, values, side="right") / left.size
    right_cdf = np.searchsorted(right, values, side="right") / right.size
    return float(np.max(np.abs(left_cdf - right_cdf)))


def _sample_eer(
    rng: np.random.Generator,
    bonafide_scores: np.ndarray,
    spoof_scores: np.ndarray,
    bonafide_count: int,
    spoof_count: int,
) -> float:
    sampled_bonafide = rng.choice(bonafide_scores, size=bonafide_count, replace=False)
    sampled_spoof = rng.choice(spoof_scores, size=spoof_count, replace=False)
    labels = np.concatenate([np.ones(sampled_bonafide.size, dtype=int), np.zeros(sampled_spoof.size, dtype=int)])
    scores = np.concatenate([sampled_bonafide, sampled_spoof])
    return eer(labels, scores)


def _percentile_interval(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    return float(np.mean(array)), float(np.percentile(array, 2.5)), float(np.percentile(array, 97.5))


def _mean_or_nan(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else float("nan")


def row_selection_table(
    full_scores: pd.DataFrame,
    matched_scores: pd.DataFrame,
    *,
    score_column: str = DEFAULT_SCORE_COLUMN,
    n_bootstrap: int = 2000,
    seed: int = 20260830,
) -> pd.DataFrame:
    _require_columns(full_scores, {"sample_id", "label", "family", score_column}, "full_scores")
    _require_columns(matched_scores, {"sample_id", "label", "family", score_column}, "matched_scores")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")

    full = full_scores.copy()
    matched = matched_scores.copy()
    full[score_column] = full[score_column].astype(float)
    matched[score_column] = matched[score_column].astype(float)
    matched_ids = set(matched["sample_id"].astype(str))
    missing_ids = sorted(matched_ids - set(full["sample_id"].astype(str)))
    if missing_ids:
        raise ValueError(f"matched scores contain {len(missing_ids)} ids absent from full scores")

    full["__matched"] = full["sample_id"].astype(str).isin(matched_ids)
    full_bonafide_scores = full.loc[_label_mask(full, 1), score_column].to_numpy(dtype=float)
    matched_bonafide_count = int(_label_mask(matched, 1).sum())
    if matched_bonafide_count == 0 or matched_bonafide_count > full_bonafide_scores.size:
        raise ValueError("matched bona fide count is incompatible with the full score file")

    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []
    for family in sorted(full.loc[_label_mask(full, 0), "family"].astype(str).unique()):
        full_spoof = full[_label_mask(full, 0) & full["family"].astype(str).eq(family)]
        matched_spoof = matched[_label_mask(matched, 0) & matched["family"].astype(str).eq(family)]
        matched_spoof_count = int(len(matched_spoof))
        if matched_spoof_count == 0:
            continue
        full_spoof_scores = full_spoof[score_column].to_numpy(dtype=float)
        if matched_spoof_count > full_spoof_scores.size:
            raise ValueError(f"matched spoof count exceeds full count for {family}")

        bootstrap_eers = [
            _sample_eer(rng, full_bonafide_scores, full_spoof_scores, matched_bonafide_count, matched_spoof_count)
            for _ in range(n_bootstrap)
        ]
        random_mean, random_low, random_high = _percentile_interval(bootstrap_eers)
        full_family_rows = full[_label_mask(full, 1) | (_label_mask(full, 0) & full["family"].astype(str).eq(family))]
        matched_family_rows = matched[
            _label_mask(matched, 1) | (_label_mask(matched, 0) & matched["family"].astype(str).eq(family))
        ]
        retained_scores = full_spoof.loc[full_spoof["__matched"], score_column].to_numpy(dtype=float)
        omitted_scores = full_spoof.loc[~full_spoof["__matched"], score_column].to_numpy(dtype=float)
        full_eer = eer(full_family_rows["label"].astype(int), full_family_rows[score_column].astype(float))
        matched_eer = eer(matched_family_rows["label"].astype(int), matched_family_rows[score_column].astype(float))
        retained_mean = _mean_or_nan(retained_scores)
        omitted_mean = _mean_or_nan(omitted_scores)
        rows.append(
            {
                "family": family,
                "full_spoof": int(len(full_spoof)),
                "matched_spoof": matched_spoof_count,
                "matched_bonafide": matched_bonafide_count,
                "retained_percent": 100.0 * matched_spoof_count / max(len(full_spoof), 1),
                "full_eer_percent": 100.0 * full_eer,
                "matched_eer_percent": 100.0 * matched_eer,
                "random_mean_eer_percent": 100.0 * random_mean,
                "random_ci_low_eer_percent": 100.0 * random_low,
                "random_ci_high_eer_percent": 100.0 * random_high,
                "matched_minus_random_mean_pp": 100.0 * (matched_eer - random_mean),
                "bootstrap_percentile": float(np.mean(np.asarray(bootstrap_eers) <= matched_eer)),
                "retained_minus_omitted_spoof_score_pp": 100.0 * (retained_mean - omitted_mean),
                "spoof_score_ks": _ks_statistic(retained_scores, omitted_scores),
                "bootstrap_trials": n_bootstrap,
            }
        )
    return pd.DataFrame(rows)


def score_retention_table(
    full_scores: pd.DataFrame,
    matched_scores: pd.DataFrame,
    *,
    score_column: str = DEFAULT_SCORE_COLUMN,
) -> pd.DataFrame:
    _require_columns(full_scores, {"sample_id", "label", score_column}, "full_scores")
    _require_columns(matched_scores, {"sample_id"}, "matched_scores")
    full = full_scores.copy()
    full[score_column] = full[score_column].astype(float)
    matched_ids = set(matched_scores["sample_id"].astype(str))
    missing_ids = sorted(matched_ids - set(full["sample_id"].astype(str)))
    if missing_ids:
        raise ValueError(f"matched scores contain {len(missing_ids)} ids absent from full scores")
    full["__matched"] = full["sample_id"].astype(str).isin(matched_ids)
    groups = {
        "all": pd.Series(True, index=full.index),
        "bonafide": _label_mask(full, 1),
        "spoof": _label_mask(full, 0),
    }
    rows = []
    for group, mask in groups.items():
        retained = full.loc[mask & full["__matched"], score_column].to_numpy(dtype=float)
        omitted = full.loc[mask & ~full["__matched"], score_column].to_numpy(dtype=float)
        retained_mean = _mean_or_nan(retained)
        omitted_mean = _mean_or_nan(omitted)
        rows.append(
            {
                "group": group,
                "retained_count": int(retained.size),
                "omitted_count": int(omitted.size),
                "retained_mean": retained_mean,
                "omitted_mean": omitted_mean,
                "retained_minus_omitted_score_pp": 100.0 * (retained_mean - omitted_mean),
                "score_ks": _ks_statistic(retained, omitted),
            }
        )
    return pd.DataFrame(rows)


def run_row_selection_check(
    full_scores: pd.DataFrame,
    matched_scores: pd.DataFrame,
    out_dir: Path,
    *,
    score_column: str = DEFAULT_SCORE_COLUMN,
    n_bootstrap: int = 2000,
    seed: int = 20260830,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    family_rows = row_selection_table(
        full_scores,
        matched_scores,
        score_column=score_column,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    if family_rows.empty:
        raise ValueError("row-selection check produced no family rows")
    retention_rows = score_retention_table(full_scores, matched_scores, score_column=score_column)
    family_path = out_dir / "retrieval_selection_check.csv"
    retention_path = out_dir / "retrieval_score_retention.csv"
    audit_path = out_dir / "retrieval_selection_check.json"
    family_rows.to_csv(family_path, index=False)
    retention_rows.to_csv(retention_path, index=False)
    outside = family_rows[
        (family_rows["matched_eer_percent"] < family_rows["random_ci_low_eer_percent"])
        | (family_rows["matched_eer_percent"] > family_rows["random_ci_high_eer_percent"])
    ]["family"].astype(str).tolist()
    spoof_retention = retention_rows[retention_rows["group"] == "spoof"].iloc[0].to_dict()
    audit = {
        "ready": True,
        "family_count": int(len(family_rows)),
        "bootstrap_trials": int(n_bootstrap),
        "families_outside_bootstrap_interval": outside,
        "max_abs_matched_minus_random_mean_pp": float(family_rows["matched_minus_random_mean_pp"].abs().max()),
        "overall_spoof_retained_count": int(spoof_retention["retained_count"]),
        "overall_spoof_omitted_count": int(spoof_retention["omitted_count"]),
        "overall_spoof_retained_minus_omitted_score_pp": float(
            spoof_retention["retained_minus_omitted_score_pp"]
        ),
        "overall_spoof_score_ks": float(spoof_retention["score_ks"]),
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return {
        "retrieval_selection_check": str(family_path),
        "retrieval_score_retention": str(retention_path),
        "retrieval_selection_audit": str(audit_path),
    }
