from __future__ import annotations

import numpy as np
import pandas as pd

from .evidence import (
    build_evidence_cards,
    family_metrics,
    leave_family_out_calibration,
    matched_evidence_metrics,
)
from .inputs import (
    validate_attack_scores,
    validate_evidence_scores,
    validate_stress_check_scores,
)
from .metrics import metric_row, operating_threshold, paired_bootstrap_delta

DEFAULT_STABILITY_SEEDS = tuple(range(20260821, 20260831))
CALIBRATION_MODES = (
    ("base_streams", "calibrated_base_streams"),
    ("passive_margin", "calibrated_passive_margin"),
    ("passive_squared", "calibrated_passive_squared"),
    ("retrieval_profile", "calibrated_retrieval_profile"),
    ("passive_retrieval_profile", "calibrated_passive_retrieval_profile"),
    ("passive_retrieval_shape", "calibrated_passive_retrieval_shape"),
    ("passive_watermark_conflict", "calibrated_passive_watermark_conflict"),
    ("card_retrieval_conflict", "calibrated_card_retrieval_conflict"),
    ("nonlinear_control", "calibrated_nonlinear_control"),
    ("squared_gap_control", "calibrated_squared_gap_control"),
    ("evidence_card", "calibrated_evidence_card"),
)
STABILITY_MODES = (
    ("base_streams", "Score fusion"),
    ("evidence_card", "Evidence card (abs. gaps)"),
    ("squared_gap_control", "Evidence card (sq. gaps)"),
)


def run_matched_evidence(scores: pd.DataFrame, out_dir) -> dict[str, str]:
    validate_evidence_scores(scores).raise_for_issues()
    out_dir.mkdir(parents=True, exist_ok=True)
    cards = build_evidence_cards(scores)
    metrics = matched_evidence_metrics(scores)
    families = family_metrics(scores)
    baseline = cards["passive_prob"].to_numpy()
    evidence_scores = cards["card_retrieval_score"].to_numpy()
    delta = paired_bootstrap_delta(cards["label"], baseline, evidence_scores, n_resamples=1000)
    deltas = pd.DataFrame([{**delta, "comparison": "card_retrieval_minus_passive"}])
    paths = {
        "evidence_cards": out_dir / "evidence_cards.csv",
        "matched_evidence_metrics": out_dir / "matched_evidence_metrics.csv",
        "family_eer": out_dir / "family_eer.csv",
        "paired_bootstrap_deltas": out_dir / "paired_bootstrap_deltas.csv",
    }
    cards.to_csv(paths["evidence_cards"], index=False)
    metrics.to_csv(paths["matched_evidence_metrics"], index=False)
    families.to_csv(paths["family_eer"], index=False)
    deltas.to_csv(paths["paired_bootstrap_deltas"], index=False)
    return {key: str(path) for key, path in paths.items()}


def run_calibration(scores: pd.DataFrame, out_dir) -> dict[str, str]:
    validate_evidence_scores(scores).raise_for_issues()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for mode, display_name in CALIBRATION_MODES:
        calibrated = leave_family_out_calibration(scores, mode=mode)
        row = metric_row(
            display_name,
            calibrated["label"],
            calibrated["calibrated_score"],
        )
        row["calibration_mode"] = mode
        rows.append(row)
    table = pd.DataFrame(rows)
    path = out_dir / "calibration_ablation.csv"
    table.to_csv(path, index=False)
    return {"calibration_ablation": str(path)}


def run_calibration_split_stability(
    scores: pd.DataFrame,
    out_dir,
    *,
    seeds=DEFAULT_STABILITY_SEEDS,
) -> dict[str, str]:
    validate_evidence_scores(scores).raise_for_issues()
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_values = tuple(int(seed) for seed in seeds)
    if not seed_values:
        raise ValueError("at least one fold seed is required")

    runs = []
    for mode, display_name in STABILITY_MODES:
        for fold_seed in seed_values:
            calibrated = leave_family_out_calibration(scores, mode=mode, fold_seed=fold_seed)
            overall = metric_row(display_name, calibrated["label"], calibrated["calibrated_score"])
            fold_eers = []
            for _, group in calibrated.groupby("calibration_fold", sort=True):
                if group["label"].nunique() == 2:
                    fold_eers.append(
                        metric_row(display_name, group["label"], group["calibrated_score"])["eer_percent"]
                    )
            runs.append(
                {
                    "calibration_mode": mode,
                    "display_name": display_name,
                    "fold_seed": fold_seed,
                    "n": overall["n"],
                    "pooled_eer_percent": overall["eer_percent"],
                    "pooled_min_dcf": overall["min_dcf"],
                    "pooled_ece": overall["ece"],
                    "fold_eer_percent_mean": float(np.mean(fold_eers)),
                    "fold_eer_percent_min": float(np.min(fold_eers)),
                    "fold_eer_percent_max": float(np.max(fold_eers)),
                }
            )
    runs_table = pd.DataFrame(runs)
    summary_rows = []
    for mode, group in runs_table.groupby("calibration_mode", sort=False):
        summary_rows.append(
            {
                "calibration_mode": mode,
                "display_name": group["display_name"].iloc[0],
                "seed_count": int(group["fold_seed"].nunique()),
                "n": int(group["n"].iloc[0]),
                "pooled_eer_percent_mean": float(group["pooled_eer_percent"].mean()),
                "pooled_eer_percent_min": float(group["pooled_eer_percent"].min()),
                "pooled_eer_percent_max": float(group["pooled_eer_percent"].max()),
                "fold_eer_percent_mean": float(group["fold_eer_percent_mean"].mean()),
                "fold_eer_percent_min": float(group["fold_eer_percent_mean"].min()),
                "fold_eer_percent_max": float(group["fold_eer_percent_mean"].max()),
                "pooled_ece_mean": float(group["pooled_ece"].mean()),
                "pooled_ece_min": float(group["pooled_ece"].min()),
                "pooled_ece_max": float(group["pooled_ece"].max()),
            }
        )
    summary_table = pd.DataFrame(summary_rows)
    run_path = out_dir / "calibration_split_stability_runs.csv"
    summary_path = out_dir / "calibration_split_stability.csv"
    runs_table.to_csv(run_path, index=False)
    summary_table.to_csv(summary_path, index=False)
    return {
        "calibration_split_stability_runs": str(run_path),
        "calibration_split_stability": str(summary_path),
    }


def run_attack_stress(scores: pd.DataFrame, out_dir) -> dict[str, str]:
    validate_attack_scores(scores).raise_for_issues()
    rows = []
    for (model, family, variant), group in scores.groupby(["model", "attack_family", "attack_variant"], sort=True):
        if group["label"].nunique() < 2:
            continue
        row = metric_row(str(model), group["label"], group["score"])
        row.update({"model": model, "attack_family": family, "attack_variant": variant})
        rows.append(row)
    table = pd.DataFrame(rows)
    path = out_dir / "attack_stress.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return {"attack_stress": str(path)}


def run_stress_checks(scores: pd.DataFrame, out_dir) -> dict[str, str]:
    validate_stress_check_scores(scores).raise_for_issues()
    rows = []
    for (check, score_name), group in scores.groupby(["check", "score_name"], sort=True):
        row = metric_row(str(score_name), group["label"], group["score"])
        row.update(
            {
                "check": check,
                "score_name": score_name,
                "positive_count": int(group["label"].sum()),
                "negative_count": int((group["label"] == 0).sum()),
            }
        )
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(["check", "score_name"])
    path = out_dir / "stress_checks.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return {"stress_checks": str(path)}


def run_active_selection(scores: pd.DataFrame, out_dir, budgets=(100, 500, 1000, 2000), *, seed: int = 2027) -> dict[str, str]:
    validate_evidence_scores(scores).raise_for_issues()
    cards = build_evidence_cards(scores)
    rng = np.random.default_rng(seed)
    rows = []
    orderings = {
        "disagreement": cards["stream_disagreement"].to_numpy(),
        "uncertainty": -cards["decision_margin"].to_numpy(),
        "watermark_low": -cards["watermark_prob"].to_numpy(),
        "random": rng.random(len(cards)),
    }
    labels = cards["label"].to_numpy(dtype=int)
    errors = (cards["decision"].to_numpy() != np.where(labels == 1, "bonafide", "spoof")).astype(float)
    for strategy, score in orderings.items():
        ranked = np.argsort(score)[::-1]
        for budget in budgets:
            chosen = ranked[: min(int(budget), len(ranked))]
            rows.append(
                {
                    "strategy": strategy,
                    "selected_samples": len(chosen),
                    "captured_error_rate": float(errors[chosen].mean()) if len(chosen) else 0.0,
                }
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "active_selection.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return {"active_selection": str(path)}


def run_evidence_card_triage(
    scores: pd.DataFrame,
    out_dir,
    budgets=(0.05, 0.10, 0.20),
    *,
    bootstrap_replicates: int = 1000,
    bootstrap_seed: int = 20260827,
) -> dict[str, str]:
    validate_evidence_scores(scores).raise_for_issues()
    cards = build_evidence_cards(scores)
    labels = cards["label"].to_numpy(dtype=int)
    card_scores = cards["card_retrieval_score"].to_numpy(dtype=float)
    card_threshold = operating_threshold(labels, card_scores)
    card_errors = (card_scores >= card_threshold) != (labels == 1)
    card_error_count = int(card_errors.sum())
    thresholds = {
        "passive_prob": operating_threshold(labels, cards["passive_prob"]),
        "retrieval_prob": operating_threshold(labels, cards["retrieval_prob"]),
        "card_retrieval_score": card_threshold,
        "card_profile_score": operating_threshold(labels, cards["card_profile_score"]),
    }
    orderings = {
        "Passive margin": ("passive score", -np.abs(cards["passive_prob"] - thresholds["passive_prob"])),
        "Retrieval margin": ("retrieval score", -np.abs(cards["retrieval_prob"] - thresholds["retrieval_prob"])),
        "card margin": ("card score", -np.abs(cards["card_retrieval_score"] - card_threshold)),
        "Profile-card margin": ("card + profile score", -np.abs(cards["card_profile_score"] - thresholds["card_profile_score"])),
        "Stream disagreement": ("all evidence scores", cards["stream_disagreement"]),
    }
    if "nearest_distance" in cards.columns:
        orderings["Retrieval distance"] = ("nearest neighbor distance", cards["nearest_distance"].astype(float))

    rows = []
    row_count = len(cards)
    groups = _group_labels(cards)
    for rule, (fields, values) in orderings.items():
        value_array = np.asarray(values, dtype=float)
        ranked = np.argsort(value_array)[::-1]
        intervals = _bootstrap_triage_intervals(
            values=value_array,
            card_errors=card_errors,
            groups=groups,
            budgets=budgets,
            n_resamples=bootstrap_replicates,
            seed=bootstrap_seed,
        )
        for budget in budgets:
            reviewed_count = round(float(budget) * row_count)
            selected = ranked[:reviewed_count]
            errors_captured = int(card_errors[selected].sum())
            interval = intervals[float(budget)]
            rows.append(
                {
                    "rule": rule,
                    "fields": fields,
                    "budget_fraction": float(budget),
                    "reviewed_count": reviewed_count,
                    "errors_captured": errors_captured,
                    "error_capture_rate": float(errors_captured / card_error_count) if card_error_count else 0.0,
                    "error_capture_rate_ci_low": interval["error_capture_rate_ci_low"],
                    "error_capture_rate_ci_high": interval["error_capture_rate_ci_high"],
                    "review_precision": float(errors_captured / reviewed_count) if reviewed_count else 0.0,
                    "review_precision_ci_low": interval["review_precision_ci_low"],
                    "review_precision_ci_high": interval["review_precision_ci_high"],
                }
            )
    random_precision_ci = _bootstrap_random_precision_interval(card_errors, groups, bootstrap_replicates, bootstrap_seed)
    for budget in budgets:
        reviewed_count = round(float(budget) * row_count)
        rows.append(
            {
                "rule": "Random order",
                "fields": "none",
                "budget_fraction": float(budget),
                "reviewed_count": reviewed_count,
                "errors_captured": float(budget) * card_error_count,
                "error_capture_rate": float(budget),
                "error_capture_rate_ci_low": float(budget),
                "error_capture_rate_ci_high": float(budget),
                "review_precision": float(card_error_count / row_count) if row_count else 0.0,
                "review_precision_ci_low": random_precision_ci[0],
                "review_precision_ci_high": random_precision_ci[1],
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "evidence_card_triage.csv"
    selective_path = out_dir / "selective_risk.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    _selective_risk_table(scores, cards, budgets=budgets).to_csv(selective_path, index=False)
    return {"evidence_card_triage": str(path), "selective_risk": str(selective_path)}


def _selective_risk_table(scores: pd.DataFrame, cards: pd.DataFrame, *, budgets: tuple[float, ...]) -> pd.DataFrame:
    labels = cards["label"].to_numpy(dtype=int)
    calibrated = leave_family_out_calibration(scores, mode="evidence_card")
    systems = [
        ("Passive", "fixed", cards["passive_prob"].to_numpy(dtype=float)),
        ("Retrieval kNN", "training-free", cards["retrieval_prob"].to_numpy(dtype=float)),
        ("Card + retrieval", "fixed", cards["card_retrieval_score"].to_numpy(dtype=float)),
        ("Card + profile", "fixed", cards["card_profile_score"].to_numpy(dtype=float)),
        ("cross-fit evidence card", "learned", calibrated["calibrated_score"].to_numpy(dtype=float)),
    ]
    rows = []
    for system, fit, values in systems:
        threshold = operating_threshold(labels, values)
        errors = (values >= threshold) != (labels == 1)
        confidence = _decision_threshold_rank_margin(values, threshold)
        error_count = int(errors.sum())
        review_order = np.argsort(confidence, kind="mergesort")
        aurc = _risk_coverage_auc(errors, confidence)
        for budget in budgets:
            reviewed_count = round(float(budget) * len(labels))
            selected = review_order[:reviewed_count]
            retained = review_order[reviewed_count:]
            errors_captured = int(errors[selected].sum()) if reviewed_count else 0
            rows.append(
                {
                    "system": system,
                    "fit": fit,
                    "budget_fraction": float(budget),
                    "row_count": len(labels),
                    "reviewed_count": int(reviewed_count),
                    "decision_error_count": error_count,
                    "decision_error_rate": float(error_count / len(labels)) if len(labels) else 0.0,
                    "confidence_method": "decision_threshold_rank_margin",
                    "errors_captured": errors_captured,
                    "error_capture_rate": float(errors_captured / error_count) if error_count else 0.0,
                    "residual_error_rate": float(errors[retained].mean()) if retained.size else 0.0,
                    "review_precision": float(errors_captured / reviewed_count) if reviewed_count else 0.0,
                    "risk_coverage_auc": aurc,
                }
            )
    return pd.DataFrame(rows)


def _risk_coverage_auc(errors: np.ndarray, confidence: np.ndarray) -> float:
    if errors.size == 0:
        return 0.0
    retained_order = np.argsort(-confidence.astype(float), kind="mergesort")
    sorted_confidence = confidence[retained_order].astype(float)
    sorted_errors = errors[retained_order].astype(float)
    cumulative_errors = np.cumsum(sorted_errors)
    _, starts, counts = np.unique(sorted_confidence, return_index=True, return_counts=True)
    expected_cumulative = cumulative_errors.copy()
    tied = counts > 1
    for start, count in zip(starts[tied], counts[tied], strict=True):
        tie_size = int(count)
        stop = int(start + tie_size)
        retained_errors = float(cumulative_errors[start - 1]) if start > 0 else 0.0
        tie_errors = float(sorted_errors[start:stop].sum())
        offsets = np.arange(1, tie_size + 1, dtype=float)
        expected_cumulative[start:stop] = retained_errors + (tie_errors / tie_size) * offsets
    retained_counts = np.arange(1, errors.size + 1, dtype=float)
    return float(np.mean(expected_cumulative / retained_counts))


def _decision_threshold_rank_margin(scores: np.ndarray, threshold: float) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if values.size == 0:
        return values.copy()
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=float)
    _, starts, counts = np.unique(sorted_values, return_index=True, return_counts=True)
    for start, count in zip(starts, counts, strict=True):
        stop = int(start + count)
        ranks[order[start:stop]] = (float(start) + float(stop - 1)) / 2.0

    left = int(np.searchsorted(sorted_values, float(threshold), side="left"))
    right = int(np.searchsorted(sorted_values, float(threshold), side="right"))
    if right > left:
        threshold_rank = (float(left) + float(right - 1)) / 2.0
    else:
        threshold_rank = min(max(float(left) - 0.5, 0.0), float(values.size - 1))
    return np.abs(ranks - threshold_rank) / max(float(values.size - 1), 1.0)


def _group_labels(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in ("speaker_id", "source", "family") if column in frame.columns]
    if columns:
        values = frame[columns].fillna("unknown")
        return ["|".join(str(value) for value in row) for row in values.to_numpy()]
    if "sample_id" in frame.columns:
        return [str(value) for value in frame["sample_id"].fillna("unknown").tolist()]
    return [str(index) for index in range(len(frame))]


def _bootstrap_triage_intervals(
    *,
    values: np.ndarray,
    card_errors: np.ndarray,
    groups: list[str],
    budgets: tuple[float, ...],
    n_resamples: int,
    seed: int,
) -> dict[float, dict[str, float]]:
    fallback = {
        float(budget): {
            "error_capture_rate_ci_low": 0.0,
            "error_capture_rate_ci_high": 0.0,
            "review_precision_ci_low": 0.0,
            "review_precision_ci_high": 0.0,
        }
        for budget in budgets
    }
    grouped_indices = _grouped_indices(groups)
    if n_resamples <= 0 or len(grouped_indices) == 0:
        return fallback
    rng = np.random.default_rng(seed)
    capture_samples = {float(budget): [] for budget in budgets}
    precision_samples = {float(budget): [] for budget in budgets}
    group_count = len(grouped_indices)
    for _ in range(n_resamples):
        sampled_groups = rng.integers(0, group_count, size=group_count)
        sampled_indices = np.concatenate([grouped_indices[group_index] for group_index in sampled_groups])
        sample_error_count = int(card_errors[sampled_indices].sum())
        if sample_error_count == 0:
            continue
        ranked = sampled_indices[np.argsort(-values[sampled_indices], kind="mergesort")]
        for budget in budgets:
            budget_value = float(budget)
            reviewed_count = round(budget_value * sampled_indices.size)
            if reviewed_count <= 0:
                continue
            selected = ranked[:reviewed_count]
            captured = int(card_errors[selected].sum())
            capture_samples[budget_value].append(float(captured / sample_error_count))
            precision_samples[budget_value].append(float(captured / reviewed_count))
    return {
        float(budget): _interval_payload(capture_samples[float(budget)], precision_samples[float(budget)])
        for budget in budgets
    }


def _bootstrap_random_precision_interval(
    card_errors: np.ndarray,
    groups: list[str],
    n_resamples: int,
    seed: int,
) -> tuple[float, float]:
    grouped_indices = _grouped_indices(groups)
    if n_resamples <= 0 or len(grouped_indices) == 0:
        value = float(card_errors.mean()) if len(card_errors) else 0.0
        return value, value
    rng = np.random.default_rng(seed + 100_000)
    samples = []
    group_count = len(grouped_indices)
    for _ in range(n_resamples):
        sampled_groups = rng.integers(0, group_count, size=group_count)
        sampled_indices = np.concatenate([grouped_indices[group_index] for group_index in sampled_groups])
        if sampled_indices.size:
            samples.append(float(card_errors[sampled_indices].mean()))
    if not samples:
        value = float(card_errors.mean()) if len(card_errors) else 0.0
        return value, value
    low, high = np.quantile(np.asarray(samples, dtype=float), [0.025, 0.975])
    return float(low), float(high)


def _grouped_indices(groups: list[str]) -> list[np.ndarray]:
    members: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        members.setdefault(group, []).append(index)
    return [np.asarray(indices, dtype=int) for indices in members.values()]


def _interval_payload(capture_samples: list[float], precision_samples: list[float]) -> dict[str, float]:
    if not capture_samples or not precision_samples:
        return {
            "error_capture_rate_ci_low": 0.0,
            "error_capture_rate_ci_high": 0.0,
            "review_precision_ci_low": 0.0,
            "review_precision_ci_high": 0.0,
        }
    capture_low, capture_high = np.quantile(np.asarray(capture_samples, dtype=float), [0.025, 0.975])
    precision_low, precision_high = np.quantile(np.asarray(precision_samples, dtype=float), [0.025, 0.975])
    return {
        "error_capture_rate_ci_low": float(capture_low),
        "error_capture_rate_ci_high": float(capture_high),
        "review_precision_ci_low": float(precision_low),
        "review_precision_ci_high": float(precision_high),
    }


def make_demo_scores(n: int = 2400, *, seed: int = 2027) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    families = np.array([f"A{i:02d}" for i in range(9, 17)])
    family = rng.choice(families, size=n)
    label = rng.binomial(1, 0.72, size=n)
    family_shift = {fam: shift for fam, shift in zip(families, np.linspace(-0.55, 0.55, len(families)))}
    latent = label * 1.4 - 0.7 + np.array([family_shift[x] for x in family])
    passive = latent + rng.normal(0, 0.85, size=n)
    watermark = 0.35 * latent + rng.normal(0, 1.1, size=n)
    retrieval = 0.9 * latent + rng.normal(0, 0.65, size=n)
    profile = 0.55 * latent + rng.normal(0, 0.75, size=n)
    evidence = pd.DataFrame(
        {
            "sample_id": [f"demo-{i:05d}" for i in range(n)],
            "label": label,
            "family": family,
            "passive_score": passive,
            "watermark_score": watermark,
            "retrieval_score": retrieval,
            "profile_margin": profile,
        }
    )
    attack_rows = []
    attack_families = ["noise", "low-pass", "high-pass", "speed", "resample", "quantize"]
    variants = ["v1", "v2", "v3", "v4"]
    for model_id, model in enumerate(["HuBERT", "Wav2Vec2", "WavLM"]):
        for fam_id, fam in enumerate(attack_families):
            for var_id, variant in enumerate(variants):
                subset_size = min(350, len(evidence))
                subset = evidence.sample(n=subset_size, replace=False, random_state=seed + model_id * 100 + fam_id * 10 + var_id)
                degradation = 0.12 * fam_id + 0.05 * var_id + 0.06 * model_id
                score = subset["passive_score"].to_numpy() - degradation + rng.normal(0, 0.3, len(subset))
                for sample_id, lab, sc in zip(subset["sample_id"], subset["label"], score):
                    attack_rows.append(
                        {
                            "sample_id": sample_id,
                            "label": int(lab),
                            "model": model,
                            "attack_family": fam,
                            "attack_variant": variant,
                            "score": float(sc),
                        }
                    )
    stress_rows = []
    stress_specs = [
        ("WavLM check", "Passive WavLM-large", 0.85),
        ("WavLM check", "Retrieval kNN", 1.30),
        ("WavLM check", "Card + retrieval", 1.45),
        ("Self-VC", "Watermark probe", 0.05),
        ("Self-VC", "Passive CNN", 0.10),
    ]
    for check, score_name, strength in stress_specs:
        subset_size = min(600, len(evidence))
        subset = evidence.sample(n=subset_size, replace=False, random_state=seed + len(stress_rows) + int(100 * strength))
        centered_label = 2 * subset["label"].to_numpy() - 1
        score = strength * centered_label + rng.normal(0, 1.0, len(subset))
        for sample_id, lab, sc in zip(subset["sample_id"], subset["label"], score):
            stress_rows.append(
                {
                    "check": check,
                    "score_name": score_name,
                    "sample_id": sample_id,
                    "label": int(lab),
                    "score": float(sc),
                }
            )
    return evidence, pd.DataFrame(attack_rows), pd.DataFrame(stress_rows)
