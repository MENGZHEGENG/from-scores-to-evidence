from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

EVIDENCE_SCORE_COLUMNS = (
    "sample_id",
    "label",
    "family",
    "passive_score",
    "watermark_score",
    "retrieval_score",
    "profile_margin",
)
EVIDENCE_NUMERIC_COLUMNS = ("label", "passive_score", "watermark_score", "retrieval_score", "profile_margin")
ATTACK_SCORE_COLUMNS = ("sample_id", "label", "model", "attack_family", "attack_variant", "score")
ATTACK_NUMERIC_COLUMNS = ("label", "score")
STRESS_CHECK_COLUMNS = ("check", "score_name", "sample_id", "label", "score")
STRESS_CHECK_NUMERIC_COLUMNS = ("label", "score")


@dataclass(frozen=True)
class ValidationReport:
    name: str
    row_count: int
    issues: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.issues

    def raise_for_issues(self) -> None:
        if self.issues:
            detail = "\n".join(f"- {issue}" for issue in self.issues)
            raise ValueError(f"{self.name} validation failed:\n{detail}")

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "row_count": self.row_count, "ready": self.ready, "issues": list(self.issues)}


def read_csv(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(path)


def _missing_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> list[str]:
    return [column for column in columns if column not in frame.columns]


def _check_nonempty_strings(frame: pd.DataFrame, columns: tuple[str, ...], issues: list[str]) -> None:
    for column in columns:
        if column not in frame.columns:
            continue
        values = frame[column].astype("string")
        if values.isna().any() or (values.str.strip() == "").any():
            issues.append(f"{column} must not contain blank values")


def _check_numeric(frame: pd.DataFrame, columns: tuple[str, ...], issues: list[str]) -> None:
    for column in columns:
        if column not in frame.columns:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any():
            issues.append(f"{column} must be numeric and non-missing")
        elif not np.isfinite(numeric.to_numpy(dtype=float)).all():
            issues.append(f"{column} must be finite")


def _check_binary_labels(frame: pd.DataFrame, issues: list[str]) -> None:
    if "label" not in frame.columns:
        return
    labels = pd.to_numeric(frame["label"], errors="coerce")
    values = set(labels.dropna().astype(int).tolist()) if not labels.isna().any() else set()
    if labels.isna().any() or not values.issubset({0, 1}):
        issues.append("label must contain only 0 for spoof/inconsistent and 1 for bonafide/consistent")
    elif values != {0, 1}:
        issues.append("label must contain both classes")


def validate_evidence_scores(frame: pd.DataFrame) -> ValidationReport:
    issues: list[str] = []
    missing = _missing_columns(frame, EVIDENCE_SCORE_COLUMNS)
    if missing:
        issues.append(f"missing columns: {missing}")
    if frame.empty:
        issues.append("file must contain at least one row")
    _check_nonempty_strings(frame, ("sample_id", "family"), issues)
    _check_numeric(frame, EVIDENCE_NUMERIC_COLUMNS, issues)
    _check_binary_labels(frame, issues)
    if "sample_id" in frame.columns and frame["sample_id"].duplicated().any():
        issues.append("sample_id values must be unique for evidence-score rows")
    if "family" in frame.columns and frame["family"].nunique(dropna=True) < 2:
        issues.append("family must contain at least two groups for leave-family-out calibration")
    return ValidationReport("evidence_scores", len(frame), tuple(issues))


def validate_attack_scores(frame: pd.DataFrame) -> ValidationReport:
    issues: list[str] = []
    missing = _missing_columns(frame, ATTACK_SCORE_COLUMNS)
    if missing:
        issues.append(f"missing columns: {missing}")
    if frame.empty:
        issues.append("file must contain at least one row")
    _check_nonempty_strings(frame, ("sample_id", "model", "attack_family", "attack_variant"), issues)
    _check_numeric(frame, ATTACK_NUMERIC_COLUMNS, issues)
    _check_binary_labels(frame, issues)
    key = ["sample_id", "model", "attack_family", "attack_variant"]
    if all(column in frame.columns for column in key) and frame.duplicated(key).any():
        issues.append("sample_id/model/attack_family/attack_variant rows must be unique")
    if all(column in frame.columns for column in ["model", "attack_family", "attack_variant", "label"]):
        labels = pd.to_numeric(frame["label"], errors="coerce")
        if not labels.isna().any():
            grouped = frame.assign(_label=labels.astype(int)).groupby(["model", "attack_family", "attack_variant"])
            one_class = ["/".join(map(str, key_values)) for key_values, group in grouped if group["_label"].nunique() < 2]
            if one_class:
                preview = one_class[:5]
                suffix = "" if len(one_class) <= 5 else f" and {len(one_class) - 5} more"
                issues.append(f"attack groups must contain both classes: {preview}{suffix}")
    return ValidationReport("ssl_attack_scores", len(frame), tuple(issues))


def validate_stress_check_scores(frame: pd.DataFrame) -> ValidationReport:
    issues: list[str] = []
    missing = _missing_columns(frame, STRESS_CHECK_COLUMNS)
    if missing:
        issues.append(f"missing columns: {missing}")
    if frame.empty:
        issues.append("file must contain at least one row")
    _check_nonempty_strings(frame, ("check", "score_name", "sample_id"), issues)
    _check_numeric(frame, STRESS_CHECK_NUMERIC_COLUMNS, issues)
    _check_binary_labels(frame, issues)
    key = ["check", "score_name", "sample_id"]
    if all(column in frame.columns for column in key) and frame.duplicated(key).any():
        issues.append("check/score_name/sample_id rows must be unique")
    if all(column in frame.columns for column in ["check", "score_name", "label"]):
        labels = pd.to_numeric(frame["label"], errors="coerce")
        if not labels.isna().any():
            grouped = frame.assign(_label=labels.astype(int)).groupby(["check", "score_name"])
            one_class = ["/".join(map(str, key_values)) for key_values, group in grouped if group["_label"].nunique() < 2]
            if one_class:
                preview = one_class[:5]
                suffix = "" if len(one_class) <= 5 else f" and {len(one_class) - 5} more"
                issues.append(f"check groups must contain both classes: {preview}{suffix}")
    return ValidationReport("stress_check_scores", len(frame), tuple(issues))
