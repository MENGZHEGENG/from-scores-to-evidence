from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class RetrievalAudit:
    ready: bool
    row_count: int
    spoof_count: int
    issues: tuple[str, ...]
    findings: tuple[str, ...]
    limitations: tuple[str, ...]
    same_heldout_topk_family_count: int | None
    same_topk_sample_count: int | None
    same_topk_speaker_count: int | None
    same_topk_source_path_count: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "row_count": self.row_count,
            "spoof_count": self.spoof_count,
            "issues": list(self.issues),
            "findings": list(self.findings),
            "limitations": list(self.limitations),
            "same_heldout_topk_family_count": self.same_heldout_topk_family_count,
            "same_topk_sample_count": self.same_topk_sample_count,
            "same_topk_speaker_count": self.same_topk_speaker_count,
            "same_topk_source_path_count": self.same_topk_source_path_count,
        }


OPTIONAL_ID_COLUMNS = ("speaker_id", "source_path", "source")
POSITIVE_LABELS = {"1", "true", "bonafide", "bona-fide", "real", "genuine", "target"}


def feature_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in frame.columns if column.startswith("feat_")]
    if not columns:
        raise ValueError("expected at least one feature column named feat_*")
    return columns


def retrieval_label_scores(support: pd.DataFrame, query: pd.DataFrame, *, k: int = 10) -> pd.DataFrame:
    result = retrieval_profile_scores(support, query, k=k)
    return result[["sample_id", "retrieval_score", "mean_neighbor_distance"]]


def retrieval_profile_scores(
    support: pd.DataFrame,
    query: pd.DataFrame,
    *,
    k: int = 10,
    id_col: str = "sample_id",
    label_col: str = "label",
    family_col: str = "family",
) -> pd.DataFrame:
    for frame, name in [(support, "support"), (query, "query")]:
        missing = {id_col, label_col, family_col} - set(frame.columns)
        if missing:
            raise ValueError(f"{name} frame missing columns: {sorted(missing)}")
    if k < 1:
        raise ValueError("k must be positive")
    cols = feature_columns(support)
    missing_features = [column for column in cols if column not in query.columns]
    if missing_features:
        raise ValueError(f"query frame missing feature columns: {missing_features}")
    scaler = StandardScaler()
    support_x = scaler.fit_transform(support[cols].to_numpy(dtype=float))
    query_x = scaler.transform(query[cols].to_numpy(dtype=float))
    support_labels = support[label_col].map(_is_positive).astype(int).to_numpy()
    query_labels = query[label_col].map(_is_positive).astype(int).to_numpy()
    support_families = support[family_col].astype(str).to_numpy()
    query_families = query[family_col].astype(str).to_numpy()
    output_rows: list[dict[str, object]] = []
    for family in _query_groups(query_families, query_labels):
        query_indices = [
            index
            for index, (query_family, query_label) in enumerate(zip(query_families, query_labels, strict=True))
            if _held_out_family(query_family, query_label) == family
        ]
        support_indices = np.asarray(
            [
                index
                for index, (support_family, support_label) in enumerate(zip(support_families, support_labels, strict=True))
                if support_label == 1 or support_family != family
            ],
            dtype=int,
        )
        n_neighbors = min(k, len(support_indices))
        if n_neighbors < 1:
            raise ValueError(f"no support rows remain for held-out family {family}")
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
        nn.fit(support_x[support_indices])
        distances, local_indices = nn.kneighbors(query_x[query_indices])
        for row_position, query_index in enumerate(query_indices):
            global_indices = support_indices[local_indices[row_position]]
            neighbor_labels = support_labels[global_indices]
            weights = 1.0 / np.maximum(distances[row_position], 1e-6)
            score = float((weights * neighbor_labels).sum() / weights.sum())
            nearest = support.iloc[int(global_indices[0])]
            neighbor_rows = [support.iloc[int(index)] for index in global_indices]
            query_row = query.iloc[query_index]
            output_rows.append(
                {
                    "sample_id": query_row[id_col],
                    "label": int(query_labels[query_index]),
                    "family": query_row[family_col],
                    "held_out_family": family,
                    "retrieval_score": score,
                    "mean_neighbor_distance": float(np.mean(distances[row_position])),
                    "nearest_distance": float(distances[row_position][0]),
                    "nearest_sample_id": nearest[id_col],
                    "nearest_family": nearest[family_col],
                    "neighbor_sample_ids": _join(neighbor[id_col] for neighbor in neighbor_rows),
                    "neighbor_families": _join(neighbor[family_col] for neighbor in neighbor_rows),
                    "neighbor_distances": _join(f"{distance:.6g}" for distance in distances[row_position]),
                    **_optional_neighbor_fields(query_row, nearest, neighbor_rows),
                }
            )
    return pd.DataFrame(output_rows).sort_values("sample_id").reset_index(drop=True)


def audit_retrieval_neighbors(
    predictions: pd.DataFrame,
    *,
    id_col: str = "sample_id",
    label_col: str = "label",
    family_col: str = "family",
) -> RetrievalAudit:
    issues: list[str] = []
    findings: list[str] = []
    limitations: list[str] = []
    required = {
        id_col,
        label_col,
        family_col,
        "held_out_family",
        "nearest_family",
        "neighbor_sample_ids",
        "neighbor_families",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        return RetrievalAudit(False, len(predictions), 0, (f"missing columns: {missing}",), (), (), None, None, None, None)
    labels = predictions[label_col].map(_is_positive).astype(bool)
    spoof = ~labels
    heldout = predictions["held_out_family"].astype(str)
    same_family = _list_contains(predictions["neighbor_families"], heldout) & spoof
    same_sample = _list_contains(predictions["neighbor_sample_ids"], predictions[id_col].astype(str))
    if int(same_family.sum()):
        issues.append(f"{int(same_family.sum())} spoof rows include a same-family top-k neighbor")
    else:
        findings.append("No spoof row includes a top-k neighbor from its held-out family.")
    if int(same_sample.sum()):
        issues.append(f"{int(same_sample.sum())} rows include the query sample among top-k neighbors")
    else:
        findings.append("No row includes the query sample among top-k neighbors.")
    same_speaker = None
    if {"query_speaker_id", "neighbor_speaker_ids"}.issubset(predictions.columns):
        same_speaker = _list_contains(predictions["neighbor_speaker_ids"], predictions["query_speaker_id"].astype(str))
        if int(same_speaker.sum()):
            limitations.append(f"{int(same_speaker.sum())} rows share a speaker identifier with a top-k neighbor")
        else:
            findings.append("No row shares a speaker identifier with a top-k neighbor.")
    else:
        limitations.append("Speaker identifiers are absent; speaker overlap is not audited.")
    same_source_path = None
    if {"query_source_path", "neighbor_source_paths"}.issubset(predictions.columns):
        same_source_path = _list_contains(predictions["neighbor_source_paths"], predictions["query_source_path"].astype(str))
        if int(same_source_path.sum()):
            issues.append(f"{int(same_source_path.sum())} rows share an audio path with a top-k neighbor")
        else:
            findings.append("No row shares an audio path with a top-k neighbor.")
    else:
        limitations.append("Audio-path identifiers are absent; path overlap is not audited.")
    return RetrievalAudit(
        ready=not issues,
        row_count=len(predictions),
        spoof_count=int(spoof.sum()),
        issues=tuple(issues),
        findings=tuple(findings),
        limitations=tuple(limitations),
        same_heldout_topk_family_count=int(same_family.sum()),
        same_topk_sample_count=int(same_sample.sum()),
        same_topk_speaker_count=int(same_speaker.sum()) if same_speaker is not None else None,
        same_topk_source_path_count=int(same_source_path.sum()) if same_source_path is not None else None,
    )


def _optional_neighbor_fields(query_row: pd.Series, nearest: pd.Series, neighbor_rows: list[pd.Series]) -> dict[str, object]:
    fields: dict[str, object] = {}
    for column in OPTIONAL_ID_COLUMNS:
        if column not in query_row.index or column not in nearest.index:
            continue
        stem = "source_path" if column == "source_path" else column
        fields[f"query_{stem}"] = query_row[column]
        fields[f"nearest_{stem}"] = nearest[column]
        fields[f"neighbor_{stem}s"] = _join(neighbor[column] for neighbor in neighbor_rows)
    return fields


def _query_groups(families: np.ndarray, labels: np.ndarray) -> list[str]:
    groups = {_held_out_family(family, label) for family, label in zip(families, labels, strict=True)}
    return sorted(groups)


def _held_out_family(family: object, label: int) -> str:
    return str(family) if int(label) == 0 else ""


def _is_positive(value: object) -> bool:
    return str(value).strip().lower() in POSITIVE_LABELS


def _join(values: Iterable[object]) -> str:
    return ";".join(str(value).strip() for value in values if str(value).strip())


def _list_contains(cells: pd.Series, targets: pd.Series) -> pd.Series:
    checks = [_contains_nonempty(cell, target) for cell, target in zip(cells, targets, strict=True)]
    return pd.Series(checks, index=cells.index, dtype=bool)


def _contains_nonempty(cell: object, target: object) -> bool:
    target_text = _clean_token(target)
    return bool(target_text) and target_text in set(_split_tokens(cell))


def _split_tokens(cell: object) -> list[str]:
    text = _clean_token(cell)
    if not text:
        return []
    return [token for token in (_clean_token(part) for part in text.split(";")) if token]


def _clean_token(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "<na>"} else text
