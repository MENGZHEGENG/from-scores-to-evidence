from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_HASH_COLUMNS = (
    "sample_id",
    "utt_id",
    "query_utt_id",
    "nearest_utt_id",
    "neighbor_utt_id",
    "speaker_id",
    "query_speaker_id",
    "nearest_speaker_id",
    "neighbor_speaker_id",
    "target_speaker_id",
)
DEFAULT_LIST_HASH_COLUMNS = (
    "neighbor_utt_ids",
    "topk_utt_ids",
    "neighbor_speaker_ids",
    "topk_speaker_ids",
)
DEFAULT_DROP_COLUMNS = (
    "path",
    "audio_path",
    "source_path",
    "query_source_path",
    "nearest_source_path",
    "neighbor_source_path",
    "neighbor_source_paths",
    "topk_source_paths",
)


@dataclass(frozen=True)
class RedactionReport:
    row_count: int
    hashed_columns: tuple[str, ...]
    hashed_list_columns: tuple[str, ...]
    dropped_columns: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "row_count": self.row_count,
            "hashed_columns": list(self.hashed_columns),
            "hashed_list_columns": list(self.hashed_list_columns),
            "dropped_columns": list(self.dropped_columns),
        }


def redact_frame(
    frame: pd.DataFrame,
    *,
    salt: str = "",
    hash_columns: tuple[str, ...] = DEFAULT_HASH_COLUMNS,
    list_hash_columns: tuple[str, ...] = DEFAULT_LIST_HASH_COLUMNS,
    drop_columns: tuple[str, ...] = DEFAULT_DROP_COLUMNS,
) -> tuple[pd.DataFrame, RedactionReport]:
    redacted = frame.copy()
    hashed: list[str] = []
    hashed_lists: list[str] = []
    dropped: list[str] = []

    for column in drop_columns:
        if column in redacted.columns:
            redacted = redacted.drop(columns=[column])
            dropped.append(column)

    for column in hash_columns:
        if column in redacted.columns:
            redacted[column] = redacted[column].map(lambda value: _hash_value(value, salt=salt))
            hashed.append(column)

    for column in list_hash_columns:
        if column in redacted.columns:
            redacted[column] = redacted[column].map(lambda value: _hash_list(value, salt=salt))
            hashed_lists.append(column)

    report = RedactionReport(
        row_count=len(redacted),
        hashed_columns=tuple(hashed),
        hashed_list_columns=tuple(hashed_lists),
        dropped_columns=tuple(dropped),
    )
    return redacted, report


def redact_csv(input_csv: str | Path, output_csv: str | Path, *, salt: str = "") -> RedactionReport:
    frame = pd.read_csv(input_csv)
    redacted, report = redact_frame(frame, salt=salt)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    redacted.to_csv(output_path, index=False)
    return report


def _hash_value(value: object, *, salt: str) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    digest = hashlib.sha256(f"{salt}\0{text}".encode()).hexdigest()[:16]
    return f"id_{digest}"


def _hash_list(value: object, *, salt: str) -> str:
    if pd.isna(value):
        return ""
    items = [item.strip() for item in str(value).split(";")]
    return ";".join(_hash_value(item, salt=salt) if item else "" for item in items)
