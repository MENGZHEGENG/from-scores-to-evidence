from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import pandas as pd
import yaml

from .inputs import validate_evidence_scores

DEFAULT_HELD_OUT_FAMILIES = tuple(f"A{index:02d}" for index in range(9, 17))


@dataclass(frozen=True)
class SplitPolicy:
    name: str
    held_out_families: tuple[str, ...]
    bonafide_label: int = 1
    spoof_label: int = 0


def load_split_policy(path: Path | str) -> SplitPolicy:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    families = tuple(str(value).strip() for value in data.get("held_out_families", DEFAULT_HELD_OUT_FAMILIES))
    if not families:
        raise ValueError("split policy must define at least one held-out family")
    labels = data.get("label_convention", {}) or {}
    return SplitPolicy(
        name=str(data.get("name", "asvspoof5_track1_dev_matched")),
        held_out_families=families,
        bonafide_label=int(labels.get("bonafide", 1)),
        spoof_label=int(labels.get("spoof", 0)),
    )


def stable_family_for_sample(
    sample_id: str,
    families: tuple[str, ...] = DEFAULT_HELD_OUT_FAMILIES,
    *,
    seed: int | None = None,
) -> str:
    if not families:
        raise ValueError("families must not be empty")
    key = sample_id if seed is None else f"{seed}:{sample_id}"
    digest = sha256(key.encode("utf-8")).digest()
    index = int.from_bytes(digest[:8], byteorder="big", signed=False) % len(families)
    return families[index]


def assign_asvspoof5_track1_folds(
    scores: pd.DataFrame,
    *,
    policy: SplitPolicy | None = None,
    allow_other_families: bool = False,
    fold_seed: int | None = None,
) -> pd.DataFrame:
    policy = policy or SplitPolicy("asvspoof5_track1_dev_matched", DEFAULT_HELD_OUT_FAMILIES)
    validate_evidence_scores(scores).raise_for_issues()

    frame = scores.copy()
    labels = pd.to_numeric(frame["label"], errors="raise").astype(int)
    families = frame["family"].astype("string").str.strip()
    sample_ids = frame["sample_id"].astype("string").str.strip()
    held_out_set = set(policy.held_out_families)

    outside = sorted(set(families[(labels == policy.spoof_label) & ~families.isin(held_out_set)].dropna()))
    if outside and not allow_other_families:
        raise ValueError(
            "spoof family values are outside the configured held-out set: " + ", ".join(outside)
        )

    assigned_folds: list[str] = []
    split_roles: list[str] = []
    for sample_id, label, family in zip(sample_ids, labels, families, strict=True):
        if label == policy.spoof_label:
            assigned_folds.append(str(family))
            split_roles.append("spoof_family" if family in held_out_set else "outside_policy")
        elif label == policy.bonafide_label:
            assigned_folds.append(
                stable_family_for_sample(str(sample_id), policy.held_out_families, seed=fold_seed)
            )
            split_roles.append("bonafide_balance")
        else:
            raise ValueError(f"unexpected label {label!r}; expected {policy.spoof_label} or {policy.bonafide_label}")

    frame["held_out_family"] = assigned_folds
    frame["split_role"] = split_roles
    return frame
