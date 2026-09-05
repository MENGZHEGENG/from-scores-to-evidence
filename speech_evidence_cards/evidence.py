from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from .metrics import metric_row
from .splits import stable_family_for_sample

REQUIRED_SCORE_COLUMNS = [
    "sample_id",
    "label",
    "family",
    "passive_score",
    "watermark_score",
    "retrieval_score",
    "profile_margin",
]

STREAMS = ["passive", "watermark", "retrieval", "profile"]


def require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def minmax(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    low = float(values.min())
    high = float(values.max())
    if high == low:
        return pd.Series(np.full(len(values), 0.5), index=values.index)
    return (values - low) / (high - low)


def add_normalized_streams(frame: pd.DataFrame) -> pd.DataFrame:
    require_columns(frame, REQUIRED_SCORE_COLUMNS)
    output = frame.copy()
    for stream in STREAMS:
        output[f"{stream}_prob"] = minmax(output[f"{stream}_score" if stream != "profile" else "profile_margin"])
    return output


def build_evidence_cards(frame: pd.DataFrame) -> pd.DataFrame:
    output = add_normalized_streams(frame)
    passive_watermark_streams = ["passive_prob", "watermark_prob"]
    base_streams = passive_watermark_streams + ["retrieval_prob"]
    all_streams = base_streams + ["profile_prob"]
    output["passive_watermark_score"] = output[passive_watermark_streams].mean(axis=1)
    output["card_retrieval_score"] = output[base_streams].mean(axis=1)
    output["card_profile_score"] = output[all_streams].mean(axis=1)
    output["stream_disagreement"] = output[all_streams].std(axis=1)
    output["passive_watermark_disagreement"] = (output["passive_prob"] - output["watermark_prob"]).abs()
    output["card_retrieval_disagreement"] = (output["passive_watermark_score"] - output["retrieval_prob"]).abs()
    output["decision_margin"] = (output["card_profile_score"] - 0.5).abs()
    output["watermark_low"] = output["watermark_prob"] < 0.25
    output["profile_mismatch"] = output["profile_prob"] < 0.35
    output["decision"] = np.where(output["card_profile_score"] >= 0.5, "bonafide", "spoof")
    return output


def matched_evidence_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    cards = build_evidence_cards(frame)
    labels = cards["label"].to_numpy(dtype=int)
    rows = [
        metric_row("passive", labels, cards["passive_prob"]),
        metric_row("watermark", labels, cards["watermark_prob"]),
        metric_row("retrieval", labels, cards["retrieval_prob"]),
        metric_row("profile", labels, cards["profile_prob"]),
        metric_row("card_retrieval", labels, cards["card_retrieval_score"]),
        metric_row("card_profile", labels, cards["card_profile_score"]),
    ]
    return pd.DataFrame(rows)


def family_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    cards = build_evidence_cards(frame)
    rows = []
    for family, group in cards.groupby("family", sort=True):
        labels = group["label"].to_numpy(dtype=int)
        if len(set(labels.tolist())) < 2:
            continue
        for name, column in [
            ("passive", "passive_prob"),
            ("retrieval", "retrieval_prob"),
            ("card_retrieval", "card_retrieval_score"),
            ("card_profile", "card_profile_score"),
        ]:
            row = metric_row(name, labels, group[column])
            row["family"] = family
            rows.append(row)
    return pd.DataFrame(rows)


def base_calibration_features(cards: pd.DataFrame) -> pd.DataFrame:
    return cards[
        [
            "passive_prob",
            "watermark_prob",
            "passive_watermark_score",
            "retrieval_prob",
            "profile_prob",
            "card_retrieval_score",
            "card_profile_score",
        ]
    ].copy()


def passive_margin_features(cards: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {"passive_margin_from_half": (cards["passive_prob"].astype(float) - 0.5).abs()},
        index=cards.index,
    )


def passive_squared_features(cards: pd.DataFrame) -> pd.DataFrame:
    output = cards[["passive_prob"]].copy()
    output["passive_squared"] = cards["passive_prob"].astype(float) ** 2
    return output


def retrieval_profile_features(cards: pd.DataFrame) -> pd.DataFrame:
    return cards[["retrieval_prob", "profile_prob"]].copy()


def passive_retrieval_profile_features(cards: pd.DataFrame) -> pd.DataFrame:
    return cards[["passive_prob", "retrieval_prob", "profile_prob"]].copy()


def passive_retrieval_shape_features(cards: pd.DataFrame) -> pd.DataFrame:
    output = passive_retrieval_profile_features(cards)
    output["passive_squared"] = cards["passive_prob"].astype(float) ** 2
    output["passive_margin_from_half"] = (cards["passive_prob"].astype(float) - 0.5).abs()
    return output


def disagreement_features(cards: pd.DataFrame) -> pd.DataFrame:
    output = base_calibration_features(cards)
    output["passive_watermark_disagreement"] = cards["passive_watermark_disagreement"]
    output["card_retrieval_disagreement"] = cards["card_retrieval_disagreement"]
    return output


def passive_watermark_conflict_features(cards: pd.DataFrame) -> pd.DataFrame:
    output = base_calibration_features(cards)
    output["passive_watermark_disagreement"] = cards["passive_watermark_disagreement"]
    return output


def card_retrieval_conflict_features(cards: pd.DataFrame) -> pd.DataFrame:
    output = base_calibration_features(cards)
    output["card_retrieval_disagreement"] = cards["card_retrieval_disagreement"]
    return output


def nonlinear_control_features(cards: pd.DataFrame) -> pd.DataFrame:
    output = base_calibration_features(cards)
    output["passive_watermark_product"] = cards["passive_prob"] * cards["watermark_prob"]
    output["card_retrieval_product"] = cards["passive_watermark_score"] * cards["retrieval_prob"]
    return output


def squared_gap_control_features(cards: pd.DataFrame) -> pd.DataFrame:
    output = base_calibration_features(cards)
    output["passive_watermark_disagreement_squared"] = cards["passive_watermark_disagreement"] ** 2
    output["card_retrieval_disagreement_squared"] = cards["card_retrieval_disagreement"] ** 2
    return output


def leave_family_out_calibration(
    frame: pd.DataFrame,
    *,
    with_disagreement: bool | None = None,
    mode: str | None = None,
    fold_seed: int | None = None,
) -> pd.DataFrame:
    cards = build_evidence_cards(frame)
    if mode is None:
        mode = "evidence_card" if with_disagreement else "base_streams"
    feature_builders = {
        "base_streams": base_calibration_features,
        "passive_margin": passive_margin_features,
        "passive_squared": passive_squared_features,
        "retrieval_profile": retrieval_profile_features,
        "passive_retrieval_profile": passive_retrieval_profile_features,
        "passive_retrieval_shape": passive_retrieval_shape_features,
        "passive_watermark_conflict": passive_watermark_conflict_features,
        "card_retrieval_conflict": card_retrieval_conflict_features,
        "nonlinear_control": nonlinear_control_features,
        "squared_gap_control": squared_gap_control_features,
        "evidence_card": disagreement_features,
    }
    if mode not in feature_builders:
        raise ValueError(f"unknown calibration mode: {mode}")
    fold_values = calibration_fold_values(cards, fold_seed=fold_seed)
    predictions = np.zeros(len(cards), dtype=float)
    for family in sorted(fold_values.unique()):
        train_mask = fold_values != family
        test_mask = ~train_mask
        if cards.loc[train_mask, "label"].nunique() < 2:
            predictions[test_mask] = cards.loc[test_mask, "card_profile_score"]
            continue
        x_train = feature_builders[mode](cards.loc[train_mask])
        x_test = feature_builders[mode](cards.loc[test_mask])
        scaler = MinMaxScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=2027)
        model.fit(x_train_scaled, cards.loc[train_mask, "label"].astype(int))
        predictions[test_mask] = model.predict_proba(x_test_scaled)[:, 1]
    output = cards[["sample_id", "label", "family"]].copy()
    if "held_out_family" in cards.columns:
        output["held_out_family"] = cards["held_out_family"].astype(str)
    output["calibration_fold"] = fold_values.to_numpy()
    output["calibrated_score"] = predictions
    output["calibration_mode"] = mode
    return output


def calibration_fold_values(cards: pd.DataFrame, *, fold_seed: int | None = None) -> pd.Series:
    labels = pd.to_numeric(cards["label"], errors="raise").astype(int)
    if fold_seed is None:
        if "held_out_family" in cards.columns:
            return cards["held_out_family"].astype(str)
        return cards["family"].astype(str)

    fold_source = "held_out_family" if "held_out_family" in cards.columns else "family"
    family_values = cards[fold_source].astype(str)
    spoof_families = tuple(sorted(set(family_values[labels == 0])))
    if not spoof_families:
        raise ValueError("calibration requires at least one spoof family")

    assigned: list[str] = []
    for sample_id, label, family in zip(cards["sample_id"].astype(str), labels, family_values, strict=True):
        if label == 0:
            assigned.append(str(family))
        elif label == 1:
            assigned.append(stable_family_for_sample(str(sample_id), spoof_families, seed=fold_seed))
        else:
            raise ValueError(f"unexpected label {label!r}; expected 0 or 1")
    return pd.Series(assigned, index=cards.index, name="calibration_fold")
