from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PALETTE = {
    "ink": "#23272f",
    "slate": "#586370",
    "teal": "#268080",
    "navy": "#1f4e79",
    "amber": "#b28025",
    "coral": "#be5c49",
    "paper": "#fafaf8",
}


def _save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.svg", bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def matched_evidence_tradeoff(results: Path, out: Path) -> None:
    frame = pd.read_csv(results / "matched_evidence_metrics.csv")
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.scatter(frame["eer_percent"], frame["ece"], s=46, color=PALETTE["teal"])
    for _, row in frame.iterrows():
        ax.annotate(row["method"], (row["eer_percent"], row["ece"]), xytext=(4, 3), textcoords="offset points", fontsize=7)
    ax.set_xlabel("EER (%, lower is better)")
    ax.set_ylabel("ECE (lower is better)")
    ax.grid(True, color="#d5d9df", linewidth=0.6)
    _save(fig, out, "matched_evidence_tradeoff")


def family_eer(results: Path, out: Path) -> None:
    frame = pd.read_csv(results / "family_eer.csv")
    pivot = frame.pivot_table(index="family", columns="method", values="eer_percent")
    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    for method in pivot.columns:
        ax.plot(pivot.index, pivot[method], marker="o", linewidth=1.3, label=method)
    ax.set_xlabel("held-out family")
    ax.set_ylabel("EER (%)")
    ax.grid(True, axis="y", color="#d5d9df", linewidth=0.6)
    ax.legend(fontsize=7, ncol=2)
    _save(fig, out, "family_eer")


def attack_stress(results: Path, out: Path) -> None:
    frame = pd.read_csv(results / "attack_stress.csv")
    frame["row"] = frame["attack_family"] + ": " + frame["attack_variant"].astype(str)
    pivot = frame.pivot_table(index="row", columns="model", values="eer_percent")
    fig, ax = plt.subplots(figsize=(5.3, 5.8))
    image = ax.imshow(pivot.values, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, fontsize=8)
    ax.set_yticks(range(len(pivot.index)), pivot.index, fontsize=7)
    for y in range(pivot.shape[0]):
        for x in range(pivot.shape[1]):
            value = pivot.values[y, x]
            ax.text(x, y, f"{value:.1f}", ha="center", va="center", fontsize=6, color="white" if value > pivot.values.max() * 0.6 else PALETTE["ink"])
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03, label="EER (%)")
    _save(fig, out, "attack_stress")


def active_selection(results: Path, out: Path) -> None:
    frame = pd.read_csv(results / "active_selection.csv")
    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    for strategy, group in frame.groupby("strategy"):
        ax.plot(group["selected_samples"], group["captured_error_rate"], marker="o", linewidth=1.2, label=strategy)
    ax.set_xscale("log")
    ax.set_xlabel("selected examples")
    ax.set_ylabel("captured error rate")
    ax.grid(True, axis="y", color="#d5d9df", linewidth=0.6)
    ax.legend(fontsize=7, ncol=2)
    _save(fig, out, "active_selection")


def evidence_cards(results: Path, out: Path, n: int = 8) -> None:
    frame = pd.read_csv(results / "evidence_cards.csv").head(n)
    fig, ax = plt.subplots(figsize=(7.0, 0.45 + 0.36 * len(frame)))
    ax.axis("off")
    table = ax.table(
        cellText=frame[["sample_id", "family", "decision", "card_profile_score", "stream_disagreement"]].round(3).values,
        colLabels=["sample", "family", "decision", "score", "disagreement"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.2)
    _save(fig, out, "evidence_cards")


def make_all(results: Path, out: Path) -> None:
    matched_evidence_tradeoff(results, out)
    family_eer(results, out)
    attack_stress(results, out)
    active_selection(results, out)
    evidence_cards(results, out)
