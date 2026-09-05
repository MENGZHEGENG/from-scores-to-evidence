from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from speech_evidence_cards.evidence import build_evidence_cards, leave_family_out_calibration
from speech_evidence_cards.experiments import (
    _decision_threshold_rank_margin,
    _risk_coverage_auc,
    _selective_risk_table,
    make_demo_scores,
    run_attack_stress,
    run_calibration,
    run_calibration_split_stability,
    run_evidence_card_triage,
    run_matched_evidence,
    run_stress_checks,
)
from speech_evidence_cards.inputs import (
    validate_attack_scores,
    validate_evidence_scores,
    validate_stress_check_scores,
)
from speech_evidence_cards.redaction import redact_frame
from speech_evidence_cards.retrieval import (
    audit_retrieval_neighbors,
    retrieval_label_scores,
    retrieval_profile_scores,
)
from speech_evidence_cards.row_selection import (
    row_selection_table,
    run_row_selection_check,
    score_retention_table,
)
from speech_evidence_cards.splits import (
    DEFAULT_HELD_OUT_FAMILIES,
    assign_asvspoof5_track1_folds,
)

ROOT = Path(__file__).resolve().parents[1]


def _clean_runtime_dirs() -> None:
    for path in [ROOT / ".pytest_cache", *ROOT.rglob("__pycache__")]:
        if path.exists():
            shutil.rmtree(path)
VALIDATOR = ROOT / "scripts" / "validate_repo.py"


spec = importlib.util.spec_from_file_location("validate_repo", VALIDATOR)
validate_repo = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(validate_repo)


def _write_release_scaffold(root: Path) -> None:
    for rel in validate_repo.REQUIRED_PATHS:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("clean reproduction code\n", encoding="utf-8")
    (root / "README.md").write_text(
        "\n".join(validate_repo.REQUIRED_README_SNIPPETS.values()) + "\n",
        encoding="utf-8",
    )
    (root / "docs" / "reproduction_plan.md").write_text(
        "\n".join(validate_repo.REQUIRED_REPRODUCTION_SNIPPETS.values()) + "\n",
        encoding="utf-8",
    )


def test_demo_pipeline_writes_main_outputs(tmp_path: Path) -> None:
    evidence, attacks, stress = make_demo_scores(n=300)
    run_matched_evidence(evidence, tmp_path)
    run_calibration(evidence, tmp_path)
    run_calibration_split_stability(evidence, tmp_path)
    run_evidence_card_triage(evidence, tmp_path)
    run_attack_stress(attacks, tmp_path)
    run_stress_checks(stress, tmp_path)
    assert (tmp_path / "matched_evidence_metrics.csv").exists()
    assert (tmp_path / "calibration_ablation.csv").exists()
    assert (tmp_path / "calibration_split_stability.csv").exists()
    assert (tmp_path / "calibration_split_stability_runs.csv").exists()
    assert (tmp_path / "evidence_card_triage.csv").exists()
    assert (tmp_path / "attack_stress.csv").exists()
    assert (tmp_path / "stress_checks.csv").exists()
    calibration = pd.read_csv(tmp_path / "calibration_ablation.csv")
    assert set(calibration["calibration_mode"]) == {
        "base_streams",
        "passive_margin",
        "passive_squared",
        "retrieval_profile",
        "passive_retrieval_profile",
        "passive_retrieval_shape",
        "passive_watermark_conflict",
        "card_retrieval_conflict",
        "nonlinear_control",
        "squared_gap_control",
        "evidence_card",
    }
    stability = pd.read_csv(tmp_path / "calibration_split_stability.csv")
    assert set(stability["calibration_mode"]) == {
        "base_streams",
        "evidence_card",
        "squared_gap_control",
    }
    assert set(stability["seed_count"]) == {10}
    metrics = pd.read_csv(tmp_path / "matched_evidence_metrics.csv")
    assert {"passive", "card_retrieval", "card_profile"}.issubset(set(metrics["method"]))
    triage = pd.read_csv(tmp_path / "evidence_card_triage.csv")
    assert "error_capture_rate_ci_low" in triage.columns
    assert "review_precision_ci_high" in triage.columns


def test_input_validators_accept_demo_scores() -> None:
    evidence, attacks, stress = make_demo_scores(n=300)

    assert validate_evidence_scores(evidence).ready
    assert validate_attack_scores(attacks).ready
    assert validate_stress_check_scores(stress).ready


def test_input_validator_reports_bad_evidence_scores() -> None:
    evidence, _, _ = make_demo_scores(n=64)
    evidence.loc[0, "sample_id"] = evidence.loc[1, "sample_id"]
    evidence.loc[2, "passive_score"] = float("nan")

    report = validate_evidence_scores(evidence)

    assert not report.ready
    assert any("sample_id" in issue for issue in report.issues)
    assert any("passive_score" in issue for issue in report.issues)


def test_validate_inputs_cli_json(tmp_path: Path) -> None:
    evidence, attacks, stress = make_demo_scores(n=300)
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    evidence_path = input_dir / "evidence_scores.csv"
    attack_path = input_dir / "ssl_attack_scores.csv"
    stress_path = input_dir / "stress_check_scores.csv"
    evidence.to_csv(evidence_path, index=False)
    attacks.to_csv(attack_path, index=False)
    stress.to_csv(stress_path, index=False)

    result = subprocess.run(
        [
            "python3",
            "scripts/validate_inputs.py",
            "--evidence-scores",
            str(evidence_path),
            "--attack-scores",
            str(attack_path),
            "--stress-check-scores",
            str(stress_path),
            "--json",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)
    assert payload["ready"] is True
    assert {report["name"] for report in payload["reports"]} == {
        "evidence_scores",
        "ssl_attack_scores",
        "stress_check_scores",
    }


def test_public_release_metadata_files_are_present() -> None:
    citation = ROOT / "CITATION.md"
    license_file = ROOT / "LICENSE"

    assert citation.is_file()
    assert license_file.is_file()
    assert "official metadata is available" in citation.read_text(encoding="utf-8")
    assert "License pending" in license_file.read_text(encoding="utf-8")


def test_asvspoof5_split_assignment_is_deterministic() -> None:
    evidence, _, _ = make_demo_scores(n=300)

    assigned_once = assign_asvspoof5_track1_folds(evidence)
    assigned_twice = assign_asvspoof5_track1_folds(evidence.sample(frac=1.0, random_state=7)).sort_values("sample_id")
    assigned_once_sorted = assigned_once.sort_values("sample_id")

    assert assigned_once_sorted["held_out_family"].tolist() == assigned_twice["held_out_family"].tolist()
    spoof_rows = assigned_once[assigned_once["label"] == 0]
    assert (spoof_rows["held_out_family"] == spoof_rows["family"]).all()
    bonafide_rows = assigned_once[assigned_once["label"] == 1]
    assert set(bonafide_rows["held_out_family"]).issubset(set(DEFAULT_HELD_OUT_FAMILIES))
    assert set(assigned_once["split_role"]) == {"bonafide_balance", "spoof_family"}


def test_calibration_split_seed_changes_only_bonafide_folds() -> None:
    evidence, _, _ = make_demo_scores(n=300)
    assigned = assign_asvspoof5_track1_folds(evidence)
    cards = build_evidence_cards(assigned)

    first = leave_family_out_calibration(cards, mode="evidence_card", fold_seed=1)
    second = leave_family_out_calibration(cards, mode="evidence_card", fold_seed=2)
    first_sorted = first.sort_values("sample_id")
    second_sorted = second.sort_values("sample_id")
    spoof_mask = first_sorted["label"].astype(int) == 0
    bonafide_mask = first_sorted["label"].astype(int) == 1

    assert first_sorted.loc[spoof_mask, "calibration_fold"].tolist() == second_sorted.loc[
        spoof_mask, "calibration_fold"
    ].tolist()
    assert first_sorted.loc[bonafide_mask, "calibration_fold"].tolist() != second_sorted.loc[
        bonafide_mask, "calibration_fold"
    ].tolist()


def test_apply_asvspoof5_splits_cli_writes_expected_columns(tmp_path: Path) -> None:
    evidence, _, _ = make_demo_scores(n=300)
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    scores_path = input_dir / "evidence_scores.csv"
    output_path = tmp_path / "evidence_scores_with_folds.csv"
    evidence.to_csv(scores_path, index=False)

    result = subprocess.run(
        [
            "python3",
            "scripts/apply_asvspoof5_splits.py",
            "--scores",
            str(scores_path),
            "--out",
            str(output_path),
            "--json",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)
    assigned = pd.read_csv(output_path)
    assert payload["rows"] == len(evidence)
    assert set(payload["held_out_families"]) == set(DEFAULT_HELD_OUT_FAMILIES)
    assert {"held_out_family", "split_role"}.issubset(assigned.columns)


def test_run_all_demo_uses_out_directory_for_inputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "demo_run"

    result = subprocess.run(
        ["python3", "scripts/run_all.py", "--demo", "--out", str(run_dir), "--skip-figures"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert str(run_dir) in result.stdout
    assert (run_dir / "inputs" / "evidence_scores.csv").exists()
    assert (run_dir / "inputs" / "ssl_attack_scores.csv").exists()
    assert (run_dir / "inputs" / "stress_check_scores.csv").exists()
    assert (run_dir / "matched_evidence_metrics.csv").exists()
    assert (run_dir / "calibration_ablation.csv").exists()
    assert (run_dir / "calibration_split_stability.csv").exists()
    assert (run_dir / "calibration_split_stability_runs.csv").exists()
    assert (run_dir / "active_selection.csv").exists()
    assert (run_dir / "evidence_card_triage.csv").exists()
    assert (run_dir / "selective_risk.csv").exists()
    assert (run_dir / "attack_stress.csv").exists()
    assert (run_dir / "stress_checks.csv").exists()


def test_selective_risk_defers_small_threshold_rank_margins(tmp_path: Path) -> None:
    evidence, _, _ = make_demo_scores(n=300)
    cards = build_evidence_cards(evidence)
    selective = _selective_risk_table(evidence, cards=cards, budgets=(0.01,))

    assert set(selective["system"]) == {
        "Passive",
        "Retrieval kNN",
        "Card + retrieval",
        "Card + profile",
        "cross-fit evidence card",
    }
    assert (selective["reviewed_count"] == 3).all()
    assert set(selective["confidence_method"]) == {"decision_threshold_rank_margin"}
    assert selective["risk_coverage_auc"].between(0.0, 1.0).all()


def test_rank_margin_is_invariant_to_monotone_rescaling() -> None:
    scores = pd.Series([0.1, 0.2, 0.5, 0.7, 0.8]).to_numpy()

    original = _decision_threshold_rank_margin(scores, threshold=0.5)
    stretched = _decision_threshold_rank_margin(scores * 10.0 + 3.0, threshold=8.0)

    assert original.tolist() == stretched.tolist()


def test_retrieval_profile_scores_exclude_held_out_family() -> None:
    support = pd.DataFrame(
        {
            "sample_id": ["s-a01", "s-a02", "s-bf1", "s-bf2"],
            "label": [0, 0, 1, 1],
            "family": ["A01", "A02", "bonafide", "bonafide"],
            "speaker_id": ["spk-a", "spk-b", "spk-c", "spk-d"],
            "source_path": ["a01.wav", "a02.wav", "bf1.wav", "bf2.wav"],
            "feat_0": [1.00, 0.80, -0.20, -0.10],
            "feat_1": [0.10, 0.25, 1.00, 0.85],
        }
    )
    query = pd.DataFrame(
        {
            "sample_id": ["q-a01", "q-bf"],
            "label": [0, 1],
            "family": ["A01", "bonafide"],
            "speaker_id": ["spk-q", "spk-e"],
            "source_path": ["q-a01.wav", "q-bf.wav"],
            "feat_0": [0.98, -0.15],
            "feat_1": [0.12, 0.95],
        }
    )

    scored = retrieval_profile_scores(support, query, k=2)
    audit = audit_retrieval_neighbors(scored)

    spoof_row = scored[scored["sample_id"] == "q-a01"].iloc[0]
    assert "A01" not in spoof_row["neighbor_families"].split(";")
    assert {"nearest_sample_id", "neighbor_sample_ids", "neighbor_families"}.issubset(scored.columns)
    assert audit.ready
    assert audit.same_heldout_topk_family_count == 0
    assert audit.same_topk_sample_count == 0


def test_retrieval_label_scores_keeps_legacy_columns() -> None:
    support = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "label": [1, 0],
            "family": ["bonafide", "A01"],
            "feat_0": [0.0, 1.0],
            "feat_1": [1.0, 0.0],
        }
    )
    query = pd.DataFrame(
        {
            "sample_id": ["q1"],
            "label": [1],
            "family": ["bonafide"],
            "feat_0": [0.1],
            "feat_1": [0.9],
        }
    )

    scored = retrieval_label_scores(support, query, k=1)

    assert list(scored.columns) == ["sample_id", "retrieval_score", "mean_neighbor_distance"]


def test_run_retrieval_profile_cli_writes_scores_and_audit(tmp_path: Path) -> None:
    support = pd.DataFrame(
        {
            "sample_id": ["s-a01", "s-a02", "s-bf"],
            "label": [0, 0, 1],
            "family": ["A01", "A02", "bonafide"],
            "speaker_id": ["spk-a", "spk-b", "spk-c"],
            "source_path": ["a01.wav", "a02.wav", "bf.wav"],
            "feat_0": [1.0, 0.8, -0.2],
            "feat_1": [0.1, 0.2, 1.0],
        }
    )
    query = pd.DataFrame(
        {
            "sample_id": ["q-a01", "q-bf"],
            "label": [0, 1],
            "family": ["A01", "bonafide"],
            "speaker_id": ["spk-q", "spk-d"],
            "source_path": ["q-a01.wav", "q-bf.wav"],
            "feat_0": [0.95, -0.1],
            "feat_1": [0.1, 0.9],
        }
    )
    support_path = tmp_path / "support_features.csv"
    query_path = tmp_path / "query_features.csv"
    out_path = tmp_path / "retrieval_profile.csv"
    audit_path = tmp_path / "retrieval_profile_audit.json"
    support.to_csv(support_path, index=False)
    query.to_csv(query_path, index=False)

    result = subprocess.run(
        [
            "python3",
            "scripts/run_retrieval_profile.py",
            "--support",
            str(support_path),
            "--query",
            str(query_path),
            "--out",
            str(out_path),
            "--audit",
            str(audit_path),
            "--k",
            "2",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)
    scored = pd.read_csv(out_path)
    audit = json.loads(audit_path.read_text())
    assert payload["ready"] is True
    assert len(scored) == 2
    assert {"nearest_sample_id", "neighbor_sample_ids", "neighbor_families"}.issubset(scored.columns)
    assert audit["same_heldout_topk_family_count"] == 0


def _row_selection_full_scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["b1", "b2", "b3", "s1", "s2", "s3", "s4", "s5"],
            "label": [1, 1, 1, 0, 0, 0, 0, 0],
            "family": ["bonafide", "bonafide", "bonafide", "A01", "A01", "A01", "A02", "A02"],
            "retrieval_score": [0.95, 0.80, 0.65, 0.10, 0.20, 0.85, 0.25, 0.35],
            "passive_score": [0.90, 0.78, 0.60, 0.15, 0.22, 0.70, 0.30, 0.40],
            "watermark_score": [0.90, 0.80, 0.70, 0.12, 0.18, 0.60, 0.20, 0.30],
            "profile_margin": [0.85, 0.75, 0.63, 0.20, 0.25, 0.50, 0.28, 0.33],
        }
    )


def _row_selection_matched_scores() -> pd.DataFrame:
    return _row_selection_full_scores()[lambda frame: frame["sample_id"].isin(["b1", "b2", "s3", "s4"])]


def test_row_selection_check_reports_retention_shift() -> None:
    family_rows = row_selection_table(
        _row_selection_full_scores(),
        _row_selection_matched_scores(),
        n_bootstrap=20,
        seed=7,
    )
    retention_rows = score_retention_table(_row_selection_full_scores(), _row_selection_matched_scores())

    assert set(family_rows["family"]) == {"A01", "A02"}
    assert family_rows.loc[family_rows["family"] == "A01", "matched_spoof"].iloc[0] == 1
    assert family_rows["spoof_score_ks"].gt(0).all()
    spoof_retention = retention_rows[retention_rows["group"] == "spoof"].iloc[0]
    assert spoof_retention["retained_count"] == 2
    assert spoof_retention["omitted_count"] == 3
    assert spoof_retention["score_ks"] > 0


def test_run_retrieval_selection_check_cli_writes_outputs(tmp_path: Path) -> None:
    full_path = tmp_path / "full_scores.csv"
    matched_path = tmp_path / "matched_scores.csv"
    out_dir = tmp_path / "out"
    _row_selection_full_scores().to_csv(full_path, index=False)
    _row_selection_matched_scores().to_csv(matched_path, index=False)

    result = subprocess.run(
        [
            "python3",
            "scripts/run_retrieval_selection_check.py",
            "--full-scores",
            str(full_path),
            "--matched-scores",
            str(matched_path),
            "--out",
            str(out_dir),
            "--n-bootstrap",
            "10",
            "--json",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)
    audit = json.loads((out_dir / "retrieval_selection_check.json").read_text())

    assert set(payload) == {
        "retrieval_selection_check",
        "retrieval_score_retention",
        "retrieval_selection_audit",
    }
    assert (out_dir / "retrieval_selection_check.csv").exists()
    assert (out_dir / "retrieval_score_retention.csv").exists()
    assert audit["ready"] is True
    assert audit["overall_spoof_retained_count"] == 2


def test_run_all_uses_optional_full_retrieval_scores(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    evidence, attacks, stress = make_demo_scores(n=300)
    evidence.to_csv(input_dir / "evidence_scores.csv", index=False)
    attacks.to_csv(input_dir / "ssl_attack_scores.csv", index=False)
    stress.to_csv(input_dir / "stress_check_scores.csv", index=False)
    full_retrieval = pd.concat(
        [
            evidence,
            evidence.sample(n=80, replace=True, random_state=17).assign(
                sample_id=lambda frame: "extra-" + frame["sample_id"].astype(str)
            ),
        ],
        ignore_index=True,
    )
    full_retrieval.to_csv(input_dir / "retrieval_full_scores.csv", index=False)
    out_dir = tmp_path / "run"

    subprocess.run(
        [
            "python3",
            "scripts/run_all.py",
            "--input-dir",
            str(input_dir),
            "--out",
            str(out_dir),
            "--skip-figures",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert (out_dir / "retrieval_selection_check.csv").exists()
    assert (out_dir / "retrieval_score_retention.csv").exists()
    assert (out_dir / "retrieval_selection_check.json").exists()


def test_risk_coverage_auc_tie_average_is_stable() -> None:
    errors = pd.Series([True, False, True, False]).to_numpy()
    confidence = pd.Series([0.2, 0.2, 0.9, 0.9]).to_numpy()

    auc = _risk_coverage_auc(errors, confidence)

    assert 0.0 <= auc <= 1.0


def test_selective_risk_uses_each_system_threshold() -> None:
    evidence = pd.DataFrame(
        {
            "sample_id": [f"u{index}" for index in range(8)],
            "utt_id": [f"u{index}" for index in range(8)],
            "label": [1, 1, 1, 1, 0, 0, 0, 0],
            "family": ["A00"] * 8,
            "speaker": [f"s{index}" for index in range(8)],
            "passive_score": [0.92, 0.88, 0.84, 0.46, 0.80, 0.20, 0.18, 0.16],
            "watermark_score": [0.92, 0.88, 0.84, 0.46, 0.80, 0.20, 0.18, 0.16],
            "retrieval_score": [0.92, 0.88, 0.84, 0.46, 0.80, 0.20, 0.18, 0.16],
            "profile_margin": [0.92, 0.88, 0.84, 0.46, 0.80, 0.20, 0.18, 0.16],
        }
    )
    cards = build_evidence_cards(evidence)

    selective = _selective_risk_table(evidence, cards=cards, budgets=(0.25,))

    passive = selective[selective["system"] == "Passive"].iloc[0]
    assert passive["decision_error_count"] == 2
    assert passive["errors_captured"] == 1


def test_redaction_hashes_identifiers_and_drops_paths() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": ["utt-a"],
            "speaker_id": ["spk-a"],
            "neighbor_utt_ids": ["utt-b;utt-c"],
            "neighbor_speaker_ids": ["spk-b;spk-c"],
            "source_path": ["/local/audio/utt-a.wav"],
            "score": [0.7],
        }
    )

    redacted, report = redact_frame(frame, salt="unit-test")

    assert redacted.loc[0, "sample_id"].startswith("id_")
    assert redacted.loc[0, "speaker_id"].startswith("id_")
    assert redacted.loc[0, "neighbor_utt_ids"].count("id_") == 2
    assert "source_path" not in redacted.columns
    assert report.row_count == 1
    assert "source_path" in report.dropped_columns


def test_redaction_cli_writes_redacted_csv(tmp_path: Path) -> None:
    source = tmp_path / "input.csv"
    target = tmp_path / "share" / "output.csv"
    report = tmp_path / "share" / "report.json"
    pd.DataFrame(
        {
            "sample_id": ["utt-a"],
            "nearest_utt_id": ["utt-b"],
            "source_path": ["/local/audio/utt-a.wav"],
            "score": [0.7],
        }
    ).to_csv(source, index=False)

    result = subprocess.run(
        [
            "python3",
            "scripts/redact_outputs_for_release.py",
            "--input",
            str(source),
            "--output",
            str(target),
            "--salt",
            "unit-test",
            "--report",
            str(report),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    redacted = pd.read_csv(target)
    payload = json.loads(report.read_text())
    assert "source_path" not in redacted.columns
    assert redacted.loc[0, "sample_id"].startswith("id_")
    assert payload["dropped_columns"] == ["source_path"]
    assert "row_count" in result.stdout


def test_validate_repo_tree_mode_passes() -> None:
    _clean_runtime_dirs()
    result = subprocess.run(
        ["python3", "scripts/validate_repo.py"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert "release tree validation passed" in result.stdout


def test_validate_history_reports_prior_sensitive_strings(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    env = {
        "GIT_AUTHOR_NAME": "Test Author",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test Author",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    }
    subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.DEVNULL)
    note = repo / "note.txt"
    note.write_text("old name: " + "speech-" + "deepfake-" + "watermarking" + "\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "commit", "-m", "legacy note"], cwd=repo, check=True, env=env, stdout=subprocess.DEVNULL)
    note.unlink()
    subprocess.run(["git", "add", "."], cwd=repo, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "commit", "-m", "remove note"], cwd=repo, check=True, env=env, stdout=subprocess.DEVNULL)

    issues = validate_repo.validate_history(repo)

    assert issues
    assert "git history" in issues[0]


def test_validate_repo_rejects_result_tables_and_forbidden_wording(tmp_path: Path) -> None:
    _write_release_scaffold(tmp_path)
    (tmp_path / "results.csv").write_text("metric,value\neer,1.0\n", encoding="utf-8")
    forbidden = "arti" + "fact"
    (tmp_path / "note.txt").write_text(f"{forbidden} wording should not ship\n", encoding="utf-8")
    second_forbidden = "mani" + "fest"
    (tmp_path / "readme-note.txt").write_text(f"{second_forbidden} wording should not ship\n", encoding="utf-8")

    issues = validate_repo.validate_tree(tmp_path)

    assert any("results.csv" in issue for issue in issues)
    assert any(forbidden in issue for issue in issues)
    assert any(second_forbidden in issue for issue in issues)


def test_validate_repo_rejects_local_generated_and_cache_dirs(tmp_path: Path) -> None:
    _write_release_scaffold(tmp_path)
    (tmp_path / "runs" / "demo").mkdir(parents=True)
    (tmp_path / "speech_evidence_cards" / "__pycache__").mkdir()

    issues = validate_repo.validate_tree(tmp_path, strict_local=True)

    assert any("runs" in issue for issue in issues)
    assert any("__pycache__" in issue for issue in issues)


def test_validate_repo_clean_removes_local_generated_and_cache_dirs(tmp_path: Path) -> None:
    _write_release_scaffold(tmp_path)
    (tmp_path / "runs" / "demo").mkdir(parents=True)
    (tmp_path / "speech_evidence_cards" / "__pycache__").mkdir()

    removed = validate_repo.clean_local_release_dirs(tmp_path)
    issues = validate_repo.validate_tree(tmp_path, strict_local=True)

    assert removed == ["runs", "speech_evidence_cards/__pycache__"]
    assert issues == []


def test_validate_repo_default_allows_local_demo_outputs(tmp_path: Path) -> None:
    _write_release_scaffold(tmp_path)
    (tmp_path / "runs" / "demo").mkdir(parents=True)
    (tmp_path / "speech_evidence_cards" / "__pycache__").mkdir()

    issues = validate_repo.validate_tree(tmp_path)

    assert not issues


def test_validate_repo_requires_release_scaffold(tmp_path: Path) -> None:
    issues = validate_repo.validate_tree(tmp_path)

    assert any("README.md" in issue for issue in issues)
    assert any("docs/reproduction_plan.md" in issue for issue in issues)
    assert any(".github/workflows/ci.yml" in issue for issue in issues)


def test_validate_repo_requires_documented_reproduction_commands(tmp_path: Path) -> None:
    _write_release_scaffold(tmp_path)
    (tmp_path / "README.md").write_text("clean reproduction code\n", encoding="utf-8")
    (tmp_path / "docs" / "reproduction_plan.md").write_text(
        "clean reproduction code\n", encoding="utf-8"
    )

    issues = validate_repo.validate_tree(tmp_path)

    assert any("README.md is missing documented main reproduction command" in issue for issue in issues)
    assert any(
        "docs/reproduction_plan.md is missing documented editable figure command" in issue
        for issue in issues
    )
