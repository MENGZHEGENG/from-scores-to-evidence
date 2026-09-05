#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def _chars(*codes: int) -> str:
    return "".join(chr(code) for code in codes)


FORBIDDEN_TEXT = [
    _chars(100, 114, 97, 119, 45, 115, 112, 101, 101, 99, 104, 45, 102, 111, 114, 101, 110, 115, 105, 99, 115),
    _chars(100, 114, 97, 119, 45, 115, 112, 101, 101, 99, 104, 45, 100, 101, 101, 112, 102, 97, 107, 101, 45, 102, 111, 114, 101, 110, 115, 105, 99, 115),
    _chars(115, 112, 101, 101, 99, 104, 45, 100, 101, 101, 112, 102, 97, 107, 101, 45, 119, 97, 116, 101, 114, 109, 97, 114, 107, 105, 110, 103),
    _chars(115, 112, 101, 101, 99, 104, 45, 100, 101, 101, 112, 102, 97, 107, 101, 45, 119, 97, 116, 101, 114, 109, 97, 114, 107, 105, 110, 103, 45, 97, 114, 116, 105, 102, 97, 99, 116, 115),
    _chars(97, 114, 116, 105, 102, 97, 99, 116),
    _chars(109, 97, 110, 105, 102, 101, 115, 116),
    _chars(115, 108, 117, 114, 109),
    _chars(115, 98, 97, 116, 99, 104),
    _chars(115, 114, 117, 110),
    _chars(115, 113, 117, 101, 117, 101),
    _chars(103, 112, 115, 99),
    _chars(97, 49, 48, 48),
    _chars(47, 117, 115, 101, 114, 115, 47, 103, 101, 110, 103, 109),
    _chars(47, 104, 111, 109, 101, 47, 109, 101, 103, 48, 48, 48),
    _chars(102, 117, 108, 108, 32, 114, 101, 112, 111),
    _chars(112, 114, 105, 118, 97, 116, 101, 32, 114, 101, 112, 111),
]
FORBIDDEN_CASE_SENSITIVE_TEXT = []
FORBIDDEN_SUFFIXES = {
    ".csv",
    ".flac",
    ".jsonl",
    ".mp3",
    ".npy",
    ".npz",
    ".onnx",
    ".parquet",
    ".pt",
    ".pth",
    ".tsv",
    ".wav",
    ".ckpt",
}
REQUIRED_PATHS = {
    ".github/workflows/ci.yml",
    "CITATION.md",
    "LICENSE",
    "README.md",
    "configs/asvspoof5_track1_dev_splits.yaml",
    "configs/main.yaml",
    "docs/data_sources.md",
    "docs/input_schemas.md",
    "docs/reproduction_plan.md",
    "speech_evidence_cards/evidence.py",
    "speech_evidence_cards/experiments.py",
    "speech_evidence_cards/figures.py",
    "speech_evidence_cards/inputs.py",
    "speech_evidence_cards/metrics.py",
    "speech_evidence_cards/redaction.py",
    "speech_evidence_cards/retrieval.py",
    "speech_evidence_cards/row_selection.py",
    "speech_evidence_cards/splits.py",
    "speech_evidence_cards/watermark.py",
    "pyproject.toml",
    "scripts/run_all.py",
    "scripts/run_retrieval_profile.py",
    "scripts/run_retrieval_selection_check.py",
    "scripts/redact_outputs_for_release.py",
    "scripts/run_matched_evidence.py",
    "scripts/apply_asvspoof5_splits.py",
    "scripts/run_calibration_split_stability.py",
    "scripts/run_evidence_card_triage.py",
    "scripts/validate_inputs.py",
    "scripts/run_stress_checks.py",
    "tests/test_smoke.py",
}
REQUIRED_README_SNIPPETS = {
    "quick-start demo command": "python scripts/run_all.py --demo --out runs/demo",
    "strict release validation command": "python scripts/validate_repo.py --clean --history --strict-local",
    "main reproduction command": "python scripts/run_all.py --input-dir prepared_scores --out runs/main",
    "matched evaluation command": "python scripts/run_matched_evidence.py --scores runs/main/evidence_scores_with_folds.csv --out runs/main",
    "calibration ablation command": "python scripts/run_calibration_ablation.py --scores runs/main/evidence_scores_with_folds.csv --out runs/main",
    "calibration stability command": "python scripts/run_calibration_split_stability.py --scores runs/main/evidence_scores_with_folds.csv --out runs/main",
    "manual-review triage command": "python scripts/run_evidence_card_triage.py --scores runs/main/evidence_scores_with_folds.csv --out runs/main",
}
REQUIRED_REPRODUCTION_SNIPPETS = {
    "main reproduction command": "python scripts/run_all.py --input-dir prepared_scores --out runs/main",
    "retrieval selection command": "python scripts/run_retrieval_selection_check.py",
    "calibration stability command": "python scripts/run_calibration_split_stability.py",
    "attack-stress command": "python scripts/run_attack_stress.py --scores ssl_attack_scores.csv --out runs/main",
    "editable figure command": "python scripts/make_figures.py --results runs/main --out runs/main/figures",
}
FORBIDDEN_TRACKED_DIRS = {
    "runs",
    "figures",
    "tables",
    "paper",
}
FORBIDDEN_CACHE_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".ipynb_checkpoints",
}
SKIP_DIRS = {".git", ".venv", "runs", "__pycache__", ".pytest_cache", ".ruff_cache"}


def tracked_like_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        yield path


def checked_files(root: Path):
    tracked = git_output(root, ["ls-files", "-z"])
    if tracked is None:
        yield from tracked_like_files(root)
        return
    for rel in tracked.split("\0"):
        if not rel:
            continue
        path = root / rel
        if path.is_file():
            yield path


def local_release_dirs(root: Path) -> list[str]:
    found = set()
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        rel_parts = path.relative_to(root).parts
        if not rel_parts or rel_parts[0] in {".git", ".venv"}:
            continue
        if rel_parts[0] in FORBIDDEN_TRACKED_DIRS:
            found.add(rel_parts[0])
            continue
        if path.name == ".pytest_cache":
            continue
        if path.name in FORBIDDEN_CACHE_DIRS:
            found.add(path.relative_to(root).as_posix())
    return sorted(found)


def validate_tree(root: Path, strict_local: bool = False) -> list[str]:
    issues = []
    for rel in sorted(REQUIRED_PATHS):
        if not (root / rel).is_file():
            issues.append(f"required release file is missing: {rel}")
    readme = root / "README.md"
    if readme.is_file():
        readme_text = readme.read_text(encoding="utf-8", errors="ignore")
        for name, snippet in REQUIRED_README_SNIPPETS.items():
            if snippet not in readme_text:
                issues.append(f"README.md is missing documented {name}: {snippet}")
    reproduction_plan = root / "docs" / "reproduction_plan.md"
    if reproduction_plan.is_file():
        reproduction_text = reproduction_plan.read_text(encoding="utf-8", errors="ignore")
        for name, snippet in REQUIRED_REPRODUCTION_SNIPPETS.items():
            if snippet not in reproduction_text:
                issues.append(f"docs/reproduction_plan.md is missing documented {name}: {snippet}")
    if strict_local:
        for rel in local_release_dirs(root):
            issues.append(f"local generated or cache directory should be removed before release: {rel}")
    for path in checked_files(root):
        rel = path.relative_to(root).as_posix()
        top = rel.split("/", 1)[0]
        if top in FORBIDDEN_TRACKED_DIRS:
            issues.append(f"generated output or presentation-only file should not be tracked: {rel}")
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            issues.append(f"table dump, large media, or model file should not be tracked: {rel}")
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        for needle in FORBIDDEN_TEXT:
            if needle in text:
                issues.append(f"forbidden release text {needle!r} in {rel}")
        original_text = path.read_text(encoding="utf-8", errors="ignore")
        for needle in FORBIDDEN_CASE_SENSITIVE_TEXT:
            if needle in original_text:
                issues.append(f"forbidden release text {needle!r} in {rel}")
    return issues


def clean_local_release_dirs(root: Path) -> list[str]:
    removed = []
    root_resolved = root.resolve()
    for rel in local_release_dirs(root):
        path = root / rel
        resolved = path.resolve()
        if root_resolved not in resolved.parents:
            continue
        if path.is_dir():
            shutil.rmtree(path)
            removed.append(rel)
    return removed


def git_output(root: Path, args: list[str]) -> str | None:
    if not (root / ".git").exists():
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def validate_history(root: Path) -> list[str]:
    commits = git_output(root, ["rev-list", "--all"])
    if not commits:
        return []
    grep_args = ["grep", "-n", "-I", "-i"]
    for needle in FORBIDDEN_TEXT:
        grep_args.extend(["-e", needle])
    sensitive_grep_args = ["grep", "-n", "-I"]
    for needle in FORBIDDEN_CASE_SENSITIVE_TEXT:
        sensitive_grep_args.extend(["-e", needle])
    issues = []
    for commit in commits.splitlines():
        outputs = [git_output(root, [*grep_args, commit, "--", "."])]
        if len(sensitive_grep_args) > 4:
            outputs.append(git_output(root, [*sensitive_grep_args, commit, "--", "."]))
        for output in outputs:
            if not output:
                continue
            for line in output.splitlines():
                try:
                    _, file_name, line_number, _ = line.split(":", 3)
                    location = f"{commit[:7]}:{file_name}:{line_number}"
                except ValueError:
                    location = f"{commit[:7]}:{line}"
                issues.append(f"forbidden release text in git history: {location}")
                if len(issues) >= 20:
                    return issues
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate that the public release tree is clean and portable.")
    parser.add_argument("--history", action="store_true", help="also scan previous Git commits before public release")
    parser.add_argument(
        "--strict-local",
        action="store_true",
        help="also fail when ignored generated or cache directories are present locally",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="remove ignored generated output and cache directories before validation",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    removed = clean_local_release_dirs(root) if args.clean else []
    issues = validate_tree(root, strict_local=args.strict_local)
    if args.history:
        issues.extend(validate_history(root))
    if issues:
        for issue in issues:
            print(issue)
        return 1
    scope_items = ["release tree"]
    if args.strict_local:
        scope_items.append("local cleanup")
    if args.history:
        scope_items.append("git history")
    scope = ", ".join(scope_items)
    if removed:
        print("removed local generated/cache directories: " + ", ".join(removed))
    print(f"{scope} validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
