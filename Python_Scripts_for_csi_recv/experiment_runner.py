#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiment Runner — Cross-Subject, Cross-Environment & Ablation Studies
========================================================================

Automated evaluation experiments for the WiFi CSI HAR pipeline.
Reuses build_dataset() and train_and_evaluate() from csi_ml_pipeline.py
without modifying any existing code.

Experiments:
  cross-env      Leave-Environment-Group-Out (LEGO): train on one
                 environment, test on all others.
  cross-subject  Leave-One-Subject-Out (LOSO): train on N-1 subjects,
                 test on the remaining one.
  ablation       Systematic ablation over preprocessing toggles, PCA
                 dimensions, window sizes, and feature groups.
  all            Run all available experiments sequentially.

Usage:
  python experiment_runner.py --experiment cross-env --model rf et
  python experiment_runner.py --experiment ablation --model rf
  python experiment_runner.py --experiment all --model rf --save
  python experiment_runner.py --experiment cross-env --simulate
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np

import config
from csi_parser import configure_console_output
configure_console_output()


# ============================================================================
# 1.  FILENAME METADATA PARSER
# ============================================================================

# Known filename patterns (order matters — most specific first):
#
#   walk_activity_<NUM>_<ENV>_<SUBJECT>_<TIMESTAMP>.txt
#     e.g. walk_activity_11_room1_vasilis_1777802888.txt
#
#   walk_activity_livroom_<NUM>_<SUBJECT>_<TIMESTAMP>.txt
#     e.g. walk_activity_livroom_05_vasilis_1776093866.txt
#
#   no_activity_<NUM>_<ENV>_<SUBJECT>_<TIMESTAMP>.txt
#     e.g. no_activity_11_room1_vasilis_1777973010.txt
#
#   no_activity_livroom_<NUM>_<TIMESTAMP>.txt  (no explicit subject)
#     e.g. no_activity_livroom_01_1776088773.txt
#
#   Legacy:  walk_activity_livroom_<NUM>_<SUBJECT>_.txt
#     e.g. walk_activity_livroom_01_vasilis_.txt

# Two activity prefixes that the naming convention uses
_ACTIVITY_PREFIXES = ("walk_activity", "no_activity")

# Pattern A: <activity>_<NUM>_<ENV>_<SUBJECT>_<TIMESTAMP>.txt
_PAT_STANDARD = re.compile(
    r"^(?P<activity>walk_activity|no_activity)"
    r"_(?P<num>\d+)"
    r"_(?P<env>[a-zA-Z]\w*?)"           # environment token (alpha-start)
    r"_(?P<subject>[a-zA-Z]\w*?)"       # subject token (alpha-start)
    r"_(?P<ts>\d{7,}|)"                 # unix timestamp or empty
    r"(?:__+)?"                          # optional trailing underscores
    r"\.(?:txt|csv)$",
    re.IGNORECASE,
)

# Pattern B: <activity>_livroom_<NUM>_<SUBJECT>_<TIMESTAMP>.txt
_PAT_LIVROOM_SUBJECT = re.compile(
    r"^(?P<activity>walk_activity|no_activity)"
    r"_(?P<env>livroom)"
    r"_(?P<num>\d+)"
    r"_(?P<subject>[a-zA-Z]\w*?)"
    r"_?(?P<ts>\d{7,}|)"
    r"(?:__+)?"
    r"\.(?:txt|csv)$",
    re.IGNORECASE,
)

# Pattern C: <activity>_livroom_<NUM>_<TIMESTAMP>.txt  (no subject)
_PAT_LIVROOM_NO_SUBJECT = re.compile(
    r"^(?P<activity>walk_activity|no_activity)"
    r"_(?P<env>livroom)"
    r"_(?P<num>\d+)"
    r"_(?P<ts>\d{7,})"
    r"\.(?:txt|csv)$",
    re.IGNORECASE,
)


def parse_recording_metadata(filepath: Path) -> dict | None:
    """
    Extract subject, environment, activity class from a recording filename.

    Returns dict with keys: subject, env, activity, num, timestamp, path
    or None if the filename does not match any known pattern.
    """
    name = filepath.name

    for pat, default_subject in [
        (_PAT_STANDARD,            None),
        (_PAT_LIVROOM_SUBJECT,     None),
        (_PAT_LIVROOM_NO_SUBJECT,  "vasilis"),
    ]:
        m = pat.match(name)
        if m:
            d = m.groupdict()
            return {
                "subject":   d.get("subject") or default_subject or "unknown",
                "env":       d["env"],
                "activity":  d["activity"],
                "num":       int(d["num"]),
                "timestamp": d.get("ts", ""),
                "path":      filepath,
            }
    return None


def discover_recordings(
    data_dir: Path,
    classes: list[str],
) -> list[dict]:
    """Scan dataset directories and return metadata for every recording."""
    _, class_dirs = config.resolve_training_classes(
        classes, data_dir=data_dir, require_existing=True, print_fn=lambda *_: None,
    )

    records: list[dict] = []
    for cls in classes:
        cls_dir = class_dirs.get(cls)
        if cls_dir is None:
            continue
        for fpath in sorted(cls_dir.glob("*.txt")) + sorted(cls_dir.glob("*.csv")):
            meta = parse_recording_metadata(fpath)
            if meta is not None:
                meta["class"] = cls   # override with the config-level class name
                records.append(meta)
            else:
                # Fallback: file exists but doesn't match naming convention
                records.append({
                    "subject":  "unknown",
                    "env":      "unknown",
                    "activity": cls,
                    "class":    cls,
                    "num":      0,
                    "timestamp": "",
                    "path":     fpath,
                })
    return records


def _group_by(records: list[dict], key: str) -> dict[str, list[dict]]:
    """Group records by a metadata key."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        groups[rec[key]].append(rec)
    return dict(groups)


# ============================================================================
# 2.  EXPERIMENT RESULT HELPERS
# ============================================================================

def _extract_model_metrics(results: dict, experiment_name: str,
                           experiment_config: dict) -> dict:
    """Flatten train_and_evaluate results into a serialisable dict."""
    rows = {}
    for model_key, res in results.items():
        rows[model_key] = {
            "experiment":     experiment_name,
            "config":         experiment_config,
            "cv_mean":        round(float(res["cv_mean"]), 4),
            "cv_std":         round(float(res["cv_std"]), 4),
            "cv_scores":      [round(float(s), 4) for s in res.get("cv_scores", [])],
            "train_accuracy": round(float(res.get("train_accuracy", 0)), 4),
            "test_accuracy":  round(float(res["test_accuracy"]), 4),
            "test_f1_macro":  round(float(res["test_f1_macro"]), 4),
            "confusion_matrix": res["confusion_matrix"].tolist()
                                if hasattr(res["confusion_matrix"], "tolist")
                                else res["confusion_matrix"],
        }
    return rows


def _print_summary_table(all_results: list[dict], title: str) -> str:
    """Print and return a formatted summary table."""
    if not all_results:
        print(f"\n[SKIP] {title}: no results to display.")
        return ""

    header = (
        f"\n{'='*90}\n"
        f"  {title}\n"
        f"{'='*90}\n"
        f"  {'Experiment':<35s} {'Model':<8s} "
        f"{'CV%':>7s} {'Test%':>7s} {'F1%':>7s}\n"
        f"  {'-'*35} {'-'*8} {'-'*7} {'-'*7} {'-'*7}"
    )
    lines = [header]
    for r in all_results:
        line = (
            f"  {r['experiment']:<35s} {r['model']:<8s} "
            f"{r['cv_mean']*100:6.1f}% "
            f"{r['test_accuracy']*100:6.1f}% "
            f"{r['test_f1_macro']*100:6.1f}%"
        )
        lines.append(line)
    lines.append(f"{'='*90}")

    table_str = "\n".join(lines)
    print(table_str)
    return table_str


# ============================================================================
# 3.  CROSS-ENVIRONMENT (LEGO)
# ============================================================================

def run_cross_environment(
    records: list[dict],
    classes: list[str],
    data_dir: Path,
    model_keys: list[str],
    common_kwargs: dict,
) -> list[dict]:
    """
    Leave-Environment-Group-Out:
    For each unique environment, train on all other environments and test on it.
    """
    from csi_ml_pipeline import build_dataset, train_and_evaluate

    env_groups = _group_by(records, "env")
    envs = sorted(env_groups.keys())

    if len(envs) < 2:
        print(f"\n[SKIP] Cross-environment requires ≥2 environments, found: {envs}")
        return []

    print(f"\n{'#'*70}")
    print(f"  CROSS-ENVIRONMENT (LEGO) — environments: {envs}")
    print(f"{'#'*70}")

    all_flat: list[dict] = []
    all_json: list[dict] = []

    for test_env in envs:
        train_envs = [e for e in envs if e != test_env]
        exp_name = f"Train={'_'.join(train_envs)} → Test={test_env}"
        print(f"\n{'─'*60}")
        print(f"  {exp_name}")
        print(f"{'─'*60}")

        # Build file lists per class
        train_files: dict[str, list] = {cls: [] for cls in classes}
        test_files:  dict[str, list] = {cls: [] for cls in classes}
        for rec in records:
            cls = rec["class"]
            if cls not in train_files:
                continue
            if rec["env"] == test_env:
                test_files[cls].append(rec["path"])
            else:
                train_files[cls].append(rec["path"])

        # Show split sizes
        for cls in classes:
            print(f"   [{cls}] train={len(train_files.get(cls, []))} "
                  f"test={len(test_files.get(cls, []))}")

        # Check we have data on both sides
        has_train = any(len(v) > 0 for v in train_files.values())
        has_test  = any(len(v) > 0 for v in test_files.values())
        if not has_train or not has_test:
            print(f"   [SKIP] Insufficient data for this split.")
            continue

        try:
            (X_train, X_train_orig, X_test,
             y_train, y_train_orig, y_test,
             train_groups_orig, le, pipeline,
             dataset_info) = build_dataset(
                data_dir=data_dir,
                classes=classes,
                train_files_override={
                    cls: [str(p) for p in paths]
                    for cls, paths in train_files.items()
                },
                test_files_override={
                    cls: [str(p) for p in paths]
                    for cls, paths in test_files.items()
                },
                **common_kwargs,
            )

            results = train_and_evaluate(
                X_train, X_train_orig, X_test,
                y_train, y_train_orig, y_test,
                train_groups_orig, le,
                cv_folds=config.CV_FOLDS,
                random_seed=common_kwargs.get("random_seed", config.RANDOM_SEED),
                target_model=model_keys,
                n_pca=dataset_info["n_pca"],
            )

            exp_config = {
                "type": "cross-env",
                "train_envs": train_envs,
                "test_env": test_env,
            }
            metrics = _extract_model_metrics(results, exp_name, exp_config)
            for mk, mval in metrics.items():
                all_flat.append({
                    "experiment": exp_name, "model": mk, **mval,
                })
            all_json.append({"experiment": exp_name, "metrics": metrics})

        except Exception as exc:
            print(f"   [ERROR] {exc}")

    _print_summary_table(all_flat, "CROSS-ENVIRONMENT RESULTS")
    return all_json


# ============================================================================
# 4.  CROSS-SUBJECT (LOSO)
# ============================================================================

def run_cross_subject(
    records: list[dict],
    classes: list[str],
    data_dir: Path,
    model_keys: list[str],
    common_kwargs: dict,
) -> list[dict]:
    """
    Leave-One-Subject-Out:
    For each unique subject, train on all other subjects and test on it.
    """
    from csi_ml_pipeline import build_dataset, train_and_evaluate

    subj_groups = _group_by(records, "subject")
    subjects = sorted(subj_groups.keys())

    if len(subjects) < 2:
        print(f"\n[SKIP] Cross-subject (LOSO) requires ≥2 subjects, found: {subjects}")
        print("       Record data from a second person to enable this experiment.")
        return []

    print(f"\n{'#'*70}")
    print(f"  CROSS-SUBJECT (LOSO) — subjects: {subjects}")
    print(f"{'#'*70}")

    all_flat: list[dict] = []
    all_json: list[dict] = []

    for test_subj in subjects:
        train_subjs = [s for s in subjects if s != test_subj]
        exp_name = f"Train={'_'.join(train_subjs)} → Test={test_subj}"
        print(f"\n{'─'*60}")
        print(f"  {exp_name}")
        print(f"{'─'*60}")

        train_files: dict[str, list] = {cls: [] for cls in classes}
        test_files:  dict[str, list] = {cls: [] for cls in classes}
        for rec in records:
            cls = rec["class"]
            if cls not in train_files:
                continue
            if rec["subject"] == test_subj:
                test_files[cls].append(rec["path"])
            else:
                train_files[cls].append(rec["path"])

        for cls in classes:
            print(f"   [{cls}] train={len(train_files.get(cls, []))} "
                  f"test={len(test_files.get(cls, []))}")

        has_train = any(len(v) > 0 for v in train_files.values())
        has_test  = any(len(v) > 0 for v in test_files.values())
        if not has_train or not has_test:
            print(f"   [SKIP] Insufficient data for this split.")
            continue

        try:
            (X_train, X_train_orig, X_test,
             y_train, y_train_orig, y_test,
             train_groups_orig, le, pipeline,
             dataset_info) = build_dataset(
                data_dir=data_dir,
                classes=classes,
                train_files_override={
                    cls: [str(p) for p in paths]
                    for cls, paths in train_files.items()
                },
                test_files_override={
                    cls: [str(p) for p in paths]
                    for cls, paths in test_files.items()
                },
                **common_kwargs,
            )

            results = train_and_evaluate(
                X_train, X_train_orig, X_test,
                y_train, y_train_orig, y_test,
                train_groups_orig, le,
                cv_folds=config.CV_FOLDS,
                random_seed=common_kwargs.get("random_seed", config.RANDOM_SEED),
                target_model=model_keys,
                n_pca=dataset_info["n_pca"],
            )

            exp_config = {
                "type": "cross-subject",
                "train_subjects": train_subjs,
                "test_subject": test_subj,
            }
            metrics = _extract_model_metrics(results, exp_name, exp_config)
            for mk, mval in metrics.items():
                all_flat.append({
                    "experiment": exp_name, "model": mk, **mval,
                })
            all_json.append({"experiment": exp_name, "metrics": metrics})

        except Exception as exc:
            print(f"   [ERROR] {exc}")

    _print_summary_table(all_flat, "CROSS-SUBJECT RESULTS")
    return all_json


# ============================================================================
# 5.  ABLATION STUDY
# ============================================================================

def run_ablation(
    classes: list[str],
    data_dir: Path,
    model_keys: list[str],
    common_kwargs: dict,
    pca_range: list[int],
    window_range: list[int],
    simulation_mode: bool = False,
) -> list[dict]:
    """
    Systematic ablation study over:
      - PCA components
      - Window size
      - Temporal diff on/off
      - Augmentation on/off
    """
    from csi_ml_pipeline import (
        build_dataset, train_and_evaluate,
        ALL_AUGMENT_TECHNIQUES, N_STATS,
    )

    print(f"\n{'#'*70}")
    print(f"  ABLATION STUDY")
    print(f"  PCA range   : {pca_range}")
    print(f"  Window range: {window_range}")
    print(f"  Models      : {model_keys}")
    print(f"{'#'*70}")

    # Define the ablation grid
    grid: list[dict] = []

    base_pca = common_kwargs.get("n_pca", config.N_PCA_COMPONENTS)
    base_window = common_kwargs.get("window_size", config.WINDOW_SIZE)
    base_cutoff = common_kwargs.get("cutoff", config.FILTER_CUTOFF_HZ)
    base_step = common_kwargs.get("step", config.PIPELINE_STEP_SIZE)

    # ---- Baseline (current config) ----
    grid.append({
        "name": "Baseline (full pipeline)",
        "use_diff": True,
        "augment": list(ALL_AUGMENT_TECHNIQUES),
        "pca": base_pca,
        "window_size": base_window,
        "cutoff": base_cutoff,
        "step": base_step,
    })

    # ---- PCA ablation ----
    for n_pca in pca_range:
        if n_pca == base_pca:
            continue
        grid.append({
            "name": f"PCA={n_pca}",
            "use_diff": True,
            "augment": list(ALL_AUGMENT_TECHNIQUES),
            "pca": n_pca,
            "window_size": base_window,
            "cutoff": base_cutoff,
            "step": base_step,
        })

    # ---- Window size ablation ----
    for ws in window_range:
        if ws == base_window:
            continue
        grid.append({
            "name": f"Window={ws}",
            "use_diff": True,
            "augment": list(ALL_AUGMENT_TECHNIQUES),
            "pca": base_pca,
            "window_size": ws,
            "cutoff": base_cutoff,
            # Adjust step to maintain ~50% overlap for new window sizes
            "step": max(1, ws // 2),
        })

    # ---- Filter (Cutoff) ablation ----
    for c in [None, 30, 50]:
        if c == base_cutoff:
            continue
        grid.append({
            "name": f"Filter={c if c is not None else 'No Filter'}",
            "use_diff": True,
            "augment": list(ALL_AUGMENT_TECHNIQUES),
            "pca": base_pca,
            "window_size": base_window,
            "cutoff": c,
            "step": base_step,
        })

    # ---- Step size (Overlap) ablation ----
    # 100 window size -> step 75 = 25% overlap, step 25 = 75% overlap
    for stp in [25, 75]:
        if stp == base_step:
            continue
        grid.append({
            "name": f"Step={stp}",
            "use_diff": True,
            "augment": list(ALL_AUGMENT_TECHNIQUES),
            "pca": base_pca,
            "window_size": base_window,
            "cutoff": base_cutoff,
            "step": stp,
        })

    # ---- Temporal diff ablation ----
    grid.append({
        "name": "No Temporal Diff",
        "use_diff": False,
        "augment": list(ALL_AUGMENT_TECHNIQUES),
        "pca": base_pca,
        "window_size": base_window,
        "cutoff": base_cutoff,
        "step": base_step,
    })

    # ---- Augmentation ablation ----
    grid.append({
        "name": "No Augmentation",
        "use_diff": True,
        "augment": [],
        "pca": base_pca,
        "window_size": base_window,
        "cutoff": base_cutoff,
        "step": base_step,
    })

    for tech in ALL_AUGMENT_TECHNIQUES:
        grid.append({
            "name": f"Augment: {tech} only",
            "use_diff": True,
            "augment": [tech],
            "pca": base_pca,
            "window_size": base_window,
            "cutoff": base_cutoff,
            "step": base_step,
        })

    # ---- Minimalistic Configuration ----
    grid.append({
        "name": "No Diff + No Augment + No Filter",
        "use_diff": False,
        "augment": [],
        "pca": base_pca,
        "window_size": base_window,
        "cutoff": None,
        "step": base_step,
    })

    print(f"\n  Total configurations: {len(grid)}")
    for i, g in enumerate(grid):
        print(f"    {i+1:2d}. {g['name']}")

    all_flat: list[dict] = []
    all_json: list[dict] = []

    for idx, cfg in enumerate(grid):
        exp_name = cfg["name"]
        print(f"\n{'─'*60}")
        print(f"  [{idx+1}/{len(grid)}] {exp_name}")
        print(f"{'─'*60}")

        kwargs = deepcopy(common_kwargs)
        fs_val = kwargs.get("pipeline_kwargs", {}).get("fs", config.SAMPLING_RATE)
        kwargs["pipeline_kwargs"] = {"fs": fs_val,
                                     "use_diff": cfg["use_diff"]}
        kwargs["augment_techniques"] = cfg["augment"] if cfg["augment"] else []
        kwargs["n_pca"] = cfg["pca"]
        kwargs["window_size"] = cfg["window_size"]
        kwargs["cutoff"] = cfg["cutoff"]
        kwargs["step"] = cfg["step"]

        try:
            (X_train, X_train_orig, X_test,
             y_train, y_train_orig, y_test,
             train_groups_orig, le, pipeline,
             dataset_info) = build_dataset(
                data_dir=data_dir,
                classes=classes,
                simulation_mode=simulation_mode,
                **kwargs,
            )

            if X_train.shape[0] == 0 or X_test.shape[0] == 0:
                print(f"   [SKIP] Empty dataset for {exp_name}")
                continue

            results = train_and_evaluate(
                X_train, X_train_orig, X_test,
                y_train, y_train_orig, y_test,
                train_groups_orig, le,
                cv_folds=config.CV_FOLDS,
                random_seed=kwargs.get("random_seed", config.RANDOM_SEED),
                target_model=model_keys,
                n_pca=dataset_info["n_pca"],
            )

            exp_config = {
                "type": "ablation",
                **cfg,
                "augment": cfg["augment"] if cfg["augment"] else "disabled",
            }
            metrics = _extract_model_metrics(results, exp_name, exp_config)
            for mk, mval in metrics.items():
                all_flat.append({
                    "experiment": exp_name, "model": mk, **mval,
                })
            all_json.append({"experiment": exp_name, "config": cfg, "metrics": metrics})

        except Exception as exc:
            print(f"   [ERROR] {exc}")
            import traceback
            traceback.print_exc()

    _print_summary_table(all_flat, "ABLATION STUDY RESULTS")
    return all_json


# ============================================================================
# 6.  REPORT GENERATION
# ============================================================================

def generate_report(
    cross_env_results: list[dict],
    cross_subj_results: list[dict],
    ablation_results: list[dict],
    output_dir: Path,
    records: list[dict],
    classes: list[str],
) -> Path:
    """Generate a thesis-ready Markdown report."""
    report_path = output_dir / "experiment_report.md"
    lines: list[str] = []

    # -- Header ---------------------------------------------------------------
    lines.append("# WiFi CSI HAR — Experiment Report")
    lines.append("")
    lines.append(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Classes**: {', '.join(classes)}")
    lines.append("")

    # -- Dataset overview -----------------------------------------------------
    env_groups  = _group_by(records, "env")
    subj_groups = _group_by(records, "subject")
    lines.append("## 1. Dataset Overview")
    lines.append("")
    lines.append(f"| Property | Value |")
    lines.append(f"|----------|-------|")
    lines.append(f"| Total recordings | {len(records)} |")
    lines.append(f"| Environments | {', '.join(sorted(env_groups.keys()))} ({len(env_groups)}) |")
    lines.append(f"| Subjects | {', '.join(sorted(subj_groups.keys()))} ({len(subj_groups)}) |")
    lines.append(f"| Classes | {', '.join(classes)} |")
    lines.append("")

    # Per-environment × class breakdown
    lines.append("### Recordings per Environment × Class")
    lines.append("")
    env_names = sorted(env_groups.keys())
    header = "| Environment | " + " | ".join(classes) + " | Total |"
    sep    = "|-------------|" + "|".join(["-----:"] * len(classes)) + "|------:|"
    lines.append(header)
    lines.append(sep)
    for env in env_names:
        recs = env_groups[env]
        counts = {cls: 0 for cls in classes}
        for r in recs:
            cls = r.get("class", r.get("activity", ""))
            if cls in counts:
                counts[cls] += 1
        row = f"| {env} | " + " | ".join(str(counts[c]) for c in classes) + f" | {sum(counts.values())} |"
        lines.append(row)
    lines.append("")

    # -- Cross-Environment Results --------------------------------------------
    if cross_env_results:
        lines.append("## 2. Cross-Environment Evaluation (LEGO)")
        lines.append("")
        lines.append("Leave-Environment-Group-Out: train on one environment, test on the other.")
        lines.append("This measures how well the model generalises across physical spaces.")
        lines.append("")
        _write_results_table(lines, cross_env_results)
    else:
        lines.append("## 2. Cross-Environment Evaluation (LEGO)")
        lines.append("")
        lines.append("> **Skipped** — requires ≥2 environments with data.")
        lines.append("")

    # -- Cross-Subject Results ------------------------------------------------
    if cross_subj_results:
        lines.append("## 3. Cross-Subject Evaluation (LOSO)")
        lines.append("")
        lines.append("Leave-One-Subject-Out: train on N-1 subjects, test on the remaining one.")
        lines.append("This measures person-independence of the HAR system.")
        lines.append("")
        _write_results_table(lines, cross_subj_results)
    else:
        lines.append("## 3. Cross-Subject Evaluation (LOSO)")
        lines.append("")
        lines.append("> **Skipped** — requires ≥2 subjects. Currently only 1 subject detected.")
        lines.append("> Record data from additional subjects to enable this experiment.")
        lines.append("")

    # -- Ablation Results -----------------------------------------------------
    if ablation_results:
        lines.append("## 4. Ablation Study")
        lines.append("")
        lines.append("Systematic evaluation of individual pipeline components.")
        lines.append("Each row disables or varies one parameter from the baseline configuration.")
        lines.append("")
        _write_ablation_table(lines, ablation_results)
    else:
        lines.append("## 4. Ablation Study")
        lines.append("")
        lines.append("> **Skipped** — ablation experiment was not requested.")
        lines.append("")

    # -- Conclusions ----------------------------------------------------------
    lines.append("## 5. Key Findings")
    lines.append("")
    _write_conclusions(lines, cross_env_results, cross_subj_results, ablation_results)

    report_text = "\n".join(lines)
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\n[SAVE] Report: {report_path}")
    return report_path


def _write_results_table(lines: list[str], json_results: list[dict]):
    """Write a markdown table from experiment JSON results."""
    lines.append("| Experiment | Model | CV (%) | Test Acc (%) | F1 Macro (%) |")
    lines.append("|------------|-------|-------:|-------------:|-------------:|")
    for exp in json_results:
        exp_name = exp["experiment"]
        for model_key, mval in exp["metrics"].items():
            lines.append(
                f"| {exp_name} | {model_key} "
                f"| {mval['cv_mean']*100:.1f} ± {mval['cv_std']*100:.1f} "
                f"| {mval['test_accuracy']*100:.1f} "
                f"| {mval['test_f1_macro']*100:.1f} |"
            )
    lines.append("")


def _write_ablation_table(lines: list[str], json_results: list[dict]):
    """Write an ablation-specific markdown table."""
    lines.append("| Configuration | Model | PCA | Window | Diff | Augment | CV (%) | Test Acc (%) | F1 (%) |")
    lines.append("|---------------|-------|----:|-------:|:----:|:-------:|-------:|-------------:|-------:|")
    for exp in json_results:
        cfg = exp.get("config", {})
        exp_name = exp["experiment"]
        pca_val = cfg.get("pca", "—")
        ws_val = cfg.get("window_size", "—")
        diff_val = "✓" if cfg.get("use_diff", True) else "✗"
        aug_val = "✓" if cfg.get("augment") and cfg["augment"] != "disabled" else "✗"
        for model_key, mval in exp["metrics"].items():
            lines.append(
                f"| {exp_name} | {model_key} "
                f"| {pca_val} | {ws_val} "
                f"| {diff_val} | {aug_val} "
                f"| {mval['cv_mean']*100:.1f} ± {mval['cv_std']*100:.1f} "
                f"| {mval['test_accuracy']*100:.1f} "
                f"| {mval['test_f1_macro']*100:.1f} |"
            )
    lines.append("")


def _write_conclusions(lines, cross_env, cross_subj, ablation):
    """Auto-generate key findings from the results."""
    findings = []

    if cross_env:
        # Find best and worst cross-env
        all_accs = []
        for exp in cross_env:
            for mk, mval in exp["metrics"].items():
                all_accs.append((exp["experiment"], mk, mval["test_accuracy"]))
        if all_accs:
            best = max(all_accs, key=lambda x: x[2])
            worst = min(all_accs, key=lambda x: x[2])
            delta = (best[2] - worst[2]) * 100
            findings.append(
                f"**Cross-Environment**: Best {best[2]*100:.1f}% "
                f"({best[0]}, {best[1]}), Worst {worst[2]*100:.1f}% "
                f"({worst[0]}, {worst[1]}). "
                f"Environment gap: **{delta:.1f}pp**."
            )

    if cross_subj:
        all_accs = []
        for exp in cross_subj:
            for mk, mval in exp["metrics"].items():
                all_accs.append((exp["experiment"], mk, mval["test_accuracy"]))
        if all_accs:
            mean_acc = np.mean([a[2] for a in all_accs]) * 100
            findings.append(
                f"**Cross-Subject**: Mean test accuracy across LOSO folds: "
                f"**{mean_acc:.1f}%**."
            )

    if ablation:
        # Find baseline and compare
        baseline_acc = None
        ablation_rows = []
        for exp in ablation:
            for mk, mval in exp["metrics"].items():
                row = (exp["experiment"], mk, mval["test_accuracy"])
                ablation_rows.append(row)
                if "baseline" in exp["experiment"].lower():
                    baseline_acc = mval["test_accuracy"]

        if baseline_acc is not None and ablation_rows:
            worst_abl = min(ablation_rows, key=lambda x: x[2])
            best_abl = max(ablation_rows, key=lambda x: x[2])
            findings.append(
                f"**Ablation Baseline**: {baseline_acc*100:.1f}%. "
                f"Removing components: worst = {worst_abl[0]} ({worst_abl[2]*100:.1f}%), "
                f"best = {best_abl[0]} ({best_abl[2]*100:.1f}%)."
            )
            # Impact of each component
            for row in ablation_rows:
                if row[0] == best_abl[0] and row[0] == worst_abl[0]:
                    continue
                delta = (row[2] - baseline_acc) * 100
                if abs(delta) > 1.0:
                    direction = "↑" if delta > 0 else "↓"
                    findings.append(
                        f"  - *{row[0]}*: {row[2]*100:.1f}% "
                        f"({direction}{abs(delta):.1f}pp vs baseline)"
                    )

    if not findings:
        findings.append("No experiments produced results to analyze.")

    for f in findings:
        lines.append(f"- {f}")
    lines.append("")


# ============================================================================
# 7.  CLI & MAIN
# ============================================================================

def parse_args() -> argparse.Namespace:
    defaults = config.get_script_defaults("experiment_runner")
    parser = argparse.ArgumentParser(
        description="WiFi CSI HAR — Cross-Subject, Cross-Environment & Ablation Experiments",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--data_dir", type=str, default=defaults["data_dir"])
    parser.add_argument("--classes", nargs="+", default=defaults["classes"])
    parser.add_argument(
        "--experiment", type=str, default=defaults["experiment"],
        choices=["cross-env", "cross-subject", "ablation", "all"],
        help="Which experiment(s) to run (default: all)",
    )
    parser.add_argument("--model", nargs="+", default=defaults["model"],
                        help="Model(s) to evaluate, e.g. rf et svm")
    parser.add_argument("-o", "--output-dir", default=defaults["output_dir"],
                        help="Directory for JSON results and report")
    parser.add_argument("--window_size", type=int, default=defaults["window_size"])
    parser.add_argument("--step", type=int, default=defaults["step"])
    parser.add_argument("--fs", type=float, default=defaults["fs"])
    parser.add_argument("--pca", type=int, default=defaults["pca"])
    parser.add_argument("--cutoff", type=float, default=defaults["cutoff"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--n_augments", type=int, default=defaults["n_augments"])
    parser.add_argument("--cv_folds", type=int, default=defaults["cv_folds"])
    config.add_bool_argument(
        parser, dest="simulate", default=defaults["simulate"],
        help="Use synthetic data (for testing the script itself).",
        positive_flags=["--simulate"], negative_flags=["--no-simulate"],
    )
    config.add_bool_argument(
        parser, dest="save", default=defaults["save"],
        help="Save JSON results and Markdown report to output dir.",
        positive_flags=["--save"], negative_flags=["--no-save"],
    )
    parser.add_argument("--ablation-pca", nargs="+", type=int,
                        default=defaults["ablation_pca"],
                        help="PCA values for ablation grid")
    parser.add_argument("--ablation-window", nargs="+", type=int,
                        default=defaults["ablation_window"],
                        help="Window sizes for ablation grid")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.monotonic()

    data_dir = config.resolve_project_path(args.data_dir)
    output_dir = config.resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  WiFi CSI HAR — EXPERIMENT RUNNER")
    print("=" * 70)
    print(f"  Experiment : {args.experiment}")
    print(f"  Models     : {args.model}")
    print(f"  Classes    : {args.classes}")
    print(f"  Data dir   : {data_dir}")
    print(f"  Output dir : {output_dir}")
    print(f"  Simulate   : {args.simulate}")
    print("=" * 70)

    # -- Common build_dataset kwargs ------------------------------------------
    # NOTE: cv_folds and fs are NOT build_dataset() parameters.
    #       They are passed separately to train_and_evaluate().
    from csi_ml_pipeline import ALL_AUGMENT_TECHNIQUES

    common_kwargs = {
        "pipeline_kwargs": {"fs": args.fs, "use_diff": True},
        "window_size":           args.window_size,
        "step":                  args.step,
        "augment_techniques":    list(ALL_AUGMENT_TECHNIQUES),
        "n_augments":            args.n_augments,
        "test_recording_ratio":  config.TEST_RATIO,
        "random_seed":           args.seed,
        "n_pca":                 args.pca,
        "cutoff":                args.cutoff,
    }

    # Extra params for train_and_evaluate (NOT for build_dataset)
    eval_kwargs = {
        "cv_folds":    args.cv_folds,
        "random_seed": args.seed,
        "fs":          args.fs,
    }

    # -- Discover recordings --------------------------------------------------
    if args.simulate:
        print("\n[INFO] Simulation mode — skipping recording discovery.")
        records = []
    else:
        records = discover_recordings(data_dir, args.classes)
        print(f"\n[OK] Discovered {len(records)} recordings")
        env_groups  = _group_by(records, "env")
        subj_groups = _group_by(records, "subject")
        for env, recs in sorted(env_groups.items()):
            print(f"   Environment '{env}': {len(recs)} recordings")
        for subj, recs in sorted(subj_groups.items()):
            print(f"   Subject '{subj}': {len(recs)} recordings")

    # -- Run experiments ------------------------------------------------------
    do_env     = args.experiment in ("cross-env", "all")
    do_subject = args.experiment in ("cross-subject", "all")
    do_abl     = args.experiment in ("ablation", "all")

    cross_env_results    = []
    cross_subj_results   = []
    ablation_results     = []

    if do_env and not args.simulate:
        cross_env_results = run_cross_environment(
            records, args.classes, data_dir, args.model, common_kwargs,
        )

    if do_subject and not args.simulate:
        cross_subj_results = run_cross_subject(
            records, args.classes, data_dir, args.model, common_kwargs,
        )

    if do_abl:
        ablation_results = run_ablation(
            args.classes, data_dir, args.model, common_kwargs,
            pca_range=args.ablation_pca,
            window_range=args.ablation_window,
            simulation_mode=args.simulate,
        )

    # -- Save results ---------------------------------------------------------
    elapsed = time.monotonic() - t0

    all_results = {
        "timestamp":    time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s":    round(elapsed, 1),
        "config": {
            "classes":     args.classes,
            "models":      args.model,
            "experiment":  args.experiment,
            "window_size": args.window_size,
            "pca":         args.pca,
            "cutoff":      args.cutoff,
            "seed":        args.seed,
            "simulate":    args.simulate,
        },
        "cross_env":     cross_env_results,
        "cross_subject": cross_subj_results,
        "ablation":      ablation_results,
    }

    if args.save or True:  # Always save
        json_path = output_dir / "experiment_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n[SAVE] JSON: {json_path}")

        report_path = generate_report(
            cross_env_results, cross_subj_results, ablation_results,
            output_dir, records, args.classes,
        )

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT RUNNER COMPLETE — {elapsed:.1f}s elapsed")
    print(f"{'='*70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
