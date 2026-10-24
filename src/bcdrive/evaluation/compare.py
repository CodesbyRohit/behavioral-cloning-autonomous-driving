"""Compare checkpoints on exactly the same held-out frames.

The frames come from the FIRST experiment's split. If the second experiment was
trained on a different dataset, some of those frames may have been in its training set;
this is detected (by session + frame index) and reported as a leakage warning.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bcdrive.data.dataset import frames_to_samples
from bcdrive.evaluation.offline import compute_metrics, frames_for_split, predict_samples
from bcdrive.utils.io import read_json, timestamp_id, unique_dir, write_json


def compare_experiments(exp_dirs: list[Path], split: str, out_root: Path) -> tuple[Path, dict]:
    ref = exp_dirs[0]
    cfg, frames, _ = frames_for_split(ref, split)
    samples = frames_to_samples(frames, None, use_side_cameras=False)
    y_true = np.array([s.steering for s in samples])
    held_out = {(f.session, f.index) for f in frames}
    rows = []
    for d in exp_dirs:
        train_ids = {tuple(x) for x in read_json(d / "splits.json")["train"]}
        overlap = len(held_out & train_ids)
        metrics = compute_metrics(y_true, predict_samples(d / "best.keras", samples, cfg), cfg.eval)
        m = read_json(d / "metrics.json") if (d / "metrics.json").is_file() else {}
        rows.append({
            "experiment": d.name, "architecture": m.get("architecture"), "best_epoch": m.get("best_epoch"),
            "mse": metrics["mse"], "mae": metrics["mae"], "r2_vs_constant": metrics["r2_vs_constant"],
            "left_mae": metrics["by_direction"]["left"].get("mae"),
            "straight_mae": metrics["by_direction"]["straight"].get("mae"),
            "right_mae": metrics["by_direction"]["right"].get("mae"),
            "eval_frames_seen_in_training": overlap,
        })
    out = unique_dir(out_root, timestamp_id("compare"))
    result = {"reference_split": f"{ref.name}/{split}", "frames": int(y_true.size), "rows": rows}
    write_json(out / "comparison.json", result)
    (out / "comparison.md").write_text(render(result), encoding="utf-8")
    return out, result


def render(result: dict) -> str:
    f = lambda v: "—" if v is None else f"{v:.4f}"  # noqa: E731
    lines = [f"# Checkpoint comparison on `{result['reference_split']}` ({result['frames']} frames)", "",
             "| experiment | arch | best epoch | MSE | MAE | R² | left MAE | straight MAE | right MAE | leaked frames |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in result["rows"]:
        lines.append(f"| {r['experiment']} | {r['architecture']} | {r['best_epoch']} | {f(r['mse'])} | {f(r['mae'])} | "
                     f"{f(r['r2_vs_constant'])} | {f(r['left_mae'])} | {f(r['straight_mae'])} | {f(r['right_mae'])} | "
                     f"{r['eval_frames_seen_in_training']} |")
    if any(r["eval_frames_seen_in_training"] for r in result["rows"]):
        lines += ["", "**Warning:** some evaluation frames were in a model's training set; its numbers are optimistic."]
    lines += ["", "Offline metrics only. Confirm any ranking with simulator runs."]
    return "\n".join(lines)
