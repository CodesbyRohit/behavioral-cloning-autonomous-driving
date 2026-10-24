"""Offline evaluation on held-out frames (never used for training).

Offline metrics answer "how well does the network imitate the recorded human steering
on frames it has not seen?". They do NOT measure whether the car can drive: small
per-frame errors compound in closed loop, and the network will meet states (drifting
off-centre) that a centred human never recorded. Use ``bcdrive drive`` for that.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from bcdrive.config import Config, EvalConfig, load_config
from bcdrive.data.dataset import Sample, frames_to_samples, load_sessions
from bcdrive.data.driving_log import FrameRecord
from bcdrive.data.split import frames_from_manifest
from bcdrive.utils.io import read_json, write_json

log = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, cfg: EvalConfig) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError(f"need equal, non-empty arrays (got {y_true.shape} and {y_pred.shape})")
    if not np.all(np.isfinite(y_pred)):
        raise ValueError("predictions contain NaN/inf")
    res = y_pred - y_true
    var = float(np.var(y_true))
    mse = float(np.mean(res ** 2))
    out = {
        "n": int(y_true.size),
        "mse": mse,
        "mae": float(np.mean(np.abs(res))),
        "rmse": float(np.sqrt(mse)),
        "bias_mean_residual": float(np.mean(res)),
        "max_abs_error": float(np.max(np.abs(res))),
        "true_std": float(np.std(y_true)),
        "pred_std": float(np.std(y_pred)),
        "pearson_r": (float(np.corrcoef(y_true, y_pred)[0, 1])
                      if np.std(y_true) > 0 and np.std(y_pred) > 0 else None),
        # A constant predictor (the mean of this split) achieves MSE == var(y_true).
        "baseline_constant_mse": var,
        "baseline_zero_mse": float(np.mean(y_true ** 2)),
        "r2_vs_constant": (1.0 - mse / var) if var > 0 else None,
    }
    edges = np.asarray(cfg.bin_edges)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (y_true >= lo) & ((y_true < hi) if hi < edges[-1] else (y_true <= hi))
        n = int(m.sum())
        row = {"range": [float(lo), float(hi)], "n": n, "sufficient": n >= cfg.min_bin_count}
        if n:
            row.update(mse=float(np.mean(res[m] ** 2)), mae=float(np.mean(np.abs(res[m]))),
                       mean_true=float(y_true[m].mean()), mean_pred=float(y_pred[m].mean()))
        bins.append(row)
    out["by_bin"] = bins
    thr = cfg.straight_threshold
    groups = {"left": y_true < -thr, "straight": np.abs(y_true) <= thr, "right": y_true > thr}
    out["by_direction"] = {}
    for g, m in groups.items():
        n = int(m.sum())
        out["by_direction"][g] = {"n": n, "sufficient": n >= cfg.min_bin_count,
                                  **({"mse": float(np.mean(res[m] ** 2)), "mae": float(np.mean(np.abs(res[m])))}
                                     if n else {})}
    out["straight_fraction"] = float(groups["straight"].mean())
    return out


def diagnose(metrics: dict, history: list[dict], synthetic: bool) -> list[str]:
    """Plain-language warnings derived only from measured numbers."""
    w = []
    if synthetic:
        w.append("SYNTHETIC smoke-test data: these numbers verify the software, not driving ability.")
    if metrics["n"] < 1000:
        w.append(f"Only {metrics['n']} evaluation frames: metrics are noisy; record more data.")
    if metrics["straight_fraction"] > 0.6:
        w.append(f"{metrics['straight_fraction']:.0%} of evaluation frames are straight driving; "
                 "curve performance is under-represented in the averages. Check by_bin / by_direction.")
    r2 = metrics.get("r2_vs_constant")
    if r2 is not None and r2 < 0.1:
        w.append(f"R^2 vs a constant predictor is {r2:.2f}: the model is barely better than predicting the "
                 "mean steering (underfitting or collapse towards zero).")
    if metrics["true_std"] > 0 and metrics["pred_std"] < 0.3 * metrics["true_std"]:
        w.append("Predictions are compressed towards zero (pred_std << true_std): typical of an "
                 "imbalanced, straight-dominated training set.")
    insufficient = [b["range"] for b in metrics["by_bin"] if 0 < b["n"] < 20 or b["n"] == 0]
    if insufficient:
        w.append(f"{len(insufficient)} steering bin(s) have too few samples to judge; see by_bin.")
    vals = [r.get("val_loss") for r in history if r.get("val_loss") is not None]
    if len(vals) >= 3:
        best_i = int(np.argmin(vals))
        if best_i < len(vals) - 2 and vals[-1] > 1.2 * vals[best_i]:
            w.append(f"Validation loss rose from {vals[best_i]:.4f} (epoch {best_i + 1}) to {vals[-1]:.4f}: "
                     "overfitting after the best epoch (best.keras keeps the best weights).")
    w.append("A low validation loss does not prove the car can drive; only a simulator run does.")
    return w


def frames_for_split(exp_dir: Path, split: str) -> tuple[Config, list[FrameRecord], dict]:
    cfg = load_config(exp_dir / "config.yaml")
    manifest = read_json(exp_dir / "dataset_manifest.json")
    dataset = load_sessions(cfg.data, [s["path"] for s in manifest["sessions"]])
    parts = frames_from_manifest(dataset.frames, read_json(exp_dir / "splits.json"))
    frames = getattr(parts, split)
    if not frames:
        raise ValueError(f"experiment {exp_dir.name} has no '{split}' frames")
    return cfg, frames, manifest


def predict_samples(model_path: Path, samples: list[Sample], cfg: Config) -> np.ndarray:
    import keras
    from bcdrive.training.sequence import DrivingSequence

    model = keras.models.load_model(model_path)
    seq = DrivingSequence(samples, cfg.train.batch_size, cfg.preprocess,
                          workers=cfg.train.workers, use_multiprocessing=False)
    return np.asarray(model.predict(seq, verbose=0), dtype=np.float64).ravel()


def evaluate_experiment(exp_dir: str | Path, split: str = "val", model_file: str = "best.keras",
                        out_dir: Path | None = None) -> dict:
    exp_dir = Path(exp_dir)
    cfg, frames, manifest = frames_for_split(exp_dir, split)
    samples = frames_to_samples(frames, None, use_side_cameras=False)
    y_true = np.array([s.steering for s in samples])
    y_pred = predict_samples(exp_dir / model_file, samples, cfg)
    metrics = compute_metrics(y_true, y_pred, cfg.eval)
    history = read_json(exp_dir / "history.json") if (exp_dir / "history.json").is_file() else []
    warnings = diagnose(metrics, history, manifest.get("synthetic_data", False))
    result = {"experiment_id": exp_dir.name, "split": split, "model_file": model_file,
              "synthetic_data": manifest.get("synthetic_data", False), "metrics": metrics, "warnings": warnings}
    out_dir = out_dir or exp_dir / "eval" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "eval_metrics.json", result)
    np.savetxt(out_dir / "predictions.csv",
               np.column_stack([y_true, y_pred]), delimiter=",", header="true,pred", comments="", fmt="%.6f")
    from bcdrive.evaluation.plots import save_eval_plots
    save_eval_plots(out_dir, y_true, y_pred, metrics, history, title=f"{exp_dir.name} [{split}]")
    (out_dir / "report.md").write_text(render_report(result), encoding="utf-8")
    log.info("%s/%s: mse=%.5f mae=%.4f r2=%s -> %s", exp_dir.name, split, metrics["mse"], metrics["mae"],
             None if metrics["r2_vs_constant"] is None else round(metrics["r2_vs_constant"], 3), out_dir)
    return result


def _fmt(v, nd=4) -> str:
    return "—" if v is None else f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def render_report(result: dict) -> str:
    m = result["metrics"]
    lines = [
        f"# Offline evaluation — `{result['experiment_id']}` / {result['split']}",
        "",
        "> Offline imitation metrics on held-out frames. They are **not** a measure of closed-loop "
        "driving performance.",
        "",
    ]
    if result["synthetic_data"]:
        lines += ["> **SYNTHETIC SMOKE-TEST DATA** — not simulator recordings.", ""]
    lines += [
        "| metric | value |", "|---|---|",
        f"| frames | {m['n']} |",
        f"| MSE | {_fmt(m['mse'], 5)} |",
        f"| MAE | {_fmt(m['mae'])} |",
        f"| RMSE | {_fmt(m['rmse'])} |",
        f"| bias (mean pred − true) | {_fmt(m['bias_mean_residual'])} |",
        f"| Pearson r | {_fmt(m['pearson_r'], 3)} |",
        f"| R² vs constant predictor | {_fmt(m['r2_vs_constant'], 3)} |",
        f"| constant-predictor MSE | {_fmt(m['baseline_constant_mse'], 5)} |",
        f"| always-zero MSE | {_fmt(m['baseline_zero_mse'], 5)} |",
        f"| std(true) / std(pred) | {_fmt(m['true_std'])} / {_fmt(m['pred_std'])} |",
        "",
        "## By direction",
        "", "| group | n | MSE | MAE |", "|---|---|---|---|",
    ]
    for g, d in m["by_direction"].items():
        flag = "" if d["sufficient"] else " (too few)"
        lines.append(f"| {g} | {d['n']}{flag} | {_fmt(d.get('mse'), 5)} | {_fmt(d.get('mae'))} |")
    lines += ["", "## By steering bin", "", "| true steering range | n | MSE | MAE | mean true | mean pred |",
              "|---|---|---|---|---|---|"]
    for b in m["by_bin"]:
        flag = "" if b["sufficient"] else " (too few)"
        lines.append(f"| [{b['range'][0]:+.2f}, {b['range'][1]:+.2f}] | {b['n']}{flag} | {_fmt(b.get('mse'), 5)} | "
                     f"{_fmt(b.get('mae'))} | {_fmt(b.get('mean_true'), 3)} | {_fmt(b.get('mean_pred'), 3)} |")
    lines += ["", "## Diagnostics", ""] + [f"- {w}" for w in result["warnings"]]
    lines += ["", "## Plots", "", "![pred vs true](pred_vs_true.png)", "", "![timeline](timeline.png)", "",
              "![residuals](residuals.png)", "", "![per-bin error](per_bin_mae.png)", "", "![loss](loss_curves.png)", ""]
    return "\n".join(lines)
