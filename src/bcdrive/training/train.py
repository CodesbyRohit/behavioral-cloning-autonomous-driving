"""Training pipeline: dataset -> split -> balanced/augmented batches -> CNN -> checkpoints.

Every run writes to ``artifacts/experiments/<experiment_id>/``:
    config.yaml            fully resolved configuration used for this run
    dataset_manifest.json  provenance: session paths, CSV hashes, issues, synthetic flag
    splits.json            exact frame membership of train/val/test (reused by eval/resume)
    balance_report.json    steering histogram before/after balancing (+ .png)
    model_summary.txt      layer table and parameter count
    history.csv/.json      per-epoch losses and wall-clock durations
    best.keras             lowest validation loss so far (never overwritten by a worse model)
    last.keras             most recent epoch (used for --resume)
    metrics.json           machine-readable training summary
    eval/<split>/          offline evaluation of best.keras
    train.log              full log
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import keras
import numpy as np

from bcdrive.config import Config, load_config, save_config
from bcdrive.data.balance import balance_report
from bcdrive.data.dataset import DatasetError, frames_to_samples, load_sessions, provenance
from bcdrive.data.split import SplitResult, frames_from_manifest, split_frames
from bcdrive.models.nvidia import build_model, model_summary_text, trainable_params
from bcdrive.preprocessing.augment import Augmenter
from bcdrive.preprocessing.image import input_shape
from bcdrive.training.sequence import DrivingSequence
from bcdrive.utils.io import add_file_log, read_json, timestamp_id, unique_dir, write_json

log = logging.getLogger(__name__)


class EpochRecorder(keras.callbacks.Callback):
    """Records per-epoch wall time and rewrites history.json after every epoch, so an
    interrupted run still leaves a complete record of the epochs it finished."""

    def __init__(self, path: Path, previous: list[dict]):
        super().__init__()
        self.path = path
        self.rows = list(previous)
        self._t0 = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        self._t0 = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        row = {"epoch": epoch + 1, "seconds": round(time.perf_counter() - self._t0, 2)}
        row.update({k: float(v) for k, v in (logs or {}).items()})
        self.rows.append(row)
        write_json(self.path, self.rows)
        log.info("epoch %d done in %.1fs: %s", epoch + 1, row["seconds"],
                 ", ".join(f"{k}={v:.5f}" for k, v in row.items() if k not in ("epoch", "seconds")))


def _save_balance_plot(report: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    edges = np.array(report["bin_edges"])
    centers, width = (edges[:-1] + edges[1:]) / 2, np.diff(edges)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(centers, report["counts_before"], width=width * 0.9, color="#c8ccd4", label="before balancing")
    ax.bar(centers, report["counts_after"], width=width * 0.55, color="#2f6fdb", label="per epoch after balancing")
    ax.set_xlabel("steering (negative = left, positive = right)")
    ax.set_ylabel("training samples")
    ax.set_title("Training steering distribution (incl. side cameras)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _seed_everything(cfg: Config) -> None:
    keras.utils.set_random_seed(cfg.train.seed)
    if cfg.train.deterministic_ops:
        import tensorflow as tf
        tf.config.experimental.enable_op_determinism()


def run_training(cfg: Config, name: str | None = None, resume_dir: Path | None = None,
                 extra_epochs: int | None = None, data_paths: list[str] | None = None,
                 evaluate_after: bool = True) -> Path:
    """Train (or resume) an experiment and return its directory."""
    previous_history: list[dict] = []
    if resume_dir is not None:
        exp_dir = Path(resume_dir)
        if not (exp_dir / "last.keras").is_file():
            raise FileNotFoundError(f"cannot resume: {exp_dir / 'last.keras'} does not exist")
        cfg = load_config(exp_dir / "config.yaml")
        previous_history = read_json(exp_dir / "history.json") if (exp_dir / "history.json").is_file() else []
    else:
        exp_dir = unique_dir(Path(cfg.train.output_dir), timestamp_id(name))
    handler = add_file_log(exp_dir / "train.log")
    try:
        return _train(cfg, exp_dir, resume_dir is not None, previous_history, extra_epochs, data_paths, evaluate_after)
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def _train(cfg: Config, exp_dir: Path, resuming: bool, previous_history: list[dict],
           extra_epochs: int | None, data_paths: list[str] | None, evaluate_after: bool) -> Path:
    log.info("experiment directory: %s (resume=%s)", exp_dir, resuming)
    _seed_everything(cfg)

    # 1-2. validate + inspect dataset
    if resuming:
        manifest = read_json(exp_dir / "dataset_manifest.json")
        data_paths = [s["path"] for s in manifest["sessions"]]
    dataset = load_sessions(cfg.data, data_paths)
    for r in dataset.reports:
        log.info("session %s: %d rows, %d usable frames, issues=%s", r.session, r.rows, len(r.frames), dict(r.issues))
    frames = dataset.frames
    if len(frames) < cfg.data.min_frames:
        raise DatasetError(
            f"only {len(frames)} usable frames (data.min_frames = {cfg.data.min_frames}). "
            "Record more driving, or lower data.min_frames for a deliberate small experiment."
        )
    prov = provenance(dataset)
    synthetic = any(p["synthetic"] for p in prov)
    if synthetic:
        log.warning("dataset contains SYNTHETIC smoke-test sessions: results say nothing about driving ability")

    # 3. split (before any augmentation); a resumed run reuses the stored split exactly
    split: SplitResult = (frames_from_manifest(frames, read_json(exp_dir / "splits.json")) if resuming
                          else split_frames(frames, cfg.split))
    log.info("split: %s", {k: v for k, v in split.summary().items() if not k.endswith("sessions")})

    use_side = cfg.augment.enabled and cfg.augment.use_side_cameras
    train_samples = frames_to_samples(split.train, cfg.augment, use_side_cameras=use_side)
    val_samples = frames_to_samples(split.val, None, use_side_cameras=False)  # center only, like inference
    steer = np.array([s.steering for s in train_samples])
    bal = balance_report(steer, cfg.balance, np.random.default_rng(cfg.train.seed))
    log.info("balancing: %d training samples -> %d per epoch (cap/bin=%s)",
             bal["samples_before"], bal["samples_after_per_epoch"], bal["cap_per_bin"])

    if not resuming:
        save_config(cfg, exp_dir / "config.yaml")
        write_json(exp_dir / "splits.json", split.to_manifest())
        write_json(exp_dir / "dataset_manifest.json", {
            "sessions": prov,
            "synthetic_data": synthetic,
            "split": split.summary(),
            "use_side_cameras": use_side,
            "train_samples_including_side_cameras": len(train_samples),
            "val_samples": len(val_samples),
        })
        write_json(exp_dir / "balance_report.json", bal)
        _save_balance_plot(bal, exp_dir / "balance_report.png")

    workers = dict(workers=cfg.train.workers, use_multiprocessing=False, max_queue_size=8)
    train_seq = DrivingSequence(train_samples, cfg.train.batch_size, cfg.preprocess,
                                augmenter=Augmenter(cfg.augment) if cfg.augment.enabled else None,
                                balance=cfg.balance, shuffle=True, seed=cfg.train.seed, **workers)
    val_seq = DrivingSequence(val_samples, cfg.train.batch_size, cfg.preprocess, **workers)

    # 4. model
    if resuming:
        model = keras.models.load_model(exp_dir / "last.keras")
        log.info("resumed weights + optimizer state from last.keras")
    else:
        model = build_model(input_shape(cfg.preprocess), cfg.model)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg.train.learning_rate),
                      loss="mse", metrics=["mae"])
        (exp_dir / "model_summary.txt").write_text(model_summary_text(model), encoding="utf-8")
    n_params = trainable_params(model)
    log.info("model %s: %d trainable parameters", model.name, n_params)

    prev_best = min((r["val_loss"] for r in previous_history if "val_loss" in r), default=None)
    initial_epoch = len(previous_history)
    epochs = initial_epoch + (extra_epochs if extra_epochs is not None else cfg.train.epochs)
    recorder = EpochRecorder(exp_dir / "history.json", previous_history)
    callbacks = [
        keras.callbacks.ModelCheckpoint(str(exp_dir / "best.keras"), monitor="val_loss", mode="min",
                                        save_best_only=True, initial_value_threshold=prev_best),
        keras.callbacks.ModelCheckpoint(str(exp_dir / "last.keras")),
        keras.callbacks.EarlyStopping(monitor="val_loss", mode="min",
                                      patience=cfg.train.early_stopping_patience),
        keras.callbacks.CSVLogger(str(exp_dir / "history.csv"), append=resuming),
        keras.callbacks.TerminateOnNaN(),
        recorder,
    ]

    # 5-7. train
    t0 = time.perf_counter()
    model.fit(train_seq, validation_data=val_seq, epochs=epochs, initial_epoch=initial_epoch,
              callbacks=callbacks, verbose=2)
    wall = time.perf_counter() - t0

    rows = recorder.rows
    new_rows = rows[initial_epoch:]
    if not new_rows:
        raise RuntimeError("training finished without completing an epoch")
    if not (exp_dir / "best.keras").is_file():
        raise RuntimeError("no best.keras was written (validation loss never improved or was NaN)")
    best = min(rows, key=lambda r: r.get("val_loss", float("inf")))
    metrics = {
        "experiment_id": exp_dir.name,
        "architecture": cfg.model.architecture,
        "trainable_params": n_params,
        "epochs_completed": len(rows),
        "stopped_early": len(rows) < epochs,
        "best_epoch": best["epoch"],
        "best_val_loss_mse": best.get("val_loss"),
        "best_val_mae": best.get("val_mae"),
        "final_train_loss_mse": rows[-1].get("loss"),
        "final_val_loss_mse": rows[-1].get("val_loss"),
        "mean_epoch_seconds": float(np.mean([r["seconds"] for r in rows])),
        "this_session_wall_seconds": round(wall, 1),
        "train_samples_per_epoch": train_seq.epoch_size,
        "val_samples": len(val_samples),
        "synthetic_data": synthetic,
        "note": "train loss is measured on augmented data and with dropout active; it is not "
                "directly comparable to val loss.",
    }
    write_json(exp_dir / "metrics.json", metrics)
    log.info("best val_loss %.5f at epoch %d -> %s", best.get("val_loss", float("nan")), best["epoch"],
             exp_dir / "best.keras")

    if evaluate_after:
        from bcdrive.evaluation.offline import evaluate_experiment
        for which in ("val", "test"):
            if (which == "val" and split.val) or (which == "test" and split.test):
                evaluate_experiment(exp_dir, which)
    return exp_dir
