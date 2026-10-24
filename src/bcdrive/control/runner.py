"""``bcdrive drive``: load the model, start the simulator adapter and record the run.

Each run writes ``artifacts/runs/<run_id>/``:
    run_meta.json   model, experiment, track, control settings
    frames.csv      every frame: speed, raw prediction, sent commands, latencies, status
    events.jsonl    connect/disconnect, emergency stops, operator marks
    summary.json    latency, steering statistics, takeovers, laps, autonomy score (if measurable)
    drive.log       full log
"""

from __future__ import annotations

import logging
from pathlib import Path

from bcdrive.config import Config, load_config
from bcdrive.control.drive import DriveSession, KerasPredictor
from bcdrive.control.operator import HELP, OperatorConsole
from bcdrive.simulator.udacity import UdacitySimulatorAdapter
from bcdrive.utils.io import add_file_log, read_json, timestamp_id, unique_dir

log = logging.getLogger(__name__)


def resolve_model(model: str | None, experiment: str | None) -> tuple[Path, Path | None]:
    if experiment:
        exp = Path(experiment)
        return exp / "best.keras", exp
    if model:
        p = Path(model)
        return p, (p.parent if (p.parent / "config.yaml").is_file() else None)
    raise ValueError("pass --experiment DIR or --model FILE")


def run_drive(cfg: Config | None, model: str | None, experiment: str | None, track: str, low_speed: bool,
              name: str | None = None, port: int | None = None, keyboard: bool = True) -> int:
    model_path, exp_dir = resolve_model(model, experiment)
    if exp_dir is not None:
        # The model must be fed exactly the preprocessing it was trained with.
        exp_cfg = load_config(exp_dir / "config.yaml")
        if cfg is None:
            cfg = exp_cfg
        elif cfg.preprocess != exp_cfg.preprocess:
            log.warning("--config preprocessing differs from the experiment's; using the experiment's")
        cfg.preprocess = exp_cfg.preprocess
    elif cfg is None:
        cfg = load_config(None)
        log.warning("model is not inside an experiment folder; assuming default preprocessing")

    run_dir = unique_dir(Path(cfg.simulator.runs_dir), timestamp_id(name or "drive"))
    handler = add_file_log(run_dir / "drive.log")
    try:
        predictor = KerasPredictor(model_path)
        synthetic = False
        if exp_dir is not None and (exp_dir / "metrics.json").is_file():
            synthetic = bool(read_json(exp_dir / "metrics.json").get("synthetic_data"))
        if synthetic:
            log.warning("this model was trained on SYNTHETIC smoke-test data; it is not expected to drive")
        session = DriveSession(predictor, cfg, run_dir, low_speed=low_speed, meta={
            "model_path": str(model_path.resolve()),
            "experiment_id": exp_dir.name if exp_dir else None,
            "trained_on_synthetic_data": synthetic,
            "simulator": "udacity-term1-v2",
            "track": track,
        })
        adapter = UdacitySimulatorAdapter(cfg.simulator.host, port or cfg.simulator.port,
                                          cfg.simulator.connect_timeout_s, cfg.simulator.stale_telemetry_s,
                                          cfg.simulator.exit_on_disconnect)
        console = OperatorConsole(session, on_quit=adapter.stop) if keyboard else None
        if console:
            print(HELP, flush=True)
            console.start()
        log.info("run directory: %s (target speed %.1f mph)", run_dir, session.target_speed)
        rc = adapter.serve(session.handle_telemetry, session.on_connect, session.on_disconnect)
        if console:
            console.stop()
        summary = session.close({"exit_code": rc, "adapter_stats": adapter.stats, "run_id": run_dir.name})
        log.info("run summary: frames=%s takeovers=%s laps=%s latency=%s", summary["frames_driven"],
                 summary["takeovers_operator_observed"], summary["laps_operator_observed"],
                 summary["inference_latency_total"])
        return rc
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
