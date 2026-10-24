"""Command-line interface: ``bcdrive <command>`` (or ``python -m bcdrive <command>``)."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TF's C++ info banner

from bcdrive.config import ConfigError, load_config  # noqa: E402
from bcdrive.utils.io import setup_logging, timestamp_id  # noqa: E402

log = logging.getLogger("bcdrive")


def cmd_env(args) -> int:
    import platform

    import engineio
    import keras
    import numpy
    import PIL
    import socketio
    import tensorflow as tf

    print(f"python      {sys.version.split()[0]} ({sys.executable})")
    print(f"platform    {platform.platform()}")
    print(f"tensorflow  {tf.__version__}   keras {keras.__version__}   numpy {numpy.__version__}   pillow {PIL.__version__}")
    print(f"socketio    {socketio.__version__}   engineio {engineio.__version__}")
    print(f"GPUs        {tf.config.list_physical_devices('GPU') or 'none (CPU training)'}")
    sim = Path("simulator/beta_simulator_windows/beta_simulator.exe")
    print(f"simulator   {sim.resolve() if sim.is_file() else 'not found at ' + str(sim)}")
    return 0


def cmd_synth(args) -> int:
    from bcdrive.data.synthetic import generate_dataset

    dirs = generate_dataset(Path(args.out), args.sessions, args.frames, seed=args.seed)
    print(f"wrote {len(dirs)} SYNTHETIC smoke-test session(s) to {Path(args.out).resolve()} "
          "(not simulator data)")
    return 0


def cmd_collect(args) -> int:
    from bcdrive.data.collect import SUGGESTED_TAGS, finalize_session, new_session

    if args.action == "new":
        tags = [t.strip() for t in (args.tags or "").split(",") if t.strip()]
        s = new_session(Path(args.root), args.name, tags, args.track, args.notes or "")
        print(f"created session {s.resolve()}")
        print("In the simulator: Training Mode -> press R (RECORD) -> select THIS folder -> drive -> press R to stop.")
        print(f"Then run:  bcdrive collect finalize \"{s}\"")
        print(f"(suggested tags: {', '.join(SUGGESTED_TAGS)})")
    else:
        meta = finalize_session(Path(args.session), load_config(args.config).data.default_columns)
        print(f"finalized {args.session}: {meta['usable_frames']} usable of {meta['rows']} rows, "
              f"issues={meta['issues'] or 'none'}, straight fraction={meta['fraction_straight_abs_lt_0.03']}")
    return 0


def cmd_inspect(args) -> int:
    from bcdrive.data.dataset import load_sessions
    from bcdrive.data.inspect import format_report, inspect_dataset

    cfg = load_config(args.config)
    ds = load_sessions(cfg.data, args.data)
    out = Path(args.out) if args.out else Path("artifacts/inspect") / timestamp_id()
    report = inspect_dataset(ds, cfg, out)
    print(format_report(report))
    print(f"report + plots: {out.resolve()}")
    return 0 if report["usable_frames"] else 1


def cmd_train(args) -> int:
    from bcdrive.training.train import run_training

    cfg = load_config(args.config)
    if args.epochs is not None and not args.resume:
        cfg.train.epochs = args.epochs
    if args.min_frames is not None:
        cfg.data.min_frames = args.min_frames
    exp = run_training(cfg, name=args.name, resume_dir=Path(args.resume) if args.resume else None,
                       extra_epochs=args.epochs if args.resume else None, data_paths=args.data,
                       evaluate_after=not args.no_eval)
    print(f"experiment: {exp.resolve()}")
    return 0


def cmd_evaluate(args) -> int:
    from bcdrive.evaluation.offline import evaluate_experiment

    r = evaluate_experiment(Path(args.experiment), args.split, args.model)
    m = r["metrics"]
    print(f"{r['experiment_id']} [{args.split}] n={m['n']} mse={m['mse']:.5f} mae={m['mae']:.4f} "
          f"r2={m['r2_vs_constant']}")
    for w in r["warnings"]:
        print(f"  - {w}")
    print(f"report: {(Path(args.experiment) / 'eval' / args.split / 'report.md').resolve()}")
    return 0


def cmd_compare(args) -> int:
    from bcdrive.evaluation.compare import compare_experiments

    out, _ = compare_experiments([Path(e) for e in args.experiments], args.split, Path(args.out))
    print((out / "comparison.md").read_text(encoding="utf-8"))
    print(f"\nsaved: {out.resolve()}")
    return 0


def cmd_drive(args) -> int:
    from bcdrive.control.runner import run_drive

    cfg = load_config(args.config) if args.config else None
    return run_drive(cfg, args.model, args.experiment, args.track, args.low_speed, args.name, args.port,
                     keyboard=not args.no_keyboard)


def cmd_runs(args) -> int:
    import json

    root = Path(args.runs_dir)
    rows = []
    for d in sorted(root.glob("*")):
        s, m = d / "summary.json", d / "run_meta.json"
        if s.is_file() and m.is_file():
            S, M = json.loads(s.read_text()), json.loads(m.read_text())
            lat = S.get("inference_latency_total", {})
            rows.append(f"| {d.name} | {M.get('experiment_id')} | {M.get('track')} | {S.get('autonomous_driving_seconds', 0) or 0:.0f} | "
                        f"{S.get('takeovers_operator_observed')} | {S.get('laps_operator_observed')} | "
                        f"{lat.get('mean_ms', float('nan')):.1f} / {lat.get('max_ms', float('nan')):.1f} |")
    print("| run | experiment | track | driving s | takeovers (obs.) | laps (obs.) | latency mean/max ms |")
    print("|---|---|---|---|---|---|---|")
    print("\n".join(rows) if rows else "| (no runs yet) | | | | | | |")
    return 0


def cmd_mock_sim(args) -> int:
    from bcdrive.data.driving_log import parse_driving_log
    from bcdrive.simulator.mock_client import jpeg_b64, replay

    frames = parse_driving_log(Path(args.session)).frames[: args.frames]
    print(f"TEST DOUBLE (not a simulator): replaying {len(frames)} frames to {args.url}")
    res = replay(args.url, [jpeg_b64(f.center) for f in frames], interval_s=args.interval)
    rt = sorted(res.round_trip_ms)
    print(f"sent {res.sent}, steer replies {len(res.replies)}, "
          f"round trip median {rt[len(rt) // 2] if rt else float('nan'):.1f} ms")
    return 0 if res.replies else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bcdrive", description="Behavioral cloning for the Udacity simulator")
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("env", help="print interpreter, framework and simulator info").set_defaults(fn=cmd_env)

    s = sub.add_parser("synth", help="write SYNTHETIC smoke-test sessions (not simulator data)")
    s.add_argument("--out", default="data/synthetic")
    s.add_argument("--sessions", type=int, default=3)
    s.add_argument("--frames", type=int, default=300)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_synth)

    c = sub.add_parser("collect", help="manage recording sessions")
    csub = c.add_subparsers(dest="action", required=True)
    cn = csub.add_parser("new", help="create a session folder to record into")
    cn.add_argument("--name", required=True)
    cn.add_argument("--tags", help="comma separated, e.g. center,recovery_left")
    cn.add_argument("--track", default="track1_lake")
    cn.add_argument("--notes")
    cn.add_argument("--root", default="data/raw")
    cf = csub.add_parser("finalize", help="validate a finished recording")
    cf.add_argument("session")
    cf.add_argument("--config")
    c.set_defaults(fn=cmd_collect)

    d = sub.add_parser("dataset", help="dataset tools")
    dsub = d.add_subparsers(dest="action", required=True)
    di = dsub.add_parser("inspect", help="report counts, issues, steering distribution, crop preview")
    di.add_argument("--config")
    di.add_argument("--data", nargs="+", help="session dirs or parents (default: data.sessions from config)")
    di.add_argument("--out")
    di.set_defaults(fn=cmd_inspect)

    t = sub.add_parser("train", help="train a model (new experiment or --resume)")
    t.add_argument("--config", default="configs/default.yaml")
    t.add_argument("--name")
    t.add_argument("--data", nargs="+")
    t.add_argument("--epochs", type=int, help="epochs (with --resume: additional epochs)")
    t.add_argument("--min-frames", type=int)
    t.add_argument("--resume", help="experiment directory to continue")
    t.add_argument("--no-eval", action="store_true")
    t.set_defaults(fn=cmd_train)

    e = sub.add_parser("evaluate", help="offline evaluation of an experiment on held-out frames")
    e.add_argument("--experiment", required=True)
    e.add_argument("--split", choices=["val", "test"], default="val")
    e.add_argument("--model", default="best.keras")
    e.set_defaults(fn=cmd_evaluate)

    cp = sub.add_parser("compare", help="compare experiments on the first one's held-out frames")
    cp.add_argument("experiments", nargs="+")
    cp.add_argument("--split", choices=["val", "test"], default="val")
    cp.add_argument("--out", default="artifacts/comparisons")
    cp.set_defaults(fn=cmd_compare)

    dr = sub.add_parser("drive", help="serve the model to the simulator (Autonomous Mode)")
    g = dr.add_mutually_exclusive_group(required=True)
    g.add_argument("--experiment")
    g.add_argument("--model")
    dr.add_argument("--config")
    dr.add_argument("--track", default="track1_lake")
    dr.add_argument("--low-speed", action="store_true")
    dr.add_argument("--name")
    dr.add_argument("--port", type=int)
    dr.add_argument("--no-keyboard", action="store_true")
    dr.set_defaults(fn=cmd_drive)

    r = sub.add_parser("runs", help="table of recorded autonomous runs")
    r.add_argument("--runs-dir", default="artifacts/runs")
    r.set_defaults(fn=cmd_runs)

    m = sub.add_parser("mock-sim", help="TEST DOUBLE: replay recorded frames to a running drive server")
    m.add_argument("session")
    m.add_argument("--url", default="http://127.0.0.1:4567")
    m.add_argument("--frames", type=int, default=100)
    m.add_argument("--interval", type=float, default=0.0)
    m.set_defaults(fn=cmd_mock_sim)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    try:
        return int(args.fn(args) or 0)
    except (ConfigError, FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        log.error("%s: %s", type(exc).__name__, exc)
        if args.log_level.upper() == "DEBUG":
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())
