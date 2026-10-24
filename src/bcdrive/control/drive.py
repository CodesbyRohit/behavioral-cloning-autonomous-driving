"""The real-time driving policy and run recorder.

``DriveSession.handle_telemetry`` is the whole closed loop for one frame:

    base64 JPEG -> decode_image_base64 -> preprocess (the SAME function as training)
    -> CNN -> SteeringFilter -> PI throttle -> validate_command -> simulator

It is simulator-agnostic (it only sees ``Telemetry``) and does no networking, so it is
tested with a mock adapter. Operator actions (emergency stop, takeover marks, lap marks)
arrive from another thread and are serialised with a lock.
"""

from __future__ import annotations

import csv
import json
import logging
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

from bcdrive.config import Config
from bcdrive.control.controller import (ControlCommand, PIThrottleController, SteeringFilter, UnsafeCommandError,
                                        brake_command)
from bcdrive.evaluation.autonomy import latency_stats, steering_stats, summarize_events
from bcdrive.preprocessing.image import ImageError, decode_image_base64, preprocess
from bcdrive.simulator.base import Telemetry
from bcdrive.utils.io import write_json

log = logging.getLogger(__name__)

Predictor = Callable[[np.ndarray], float]  # preprocessed (66, 200, 3) -> raw steering

MAX_CONSECUTIVE_INVALID = 15  # ~1 s of bad frames -> automatic emergency stop
STUCK_SECONDS = 5.0


class KerasPredictor:
    """Loads the model once and runs single-frame inference."""

    def __init__(self, model_path: str | Path):
        import keras

        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"model file not found: {path}")
        try:
            self.model = keras.models.load_model(path, compile=False)
        except Exception as exc:
            raise RuntimeError(f"could not load model {path}: {exc}") from exc
        shape = tuple(self.model.input_shape[1:])
        self.input_shape = shape
        self(np.zeros(shape, dtype=np.float32))  # warm-up: first call builds the graph

    def __call__(self, x: np.ndarray) -> float:
        import keras
        y = self.model(x[None, ...], training=False)
        return float(keras.ops.convert_to_numpy(y).reshape(-1)[0])


FRAME_FIELDS = ["t", "frame", "speed", "sim_steering", "sim_throttle", "raw_pred", "steering_cmd",
                "throttle_cmd", "decode_ms", "preprocess_ms", "inference_ms", "total_ms", "status"]


class DriveSession:
    def __init__(self, predictor: Predictor, cfg: Config, run_dir: Path | None = None,
                 low_speed: bool = False, meta: dict | None = None):
        self.predictor = predictor
        self.cfg = cfg
        self.target_speed = cfg.control.low_speed_target if low_speed else cfg.control.target_speed
        self.steer_filter = SteeringFilter(cfg.control)
        self.throttle = PIThrottleController(cfg.control, self.target_speed)
        self.lock = threading.Lock()
        self.t0 = time.monotonic()
        self.frame = 0
        self.invalid_streak = 0
        self.estopped = False
        self.quit_requested = False
        self.events: list[dict] = []
        self.latencies: list[float] = []
        self.steering_log: list[float] = []
        self.steering_t: list[float] = []
        self.drive_start: float | None = None
        self.drive_end: float | None = None
        self._slow_since: float | None = None
        self._stuck_flagged = False
        self.run_dir = run_dir
        self._csv_file = None
        self._writer = None
        self._events_file = None
        if run_dir is not None:
            run_dir.mkdir(parents=True, exist_ok=True)
            write_json(run_dir / "run_meta.json", {**(meta or {}), "target_speed": self.target_speed,
                                                   "low_speed_mode": low_speed, "control": vars(cfg.control),
                                                   "preprocess": vars(cfg.preprocess)})
            self._csv_file = (run_dir / "frames.csv").open("w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._csv_file, fieldnames=FRAME_FIELDS)
            self._writer.writeheader()
            self._events_file = (run_dir / "events.jsonl").open("a", encoding="utf-8")

    # ------------------------------------------------------------------ events
    def now(self) -> float:
        return time.monotonic() - self.t0

    def record_event(self, event: str, **fields) -> None:
        e = {"t": round(self.now(), 3), "event": event, "frame": self.frame, **fields}
        self.events.append(e)
        if self._events_file:
            self._events_file.write(json.dumps(e) + "\n")
            self._events_file.flush()
        log.info("EVENT %s %s", event, fields or "")

    def on_connect(self) -> None:
        with self.lock:
            self.steer_filter.reset()
            self.throttle.reset()
            self.invalid_streak = 0
            self.record_event("connect")

    def on_disconnect(self) -> None:
        with self.lock:
            self.steer_filter.reset()
            self.throttle.reset()
            self.record_event("disconnect")

    def emergency_stop(self, source: str = "operator", reason: str = "") -> None:
        with self.lock:
            if not self.estopped:
                self.estopped = True
                self.record_event("estop", source=source, reason=reason)

    def resume(self) -> None:
        with self.lock:
            if self.estopped:
                self.estopped = False
                self.invalid_streak = 0
                self.steer_filter.reset()
                self.throttle.reset()
                self.record_event("resume")

    def mark(self, event: str) -> None:
        """Operator marks: 'intervention', 'off_track', 'lap_complete'."""
        with self.lock:
            self.record_event(event, source="operator")

    def request_quit(self) -> None:
        with self.lock:
            self.quit_requested = True
            self.record_event("quit", source="operator")

    # ------------------------------------------------------------------ loop
    def handle_telemetry(self, t: Telemetry) -> ControlCommand | None:
        with self.lock:
            return self._handle(t)

    def _handle(self, t: Telemetry) -> ControlCommand | None:
        start = time.perf_counter()
        self.frame += 1
        now = self.now()
        row = {"t": round(now, 4), "frame": self.frame, "speed": t.speed, "sim_steering": t.steering_angle,
               "sim_throttle": t.throttle}

        if self.estopped or self.quit_requested:
            # Brake to a standstill, then stop sending commands entirely.
            if t.speed > 0.5:
                cmd = brake_command(t.speed, self.cfg.control)
                self._write(row, cmd, status="estop_braking")
                return cmd.validated()
            self._write(row, None, status="estop_idle")
            return None

        timings = {}
        try:
            t1 = time.perf_counter()
            image = decode_image_base64(t.image_b64)
            t2 = time.perf_counter()
            x = preprocess(image, self.cfg.preprocess)
            t3 = time.perf_counter()
            raw = float(self.predictor(x))
            t4 = time.perf_counter()
            timings = {"decode_ms": (t2 - t1) * 1e3, "preprocess_ms": (t3 - t2) * 1e3,
                       "inference_ms": (t4 - t3) * 1e3}
            steering = self.steer_filter(raw)
        except (ImageError, UnsafeCommandError) as exc:
            return self._invalid_frame(row, t, str(exc))

        self.invalid_streak = 0
        throttle = self.throttle(t.speed)
        cmd = ControlCommand(steering, throttle).validated()
        total = (time.perf_counter() - start) * 1e3
        self.latencies.append(total)
        self.steering_log.append(steering)
        self.steering_t.append(now)
        if self.drive_start is None:
            self.drive_start = now
        self.drive_end = now
        self._check_stuck(now, t.speed, throttle)
        self._write({**row, **timings, "raw_pred": raw, "total_ms": total}, cmd, status="ok")
        return cmd

    def _invalid_frame(self, row: dict, t: Telemetry, reason: str) -> ControlCommand | None:
        self.invalid_streak += 1
        log.warning("invalid frame %d (%d in a row): %s", self.frame, self.invalid_streak, reason)
        if self.invalid_streak >= MAX_CONSECUTIVE_INVALID:
            if not self.estopped:
                self.estopped = True
                self.record_event("estop", source="auto", reason=f"{self.invalid_streak} invalid frames: {reason}")
            cmd = brake_command(t.speed, self.cfg.control)
        else:
            # Hold the last valid steering, cut the throttle.
            held = self.steer_filter.prev if self.steer_filter.prev is not None else 0.0
            cmd = ControlCommand(held, 0.0)
        self._write(row, cmd, status=f"invalid: {reason[:80]}")
        return cmd.validated()

    def _check_stuck(self, now: float, speed: float, throttle: float) -> None:
        if speed < 0.5 and throttle > 0.2:
            self._slow_since = self._slow_since if self._slow_since is not None else now
            if not self._stuck_flagged and now - self._slow_since > STUCK_SECONDS:
                self._stuck_flagged = True
                self.record_event("possible_stuck", source="heuristic",
                                  reason=f"speed < 0.5 mph for {STUCK_SECONDS:.0f}s with throttle applied")
        else:
            self._slow_since, self._stuck_flagged = None, False

    def _write(self, row: dict, cmd: ControlCommand | None, status: str) -> None:
        if not self._writer:
            return
        row = dict(row)
        if cmd is not None:
            row["steering_cmd"], row["throttle_cmd"] = round(cmd.steering, 6), round(cmd.throttle, 6)
        row["status"] = status
        for k in ("decode_ms", "preprocess_ms", "inference_ms", "total_ms", "raw_pred"):
            if k in row and row[k] is not None:
                row[k] = round(row[k], 4)
        self._writer.writerow(row)

    # ------------------------------------------------------------------ results
    def summary(self) -> dict:
        with self.lock:
            return {
                "frames_received": self.frame,
                "frames_driven": len(self.steering_log),
                "run_seconds": round(self.now(), 2),
                "target_speed": self.target_speed,
                "inference_latency_total": latency_stats(self.latencies),
                "steering_commands": steering_stats(self.steering_log, self.steering_t),
                **summarize_events(self.events, self.drive_start, self.drive_end),
            }

    def close(self, extra: dict | None = None) -> dict:
        s = self.summary()
        if extra:
            s.update(extra)
        if self.run_dir is not None:
            write_json(self.run_dir / "summary.json", s)
        for f in (self._csv_file, self._events_file):
            if f:
                f.close()
        self._csv_file = self._events_file = self._writer = None
        return s
