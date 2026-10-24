"""Vehicle control outside the neural network.

The CNN only predicts steering. Speed is held by a separate, simple PI controller on
the simulator-reported speed. Everything sent to the simulator passes through
``validate_command``: finite numbers inside the simulator's [-1, 1] ranges, always.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from bcdrive.config import ControlConfig


class UnsafeCommandError(ValueError):
    pass


@dataclass(frozen=True)
class ControlCommand:
    steering: float  # [-1, 1], positive = right
    throttle: float  # [-1, 1], negative = brake

    def validated(self) -> "ControlCommand":
        validate_command(self)
        return self


def validate_command(cmd: ControlCommand) -> None:
    for name in ("steering", "throttle"):
        v = getattr(cmd, name)
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            raise UnsafeCommandError(f"{name} is not a finite number: {v!r}")
        if not -1.0 <= v <= 1.0:
            raise UnsafeCommandError(f"{name}={v} outside [-1, 1]")


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


class SteeringFilter:
    """raw prediction -> clip -> optional EMA smoothing -> rate limit -> clip to max."""

    def __init__(self, cfg: ControlConfig):
        self.cfg = cfg
        self.prev: float | None = None

    def reset(self) -> None:
        self.prev = None

    def __call__(self, raw: float) -> float:
        if not math.isfinite(raw):
            raise UnsafeCommandError(f"model produced non-finite steering {raw!r}")
        c = self.cfg
        x = clamp(raw, -1.0, 1.0)
        if self.prev is not None:
            x = c.steering_smoothing * x + (1.0 - c.steering_smoothing) * self.prev
            x = clamp(x, self.prev - c.max_steering_delta, self.prev + c.max_steering_delta)
        x = clamp(x, -c.max_steering, c.max_steering)
        self.prev = x
        return x


class PIThrottleController:
    """Holds a target speed. Output in [-max_brake, max_throttle] (negative = brake)."""

    def __init__(self, cfg: ControlConfig, target_speed: float):
        self.cfg = cfg
        self.target = target_speed
        self.integral = 0.0

    def reset(self) -> None:
        self.integral = 0.0

    def __call__(self, speed: float) -> float:
        if not math.isfinite(speed):
            raise UnsafeCommandError(f"non-finite speed {speed!r}")
        c = self.cfg
        error = self.target - speed
        self.integral = clamp(self.integral + error, -c.integral_limit, c.integral_limit)
        return clamp(c.kp * error + c.ki * self.integral, -c.max_brake, c.max_throttle)


def brake_command(speed: float, cfg: ControlConfig) -> ControlCommand:
    """Straight wheels and brake (or coast once stopped)."""
    return ControlCommand(0.0, -cfg.max_brake if speed > 0.5 else 0.0)
