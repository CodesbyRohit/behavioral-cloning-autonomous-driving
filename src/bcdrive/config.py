"""Centralised, validated configuration.

Every tunable number in the project lives in one of these dataclasses. A YAML file
(see ``configs/default.yaml``) overrides the defaults; unknown keys are rejected so a
typo cannot silently fall back to a default value.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a configuration value is missing, unknown or out of range."""


@dataclass
class DataConfig:
    # Directories that each contain a driving_log.csv (+ IMG/). A directory without a
    # driving_log.csv is scanned one level deep for session folders.
    sessions: list[str] = field(default_factory=lambda: ["data/raw"])
    # Column names for header-less logs (the Udacity simulator writes no header).
    default_columns: list[str] = field(
        default_factory=lambda: ["center", "left", "right", "steering", "throttle", "brake", "speed"]
    )
    min_frames: int = 500  # training refuses to start with fewer usable frames


@dataclass
class PreprocessConfig:
    # Pixels removed from each edge of the raw camera frame before resizing.
    # Defaults target the Udacity simulator's 320x160 frames: 60 px of sky/trees at the
    # top and 25 px of bonnet at the bottom. Verify with `bcdrive dataset inspect`.
    crop_top: int = 60
    crop_bottom: int = 25
    crop_left: int = 0
    crop_right: int = 0
    out_height: int = 66
    out_width: int = 200
    color_space: str = "yuv"  # "yuv" (as in the NVIDIA paper) or "rgb"
    # Reject frames whose (height, width) differs from this. null disables the check.
    expected_input_hw: list[int] | None = field(default_factory=lambda: [160, 320])


@dataclass
class SplitConfig:
    strategy: str = "auto"  # "auto" | "session" | "chunk"
    val_fraction: float = 0.2
    test_fraction: float = 0.0
    chunk_size: int = 100  # consecutive frames kept together in chunk splitting
    gap: int = 10  # frames dropped at both ends of every val/test chunk (anti-leakage)
    min_sessions_for_session_split: int = 3
    seed: int = 42


@dataclass
class AugmentConfig:
    enabled: bool = True
    use_side_cameras: bool = True
    # Added to the label of the LEFT camera, subtracted for the RIGHT camera. Correct for
    # the Udacity convention (positive steering = turn right); see docs/DATA_COLLECTION.md.
    side_camera_correction: float = 0.2
    flip_probability: float = 0.5
    brightness_probability: float = 0.5
    brightness_range: list[float] = field(default_factory=lambda: [0.5, 1.2])
    shift_probability: float = 0.3
    max_shift_px: int = 20
    steering_per_shift_px: float = 0.004


@dataclass
class BalanceConfig:
    enabled: bool = True
    bins: int = 21
    # Maximum samples per steering bin. "auto" = 2 x mean count of the non-empty bins.
    max_per_bin: Any = "auto"


@dataclass
class ModelConfig:
    architecture: str = "nvidia"  # "nvidia" | "tiny" (CPU smoke-test fallback)
    activation: str = "elu"
    dropout: float = 0.3
    l2: float = 0.0


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-3
    early_stopping_patience: int = 4
    workers: int = 4
    seed: int = 42
    deterministic_ops: bool = False  # tf op determinism: reproducible but slower
    output_dir: str = "artifacts/experiments"


@dataclass
class EvalConfig:
    straight_threshold: float = 0.03  # |steering| below this counts as "straight"
    bin_edges: list[float] = field(
        default_factory=lambda: [-1.0, -0.5, -0.25, -0.1, -0.03, 0.03, 0.1, 0.25, 0.5, 1.0]
    )
    min_bin_count: int = 20  # bins with fewer samples are reported as insufficient


@dataclass
class ControlConfig:
    target_speed: float = 12.0  # mph, as reported by the simulator
    low_speed_target: float = 6.0
    kp: float = 0.1
    ki: float = 0.002
    integral_limit: float = 50.0
    max_throttle: float = 0.6
    max_brake: float = 0.3  # sent as negative throttle
    max_steering: float = 1.0
    steering_smoothing: float = 1.0  # EMA weight of the newest prediction; 1.0 = off
    max_steering_delta: float = 1.0  # per-frame rate limit; 1.0 effectively off


@dataclass
class SimulatorConfig:
    host: str = "0.0.0.0"
    port: int = 4567
    connect_timeout_s: float = 120.0  # give up if the simulator never connects
    stale_telemetry_s: float = 2.0  # warn when a connected simulator goes quiet
    exit_on_disconnect: bool = False  # otherwise wait (connect_timeout_s) for a reconnect
    runs_dir: str = "artifacts/runs"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    balance: BalanceConfig = field(default_factory=BalanceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def validate(self) -> "Config":
        _validate(self)
        return self


def _merge(dc_obj: Any, overrides: dict[str, Any], prefix: str = "") -> None:
    if not isinstance(overrides, dict):
        raise ConfigError(f"section '{prefix or 'root'}' must be a mapping, got {type(overrides).__name__}")
    valid = {f.name: f for f in dataclasses.fields(dc_obj)}
    for key, value in overrides.items():
        if key not in valid:
            raise ConfigError(f"unknown config key '{prefix}{key}' (valid: {', '.join(sorted(valid))})")
        current = getattr(dc_obj, key)
        if dataclasses.is_dataclass(current):
            _merge(current, value or {}, prefix=f"{prefix}{key}.")
        else:
            setattr(dc_obj, key, value)


def config_from_dict(raw: dict[str, Any] | None) -> Config:
    cfg = Config()
    _merge(cfg, raw or {})
    return cfg.validate()


def load_config(path: str | Path | None = None) -> Config:
    """Load ``path`` (YAML) over the defaults. ``None`` returns validated defaults."""
    if path is None:
        return Config().validate()
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return config_from_dict(raw)


def save_config(cfg: Config, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def _validate(c: Config) -> None:
    p = c.preprocess
    for name in ("crop_top", "crop_bottom", "crop_left", "crop_right"):
        _require(isinstance(getattr(p, name), int) and getattr(p, name) >= 0, f"preprocess.{name} must be an int >= 0")
    _require(p.out_height > 0 and p.out_width > 0, "preprocess output size must be positive")
    _require(p.color_space in ("yuv", "rgb"), "preprocess.color_space must be 'yuv' or 'rgb'")
    if p.expected_input_hw is not None:
        _require(len(p.expected_input_hw) == 2, "preprocess.expected_input_hw must be [height, width] or null")
        h, w = p.expected_input_hw
        _require(p.crop_top + p.crop_bottom < h, "vertical crop removes the entire expected frame")
        _require(p.crop_left + p.crop_right < w, "horizontal crop removes the entire expected frame")

    s = c.split
    _require(s.strategy in ("auto", "session", "chunk"), "split.strategy must be auto|session|chunk")
    _require(0 < s.val_fraction < 1, "split.val_fraction must be in (0, 1)")
    _require(0 <= s.test_fraction < 1, "split.test_fraction must be in [0, 1)")
    _require(s.val_fraction + s.test_fraction < 1, "val_fraction + test_fraction must be < 1")
    _require(s.chunk_size >= 1 and s.gap >= 0, "split.chunk_size >= 1 and split.gap >= 0 required")
    _require(2 * s.gap < s.chunk_size, "split.gap must be smaller than half of split.chunk_size")

    a = c.augment
    for name in ("flip_probability", "brightness_probability", "shift_probability"):
        _require(0.0 <= getattr(a, name) <= 1.0, f"augment.{name} must be in [0, 1]")
    _require(len(a.brightness_range) == 2 and 0 < a.brightness_range[0] <= a.brightness_range[1],
             "augment.brightness_range must be [low, high] with 0 < low <= high")
    _require(0.0 <= a.side_camera_correction <= 0.5, "augment.side_camera_correction must be in [0, 0.5]")
    _require(a.max_shift_px >= 0, "augment.max_shift_px must be >= 0")

    b = c.balance
    _require(b.bins >= 2, "balance.bins must be >= 2")
    _require(b.max_per_bin == "auto" or (isinstance(b.max_per_bin, int) and b.max_per_bin > 0),
             "balance.max_per_bin must be 'auto' or a positive int")

    m = c.model
    _require(m.architecture in ("nvidia", "tiny"), "model.architecture must be 'nvidia' or 'tiny'")
    _require(0.0 <= m.dropout < 1.0, "model.dropout must be in [0, 1)")

    t = c.train
    _require(t.epochs >= 1 and t.batch_size >= 1, "train.epochs and train.batch_size must be >= 1")
    _require(t.learning_rate > 0, "train.learning_rate must be > 0")
    _require(t.workers >= 1, "train.workers must be >= 1")

    e = c.eval
    _require(len(e.bin_edges) >= 2 and all(x < y for x, y in zip(e.bin_edges, e.bin_edges[1:])),
             "eval.bin_edges must be strictly increasing")

    k = c.control
    _require(0 < k.max_throttle <= 1.0, "control.max_throttle must be in (0, 1]")
    _require(0 <= k.max_brake <= 1.0, "control.max_brake must be in [0, 1]")
    _require(0 < k.max_steering <= 1.0, "control.max_steering must be in (0, 1]")
    _require(0 < k.steering_smoothing <= 1.0, "control.steering_smoothing must be in (0, 1]")
    _require(0 < k.max_steering_delta <= 2.0, "control.max_steering_delta must be in (0, 2]")
    _require(0 <= k.low_speed_target <= k.target_speed <= 30.5,
             "need 0 <= control.low_speed_target <= control.target_speed <= 30 (simulator top speed)")

    sim = c.simulator
    _require(1 <= sim.port <= 65535, "simulator.port must be a valid TCP port")
    _require(sim.connect_timeout_s > 0 and sim.stale_telemetry_s > 0, "simulator timeouts must be > 0")

    _require(c.data.min_frames >= 1, "data.min_frames must be >= 1")
    _require(len(c.data.sessions) >= 1, "data.sessions must list at least one directory")
