"""Small filesystem / logging helpers shared by the CLI commands."""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        if getattr(h, "_bcdrive", False):
            root.removeHandler(h)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    for h in handlers:
        h.setFormatter(logging.Formatter(LOG_FORMAT))
        h._bcdrive = True  # type: ignore[attr-defined]
        root.addHandler(h)


def add_file_log(log_file: Path) -> logging.Handler:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    h = logging.FileHandler(log_file, encoding="utf-8")
    h.setFormatter(logging.Formatter(LOG_FORMAT))
    h._bcdrive = True  # type: ignore[attr-defined]
    logging.getLogger().addHandler(h)
    return h


def timestamp_id(name: str | None = None) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if not name:
        return stamp
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-")
    return f"{stamp}_{safe}" if safe else stamp


def unique_dir(parent: Path, name: str) -> Path:
    """Create ``parent/name`` (adding -2, -3, ... if it exists) and return it."""
    parent.mkdir(parents=True, exist_ok=True)
    candidate, n = parent / name, 2
    while candidate.exists():
        candidate, n = parent / f"{name}-{n}", n + 1
    candidate.mkdir()
    return candidate


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item"):  # numpy scalars
        return _jsonable(obj.item())
    return obj


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(data), indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
