"""Closed-loop run metrics. Only measured or operator-marked inputs are used.

The Udacity simulator's telemetry contains steering, throttle, speed and an image, and
nothing about lap progress, track position or crashes. Lap completion, takeovers and
off-track events are therefore recorded by the operator with keys in the drive console
(see docs/EVALUATION_PROTOCOL.md) and are labelled ``operator_observed`` everywhere.
"""

from __future__ import annotations

import math

import numpy as np

TAKEOVER_PENALTY_S = 6.0  # NVIDIA's convention, not a universal safety measure


def autonomy_score(takeovers: int, elapsed_s: float, penalty_s: float = TAKEOVER_PENALTY_S) -> float | None:
    """NVIDIA-style autonomy = (1 - takeovers * 6 s / elapsed seconds) * 100.

    Approximate metric (Bojarski et al. 2016). Can go negative for very frequent
    takeovers; returned unclipped so that is visible. None when elapsed time is unknown.
    """
    if elapsed_s is None or not math.isfinite(elapsed_s) or elapsed_s <= 0:
        return None
    if takeovers < 0:
        raise ValueError("takeovers must be >= 0")
    return (1.0 - takeovers * penalty_s / elapsed_s) * 100.0


def steering_stats(steering: list[float], timestamps: list[float]) -> dict:
    if not steering:
        return {"frames": 0}
    s = np.asarray(steering, dtype=np.float64)
    duration = (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0
    signs = np.sign(s[np.abs(s) > 0.02])
    flips = int(np.sum(signs[1:] != signs[:-1])) if signs.size > 1 else 0
    return {
        "frames": int(s.size),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "mean_abs": float(np.abs(s).mean()),
        "max_abs": float(np.abs(s).max()),
        "mean_abs_frame_delta": float(np.abs(np.diff(s)).mean()) if s.size > 1 else 0.0,
        # left/right reversals per second: a rough oscillation indicator
        "sign_reversals_per_s": (flips / duration) if duration > 0 else None,
    }


def latency_stats(latencies_ms: list[float]) -> dict:
    if not latencies_ms:
        return {"frames": 0}
    a = np.asarray(latencies_ms)
    return {"frames": int(a.size), "mean_ms": float(a.mean()), "p95_ms": float(np.percentile(a, 95)),
            "max_ms": float(a.max())}


def summarize_events(events: list[dict], drive_start: float | None, drive_end: float | None) -> dict:
    """Takeovers / off-track / laps from operator-marked events (times in run seconds)."""
    kinds = lambda k: [e for e in events if e["event"] == k]  # noqa: E731
    takeovers = kinds("intervention") + kinds("estop")
    laps = kinds("lap_complete")
    out = {
        "takeovers_operator_observed": len(takeovers),
        "off_track_operator_observed": len(kinds("off_track")),
        "possible_stuck_heuristic": len(kinds("possible_stuck")),
        "auto_estops_invalid_frames": len([e for e in kinds("estop") if e.get("source") == "auto"]),
        "laps_operator_observed": len(laps),
        "laps": [],
        "autonomy_score_note": "computed only for operator-marked laps; approximate NVIDIA-style metric",
    }
    prev = drive_start
    for lap in laps:
        if prev is None:
            break
        t_end = lap["t"]
        n_take = len([e for e in takeovers if prev <= e["t"] <= t_end])
        lap_time = t_end - prev
        out["laps"].append({
            "lap_time_s": lap_time,
            "takeovers": n_take,
            "autonomy_score_pct": autonomy_score(n_take, lap_time),
            "verified_by": "operator_observed",
        })
        prev = t_end
    if drive_start is not None and drive_end is not None:
        out["autonomous_driving_seconds"] = drive_end - drive_start
        out["autonomy_score_whole_run_pct"] = (autonomy_score(len(takeovers), drive_end - drive_start)
                                               if laps or takeovers else None)
    return out
