"""ExpDash-compatible progress reporting for locally scheduled experiments.

The ``exp run`` launcher exports ``EXP_METRICS_FILE``. This module atomically
updates that file using ExpDash's documented JSON schema, so no HTTP client or
socket connection is required.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from threading import Lock
from typing import Union

MetricValue = Union[int, float, str]
_HISTORY_MAX = 240
_history_by_file: dict[str, list[list[object]]] = {}
_lock = Lock()


def validate_history(history: object) -> list[list[object]]:
    if not isinstance(history, list):
        raise ValueError("ExpDash history must be a list")
    last_step = -1
    for entry in history:
        if (not isinstance(entry, list) or len(entry) != 3
                or not isinstance(entry[0], (int, float)) or not math.isfinite(entry[0])
                or not isinstance(entry[1], int) or entry[1] < last_step
                or not isinstance(entry[2], dict)):
            raise ValueError("ExpDash history must contain ordered [timestamp, step, metrics] rows")
        last_step = entry[1]
    return history


def resume_history(source: Path) -> None:
    """Seed a new scheduled job's chart with its parent run's sample history."""
    destination = os.environ.get("EXP_METRICS_FILE")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("values", {}).get("progress_unit") != "samples":
        raise ValueError("Parent history must use cumulative samples")
    history = validate_history(payload["history"])
    if destination:
        with _lock:
            _history_by_file[destination] = list(history)


def is_enabled() -> bool:
    """Return whether this process was started through the ExpDash launcher."""
    return bool(os.environ.get("EXP_METRICS_FILE"))


def report(
    *,
    step: int | None = None,
    total: int | None = None,
    step_offset: int | None = None,
    **values: MetricValue,
) -> bool:
    """Publish scalar progress values to the active ExpDash run.

    Returns ``False`` when the process has no ``EXP_METRICS_FILE`` because it
    was not launched through ``exp run``. Otherwise the metrics file is
    atomically replaced and ``True`` is returned. Invalid metric values raise
    ``TypeError`` instead of being silently omitted.
    """
    metrics_file = os.environ.get("EXP_METRICS_FILE")
    if not metrics_file:
        return False
    if step_offset is None:
        raw_offset = os.environ.get("EXP_STEP_OFFSET", "0")
        try:
            step_offset = int(raw_offset)
        except ValueError as error:
            raise ValueError("EXP_STEP_OFFSET must be an integer") from error
    if step_offset < 0:
        raise ValueError("step_offset must be non-negative")
    if step is not None and step < 0:
        raise ValueError("step must be non-negative")
    if total is not None and total < 0:
        raise ValueError("total must be non-negative")
    if step is not None and total is not None and step > total:
        raise ValueError("step cannot exceed total")
    if any(not isinstance(value, (int, float, str)) for value in values.values()):
        raise TypeError("metric values must be integers, floats, or strings")
    if any(isinstance(value, float) and not math.isfinite(value) for value in values.values()):
        raise ValueError("metrics must be finite")

    absolute_step = step + step_offset if step is not None else None
    absolute_total = total + step_offset if total is not None else None
    now = time.time()
    numeric_values = {
        name: value for name, value in values.items() if isinstance(value, (int, float))
    }
    with _lock:
        destination = Path(metrics_file)
        history = _history_by_file.get(metrics_file)
        if history is None:
            history = []
            if destination.is_file():
                try:
                    previous = json.loads(destination.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as error:
                    raise ValueError(
                        f"cannot resume malformed ExpDash metrics file: {destination}"
                    ) from error
                history.extend(validate_history(previous.get("history", [])))
            _history_by_file[metrics_file] = history
        if absolute_step is not None:
            if history and absolute_step < history[-1][1]:
                raise ValueError("Cumulative progress cannot move backwards")
            if history and absolute_step == history[-1][1]:
                history.pop()
            history.append([now, absolute_step, numeric_values])
            if len(history) > _HISTORY_MAX:
                history[:] = history[:-1:2] + [history[-1]]
        payload = {
            "ts": now,
            "step": absolute_step,
            "total": absolute_total,
            "local_step": step,
            "step_offset": step_offset,
            "values": values,
            "history": history,
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f"{destination.name}.tmp")
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        temporary.replace(destination)
    return True
