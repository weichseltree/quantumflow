"""ExpDash-compatible progress reporting for locally scheduled experiments.

The ``exp run`` launcher exports ``EXP_METRICS_FILE``. This module atomically
updates that file using ExpDash's documented JSON schema, so no HTTP client or
socket connection is required.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from threading import Lock
from typing import Union

MetricValue = Union[int, float, str]
_HISTORY_MAX = 240
_history_by_file: dict[str, list[list[object]]] = {}
_lock = Lock()


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
                previous_history = previous.get("history", [])
                if not isinstance(previous_history, list):
                    raise ValueError("existing ExpDash history must be a list")
                history.extend(previous_history)
            _history_by_file[metrics_file] = history
        if absolute_step is not None:
            history.append([now, absolute_step, numeric_values])
            if len(history) > _HISTORY_MAX:
                history[:] = history[::2]
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
