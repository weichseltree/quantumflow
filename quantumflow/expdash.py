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
_history: list[list[object]] = []
_lock = Lock()


def is_enabled() -> bool:
    """Return whether this process was started through the ExpDash launcher."""
    return bool(os.environ.get("EXP_METRICS_FILE"))


def report(
    *,
    step: int | None = None,
    total: int | None = None,
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
    if step is not None and step < 0:
        raise ValueError("step must be non-negative")
    if total is not None and total < 0:
        raise ValueError("total must be non-negative")
    if step is not None and total is not None and step > total:
        raise ValueError("step cannot exceed total")
    if any(not isinstance(value, (int, float, str)) for value in values.values()):
        raise TypeError("metric values must be integers, floats, or strings")

    now = time.time()
    numeric_values = {
        name: value for name, value in values.items() if isinstance(value, (int, float))
    }
    with _lock:
        if step is not None:
            _history.append([now, step, numeric_values])
            if len(_history) > _HISTORY_MAX:
                _history[:] = _history[::2]
        payload = {
            "ts": now,
            "step": step,
            "total": total,
            "values": values,
            "history": _history,
        }
        destination = Path(metrics_file)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f"{destination.name}.tmp")
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        temporary.replace(destination)
    return True
