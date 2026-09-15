"""Convert legacy pilot ExpDash histories from optimizer steps to samples."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path


def batch_size_from_command(command: str) -> int:
    tokens = shlex.split(command)
    try:
        index = tokens.index("--batch-size")
        return int(tokens[index + 1])
    except (ValueError, IndexError) as error:
        raise ValueError("status command has no valid --batch-size") from error


def migrate_record(status_path: Path) -> bool:
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if not isinstance(status, dict):
        return False
    if status.get("sweep") != "transport-beta-pilot":
        return False
    metrics_path = status_path.with_suffix(".metrics")
    if not metrics_path.is_file():
        return False

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    values = payload.get("values", {})
    if values.get("progress_unit") == "samples":
        return False

    batch_size = batch_size_from_command(status["cmd"])
    local_step = payload.get("step")
    local_total = payload.get("total")
    if local_step is not None:
        payload["step"] = local_step * batch_size
        payload["local_step"] = local_step
    if local_total is not None:
        payload["total"] = local_total * batch_size
    payload["step_offset"] = 0
    payload["values"] = {
        **values,
        "optimizer_step": local_step or 0,
        "samples_seen": (local_step or 0) * batch_size,
        "progress_unit": "samples",
    }
    for entry in payload.get("history", []):
        if len(entry) >= 2 and isinstance(entry[1], int):
            entry[1] *= batch_size

    temporary = metrics_path.with_suffix(".metrics.tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    temporary.replace(metrics_path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--status-dir",
        type=Path,
        default=Path.home() / ".exp_status",
    )
    args = parser.parse_args()

    migrated = sum(
        migrate_record(path) for path in sorted(args.status_dir.glob("*.json"))
    )
    print(f"Migrated {migrated} pilot metric histories to cumulative samples.")


if __name__ == "__main__":
    main()
