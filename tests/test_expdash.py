import json

import pytest

from quantumflow import expdash


def test_report_is_inactive_without_expdash_launcher(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EXP_METRICS_FILE", raising=False)

    assert expdash.report(step=1, total=2, loss=0.5) is False


def test_report_writes_expdash_metrics_schema(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    metrics_path = tmp_path / "run.metrics"
    monkeypatch.setenv("EXP_METRICS_FILE", str(metrics_path))
    expdash._history_by_file.clear()

    assert expdash.report(step=5, total=10, loss=0.25, phase="train") is True

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["step"] == 5
    assert payload["total"] == 10
    assert payload["values"] == {"loss": 0.25, "phase": "train"}
    assert payload["history"][0][1:] == [5, {"loss": 0.25}]


def test_report_uses_absolute_steps_and_resumes_history(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    metrics_path = tmp_path / "continuation.metrics"
    monkeypatch.setenv("EXP_METRICS_FILE", str(metrics_path))
    expdash._history_by_file.clear()

    expdash.report(step=2000, total=2000, loss=0.3)
    expdash._history_by_file.clear()
    expdash.report(step=1, total=2000, step_offset=2000, loss=0.2)

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["step"] == 2001
    assert payload["total"] == 4000
    assert payload["local_step"] == 1
    assert payload["step_offset"] == 2000
    assert [entry[1] for entry in payload["history"]] == [2000, 2001]


def test_report_rejects_invalid_progress(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("EXP_METRICS_FILE", str(tmp_path / "run.metrics"))

    with pytest.raises(ValueError, match="cannot exceed"):
        expdash.report(step=2, total=1)


def test_continuation_copies_parent_sample_history(monkeypatch, tmp_path):
    parent = tmp_path / "parent.json"
    parent.write_text(json.dumps({
        "values": {"progress_unit": "samples"},
        "history": [[1.0, 512000, {"loss": 0.2}]],
    }))
    child = tmp_path / "child.metrics"
    monkeypatch.setenv("EXP_METRICS_FILE", str(child))
    expdash.resume_history(parent)
    expdash.report(step=524800, total=1024000, loss=0.1, progress_unit="samples")
    assert [row[1] for row in json.loads(child.read_text())["history"]] == [512000, 524800]
    with pytest.raises(ValueError, match="backwards"):
        expdash.report(step=1, loss=0.2)


def test_downsampling_keeps_latest_sample(monkeypatch, tmp_path):
    path = tmp_path / "large.metrics"
    monkeypatch.setenv("EXP_METRICS_FILE", str(path))
    for step in range(1, 243):
        expdash.report(step=step, loss=0.2)
    assert json.loads(path.read_text())["history"][-1][1] == 242
