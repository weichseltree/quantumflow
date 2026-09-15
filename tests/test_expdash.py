import json

import pytest

from quantumflow import expdash


def test_report_is_inactive_without_expdash_launcher(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EXP_METRICS_FILE", raising=False)

    assert expdash.report(step=1, total=2, loss=0.5) is False


def test_report_writes_expdash_metrics_schema(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    metrics_path = tmp_path / "run.metrics"
    monkeypatch.setenv("EXP_METRICS_FILE", str(metrics_path))
    expdash._history.clear()

    assert expdash.report(step=5, total=10, loss=0.25, phase="train") is True

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["step"] == 5
    assert payload["total"] == 10
    assert payload["values"] == {"loss": 0.25, "phase": "train"}
    assert payload["history"][0][1:] == [5, {"loss": 0.25}]


def test_report_rejects_invalid_progress(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("EXP_METRICS_FILE", str(tmp_path / "run.metrics"))

    with pytest.raises(ValueError, match="cannot exceed"):
        expdash.report(step=2, total=1)
