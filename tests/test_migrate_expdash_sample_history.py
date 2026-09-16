import json

from scripts.migrate_expdash_sample_history import migrate_record


def test_migrates_legacy_step_history_to_samples(tmp_path) -> None:
    status_path = tmp_path / "run.json"
    metrics_path = tmp_path / "run.metrics"
    status_path.write_text(
        json.dumps(
            {
                "sweep": "transport-beta-pilot",
                "cmd": "python run.py --batch-size 256 --steps 2000",
            }
        ),
        encoding="utf-8",
    )
    metrics_path.write_text(
        json.dumps(
            {
                "step": 2000,
                "total": 2000,
                "values": {"loss": 0.1},
                "history": [[1.0, 50, {"loss": 0.5}], [2.0, 2000, {"loss": 0.1}]],
            }
        ),
        encoding="utf-8",
    )

    assert migrate_record(status_path)
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["step"] == 512000
    assert payload["total"] == 512000
    assert [entry[1] for entry in payload["history"]] == [12800, 512000]
    assert payload["values"]["progress_unit"] == "samples"


def test_skips_non_record_json(tmp_path) -> None:
    path = tmp_path / "events.json"
    path.write_text("[]", encoding="utf-8")
    assert not migrate_record(path)
