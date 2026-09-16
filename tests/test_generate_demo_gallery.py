import json
from pathlib import Path

import pytest

from scripts.generate_demo_gallery import generate_demo_gallery

_TINY_KWARGS = {
    "steps": 5,
    "batch_size": 4,
    "eval_samples": 4,
    "num_particles": 4,
    "tape_steps": 2,
}


def test_training_writes_model_before_gallery_export_is_attempted(tmp_path: Path) -> None:
    output_dir = tmp_path / "demo"

    with pytest.raises(RuntimeError, match="official 'orchard_tape' package"):
        generate_demo_gallery(output_dir, **_TINY_KWARGS)

    assert (output_dir / "model.npz").is_file()
    assert (output_dir / "metrics.json").is_file()
    assert not (output_dir / "gallery").exists()


def test_skip_train_requires_an_existing_model(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="model.npz"):
        generate_demo_gallery(tmp_path / "missing", skip_train=True, **_TINY_KWARGS)


@pytest.mark.orchard
def test_full_demo_gallery_pipeline(tmp_path: Path) -> None:
    output_dir = tmp_path / "demo"

    paths = generate_demo_gallery(output_dir, **_TINY_KWARGS)

    assert paths["model"].is_file()
    assert (paths["tape_dir"] / "webxr_particles.json").is_file()
    assert paths["webxr_html"].is_file()
    assert paths["orchard_bundle"].is_dir()

    manifest = json.loads(paths["gallery_manifest"].read_text(encoding="utf-8"))
    assert manifest["schema"] == "quantumflow/gallery/2"
    gallery_dir = paths["gallery_manifest"].parent
    assert manifest["orchard_bundle"] == str(paths["orchard_bundle"].relative_to(gallery_dir))


@pytest.mark.orchard
def test_no_bundle_skips_bundle_but_keeps_tape_and_viewer(tmp_path: Path) -> None:
    output_dir = tmp_path / "demo"

    paths = generate_demo_gallery(output_dir, skip_bundle=True, **_TINY_KWARGS)

    assert "orchard_bundle" not in paths
    assert paths["webxr_html"].is_file()
    manifest = json.loads(paths["gallery_manifest"].read_text(encoding="utf-8"))
    assert "orchard_bundle" not in manifest
