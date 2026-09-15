import importlib
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from quantumflow.orchard_export import (
    create_orchard_bundle,
    export_trajectory_tape,
    generate_webxr_gallery_html,
)
from quantumflow.ot_cfm import ModelParams


def _linear_potential() -> ModelParams:
    # Phi(t, q1, q2) = t + 2*q1 + 3*q2: v=(2,3), dPhi/dt along flow = 14.
    return ModelParams(
        weights=[jnp.asarray([[1.0], [2.0], [3.0]])],
        biases=[jnp.asarray([0.0])],
    )


def test_export_round_trips_without_clipping_and_has_correct_lift_velocity(
    tmp_path: Path,
) -> None:
    orchard_tape = importlib.import_module("orchard_tape")
    tape_dir = export_trajectory_tape(
        _linear_potential(),
        tmp_path / "tape",
        num_particles=12,
        num_steps=4,
        seed=7,
        provenance={"synthetic": True, "purpose": "unit-test smoke trajectory"},
        source_run="synthetic-linear-potential",
        source_model="analytic Phi=t+2q1+3q2",
        git_sha="test-fixture",
    )

    payload = json.loads((tape_dir / "webxr_particles.json").read_text())
    header = json.loads((tape_dir / "header.json").read_text())
    trailer = json.loads((tape_dir / "trailer.json").read_text())
    reader = orchard_tape.TapeReader(tape_dir)

    assert header["schema"] == "video/tape/1"
    assert header["git"] == "test-fixture"
    assert header["meta"]["provenance"]["synthetic"] is True
    assert header["meta"]["time"]["interval"] == [0.0, 1.0]
    assert "gauge" in header["meta"]["potential_gauge"]
    assert trailer["clamped_positions_total"] == 0
    assert len(reader.frames) == payload["num_steps"] + 1 == 5
    assert [record["t"] for record in reader.frames] == pytest.approx(
        np.linspace(0.0, 1.0, 5)
    )

    origin = np.asarray(header["meta"]["coordinate_convention"]["tape_origin"])
    tolerance = max(header["box"]) / 65535.0 + 1e-6
    for index in range(5):
        tape_frame = reader.frame(index)
        source_3d = np.column_stack(
            (payload["frames"][index]["positions"], payload["frames"][index]["potentials"])
        )
        np.testing.assert_allclose(tape_frame["pos"] + origin, source_3d, atol=tolerance)
        np.testing.assert_allclose(
            tape_frame["vel"][:, :2],
            np.tile((2.0, 3.0), (payload["num_particles"], 1)),
            atol=1e-6,
        )
        np.testing.assert_allclose(tape_frame["vel"][:, 2], 14.0, atol=1e-6)


def test_export_rejects_non_finite_model_output(tmp_path: Path) -> None:
    params = ModelParams(
        weights=[jnp.asarray([[np.nan], [0.0], [0.0]])],
        biases=[jnp.asarray([0.0])],
    )
    with pytest.raises(ValueError, match="non-finite"):
        export_trajectory_tape(params, tmp_path / "bad", num_particles=2, num_steps=1)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_particles": 0}, "num_particles"),
        ({"num_steps": 0}, "num_steps"),
        ({"box_size": float("nan")}, "box_size"),
    ],
)
def test_export_validates_dimensions(
    tmp_path: Path, kwargs: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        export_trajectory_tape(_linear_potential(), tmp_path / "bad", **kwargs)


def test_missing_orchard_dependency_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_import = importlib.import_module

    def missing(name: str, package: str | None = None):
        if name == "orchard_tape":
            raise ImportError("not installed")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="official 'orchard_tape' package"):
        export_trajectory_tape(_linear_potential(), tmp_path / "tape")


def test_official_bundle_is_verified(tmp_path: Path) -> None:
    tape_dir = export_trajectory_tape(
        _linear_potential(),
        tmp_path / "tape",
        num_particles=8,
        num_steps=2,
        provenance={"synthetic": True, "purpose": "bundle smoke test"},
        git_sha="test-fixture",
    )
    bundle_dir = create_orchard_bundle(
        tape_dir, tmp_path / "bundles", title="Synthetic bundle test", commit="test-fixture"
    )
    bundle = importlib.import_module("orchard.bundle")
    manifest = json.loads((bundle_dir / "bundle.json").read_text())

    assert manifest["schema"] == "orchard/bundle/1"
    assert manifest["source"]["clamped_positions"] == 0
    assert bundle.verify_bundle(bundle_dir)["ok"] is True


def test_gallery_is_offline_accessible_and_copies_pinned_assets(tmp_path: Path) -> None:
    gallery_html = tmp_path / "gallery" / "index.html"
    generate_webxr_gallery_html(
        gallery_html,
        particles_json_rel_path="tape/webxr_particles.json",
        title='Research <flow> "A"',
    )
    document = gallery_html.read_text(encoding="utf-8")

    assert "https://" not in document
    assert "http://" not in document
    assert "Research &lt;flow&gt; &quot;A&quot;" in document
    assert "VRButton.createButton" in document
    assert 'id="scrubber"' in document
    assert "prefers-reduced-motion" in document
    assert "Could not load trajectory data" in document
    assert "Data preview mode" not in document
    assert {path.name for path in (gallery_html.parent / "assets").iterdir()} == {
        "three.module.min.js",
        "OrbitControls.js",
        "VRButton.js",
        "THREE-LICENSE.txt",
    }
