from pathlib import Path

import jax

from quantumflow.orchard_export import (
    export_trajectory_tape,
    generate_webxr_gallery_html,
)
from quantumflow.ot_cfm import init_potential_network


def test_export_trajectory_tape_and_gallery(tmp_path: Path) -> None:
    key = jax.random.key(42)
    params = init_potential_network(key, in_dim=2, hidden_dims=(16, 16))

    tape_dir = tmp_path / "test_tape"
    export_trajectory_tape(
        params=params,
        output_tape_dir=tape_dir,
        num_particles=32,
        num_steps=5,
        seed=42,
    )

    assert (tape_dir / "webxr_particles.json").exists()
    if (tape_dir / "header.json").exists():
        assert (tape_dir / "data.bin").exists()

    gallery_html = tmp_path / "gallery.html"
    generate_webxr_gallery_html(gallery_html, particles_json_rel_path="webxr_particles.json")
    assert gallery_html.exists()
    assert "Three.js" in gallery_html.read_text(encoding="utf-8")
