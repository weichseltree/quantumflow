# ruff: noqa: E501
"""Export QuantumFlow trajectories to Orchard tapes, bundles, and a WebXR viewer."""

from __future__ import annotations

import html
import importlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.distance import cdist

from quantumflow.ot_cfm import ModelParams, evaluate_potential, integrate_ode, velocity

_ASSET_DIR = Path(__file__).with_name("gallery_assets")
_ASSETS = ("three.module.min.js", "OrbitControls.js", "VRButton.js", "THREE-LICENSE.txt")


def _require_module(name: str, purpose: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise RuntimeError(
            f"{purpose} requires the official {name!r} package. "
            "Install Orchard and orchard-tape in this environment; no fallback format is written."
        ) from exc


def _finite(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value)
    if not np.isfinite(array).all():
        bad = np.argwhere(~np.isfinite(array))[0].tolist()
        raise ValueError(f"{name} contains a non-finite value at index {bad}")
    return array


def _validated_provenance(provenance: Mapping[str, Any] | None) -> dict[str, Any]:
    result = dict(provenance or {})
    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("provenance must be finite JSON-serializable data") from exc
    return result


def export_trajectory_tape(
    params: ModelParams,
    output_tape_dir: Path,
    num_particles: int = 512,
    num_steps: int = 60,
    seed: int = 42,
    box_size: float | None = None,
    title: str = "Isotropic-Hessian OT-CFM Particle Flow",
    *,
    provenance: Mapping[str, Any] | None = None,
    source_run: str | None = None,
    source_model: str | None = None,
    git_sha: str | None = None,
) -> Path:
    """Write a ``video/tape/1`` trajectory and its browser-viewer sidecar.

    Scientific coordinates are ``(q1, q2, Phi(t, q))``. The tape is translated
    into a tight positive quantization box; ``meta.coordinate_convention``
    records the inverse translation. Flow time is dimensionless and normalized
    to ``[0, 1]``. ``Phi`` has an arbitrary time-dependent additive gauge, so
    absolute vertical position is not comparable across independently trained
    models unless their gauges are fixed.

    ``box_size`` is retained for compatibility and, when supplied, is a minimum
    per-axis box extent. It never clips a trajectory.
    """
    orchard_tape = _require_module("orchard_tape", "trajectory export")
    if not isinstance(num_particles, int) or num_particles <= 0:
        raise ValueError("num_particles must be a positive integer")
    if not isinstance(num_steps, int) or num_steps <= 0:
        raise ValueError("num_steps must be a positive integer")
    if box_size is not None and (not np.isfinite(box_size) or box_size <= 0):
        raise ValueError("box_size must be finite and positive when supplied")

    scientific_provenance = _validated_provenance(provenance)
    if source_run is not None:
        scientific_provenance["source_run"] = source_run
    if source_model is not None:
        scientific_provenance["source_model"] = source_model

    key = jax.random.key(seed)
    x0_2d = jax.random.normal(key, shape=(num_particles, 2))
    _, trajectory_2d = integrate_ode(params, x0_2d, num_steps=num_steps)

    times = np.linspace(0.0, 1.0, num_steps + 1, dtype=np.float64)
    positions: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    potentials: list[np.ndarray] = []
    planar_positions: list[np.ndarray] = []
    planar_velocities: list[np.ndarray] = []

    for step_idx, x_t in enumerate(trajectory_2d):
        t = jnp.asarray(times[step_idx])
        phi = jax.vmap(lambda point: evaluate_potential(params, t, point))(x_t)
        v_xy = jax.vmap(lambda point: velocity(params, t, point))(x_t)
        partial_t = jax.vmap(
            lambda point: jax.grad(lambda time: evaluate_potential(params, time, point))(t)
        )(x_t)
        # Along a trajectory, dPhi/dt = partial_t Phi + grad(Phi) dot dq/dt.
        v_phi = partial_t + jnp.sum(jnp.square(v_xy), axis=1)

        xy = _finite(f"trajectory frame {step_idx}", x_t).astype(np.float64)
        phi_np = _finite(f"potential frame {step_idx}", phi).astype(np.float64)
        v_xy_np = _finite(f"velocity frame {step_idx}", v_xy).astype(np.float64)
        v_phi_np = _finite(f"potential derivative frame {step_idx}", v_phi).astype(np.float64)
        planar_positions.append(xy)
        planar_velocities.append(v_xy_np)
        potentials.append(phi_np)
        positions.append(np.column_stack((xy, phi_np)))
        velocities.append(np.column_stack((v_xy_np, v_phi_np)))

    all_positions = np.concatenate(positions, axis=0)
    lower_raw = all_positions.min(axis=0)
    upper_raw = all_positions.max(axis=0)
    span = upper_raw - lower_raw
    padding = np.maximum(span * 0.025, 1e-4)
    extent = span + 2.0 * padding
    if box_size is not None:
        extent = np.maximum(extent, float(box_size))
    center = (lower_raw + upper_raw) / 2.0
    origin = center - extent / 2.0
    tape_positions = [frame - origin for frame in positions]

    final_pts = planar_positions[-1]
    angles = np.linspace(0.0, 2.0 * np.pi, 9)[:-1]
    centers = np.stack((2.0 * np.cos(angles), 2.0 * np.sin(angles)), axis=1)
    species_labels = np.argmin(cdist(final_pts, centers), axis=1).astype(np.uint8)

    coordinate_convention = {
        "scientific": ["q1", "q2", "Phi(t,q)"],
        "tape": "scientific - tape_origin",
        "tape_origin": origin.tolist(),
        "viewer": ["q1", "Phi(t,q)", "q2"],
    }
    meta = {
        "title": title,
        "description": "Optimal Transport Conditional Flow Matching trajectory",
        "species_names": [f"Target mode {index}" for index in range(8)],
        "coordinate_convention": coordinate_convention,
        "time": {
            "coordinate": "normalized flow time",
            "unit": "dimensionless",
            "interval": [0.0, 1.0],
        },
        "units": {
            "q1": "dimensionless model coordinate",
            "q2": "dimensionless model coordinate",
            "Phi": "dimensionless learned potential",
        },
        "potential_gauge": (
            "Phi is defined only up to a time-dependent additive gauge; absolute vertical "
            "offsets are not comparable across models unless the gauge is fixed."
        ),
        "provenance": scientific_provenance,
    }

    output_tape_dir = Path(output_tape_dir)
    output_tape_dir.mkdir(parents=True, exist_ok=True)
    with orchard_tape.TapeWriter(
        path=output_tape_dir,
        box=tuple(float(value) for value in extent),
        n_total=num_particles,
        run_seed=seed,
        quantize="uint16",
        periodic=(False, False, False),
        velocity=True,
        scalars=(("species", "uint8"), ("potential", "float32")),
        units="dimensionless model coordinates; normalized flow time [0,1]",
        t0_origin="start of normalized OT-CFM integration",
        t0_offset_tau=0.0,
        meta=meta,
        git_sha=git_sha,
    ) as writer:
        for step_idx, time in enumerate(times):
            writer.append(
                step=step_idx,
                time=float(time),
                pos=tape_positions[step_idx],
                vel=velocities[step_idx],
                species=species_labels,
                potential=potentials[step_idx].astype(np.float32),
            )

    payload = {
        "title": title,
        "synthetic": bool(scientific_provenance.get("synthetic", False)),
        "num_particles": num_particles,
        "num_steps": num_steps,
        "coordinate_convention": coordinate_convention,
        "time_unit": "dimensionless normalized flow time",
        "potential_gauge": meta["potential_gauge"],
        "provenance": scientific_provenance,
        "species": species_labels.tolist(),
        "bounds": {"min": lower_raw.tolist(), "max": upper_raw.tolist()},
        "frames": [
            {
                "step": index,
                "time": float(times[index]),
                "positions": planar_positions[index].tolist(),
                "potentials": potentials[index].tolist(),
                "velocities": planar_velocities[index].tolist(),
            }
            for index in range(num_steps + 1)
        ],
    }
    (output_tape_dir / "webxr_particles.json").write_text(
        json.dumps(payload, allow_nan=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return output_tape_dir


def create_orchard_bundle(
    tape_dir: Path,
    bundle_output_dir: Path,
    title: str = "QuantumFlow OT-CFM Transport Flow",
    *,
    commit: str | None = None,
) -> Path:
    """Create and verify an official ``orchard/bundle/1`` tape bundle."""
    bundle = _require_module("orchard.bundle", "bundle creation")
    bundle_dir = Path(
        bundle.bundle_tape(
            tape_dir=tape_dir,
            tree="quantumflow",
            title=title,
            out_root=bundle_output_dir,
            verbose=False,
            commit=commit,
        )
    )
    report = bundle.verify_bundle(bundle_dir)
    if not report.get("ok"):
        raise RuntimeError(f"Orchard bundle verification failed: {report}")
    return bundle_dir


def generate_webxr_gallery_html(
    output_html_path: Path,
    particles_json_rel_path: str = "webxr_particles.json",
    title: str = "QuantumFlow OT-CFM VR Gallery",
) -> None:
    """Write an offline-capable Three.js gallery with graceful WebXR fallback."""
    output_html_path = Path(output_html_path)
    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    assets_dir = output_html_path.parent / "assets"
    assets_dir.mkdir(exist_ok=True)
    for asset in _ASSETS:
        source = _ASSET_DIR / asset
        if not source.is_file():
            raise RuntimeError(f"Vendored gallery asset is missing: {source}")
        shutil.copy2(source, assets_dir / asset)

    document = _GALLERY_HTML.replace("__TITLE_HTML__", html.escape(title)).replace(
        "__TITLE_JSON__", json.dumps(title)
    ).replace("__DATA_URL__", json.dumps(particles_json_rel_path))
    output_html_path.write_text(document, encoding="utf-8")


_GALLERY_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
  <meta name="color-scheme" content="dark">
  <title>__TITLE_HTML__</title>
  <style>
    :root { color-scheme: dark; font-family: ui-sans-serif, system-ui, sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; min-height: 100dvh; overflow: hidden; background: #07090d; color: #f6f7f9; }
    canvas { display: block; width: 100%; height: 100%; }
    .panel { position: fixed; z-index: 2; background: rgba(12,15,21,.92); border: 1px solid #353b46;
      box-shadow: 0 14px 36px rgba(0,0,0,.42); }
    #info { inset: max(16px,env(safe-area-inset-top)) auto auto max(16px,env(safe-area-inset-left));
      width: min(430px,calc(100vw - 32px)); padding: 18px 20px; }
    h1 { margin: 0 0 8px; max-width: 20ch; font-size: clamp(1.25rem,3vw,2rem); line-height: 1.05;
      letter-spacing: -.03em; overflow-wrap: anywhere; }
    p { margin: 5px 0; max-width: 65ch; color: #b9c1cc; font-size: .9rem; line-height: 1.45; }
    #status[data-state="error"] { color: #ffb4aa; }
    #status[data-state="ready"] { color: #a9dfc2; }
    #meta { font-variant-numeric: tabular-nums; }
    #retry { display: none; margin-top: 12px; }
    #controls { inset: auto auto max(16px,env(safe-area-inset-bottom)) 50%; transform: translateX(-50%);
      width: min(720px,calc(100vw - 32px)); padding: 12px; display: grid;
      grid-template-columns: auto auto minmax(150px,1fr) auto; gap: 10px; align-items: center; }
    button, select, input { font: inherit; }
    button, select { min-height: 44px; border: 1px solid #515968; background: #171c24; color: #fff;
      padding: 8px 14px; cursor: pointer; }
    button:hover, select:hover { background: #242b36; }
    button:focus-visible, select:focus-visible, input:focus-visible { outline: 3px solid #7dd3fc; outline-offset: 2px; }
    button:disabled, select:disabled, input:disabled { opacity: .48; cursor: not-allowed; }
    input[type="range"] { width: 100%; min-height: 44px; accent-color: #ff7657; cursor: pointer; }
    #xr-slot { position: fixed; right: max(16px,env(safe-area-inset-right));
      top: max(16px,env(safe-area-inset-top)); z-index: 3; }
    #xr-slot button { position: static !important; width: auto !important; opacity: 1 !important; }
    .sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px;
      overflow: hidden; clip: rect(0,0,0,0); white-space: nowrap; border: 0; }
    @media (max-width: 620px) {
      #info { padding: 14px 16px; }
      #info p:not(#status):not(#meta) { display: none; }
      #xr-slot { top: auto; bottom: calc(86px + env(safe-area-inset-bottom)); }
      #controls { grid-template-columns: 1fr 1fr auto; }
      #scrubber-wrap { grid-column: 1 / -1; grid-row: 1; }
    }
    @media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto !important; transition: none !important; } }
  </style>
  <script type="importmap">{"imports":{"three":"./assets/three.module.min.js"}}</script>
</head>
<body>
  <section id="info" class="panel" aria-labelledby="title">
    <h1 id="title">__TITLE_HTML__</h1>
    <p>Coordinates: q1 × q2, lifted vertically by the learned potential Φ(t,q).</p>
    <p id="status" data-state="loading" role="status" aria-live="polite">Loading trajectory…</p>
    <p id="meta">Frame — · t = —</p>
    <button id="retry" type="button">Retry data load</button>
  </section>
  <div id="xr-slot" aria-label="Immersive viewing"></div>
  <section id="controls" class="panel" aria-label="Playback controls">
    <button id="play" type="button" disabled>Play</button>
    <button id="restart" type="button" disabled>Restart</button>
    <label id="scrubber-wrap"><span class="sr-only">Trajectory frame</span>
      <input id="scrubber" type="range" min="0" max="0" value="0" disabled>
    </label>
    <label><span class="sr-only">Playback speed</span>
      <select id="speed" disabled aria-label="Playback speed">
        <option value=".5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option>
      </select>
    </label>
  </section>
  <script type="module">
    import * as THREE from 'three';
    import { OrbitControls } from './assets/OrbitControls.js';
    import { VRButton } from './assets/VRButton.js';

    const DATA_URL = __DATA_URL__;
    const TITLE = __TITLE_JSON__;
    const palette = [0xff7657,0x59c3ff,0xa7df78,0xf4c95d,0xb9a1ff,0xff9fc9,0x75e6c2,0xe8edf2];
    const ui = Object.fromEntries(['status','meta','retry','play','restart','scrubber','speed','xr-slot']
      .map(id => [id, document.getElementById(id)]));
    const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
    let data, points, frame = 0, playing = false, elapsed = 0, lastTime = 0;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x07090d);
    scene.fog = new THREE.FogExp2(0x07090d, .018);
    const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, .01, 1000);
    const renderer = new THREE.WebGLRenderer({antialias: true, powerPreference: 'high-performance'});
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.setSize(innerWidth, innerHeight);
    renderer.xr.enabled = true;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    document.body.prepend(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = !reducedMotion;
    controls.target.set(0, 0, 0);
    scene.add(new THREE.HemisphereLight(0xbfe6ff, 0x242018, 2.2));
    const grid = new THREE.GridHelper(10, 20, 0x485260, 0x202733);
    scene.add(grid);
    const xrButton = VRButton.createButton(renderer);
    ui['xr-slot'].append(xrButton);

    function fail(message) {
      playing = false;
      ui.status.dataset.state = 'error';
      ui.status.textContent = message;
      ui.retry.style.display = 'inline-block';
      for (const control of [ui.play, ui.restart, ui.scrubber, ui.speed]) control.disabled = true;
    }

    function validate(raw) {
      if (!raw || !Number.isInteger(raw.num_particles) || raw.num_particles < 1 ||
          !Array.isArray(raw.frames) || raw.frames.length < 2 ||
          !Array.isArray(raw.species) || raw.species.length !== raw.num_particles) {
        throw new Error('The trajectory file has an invalid shape.');
      }
      raw.frames.forEach((item, index) => {
        if (!Number.isFinite(item.time) || !Array.isArray(item.positions) ||
            !Array.isArray(item.potentials) || item.positions.length !== raw.num_particles ||
            item.potentials.length !== raw.num_particles) {
          throw new Error(`Frame ${index} is incomplete.`);
        }
        item.positions.forEach((position, particle) => {
          if (!Array.isArray(position) || position.length !== 2 ||
              !position.every(Number.isFinite) || !Number.isFinite(item.potentials[particle])) {
            throw new Error(`Frame ${index}, particle ${particle} is not finite.`);
          }
        });
      });
      return raw;
    }

    function setFrame(index) {
      frame = Math.max(0, Math.min(data.frames.length - 1, index));
      const source = data.frames[frame];
      const positions = points.geometry.attributes.position;
      for (let i = 0; i < data.num_particles; i++) {
        positions.setXYZ(i, source.positions[i][0], source.potentials[i], source.positions[i][1]);
      }
      positions.needsUpdate = true;
      ui.scrubber.value = String(frame);
      ui.meta.textContent = `Frame ${frame + 1} of ${data.frames.length} · t = ${source.time.toFixed(3)}`;
    }

    function buildParticles() {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(data.num_particles * 3), 3));
      const colors = new Float32Array(data.num_particles * 3);
      data.species.forEach((species, index) => {
        const color = new THREE.Color(palette[Math.abs(Number(species) || 0) % palette.length]);
        color.toArray(colors, index * 3);
      });
      geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
      points = new THREE.Points(geometry, new THREE.PointsMaterial({
        size: .075, sizeAttenuation: true, vertexColors: true, transparent: true, opacity: .92
      }));
      scene.add(points);
      setFrame(0);
      geometry.computeBoundingSphere();
      const bounds = data.bounds;
      const center = new THREE.Vector3(
        (bounds.min[0] + bounds.max[0]) / 2,
        (bounds.min[2] + bounds.max[2]) / 2,
        (bounds.min[1] + bounds.max[1]) / 2
      );
      const span = Math.max(...bounds.max.map((value, index) => value - bounds.min[index]), .1);
      controls.target.copy(center);
      camera.position.copy(center).add(new THREE.Vector3(span * 1.15, span * .8, span * 1.45));
      camera.near = Math.max(span / 1000, .001);
      camera.far = Math.max(span * 30, 20);
      camera.updateProjectionMatrix();
      grid.scale.setScalar(Math.max(span / 10, .1));
      grid.position.y = bounds.min[2];
    }

    async function loadData() {
      ui.retry.style.display = 'none';
      ui.status.dataset.state = 'loading';
      ui.status.textContent = 'Loading trajectory…';
      try {
        const response = await fetch(DATA_URL, {cache: 'no-store'});
        if (!response.ok) throw new Error(`HTTP ${response.status} ${response.statusText}`);
        data = validate(await response.json());
        if (points) { scene.remove(points); points.geometry.dispose(); points.material.dispose(); }
        buildParticles();
        ui.scrubber.max = String(data.frames.length - 1);
        for (const control of [ui.play, ui.restart, ui.scrubber, ui.speed]) control.disabled = false;
        playing = !reducedMotion;
        ui.play.textContent = playing ? 'Pause' : 'Play';
        const synthetic = data.synthetic ? ' Synthetic smoke data.' : '';
        ui.status.dataset.state = 'ready';
        ui.status.textContent = `${data.num_particles.toLocaleString()} particles loaded.${synthetic}`;
      } catch (error) {
        fail(`Could not load trajectory data: ${error.message} Serve this directory over HTTP and retry.`);
      }
    }

    ui.retry.addEventListener('click', loadData);
    ui.play.addEventListener('click', () => { playing = !playing; ui.play.textContent = playing ? 'Pause' : 'Play'; });
    ui.restart.addEventListener('click', () => { elapsed = 0; setFrame(0); });
    ui.scrubber.addEventListener('input', event => { playing = false; ui.play.textContent = 'Play'; setFrame(Number(event.target.value)); });
    addEventListener('resize', () => {
      camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
      renderer.setSize(innerWidth, innerHeight); renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    });

    renderer.setAnimationLoop(time => {
      const delta = Math.min((time - lastTime) / 1000, .1); lastTime = time;
      if (playing && data) {
        elapsed += delta * Number(ui.speed.value);
        const next = Math.floor(elapsed * 12) % data.frames.length;
        if (next !== frame) setFrame(next);
      }
      controls.update();
      renderer.render(scene, camera);
    });
    document.title = TITLE;
    loadData();
  </script>
</body>
</html>
"""
