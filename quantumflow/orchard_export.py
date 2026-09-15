# ruff: noqa: E501
"""Orchard VR Gallery and particle tape export for QuantumFlow transport flows."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

try:
    import orchard_tape
except ImportError:
    orchard_tape = None

try:
    from orchard.bundle import bundle_tape
except ImportError:
    bundle_tape = None

from quantumflow.ot_cfm import (
    ModelParams,
    evaluate_potential,
    integrate_ode,
    velocity,
)


def export_trajectory_tape(
    params: ModelParams,
    output_tape_dir: Path,
    num_particles: int = 512,
    num_steps: int = 60,
    seed: int = 42,
    box_size: float = 8.0,
    title: str = "Isotropic-Hessian OT-CFM Particle Flow",
) -> Path:
    """Export particle flow simulation into orchard video/tape/1 format."""
    output_tape_dir = Path(output_tape_dir)
    output_tape_dir.mkdir(parents=True, exist_ok=True)

    key = jax.random.key(seed)
    k_x0, k_target = jax.random.split(key)
    x0_2d = jax.random.normal(k_x0, shape=(num_particles, 2))

    # Integrate 2D ODE with RK4
    _, trajectory_2d = integrate_ode(params, x0_2d, num_steps=num_steps)

    # Convert 2D trajectory + potential into 3D positions [x, y, z]
    # Center the coordinate box around origin by offsetting [x + box_size/2, y + box_size/2, z + box_size/2]
    # for uint16 quantization in [0, box_size]
    box = (box_size, box_size, box_size)
    half_box = box_size / 2.0

    # Determine species based on which of 8 modes each final particle landed closest to
    angles = np.linspace(0, 2 * np.pi, 9)[:-1]
    centers = np.stack([2.0 * np.cos(angles), 2.0 * np.sin(angles)], axis=1)
    final_pts = np.asarray(trajectory_2d[-1])
    from scipy.spatial.distance import cdist
    dists = cdist(final_pts, centers)
    species_labels = np.argmin(dists, axis=1).astype(np.uint8)

    if orchard_tape is not None:
        writer = orchard_tape.TapeWriter(
            path=output_tape_dir,
            box=box,
            n_total=num_particles,
            run_seed=seed,
            quantize="uint16",
            periodic=(False, False, False),
            velocity=True,
            scalars=(("species", "uint8"), ("potential", "float32")),
            units="reduced (sigma, tau)",
            meta={
                "title": title,
                "description": "Optimal Transport Conditional Flow Matching in VR",
                "species_names": [f"Mode {i}" for i in range(8)],
                "box_offset": [-half_box, -half_box, -half_box],
            },
        )

        dt = 1.0 / num_steps
        for step_idx, x_t in enumerate(trajectory_2d):
            t_val = step_idx * dt
            # Evaluate potential z = Phi(t, x)
            phi_vals = jax.vmap(lambda p_: evaluate_potential(params, jnp.array(t_val), p_))(x_t)
            v_2d = jax.vmap(lambda p_: velocity(params, jnp.array(t_val), p_))(x_t)
            partial_t = jax.vmap(
                lambda p_: jax.grad(
                    lambda t_: evaluate_potential(params, t_, p_)
                )(jnp.array(t_val))
            )(x_t)

            # 3D positions with z scaled
            pos_x = np.asarray(x_t[:, 0]) + half_box
            pos_y = np.asarray(x_t[:, 1]) + half_box
            pos_z = np.asarray(phi_vals) * 0.5 + half_box
            pos_3d = np.stack([pos_x, pos_y, pos_z], axis=1)

            # 3D velocities
            vel_x = np.asarray(v_2d[:, 0])
            vel_y = np.asarray(v_2d[:, 1])
            vel_z = np.asarray(partial_t + jnp.sum(jnp.square(v_2d), axis=1)) * 0.5
            vel_3d = np.stack([vel_x, vel_y, vel_z], axis=1)

            writer.append(
                step=step_idx,
                time=float(t_val),
                pos=pos_3d,
                vel=vel_3d,
                species=species_labels,
                potential=np.asarray(phi_vals, dtype=np.float32),
            )
        writer.close()

    # Also export standalone JSON particle trajectory for direct WebXR viewer
    frames_json = []
    dt = 1.0 / num_steps
    for step_idx, x_t in enumerate(trajectory_2d):
        t_val = step_idx * dt
        phi_vals = jax.vmap(lambda p_: evaluate_potential(params, jnp.array(t_val), p_))(x_t)
        v_2d = jax.vmap(lambda p_: velocity(params, jnp.array(t_val), p_))(x_t)

        frames_json.append({
            "step": step_idx,
            "time": float(t_val),
            "positions": np.asarray(x_t).tolist(),
            "potentials": np.asarray(phi_vals).tolist(),
            "velocities": np.asarray(v_2d).tolist(),
        })

    payload = {
        "title": title,
        "num_particles": num_particles,
        "num_steps": num_steps,
        "species": species_labels.tolist(),
        "frames": frames_json,
    }
    with open(output_tape_dir / "webxr_particles.json", "w", encoding="utf-8") as f:
        json.dump(payload, f)

    return output_tape_dir


def create_orchard_bundle(
    tape_dir: Path,
    bundle_output_dir: Path,
    title: str = "QuantumFlow OT-CFM Transport Flow",
) -> Path | None:
    """Bundle the particle tape using orchard/bundle/1 format."""
    if bundle_tape is None:
        return None
    bundle_dir = bundle_tape(
        tape_dir=tape_dir,
        tree="quantumflow",
        title=title,
        out_root=bundle_output_dir,
    )
    return bundle_dir


def generate_webxr_gallery_html(
    output_html_path: Path,
    particles_json_rel_path: str = "webxr_particles.json",
    title: str = "QuantumFlow OT-CFM VR Gallery",
) -> None:
    """Generate a self-contained WebXR 3D VR gallery HTML page."""
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            margin: 0;
            overflow: hidden;
            background-color: #050811;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            color: #e2e8f0;
        }}
        #info-overlay {{
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(15, 23, 42, 0.85);
            padding: 16px 20px;
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(8px);
            z-index: 100;
            max-width: 360px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
        }}
        h1 {{
            font-size: 1.1rem;
            margin: 0 0 8px 0;
            color: #38bdf8;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }}
        p {{
            font-size: 0.85rem;
            margin: 4px 0;
            color: #94a3b8;
        }}
        .metric-badge {{
            display: inline-block;
            background: rgba(56, 189, 248, 0.15);
            color: #38bdf8;
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 0.8rem;
            margin-top: 6px;
        }}
        #controls {{
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            gap: 12px;
            z-index: 100;
            background: rgba(15, 23, 42, 0.85);
            padding: 10px 16px;
            border-radius: 30px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(8px);
        }}
        button {{
            background: #0284c7;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 20px;
            cursor: pointer;
            font-weight: 500;
            font-size: 0.85rem;
            transition: all 0.2s ease;
        }}
        button:hover {{
            background: #0369a1;
            transform: translateY(-1px);
        }}
    </style>
    <!-- Three.js and VRButton -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/webxr/VRButton.js"></script>
</head>
<body>
    <div id="info-overlay">
        <h1>{title}</h1>
        <p>Optimal Transport Conditional Flow Matching</p>
        <p id="status-text">Loading simulation data...</p>
        <div class="metric-badge" id="step-badge">Step: 0 / 60 (t = 0.00)</div>
    </div>

    <div id="controls">
        <button id="btn-play">Pause</button>
        <button id="btn-restart">Restart Flow</button>
        <button id="btn-speed">1x Speed</button>
    </div>

    <script>
        let scene, camera, renderer, controls;
        let particleSystem, particleGeometry;
        let flowData = null;
        let currentFrame = 0;
        let isPlaying = true;
        let playSpeed = 1.0;
        let lastTime = 0;

        const PALETTE = [
            0x38bdf8, 0x818cf8, 0xc084fc, 0xf472b6,
            0xfb7185, 0xfb923c, 0xfacc15, 0x4ade80
        ];

        async function init() {{
            scene = new THREE.Scene();
            scene.fog = new THREE.FogExp2(0x050811, 0.04);

            camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 100);
            camera.position.set(0, 6, 8);

            renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            renderer.xr.enabled = true;
            document.body.appendChild(renderer.domElement);

            document.body.appendChild(VRButton.createButton(renderer));

            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.target.set(0, 0, 0);

            // Lighting & Grid
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
            scene.add(ambientLight);

            const dirLight = new THREE.DirectionalLight(0x38bdf8, 0.8);
            dirLight.position.set(5, 10, 7);
            scene.add(dirLight);

            const grid = new THREE.GridHelper(12, 24, 0x1e293b, 0x0f172a);
            grid.position.y = -0.01;
            scene.add(grid);

            // Load particles JSON
            try {{
                const res = await fetch('{particles_json_rel_path}');
                flowData = await res.json();
                document.getElementById('status-text').innerText = 
                    `Loaded ${{flowData.num_particles}} particles across ${{flowData.num_steps}} steps.`;
                setupParticleSystem();
            }} catch (err) {{
                document.getElementById('status-text').innerText = "Data preview mode";
            }}

            window.addEventListener('resize', onWindowResize);
            setupUI();

            renderer.setAnimationLoop(render);
        }}

        function setupParticleSystem() {{
            if (!flowData) return;
            const count = flowData.num_particles;
            particleGeometry = new THREE.BufferGeometry();
            const positions = new Float32Array(count * 3);
            const colors = new Float32Array(count * 3);

            for (let i = 0; i < count; i++) {{
                const sp = flowData.species[i] % PALETTE.length;
                const c = new THREE.Color(PALETTE[sp]);
                colors[i * 3] = c.r;
                colors[i * 3 + 1] = c.g;
                colors[i * 3 + 2] = c.b;
            }}

            particleGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            particleGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

            const pMaterial = new THREE.PointsMaterial({{
                size: 0.12,
                vertexColors: true,
                transparent: true,
                opacity: 0.85,
                blending: THREE.AdditiveBlending
            }});

            particleSystem = new THREE.Points(particleGeometry, pMaterial);
            scene.add(particleSystem);
            updatePositions(0);
        }}

        function updatePositions(frameIdx) {{
            if (!flowData || !particleGeometry) return;
            const frame = flowData.frames[frameIdx];
            if (!frame) return;

            const posAttr = particleGeometry.attributes.position;
            const count = flowData.num_particles;
            for (let i = 0; i < count; i++) {{
                const p = frame.positions[i];
                const phi = frame.potentials ? frame.potentials[i] : 0.0;
                posAttr.setXYZ(i, p[0], phi * 0.4, p[1]);
            }}
            posAttr.needsUpdate = true;

            document.getElementById('step-badge').innerText = 
                `Step: ${{frame.step}} / ${{flowData.num_steps}} (t = ${{frame.time.toFixed(2)}})`;
        }}

        function setupUI() {{
            const btnPlay = document.getElementById('btn-play');
            btnPlay.addEventListener('click', () => {{
                isPlaying = !isPlaying;
                btnPlay.innerText = isPlaying ? 'Pause' : 'Play';
            }});

            document.getElementById('btn-restart').addEventListener('click', () => {{
                currentFrame = 0;
                updatePositions(0);
            }});

            const btnSpeed = document.getElementById('btn-speed');
            btnSpeed.addEventListener('click', () => {{
                if (playSpeed === 1.0) {{ playSpeed = 2.0; btnSpeed.innerText = '2x Speed'; }}
                else if (playSpeed === 2.0) {{ playSpeed = 0.5; btnSpeed.innerText = '0.5x Speed'; }}
                else {{ playSpeed = 1.0; btnSpeed.innerText = '1x Speed'; }}
            }});
        }}

        function onWindowResize() {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }}

        function render(time) {{
            const delta = (time - lastTime) * 0.001;
            lastTime = time;

            if (isPlaying && flowData) {{
                currentFrame += delta * 30 * playSpeed;
                if (currentFrame >= flowData.frames.length) {{
                    currentFrame = 0;
                }}
                updatePositions(Math.floor(currentFrame));
            }}

            controls.update();
            renderer.render(scene, camera);
        }}

        init();
    </script>
</body>
</html>
"""
    output_html_path = Path(output_html_path)
    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    output_html_path.write_text(html_content, encoding="utf-8")
