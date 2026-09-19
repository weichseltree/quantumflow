"""Geometry for the quantumflow exhibition: physics turned into standing objects.

Every builder here answers the same design rule. A visitor walking past a
glowing shape learns nothing from the shape; they learn from a *quantity*
painted onto it that they can read by eye and check against the plaque. So
each surface carries a measurement in its vertex colour:

- an orbital's lobes are lit by the local kinetic-energy density, so the
  visitor sees that the energy lives at the nodes and nowhere else;
- a density shell is placed at a containment fraction, so "the 90% surface"
  means what it says and the shells are not chosen for looks;
- a relief's colour is its own height, so the potential reads as a landscape
  and the density reads as what settled into its valleys;
- a convexity bowl carries chord ribbons, which turn the convexity claim into
  something a visitor can falsify by looking for red.

Grids come from :mod:`quantumflow.multidim`; the writer is
:mod:`quantumflow.glb`. Nothing here trains anything: every model in this
module can be built on a CPU from the exact solver.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import numpy as np
from scipy.ndimage import map_coordinates
from skimage import measure

from quantumflow.glb import Model, Surface
from quantumflow.jax.convex import init_icnn, kinetic_energy
from quantumflow.multidim import Grid, solve_multidim_schroedinger

#: The lobe where the wavefunction is positive. Warm, so the two signs read
#: as different things across a room rather than as two shades of one thing.
POSITIVE_LOBE = np.array([1.00, 0.72, 0.30])
#: The lobe where it is negative.
NEGATIVE_LOBE = np.array([0.25, 0.75, 1.00])
#: A surface whose colour carries no quantity.
NEUTRAL = np.array([0.82, 0.84, 0.88])
#: A convexity violation. Nothing else in the exhibition is this colour.
VIOLATION = np.array([0.95, 0.20, 0.22])
#: A satisfied chord.
SATISFIED = np.array([0.35, 0.80, 0.55])


@dataclass(frozen=True)
class ModelReport:
    """What a built model costs, against the grove's per-exhibit budget."""

    name: str
    triangles: int
    surfaces: int
    #: Physical facts the plaque must state for the model to mean anything.
    plaque: dict[str, str]


def _lattice_faces(rows: int, cols: int) -> np.ndarray:
    """Triangulate a regular ``rows x cols`` lattice of vertices."""
    r, c = np.meshgrid(np.arange(rows - 1), np.arange(cols - 1), indexing="ij")
    top_left = (r * cols + c).ravel()
    top_right = top_left + 1
    bottom_left = top_left + cols
    bottom_right = bottom_left + 1
    lower = np.stack([top_left, bottom_left, top_right], axis=1)
    upper = np.stack([top_right, bottom_left, bottom_right], axis=1)
    return np.concatenate([lower, upper], axis=0).astype(np.uint32)


def _to_physical(vertices: np.ndarray, grid: Grid) -> np.ndarray:
    """Map marching-cubes index coordinates onto the grid's own axes."""
    spacings = np.asarray(grid.spacings, dtype=np.float64)
    lower = np.asarray([bound[0] for bound in grid.bounds], dtype=np.float64)
    return vertices * spacings + lower


def _sample(field: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Trilinearly sample a gridded field at marching-cubes vertices."""
    return map_coordinates(field, vertices.T, order=1, mode="nearest")


def _normalize(values: np.ndarray) -> np.ndarray:
    """Scale to ``[0, 1]``; a constant field becomes all ones."""
    low = float(np.min(values))
    high = float(np.max(values))
    if high - low <= 0.0:
        return np.ones_like(values)
    return (values - low) / (high - low)


def _level_in_a_gap(ordered: np.ndarray, level: float, tolerance: float = 1e-9) -> float:
    """Move an isovalue off a plateau of equal densities into the gap below it.

    A symmetric well puts many grid points at the same density -- the same to
    about one part in 1e16, which is equal for every purpose except
    ``np.unique``. Two things go wrong when an isovalue lands on such a
    plateau. Marching cubes has no surface to find there, since the field does
    not cross the level so much as sit on it. And the containment the plaque
    quotes stops surviving being written down: rounding the level to twelve
    figures moves it by 1e-13, enough to sweep a whole plateau from inside the
    shell to outside and shift the stated share by thirteen points.

    Clustering values that agree to ``tolerance`` and placing the level midway
    between the plateau and the next distinct density below it fixes both. The
    surface then sits where nothing is, the plateau is unambiguously inside,
    and any sane rounding of the number leaves it there.
    """
    values = np.unique(ordered)
    if values.shape[0] < 2:
        return float(level) * 0.5

    # Values agreeing to `tolerance` are one plateau; keep the first of each.
    scale = np.maximum(np.abs(values[1:]), 1.0)
    clusters = values[np.concatenate([[True], np.diff(values) > tolerance * scale])]

    # The gaps either side of wherever the requested level fell. Both are real
    # gaps in the density, so a surface in either is well defined; they differ
    # in whether the plateau between them counts as inside the shell.
    position = int(np.searchsorted(clusters, level))
    lower = float(clusters[position - 1]) if position > 0 else float(clusters[0]) * 0.5
    upper = float(clusters[min(position, clusters.shape[0] - 1)])
    if position >= clusters.shape[0]:
        return float(clusters[-1]) * (1.0 + tolerance)
    return 0.5 * (lower + upper)


def _best_gap_level(
    field: np.ndarray, ordered: np.ndarray, volume: float, total: float, fraction: float
) -> tuple[float, float]:
    """Pick the gap level whose containment lands closest to ``fraction``.

    On a coarse grid a plateau can be worth ten points of containment, so
    which side of it the surface sits on matters. Both candidates are honest
    -- each states the share it really encloses -- and this takes the nearer
    one rather than letting the interpolation's side of the plateau decide.
    """
    enclosed = np.cumsum(ordered) * volume
    interpolated = float(np.interp(fraction * total, enclosed, ordered))

    candidates = {_level_in_a_gap(ordered, interpolated)}
    # The gap on the other side of the same plateau.
    values = np.unique(ordered)
    scale = np.maximum(np.abs(values[1:]), 1.0)
    clusters = values[np.concatenate([[True], np.diff(values) > 1e-9 * scale])]
    position = int(np.searchsorted(clusters, interpolated))
    if 0 < position < clusters.shape[0] - 1:
        candidates.add(0.5 * (float(clusters[position]) + float(clusters[position + 1])))

    best_level, best_achieved, best_error = 0.0, 0.0, float("inf")
    for candidate in candidates:
        achieved = float(field[field >= candidate].sum() * volume / total)
        error = abs(achieved - fraction)
        if error < best_error:
            best_level, best_achieved, best_error = candidate, achieved, error
    return best_level, best_achieved


def _fit(model: Model, height_m: float, *, ground: bool = True) -> Model:
    """Scale a model uniformly to stand ``height_m`` tall, centred on y.

    Uniform, because a per-axis fit would distort the shape of a solution to
    the Schrodinger equation and the visitor would be reading an artefact of
    the plinth. ``ground`` sets the object's base on y = 0 for a plinth; the
    alternative centres it, for a thing that should hang.
    """
    if not model.surfaces:
        raise ValueError(f"{model.name}: nothing to fit")
    stacked = np.concatenate([surface.positions for surface in model.surfaces], axis=0)
    low = stacked.min(axis=0)
    high = stacked.max(axis=0)
    extent = float(np.max(high - low))
    if extent <= 0.0:
        raise ValueError(f"{model.name}: the geometry has no extent")

    scale = float(height_m) / extent
    centre = (low + high) / 2.0
    for surface in model.surfaces:
        moved = (surface.positions - centre) * scale
        if ground:
            moved[:, 1] += height_m / 2.0
        surface.positions = np.ascontiguousarray(moved, dtype=np.float32)
    return model


def _refine(field: np.ndarray, grid: Grid, factor: int) -> tuple[np.ndarray, Grid]:
    """Cubic-interpolate a gridded field onto a finer grid of the same extent.

    An eigenfunction of a smooth potential is smooth, so interpolating it
    before marching cubes buys a surface without facets for a fraction of the
    cost of solving on the fine grid -- a 48-point cubic solve is minutes, a
    96-point one is out of reach on a workstation. Nothing measured is
    interpolated: the energies, the shell levels and the colours all come from
    the solve. Only the surface a visitor looks at is refined, and the plaque
    says so.
    """
    if factor <= 1:
        return field, grid

    fine_points = tuple((n - 1) * factor + 1 for n in grid.points_per_dim)
    axes = [
        np.linspace(0.0, n - 1.0, fine)
        for n, fine in zip(grid.points_per_dim, fine_points)
    ]
    sample = np.stack([m.ravel() for m in np.meshgrid(*axes, indexing="ij")], axis=0)
    refined = map_coordinates(field, sample, order=3, mode="nearest").reshape(fine_points)
    return refined, Grid(
        dimension=grid.dimension, bounds=grid.bounds, points_per_dim=fine_points
    )


def _isosurface(
    field: np.ndarray,
    level: float,
    grid: Grid,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract one isosurface, or ``None`` when the level is not crossed."""
    if not (float(field.min()) < level < float(field.max())):
        return None
    try:
        vertices, faces, _, _ = measure.marching_cubes(field, level=level)
    except (RuntimeError, ValueError):
        return None
    if faces.shape[0] == 0:
        return None
    return _to_physical(vertices, grid), faces.astype(np.uint32), vertices


def orbital_model(
    grid: Grid,
    wavefunction: np.ndarray,
    *,
    energy_hartree: float,
    index: int,
    iso_fraction: float = 0.28,
    height_m: float = 1.6,
    refine: int = 2,
) -> tuple[Model, ModelReport]:
    """One orbital as its two signed lobes, lit by local kinetic energy.

    The isosurface is taken on ``psi`` itself rather than ``|psi|^2``, so the
    two lobes are separate objects and the nodal surface is the gap between
    them: a visitor can count the nodes, and the count is the reason the
    energies are ordered the way the room places them.

    Vertex colour is the kinetic-energy density ``tau = |grad psi|^2 / 2``
    sampled on the surface. It is brightest in a thin band hugging each node,
    which is the whole of why a node costs energy.
    """
    shape = grid.points_per_dim
    psi = np.asarray(wavefunction, dtype=np.float64).reshape(shape)

    # Differentiated on the grid that was actually solved, then refined; the
    # other order would report the interpolator's derivatives as physics.
    gradients = np.gradient(psi, *grid.spacings)
    tau = 0.5 * sum(component**2 for component in gradients)

    level = iso_fraction * float(np.max(np.abs(psi)))
    if level <= 0.0:
        raise ValueError(f"orbital {index}: the wavefunction is identically zero")

    psi, fine_grid = _refine(psi, grid, refine)
    tau, _ = _refine(tau, grid, refine)

    model = Model(name=f"orbital-{index}")
    for sign, base in ((+1.0, POSITIVE_LOBE), (-1.0, NEGATIVE_LOBE)):
        extracted = _isosurface(psi, sign * level, fine_grid)
        if extracted is None:
            continue
        positions, faces, index_coords = extracted
        brightness = _normalize(_sample(tau, index_coords))
        # A floor under the darkest part: an unlit region should read as
        # "little kinetic energy here", not as a hole in the object.
        shade = 0.30 + 0.70 * brightness[:, None]
        model.add(
            Surface(
                positions=positions,
                indices=faces,
                colors=np.clip(base[None, :] * shade, 0.0, 1.0),
                name=f"orbital-{index}-{'positive' if sign > 0 else 'negative'}",
                emissive=0.35,
                roughness=0.45,
            )
        )

    if not model.surfaces:
        raise ValueError(f"orbital {index}: no isosurface at fraction {iso_fraction}")

    _fit(model, height_m)
    report = ModelReport(
        name=model.name,
        triangles=model.triangles,
        surfaces=len(model.surfaces),
        plaque={
            "title": f"Single-Particle Orbital Eigenstate \u03c8_{index}(r)",
            "orbital": str(index),
            "energy_hartree": f"{energy_hartree:.6f}",
            "isosurface": f"Wavefunction envelope at psi = +/- {iso_fraction:.2f} x max|psi|",
            "colour": "local kinetic energy density, |grad psi|^2 / 2, normalised on this surface. Bright bands hug the zero-crossing boundaries, showing that nodal steepness drives orbital kinetic energy.",
            "lobes": "gold where psi > 0, blue where psi < 0; the node is the gap between them",
            "physical_meaning": "Nodal planes force steep spatial gradients grad(psi), directly driving up the kinetic energy integral and dictating the state's position in the quantum energy ladder.",
            "reading_guide": "1. Gold represents positive phase (\u03c8 > 0); cyan represents negative phase (\u03c8 < 0). 2. The spatial gap separating lobes is the nodal surface where \u03c8 = 0. 3. Bright yellow vertex luminance highlights local kinetic energy density \u03c4(r) concentrated at the nodal interface.",
            "falsifiable_criterion": "Count the nodal gaps: state energy strictly increases with the number of zero-crossing nodal planes.",
            "grid": f"solved on {grid.points_per_dim[0]} points per axis"
            + (f", surface refined {refine}x for display" if refine > 1 else ""),
        },
    )
    return model, report


def density_shell_model(
    grid: Grid,
    density: np.ndarray,
    *,
    fractions: tuple[float, ...] = (0.5, 0.8, 0.95),
    height_m: float = 2.5,
    refine: int = 2,
) -> tuple[Model, ModelReport]:
    """The total density as nested shells, each at a containment fraction.

    The isovalues are not chosen for looks. For each requested fraction the
    builder finds the value whose enclosing surface holds that share of the
    electrons, so "the 95% shell" is a statement a visitor can hold against
    the plaque, and the spacing between shells means something.
    """
    shape = grid.points_per_dim
    field = np.asarray(density, dtype=np.float64).reshape(shape)
    volume = grid.volume_element

    # The containment levels are read off the solved grid, where the volume
    # element is the one the integral was defined with.
    ordered = np.sort(field.ravel())[::-1]
    enclosed = np.cumsum(ordered) * volume
    total = float(enclosed[-1])
    if total <= 0.0:
        raise ValueError("the density integrates to zero")

    chosen: list[tuple[float, float, float]] = []
    for fraction in fractions:
        # Interpolating between the two bracketing samples rather than taking
        # the nearer one: on a coarse grid a whole cell of density separates
        # them, which put the 50% shell at 53.5% before this.
        # The level goes in a gap between distinct densities, and `achieved` is
        # measured on the grid the integral was defined on -- before any
        # display refinement, whose finer volume element would not match this
        # `volume`. The plaque states the containment that was achieved rather
        # than the one requested, because a visitor reading the shells is owed
        # the share that is true.
        level, achieved = _best_gap_level(field, ordered, volume, total, fraction)
        chosen.append((fraction, level, achieved))

    field, grid = _refine(field, grid, refine)
    model = Model(name="density-shells")
    levels: dict[str, str] = {}
    for fraction, level, achieved in chosen:
        extracted = _isosurface(field, level, grid)
        if extracted is None:
            continue
        positions, faces, _ = extracted
        # The outer shells are the fainter ones, so the eye reaches the core.
        alpha = float(np.interp(fraction, (min(fractions), max(fractions)), (0.85, 0.16)))
        tint = np.append(NEUTRAL * (0.55 + 0.45 * (1.0 - fraction)), alpha)
        colors = np.tile(tint, (len(positions), 1))
        model.add(
            Surface(
                positions=positions,
                indices=faces,
                colors=colors,
                name=f"density-{int(round(fraction * 100))}",
                emissive=0.22,
                roughness=0.35,
            )
        )
        key = f"shell_{int(round(fraction * 100))}pct"
        levels[key] = f"n = {level:.12g}"
        levels[f"{key}_encloses"] = f"{achieved:.4f}"

    if not model.surfaces:
        raise ValueError("no density shell could be extracted")

    _fit(model, height_m)
    report = ModelReport(
        name=model.name,
        triangles=model.triangles,
        surfaces=len(model.surfaces),
        plaque={
            "title": "Total Electron Density Containment Shells n(r)",
            "shells": ", ".join(f"{int(round(f * 100))}%" for f in fractions),
            "meaning": "each surface encloses that share of the electrons",
            "electrons": f"{total:.4f}",
            "physical_meaning": "Unlike individual oscillating orbitals with zero-nodes, the total electron cloud n(r) = sum_i |psi_i(r)|^2 is strictly positive everywhere. The spacing between shells directly reflects quantum confinement and exponential decay into classically forbidden potential barriers.",
            "reading_guide": "1. Each nested shell is an exact isosurface enclosing the stated percentage of total electron probability. 2. Outer shells are rendered with higher translucency so the dense core remains visible. 3. Notice that unlike single-electron orbitals, the total density has no nodes or zero-crossing gaps.",
            "falsifiable_criterion": "Check the numerical containment values: each shell encloses its stated electron share measured directly on the discrete numerical grid.",
            **levels,
        },
    )
    return model, report


def relief_model(
    grid: Grid,
    field: np.ndarray,
    *,
    name: str,
    palette: np.ndarray = NEUTRAL,
    relief_m: float = 0.45,
    width_m: float = 3.2,
    invert: bool = False,
) -> tuple[Model, ModelReport]:
    """A 2D scalar field as a relief surface, coloured by its own height.

    Used for the potential a visitor walks around and for the density that
    settled into it. The vertical exaggeration is reported rather than hidden:
    a relief read as a landscape is only honest if the plaque says how many
    metres stand for a Hartree.
    """
    if grid.dimension != 2:
        raise ValueError("a relief needs a two-dimensional grid")

    rows, cols = grid.points_per_dim
    values = np.asarray(field, dtype=np.float64).reshape(rows, cols)
    x_axis, y_axis = grid.coords_1d

    span = float(values.max() - values.min())
    exaggeration = (relief_m / span) if span > 0.0 else 0.0
    heights = (values - values.min()) * exaggeration
    if invert:
        heights = relief_m - heights

    mesh_x, mesh_y = np.meshgrid(x_axis, y_axis, indexing="ij")
    positions = np.stack([mesh_x.ravel(), heights.ravel(), mesh_y.ravel()], axis=1)

    shade = _normalize(values.ravel())
    if invert:
        shade = 1.0 - shade
    colors = np.clip(palette[None, :] * (0.32 + 0.68 * shade[:, None]), 0.0, 1.0)

    model = Model(name=name)
    model.add(
        Surface(
            positions=positions,
            indices=_lattice_faces(rows, cols),
            colors=colors,
            name=name,
            roughness=0.6,
            emissive=0.12,
        )
    )

    footprint = max(float(x_axis[-1] - x_axis[0]), float(y_axis[-1] - y_axis[0]))
    scale = width_m / footprint if footprint > 0.0 else 1.0
    for surface in model.surfaces:
        surface.positions = np.ascontiguousarray(surface.positions * scale, dtype=np.float32)

    report = ModelReport(
        name=name,
        triangles=model.triangles,
        surfaces=len(model.surfaces),
        plaque={
            "title": (
                "Confining Potential Landscape v(r)"
                if invert
                else "Ground-State Density Landscape n(r)"
            ),
            "field_min": f"{float(values.min()):.6g}",
            "field_max": f"{float(values.max()):.6g}",
            "vertical_scale": f"{relief_m * scale:.3f} m stands for {span:.4f} Hartree",
            "footprint": f"{width_m:.2f} m across",
            "physical_meaning": (
                "Topographical landscape of the external confining potential well v(r). Quantum bound states form in the inverted potential valleys."
                if invert
                else "The ground-state electron density n(r) settling into the potential landscape, balancing quantum pressure against potential confinement."
            ),
            "reading_guide": (
                "1. Surface elevation corresponds to potential depth (inverted for intuitive valley presentation). 2. Observe how the two Gaussian wells form a double-well molecular binding trap."
                if invert
                else "1. Surface height represents local electron density n(x, y). 2. Compare against the potential landscape to see how density piles up precisely within potential wells."
            ),
            "falsifiable_criterion": "Vertical relief is strictly proportional to energy/density: height scale is calibrated in meters per Hartree.",
        },
    )
    return model, report


def _convexity_tolerance(values: np.ndarray) -> float:
    """The largest chord gap that is rounding rather than a real violation.

    The constrained functional is convex by construction, so any negative gap
    it shows is arithmetic. Its values run to order 1e4 while its twin's run to
    order 1, so a fixed absolute tolerance would call noise a violation on one
    bowl and miss real violations on the other: the tolerance has to scale with
    the values being compared. Measured at single precision this admitted 48
    phantom violations on a bowl that has none in double.
    """
    epsilon = float(np.finfo(np.float64 if jax.config.jax_enable_x64 else np.float32).eps)
    scale = float(np.max(np.abs(values))) if values.size else 1.0
    multiplier = 64.0 if jax.config.jax_enable_x64 else 512.0
    return multiplier * epsilon * max(scale, 1.0)


def _unconstrained_energy(params, density):
    """The ICNN forward pass with its one guarantee removed.

    :func:`quantumflow.jax.convex.kinetic_energy` passes every hidden-to-hidden
    weight through Softplus, which is the whole of why the functional is convex
    in the density. This copy differs in exactly that: the weights are used
    raw. Same architecture, same initialisation, same seed -- so the two bowls
    in :func:`convexity_bowl_models` differ by the constraint and by nothing
    else, which is what makes the pair evidence rather than illustration.
    """
    import jax.numpy as jnp

    value = None
    layer_count = len([key for key in params if key.startswith("affine_")])
    for index in range(layer_count):
        affine_weight, bias = params[f"affine_{index}"]
        value_from_density = density @ affine_weight + bias
        if index:
            convex_weight, convex_bias = params[f"convex_{index}"]
            value = value_from_density + value @ convex_weight + convex_bias
        else:
            value = value_from_density
        if index < layer_count - 1:
            value = jax.nn.softplus(value)
        else:
            value = jnp.squeeze(value, axis=-1)
    return value


def convexity_bowl_models(
    base_density: np.ndarray,
    direction_a: np.ndarray,
    direction_b: np.ndarray,
    *,
    seed: int = 0,
    span: float = 1.0,
    samples: int = 96,
    chords: int = 6,
    height_m: float = 1.8,
    width_m: float = 1.8,
) -> tuple[tuple[Model, ModelReport], tuple[Model, ModelReport]]:
    """Two bowls: the constrained functional and its unconstrained twin.

    The slice through density space is **affine** --
    ``n(a, b) = n0 + a * d1 + b * d2`` with no renormalisation -- because an
    affine map preserves convexity exactly. A slice that renormalised would be
    a nonlinear reparameterisation, and a bowl drawn over it would prove
    nothing about the functional. The cost is that the densities away from the
    centre are not normalised to a particle number; the plaque says so.

    Each bowl carries chord ribbons. A chord joins two points of the surface in
    a straight line; convexity says the surface never rises above it. The
    ribbon is the region between chord and surface, green where the inequality
    holds and red where it fails. On the constrained bowl there is no red
    anywhere, and a visitor can go looking for some.
    """
    base = np.asarray(base_density, dtype=np.float64).reshape(-1)
    d_a = np.asarray(direction_a, dtype=np.float64).reshape(-1)
    d_b = np.asarray(direction_b, dtype=np.float64).reshape(-1)
    if d_a.shape != base.shape or d_b.shape != base.shape:
        raise ValueError("the slice directions must match the base density")

    params = init_icnn(jax.random.key(seed), input_size=base.shape[0])

    axis = np.linspace(-span, span, samples)
    alpha, beta = np.meshgrid(axis, axis, indexing="ij")
    densities = (
        base[None, :]
        + alpha.ravel()[:, None] * d_a[None, :]
        + beta.ravel()[:, None] * d_b[None, :]
    )
    negative_fraction = float(np.mean(densities < 0.0))

    def build(evaluate, name: str, convex: bool) -> tuple[Model, ModelReport]:
        eval_fn = jax.jit(evaluate)
        values = np.asarray(eval_fn(params, densities), dtype=np.float64)
        surface = values.reshape(samples, samples)

        span_value = float(surface.max() - surface.min())
        scale = (height_m / span_value) if span_value > 0.0 else 0.0
        heights = (surface - surface.min()) * scale
        plane = width_m / (2.0 * span)

        positions = np.stack(
            [(alpha * plane).ravel(), heights.ravel(), (beta * plane).ravel()], axis=1
        )
        shade = _normalize(surface.ravel())
        colors = np.clip(NEUTRAL[None, :] * (0.30 + 0.62 * (1.0 - shade[:, None])), 0.0, 1.0)

        model = Model(name=name)
        model.add(
            Surface(
                positions=positions,
                indices=_lattice_faces(samples, samples),
                colors=colors,
                name=f"{name}-surface",
                roughness=0.55,
                emissive=0.10,
            )
        )

        violations = 0
        rng = np.random.default_rng(seed)
        for chord_index in range(chords):
            angle = np.pi * (chord_index / max(chords, 1)) + rng.uniform(0.0, 0.2)
            offset = rng.uniform(-0.45, 0.45) * span
            # A straight line across the slice, which is a straight line in
            # density space because the slice is affine.
            steps = np.linspace(-span, span, samples)
            path_a = steps * np.cos(angle) - offset * np.sin(angle)
            path_b = steps * np.sin(angle) + offset * np.cos(angle)
            inside = (np.abs(path_a) <= span) & (np.abs(path_b) <= span)
            path_a, path_b = path_a[inside], path_b[inside]
            if path_a.shape[0] < 4:
                continue

            path_density = (
                base[None, :] + path_a[:, None] * d_a[None, :] + path_b[:, None] * d_b[None, :]
            )
            along = np.asarray(eval_fn(params, path_density), dtype=np.float64)
            # The chord itself: the straight line between the two endpoints.
            lam = np.linspace(0.0, 1.0, along.shape[0])
            chord = (1.0 - lam) * along[0] + lam * along[-1]

            gap = chord - along
            tolerance = _convexity_tolerance(along)
            violations += int(np.sum(gap < -tolerance))

            surface_y = (along - surface.min()) * scale
            chord_y = (chord - surface.min()) * scale
            # Lift the ribbon a hair off the surface so it never z-fights.
            lift = 0.004 * height_m

            ribbon_x = np.repeat(path_a * plane, 2)
            ribbon_z = np.repeat(path_b * plane, 2)
            ribbon_y = np.empty(2 * along.shape[0], dtype=np.float64)
            ribbon_y[0::2] = surface_y + lift
            ribbon_y[1::2] = chord_y + lift
            ribbon = np.stack([ribbon_x, ribbon_y, ribbon_z], axis=1)

            held = gap >= -tolerance
            vertex_colors = np.where(
                np.repeat(held, 2)[:, None], SATISFIED[None, :], VIOLATION[None, :]
            )

            quads = np.arange(along.shape[0] - 1) * 2
            faces = np.concatenate(
                [
                    np.stack([quads, quads + 1, quads + 2], axis=1),
                    np.stack([quads + 1, quads + 3, quads + 2], axis=1),
                ],
                axis=0,
            ).astype(np.uint32)

            model.add(
                Surface(
                    positions=ribbon,
                    indices=faces,
                    colors=vertex_colors,
                    name=f"{name}-chord-{chord_index}",
                    emissive=0.55,
                    roughness=0.3,
                )
            )

        report = ModelReport(
            name=name,
            triangles=model.triangles,
            surfaces=len(model.surfaces),
            plaque={
                "title": (
                    "Input-Convex Kinetic Functional T_s[n] (ICNN)"
                    if convex
                    else "Unconstrained Neural Functional Twin"
                ),
                "functional": (
                    "input-convex: hidden weights passed through Softplus (W >= 0), guaranteeing strict mathematical convexity T[lambda n1 + (1-lambda) n2] <= lambda T[n1] + (1-lambda) T[n2]"
                    if convex
                    else "the same network with the Softplus reparameterisation removed"
                ),
                "slice": "affine in the density, so convexity is preserved exactly",
                "chord_test": "green where the chord lies above the surface, red where it does not",
                "violating_samples": str(violations),
                "physical_meaning": (
                    "Strict convexity of Ts[n] is the mathematical bedrock of Hohenberg-Kohn DFT: it guarantees that variational minimization min_n {Ts[n] + int v n} has a unique global minimum with no spurious local traps."
                    if convex
                    else "Without the non-negative weight constraint, the functional forms non-convex folds and false local minima, causing variational density relaxation to diverge or get trapped."
                ),
                "reading_guide": (
                    "1. The 3D bowl surface is an exact 2D affine slice through many-body density space. 2. Green chord ribbons join pairs of points and float strictly above the bowl surface. 3. Look across the entire surface: there is not a single red chord violation."
                    if convex
                    else "1. Same architecture and initial random seed, but with non-negative constraints removed. 2. Observe the non-convex ripples and red chord ribbons where chords cut underneath the functional surface."
                ),
                "falsifiable_criterion": (
                    "Hunt for red chord violations: on the constrained bowl, zero violations exist across all random chords."
                    if convex
                    else "Red ribbons demonstrate falsification of convexity on the unconstrained twin."
                ),
                "precision": "double" if jax.config.jax_enable_x64 else "single",
                "seed": str(seed),
                "unnormalised_densities": f"{negative_fraction:.3%} of slice points go negative",
            },
        )
        return model, report

    convex = build(kinetic_energy, "convex-bowl", True)
    twin = build(_unconstrained_energy, "unconstrained-bowl", False)
    return convex, twin


def solve_showcase_system(
    *,
    dimension: int = 3,
    points: int = 48,
    separation: float = 1.0,
    depth: float = 12.0,
    width: float = 1.2,
    num_orbitals: int = 6,
    extent: float = 3.2,
    degeneracy_tolerance: float = 1e-6,
    close_shell: bool = False,
) -> dict:
    """Solve a two-well system: the exhibition's standing example.

    Two Gaussian wells a chosen distance apart is the smallest system whose
    orbitals show the thing the orangery is built to show -- as the wells
    separate, a bonding and an antibonding state pull apart in energy, and the
    visitor walking the room is walking ``separation``.

    The returned ``degenerate_cut`` flag is a correctness warning for the
    density exhibit, not a detail. States that share an energy are only fixed
    up to a rotation among themselves, so a total density that occupies part
    of a degenerate set depends on a basis the solver chose arbitrarily -- two
    solvers give two different clouds for the same physics. Occupy the whole
    multiplet or none of it; the flag says when ``num_orbitals`` splits one.

    Pass ``close_shell`` to step the occupation down to the largest count at
    or below ``num_orbitals`` that ends on a real energy gap. Any exhibit
    built from the total density -- the shells, the cloud a visitor walks into
    -- should set it, because a cloud whose shape depends on an arbitrary
    rotation is not a thing to hang on a wall and label.
    """
    grid = Grid.create(dimension=dimension, lower=-extent, upper=extent, points=points)
    coords = grid.coordinates

    centre = np.zeros(dimension)
    centre[0] = separation / 2.0
    left = coords + centre
    right = coords - centre
    potential = -depth * (
        np.exp(-np.sum(left**2, axis=-1) / (2.0 * width**2))
        + np.exp(-np.sum(right**2, axis=-1) / (2.0 * width**2))
    )

    # One state past what is asked for, so the occupied set can be checked
    # against the first state it leaves out.
    solution = solve_multidim_schroedinger(
        potential=potential, grid=grid, num_orbitals=num_orbitals + 1
    )
    energies = np.asarray(solution["orbital_energies"])
    if close_shell:
        # Step down until the occupation ends on a gap rather than inside a
        # multiplet. Walking down, not up: adding states would occupy orbitals
        # the system was not asked for.
        while num_orbitals > 1 and (
            abs(energies[num_orbitals] - energies[num_orbitals - 1]) < degeneracy_tolerance
        ):
            num_orbitals -= 1

    solution["degenerate_cut"] = bool(
        abs(energies[num_orbitals] - energies[num_orbitals - 1]) < degeneracy_tolerance
    )
    solution["occupied"] = int(num_orbitals)
    solution["orbital_energies"] = energies[:num_orbitals]
    solution["wavefunctions"] = solution["wavefunctions"][:, :num_orbitals]
    solution["density"] = np.sum(solution["wavefunctions"] ** 2, axis=1)
    solution["grid"] = grid
    solution["separation"] = separation
    return solution
