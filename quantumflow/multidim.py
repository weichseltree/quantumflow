"""Multi-dimensional quantum mechanics grid and Schrödinger solver in JAX and NumPy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg


@dataclass(frozen=True)
class Grid:
    """Discretized spatial grid in 1D, 2D, or 3D."""

    dimension: int
    bounds: tuple[tuple[float, float], ...]
    points_per_dim: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.dimension not in (1, 2, 3):
            raise ValueError(f"dimension must be 1, 2, or 3, got {self.dimension}")
        if len(self.bounds) != self.dimension:
            raise ValueError(
                f"bounds length {len(self.bounds)} does not match dimension {self.dimension}"
            )
        if len(self.points_per_dim) != self.dimension:
            raise ValueError(
                f"points_per_dim length {len(self.points_per_dim)} "
                f"does not match dimension {self.dimension}"
            )
        for pts in self.points_per_dim:
            if pts < 2:
                raise ValueError(f"points_per_dim elements must be >= 2, got {pts}")

    @classmethod
    def create(
        cls,
        dimension: int = 2,
        lower: float = -4.0,
        upper: float = 4.0,
        points: int = 32,
    ) -> Grid:
        """Create a symmetric cubic/square grid with uniform resolution."""
        bounds = tuple((lower, upper) for _ in range(dimension))
        points_per_dim = tuple(points for _ in range(dimension))
        return cls(dimension=dimension, bounds=bounds, points_per_dim=points_per_dim)

    @property
    def total_points(self) -> int:
        return int(np.prod(self.points_per_dim))

    @property
    def spacings(self) -> tuple[float, ...]:
        return tuple((b[1] - b[0]) / (n - 1) for b, n in zip(self.bounds, self.points_per_dim))

    @property
    def volume_element(self) -> float:
        return float(np.prod(self.spacings))

    @property
    def coords_1d(self) -> list[np.ndarray]:
        return [
            np.linspace(b[0], b[1], n, dtype=np.float64)
            for b, n in zip(self.bounds, self.points_per_dim)
        ]

    @property
    def meshgrid(self) -> list[np.ndarray]:
        coords_1d = self.coords_1d
        return np.meshgrid(*coords_1d, indexing="ij")

    @property
    def coordinates(self) -> np.ndarray:
        """Return (total_points, dimension) coordinates array."""
        mesh = self.meshgrid
        stacked = np.stack([m.ravel() for m in mesh], axis=-1)
        return stacked


def laplacian_matrix_1d(n: int, h: float) -> np.ndarray:
    """Standard 1D central difference 3-point Laplacian matrix with Dirichlet boundary."""
    main_diag = -2.0 * np.ones(n, dtype=np.float64)
    off_diag = np.ones(n - 1, dtype=np.float64)
    return (np.diag(main_diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)) / (h**2)


def laplacian_matrix(grid: Grid) -> np.ndarray:
    """Build the d-dimensional grid Laplacian operator using Kronecker products."""
    l_1ds = [laplacian_matrix_1d(n, h) for n, h in zip(grid.points_per_dim, grid.spacings)]
    if grid.dimension == 1:
        return l_1ds[0]
    if grid.dimension == 2:
        eye_0 = np.eye(grid.points_per_dim[0], dtype=np.float64)
        eye_1 = np.eye(grid.points_per_dim[1], dtype=np.float64)
        return np.kron(l_1ds[0], eye_1) + np.kron(eye_0, l_1ds[1])
    if grid.dimension == 3:
        eye_0 = np.eye(grid.points_per_dim[0], dtype=np.float64)
        eye_1 = np.eye(grid.points_per_dim[1], dtype=np.float64)
        eye_2 = np.eye(grid.points_per_dim[2], dtype=np.float64)
        return (
            np.kron(np.kron(l_1ds[0], eye_1), eye_2)
            + np.kron(np.kron(eye_0, l_1ds[1]), eye_2)
            + np.kron(np.kron(eye_0, eye_1), l_1ds[2])
        )
    raise ValueError(f"Unsupported dimension {grid.dimension}")


def laplacian_matrix_sparse(grid: Grid) -> sp.csr_matrix:
    """Build the d-dimensional grid Laplacian as a sparse operator.

    Identical to :func:`laplacian_matrix` in value, but the Kronecker products
    are taken over sparse factors. The dense builder allocates
    ``total_points**2`` doubles, which is 1.5 GB at a 24-point cubic grid and
    8.6 GB at 32; the sparse operator holds ``2 * dimension + 1`` diagonals and
    makes the resolutions an isosurface needs reachable.
    """
    l_1ds = [
        sp.csr_matrix(laplacian_matrix_1d(n, h)) for n, h in zip(grid.points_per_dim, grid.spacings)
    ]
    if grid.dimension == 1:
        return l_1ds[0].tocsr()

    eyes = [sp.identity(n, dtype=np.float64, format="csr") for n in grid.points_per_dim]
    total = None
    for axis, l_1d in enumerate(l_1ds):
        # The term acting along `axis` is the identity on every other axis.
        factors = [l_1d if index == axis else eyes[index] for index in range(grid.dimension)]
        term = factors[0]
        for factor in factors[1:]:
            term = sp.kron(term, factor, format="csr")
        total = term if total is None else total + term
    return total.tocsr()


def kinetic_operator(grid: Grid) -> np.ndarray:
    """Return kinetic energy operator T = -1/2 nabla^2."""
    return -0.5 * laplacian_matrix(grid)


def kinetic_operator_sparse(grid: Grid) -> sp.csr_matrix:
    """Return the sparse kinetic energy operator T = -1/2 nabla^2."""
    return -0.5 * laplacian_matrix_sparse(grid)


def solve_multidim_schroedinger(
    potential: np.ndarray | Any,
    grid: Grid,
    num_orbitals: int = 4,
    kinetic_matrix: np.ndarray | None = None,
) -> dict[str, Any]:
    """Solve the multi-dimensional non-interacting Schrödinger equation on grid.

    Parameters
    ----------
    potential : array-like
        Flat potential array of shape ``(total_points,)`` or shaped as ``grid.points_per_dim``.
    grid : Grid
        Spatial grid specification.
    num_orbitals : int
        Number of lowest eigenstates (occupied orbitals) to compute.
    kinetic_matrix : np.ndarray, optional
        Precomputed kinetic operator matrix ``-0.5 * laplacian``.

    Returns
    -------
    dict with keys:
        - ``orbital_energies``: shape ``(num_orbitals,)``
        - ``chemical_potential``: HOMO eigenvalue ``epsilon_N``
        - ``density``: total density ``n(r) = sum_i |psi_i(r)|^2`` (normalized to ``num_orbitals``)
        - ``kinetic_energy``: non-interacting kinetic energy ``T_s``
        - ``potential_energy``: external potential energy ``int v(r) n(r) dr``
        - ``total_energy``: sum of orbital energies
        - ``wavefunctions``: shape ``(total_points, num_orbitals)``
        - ``potential``: flat potential array
    """
    v_flat = np.asarray(potential, dtype=np.float64).reshape(-1)
    if v_flat.shape[0] != grid.total_points:
        raise ValueError(
            f"Potential size {v_flat.shape[0]} does not match grid total points {grid.total_points}"
        )

    if kinetic_matrix is None:
        # Past a thousand points the dense Kronecker builder is the memory
        # ceiling, not the eigensolver: it wants total_points**2 doubles.
        kinetic_matrix = (
            kinetic_operator(grid) if grid.total_points <= 1000 else kinetic_operator_sparse(grid)
        )

    if sp.issparse(kinetic_matrix):
        h_mat = (kinetic_matrix + sp.diags(v_flat)).tocsr()
    else:
        h_mat = kinetic_matrix + np.diag(v_flat)

    if not sp.issparse(h_mat) and grid.total_points <= 1000:
        evals, evecs = scipy.linalg.eigh(h_mat, subset_by_index=[0, num_orbitals - 1])
        orbital_energies = evals
        wavefunctions = evecs
    else:
        h_sparse = h_mat if sp.issparse(h_mat) else sp.csr_matrix(h_mat)
        # ARPACK starts from a random vector unless given one. Inside a
        # degenerate subspace the basis it returns is then different every
        # call, so a density summed over part of a degenerate set -- and any
        # geometry exported from it -- would change run to run, which a bundle
        # recording a sha256 cannot live with. A fixed start makes the export
        # reproducible; it does not make a split multiplet physical, which is
        # what `degenerate_cut` is for.
        start = np.ones(grid.total_points, dtype=np.float64)
        # `sigma` shifts the spectrum so the solver converges on the lowest
        # states directly; 'SA' on an unshifted operator stalls on the fine
        # grids an isosurface needs.
        evals, evecs = scipy.sparse.linalg.eigsh(
            h_sparse,
            k=num_orbitals,
            sigma=float(v_flat.min()) - 1.0,
            which="LM",
            v0=start,
        )
        sort_idx = np.argsort(evals)
        orbital_energies = evals[sort_idx]
        wavefunctions = evecs[:, sort_idx]

    # Normalize wavefunctions such that sum |psi_i|^2 * dV = 1
    dv = grid.volume_element
    norms = np.sqrt(np.sum(wavefunctions**2, axis=0, keepdims=True) * dv)
    wavefunctions = wavefunctions / norms

    # Total density n(r) = sum_i |psi_i(r)|^2
    density = np.sum(wavefunctions**2, axis=1)

    # Potential energy = sum v(r) * n(r) * dV
    potential_energy = float(np.sum(v_flat * density) * dv)

    # Total energy = sum epsilon_i
    total_energy = float(np.sum(orbital_energies))

    # Kinetic energy = total_energy - potential_energy
    kinetic_energy = total_energy - potential_energy

    # Chemical potential (HOMO)
    chemical_potential = float(orbital_energies[num_orbitals - 1])

    return {
        "orbital_energies": orbital_energies,
        "chemical_potential": chemical_potential,
        "density": density,
        "kinetic_energy": kinetic_energy,
        "potential_energy": potential_energy,
        "total_energy": total_energy,
        "wavefunctions": wavefunctions,
        "potential": v_flat,
    }


def generate_multidim_potentials(
    grid: Grid,
    dataset_size: int = 100,
    num_wells: int = 3,
    depth_range: tuple[float, float] = (5.0, 20.0),
    width_range: tuple[float, float] = (0.6, 1.6),
    seed: int = 42,
) -> np.ndarray:
    """Generate smooth Gaussian-mixture well potentials in 1D, 2D, or 3D."""
    rng = np.random.default_rng(seed)
    coords = grid.coordinates  # shape: (total_points, d)

    potentials = np.zeros((dataset_size, grid.total_points), dtype=np.float64)

    for i in range(dataset_size):
        v = np.zeros(grid.total_points, dtype=np.float64)
        for _ in range(num_wells):
            depth = rng.uniform(depth_range[0], depth_range[1])
            width = rng.uniform(width_range[0], width_range[1])
            center = np.array(
                [rng.uniform(b[0] * 0.5, b[1] * 0.5) for b in grid.bounds],
                dtype=np.float64,
            )
            # Gaussian well: -depth * exp(-||r - center||^2 / (2 * width^2))
            diff = coords - center
            r_sq = np.sum(diff**2, axis=-1)
            v += -depth * np.exp(-r_sq / (2.0 * width**2))
        potentials[i] = v

    return potentials


def generate_piecewise_constant_potentials(
    grid: Grid,
    dataset_size: int = 100,
    num_wells: int = 3,
    depth_range: tuple[float, float] = (5.0, 20.0),
    width_range: tuple[float, float] = (0.6, 1.6),
    seed: int = 42,
    shape: str = "box",
) -> np.ndarray:
    """Generate piecewise-constant step/box well potentials in 1D, 2D, or 3D.

    Parameters
    ----------
    grid : Grid
        Spatial grid specification.
    dataset_size : int
        Number of potentials to generate.
    num_wells : int
        Number of step/box wells per potential.
    depth_range : tuple[float, float]
        Range of well depths.
    width_range : tuple[float, float]
        Range of half-widths along each axis.
    seed : int
        Random seed.
    shape : str
        'box' for axis-aligned rectangular boxes / cuboids (L_infinity norm),
        'sphere' for spherical / circular step wells (L_2 norm),
        'staircase' for concentric nested convex boxes with ascending step heights.
    """
    rng = np.random.default_rng(seed)
    coords = grid.coordinates  # shape: (total_points, d)
    dim = grid.dimension

    potentials = np.zeros((dataset_size, grid.total_points), dtype=np.float64)

    for i in range(dataset_size):
        v = np.zeros(grid.total_points, dtype=np.float64)
        if shape == "staircase":
            # Concentric nested convex boxes centered near origin
            center = np.array(
                [rng.uniform(b[0] * 0.25, b[1] * 0.25) for b in grid.bounds],
                dtype=np.float64,
            )
            base_depth = rng.uniform(depth_range[0], depth_range[1])
            widths = np.sort(rng.uniform(width_range[0], width_range[1] * 1.5, size=num_wells))
            steps = np.linspace(base_depth, 0.0, num_wells + 1)
            diff = np.abs(coords - center)
            max_dist = np.max(diff, axis=-1)
            v_val = np.zeros(grid.total_points, dtype=np.float64)
            for w, s in zip(reversed(widths), reversed(steps[:-1])):
                mask = max_dist <= w
                v_val[mask] = -s
            v = v_val
        elif shape == "sphere":
            for _ in range(num_wells):
                depth = rng.uniform(depth_range[0], depth_range[1])
                radius = rng.uniform(width_range[0], width_range[1])
                center = np.array(
                    [rng.uniform(b[0] * 0.4, b[1] * 0.4) for b in grid.bounds],
                    dtype=np.float64,
                )
                r = np.sqrt(np.sum((coords - center) ** 2, axis=-1))
                mask = r <= radius
                v[mask] -= depth
        else:  # "box" (default cuboid / hyper-rectangle)
            for _ in range(num_wells):
                depth = rng.uniform(depth_range[0], depth_range[1])
                half_widths = np.array(
                    [rng.uniform(width_range[0], width_range[1]) for _ in range(dim)],
                    dtype=np.float64,
                )
                center = np.array(
                    [rng.uniform(b[0] * 0.4, b[1] * 0.4) for b in grid.bounds],
                    dtype=np.float64,
                )
                diff = np.abs(coords - center)
                mask = np.all(diff <= half_widths, axis=-1)
                v[mask] -= depth
        potentials[i] = v

    return potentials


def generate_piecewise_linear_potentials(
    grid: Grid,
    dataset_size: int = 100,
    num_facets: int = 6,
    slope_range: tuple[float, float] = (1.5, 5.0),
    depth_range: tuple[float, float] = (5.0, 20.0),
    seed: int = 42,
    mode: str = "convex_polyhedral",
) -> np.ndarray:
    """Generate piecewise-linear (continuous, polyhedral or pyramid) potentials in 1D, 2D, or 3D.

    In 'convex_polyhedral' mode, potentials are formed as the pointwise maximum
    of affine functions: v(r) = max_{j=1..M} (a_j . (r - c) + b_j) - V0,
    which is mathematically guaranteed to be convex and piecewise linear with
    facet ridges and corners.

    In 'l1_pyramid' mode, v(r) = sum_d s_d * |x_d - c_d| - V0 (convex L1 tent/pyramid).

    Parameters
    ----------
    grid : Grid
        Spatial grid specification.
    dataset_size : int
        Number of potentials to generate.
    num_facets : int
        Number of affine planar facets (for 'convex_polyhedral').
    slope_range : tuple[float, float]
        Range of facet slopes.
    depth_range : tuple[float, float]
        Range of well depths (shifting the minimum).
    seed : int
        Random seed.
    mode : str
        'convex_polyhedral' (max of affine planes) or 'l1_pyramid' (L1 pyramid).
    """
    rng = np.random.default_rng(seed)
    coords = grid.coordinates  # shape: (total_points, d)
    dim = grid.dimension

    potentials = np.zeros((dataset_size, grid.total_points), dtype=np.float64)

    for i in range(dataset_size):
        center = np.array(
            [rng.uniform(b[0] * 0.25, b[1] * 0.25) for b in grid.bounds],
            dtype=np.float64,
        )
        depth = rng.uniform(depth_range[0], depth_range[1])
        diff = coords - center  # (total_points, dim)

        if mode == "l1_pyramid":
            slopes = rng.uniform(slope_range[0], slope_range[1], size=dim)
            v = np.sum(slopes * np.abs(diff), axis=-1) - depth
        else:  # convex_polyhedral: max_j (a_j . diff + b_j) - depth
            normals = []
            for d in range(dim):
                e_pos = np.zeros(dim)
                e_pos[d] = 1.0
                e_neg = np.zeros(dim)
                e_neg[d] = -1.0
                normals.extend([e_pos, e_neg])

            extra_facets = max(0, num_facets - len(normals))
            if extra_facets > 0:
                rand_vecs = rng.normal(size=(extra_facets, dim))
                rand_norms = np.linalg.norm(rand_vecs, axis=-1, keepdims=True)
                rand_vecs = rand_vecs / np.maximum(rand_norms, 1e-8)
                normals.extend(list(rand_vecs))

            facet_planes = []
            for n in normals:
                slope = rng.uniform(slope_range[0], slope_range[1])
                offset = rng.uniform(0.0, 1.0)
                plane_val = slope * np.dot(diff, n) + offset
                facet_planes.append(plane_val)

            # Convex upper envelope
            v = np.max(np.stack(facet_planes, axis=0), axis=0)
            # Normalize so minimum is at -depth
            v = (v - np.min(v)) - depth

        potentials[i] = v

    return potentials


def generate_multidim_dataset(
    grid: Grid,
    dataset_size: int = 100,
    num_orbitals: int = 3,
    num_wells: int = 3,
    seed: int = 42,
    depth_range: tuple[float, float] = (5.0, 20.0),
    width_range: tuple[float, float] = (0.6, 1.6),
    potential_type: str = "gaussian",
    **potential_kwargs: Any,
) -> dict[str, Any]:
    """Generate complete dataset with potentials, densities, kinetic and orbital energies.

    Parameters
    ----------
    grid : Grid
        Spatial grid specification.
    dataset_size : int
        Number of configurations.
    num_orbitals : int
        Number of occupied orbitals.
    num_wells : int
        Number of wells (for Gaussian or piecewise constant).
    seed : int
        Random seed.
    depth_range : tuple[float, float]
        Range of potential depths.
    width_range : tuple[float, float]
        Range of potential widths.
    potential_type : str
        'gaussian', 'piecewise_constant', or 'piecewise_linear'.
    **potential_kwargs : Any
        Additional keyword arguments forwarded to the specific potential generator.
    """
    if potential_type == "piecewise_constant":
        potentials = generate_piecewise_constant_potentials(
            grid=grid,
            dataset_size=dataset_size,
            num_wells=num_wells,
            depth_range=depth_range,
            width_range=width_range,
            seed=seed,
            **potential_kwargs,
        )
    elif potential_type == "piecewise_linear":
        potentials = generate_piecewise_linear_potentials(
            grid=grid,
            dataset_size=dataset_size,
            depth_range=depth_range,
            seed=seed,
            **potential_kwargs,
        )
    else:
        potentials = generate_multidim_potentials(
            grid=grid,
            dataset_size=dataset_size,
            num_wells=num_wells,
            depth_range=depth_range,
            width_range=width_range,
            seed=seed,
        )

    t_mat = kinetic_operator(grid)

    densities = np.zeros((dataset_size, grid.total_points), dtype=np.float64)
    kinetic_energies = np.zeros(dataset_size, dtype=np.float64)
    potential_energies = np.zeros(dataset_size, dtype=np.float64)
    total_energies = np.zeros(dataset_size, dtype=np.float64)
    chemical_potentials = np.zeros(dataset_size, dtype=np.float64)
    orbital_energies = np.zeros((dataset_size, num_orbitals), dtype=np.float64)
    derivatives = np.zeros((dataset_size, grid.total_points), dtype=np.float64)

    for i in range(dataset_size):
        sol = solve_multidim_schroedinger(
            potential=potentials[i],
            grid=grid,
            num_orbitals=num_orbitals,
            kinetic_matrix=t_mat,
        )
        densities[i] = sol["density"]
        kinetic_energies[i] = sol["kinetic_energy"]
        potential_energies[i] = sol["potential_energy"]
        total_energies[i] = sol["total_energy"]
        chemical_potentials[i] = sol["chemical_potential"]
        orbital_energies[i] = sol["orbital_energies"]
        # Euler equation derivative: delta T / delta n = mu - v(r)
        derivatives[i] = sol["chemical_potential"] - potentials[i]

    return {
        "grid": grid,
        "potentials": potentials,
        "densities": densities,
        "kinetic_energies": kinetic_energies,
        "potential_energies": potential_energies,
        "total_energies": total_energies,
        "chemical_potentials": chemical_potentials,
        "orbital_energies": orbital_energies,
        "derivatives": derivatives,
    }


def von_weizsaecker_multidim(
    density: np.ndarray,
    grid: Grid,
    kinetic_matrix: np.ndarray | sp.csr_matrix | None = None,
    eps: float = 1e-12,
) -> np.ndarray | float:
    """Evaluate multi-dimensional von Weizsaecker kinetic energy on grid:

    T_W[n] = 1/2 int ||grad sqrt(n)||^2 dV = <sqrt(n), -1/2 nabla^2 sqrt(n)> dV
    """
    d_arr = np.asarray(density, dtype=np.float64)
    if kinetic_matrix is None:
        kinetic_matrix = (
            kinetic_operator(grid) if grid.total_points <= 1000 else kinetic_operator_sparse(grid)
        )

    dv = grid.volume_element
    if d_arr.ndim == 1 or d_arr.shape == grid.points_per_dim:
        sqrt_n = np.sqrt(np.maximum(d_arr.reshape(-1), 0.0) + eps)
        t_sqrt_n = (
            kinetic_matrix.dot(sqrt_n) if sp.issparse(kinetic_matrix) else kinetic_matrix @ sqrt_n
        )
        return float(np.sum(sqrt_n * t_sqrt_n) * dv)
    else:
        # Batched (batch, total_points)
        batch_size = d_arr.shape[0]
        results = np.zeros(batch_size, dtype=np.float64)
        for i in range(batch_size):
            sqrt_n = np.sqrt(np.maximum(d_arr[i].reshape(-1), 0.0) + eps)
            t_sqrt_n = (
                kinetic_matrix.dot(sqrt_n)
                if sp.issparse(kinetic_matrix)
                else kinetic_matrix @ sqrt_n
            )
            results[i] = float(np.sum(sqrt_n * t_sqrt_n) * dv)
        return results


def quantum_potential_multidim(
    density: np.ndarray,
    grid: Grid,
    kinetic_matrix: np.ndarray | sp.csr_matrix | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """Evaluate the Bohmian quantum potential Q[n] on multi-dimensional grid:

    Q[n](r) = -1/2 (nabla^2 sqrt(n)) / (sqrt(n) + eps) = (T_kin sqrt(n)) / (sqrt(n) + eps)
    """
    d_arr = np.asarray(density, dtype=np.float64)
    if kinetic_matrix is None:
        kinetic_matrix = (
            kinetic_operator(grid) if grid.total_points <= 1000 else kinetic_operator_sparse(grid)
        )

    if d_arr.ndim == 1 or d_arr.shape == grid.points_per_dim:
        sqrt_n = np.sqrt(np.maximum(d_arr.reshape(-1), 0.0) + eps)
        t_sqrt_n = (
            kinetic_matrix.dot(sqrt_n) if sp.issparse(kinetic_matrix) else kinetic_matrix @ sqrt_n
        )
        return (t_sqrt_n / (sqrt_n + eps)).reshape(d_arr.shape)
    else:
        # Batched (batch, total_points)
        batch_size = d_arr.shape[0]
        q_pot = np.zeros_like(d_arr)
        for i in range(batch_size):
            sqrt_n = np.sqrt(np.maximum(d_arr[i].reshape(-1), 0.0) + eps)
            t_sqrt_n = (
                kinetic_matrix.dot(sqrt_n)
                if sp.issparse(kinetic_matrix)
                else kinetic_matrix @ sqrt_n
            )
            q_pot[i] = (t_sqrt_n / (sqrt_n + eps)).reshape(d_arr[i].shape)
        return q_pot


def generate_bosonic_dataset(
    grid: Grid,
    dataset_size: int = 100,
    num_particles: float = 10.0,
    seed: int = 42,
    potential_type: str = "gaussian",
    **potential_kwargs: Any,
) -> dict[str, Any]:
    """Generate dataset for non-interacting bosons condensed into lowest state psi_0(r).

    Density is n(r) = N * |psi_0(r)|^2, where T_s[n] = T_W[n] exactly.
    """
    if potential_type == "piecewise_constant":
        potentials = generate_piecewise_constant_potentials(
            grid=grid, dataset_size=dataset_size, seed=seed, **potential_kwargs
        )
    elif potential_type == "piecewise_linear":
        potentials = generate_piecewise_linear_potentials(
            grid=grid, dataset_size=dataset_size, seed=seed, **potential_kwargs
        )
    else:
        potentials = generate_multidim_potentials(
            grid=grid, dataset_size=dataset_size, seed=seed, **potential_kwargs
        )

    t_mat = kinetic_operator(grid)
    dv = grid.volume_element

    densities = np.zeros((dataset_size, grid.total_points), dtype=np.float64)
    kinetic_energies = np.zeros(dataset_size, dtype=np.float64)
    potential_energies = np.zeros(dataset_size, dtype=np.float64)
    total_energies = np.zeros(dataset_size, dtype=np.float64)
    chemical_potentials = np.zeros(dataset_size, dtype=np.float64)
    derivatives = np.zeros((dataset_size, grid.total_points), dtype=np.float64)

    for i in range(dataset_size):
        sol = solve_multidim_schroedinger(
            potential=potentials[i],
            grid=grid,
            num_orbitals=1,
            kinetic_matrix=t_mat,
        )
        psi_0 = sol["wavefunctions"][:, 0]
        # Normalized to num_particles
        n_boson = num_particles * (psi_0**2)
        eps_0 = float(sol["orbital_energies"][0])
        v_ext_val = float(np.sum(potentials[i] * n_boson) * dv)
        tot_e = float(num_particles * eps_0)
        kin_e = tot_e - v_ext_val

        densities[i] = n_boson
        kinetic_energies[i] = kin_e
        potential_energies[i] = v_ext_val
        total_energies[i] = tot_e
        chemical_potentials[i] = eps_0
        derivatives[i] = eps_0 - potentials[i]

    return {
        "grid": grid,
        "num_particles": num_particles,
        "potentials": potentials,
        "densities": densities,
        "kinetic_energies": kinetic_energies,
        "potential_energies": potential_energies,
        "total_energies": total_energies,
        "chemical_potentials": chemical_potentials,
        "derivatives": derivatives,
    }


def generate_spin_polarized_dataset(
    grid: Grid,
    dataset_size: int = 100,
    num_up: int = 2,
    num_down: int = 1,
    seed: int = 42,
    potential_type: str = "gaussian",
    **potential_kwargs: Any,
) -> dict[str, Any]:
    """Generate spin-polarized fermionic dataset with (num_up, num_down) particles."""
    max_orbitals = max(num_up, num_down)
    if potential_type == "piecewise_constant":
        potentials = generate_piecewise_constant_potentials(
            grid=grid, dataset_size=dataset_size, seed=seed, **potential_kwargs
        )
    elif potential_type == "piecewise_linear":
        potentials = generate_piecewise_linear_potentials(
            grid=grid, dataset_size=dataset_size, seed=seed, **potential_kwargs
        )
    else:
        potentials = generate_multidim_potentials(
            grid=grid, dataset_size=dataset_size, seed=seed, **potential_kwargs
        )

    t_mat = kinetic_operator(grid)
    dv = grid.volume_element

    densities_up = np.zeros((dataset_size, grid.total_points), dtype=np.float64)
    densities_down = np.zeros((dataset_size, grid.total_points), dtype=np.float64)
    densities_total = np.zeros((dataset_size, grid.total_points), dtype=np.float64)
    kinetic_energies = np.zeros(dataset_size, dtype=np.float64)
    total_energies = np.zeros(dataset_size, dtype=np.float64)

    for i in range(dataset_size):
        sol = solve_multidim_schroedinger(
            potential=potentials[i],
            grid=grid,
            num_orbitals=max_orbitals,
            kinetic_matrix=t_mat,
        )
        wfs = sol["wavefunctions"]
        evals = sol["orbital_energies"]

        n_up = np.sum(wfs[:, :num_up] ** 2, axis=1) if num_up > 0 else np.zeros(grid.total_points)
        n_down = (
            np.sum(wfs[:, :num_down] ** 2, axis=1) if num_down > 0 else np.zeros(grid.total_points)
        )
        n_tot = n_up + n_down

        tot_e = float(np.sum(evals[:num_up]) + np.sum(evals[:num_down]))
        pot_e = float(np.sum(potentials[i] * n_tot) * dv)
        kin_e = tot_e - pot_e

        densities_up[i] = n_up
        densities_down[i] = n_down
        densities_total[i] = n_tot
        kinetic_energies[i] = kin_e
        total_energies[i] = tot_e

    return {
        "grid": grid,
        "num_up": num_up,
        "num_down": num_down,
        "potentials": potentials,
        "densities_up": densities_up,
        "densities_down": densities_down,
        "densities_total": densities_total,
        "kinetic_energies": kinetic_energies,
        "total_energies": total_energies,
    }
