"""Tests for the exhibition geometry, the GLB writer and the shooting solver.

The exhibition makes claims a visitor is invited to check by eye -- that the
constrained functional has no convexity violation, that the shooting method
finds the published eigenvalues. These tests are what make those claims
something other than a caption.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import h5py
import jax
import numpy as np
import pytest

from quantumflow.exhibition import (
    _unconstrained_energy,
    convexity_bowl_models,
    density_shell_model,
    orbital_model,
    relief_model,
    solve_showcase_system,
)
from quantumflow.glb import Model, Surface, vertex_normals, write_glb
from quantumflow.jax.convex import init_icnn, kinetic_energy
from quantumflow.multidim import (
    Grid,
    kinetic_operator,
    kinetic_operator_sparse,
    solve_multidim_schroedinger,
)
from quantumflow.shooting import find_eigenvalues, integrate_numerov, sweep

BENCHMARK = Path("datasets/snyder_2012/validate/dataset.hdf5")

#: The grove refuses a model over any of these.
MAX_BYTES = 8_000_000
MAX_TRIANGLES = 150_000
MAX_DRAWS = 256


def _tetrahedron() -> Surface:
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    indices = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.uint32)
    return Surface(positions=positions, indices=indices, name="tetrahedron")


class TestGlbWriter:
    def test_writes_a_parseable_glb(self, tmp_path: Path) -> None:
        path = write_glb(tmp_path / "t.glb", Model(name="t", surfaces=[_tetrahedron()]))
        raw = path.read_bytes()

        assert raw[:4] == b"glTF"
        version, total = struct.unpack("<II", raw[4:12])
        assert version == 2
        assert total == len(raw)

        json_length, json_type = struct.unpack("<II", raw[12:20])
        assert json_type == 0x4E4F534A
        document = json.loads(raw[20 : 20 + json_length])
        assert document["asset"]["version"] == "2.0"
        assert document["scene"] == 0
        # Nothing the grove would need a decoder for.
        assert "extensionsRequired" not in document
        assert "extensionsUsed" not in document

    def test_every_chunk_is_four_byte_aligned(self, tmp_path: Path) -> None:
        path = write_glb(tmp_path / "t.glb", Model(name="t", surfaces=[_tetrahedron()]))
        raw = path.read_bytes()
        json_length, _ = struct.unpack("<II", raw[12:20])
        assert json_length % 4 == 0
        binary_length, _ = struct.unpack("<II", raw[20 + json_length : 28 + json_length])
        assert binary_length % 4 == 0
        assert len(raw) == 28 + json_length + binary_length

    def test_position_accessor_carries_bounds(self, tmp_path: Path) -> None:
        # The bundler places a model from these before the glb has downloaded,
        # so a missing bound is not cosmetic.
        path = write_glb(tmp_path / "t.glb", Model(name="t", surfaces=[_tetrahedron()]))
        raw = path.read_bytes()
        json_length, _ = struct.unpack("<II", raw[12:20])
        document = json.loads(raw[20 : 20 + json_length])
        position = document["accessors"][0]
        assert position["min"] == [0.0, 0.0, 0.0]
        assert position["max"] == [1.0, 1.0, 1.0]

    def test_vertex_colours_are_written_as_normalised_bytes(self, tmp_path: Path) -> None:
        surface = _tetrahedron()
        surface.colors = np.ones((4, 4), dtype=np.float32)
        path = write_glb(tmp_path / "t.glb", Model(name="t", surfaces=[surface]))
        raw = path.read_bytes()
        json_length, _ = struct.unpack("<II", raw[12:20])
        document = json.loads(raw[20 : 20 + json_length])
        attributes = document["meshes"][0]["primitives"][0]["attributes"]
        colour = document["accessors"][attributes["COLOR_0"]]
        assert colour["componentType"] == 5121
        assert colour["normalized"] is True
        assert colour["type"] == "VEC4"

    def test_refuses_an_index_past_the_end(self) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            Surface(
                positions=np.zeros((3, 3), dtype=np.float32),
                indices=np.array([[0, 1, 7]], dtype=np.uint32),
            )

    def test_refuses_a_non_finite_position(self) -> None:
        positions = np.zeros((3, 3), dtype=np.float32)
        positions[1, 1] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            Surface(positions=positions, indices=np.array([[0, 1, 2]], dtype=np.uint32))

    def test_normals_are_unit_length(self) -> None:
        surface = _tetrahedron()
        lengths = np.linalg.norm(surface.normals, axis=1)
        np.testing.assert_allclose(lengths, 1.0, atol=1e-6)

    def test_degenerate_normal_does_not_divide_by_zero(self) -> None:
        # Two faces wound against each other cancel exactly.
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        indices = np.array([[0, 1, 2], [0, 2, 1]], dtype=np.uint32)
        normals = vertex_normals(positions, indices)
        assert np.isfinite(normals).all()
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-6)


class TestSparseSolver:
    def test_sparse_laplacian_matches_the_dense_one(self) -> None:
        grid = Grid.create(dimension=3, lower=-2.0, upper=2.0, points=7)
        np.testing.assert_allclose(
            kinetic_operator_sparse(grid).toarray(), kinetic_operator(grid), atol=1e-12
        )

    def test_sparse_and_dense_paths_agree_on_the_spectrum(self) -> None:
        grid = Grid.create(dimension=3, lower=-3.0, upper=3.0, points=8)
        coords = grid.coordinates
        potential = -10.0 * np.exp(-np.sum(coords**2, axis=-1) / 2.0)

        dense = solve_multidim_schroedinger(
            potential, grid, num_orbitals=3, kinetic_matrix=kinetic_operator(grid)
        )
        sparse = solve_multidim_schroedinger(
            potential, grid, num_orbitals=3, kinetic_matrix=kinetic_operator_sparse(grid)
        )
        np.testing.assert_allclose(dense["orbital_energies"], sparse["orbital_energies"], atol=1e-9)

    def test_the_non_degenerate_ground_state_density_agrees(self) -> None:
        # Only the ground state: states 1 and 2 of this well are exactly
        # degenerate, and a density built from part of a degenerate set is
        # fixed only up to a rotation the solver picked, so comparing it
        # across two solvers compares arbitrary choices, not physics.
        grid = Grid.create(dimension=3, lower=-3.0, upper=3.0, points=8)
        coords = grid.coordinates
        potential = -10.0 * np.exp(-np.sum(coords**2, axis=-1) / 2.0)

        dense = solve_multidim_schroedinger(
            potential, grid, num_orbitals=1, kinetic_matrix=kinetic_operator(grid)
        )
        sparse = solve_multidim_schroedinger(
            potential, grid, num_orbitals=1, kinetic_matrix=kinetic_operator_sparse(grid)
        )
        np.testing.assert_allclose(dense["density"], sparse["density"], atol=1e-9)

    def test_closing_the_shell_steps_the_occupation_down(self) -> None:
        # Six requested, but the sixth is degenerate with a seventh that is
        # left out; the closed occupation stops at the last real gap.
        opened = solve_showcase_system(points=20, separation=1.6, num_orbitals=6)
        closed = solve_showcase_system(points=20, separation=1.6, num_orbitals=6, close_shell=True)
        assert opened["degenerate_cut"] is True
        assert closed["degenerate_cut"] is False
        assert closed["occupied"] < opened["occupied"]
        # The density must be the sum over exactly the states it reports.
        np.testing.assert_allclose(
            closed["density"],
            np.sum(closed["wavefunctions"] ** 2, axis=1),
            atol=1e-12,
        )

    def test_a_split_degenerate_shell_is_flagged(self) -> None:
        # The p-like triplet of a symmetric well: occupying two of three is
        # exactly the mistake the flag exists to catch.
        split = solve_showcase_system(dimension=3, points=10, separation=0.0, num_orbitals=2)
        assert split["degenerate_cut"] is True

        whole = solve_showcase_system(dimension=3, points=10, separation=0.0, num_orbitals=1)
        assert whole["degenerate_cut"] is False


@pytest.mark.skipif(not BENCHMARK.exists(), reason="the Snyder benchmark is not on this machine")
class TestShooting:
    """The room's claim is that these energies were not put in by hand."""

    def test_reproduces_the_benchmark_eigenvalues(self) -> None:
        with h5py.File(BENCHMARK, "r") as handle:
            potentials = handle["potential"][:3]
            reference = handle["energies"][:3]
            spacing = float(handle.attrs["h"])

        for index in range(3):
            found = find_eigenvalues(
                potentials[index],
                spacing,
                energy_min=float(potentials[index].min()) - 1.0,
                energy_max=float(reference[index][3]) + 20.0,
            )
            assert found.shape[0] >= 4
            np.testing.assert_allclose(found[:4], reference[index][:4], atol=1e-8)

    def test_a_trial_that_is_not_an_eigenvalue_misses_the_wall(self) -> None:
        with h5py.File(BENCHMARK, "r") as handle:
            potential = np.asarray(handle["potential"][0], dtype=np.float64)
            reference = np.asarray(handle["energies"][0], dtype=np.float64)
            spacing = float(handle.attrs["h"])

        landed = integrate_numerov(potential, float(reference[0]), spacing)
        missed = integrate_numerov(potential, float(reference[0]) + 2.0, spacing)
        # Normalised, because the two rays differ by orders of magnitude.
        assert abs(landed[-1]) / np.max(np.abs(landed)) < 1e-6
        assert abs(missed[-1]) / np.max(np.abs(missed)) > 0.5

    def test_the_sweep_uses_the_spacing_it_is_given(self) -> None:
        # The benchmark's stored h differs from x[1] - x[0] in the eighth
        # figure, and only the stored one reproduces its published energies.
        # Deriving the spacing from the coordinates costs 5e-07 Hartree, which
        # is the difference between a plaque that is true and one that is not.
        with h5py.File(BENCHMARK, "r") as handle:
            potential = np.asarray(handle["potential"][0], dtype=np.float64)
            reference = np.asarray(handle["energies"][0], dtype=np.float64)
            x = np.asarray(handle.attrs["x"], dtype=np.float64)
            spacing = float(handle.attrs["h"])

        assert spacing != float(x[1] - x[0])

        stated = sweep(potential, x, energy_min=-17.3, energy_max=25.1, trials=8, h=spacing)
        np.testing.assert_allclose(stated.eigen_energies[:2], reference[:2], atol=1e-10)

        derived = sweep(potential, x, energy_min=-17.3, energy_max=25.1, trials=8)
        assert np.abs(derived.eigen_energies[:2] - reference[:2]).max() > 1e-8

    def test_the_sweep_normalises_every_frame(self) -> None:
        with h5py.File(BENCHMARK, "r") as handle:
            potential = np.asarray(handle["potential"][0], dtype=np.float64)
            x = np.asarray(handle.attrs["x"], dtype=np.float64)

        result = sweep(potential, x, energy_min=-17.0, energy_max=25.0, trials=40)
        assert result.waves.shape == (40, x.shape[0])
        np.testing.assert_allclose(np.max(np.abs(result.waves), axis=1), 1.0, atol=1e-9)


class TestExhibitionModels:
    @pytest.fixture(scope="class")
    def solved(self) -> dict:
        grid = Grid.create(dimension=3, lower=-3.0, upper=3.0, points=12)
        coords = grid.coordinates
        offset = np.array([0.8, 0.0, 0.0])
        potential = -12.0 * (
            np.exp(-np.sum((coords + offset) ** 2, axis=-1) / 2.0)
            + np.exp(-np.sum((coords - offset) ** 2, axis=-1) / 2.0)
        )
        solution = solve_multidim_schroedinger(potential, grid, num_orbitals=3)
        solution["grid"] = grid
        return solution

    def test_an_orbital_stands_at_the_requested_height(self, solved: dict) -> None:
        model, _ = orbital_model(
            solved["grid"],
            solved["wavefunctions"][:, 0],
            energy_hartree=float(solved["orbital_energies"][0]),
            index=0,
            height_m=1.6,
            refine=1,
        )
        stacked = np.concatenate([s.positions for s in model.surfaces], axis=0)
        extent = stacked.max(axis=0) - stacked.min(axis=0)
        assert float(np.max(extent)) == pytest.approx(1.6, abs=1e-4)
        # Standing on a plinth, not sunk into it.
        assert float(stacked[:, 1].min()) >= -1e-4

    def test_an_orbital_stays_inside_the_grove_budget(self, solved: dict) -> None:
        model, report = orbital_model(
            solved["grid"],
            solved["wavefunctions"][:, 1],
            energy_hartree=float(solved["orbital_energies"][1]),
            index=1,
        )
        assert report.triangles < MAX_TRIANGLES
        assert len(model.surfaces) < MAX_DRAWS

    def test_the_plaque_states_what_the_colour_means(self, solved: dict) -> None:
        _, report = orbital_model(
            solved["grid"],
            solved["wavefunctions"][:, 1],
            energy_hartree=float(solved["orbital_energies"][1]),
            index=1,
        )
        # An unexplained colour is decoration; the caption is part of the exhibit.
        assert "kinetic energy" in report.plaque["colour"]
        assert "energy_hartree" in report.plaque

    def test_density_shells_enclose_the_share_they_claim(self, solved: dict) -> None:
        grid = solved["grid"]
        field = np.asarray(solved["density"], dtype=np.float64)
        _, report = density_shell_model(grid, field, fractions=(0.5, 0.9), refine=1)

        total = float(field.sum() * grid.volume_element)
        assert float(report.plaque["electrons"]) == pytest.approx(total, rel=1e-9)

        for key in ("shell_50pct", "shell_90pct"):
            level = float(report.plaque[key].split("=")[1])
            enclosed = float(field[field >= level].sum() * grid.volume_element)
            # The plaque states the containment it achieved, not the one that
            # was asked for; on a coarse grid those differ, and the visitor is
            # owed the true one.
            # The plaque rounds to four decimals; it must still describe the
            # level printed beside it.
            assert float(report.plaque[f"{key}_encloses"]) == pytest.approx(
                enclosed / total, abs=1e-4
            )

    def test_a_relief_reports_its_vertical_exaggeration(self) -> None:
        grid = Grid.create(dimension=2, lower=-2.0, upper=2.0, points=24)
        coords = grid.coordinates
        field = -np.exp(-np.sum(coords**2, axis=-1))
        _, report = relief_model(grid, field, name="r", relief_m=0.4, width_m=3.0)
        # A landscape read by eye is only honest if the scale is stated.
        assert "stands for" in report.plaque["vertical_scale"]


class TestConvexityExhibit:
    """The bowls are an invitation to falsify a claim; the claim must hold."""

    @pytest.fixture(scope="class")
    def slice_directions(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(7)
        base = rng.uniform(0.2, 1.0, 64)
        return base, rng.normal(0, 0.05, 64), rng.normal(0, 0.05, 64)

    def test_the_constrained_bowl_has_no_violation(self, slice_directions) -> None:
        base, d_a, d_b = slice_directions
        (_, convex_report), _ = convexity_bowl_models(base, d_a, d_b, samples=32, chords=4)
        assert convex_report.plaque["violating_samples"] == "0"

    def test_the_twin_differs_only_in_the_constraint(self, slice_directions) -> None:
        base, _, _ = slice_directions
        params = init_icnn(jax.random.key(0), input_size=base.shape[0])
        batch = np.stack([base, base * 1.05])
        # Same parameters into both; the values must differ, or the pair of
        # bowls would be showing the same network twice.
        constrained = np.asarray(kinetic_energy(params, batch))
        unconstrained = np.asarray(_unconstrained_energy(params, batch))
        assert not np.allclose(constrained, unconstrained)

    def test_convexity_holds_on_random_chords(self, slice_directions) -> None:
        base, d_a, d_b = slice_directions
        params = init_icnn(jax.random.key(0), input_size=base.shape[0])
        rng = np.random.default_rng(3)

        first = rng.uniform(-1, 1, (256, 2))
        second = rng.uniform(-1, 1, (256, 2))
        lam = rng.uniform(0, 1, (256, 1))

        def densities(points: np.ndarray) -> np.ndarray:
            return base[None, :] + points[:, :1] * d_a[None, :] + points[:, 1:2] * d_b[None, :]

        value_first = np.asarray(kinetic_energy(params, densities(first)))
        value_second = np.asarray(kinetic_energy(params, densities(second)))
        midpoint = np.asarray(kinetic_energy(params, densities(lam * first + (1 - lam) * second)))

        chord = lam[:, 0] * value_first + (1 - lam[:, 0]) * value_second
        scale = float(np.max(np.abs(chord)))
        # Convexity in the density, and the slice is affine, so it survives.
        assert np.all(chord - midpoint >= -1e-4 * scale)
