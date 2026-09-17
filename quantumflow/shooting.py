"""Numerov shooting in NumPy: quantization as something you can watch happen.

The Shooting Gallery's exhibit is this method run as a sweep. Integrate the
wave outward from the left wall at some trial energy and it will, in general,
miss the right wall -- diverging away from zero as it goes. Only at particular
energies does it land. Nobody puts those energies in; they are what survives
the demand that the wave fit inside its box.

The repository's existing Numerov solver lives in
``quantumflow.noninteracting_1d.numerov_solver`` and is written in TensorFlow,
which is the legacy stack here. Rendering an exhibition asset should not drag
that in, so the recurrence is reimplemented on NumPy. It is checked against the
Snyder benchmark's own stored eigenvalues in the tests, which is the only
reason to trust what the room shows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ShootingSweep:
    """One potential, swept over trial energies."""

    x: np.ndarray
    potential: np.ndarray
    energies: np.ndarray
    #: ``(trials, points)``; each row the wave shot at that trial energy,
    #: normalised to unit maximum so a diverging solution stays on the page.
    waves: np.ndarray
    #: ``(trials,)``; the wave's value at the far wall, the quantity whose
    #: zeros are the eigenvalues.
    endpoint: np.ndarray
    #: Trial energies whose endpoint changed sign since the previous trial.
    eigen_energies: np.ndarray


def integrate_numerov(potential: np.ndarray, energy: float, h: float) -> np.ndarray:
    """Shoot one wave from the left wall at a single trial energy.

    Solves ``psi'' = 2 (v - E) psi`` with ``psi(0) = 0`` by the Numerov
    recurrence, which is accurate to ``h**4`` because the ``psi'`` term is
    absent -- the reason this method, and not Euler, is what the room shows.
    """
    v = np.asarray(potential, dtype=np.float64)
    f = 2.0 * (v - energy)
    factor = 1.0 - (h * h / 12.0) * f

    psi = np.zeros_like(v)
    # psi(0) = 0 is the left boundary condition; the second point sets the
    # arbitrary overall scale, which no observable depends on.
    psi[1] = h

    for n in range(1, v.shape[0] - 1):
        psi[n + 1] = (
            2.0 * psi[n] * (1.0 + (5.0 * h * h / 12.0) * f[n]) - psi[n - 1] * factor[n - 1]
        ) / factor[n + 1]
        # A diverging trial runs away exponentially and will overflow long
        # before the far wall. Rescaling the whole ray preserves the shape and
        # the sign of the endpoint, which is all the sweep reads.
        if abs(psi[n + 1]) > 1e100:
            psi[: n + 2] /= 1e100

    return psi


def find_eigenvalues(
    potential: np.ndarray,
    h: float,
    *,
    energy_min: float,
    energy_max: float,
    coarse_steps: int = 2000,
    refine_steps: int = 60,
) -> np.ndarray:
    """Bracket every sign change of the endpoint, then bisect each one."""
    grid = np.linspace(energy_min, energy_max, coarse_steps)
    endpoints = np.array([integrate_numerov(potential, e, h)[-1] for e in grid])

    # A sign change of psi(L) between two trials brackets an eigenvalue.
    signs = np.sign(endpoints)
    crossings = np.flatnonzero(signs[:-1] * signs[1:] < 0)

    found = []
    for index in crossings:
        low, high = grid[index], grid[index + 1]
        low_sign = np.sign(endpoints[index])
        for _ in range(refine_steps):
            middle = 0.5 * (low + high)
            middle_sign = np.sign(integrate_numerov(potential, middle, h)[-1])
            # Comparing signs, not their product. A diverging ray's endpoint
            # runs to 1e300 or down to 1e-300, so the product can overflow to
            # inf or underflow to exactly 0.0 -- and `product <= 0` is then
            # true whatever the signs were, sending the bisection down the
            # wrong half.
            if middle_sign != low_sign:
                high = middle
            else:
                low, low_sign = middle, middle_sign
        found.append(0.5 * (low + high))
    return np.array(found, dtype=np.float64)


def sweep(
    potential: np.ndarray,
    x: np.ndarray,
    *,
    energy_min: float,
    energy_max: float,
    trials: int = 420,
    h: float | None = None,
) -> ShootingSweep:
    """Shoot the wave at every trial energy across a range.

    Each wave is normalised by its own largest magnitude. That is what makes
    the exhibit legible: a trial that misses the wall is dominated by its
    runaway tail, so it reads as a flat line with a spike at the far wall,
    while a trial that lands fills the box as a standing wave whose nodes can
    be counted.

    Pass ``h`` when the grid came from a dataset that recorded its own spacing.
    The Snyder benchmark stores an ``h`` that differs from ``x[1] - x[0]`` in
    the eighth significant figure, and reproducing its published eigenvalues
    needs the spacing its generator actually used: deriving the spacing from
    the coordinates instead moved the second eigenvalue by 5e-07 Hartree, which
    is small physics and a large embarrassment on a plaque claiming 1e-12.
    """
    x = np.asarray(x, dtype=np.float64)
    potential = np.asarray(potential, dtype=np.float64)
    if x.shape != potential.shape:
        raise ValueError("x and potential must have the same shape")
    h = float(x[1] - x[0]) if h is None else float(h)

    grid = np.linspace(energy_min, energy_max, trials)
    waves = np.empty((trials, x.shape[0]), dtype=np.float64)
    endpoint = np.empty(trials, dtype=np.float64)

    for index, energy in enumerate(grid):
        psi = integrate_numerov(potential, float(energy), h)
        scale = float(np.max(np.abs(psi)))
        endpoint[index] = psi[-1] / scale if scale > 0.0 else 0.0
        waves[index] = psi / scale if scale > 0.0 else psi

    return ShootingSweep(
        x=x,
        potential=potential,
        energies=grid,
        waves=waves,
        endpoint=endpoint,
        eigen_energies=find_eigenvalues(
            potential, h, energy_min=energy_min, energy_max=energy_max
        ),
    )
