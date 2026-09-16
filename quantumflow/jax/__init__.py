"""JAX implementations of QuantumFlow models and numerical utilities."""

from .convex import (
    composite_training_step,
    functional_derivative,
    init_icnn,
    kinetic_energy,
    make_composite_training_step,
    make_training_step,
    potential_from_kinetic_derivative,
    reconstruct_potential,
    solve_ground_state_density,
    training_step,
)

__all__ = [
    "composite_training_step",
    "functional_derivative",
    "init_icnn",
    "kinetic_energy",
    "make_composite_training_step",
    "make_training_step",
    "potential_from_kinetic_derivative",
    "reconstruct_potential",
    "solve_ground_state_density",
    "training_step",
]
