"""JAX implementations of QuantumFlow models and numerical utilities."""

from .convex import (
    composite_training_step,
    functional_derivative,
    init_icnn,
    kinetic_energy,
    make_composite_training_step,
    make_training_step,
    multispecies_kinetic_energy,
    potential_from_kinetic_derivative,
    quantum_potential,
    reconstruct_potential,
    regularized_kinetic_energy,
    solve_ground_state_density,
    solve_ground_state_density_stabilized,
    spin_resolved_kinetic_energy,
    training_step,
    von_weizsaecker_kinetic_energy,
)

__all__ = [
    "composite_training_step",
    "functional_derivative",
    "init_icnn",
    "kinetic_energy",
    "make_composite_training_step",
    "make_training_step",
    "multispecies_kinetic_energy",
    "potential_from_kinetic_derivative",
    "quantum_potential",
    "reconstruct_potential",
    "regularized_kinetic_energy",
    "solve_ground_state_density",
    "solve_ground_state_density_stabilized",
    "spin_resolved_kinetic_energy",
    "training_step",
    "von_weizsaecker_kinetic_energy",
]
