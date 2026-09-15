import pytest

tensorflow = pytest.importorskip("tensorflow")

from quantumflow.noninteracting_1d.convex import InputConvexKineticEnergyFunctional
from quantumflow.noninteracting_1d.derivative_model import potential_from_kinetic_derivative


class _Dataset:
    density = tensorflow.zeros((1, 8), dtype=tensorflow.float32)


def test_input_convex_functional_satisfies_jensen_inequality() -> None:
    tensorflow.keras.utils.set_random_seed(1)
    model = InputConvexKineticEnergyFunctional(None, _Dataset(), hidden_units=(4, 4))
    first_density = tensorflow.constant([[0.1] * 8], dtype=tensorflow.float32)
    second_density = tensorflow.constant([[0.8] * 8], dtype=tensorflow.float32)
    midpoint = (first_density + second_density) / 2

    midpoint_energy = model({"density": midpoint})["kinetic_energy"]
    mean_energy = (
        model({"density": first_density})["kinetic_energy"]
        + model({"density": second_density})["kinetic_energy"]
    ) / 2

    assert float(midpoint_energy[0]) <= float(mean_energy[0]) + 1e-6


def test_potential_is_euler_equation_complement() -> None:
    derivative = tensorflow.constant([[1.0, 2.0]], dtype=tensorflow.float32)

    potential = potential_from_kinetic_derivative(derivative, 3.0)

    tensorflow.debugging.assert_near(potential, [[2.0, 1.0]])
