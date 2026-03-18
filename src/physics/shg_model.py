"""SHG forward model."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt

from src.physics.optics import nlimeglass, rij, tij

FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]

MIN_DENOMINATOR_ABS: float = 1e-14
NON_NEGATIVE_TOLERANCE: float = 1e-12


@dataclass
class SHGParams:
    """Parameters of the SHG simulation."""

    lambda_m: float
    n21w: complex
    n22w: complex


def _ensure_finite_real_array(name: str, values: FloatArray) -> None:
    """Validate that a real array is finite."""
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains NaN or inf.")


def _ensure_finite_complex(name: str, value: complex) -> None:
    """Validate that a complex scalar is finite."""
    if not np.isfinite(value.real) or not np.isfinite(value.imag):
        raise ValueError(f"{name} must be finite.")


def _ensure_finite_complex_array(name: str, values: ComplexArray) -> None:
    """Validate that a complex array is finite."""
    if not np.all(np.isfinite(values.real)) or not np.all(np.isfinite(values.imag)):
        raise ValueError(f"{name} contains NaN or inf.")


def _guard_denominator(name: str, denominator: complex) -> complex:
    """Validate a scalar denominator before division."""
    _ensure_finite_complex(name, denominator)
    if abs(denominator) <= MIN_DENOMINATOR_ABS:
        raise ZeroDivisionError(f"{name} is too close to zero for a stable division.")
    return denominator


def _checked_divide(numerator: complex | ComplexArray, denominator: complex, name: str) -> complex | ComplexArray:
    """Divide after validating the denominator and the result."""
    stable_denominator = _guard_denominator(name, denominator)
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        result = numerator / stable_denominator

    result_array = np.asarray(result)
    if np.iscomplexobj(result_array):
        _ensure_finite_complex_array(name, result_array.astype(np.complex128, copy=False))
    else:
        _ensure_finite_real_array(name, result_array.astype(np.float64, copy=False))
    return result


def _validate_inputs(params: SHGParams, d_nm: FloatArray) -> FloatArray:
    """Validate SHG model inputs."""
    if not np.isfinite(params.lambda_m) or params.lambda_m <= 0.0:
        raise ValueError("params.lambda_m must be positive and finite.")

    _ensure_finite_complex("params.n21w", params.n21w)
    _ensure_finite_complex("params.n22w", params.n22w)

    thickness_nm = np.asarray(d_nm, dtype=np.float64)
    if thickness_nm.ndim != 1:
        raise ValueError("d_nm must be a one-dimensional array.")

    _ensure_finite_real_array("d_nm", thickness_nm)
    if np.any(thickness_nm < 0.0):
        raise ValueError("d_nm must be non-negative.")

    return thickness_nm


def default_shg_params() -> SHGParams:
    """Return the current default SHG parameters."""
    return SHGParams(
        lambda_m=1560e-9,
        n21w=complex(5.6428, 0.0849),
        n22w=complex(2.8698, 0.4492),
    )


def simulate_shg(params: SHGParams, d_nm: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Simulate transmitted and reflected SHG intensities."""
    thickness_nm = _validate_inputs(params, d_nm)

    imaginary_unit = 1j
    lambda_fundamental_m = params.lambda_m
    vacuum_wavenumber = _checked_divide(2.0 * np.pi, lambda_fundamental_m, "vacuum wavenumber")
    thickness_m = thickness_nm * 1e-9

    n_incident_fundamental = 1.0
    n_substrate_fundamental = nlimeglass(lambda_fundamental_m)
    pump_roundtrip_phase_argument = 2.0 * params.n21w * vacuum_wavenumber * thickness_m

    r_air_film_fundamental = rij(n_incident_fundamental, params.n21w, 0.0)
    r_film_substrate_fundamental = rij(params.n21w, n_substrate_fundamental, 0.0)
    r_film_air_fundamental = -r_air_film_fundamental
    t_film_air_fundamental = tij(params.n21w, n_incident_fundamental, 0.0)

    pump_roundtrip_phase = np.exp(imaginary_unit * pump_roundtrip_phase_argument)

    # Physically sensitive: this denominator encodes the effective cavity response of the pump field.
    pump_reflection_denominator = 1.0 + r_air_film_fundamental * r_film_substrate_fundamental * pump_roundtrip_phase
    if np.any(np.abs(pump_reflection_denominator) <= MIN_DENOMINATOR_ABS):
        raise ZeroDivisionError("Pump cavity denominator is too close to zero.")

    effective_pump_reflection = (r_air_film_fundamental + r_film_substrate_fundamental * pump_roundtrip_phase) / (
        pump_reflection_denominator
    )
    _ensure_finite_complex_array(
        "effective pump reflection",
        np.asarray(effective_pump_reflection, dtype=np.complex128),
    )

    pump_forward_field = _checked_divide(
        1.0 + r_film_air_fundamental * effective_pump_reflection,
        t_film_air_fundamental,
        "pump forward field",
    )
    pump_backward_field = _checked_divide(
        r_film_air_fundamental + effective_pump_reflection,
        t_film_air_fundamental,
        "pump backward field",
    )

    pump_forward_sq_2k = pump_forward_field ** 2
    pump_backward_sq_2k = pump_backward_field ** 2
    pump_mixed_forward_0k = pump_forward_field * pump_backward_field

    # TODO: confirm whether the backward 0k source term should differ from the forward one.
    # The current expression is algebraically identical to the forward product and is preserved from the original code.
    pump_mixed_backward_0k = pump_backward_field * pump_forward_field

    lambda_shg_m = lambda_fundamental_m / 2.0
    n_incident_shg = 1.0
    n_substrate_shg = nlimeglass(lambda_shg_m)

    film_index_sum = params.n22w + params.n21w
    r_source_2k = _checked_divide(params.n22w - params.n21w, film_index_sum, "2k source reflection coefficient")
    t_source_2k = _checked_divide(2.0 * n_incident_shg, film_index_sum, "2k source transmission coefficient")

    r_source_0k = 1.0
    t_source_0k = _checked_divide(2.0 * n_incident_shg, params.n22w, "0k source transmission coefficient")

    r_air_film_shg = rij(n_incident_shg, params.n22w, 0.0)
    t_air_film_shg = tij(n_incident_shg, params.n22w, 0.0)
    r_film_substrate_shg = rij(params.n22w, n_substrate_shg, 0.0)
    t_film_substrate_shg = tij(params.n22w, n_substrate_shg, 0.0)

    transmitted_intensity_2k = np.zeros_like(thickness_m, dtype=np.complex128)
    reflected_intensity_2k = np.zeros_like(thickness_m, dtype=np.complex128)
    transmitted_intensity_0k = np.zeros_like(thickness_m, dtype=np.complex128)
    reflected_intensity_0k = np.zeros_like(thickness_m, dtype=np.complex128)

    # Physically sensitive: n22w**2 - n21w**2 acts like a phase-mismatch denominator and can strongly amplify noise.
    source_prefactor_2k = _checked_divide(
        4.0 * np.pi,
        params.n22w ** 2 - params.n21w ** 2,
        "2k source prefactor",
    )

    # Physically sensitive: the 0k source term scales as 1 / n22w**2 and becomes unstable near vanishing SHG index.
    source_prefactor_0k = _checked_divide(4.0 * np.pi, params.n22w ** 2, "0k source prefactor")

    for index, thickness_value_m in enumerate(thickness_m):
        shg_layer_phase = np.exp(imaginary_unit * params.n22w * 2.0 * vacuum_wavenumber * thickness_value_m)
        source_phase_2k = np.exp(imaginary_unit * 2.0 * params.n21w * vacuum_wavenumber * thickness_value_m)

        pump_source_vector_2k = np.array(
            [[pump_forward_sq_2k[index]], [pump_backward_sq_2k[index]]],
            dtype=np.complex128,
        )

        source_to_shg_ratio_2k = _checked_divide(source_phase_2k, shg_layer_phase, "2k source/shg phase ratio")
        inverse_phase_product_2k = _checked_divide(
            1.0,
            source_phase_2k * shg_layer_phase,
            "2k inverse phase product",
        )
        shg_to_source_ratio_2k = _checked_divide(shg_layer_phase, source_phase_2k, "2k shg/source phase ratio")

        source_matrix_2k = np.array(
            [
                [
                    _checked_divide(source_to_shg_ratio_2k - 1.0, t_source_2k, "2k source matrix (0,0)"),
                    _checked_divide(
                        (inverse_phase_product_2k - 1.0) * r_source_2k,
                        t_source_2k,
                        "2k source matrix (0,1)",
                    ),
                ],
                [
                    _checked_divide(
                        (source_phase_2k * shg_layer_phase - 1.0) * r_source_2k,
                        t_source_2k,
                        "2k source matrix (1,0)",
                    ),
                    _checked_divide(shg_to_source_ratio_2k - 1.0, t_source_2k, "2k source matrix (1,1)"),
                ],
            ],
            dtype=np.complex128,
        )
        nonlinear_source_2k = source_prefactor_2k * source_matrix_2k @ pump_source_vector_2k
        source_forward_2k = nonlinear_source_2k[0, 0]
        source_backward_2k = nonlinear_source_2k[1, 0]

        # Physically sensitive: this Fabry-Perot-like denominator can become singular close to SHG cavity resonances.
        shg_cavity_denominator = 1.0 + r_air_film_shg * r_film_substrate_shg * (shg_layer_phase ** 2)
        shg_cavity_denominator = _guard_denominator("SHG cavity denominator", shg_cavity_denominator)

        transmitted_field_2k = _checked_divide(
            shg_layer_phase * t_film_substrate_shg * (source_forward_2k + r_air_film_shg * source_backward_2k),
            shg_cavity_denominator,
            "2k transmitted field",
        )
        reflected_field_2k = _checked_divide(
            t_air_film_shg * (source_forward_2k * r_film_substrate_shg * (shg_layer_phase ** 2) - source_backward_2k),
            shg_cavity_denominator,
            "2k reflected field",
        )
        transmitted_intensity_2k[index] = transmitted_field_2k * np.conj(transmitted_field_2k)
        reflected_intensity_2k[index] = reflected_field_2k * np.conj(reflected_field_2k)

        source_phase_0k = 1.0
        pump_source_vector_0k = np.array(
            [[pump_mixed_forward_0k[index]], [pump_mixed_backward_0k[index]]],
            dtype=np.complex128,
        )
        inverse_phase_product_0k = _checked_divide(
            1.0,
            source_phase_0k * shg_layer_phase,
            "0k inverse phase product",
        )
        shg_to_source_ratio_0k = _checked_divide(shg_layer_phase, source_phase_0k, "0k shg/source phase ratio")

        source_matrix_0k = np.array(
            [
                [
                    _checked_divide(
                        _checked_divide(source_phase_0k, shg_layer_phase, "0k source/shg phase ratio") - 1.0,
                        t_source_0k,
                        "0k source matrix (0,0)",
                    ),
                    _checked_divide(
                        (inverse_phase_product_0k - 1.0) * r_source_0k,
                        t_source_0k,
                        "0k source matrix (0,1)",
                    ),
                ],
                [
                    _checked_divide(
                        (source_phase_0k * shg_layer_phase - 1.0) * r_source_0k,
                        t_source_0k,
                        "0k source matrix (1,0)",
                    ),
                    _checked_divide(shg_to_source_ratio_0k - 1.0, t_source_0k, "0k source matrix (1,1)"),
                ],
            ],
            dtype=np.complex128,
        )
        nonlinear_source_0k = source_prefactor_0k * source_matrix_0k @ pump_source_vector_0k
        source_forward_0k = nonlinear_source_0k[0, 0]
        source_backward_0k = nonlinear_source_0k[1, 0]

        transmitted_field_0k = _checked_divide(
            shg_layer_phase * t_film_substrate_shg * (source_forward_0k + r_air_film_shg * source_backward_0k),
            shg_cavity_denominator,
            "0k transmitted field",
        )
        reflected_field_0k = _checked_divide(
            t_air_film_shg * (source_forward_0k * r_film_substrate_shg * (shg_layer_phase ** 2) - source_backward_0k),
            shg_cavity_denominator,
            "0k reflected field",
        )
        transmitted_intensity_0k[index] = transmitted_field_0k * np.conj(transmitted_field_0k)
        reflected_intensity_0k[index] = reflected_field_0k * np.conj(reflected_field_0k)

    transmitted_intensity = (transmitted_intensity_2k + transmitted_intensity_0k).real.astype(np.float64)
    reflected_intensity = (reflected_intensity_2k + reflected_intensity_0k).real.astype(np.float64)

    _ensure_finite_real_array("transmitted intensity", transmitted_intensity)
    _ensure_finite_real_array("reflected intensity", reflected_intensity)

    return transmitted_intensity, reflected_intensity


def validate_default_simulation(
    params: Optional[SHGParams] = None,
    d_nm: Optional[FloatArray] = None,
    non_negative_tolerance: float = NON_NEGATIVE_TOLERANCE,
) -> tuple[FloatArray, FloatArray]:
    """Run a basic self-check of the SHG simulation."""
    simulation_params = params if params is not None else default_shg_params()
    thickness_nm = (
        np.asarray(d_nm, dtype=np.float64)
        if d_nm is not None
        else np.arange(0.0, 601.0, 1.0, dtype=np.float64)
    )

    transmitted_intensity, reflected_intensity = simulate_shg(simulation_params, thickness_nm)

    if transmitted_intensity.shape != thickness_nm.shape or reflected_intensity.shape != thickness_nm.shape:
        raise ValueError("Simulation outputs must have the same shape as d_nm.")

    _ensure_finite_real_array("transmitted intensity", transmitted_intensity)
    _ensure_finite_real_array("reflected intensity", reflected_intensity)

    if np.any(transmitted_intensity < -non_negative_tolerance):
        raise ValueError("Transmitted intensity contains negative values beyond tolerance.")
    if np.any(reflected_intensity < -non_negative_tolerance):
        raise ValueError("Reflected intensity contains negative values beyond tolerance.")

    return transmitted_intensity, reflected_intensity
