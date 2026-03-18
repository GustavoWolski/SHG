"""Optical helper functions."""

import numpy as np

from src.physics.constants import Z0

MIN_DENOMINATOR_ABS: float = 1e-14


def _ensure_finite_complex(name: str, value: complex) -> None:
    """Validate that a complex scalar is finite."""
    if not np.isfinite(value.real) or not np.isfinite(value.imag):
        raise ValueError(f"{name} must be finite.")


def _ensure_nonzero_denominator(name: str, value: complex) -> None:
    """Validate that a denominator is finite and not too small."""
    _ensure_finite_complex(name, value)
    if abs(value) <= MIN_DENOMINATOR_ABS:
        raise ZeroDivisionError(f"{name} is too close to zero for a stable division.")


def rij(n1: complex, n2: complex, sig_s: float = 0.0) -> complex:
    """Return the Fresnel reflection coefficient."""
    denominator = n1 + n2 + Z0 * sig_s
    _ensure_nonzero_denominator("Fresnel reflection denominator", denominator)
    return (n1 - n2 - Z0 * sig_s) / denominator


def tij(n1: complex, n2: complex, sig_s: float = 0.0) -> complex:
    """Return the Fresnel transmission coefficient."""
    denominator = n1 + n2 + Z0 * sig_s
    _ensure_nonzero_denominator("Fresnel transmission denominator", denominator)
    return 2.0 * n1 / denominator


def nlimeglass(lambda_m: float) -> float:
    """Return the lime-glass refractive index."""
    if not np.isfinite(lambda_m) or lambda_m <= 0.0:
        raise ValueError("lambda_m must be positive and finite.")

    wavelength_um = lambda_m / 1e-6
    if abs(wavelength_um) <= MIN_DENOMINATOR_ABS:
        raise ZeroDivisionError("lambda_m is too close to zero for the glass-index model.")

    return 1.5130 - 0.003169 * (wavelength_um ** 2) + 0.003962 / (wavelength_um ** 2)
