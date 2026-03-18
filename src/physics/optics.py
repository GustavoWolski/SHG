"""Optical helper functions."""

from src.physics.constants import Z0


def rij(n1: complex, n2: complex, sig_s: float = 0.0) -> complex:
    """Return the Fresnel reflection coefficient."""
    return (n1 - n2 - Z0 * sig_s) / (n1 + n2 + Z0 * sig_s)


def tij(n1: complex, n2: complex, sig_s: float = 0.0) -> complex:
    """Return the Fresnel transmission coefficient."""
    return 2.0 * n1 / (n1 + n2 + Z0 * sig_s)


def nlimeglass(lambda_m: float) -> float:
    """Return the lime-glass refractive index."""
    l_um = lambda_m / 1e-6
    return 1.5130 - 0.003169 * (l_um ** 2) + 0.003962 / (l_um ** 2)
