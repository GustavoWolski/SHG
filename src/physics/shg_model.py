"""SHG forward model."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from src.physics.optics import nlimeglass, rij, tij

FloatArray = npt.NDArray[np.float64]


@dataclass
class SHGParams:
    """Parameters of the SHG simulation."""

    lambda_m: float
    n21w: complex
    n22w: complex


def simulate_shg(params: SHGParams, d_nm: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Simulate transmitted and reflected SHG intensities."""
    i = 1j
    lambda_m = params.lambda_m
    k0 = 2.0 * np.pi / lambda_m
    d_m = np.asarray(d_nm, dtype=np.float64) * 1e-9

    n11w = 1.0
    n31w = nlimeglass(lambda_m)
    fase21w = 2.0 * params.n21w * k0 * d_m

    r12b = rij(n11w, params.n21w, 0.0)
    r23b = rij(params.n21w, n31w, 0.0)
    r21b = -r12b
    t21b = tij(params.n21w, n11w, 0.0)

    phi2 = np.exp(i * fase21w)
    r = (r12b + r23b * phi2) / (1.0 + r12b * r23b * phi2)

    emas = (1.0 + r21b * r) / t21b
    emen = (r21b + r) / t21b

    emas2k = emas ** 2
    emen2k = emen ** 2
    emas0k = emas * emen
    emen0k = emen * emas

    lamb_shg = lambda_m / 2.0
    n12w = 1.0
    n32w = nlimeglass(lamb_shg)

    r2s2k = (params.n22w - params.n21w) / (params.n22w + params.n21w)
    t2s2k = 2.0 * n12w / (params.n22w + params.n21w)

    r2s0k = 1.0
    t2s0k = 2.0 * n12w / params.n22w

    r122w = rij(n12w, params.n22w, 0.0)
    t122w = tij(n12w, params.n22w, 0.0)
    r232w = rij(params.n22w, n32w, 0.0)
    t232w = tij(params.n22w, n32w, 0.0)

    i3_2k = np.zeros_like(d_m, dtype=np.complex128)
    i1_2k = np.zeros_like(d_m, dtype=np.complex128)
    i3_0k = np.zeros_like(d_m, dtype=np.complex128)
    i1_0k = np.zeros_like(d_m, dtype=np.complex128)

    for idx, di in enumerate(d_m):
        phi22w = np.exp(i * params.n22w * 2.0 * k0 * di)
        phi2s2k = np.exp(i * 2.0 * params.n21w * k0 * di)

        es2k = np.array([[emas2k[idx]], [emen2k[idx]]], dtype=np.complex128)
        s2k = np.array(
            [
                [(phi2s2k / phi22w - 1.0) / t2s2k, (1.0 / (phi2s2k * phi22w) - 1.0) * r2s2k / t2s2k],
                [(phi2s2k * phi22w - 1.0) * r2s2k / t2s2k, (phi22w / phi2s2k - 1.0) / t2s2k],
            ],
            dtype=np.complex128,
        )
        cs2k = 4.0 * np.pi / (params.n22w ** 2 - params.n21w ** 2)
        smm = cs2k * s2k @ es2k
        smas = smm[0, 0]
        smen = smm[1, 0]

        num = 1.0 + r122w * r232w * (phi22w ** 2)
        e3mas2k = phi22w * t232w * (smas + r122w * smen) / num
        e3mem2k = t122w * (smas * r232w * (phi22w ** 2) - smen) / num
        i3_2k[idx] = e3mas2k * np.conj(e3mas2k)
        i1_2k[idx] = e3mem2k * np.conj(e3mem2k)

        phi2s0k = 1.0
        es0k = np.array([[emas0k[idx]], [emen0k[idx]]], dtype=np.complex128)
        s0k = np.array(
            [
                [(phi2s0k / phi22w - 1.0) / t2s0k, (1.0 / (phi2s0k * phi22w) - 1.0) * r2s0k / t2s0k],
                [(phi2s0k * phi22w - 1.0) * r2s0k / t2s0k, (phi22w / phi2s0k - 1.0) / t2s0k],
            ],
            dtype=np.complex128,
        )
        cs0k = 4.0 * np.pi / (params.n22w ** 2)
        smm0 = cs0k * s0k @ es0k
        smas0 = smm0[0, 0]
        smen0 = smm0[1, 0]

        num0 = 1.0 + r122w * r232w * (phi22w ** 2)
        e3mas0k = phi22w * t232w * (smas0 + r122w * smen0) / num0
        e3mem0k = t122w * (smas0 * r232w * (phi22w ** 2) - smen0) / num0
        i3_0k[idx] = e3mas0k * np.conj(e3mas0k)
        i1_0k[idx] = e3mem0k * np.conj(e3mem0k)

    i3 = (i3_2k + i3_0k).real.astype(np.float64)
    i1 = (i1_2k + i1_0k).real.astype(np.float64)
    return i3, i1
