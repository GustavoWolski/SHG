"""Loading helpers for SHG datasets."""

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import numpy.typing as npt

from src.data.synthetic_generator import NormalizationMode, SyntheticSHGDataset

FloatArray = npt.NDArray[np.float64]


@dataclass
class ExperimentalSHGData:
    """Experimental SHG curves loaded from an external text file."""

    d_nm: FloatArray
    i3: FloatArray
    i1: FloatArray
    i3_mask: npt.NDArray[np.bool_]
    i1_mask: npt.NDArray[np.bool_]


def load_columns(file_path: str | Path, delimiter: str = ",", skiprows: int = 0) -> FloatArray:
    """Load numeric columns from a text file."""
    return np.loadtxt(Path(file_path), delimiter=delimiter, dtype=np.float64, skiprows=skiprows)


EXPECTED_COLUMN_NAMES: set[str] = {"d_nm", "i3", "i1"}


def _parse_normalization_mode(raw_value: object) -> NormalizationMode:
    """Validate a serialized normalization mode loaded from disk."""
    normalization = str(raw_value)
    if normalization == "none":
        return "none"
    if normalization == "global":
        return "global"
    if normalization == "separate":
        return "separate"
    raise ValueError(f"Unknown normalization mode in dataset file: {normalization!r}")


def _detect_header_order(
    file_path: Path,
    delimiter: str,
    skiprows: int,
) -> tuple[bool, list[str]]:
    """Try to read the first non-skipped line as a header with known column names.

    Returns ``(has_header, column_order)`` where *column_order* is always a
    three-element list of ``"d_nm"``, ``"i3"`` and ``"i1"`` (in whichever
    order they appear) when a header is found, or ``["d_nm", "i3", "i1"]``
    as default when no header is detected.
    """
    default_order = ["d_nm", "i3", "i1"]
    try:
        with open(file_path, encoding="utf-8") as fh:
            for _ in range(skiprows):
                next(fh, None)
            first_line = next(fh, None)
    except (OSError, StopIteration):
        return False, default_order

    if first_line is None:
        return False, default_order

    tokens = [token.strip().lower() for token in first_line.split(delimiter)]
    if set(tokens) == EXPECTED_COLUMN_NAMES and len(tokens) == 3:
        return True, tokens
    return False, default_order


def load_experimental_shg_data(
    file_path: str | Path,
    delimiter: str = ",",
    skiprows: int = 0,
) -> ExperimentalSHGData:
    """Load experimental SHG data with optional missing i3/i1 values.

    The loader auto-detects the header row when ``skiprows >= 1``.  If the
    first non-skipped line contains exactly the tokens ``d_nm``, ``i3`` and
    ``i1`` (in any order), the columns are assigned by name instead of by
    position.  This allows CSV files with ``d_nm,i1,i3`` to be loaded
    correctly without manual column reordering.
    """
    resolved_path = Path(file_path)

    has_header, column_order = _detect_header_order(resolved_path, delimiter, skiprows)
    actual_skiprows = skiprows + 1 if has_header else skiprows

    columns = np.genfromtxt(
        resolved_path,
        delimiter=delimiter,
        dtype=np.float64,
        skip_header=actual_skiprows,
        filling_values=np.nan,
    )
    columns = np.asarray(np.atleast_2d(columns), dtype=np.float64)
    if columns.ndim != 2 or columns.shape[1] != 3:
        raise ValueError("Experimental data file must contain exactly 3 numeric columns: d_nm, i3, i1.")

    col_index = {name: idx for idx, name in enumerate(column_order)}
    d_nm = np.asarray(columns[:, col_index["d_nm"]], dtype=np.float64)
    i3 = np.asarray(columns[:, col_index["i3"]], dtype=np.float64)
    i1 = np.asarray(columns[:, col_index["i1"]], dtype=np.float64)
    i3_mask = np.isfinite(i3)
    i1_mask = np.isfinite(i1)

    if not np.all(np.isfinite(d_nm)):
        raise ValueError("Experimental thickness values d_nm must be finite.")
    if not np.any(i3_mask | i1_mask):
        raise ValueError("Experimental data must contain at least one finite i3 or i1 value.")

    return ExperimentalSHGData(
        d_nm=d_nm,
        i3=i3,
        i1=i1,
        i3_mask=i3_mask,
        i1_mask=i1_mask,
    )


def load_synthetic_dataset(file_path: str | Path) -> SyntheticSHGDataset:
    """Load a synthetic SHG dataset saved as NPZ."""
    with np.load(Path(file_path), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        seed_value = int(data["seed"].item())
        normalization_mode = _parse_normalization_mode(data["normalization"].item())
        return SyntheticSHGDataset(
            d_nm=np.asarray(data["d_nm"], dtype=np.float64),
            i3=np.asarray(data["i3"], dtype=np.float64),
            i1=np.asarray(data["i1"], dtype=np.float64),
            curves=np.asarray(data["curves"], dtype=np.float64),
            parameters=np.asarray(data["parameters"], dtype=np.float64),
            lambda_m=float(data["lambda_m"].item()),
            bounds={name: tuple(values) for name, values in metadata["bounds"].items()},
            normalization=normalization_mode,
            seed=None if seed_value < 0 else seed_value,
        )
