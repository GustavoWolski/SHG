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


def _normalize_delimiter(delimiter: str | None) -> str | None:
    """Normalize user-facing delimiter aliases before calling NumPy loaders."""
    if delimiter is None:
        return None
    if delimiter.lower() in {"", "none", "space", "whitespace", "\\s+"}:
        return None
    return delimiter


def _split_header_tokens(line: str, delimiter: str | None) -> list[str]:
    """Split one possible header line using the same delimiter as numeric data."""
    if delimiter is None:
        return [token.strip().lower() for token in line.split()]
    return [token.strip().lower() for token in line.split(delimiter)]


def _detect_header_order(
    file_path: Path,
    delimiter: str | None,
    skiprows: int,
) -> tuple[bool, list[str]]:
    """Try to read the first non-skipped line as a header with known column names.

    Returns ``(has_header, column_order)`` where *column_order* contains the
    recognized columns when a header is found, or ``["d_nm", "i3", "i1"]``
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

    tokens = _split_header_tokens(first_line, delimiter)
    if (
        len(tokens) in {1, 2, 3}
        and len(set(tokens)) == len(tokens)
        and "d_nm" in tokens
        and set(tokens).issubset(EXPECTED_COLUMN_NAMES)
    ):
        return True, tokens
    return False, default_order


def _read_numeric_table(file_path: Path, delimiter: str | None, skiprows: int) -> FloatArray:
    """Read a text table into a two-dimensional float array."""
    columns = np.genfromtxt(
        file_path,
        delimiter=delimiter,
        dtype=np.float64,
        skip_header=skiprows,
        filling_values=np.nan,
    )
    return np.asarray(np.atleast_2d(columns), dtype=np.float64)


def _validate_thickness_grid(d_nm: FloatArray) -> FloatArray:
    """Validate a one-dimensional experimental thickness grid."""
    thickness_nm = np.asarray(d_nm, dtype=np.float64)
    if thickness_nm.ndim != 1 or thickness_nm.size == 0:
        raise ValueError("Experimental thickness grid must contain at least one d_nm value.")
    if not np.all(np.isfinite(thickness_nm)):
        raise ValueError("Experimental thickness values d_nm must be finite.")
    return thickness_nm


def load_experimental_thickness_grid(
    file_path: str | Path,
    delimiter: str | None = ",",
    skiprows: int = 0,
) -> FloatArray:
    """Load only the experimental thickness grid used for synthetic datasets.

    The file may contain one, two or three numeric columns.  If a recognized
    header is present, the ``d_nm`` column is selected by name; otherwise the
    first numeric column is used.
    """
    resolved_path = Path(file_path)
    resolved_delimiter = _normalize_delimiter(delimiter)
    has_header, column_order = _detect_header_order(resolved_path, resolved_delimiter, skiprows)
    actual_skiprows = skiprows + 1 if has_header else skiprows

    columns = _read_numeric_table(resolved_path, resolved_delimiter, actual_skiprows)
    if columns.ndim != 2 or columns.shape[1] < 1:
        raise ValueError("Experimental grid file must contain at least one numeric d_nm column.")

    d_nm_index = column_order.index("d_nm") if has_header else 0
    if d_nm_index >= columns.shape[1]:
        raise ValueError("Experimental grid header references d_nm but the numeric table is missing that column.")
    return _validate_thickness_grid(np.asarray(columns[:, d_nm_index], dtype=np.float64))


def load_experimental_shg_data(
    file_path: str | Path,
    delimiter: str | None = ",",
    skiprows: int = 0,
) -> ExperimentalSHGData:
    """Load experimental SHG data with optional missing i3/i1 channels.

    Headers with ``d_nm``, ``i3`` and/or ``i1`` are assigned by name.  This
    allows reflection-only files such as ``d_nm,i1``; the missing channel is
    represented as ``NaN`` and ignored through its observation mask.
    """
    resolved_path = Path(file_path)
    resolved_delimiter = _normalize_delimiter(delimiter)

    has_header, column_order = _detect_header_order(resolved_path, resolved_delimiter, skiprows)
    actual_skiprows = skiprows + 1 if has_header else skiprows

    columns = _read_numeric_table(resolved_path, resolved_delimiter, actual_skiprows)
    if columns.ndim != 2 or columns.shape[1] not in {2, 3}:
        raise ValueError("Experimental data file must contain d_nm plus at least one channel column: i3 and/or i1.")

    if not has_header:
        column_order = ["d_nm", "i3", "i1"] if columns.shape[1] == 3 else ["d_nm", "i1"]
    if len(column_order) != columns.shape[1]:
        raise ValueError("Experimental data header does not match the number of numeric columns.")

    col_index = {name: idx for idx, name in enumerate(column_order)}
    d_nm = np.asarray(columns[:, col_index["d_nm"]], dtype=np.float64)
    i3 = (
        np.asarray(columns[:, col_index["i3"]], dtype=np.float64)
        if "i3" in col_index
        else np.full(d_nm.shape, np.nan, dtype=np.float64)
    )
    i1 = (
        np.asarray(columns[:, col_index["i1"]], dtype=np.float64)
        if "i1" in col_index
        else np.full(d_nm.shape, np.nan, dtype=np.float64)
    )
    i3_mask = np.isfinite(i3)
    i1_mask = np.isfinite(i1)

    d_nm = _validate_thickness_grid(d_nm)
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
