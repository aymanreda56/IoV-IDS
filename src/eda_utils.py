"""Shared helpers for the CICIoV2024 EDA notebook.

This module centralizes the heavy / reusable logic (file discovery, robust
loading with schema reconciliation, lossless dtype downcasting, chunked reads,
plotting palette and export helpers) so the notebook stays readable and fast.

The CICIoV2024 dataset is provided in three equivalent encodings (binary,
decimal, hexadecimal). Each encoding lives in its own folder and contains six
label-pure CSV files (one per traffic class). This module loads them, aligns
columns by name (tolerating the hexadecimal DoS file that is missing the ``DLC``
column), and attaches a clean ``specific_class`` target.
"""

# ``from __future__ import annotations`` lets us write type hints (like
# ``Dict[str, Path]``) without runtime cost; they are treated as strings.
from __future__ import annotations

# ``Path`` gives us safe, OS-independent file paths (no backslash/slash issues).
from pathlib import Path

# These are just type-hint helpers used in the function signatures below.
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

# ``__file__`` is THIS file (src/eda_utils.py). ``.resolve()`` makes it an
# absolute path, ``.parent`` is the ``src/`` folder, and ``.parent.parent`` is
# the project root (one level above ``src/``). Building paths this way means the
# notebook works no matter what the current working directory is.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT  # the 3 encoding folders live directly in the project root
OUTPUTS_DIR = PROJECT_ROOT / "outputs"        # the "/" operator joins paths
FIGURES_DIR = OUTPUTS_DIR / "figures"         # PNG charts go here
TABLES_DIR = OUTPUTS_DIR / "tables"           # CSV summary tables go here

# Maps a human-readable class name -> the suffix used in its CSV file name.
# e.g. for the decimal encoding, "GAS" -> "decimal_spoofing-GAS.csv".
# We list files explicitly (rather than scanning the folder) so macOS junk
# files like ".DS_Store" are never accidentally read.
CLASS_FILE_SUFFIX: Dict[str, str] = {
    "BENIGN": "benign",
    "DoS": "DoS",
    "GAS": "spoofing-GAS",
    "RPM": "spoofing-RPM",
    "SPEED": "spoofing-SPEED",
    "STEERING_WHEEL": "spoofing-STEERING_WHEEL",
}

# The three equivalent number-base encodings of the same CAN frames.
ENCODINGS = ("decimal", "hexadecimal", "binary")

# The three "answer" columns. They are NOT features; we exclude them when we
# look at the inputs the model would use.
LABEL_COLS = ["label", "category", "specific_class"]


def ensure_output_dirs() -> None:
    """Create the ``outputs/figures`` and ``outputs/tables`` directories."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# File discovery
# --------------------------------------------------------------------------- #


def encoding_files(encoding: str) -> Dict[str, Path]:
    """Return a mapping of class name -> CSV path for a given encoding.

    macOS junk files (``.DS_Store``, ``._*``) are never referenced because we
    build paths explicitly from :data:`CLASS_FILE_SUFFIX` rather than globbing
    the directory contents.

    Args:
        encoding: One of ``"decimal"``, ``"hexadecimal"``, ``"binary"``.

    Returns:
        Dict mapping each class name to its existing CSV ``Path``.

    Raises:
        FileNotFoundError: If a required class file is missing.
    """
    # Guard against typos like "decimel" before we try to build paths.
    if encoding not in ENCODINGS:
        raise ValueError(f"Unknown encoding {encoding!r}; expected one of {ENCODINGS}")

    folder = DATA_DIR / encoding  # e.g. <project>/decimal
    paths: Dict[str, Path] = {}
    # Build one path per class, e.g. <project>/decimal/decimal_spoofing-GAS.csv
    for class_name, suffix in CLASS_FILE_SUFFIX.items():
        path = folder / f"{encoding}_{suffix}.csv"
        # Fail early with a clear message if a file is missing, rather than
        # producing a confusing error deep inside pandas later.
        if not path.is_file():
            raise FileNotFoundError(f"Expected dataset file not found: {path}")
        paths[class_name] = path
    return paths


# --------------------------------------------------------------------------- #
# Dtype downcasting (lossless)
# --------------------------------------------------------------------------- #


def downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast integer columns to the smallest lossless unsigned/signed type.

    This only changes how values are *stored*, not the values themselves: every
    value still fits its target dtype (data bytes are 0-255 -> uint8, CAN IDs are
    0-2047 -> uint16, bits are 0/1 -> uint8). No rounding or truncation occurs.

    Args:
        df: Input dataframe (may contain non-numeric label columns, untouched).

    Returns:
        The same dataframe with numeric columns downcast in place.
    """
    for col in df.columns:
        # Skip the text label columns (label/category/specific_class).
        if col in LABEL_COLS:
            continue
        # Only touch integer columns. ``downcast="unsigned"`` asks pandas to use
        # the smallest unsigned integer type that still holds every value:
        #   bytes 0-255   -> uint8  (1 byte each, vs 8 bytes for default int64)
        #   IDs   0-2047  -> uint16 (2 bytes each)
        #   bits  0/1     -> uint8
        # Because the chosen type fits the value range, NOTHING is lost.
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="unsigned")
    return df


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Read a single CSV, dropping any stray fully-empty 'Unnamed' columns."""
    # low_memory=False reads each column in one pass so pandas infers a single
    # consistent dtype. Without it, the malformed hex SPEED DATA_6 column (which
    # mixes "02" and "2.0") triggers a noisy mixed-type warning.
    kwargs.setdefault("low_memory", False)
    df = pd.read_csv(path, **kwargs)
    # A trailing comma in a CSV header creates a phantom "Unnamed: N" column.
    # We detect and drop any such columns so they don't pollute the analysis.
    junk = [c for c in df.columns if str(c).startswith("Unnamed")]
    if junk:
        df = df.drop(columns=junk)
    return df


def load_encoding(
    encoding: str,
    classes: Optional[Iterable[str]] = None,
    downcast: bool = True,
    benign_nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load and concatenate all class files for one encoding.

    Columns are aligned by name across files, which tolerates the hexadecimal
    DoS file that is missing the ``DLC`` column (it becomes NaN for those rows).

    Args:
        encoding: ``"decimal"``, ``"hexadecimal"`` or ``"binary"``.
        classes: Optional subset of class names to load (default: all six).
        downcast: If True, losslessly downcast integer columns to save memory.
        benign_nrows: Optional cap on benign rows (useful for the huge binary
            file). ``None`` loads all benign rows.

    Returns:
        A single concatenated dataframe with a reset index. The
        ``specific_class`` column is the recommended multi-class target.
    """
    # Get {class_name: path} for all six files in this encoding.
    paths = encoding_files(encoding)
    # If the caller asked for only some classes, keep just those.
    if classes is not None:
        paths = {c: paths[c] for c in classes}

    frames: List[pd.DataFrame] = []  # will hold one dataframe per class file
    for class_name, path in paths.items():
        # Only the benign file can be huge (~1.2M rows); allow capping it.
        nrows = benign_nrows if class_name == "BENIGN" else None
        df = _read_csv(path, nrows=nrows)
        frames.append(df)

    # ``pd.concat`` stacks the per-class tables on top of each other (axis=0).
    #   - ``ignore_index=True`` renumbers the rows 0..N-1 cleanly.
    #   - ``sort=False`` keeps column order stable.
    # Crucially, concat aligns columns BY NAME: the hex DoS file has no ``DLC``
    # column, so those rows simply get ``NaN`` for ``DLC`` instead of shifting
    # every value one column to the left.
    combined = pd.concat(frames, axis=0, ignore_index=True, sort=False)

    if downcast:
        combined = downcast_numeric(combined)  # shrink memory, lossless
    return combined


def iter_binary_benign_chunks(chunksize: int = 200_000):
    """Yield chunks of the 400MB binary benign file for aggregate stats.

    Use this instead of loading the whole file when you only need streaming
    aggregates (e.g. per-column means) and want to bound memory use.

    Args:
        chunksize: Number of rows per chunk.

    Yields:
        ``pandas.DataFrame`` chunks.
    """
    path = encoding_files("binary")["BENIGN"]
    # Passing ``chunksize`` makes pandas return an iterator that reads the file
    # in pieces of ``chunksize`` rows. We never hold the whole 400MB file in
    # memory at once. ``yield`` hands each downcast chunk back to the caller,
    # who can accumulate aggregates (e.g. running sums) across chunks.
    for chunk in pd.read_csv(path, chunksize=chunksize):
        yield downcast_numeric(chunk)


# --------------------------------------------------------------------------- #
# Analysis helpers
# --------------------------------------------------------------------------- #

# Stable, colorblind-friendly palette keyed by specific_class.
CLASS_ORDER = ["BENIGN", "DoS", "GAS", "RPM", "SPEED", "STEERING_WHEEL"]
CLASS_PALETTE = {
    "BENIGN": "#4C72B0",
    "DoS": "#C44E52",
    "GAS": "#DD8452",
    "RPM": "#55A868",
    "SPEED": "#8172B3",
    "STEERING_WHEEL": "#937860",
}


def feature_columns(df: pd.DataFrame) -> List[str]:
    """Return the model-feature columns (everything except the 3 label cols)."""
    return [c for c in df.columns if c not in LABEL_COLS]


def unique_frame_counts(df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Count total vs unique feature-frames per ``specific_class``.

    A "frame" is a unique combination of the feature columns (e.g. ID + the 8
    data bytes). This exposes how few distinct patterns each attack class has,
    which is the root cause of train/test leakage under a naive random split.

    Args:
        df: Combined dataframe containing ``specific_class``.
        feature_cols: Columns that define a frame (default: all feature cols).

    Returns:
        DataFrame indexed by class with ``rows`` and ``unique_frames`` columns.
    """
    # Default to all feature columns (ID + the 8 data bytes for decimal).
    if feature_cols is None:
        feature_cols = feature_columns(df)
    # ``groupby(...).size()`` = how many rows each class has (total frames).
    rows = df.groupby("specific_class", observed=True).size()
    # For each class, drop duplicate feature-rows and count what remains =
    # the number of DISTINCT frames that class actually contains.
    uniques = df.groupby("specific_class", observed=True)[feature_cols].apply(
        lambda g: g.drop_duplicates().shape[0]  # .shape[0] = number of rows
    )
    # Combine both numbers side by side into one small table.
    out = pd.DataFrame({"rows": rows, "unique_frames": uniques})
    # Reorder the rows into our preferred class order (BENIGN, DoS, ...).
    return out.reindex([c for c in CLASS_ORDER if c in out.index])


def hex_series_to_int(s: pd.Series) -> pd.Series:
    """Convert a Series of hex strings (e.g. '42C', '0D') to integers.

    Handles a known data-quality defect in the CICIoV2024 hexadecimal SPEED
    file, where ``DATA_6`` is written as a float string (e.g. ``'2.0'``) instead
    of a hex byte (``'02'``). Such values are parsed as decimals; everything else
    is parsed as base-16.
    """
    def _parse(x: str) -> int:
        x = str(x).strip()
        # Malformed float-like cells (e.g. "2.0") -> take the integer value.
        if "." in x:
            return int(float(x))
        # Normal case: interpret as hexadecimal.
        return int(x, 16)

    return s.apply(_parse)


def save_fig(fig, name: str, dpi: int = 150) -> Path:
    """Save a matplotlib figure to ``outputs/figures/<name>.png``."""
    ensure_output_dirs()
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def save_table(df: pd.DataFrame, name: str, index: bool = True) -> Path:
    """Save a dataframe to ``outputs/tables/<name>.csv``."""
    ensure_output_dirs()
    path = TABLES_DIR / f"{name}.csv"
    df.to_csv(path, index=index)
    return path
