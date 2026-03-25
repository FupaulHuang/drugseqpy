"""
utils.py
--------
Shared internal utilities for drugseqpy.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def require_pkg(pkg: str, install_hint: str | None = None) -> Any:
    """Import a package, raising a helpful ImportError if absent."""
    try:
        return importlib.import_module(pkg)
    except ImportError:
        hint = install_hint or f"pip install {pkg}"
        raise ImportError(f"Package '{pkg}' is required. Install: {hint}")


def parse_well_id(well_id: str) -> tuple[int, int]:
    """
    Parse a well ID like 'A01' or 'H12' into (row, col) 1-indexed integers.
    Returns (-1, -1) if parsing fails.
    """
    import re
    m = re.match(r"([A-Z]+)(\d+)", well_id.strip().upper())
    if not m:
        return -1, -1
    row = sum((ord(c) - 64) for c in m.group(1))
    col = int(m.group(2))
    return row, col


def well_ids_to_grid(obs: pd.DataFrame,
                     well_col: str = "well_id",
                     value_col: str | None = None) -> np.ndarray:
    """
    Build a plate layout grid from a subset of sample metadata.

    Returns an (n_rows, n_cols) float array with NaN for empty wells.
    """
    rows, cols, vals = [], [], []
    for _, row in obs.iterrows():
        r, c = parse_well_id(str(row.get(well_col, "")))
        if r < 0:
            continue
        rows.append(r)
        cols.append(c)
        vals.append(float(row[value_col]) if value_col else 1.0)

    if not rows:
        return np.full((1, 1), np.nan)

    mat = np.full((max(rows), max(cols)), np.nan)
    for r, c, v in zip(rows, cols, vals):
        mat[r - 1, c - 1] = v
    return mat


def plate384_layout() -> dict:
    """Standard 384-well plate dimensions."""
    import string
    return {
        "rows":    list(string.ascii_uppercase[:16]),
        "cols":    [f"{i:02d}" for i in range(1, 25)],
        "n_wells": 384,
    }


def plate96_layout() -> dict:
    """Standard 96-well plate dimensions."""
    import string
    return {
        "rows":    list(string.ascii_uppercase[:8]),
        "cols":    [f"{i:02d}" for i in range(1, 13)],
        "n_wells": 96,
    }


def make_dummy_screen(
    n_genes: int = 200,
    n_plates: int = 2,
    n_dmso_per_plate: int = 8,
    n_compounds: int = 4,
    n_reps: int = 4,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate a synthetic Drug-seq count matrix and metadata for testing.

    Returns
    -------
    counts : pd.DataFrame  (genes × samples)
    obs    : pd.DataFrame  sample metadata
    """
    import string
    rng = np.random.default_rng(seed)

    n_per_plate = n_dmso_per_plate + n_compounds * n_reps
    n_samples   = n_plates * n_per_plate

    gene_names = (
        [f"MT-G{i}" for i in range(1, 7)] +
        [f"RPL{i}"   for i in range(1, 7)] +
        [f"RPS{i}"   for i in range(1, 7)] +
        [f"Gene{i:04d}" for i in range(1, n_genes - 17)]
    )[:n_genes]

    # baseline NB counts
    mu   = rng.lognormal(mean=5, sigma=2, size=n_genes)
    mat  = rng.negative_binomial(n=5, p=5 / (5 + mu), size=(n_samples, n_genes))

    # add compound-specific signal
    rows_list, meta_rows = [], []
    for p in range(1, n_plates + 1):
        for _ in range(n_dmso_per_plate):
            meta_rows.append({
                "plate_id": f"Plate{p:02d}",
                "compound": "DMSO",
                "dose": 0.0, "dose_unit": "uM",
                "sample_type": "DMSO",
            })
        for ci, cmpd in enumerate([f"Cmpd{j+1:02d}" for j in range(n_compounds)]):
            for ri in range(n_reps):
                meta_rows.append({
                    "plate_id": f"Plate{p:02d}",
                    "compound": cmpd,
                    "dose": 10.0, "dose_unit": "uM",
                    "sample_type": "treatment",
                })
                # add signal in distinct gene block
                g_start = 18 + ci * 10
                g_end   = g_start + 10
                idx     = len(meta_rows) - 1
                mat[idx, g_start:g_end] = (
                    mat[idx, g_start:g_end] * rng.integers(4, 9)
                )

    meta = pd.DataFrame(meta_rows)
    meta.index = [f"S{i+1:04d}" for i in range(n_samples)]

    # add well IDs
    n_rows = int(np.ceil(n_per_plate / 12))
    row_letters = string.ascii_uppercase[:n_rows]
    well_ids = [
        f"{row_letters[i % len(row_letters)]}{i % 12 + 1:02d}"
        for i in range(n_per_plate)
    ]
    meta["well_id"] = well_ids * n_plates

    counts_df = pd.DataFrame(
        mat.T,
        index=gene_names,
        columns=meta.index,
    )
    return counts_df, meta
