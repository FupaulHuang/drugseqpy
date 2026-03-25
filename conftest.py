"""
conftest.py — shared pytest fixtures for drugseqpy tests.
"""

import numpy as np
import pandas as pd
import pytest

from drugseqpy import create_drugseq_object, DrugSeqData
from drugseqpy.utils import make_dummy_screen


@pytest.fixture(scope="module")
def mini_screen() -> DrugSeqData:
    """Minimal 2-plate, 4-compound screen (n_genes=150, fast)."""
    counts, obs = make_dummy_screen(
        n_genes=150, n_plates=2,
        n_dmso_per_plate=6, n_compounds=4, n_reps=4, seed=0,
    )
    return create_drugseq_object(counts, obs)


@pytest.fixture(scope="module")
def dsd_qc(mini_screen) -> DrugSeqData:
    """Screen with QC metrics pre-computed."""
    from drugseqpy import compute_qc_metrics, compute_plate_qc
    dsd = mini_screen
    compute_qc_metrics(dsd, mito_pattern="^MT-", ribo_pattern=r"^RP[SL]", inplace=True)
    compute_plate_qc(dsd, signal_col="log10_total_umi", inplace=True)
    return dsd


@pytest.fixture(scope="module")
def dsd_normed(dsd_qc) -> DrugSeqData:
    """Screen with limma_voom normalization applied."""
    from drugseqpy import normalize_counts, filter_genes
    dsd = dsd_qc
    normalize_counts(dsd, method="limma_voom", inplace=True)
    return filter_genes(dsd, min_count=1, min_samples=2, group_aware=True, verbose=False)


@pytest.fixture(scope="module")
def dsd_de(dsd_normed) -> DrugSeqData:
    """Screen with per-compound DE results."""
    from drugseqpy import compute_multi_de
    compute_multi_de(
        dsd_normed,
        reference="DMSO",
        method="ols_voom",
        within_plate=True,
        fdr_threshold=0.05,
        lfc_threshold=0.5,
        n_jobs=1,
        inplace=True,
    )
    return dsd_normed
