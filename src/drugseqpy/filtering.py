"""
filtering.py
------------
Quality-based removal of low-quality samples and lowly expressed genes.

Group-aware gene filtering (macpie strategy): a gene is kept if it meets
the expression threshold in at least one compound group, preventing removal
of drug-induced genes that are absent in DMSO controls.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .core import DrugSeqData


# ---------------------------------------------------------------------------
# filter_samples
# ---------------------------------------------------------------------------

def filter_samples(
    dsd: DrugSeqData,
    min_umi: float = 5000,
    max_umi: float = np.inf,
    min_genes: int = 200,
    max_pct_mito: float = 25.0,
    max_pct_ribo: float = np.inf,
    min_hk_mean: float = 0.0,
    max_hk_cv: float = np.inf,
    max_outlier_score: float = 10.0,
    keep_sample_types: list[str] | None = None,
    verbose: bool = True,
) -> DrugSeqData:
    """
    Remove low-quality samples based on QC metric thresholds.

    Control wells (DMSO, positive_ctrl, negative_ctrl) are exempt from
    library QC thresholds by default to preserve assay integrity metrics.

    Parameters
    ----------
    dsd : DrugSeqData  (must have compute_qc_metrics run first)
    min_umi / max_umi : total UMI bounds
    min_genes : minimum detected genes
    max_pct_mito : maximum % mitochondrial reads
    max_pct_ribo : maximum % ribosomal reads
    min_hk_mean : minimum HK gene mean log-expression
    max_hk_cv : maximum HK gene CV
    max_outlier_score : MAD-based outlier score cutoff
    keep_sample_types : sample types never removed (default DMSO/controls)
    verbose : print summary

    Returns
    -------
    DrugSeqData  (new filtered object)
    """
    if "total_umi" not in dsd.obs.columns:
        raise KeyError("QC metrics missing. Run compute_qc_metrics() first.")

    if keep_sample_types is None:
        keep_sample_types = ["DMSO", "positive_ctrl", "negative_ctrl", "vehicle"]

    obs = dsd.obs
    protected = obs.get("sample_type", pd.Series("treatment", index=obs.index))\
                   .isin(keep_sample_types).values

    def _below(col, val):
        if col not in obs.columns:
            return np.zeros(len(obs), dtype=bool)
        return ~protected & (obs[col].values < val)

    def _above(col, val):
        if col not in obs.columns:
            return np.zeros(len(obs), dtype=bool)
        return ~protected & (obs[col].values > val)

    flags = {
        "low_umi":    _below("total_umi",       min_umi),
        "high_umi":   _above("total_umi",        max_umi),
        "low_genes":  _below("n_genes_det",      min_genes),
        "high_mito":  _above("pct_mito",         max_pct_mito),
        "high_ribo":  _above("pct_ribo",         max_pct_ribo),
        "low_hk":     _below("hk_mean_log",      min_hk_mean),
        "high_hk_cv": _above("hk_cv",            max_hk_cv),
        "outlier":    _above("outlier_score",     max_outlier_score),
    }

    fail = np.zeros(len(obs), dtype=bool)
    for v in flags.values():
        fail |= v
    pass_mask = ~fail

    if verbose:
        n_keep   = pass_mask.sum()
        n_remove = fail.sum()
        print(f"filter_samples: keeping {n_keep} / {len(obs)} samples "
              f"({n_remove} removed).")
        for name, flag in flags.items():
            if flag.sum() > 0:
                print(f"  {name:<18}: {int(flag.sum())} sample(s)")

    if not pass_mask.any():
        raise ValueError("All samples were removed. Relax thresholds.")

    return DrugSeqData(dsd.adata[pass_mask].copy())


# ---------------------------------------------------------------------------
# filter_genes
# ---------------------------------------------------------------------------

def filter_genes(
    dsd: DrugSeqData,
    min_count: int = 1,
    min_samples: int = 2,
    group_aware: bool = True,
    group_col: str = "compound",
    min_cpm: float = 0.0,
    min_cpm_samples: int = 2,
    remove_mito: bool = False,
    remove_ribo: bool = False,
    mito_pattern: str = "^MT-",
    ribo_pattern: str = r"^RP[SL]",
    verbose: bool = True,
) -> DrugSeqData:
    """
    Remove lowly expressed and optionally mitochondrial/ribosomal genes.

    Group-aware strategy (macpie ``filter_genes_by_expression``):
    A gene is retained if it passes the threshold in at least one treatment
    group. This prevents removing drug-induced genes absent in DMSO controls.

    Parameters
    ----------
    dsd : DrugSeqData
    min_count : minimum count to call a gene expressed in a sample
    min_samples : minimum samples per group that must express the gene
    group_aware : use group-wise filter (default True; macpie strategy)
    group_col : metadata column defining groups
    min_cpm : CPM threshold (0 = disabled)
    remove_mito / remove_ribo : drop mito/ribo genes
    """
    mat = dsd.adata.layers["counts"]
    if sp.issparse(mat):
        mat = mat.toarray()
    mat = mat.astype(float)  # (n_samples, n_genes)
    n_genes = mat.shape[1]

    # -- expression filter -------------------------------------------------
    if group_aware and group_col in dsd.obs.columns:
        groups = dsd.obs[group_col].values
        unique_groups = pd.unique(groups[~pd.isna(groups)])
        keep_expr = np.zeros(n_genes, dtype=bool)
        for g in unique_groups:
            idx = groups == g
            g_mat = mat[idx]
            passes = (g_mat >= min_count).sum(axis=0) >= min(min_samples, idx.sum())
            keep_expr |= passes
    else:
        keep_expr = (mat >= min_count).sum(axis=0) >= min_samples

    # -- CPM filter --------------------------------------------------------
    keep_cpm = np.ones(n_genes, dtype=bool)
    if min_cpm > 0:
        lib = mat.sum(axis=1, keepdims=True)
        cpm_mat = mat / (lib + 1e-8) * 1e6
        keep_cpm = (cpm_mat >= min_cpm).sum(axis=0) >= min_cpm_samples

    # -- biotype filters ---------------------------------------------------
    gene_names = pd.Series(dsd.var_names)
    keep_mito = (~gene_names.str.contains(mito_pattern, regex=True)).values \
        if remove_mito else np.ones(n_genes, dtype=bool)
    keep_ribo = (~gene_names.str.contains(ribo_pattern, regex=True)).values \
        if remove_ribo else np.ones(n_genes, dtype=bool)

    keep = keep_expr & keep_cpm & keep_mito & keep_ribo

    if verbose:
        print(f"filter_genes: retaining {keep.sum()} / {n_genes} genes "
              f"({(~keep).sum()} removed).")
        if remove_mito:
            print(f"  Mitochondrial removed: {(~keep_mito).sum()}")
        if remove_ribo:
            print(f"  Ribosomal removed    : {(~keep_ribo).sum()}")
        print(f"  Low-expression removed: {(~(keep_expr & keep_cpm)).sum()}")

    if not keep.any():
        raise ValueError("All genes were removed. Relax filter thresholds.")

    return DrugSeqData(dsd.adata[:, keep].copy())
