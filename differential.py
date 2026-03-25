"""
differential.py
---------------
Per-compound differential expression.

Methods
-------
pydeseq2   : Python port of DESeq2 (negative binomial GLM)
ols_voom   : OLS on voom log-CPM (limma-voom equivalent)
t_test     : simple t-test on log-CPM (fast screening)
"""

from __future__ import annotations

import warnings
from typing import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.stats as spstats

from .core import DrugSeqData


# ---------------------------------------------------------------------------
# run_de  (single compound)
# ---------------------------------------------------------------------------

def run_de(
    dsd: DrugSeqData,
    compound: str,
    reference: str = "DMSO",
    compound_col: str = "compound",
    within_plate: bool = True,
    method: str = "pydeseq2",
    fdr_threshold: float = 0.05,
    lfc_threshold: float = 0.5,
    min_replicates: int = 2,
) -> pd.DataFrame:
    """
    Run differential expression for one compound vs reference.

    Returns
    -------
    DataFrame: gene, logFC, stat, pvalue, padj, significant
    """
    obs = dsd.obs

    if within_plate and "plate_id" in obs.columns:
        cmpd_plates = obs.loc[obs[compound_col] == compound, "plate_id"].unique()
        idx = obs[compound_col].isin([compound, reference]) & \
              obs["plate_id"].isin(cmpd_plates)
    else:
        idx = obs[compound_col].isin([compound, reference])

    n_treat = (obs.loc[idx, compound_col] == compound).sum()
    if n_treat < min_replicates:
        raise ValueError(
            f"Compound '{compound}' has only {n_treat} replicate(s) "
            f"(need >= {min_replicates})."
        )

    sub = dsd.adata[idx].copy()
    group = (sub.obs[compound_col] == compound).map({True: "treatment",
                                                       False: "control"})

    if method == "pydeseq2":
        return _run_pydeseq2(sub, group, fdr_threshold, lfc_threshold)
    elif method == "ols_voom":
        return _run_ols_voom(sub, group, fdr_threshold, lfc_threshold)
    elif method == "t_test":
        return _run_ttest(sub, group, fdr_threshold, lfc_threshold)
    else:
        raise ValueError(f"method must be pydeseq2, ols_voom, or t_test")


def _run_pydeseq2(adata, group, fdr_threshold, lfc_threshold):
    try:
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats
    except ImportError:
        raise ImportError(
            "pydeseq2 is required. Install with: pip install pydeseq2"
        )
    counts = adata.layers["counts"]
    if sp.issparse(counts):
        counts = counts.toarray()
    counts = counts.astype(int)

    meta = adata.obs[["plate_id"]].copy() if "plate_id" in adata.obs.columns \
           else pd.DataFrame(index=adata.obs_names)
    meta["condition"] = group.values

    dds = DeseqDataSet(
        counts=pd.DataFrame(counts, index=adata.obs_names,
                             columns=adata.var_names),
        metadata=meta,
        design_factors="condition",
        ref_level=["condition", "control"],
        quiet=True,
    )
    dds.deseq2()
    ds = DeseqStats(dds, contrast=["condition", "treatment", "control"],
                    quiet=True)
    ds.summary()
    res = ds.results_df.copy()
    res = res.rename(columns={
        "log2FoldChange": "logFC",
        "pvalue":         "pvalue",
        "padj":           "padj",
        "stat":           "stat",
        "baseMean":       "base_mean",
    })
    res["gene"] = res.index
    res["significant"] = (
        res["padj"].notna() &
        (res["padj"] < fdr_threshold) &
        (res["logFC"].abs() >= lfc_threshold)
    )
    return res[["gene","logFC","base_mean","stat","pvalue","padj","significant"]]\
              .reset_index(drop=True)


def _run_ols_voom(adata, group, fdr_threshold, lfc_threshold):
    """limma-voom equivalent using OLS on voom log-CPM with eBayes shrinkage."""
    from statsmodels.stats.multitest import multipletests

    counts = adata.layers["counts"]
    if sp.issparse(counts):
        counts = counts.toarray()
    counts = counts.astype(float)

    from .normalization import _limma_voom, _tmm_size_factors
    log_cpm = _limma_voom(counts)  # (n_samples, n_genes)

    is_treat = (group == "treatment").values.astype(float)
    X = np.column_stack([np.ones(len(is_treat)), is_treat])  # intercept + group

    # OLS per gene
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ log_cpm         # (2, n_genes)
    fitted = X @ beta
    resid  = log_cpm - fitted
    sigma2 = (resid ** 2).sum(axis=0) / (len(is_treat) - 2)

    # logFC = treatment coefficient
    logfc = beta[1]
    se    = np.sqrt(sigma2 * XtX_inv[1, 1])

    # empirical Bayes shrinkage of variance (simplified Smyth 2004)
    n_genes = log_cpm.shape[1]
    df_resid = len(is_treat) - 2
    prior_df  = 3.0
    prior_var = np.median(sigma2)
    post_var  = (prior_df * prior_var + df_resid * sigma2) / (prior_df + df_resid)
    post_se   = np.sqrt(post_var * XtX_inv[1, 1])

    t_stat = logfc / (post_se + 1e-10)
    df_post = df_resid + prior_df
    pvals   = 2 * spstats.t.sf(np.abs(t_stat), df=df_post)
    _, padj, _, _ = multipletests(pvals, method="fdr_bh")

    genes = np.array(adata.var_names)
    return pd.DataFrame({
        "gene":        genes,
        "logFC":       logfc,
        "base_mean":   log_cpm.mean(axis=0),
        "stat":        t_stat,
        "pvalue":      pvals,
        "padj":        padj,
        "significant": (padj < fdr_threshold) & (np.abs(logfc) >= lfc_threshold),
    })


def _run_ttest(adata, group, fdr_threshold, lfc_threshold):
    from statsmodels.stats.multitest import multipletests

    counts = adata.layers["counts"]
    if sp.issparse(counts):
        counts = counts.toarray()
    log_cpm = np.log1p(counts.astype(float))

    treat_mask   = (group == "treatment").values
    control_mask = ~treat_mask

    logfc  = log_cpm[treat_mask].mean(0) - log_cpm[control_mask].mean(0)
    t_stat, pvals = spstats.ttest_ind(
        log_cpm[treat_mask], log_cpm[control_mask], axis=0, equal_var=False
    )
    pvals = np.nan_to_num(pvals, nan=1.0)
    _, padj, _, _ = multipletests(pvals, method="fdr_bh")

    return pd.DataFrame({
        "gene":        np.array(adata.var_names),
        "logFC":       logfc,
        "base_mean":   log_cpm.mean(axis=0),
        "stat":        t_stat,
        "pvalue":      pvals,
        "padj":        padj,
        "significant": (padj < fdr_threshold) & (np.abs(logfc) >= lfc_threshold),
    })


# ---------------------------------------------------------------------------
# compute_multi_de  (parallelised)
# ---------------------------------------------------------------------------

def compute_multi_de(
    dsd: DrugSeqData,
    compounds: list[str] | None = None,
    reference: str = "DMSO",
    compound_col: str = "compound",
    method: str = "pydeseq2",
    within_plate: bool = True,
    min_replicates: int = 2,
    n_jobs: int = 1,
    fdr_threshold: float = 0.05,
    lfc_threshold: float = 0.5,
    inplace: bool = True,
) -> DrugSeqData | None:
    """
    Parallelised differential expression for all compounds vs reference.

    Results are stored in ``adata.uns['de_results']`` as a dict of DataFrames.
    """
    if not inplace:
        dsd = DrugSeqData(dsd.adata.copy())

    obs = dsd.obs
    if compounds is None:
        all_cpds = obs[compound_col].dropna().unique().tolist()
        compounds = [c for c in all_cpds if c != reference and
                     c not in ("vehicle","media","Media")]

    print(f"compute_multi_de: testing {len(compounds)} compound(s) "
          f"using {n_jobs} worker(s) [{method}].")

    de_results = dsd.adata.uns.get("de_results", {})

    def _run_one(cmpd):
        try:
            return cmpd, run_de(
                dsd, cmpd, reference, compound_col, within_plate,
                method, fdr_threshold, lfc_threshold, min_replicates,
            )
        except Exception as e:
            warnings.warn(f"  Failed for '{cmpd}': {e}")
            return cmpd, None

    if n_jobs == 1:
        results = [_run_one(c) for c in compounds]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = {ex.submit(_run_one, c): c for c in compounds}
            results = [f.result() for f in as_completed(futures)]

    done = 0
    for cmpd, df in results:
        if df is not None:
            de_results[cmpd] = df
            done += 1

    dsd.adata.uns["de_results"] = de_results
    print(f"compute_multi_de complete: {done} / {len(compounds)} compounds processed.")
    return dsd if not inplace else None


# ---------------------------------------------------------------------------
# summarise_de
# ---------------------------------------------------------------------------

def summarise_de(
    dsd: DrugSeqData,
    fdr_threshold: float = 0.05,
    lfc_threshold: float = 0.5,
) -> pd.DataFrame:
    """Return a tidy summary DataFrame: one row per compound."""
    de = dsd.adata.uns.get("de_results", {})
    if not de:
        raise ValueError("de_results is empty. Run compute_multi_de() first.")

    rows = []
    for cmpd, df in de.items():
        sig = df["padj"].notna() & (df["padj"] < fdr_threshold) & \
              (df["logFC"].abs() >= lfc_threshold)
        top_gene = df.loc[df["padj"].idxmin(), "gene"] if sig.any() else None
        rows.append({
            "compound":    cmpd,
            "n_tested":    len(df),
            "n_sig_up":    int((sig & (df["logFC"] > 0)).sum()),
            "n_sig_down":  int((sig & (df["logFC"] < 0)).sum()),
            "n_sig_total": int(sig.sum()),
            "top_gene":    top_gene,
            "min_padj":    float(df["padj"].min()),
        })
    return pd.DataFrame(rows)
