"""
qc.py
-----
Drug-seq quality control:
  - Per-sample library metrics (total UMI, detected genes, %mito, %ribo)
  - Housekeeping gene stability (CV, dropout fraction)
  - MAD-based multivariate outlier scoring
  - Plate-level metrics (Z'-factor, SSMD, DMSO CV/correlation)
  - Group-level dispersion (sd, MAD, z-score, IQR, CV)
  - Replicate intra-class correlation (ICC)
  - Robust control selection (Pearson correlation filter)
  - Zero-inflation test relative to Negative Binomial baseline
  - Metadata validation
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.stats as stats
from scipy.special import gammaln

from .core import DrugSeqData, VALID_SAMPLE_TYPES, REQUIRED_OBS_COLS


# ---------------------------------------------------------------------------
# Housekeeping gene list
# ---------------------------------------------------------------------------

def _load_hk_genes() -> list[str]:
    """Load the bundled Eisenberg & Levanon 2013 housekeeping gene list."""
    hk_path = Path(__file__).parent.parent / "data" / "housekeeping_genes.txt"
    if hk_path.exists():
        return [g.strip() for g in hk_path.read_text().splitlines() if g.strip()]
    # fallback: core stable genes
    return ["ACTB", "GAPDH", "B2M", "HPRT1", "PPIA", "RPL13A", "RPLP0",
            "SDHA", "TBP", "YWHAZ", "HMBS", "UBC", "GUSB", "PGK1", "RPS18"]


# ---------------------------------------------------------------------------
# Gini index
# ---------------------------------------------------------------------------

def _gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.sum() == 0:
        return np.nan
    x = np.sort(x)
    n = len(x)
    return 2 * np.sum(x * np.arange(1, n + 1)) / (n * x.sum()) - (n + 1) / n


# ---------------------------------------------------------------------------
# compute_qc_metrics
# ---------------------------------------------------------------------------

def compute_qc_metrics(
    dsd: DrugSeqData,
    mito_pattern: str = "^MT-",
    ribo_pattern: str = r"^RP[SL]",
    hk_genes: list[str] | None = None,
    min_hk_detected: int = 5,
    inplace: bool = True,
) -> DrugSeqData | None:
    """
    Compute per-sample QC metrics and add them to ``adata.obs``.

    Metrics added
    -------------
    total_umi, log10_total_umi, n_genes_det, genes_per_umi,
    pct_mito, pct_ribo, hk_mean_log, hk_cv, hk_dropout_frac,
    gini_index, outlier_score

    Parameters
    ----------
    dsd : DrugSeqData
    mito_pattern : regex for mitochondrial genes (default ``^MT-``)
    ribo_pattern : regex for ribosomal genes
    hk_genes : housekeeping gene list (None = built-in list)
    min_hk_detected : minimum HK genes required to compute HK metrics
    inplace : modify *dsd* in place (default True)
    """
    if not inplace:
        dsd = DrugSeqData(dsd.adata.copy())

    adata = dsd.adata
    # raw counts matrix: (n_samples, n_genes)
    if "counts" in adata.layers:
        mat = sp.csr_matrix(adata.layers["counts"])
    else:
        mat = sp.csr_matrix(adata.X)

    gene_names = np.array(adata.var_names)

    # -- library metrics ---------------------------------------------------
    total_umi = np.asarray(mat.sum(axis=1)).ravel().astype(float)
    n_genes = np.asarray((mat > 0).sum(axis=1)).ravel().astype(float)

    is_mito = pd.Series(gene_names).str.contains(mito_pattern, regex=True).values
    is_ribo = pd.Series(gene_names).str.contains(ribo_pattern, regex=True).values

    print(f"  Found {is_mito.sum()} mitochondrial and {is_ribo.sum()} ribosomal genes.")

    mito_counts = np.asarray(mat[:, is_mito].sum(axis=1)).ravel()
    ribo_counts = np.asarray(mat[:, is_ribo].sum(axis=1)).ravel()
    pct_mito = 100 * mito_counts / (total_umi + 1e-8)
    pct_ribo = 100 * ribo_counts / (total_umi + 1e-8)

    # -- housekeeping gene stability ---------------------------------------
    if hk_genes is None:
        hk_genes = _load_hk_genes()
    hk_present = [g for g in hk_genes if g in adata.var_names]

    hk_mean_log = np.full(adata.n_obs, np.nan)
    hk_cv = np.full(adata.n_obs, np.nan)
    hk_dropout = np.full(adata.n_obs, np.nan)

    if len(hk_present) >= min_hk_detected:
        print(f"  Computing HK metrics using {len(hk_present)} genes.")
        hk_idx = [adata.var_names.get_loc(g) for g in hk_present]
        hk_mat = mat[:, hk_idx].toarray().astype(float)  # (samples, hk_genes)
        hk_mean_log = np.log1p(hk_mat).mean(axis=1)
        means = hk_mat.mean(axis=1)
        stds  = hk_mat.std(axis=1)
        hk_cv = np.where(means > 1e-8, stds / means, np.nan)
        hk_dropout = (hk_mat == 0).mean(axis=1)
    else:
        warnings.warn(
            f"Only {len(hk_present)} housekeeping genes detected "
            f"(need >= {min_hk_detected}). Skipping HK metrics."
        )

    # -- Gini index --------------------------------------------------------
    mat_dense = mat.toarray()
    gini_idx = np.array([_gini(mat_dense[i]) for i in range(adata.n_obs)])

    # -- MAD-based multivariate outlier score ------------------------------
    qc_for_outlier = pd.DataFrame({
        "log10_umi": np.log10(total_umi + 1),
        "n_genes":   n_genes,
        "pct_mito":  pct_mito,
        "pct_ribo":  pct_ribo,
        "gini":      gini_idx,
    })
    outlier_score = _mad_outlier_score(qc_for_outlier.values)

    # -- store in obs ------------------------------------------------------
    adata.obs["total_umi"]       = total_umi
    adata.obs["log10_total_umi"] = np.log10(total_umi + 1)
    adata.obs["n_genes_det"]     = n_genes
    adata.obs["genes_per_umi"]   = n_genes / (total_umi + 1)
    adata.obs["pct_mito"]        = pct_mito
    adata.obs["pct_ribo"]        = pct_ribo
    adata.obs["hk_mean_log"]     = hk_mean_log
    adata.obs["hk_cv"]           = hk_cv
    adata.obs["hk_dropout_frac"] = hk_dropout
    adata.obs["gini_index"]      = gini_idx
    adata.obs["outlier_score"]   = outlier_score

    # mark mito/ribo in var
    adata.var["is_mito"] = is_mito
    adata.var["is_ribo"] = is_ribo

    print(f"QC metrics computed for {adata.n_obs} samples.")
    return dsd if not inplace else None


def _mad_outlier_score(mat: np.ndarray) -> np.ndarray:
    """MAD-scaled Euclidean distance from the per-column median."""
    ok = ~np.any(np.isnan(mat), axis=0)
    m = mat[:, ok]
    center = np.median(m, axis=0)
    spread = stats.median_abs_deviation(m, axis=0)
    spread[spread == 0] = 1.0
    scaled = (m - center) / spread
    return np.sqrt((scaled ** 2).sum(axis=1))


# ---------------------------------------------------------------------------
# compute_plate_qc
# ---------------------------------------------------------------------------

def compute_plate_qc(
    dsd: DrugSeqData,
    signal_col: str = "log10_total_umi",
    pos_ctrl_label: str = "positive_ctrl",
    neg_ctrl_label: str = "DMSO",
    inplace: bool = True,
) -> DrugSeqData | None:
    """
    Compute plate-level QC statistics and store in ``adata.uns['plate_qc']``.

    Computes per-plate: Z'-factor, SSMD, DMSO inter-well CV, mean pairwise
    DMSO Pearson correlation, and a well-position matrix for spatial plots.
    """
    if not inplace:
        dsd = DrugSeqData(dsd.adata.copy())

    if signal_col not in dsd.obs.columns:
        raise KeyError(
            f"signal_col '{signal_col}' not found in obs. "
            "Run compute_qc_metrics() first."
        )

    obs = dsd.obs
    plates = obs["plate_id"].unique()
    plate_qc: dict[str, dict] = {}

    for pid in plates:
        pm = obs[obs["plate_id"] == pid]
        pos = pm.loc[pm["sample_type"] == pos_ctrl_label, signal_col].dropna().values
        neg = pm.loc[pm["sample_type"] == neg_ctrl_label, signal_col].dropna().values

        zp   = _zprime(pos, neg)
        ssmd = _ssmd(pos, neg)
        dcv  = _coef_var(neg)
        dcor = _dmso_pairwise_cor(dsd, pid, neg_ctrl_label)
        wmat = _build_well_matrix(pm, signal_col)
        flag = _plate_flag(zp, ssmd, dcv)

        plate_qc[pid] = {
            "plate_id":    pid,
            "n_samples":   len(pm),
            "n_dmso":      len(neg),
            "n_pos_ctrl":  len(pos),
            "zprime":      zp,
            "ssmd":        ssmd,
            "dmso_cv":     dcv,
            "dmso_cor":    dcor,
            "well_matrix": wmat,
            "flag":        flag,
        }

    dsd.adata.uns["plate_qc"] = plate_qc
    flags = {pid: d["flag"] for pid, d in plate_qc.items()}
    n_pass = sum(v == "pass" for v in flags.values())
    n_warn = sum(v == "warn" for v in flags.values())
    n_fail = sum(v == "fail" for v in flags.values())
    print(f"Plate QC computed for {len(plates)} plate(s): "
          f"{n_pass} pass, {n_warn} warn, {n_fail} fail.")
    return dsd if not inplace else None


def _zprime(pos, neg):
    if len(pos) < 2 or len(neg) < 2:
        return np.nan
    denom = abs(pos.mean() - neg.mean())
    return np.nan if denom < 1e-10 else 1 - 3 * (pos.std() + neg.std()) / denom


def _ssmd(pos, neg):
    if len(pos) < 1 or len(neg) < 1:
        return np.nan
    denom = np.sqrt(pos.var() + neg.var() + 1e-10)
    return (pos.mean() - neg.mean()) / denom


def _coef_var(x):
    x = x[~np.isnan(x)]
    return np.nan if len(x) < 2 else 100 * x.std() / (x.mean() + 1e-8)


def _dmso_pairwise_cor(dsd, plate_id, neg_ctrl_label):
    mask = (dsd.obs["plate_id"] == plate_id) & \
           (dsd.obs["sample_type"] == neg_ctrl_label)
    if mask.sum() < 2:
        return np.nan
    mat = np.log1p(dsd.adata[mask].layers["counts"].toarray().astype(float))
    cc = np.corrcoef(mat)
    upper = cc[np.triu_indices_from(cc, k=1)]
    return float(np.nanmean(upper))


def _build_well_matrix(plate_meta, value_col):
    import re
    wm = {}
    for _, row in plate_meta.iterrows():
        wid = str(row.get("well_id", ""))
        m = re.match(r"([A-Z]+)(\d+)", wid)
        if not m:
            continue
        r = sum((ord(c) - 64) for c in m.group(1))
        c = int(m.group(2))
        wm[(r, c)] = row[value_col]
    if not wm:
        return None
    max_r = max(k[0] for k in wm)
    max_c = max(k[1] for k in wm)
    mat = np.full((max_r, max_c), np.nan)
    for (r, c), v in wm.items():
        mat[r - 1, c - 1] = v
    return mat


def _plate_flag(zprime, ssmd, dmso_cv):
    if np.isnan(zprime) or np.isnan(ssmd):
        return "warn"
    if zprime > 0.5 and abs(ssmd) > 3 and (np.isnan(dmso_cv) or dmso_cv < 25):
        return "pass"
    if zprime > 0 and abs(ssmd) > 1:
        return "warn"
    return "fail"


# ---------------------------------------------------------------------------
# plate_qc_summary
# ---------------------------------------------------------------------------

def plate_qc_summary(dsd: DrugSeqData) -> pd.DataFrame:
    """Return plate QC results as a tidy DataFrame."""
    pqc = dsd.adata.uns.get("plate_qc", {})
    if not pqc:
        raise ValueError("plate_qc is empty. Run compute_plate_qc() first.")
    rows = []
    for pid, d in pqc.items():
        rows.append({k: v for k, v in d.items() if k != "well_matrix"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# compute_group_qc
# ---------------------------------------------------------------------------

def compute_group_qc(
    dsd: DrugSeqData,
    group_by: str = "compound",
    readout_col: str = "total_umi",
    order_by: str = "median",
) -> pd.DataFrame:
    """
    Compute group-level dispersion statistics: sd, MAD, z-score, IQR, CV.

    This is the Python equivalent of macpie's ``compute_qc_metrics()``
    group-level statistics.

    Returns
    -------
    pd.DataFrame  one row per group
    """
    if readout_col not in dsd.obs.columns:
        raise KeyError(
            f"'{readout_col}' not in obs. Run compute_qc_metrics() first."
        )
    df = dsd.obs[[group_by, readout_col]].dropna()
    groups = df.groupby(group_by)[readout_col]

    stats_df = groups.agg(
        group_median="median",
        sd_value="std",
        IQR=lambda x: x.quantile(0.75) - x.quantile(0.25),
        n_samples="count",
    ).reset_index()
    stats_df["mad_value"] = groups.apply(
        lambda x: stats.median_abs_deviation(x, nan_policy="omit")
    ).values
    stats_df["cv_pct"] = 100 * stats_df["sd_value"] / (
        groups.mean().values + 1e-8
    )

    # robust z-score across groups
    med = stats_df["group_median"].median()
    mad = stats.median_abs_deviation(stats_df["group_median"], nan_policy="omit")
    stats_df["z_score"] = (stats_df["group_median"] - med) / (mad + 1e-8)

    sort_col = {
        "median": "group_median",
        "sd": "sd_value",
        "mad": "mad_value",
        "name": group_by,
    }.get(order_by, "group_median")
    return stats_df.sort_values(sort_col).reset_index(drop=True)


# ---------------------------------------------------------------------------
# compute_replicate_icc
# ---------------------------------------------------------------------------

def compute_replicate_icc(
    dsd: DrugSeqData,
    group_by: str = "compound",
    metric: str = "total_umi",
    min_reps: int = 2,
) -> dict:
    """
    Compute one-way ANOVA intra-class correlation coefficient for replicate QC.

    ICC = (MS_between - MS_within) / (MS_between + (k-1)*MS_within)

    Higher ICC → replicates are more consistent relative to between-group variance.

    Returns
    -------
    dict with keys: icc, anova_table, per_group_cv
    """
    if metric not in dsd.obs.columns:
        raise KeyError(f"'{metric}' not in obs.")
    df = dsd.obs[[group_by, metric]].dropna()
    grp_counts = df.groupby(group_by).size()
    df = df[df[group_by].isin(grp_counts[grp_counts >= min_reps].index)]

    groups = [g[metric].values for _, g in df.groupby(group_by)]
    grand_mean = df[metric].mean()
    k_bar = np.mean([len(g) for g in groups])
    n_groups = len(groups)
    N = sum(len(g) for g in groups)

    ss_b = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_w = sum(((g - g.mean()) ** 2).sum() for g in groups)
    df_b, df_w = n_groups - 1, N - n_groups
    ms_b = ss_b / max(df_b, 1)
    ms_w = ss_w / max(df_w, 1)
    icc = max(0.0, (ms_b - ms_w) / (ms_b + (k_bar - 1) * ms_w + 1e-10))

    per_group_cv = df.groupby(group_by)[metric].agg(
        n="count",
        cv_pct=lambda x: 100 * x.std() / (x.mean() + 1e-8),
    ).reset_index()

    print(f"Replicate ICC ({group_by}, metric={metric}): {icc:.3f}")
    return {
        "icc": icc,
        "anova": {
            "MS_between": ms_b, "MS_within": ms_w,
            "df_between": df_b, "df_within": df_w,
        },
        "per_group_cv": per_group_cv,
    }


# ---------------------------------------------------------------------------
# select_robust_controls
# ---------------------------------------------------------------------------

def select_robust_controls(
    dsd: DrugSeqData,
    neg_ctrl_label: str = "DMSO",
    min_cor: float = 0.90,
    method: str = "pearson",
) -> list[str]:
    """
    Return sample IDs for high-quality control wells (TMMwsp Pearson ≥ min_cor).

    Mirrors macpie's ``select_robust_controls()``.
    """
    if "sample_type" not in dsd.obs.columns:
        raise KeyError("'sample_type' column required in obs.")

    mask = dsd.obs["sample_type"] == neg_ctrl_label
    ctrl_ids = dsd.obs_names[mask].tolist()
    if len(ctrl_ids) < 3:
        warnings.warn("Fewer than 3 control wells; returning all.")
        return ctrl_ids

    mat = np.log1p(
        dsd.adata[mask].layers["counts"].toarray().astype(float)
    )  # (n_ctrl, n_genes)
    cc = np.corrcoef(mat)
    mean_cor = np.array([
        np.mean(np.delete(cc[i], i)) for i in range(len(ctrl_ids))
    ])
    keep = [cid for cid, mc in zip(ctrl_ids, mean_cor) if mc >= min_cor]
    print(
        f"select_robust_controls: retaining {len(keep)} / {len(ctrl_ids)} "
        f"control wells (min_cor={min_cor:.2f})."
    )
    return keep


# ---------------------------------------------------------------------------
# check_zero_inflation
# ---------------------------------------------------------------------------

def check_zero_inflation(
    dsd: DrugSeqData,
    max_genes: int = 2000,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Test genes for zero-inflation relative to a Negative Binomial baseline.

    For each gene, fits mu and dispersion from the count data, computes
    expected zero fraction under NB, then applies a one-sided binomial test.

    Returns
    -------
    pd.DataFrame  gene × (obs_zeros, exp_zeros, zero_ratio, pvalue, padj,
                           zero_inflated)
    """
    from scipy.stats import binom_test
    from statsmodels.stats.multitest import multipletests

    mat = dsd.adata.layers["counts"].toarray().astype(float)
    genes = np.array(dsd.var_names)

    # subsample top expressed genes
    if len(genes) > max_genes:
        means = mat.mean(axis=0)
        idx = np.argsort(means)[::-1][:max_genes]
        mat = mat[:, idx]
        genes = genes[idx]

    n_samp = mat.shape[0]
    mu_hat = mat.mean(axis=0)

    # method-of-moments dispersion estimate
    var_hat = mat.var(axis=0)
    # NB: var = mu + mu^2/r  =>  r = mu^2 / (var - mu)
    with np.errstate(divide="ignore", invalid="ignore"):
        disp = np.where(
            var_hat > mu_hat,
            mu_hat ** 2 / (var_hat - mu_hat),
            np.inf,
        )

    # expected zero fraction: P(X=0) = (r/(r+mu))^r
    with np.errstate(divide="ignore", invalid="ignore"):
        exp_zero_frac = np.where(
            np.isfinite(disp),
            (disp / (disp + mu_hat + 1e-10)) ** disp,
            np.exp(-mu_hat),  # Poisson limit
        )
    exp_zeros = exp_zero_frac * n_samp
    obs_zeros = (mat == 0).sum(axis=0)

    pvals = np.array([
        binom_test(int(obs), n_samp, max(1e-10, ef), alternative="greater")
        for obs, ef in zip(obs_zeros, exp_zero_frac)
    ])
    _, padj, _, _ = multipletests(pvals, method="fdr_bh")

    return pd.DataFrame({
        "gene":          genes,
        "obs_zeros":     obs_zeros.astype(int),
        "exp_zeros":     np.round(exp_zeros, 1),
        "zero_ratio":    obs_zeros / (exp_zeros + 1e-8),
        "pvalue":        pvals,
        "padj":          padj,
        "zero_inflated": padj < alpha,
    })


# ---------------------------------------------------------------------------
# validate_metadata
# ---------------------------------------------------------------------------

def validate_metadata(
    dsd: DrugSeqData,
    required_cols: list[str] | None = None,
    verbose: bool = True,
) -> dict:
    """
    Validate sample metadata for common issues.

    Checks for missing required columns, NA values, unexpected sample_type
    values, and prints a grouped summary by plate.
    """
    if required_cols is None:
        required_cols = REQUIRED_OBS_COLS

    obs = dsd.obs
    issues = []

    for col in required_cols:
        if col not in obs.columns:
            issues.append(f"MISSING column: '{col}'")
        elif obs[col].isna().any():
            n = obs[col].isna().sum()
            issues.append(f"Column '{col}' has {n} NA value(s).")

    if "sample_type" in obs.columns:
        bad = set(obs["sample_type"].dropna()) - VALID_SAMPLE_TYPES
        if bad:
            issues.append(f"Unexpected sample_type values: {bad}")

    if verbose:
        if issues:
            print("Metadata validation issues:")
            for iss in issues:
                print(f"  - {iss}")
        else:
            print("Metadata validation passed.")

        if "plate_id" in obs.columns:
            print("\nSummary by plate:")
            print(obs.groupby("plate_id")["sample_type"].value_counts().to_string())

    return {"ok": len(issues) == 0, "issues": issues}
