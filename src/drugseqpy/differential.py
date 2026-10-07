"""
differential.py
---------------
Per-contrast differential expression.

Methods
-------
limma_voom : Python limma-voom equivalent (OLS on voom log-CPM + eBayes)
deseq      : PyDESeq2 (negative binomial GLM, optional batch factors)
edgeR      : R edgeR via rpy2 (optional dependency)
t_test     : simple t-test on log-CPM (fast screening, legacy)

Aliases for backward compatibility: ``ols_voom`` -> ``limma_voom``,
``pydeseq2`` -> ``deseq``.

General contrast results are stored in ``adata.uns['comparison_results']``;
default compound-vs-reference runs additionally populate the legacy
``adata.uns['de_results']`` mapping.
"""

from __future__ import annotations

import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.stats as spstats

from .core import DrugSeqData


# ---------------------------------------------------------------------------
# Method name resolution
# ---------------------------------------------------------------------------

METHOD_ALIASES = {
    "ols_voom": "limma_voom",
    "limma-voom": "limma_voom",
    "pydeseq2": "deseq",
    "deseq2": "deseq",
    "edger": "edgeR",
    "edge_r": "edgeR",
    "edge-r": "edgeR",
    "ttest": "t_test",
    "t-test": "t_test",
}
VALID_METHODS = ("limma_voom", "edgeR", "deseq", "t_test")
# Common negative-control labels available to callers through the optional
# ``exclude_levels`` argument. They are not removed implicitly because a
# grouping column without ``sample_type`` may intentionally include them.
DEFAULT_EXCLUDED_LEVELS = ("vehicle", "media", "water", "blank")


def _level_list(value) -> list:
    """Return a scalar-or-sequence group selector as a list of levels."""
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=str)
    if isinstance(value, (list, tuple, pd.Index, pd.Series, np.ndarray)):
        return list(value)
    return [value]


def _level_label(value) -> str:
    """Stable display/storage label for a scalar or pooled selector."""
    return "|".join(str(level) for level in _level_list(value))


def _resolve_method(method: str) -> str:
    """Canonicalize a DE method name through the alias table."""
    raw = str(method).strip()
    canonical = METHOD_ALIASES.get(raw, METHOD_ALIASES.get(raw.lower(), raw))
    if str(canonical).lower() == "edger":
        canonical = "edgeR"
    elif str(canonical).lower() in {"deseq", "deseq2"}:
        canonical = "deseq"
    elif str(canonical).lower() in {"limma_voom", "limma-voom"}:
        canonical = "limma_voom"
    if canonical not in VALID_METHODS:
        raise ValueError(
            f"method must be one of {VALID_METHODS} (aliases {METHOD_ALIASES}), "
            f"got '{method}'"
        )
    return canonical


# ---------------------------------------------------------------------------
# Contrast normalization
# ---------------------------------------------------------------------------

def normalize_contrasts(
    comparisons: dict | list | None = None,
    obs: pd.DataFrame | None = None,
    group_col: str = "compound",
    reference: str = "DMSO",
    exclude_levels: list | tuple | set | None = None,
) -> list[dict]:
    """
    Normalize user-supplied contrast definitions into a canonical list.

    Each contrast is a dict with keys ``name``, ``group_col``, ``case`` and
    ``control``.

    Parameters
    ----------
    comparisons : None, dict or sequence
        - ``None``: one contrast per non-control level of *group_col* with
          *control* = *reference* ("all samples vs one control").
        - dict: ``{case: control, ...}`` or ``{name: (case, control), ...}``.
        - sequence of 2-tuples ``(case, control)`` or of dicts with keys
          ``case``/``control`` (plus optional ``name``/``group_col``). Either
          selector may be a sequence of levels; ``control='rest'`` expands to
          all levels other than the case selector.
    obs : sample metadata used to validate that levels exist.
    group_col : default metadata column defining the groups.
    reference : default control level used when ``comparisons is None``.
    exclude_levels : optional additional levels omitted from auto-generated
        contrasts. Explicit ``comparisons`` are never filtered by this list.

    Returns
    -------
    list[dict]
    """
    if obs is None:
        raise ValueError("obs is required to normalize contrasts.")
    if group_col not in obs.columns:
        raise KeyError(
            f"group column '{group_col}' not found in obs. "
            f"Available columns: {list(obs.columns)}"
        )

    if comparisons is None:
        observed_values = obs[group_col].dropna().unique().tolist()
        reference_value = next(
            (value for value in observed_values
             if str(value).lower() == str(reference).lower()),
            None,
        )
        observed = {str(value) for value in observed_values}
        if reference_value is None:
            raise ValueError(
                f"reference '{reference}' not found in obs['{group_col}']. "
                f"Available levels: {sorted(observed)}"
            )
        levels = [lv for lv in observed_values
                  if str(lv).lower() != str(reference_value).lower()]
        if exclude_levels:
            excluded = {str(lv).lower() for lv in exclude_levels}
            levels = [lv for lv in levels
                      if str(lv).lower() not in excluded]
        # Do not infer exclusions from sample_type.  A caller may intentionally
        # compare a media/water/vehicle level, so only the configured reference
        # and explicit ``exclude_levels`` are omitted here.
        return [
            {"name": str(c), "group_col": group_col,
             "case": c, "control": reference_value}
            for c in levels
        ]

    # Accept one named contrast directly as a mapping, in addition to a list
    # of contrast mappings and the compact {case: control} form.
    if isinstance(comparisons, dict) and {
        "case", "control"
    }.issubset(comparisons):
        comparisons = [comparisons]

    # A bare pair ("case", "control") is a convenient one-contrast form.
    # Do not reinterpret a pair of nested selectors as a bare pair.
    if (isinstance(comparisons, (tuple, list)) and len(comparisons) == 2 and
            all(not isinstance(value, (tuple, list, dict))
                for value in comparisons)):
        comparisons = [tuple(comparisons)]

    out: list[dict] = []
    items = comparisons.items() if isinstance(comparisons, dict) else comparisons
    observed_levels = set(obs[group_col].dropna().astype(str).unique())
    for item in items:
        if isinstance(comparisons, dict):
            key, value = item
            # ``{"A vs B": ("A", "B")}`` is the named-pair form.  When
            # the mapping key is itself an observed level, a sequence value
            # is instead a pooled control selector (``{"A": ["B", "C"]}``).
            if (isinstance(value, (tuple, list)) and len(value) == 2 and
                    str(key) not in observed_levels):
                case, control = value
                name = str(key)
            else:
                case, control = key, value
                name = str(case)
            gcol = group_col
        elif isinstance(item, dict):
            if "case" not in item or "control" not in item:
                raise ValueError(
                    f"Contrast dict must contain 'case' and 'control': {item}"
                )
            case, control = item["case"], item["control"]
            name = str(item.get("name", f"{case} vs {control}"))
            gcol = item.get("group_col", group_col)
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            case, control = item
            name = f"{case} vs {control}"
            gcol = group_col
        else:
            raise ValueError(
                f"Unsupported contrast definition: {item!r}. Use "
                "'case=control', (case, control), or a dict with "
                "'case'/'control'."
            )
        out.append({"name": name, "group_col": gcol,
                    "case": case, "control": control})

    # validate levels exist in metadata
    for c in out:
        if c["group_col"] not in obs.columns:
            raise KeyError(
                f"group column '{c['group_col']}' not found in obs. "
                f"Available columns: {list(obs.columns)}"
            )
        levels = set(obs[c["group_col"]].dropna().astype(str).unique())
        level_lookup = {
            str(value).lower(): value
            for value in obs[c["group_col"]].dropna().unique()
        }
        for field in ("case", "control"):
            values = _level_list(c[field])
            if len(values) == 1 and str(values[0]).lower() == "rest":
                continue
            missing = [str(value) for value in values
                       if str(value).lower() not in level_lookup]
            if missing:
                raise ValueError(
                    f"{field} '{missing[0]}' not found in obs['{c['group_col']}']. "
                    f"Available levels: {sorted(levels)}"
                )
    return out


# ---------------------------------------------------------------------------
# run_comparison  (one case vs one control)
# ---------------------------------------------------------------------------

def run_comparison(
    dsd: DrugSeqData,
    case: str,
    control: str,
    group_col: str = "compound",
    method: str = "limma_voom",
    batch_cols: list[str] | None = None,
    within_plate: bool = True,
    min_replicates: int = 2,
    fdr_threshold: float = 0.05,
    lfc_threshold: float = 0.5,
    contrast_name: str | None = None,
) -> pd.DataFrame:
    """
    Run differential expression for one case group versus one control group.

    Parameters
    ----------
    dsd : DrugSeqData
    case, control : group levels in ``obs[group_col]`` to compare
        (direction of logFC: case - control). Each may be a sequence to pool
        levels; ``control='rest'`` means every other level.
    group_col : metadata column defining the groups.
    method : one of ``'limma_voom'``, ``'edgeR'``, ``'deseq'``, ``'t_test'``
        (aliases ``'ols_voom'``/``'pydeseq2'`` accepted).
    batch_cols : optional metadata columns added as covariates
        (e.g. ``['plate_id']``).
    within_plate : restrict the comparison to plates shared by case and
        control when ``plate_id`` is present.
    min_replicates : minimum samples required in *each* group.

    Returns
    -------
    DataFrame with columns gene, logFC, base_mean, stat, pvalue, padj,
    significant, direction, contrast, case, control, method.
    """
    if not isinstance(dsd, DrugSeqData):
        dsd = DrugSeqData(dsd)
    if min_replicates < 1:
        raise ValueError("min_replicates must be >= 1.")
    if isinstance(batch_cols, str):
        batch_cols = [batch_cols]
    method = _resolve_method(method)
    obs = dsd.obs
    if group_col not in obs.columns:
        raise KeyError(
            f"group column '{group_col}' not found in obs. "
            f"Available columns: {list(obs.columns)}"
        )
    if "counts" not in dsd.adata.layers:
        raise KeyError(
            "'counts' layer not found. Add raw integer counts to "
            "adata.layers['counts'] before running differential expression."
        )
    valid_batch_cols = []
    for col in batch_cols or []:
        if col in obs.columns:
            valid_batch_cols.append(col)
        else:
            warnings.warn(f"batch column '{col}' not in obs; skipping.")
    batch_cols = valid_batch_cols
    vals = obs[group_col].astype(str)
    observed_lookup = {
        str(value).lower(): str(value)
        for value in obs[group_col].dropna().unique()
    }

    def _canonical_levels(selector):
        levels = []
        for value in _level_list(selector):
            key = str(value).lower()
            levels.append(observed_lookup.get(key, str(value)))
        return levels

    case_levels = _canonical_levels(case)
    control_levels = _canonical_levels(control)
    if len(control_levels) == 1 and control_levels[0].lower() == "rest":
        control_levels = sorted(
            level for level in vals.dropna().unique()
            if level not in set(case_levels)
        )
    overlap = sorted(set(case_levels) & set(control_levels))
    if overlap:
        raise ValueError(
            f"case and control selectors overlap: {overlap}"
        )
    case_mask = vals.isin(case_levels)
    control_mask = vals.isin(control_levels)

    if not case_mask.any():
        raise ValueError(
            f"case '{_level_label(case)}' not found in obs['{group_col}']. "
            f"Available levels: {sorted(observed_lookup.values())}"
        )
    if not control_mask.any():
        raise ValueError(
            f"control '{_level_label(control)}' not found in obs['{group_col}']. "
            f"Available levels: {sorted(observed_lookup.values())}"
        )

    idx = case_mask | control_mask
    if within_plate and "plate_id" in obs.columns:
        case_plates = obs.loc[case_mask, "plate_id"].unique()
        idx &= obs["plate_id"].isin(case_plates)
        if not (idx & control_mask).any():
            raise ValueError(
                f"no '{control}' samples share a plate with '{case}' "
                f"(plates: {list(case_plates)}). Use within_plate=False "
                "to pool across plates."
            )

    n_case = int((idx & case_mask).sum())
    n_ctrl = int((idx & control_mask).sum())
    if n_case < min_replicates:
        raise ValueError(
            f"case '{case}' has only {n_case} replicate(s) "
            f"(need >= {min_replicates})."
        )
    if n_ctrl < min_replicates:
        raise ValueError(
            f"control '{control}' has only {n_ctrl} replicate(s) "
            f"(need >= {min_replicates})."
        )

    sub = dsd.adata[idx].copy()
    group = pd.Series(
        np.where(vals[idx].isin(case_levels), "treatment", "control"),
        index=sub.obs_names,
    )

    if method == "limma_voom":
        res = _run_limma_voom(sub, group, batch_cols, fdr_threshold, lfc_threshold)
    elif method == "deseq":
        res = _run_deseq2(sub, group, batch_cols, fdr_threshold, lfc_threshold)
    elif method == "edgeR":
        res = _run_edger(sub, group, batch_cols, fdr_threshold, lfc_threshold)
    else:  # t_test
        if batch_cols:
            warnings.warn("t_test ignores batch_cols; use limma_voom or deseq.")
        res = _run_ttest(sub, group, fdr_threshold, lfc_threshold)

    res["contrast"] = contrast_name or _level_label(case)
    res["case"] = _level_label(case_levels)
    res["control"] = _level_label(control_levels) if not (
        len(_level_list(control)) == 1 and
        str(_level_list(control)[0]).lower() == "rest"
    ) else _level_label(control_levels)
    res["method"] = method
    sig = res["significant"].fillna(False).astype(bool)
    res["direction"] = np.select(
        [sig & (res["logFC"] > 0), sig & (res["logFC"] < 0)],
        ["up", "down"], default="ns",
    )
    return res[["gene", "logFC", "base_mean", "stat", "pvalue", "padj",
                "significant", "direction", "contrast", "case", "control",
                "method"]]


# ---------------------------------------------------------------------------
# DE backends
# ---------------------------------------------------------------------------

def _build_design(sub_obs: pd.DataFrame, group: pd.Series,
                  batch_cols: list[str] | None) -> np.ndarray:
    """Design matrix: intercept + batch dummies + treatment indicator."""
    cols = [np.ones((len(group), 1))]
    if batch_cols:
        for col in batch_cols:
            if col not in sub_obs.columns:
                warnings.warn(f"batch column '{col}' not in obs; skipping.")
                continue
            dummies = pd.get_dummies(sub_obs[col].astype(str),
                                     prefix=col, drop_first=True)
            cols.append(dummies.astype(float).values)
    cols.append((group == "treatment").astype(float).values[:, None])
    return np.hstack(cols)


def _run_limma_voom(adata, group, batch_cols, fdr_threshold, lfc_threshold):
    """limma-voom equivalent: OLS on voom log-CPM with eBayes shrinkage."""
    from statsmodels.stats.multitest import multipletests
    from .normalization import _limma_voom

    counts = adata.layers["counts"]
    if sp.issparse(counts):
        counts = counts.toarray()
    counts = counts.astype(float)

    log_cpm = _limma_voom(counts)  # (n_samples, n_genes)
    X = _build_design(adata.obs, group, batch_cols)

    # OLS per gene
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ log_cpm
    fitted = X @ beta
    resid = log_cpm - fitted
    df_resid = max(len(group) - X.shape[1], 1)
    sigma2 = (resid ** 2).sum(axis=0) / df_resid

    # logFC = treatment coefficient (last column)
    treat_idx = X.shape[1] - 1
    logfc = beta[treat_idx]
    se = np.sqrt(np.maximum(sigma2, 1e-10) * XtX_inv[treat_idx, treat_idx])

    # empirical Bayes shrinkage of variance (simplified Smyth 2004)
    prior_df = 3.0
    prior_var = np.median(sigma2)
    post_var = (prior_df * prior_var + df_resid * sigma2) / (prior_df + df_resid)
    post_se = np.sqrt(post_var * XtX_inv[treat_idx, treat_idx])

    t_stat = logfc / (post_se + 1e-10)
    df_post = df_resid + prior_df
    pvals = 2 * spstats.t.sf(np.abs(t_stat), df=df_post)
    _, padj, _, _ = multipletests(pvals, method="fdr_bh")

    return pd.DataFrame({
        "gene": np.array(adata.var_names),
        "logFC": logfc,
        "base_mean": log_cpm.mean(axis=0),
        "stat": t_stat,
        "pvalue": pvals,
        "padj": padj,
        "significant": (padj < fdr_threshold) & (np.abs(logfc) >= lfc_threshold),
    })


def _run_deseq2(adata, group, batch_cols, fdr_threshold, lfc_threshold):
    """DESeq2 equivalent via PyDESeq2, with optional batch covariates."""
    try:
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats
    except ImportError:
        raise ImportError(
            "pydeseq2 is required for method='deseq'. "
            "Install with: pip install pydeseq2"
        )

    counts = adata.layers["counts"]
    if sp.issparse(counts):
        counts = counts.toarray()
    counts = counts.astype(int)

    meta_cols = list(batch_cols or [])
    meta = adata.obs[meta_cols].copy() if meta_cols \
        else pd.DataFrame(index=adata.obs_names)
    for col in meta_cols:
        meta[col] = meta[col].astype(str)
    meta["condition"] = group.values

    counts_df = pd.DataFrame(
        counts, index=adata.obs_names, columns=adata.var_names,
    )
    formula = "~ " + " + ".join(meta_cols + ["condition"])
    try:
        # Formulaic designs are the current PyDESeq2 API and allow arbitrary
        # batch factors without relying on the deprecated design_factors path.
        dds = DeseqDataSet(
            counts=counts_df, metadata=meta, design=formula, quiet=True,
        )
    except (TypeError, ValueError):
        # Compatibility with older PyDESeq2 releases.
        dds = DeseqDataSet(
            counts=counts_df, metadata=meta,
            design_factors=meta_cols + ["condition"],
            ref_level=["condition", "control"], quiet=True,
        )
    dds.deseq2()
    ds = DeseqStats(dds, contrast=["condition", "treatment", "control"],
                    quiet=True)
    ds.summary()
    res = ds.results_df.copy()
    res = res.rename(columns={
        "log2FoldChange": "logFC",
        "pvalue": "pvalue",
        "padj": "padj",
        "stat": "stat",
        "baseMean": "base_mean",
    })
    res["gene"] = res.index
    res["significant"] = (
        res["padj"].notna() &
        (res["padj"] < fdr_threshold) &
        (res["logFC"].abs() >= lfc_threshold)
    )
    return res[["gene", "logFC", "base_mean", "stat", "pvalue", "padj",
                "significant"]].reset_index(drop=True)


def _run_edger(adata, group, batch_cols, fdr_threshold, lfc_threshold):
    """edgeR exact/QLF test via an optional R bridge (rpy2 + edgeR)."""
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr
        pandas2ri.activate()
    except ImportError:
        raise ImportError(
            "method='edgeR' requires R with the edgeR package and rpy2. "
            "Install with: conda install -c conda-forge rpy2 r-edger"
        )
    try:
        edger = importr("edgeR")
        base = importr("base")
        stats_r = importr("stats")
    except Exception as e:  # package missing in R
        raise ImportError(
            f"R package 'edgeR' not found ({e}). "
            "Install with: conda install -c conda-forge r-edger"
        )

    counts = adata.layers["counts"]
    if sp.issparse(counts):
        counts = counts.toarray()
    counts = counts.astype(int).T  # (n_genes, n_samples)

    group_r = ro.StrVector(["treatment" if g == "treatment" else "control"
                            for g in group.values])
    dge = edger.DGEList(counts=counts, group=group_r)
    dge = edger.calcNormFactors(dge, method="TMM")

    try:
        if batch_cols:
            meta = adata.obs[list(batch_cols)].astype(str).copy()
            meta["group"] = group_r
            design = stats_r.model_matrix(
                ro.Formula("~ 0 + " + " + ".join(list(batch_cols)) + " + group"),
                data=pandas2ri.py2rpy(meta),
            )
            dge = edger.estimateDisp(dge, design)
            fit = edger.glmQLFit(dge, design)
            tst = edger.glmQLFTest(fit, coef=base.ncol(design))
        else:
            dge = edger.estimateDisp(dge)
            tst = edger.exactTest(dge)
    except Exception as e:
        raise RuntimeError(f"edgeR analysis failed in R: {e}")

    top = edger.topTags(tst, n=base.nrow(dge), sort_by="none")
    get_col = ro.r("function(t, col) t$table[[col]]")
    logfc = np.array(base.as_numeric(get_col(top, "logFC")), dtype=float)
    logcpm = np.array(base.as_numeric(get_col(top, "logCPM")), dtype=float)
    pvals = np.array(base.as_numeric(get_col(top, "PValue")), dtype=float)
    pvals = np.nan_to_num(pvals, nan=1.0)

    from statsmodels.stats.multitest import multipletests
    _, padj, _, _ = multipletests(pvals, method="fdr_bh")

    return pd.DataFrame({
        "gene": np.array(adata.var_names.astype(str)),
        "logFC": logfc,
        "base_mean": 2.0 ** logcpm - 1.0,
        "stat": np.nan,
        "pvalue": pvals,
        "padj": padj,
        "significant": (padj < fdr_threshold) &
                       (np.abs(logfc) >= lfc_threshold),
    })


def _run_ttest(adata, group, fdr_threshold, lfc_threshold):
    from statsmodels.stats.multitest import multipletests

    counts = adata.layers["counts"]
    if sp.issparse(counts):
        counts = counts.toarray()
    log_cpm = np.log1p(counts.astype(float))

    treat_mask = (group == "treatment").values
    control_mask = ~treat_mask

    logfc = log_cpm[treat_mask].mean(0) - log_cpm[control_mask].mean(0)
    t_stat, pvals = spstats.ttest_ind(
        log_cpm[treat_mask], log_cpm[control_mask], axis=0, equal_var=False
    )
    pvals = np.nan_to_num(pvals, nan=1.0)
    _, padj, _, _ = multipletests(pvals, method="fdr_bh")

    return pd.DataFrame({
        "gene": np.array(adata.var_names),
        "logFC": logfc,
        "base_mean": log_cpm.mean(axis=0),
        "stat": t_stat,
        "pvalue": pvals,
        "padj": padj,
        "significant": (padj < fdr_threshold) & (np.abs(logfc) >= lfc_threshold),
    })


# Backward-compatible backend aliases
_run_ols_voom = _run_limma_voom
_run_pydeseq2 = _run_deseq2


# ---------------------------------------------------------------------------
# run_de  (legacy single-compound wrapper)
# ---------------------------------------------------------------------------

def run_de(
    dsd: DrugSeqData,
    compound: str,
    reference: str = "DMSO",
    compound_col: str = "compound",
    within_plate: bool = True,
    method: str = "limma_voom",
    fdr_threshold: float = 0.05,
    lfc_threshold: float = 0.5,
    min_replicates: int = 2,
) -> pd.DataFrame:
    """
    Run differential expression for one compound vs reference (legacy API).

    Delegates to :func:`run_comparison`.

    Returns
    -------
    DataFrame: gene, logFC, base_mean, stat, pvalue, padj, significant,
    direction, contrast, case, control, method
    """
    return run_comparison(
        dsd, case=compound, control=reference, group_col=compound_col,
        method=method, within_plate=within_plate,
        min_replicates=min_replicates, fdr_threshold=fdr_threshold,
        lfc_threshold=lfc_threshold, contrast_name=str(compound),
    )


# ---------------------------------------------------------------------------
# compute_multi_de  (parallelised, contrast-aware)
# ---------------------------------------------------------------------------

def _run_comparison_task(dsd, case, control, group_col, method, batch_cols,
                         within_plate, min_replicates, fdr_threshold,
                         lfc_threshold, name):
    """Picklable task wrapper used by ProcessPoolExecutor."""
    return name, run_comparison(
        dsd, case=case, control=control, group_col=group_col, method=method,
        batch_cols=batch_cols, within_plate=within_plate,
        min_replicates=min_replicates, fdr_threshold=fdr_threshold,
        lfc_threshold=lfc_threshold, contrast_name=name,
    )


def compute_multi_de(
    dsd: DrugSeqData,
    compounds: list[str] | None = None,
    reference: str = "DMSO",
    compound_col: str = "compound",
    method: str = "limma_voom",
    within_plate: bool = True,
    min_replicates: int = 2,
    n_jobs: int = 1,
    fdr_threshold: float = 0.05,
    lfc_threshold: float = 0.5,
    inplace: bool = True,
    *,
    methods: list[str] | None = None,
    comparisons: dict | list | None = None,
    group_col: str | None = None,
    batch_cols: list[str] | None = None,
    exclude_levels: list[str] | tuple[str, ...] | None = None,
) -> DrugSeqData | None:
    """
    Differential expression across multiple contrasts, optionally with
    several methods.

    Contrast results are stored in ``adata.uns['comparison_results']`` keyed
    by contrast name (or ``"<contrast>::<method>"`` when several methods are
    requested).  When ``comparisons`` is None (the default
    compound-vs-reference mode), the legacy ``adata.uns['de_results']``
    mapping is also populated from the first method so existing downstream
    helpers keep working.
    """
    if not isinstance(dsd, DrugSeqData):
        dsd = DrugSeqData(dsd)
    if n_jobs < 1:
        raise ValueError("n_jobs must be >= 1.")
    if min_replicates < 1:
        raise ValueError("min_replicates must be >= 1.")
    if "counts" not in dsd.adata.layers:
        raise KeyError(
            "'counts' layer not found. Add raw integer counts to "
            "adata.layers['counts'] before running differential expression."
        )
    if isinstance(batch_cols, str):
        batch_cols = [batch_cols]
    if not inplace:
        dsd = DrugSeqData(dsd.adata.copy())

    gcol = group_col or compound_col
    if isinstance(methods, str):
        method_list = ["limma_voom", "edgeR", "deseq"] \
            if methods.strip().lower() == "all" else [methods]
    else:
        method_list = list(methods) if methods else [method]
    method_list = [_resolve_method(m) for m in method_list]

    if comparisons is not None and compounds is not None:
        raise ValueError("Pass either 'comparisons' or 'compounds', not both.")

    contrasts = normalize_contrasts(
        comparisons, dsd.obs, gcol, reference,
        exclude_levels=exclude_levels,
    )
    if isinstance(compounds, str):
        compounds = [compounds]
    if compounds is not None:
        wanted = {str(c) for c in compounds}
        contrasts = [c for c in contrasts if c["name"] in wanted]

    if not contrasts:
        raise ValueError(
            f"No contrasts generated from obs['{gcol}']. Pass explicit "
            "'comparisons' or check group levels."
        )

    print(f"compute_multi_de: {len(contrasts)} contrast(s) × "
          f"{len(method_list)} method(s) [{', '.join(method_list)}], "
          f"{n_jobs} worker(s).")

    tasks = [
        (c, m, c["name"] if len(method_list) == 1 else f"{c['name']}::{m}")
        for c in contrasts for m in method_list
    ]

    failures = {}
    # A run represents one complete comparison configuration. Rebuild the
    # mapping so reruns cannot silently retain stale methods or contrasts.
    comparison_results = {}

    def _run_one(task):
        contrast, meth, key = task
        try:
            return key, _run_comparison_task(
                dsd, contrast["case"], contrast["control"],
                contrast["group_col"], meth, batch_cols, within_plate,
                min_replicates, fdr_threshold, lfc_threshold, contrast["name"],
            )[1]
        except Exception as e:
            failures[key] = str(e)
            warnings.warn(
                f"  Failed for '{contrast['name']}' [{meth}]: {e}"
            )
            return key, None

    if n_jobs == 1:
        results = [_run_one(t) for t in tasks]
    else:
        # Submit the module-level task wrapper directly.  A nested callable
        # cannot be pickled by ProcessPoolExecutor (notably under the
        # ``spawn`` start method), which previously made ``n_jobs > 1`` fail
        # before any comparison was executed.
        futures = {}
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            for contrast, meth, key in tasks:
                future = ex.submit(
                    _run_comparison_task,
                    dsd, contrast["case"], contrast["control"],
                    contrast["group_col"], meth, batch_cols, within_plate,
                    min_replicates, fdr_threshold, lfc_threshold,
                    contrast["name"],
                )
                futures[future] = (key, meth, contrast["name"])
            results = []
            for future in as_completed(futures):
                key, meth, contrast_name = futures[future]
                try:
                    _, df = future.result()
                except Exception as e:
                    failures[key] = str(e)
                    warnings.warn(
                        f"  Failed for '{contrast_name}' [{meth}]: {e}"
                    )
                    df = None
                results.append((key, df))
        # Restore the caller's contrast/method order after asynchronous
        # completion so dictionary insertion order and downstream reports are
        # reproducible across runs.
        result_by_key = dict(results)
        results = [(key, result_by_key[key])
                   for _, _, key in tasks]

    done = 0
    for key, df in results:
        if df is not None:
            comparison_results[key] = df
            done += 1
    dsd.adata.uns["comparison_results"] = comparison_results
    dsd.adata.uns["comparison_failures"] = failures

    # legacy de_results mirror for default compound-vs-reference runs
    if comparisons is None:
        # Rebuild the compatibility mirror from this run so removed contrasts
        # or methods cannot survive a rerun as stale legacy results.
        de_results = {}
        for c in contrasts:
            candidate_keys = ([c["name"]] if len(method_list) == 1 else
                              [f"{c['name']}::{method}" for method in method_list])
            for key in candidate_keys:
                df = comparison_results.get(key)
                if df is not None:
                    de_results[c["name"]] = df
                    break
        dsd.adata.uns["de_results"] = de_results
    else:
        # ``de_results`` is intentionally a compatibility mirror for the
        # default compound-vs-reference mode only; custom contrasts should
        # not leave stale legacy tables that downstream code could mistake
        # for the current analysis.
        dsd.adata.uns["de_results"] = {}

    print(f"compute_multi_de complete: {done} / {len(tasks)} runs succeeded.")
    return dsd if not inplace else None


# ---------------------------------------------------------------------------
# summarise_de  /  summarise_comparison
# ---------------------------------------------------------------------------

def summarise_de(
    dsd: DrugSeqData,
    fdr_threshold: float = 0.05,
    lfc_threshold: float = 0.5,
) -> pd.DataFrame:
    """Return a tidy summary DataFrame: one row per compound (legacy)."""
    if not isinstance(dsd, DrugSeqData):
        dsd = DrugSeqData(dsd)
    de = dsd.adata.uns.get("de_results", {})
    if not de:
        raise ValueError("de_results is empty. Run compute_multi_de() first.")

    rows = []
    for cmpd, df in de.items():
        sig = df["padj"].notna() & (df["padj"] < fdr_threshold) & \
              (df["logFC"].abs() >= lfc_threshold)
        top_gene = (df.loc[df["padj"].idxmin(), "gene"]
                    if sig.any() else None)
        rows.append({
            "compound": cmpd,
            "n_tested": len(df),
            "n_sig_up": int((sig & (df["logFC"] > 0)).sum()),
            "n_sig_down": int((sig & (df["logFC"] < 0)).sum()),
            "n_sig_total": int(sig.sum()),
            "top_gene": top_gene,
            "min_padj": float(df["padj"].min()),
        })
    return pd.DataFrame(rows)


def summarise_comparison(
    dsd: DrugSeqData,
    fdr_threshold: float = 0.05,
    lfc_threshold: float = 0.5,
) -> pd.DataFrame:
    """Return a tidy summary of ``uns['comparison_results']``.

    One row per (contrast, method) with up/down significant gene counts.
    Falls back to ``de_results`` when no comparison results are stored.
    """
    if not isinstance(dsd, DrugSeqData):
        dsd = DrugSeqData(dsd)
    tables = dsd.adata.uns.get("comparison_results") or \
        dsd.adata.uns.get("de_results", {})
    if not tables:
        raise ValueError(
            "comparison_results is empty. Run compute_multi_de() first."
        )

    rows = []
    for key, df in tables.items():
        sig = df["padj"].notna() & (df["padj"] < fdr_threshold) & \
              (df["logFC"].abs() >= lfc_threshold)
        top_gene = (df.loc[df["padj"].idxmin(), "gene"]
                    if sig.any() else None)
        first = df.iloc[0] if not df.empty else pd.Series(dtype=object)
        rows.append({
            "contrast": key,
            "case": first.get("case") if "case" in df.columns else None,
            "control": first.get("control") if "control" in df.columns else None,
            "method": first.get("method") if "method" in df.columns else None,
            "n_tested": len(df),
            "n_sig_up": int((sig & (df["logFC"] > 0)).sum()),
            "n_sig_down": int((sig & (df["logFC"] < 0)).sum()),
            "n_sig_total": int(sig.sum()),
            "top_gene": top_gene,
            "min_padj": float(df["padj"].min()),
        })
    return pd.DataFrame(rows)
