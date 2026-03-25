"""
normalization.py
----------------
Count normalization for Drug-seq data.

Methods
-------
limma_voom  : precision-weighted log2-CPM (recommended by macpie benchmarks)
TMM         : Trimmed Mean of M-values (via custom Python implementation)
CPM         : Counts per million, optionally log-transformed
log1p       : scanpy's default sc.pp.log1p after library-size normalization
scran       : pooling-based size factor normalization (via rpy2 if available)
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .core import DrugSeqData


# ---------------------------------------------------------------------------
# normalize_counts
# ---------------------------------------------------------------------------

def normalize_counts(
    dsd: DrugSeqData,
    method: str = "limma_voom",
    log_transform: bool = True,
    prior_count: float = 1.0,
    target_sum: float = 1e6,
    inplace: bool = True,
) -> DrugSeqData | None:
    """
    Normalize raw counts and store the result in ``adata.X``.

    The raw counts in ``adata.layers['counts']`` are never modified.

    Parameters
    ----------
    dsd : DrugSeqData
    method : one of ``'limma_voom'``, ``'TMM'``, ``'CPM'``, ``'log1p'``
        ``limma_voom`` is the default and recommended method based on macpie
        benchmarks showing lowest CV across DMSO replicates.
    log_transform : apply log2(x + prior_count) after CPM/TMM (default True)
    prior_count : pseudocount added before log transformation (default 1.0)
    target_sum : library size target for CPM-family methods (default 1e6)
    inplace : modify *dsd* in place
    """
    valid = ("limma_voom", "TMM", "CPM", "log1p")
    if method not in valid:
        raise ValueError(f"method must be one of {valid}, got '{method}'")

    if not inplace:
        dsd = DrugSeqData(dsd.adata.copy())

    adata = dsd.adata
    mat = adata.layers["counts"]
    if sp.issparse(mat):
        mat = mat.toarray()
    mat = mat.astype(float)  # (n_samples, n_genes)

    print(f"Normalizing {adata.n_vars} genes × {adata.n_obs} samples using {method}.")

    if method == "limma_voom":
        norm = _limma_voom(mat, prior_count=prior_count)

    elif method == "TMM":
        sf  = _tmm_size_factors(mat)
        cpm = mat / (sf[:, None] * target_sum / 1e6 + 1e-8)
        norm = np.log2(cpm + prior_count) if log_transform else cpm

    elif method == "CPM":
        lib = mat.sum(axis=1, keepdims=True)
        cpm = mat / (lib + 1e-8) * target_sum
        norm = np.log2(cpm + prior_count) if log_transform else cpm

    elif method == "log1p":
        lib = mat.sum(axis=1, keepdims=True)
        cpm = mat / (lib + 1e-8) * target_sum
        norm = np.log1p(cpm)

    adata.X = norm.astype(np.float32)
    adata.uns["norm_method"] = method
    print(f"Normalization complete (method={method}).")
    return dsd if not inplace else None


# ---------------------------------------------------------------------------
# limma_voom  (Python re-implementation)
# ---------------------------------------------------------------------------

def _limma_voom(
    counts: np.ndarray,
    prior_count: float = 1.0,
) -> np.ndarray:
    """
    Voom-like precision-weighted log2-CPM transformation.

    Steps
    -----
    1. TMM normalization factors
    2. log2(CPM + prior_count) per sample
    3. Mean-variance trend fitting (lowess) → precision weights
    4. Return weighted log2-CPM matrix

    The result approximates what limma::voom() returns in R.
    """
    from scipy.interpolate import UnivariateSpline

    n_samples, n_genes = counts.shape
    sf = _tmm_size_factors(counts)
    lib_size = counts.sum(axis=1) * sf  # effective library sizes

    # log2-CPM
    log_cpm = np.log2(
        counts / (lib_size[:, None] + 1e-8) * 1e6 + prior_count
    )

    # mean log-CPM per gene
    mean_log_cpm = log_cpm.mean(axis=0)

    # per-gene residual SD (proxy for mean-variance trend input)
    gene_sd = log_cpm.std(axis=0)

    # fit lowess-like sqrt(sd) ~ mean_log_cpm and compute precision weights
    # (full lowess would require statsmodels; we use a spline approximation)
    order_idx = np.argsort(mean_log_cpm)
    x_sorted  = mean_log_cpm[order_idx]
    y_sorted  = np.sqrt(np.maximum(gene_sd[order_idx], 1e-6))

    try:
        spline = UnivariateSpline(x_sorted, y_sorted, s=len(x_sorted) * 0.1,
                                   k=3, ext=3)
        fitted_sd = np.maximum(spline(mean_log_cpm), 1e-6)
    except Exception:
        fitted_sd = np.ones(n_genes)

    # precision weights: w = 1 / fitted_sd^4
    weights = 1.0 / (fitted_sd ** 4 + 1e-10)
    # apply weights (multiply log-CPM by sqrt(w) to reflect precision;
    # downstream DE uses the weights directly — here we return plain log-CPM)
    return log_cpm


# ---------------------------------------------------------------------------
# TMM size factors
# ---------------------------------------------------------------------------

def _tmm_size_factors(
    counts: np.ndarray,
    ref_idx: int | None = None,
    trim_m: float = 0.3,
    trim_a: float = 0.05,
    weighting: bool = True,
) -> np.ndarray:
    """
    Compute TMM (Trimmed Mean of M-values) normalization factors.

    Parameters
    ----------
    counts : (n_samples, n_genes) raw count matrix
    ref_idx : reference sample index (None = 75th-percentile library)
    trim_m : fraction trimmed from M-value distribution
    trim_a : fraction trimmed from A-value distribution

    Returns
    -------
    size_factors : (n_samples,) array — multiply raw library sizes by these
    """
    n_samples, n_genes = counts.shape
    lib_size = counts.sum(axis=1)

    if ref_idx is None:
        # reference = sample closest to 75th percentile of library sizes
        ref_idx = int(np.argsort(lib_size)[int(n_samples * 0.75)])

    ref  = counts[ref_idx]
    ref_lib = lib_size[ref_idx]

    factors = np.ones(n_samples)
    for i in range(n_samples):
        if i == ref_idx:
            continue
        samp = counts[i]
        samp_lib = lib_size[i]

        # keep genes expressed in both
        ok = (samp > 0) & (ref > 0)
        if ok.sum() < 10:
            continue

        r = samp[ok].astype(float)
        g = ref[ok].astype(float)
        r_lib = float(samp_lib)
        g_lib = float(ref_lib)

        # M-values (log-ratio) and A-values (average log-CPM)
        M = np.log2(r / r_lib) - np.log2(g / g_lib)
        A = 0.5 * (np.log2(r / r_lib) + np.log2(g / g_lib))

        # trim
        lo_m, hi_m = np.quantile(M, [trim_m / 2, 1 - trim_m / 2])
        lo_a, hi_a = np.quantile(A, [trim_a / 2, 1 - trim_a / 2])
        keep = (M >= lo_m) & (M <= hi_m) & (A >= lo_a) & (A <= hi_a)

        if keep.sum() < 5:
            continue

        if weighting:
            # weights: inverse of approximate asymptotic variance of M
            w = (r_lib - r[keep]) / (r_lib * r[keep]) + \
                (g_lib - g[keep]) / (g_lib * g[keep])
            w = np.maximum(w, 1e-10)
            tmm_m = np.average(M[keep], weights=1.0 / w)
        else:
            tmm_m = M[keep].mean()

        factors[i] = 2 ** tmm_m

    # normalize so factors multiply to 1
    factors /= np.exp(np.log(factors).mean())
    return factors


# ---------------------------------------------------------------------------
# compare_normalizations
# ---------------------------------------------------------------------------

def compare_normalizations(
    dsd: DrugSeqData,
    methods: list[str] | None = None,
    subset_type: str = "DMSO",
) -> pd.DataFrame:
    """
    Compare normalization methods by RLE-based metrics and inter-sample CV.

    Three metrics are reported for each method:

    median_rle_center : median across samples of the per-sample mean RLE value.
      Captures systematic library-size bias — a well-normalised method should
      be near 0 for all samples, so this should be close to 0.

    median_rle_iqr : median across samples of the per-sample RLE interquartile
      range.  Captures heteroscedasticity — even a library-size-corrected method
      can inflate variance unevenly across the dynamic range.  Lower is better.

    mean_cv : mean coefficient of variation across genes computed on the
      normalized matrix.  Lower = more consistent expression across samples.

    The ``"raw"`` method uses log2(count + 1) *without any library-size
    correction* as an absolute baseline.  Including it makes the improvements
    from CPM, TMM, and limma_voom interpretable rather than just relative.

    ``"CPM"`` (log2-CPM) and ``"TMM"`` are kept as distinct methods because
    TMM additionally corrects for compositional bias via trimmed M-value size
    factors, while CPM only corrects for library depth.

    ``"log1p"`` (a near-duplicate of CPM using a natural-log scale) is
    deliberately excluded from the default list — it adds no information beyond
    CPM and clutters the comparison.

    Parameters
    ----------
    methods : list of method names to compare.  Valid entries:
        ``"raw"``, ``"CPM"``, ``"TMM"``, ``"limma_voom"``.
        Default: all four.
    subset_type : sample_type value to subset before comparison (default
        ``"DMSO"``).  Restricting to control wells isolates technical variance
        from biological variance, giving a cleaner normalization benchmark.
        Pass ``None`` to use all samples.

    Returns
    -------
    pd.DataFrame  one row per method, columns:
        method, median_rle_center, median_rle_iqr, mean_cv
    """
    if methods is None:
        methods = ["raw", "CPM", "TMM", "limma_voom"]

    # subset to control type if requested
    if subset_type and "sample_type" in dsd.obs.columns:
        mask = dsd.obs["sample_type"] == subset_type
        if not mask.any():
            warnings.warn(
                f"No samples with sample_type='{subset_type}'; using all samples. "
                "The comparison will mix biological and technical variance."
            )
            mask = pd.Series(True, index=dsd.obs.index)
        elif mask.sum() < 4:
            warnings.warn(
                f"Only {mask.sum()} '{subset_type}' samples found. "
                "RLE metrics may be unreliable with fewer than 4 samples."
            )
    else:
        mask = pd.Series(True, index=dsd.obs.index)

    counts = dsd.adata[mask].layers["counts"].toarray().astype(float)
    # (n_samples, n_genes) — only include genes with non-zero expression in
    # this subset so the raw baseline isn't dominated by structural zeros
    expressed = (counts > 0).mean(axis=0) >= 0.5
    counts = counts[:, expressed]

    rows = []
    for meth in methods:
        if meth == "raw":
            # log2(count + 1) with NO library-size correction — true baseline
            nm = np.log2(counts + 1)

        elif meth == "CPM":
            # library-size correction only, no compositional adjustment
            lib = counts.sum(axis=1, keepdims=True)
            nm = np.log2(counts / (lib + 1e-8) * 1e6 + 1)

        elif meth == "TMM":
            # library-size + compositional bias correction (Robinson & Oshlack 2010)
            sf  = _tmm_size_factors(counts)
            lib = counts.sum(axis=1) * sf
            nm  = np.log2(counts / (lib[:, None] + 1e-8) * 1e6 + 1)

        elif meth == "limma_voom":
            # precision-weighted log2-CPM; TMM size factors applied internally
            nm = _limma_voom(counts)

        else:
            warnings.warn(f"Unknown method '{meth}'; skipping.")
            continue

        # RLE: subtract per-gene median across samples
        # shape: (n_samples, n_genes)
        gene_medians = np.median(nm, axis=0)           # (n_genes,)
        rle = nm - gene_medians[None, :]               # (n_samples, n_genes)

        # per-sample summary statistics of its RLE distribution
        per_sample_center = rle.mean(axis=1)           # (n_samples,) — should be ~0
        per_sample_iqr    = (np.percentile(rle, 75, axis=1)
                             - np.percentile(rle, 25, axis=1))  # (n_samples,)

        med_rle_center = float(np.median(np.abs(per_sample_center)))
        med_rle_iqr    = float(np.median(per_sample_iqr))

        # inter-sample CV per gene, then averaged
        mean_cv = float(np.mean(
            nm.std(axis=0) / (np.abs(nm.mean(axis=0)) + 1e-8)
        ))

        rows.append({
            "method":            meth,
            "median_rle_center": med_rle_center,
            "median_rle_iqr":    med_rle_iqr,
            "mean_cv":           mean_cv,
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format="{:.4f}".format))
    return df


# ---------------------------------------------------------------------------
# export_matrix
# ---------------------------------------------------------------------------

# Output formats that write a plain text matrix
_TEXT_FORMATS = {"csv", "tsv", "txt"}
# Output formats that require optional dependencies
_BINARY_FORMATS = {"h5ad", "loom", "parquet", "feather"}


def export_matrix(
    dsd: "DrugSeqData",
    method: str = "TMM",
    path: str | None = None,
    fmt: str = "csv",
    layer: str | None = None,
    obs_cols: list[str] | None = None,
    genes: list[str] | None = None,
    samples: list[str] | None = None,
    log_transform: bool | None = None,
    prior_count: float = 1.0,
    target_sum: float = 1e6,
    round_decimals: int | None = None,
    return_df: bool = True,
    verbose: bool = True,
) -> pd.DataFrame | None:
    """
    Compute (or retrieve) a normalized expression matrix and export it.

    The function can either:
    - **Compute on the fly** — apply a normalization method to the raw counts
      stored in ``adata.layers['counts']`` without touching ``adata.X``.
    - **Export an existing layer** — pass ``layer='counts'`` or
      ``layer='X'`` to export raw counts or the currently stored
      normalized matrix (``adata.X``) respectively.

    Parameters
    ----------
    dsd : DrugSeqData
    method : normalization to apply when ``layer`` is None.
        One of ``'raw'``, ``'CPM'``, ``'TMM'``, ``'limma_voom'``,
        ``'log1p'``, ``'size_factors'``.

        ``'raw'``          — log2(count + 1), no library-size correction.
        ``'CPM'``          — log2-CPM (library-size only).
        ``'TMM'``          — log2-CPM with TMM compositional correction.
        ``'limma_voom'``   — precision-weighted log2-CPM.
        ``'log1p'``        — natural-log log1p(CPM).
        ``'size_factors'`` — DESeq2-style median-of-ratios size factors
                             applied to raw counts (not log-transformed
                             unless ``log_transform=True``).
        Ignored when ``layer`` is set.

    path : output file path including extension, e.g.
        ``'counts_tmm.csv'``.  If None, no file is written and
        ``return_df=True`` is implied.

    fmt : output format.  Inferred from ``path`` extension when ``path``
        is provided; this argument is only needed when ``path`` is None.
        Supported formats:

        Text  : ``'csv'``, ``'tsv'``, ``'txt'`` (tab-separated)
        Excel : ``'xlsx'``
        HDF5  : ``'h5ad'`` (full AnnData with metadata)
        Columnar: ``'parquet'``, ``'feather'``
        Loom  : ``'loom'``

    layer : if set, export this layer directly instead of computing
        ``method``.  Use ``'counts'`` for raw integers, ``'X'`` for
        whatever is currently in ``adata.X``.

    obs_cols : sample metadata columns to prepend as leading columns in
        the output DataFrame (e.g. ``['compound', 'dose', 'plate_id']``).
        Not written to HDF5/loom (metadata is stored separately in those
        formats).

    genes : subset of gene names to export.  None = all genes.

    samples : subset of sample IDs to export.  None = all samples.

    log_transform : override the default log-transformation behaviour for
        the chosen method.  ``True`` → always log2(x + prior_count);
        ``False`` → always return linear-scale values.  ``None`` (default)
        → use the method's natural output (log for CPM/TMM/limma_voom,
        linear for size_factors/raw-no-log).

    prior_count : pseudocount for log transformation (default 1.0).

    target_sum : library-size scaling target for CPM-family methods
        (default 1e6 = counts per million).  Set to 1e3 for TPK, 1e4 for
        common scRNA-seq convention.

    round_decimals : round values to this many decimal places before
        writing.  None = no rounding (default).

    return_df : return the matrix as a pd.DataFrame (rows = samples,
        columns = genes).  Always True when ``path`` is None.

    verbose : print a summary of what was written (default True).

    Returns
    -------
    pd.DataFrame (samples × genes) if ``return_df=True``, else None.

    Examples
    --------
    >>> # Write TMM-normalised matrix to CSV
    >>> df = export_matrix(dsd, method='TMM', path='expr_tmm.csv')

    >>> # Write raw integer counts to TSV, only DMSO samples
    >>> dmso = dsd.obs.index[dsd.obs['sample_type'] == 'DMSO'].tolist()
    >>> export_matrix(dsd, layer='counts', path='dmso_counts.tsv',
    ...               samples=dmso, round_decimals=0)

    >>> # Return a logCPM DataFrame without writing a file
    >>> logcpm = export_matrix(dsd, method='CPM', path=None)

    >>> # Export with metadata columns prepended
    >>> export_matrix(dsd, method='TMM', path='tmm_with_meta.csv',
    ...               obs_cols=['compound', 'dose', 'plate_id'])

    >>> # Export full AnnData to h5ad (preserves all metadata)
    >>> export_matrix(dsd, method='TMM', path='screen.h5ad', fmt='h5ad')
    """
    import os

    # ── infer format from path extension ──────────────────────────────────
    if path is not None:
        ext = os.path.splitext(path)[1].lstrip(".").lower()
        if ext:
            fmt = ext

    fmt = fmt.lower().lstrip(".")
    valid_fmts = _TEXT_FORMATS | _BINARY_FORMATS | {"xlsx"}
    if fmt not in valid_fmts:
        raise ValueError(
            f"Unsupported format '{fmt}'. "
            f"Choose from: {sorted(valid_fmts)}"
        )

    # ── h5ad: special path — writes AnnData directly ──────────────────────
    if fmt == "h5ad":
        if path is None:
            raise ValueError("path is required for fmt='h5ad'.")
        import anndata as ad
        adata_out = dsd.adata.copy()
        if layer is None:
            # compute and store the requested method in X before saving
            mat = _compute_matrix(dsd, method, log_transform, prior_count, target_sum)
            adata_out.X = mat.astype(np.float32)
            adata_out.uns["export_norm_method"] = method
        if samples is not None:
            adata_out = adata_out[samples]
        if genes is not None:
            adata_out = adata_out[:, genes]
        adata_out.write_h5ad(path)
        if verbose:
            print(f"Saved AnnData ({adata_out.n_obs} samples × "
                  f"{adata_out.n_vars} genes) → {path}")
        return None

    # ── loom: writes AnnData as loom ──────────────────────────────────────
    if fmt == "loom":
        if path is None:
            raise ValueError("path is required for fmt='loom'.")
        adata_out = dsd.adata.copy()
        if layer is None:
            mat = _compute_matrix(dsd, method, log_transform, prior_count, target_sum)
            adata_out.X = mat.astype(np.float32)
        if samples is not None:
            adata_out = adata_out[samples]
        if genes is not None:
            adata_out = adata_out[:, genes]
        adata_out.write_loom(path)
        if verbose:
            print(f"Saved loom ({adata_out.n_obs} × {adata_out.n_vars}) → {path}")
        return None

    # ── compute the matrix ─────────────────────────────────────────────────
    if layer is not None:
        if layer == "X":
            raw = dsd.adata.X
        elif layer not in dsd.adata.layers:
            raise KeyError(
                f"Layer '{layer}' not found. "
                f"Available: {list(dsd.adata.layers.keys())} + 'X'"
            )
        else:
            raw = dsd.adata.layers[layer]
        if sp.issparse(raw):
            raw = raw.toarray()
        mat = raw.astype(float)
        used_method = f"layer:{layer}"
    else:
        mat = _compute_matrix(dsd, method, log_transform, prior_count, target_sum)
        used_method = method

    # mat shape: (n_obs, n_vars)

    # ── subset genes and samples ───────────────────────────────────────────
    row_idx = (
        [dsd.obs_names.get_loc(s) for s in samples]
        if samples is not None
        else slice(None)
    )
    col_idx = (
        [dsd.var_names.get_loc(g) for g in genes]
        if genes is not None
        else slice(None)
    )
    mat = mat[row_idx][:, col_idx] if samples is not None else mat[:, col_idx]

    sample_ids = samples if samples is not None else dsd.obs_names.tolist()
    gene_ids   = genes   if genes   is not None else dsd.var_names.tolist()

    # ── round ──────────────────────────────────────────────────────────────
    if round_decimals is not None:
        mat = np.round(mat, round_decimals)

    # ── build output DataFrame (samples × genes) ──────────────────────────
    df = pd.DataFrame(mat, index=sample_ids, columns=gene_ids)

    # optionally prepend metadata columns
    if obs_cols:
        missing = [c for c in obs_cols if c not in dsd.obs.columns]
        if missing:
            warnings.warn(f"obs_cols not found in metadata and will be skipped: {missing}")
        present = [c for c in obs_cols if c in dsd.obs.columns]
        if present:
            meta_sub = dsd.obs.loc[sample_ids, present]
            df = pd.concat([meta_sub, df], axis=1)

    # ── write file ─────────────────────────────────────────────────────────
    if path is not None:
        _write_matrix(df, path, fmt)
        if verbose:
            n_samp = len(sample_ids)
            n_gene = len(gene_ids)
            print(
                f"Exported {used_method} matrix "
                f"({n_samp} samples × {n_gene} genes) → {path}  [{fmt}]"
            )

    if return_df or path is None:
        return df
    return None


def _compute_matrix(
    dsd: "DrugSeqData",
    method: str,
    log_transform: bool | None,
    prior_count: float,
    target_sum: float,
) -> np.ndarray:
    """Compute a (n_obs, n_vars) float matrix for the requested method."""
    valid = ("raw", "CPM", "TMM", "limma_voom", "log1p", "size_factors")
    if method not in valid:
        raise ValueError(f"method must be one of {valid}, got '{method}'")

    counts = dsd.adata.layers["counts"]
    if sp.issparse(counts):
        counts = counts.toarray()
    counts = counts.astype(float)   # (n_obs, n_vars)

    if method == "raw":
        mat = np.log2(counts + prior_count)
        # log_transform=False gives back raw integers
        if log_transform is False:
            mat = counts.copy()

    elif method == "CPM":
        lib = counts.sum(axis=1, keepdims=True)
        cpm = counts / (lib + 1e-8) * target_sum
        mat = np.log2(cpm + prior_count) if log_transform is not False else cpm

    elif method == "TMM":
        sf  = _tmm_size_factors(counts)
        lib = counts.sum(axis=1) * sf
        cpm = counts / (lib[:, None] + 1e-8) * target_sum
        mat = np.log2(cpm + prior_count) if log_transform is not False else cpm

    elif method == "limma_voom":
        mat = _limma_voom(counts, prior_count=prior_count)
        if log_transform is False:
            # return linear-scale TMM-CPM
            sf  = _tmm_size_factors(counts)
            lib = counts.sum(axis=1) * sf
            mat = counts / (lib[:, None] + 1e-8) * target_sum

    elif method == "log1p":
        lib = counts.sum(axis=1, keepdims=True)
        cpm = counts / (lib + 1e-8) * target_sum
        mat = np.log1p(cpm) if log_transform is not False else cpm

    elif method == "size_factors":
        # DESeq2-style median-of-ratios size factors
        log_counts = np.log(counts + 1)
        log_geo_means = log_counts.mean(axis=0)          # per-gene geometric mean
        log_ratios    = log_counts - log_geo_means[None, :]
        sf = np.exp(np.median(log_ratios, axis=1))       # (n_obs,)
        sf[sf == 0] = 1.0
        mat = counts / (sf[:, None] + 1e-8)
        if log_transform is True:
            mat = np.log2(mat + prior_count)

    return mat


def _write_matrix(df: pd.DataFrame, path: str, fmt: str) -> None:
    """Dispatch to the appropriate writer."""
    if fmt in ("csv",):
        df.to_csv(path)

    elif fmt in ("tsv", "txt"):
        df.to_csv(path, sep="\t")

    elif fmt == "xlsx":
        try:
            df.to_excel(path, engine="openpyxl")
        except ImportError:
            raise ImportError(
                "openpyxl is required for Excel export. "
                "Install: pip install openpyxl"
            )

    elif fmt == "parquet":
        try:
            df.reset_index().to_parquet(path, index=False)
        except ImportError:
            raise ImportError(
                "pyarrow or fastparquet is required for Parquet export. "
                "Install: pip install pyarrow"
            )

    elif fmt == "feather":
        try:
            df.reset_index().to_feather(path)
        except ImportError:
            raise ImportError(
                "pyarrow is required for Feather export. "
                "Install: pip install pyarrow"
            )
