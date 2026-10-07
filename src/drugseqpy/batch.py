"""
batch.py
--------
Plate/batch effect correction for Drug-seq data.

Methods
-------
limma_voom : TMM-normalized log2-CPM followed by limma-style batch removal
edgeR      : edgeR-style TMM log2-CPM followed by batch removal
none       : No correction
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .core import DrugSeqData
from .normalization import _limma_voom, _tmm_size_factors


def correct_batch(
    dsd: DrugSeqData,
    batch_col: str = "plate_id",
    method: str = "limma_voom",
    covariate_cols: list[str] | tuple[str, ...] | None = None,
    prior_count: float = 1.0,
    inplace: bool = True,
) -> DrugSeqData | None:
    """Remove batch effects from a normalized expression view.

    The implementation mirrors the role of ``limma::removeBatchEffect``:
    raw counts are normalized, a linear model containing biological
    covariates and the batch factor is fitted gene by gene, and only the
    fitted batch component is subtracted. Raw counts in
    ``layers['counts']`` are never modified.

    Parameters
    ----------
    dsd
        DrugSeqData containing raw counts in ``layers['counts']``.
    batch_col
        Observation column containing plate or batch labels.
    method
        ``'limma_voom'`` (default), ``'edgeR'``, or ``'none'``. The edgeR
        path uses TMM effective library sizes and log2-CPM.
    covariate_cols
        Biological observation columns whose fitted effects must be
        preserved, such as ``['compound']``. The covariates must not be
        perfectly confounded with ``batch_col``.
    prior_count
        Pseudocount used by the log2-CPM transformations.
    inplace
        Modify *dsd* in place when ``True``.

    Notes
    -----
    The corrected matrix is intended for PCA, UMAP, t-SNE, MDS, clustering,
    and expression visualization. Differential expression should continue to
    use ``layers['counts']`` and include the batch column in its design.
    """
    aliases = {"limma": "limma_voom", "edger": "edgeR", "tmm": "edgeR"}
    canonical_method = aliases.get(str(method).casefold(), method)
    valid = ("limma_voom", "edgeR", "none")
    if canonical_method not in valid:
        raise ValueError(f"method must be one of {valid}, got '{method}'")

    if canonical_method == "none":
        print("No batch correction applied.")
        return dsd if not inplace else None

    if batch_col not in dsd.obs.columns:
        raise KeyError(f"batch_col '{batch_col}' not found in obs.")

    covariate_cols = list(covariate_cols or [])
    missing = [column for column in covariate_cols if column not in dsd.obs.columns]
    if missing:
        raise KeyError(f"covariate columns not found in obs: {missing}")
    if batch_col in covariate_cols:
        raise ValueError("batch_col must not also appear in covariate_cols.")

    if not inplace:
        dsd = dsd.copy()

    counts = dsd.adata.layers.get("counts")
    if counts is None:
        raise KeyError("'counts' layer not found; batch correction requires raw counts.")
    if sp.issparse(counts):
        counts = counts.toarray()
    counts = np.asarray(counts, dtype=float)

    print(
        f"Applying batch correction: method={canonical_method}, "
        f"batch_col={batch_col}"
    )
    normalized = _normalize_for_batch(counts, canonical_method, prior_count)
    corrected, design_columns = _remove_batch_effect(
        normalized,
        dsd.obs,
        batch_col=batch_col,
        covariate_cols=covariate_cols,
    )

    dsd.adata.layers["batch_uncorrected"] = normalized.astype(np.float32)
    dsd.adata.layers["batch_corrected"] = corrected.astype(np.float32)
    dsd.adata.X = corrected.astype(np.float32)
    dsd.adata.uns["batch_method"] = canonical_method
    dsd.adata.uns["batch_col"] = batch_col
    dsd.adata.uns["batch_covariates"] = covariate_cols
    dsd.adata.uns["batch_design_columns"] = design_columns
    print("Batch correction complete; corrected expression stored in adata.X.")
    return dsd if not inplace else None


def compute_batch_effect_metrics(
    before: DrugSeqData,
    after: DrugSeqData,
    batch_col: str = "plate_id",
    group_col: str = "compound",
    reduction: str = "X_pca",
    n_dims: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compare batch separation and biological alignment before/after correction.

    Both objects must contain the requested embedding. The returned tidy table
    has one row per diagnostic and ``before``/``after`` value columns.
    """
    before = before if isinstance(before, DrugSeqData) else DrugSeqData(before)
    after = after if isinstance(after, DrugSeqData) else DrugSeqData(after)
    if before.obs_names.tolist() != after.obs_names.tolist():
        raise ValueError("before and after must contain samples in the same order.")

    before_metrics = _embedding_batch_metrics(
        before, batch_col, group_col, reduction, n_dims, random_state
    )
    after_metrics = _embedding_batch_metrics(
        after, batch_col, group_col, reduction, n_dims, random_state
    )
    return pd.DataFrame([
        {
            "metric": metric,
            "before": before_metrics[metric],
            "after": after_metrics[metric],
        }
        for metric in before_metrics
    ])


def _embedding_batch_metrics(
    dsd: DrugSeqData,
    batch_col: str,
    group_col: str,
    reduction: str,
    n_dims: int,
    random_state: int,
) -> dict[str, float]:
    from itertools import combinations

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, silhouette_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    if batch_col not in dsd.obs.columns:
        raise KeyError(f"batch_col '{batch_col}' not found in obs.")
    if group_col not in dsd.obs.columns:
        raise KeyError(f"group_col '{group_col}' not found in obs.")
    if reduction not in dsd.obsm:
        raise KeyError(f"reduction '{reduction}' not found in obsm.")

    labels = dsd.obs[batch_col].astype(str).to_numpy()
    x = np.asarray(dsd.obsm[reduction][:, :n_dims], dtype=float)
    unique_batches, counts = np.unique(labels, return_counts=True)
    if len(unique_batches) < 2:
        raise ValueError("Batch diagnostics require at least two batch levels.")

    silhouette = float(silhouette_score(x, labels))
    grand_mean = x.mean(axis=0)
    total_ss = float(np.square(x - grand_mean).sum())
    between_ss = sum(
        int((labels == batch).sum())
        * float(np.square(x[labels == batch].mean(axis=0) - grand_mean).sum())
        for batch in unique_batches
    )
    plate_variance_fraction = between_ss / total_ss if total_ss > 0 else np.nan

    n_splits = min(5, int(counts.min()))
    if n_splits >= 2:
        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        predicted = cross_val_predict(
            LogisticRegression(max_iter=2000), x, labels, cv=splitter
        )
        classifier_accuracy = float(
            balanced_accuracy_score(labels, predicted)
        )
    else:
        classifier_accuracy = np.nan

    centroids = (
        pd.DataFrame(x, index=dsd.obs.index)
        .assign(
            batch=labels,
            group=dsd.obs[group_col].astype(str).to_numpy(),
        )
        .groupby(["batch", "group"], sort=True)
        .mean()
    )
    distances: list[float] = []
    retrieval: list[bool] = []
    for first, second in combinations(sorted(unique_batches), 2):
        a = centroids.loc[first]
        b = centroids.loc[second]
        shared = sorted(a.index.intersection(b.index))
        for group in shared:
            distances.append(float(np.linalg.norm(a.loc[group] - b.loc[group])))
            candidate_distances = {
                other: float(np.linalg.norm(a.loc[group] - b.loc[other]))
                for other in shared
            }
            retrieval.append(
                min(candidate_distances, key=candidate_distances.get) == group
            )

    return {
        "plate_silhouette": silhouette,
        "plate_variance_fraction": plate_variance_fraction,
        "plate_classifier_balanced_accuracy": classifier_accuracy,
        "median_matched_compound_centroid_distance": (
            float(np.median(distances)) if distances else np.nan
        ),
        "cross_plate_compound_retrieval_accuracy": (
            float(np.mean(retrieval)) if retrieval else np.nan
        ),
    }


def _normalize_for_batch(
    counts: np.ndarray,
    method: str,
    prior_count: float,
) -> np.ndarray:
    """Create the log-expression matrix used for batch-effect fitting."""
    if method == "limma_voom":
        return _limma_voom(counts, prior_count=prior_count)

    factors = _tmm_size_factors(counts)
    effective_library_size = counts.sum(axis=1) * factors
    return np.log2(
        counts / (effective_library_size[:, None] + 1e-8) * 1e6
        + prior_count
    )


def _design_block(
    obs: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Encode numeric and categorical model terms without an intercept."""
    blocks: list[pd.DataFrame] = []
    for column in columns:
        values = obs[column]
        if values.isna().any():
            raise ValueError(f"Column '{column}' contains missing values.")
        if pd.api.types.is_numeric_dtype(values):
            numeric = pd.to_numeric(values, errors="raise").astype(float)
            blocks.append(pd.DataFrame({column: numeric.to_numpy()}, index=obs.index))
        else:
            encoded = pd.get_dummies(
                values.astype(str),
                prefix=column,
                drop_first=True,
                dtype=float,
            )
            encoded.index = obs.index
            if not encoded.empty:
                blocks.append(encoded)
    if not blocks:
        return pd.DataFrame(index=obs.index)
    return pd.concat(blocks, axis=1)


def _remove_batch_effect(
    expression: np.ndarray,
    obs: pd.DataFrame,
    *,
    batch_col: str,
    covariate_cols: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Subtract fitted batch coefficients while retaining covariate effects."""
    if obs[batch_col].isna().any():
        raise ValueError(f"Column '{batch_col}' contains missing values.")

    batch = pd.get_dummies(
        obs[batch_col].astype(str),
        prefix=batch_col,
        drop_first=True,
        dtype=float,
    )
    batch.index = obs.index
    if batch.empty:
        warnings.warn(
            f"Only one level found in '{batch_col}'; returning normalized expression."
        )
        return expression.copy(), ["intercept"]

    preserved = _design_block(obs, covariate_cols)
    intercept = pd.DataFrame({"intercept": np.ones(len(obs))}, index=obs.index)
    design = pd.concat([intercept, preserved, batch], axis=1)
    design_matrix = design.to_numpy(dtype=float)
    if np.linalg.matrix_rank(design_matrix) < design_matrix.shape[1]:
        raise ValueError(
            "The batch/covariate design is rank deficient. Batch is confounded "
            "with a preserved covariate or redundant covariates were supplied."
        )

    coefficients, _, _, _ = np.linalg.lstsq(design_matrix, expression, rcond=None)
    batch_start = 1 + preserved.shape[1]
    batch_effect = batch.to_numpy(dtype=float) @ coefficients[batch_start:, :]
    corrected = expression - batch_effect
    return corrected, design.columns.astype(str).tolist()
