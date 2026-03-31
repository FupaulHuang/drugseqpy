"""
batch.py
--------
Plate/batch effect correction for Drug-seq data.

Methods
-------
harmony         : Harmony integration (harmonypy)
combat          : ComBat parametric empirical Bayes (pycombat / inmoose)
dmso_regression : Regress out DMSO PC variation per plate (Drug-seq native)
none            : No correction
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .core import DrugSeqData


# ---------------------------------------------------------------------------
# correct_batch
# ---------------------------------------------------------------------------

def correct_batch(
    dsd: DrugSeqData,
    batch_col: str = "plate_id",
    method: str = "harmony",
    covariate_cols: list[str] | None = None,
    n_pcs: int = 30,
    n_dmso_pcs: int = 3,
    neg_ctrl_label: str = "DMSO",
    theta: float = 2.0,
    random_state: int = 42,
    inplace: bool = True,
) -> DrugSeqData | None:
    """
    Correct for plate/batch effects.

    Parameters
    ----------
    dsd : DrugSeqData with normalized counts (adata.X) and PCA in obsm['X_pca']
        if using harmony.
    batch_col : metadata column encoding the batch/plate variable.
    method : one of ``'harmony'``, ``'combat'``, ``'dmso_regression'``,
        ``'none'``.
        - **harmony** (default): corrects PCA embeddings; does NOT modify X.
          Run run_pca() first.  The corrected embedding is stored in
          ``obsm['X_pca_harmony']``.
        - **combat**: corrects adata.X directly.  Requires inmoose or sklearn.
        - **dmso_regression**: Drug-seq native; regresses out DMSO PC variation
          per plate from adata.X.
    covariate_cols : biological covariates to preserve during ComBat.
    n_pcs : number of PCs used for Harmony (default 30).
    n_dmso_pcs : DMSO PCs to regress out per plate (dmso_regression).
    theta : Harmony diversity penalty (default 2.0).
    random_state : reproducibility seed.
    """
    valid = ("harmony", "combat", "dmso_regression", "none")
    if method not in valid:
        raise ValueError(f"method must be one of {valid}")

    if method == "none":
        print("No batch correction applied.")
        return dsd if not inplace else None

    if batch_col not in dsd.obs.columns:
        raise KeyError(f"batch_col '{batch_col}' not found in obs.")

    if not inplace:
        dsd = DrugSeqData(dsd.adata.copy())

    print(f"Applying batch correction: method={method}, batch_col={batch_col}")

    if method == "harmony":
        _correct_harmony(dsd, batch_col, n_pcs, theta, random_state)

    elif method == "combat":
        _correct_combat(dsd, batch_col, covariate_cols)

    elif method == "dmso_regression":
        _correct_dmso_regression(dsd, batch_col, neg_ctrl_label, n_dmso_pcs)

    dsd.adata.uns["batch_method"] = method
    dsd.adata.uns["batch_col"]    = batch_col
    print("Batch correction complete.")
    return dsd if not inplace else None


# ---------------------------------------------------------------------------
# Harmony
# ---------------------------------------------------------------------------

def _correct_harmony(dsd, batch_col, n_pcs, theta, random_state):
    try:
        import harmonypy as hm
    except ImportError:
        raise ImportError(
            "harmonypy is required for method='harmony'. "
            "Install with: pip install harmonypy"
        )
    if "X_pca" not in dsd.adata.obsm:
        raise KeyError(
            "'X_pca' not found in obsm. Run run_pca() before correct_batch()."
        )

    pca_emb = dsd.adata.obsm["X_pca"][:, :n_pcs].copy()
    meta    = dsd.obs[[batch_col]]

    ho = hm.run_harmony(
        pca_emb.T,
        meta,
        batch_col,
        theta=theta,
        random_state=random_state,
        verbose=False,
    )
    dsd.adata.obsm["X_pca_harmony"] = ho.Z_corr.astype(np.float32)
    print(f"  Harmony corrected PCA stored in obsm['X_pca_harmony'] ({n_pcs} dims).")


# ---------------------------------------------------------------------------
# ComBat
# ---------------------------------------------------------------------------

def _correct_combat(dsd, batch_col, covariate_cols):
    X = dsd.adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = X.astype(float)

    batch = dsd.obs[batch_col].values
    try:
        # inmoose is the maintained Python ComBat implementation
        from inmoose.pycombat import pycombat_norm
        covariates = None
        if covariate_cols:
            covariates = dsd.obs[covariate_cols].values
        corrected = pycombat_norm(X.T, batch, covariates=covariates)
        dsd.adata.X = corrected.T.astype(np.float32)
    except ImportError:
        warnings.warn(
            "inmoose not installed; using simple per-batch mean centering. "
            "Install with: pip install inmoose"
        )
        corrected = X.copy()
        grand_mean = X.mean(axis=0)
        for b in np.unique(batch):
            idx = batch == b
            batch_mean = X[idx].mean(axis=0)
            corrected[idx] = X[idx] - batch_mean + grand_mean
        dsd.adata.X = corrected.astype(np.float32)


# ---------------------------------------------------------------------------
# DMSO-regression
# ---------------------------------------------------------------------------

def _correct_dmso_regression(dsd, batch_col, neg_ctrl_label, n_pcs):
    """
    For each plate, compute PCA on DMSO wells, then regress out those PCs
    from all wells on the plate.
    """
    from sklearn.decomposition import PCA as skPCA

    X = dsd.adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = X.astype(float).copy()  # (n_samples, n_genes)

    obs = dsd.obs
    plates = obs[batch_col].unique()

    for pid in plates:
        plate_idx = np.where(obs[batch_col] == pid)[0]
        dmso_idx  = np.where(
            (obs[batch_col] == pid) &
            (obs.get("sample_type", pd.Series("x")) == neg_ctrl_label)
        )[0]

        if len(dmso_idx) < 3:
            warnings.warn(
                f"Plate {pid}: fewer than 3 DMSO wells; skipping regression."
            )
            continue

        k = min(n_pcs, len(dmso_idx) - 1)
        pca = skPCA(n_components=k, random_state=42)
        pca.fit(X[dmso_idx])

        # project all plate samples, reconstruct, subtract
        plate_mat     = X[plate_idx]
        plate_c       = plate_mat - pca.mean_
        scores        = plate_c @ pca.components_.T
        reconstructed = scores @ pca.components_
        X[plate_idx]  = plate_mat - reconstructed

    dsd.adata.X = X.astype(np.float32)
    print(f"  DMSO-regression applied across {len(plates)} plate(s).")
