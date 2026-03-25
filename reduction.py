"""
reduction.py
------------
Dimensionality reduction for Drug-seq data.

Functions
---------
run_pca          : scanpy-backed PCA on normalized counts
run_umap         : UMAP via umap-learn
embed_dmso       : DMSO-anchored embedding (perturbation score)
cluster_compounds: graph-based or k-means compound clustering
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .core import DrugSeqData


# ---------------------------------------------------------------------------
# run_pca
# ---------------------------------------------------------------------------

def run_pca(
    dsd: DrugSeqData,
    n_pcs: int = 50,
    n_variable_genes: int = 2000,
    use_highly_variable: bool = True,
    random_state: int = 42,
    inplace: bool = True,
) -> DrugSeqData | None:
    """
    Run PCA on normalized counts, storing results in ``obsm['X_pca']``.

    Also marks highly variable genes in ``var['highly_variable']`` and stores
    PCA loadings in ``varm['PCs']``, variance ratios in ``uns['pca']``.
    """
    import scanpy as sc

    if not inplace:
        dsd = DrugSeqData(dsd.adata.copy())

    adata = dsd.adata
    if adata.X is None or (sp.issparse(adata.X) and adata.X.nnz == 0):
        raise ValueError("adata.X is empty. Run normalize_counts() first.")

    # select highly variable genes
    if use_highly_variable:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=min(n_variable_genes, adata.n_vars),
            flavor="seurat",
            inplace=True,
        )
        print(
            f"  Selected {adata.var['highly_variable'].sum()} highly variable genes."
        )
    else:
        adata.var["highly_variable"] = True

    n_pcs_actual = min(n_pcs, adata.n_obs - 1,
                       adata.var["highly_variable"].sum() - 1)
    sc.pp.pca(
        adata,
        n_comps=n_pcs_actual,
        use_highly_variable=True,
        random_state=random_state,
        svd_solver="arpack",
    )

    pct_var = adata.uns["pca"]["variance_ratio"] * 100
    print(
        f"PCA complete. Top PC: {pct_var[0]:.1f}% variance; "
        f"cumulative (10 PCs): {pct_var[:10].sum():.1f}%."
    )
    return dsd if not inplace else None


# ---------------------------------------------------------------------------
# run_umap
# ---------------------------------------------------------------------------

def run_umap(
    dsd: DrugSeqData,
    dims: int | list[int] = 20,
    n_neighbors: int = 15,
    min_dist: float = 0.3,
    metric: str = "euclidean",
    use_harmony: bool = False,
    random_state: int = 42,
    inplace: bool = True,
) -> DrugSeqData | None:
    """
    Compute UMAP embedding from PCA coordinates.

    Requires ``run_pca()`` to have been called first.

    Parameters
    ----------
    dims : number of PCs to use (int) or explicit list of PC indices
    use_harmony : use Harmony-corrected PCA (obsm['X_pca_harmony']) if available
    """
    import scanpy as sc

    if not inplace:
        dsd = DrugSeqData(dsd.adata.copy())

    adata = dsd.adata
    if "X_pca" not in adata.obsm:
        raise KeyError("PCA not found. Run run_pca() first.")

    # select PCA representation
    rep_key = "X_pca_harmony" if (use_harmony and "X_pca_harmony" in adata.obsm) \
              else "X_pca"

    if isinstance(dims, int):
        n_dims = min(dims, adata.obsm[rep_key].shape[1])
        use_rep_slice = adata.obsm[rep_key][:, :n_dims]
    else:
        use_rep_slice = adata.obsm[rep_key][:, list(dims)]
        n_dims = len(dims)

    # store a trimmed copy for neighbors
    adata.obsm["_pca_for_umap"] = use_rep_slice.astype(np.float32)

    sc.pp.neighbors(
        adata,
        use_rep="_pca_for_umap",
        n_neighbors=n_neighbors,
        metric=metric,
        random_state=random_state,
    )
    sc.tl.umap(adata, min_dist=min_dist, random_state=random_state)

    # clean up temporary key
    del adata.obsm["_pca_for_umap"]
    print(f"UMAP complete (n_neighbors={n_neighbors}, min_dist={min_dist}).")
    return dsd if not inplace else None


# ---------------------------------------------------------------------------
# embed_dmso
# ---------------------------------------------------------------------------

def embed_dmso(
    dsd: DrugSeqData,
    n_pcs: int = 20,
    n_variable_genes: int = 2000,
    neg_ctrl_label: str = "DMSO",
    random_state: int = 42,
    inplace: bool = True,
) -> DrugSeqData | None:
    """
    Compute a DMSO-anchored embedding.

    Fits PCA exclusively on DMSO wells, projects all samples into that
    DMSO-defined space, and computes a perturbation score (z-scored Euclidean
    distance from the DMSO centroid).  High perturbation score → strong
    transcriptional response.

    Results stored in ``obsm['X_dmso_pca']`` and ``obs['pert_score']``.
    """
    from sklearn.decomposition import PCA as skPCA

    if "sample_type" not in dsd.obs.columns:
        raise KeyError("'sample_type' column required in obs.")

    if not inplace:
        dsd = DrugSeqData(dsd.adata.copy())

    adata = dsd.adata
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = X.astype(float)

    # select variable genes
    gene_var = X.var(axis=0)
    top_idx  = np.argsort(gene_var)[::-1][:min(n_variable_genes, X.shape[1])]
    X_sub    = X[:, top_idx]

    dmso_mask = (adata.obs["sample_type"] == neg_ctrl_label).values
    if dmso_mask.sum() < 3:
        raise ValueError(
            f"Only {dmso_mask.sum()} DMSO wells found; need >= 3."
        )
    print(f"embedDMSO: fitting PCA on {dmso_mask.sum()} DMSO wells.")

    k = min(n_pcs, dmso_mask.sum() - 1)
    pca = skPCA(n_components=k, random_state=random_state)
    pca.fit(X_sub[dmso_mask])

    emb = pca.transform(X_sub)  # (n_samples, k)
    colnames = [f"DMSO_PC{i+1}" for i in range(k)]

    # distance from DMSO centroid
    dmso_center = emb[dmso_mask].mean(axis=0)
    distances   = np.sqrt(((emb - dmso_center) ** 2).sum(axis=1))

    # z-score relative to DMSO distribution
    dmso_dists = distances[dmso_mask]
    pert_score = (distances - dmso_dists.mean()) / (dmso_dists.std() + 1e-8)

    adata.obsm["X_dmso_pca"]     = emb.astype(np.float32)
    adata.obs["pert_score"]       = pert_score
    adata.obs["dmso_dist"]        = distances
    adata.uns["dmso_pca_cols"]    = colnames
    adata.uns["dmso_pca_center"]  = dmso_center

    print("embedDMSO complete. 'pert_score' reflects perturbation from DMSO centroid.")
    return dsd if not inplace else None


# ---------------------------------------------------------------------------
# cluster_compounds
# ---------------------------------------------------------------------------

def cluster_compounds(
    dsd: DrugSeqData,
    reduction: str = "X_pca",
    n_dims: int = 20,
    compound_col: str = "compound",
    method: str = "leiden",
    resolution: float = 0.8,
    k: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Cluster compounds by their mean transcriptomic embedding.

    Parameters
    ----------
    reduction : embedding key in obsm
    n_dims : number of dimensions to use
    method : ``'leiden'``, ``'louvain'``, or ``'kmeans'``
    resolution : Leiden/Louvain resolution
    k : k-means clusters (only used if method='kmeans')

    Returns
    -------
    DataFrame with columns: compound, cluster
    """
    if reduction not in dsd.adata.obsm:
        raise KeyError(f"'{reduction}' not found in obsm.")

    emb   = dsd.adata.obsm[reduction][:, :n_dims]
    obs   = dsd.obs
    cpds  = obs[compound_col].values

    # compound-level mean embeddings
    unique_cpds = pd.unique(cpds)
    centroids   = np.vstack([
        emb[cpds == c].mean(axis=0) for c in unique_cpds
    ])

    if method == "kmeans":
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, random_state=random_state, n_init=25)
        labels = km.fit_predict(centroids).astype(str)

    else:
        import scanpy as sc
        import anndata as ad
        agg_adata = ad.AnnData(
            X=centroids.astype(np.float32),
            obs=pd.DataFrame({"compound": unique_cpds}),
        )
        agg_adata.obsm["X_emb"] = centroids.astype(np.float32)
        sc.pp.neighbors(agg_adata, use_rep="X_emb", n_neighbors=min(15, len(unique_cpds)-1))
        if method == "leiden":
            sc.tl.leiden(agg_adata, resolution=resolution, random_state=random_state)
            labels = agg_adata.obs["leiden"].values
        else:
            sc.tl.louvain(agg_adata, resolution=resolution, random_state=random_state)
            labels = agg_adata.obs["louvain"].values

    return pd.DataFrame({"compound": unique_cpds, "cluster": labels})
