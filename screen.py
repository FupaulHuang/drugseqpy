"""
screen.py
---------
Screen-level compound analysis.

Functions
---------
aggregate_by_de              : compound-level DE signature matrix
compute_compound_umap        : UMAP on aggregated DE signatures
compute_compound_fingerprint : signed binary {-1,0,+1} compound fingerprints
compute_compound_similarity_network : igraph similarity network
compute_screen_profile       : fgsea-like similarity to a target profile
plot_screen_overview         : compact dot plot of perturbation × DE size
plot_screen_heatmap          : logFC heatmap across all compounds
plot_gene_counts             : CPM box plots per gene per treatment
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from .core import DrugSeqData
from .differential import summarise_de


# ---------------------------------------------------------------------------
# aggregate_by_de
# ---------------------------------------------------------------------------

def aggregate_by_de(
    dsd: DrugSeqData,
    value_col: str = "logFC",
) -> pd.DataFrame:
    """
    Build a genes × compounds DE signature matrix.

    Collapses replicate wells into compound-level log-fold-change vectors by
    taking the DE result for each compound.  Missing genes are filled with 0.

    Returns
    -------
    pd.DataFrame  shape (n_genes, n_compounds)
    """
    de = dsd.adata.uns.get("de_results", {})
    if not de:
        raise ValueError("de_results is empty. Run compute_multi_de() first.")

    compounds = list(de.keys())
    all_genes = dsd.var_names.tolist()

    mat = pd.DataFrame(0.0, index=all_genes, columns=compounds)
    for cmpd, df in de.items():
        col = value_col if value_col in df.columns else "logFC"
        s = df.set_index("gene")[col]
        shared = s.index.intersection(all_genes)
        mat.loc[shared, cmpd] = s[shared].values

    return mat


# ---------------------------------------------------------------------------
# compute_compound_umap
# ---------------------------------------------------------------------------

def compute_compound_umap(
    dsd: DrugSeqData,
    n_pcs: int = 20,
    n_variable_genes: int = 2000,
    n_neighbors: int = 5,
    min_dist: float = 0.3,
    use_leiden: bool = True,
    resolution: float = 0.8,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compound-level UMAP based on DE signatures.

    Aggregates replicates into compound-level logFC vectors, runs PCA then
    UMAP on the compound axis, and applies Leiden clustering.

    Returns
    -------
    pd.DataFrame  compounds × (UMAP_1, UMAP_2, cluster)
    """
    from sklearn.decomposition import PCA
    from umap import UMAP
    import anndata as ad
    import scanpy as sc

    de_mat = aggregate_by_de(dsd)  # (n_genes, n_compounds)

    # select variable genes
    gene_var = de_mat.var(axis=1)
    top_g    = gene_var.nlargest(min(n_variable_genes, len(gene_var))).index
    sub_mat  = de_mat.loc[top_g].T.values  # (n_compounds, n_genes_selected)

    # PCA
    k = min(n_pcs, sub_mat.shape[0] - 1, sub_mat.shape[1] - 1)
    pca = PCA(n_components=k, random_state=random_state)
    pca_emb = pca.fit_transform(sub_mat)

    # UMAP
    n_n = min(n_neighbors, pca_emb.shape[0] - 1)
    reducer = UMAP(n_neighbors=n_n, min_dist=min_dist,
                   random_state=random_state, verbose=False)
    umap_emb = reducer.fit_transform(pca_emb)

    # Leiden clustering via scanpy
    compounds = de_mat.columns.tolist()
    if use_leiden and len(compounds) > 3:
        agg = ad.AnnData(
            X=pca_emb.astype(np.float32),
            obs=pd.DataFrame({"compound": compounds}),
        )
        agg.obsm["X_pca"] = pca_emb.astype(np.float32)
        sc.pp.neighbors(agg, use_rep="X_pca",
                        n_neighbors=n_n, random_state=random_state)
        sc.tl.leiden(agg, resolution=resolution, random_state=random_state)
        clusters = agg.obs["leiden"].values
    else:
        clusters = np.zeros(len(compounds), dtype=str)

    return pd.DataFrame({
        "compound": compounds,
        "UMAP_1":   umap_emb[:, 0],
        "UMAP_2":   umap_emb[:, 1],
        "cluster":  clusters,
    })


# ---------------------------------------------------------------------------
# compute_compound_fingerprint
# ---------------------------------------------------------------------------

def compute_compound_fingerprint(
    dsd: DrugSeqData,
    fdr_threshold: float = 0.05,
    lfc_threshold: float = 0.5,
    union_genes: bool = True,
) -> pd.DataFrame:
    """
    Build a signed binary {-1, 0, +1} compound fingerprint matrix.

    Returns
    -------
    pd.DataFrame  shape (n_sig_genes, n_compounds)  with values in {-1, 0, 1}
    """
    de = dsd.adata.uns.get("de_results", {})
    if not de:
        raise ValueError("de_results is empty.")

    sig_genes_per_cmpd = {}
    for cmpd, df in de.items():
        sig = df[
            df["padj"].notna() &
            (df["padj"] < fdr_threshold) &
            (df["logFC"].abs() >= lfc_threshold)
        ]
        sig_genes_per_cmpd[cmpd] = sig.set_index("gene")["logFC"]

    if union_genes:
        all_sig = sorted(set().union(*[s.index for s in sig_genes_per_cmpd.values()]))
    else:
        all_sig = sorted(dsd.var_names)

    fp = pd.DataFrame(0, index=all_sig, columns=list(de.keys()), dtype=np.int8)
    for cmpd, lfc_s in sig_genes_per_cmpd.items():
        present = lfc_s.index.intersection(all_sig)
        fp.loc[present, cmpd] = np.sign(lfc_s[present]).astype(np.int8)

    return fp


# ---------------------------------------------------------------------------
# compute_compound_similarity_network
# ---------------------------------------------------------------------------

def compute_compound_similarity_network(
    dsd: DrugSeqData,
    method: str = "cosine",
    threshold: float = 0.4,
) -> dict:
    """
    Build a compound similarity network.

    Parameters
    ----------
    method : ``'cosine'``, ``'pearson'``, or ``'jaccard'``
    threshold : minimum similarity to draw an edge

    Returns
    -------
    dict with keys: similarity (DataFrame), graph (igraph if available),
    clusters (Series), plot (matplotlib Figure if igraph not available)
    """
    from .enrichment import connectivity_score

    if method == "jaccard":
        fp = compute_compound_fingerprint(dsd).abs()
        fp_bool = (fp != 0).astype(float)
        intersect = fp_bool.T @ fp_bool
        n = fp_bool.sum(axis=0)
        union = n.values[:, None] + n.values[None, :] - intersect.values
        sim = pd.DataFrame(
            intersect.values / (union + 1e-10),
            index=intersect.index, columns=intersect.columns
        )
    else:
        sim_mat = connectivity_score(dsd, method=method)
        sim = pd.DataFrame(sim_mat,
                           index=list(dsd.adata.uns["de_results"].keys()),
                           columns=list(dsd.adata.uns["de_results"].keys()))

    np.fill_diagonal(sim.values, 0)
    adj = sim.copy()
    adj[adj < threshold] = 0

    graph = None
    clusters = pd.Series(dtype=str)
    try:
        import igraph as ig
        g = ig.Graph.Weighted_Adjacency(
            adj.values.tolist(), mode="undirected"
        )
        g.vs["name"] = adj.index.tolist()
        comm = g.community_leiden(weights="weight", resolution_parameter=0.8)
        membership = dict(zip(adj.index, map(str, comm.membership)))
        clusters = pd.Series(membership)
        graph = g
        print(
            f"Compound network: {g.vcount()} nodes, {g.ecount()} edges, "
            f"{len(comm)} Leiden communities (threshold={threshold:.2f})."
        )
    except ImportError:
        warnings.warn("igraph not installed; graph object not created.")

    return {"similarity": sim, "graph": graph, "clusters": clusters}


# ---------------------------------------------------------------------------
# compute_screen_profile
# ---------------------------------------------------------------------------

def compute_screen_profile(
    dsd: DrugSeqData,
    target: str | list[str],
    rank_by: str = "stat",
    fdr_threshold: float = 0.1,
    n_perm: int = 1000,
) -> pd.DataFrame:
    """
    Score each compound's similarity to a target profile using GSEA.

    Parameters
    ----------
    target : compound name (str) or gene list (list of str)
    rank_by : DE column used to rank genes per compound

    Returns
    -------
    pd.DataFrame  compounds × (NES, pvalue, padj)
    """
    try:
        import gseapy as gp
    except ImportError:
        raise ImportError("gseapy required. Install: pip install gseapy")

    de = dsd.adata.uns.get("de_results", {})
    if not de:
        raise ValueError("de_results is empty. Run compute_multi_de() first.")

    # resolve target gene set
    if isinstance(target, str):
        if target not in de:
            raise ValueError(f"Target compound '{target}' not in de_results.")
        tgt_de = de[target]
        col = rank_by if rank_by in tgt_de.columns else "logFC"
        sig = tgt_de[
            tgt_de["padj"].notna() & (tgt_de["padj"] < fdr_threshold)
        ]
        target_set = {
            "up":   sig.loc[sig[col] > 0, "gene"].tolist(),
            "down": sig.loc[sig[col] < 0, "gene"].tolist(),
        }
        target_set = {k: v for k, v in target_set.items() if len(v) >= 10}
    else:
        target_set = {"target_set": list(target)}

    if not target_set:
        raise ValueError("Target profile has no significant genes at FDR < 0.1.")

    compounds = [c for c in de if c != (target if isinstance(target, str) else None)]
    rows = []
    for cmpd in compounds:
        df  = de[cmpd]
        col = rank_by if rank_by in df.columns else "logFC"
        ranked = df.dropna(subset=[col]).set_index("gene")[col].sort_values(ascending=False)
        try:
            res = gp.prerank(
                rnk=ranked,
                gene_sets=target_set,
                permutation_num=n_perm,
                outdir=None,
                verbose=False,
            )
            for gs, nes, pval, padj in zip(
                res.res2d["Term"],
                res.res2d["NES"],
                res.res2d["NOM p-val"],
                res.res2d["FDR q-val"],
            ):
                rows.append({"compound": cmpd, "gene_set": gs,
                             "NES": nes, "pvalue": pval, "padj": padj})
        except Exception as e:
            warnings.warn(f"  GSEA failed for '{cmpd}': {e}")

    if not rows:
        return pd.DataFrame(columns=["compound","gene_set","NES","pvalue","padj"])
    df_out = pd.DataFrame(rows)
    return df_out.sort_values("NES", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# plot_screen_overview
# ---------------------------------------------------------------------------

def plot_screen_overview(
    dsd: DrugSeqData,
    fdr_threshold: float = 0.05,
    lfc_threshold: float = 0.5,
    top_n_label: int = 15,
    use_pert_score: bool = True,
    figsize: tuple = (5, 8),
) -> plt.Figure:
    """
    Compact dot plot: perturbation score (y) × DE count (size) × direction (color).

    Novel function — not available in macpie or Seurat.
    """
    de_sum = summarise_de(dsd, fdr_threshold, lfc_threshold)

    # perturbation score
    if use_pert_score and "pert_score" in dsd.obs.columns:
        pert = dsd.obs.groupby("compound")["pert_score"].median()
        de_sum["pert_score"] = de_sum["compound"].map(pert)
    else:
        # fallback: mean |logFC|
        de = dsd.adata.uns.get("de_results", {})
        de_sum["pert_score"] = de_sum["compound"].map(
            {c: df["logFC"].abs().mean() for c, df in de.items()}
        )

    de_sum["direction_balance"] = (
        (de_sum["n_sig_up"] - de_sum["n_sig_down"]) /
        (de_sum["n_sig_total"] + 1)
    )
    de_sum = de_sum.sort_values("pert_score")

    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(
        de_sum["pert_score"],
        range(len(de_sum)),
        s=np.clip(de_sum["n_sig_total"] * 3, 20, 300),
        c=de_sum["direction_balance"],
        cmap="RdBu_r",
        vmin=-1, vmax=1,
        alpha=0.85,
        edgecolors="none",
    )
    ax.set_yticks(range(len(de_sum)))
    ax.set_yticklabels(de_sum["compound"], fontsize=7)
    ax.axvline(0, color="grey", linestyle="--", linewidth=0.5)
    ax.set_xlabel(
        "Perturbation score (DMSO-anchored)" if use_pert_score else "Mean |logFC|"
    )
    ax.set_title("Screen overview")
    plt.colorbar(sc, ax=ax, label="Direction (+up/−down)", shrink=0.5)

    # label top active
    top_idx = de_sum["n_sig_total"].nlargest(top_n_label).index
    for pos, (_, row) in enumerate(de_sum.iterrows()):
        if row.name in top_idx:
            ax.text(
                row["pert_score"], pos, f"  {row['compound']}",
                fontsize=6, va="center",
            )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# plot_screen_heatmap
# ---------------------------------------------------------------------------

def plot_screen_heatmap(
    dsd: DrugSeqData,
    gene_list: list[str] | None = None,
    n_top: int = 50,
    value_col: str = "logFC",
    fdr_threshold: float = 0.05,
    cluster_rows: bool = True,
    cluster_cols: bool = True,
    figsize: tuple | None = None,
    cmap: str = "RdBu_r",
) -> plt.Figure:
    """
    Heatmap of logFC across all compounds for selected genes.

    Mirrors macpie's ``plot_multi_de()``.
    """
    de_mat = aggregate_by_de(dsd, value_col)

    # filter to significant genes
    if fdr_threshold < 1.0:
        de = dsd.adata.uns["de_results"]
        sig_genes = set().union(*[
            set(df.loc[df["padj"].notna() & (df["padj"] < fdr_threshold), "gene"])
            for df in de.values()
        ])
        de_mat = de_mat.loc[de_mat.index.isin(sig_genes)]

    if gene_list is None:
        mean_lfc = de_mat.abs().mean(axis=1)
        gene_list = mean_lfc.nlargest(min(n_top, len(mean_lfc))).index.tolist()

    de_sub = de_mat.loc[de_mat.index.isin(gene_list)]
    if de_sub.empty:
        raise ValueError("No genes found in DE results.")

    if figsize is None:
        figsize = (max(6, len(de_sub.columns) * 0.5),
                   max(5, len(de_sub) * 0.15))

    fig = plt.figure(figsize=figsize)
    g = sns.clustermap(
        de_sub,
        cmap=cmap,
        center=0,
        row_cluster=cluster_rows,
        col_cluster=cluster_cols,
        figsize=figsize,
        yticklabels=len(de_sub) <= 60,
    )
    g.ax_heatmap.set_title(f"Screen logFC heatmap (n={len(de_sub)} genes)")
    return g.fig


# ---------------------------------------------------------------------------
# plot_gene_counts
# ---------------------------------------------------------------------------

def plot_gene_counts(
    dsd: DrugSeqData,
    genes: list[str],
    group_by: str = "compound",
    treatments: list[str] | None = None,
    control: str = "DMSO",
    normalization: str = "cpm",
    figsize: tuple | None = None,
) -> plt.Figure:
    """
    Box plots of expression per gene per treatment group.

    Mirrors macpie's ``plot_counts()``.
    """
    obs = dsd.obs
    if treatments is None:
        all_cpds = obs[group_by].unique().tolist()
        treatments = [c for c in all_cpds
                      if c not in (control, "vehicle", "media") ][:8]

    show_groups = [control] + treatments
    mask = obs[group_by].isin(show_groups)
    sub = dsd.adata[mask]
    sub_obs = sub.obs

    mat = sub.layers["counts"].toarray().astype(float)
    if normalization == "cpm":
        lib = mat.sum(axis=1, keepdims=True)
        nm  = mat / (lib + 1e-8) * 1e6
    else:
        nm = np.log1p(mat)

    genes_present = [g for g in genes if g in dsd.var_names]
    if not genes_present:
        raise ValueError("None of the requested genes found in var_names.")

    n_genes = len(genes_present)
    ncols = min(3, n_genes)
    nrows = int(np.ceil(n_genes / ncols))
    if figsize is None:
        figsize = (ncols * 4, nrows * 3.5)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for idx, gene in enumerate(genes_present):
        ax = axes[idx // ncols][idx % ncols]
        gene_idx = dsd.var_names.get_loc(gene)
        vals = nm[:, gene_idx]
        data = pd.DataFrame({
            "expression": vals,
            "group": sub_obs[group_by].values,
        })
        data = data[data["group"].isin(show_groups)]
        data["group"] = pd.Categorical(data["group"],
                                        categories=show_groups, ordered=True)
        sns.boxplot(data=data, x="group", y="expression",
                    palette=["#A8D5BA"] + ["#85B7EB"] * len(treatments),
                    ax=ax, linewidth=0.8, fliersize=2)
        ax.set_title(gene, fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("CPM" if normalization == "cpm" else "log1p", fontsize=7)
        ax.tick_params(axis="x", labelrotation=40, labelsize=7)

    # hide empty subplots
    for idx in range(n_genes, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(f"Gene counts: {treatments} vs {control}", fontsize=10)
    fig.tight_layout()
    return fig
