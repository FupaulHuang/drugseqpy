"""
plots.py
--------
All visualization functions for Drug-seq QC and analysis.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.collections import PathCollection

from .core import DrugSeqData


# ---------------------------------------------------------------------------
# Shared style helpers
# ---------------------------------------------------------------------------

def _drugseq_palette(n: int) -> list[str]:
    """Qualitative palette cycled to n colors."""
    base = ["#2980B9","#C0392B","#2ECC71","#E67E22","#9B59B6",
            "#1ABC9C","#E74C3C","#3498DB","#F39C12","#16A085"]
    return [base[i % len(base)] for i in range(n)]


FLAG_COLORS = {"pass": "#2ECC71", "warn": "#F39C12", "fail": "#E74C3C"}


# ---------------------------------------------------------------------------
# plot_qc_summary
# ---------------------------------------------------------------------------

def plot_qc_summary(
    dsd: DrugSeqData,
    metrics: list[str] | None = None,
    group_by: str = "plate_id",
    thresholds: dict | None = None,
    ncol: int = 3,
    figsize: tuple | None = None,
) -> plt.Figure:
    """
    Violin + strip plots for QC metrics, faceted by group.

    Mirrors the DRUGseqR plotQCSummary().
    """
    if metrics is None:
        metrics = ["total_umi","n_genes_det","pct_mito",
                   "pct_ribo","hk_cv","gini_index","outlier_score"]
    if thresholds is None:
        thresholds = {"pct_mito": 20, "n_genes_det": 300}

    metrics = [m for m in metrics if m in dsd.obs.columns]
    if not metrics:
        raise ValueError("No QC metrics found in obs. Run compute_qc_metrics().")

    nrow = int(np.ceil(len(metrics) / ncol))
    if figsize is None:
        figsize = (ncol * 4, nrow * 3.5)
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)

    groups = dsd.obs[group_by].unique() if group_by in dsd.obs.columns \
             else ["all"]
    palette = _drugseq_palette(len(groups))

    for idx, metric in enumerate(metrics):
        ax = axes[idx // ncol][idx % ncol]
        data = dsd.obs[[group_by, metric]].dropna() if group_by in dsd.obs.columns \
               else dsd.obs[[metric]].dropna().assign(**{group_by: "all"})

        sns.violinplot(data=data, x=group_by, y=metric,
                       palette=palette, ax=ax,
                       inner="box", linewidth=0.5, alpha=0.7)
        sns.stripplot(data=data, x=group_by, y=metric,
                      color="black", size=1.5, alpha=0.3, ax=ax, jitter=True)

        if metric in (thresholds or {}):
            ax.axhline(thresholds[metric], color="#C0392B",
                       linestyle="--", linewidth=0.8)

        ax.set_title(metric, fontsize=9)
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelrotation=40, labelsize=7)

    for idx in range(len(metrics), nrow * ncol):
        axes[idx // ncol][idx % ncol].set_visible(False)

    fig.suptitle("Per-sample QC metrics", fontsize=11, y=1.01)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# plot_qc_scatter
# ---------------------------------------------------------------------------

def plot_qc_scatter(
    dsd: DrugSeqData,
    x_metric: str = "log10_total_umi",
    y_metric: str = "n_genes_det",
    color_by: str = "sample_type",
    x_threshold: float | None = None,
    y_threshold: float | None = None,
    label_outliers: bool = True,
    figsize: tuple = (6, 5),
) -> plt.Figure:
    """Scatter plot of two QC metrics, outliers labeled."""
    df = dsd.obs[[x_metric, y_metric, color_by]].dropna()
    groups = df[color_by].unique()
    palette = dict(zip(groups, _drugseq_palette(len(groups))))

    fig, ax = plt.subplots(figsize=figsize)
    for grp in groups:
        sub = df[df[color_by] == grp]
        ax.scatter(sub[x_metric], sub[y_metric], s=15, alpha=0.7,
                   label=str(grp), color=palette[grp], edgecolors="none")

    if x_threshold:
        ax.axvline(x_threshold, color="#C0392B", linestyle="--", linewidth=0.7)
    if y_threshold:
        ax.axhline(y_threshold, color="#C0392B", linestyle="--", linewidth=0.7)

    if label_outliers and "outlier_score" in dsd.obs.columns:
        outliers = dsd.obs[dsd.obs["outlier_score"] > 5]
        for sid, row in outliers.iterrows():
            if sid in df.index:
                ax.annotate(sid, (df.loc[sid, x_metric], df.loc[sid, y_metric]),
                            fontsize=5, alpha=0.7)

    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    ax.legend(fontsize=7, markerscale=1.5)
    ax.set_title(f"{x_metric} vs {y_metric}")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# plot_plate_heatmap
# ---------------------------------------------------------------------------

def plot_plate_heatmap(
    dsd: DrugSeqData,
    plate_id: str | None = None,
    value_col: str = "log10_total_umi",
    show_sample_type: bool = False,
    cmap: str = "magma",
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """Spatial heatmap of a QC metric across the plate layout."""
    pqc = dsd.adata.uns.get("plate_qc", {})
    if not pqc:
        raise ValueError("plate_qc empty. Run compute_plate_qc() first.")
    if plate_id is None:
        plate_id = list(pqc.keys())[0]

    wm = pqc[plate_id].get("well_matrix")
    if wm is None:
        raise ValueError(f"No well_matrix for plate '{plate_id}'.")

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(wm, cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax, label=value_col, shrink=0.6)

    n_rows, n_cols = wm.shape
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([str(i + 1) for i in range(n_cols)], fontsize=6)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([chr(65 + i) for i in range(n_rows)], fontsize=6)
    ax.set_title(f"Plate: {plate_id}  |  {value_col}")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# plot_zprime
# ---------------------------------------------------------------------------

def plot_zprime(
    dsd: DrugSeqData,
    show_ssmd: bool = True,
    figsize: tuple | None = None,
) -> plt.Figure:
    """Bar chart of Z'-factor (and optionally SSMD) per plate."""
    from .qc import plate_qc_summary

    df = plate_qc_summary(dsd)
    df = df.sort_values("zprime")
    colors = [FLAG_COLORS.get(f, "grey") for f in df["flag"]]

    ncols = 2 if show_ssmd else 1
    if figsize is None:
        figsize = (ncols * 4, max(3, len(df) * 0.5))
    fig, axes = plt.subplots(1, ncols, figsize=figsize, sharey=True)
    if ncols == 1:
        axes = [axes]

    axes[0].barh(df["plate_id"], df["zprime"], color=colors)
    axes[0].axvline(0.5, color="grey", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel("Z'-factor")
    axes[0].set_title("Z'-factor per plate")

    if show_ssmd:
        axes[1].barh(df["plate_id"], df["ssmd"].abs(), color=colors)
        axes[1].axvline(3, color="grey", linestyle="--", linewidth=0.8)
        axes[1].set_xlabel("|SSMD|")
        axes[1].set_title("SSMD per plate")

    from matplotlib.patches import Patch
    legend = [Patch(color=c, label=l) for l, c in FLAG_COLORS.items()]
    axes[-1].legend(handles=legend, fontsize=7, loc="lower right")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# plot_hk_genes
# ---------------------------------------------------------------------------

def plot_hk_genes(
    dsd: DrugSeqData,
    hk_genes: list[str] | None = None,
    n_top: int = 50,
    group_by: str = "plate_id",
    figsize: tuple | None = None,
) -> plt.Figure:
    """Heatmap of housekeeping gene expression across samples."""
    from .qc import _load_hk_genes

    if hk_genes is None:
        hk_genes = _load_hk_genes()
    present = [g for g in hk_genes if g in dsd.var_names][:n_top]
    if not present:
        raise ValueError("No housekeeping genes found in var_names.")

    gi = [dsd.var_names.get_loc(g) for g in present]
    mat = dsd.adata.X[:, gi]
    if sp.issparse(mat):
        mat = mat.toarray()
    mat = np.log1p(mat.astype(float))  # (n_samples, n_hk)

    df = pd.DataFrame(mat.T, index=present)
    z  = df.subtract(df.mean(axis=1), axis=0).divide(df.std(axis=1) + 1e-8, axis=0)

    if figsize is None:
        figsize = (max(8, dsd.n_obs * 0.08), max(5, len(present) * 0.18))

    row_colors = None
    if group_by in dsd.obs.columns:
        grps = dsd.obs[group_by]
        unique = grps.unique()
        cmap_g = dict(zip(unique, _drugseq_palette(len(unique))))
        row_colors = grps.map(cmap_g)

    g = sns.clustermap(
        z,
        col_colors=row_colors,
        cmap="RdBu_r",
        center=0,
        figsize=figsize,
        yticklabels=len(present) <= 60,
    )
    g.ax_heatmap.set_title(f"Housekeeping genes (n={len(present)})", fontsize=9)
    return g.fig


# ---------------------------------------------------------------------------
# plot_embedding
# ---------------------------------------------------------------------------

def plot_embedding(
    dsd: DrugSeqData,
    reduction: str = "X_umap",
    dims: tuple[int, int] = (0, 1),
    color_by: str = "compound",
    label_by: str | None = None,
    point_size: float = 20,
    alpha: float = 0.8,
    figsize: tuple = (7, 6),
    show_pct_var: bool = True,
) -> plt.Figure:
    """Scatter plot of any obsm embedding."""
    if reduction not in dsd.adata.obsm:
        available = list(dsd.adata.obsm.keys())
        raise KeyError(f"'{reduction}' not in obsm. Available: {available}")

    emb   = dsd.adata.obsm[reduction]
    x     = emb[:, dims[0]]
    y     = emb[:, dims[1]]
    meta  = dsd.obs

    fig, ax = plt.subplots(figsize=figsize)

    if color_by in meta.columns:
        groups = meta[color_by].values
        unique = pd.unique(groups)
        pal    = dict(zip(unique, _drugseq_palette(len(unique))))
        for grp in unique:
            mask = groups == grp
            ax.scatter(x[mask], y[mask], s=point_size, alpha=alpha,
                       label=str(grp), color=pal[grp], edgecolors="none")
        ax.legend(fontsize=7, markerscale=1.5,
                  bbox_to_anchor=(1.02, 1), loc="upper left")
    elif color_by in dsd.obs.columns:
        vals = dsd.obs[color_by].values.astype(float)
        sc   = ax.scatter(x, y, c=vals, s=point_size, alpha=alpha,
                          cmap="viridis", edgecolors="none")
        plt.colorbar(sc, ax=ax, label=color_by, shrink=0.7)

    if label_by and label_by in meta.columns:
        for xi, yi, lab in zip(x, y, meta[label_by]):
            ax.annotate(lab, (xi, yi), fontsize=5, alpha=0.6)

    # axis labels with variance explained for PCA
    xlabel = f"Dim {dims[0]+1}"
    ylabel = f"Dim {dims[1]+1}"
    if show_pct_var and reduction == "X_pca":
        pct = dsd.adata.uns.get("pca", {}).get("variance_ratio", None)
        if pct is not None:
            xlabel = f"PC{dims[0]+1} ({pct[dims[0]]*100:.1f}%)"
            ylabel = f"PC{dims[1]+1} ({pct[dims[1]]*100:.1f}%)"

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{reduction.replace('X_','').upper()} — {color_by}")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# plot_mds
# ---------------------------------------------------------------------------

def plot_mds(
    dsd: DrugSeqData,
    group_by: str = "sample_type",
    label_by: str | None = "compound",
    n_top_genes: int = 500,
    use_norm: bool = True,
    figsize: tuple = (7, 6),
) -> plt.Figure:
    """
    MDS plot using limma-style leading log-fold-change distances.

    Mirrors macpie's plot_mds().  Uses the top *n_top_genes* genes ranked
    by pairwise log-fold-change for each pair of samples.
    """
    from sklearn.manifold import MDS

    X = dsd.adata.X if use_norm else dsd.adata.layers["counts"].toarray()
    if sp.issparse(X):
        X = X.toarray()
    X = np.log2(X.astype(float) + 1)

    # gene selection: top genes by variance
    gene_var = X.var(axis=0)
    top_idx  = np.argsort(gene_var)[::-1][:n_top_genes]
    X_sub    = X[:, top_idx]

    # pairwise leading-logFC distance
    n = X_sub.shape[0]
    k = min(500, top_idx.size)
    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            diff   = np.abs(X_sub[i] - X_sub[j])
            lead   = np.sort(diff)[::-1][:k].mean()
            dist_mat[i, j] = dist_mat[j, i] = lead

    mds  = MDS(n_components=2, dissimilarity="precomputed",
               random_state=42, normalized_stress="auto")
    coords = mds.fit_transform(dist_mat)

    fig, ax = plt.subplots(figsize=figsize)
    obs = dsd.obs
    if group_by in obs.columns:
        groups = obs[group_by].values
        unique = pd.unique(groups)
        pal    = dict(zip(unique, _drugseq_palette(len(unique))))
        for grp in unique:
            mask = groups == grp
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       s=30, alpha=0.8, label=str(grp),
                       color=pal[grp], edgecolors="none")
        ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")

    if label_by and label_by in obs.columns:
        for xi, yi, lab in zip(coords[:, 0], coords[:, 1], obs[label_by]):
            ax.annotate(lab, (xi, yi), fontsize=5, alpha=0.6)

    ax.set_xlabel(f"Leading logFC dim 1 (top {n_top_genes} genes)")
    ax.set_ylabel("Leading logFC dim 2")
    ax.set_title("MDS — sample grouping")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# plot_rle
# ---------------------------------------------------------------------------

def plot_rle(
    dsd: DrugSeqData,
    subset_type: str | None = "DMSO",
    normalization: str = "limma_voom",
    label_col: str = "well_id",
    color_col: str = "plate_id",
    figsize: tuple | None = None,
) -> plt.Figure:
    """
    Relative Log Expression (RLE) plot.

    Boxes should be centered on 0 after good normalization.
    Mirrors macpie's plot_rle().
    """
    from .normalization import _limma_voom, _tmm_size_factors

    adata = dsd.adata
    if subset_type and "sample_type" in adata.obs.columns:
        mask = adata.obs["sample_type"] == subset_type
        if not mask.any():
            warnings.warn(f"No samples with sample_type='{subset_type}'; using all.")
            mask = pd.Series(True, index=adata.obs_names)
        sub = adata[mask]
    else:
        sub = adata

    counts = sub.layers["counts"].toarray().astype(float)

    if normalization == "raw":
        nm = np.log2(counts + 1)
    elif normalization in ("CPM", "TMM", "limma_voom"):
        from .normalization import _tmm_size_factors, _limma_voom
        sf  = _tmm_size_factors(counts)
        lib = counts.sum(axis=1) * sf
        if normalization == "limma_voom":
            nm = _limma_voom(counts)
        else:
            nm = np.log2(counts / (lib[:, None] + 1e-8) * 1e6 + 1)
    else:
        nm = np.log2(counts + 1)

    row_meds = np.median(nm, axis=0)
    rle      = nm - row_meds[None, :]  # (n_samples, n_genes)

    labels = sub.obs.get(label_col, sub.obs_names)
    colors_col = sub.obs.get(color_col, pd.Series("all", index=sub.obs_names))
    unique_colors = colors_col.unique()
    pal = dict(zip(unique_colors, _drugseq_palette(len(unique_colors))))
    box_colors = [pal[c] for c in colors_col]

    n = rle.shape[0]
    if figsize is None:
        figsize = (max(8, n * 0.35), 4)
    fig, ax = plt.subplots(figsize=figsize)

    bp = ax.boxplot(
        rle.T.tolist(),
        patch_artist=True,
        widths=0.6,
        flierprops={"marker": ".", "markersize": 2, "alpha": 0.3},
        medianprops={"color": "black", "linewidth": 1},
    )
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(0, color="#C0392B", linestyle="--", linewidth=0.7)
    ax.set_xticks(range(1, n + 1))
    ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax.set_ylabel("RLE")
    ax.set_title(
        f"RLE plot — {subset_type or 'all'} samples ({normalization})"
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# plot_volcano
# ---------------------------------------------------------------------------

def plot_volcano(
    dsd: DrugSeqData,
    compound: str | None = None,
    lfc_threshold: float = 0.5,
    fdr_threshold: float = 0.05,
    n_label: int = 15,
    figsize: tuple = (6, 5),
) -> plt.Figure:
    """Volcano plot for a single compound."""
    de = dsd.adata.uns.get("de_results", {})
    if not de:
        raise ValueError("de_results empty.")
    if compound is None:
        compound = list(de.keys())[0]
        print(f"No compound specified; using '{compound}'.")
    if compound not in de:
        raise KeyError(f"'{compound}' not in de_results.")

    df = de[compound].copy()
    df["-log10_padj"] = -np.log10(df["padj"].clip(1e-300))
    df["direction"]   = "NS"
    df.loc[df["significant"] & (df["logFC"] > 0), "direction"] = "Up"
    df.loc[df["significant"] & (df["logFC"] < 0), "direction"] = "Down"

    color_map = {"Up": "#C0392B", "Down": "#2980B9", "NS": "#AAAAAA"}
    fig, ax   = plt.subplots(figsize=figsize)
    for direc in ("NS", "Up", "Down"):
        sub = df[df["direction"] == direc]
        ax.scatter(sub["logFC"], sub["-log10_padj"], s=8, alpha=0.6,
                   color=color_map[direc], label=direc, edgecolors="none")

    ax.axvline(-lfc_threshold, color="grey", linestyle="--", linewidth=0.6)
    ax.axvline( lfc_threshold, color="grey", linestyle="--", linewidth=0.6)
    ax.axhline(-np.log10(fdr_threshold), color="grey",
               linestyle="--", linewidth=0.6)

    # label top genes
    if n_label > 0:
        top = df[df["significant"]].nsmallest(n_label, "padj")
        for _, row in top.iterrows():
            ax.annotate(row["gene"], (row["logFC"], row["-log10_padj"]),
                        fontsize=5, alpha=0.8)

    n_up   = (df["direction"] == "Up").sum()
    n_down = (df["direction"] == "Down").sum()
    ax.set_xlabel("log₂ fold change")
    ax.set_ylabel("-log₁₀(adjusted p-value)")
    ax.set_title(f"Volcano: {compound} vs DMSO")
    ax.legend(fontsize=7)
    ax.text(0.02, 0.98, f"↑{n_up}  ↓{n_down}", transform=ax.transAxes,
            va="top", fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# plot_ma
# ---------------------------------------------------------------------------

def plot_ma(
    dsd: DrugSeqData,
    compound: str | None = None,
    lfc_threshold: float = 0.5,
    fdr_threshold: float = 0.05,
    n_label: int = 10,
    figsize: tuple = (6, 5),
) -> plt.Figure:
    """MA plot for a single compound."""
    de = dsd.adata.uns.get("de_results", {})
    if compound is None:
        compound = list(de.keys())[0]
    df = de[compound].copy()

    fig, ax = plt.subplots(figsize=figsize)
    sig = df["significant"].fillna(False)
    ax.scatter(df.loc[~sig, "base_mean"], df.loc[~sig, "logFC"],
               s=6, alpha=0.4, color="#AAAAAA", edgecolors="none")
    ax.scatter(df.loc[sig, "base_mean"], df.loc[sig, "logFC"],
               s=8, alpha=0.7, color="#C0392B", edgecolors="none")

    ax.axhline(0,              color="black",  linewidth=0.6)
    ax.axhline( lfc_threshold, color="grey", linestyle="--", linewidth=0.6)
    ax.axhline(-lfc_threshold, color="grey", linestyle="--", linewidth=0.6)

    if n_label > 0:
        top = df[sig].nlargest(n_label, "logFC")
        for _, row in top.iterrows():
            ax.annotate(row["gene"], (row["base_mean"], row["logFC"]),
                        fontsize=5, alpha=0.8)

    ax.set_xscale("log")
    ax.set_xlabel("Mean expression (log scale)")
    ax.set_ylabel("log₂ fold change")
    ax.set_title(f"MA plot: {compound} vs DMSO")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# plot_pc_elbow
# ---------------------------------------------------------------------------

def plot_pc_elbow(
    dsd: DrugSeqData,
    n_pcs: int = 30,
    figsize: tuple = (5, 4),
) -> plt.Figure:
    """Elbow plot of PCA variance explained."""
    pca_uns = dsd.adata.uns.get("pca", {})
    pct_var = pca_uns.get("variance_ratio", None)
    if pct_var is None:
        raise ValueError("PCA not found. Run run_pca() first.")
    n = min(n_pcs, len(pct_var))
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(range(1, n + 1), pct_var[:n] * 100, "o-", markersize=4,
            linewidth=1, color="#2980B9")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("% variance explained")
    ax.set_title("PCA elbow plot")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# plot_replicate_distance
# ---------------------------------------------------------------------------

def plot_replicate_distance(
    dsd: DrugSeqData,
    treatment: str | None = None,
    group_by: str = "compound",
    use_norm: bool = True,
    figsize: tuple = (6, 5),
) -> plt.Figure:
    """Pairwise distance heatmap within a treatment group."""
    obs = dsd.obs
    if treatment is None:
        cpds = [c for c in obs[group_by].unique() if c != "DMSO"]
        treatment = cpds[0] if cpds else obs[group_by].iloc[0]

    mask = obs[group_by] == treatment
    sub  = dsd.adata[mask]
    X    = sub.X if use_norm else sub.layers["counts"].toarray().astype(float)
    if sp.issparse(X):
        X = X.toarray()

    from sklearn.metrics import pairwise_distances
    dist_mat = pairwise_distances(X, metric="euclidean")
    ids = sub.obs_names.tolist()

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(dist_mat, cmap="inferno_r", aspect="auto")
    ax.set_xticks(range(len(ids))); ax.set_xticklabels(ids, rotation=90, fontsize=6)
    ax.set_yticks(range(len(ids))); ax.set_yticklabels(ids, fontsize=6)
    plt.colorbar(im, ax=ax, label="Euclidean distance", shrink=0.6)
    ax.set_title(f"Replicate distance — {treatment}")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# plot_group_qc_heatmap
# ---------------------------------------------------------------------------

def plot_group_qc_heatmap(
    stats_df: pd.DataFrame,
    figsize: tuple | None = None,
) -> plt.Figure:
    """Heatmap of group-level QC metrics (sd, mad, z_score, IQR, cv_pct)."""
    grp_col  = stats_df.columns[0]
    mat_cols = [c for c in ["sd_value","mad_value","z_score","IQR","cv_pct"]
                if c in stats_df.columns]
    mat = stats_df.set_index(grp_col)[mat_cols]
    z   = (mat - mat.mean()) / (mat.std() + 1e-8)

    if figsize is None:
        figsize = (max(5, len(mat_cols)), max(4, len(mat) * 0.3))
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(z.T, cmap="RdBu_r", center=0, ax=ax,
                cbar_kws={"label": "z-score"}, linewidths=0.2)
    ax.set_title("Group-level QC metrics (z-scaled)")
    ax.set_xlabel("")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# plot_norm_comparison
# ---------------------------------------------------------------------------

def plot_norm_comparison(
    dsd: DrugSeqData,
    methods: list[str] | None = None,
    subset_type: str | None = "DMSO",
    figsize: tuple = (11, 3.5),
) -> plt.Figure:
    """
    Three-panel normalization comparison: RLE center, RLE IQR, and CV.

    - **RLE center** (median |per-sample mean RLE|): measures library-size
      bias.  A perfectly normalized dataset has all boxes centered at 0, so
      lower is better.
    - **RLE IQR** (median per-sample RLE IQR): measures heteroscedasticity
      across the dynamic range.  CPM and TMM can both achieve low center but
      differ here if one inflates variance at low-count genes.
    - **Mean CV**: inter-sample coefficient of variation averaged over genes.
      Lower = more reproducible expression across samples.

    The ``"raw"`` bar is always shown as a grey anchor so improvements from
    each method are visually interpretable rather than just relative to each
    other.
    """
    from .normalization import compare_normalizations

    if methods is None:
        methods = ["raw", "CPM", "TMM", "limma_voom"]

    df = compare_normalizations(dsd, methods, subset_type)

    pal = ["#AAAAAA" if m == "raw" else c
           for m, c in zip(df["method"], _drugseq_palette(len(df)))]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    metrics = [
        ("median_rle_center", "Median |RLE center|\n(library-size bias)", "lower = better"),
        ("median_rle_iqr",    "Median RLE IQR\n(heteroscedasticity)",     "lower = better"),
        ("mean_cv",           "Mean inter-sample CV\n(reproducibility)",   "lower = better"),
    ]
    for ax, (col, title, subtitle) in zip(axes, metrics):
        ax.bar(df["method"], df[col], color=pal, edgecolor="none")
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(subtitle, fontsize=7)
        ax.tick_params(axis="x", labelrotation=25, labelsize=8)

    fig.suptitle(
        f"Normalization comparison  (subset: {subset_type or 'all samples'})",
        fontsize=10
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# plot_compound_umap
# ---------------------------------------------------------------------------

def plot_compound_umap(
    umap_df: pd.DataFrame,
    color_by: str = "cluster",
    label_compounds: bool = True,
    figsize: tuple = (7, 6),
) -> plt.Figure:
    """Scatter plot of compound-level UMAP."""
    groups = umap_df[color_by].unique()
    pal    = dict(zip(groups, _drugseq_palette(len(groups))))

    fig, ax = plt.subplots(figsize=figsize)
    for grp in groups:
        sub = umap_df[umap_df[color_by] == grp]
        ax.scatter(sub["UMAP_1"], sub["UMAP_2"], s=60, alpha=0.85,
                   label=str(grp), color=pal[grp], edgecolors="none")

    if label_compounds:
        for _, row in umap_df.iterrows():
            ax.annotate(row["compound"], (row["UMAP_1"], row["UMAP_2"]),
                        fontsize=5, alpha=0.7)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Compound-level UMAP (DE signatures)")
    ax.legend(title=color_by, fontsize=7, markerscale=1.5,
              bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig
