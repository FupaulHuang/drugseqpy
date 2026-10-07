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
    """Return a qualitative palette with one color per requested level."""
    base = ["#2980B9","#C0392B","#2ECC71","#E67E22","#9B59B6",
            "#1ABC9C","#E74C3C","#3498DB","#F39C12","#16A085"]
    if n <= len(base):
        return base[:n]
    return [mcolors.to_hex(color) for color in sns.color_palette("tab20", n)]


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
    split_by: str | None = None,
    label_by: str | None = None,
    point_size: float = 20,
    alpha: float = 0.8,
    figsize: tuple | None = None,
    show_pct_var: bool = True,
) -> plt.Figure:
    """Scatter plot of any obsm embedding, optionally faceted by metadata."""
    if reduction not in dsd.adata.obsm:
        available = list(dsd.adata.obsm.keys())
        raise KeyError(f"'{reduction}' not in obsm. Available: {available}")

    emb   = dsd.adata.obsm[reduction]
    x     = emb[:, dims[0]]
    y     = emb[:, dims[1]]
    meta  = dsd.obs
    if color_by not in meta.columns:
        raise KeyError(f"'{color_by}' not found in obs.")
    if split_by is not None and split_by not in meta.columns:
        raise KeyError(f"'{split_by}' not found in obs.")

    # axis labels with variance explained for PCA
    xlabel = f"Dim {dims[0]+1}"
    ylabel = f"Dim {dims[1]+1}"
    if show_pct_var and reduction == "X_pca":
        pct = dsd.adata.uns.get("pca", {}).get("variance_ratio", None)
        if pct is not None:
            xlabel = f"PC{dims[0]+1} ({pct[dims[0]]*100:.1f}%)"
            ylabel = f"PC{dims[1]+1} ({pct[dims[1]]*100:.1f}%)"

    return _plot_coordinate_panels(
        x,
        y,
        meta,
        color_by=color_by,
        split_by=split_by,
        label_by=label_by,
        point_size=point_size,
        alpha=alpha,
        figsize=figsize,
        xlabel=xlabel,
        ylabel=ylabel,
        title=reduction.replace("X_", "").upper(),
    )


def _plot_coordinate_panels(
    x: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    *,
    color_by: str,
    split_by: str | None,
    label_by: str | None,
    point_size: float,
    alpha: float,
    figsize: tuple | None,
    xlabel: str,
    ylabel: str,
    title: str,
) -> plt.Figure:
    """Render coordinates with a shared palette across optional facets."""
    groups = meta[color_by].astype(str).to_numpy()
    group_levels = list(pd.unique(groups))
    palette = dict(zip(group_levels, _drugseq_palette(len(group_levels))))

    if split_by is None:
        split_levels = [None]
    else:
        split_levels = list(pd.unique(meta[split_by].astype(str)))
        if not split_levels:
            raise ValueError(f"'{split_by}' has no values to plot.")

    n_panels = len(split_levels)
    ncols = min(n_panels, 3)
    nrows = int(np.ceil(n_panels / ncols))
    if figsize is None:
        figsize = (7, 6) if split_by is None else (6 * ncols, 5 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    flat_axes = axes.ravel()

    for panel_index, level in enumerate(split_levels):
        ax = flat_axes[panel_index]
        panel_mask = np.ones(len(meta), dtype=bool)
        if split_by is not None:
            panel_mask = meta[split_by].astype(str).eq(level).to_numpy()

        for group in group_levels:
            mask = panel_mask & (groups == group)
            if not mask.any():
                continue
            ax.scatter(
                x[mask], y[mask], s=point_size, alpha=alpha,
                label=group, color=palette[group], edgecolors="none",
            )

        if label_by and label_by in meta.columns:
            labels = meta[label_by].astype(str).to_numpy()
            for xi, yi, label in zip(x[panel_mask], y[panel_mask], labels[panel_mask]):
                ax.annotate(label, (xi, yi), fontsize=5, alpha=0.6)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"{split_by} = {level}" if split_by is not None else f"{title} — {color_by}"
        )

    for ax in flat_axes[n_panels:]:
        ax.set_visible(False)

    handles = [
        plt.Line2D(
            [], [], linestyle="", marker="o", markersize=5,
            color=palette[group], label=group,
        )
        for group in group_levels
    ]
    if split_by is None:
        flat_axes[0].legend(
            handles=handles, fontsize=7, markerscale=1.2,
            bbox_to_anchor=(1.02, 1), loc="upper left", title=color_by,
        )
    else:
        fig.suptitle(f"{title} — colored by {color_by}, split by {split_by}")
        fig.legend(
            handles=handles, fontsize=7, ncol=1,
            bbox_to_anchor=(1.01, 0.5), loc="center left", title=color_by,
        )
    fig.tight_layout(rect=(0, 0, 0.86 if split_by is not None else 1, 0.95))
    return fig


def plot_pca(
    dsd: DrugSeqData,
    color_by: str = "compound",
    **kwargs,
) -> plt.Figure:
    """Plot the PCA coordinates created by :func:`run_pca`."""
    return plot_embedding(
        dsd, reduction="X_pca", color_by=color_by, show_pct_var=True, **kwargs
    )


def plot_umap(
    dsd: DrugSeqData,
    color_by: str = "compound",
    **kwargs,
) -> plt.Figure:
    """Plot the UMAP coordinates created by :func:`run_umap`."""
    return plot_embedding(
        dsd, reduction="X_umap", color_by=color_by, show_pct_var=False, **kwargs
    )


def plot_tsne(
    dsd: DrugSeqData,
    color_by: str = "compound",
    **kwargs,
) -> plt.Figure:
    """Plot the t-SNE coordinates created by :func:`run_tsne`."""
    return plot_embedding(
        dsd, reduction="X_tsne", color_by=color_by, show_pct_var=False, **kwargs
    )


# ---------------------------------------------------------------------------
# plot_mds
# ---------------------------------------------------------------------------

def plot_mds(
    dsd: DrugSeqData,
    group_by: str = "sample_type",
    split_by: str | None = None,
    label_by: str | None = "compound",
    n_top_genes: int = 500,
    use_norm: bool = True,
    figsize: tuple | None = None,
) -> plt.Figure:
    """
    MDS plot using limma-style leading log-fold-change distances.

    Mirrors macpie's plot_mds().  Uses the top *n_top_genes* genes ranked
    by pairwise log-fold-change for each pair of samples.
    """
    from sklearn.manifold import MDS

    X = dsd.adata.X if use_norm else dsd.adata.layers["counts"]
    if sp.issparse(X):
        X = X.toarray()
    X = X.astype(float)
    if not use_norm:
        X = np.log2(X + 1)

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

    obs = dsd.obs
    if group_by not in obs.columns:
        raise KeyError(f"'{group_by}' not found in obs.")
    if split_by is not None and split_by not in obs.columns:
        raise KeyError(f"'{split_by}' not found in obs.")
    return _plot_coordinate_panels(
        coords[:, 0],
        coords[:, 1],
        obs,
        color_by=group_by,
        split_by=split_by,
        label_by=label_by,
        point_size=30,
        alpha=0.8,
        figsize=figsize,
        xlabel=f"Leading logFC dim 1 (top {n_top_genes} genes)",
        ylabel="Leading logFC dim 2",
        title="MDS",
    )


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
    ax.axhline(-np.log10(max(float(fdr_threshold), 1e-300)), color="grey",
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
    ax.text(
        0.98, 0.02, f"↑{n_up}  ↓{n_down}", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=8,
    )
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


# ---------------------------------------------------------------------------
# Comparison-aware DEG plots
# ---------------------------------------------------------------------------

UP_COLOR = "#C0392B"
DOWN_COLOR = "#2980B9"
NS_COLOR = "#AAAAAA"


def _coerce_dsd(dsd) -> DrugSeqData:
    """Accept either the package wrapper or a plain AnnData object."""
    return dsd if isinstance(dsd, DrugSeqData) else DrugSeqData(dsd)


def _comparison_tables(dsd: DrugSeqData) -> dict:
    """Comparison tables, preferring comparison_results over de_results."""
    tables = dsd.adata.uns.get("comparison_results") or \
        dsd.adata.uns.get("de_results", {})
    if not tables:
        raise ValueError(
            "No comparison results found. Run compute_multi_de() first."
        )
    return tables


def _significant_mask(df: pd.DataFrame, fdr_threshold: float,
                      lfc_threshold: float) -> pd.Series:
    """Use backend significance flags when available, otherwise thresholds."""
    if "significant" in df.columns:
        return df["significant"].fillna(False).astype(bool)
    if {"padj", "logFC"}.issubset(df.columns):
        return (df["padj"].notna() & (df["padj"] < fdr_threshold) &
                (df["logFC"].abs() >= lfc_threshold))
    return pd.Series(False, index=df.index, dtype=bool)


def _pick_comparison_key(tables: dict, contrast: str | None,
                         method: str | None) -> str:
    """Select a result key from the comparison tables."""
    keys = list(tables.keys())
    canonical_method = method
    if method is not None:
        try:
            from .differential import _resolve_method
            canonical_method = _resolve_method(method)
        except (TypeError, ValueError):
            canonical_method = str(method)
    if contrast is None:
        if canonical_method is not None:
            matching = [key for key in keys
                        if "method" in tables[key].columns and
                        not tables[key].empty and
                        str(tables[key]["method"].iloc[0]) == str(canonical_method)]
            if matching:
                return matching[0]
        simple = [k for k in keys if "::" not in str(k)]
        return simple[0] if simple else keys[0]
    if canonical_method is not None:
        qualified = f"{contrast}::{canonical_method}"
        if qualified in tables:
            return qualified
        matching = [key for key in keys
                    if str(key).startswith(f"{contrast}::") and
                    "method" in tables[key].columns and
                    not tables[key].empty and
                    str(tables[key]["method"].iloc[0]) == str(canonical_method)]
        if matching:
            return matching[0]
    if contrast in tables:
        return contrast
    prefix = [k for k in keys if str(k).startswith(f"{contrast}::")]
    if prefix:
        return prefix[0]
    raise KeyError(
        f"Contrast '{contrast}' not in comparison results. "
        f"Available: {keys}"
    )


def plot_comparison_volcano(
    dsd: DrugSeqData,
    contrast: str | None = None,
    method: str | None = None,
    lfc_threshold: float = 0.5,
    fdr_threshold: float = 0.05,
    n_label: int = 15,
    figsize: tuple = (6, 5),
) -> plt.Figure:
    """Volcano plot for a contrast stored in comparison_results."""
    dsd = _coerce_dsd(dsd)
    tables = _comparison_tables(dsd)
    key = _pick_comparison_key(tables, contrast, method)
    df = tables[key].copy()

    df["-log10_padj"] = -np.log10(df["padj"].astype(float).clip(1e-300))
    sig_mask = _significant_mask(df, fdr_threshold, lfc_threshold)
    df["direction"] = "NS"
    df.loc[sig_mask & (df["logFC"] > 0), "direction"] = "Up"
    df.loc[sig_mask & (df["logFC"] < 0), "direction"] = "Down"

    color_map = {"Up": UP_COLOR, "Down": DOWN_COLOR, "NS": NS_COLOR}
    fig, ax = plt.subplots(figsize=figsize)
    for direc in ("NS", "Up", "Down"):
        sub = df[df["direction"] == direc]
        ax.scatter(sub["logFC"], sub["-log10_padj"], s=8, alpha=0.6,
                   color=color_map[direc], label=direc, edgecolors="none")

    ax.axvline(-lfc_threshold, color="grey", linestyle="--", linewidth=0.6)
    ax.axvline(lfc_threshold, color="grey", linestyle="--", linewidth=0.6)
    ax.axhline(-np.log10(fdr_threshold), color="grey",
               linestyle="--", linewidth=0.6)

    if n_label > 0:
        maximum = float(df["-log10_padj"].max())
        ax.set_ylim(top=maximum + max(0.8, maximum * 0.08))
        top = df[sig_mask].nsmallest(n_label, "padj")
        for index, (_, row) in enumerate(top.iterrows()):
            ax.annotate(
                row["gene"], (row["logFC"], row["-log10_padj"]),
                xytext=(0.02, 0.97 - 0.035 * index),
                textcoords="axes fraction",
                ha="left",
                va="top",
                fontsize=5,
                alpha=0.8,
                arrowprops={"arrowstyle": "-", "color": "#6e7b7f", "lw": 0.35},
            )

    n_up = (df["direction"] == "Up").sum()
    n_down = (df["direction"] == "Down").sum()
    case = df["case"].iloc[0] if "case" in df.columns else key
    control = df["control"].iloc[0] if "control" in df.columns else "control"
    meth = df["method"].iloc[0] if "method" in df.columns else ""
    ax.set_xlabel("log₂ fold change")
    ax.set_ylabel("-log₁₀(adjusted p-value)")
    ax.set_title(f"Volcano: {case} vs {control} ({meth})")
    ax.legend(fontsize=7)
    ax.text(0.02, 0.98, f"↑{n_up}  ↓{n_down}", transform=ax.transAxes,
            va="top", fontsize=8)
    fig.tight_layout()
    return fig


def plot_comparison_heatmap(
    dsd: DrugSeqData,
    contrasts: list[str] | None = None,
    n_top_genes: int = 60,
    value_col: str = "logFC",
    fdr_threshold: float = 0.05,
    lfc_threshold: float = 0.5,
    significant_only: bool = True,
    cluster_rows: bool = True,
    cluster_cols: bool = True,
    cmap: str = "RdBu_r",
    figsize: tuple | None = None,
) -> plt.Figure:
    """
    Heatmap of DEG logFC across contrasts (reference-script style).

    Genes are the union of the top *n_top_genes* by absolute logFC across
    the selected contrasts (significant genes only by default).
    """
    dsd = _coerce_dsd(dsd)
    tables = _comparison_tables(dsd)
    keys = ([contrasts] if isinstance(contrasts, str) else contrasts) or list(tables.keys())
    keys = [k for k in keys if k in tables]
    if not keys:
        raise ValueError("None of the requested contrasts have results.")

    per_contrast = []
    for key in keys:
        df = tables[key]
        if significant_only:
            df = df[_significant_mask(df, fdr_threshold, lfc_threshold)]
        top = df.reindex(df["logFC"].abs().sort_values(ascending=False).index)\
                 .head(n_top_genes)
        per_contrast.append(top[["gene", value_col]].rename(
            columns={value_col: key}))

    mat = pd.concat(
        [t.set_index("gene") for t in per_contrast], axis=1
    ).fillna(0.0)
    if len(mat) > n_top_genes:
        keep = mat.abs().max(axis=1).sort_values(ascending=False, kind="mergesort")\
               .head(n_top_genes).index
        mat = mat.loc[keep]
    if mat.empty:
        raise ValueError(
            "No significant genes at the given thresholds — nothing to plot."
        )

    if figsize is None:
        figsize = (max(6, len(mat.columns) * 0.8),
                   max(5, len(mat) * 0.12))

    # hierarchical clustering needs >= 2 rows / columns
    cluster_rows = cluster_rows and mat.shape[0] >= 2
    cluster_cols = cluster_cols and mat.shape[1] >= 2
    if mat.shape[0] >= 2:
        g = sns.clustermap(
            mat, cmap=cmap, center=0,
            row_cluster=cluster_rows, col_cluster=cluster_cols,
            figsize=figsize, yticklabels=len(mat) <= 80,
        )
        g.ax_heatmap.set_title(f"DEG logFC heatmap (n={len(mat)} genes)")
        return g.fig

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(mat, cmap=cmap, center=0, ax=ax,
                yticklabels=True, cbar_kws={"label": value_col})
    ax.set_title(f"DEG logFC heatmap (n={len(mat)} genes)")
    fig.tight_layout()
    return fig


def plot_deg_counts(
    dsd: DrugSeqData,
    contrasts: list[str] | None = None,
    fdr_threshold: float = 0.05,
    lfc_threshold: float = 0.5,
    figsize: tuple | None = None,
) -> plt.Figure:
    """Mirrored bar plot of significant up/down DEG counts per contrast."""
    dsd = _coerce_dsd(dsd)
    tables = _comparison_tables(dsd)
    keys = ([contrasts] if isinstance(contrasts, str) else contrasts) or list(tables.keys())
    keys = [k for k in keys if k in tables]
    if not keys:
        raise ValueError("None of the requested contrasts have results.")

    rows = []
    for key in keys:
        df = tables[key]
        sig = _significant_mask(df, fdr_threshold, lfc_threshold)
        rows.append({
            "contrast": str(key),
            "up": int((sig & (df["logFC"] > 0)).sum()),
            "down": int((sig & (df["logFC"] < 0)).sum()),
        })
    counts = pd.DataFrame(rows).set_index("contrast")

    if figsize is None:
        figsize = (6, max(3, len(counts) * 0.5))
    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(counts))
    ax.barh(y, counts["up"], color=UP_COLOR, label="up", height=0.6)
    ax.barh(y, -counts["down"], color=DOWN_COLOR, label="down", height=0.6)
    for i, (u, d) in enumerate(zip(counts["up"], counts["down"])):
        if u:
            ax.text(u, i, f" {u}", va="center", fontsize=7)
        if d:
            ax.text(-d, i, f" {d} ", va="center", ha="right", fontsize=7)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(counts.index, fontsize=7)
    ax.set_xlabel("Number of significant DEGs")
    ax.set_title("DEG counts per contrast")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def _gene_expression_groups(dsd, contrast, group_col, show_groups):
    """Resolve the sample groups shown for gene box/violin plots."""
    obs = dsd.obs

    def _selector_values(value):
        if isinstance(value, (list, tuple, set, frozenset, pd.Index, np.ndarray)):
            return [str(item) for item in value]
        text = str(value)
        return text.split("|") if "|" in text else [text]

    if contrast is not None:
        tables = _comparison_tables(dsd)
        key = _pick_comparison_key(tables, contrast, None)
        df = tables[key]
        case, control = df["case"].iloc[0], df["control"].iloc[0]
        groups = _selector_values(control) + _selector_values(case)
        groups = list(dict.fromkeys(groups))
        gcol = group_col
        if gcol not in obs.columns:
            raise KeyError(f"group column '{gcol}' not in obs.")
        return gcol, groups
    if show_groups is not None:
        selected = ([show_groups] if isinstance(show_groups, str)
                    else list(show_groups))
        selected = [str(value) for value in selected]
        return group_col, selected
    if group_col not in obs.columns:
        raise KeyError(f"group column '{group_col}' not in obs.")
    levels = [lv for lv in obs[group_col].dropna().unique()
              if str(lv) not in ("vehicle", "media", "Media")]
    return group_col, levels[:8]


def _resolve_gene_indices(dsd, genes):
    """Resolve requested gene IDs or symbols to unique var column indices."""
    names = [str(value) for value in dsd.var_names]
    exact = {value: i for i, value in enumerate(names)}
    lower = {value.lower(): i for i, value in enumerate(names)}
    symbols = {}
    if "gene_symbol" in dsd.var.columns:
        for i, value in enumerate(dsd.var["gene_symbol"].astype(str)):
            key = value.strip().lower()
            if key and key not in {"nan", "none"}:
                symbols.setdefault(key, i)
    found = []
    missing = []
    for gene in genes:
        text = str(gene)
        idx = exact.get(text)
        if idx is None:
            idx = lower.get(text.lower(), symbols.get(text.lower()))
        if idx is None:
            missing.append(gene)
        else:
            found.append((text, idx))
    return found, missing


def _gene_expression_matrix(dsd, use_norm):
    """Expression values (n_samples, n_genes) for box/violin plots."""
    if use_norm:
        X = dsd.adata.X
        if X is None:
            X = dsd.adata.layers.get("counts")
    else:
        if "counts" not in dsd.adata.layers:
            raise KeyError("'counts' layer is required for expression plots.")
        X = dsd.adata.layers["counts"]
        lib = np.asarray(X.sum(axis=1)).reshape(-1, 1)
        X = X / (lib + 1e-8) * 1e6
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=float)


def plot_gene_boxplot(
    dsd: DrugSeqData,
    genes: list[str],
    group_col: str = "compound",
    contrast: str | None = None,
    show_groups: list[str] | None = None,
    use_norm: bool = True,
    figsize: tuple | None = None,
) -> plt.Figure:
    """Box plots of expression for one or more genes across groups."""
    dsd = _coerce_dsd(dsd)
    if isinstance(genes, str):
        genes = [genes]
    gcol, groups = _gene_expression_groups(dsd, contrast, group_col,
                                           show_groups)
    mask = dsd.obs[gcol].isin(groups)
    sub = dsd.adata[mask]
    vals = _gene_expression_matrix(DrugSeqData(sub), use_norm)

    resolved_genes, _ = _resolve_gene_indices(dsd, genes)
    if not resolved_genes:
        raise ValueError(
            f"None of the requested genes found in var_names: {genes}"
        )

    n_genes = len(resolved_genes)
    ncols = min(3, n_genes)
    nrows = int(np.ceil(n_genes / ncols))
    if figsize is None:
        figsize = (ncols * 4, nrows * 3.5)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    palette = [NS_COLOR if g == groups[0] else DOWN_COLOR for g in groups]
    for idx, (gene, gene_idx) in enumerate(resolved_genes):
        ax = axes[idx // ncols][idx % ncols]
        data = pd.DataFrame({
            "expression": vals[:, gene_idx],
            "group": pd.Categorical(sub.obs[gcol].values,
                                    categories=groups, ordered=True),
        })
        sns.boxplot(data=data, x="group", y="expression",
                    hue="group", palette=palette, legend=False,
                    ax=ax, linewidth=0.8, fliersize=2)
        ax.set_title(gene, fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("expression" if use_norm else "CPM", fontsize=7)
        ax.tick_params(axis="x", labelrotation=40, labelsize=7)

    for idx in range(n_genes, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Gene expression boxplots", fontsize=10)
    fig.tight_layout()
    return fig


def plot_gene_violin(
    dsd: DrugSeqData,
    genes: list[str],
    group_col: str = "compound",
    contrast: str | None = None,
    show_groups: list[str] | None = None,
    use_norm: bool = True,
    figsize: tuple | None = None,
) -> plt.Figure:
    """Violin plots (inner box + strip overlay) of expression per group."""
    dsd = _coerce_dsd(dsd)
    if isinstance(genes, str):
        genes = [genes]
    gcol, groups = _gene_expression_groups(dsd, contrast, group_col,
                                           show_groups)
    mask = dsd.obs[gcol].isin(groups)
    sub = dsd.adata[mask]
    vals = _gene_expression_matrix(DrugSeqData(sub), use_norm)

    resolved_genes, _ = _resolve_gene_indices(dsd, genes)
    if not resolved_genes:
        raise ValueError(
            f"None of the requested genes found in var_names: {genes}"
        )

    n_genes = len(resolved_genes)
    ncols = min(3, n_genes)
    nrows = int(np.ceil(n_genes / ncols))
    if figsize is None:
        figsize = (ncols * 4, nrows * 3.5)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    palette = [NS_COLOR if g == groups[0] else DOWN_COLOR for g in groups]
    for idx, (gene, gene_idx) in enumerate(resolved_genes):
        ax = axes[idx // ncols][idx % ncols]
        data = pd.DataFrame({
            "expression": vals[:, gene_idx],
            "group": pd.Categorical(sub.obs[gcol].values,
                                    categories=groups, ordered=True),
        })
        sns.violinplot(data=data, x="group", y="expression",
                       hue="group", palette=palette, legend=False,
                       ax=ax, inner="box", cut=0, linewidth=0.6, alpha=0.7)
        sns.stripplot(data=data, x="group", y="expression",
                      color="black", size=1.5, alpha=0.4, ax=ax, jitter=True)
        ax.set_title(gene, fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("expression" if use_norm else "CPM", fontsize=7)
        ax.tick_params(axis="x", labelrotation=40, labelsize=7)

    for idx in range(n_genes, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Gene expression violin plots", fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Enrichment plots
# ---------------------------------------------------------------------------

def _enrichment_slot(dsd, contrast, library, mode, direction):
    """Fetch one enrichment DataFrame from uns['enrichment_results']."""
    enr = dsd.adata.uns.get("enrichment_results", {})
    if not enr:
        raise ValueError(
            "No enrichment results found. Run run_enrichment() first."
        )
    if contrast is None:
        contrast = list(enr.keys())[0]
    if contrast not in enr:
        raise KeyError(f"Contrast '{contrast}' not in enrichment_results.")
    if library is None:
        library = list(enr[contrast].keys())[0]
    if library not in enr[contrast]:
        raise KeyError(f"Library '{library}' not found for '{contrast}'.")
    slot = f"{mode}_{direction}" if mode == "ora" else "gsea"
    df = enr[contrast][library].get(slot)
    if df is None:
        df = pd.DataFrame()
    return contrast, library, df


def _top_terms(df, n_terms, mode):
    """Top pathways by adjusted p-value (per mode's column names)."""
    padj_col = "Adjusted P-value" if mode == "ora" else "FDR q-val"
    if padj_col not in df.columns or df.empty:
        return df
    sub = df[df[padj_col].notna()].sort_values(padj_col)
    return sub.head(n_terms)


def _empty_axes_message(fig, ax, message):
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=10,
            color="grey", transform=ax.transAxes)
    return fig


def plot_enrichment_dotplot(
    dsd: DrugSeqData,
    contrast: str | None = None,
    libraries: list[str] | None = None,
    direction: str = "up",
    mode: str = "ora",
    n_terms: int = 20,
    figsize: tuple | None = None,
) -> plt.Figure:
    """
    Dot plot of top enriched pathways.

    For ORA mode, dot size = overlap count and color encodes direction
    (``direction='both'`` plots up and down together).  For GSEA mode,
    dot size = |NES| and color = NES sign.
    """
    dsd = _coerce_dsd(dsd)
    if mode not in {"ora", "gsea"}:
        raise ValueError("mode must be 'ora' or 'gsea'.")
    if direction not in {"up", "down", "both"}:
        raise ValueError("direction must be 'up', 'down', or 'both'.")
    enr = dsd.adata.uns.get("enrichment_results", {})
    if not enr:
        fig, ax = plt.subplots(figsize=(6, 2))
        return _empty_axes_message(fig, ax, "No enrichment results found.")

    if contrast is None:
        contrast = list(enr.keys())[0]
    if contrast not in enr:
        raise KeyError(f"Contrast '{contrast}' not in enrichment_results.")
    libs = ([libraries] if isinstance(libraries, str) else list(libraries)) \
        if libraries else list(enr[contrast].keys())

    directions = ["up", "down"] if direction == "both" else [direction]
    if figsize is None:
        figsize = (4.5 * len(libs), 5)
    fig, axes = plt.subplots(1, len(libs), figsize=figsize, squeeze=False)

    for ax, lib in zip(axes[0], libs):
        frames = {}
        for direc in directions:
            slot = f"{mode}_{direc}" if mode == "ora" else "gsea"
            df = enr[contrast].get(lib, {}).get(slot, pd.DataFrame())
            frames[direc] = _top_terms(df, n_terms, mode)
        ax.set_title(lib, fontsize=9)

        plot_rows = []
        for direc, df in frames.items():
            if df.empty:
                continue
            padj_col = "Adjusted P-value" if mode == "ora" else "FDR q-val"
            for _, r in df.iterrows():
                if mode == "ora":
                    overlap = str(r.get("Overlap", ""))
                    magnitude = int(overlap.split("/")[0]) \
                        if overlap and overlap.split("/")[0].isdigit() else 10
                    color = UP_COLOR if direc == "up" else DOWN_COLOR
                    size = 24 + 18 * np.sqrt(magnitude)
                    eff = magnitude
                else:
                    nes = float(r.get("NES", np.nan))
                    magnitude = abs(nes)
                    size = magnitude * 30 + 10
                    color = UP_COLOR if nes > 0 else DOWN_COLOR
                    eff = nes
                plot_rows.append({
                    "pathway": str(r.get("Term", "")),
                    "x": -np.log10(max(float(r.get(padj_col, np.nan) or 1e-300), 1e-300)),
                    "size": size,
                    "color": color,
                    "effect": eff,
                })
        if not plot_rows:
            _empty_axes_message(fig, ax,
                                "No significant enrichment" if frames
                                else f"Library '{lib}' not available")
            continue
        pdf = pd.DataFrame(plot_rows)
        pdf["y"] = range(len(pdf))
        ax.scatter(pdf["x"], pdf["y"], s=pdf["size"], c=pdf["color"],
                   alpha=0.8, edgecolors="none")
        ax.set_yticks(pdf["y"])
        labels = [p[:40] + ("…" if len(p) > 40 else "")
                  for p in pdf["pathway"]]
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("-log₁₀ adjusted p-value", fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
        magnitudes = np.asarray([abs(value) for value in pdf["effect"]], dtype=float)
        legend_values = np.unique(
            np.quantile(magnitudes, [0, 0.5, 1]).round(1)
        )
        size_handles = []
        for value in legend_values:
            marker_size = (
                24 + 18 * np.sqrt(value)
                if mode == "ora" else value * 30 + 10
            )
            label = f"{int(value)}" if mode == "ora" else f"{value:g}"
            size_handles.append(
                ax.scatter([], [], s=marker_size, color="#738185", alpha=0.8,
                           edgecolors="none", label=label)
            )
        size_legend = ax.legend(
            handles=size_handles,
            title="Overlap genes" if mode == "ora" else "|NES|",
            fontsize=7,
            title_fontsize=7,
            loc="lower right" if direction == "both" else "upper right",
            frameon=False,
        )
        if direction == "both":
            from matplotlib.patches import Patch
            ax.add_artist(size_legend)
            ax.legend(handles=[Patch(color=UP_COLOR, label="up"),
                               Patch(color=DOWN_COLOR, label="down")],
                      fontsize=7, loc="upper right", frameon=False)
    direction_label = {
        "up": "upregulated DEGs",
        "down": "downregulated DEGs",
        "both": "up- and downregulated DEGs",
    }[direction]
    fig.suptitle(
        f"Enrichment dot plot — {contrast}, {direction_label} ({mode})",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def plot_enrichment_barplot(
    dsd: DrugSeqData,
    contrast: str | None = None,
    library: str | None = None,
    direction: str = "up",
    mode: str = "ora",
    n_terms: int = 20,
    figsize: tuple | None = None,
) -> plt.Figure:
    """Horizontal bar plot of top pathways by -log10 adjusted p-value."""
    dsd = _coerce_dsd(dsd)
    if mode not in {"ora", "gsea"}:
        raise ValueError("mode must be 'ora' or 'gsea'.")
    if direction not in {"up", "down"}:
        raise ValueError("direction must be 'up' or 'down'.")
    contrast, library, df = _enrichment_slot(dsd, contrast, library,
                                             mode, direction)
    df = _top_terms(df, n_terms, mode)

    if figsize is None:
        figsize = (6.5, max(3, len(df) * 0.35))
    fig, ax = plt.subplots(figsize=figsize)

    if df.empty:
        return _empty_axes_message(fig, ax, "No significant enrichment")

    padj_col = "Adjusted P-value" if mode == "ora" else "FDR q-val"
    df = df.copy()
    df["-log10_padj"] = -np.log10(
        df[padj_col].astype(float).clip(lower=1e-300))
    df = df.sort_values("-log10_padj", ascending=False)
    color = UP_COLOR if direction == "up" else DOWN_COLOR

    ax.barh(range(len(df)), df["-log10_padj"], color=color, height=0.6)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([t[:50] + ("…" if len(t) > 50 else "")
                        for t in df["Term"]], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("-log₁₀ adjusted p-value")
    ax.set_title(f"{contrast} — {library} ({mode} {direction})", fontsize=10)
    fig.tight_layout()
    return fig
