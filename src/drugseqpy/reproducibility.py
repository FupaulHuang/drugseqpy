"""Repeatability and cross-plate reproducibility for Drug-seq experiments."""

from __future__ import annotations

from itertools import combinations
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import pearsonr, spearmanr

from .core import DrugSeqData
from .differential import run_comparison
from .plate import plot_plate_layout


def _coerce_dsd(dsd) -> DrugSeqData:
    return dsd if isinstance(dsd, DrugSeqData) else DrugSeqData(dsd)


def _natural_key(value: str) -> tuple:
    parts = re.split(r"(\d+)", str(value))
    return tuple(int(part) if part.isdigit() else part.casefold() for part in parts)


def compute_within_plate_repeatability(
    dsd: DrugSeqData,
    plate_col: str = "plate_id",
    group_col: str = "compound",
    min_replicates: int = 2,
    use_norm: bool = False,
    method: str = "pearson",
) -> pd.DataFrame:
    """Compute replicate-pair correlations within plate/group strata.

    Raw counts transformed as ``log10(count + 1)`` are used by default.
    Set ``use_norm=True`` to correlate normalized expression instead.
    """
    dsd = _coerce_dsd(dsd)
    for column in (plate_col, group_col):
        if column not in dsd.obs.columns:
            raise KeyError(f"'{column}' not found in obs.")
    if min_replicates < 2:
        raise ValueError("min_replicates must be at least 2.")
    method = str(method).casefold()
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be 'pearson' or 'spearman'.")

    matrix = _repeatability_matrix(dsd, use_norm=use_norm)

    rows = []
    obs = dsd.obs.copy()
    obs["_sample_id"] = dsd.obs_names.astype(str)
    obs = obs.reset_index(drop=True)
    for (plate, group), indexes in obs.groupby(
        [plate_col, group_col], sort=True
    ).groups.items():
        indexes = np.asarray(list(indexes), dtype=int)
        if len(indexes) < min_replicates:
            continue
        correlations = _sample_correlation_matrix(matrix[indexes], method)
        for first, second in combinations(range(len(indexes)), 2):
            rows.append({
                plate_col: plate,
                group_col: group,
                "sample_a": obs.loc[indexes[first], "_sample_id"],
                "sample_b": obs.loc[indexes[second], "_sample_id"],
                "n_replicates": len(indexes),
                "n_genes": matrix.shape[1],
                "method": method,
                "expression_source": (
                    "normalized" if use_norm else "log10_raw_umi"
                ),
                "correlation": float(correlations[first, second]),
            })
    if not rows:
        raise ValueError("No plate/group combination has enough replicates.")
    return pd.DataFrame(rows).sort_values(
        [plate_col, group_col, "sample_a", "sample_b"]
    ).reset_index(drop=True)


def _repeatability_matrix(dsd: DrugSeqData, use_norm: bool) -> np.ndarray:
    matrix = dsd.X if use_norm else dsd.layers["counts"]
    if sp.issparse(matrix):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=float)
    return matrix if use_norm else np.log10(matrix + 1)


def _sample_correlation_matrix(matrix: np.ndarray, method: str) -> np.ndarray:
    if method == "pearson":
        result = np.corrcoef(matrix)
    else:
        result = spearmanr(matrix, axis=1, nan_policy="omit").statistic
    if np.ndim(result) == 0:
        value = float(result)
        return np.array([[1.0, value], [value, 1.0]])
    return np.asarray(result, dtype=float)


def plot_within_plate_repeatability(
    repeatability: pd.DataFrame,
    plate_col: str = "plate_id",
    group_col: str = "compound",
    value_col: str = "correlation",
    figsize: tuple[float, float] = (14, 6.2),
) -> plt.Figure:
    """Plot replicate-pair correlations without pooling different compounds."""
    missing = {plate_col, group_col, value_col} - set(repeatability.columns)
    if missing:
        raise KeyError(f"repeatability table is missing columns: {sorted(missing)}")

    fig, ax = plt.subplots(figsize=figsize)
    plates = sorted(
        repeatability[plate_col].astype(str).unique(), key=_natural_key
    )
    groups = sorted(
        repeatability[group_col].astype(str).unique(), key=_natural_key
    )
    colors = ["#147d75", "#d08a1b", "#7c5aa6", "#397bb5"]
    width = min(0.32, 0.72 / max(len(plates), 1))
    offsets = (np.arange(len(plates)) - (len(plates) - 1) / 2) * width
    rng = np.random.default_rng(42)
    for plate_index, plate in enumerate(plates):
        color = colors[plate_index % len(colors)]
        for group_index, group in enumerate(groups):
            values = repeatability.loc[
                repeatability[plate_col].astype(str).eq(plate)
                & repeatability[group_col].astype(str).eq(group),
                value_col,
            ].dropna().to_numpy(dtype=float)
            if not len(values):
                continue
            position = group_index + offsets[plate_index]
            box = ax.boxplot(
                [values], positions=[position], widths=width * 0.82,
                patch_artist=True, showfliers=False,
            )
            box["boxes"][0].set(facecolor=color, edgecolor=color, alpha=0.48)
            for median in box["medians"]:
                median.set(color=color, linewidth=1.6)
            jitter = rng.uniform(-width * 0.22, width * 0.22, len(values))
            ax.scatter(
                position + jitter, values, s=10, color=color,
                alpha=0.38, edgecolors="none", zorder=3,
            )
        ax.plot([], [], color=color, linewidth=7, alpha=0.55, label=plate)
    methods = (
        repeatability["method"].dropna().astype(str).unique()
        if "method" in repeatability.columns else []
    )
    method_label = methods[0].title() if len(methods) == 1 else "Profile"
    ax.set(
        title="Within-plate replicate correlation by compound",
        ylabel=f"{method_label} correlation for replicate-well pairs",
        xlabel="Compound",
        xticks=np.arange(len(groups)),
        xticklabels=groups,
    )
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title=plate_col, frameon=False, ncol=min(len(plates), 4))
    ax.grid(axis="y", color="#dfe6e7", linestyle="--", linewidth=0.7)
    fig.tight_layout()
    return fig


def summarise_within_plate_repeatability(
    repeatability: pd.DataFrame,
    plate_col: str = "plate_id",
    group_col: str = "compound",
    value_col: str = "correlation",
) -> pd.DataFrame:
    """Summarize pair-level repeatability to one row per plate and compound."""
    missing = {plate_col, group_col, value_col} - set(repeatability.columns)
    if missing:
        raise KeyError(f"repeatability table is missing columns: {sorted(missing)}")
    grouping = [plate_col, group_col]
    if "method" in repeatability.columns:
        grouping.append("method")
    if "expression_source" in repeatability.columns:
        grouping.append("expression_source")
    return (
        repeatability.groupby(grouping, sort=True, dropna=False)
        .agg(
            n_replicates=("n_replicates", "max"),
            n_pairs=(value_col, "size"),
            median_correlation=(value_col, "median"),
            minimum_correlation=(value_col, "min"),
        )
        .reset_index()
    )


def _select_replicate_group(
    dsd: DrugSeqData,
    *,
    plate: str,
    compound: str,
    plate_col: str,
    group_col: str,
    use_norm: bool,
) -> tuple[np.ndarray, pd.DataFrame]:
    dsd = _coerce_dsd(dsd)
    for column in (plate_col, group_col):
        if column not in dsd.obs.columns:
            raise KeyError(f"'{column}' not found in obs.")
    mask = (
        dsd.obs[plate_col].astype(str).eq(str(plate))
        & dsd.obs[group_col].astype(str).eq(str(compound))
    ).to_numpy()
    if mask.sum() < 2:
        raise ValueError(
            f"Need at least two replicates for {plate_col}={plate!r}, "
            f"{group_col}={compound!r}."
        )
    obs = dsd.obs.loc[mask].copy()
    obs["_sample_id"] = dsd.obs_names[mask].astype(str)
    obs["_matrix_position"] = np.flatnonzero(mask)
    sort_column = "well_id" if "well_id" in obs.columns else "_sample_id"
    order = sorted(
        range(len(obs)), key=lambda index: _natural_key(obs.iloc[index][sort_column])
    )
    obs = obs.iloc[order].copy()
    positions = obs["_matrix_position"].to_numpy(dtype=int)
    return _repeatability_matrix(dsd, use_norm=use_norm)[positions], obs


def plot_within_plate_replicate_scatter(
    dsd: DrugSeqData,
    plate: str,
    compound: str,
    method: str = "pearson",
    plate_col: str = "plate_id",
    group_col: str = "compound",
    label_col: str = "well_id",
    use_norm: bool = False,
    max_pairs: int | None = 4,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Plot replicate expression using raw log10 counts or normalized values."""
    method = str(method).casefold()
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be 'pearson' or 'spearman'.")
    matrix, obs = _select_replicate_group(
        dsd,
        plate=plate,
        compound=compound,
        plate_col=plate_col,
        group_col=group_col,
        use_norm=use_norm,
    )
    pairs = list(combinations(range(len(obs)), 2))
    if max_pairs is not None:
        if max_pairs < 1:
            raise ValueError("max_pairs must be positive or None.")
        if len(pairs) > max_pairs:
            selected = np.linspace(0, len(pairs) - 1, max_pairs, dtype=int)
            pairs = [pairs[index] for index in selected]
    ncols = min(2, len(pairs))
    nrows = int(np.ceil(len(pairs) / ncols))
    if figsize is None:
        figsize = (6.2 * ncols, 5.4 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    correlations = _sample_correlation_matrix(matrix, method)
    low = float(np.nanmin(matrix))
    high = float(np.nanmax(matrix))
    padding = max((high - low) * 0.03, 0.05)
    if label_col in obs.columns:
        labels = np.array([
            f"{plate}_{value}" for value in obs[label_col].astype(str)
        ])
    else:
        labels = obs["_sample_id"].astype(str).to_numpy()
    axis_label = "Normalized expression" if use_norm else "log10(raw UMI + 1)"

    for axis, (first, second) in zip(axes.flat, pairs):
        x = matrix[first]
        y = matrix[second]
        axis.scatter(
            x, y, s=5, color="#167d9a", alpha=0.24,
            linewidth=0, rasterized=True,
        )
        axis.plot(
            [low - padding, high + padding], [low - padding, high + padding],
            color="#6b7280", linestyle="--", linewidth=0.8,
        )
        if np.nanstd(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            axis.plot(
                [low, high],
                [intercept + slope * low, intercept + slope * high],
                color="#d55e00", linewidth=1.0,
            )
        symbol = "r" if method == "pearson" else "rho"
        axis.set(
            title=(
                f"{labels[first]} vs {labels[second]}\n"
                f"{method.title()} {symbol} = {correlations[first, second]:.3f}"
            ),
            xlabel=f"{labels[first]}: {axis_label}",
            ylabel=f"{labels[second]}: {axis_label}",
            xlim=(low - padding, high + padding),
            ylim=(low - padding, high + padding),
        )
        axis.grid(color="#e5e7eb", linewidth=0.5)
    for axis in axes.flat[len(pairs):]:
        axis.set_visible(False)
    fig.suptitle(
        f"Within-plate replicate expression: {plate} / {compound}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_within_plate_correlation_heatmap(
    dsd: DrugSeqData,
    plate: str,
    compound: str,
    method: str = "pearson",
    plate_col: str = "plate_id",
    group_col: str = "compound",
    label_col: str = "well_id",
    use_norm: bool = False,
    figsize: tuple[float, float] = (7.2, 6.2),
) -> plt.Figure:
    """Plot one raw-count or normalized replicate correlation matrix."""
    method = str(method).casefold()
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be 'pearson' or 'spearman'.")
    matrix, obs = _select_replicate_group(
        dsd,
        plate=plate,
        compound=compound,
        plate_col=plate_col,
        group_col=group_col,
        use_norm=use_norm,
    )
    correlations = _sample_correlation_matrix(matrix, method)
    labels = (
        obs[label_col].astype(str).tolist()
        if label_col in obs.columns else obs["_sample_id"].astype(str).tolist()
    )
    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(correlations, cmap="RdYlBu_r", vmin=-1, vmax=1)
    expression_label = (
        "normalized expression" if use_norm else "log10(raw UMI + 1)"
    )
    ax.set(
        title=(
            f"{plate} / {compound}: {method.title()} replicate correlation\n"
            f"{expression_label}"
        ),
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
    )
    ax.tick_params(axis="x", rotation=45)
    if len(labels) <= 12:
        for row in range(len(labels)):
            for column in range(len(labels)):
                value = correlations[row, column]
                color = "white" if abs(value) > 0.65 else "#203033"
                ax.text(
                    column, row, f"{value:.2f}", ha="center", va="center",
                    fontsize=8, color=color,
                )
    colorbar = fig.colorbar(image, ax=ax, shrink=0.82)
    colorbar.set_label(f"{method.title()} correlation")
    fig.tight_layout()
    return fig


def compute_cross_plate_reproducibility(
    dsd: DrugSeqData,
    plate_col: str = "plate_id",
    group_col: str = "compound",
    reference: str = "DMSO",
    exclude_levels: tuple[str, ...] | list[str] = ("water",),
    method: str = "limma_voom",
    min_replicates: int = 2,
    fdr_threshold: float = 0.05,
    lfc_threshold: float = 0.5,
) -> dict:
    """Fit each plate independently and compare treatment logFC signatures.

    Returns a dictionary containing a tidy ``summary`` table, per-plate DE
    ``plate_results``, and gene-level pairwise ``gene_results`` used by the
    dedicated reproducibility plots.
    """
    dsd = _coerce_dsd(dsd)
    for column in (plate_col, group_col):
        if column not in dsd.obs.columns:
            raise KeyError(f"'{column}' not found in obs.")

    plates = sorted(dsd.obs[plate_col].astype(str).unique(), key=_natural_key)
    if len(plates) < 2:
        raise ValueError("Cross-plate reproducibility requires at least two plates.")
    excluded = {str(value) for value in exclude_levels}
    excluded.add(str(reference))
    groups = sorted(
        set(dsd.obs[group_col].dropna().astype(str)) - excluded,
        key=_natural_key,
    )
    if not groups:
        raise ValueError("No treatment groups remain after exclusions.")

    plate_results: dict[tuple[str, str], pd.DataFrame] = {}
    for plate in plates:
        plate_dsd = dsd.subset(
            obs_mask=dsd.obs[plate_col].astype(str).eq(plate).to_numpy()
        )
        for group in groups:
            result = run_comparison(
                plate_dsd,
                case=group,
                control=reference,
                group_col=group_col,
                method=method,
                within_plate=False,
                min_replicates=min_replicates,
                fdr_threshold=fdr_threshold,
                lfc_threshold=lfc_threshold,
            )
            group_mask = plate_dsd.obs[group_col].astype(str).eq(group).to_numpy()
            raw = plate_dsd.layers["counts"][group_mask]
            normalized = plate_dsd.X[group_mask]
            raw_mean = np.asarray(raw.mean(axis=0)).ravel()
            normalized_mean = np.asarray(normalized.mean(axis=0)).ravel()
            expression = pd.DataFrame({
                "gene": plate_dsd.var_names.astype(str),
                "log10_mean_umi": np.log10(raw_mean + 1),
                "mean_normalized_expression": normalized_mean,
            })
            plate_results[(plate, group)] = result.merge(
                expression, on="gene", how="left", validate="one_to_one"
            )

    rows = []
    gene_results: dict[tuple[str, str, str], pd.DataFrame] = {}
    for first, second in combinations(plates, 2):
        for group in groups:
            first_result = plate_results[(first, group)][
                [
                    "gene", "logFC", "significant", "log10_mean_umi",
                    "mean_normalized_expression",
                ]
            ]
            second_result = plate_results[(second, group)][
                [
                    "gene", "logFC", "significant", "log10_mean_umi",
                    "mean_normalized_expression",
                ]
            ]
            merged = first_result.merge(
                second_result,
                on="gene",
                suffixes=(f"_{first}", f"_{second}"),
            )
            gene_results[(first, second, group)] = merged
            lfc_first = merged[f"logFC_{first}"].astype(float)
            lfc_second = merged[f"logFC_{second}"].astype(float)
            sig_first = merged[f"significant_{first}"].astype(bool)
            sig_second = merged[f"significant_{second}"].astype(bool)
            union = sig_first | sig_second
            overlap = sig_first & sig_second
            rows.append({
                "plate_a": first,
                "plate_b": second,
                "compound": group,
                "n_genes": len(merged),
                "pearson_logfc": float(pearsonr(lfc_first, lfc_second).statistic),
                "spearman_logfc": float(spearmanr(lfc_first, lfc_second).statistic),
                "n_significant_a": int(sig_first.sum()),
                "n_significant_b": int(sig_second.sum()),
                "n_significant_union": int(union.sum()),
                "n_significant_overlap": int(overlap.sum()),
                "significant_jaccard": (
                    float(overlap.sum() / union.sum()) if union.any() else np.nan
                ),
                "sign_concordance_union": (
                    float((np.sign(lfc_first[union]) ==
                           np.sign(lfc_second[union])).mean())
                    if union.any() else np.nan
                ),
            })

    summary = pd.DataFrame(rows).sort_values(
        ["plate_a", "plate_b", "compound"]
    ).reset_index(drop=True)
    return {
        "summary": summary,
        "plate_results": plate_results,
        "gene_results": gene_results,
        "config": {
            "plate_col": plate_col,
            "group_col": group_col,
            "reference": reference,
            "exclude_levels": list(exclude_levels),
            "method": method,
            "fdr_threshold": fdr_threshold,
            "lfc_threshold": lfc_threshold,
        },
    }


def _single_plate_de_table(
    reproducibility: dict,
    plate: str,
    compound: str,
) -> pd.DataFrame:
    if not isinstance(reproducibility, dict):
        raise TypeError("reproducibility must be the compute function result.")
    key = (str(plate), str(compound))
    if key not in reproducibility.get("plate_results", {}):
        available = sorted(reproducibility.get("plate_results", {}).keys())
        raise KeyError(f"No single-plate DE result for {key}. Available: {available}")
    return reproducibility["plate_results"][key].copy()


def plot_single_plate_de_volcano(
    reproducibility: dict,
    plate: str,
    compound: str,
    fdr_threshold: float = 0.05,
    lfc_threshold: float = 0.5,
    n_label: int = 5,
    figsize: tuple[float, float] = (7, 5.8),
) -> plt.Figure:
    """Plot one independently fitted plate/compound DE result."""
    table = _single_plate_de_table(reproducibility, plate, compound)
    table["minus_log10_padj"] = -np.log10(
        table["padj"].astype(float).clip(lower=1e-300)
    )
    significant = (
        table["padj"].astype(float).lt(fdr_threshold)
        & table["logFC"].astype(float).abs().ge(lfc_threshold)
    )
    direction = np.where(
        significant & table["logFC"].gt(0), "Up",
        np.where(significant & table["logFC"].lt(0), "Down", "Not significant"),
    )
    colors = {
        "Not significant": "#aeb9bc",
        "Up": "#b8473b",
        "Down": "#2f78a8",
    }
    fig, ax = plt.subplots(figsize=figsize)
    for label in ("Not significant", "Up", "Down"):
        mask = direction == label
        ax.scatter(
            table.loc[mask, "logFC"],
            table.loc[mask, "minus_log10_padj"],
            s=7 if label == "Not significant" else 11,
            color=colors[label],
            alpha=0.28 if label == "Not significant" else 0.68,
            edgecolors="none",
            label=label,
        )
    ax.axvline(-lfc_threshold, color="#6e7b7f", linestyle="--", linewidth=0.7)
    ax.axvline(lfc_threshold, color="#6e7b7f", linestyle="--", linewidth=0.7)
    ax.axhline(
        -np.log10(fdr_threshold), color="#6e7b7f", linestyle="--", linewidth=0.7
    )
    if n_label > 0:
        maximum = float(table["minus_log10_padj"].max())
        ax.set_ylim(top=maximum + max(0.8, maximum * 0.08))
        top = table.loc[significant].nsmallest(n_label, "padj")
        for index, row in enumerate(top.itertuples()):
            ax.annotate(
                row.gene, (row.logFC, row.minus_log10_padj),
                xytext=(0.02, 0.97 - 0.035 * index),
                textcoords="axes fraction",
                ha="left",
                va="top",
                fontsize=6,
                arrowprops={"arrowstyle": "-", "color": "#6e7b7f", "lw": 0.4},
            )
    reference = reproducibility.get("config", {}).get("reference", "control")
    ax.set(
        title=f"{plate}: {compound} vs {reference}",
        xlabel="log2 fold change",
        ylabel="-log10(adjusted p-value)",
    )
    ax.legend(frameon=False, fontsize=8)
    ax.grid(color="#e4e9ea", linewidth=0.5)
    fig.tight_layout()
    return fig


def _top_single_plate_de_genes(
    table: pd.DataFrame,
    n_top_genes: int,
) -> pd.DataFrame:
    significant = table.loc[table["significant"].astype(bool)].copy()
    candidates = significant if not significant.empty else table.copy()
    return candidates.sort_values(
        ["padj", "logFC"], ascending=[True, False]
    ).head(n_top_genes)


def plot_single_plate_de_heatmap(
    dsd: DrugSeqData,
    reproducibility: dict,
    plate: str,
    compound: str,
    plate_col: str = "plate_id",
    group_col: str = "compound",
    n_top_genes: int = 30,
    use_norm: bool = True,
    figsize: tuple[float, float] = (11, 7),
) -> plt.Figure:
    """Plot top single-plate DEGs across treatment and reference wells."""
    dsd = _coerce_dsd(dsd)
    table = _single_plate_de_table(reproducibility, plate, compound)
    top = _top_single_plate_de_genes(table, n_top_genes)
    reference = reproducibility.get("config", {}).get("reference", "DMSO")
    sample_mask = (
        dsd.obs[plate_col].astype(str).eq(str(plate))
        & dsd.obs[group_col].astype(str).isin([str(reference), str(compound)])
    ).to_numpy()
    gene_index = dsd.var_names.astype(str).get_indexer(top["gene"].astype(str))
    valid = gene_index >= 0
    gene_index = gene_index[valid]
    genes = top.loc[valid, "gene"].astype(str).tolist()
    if not len(gene_index):
        raise ValueError("No selected DE genes were found in the expression matrix.")
    matrix = dsd.X if use_norm else dsd.layers["counts"]
    matrix = matrix[sample_mask][:, gene_index]
    if sp.issparse(matrix):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=float)
    if not use_norm:
        matrix = np.log10(matrix + 1)
    gene_by_sample = matrix.T
    means = gene_by_sample.mean(axis=1, keepdims=True)
    standard_deviations = gene_by_sample.std(axis=1, keepdims=True)
    z_scores = (gene_by_sample - means) / np.where(
        standard_deviations == 0, 1, standard_deviations
    )
    sample_obs = dsd.obs.loc[sample_mask]
    sample_labels = (
        sample_obs["well_id"].astype(str).tolist()
        if "well_id" in sample_obs.columns else sample_obs.index.astype(str).tolist()
    )
    sample_groups = sample_obs[group_col].astype(str).tolist()

    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(z_scores, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    ax.set(
        title=f"{plate}: top {compound} DEGs",
        xticks=np.arange(len(sample_labels)),
        yticks=np.arange(len(genes)),
        xticklabels=[
            f"{well}\n{group}" for well, group in zip(sample_labels, sample_groups)
        ],
        yticklabels=genes,
        xlabel="Within-plate treatment and reference wells",
        ylabel="Gene",
    )
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.8)
    colorbar.set_label("Per-gene expression z-score")
    fig.tight_layout()
    return fig


def plot_single_plate_de_layout(
    dsd: DrugSeqData,
    reproducibility: dict,
    plate: str,
    compound: str,
    plate_col: str = "plate_id",
    n_top_genes: int = 30,
    nrow: int = 8,
    ncol: int = 12,
) -> plt.Figure:
    """Map a signed top-DEG expression score to physical wells on one plate."""
    dsd = _coerce_dsd(dsd)
    table = _single_plate_de_table(reproducibility, plate, compound)
    top = _top_single_plate_de_genes(table, n_top_genes)
    gene_index = dsd.var_names.astype(str).get_indexer(top["gene"].astype(str))
    valid = gene_index >= 0
    gene_index = gene_index[valid]
    top = top.loc[valid].copy()
    if not len(gene_index):
        raise ValueError("No selected DE genes were found in the expression matrix.")
    plate_mask = dsd.obs[plate_col].astype(str).eq(str(plate)).to_numpy()
    expression = dsd.X[plate_mask][:, gene_index]
    if sp.issparse(expression):
        expression = expression.toarray()
    expression = np.asarray(expression, dtype=float)
    means = expression.mean(axis=0, keepdims=True)
    standard_deviations = expression.std(axis=0, keepdims=True)
    z_scores = (expression - means) / np.where(
        standard_deviations == 0, 1, standard_deviations
    )
    direction = np.sign(top["logFC"].to_numpy(dtype=float))
    score = np.nanmean(z_scores * direction, axis=1)
    plate_obs = dsd.obs.loc[plate_mask].copy()
    plate_obs["de_signature_score"] = score
    fig, _ = plot_plate_layout(
        plate_obs,
        value_col="de_signature_score",
        plate_id=plate_col,
        continuous=True,
        log_values=False,
        nrow=nrow,
        ncol=ncol,
        cmap="RdBu_r",
        show_legend=True,
    )
    fig.suptitle(
        f"{plate}: {compound} signed DEG-expression score by well",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    return fig


def plot_cross_plate_reproducibility(
    reproducibility: pd.DataFrame | dict,
    correlation_threshold: float = 0.5,
    correlation_method: str = "pearson",
    figsize: tuple[float, float] = (13, 6.2),
) -> plt.Figure:
    """Plot signature correlation and DEG overlap from one summary table."""
    table = (reproducibility["summary"]
             if isinstance(reproducibility, dict) else reproducibility)
    correlation_method = str(correlation_method).casefold()
    if correlation_method not in {"pearson", "spearman"}:
        raise ValueError("correlation_method must be 'pearson' or 'spearman'.")
    correlation_col = f"{correlation_method}_logfc"
    required = {
        "compound", correlation_col, "significant_jaccard",
        "n_significant_union",
    }
    missing = required - set(table.columns)
    if missing:
        raise KeyError(f"reproducibility table is missing columns: {sorted(missing)}")

    plot_table = table.sort_values(correlation_col)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    colors = np.where(
        plot_table[correlation_col] >= correlation_threshold,
        "#147d75", "#d08a1b",
    )
    axes[0].barh(plot_table["compound"], plot_table[correlation_col],
                 color=colors)
    axes[0].axvline(
        correlation_threshold, color="#6e7b7f", linestyle="--", linewidth=0.9
    )
    finite_correlations = plot_table[correlation_col].dropna()
    lower = min(-0.05, float(finite_correlations.min()) - 0.05) \
        if not finite_correlations.empty else -0.05
    axes[0].set(
        title="Cross-plate treatment-effect correlation",
        xlabel=f"{correlation_method.title()} correlation of gene logFC",
        ylabel="",
        xlim=(lower, 1),
    )

    sizes = 26 + 1.2 * np.sqrt(
        plot_table["n_significant_union"].clip(lower=0)
    )
    axes[1].scatter(
        plot_table[correlation_col],
        plot_table["significant_jaccard"].fillna(0),
        s=sizes,
        color="#3b7fa5",
        alpha=0.72,
        edgecolor="white",
        linewidth=0.6,
    )
    for row in plot_table.itertuples():
        if (row.n_significant_union < 10 and
                getattr(row, correlation_col) < correlation_threshold):
            continue
        axes[1].annotate(
            row.compound,
            (getattr(row, correlation_col),
             0 if pd.isna(row.significant_jaccard) else row.significant_jaccard),
            xytext=(4, 3), textcoords="offset points", fontsize=7,
        )
    axes[1].set(
        title="Effect agreement versus DEG overlap",
        xlabel=f"{correlation_method.title()} correlation of gene logFC",
        ylabel="Jaccard index of significant DEGs",
        xlim=(-0.05, 1),
        ylim=(-0.02, 1.02),
    )
    for axis in axes:
        axis.grid(color="#e1e7e8", linestyle="--", linewidth=0.6)
    fig.tight_layout()
    return fig


def plot_plate_logfc_concordance(
    reproducibility: dict,
    compounds: list[str] | None = None,
    max_panels: int = 4,
    value_type: str = "log2fc",
    correlation_method: str = "pearson",
    figsize: tuple[float, float] = (11.5, 10.5),
) -> plt.Figure:
    """Plot plate concordance using effects, raw UMI, or normalized expression."""
    if not isinstance(reproducibility, dict):
        raise TypeError("reproducibility must be the compute function result.")
    value_type = str(value_type).casefold()
    value_aliases = {
        "log2fc": "log2fc",
        "logfc": "log2fc",
        "log10_umi": "log10_umi",
        "log10(umi+1)": "log10_umi",
        "normalized": "normalized",
        "normalized_expression": "normalized",
    }
    if value_type not in value_aliases:
        raise ValueError(
            "value_type must be 'log2fc', 'log10_umi', or 'normalized'."
        )
    value_type = value_aliases[value_type]
    correlation_method = str(correlation_method).casefold()
    if correlation_method not in {"pearson", "spearman"}:
        raise ValueError("correlation_method must be 'pearson' or 'spearman'.")
    table = reproducibility["summary"]
    gene_results = reproducibility["gene_results"]
    selected = table.copy()
    if compounds is not None:
        selected = selected[selected["compound"].isin(list(compounds))]
    selected = selected.sort_values(
        "n_significant_union", ascending=False
    ).head(max_panels)
    if selected.empty:
        raise ValueError("No cross-plate comparisons are available to plot.")

    n_panels = len(selected)
    ncols = min(2, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for axis, row in zip(axes.flat, selected.itertuples(index=False)):
        key = (str(row.plate_a), str(row.plate_b), str(row.compound))
        merged = gene_results[key]
        column, axis_name = {
            "log2fc": ("logFC", "log2FC"),
            "log10_umi": ("log10_mean_umi", "log10(mean UMI + 1)"),
            "normalized": (
                "mean_normalized_expression", "Mean normalized expression",
            ),
        }[value_type]
        x = merged[f"{column}_{row.plate_a}"].astype(float)
        y = merged[f"{column}_{row.plate_b}"].astype(float)
        significant_a = merged[f"significant_{row.plate_a}"].astype(bool)
        significant_b = merged[f"significant_{row.plate_b}"].astype(bool)
        categories = [
            ("Neither", ~significant_a & ~significant_b, "#aeb9bc", 5, 0.18),
            (f"{row.plate_a} only", significant_a & ~significant_b,
             "#2f78a8", 9, 0.62),
            (f"{row.plate_b} only", ~significant_a & significant_b,
             "#d8892b", 9, 0.62),
            ("Shared", significant_a & significant_b, "#8b3f91", 11, 0.72),
        ]
        for label, mask, color, size, alpha in categories:
            if mask.any():
                axis.scatter(
                    x[mask], y[mask], s=size, color=color, alpha=alpha,
                    edgecolors="none", label=label,
                )
        low = float(min(x.min(), y.min()))
        high = float(max(x.max(), y.max()))
        axis.plot([low, high], [low, high], color="#526267",
                  linestyle="--", linewidth=0.8)
        correlation = (
            pearsonr(x, y).statistic
            if correlation_method == "pearson" else spearmanr(x, y).statistic
        )
        symbol = "r" if correlation_method == "pearson" else "rho"
        axis.set(
            title=(
                f"{row.compound}: {correlation_method.title()} "
                f"{symbol} = {correlation:.2f}"
            ),
            xlabel=f"{row.plate_a} {axis_name}",
            ylabel=f"{row.plate_b} {axis_name}",
        )
        axis.grid(color="#e4e9ea", linewidth=0.5)
        axis.legend(frameon=False, fontsize=7, loc="best")
    for axis in axes.flat[n_panels:]:
        axis.set_visible(False)
    fig.suptitle(
        f"Independent plate-specific concordance: {axis_name}",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig
