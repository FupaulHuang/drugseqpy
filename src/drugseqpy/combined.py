"""Combined-library QC figures with shared wells and a fixed species baseline.

The calculations follow the S6/S7 combined-figure workflow. Inputs are AnnData
or DrugSeqData, with observations as wells and variables as feature rows.
Plot functions return Matplotlib figures; the report also exports source tables.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, StrMethodFormatter
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import gaussian_kde
import seaborn as sns

from .core import DrugSeqData
from .plate import plot_plate_layout


HUMAN_MARKERS = (
    "VIM", "FN1", "CDH2", "CD44", "EGFR", "MMP9", "TGFB1", "SNAI1", "TWIST1", "ITGB1",
)
MOUSE_3T3_MARKERS = (
    "Pdgfra", "Col1a1", "Col1a2", "Col3a1", "Col6a1", "Vim", "Fn1", "Sparc", "Thy1", "Dcn",
)
HUMAN_HOUSEKEEPING = (
    "GAPDH", "ACTB", "B2M", "PPIA", "RPLP0", "HPRT1", "TBP", "PGK1", "UBC", "GUSB",
    "LDHA", "RPL19", "RPL11",
)
MOUSE_HOUSEKEEPING = (
    "Gapdh", "Actb", "B2m", "Ppia", "Rplp0", "Hprt1", "Tbp", "Pgk1", "Ubc", "Gusb",
)
SPECIES_COLORS = {"human": "#00B4D8", "mouse": "#0A192F", "Excluded": "#E2E8F0"}
READ_METRICS = {
    "Valid Barcodes": "Reads With Valid Barcodes",
    "Q30 CB+UMI": "Q30 Bases in CB+UMI",
    "Q30 RNA": "Q30 Bases in RNA read",
    "Mapped to Genome": "Reads Mapped to Genome: Unique",
    "Mapped to Gene": "Reads Mapped to Gene: Unique Gene",
}
RING_COLORS = ["#A8E6CF", "#FFD3B6", "#A8D8EA", "#D4A5D9", "#F7A8B8"]
COMPOSITION_LABELS = {
    "pct_mito": "Mitochondrial UMIs (%)",
    "pct_ribo": "Cytosolic ribosomal UMIs (%)",
    "human_housekeeping_mean_umi": "Human housekeeping mean UMI - human wells",
    "mouse_housekeeping_mean_umi": "Mouse housekeeping mean UMI - mouse wells",
}


def _sum(x, axis=1):
    return np.asarray(x.sum(axis=axis)).ravel()


def _ratio(numerator, denominator):
    return np.divide(numerator, denominator, out=np.full(len(denominator), np.nan),
                     where=denominator > 0)


def _require(table, columns):
    missing = set(columns) - set(table.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    if table.empty:
        raise ValueError("The plotting table is empty.")


def _medians(values):
    return values.groupby(["metric", "group"], sort=False, observed=True)["value"].agg(
        n="count", median="median",
    ).reset_index()


def compute_combined_qc(
    dsd,
    *,
    baseline_group: str,
    groupby: str = "plate_id",
    barcode_col: str = "barcode",
    filtered_col: str | None = None,
    reads_col: str = "reads_per_well",
    layer: str | None = "counts",
    gene_id_col: str | None = None,
    symbol_col: str = "gene_symbol",
    marker_sets: Mapping[str, Sequence[str]] | None = None,
    housekeeping_sets: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute auditable tables for the combined human/mouse QC figure suite.

    Supply raw counts for all wells (including the baseline's unfiltered wells)
    and, when available, a boolean ``filtered_col``. The plotted population is
    the intersection of filtered barcodes across *all* groups. With no filter
    column, all supplied observations are eligible. Barcodes must be unique
    within each group and use identical spelling across groups.

    Baseline human/(human+mouse) >= 0.5 defines human; otherwise mouse. A zero
    species denominator gives an unknown baseline and is rejected for shared
    wells. Percentages use human+mouse for purity and all UMIs for mt/ribo.
    Human/mouse feature IDs start with ENSG/ENSMUSG. ``gene_id_col=None`` uses
    var['gene_id'] when present, otherwise var_names. ``layer=None`` explicitly
    selects X; the default requires raw counts in layers['counts'].

    Feature distributions retain duplicate symbols as separate rows. Marker
    and housekeeping symbols use the first matching feature, as in the source
    R script. Missing housekeeping genes raise; missing markers warn and are
    omitted. Housekeeping means include zeros and are plotted only in the
    corresponding baseline species. Inputs are never modified or densified.
    """
    adata = dsd.adata if isinstance(dsd, DrugSeqData) else dsd
    obs = adata.obs.copy()
    _require(obs, [groupby, barcode_col, reads_col])
    for col in (groupby, barcode_col):
        if obs[col].isna().any() or obs[col].astype(str).eq("").any():
            raise ValueError(f"{col!r} must contain nonempty identifiers.")
    obs["group"] = obs[groupby].astype(str)
    obs["barcode"] = obs[barcode_col].astype(str)
    if obs.duplicated(["group", "barcode"]).any():
        raise ValueError("Barcodes must be unique within each group.")
    groups = list(pd.unique(obs["group"]))
    baseline_group = str(baseline_group)
    if baseline_group not in groups:
        raise ValueError(f"Baseline group {baseline_group!r} is absent.")
    if layer is not None and layer not in adata.layers:
        raise ValueError(f"Raw count layer {layer!r} is absent; use layer=None only for raw X.")
    x = adata.X if layer is None else adata.layers[layer]
    x = x.tocsr(copy=True) if sparse.issparse(x) else np.asarray(x)
    if sparse.issparse(x):
        x.sum_duplicates()
    vals = x.data if sparse.issparse(x) else x
    if not np.isfinite(vals).all() or (vals < 0).any() or (vals != np.floor(vals)).any():
        raise ValueError("Expected finite, nonnegative integer raw UMI counts.")
    reads = pd.to_numeric(obs[reads_col], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(reads).all() or (reads < 0).any():
        raise ValueError("Read counts must be finite and nonnegative.")
    obs["reads_per_well"] = reads
    if filtered_col is None:
        eligible = np.ones(len(obs), dtype=bool)
    else:
        _require(obs, [filtered_col])
        flag = obs[filtered_col]
        if flag.isna().any() or not pd.api.types.is_bool_dtype(flag):
            raise ValueError("filtered_col must be boolean without missing values.")
        eligible = flag.to_numpy(dtype=bool)
    shared = set.intersection(*[
        set(obs.loc[eligible & obs["group"].eq(g), "barcode"]) for g in groups
    ])
    if not shared:
        raise ValueError("No shared eligible barcodes across groups.")
    selected = eligible & obs["barcode"].isin(shared).to_numpy()
    obs["eligible"] = eligible
    obs["shared_well"] = selected
    if gene_id_col is None:
        ids = adata.var.get("gene_id", pd.Series(adata.var_names, index=adata.var_names))
    else:
        if gene_id_col not in adata.var:
            raise ValueError(f"Missing var column {gene_id_col!r}.")
        ids = adata.var[gene_id_col]
    ids = pd.Series(np.asarray(ids)).fillna("").astype(str)
    if symbol_col not in adata.var:
        raise ValueError(f"Missing gene-symbol column {symbol_col!r} in var.")
    symbols = pd.Series(adata.var[symbol_col].to_numpy()).fillna("").astype(str)
    human = ids.str.startswith("ENSG").to_numpy()
    mouse = ids.str.startswith("ENSMUSG").to_numpy()
    if not human.any() or not mouse.any():
        raise ValueError("Both ENSG human and ENSMUSG mouse feature IDs are required.")
    masks = {
        "mitochondrial": symbols.str.match(r"^(MT-|mt-)").to_numpy(),
        "ribosomal": symbols.str.match(
            r"^(RPL[0-9]+|RPS[0-9]+|RPLP[0-9]+|RPSA|Rpl[0-9]+|Rps[0-9]+|Rplp[0-9]+|Rpsa)$"
        ).to_numpy(),
    }
    obs["total_umi"] = _sum(x)
    obs["n_genes_det"] = _sum(x > 0)
    obs["human_umi"] = _sum(x[:, human])
    obs["mouse_umi"] = _sum(x[:, mouse])
    species_total = (obs["human_umi"] + obs["mouse_umi"]).to_numpy()
    obs["human_fraction"] = _ratio(obs["human_umi"].to_numpy(), species_total)
    obs["mouse_fraction"] = _ratio(obs["mouse_umi"].to_numpy(), species_total)
    base = obs.loc[obs["group"].eq(baseline_group), ["barcode", "human_fraction"]].copy()
    base["baseline_species"] = np.where(base["human_fraction"] >= .5, "human", "mouse")
    base.loc[base["human_fraction"].isna(), "baseline_species"] = pd.NA
    obs["baseline_species"] = obs["barcode"].map(base.set_index("barcode")["baseline_species"])
    if obs.loc[selected, "baseline_species"].isna().any():
        raise ValueError("Shared wells have missing or zero-count baseline species assignments.")
    total = obs["total_umi"].to_numpy()
    for name, pct in (("mitochondrial", "pct_mito"), ("ribosomal", "pct_ribo")):
        obs[f"{name}_umi"] = _sum(x[:, masks[name]])
        obs[pct] = 100 * _ratio(obs[f"{name}_umi"].to_numpy(), total)
    hk = housekeeping_sets if housekeeping_sets is not None else {
        "human": HUMAN_HOUSEKEEPING, "mouse": MOUSE_HOUSEKEEPING,
    }
    if set(hk) != {"human", "mouse"}:
        raise ValueError("housekeeping_sets must define human and mouse gene sets.")
    symbol_index = {}
    for i, symbol in enumerate(symbols):
        symbol_index.setdefault(symbol, i)
    for species, genes in hk.items():
        genes = list(dict.fromkeys(genes))
        missing = [g for g in genes if g not in symbol_index]
        if not genes or missing:
            raise ValueError(f"Missing {species} housekeeping genes: {missing}; supply a complete set.")
        idx = np.array([symbol_index[g] for g in genes])
        if not (human if species == "human" else mouse)[idx].all():
            raise ValueError(f"{species} housekeeping genes do not match their species feature IDs.")
        mask = np.zeros(adata.n_vars, dtype=bool)
        mask[idx] = True
        masks[f"housekeeping_{species}"] = mask
        obs[f"{species}_housekeeping_mean_umi"] = _sum(x[:, idx]) / len(idx)
    symbol_mask = (symbols.ne("") & ~symbols.str.startswith("ENS")).to_numpy()
    ensembl_mask = ids.str.startswith("ENS").to_numpy()
    obs["detected_gene_symbol"] = _sum(x[:, symbol_mask] > 0)
    obs["detected_ensembl_id"] = _sum(x[:, ensembl_mask] > 0)
    wells = obs.loc[selected].copy()
    totals = []
    marker_rows = []
    panels = marker_sets if marker_sets is not None else {
        "Human_MDA-MB-231_Markers": HUMAN_MARKERS, "Mouse_3T3_Markers": MOUSE_3T3_MARKERS,
    }
    for panel, genes in panels.items():
        missing = [g for g in genes if g not in symbol_index]
        if missing:
            warnings.warn(f"{panel}: missing markers {missing}", UserWarning, stacklevel=2)
    for group in groups:
        rows = selected & obs["group"].eq(group).to_numpy()
        feature_totals = _sum(x[rows], axis=0)
        frame = pd.DataFrame({"group": group, "gene_id": ids, "gene_symbol": symbols,
                              "total_umi": feature_totals})
        totals.append(frame.loc[frame["total_umi"] > 0])
        for panel, genes in panels.items():
            for gene in dict.fromkeys(genes):
                if gene not in symbol_index:
                    continue
                for species in ("human", "mouse"):
                    subset = rows & obs["baseline_species"].eq(species).fillna(False).to_numpy()
                    if subset.any():
                        marker_rows.append({
                            "panel": panel, "group": group, "baseline_species": species,
                            "gene": gene, "mean_umi": float(x[subset, symbol_index[gene]].mean()),
                            "n_wells": int(subset.sum()), "gene_id": ids.iloc[symbol_index[gene]],
                        })
    purity_rows = []
    for (group, species), sub in wells.groupby(["group", "baseline_species"], sort=False):
        values = sub[f"{species}_fraction"]
        purity_rows.append({"group": group, "baseline_species": species, "n_wells": len(sub),
                            "n_valid": int(values.notna().sum()), "min": values.min(),
                            "max": values.max(), "median": values.median(), "mean": values.mean()})
    keys = ["group", "barcode", "baseline_species"]
    qc = wells.melt(id_vars=keys, value_vars=["reads_per_well", "total_umi", "n_genes_det"],
                    var_name="metric", value_name="value")
    composition = []
    for metric in COMPOSITION_LABELS:
        sub = wells
        if "housekeeping" in metric:
            sub = sub.loc[sub["baseline_species"].eq(metric.split("_")[0])]
        composition.append(sub[keys + [metric]].rename(columns={metric: "value"}).assign(metric=metric))
    composition = pd.concat(composition, ignore_index=True)
    gene_sets = pd.concat([
        pd.DataFrame({"gene_set": name, "gene_id": ids[mask], "gene_symbol": symbols[mask]})
        for name, mask in masks.items()
    ], ignore_index=True)
    return {
        "wells": wells, "all_wells": obs, "baseline": base,
        "excluded_wells": obs.loc[~selected].copy(),
        "gene_totals": pd.concat(totals, ignore_index=True),
        "markers": pd.DataFrame(marker_rows, columns=[
            "panel", "group", "baseline_species", "gene", "mean_umi", "n_wells", "gene_id",
        ]),
        "purity_statistics": pd.DataFrame(purity_rows), "gene_sets": gene_sets,
        "qc_values": qc, "qc_medians": _medians(qc),
        "composition_values": composition, "composition_medians": _medians(composition),
    }


def read_star_summary_metrics(summary_files: Mapping[str, str | Path]) -> pd.DataFrame:
    """Read full-library STARsolo Summary.csv files into group/metric/percent rows.

    Values in Summary.csv must be fractions in [0, 1]. Q30 metrics describe
    bases; the other metrics describe reads. They are not shared-well metrics.
    """
    rows = []
    for group, path in summary_files.items():
        frame = pd.read_csv(path, header=None, names=["key", "value"])
        if frame["key"].duplicated().any():
            raise ValueError(f"Duplicate Summary.csv keys in {path}.")
        values = frame.set_index("key")["value"]
        for label, key in READ_METRICS.items():
            if key not in values:
                raise ValueError(f"Missing STAR summary metric {key!r} in {path}.")
            fraction = float(values[key])
            if not np.isfinite(fraction) or not 0 <= fraction <= 1:
                raise ValueError(f"STAR summary {key!r} must be a fraction in [0, 1].")
            rows.append({"group": str(group), "metric": label, "percent": 100 * fraction})
    return pd.DataFrame(rows, columns=["group", "metric", "percent"])


def _colors(groups, palette=None):
    defaults = dict(zip(groups, sns.color_palette("husl", len(groups)).as_hex()))
    if isinstance(palette, Mapping):
        defaults.update(palette)
    elif palette is not None:
        defaults = dict(zip(groups, sns.color_palette(palette, len(groups)).as_hex()))
    return defaults


def _grid(n, ncols=2, figsize=None):
    if n < 1:
        raise ValueError("No panels to plot.")
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=figsize or (5 * ncols, 4 * nrows))
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, list(axes.flat[:n])


def _finish(fig, title):
    for ax in fig.axes:
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E2E8F0", linestyle="--", linewidth=.6)
    fig.suptitle(title, weight="bold", color="#1E293B")
    fig.tight_layout(rect=(0, 0, 1, .95))
    return fig


def plot_combined_barnyard(wells, *, palette=None, figsize=None):
    """Plot human/mouse UMIs with separate baseline-species purity statistics."""
    _require(wells, ["group", "baseline_species", "human_umi", "mouse_umi",
                     "human_fraction", "mouse_fraction"])
    groups = list(pd.unique(wells["group"]))
    colors = {**SPECIES_COLORS, **(palette or {})}
    fig, axes = _grid(len(groups), figsize=figsize or (10, 6.5))
    xlimit = max(wells["human_umi"].max(), 1) * 1.08
    ylimit = max(wells["mouse_umi"].max(), 1) * 1.08
    for ax, group in zip(axes, groups):
        sub = wells.loc[wells["group"].eq(group)]
        lines = []
        for species in ("human", "mouse"):
            one = sub.loc[sub["baseline_species"].eq(species)]
            ax.scatter(one["human_umi"], one["mouse_umi"], color=colors[species],
                       label=species, s=20, alpha=.6)
            purity = 100 * one[f"{species}_fraction"].dropna()
            if len(one):
                lines.append(f"{species}: n={len(one)}\n"
                             f"min {purity.min():.2f}%, max {purity.max():.2f}%\n"
                             f"median {purity.median():.2f}%, mean {purity.mean():.2f}%")
        limit = min(xlimit, ylimit)
        ax.plot([0, limit], [0, limit], "--", color="grey", lw=.8)
        ax.set(xlim=(0, xlimit), ylim=(0, ylimit), title=str(group),
               xlabel="Human UMI counts", ylabel="Mouse UMI counts")
        ax.text(.98, .98, "\n".join(lines), transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color="#D32F2F", bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none"})
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.tick_params(axis="x", labelrotation=25)
        ax.legend(title="Baseline species", fontsize=8, loc="lower right")
    return _finish(fig, "Combined barnyard plot")


def plot_combined_species_purity(wells, *, palette=None, figsize=None):
    """Plot human versus mouse UMI percentages in each library."""
    _require(wells, ["group", "baseline_species", "human_fraction", "mouse_fraction"])
    groups = list(pd.unique(wells["group"]))
    fig, axes = _grid(len(groups), figsize=figsize or (9, 6))
    colors = {**SPECIES_COLORS, **(palette or {})}
    for ax, group in zip(axes, groups):
        for species in ("human", "mouse"):
            sub = wells.loc[wells["group"].eq(group) & wells["baseline_species"].eq(species)]
            ax.scatter(100 * sub["human_fraction"], 100 * sub["mouse_fraction"],
                       color=colors[species], label=species, alpha=.7, s=30)
        ax.set(xlim=(0, 103), ylim=(0, 103), xlabel="Human UMIs (%)", ylabel="Mouse UMIs (%)",
               title=str(group))
        ax.legend(title="Baseline species")
    return _finish(fig, "Species purity")


def _density(ax, values, color, label=None, count=False):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) > 1 and np.ptp(values) > 0:
        kde = gaussian_kde(values, bw_method="silverman")
        bandwidth = float(np.sqrt(kde.covariance[0, 0]))
        grid = np.linspace(values.min() - 3 * bandwidth, values.max() + 3 * bandwidth, 512)
        y = kde(grid) * (len(values) if count else 1)
        ax.plot(grid, y, color=color, label=label)
        ax.fill_between(grid, y, color=color, alpha=.4)
    elif len(values):
        ax.axvline(values[0], color=color, label=label)
    else:
        ax.text(.5, .5, "No positive features", transform=ax.transAxes, ha="center")


def plot_combined_density(wells, metric="total_umi", *, palette=None, figsize=(9, 5.5)):
    """Overlay per-library well densities on a linear count axis."""
    _require(wells, ["group", metric])
    groups = list(pd.unique(wells["group"]))
    colors = _colors(groups, palette)
    fig, ax = plt.subplots(figsize=figsize)
    for group in groups:
        _density(ax, wells.loc[wells["group"].eq(group), metric], colors[group], str(group))
    ax.set(xlabel={"total_umi": "Total UMIs per well", "n_genes_det": "Detected features per well"}.get(metric, metric),
           ylabel="Density")
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.legend(title="Group")
    return _finish(fig, "Combined well distribution")


def plot_combined_gene_umi(gene_totals, identifier="all", *, figsize=None):
    """Count-scaled KDE of log2(feature UMI totals + 1), retaining feature rows.

    ``all`` facets by library; ``gene_symbol``/``ensembl_id`` pool the library-
    feature rows (without summing libraries), with a raw-count median annotation.
    Zero-total features are omitted, matching the reference figures.
    """
    _require(gene_totals, ["group", "gene_id", "gene_symbol", "total_umi"])
    if identifier not in {"all", "gene_symbol", "ensembl_id"}:
        raise ValueError("identifier must be all, gene_symbol, or ensembl_id.")
    data = gene_totals.loc[gene_totals["total_umi"] > 0].copy()
    if identifier == "gene_symbol":
        sym = data["gene_symbol"].fillna("").astype(str)
        data = data.loc[sym.ne("") & ~sym.str.startswith("ENS")]
    elif identifier == "ensembl_id":
        data = data.loc[data["gene_id"].astype(str).str.startswith("ENS")]
    groups = list(pd.unique(gene_totals["group"])) if identifier == "all" else [identifier]
    fig, axes = _grid(len(groups), figsize=figsize or ((10, 6) if identifier == "all" else (9, 5.5)))
    for ax, group in zip(axes, groups):
        sub = data.loc[data["group"].eq(group)] if identifier == "all" else data
        values = sub["total_umi"]
        _density(ax, np.log2(values + 1), "#69b3a2", count=True)
        if identifier != "all" and len(values):
            median = values.median()
            ax.axvline(np.log2(median + 1), color="#D32F2F", ls="--")
            ax.text(.98, .95, f"Median: {median:,.0f}\nn = {len(values):,} feature/library rows",
                    transform=ax.transAxes, ha="right", va="top", color="#D32F2F")
        ax.set(title=str(group), xlabel="Total UMI count per feature: log2(x + 1)",
               ylabel="Number of features (count-scaled density)")
    return _finish(fig, "UMIs per feature")


def plot_combined_markers(markers, panel, *, palette=None, figsize=(16, 9)):
    """Grouped mean raw-UMI bars by library and baseline species, one facet per gene."""
    _require(markers, ["panel", "group", "baseline_species", "gene", "mean_umi"])
    data = markers.loc[markers["panel"].eq(panel)]
    _require(data, [])
    genes = list(pd.unique(data["gene"]))
    groups = list(pd.unique(data["group"]))
    colors = {**SPECIES_COLORS, **(palette or {})}
    fig, axes = _grid(len(genes), ncols=5, figsize=figsize)
    for ax, gene in zip(axes, genes):
        for j, species in enumerate(("human", "mouse")):
            sub = data.loc[data["gene"].eq(gene) & data["baseline_species"].eq(species)]
            y = sub.set_index("group")["mean_umi"].reindex(groups)
            ax.bar(np.arange(len(groups)) + (j - .5) * .36, y, width=.36,
                   color=colors[species], label=species)
        ax.set_xticks(np.arange(len(groups)), groups)
        ax.set(title=gene, ylabel="Mean raw UMIs / well")
    _finish(fig, str(panel).replace("_", " "))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Baseline species", ncol=2, frameon=False,
               loc="upper center", bbox_to_anchor=(.5, .95))
    fig.subplots_adjust(top=.86)
    return fig


def plot_combined_saturation(wells, identifier="gene_symbol", *, palette=None, figsize=(9, 6)):
    """Reads-versus-detected-features scatter; this is not a downsampling curve."""
    if identifier not in {"gene_symbol", "ensembl_id"}:
        raise ValueError("identifier must be gene_symbol or ensembl_id.")
    metric = f"detected_{identifier}"
    _require(wells, ["group", "reads_per_well", metric])
    groups = list(pd.unique(wells["group"]))
    colors = _colors(groups, palette)
    fig, ax = plt.subplots(figsize=figsize)
    for group in groups:
        sub = wells.loc[wells["group"].eq(group)]
        ax.scatter(sub["reads_per_well"], sub[metric], color=colors[group], label=str(group),
                   s=25, alpha=.7)
    median = wells[metric].median()
    ax.axhline(median, color="#D32F2F", ls="--")
    ax.annotate(f"Median: {median:,.0f}", xy=(.98, median), xycoords=("axes fraction", "data"),
                xytext=(0, 5), textcoords="offset points", ha="right", color="#D32F2F")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x / 1e6:.1f}"))
    ax.set(xlabel="Total reads per well (cbMatch, millions)",
           ylabel=f"Detected features ({identifier.replace('_', ' ')})")
    ax.legend(title="Group")
    return _finish(fig, "Sequencing saturation: reads versus detected features")


def plot_combined_qc_boxplots(values, *, ncols=1, decimals=0, figsize=(8, 10), seed=20261013):
    """Plot long-form group/metric/value data with jitter and per-group medians."""
    _require(values, ["group", "metric", "value"])
    metrics = list(pd.unique(values["metric"]))
    groups = list(pd.unique(values["group"]))
    if ncols < 1:
        raise ValueError("ncols must be positive.")
    fig, axes = _grid(len(metrics), ncols=ncols, figsize=figsize)
    rng = np.random.default_rng(seed)
    labels = {"reads_per_well": "Read count (cbMatch)", "total_umi": "UMI count",
              "n_genes_det": "Detected features", **COMPOSITION_LABELS}
    for ax, metric in zip(axes, metrics):
        notes = []
        for j, group in enumerate(groups):
            vals = values.loc[values["metric"].eq(metric) & values["group"].eq(group), "value"]
            vals = vals[np.isfinite(vals)]
            if not len(vals):
                continue
            ax.boxplot([vals], positions=[j], widths=.6, showfliers=False, patch_artist=True,
                       boxprops={"facecolor": "#B2DFB2", "alpha": .8},
                       medianprops={"color": "black"})
            ax.scatter(j + rng.uniform(-.1, .1, len(vals)), vals, color="black", s=12, alpha=.6)
            notes.append(f"{group} median: {vals.median():,.{decimals}f}")
        ax.set_xticks(range(len(groups)), groups)
        ax.set_ylabel(labels.get(metric, metric))
        ax.margins(y=.25)
        ax.text(.98, .98, "\n".join(notes), transform=ax.transAxes, ha="right", va="top",
                color="#D32F2F", weight="bold", fontsize=9)
    return _finish(fig, "Combined QC metrics")


def plot_mt_ribo_housekeeping(values, *, figsize=(12, 9), seed=20261013):
    """Four-panel composition/housekeeping boxplot from composition_values."""
    return plot_combined_qc_boxplots(values, ncols=2, decimals=2, figsize=figsize, seed=seed)


def _validate_read_metrics(metrics):
    _require(metrics, ["group", "metric", "percent"])
    if metrics.duplicated(["group", "metric"]).any():
        raise ValueError("Duplicate library/read-metric rows.")
    for group, sub in metrics.groupby("group", observed=True):
        if set(sub["metric"]) != set(READ_METRICS):
            raise ValueError(f"Expected all five STAR read/base metrics for {group}.")
    values = metrics["percent"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 100).any():
        raise ValueError("Read/base percentages must be finite and in [0, 100].")


def plot_combined_read_doughnut(metrics, *, figsize=(11, 4.5)):
    """Concentric STAR read/base percentages with arcs on a fixed 0–100% scale.

    Unlike the reference's auto-scaled polar axis, every ring here represents
    a complete 100% circle. Printed percentages are the unaltered STAR values.
    """
    _validate_read_metrics(metrics)
    groups = list(pd.unique(metrics["group"]))
    fig, axes = _grid(len(groups), ncols=len(groups), figsize=figsize)
    for ax, group in zip(axes, groups):
        values = metrics.loc[metrics["group"].eq(group)].set_index("metric")["percent"]
        for i, (label, color) in enumerate(zip(READ_METRICS, RING_COLORS)):
            pct = float(values[label])
            radius = 1 - i * .16
            ax.pie([pct, 100 - pct], radius=radius, colors=[color, "#F1F5F9"], startangle=90,
                   counterclock=False, wedgeprops={"width": .135, "edgecolor": "white"})
            angle = np.deg2rad(90 - 1.8 * pct)
            ax.text((radius - .0675) * np.cos(angle), (radius - .0675) * np.sin(angle),
                    f"{pct:.1f}%", ha="center", va="center", fontsize=8, weight="bold")
        ax.set_title(str(group), weight="bold")
    fig.legend([Patch(facecolor=c) for c in RING_COLORS], list(READ_METRICS),
               loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Full-library read and base-level metrics (%)", weight="bold")
    fig.tight_layout(rect=(0, .12, 1, .94))
    return fig


def generate_combined_qc_report(
    dsd,
    output_dir: str | Path,
    *,
    baseline_group: str,
    read_metrics: pd.DataFrame | None = None,
    summary_files: Mapping[str, str | Path] | None = None,
    include_plate_maps: bool = True,
    planned_col: str = "expected_cells",
    nrow: int = 8,
    ncol: int = 12,
    formats=("pdf", "png"),
    report_label: str | None = None,
    **compute_kwargs,
):
    """Generate the combined QC suite, optional plate maps, and source CSV tables.

    Pass either ``summary_files`` or group/metric/percent ``read_metrics`` for
    the doughnut. Without them it is explicitly skipped. Plate maps require
    planned labels and physical well coordinates in obs; quantitative maps
    contain only shared wells, while identity maps retain all supplied wells.
    ``expected_row``/``expected_column`` can override numbered well-dT IDs.
    Other keywords are forwarded to :func:`compute_combined_qc`.

    Returns a dict with ``tables``, ``files``, and ``skipped``. No input is
    mutated. The 14 reference QC figures and six maps are produced with the
    default gene panels and all required inputs. PDF sizes are preserved.
    """
    if summary_files is not None and read_metrics is not None:
        raise ValueError("Supply either summary_files or read_metrics, not both.")
    formats = tuple(formats)
    if not formats or not set(formats) <= {"pdf", "png", "svg"}:
        raise ValueError("formats must contain pdf, png, and/or svg.")
    tables = compute_combined_qc(dsd, baseline_group=baseline_group, **compute_kwargs)
    wells = tables["wells"]
    groups = list(pd.unique(wells["group"]))
    if summary_files is not None:
        read_metrics = read_star_summary_metrics(summary_files)
    if read_metrics is not None:
        _validate_read_metrics(read_metrics)
        if set(read_metrics["group"]) != set(groups):
            raise ValueError("Read-metric groups must match the count-matrix groups.")
        tables["read_metrics"] = read_metrics.copy()
    if include_plate_maps:
        _require(tables["all_wells"], [planned_col])
    label = report_label or "Combined_" + "_".join(groups)
    label = _safe_stem(label)
    out = Path(output_dir)
    fig_dir, table_dir = out / "figures", out / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    files, skipped = [], []

    def save(fig, name):
        try:
            for fmt in formats:
                path = fig_dir / f"{label}_{name}.{fmt}"
                # Do not use bbox_inches='tight': preserve reference page dimensions.
                fig.savefig(path, dpi=180)
                files.append(str(path))
        finally:
            plt.close(fig)

    save(plot_combined_barnyard(wells), "Barnyard_Plot")
    save(plot_combined_species_purity(wells), "Species_Purity")
    for metric, name in (("total_umi", "UMI_Density"), ("n_genes_det", "Gene_Density")):
        save(plot_combined_density(wells, metric), name)
    for identifier, name in (("all", "Gene"), ("gene_symbol", "Gene_Symbol"),
                             ("ensembl_id", "Entrez_ID")):
        save(plot_combined_gene_umi(tables["gene_totals"], identifier), f"UMI_per_{name}")
        if identifier != "all":
            save(plot_combined_saturation(wells, identifier), f"Saturation_{name}")
    marker_panels = list(pd.unique(tables["markers"]["panel"]))
    requested_panels = compute_kwargs.get("marker_sets")
    if requested_panels is None:
        requested_panels = {"Human_MDA-MB-231_Markers": HUMAN_MARKERS,
                            "Mouse_3T3_Markers": MOUSE_3T3_MARKERS}
    for panel in requested_panels:
        if panel not in marker_panels:
            skipped.append(f"{panel}: none of the requested marker symbols were found.")
    for panel in marker_panels:
        save(plot_combined_markers(tables["markers"], panel), _safe_stem(panel))
    save(plot_combined_qc_boxplots(tables["qc_values"]), "QC_Boxplots")
    save(plot_mt_ribo_housekeeping(tables["composition_values"]), "MT_Ribo_Housekeeping_Boxplots")
    if read_metrics is not None:
        save(plot_combined_read_doughnut(read_metrics), "Multilayered_Doughnut")
    else:
        skipped.append("Multilayered_Doughnut: supply full-library STAR summary metrics.")
    if include_plate_maps:
        for name, frame, field, continuous, log_values in [
            ("Detected_Genes", wells, "n_genes_det", True, False),
            ("UMI_Count", wells, "total_umi", True, True),
            ("Read_Count", wells, "reads_per_well", True, True),
            ("MT_Pct", wells, "pct_mito", True, False),
            ("Human_Mouse", tables["all_wells"], "baseline_species", False, False),
            ("231_3T3_Water", tables["all_wells"], planned_col, False, False),
        ]:
            frame = frame.copy()
            # Existing numbered-well conventions differ between experiments.
            # Explicit physical coordinates take precedence only in this report.
            if {"expected_row", "expected_column"} <= set(frame.columns):
                from .plate import _row_label
                row = pd.to_numeric(frame["expected_row"], errors="raise")
                col = pd.to_numeric(frame["expected_column"], errors="raise")
                if (row.isna().any() or col.isna().any() or (row % 1 != 0).any()
                        or (col % 1 != 0).any()):
                    raise ValueError("Physical plate coordinates must be nonmissing integers.")
                frame["well_id"] = [_row_label(int(r)) + str(int(c)) for r, c in zip(row, col)]
            frame["baseline_species"] = frame["baseline_species"].where(frame["shared_well"], "Excluded")
            code = frame["baseline_species"].map({"human": "H", "mouse": "M", "Excluded": ""})
            planned = frame[planned_col].astype(object).fillna("").astype(str)
            frame["plot_label"] = (planned + " " + code.fillna("")).str.strip()
            if continuous:
                frame["plot_label"] = [
                    f"{v:,.2f}\n{s}" if field == "pct_mito" else f"{v:,.0f}\n{s}"
                    for v, s in zip(frame[field], frame["plot_label"])
                ]
            palette = SPECIES_COLORS if field == "baseline_species" else {
                "231": "#8CCAD9", "3T3": "#F2BF83", "water": "#F5F7FA",
            }
            fig, layout = plot_plate_layout(
                frame, field, plate_id="group", label_col="plot_label", continuous=continuous,
                log_values=log_values, nrow=nrow, ncol=ncol, nrow_plate=1,
                figsize=(10, 7.25 * len(groups)), palette=palette, legend_per_plate=True,
                label_size=7, well_size=.88,
            )
            fig.suptitle(name.replace("_", " "), fontsize=14, weight="bold", y=.985)
            population = ("Shared wells; empty circles = other positions" if continuous else
                          "Full supplied layout; Excluded = outside the shared-well set")
            fig.text(.5, .95, f"{population}\nH/M = {baseline_group}-baseline human/mouse; "
                     "other labels = planned cell types", ha="center", va="center", fontsize=8)
            save(fig, "Plate_Map_" + name)
            tables["plate_map_" + name] = layout
    for name, table in tables.items():
        path = table_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        files.append(str(path))
    return {"tables": tables, "files": files, "skipped": skipped}


def _safe_stem(value):
    stem = "".join(c if c.isalnum() or c in "_-" else "_" for c in str(value)).strip("_")
    if not stem:
        raise ValueError("Report and panel labels must contain letters or numbers.")
    return stem
