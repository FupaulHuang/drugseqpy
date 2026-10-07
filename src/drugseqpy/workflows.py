"""High-level standard QC and independent mix-species report workflows."""
from __future__ import annotations

import gzip
from pathlib import Path
from collections.abc import Mapping, Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter, StrMethodFormatter

from .core import DrugSeqData
from .qc import compute_qc_metrics, compute_plate_qc, plate_qc_summary, validate_metadata
from .plate import adaptive_figsize, plot_plate_layout
from .mix_species import compute_mix_species_qc


REFERENCE_COLORS = {
    "F3": "#167d9a",
    "F4": "#d46b3d",
    "231": "#00B4D8",
    "3T3": "#0A192F",
    "water": "#64748B",
}


MDA_MB_231_MARKERS = (
    "VIM", "FN1", "CDH2", "CD44", "EGFR",
    "MMP9", "TGFB1", "SNAI1", "TWIST1", "ITGB1",
)


def _save(fig, out: Path, stem: str, formats):
    paths = []
    for fmt in formats:
        path = out / f"{stem}.{fmt}"
        fig.savefig(path, bbox_inches="tight", dpi=180 if fmt == "png" else 300)
        paths.append(str(path))
    plt.close(fig)
    return paths


PaletteSpec = str | Sequence[str] | Mapping[str, str] | None


def _palette(values, palette: PaletteSpec = None):
    """Resolve a named, sequential, or category-to-color palette."""
    values = [str(x) for x in pd.unique(pd.Series(values).dropna().astype(str))]
    remaining = [x for x in values if x not in REFERENCE_COLORS]
    colors = sns.color_palette("colorblind", max(1, len(remaining))).as_hex()
    generated = dict(zip(remaining, colors))
    defaults = {
        x: REFERENCE_COLORS.get(x, generated.get(x, "#6b7280"))
        for x in values
    }
    if palette is None:
        return defaults
    if isinstance(palette, Mapping):
        return {x: palette.get(x, defaults[x]) for x in values}
    custom = sns.color_palette(palette, n_colors=max(1, len(values))).as_hex()
    return dict(zip(values, custom))


def _label_stem(value) -> str:
    text = "".join(
        char if char.isalnum() or char in {"_", "-"} else "_"
        for char in str(value)
    ).strip("_")
    return text or "all"


def _qc_report_labels(plate_order, report_label: str | None = None) -> tuple[str, str]:
    plates = [
        str(plate) for plate in plate_order
        if str(plate).strip() and str(plate).lower() != "nan"
    ]
    if not plates:
        plates = ["all"]
    display = "/".join(plates)
    if report_label is not None:
        return _label_stem(report_label), display
    stem = "_".join(_label_stem(plate) for plate in plates)
    if len(plates) > 1:
        stem = f"Combined_{stem}"
    return stem, display


def _matrix(dsd):
    x = dsd.adata.layers.get("counts", dsd.adata.X)
    if hasattr(x, "toarray"): x = x.toarray()
    return np.asarray(x, dtype=float)


def _gene_tables(dsd, out):
    x = _matrix(dsd); genes = pd.Index(dsd.var_names.astype(str))
    totals = x.sum(axis=0)
    symbols = dsd.var["gene_symbol"].astype(str).values if "gene_symbol" in dsd.var else genes.astype(str).values
    base = pd.DataFrame({"gene": genes, "gene_symbol": symbols, "total_umi": totals, "detected_samples": (x > 0).sum(axis=0)})
    base["gene_label"] = base["gene_symbol"].where(base["gene_symbol"].ne("nan") & base["gene_symbol"].ne(""), base["gene"])
    base.to_csv(out / "gene_umi_summary.csv", index=False)
    base[~base["gene_symbol"].str.startswith("ENS")].to_csv(out / "gene_symbol_umi_summary.csv", index=False)
    base[base["gene"].str.startswith("ENS")].to_csv(out / "entrez_umi_summary.csv", index=False)
    return base


def _grouped_gene_umi_table(
    dsd,
    *,
    groupby: str,
    split_by: str | None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Calculate feature totals independently for color groups and facets."""
    matrix = _matrix(dsd)
    genes = pd.Index(dsd.var_names.astype(str))
    symbols = (
        dsd.var["gene_symbol"].fillna("").astype(str).values
        if "gene_symbol" in dsd.var else genes.astype(str).values
    )
    groups, group_order = _groups(dsd, groupby)
    splits, split_order = _split_groups(dsd, split_by)
    rows = []
    group_values = groups.astype(str).to_numpy()
    split_values = splits.astype(str).to_numpy()
    for split_value in split_order:
        for group in group_order:
            mask = (
                (split_values == str(split_value))
                & (group_values == str(group))
            )
            if not mask.any():
                continue
            totals = matrix[mask].sum(axis=0)
            detected = (matrix[mask] > 0).sum(axis=0)
            frame = pd.DataFrame({
                "gene": genes,
                "gene_symbol": symbols,
                "total_umi": np.asarray(totals).ravel().astype(float),
                "detected_samples": np.asarray(detected).ravel().astype(int),
                groupby: str(group),
                "_split_value": str(split_value),
            })
            if split_by is not None and split_by != groupby:
                frame[split_by] = str(split_value)
            rows.append(frame)
    return pd.concat(rows, ignore_index=True), group_order, split_order


def _plate_groups(dsd):
    """Plate group labels as sorted strings (never positional indices)."""
    groups = dsd.obs.get("plate_id", pd.Series("all", index=dsd.obs.index)).astype(str)
    order = sorted(pd.unique(groups[groups != "nan"]))
    return groups.where(groups != "nan", "unknown"), order


def _groups(dsd, groupby: str = "plate_id"):
    """Return display-safe labels and a stable order for an obs grouping."""
    if groupby not in dsd.obs.columns:
        if groupby == "plate_id":
            groups = pd.Series("all", index=dsd.obs.index, dtype=str)
        else:
            raise KeyError(
                f"groupby={groupby!r} is not present in dsd.obs. "
                f"Available columns: {list(dsd.obs.columns)}"
            )
    else:
        groups = dsd.obs[groupby].fillna("unknown").astype(str)
    order = sorted(pd.unique(groups))
    return groups, order


def _split_groups(dsd, split_by: str | None):
    """Return facet labels while keeping ``None`` as one combined panel."""
    if split_by is None:
        return (
            pd.Series("all", index=dsd.obs.index, dtype=str),
            ["all"],
        )
    return _groups(dsd, split_by)


def _split_stem(stem: str, split_by: str | None) -> str:
    """Give explicitly faceted figures a non-overwriting filename suffix."""
    if split_by is None:
        return stem
    return f"{stem}_split_by_{_label_stem(split_by)}"


def _density_figure(
    dsd,
    metric,
    title,
    output_name,
    out,
    formats,
    *,
    groupby: str = "plate_id",
    split_by: str | None = None,
    nrow_plate: int | None = None,
    palette: PaletteSpec = None,
):
    groups, order = _groups(dsd, groupby)
    splits, split_order = _split_groups(dsd, split_by)
    data = pd.DataFrame({
        "metadata_index": dsd.obs.index.astype(str),
        groupby: groups.astype(str).values,
        "_split_value": splits.astype(str).values,
        metric: dsd.obs[metric].values,
    }).dropna(subset=[metric])
    if split_by is not None:
        data[split_by] = data["_split_value"]
    data.drop(columns="_split_value").to_csv(
        out / f"{_split_stem(output_name, split_by)}_plot_table.csv",
        index=False,
    )
    fig, axes = _facet_grid(
        len(split_order), nrow_plate=nrow_plate, panel=(8.0, 5.2),
    )
    colors = _palette(order, palette)
    metric_labels = {
        "total_umi": "Total UMI Counts per Well",
        "n_genes_det": "Genes Detected per Well",
    }
    for ax, split_value in zip(axes.flat, split_order):
        panel = data[data["_split_value"] == str(split_value)]
        for group in order:
            vals = panel.loc[panel[groupby] == group, metric]
            if len(vals) > 1 and vals.nunique() > 1:
                sns.kdeplot(
                    x=vals, ax=ax, fill=True, alpha=.28, linewidth=1.1,
                    color=colors.get(group), label=group,
                )
            elif len(vals):
                ax.axvline(
                    float(vals.iloc[0]), color=colors.get(group),
                    linewidth=1.5, label=group,
                )
        panel_title = title
        if split_by is not None:
            panel_title = f"{split_by}: {split_value}"
        ax.set_title(panel_title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel(
            metric_labels.get(metric, metric), fontsize=10,
            fontweight="bold",
        )
        ax.set_ylabel("Density", fontsize=10, fontweight="bold")
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.tick_params(axis="both", labelsize=9)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                title=groupby, fontsize=8, title_fontsize=9,
                frameon=False, loc="best",
            )
        sns.despine(ax=ax)
    for ax in axes.flat[len(split_order):]:
        ax.set_visible(False)
    if split_by is not None:
        fig.suptitle(title, fontsize=14, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, .96))
    else:
        fig.tight_layout()
    return _save(
        fig, out, _split_stem(output_name, split_by), formats,
    )


def _filtered_barcodes(gene_dir: Path) -> set[str]:
    barcode_path = gene_dir / "filtered" / "barcodes.tsv.gz"
    if not barcode_path.exists():
        return set()
    with gzip.open(barcode_path, "rt") as handle:
        return {line.strip().removesuffix("-1") for line in handle if line.strip()}


def _star_well_read_table(
    dsd: DrugSeqData,
    star_root: str | Path,
    *,
    groupby: str = "plate_id",
    split_by: str | None = None,
) -> pd.DataFrame:
    """Return one STAR read-count row for every well in ``dsd``."""
    root = Path(star_root)
    sample_table = _expression_saturation_table(
        dsd, groupby=groupby, split_by=split_by,
    )
    rows = []
    found_valid_stats = False
    for plate, expected in sample_table.groupby("plate_id", sort=False):
        gene_dir = root / str(plate) / "library_seq_Solo.out" / "Gene"
        stats_path = gene_dir / "CellReads.stats"
        if stats_path.exists():
            stats = pd.read_csv(stats_path, sep="\t")
        else:
            stats = pd.DataFrame(columns=["CB", "cbMatch"])
        if {"CB", "cbMatch"}.issubset(stats.columns):
            found_valid_stats = True
            stats = stats[["CB", "cbMatch"]].copy()
            stats["barcode_norm"] = (
                stats["CB"].astype(str).str.replace(r"-1$", "", regex=True)
            )
            filtered = _filtered_barcodes(gene_dir)
            if filtered:
                stats = stats[stats["barcode_norm"].isin(filtered)]
            else:
                stats = stats[
                    stats["CB"].astype(str) != "CBnotInPasslist"
                ]
            stats["read_count"] = pd.to_numeric(
                stats["cbMatch"], errors="coerce",
            )
            stats = stats.sort_values("read_count").drop_duplicates(
                "barcode_norm", keep="last",
            )
        else:
            stats = pd.DataFrame(columns=["barcode_norm", "read_count"])
        merged = expected.merge(
            stats[["barcode_norm", "read_count"]],
            on="barcode_norm",
            how="left",
        )
        merged["matched_star_record"] = merged["read_count"].notna()
        merged["read_count"] = merged["read_count"].fillna(0).astype(float)
        rows.append(merged)
    if not rows or not found_valid_stats:
        columns = [
            "plate_id", "barcode_norm", "sample_id", groupby,
            "_star_split", "read_count", "matched_star_record",
        ]
        return pd.DataFrame(columns=list(dict.fromkeys(columns)))
    table = pd.concat(rows, ignore_index=True)
    table[groupby] = table["_star_group"].fillna("unknown").astype(str)
    if split_by is not None:
        table[split_by] = table["_star_split"].fillna("unknown").astype(str)
    return table


def _expression_saturation_table(
    dsd,
    *,
    groupby: str = "plate_id",
    split_by: str | None = None,
) -> pd.DataFrame:
    obs = dsd.obs.copy()
    obs["plate_id"] = obs.get("plate_id", pd.Series("all", index=obs.index)).astype(str)
    groups, _ = _groups(dsd, groupby)
    splits, _ = _split_groups(dsd, split_by)
    obs["_star_group"] = groups.astype(str).values
    obs["_star_split"] = splits.astype(str).values
    if "barcode" in obs.columns:
        barcode = obs["barcode"].astype(str)
    elif "sample_id" in obs.columns:
        # Imported objects normally preserve ``barcode``. This fallback also
        # supports older objects whose observation names were replaced by
        # ``sample_id = plate_id + '_' + barcode`` before plates were merged.
        sample_ids = obs["sample_id"].astype(str)
        prefixes = obs["plate_id"].astype(str) + "_"
        barcode = pd.Series(
            [
                sample_id[len(prefix):]
                if sample_id.startswith(prefix) else sample_id
                for sample_id, prefix in zip(sample_ids, prefixes)
            ],
            index=obs.index,
            dtype=str,
        )
    else:
        barcode = pd.Series(obs.index, index=obs.index, dtype=str)
    obs["barcode_norm"] = barcode.str.replace(r"-1$", "", regex=True)
    mat = dsd.adata.layers.get("counts", dsd.adata.X)
    symbols = dsd.var["gene_symbol"].astype(str).values if "gene_symbol" in dsd.var else dsd.var_names.astype(str).values
    genes = dsd.var_names.astype(str).values
    symbol_series = pd.Series(symbols).astype(str)
    symbol_mask = symbol_series.notna().values & ~symbol_series.isin(["", "nan", "None"]).values & ~symbol_series.str.startswith("ENS").values
    entrez_mask = pd.Series(genes).astype(str).str.startswith("ENS").values
    obs["symbol_genes"] = np.asarray((mat[:, symbol_mask] > 0).sum(axis=1)).ravel().astype(float)
    obs["entrez_genes"] = np.asarray((mat[:, entrez_mask] > 0).sum(axis=1)).ravel().astype(float)
    cols = [
        "plate_id", "barcode_norm", "_star_group", "_star_split", "symbol_genes",
        "entrez_genes",
    ]
    if "sample_id" in obs.columns:
        cols.append("sample_id")
    return obs[cols]


def _plot_star_doughnut(
    summary: pd.DataFrame,
    out: Path,
    formats,
    *,
    report_label: str,
    groupby: str = "plate_id",
    split_by: str | None = None,
    nrow_plate: int | None = None,
    palette: PaletteSpec = None,
):
    metric_defaults = [
        ("Valid Barcodes", "#A8E6CF"),
        ("Q30 CB+UMI", "#FFD3B6"),
        ("Q30 RNA", "#A8D8EA"),
        ("Mapped to Genome", "#D4A5D9"),
        ("Mapped to Gene", "#F7A8B8"),
    ]
    if palette is None:
        metrics = metric_defaults
    else:
        colors = _palette([metric for metric, _ in metric_defaults], palette)
        metrics = [(metric, colors[metric]) for metric, _ in metric_defaults]
    panel_col = split_by if split_by is not None else groupby
    groups = sorted(pd.unique(summary[panel_col].astype(str)))
    panels_per_row = 2 if nrow_plate is None else int(nrow_plate)
    if panels_per_row < 1:
        raise ValueError("nrow_plate must be None or a positive integer")
    ncols = min(panels_per_row, max(1, len(groups)))
    nrows = int(np.ceil(max(1, len(groups)) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.2 * ncols, 4.6 * nrows),
        squeeze=False,
        subplot_kw={"aspect": "equal"},
    )
    grouped = summary.groupby([panel_col, "metric"], as_index=False)["percent"].mean()
    for ax, group in zip(axes.flat, groups):
        pdata = grouped[grouped[panel_col].astype(str) == group].set_index("metric")
        for i, (metric, color) in enumerate(metrics):
            pct = float(pdata.loc[metric, "percent"]) if metric in pdata.index else 0.0
            shown = min(max(pct, 0.0), 100.0)
            radius = 1.00 - i * 0.145
            width = 0.105
            ax.pie(
                [shown, max(100.0 - shown, 0.0)],
                radius=radius,
                startangle=90,
                counterclock=False,
                colors=[color, "#E5E7EB"],
                wedgeprops={"width": width, "edgecolor": "white", "linewidth": 1},
            )
            theta = np.deg2rad(90 - shown * 1.8)
            r = radius - width / 2
            ax.text(np.cos(theta) * r, np.sin(theta) * r, f"{pct:.1f}%", ha="center", va="center", fontsize=6.5, fontweight="bold", color="#1f2937")
        ax.text(0, 0, str(group), ha="center", va="center", fontsize=14, fontweight="bold")
        ax.set_axis_off()
    for ax in axes.flat[len(groups):]:
        ax.set_visible(False)
    handles = [Patch(facecolor=color, edgecolor="none", label=metric) for metric, color in metrics]
    fig.legend(handles=handles, loc="center right", frameon=False, title="Ring", fontsize=8, title_fontsize=9)
    fig.suptitle("Read-Level STARsolo QC (% of Total Reads)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, .84, .92))
    return _save(
        fig,
        out,
        _split_stem(f"{report_label}_Multilayered_Doughnut", split_by),
        formats,
    )


def _star_figures(
    star_root,
    out,
    formats,
    dsd=None,
    *,
    plate_order=None,
    report_label: str | None = None,
    groupby: str = "plate_id",
    split_by: str | None = None,
    nrow_plate: int | None = None,
    palette: PaletteSpec = None,
    include_read_summary: bool = True,
    include_saturation: bool = True,
):
    root = Path(star_root); summaries = []; saturation = []
    expr_sat = (
        _expression_saturation_table(
            dsd, groupby=groupby, split_by=split_by,
        )
        if dsd is not None and include_saturation else None
    )
    plate_group = {}
    plate_split = {}
    if dsd is not None:
        if groupby not in dsd.obs.columns:
            raise KeyError(
                f"groupby={groupby!r} is not present in dsd.obs"
            )
        groups, _ = _groups(dsd, groupby)
        splits, _ = _split_groups(dsd, split_by)
        mapping = pd.DataFrame({
            "plate_id": dsd.obs["plate_id"].astype(str).values,
            "_star_group": groups.astype(str).values,
            "_star_split": splits.astype(str).values,
        })
        for plate, sub in mapping.groupby("plate_id", sort=False):
            labels = list(pd.unique(sub["_star_group"]))
            if include_read_summary and len(labels) != 1:
                raise ValueError(
                    f"STAR read-summary groupby={groupby!r} must have one "
                    f"value per plate; plate {plate!r} has {labels}"
                )
            plate_group[plate] = labels[0] if labels else "unknown"
            split_labels = list(pd.unique(sub["_star_split"]))
            if include_read_summary and len(split_labels) != 1:
                raise ValueError(
                    f"STAR read-summary split_by={split_by!r} must have one "
                    f"value per plate; plate {plate!r} has {split_labels}"
                )
            plate_split[plate] = (
                split_labels[0] if split_labels else "unknown"
            )
    if plate_order is None:
        if dsd is not None and "plate_id" in dsd.obs:
            plate_order = sorted(pd.unique(dsd.obs["plate_id"].dropna().astype(str)))
        elif root.exists():
            plate_order = sorted(
                path.name for path in root.iterdir()
                if (path / "library_seq_Solo.out" / "Gene").exists()
            )
        else:
            plate_order = []
    report_label, _ = _qc_report_labels(plate_order, report_label)
    for plate in plate_order:
        gene_dir = root / plate / "library_seq_Solo.out" / "Gene"
        summary = gene_dir / "Summary.csv"
        stats = gene_dir / "CellReads.stats"
        if include_read_summary and summary.exists():
            kv = pd.read_csv(summary, header=None, names=["metric", "value"])
            values = dict(zip(kv.metric, pd.to_numeric(kv.value, errors="coerce")))
            metric_map = {
                "Reads With Valid Barcodes": "Valid Barcodes",
                "Q30 Bases in CB+UMI": "Q30 CB+UMI",
                "Q30 Bases in RNA read": "Q30 RNA",
                "Reads Mapped to Genome: Unique": "Mapped to Genome",
                "Reads Mapped to Gene: Unique Gene": "Mapped to Gene",
            }
            for raw, label in metric_map.items():
                if raw in values:
                    value = values[raw]
                    summaries.append({
                        "plate_id": plate,
                        groupby: plate_group.get(str(plate), str(plate)),
                        "_star_split": plate_split.get(str(plate), "all"),
                        "metric": label,
                        "fraction": value,
                        "percent": value * 100 if value <= 1.5 else value,
                    })
        if include_saturation and stats.exists():
            df = pd.read_csv(stats, sep="\t")
            if {"CB", "cbMatch", "nGenesUnique"}.issubset(df):
                df = df.copy()
                df["barcode_norm"] = df["CB"].astype(str).str.replace(r"-1$", "", regex=True)
                barcodes = _filtered_barcodes(gene_dir)
                if barcodes:
                    df = df[df["barcode_norm"].isin(barcodes)]
                else:
                    df = df[df["CB"].astype(str) != "CBnotInPasslist"]
                base = df[["CB", "barcode_norm", "cbMatch", "nGenesUnique"]].assign(plate_id=plate)
                if expr_sat is not None:
                    base = base.merge(expr_sat, on=["plate_id", "barcode_norm"], how="left")
                if groupby == "plate_id":
                    base["_star_group"] = base["plate_id"].astype(str)
                elif "_star_group" not in base:
                    base["_star_group"] = base["plate_id"].map(plate_group)
                if "_star_split" not in base:
                    base["_star_split"] = base["plate_id"].map(plate_split)
                if "symbol_genes" not in base:
                    base["symbol_genes"] = base["nGenesUnique"]
                if "entrez_genes" not in base:
                    base["entrez_genes"] = base["nGenesUnique"]
                base["cbMatch"] = pd.to_numeric(base["cbMatch"], errors="coerce")
                base["nGenesUnique"] = pd.to_numeric(base["nGenesUnique"], errors="coerce")
                base["symbol_genes"] = pd.to_numeric(base["symbol_genes"], errors="coerce").fillna(base["nGenesUnique"])
                base["entrez_genes"] = pd.to_numeric(base["entrez_genes"], errors="coerce").fillna(base["nGenesUnique"])
                saturation.append(base)
    files = []
    if summaries:
        s = pd.DataFrame(summaries)
        if split_by is not None:
            s[split_by] = s["_star_split"].astype(str)
        s.drop(columns="_star_split").to_csv(
            out / "star_group_summary.csv", index=False,
        )
        files += _plot_star_doughnut(
            s,
            out,
            formats,
            report_label=report_label,
            groupby=groupby,
            split_by=split_by,
            nrow_plate=nrow_plate,
            palette=palette,
        )
    if saturation:
        sat = pd.concat(saturation, ignore_index=True)
        sat[groupby] = sat["_star_group"].fillna("unknown").astype(str)
        if split_by is None:
            sat["_plot_split"] = "all"
        else:
            sat[split_by] = sat["_star_split"].fillna("unknown").astype(str)
            sat["_plot_split"] = sat[split_by]
        sat.drop(columns="_plot_split").to_csv(
            out / "star_saturation_metrics.csv", index=False,
        )
        group_order = sorted(pd.unique(sat[groupby]))
        colors = _palette(group_order, palette)
        split_order = sorted(pd.unique(sat["_plot_split"].astype(str)))
        for suffix, y in [("Entrez_ID", "entrez_genes"), ("Gene_Symbol", "symbol_genes")]:
            # Original (non-log) values: x = reads (cbMatch),
            # y = detected genes from the expression matrix when available.
            fig, axes = _facet_grid(
                len(split_order), nrow_plate=nrow_plate, panel=(7.5, 5.5),
            )
            for ax, split_value in zip(axes.flat, split_order):
                sub = sat[sat["_plot_split"].astype(str) == split_value]
                panel_group_order = [
                    group for group in group_order
                    if group in set(sub[groupby].astype(str))
                ]
                sns.scatterplot(
                    data=sub, x="cbMatch", y=y, hue=groupby,
                    hue_order=panel_group_order,
                    palette={group: colors[group] for group in panel_group_order},
                    s=28, alpha=.7, ax=ax,
                )
                med = (
                    float(np.nanmedian(sub[y]))
                    if sub[y].notna().any() else np.nan
                )
                if np.isfinite(med):
                    ax.axhline(
                        med, linestyle="--", linewidth=1, color="#D32F2F",
                    )
                    ax.text(
                        .02, 1.01, f"Median: {med:,.0f}",
                        transform=ax.transAxes, ha="left", va="bottom",
                        fontsize=9, color="#D32F2F", fontweight="bold",
                        clip_on=False,
                    )
                ax.set_xlabel("Reads count", fontsize=10, fontweight="bold")
                ax.set_ylabel("Detected genes", fontsize=10, fontweight="bold")
                panel_title = "All samples"
                if split_by is not None:
                    panel_title = f"{split_by}: {split_value}"
                ax.set_title(panel_title, fontsize=11, fontweight="bold")
                ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
                ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
                ax.tick_params(axis="both", labelsize=8)
                handles, _ = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(
                        title=groupby, fontsize=8, title_fontsize=9,
                        frameon=False,
                    )
                sns.despine(ax=ax)
            for ax in axes.flat[len(split_order):]:
                ax.set_visible(False)
            fig.suptitle(
                f"STAR Saturation - {suffix.replace('_', ' ')}",
                fontsize=14, fontweight="bold",
            )
            fig.tight_layout(rect=(0, 0, 1, .95))
            files += _save(
                fig,
                out,
                _split_stem(
                    f"{report_label}_Saturation_{suffix}", split_by,
                ),
                formats,
            )
    return files


def _ensure_qc_metrics(dsd: DrugSeqData) -> None:
    """Populate the sample metrics required by the public QC figure groups."""
    required = {
        "total_umi", "n_genes_det", "pct_mito", "pct_ribo",
        "outlier_score",
    }
    if not required.issubset(dsd.obs.columns):
        compute_qc_metrics(dsd, inplace=True)
    if "pct_mito" in dsd.obs:
        dsd.obs["mt.pct"] = dsd.obs["pct_mito"]
    if "pct_ribo" in dsd.obs:
        dsd.obs["ribo.pct"] = dsd.obs["pct_ribo"]


def _plate_gene_expression(
    dsd: DrugSeqData,
    plate_gene: str | Sequence[str],
    aggregation: str,
) -> tuple[list[str], np.ndarray, pd.DataFrame, pd.DataFrame, bool]:
    """Resolve one gene or a gene set and return per-sample expression."""
    is_gene_set = not isinstance(plate_gene, str)
    if isinstance(plate_gene, str):
        genes = [plate_gene.strip()]
    elif isinstance(plate_gene, Sequence):
        genes = list(dict.fromkeys(str(gene).strip() for gene in plate_gene))
    else:
        raise TypeError("plate_gene must be a string or a sequence of strings")
    genes = [gene for gene in genes if gene]
    if not genes:
        raise ValueError("plate_gene must contain at least one non-empty gene")

    aggregation = str(aggregation).lower()
    allowed = {"mean", "sum", "median"}
    if aggregation not in allowed:
        raise ValueError(
            f"gene_aggregation must be one of {sorted(allowed)}, got {aggregation!r}"
        )

    symbols = (
        dsd.var["gene_symbol"].astype(str).to_numpy()
        if "gene_symbol" in dsd.var else dsd.var_names.astype(str).to_numpy()
    )
    ids = dsd.var_names.astype(str).to_numpy()
    counts = dsd.adata.layers.get("counts", dsd.adata.X)
    vectors = []
    resolution_rows = []
    missing = []
    for gene in genes:
        mask = (symbols == gene) | (ids == gene)
        if not mask.any():
            missing.append(gene)
            continue
        values = counts[:, mask].sum(axis=1)
        vectors.append(np.asarray(values).ravel().astype(float))
        for feature_id, symbol in zip(ids[mask], symbols[mask]):
            resolution_rows.append({
                "requested_gene": gene,
                "feature_id": feature_id,
                "gene_symbol": symbol,
                "matched_by": (
                    "feature_id" if feature_id == gene and symbol != gene
                    else "gene_symbol" if symbol == gene and feature_id != gene
                    else "both"
                ),
            })
    if missing:
        raise KeyError(
            "plate_gene entries were not found in gene symbols or IDs: "
            + ", ".join(missing)
        )

    per_gene = np.column_stack(vectors)
    if is_gene_set:
        if aggregation == "mean":
            expression = per_gene.mean(axis=1)
        elif aggregation == "sum":
            expression = per_gene.sum(axis=1)
        else:
            expression = np.median(per_gene, axis=1)
    else:
        expression = per_gene[:, 0]

    per_sample = pd.DataFrame(per_gene, columns=genes, index=dsd.obs.index)
    per_sample.insert(0, "metadata_index", dsd.obs.index.astype(str))
    if "sample_id" in dsd.obs.columns:
        per_sample.insert(1, "sample_id", dsd.obs["sample_id"].astype(str).values)
    per_sample["aggregate_expression"] = expression
    per_sample["aggregation"] = aggregation if is_gene_set else "single_gene"
    resolution = pd.DataFrame(resolution_rows)
    resolution["aggregation"] = aggregation if is_gene_set else "single_gene"
    return genes, expression, resolution, per_sample, is_gene_set


def generate_plate_layout_figures(
    dsd: DrugSeqData,
    output_dir: str | Path,
    *,
    formats=("pdf", "png"),
    plate_value: str | None = None,
    plate_gene: str | Sequence[str] | None = None,
    gene_aggregation: str = "mean",
    well_color: str | None = None,
    plate_id: str = "plate_id",
    split_by: str | None = None,
    label_col: str | None = None,
    continuous: bool = False,
    log_values: bool = True,
    show_legend: bool = True,
    nrow: int = 8,
    ncol: int = 12,
    nrow_plate: int | None = None,
    output_name: str | None = None,
    figsize: tuple[float, float] | None = None,
    well_size: float = 0.76,
    well_border: bool = True,
    well_border_color: str = "#111827",
    well_border_width: float = 0.8,
    palette: PaletteSpec = None,
    cmap: str = "RdYlBu_r",
    label_color: str = "#111827",
    label_size: float | None = None,
    label_style: str = "normal",
    label_weight: str | int = "normal",
) -> list[str]:
    """Generate one universal plate-layout figure and its source tables.

    Exactly one of ``plate_value``, ``plate_gene``, or ``well_color`` must be
    supplied. ``plate_value`` names an ``obs`` field; a string ``plate_gene``
    colors wells by one gene, while a sequence aggregates per-gene expression
    with ``gene_aggregation`` (mean, sum, or median); and ``well_color`` is a
    literal fixed fill color. Values are categorical by default. Set
    ``continuous=True`` for numeric metrics or gene expression; these use
    ``log1p`` by default.
    ``label_col=None`` leaves wells unlabeled. ``plate_id`` names the metadata
    column containing actual plate identities. ``split_by`` optionally creates
    one layout for each plate-by-split combination. ``nrow`` and ``ncol``
    define the exact plate format, such as 8 by 12 for 96-well plates or 16 by
    24 for 384-well plates. ``nrow_plate`` controls how many plate panels
    appear in one figure row.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    choices = [plate_value is not None, plate_gene is not None, well_color is not None]
    if sum(choices) != 1:
        raise ValueError(
            "Exactly one of plate_value, plate_gene, or well_color must be supplied"
        )
    qc_fields = {
        "total_umi", "n_genes_det", "pct_mito", "pct_ribo", "mt.pct",
        "ribo.pct", "gini_index", "outlier_score",
    }
    if plate_value in qc_fields and plate_value not in dsd.obs.columns:
        _ensure_qc_metrics(dsd)
    if plate_id not in dsd.obs.columns:
        raise KeyError(f"plate_id={plate_id!r} is not present in dsd.obs")
    if split_by is not None and split_by not in dsd.obs.columns:
        raise KeyError(f"split_by={split_by!r} is not present in dsd.obs")

    plot_obs = dsd.obs.copy()
    value_col = plate_value
    gene_resolution = None
    per_sample_gene_expression = None
    plate_genes = None
    is_gene_set = False
    if plate_value is not None and plate_value not in plot_obs.columns:
        raise KeyError(f"plate_value={plate_value!r} is not present in dsd.obs")
    if plate_gene is not None:
        (
            plate_genes,
            expression,
            gene_resolution,
            per_sample_gene_expression,
            is_gene_set,
        ) = _plate_gene_expression(dsd, plate_gene, gene_aggregation)
        if is_gene_set:
            value_col = f"{gene_aggregation.lower()} expression ({len(plate_genes)} genes)"
            plot_obs["plate_gene_set"] = ";".join(plate_genes)
            plot_obs["gene_aggregation"] = gene_aggregation.lower()
        else:
            value_col = f"{plate_genes[0]} expression"
        plot_obs[value_col] = expression

    if output_name is None:
        if plate_value is not None:
            output_name = f"plate_layout_{_label_stem(plate_value)}"
        elif plate_gene is not None and is_gene_set:
            gene_label = (
                "_".join(_label_stem(gene) for gene in plate_genes)
                if len(plate_genes) <= 4 else f"{len(plate_genes)}_genes"
            )
            output_name = (
                f"plate_layout_geneset_{gene_label}_"
                f"{gene_aggregation.lower()}_expression"
            )
        elif plate_gene is not None:
            output_name = f"plate_layout_{_label_stem(plate_genes[0])}_expression"
        else:
            output_name = "plate_layout_wells"
    output_name = _label_stem(output_name)
    output_name = _split_stem(output_name, split_by)

    fig, layout = plot_plate_layout(
        plot_obs,
        value_col=value_col,
        well_color=well_color,
        plate_id=plate_id,
        split_by=split_by,
        label_col=label_col,
        continuous=continuous,
        log_values=log_values,
        show_legend=show_legend,
        nrow=nrow,
        ncol=ncol,
        nrow_plate=nrow_plate,
        figsize=figsize,
        well_size=well_size,
        well_border=well_border,
        well_border_color=well_border_color,
        well_border_width=well_border_width,
        palette=palette,
        cmap=cmap,
        label_color=label_color,
        label_size=label_size,
        label_style=label_style,
        label_weight=label_weight,
    )
    sort_cols = [c for c in ["plate_id", "row", "column"] if c in layout.columns]
    layout_sorted = layout.sort_values(sort_cols) if sort_cols else layout
    layout_sorted.to_csv(out / f"{output_name}_plot_table.csv", index=False)
    if gene_resolution is not None:
        gene_resolution.to_csv(
            out / f"{output_name}_gene_resolution.csv", index=False,
        )
        per_sample_gene_expression.to_csv(
            out / f"{output_name}_per_gene_expression.csv", index=False,
        )
    coordinate_cols = [
        c for c in [
            "plate_id", "physical_plate_id", "sample_id", "metadata_index", "well_id", "row",
            "column", "row_label", "is_missing_well",
        ] if c in layout_sorted
    ]
    layout_sorted[coordinate_cols].to_csv(
        out / "plate_layout_coordinates.csv", index=False,
    )
    files = _save(fig, out, output_name, formats)
    if dsd.adata.uns.get("plate_qc"):
        plate_qc_summary(dsd).to_csv(out / "plate_summary.csv", index=False)
    return files


def generate_gene_set_violin_figure(
    dsd: DrugSeqData,
    output_dir: str | Path,
    gene_sets: Mapping[str, Sequence[str]],
    *,
    gene_aggregation: str = "mean",
    formats=("pdf", "png"),
    groupby: str = "plate_id",
    split_by: str | None = None,
    nrow_plate: int | None = None,
    palette: PaletteSpec = None,
    log_values: bool = True,
    point_color: str = "#253746",
    annotation_color: str = "#D32F2F",
    output_name: str = "gene_set_expression_violin",
) -> list[str]:
    """Plot aggregated per-well expression for one or more gene sets.

    Gene symbols/IDs are resolved with the same exact-match logic used by
    :func:`generate_plate_layout_figures`, making spatial maps and violin
    distributions directly comparable.
    """
    if not isinstance(gene_sets, Mapping) or not gene_sets:
        raise ValueError("gene_sets must be a non-empty name-to-gene-list mapping")
    groups, group_order = _groups(dsd, groupby)
    splits, split_order = _split_groups(dsd, split_by)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    output_name = _label_stem(output_name)

    plot_rows = []
    resolution_tables = []
    per_gene_tables = []
    set_order = []
    for set_name, genes in gene_sets.items():
        label = str(set_name).strip()
        if not label:
            raise ValueError("gene-set names must be non-empty")
        if isinstance(genes, str):
            raise TypeError(
                f"gene_sets[{set_name!r}] must be a sequence, not one string"
            )
        (
            resolved_genes,
            expression,
            resolution,
            per_gene,
            _is_gene_set,
        ) = _plate_gene_expression(dsd, genes, gene_aggregation)
        set_order.append(label)
        frame = pd.DataFrame({
            "metadata_index": dsd.obs.index.astype(str),
            "gene_set": label,
            "expression": expression,
            groupby: groups.astype(str).values,
            "_split_value": splits.astype(str).values,
        })
        if "sample_id" in dsd.obs:
            frame["sample_id"] = dsd.obs["sample_id"].astype(str).values
        if split_by is not None:
            frame[split_by] = splits.astype(str).values
        frame["plot_expression"] = (
            np.log1p(frame["expression"])
            if log_values else frame["expression"]
        )
        frame["aggregation"] = gene_aggregation.lower()
        frame["genes"] = ";".join(resolved_genes)
        plot_rows.append(frame)
        resolution.insert(0, "gene_set", label)
        resolution_tables.append(resolution)
        per_gene.insert(0, "gene_set", label)
        per_gene_tables.append(per_gene)

    plot_table = pd.concat(plot_rows, ignore_index=True)
    plot_table.drop(columns="_split_value").to_csv(
        out / f"{output_name}_plot_table.csv", index=False,
    )
    pd.concat(resolution_tables, ignore_index=True).to_csv(
        out / f"{output_name}_gene_resolution.csv", index=False,
    )
    pd.concat(per_gene_tables, ignore_index=True).to_csv(
        out / f"{output_name}_per_gene_expression.csv", index=False,
    )

    panel_specs = [
        (split_value, set_name)
        for split_value in split_order
        for set_name in set_order
    ]
    panels_per_row = min(2, len(panel_specs)) if nrow_plate is None else int(nrow_plate)
    if panels_per_row < 1:
        raise ValueError("nrow_plate must be None or a positive integer")
    ncols = min(panels_per_row, max(1, len(panel_specs)))
    nrows = int(np.ceil(len(panel_specs) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.0 * ncols, 3.8 * nrows + .5),
        squeeze=False,
        sharey=False,
    )
    colors = _palette(group_order, palette)
    for ax, (split_value, set_name) in zip(axes.flat, panel_specs):
        sub = plot_table[
            (plot_table["_split_value"] == str(split_value))
            & (plot_table["gene_set"] == set_name)
        ]
        sns.violinplot(
            data=sub, x=groupby, y="plot_expression", hue=groupby,
            order=group_order, hue_order=group_order, palette=colors,
            inner="box", cut=0, density_norm="width", legend=False, ax=ax,
        )
        sns.stripplot(
            data=sub, x=groupby, y="plot_expression", order=group_order,
            color=point_color,
            size=max(1.5, min(3.2, 300 / max(len(sub), 1))),
            alpha=.28, ax=ax,
        )
        medians = sub.groupby(groupby)["plot_expression"].median()
        maxima = sub.groupby(groupby)["plot_expression"].max()
        ymin, ymax = ax.get_ylim()
        yrange = max(ymax - ymin, 1e-12)
        ax.set_ylim(ymin, ymax + yrange * .16)
        for xpos, group in enumerate(group_order):
            if group not in medians or pd.isna(medians[group]):
                continue
            median = float(medians[group])
            group_max = float(maxima[group])
            ax.text(
                xpos, group_max + yrange * .035,
                f"Median: {median:.2f}",
                ha="center", va="bottom", fontsize=7.5,
                fontweight="bold", color=annotation_color, clip_on=False,
            )
        title = set_name
        if split_by is not None:
            title = f"{set_name} · {split_by}: {split_value}"
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(groupby)
        ax.set_ylabel(
            f"log1p({gene_aggregation} expression)"
            if log_values else f"{gene_aggregation} expression",
        )
        _style_axis(ax)
    for ax in axes.flat[len(panel_specs):]:
        ax.set_visible(False)
    fig.suptitle(
        "Gene-set expression per well", fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, .96))
    return _save(fig, out, output_name, formats)


def generate_qc_metric_violin_figure(
    dsd: DrugSeqData,
    output_dir: str | Path,
    *,
    formats=("pdf", "png"),
    report_label: str | None = None,
    metrics: Sequence[str] | None = None,
    groupby: str = "plate_id",
    split_by: str | None = None,
    nrow_plate: int | None = None,
    palette: PaletteSpec = None,
) -> list[str]:
    """Generate the essential sample-QC metric violin panel by group.

    By default the panels show mitochondrial percentage, ribosomal
    percentage, detected UMI count, and detected-gene count. ``metrics`` may
    select any non-empty subset. Each panel has an independent y-axis and
    labels the per-group median above, rather than inside, each violin.
    ``groupby`` must name a column in ``dsd.obs`` and defaults to ``plate_id``.
    ``split_by=None`` keeps groups together; a metadata column creates facets.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _ensure_qc_metrics(dsd)
    if "pct_ribo" not in dsd.obs:
        dsd.obs["pct_ribo"] = 0.0
    metric_labels = {
        "mt.pct": "mt.pct (%)",
        "pct_mito": "pct_mito (%)",
        "ribo.pct": "ribo.pct (%)",
        "pct_ribo": "pct_ribo (%)",
        "total_umi": "Detected UMI",
        "n_genes_det": "Detected genes",
    }
    default_metrics = ["mt.pct", "ribo.pct", "total_umi", "n_genes_det"]
    metrics = default_metrics if metrics is None else list(dict.fromkeys(metrics))
    if not metrics:
        raise ValueError("metrics must contain at least one QC metric.")
    missing_metrics = [metric for metric in metrics if metric not in dsd.obs]
    if missing_metrics:
        raise KeyError(
            f"QC metrics not found in dsd.obs: {missing_metrics}. "
            f"Available columns: {list(dsd.obs.columns)}"
        )
    non_numeric = [
        metric for metric in metrics
        if not pd.api.types.is_numeric_dtype(dsd.obs[metric])
    ]
    if non_numeric:
        raise TypeError(f"QC violin metrics must be numeric: {non_numeric}")
    groups, group_order = _groups(dsd, groupby)
    splits, split_order = _split_groups(dsd, split_by)
    _, plate_order = _plate_groups(dsd)
    report_label, title_label = _qc_report_labels(plate_order, report_label)
    colors = _palette(group_order, palette)
    plot_table = pd.DataFrame({
        "metadata_index": dsd.obs.index.astype(str),
        groupby: groups.astype(str).values,
        "_split_value": splits.astype(str).values,
    })
    if split_by is not None:
        plot_table[split_by] = plot_table["_split_value"]
    for metric in metrics:
        plot_table[metric] = dsd.obs[metric].values
    plot_table.drop(columns="_split_value").to_csv(
        out / "qc_metric_violin_plot_table.csv", index=False,
    )
    panel_specs = [
        (split_value, metric)
        for split_value in split_order
        for metric in metrics
    ]
    default_cols = min(2, len(metrics))
    panels_per_row = default_cols if nrow_plate is None else int(nrow_plate)
    if panels_per_row < 1:
        raise ValueError("nrow_plate must be None or a positive integer")
    ncols = min(panels_per_row, len(panel_specs))
    nrows = int(np.ceil(len(panel_specs) / ncols))
    panel_width = max(5.0, min(12.0, 3.5 + 0.8 * len(group_order)))
    figure_width = panel_width * ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(figure_width, 3.8 * nrows), squeeze=False)
    for ax, (split_value, metric) in zip(axes.flat, panel_specs):
        panel = plot_table[plot_table["_split_value"] == str(split_value)]
        data = panel[[groupby, metric]].rename(columns={metric: "value"})
        sns.violinplot(data=data, x=groupby, y="value", hue=groupby, order=group_order, hue_order=group_order, ax=ax, inner="box", cut=0, density_norm="width", palette=colors, legend=False)
        sns.stripplot(data=data, x=groupby, y="value", order=group_order, ax=ax, color="#253746", size=max(1.5, min(3.5, 50 / max(dsd.n_obs, 1))), alpha=.3)
        medians = data.groupby(groupby)["value"].median()
        maxima = data.groupby(groupby)["value"].max()
        if metric == "total_umi":
            ax.set_yscale("log")
            ymin, ymax = ax.get_ylim()
            positive_values = data.loc[data["value"] > 0, "value"]
            global_max = float(positive_values.max()) if len(positive_values) else 1.0
            ax.set_ylim(ymin, max(ymax, global_max * 1.35))
        else:
            ymin, ymax = ax.get_ylim()
            yrange = max(ymax - ymin, 1e-12)
            ax.set_ylim(ymin, ymax + yrange * .16)
        for xpos, plate in enumerate(group_order):
            if plate not in medians or pd.isna(medians[plate]):
                continue
            median = float(medians[plate])
            if metric in {"total_umi", "n_genes_det"}:
                label = f"{median:,.0f}"
            else:
                label = f"{median:.3f}"
            group_max = float(maxima.get(plate, median))
            if metric == "total_umi":
                y_text = max(group_max, 1e-12) * 1.08
            else:
                y_text = group_max + yrange * .035
            ax.text(
                xpos,
                y_text,
                f"Median: {label}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold",
                color="#D32F2F",
                clip_on=False,
            )
        panel_title = metric
        if split_by is not None:
            panel_title = f"{metric} · {split_by}: {split_value}"
        ax.set_title(panel_title)
        ax.set_xlabel(groupby)
        ax.set_ylabel(metric_labels.get(metric, metric))
        if len(group_order) > 4:
            ax.tick_params(axis="x", labelrotation=45, labelsize=max(6, 9 - len(group_order) // 8))
            for tick in ax.get_xticklabels():
                tick.set_horizontalalignment("right")
    for ax in axes.flat[len(panel_specs):]:
        ax.set_visible(False)
    split_text = f", split by {split_by}" if split_by is not None else ""
    fig.suptitle(
        f"{title_label} QC metric distributions by {groupby}{split_text}",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    return _save(
        fig,
        out,
        _split_stem(f"{report_label}_QC_Boxplots", split_by),
        formats,
    )


def generate_outlier_score_scatter_figure(
    dsd: DrugSeqData,
    output_dir: str | Path,
    *,
    formats=("pdf", "png"),
    report_label: str | None = None,
    groupby: str = "plate_id",
    split_by: str | None = None,
    nrow_plate: int | None = None,
    palette: PaletteSpec = None,
    threshold: float | None = None,
    label_outliers: bool = False,
    label_col: str = "well_id",
    max_labels: int = 10,
    output_name: str | None = None,
) -> list[str]:
    """Plot one sample-level MAD outlier score per point.

    Samples are ordered by ``groupby`` while preserving their input order
    within each group. Colors identify groups. A threshold line and labels are
    optional because the score is intended to prioritize review rather than
    impose a universal filtering cutoff. ``split_by`` optionally creates
    separate scatter panels.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _ensure_qc_metrics(dsd)
    groups, group_order = _groups(dsd, groupby)
    splits, split_order = _split_groups(dsd, split_by)
    _, plate_order = _plate_groups(dsd)
    report_label, title_label = _qc_report_labels(plate_order, report_label)
    colors = _palette(group_order, palette)

    obs = dsd.obs.copy()
    table = pd.DataFrame({
        "metadata_index": obs.index.astype(str),
        "sample_id": obs.get(
            "sample_id", pd.Series(obs.index.astype(str), index=obs.index),
        ).astype(str).values,
        groupby: groups.values,
        "_split_value": splits.astype(str).values,
        "outlier_score": pd.to_numeric(obs["outlier_score"], errors="coerce").values,
        "input_order": np.arange(len(obs)),
    })
    for column in ["plate_id", "well_id", "barcode", "compound", "treatment"]:
        if column in obs.columns and column not in table.columns:
            table[column] = obs[column].values
    if split_by is not None:
        table[split_by] = table["_split_value"]
    table[groupby] = pd.Categorical(
        table[groupby].astype(str), categories=group_order, ordered=True,
    )
    table = table.sort_values(
        ["_split_value", groupby, "input_order"], kind="stable",
    ).reset_index(drop=True)
    table[groupby] = table[groupby].astype("object").astype(str)
    table["sample_order"] = (
        table.groupby("_split_value", sort=False).cumcount() + 1
    )
    table["above_threshold"] = (
        table["outlier_score"].gt(float(threshold))
        if threshold is not None else False
    )
    figure_stem = (
        _label_stem(output_name)
        if output_name is not None
        else f"{report_label}_Outlier_Score_Scatter"
    )
    table_stem = (
        f"{figure_stem}_plot_table"
        if output_name is not None else "outlier_score_plot_table"
    )
    table.drop(columns="_split_value").to_csv(
        out / f"{table_stem}.csv", index=False,
    )

    fig, axes = _facet_grid(
        len(split_order), nrow_plate=nrow_plate, panel=(9.0, 5.8),
    )
    for ax, split_value in zip(axes.flat, split_order):
        panel = table[table["_split_value"] == str(split_value)].copy()
        sns.scatterplot(
            data=panel,
            x="sample_order",
            y="outlier_score",
            hue=groupby,
            hue_order=group_order,
            palette=colors,
            s=max(24.0, min(58.0, 1800.0 / max(len(panel), 1))),
            alpha=.8,
            edgecolor="white",
            linewidth=.45,
            ax=ax,
            zorder=3,
        )
        for boundary in panel.groupby(
            groupby, sort=False,
        )["sample_order"].max().iloc[:-1]:
            ax.axvline(
                float(boundary) + .5, color="#CBD5E1",
                linewidth=.7, zorder=1,
            )
        if threshold is not None:
            ax.axhline(
                float(threshold), color="#D32F2F", linestyle="--",
                linewidth=1.1, zorder=2,
            )
            ax.text(
                .995, float(threshold), f" Threshold: {threshold:g}",
                transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                fontsize=8, color="#D32F2F", fontweight="bold",
            )

        if label_outliers and max_labels > 0:
            candidates = (
                panel[panel["above_threshold"]]
                if threshold is not None else panel
            ).nlargest(max_labels, "outlier_score")
            effective_label_col = (
                label_col if label_col in candidates.columns else "sample_id"
            )
            if not candidates.empty:
                ymin, ymax = ax.get_ylim()
                data_top = float(candidates["outlier_score"].max())
                span = max(ymax - ymin, data_top * .1, 1e-9)
                ax.set_ylim(ymin, max(ymax, data_top) + span * .18)
            for offset, (_, row) in enumerate(candidates.iterrows()):
                right_side = offset % 2 == 0
                ax.annotate(
                    str(row[effective_label_col]),
                    (row["sample_order"], row["outlier_score"]),
                    xytext=((5 if right_side else -5), 6 + (offset % 4) * 4),
                    textcoords="offset points",
                    ha="left" if right_side else "right",
                    va="bottom",
                    fontsize=6.5,
                    color="#374151",
                    clip_on=False,
                    arrowprops={
                        "arrowstyle": "-", "color": "#94A3B8",
                        "linewidth": .45,
                    },
                    zorder=4,
                )

        group_positions = panel.groupby(
            groupby, sort=False,
        )["sample_order"].median()
        ax.set_xticks(group_positions.values)
        ax.set_xticklabels(
            group_positions.index.astype(str),
            rotation=35 if len(group_positions) > 4 else 0,
            ha="right" if len(group_positions) > 4 else "center",
        )
        ax.set_xlabel(f"Samples ordered by {groupby}")
        ax.set_ylabel("MAD outlier score")
        panel_title = f"{title_label} sample outlier scores by {groupby}"
        if split_by is not None:
            panel_title = f"{split_by}: {split_value}"
        ax.set_title(panel_title, fontsize=12, fontweight="bold")
        ax.grid(color="#E5E7EB", linestyle="--", linewidth=.6, axis="y")
        ax.set_axisbelow(True)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                title=groupby, frameon=False, fontsize=8,
                title_fontsize=9,
            )
        sns.despine(ax=ax)
    for ax in axes.flat[len(split_order):]:
        ax.set_visible(False)
    if split_by is not None:
        fig.suptitle(
            f"{title_label} sample outlier scores by {groupby}, "
            f"split by {split_by}",
            fontsize=14, fontweight="bold",
        )
        fig.tight_layout(rect=(0, 0, 1, .96))
    else:
        fig.tight_layout()
    return _save(
        fig,
        out,
        _split_stem(figure_stem, split_by),
        formats,
    )


def generate_umi_density_figure(
    dsd: DrugSeqData,
    output_dir: str | Path,
    *,
    formats=("pdf", "png"),
    report_label: str | None = None,
    groupby: str = "plate_id",
    split_by: str | None = None,
    nrow_plate: int | None = None,
    palette: PaletteSpec = None,
) -> list[str]:
    """Generate the per-well total-UMI density figure grouped by metadata."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _ensure_qc_metrics(dsd)
    _, group_order = _plate_groups(dsd)
    report_label, title_label = _qc_report_labels(group_order, report_label)
    return _density_figure(
        dsd,
        "total_umi",
        f"{title_label} UMI density by {groupby}",
        f"{report_label}_UMI_Density",
        out,
        formats,
        groupby=groupby,
        split_by=split_by,
        nrow_plate=nrow_plate,
        palette=palette,
    )


def generate_detected_gene_density_figure(
    dsd: DrugSeqData,
    output_dir: str | Path,
    *,
    formats=("pdf", "png"),
    report_label: str | None = None,
    groupby: str = "plate_id",
    split_by: str | None = None,
    nrow_plate: int | None = None,
    palette: PaletteSpec = None,
) -> list[str]:
    """Generate the per-well detected-gene density figure grouped by metadata."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _ensure_qc_metrics(dsd)
    _, group_order = _plate_groups(dsd)
    report_label, title_label = _qc_report_labels(group_order, report_label)
    return _density_figure(
        dsd,
        "n_genes_det",
        f"{title_label} gene density by {groupby}",
        f"{report_label}_Gene_Density",
        out,
        formats,
        groupby=groupby,
        split_by=split_by,
        nrow_plate=nrow_plate,
        palette=palette,
    )


def generate_qc_distribution_figures(
    dsd: DrugSeqData,
    output_dir: str | Path,
    *,
    formats=("pdf", "png"),
    report_label: str | None = None,
    metrics: Sequence[str] | None = None,
    groupby: str = "plate_id",
    split_by: str | None = None,
    nrow_plate: int | None = None,
    palette: PaletteSpec = None,
    outlier_threshold: float | None = None,
    label_outliers: bool = False,
    outlier_label_col: str = "well_id",
    max_outlier_labels: int = 10,
) -> list[str]:
    """Generate the QC violin, outlier scatter, and two density figures.

    This compatibility wrapper delegates to all four focused sample-QC
    visualization functions. Call those functions directly when a notebook or
    tutorial should place code immediately beside one figure.
    """
    files = generate_qc_metric_violin_figure(
        dsd, output_dir, formats=formats, report_label=report_label,
        metrics=metrics,
        groupby=groupby, split_by=split_by,
        nrow_plate=nrow_plate, palette=palette,
    )
    files += generate_outlier_score_scatter_figure(
        dsd,
        output_dir,
        formats=formats,
        report_label=report_label,
        groupby=groupby,
        split_by=split_by,
        nrow_plate=nrow_plate,
        palette=palette,
        threshold=outlier_threshold,
        label_outliers=label_outliers,
        label_col=outlier_label_col,
        max_labels=max_outlier_labels,
    )
    files += generate_umi_density_figure(
        dsd, output_dir, formats=formats, report_label=report_label,
        groupby=groupby, split_by=split_by,
        nrow_plate=nrow_plate, palette=palette,
    )
    files += generate_detected_gene_density_figure(
        dsd, output_dir, formats=formats, report_label=report_label,
        groupby=groupby, split_by=split_by,
        nrow_plate=nrow_plate, palette=palette,
    )
    return files


def _normalize_identifier_types(
    identifier_types: str | Sequence[str] | None,
) -> list[str]:
    """Normalize requested gene-identifier distribution types."""
    if identifier_types is None:
        requested = ["ensembl_id", "gene_symbol"]
    elif isinstance(identifier_types, str):
        requested = [identifier_types]
    else:
        requested = list(identifier_types)
    aliases = {
        "ensembl": "ensembl_id",
        "ensembl_id": "ensembl_id",
        "ensembl id": "ensembl_id",
        "gene_symbol": "gene_symbol",
        "gene symbol": "gene_symbol",
        "symbol": "gene_symbol",
    }
    normalized = []
    invalid = []
    for value in requested:
        key = str(value).strip().lower().replace("-", "_")
        if key not in aliases:
            invalid.append(str(value))
            continue
        resolved = aliases[key]
        if resolved not in normalized:
            normalized.append(resolved)
    if invalid:
        raise ValueError(
            "identifier_types supports only 'ensembl_id' and 'gene_symbol'; "
            f"invalid values: {invalid}"
        )
    if not normalized:
        raise ValueError("identifier_types must contain at least one identifier type")
    return normalized


def generate_gene_umi_distribution_figures(
    dsd: DrugSeqData,
    output_dir: str | Path,
    *,
    formats=("pdf", "png"),
    identifier_types: str | Sequence[str] | None = None,
    report_label: str | None = None,
    groupby: str = "plate_id",
    split_by: str | None = None,
    nrow_plate: int | None = None,
    palette: PaletteSpec = None,
) -> list[str]:
    """Generate gene-level total-UMI distributions by identifier type.

    ``identifier_types=None`` produces both the Ensembl-ID and gene-symbol
    distributions. Pass one name or a sequence containing ``ensembl_id``
    and/or ``gene_symbol`` to select a subset. Feature totals are calculated
    separately for ``groupby`` and overlaid by color; ``split_by`` optionally
    creates separate panels.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _, group_order = _plate_groups(dsd)
    report_label, _ = _qc_report_labels(group_order, report_label)
    _gene_tables(dsd, out)
    grouped_genes, group_order, split_order = _grouped_gene_umi_table(
        dsd, groupby=groupby, split_by=split_by,
    )
    selected = _normalize_identifier_types(identifier_types)
    files = []
    density_colors = _palette(group_order, palette)
    density_specs = {
        "ensembl_id": (
            grouped_genes[grouped_genes["gene"].str.startswith("ENS")],
            f"{report_label}_UMI_per_Entrez_ID",
            "UMIs per Ensembl ID",
        ),
        "gene_symbol": (
            grouped_genes[
                grouped_genes["gene_symbol"].notna()
                & ~grouped_genes["gene_symbol"].isin(["", "nan", "None"])
                & ~grouped_genes["gene_symbol"].str.startswith("ENS")
            ],
            f"{report_label}_UMI_per_Gene_Symbol",
            "UMIs per gene symbol",
        ),
    }
    plot_tables = []
    for identifier_type in selected:
        subset, stem, title = density_specs[identifier_type]
        subset = subset[subset["total_umi"] > 0].copy()
        subset["log2_total_umi"] = np.log2(subset["total_umi"] + 1)
        subset["identifier_type"] = identifier_type
        if split_by is not None:
            subset["split_by"] = split_by
            subset["split_by_value"] = subset["_split_value"]
        plot_tables.append(subset)
        fig, axes = _facet_grid(
            len(split_order), nrow_plate=nrow_plate, panel=(8.0, 5.2),
        )
        for ax, split_value in zip(axes.flat, split_order):
            panel = subset[subset["_split_value"] == str(split_value)]
            for group in group_order:
                vals = panel.loc[
                    panel[groupby].astype(str) == str(group),
                    "log2_total_umi",
                ]
                if len(vals) > 1 and vals.nunique() > 1:
                    sns.kdeplot(
                        x=vals, fill=True, alpha=.25, linewidth=1.1,
                        color=density_colors[group], label=group, ax=ax,
                    )
                elif len(vals):
                    ax.axvline(
                        float(vals.iloc[0]), color=density_colors[group],
                        linewidth=1.5, label=group,
                    )
            panel_title = title
            if split_by is not None:
                panel_title = f"{split_by}: {split_value}"
            ax.set_title(panel_title, fontsize=12, fontweight="bold")
            ax.set_xlabel("log2(total UMI + 1)")
            ax.set_ylabel("Number of genes (density)")
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend(
                    title=groupby, frameon=False, fontsize=8,
                    title_fontsize=9,
                )
            sns.despine(ax=ax)
        for ax in axes.flat[len(split_order):]:
            ax.set_visible(False)
        if split_by is not None:
            fig.suptitle(title, fontsize=14, fontweight="bold")
            fig.tight_layout(rect=(0, 0, 1, .96))
        else:
            fig.tight_layout()
        files += _save(
            fig, out, _split_stem(stem, split_by), formats,
        )
    pd.concat(plot_tables, ignore_index=True).drop(
        columns="_split_value",
    ).to_csv(out / "gene_umi_distribution_plot_table.csv", index=False)
    return files


def _marker_expression_data(
    dsd: DrugSeqData,
    out: Path,
    *,
    marker_genes: Sequence[str] | None,
    groupby: str,
    split_by: str | None,
    report_label: str | None,
):
    """Prepare reusable sample-level marker expression and display orders."""
    groups, group_order = _groups(dsd, groupby)
    splits, split_order = _split_groups(dsd, split_by)
    _, plate_order = _plate_groups(dsd)
    report_label, title_label = _qc_report_labels(plate_order, report_label)
    gene_table = _gene_tables(dsd, out)
    if marker_genes is None:
        marker_genes = MDA_MB_231_MARKERS
    marker_genes = list(dict.fromkeys(str(gene) for gene in marker_genes))
    if not marker_genes:
        return None
    available = [
        gene for gene in marker_genes
        if gene in set(gene_table["gene_symbol"])
    ]
    pd.DataFrame({
        "requested": marker_genes,
        "available": [gene in available for gene in marker_genes],
    }).to_csv(out / "marker_gene_summary.csv", index=False)
    if not available:
        return None

    x = _matrix(dsd)
    symbols = gene_table["gene_symbol"].values
    compounds = dsd.obs.get(
        "compound", pd.Series("unknown", index=dsd.obs.index),
    ).astype(str).values
    rows = []
    for gene in available:
        frame = pd.DataFrame({
            "gene": gene,
            "expression": x[:, symbols == gene].sum(1),
            groupby: groups.astype(str).values,
            "_split_value": splits.astype(str).values,
            "compound": compounds,
        })
        if split_by is not None:
            frame[split_by] = splits.astype(str).values
        rows.append(frame)
    marker_long = pd.concat(rows, ignore_index=True)
    return (
        marker_long,
        available,
        group_order,
        split_order,
        report_label,
        title_label,
    )


def generate_marker_gene_figure(
    dsd: DrugSeqData,
    output_dir: str | Path,
    *,
    marker_genes: Sequence[str] | None = MDA_MB_231_MARKERS,
    formats=("pdf", "png"),
    report_label: str | None = None,
    groupby: str = "plate_id",
    split_by: str | None = None,
    nrow_plate: int | None = None,
    palette: PaletteSpec = None,
) -> list[str]:
    """Generate a grouped expression panel for specific marker genes."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    prepared = _marker_expression_data(
        dsd,
        out,
        marker_genes=marker_genes,
        groupby=groupby,
        split_by=split_by,
        report_label=report_label,
    )
    if prepared is None:
        return []
    (
        marker_long,
        available,
        group_order,
        split_order,
        report_label,
        title_label,
    ) = prepared
    marker_long.drop(columns="_split_value").to_csv(
        out / "marker_expression_plot_table.csv", index=False,
    )
    summary_groups = ["_split_value", groupby, "gene"]
    marker_summary = marker_long.groupby(
        summary_groups, dropna=False,
    )["expression"].mean().reset_index(name="mean_expression")
    if split_by is not None:
        marker_summary[split_by] = marker_summary["_split_value"]
    marker_summary.drop(columns="_split_value").to_csv(
        out / "marker_expression_summary.csv", index=False,
    )
    group_colors = _palette(group_order, palette)
    panel_specs = [
        (split_value, gene)
        for split_value in split_order
        for gene in available
    ]
    panels_per_row = 5 if nrow_plate is None else int(nrow_plate)
    if panels_per_row < 1:
        raise ValueError("nrow_plate must be None or a positive integer")
    ncols = min(panels_per_row, max(1, len(panel_specs)))
    nrows = int(np.ceil(len(panel_specs) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.25 * ncols, 2.75 * nrows + .6),
        squeeze=False,
    )
    for ax, (split_value, gene) in zip(axes.flat, panel_specs):
        sub = marker_summary[
            (marker_summary["gene"] == gene)
            & (marker_summary["_split_value"] == str(split_value))
        ]
        sns.barplot(
            data=sub, x=groupby, y="mean_expression", hue=groupby,
            order=group_order, hue_order=group_order,
            palette=group_colors, legend=False, ax=ax,
        )
        panel_title = gene
        if split_by is not None:
            panel_title = f"{gene} · {split_by}: {split_value}"
        ax.set_title(panel_title, fontsize=9.5, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Average expression", fontsize=8)
        ax.tick_params(axis="both", labelsize=8)
        if ax.get_legend():
            ax.get_legend().remove()
    for ax in axes.flat[len(panel_specs):]:
        ax.set_visible(False)
    handles = [
        Patch(facecolor=group_colors[group], edgecolor="none", label=group)
        for group in group_order
    ]
    fig.legend(
        handles=handles, title=groupby, loc="upper center",
        bbox_to_anchor=(.5, .94), ncol=min(len(handles), 6),
        frameon=False, fontsize=8, title_fontsize=9,
    )
    split_text = f" · split by {split_by}" if split_by is not None else ""
    fig.suptitle(
        f"{title_label} - Human MDA-MB-231 Marker Genes{split_text}",
        fontsize=14, fontweight="bold", y=.995,
    )
    fig.tight_layout(rect=(0, 0, 1, .86))
    return _save(
        fig,
        out,
        _split_stem(
            f"{report_label}_Human_MDA-MB-231_Markers", split_by,
        ),
        formats,
    )


def generate_marker_gene_dotplot_figure(
    dsd: DrugSeqData,
    output_dir: str | Path,
    *,
    marker_genes: Sequence[str] | None = MDA_MB_231_MARKERS,
    formats=("pdf", "png"),
    report_label: str | None = None,
    groupby: str = "plate_id",
    split_by: str | None = None,
    nrow_plate: int | None = None,
    cmap: str = "RdYlBu_r",
    dot_min_size: float = 18,
    dot_max_size: float = 280,
) -> list[str]:
    """Plot marker means by color and percent-expressing wells by dot size."""
    if dot_min_size <= 0 or dot_max_size <= dot_min_size:
        raise ValueError(
            "dot sizes must satisfy 0 < dot_min_size < dot_max_size"
        )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    prepared = _marker_expression_data(
        dsd,
        out,
        marker_genes=marker_genes,
        groupby=groupby,
        split_by=split_by,
        report_label=report_label,
    )
    if prepared is None:
        return []
    (
        marker_long,
        available,
        group_order,
        split_order,
        report_label,
        title_label,
    ) = prepared
    dot_table = marker_long.groupby(
        ["_split_value", groupby, "gene"], dropna=False,
    )["expression"].agg(
        mean_expression="mean",
        pct_expressing=lambda values: 100 * np.mean(values > 0),
        n_samples="size",
    ).reset_index()
    if split_by is not None:
        dot_table[split_by] = dot_table["_split_value"]
    dot_table.drop(columns="_split_value").to_csv(
        out / "marker_dotplot_table.csv", index=False,
    )

    panels_per_row = 2 if nrow_plate is None else int(nrow_plate)
    if panels_per_row < 1:
        raise ValueError("nrow_plate must be None or a positive integer")
    ncols = min(panels_per_row, max(1, len(split_order)))
    nrows = int(np.ceil(len(split_order) / ncols))
    panel_width = max(6.4, .72 * len(available) + 2.2)
    panel_height = max(3.8, .52 * len(group_order) + 2.5)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(panel_width * ncols, panel_height * nrows + 1.0),
        squeeze=False,
    )
    vmax = float(dot_table["mean_expression"].max())
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    scatter = None
    used_axes = []
    for ax, split_value in zip(axes.flat, split_order):
        used_axes.append(ax)
        sub = dot_table[dot_table["_split_value"] == str(split_value)].copy()
        sub["gene_position"] = sub["gene"].map(
            {gene: index for index, gene in enumerate(available)}
        )
        sub["group_position"] = sub[groupby].astype(str).map(
            {group: index for index, group in enumerate(group_order)}
        )
        sizes = dot_min_size + (
            sub["pct_expressing"].clip(0, 100) / 100
        ) * (dot_max_size - dot_min_size)
        scatter = ax.scatter(
            sub["gene_position"],
            sub["group_position"],
            s=sizes,
            c=sub["mean_expression"],
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            edgecolor="#334155",
            linewidth=.55,
            alpha=.95,
        )
        ax.set_xticks(range(len(available)))
        ax.set_xticklabels(available, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(group_order)))
        ax.set_yticklabels(group_order, fontsize=9)
        ax.set_xlabel("Marker gene", fontsize=9)
        ax.set_ylabel(groupby, fontsize=9)
        panel_title = "All samples"
        if split_by is not None:
            panel_title = f"{split_by}: {split_value}"
        ax.set_title(panel_title, fontsize=11, fontweight="bold")
        ax.set_xlim(-.65, len(available) - .35)
        ax.set_ylim(len(group_order) - .5, -.5)
        ax.grid(color="#E2E8F0", linewidth=.6)
        ax.set_axisbelow(True)
        sns.despine(ax=ax)
    for ax in axes.flat[len(split_order):]:
        ax.set_visible(False)
    fig.suptitle(
        f"{title_label} - Marker Gene Dot Plot",
        fontsize=14,
        fontweight="bold",
        y=.995,
    )
    fig.subplots_adjust(
        left=.07, right=.90, bottom=.18, top=.86, wspace=.25, hspace=.34,
    )
    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=used_axes, fraction=.025, pad=.025)
        cbar.set_label("Mean expression", fontsize=9)
        cbar.ax.tick_params(labelsize=8)
    size_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none",
            markerfacecolor="#CBD5E1", markeredgecolor="#334155",
            markersize=np.sqrt(
                dot_min_size + (pct / 100) * (dot_max_size - dot_min_size)
            ),
            label=f"{pct}%",
        )
        for pct in (0, 50, 100)
    ]
    fig.legend(
        handles=size_handles,
        title="Wells expressing",
        loc="lower center",
        bbox_to_anchor=(.48, .015),
        ncol=3,
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )
    return _save(
        fig,
        out,
        _split_stem(
            f"{report_label}_Human_MDA-MB-231_Markers_Dotplot", split_by,
        ),
        formats,
    )


def generate_marker_gene_violin_figure(
    dsd: DrugSeqData,
    output_dir: str | Path,
    *,
    marker_genes: Sequence[str] | None = MDA_MB_231_MARKERS,
    formats=("pdf", "png"),
    report_label: str | None = None,
    groupby: str = "plate_id",
    split_by: str | None = None,
    nrow_plate: int | None = None,
    palette: PaletteSpec = None,
    log_values: bool = False,
) -> list[str]:
    """Plot one grouped per-well expression violin panel per marker gene."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    prepared = _marker_expression_data(
        dsd,
        out,
        marker_genes=marker_genes,
        groupby=groupby,
        split_by=split_by,
        report_label=report_label,
    )
    if prepared is None:
        return []
    (
        marker_long,
        available,
        group_order,
        split_order,
        report_label,
        title_label,
    ) = prepared
    marker_long["plot_expression"] = (
        np.log1p(marker_long["expression"])
        if log_values else marker_long["expression"]
    )
    marker_long.drop(columns="_split_value").to_csv(
        out / "marker_violin_plot_table.csv", index=False,
    )
    group_colors = _palette(group_order, palette)
    panel_specs = [
        (split_value, gene)
        for split_value in split_order
        for gene in available
    ]
    panels_per_row = 5 if nrow_plate is None else int(nrow_plate)
    if panels_per_row < 1:
        raise ValueError("nrow_plate must be None or a positive integer")
    ncols = min(panels_per_row, max(1, len(panel_specs)))
    nrows = int(np.ceil(len(panel_specs) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.25 * ncols, 2.9 * nrows + .8),
        squeeze=False,
        sharey=False,
    )
    for ax, (split_value, gene) in zip(axes.flat, panel_specs):
        sub = marker_long[
            (marker_long["_split_value"] == str(split_value))
            & (marker_long["gene"] == gene)
        ]
        sns.violinplot(
            data=sub,
            x=groupby,
            y="plot_expression",
            hue=groupby,
            order=group_order,
            hue_order=group_order,
            palette=group_colors,
            cut=0,
            inner="quartile",
            density_norm="width",
            common_norm=False,
            linewidth=.8,
            dodge=True,
            legend=False,
            ax=ax,
        )
        if ax.get_legend():
            ax.get_legend().remove()
        ax.tick_params(axis="x", labelsize=8)
        ax.set_xlabel("", fontsize=9)
        ylabel = "log1p(expression)" if log_values else "Expression per well"
        ax.set_ylabel(ylabel, fontsize=9)
        panel_title = gene
        if split_by is not None:
            panel_title = f"{gene} · {split_by}: {split_value}"
        ax.set_title(panel_title, fontsize=9.5, fontweight="bold")
        ax.grid(color="#E2E8F0", linewidth=.6, axis="y")
        ax.set_axisbelow(True)
        sns.despine(ax=ax)
    for ax in axes.flat[len(panel_specs):]:
        ax.set_visible(False)
    handles = [
        Patch(facecolor=group_colors[group], edgecolor="none", label=group)
        for group in group_order
    ]
    fig.legend(
        handles=handles,
        title=groupby,
        loc="upper center",
        bbox_to_anchor=(.5, .94),
        ncol=min(len(handles), 6),
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )
    fig.suptitle(
        f"{title_label} - Marker Gene Violin Plot",
        fontsize=14,
        fontweight="bold",
        y=.995,
    )
    fig.tight_layout(rect=(0, 0, 1, .86))
    return _save(
        fig,
        out,
        _split_stem(
            f"{report_label}_Human_MDA-MB-231_Markers_Violin", split_by,
        ),
        formats,
    )


def generate_gene_summary_figures(
    dsd: DrugSeqData,
    output_dir: str | Path,
    *,
    formats=("pdf", "png"),
    identifier_types: str | Sequence[str] | None = None,
    marker_genes: Sequence[str] | None = MDA_MB_231_MARKERS,
    report_label: str | None = None,
    groupby: str = "plate_id",
    split_by: str | None = None,
    nrow_plate: int | None = None,
    palette: PaletteSpec = None,
) -> list[str]:
    """Compatibility wrapper for gene UMI distributions and marker panels."""
    files = generate_gene_umi_distribution_figures(
        dsd,
        output_dir,
        formats=formats,
        identifier_types=identifier_types,
        report_label=report_label,
        groupby=groupby,
        split_by=split_by,
        nrow_plate=nrow_plate,
        palette=palette,
    )
    if marker_genes is not None and len(marker_genes):
        files += generate_marker_gene_figure(
            dsd,
            output_dir,
            marker_genes=marker_genes,
            formats=formats,
            report_label=report_label,
            groupby=groupby,
            split_by=split_by,
            nrow_plate=nrow_plate,
            palette=palette,
        )
    return files


def generate_star_read_qc_figure(
    dsd: DrugSeqData,
    star_root: str | Path,
    output_dir: str | Path,
    *,
    formats=("pdf", "png"),
    report_label: str | None = None,
    groupby: str = "plate_id",
    split_by: str | None = None,
    nrow_plate: int | None = None,
    palette: PaletteSpec = None,
) -> list[str]:
    """Generate the STARsolo read-summary multilayered doughnut figure."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _, group_order = _plate_groups(dsd)
    return _star_figures(
        star_root,
        out,
        formats,
        dsd=dsd,
        plate_order=group_order,
        report_label=report_label,
        groupby=groupby,
        split_by=split_by,
        nrow_plate=nrow_plate,
        palette=palette,
        include_read_summary=True,
        include_saturation=False,
    )


def generate_star_read_count_violin_figure(
    dsd: DrugSeqData,
    star_root: str | Path,
    output_dir: str | Path,
    *,
    formats=("pdf", "png"),
    report_label: str | None = None,
    groupby: str = "plate_id",
    split_by: str | None = None,
    nrow_plate: int | None = None,
    palette: PaletteSpec = None,
    point_color: str = "#253746",
    annotation_color: str = "#D32F2F",
) -> list[str]:
    """Plot STAR matched-read counts for every analyzed well as violins.

    ``cbMatch`` from ``CellReads.stats`` is the per-well read count. Every
    sample in ``dsd`` is retained; wells without a matching STAR record are
    assigned zero and marked by ``matched_star_record=False`` in the source
    table. Mean and median values are annotated above each group.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _, plate_order = _plate_groups(dsd)
    report_label, title_label = _qc_report_labels(plate_order, report_label)
    table = _star_well_read_table(
        dsd, star_root, groupby=groupby, split_by=split_by,
    )
    if table.empty:
        return []
    if split_by is None:
        table["_plot_split"] = "all"
    else:
        table["_plot_split"] = table[split_by].astype(str)
    group_order = sorted(pd.unique(table[groupby].astype(str)))
    split_order = sorted(pd.unique(table["_plot_split"].astype(str)))
    colors = _palette(group_order, palette)

    source_cols = [
        column for column in [
            "sample_id", "plate_id", "barcode_norm", groupby, split_by,
            "read_count", "matched_star_record",
        ]
        if column is not None and column in table.columns
    ]
    source_cols = list(dict.fromkeys(source_cols))
    source = table[source_cols].copy().rename(
        columns={"barcode_norm": "barcode"},
    )
    source.to_csv(out / "star_read_count_per_well.csv", index=False)
    summary_groups = [groupby]
    if split_by is not None and split_by != groupby:
        summary_groups.insert(0, split_by)
    summary = table.groupby(summary_groups, dropna=False).agg(
        n_wells=("read_count", "size"),
        n_matched_star_records=("matched_star_record", "sum"),
        mean_read_count=("read_count", "mean"),
        median_read_count=("read_count", "median"),
        min_read_count=("read_count", "min"),
        max_read_count=("read_count", "max"),
    ).reset_index()
    summary.to_csv(out / "star_read_count_summary.csv", index=False)

    fig, axes = _facet_grid(
        len(split_order), nrow_plate=nrow_plate, panel=(7.2, 5.4),
    )
    for ax, split_value in zip(axes.flat, split_order):
        panel = table[table["_plot_split"].astype(str) == split_value]
        panel_groups = [
            group for group in group_order
            if group in set(panel[groupby].astype(str))
        ]
        sns.violinplot(
            data=panel,
            x=groupby,
            y="read_count",
            hue=groupby,
            order=panel_groups,
            hue_order=panel_groups,
            palette={group: colors[group] for group in panel_groups},
            inner="box",
            cut=0,
            density_norm="width",
            legend=False,
            ax=ax,
        )
        sns.stripplot(
            data=panel,
            x=groupby,
            y="read_count",
            order=panel_groups,
            color=point_color,
            size=max(1.8, min(3.5, 500 / max(len(panel), 1))),
            alpha=.3,
            ax=ax,
        )
        stats = panel.groupby(groupby, dropna=False)["read_count"].agg(
            ["mean", "median", "max"],
        )
        ymin, ymax = ax.get_ylim()
        span = max(ymax - ymin, float(panel["read_count"].max()) * .1, 1.0)
        ax.set_ylim(ymin, ymax + span * .24)
        for xpos, group in enumerate(panel_groups):
            if group not in stats.index:
                continue
            row = stats.loc[group]
            ax.text(
                xpos,
                float(row["max"]) + span * .045,
                f"Median: {row['median']:,.0f}\nMean: {row['mean']:,.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color=annotation_color,
                clip_on=False,
            )
        panel_title = f"{title_label} wells"
        if split_by is not None:
            panel_title = f"{split_by}: {split_value}"
        ax.set_title(panel_title, fontsize=11, fontweight="bold")
        ax.set_xlabel(groupby)
        ax.set_ylabel("Matched reads per well (cbMatch)")
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        if len(panel_groups) > 4:
            ax.tick_params(axis="x", labelrotation=35)
        ax.grid(color="#E5E7EB", linestyle="--", linewidth=.6, axis="y")
        ax.set_axisbelow(True)
        sns.despine(ax=ax)
    for ax in axes.flat[len(split_order):]:
        ax.set_visible(False)
    split_text = f" · split by {split_by}" if split_by is not None else ""
    fig.suptitle(
        f"{title_label} STARsolo read counts per well{split_text}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, .95))
    return _save(
        fig,
        out,
        _split_stem(
            f"{report_label}_STAR_Read_Count_Violin", split_by,
        ),
        formats,
    )


def generate_star_saturation_figures(
    dsd: DrugSeqData,
    star_root: str | Path,
    output_dir: str | Path,
    *,
    formats=("pdf", "png"),
    report_label: str | None = None,
    groupby: str = "plate_id",
    split_by: str | None = None,
    nrow_plate: int | None = None,
    palette: PaletteSpec = None,
) -> list[str]:
    """Generate STARsolo read-count versus detected-gene saturation plots.

    ``groupby`` controls point colors in a combined axes. ``split_by=None`` is
    the default; providing a metadata column creates separate panels.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _, plate_order = _plate_groups(dsd)
    return _star_figures(
        star_root,
        out,
        formats,
        dsd=dsd,
        plate_order=plate_order,
        report_label=report_label,
        groupby=groupby,
        split_by=split_by,
        nrow_plate=nrow_plate,
        palette=palette,
        include_read_summary=False,
        include_saturation=True,
    )


def generate_star_qc_figures(
    dsd: DrugSeqData,
    star_root: str | Path,
    output_dir: str | Path,
    *,
    formats=("pdf", "png"),
    report_label: str | None = None,
    groupby: str = "plate_id",
    split_by: str | None = None,
    nrow_plate: int | None = None,
    palette: PaletteSpec = None,
) -> list[str]:
    """Compatibility wrapper for the three focused STARsolo figure functions."""
    files = generate_star_read_qc_figure(
        dsd,
        star_root,
        output_dir,
        formats=formats,
        report_label=report_label,
        groupby=groupby,
        split_by=split_by,
        nrow_plate=nrow_plate,
        palette=palette,
    )
    files += generate_star_read_count_violin_figure(
        dsd,
        star_root,
        output_dir,
        formats=formats,
        report_label=report_label,
        groupby=groupby,
        split_by=split_by,
        nrow_plate=nrow_plate,
        palette=palette,
    )
    files += generate_star_saturation_figures(
        dsd,
        star_root,
        output_dir,
        formats=formats,
        report_label=report_label,
        groupby=groupby,
        split_by=split_by,
        nrow_plate=nrow_plate,
        palette=palette,
    )
    return files


def generate_qc_report(
    dsd: DrugSeqData,
    output_dir: str | Path,
    *,
    formats=("pdf", "png"),
    star_root: str | Path | None = None,
    marker_genes: Sequence[str] | None = MDA_MB_231_MARKERS,
    gene_umi_identifier_types: str | Sequence[str] | None = None,
    plate_gene: str | Sequence[str] | None = "FN1",
    plate_gene_set: Sequence[str] | None = None,
    plate_gene_aggregation: str = "mean",
    plate_nrow: int = 8,
    plate_ncol: int = 12,
    nrow_plate: int | None = None,
    report_label: str | None = None,
    violin_metrics: Sequence[str] | None = None,
    groupby: str = "plate_id",
    split_by: str | None = None,
    palette: PaletteSpec = None,
    cmap: str = "RdYlBu_r",
    outlier_threshold: float | None = None,
    label_outliers: bool = False,
    outlier_label_col: str = "well_id",
    max_outlier_labels: int = 10,
) -> list[str]:
    """Run all public QC figure groups and write a producer-aware manifest.

    This convenience function remains the batch-mode entry point. Tutorials
    and interactive analyses may call each focused figure function
    independently.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _ensure_qc_metrics(dsd)
    if {"plate_id", "sample_type"}.issubset(dsd.obs.columns):
        compute_plate_qc(dsd, inplace=True)
    dsd.obs.to_csv(out / "sample_qc_metrics.csv")
    audit = validate_metadata(dsd, verbose=False)
    pd.DataFrame({
        "ok": [audit["ok"]],
        "issues": ["; ".join(audit["issues"])],
    }).to_csv(out / "metadata_audit.csv", index=False)

    plate_files = []
    plate_files += generate_plate_layout_figures(
        dsd, out, formats=formats, plate_value="total_umi", continuous=True,
        split_by=split_by,
        nrow=plate_nrow, ncol=plate_ncol,
        nrow_plate=nrow_plate,
        output_name="plate_layout_detected_umi", cmap=cmap,
    )
    plate_files += generate_plate_layout_figures(
        dsd, out, formats=formats, plate_value="n_genes_det", continuous=True,
        split_by=split_by,
        nrow=plate_nrow, ncol=plate_ncol,
        nrow_plate=nrow_plate,
        output_name="plate_layout_detected_genes", cmap=cmap,
    )
    plate_files += generate_plate_layout_figures(
        dsd, out, formats=formats, plate_value="compound",
        split_by=split_by,
        nrow=plate_nrow, ncol=plate_ncol,
        nrow_plate=nrow_plate,
        output_name="plate_layout_compound", palette=palette,
    )
    plate_files += generate_plate_layout_figures(
        dsd, out, formats=formats, well_color="#E2E8F0", label_col="well_id",
        split_by=split_by,
        show_legend=False, nrow=plate_nrow, ncol=plate_ncol,
        nrow_plate=nrow_plate,
        output_name="plate_layout_well_id", well_size=.88, label_size=6,
    )
    if plate_gene is not None:
        try:
            plate_files += generate_plate_layout_figures(
                dsd, out, formats=formats, plate_gene=plate_gene,
                split_by=split_by,
                gene_aggregation=plate_gene_aggregation, continuous=True,
                nrow=plate_nrow, ncol=plate_ncol,
                nrow_plate=nrow_plate,
                output_name="plate_layout_gene_expression", cmap=cmap,
            )
        except KeyError:
            # The optional expression map is omitted when its requested genes
            # are absent, matching the prior single-gene report behavior.
            pass
    if plate_gene_set is not None:
        try:
            plate_files += generate_plate_layout_figures(
                dsd, out, formats=formats, plate_gene=plate_gene_set,
                split_by=split_by,
                gene_aggregation=plate_gene_aggregation, continuous=True,
                nrow=plate_nrow, ncol=plate_ncol,
                nrow_plate=nrow_plate,
                output_name="plate_layout_gene_set_expression", cmap=cmap,
            )
        except KeyError:
            pass

    figure_groups = [
        (
            "generate_plate_layout_figures",
            plate_files,
        ),
        (
            "generate_qc_metric_violin_figure",
            generate_qc_metric_violin_figure(
                dsd, out, formats=formats, report_label=report_label,
                metrics=violin_metrics,
                groupby=groupby, split_by=split_by,
                nrow_plate=nrow_plate if split_by is not None else None,
                palette=palette,
            ),
        ),
        (
            "generate_outlier_score_scatter_figure",
            generate_outlier_score_scatter_figure(
                dsd, out, formats=formats, report_label=report_label,
                groupby=groupby, split_by=split_by,
                nrow_plate=nrow_plate if split_by is not None else None,
                palette=palette,
                threshold=outlier_threshold,
                label_outliers=label_outliers,
                label_col=outlier_label_col,
                max_labels=max_outlier_labels,
            ),
        ),
        (
            "generate_umi_density_figure",
            generate_umi_density_figure(
                dsd, out, formats=formats, report_label=report_label,
                groupby=groupby, split_by=split_by,
                nrow_plate=nrow_plate if split_by is not None else None,
                palette=palette,
            ),
        ),
        (
            "generate_detected_gene_density_figure",
            generate_detected_gene_density_figure(
                dsd, out, formats=formats, report_label=report_label,
                groupby=groupby, split_by=split_by,
                nrow_plate=nrow_plate if split_by is not None else None,
                palette=palette,
            ),
        ),
        (
            "generate_gene_umi_distribution_figures",
            generate_gene_umi_distribution_figures(
                dsd,
                out,
                formats=formats,
                identifier_types=gene_umi_identifier_types,
                report_label=report_label,
                groupby=groupby,
                split_by=split_by,
                nrow_plate=nrow_plate if split_by is not None else None,
                palette=palette,
            ),
        ),
    ]
    if marker_genes is not None and len(marker_genes):
        figure_groups.append(
            (
                "generate_marker_gene_figure",
                generate_marker_gene_figure(
                    dsd,
                    out,
                    marker_genes=marker_genes,
                    formats=formats,
                    report_label=report_label,
                    groupby=groupby,
                    split_by=split_by,
                    nrow_plate=nrow_plate if split_by is not None else None,
                    palette=palette,
                ),
            )
        )
    if star_root:
        figure_groups.extend([
            (
                "generate_star_read_qc_figure",
                generate_star_read_qc_figure(
                    dsd, star_root, out, formats=formats,
                    report_label=report_label, groupby=groupby,
                    split_by=split_by,
                    nrow_plate=nrow_plate, palette=palette,
                ),
            ),
            (
                "generate_star_read_count_violin_figure",
                generate_star_read_count_violin_figure(
                    dsd, star_root, out, formats=formats,
                    report_label=report_label, groupby=groupby,
                    split_by=split_by,
                    nrow_plate=nrow_plate, palette=palette,
                ),
            ),
            (
                "generate_star_saturation_figures",
                generate_star_saturation_figures(
                    dsd, star_root, out, formats=formats,
                    report_label=report_label, groupby=groupby,
                    split_by=split_by,
                    nrow_plate=nrow_plate,
                    palette=palette,
                ),
            ),
        ])

    manifest_rows = [
        {"figure": figure, "producer": producer}
        for producer, figures in figure_groups
        for figure in figures
    ]
    pd.DataFrame(manifest_rows, columns=["figure", "producer"]).to_csv(
        out / "qc_figure_manifest.csv", index=False,
    )
    return [row["figure"] for row in manifest_rows]


def _mix_species_table(
    dsd: DrugSeqData,
    *,
    expected_col: str | None,
    species_a_prefix: str,
    species_b_prefix: str,
    species_a_name: str,
    species_b_name: str,
    expected_species_map: Mapping[str, str] | None = None,
    assignment_threshold: float = .5,
    min_species_umi: float = 0,
    human_prefix: str | None = None,
    mouse_prefix: str | None = None,
) -> pd.DataFrame:
    table = compute_mix_species_qc(
        dsd,
        species_a_prefix=species_a_prefix,
        species_b_prefix=species_b_prefix,
        species_a_name=species_a_name,
        species_b_name=species_b_name,
        expected_species_map=expected_species_map,
        assignment_threshold=assignment_threshold,
        min_species_umi=min_species_umi,
        human_prefix=human_prefix,
        mouse_prefix=mouse_prefix,
        expected_col=expected_col,
    ).copy()
    table["plate_id"] = table.get("plate_id", pd.Series("all", index=table.index)).astype(str)
    if expected_col and expected_col in table.columns:
        table["expected_label"] = table[expected_col].fillna("unknown").astype(str)
    elif "expected_species" in table.columns:
        table["expected_label"] = table["expected_species"].fillna("unknown").astype(str)
    else:
        table["expected_label"] = table["observed_species"].fillna("unknown").astype(str)
    table["species_a_pct"] = table["species_a_fraction"] * 100
    table["species_b_pct"] = table["species_b_fraction"] * 100
    table["species_purity_pct"] = np.nanmax(
        table[["species_a_pct", "species_b_pct"]].to_numpy(dtype=float), axis=1
    )
    if "cross_contamination_fraction" in table:
        table["cross_contamination_pct"] = (
            table["cross_contamination_fraction"] * 100
        )
    # Preserve the historical source-table columns for the default pair.
    if "human_fraction" in table:
        table["human_pct"] = table["human_fraction"] * 100
    if "mouse_fraction" in table:
        table["mouse_pct"] = table["mouse_fraction"] * 100
    if "sample_id" not in table.columns:
        table["sample_id"] = table.index.astype(str)
    return table


def _mix_species_names(table: pd.DataFrame) -> tuple[str, str]:
    """Return the configured display names stored in the plot table."""
    return (
        str(table["species_a_name"].iloc[0]),
        str(table["species_b_name"].iloc[0]),
    )


def _facet_grid(
    n_panels: int,
    *,
    nrow_plate: int | None = None,
    max_cols: int = 2,
    panel: tuple[float, float] = (5.2, 4.4),
):
    panels_per_row = max_cols if nrow_plate is None else int(nrow_plate)
    if panels_per_row < 1:
        raise ValueError("nrow_plate must be None or a positive integer")
    ncols = min(panels_per_row, max(1, n_panels))
    nrows = int(np.ceil(max(1, n_panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel[0] * ncols, panel[1] * nrows), squeeze=False)
    return fig, axes


def _style_axis(
    ax,
    *,
    grid_color: str = "#DCE6F2",
    panel_border_color: str = "#334155",
):
    """Apply the bordered, light-grid report style used by mix-species plots."""
    ax.grid(color=grid_color, linestyle=(0, (4, 4)), linewidth=.75, axis="y")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(panel_border_color)
        spine.set_linewidth(.8)
    ax.tick_params(colors=panel_border_color)
    ax.xaxis.label.set_color(panel_border_color)
    ax.yaxis.label.set_color(panel_border_color)


def _add_panel_note(ax, text: str, *, x: float = .5, ha: str = "center", color: str = "#D32F2F"):
    ax.text(
        x,
        1.04,
        text,
        transform=ax.transAxes,
        ha=ha,
        va="bottom",
        fontsize=7.2,
        fontweight="bold",
        color=color,
        clip_on=False,
        bbox={
            "boxstyle": "round,pad=.2",
            "facecolor": "white",
            "edgecolor": "#CBD5E1",
            "alpha": .92,
        },
        zorder=6,
    )


def _scatter_size(n: int) -> float:
    return max(18.0, min(55.0, 1200.0 / max(n, 1)))


def _compact_count_formatter(x, _pos):
    if not np.isfinite(x):
        return ""
    value = abs(float(x))
    sign = "-" if x < 0 else ""
    if value >= 1_000_000:
        label = f"{value / 1_000_000:.1f}M"
        return sign + label.replace(".0M", "M")
    if value >= 1_000:
        return f"{sign}{value / 1_000:.0f}k"
    return f"{x:.0f}"


def _annotate_group_median(
    ax,
    data: pd.DataFrame,
    order: list[str],
    metric: str,
    *,
    percent: bool = False,
    color: str = "#D32F2F",
    above_data: bool = False,
):
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin
    grouped = data.groupby("plate_id", dropna=False)[metric]
    medians = grouped.median()
    maxima = grouped.max()
    label_tops = []
    for xpos, plate in enumerate(order):
        if plate not in medians or pd.isna(medians[plate]):
            continue
        median = float(medians[plate])
        label = f"{median:.2f}%" if percent and abs(median) < 1 else (f"{median:.1f}%" if percent else f"{median:,.0f}")
        anchor = float(maxima[plate]) if above_data else median
        y = anchor + yrange * .04
        label_tops.append(y)
        ax.text(
            xpos, y, f"Median: {label}", ha="center", va="bottom",
            fontsize=7.5, fontweight="bold", color=color, clip_on=False,
        )
    required_top = max(label_tops, default=ymax) + yrange * .07
    ax.set_ylim(ymin, max(ymax + yrange * .1, required_top))


def _resolve_barnyard_metric_species(
    table: pd.DataFrame,
    metric_species: str | Sequence[str] | None,
) -> list[str]:
    """Resolve which assigned base species receive barnyard annotations."""
    species_a_name, species_b_name = _mix_species_names(table)
    configured = [species_a_name, species_b_name]
    available = set(
        table.get(
            "expected_species", pd.Series(index=table.index, dtype=str),
        ).dropna().astype(str)
    )
    aliases = {
        "species_a": [species_a_name],
        "a": [species_a_name],
        "species_b": [species_b_name],
        "b": [species_b_name],
        "both": configured,
        "all": configured,
        "base": [name for name in configured if name in available],
        "none": [],
    }

    if metric_species is None:
        return []
    requested = [metric_species] if isinstance(metric_species, str) else list(metric_species)
    selected: list[str] = []
    for value in requested:
        key = str(value).strip().casefold()
        if key in aliases:
            matches = aliases[key]
        else:
            matches = [name for name in configured if name.casefold() == key]
            if not matches:
                raise ValueError(
                    "metric_species must be None, 'base', 'both', "
                    "'species_a', 'species_b', or configured species names "
                    f"{configured}; received {value!r}"
                )
        for name in matches:
            if name not in selected:
                selected.append(name)
    return selected


def _mix_barnyard_metric_summary(
    table: pd.DataFrame,
    metric_species: str | Sequence[str] | None,
) -> pd.DataFrame:
    """Summarize base-species purity for barnyard annotations and auditing."""
    species_a_name, species_b_name = _mix_species_names(table)
    selected = _resolve_barnyard_metric_species(table, metric_species)
    columns = [
        "plate_id", "base_species", "metric", "n_wells",
        "minimum", "mean", "median", "maximum",
    ]
    if "expected_species" not in table or not selected:
        return pd.DataFrame(columns=columns)
    purity_columns = {
        species_a_name: "species_a_pct",
        species_b_name: "species_b_pct",
    }
    rows = []
    for plate in sorted(pd.unique(table["plate_id"].astype(str))):
        panel = table[table["plate_id"].astype(str) == plate]
        expected = panel["expected_species"].astype(str)
        for base_species in selected:
            values = panel.loc[
                expected == base_species, purity_columns[base_species],
            ].dropna()
            if values.empty:
                continue
            rows.append({
                "plate_id": plate,
                "base_species": base_species,
                "metric": "base_species_purity_pct",
                "n_wells": int(values.size),
                "minimum": float(values.min()),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "maximum": float(values.max()),
            })
    return pd.DataFrame(rows, columns=columns)


def _plot_mix_barnyard(
    table: pd.DataFrame, out: Path, formats, *,
    nrow_plate: int | None = None, palette: PaletteSpec = None,
    reference_line_color: str = "#7F7F7F",
    annotation_color: str = "#D32F2F",
    edge_color: str = "white",
    grid_color: str = "#DCE6F2",
    panel_border_color: str = "#334155",
    point_alpha: float = .72,
    show_observed_style: bool = True,
    metric_species: str | Sequence[str] | None = "base",
) -> list[str]:
    species_a_name, species_b_name = _mix_species_names(table)
    metric_summary = _mix_barnyard_metric_summary(table, metric_species)
    metric_summary.to_csv(
        out / "species_barnyard_metric_summary.csv", index=False,
    )
    plates = sorted(pd.unique(table["plate_id"].astype(str)))
    hue_order = sorted(pd.unique(table["expected_label"].astype(str)))
    style_order = [
        value for value in [
            species_a_name, species_b_name, "mixed", "low_umi", "undetermined",
        ] if value in set(table["observed_species"].astype(str))
    ]
    colors = _palette(hue_order, palette)
    lim = max(
        float(table["species_a_umi"].max()),
        float(table["species_b_umi"].max()),
        1.0,
    ) * 1.08
    fig, axes = _facet_grid(
        len(plates), nrow_plate=nrow_plate, panel=(5.6, 5.6),
    )
    handles = labels = None
    for ax, plate in zip(axes.flat, plates):
        sub = table[table["plate_id"] == plate]
        ax.plot(
            [0, lim], [0, lim], "--", color=reference_line_color,
            linewidth=1, zorder=1,
        )
        scatter_kwargs = dict(
            data=sub,
            x="species_a_umi",
            y="species_b_umi",
            hue="expected_label",
            hue_order=hue_order,
            palette=colors,
            s=_scatter_size(len(table)),
            alpha=point_alpha,
            edgecolor=edge_color,
            linewidth=.35,
            zorder=3,
            ax=ax,
        )
        if show_observed_style:
            scatter_kwargs.update(
                style="observed_species",
                style_order=style_order or None,
            )
        sns.scatterplot(**scatter_kwargs)
        base_notes = []
        panel_metrics = metric_summary[
            metric_summary["plate_id"].astype(str) == str(plate)
        ]
        for row in panel_metrics.itertuples(index=False):
            base_notes.extend([
                f"Base {row.base_species} purity: "
                f"min {row.minimum:.1f}% · max {row.maximum:.1f}%",
                f"mean {row.mean:.1f}% · median {row.median:.1f}%",
            ])
        if base_notes:
            _add_panel_note(
                ax, "\n".join(base_notes), color=annotation_color,
            )
        title_pad = 42 + max(0, len(base_notes) - 2) * 10
        ax.set_title(
            str(plate), fontsize=11, fontweight="bold", pad=title_pad,
        )
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        # Both species counts share one numeric scale. A square plotting
        # box keeps equal count differences visually equal on both axes and
        # prevents the barnyard cloud/diagonal from being stretched.
        ax.set_aspect("equal", adjustable="box")
        ax.set_box_aspect(1)
        ax.set_xlabel(f"{species_a_name.capitalize()} UMI counts")
        ax.set_ylabel(f"{species_b_name.capitalize()} UMI counts")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
        ax.xaxis.set_major_formatter(FuncFormatter(_compact_count_formatter))
        ax.yaxis.set_major_formatter(FuncFormatter(_compact_count_formatter))
        ax.tick_params(axis="x", labelsize=8.5)
        _style_axis(
            ax,
            grid_color=grid_color,
            panel_border_color=panel_border_color,
        )
        if handles is None and ax.get_legend():
            handles, labels = ax.get_legend_handles_labels()
        if ax.get_legend():
            ax.get_legend().remove()
    for ax in axes.flat[len(plates):]:
        ax.set_visible(False)
    if handles:
        fig.legend(
            handles, labels, title="Expected/base label", loc="upper center",
            bbox_to_anchor=(.5, .92), ncol=min(6, len(labels)),
            frameon=False, fontsize=8, title_fontsize=9,
        )
    fig.suptitle(
        f"Combined {species_a_name}–{species_b_name} barnyard plot "
        "(raw counts; no base species)",
        fontsize=14,
        fontweight="bold",
        y=.995,
    )
    top = .72 if len(metric_summary["base_species"].unique()) > 1 else .80
    fig.tight_layout(rect=(.01, .02, .99, top))
    return _save(fig, out, "species_barnyard", formats)


def _plot_mix_purity(
    table: pd.DataFrame, out: Path, formats, *,
    nrow_plate: int | None = None, palette: PaletteSpec = None,
    reference_line_color: str = "#7F7F7F",
    guide_line_color: str = "#CBD5E1",
    edge_color: str = "white",
    grid_color: str = "#DCE6F2",
    panel_border_color: str = "#334155",
    point_alpha: float = .72,
    show_observed_style: bool = True,
) -> list[str]:
    species_a_name, species_b_name = _mix_species_names(table)
    plates = sorted(pd.unique(table["plate_id"].astype(str)))
    hue_order = sorted(pd.unique(table["expected_label"].astype(str)))
    style_order = [
        value for value in [
            species_a_name, species_b_name, "mixed", "low_umi", "undetermined",
        ] if value in set(table["observed_species"].astype(str))
    ]
    colors = _palette(hue_order, palette)
    fig, axes = _facet_grid(
        len(plates), nrow_plate=nrow_plate, panel=(5.4, 5.4),
    )
    handles = labels = None
    for ax, plate in zip(axes.flat, plates):
        sub = table[table["plate_id"] == plate]
        ax.plot(
            [0, 100], [100, 0], "--", color=reference_line_color,
            linewidth=1, zorder=1,
        )
        ax.axvline(50, color=guide_line_color, linewidth=.9, zorder=1)
        ax.axhline(50, color=guide_line_color, linewidth=.9, zorder=1)
        scatter_kwargs = dict(
            data=sub,
            x="species_a_pct",
            y="species_b_pct",
            hue="expected_label",
            hue_order=hue_order,
            palette=colors,
            s=_scatter_size(len(table)),
            alpha=point_alpha,
            edgecolor=edge_color,
            linewidth=.35,
            zorder=3,
            ax=ax,
        )
        if show_observed_style:
            scatter_kwargs.update(
                style="observed_species",
                style_order=style_order or None,
            )
        sns.scatterplot(**scatter_kwargs)
        discordant = int((sub.get("concordance", pd.Series(index=sub.index, dtype=str)) == "discordant").sum())
        controls = int((sub.get("concordance", pd.Series(index=sub.index, dtype=str)) == "signal_control").sum())
        ax.set_title(
            f"{plate}\ndiscordant: {discordant}   signal controls: {controls}",
            fontsize=10.5,
            fontweight="bold",
            pad=12,
        )
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        # Complementary species percentages share the same 0–100 scale.
        # A square plotting box prevents the purity diagonal and point cloud
        # from being stretched in either direction.
        ax.set_aspect("equal", adjustable="box")
        ax.set_box_aspect(1)
        ax.set_xlabel(f"{species_a_name.capitalize()} UMI fraction (%)")
        ax.set_ylabel(f"{species_b_name.capitalize()} UMI fraction (%)")
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        _style_axis(
            ax,
            grid_color=grid_color,
            panel_border_color=panel_border_color,
        )
        if handles is None and ax.get_legend():
            handles, labels = ax.get_legend_handles_labels()
        if ax.get_legend():
            ax.get_legend().remove()
    for ax in axes.flat[len(plates):]:
        ax.set_visible(False)
    if handles:
        fig.legend(
            handles, labels, title="Expected/base label", loc="lower center",
            ncol=min(6, len(labels)), frameon=False, fontsize=8,
            title_fontsize=9,
        )
    fig.suptitle(
        f"Species purity: {species_a_name} and {species_b_name}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, .1, 1, .90))
    return _save(fig, out, "species_purity", formats)


def _plot_mix_concordance(
    table: pd.DataFrame, out: Path, formats, *,
    nrow_plate: int | None = None, palette: PaletteSpec = None,
    edge_color: str = "white",
    label_color: str = "white",
    grid_color: str = "#DCE6F2",
    panel_border_color: str = "#334155",
    fill_alpha: float = .95,
) -> list[str]:
    species_a_name, species_b_name = _mix_species_names(table)
    counts = table.groupby(["plate_id", "expected_label", "observed_species"], dropna=False).size().reset_index(name="n_samples")
    plates = sorted(pd.unique(counts["plate_id"].astype(str)))
    expected_order = sorted(pd.unique(counts["expected_label"].astype(str)))
    observed_values = sorted(pd.unique(counts["observed_species"].astype(str)))
    species_order = [
        value for value in [
            species_a_name, species_b_name, "mixed", "low_umi", "undetermined",
        ] if value in set(observed_values)
    ]
    species_order += [x for x in observed_values if x not in species_order]
    colors = _palette(species_order, palette)
    fig, axes = _facet_grid(
        len(plates), nrow_plate=nrow_plate, panel=(5.2, 4.0),
    )
    for ax, plate in zip(axes.flat, plates):
        sub = counts[counts["plate_id"] == plate]
        bottom = np.zeros(len(expected_order))
        for species in species_order:
            vals = []
            for expected in expected_order:
                match = sub[(sub["expected_label"] == expected) & (sub["observed_species"] == species)]
                vals.append(int(match["n_samples"].iloc[0]) if len(match) else 0)
            bars = ax.bar(
                expected_order, vals, bottom=bottom,
                color=colors.get(species), label=species,
                edgecolor=edge_color, linewidth=.5, alpha=fill_alpha,
            )
            for bar, value, base in zip(bars, vals, bottom):
                if value:
                    ax.text(bar.get_x() + bar.get_width() / 2, base + value / 2, str(value), ha="center", va="center", fontsize=8, fontweight="bold", color=label_color)
            bottom += np.asarray(vals)
        ax.set_title(str(plate), fontsize=11, fontweight="bold")
        ax.set_xlabel("Expected label")
        ax.set_ylabel("Retained wells")
        ax.tick_params(axis="x", labelrotation=25)
        _style_axis(
            ax,
            grid_color=grid_color,
            panel_border_color=panel_border_color,
        )
    for ax in axes.flat[len(plates):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Observed species", loc="lower center", ncol=min(5, len(labels)), frameon=False, fontsize=8, title_fontsize=9)
    fig.suptitle("Expected label vs observed species", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, .12, 1, .94))
    return _save(fig, out, "species_concordance_counts", formats)


def _plot_mix_metric_distributions(
    table: pd.DataFrame, out: Path, formats, *, palette: PaletteSpec = None,
    point_color: str = "#253746",
    median_color: str = "#D32F2F",
    grid_color: str = "#DCE6F2",
    panel_border_color: str = "#334155",
) -> list[str]:
    metric_specs = [
        ("species_purity_pct", "Dominant species fraction (%)", True),
        (
            "cross_contamination_pct",
            "Off-target species fraction within assigned wells (%)",
            True,
        ),
        ("total_umi", "Total UMI", False),
        ("n_genes_det", "Detected genes", False),
        ("pct_mito", "Mitochondrial UMI fraction (%)", True),
    ]
    metrics = [
        spec for spec in metric_specs
        if spec[0] in table.columns and table[spec[0]].notna().any()
    ]
    ncols = 2
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.5, 3.8 * nrows), squeeze=False)
    order = sorted(pd.unique(table["plate_id"].astype(str)))
    colors = _palette(order, palette)
    for ax, (metric, label, is_percent) in zip(axes.flat, metrics):
        data = table[["plate_id", metric]].rename(columns={metric: "value"}).dropna()
        sns.violinplot(data=data, x="plate_id", y="value", hue="plate_id", order=order, hue_order=order, palette=colors, inner="box", cut=0, density_norm="width", legend=False, ax=ax)
        sns.stripplot(data=data, x="plate_id", y="value", order=order, color=point_color, size=max(1.5, min(3.2, 500 / max(len(table), 1))), alpha=.28, ax=ax)
        _annotate_group_median(
            ax, data, order, "value", percent=is_percent,
            color=median_color, above_data=True,
        )
        if metric == "total_umi":
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
        elif is_percent:
            decimals = 1 if data["value"].max() < 10 else 0
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=decimals))
        else:
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.set_title(metric, fontsize=11, fontweight="bold")
        ax.set_xlabel("plate_id")
        ax.set_ylabel(label)
        _style_axis(
            ax,
            grid_color=grid_color,
            panel_border_color=panel_border_color,
        )
    for ax in axes.flat[len(metrics):]:
        ax.set_visible(False)
    fig.suptitle("Mix-species technical QC metrics by plate_id", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .96))
    return _save(fig, out, "mix_species_qc_metrics", formats)


def _plot_mix_fraction_by_expected(
    table: pd.DataFrame, out: Path, formats, *, palette: PaletteSpec = None,
    edge_color: str = "white",
    grid_color: str = "#DCE6F2",
    panel_border_color: str = "#334155",
    fill_alpha: float = .9,
    point_alpha: float = .35,
) -> list[str]:
    species_a_name, species_b_name = _mix_species_names(table)
    long = pd.concat([
        table[["plate_id", "expected_label", "species_a_pct"]]
        .rename(columns={"species_a_pct": "fraction"})
        .assign(species=species_a_name),
        table[["plate_id", "expected_label", "species_b_pct"]]
        .rename(columns={"species_b_pct": "fraction"})
        .assign(species=species_b_name),
    ], ignore_index=True).dropna(subset=["fraction"])
    long.to_csv(out / "species_fraction_plot_table.csv", index=False)
    expected_order = sorted(pd.unique(long["expected_label"].astype(str)))
    fig, axes = _facet_grid(2, max_cols=2, panel=(5.3, 4.2))
    colors = _palette(sorted(pd.unique(long["plate_id"].astype(str))), palette)
    for ax, species in zip(axes.flat, [species_a_name, species_b_name]):
        sub = long[long["species"] == species]
        sns.boxplot(
            data=sub, x="expected_label", y="fraction", hue="plate_id",
            order=expected_order, palette=colors, showfliers=False,
            boxprops={"alpha": fill_alpha}, ax=ax,
        )
        sns.stripplot(
            data=sub, x="expected_label", y="fraction", hue="plate_id",
            order=expected_order, palette=colors, dodge=True,
            size=max(1.5, min(3.0, 500 / max(len(long), 1))),
            alpha=point_alpha, edgecolor=edge_color, linewidth=.25,
            legend=False, ax=ax,
        )
        if ax.get_legend():
            ax.get_legend().remove()
        ax.set_title(f"{species.capitalize()} fraction", fontsize=11, fontweight="bold")
        ax.set_xlabel("Expected label")
        ax.set_ylabel("UMI fraction (%)")
        ax.tick_params(axis="x", labelrotation=25)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        _style_axis(
            ax,
            grid_color=grid_color,
            panel_border_color=panel_border_color,
        )
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title="plate_id", loc="lower center", ncol=min(4, len(labels)), frameon=False, fontsize=8, title_fontsize=9)
    fig.suptitle("Species fractions by expected label", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, .12, 1, .94))
    return _save(fig, out, "species_fraction_by_expected", formats)


def _plot_mix_contamination_by_base(
    table: pd.DataFrame,
    out: Path,
    formats,
    *,
    palette: PaletteSpec = None,
    point_color: str = "#253746",
    median_color: str = "#D32F2F",
    grid_color: str = "#DCE6F2",
    panel_border_color: str = "#334155",
) -> list[str]:
    """Plot off-target fractions separately for each assigned base species."""
    species_a_name, species_b_name = _mix_species_names(table)
    plot_columns = [
        "sample_id", "plate_id", "expected_label", "base_species",
        "cross_contamination_pct", "on_target_fraction",
        "off_target_umi", "on_target_umi",
    ]
    if {"base_species", "cross_contamination_pct"}.issubset(table.columns):
        plot_table = table.loc[
            table["base_species"].isin([species_a_name, species_b_name]),
            plot_columns,
        ].copy()
    else:
        plot_table = pd.DataFrame(columns=plot_columns)
    plot_table.to_csv(
        out / "species_cross_contamination_by_base_plot_table.csv", index=False,
    )
    fig, axes = _facet_grid(2, max_cols=2, panel=(5.3, 4.2))
    plate_order = sorted(pd.unique(table["plate_id"].astype(str)))
    colors = _palette(plate_order, palette)
    for ax, (base_species, contaminant_species) in zip(
        axes.flat,
        [
            (species_a_name, species_b_name),
            (species_b_name, species_a_name),
        ],
    ):
        sub = plot_table[plot_table["base_species"] == base_species]
        if sub.empty:
            ax.text(
                .5, .5, f"No wells assigned to {base_species}",
                transform=ax.transAxes, ha="center", va="center",
            )
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            sns.violinplot(
                data=sub,
                x="plate_id",
                y="cross_contamination_pct",
                hue="plate_id",
                order=plate_order,
                hue_order=plate_order,
                palette=colors,
                inner="box",
                cut=0,
                density_norm="width",
                legend=False,
                ax=ax,
            )
            sns.stripplot(
                data=sub,
                x="plate_id",
                y="cross_contamination_pct",
                order=plate_order,
                color=point_color,
                size=max(1.8, min(3.5, 220 / max(len(sub), 1))),
                alpha=.35,
                ax=ax,
            )
            median_data = sub[["plate_id", "cross_contamination_pct"]].rename(
                columns={"cross_contamination_pct": "value"}
            )
            _annotate_group_median(
                ax,
                median_data,
                plate_order,
                "value",
                percent=True,
                color=median_color,
            )
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=1))
            ax.set_xlabel("plate_id")
            ax.set_ylabel(
                f"Off-target {contaminant_species} fraction (%)"
            )
        ax.set_title(
            f"Base species: {base_species}\n"
            f"Contaminating species: {contaminant_species}",
            fontsize=11,
            fontweight="bold",
        )
        _style_axis(
            ax,
            grid_color=grid_color,
            panel_border_color=panel_border_color,
        )
    fig.suptitle(
        "Cross-well contamination by assigned base species",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, .94))
    return _save(fig, out, "species_cross_contamination_by_base", formats)


def _mix_species_figure_input(
    dsd: DrugSeqData,
    output_dir: str | Path,
    *,
    expected_col: str | None,
    species_a_prefix: str = "ENSG",
    species_b_prefix: str = "ENSMUSG",
    species_a_name: str = "human",
    species_b_name: str = "mouse",
    expected_species_map: Mapping[str, str] | None = None,
    assignment_threshold: float = .5,
    min_species_umi: float = 0,
    human_prefix: str | None = None,
    mouse_prefix: str | None = None,
) -> tuple[pd.DataFrame, Path]:
    """Prepare the shared table and output directory for one mix figure."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    table = _mix_species_table(
        dsd,
        expected_col=expected_col,
        species_a_prefix=species_a_prefix,
        species_b_prefix=species_b_prefix,
        species_a_name=species_a_name,
        species_b_name=species_b_name,
        expected_species_map=expected_species_map,
        assignment_threshold=assignment_threshold,
        min_species_umi=min_species_umi,
        human_prefix=human_prefix,
        mouse_prefix=mouse_prefix,
    )
    return table, out


def _validate_mix_alpha(value: float, name: str) -> None:
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")


def generate_mix_species_barnyard_figure(
    dsd: DrugSeqData,
    output_dir: str | Path,
    *,
    formats=("pdf", "png"),
    expected_col: str | None = "compound",
    species_a_prefix: str = "ENSG",
    species_b_prefix: str = "ENSMUSG",
    species_a_name: str = "human",
    species_b_name: str = "mouse",
    expected_species_map: Mapping[str, str] | None = None,
    assignment_threshold: float = .5,
    min_species_umi: float = 0,
    human_prefix: str | None = None,
    mouse_prefix: str | None = None,
    nrow_plate: int | None = None,
    metric_species: str | Sequence[str] | None = "base",
    expected_palette: PaletteSpec = None,
    reference_line_color: str = "#7F7F7F",
    annotation_color: str = "#D32F2F",
    edge_color: str = "white",
    grid_color: str = "#DCE6F2",
    panel_border_color: str = "#334155",
    point_alpha: float = .72,
    show_observed_style: bool = True,
) -> list[str]:
    """Generate only the configurable two-species UMI barnyard figure.

    Every facet uses identical x/y limits and a square 1:1 plotting box so
    species-A and species-B count distances have the same visual scale.
    ``metric_species`` controls which assigned base-species purity summaries
    are shown; ``None`` hides them.
    """
    _validate_mix_alpha(point_alpha, "point_alpha")
    table, out = _mix_species_figure_input(
        dsd, output_dir, expected_col=expected_col,
        species_a_prefix=species_a_prefix,
        species_b_prefix=species_b_prefix,
        species_a_name=species_a_name,
        species_b_name=species_b_name,
        expected_species_map=expected_species_map,
        assignment_threshold=assignment_threshold,
        min_species_umi=min_species_umi,
        human_prefix=human_prefix, mouse_prefix=mouse_prefix,
    )
    table.to_csv(out / "species_barnyard_plot_table.csv", index_label="metadata_index")
    return _plot_mix_barnyard(
        table, out, formats, nrow_plate=nrow_plate,
        palette=expected_palette,
        reference_line_color=reference_line_color,
        annotation_color=annotation_color,
        edge_color=edge_color,
        grid_color=grid_color,
        panel_border_color=panel_border_color,
        point_alpha=point_alpha,
        show_observed_style=show_observed_style,
        metric_species=metric_species,
    )


def generate_mix_species_purity_figure(
    dsd: DrugSeqData,
    output_dir: str | Path,
    *,
    formats=("pdf", "png"),
    expected_col: str | None = "compound",
    species_a_prefix: str = "ENSG",
    species_b_prefix: str = "ENSMUSG",
    species_a_name: str = "human",
    species_b_name: str = "mouse",
    expected_species_map: Mapping[str, str] | None = None,
    assignment_threshold: float = .5,
    min_species_umi: float = 0,
    human_prefix: str | None = None,
    mouse_prefix: str | None = None,
    nrow_plate: int | None = None,
    expected_palette: PaletteSpec = None,
    reference_line_color: str = "#7F7F7F",
    guide_line_color: str = "#CBD5E1",
    edge_color: str = "white",
    grid_color: str = "#DCE6F2",
    panel_border_color: str = "#334155",
    point_alpha: float = .72,
    show_observed_style: bool = True,
) -> list[str]:
    """Generate only the configurable two-species purity figure."""
    _validate_mix_alpha(point_alpha, "point_alpha")
    table, out = _mix_species_figure_input(
        dsd, output_dir, expected_col=expected_col,
        species_a_prefix=species_a_prefix,
        species_b_prefix=species_b_prefix,
        species_a_name=species_a_name,
        species_b_name=species_b_name,
        expected_species_map=expected_species_map,
        assignment_threshold=assignment_threshold,
        min_species_umi=min_species_umi,
        human_prefix=human_prefix, mouse_prefix=mouse_prefix,
    )
    table.to_csv(out / "species_purity_plot_table.csv", index_label="metadata_index")
    return _plot_mix_purity(
        table, out, formats, nrow_plate=nrow_plate,
        palette=expected_palette,
        reference_line_color=reference_line_color,
        guide_line_color=guide_line_color,
        edge_color=edge_color,
        grid_color=grid_color,
        panel_border_color=panel_border_color,
        point_alpha=point_alpha,
        show_observed_style=show_observed_style,
    )


def generate_mix_species_concordance_figure(
    dsd: DrugSeqData,
    output_dir: str | Path,
    *,
    formats=("pdf", "png"),
    expected_col: str | None = "compound",
    species_a_prefix: str = "ENSG",
    species_b_prefix: str = "ENSMUSG",
    species_a_name: str = "human",
    species_b_name: str = "mouse",
    expected_species_map: Mapping[str, str] | None = None,
    assignment_threshold: float = .5,
    min_species_umi: float = 0,
    human_prefix: str | None = None,
    mouse_prefix: str | None = None,
    nrow_plate: int | None = None,
    observed_palette: PaletteSpec = None,
    edge_color: str = "white",
    label_color: str = "white",
    grid_color: str = "#DCE6F2",
    panel_border_color: str = "#334155",
    fill_alpha: float = .95,
) -> list[str]:
    """Generate only the expected-label versus observed-species bars."""
    _validate_mix_alpha(fill_alpha, "fill_alpha")
    table, out = _mix_species_figure_input(
        dsd, output_dir, expected_col=expected_col,
        species_a_prefix=species_a_prefix,
        species_b_prefix=species_b_prefix,
        species_a_name=species_a_name,
        species_b_name=species_b_name,
        expected_species_map=expected_species_map,
        assignment_threshold=assignment_threshold,
        min_species_umi=min_species_umi,
        human_prefix=human_prefix, mouse_prefix=mouse_prefix,
    )
    plot_table = table.groupby(
        ["plate_id", "expected_label", "observed_species"], dropna=False,
    ).size().reset_index(name="n_samples")
    plot_table.to_csv(out / "species_concordance_plot_table.csv", index=False)
    return _plot_mix_concordance(
        table, out, formats, nrow_plate=nrow_plate,
        palette=observed_palette,
        edge_color=edge_color,
        label_color=label_color,
        grid_color=grid_color,
        panel_border_color=panel_border_color,
        fill_alpha=fill_alpha,
    )


def generate_mix_species_qc_metrics_figure(
    dsd: DrugSeqData,
    output_dir: str | Path,
    *,
    formats=("pdf", "png"),
    expected_col: str | None = "compound",
    species_a_prefix: str = "ENSG",
    species_b_prefix: str = "ENSMUSG",
    species_a_name: str = "human",
    species_b_name: str = "mouse",
    expected_species_map: Mapping[str, str] | None = None,
    assignment_threshold: float = .5,
    min_species_umi: float = 0,
    human_prefix: str | None = None,
    mouse_prefix: str | None = None,
    plate_palette: PaletteSpec = None,
    point_color: str = "#253746",
    annotation_color: str = "#D32F2F",
    grid_color: str = "#DCE6F2",
    panel_border_color: str = "#334155",
) -> list[str]:
    """Generate only the plate-level mix-species technical-QC panel."""
    table, out = _mix_species_figure_input(
        dsd, output_dir, expected_col=expected_col,
        species_a_prefix=species_a_prefix,
        species_b_prefix=species_b_prefix,
        species_a_name=species_a_name,
        species_b_name=species_b_name,
        expected_species_map=expected_species_map,
        assignment_threshold=assignment_threshold,
        min_species_umi=min_species_umi,
        human_prefix=human_prefix, mouse_prefix=mouse_prefix,
    )
    columns = [
        column for column in [
            "sample_id", "plate_id", "species_purity_pct", "total_umi",
            "cross_contamination_pct", "n_genes_det", "pct_mito",
        ] if column in table
    ]
    table[columns].to_csv(out / "mix_species_qc_metrics_plot_table.csv", index=False)
    return _plot_mix_metric_distributions(
        table, out, formats,
        palette=plate_palette,
        point_color=point_color,
        median_color=annotation_color,
        grid_color=grid_color,
        panel_border_color=panel_border_color,
    )


def generate_mix_species_fraction_figure(
    dsd: DrugSeqData,
    output_dir: str | Path,
    *,
    formats=("pdf", "png"),
    expected_col: str | None = "compound",
    species_a_prefix: str = "ENSG",
    species_b_prefix: str = "ENSMUSG",
    species_a_name: str = "human",
    species_b_name: str = "mouse",
    expected_species_map: Mapping[str, str] | None = None,
    assignment_threshold: float = .5,
    min_species_umi: float = 0,
    human_prefix: str | None = None,
    mouse_prefix: str | None = None,
    plate_palette: PaletteSpec = None,
    edge_color: str = "white",
    grid_color: str = "#DCE6F2",
    panel_border_color: str = "#334155",
    fill_alpha: float = .95,
    point_alpha: float = .35,
) -> list[str]:
    """Generate only the species-fraction-by-expected-label figure."""
    _validate_mix_alpha(fill_alpha, "fill_alpha")
    _validate_mix_alpha(point_alpha, "point_alpha")
    table, out = _mix_species_figure_input(
        dsd, output_dir, expected_col=expected_col,
        species_a_prefix=species_a_prefix,
        species_b_prefix=species_b_prefix,
        species_a_name=species_a_name,
        species_b_name=species_b_name,
        expected_species_map=expected_species_map,
        assignment_threshold=assignment_threshold,
        min_species_umi=min_species_umi,
        human_prefix=human_prefix, mouse_prefix=mouse_prefix,
    )
    return _plot_mix_fraction_by_expected(
        table, out, formats,
        palette=plate_palette,
        edge_color=edge_color,
        grid_color=grid_color,
        panel_border_color=panel_border_color,
        fill_alpha=fill_alpha,
        point_alpha=point_alpha,
    )


def generate_mix_species_contamination_figure(
    dsd: DrugSeqData,
    output_dir: str | Path,
    *,
    formats=("pdf", "png"),
    expected_col: str | None = "expected_species",
    species_a_prefix: str = "ENSG",
    species_b_prefix: str = "ENSMUSG",
    species_a_name: str = "human",
    species_b_name: str = "mouse",
    expected_species_map: Mapping[str, str] | None = None,
    assignment_threshold: float = .5,
    min_species_umi: float = 0,
    human_prefix: str | None = None,
    mouse_prefix: str | None = None,
    plate_palette: PaletteSpec = None,
    point_color: str = "#253746",
    annotation_color: str = "#D32F2F",
    grid_color: str = "#DCE6F2",
    panel_border_color: str = "#334155",
) -> list[str]:
    """Plot cross-contamination separately for both assigned base species."""
    table, out = _mix_species_figure_input(
        dsd,
        output_dir,
        expected_col=expected_col,
        species_a_prefix=species_a_prefix,
        species_b_prefix=species_b_prefix,
        species_a_name=species_a_name,
        species_b_name=species_b_name,
        expected_species_map=expected_species_map,
        assignment_threshold=assignment_threshold,
        min_species_umi=min_species_umi,
        human_prefix=human_prefix,
        mouse_prefix=mouse_prefix,
    )
    return _plot_mix_contamination_by_base(
        table,
        out,
        formats,
        palette=plate_palette,
        point_color=point_color,
        median_color=annotation_color,
        grid_color=grid_color,
        panel_border_color=panel_border_color,
    )


def generate_mix_species_report(
    dsd: DrugSeqData,
    output_dir: str | Path,
    *,
    formats=("pdf", "png"),
    expected_col: str | None = "compound",
    species_a_prefix: str = "ENSG",
    species_b_prefix: str = "ENSMUSG",
    species_a_name: str = "human",
    species_b_name: str = "mouse",
    expected_species_map: Mapping[str, str] | None = None,
    assignment_threshold: float = .5,
    min_species_umi: float = 0,
    human_prefix: str | None = None,
    mouse_prefix: str | None = None,
    nrow_plate: int | None = None,
    barnyard_metric_species: str | Sequence[str] | None = "base",
    palette: PaletteSpec = None,
    expected_palette: PaletteSpec = None,
    observed_palette: PaletteSpec = None,
    plate_palette: PaletteSpec = None,
    reference_line_color: str = "#7F7F7F",
    guide_line_color: str = "#CBD5E1",
    annotation_color: str = "#D32F2F",
    edge_color: str = "white",
    grid_color: str = "#DCE6F2",
    panel_border_color: str = "#334155",
    point_color: str = "#253746",
    label_color: str = "white",
    point_alpha: float = .72,
    fill_alpha: float = .95,
    show_observed_style: bool = True,
) -> list[str]:
    """Generate all independent mix-species figures and source tables.

    ``palette`` remains a backward-compatible fallback for every categorical
    layer. The role-specific palettes override it for expected labels,
    observed-species fills, and plate groups. Line, annotation, edge, grid,
    panel-border, point, and in-bar label colors are independently adjustable.
    """
    _validate_mix_alpha(point_alpha, "point_alpha")
    _validate_mix_alpha(fill_alpha, "fill_alpha")
    expected_palette = palette if expected_palette is None else expected_palette
    observed_palette = palette if observed_palette is None else observed_palette
    plate_palette = palette if plate_palette is None else plate_palette
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    table = _mix_species_table(
        dsd,
        expected_col=expected_col,
        species_a_prefix=species_a_prefix,
        species_b_prefix=species_b_prefix,
        species_a_name=species_a_name,
        species_b_name=species_b_name,
        expected_species_map=expected_species_map,
        assignment_threshold=assignment_threshold,
        min_species_umi=min_species_umi,
        human_prefix=human_prefix,
        mouse_prefix=mouse_prefix,
    )
    table.to_csv(out / "per_sample_species_qc.csv", index_label="metadata_index")
    resolved_species_a, resolved_species_b = _mix_species_names(table)
    summary_group_cols = ["plate_id", "expected_label"]
    if "expected_species" in table:
        summary_group_cols.append("expected_species")
    species_summary = table.groupby(summary_group_cols, dropna=False).agg(
        n_retained=("sample_id", "size"),
        species_a_majority=("observed_species", lambda x: int((x == resolved_species_a).sum())),
        species_b_majority=("observed_species", lambda x: int((x == resolved_species_b).sum())),
        mixed_or_undetermined=("observed_species", lambda x: int(x.isin(["mixed", "low_umi", "undetermined"]).sum())),
        discordant=("concordance", lambda x: int((x == "discordant").sum())) if "concordance" in table else ("observed_species", lambda x: 0),
        signal_controls=("concordance", lambda x: int((x == "signal_control").sum())) if "concordance" in table else ("observed_species", lambda x: 0),
        median_species_a_pct=("species_a_pct", "median"),
        median_species_b_pct=("species_b_pct", "median"),
        median_cross_contamination_pct=("cross_contamination_pct", "median") if "cross_contamination_pct" in table else ("species_purity_pct", lambda x: np.nan),
        median_total_umi=("total_umi", "median"),
        median_n_genes=("n_genes_det", "median"),
    ).reset_index()
    # Keep the historical summary names for existing human–mouse consumers.
    if resolved_species_a.casefold() == "human":
        species_summary["human_majority"] = species_summary["species_a_majority"]
        species_summary["median_human_pct"] = species_summary["median_species_a_pct"]
    elif resolved_species_b.casefold() == "human":
        species_summary["human_majority"] = species_summary["species_b_majority"]
        species_summary["median_human_pct"] = species_summary["median_species_b_pct"]
    if resolved_species_a.casefold() == "mouse":
        species_summary["mouse_majority"] = species_summary["species_a_majority"]
        species_summary["median_mouse_pct"] = species_summary["median_species_a_pct"]
    elif resolved_species_b.casefold() == "mouse":
        species_summary["mouse_majority"] = species_summary["species_b_majority"]
        species_summary["median_mouse_pct"] = species_summary["median_species_b_pct"]
    species_summary.to_csv(out / "species_summary.csv", index=False)
    concordance = table.groupby(["plate_id", "expected_label", "observed_species", "concordance"] if "concordance" in table else ["plate_id", "expected_label", "observed_species"], dropna=False).size().reset_index(name="n_samples")
    concordance.to_csv(out / "species_concordance.csv", index=False)
    audit_cols = [c for c in [
        "sample_id", "plate_id", "well_id", "barcode", "expected_label",
        "expected_species", "base_species", "observed_species", "concordance",
        "species_a_name", "species_b_name", "species_a_pct", "species_b_pct",
        "species_a_umi", "species_b_umi", "on_target_umi", "off_target_umi",
        "cross_contamination_pct", "total_umi",
    ] if c in table]
    audit_mask = table["observed_species"].isin(["low_umi", "mixed", "undetermined"])
    if "concordance" in table:
        audit_mask = audit_mask | table["concordance"].isin(["discordant", "signal_control"])
    table.loc[audit_mask, audit_cols].to_csv(out / "mix_species_outlier_audit.csv", index=False)
    figure_groups = []
    barnyard_files = _plot_mix_barnyard(
        table, out, formats, nrow_plate=nrow_plate,
        palette=expected_palette,
        reference_line_color=reference_line_color,
        annotation_color=annotation_color,
        edge_color=edge_color,
        grid_color=grid_color,
        panel_border_color=panel_border_color,
        point_alpha=point_alpha,
        show_observed_style=show_observed_style,
        metric_species=barnyard_metric_species,
    )
    figure_groups.append(("generate_mix_species_barnyard_figure", barnyard_files))
    purity_files = _plot_mix_purity(
        table, out, formats, nrow_plate=nrow_plate,
        palette=expected_palette,
        reference_line_color=reference_line_color,
        guide_line_color=guide_line_color,
        edge_color=edge_color,
        grid_color=grid_color,
        panel_border_color=panel_border_color,
        point_alpha=point_alpha,
        show_observed_style=show_observed_style,
    )
    figure_groups.append(("generate_mix_species_purity_figure", purity_files))
    concordance_files = _plot_mix_concordance(
        table, out, formats, nrow_plate=nrow_plate,
        palette=observed_palette,
        edge_color=edge_color,
        label_color=label_color,
        grid_color=grid_color,
        panel_border_color=panel_border_color,
        fill_alpha=fill_alpha,
    )
    figure_groups.append(("generate_mix_species_concordance_figure", concordance_files))
    metric_files = _plot_mix_metric_distributions(
        table, out, formats,
        palette=plate_palette,
        point_color=point_color,
        median_color=annotation_color,
        grid_color=grid_color,
        panel_border_color=panel_border_color,
    )
    figure_groups.append(("generate_mix_species_qc_metrics_figure", metric_files))
    fraction_files = _plot_mix_fraction_by_expected(
        table, out, formats,
        palette=plate_palette,
        edge_color=edge_color,
        grid_color=grid_color,
        panel_border_color=panel_border_color,
        fill_alpha=fill_alpha,
        point_alpha=min(point_alpha, .55),
    )
    figure_groups.append(("generate_mix_species_fraction_figure", fraction_files))
    contamination_files = _plot_mix_contamination_by_base(
        table, out, formats,
        palette=plate_palette,
        point_color=point_color,
        median_color=annotation_color,
        grid_color=grid_color,
        panel_border_color=panel_border_color,
    )
    figure_groups.append((
        "generate_mix_species_contamination_figure", contamination_files,
    ))
    manifest_rows = [
        {"figure": figure, "producer": producer}
        for producer, figures in figure_groups
        for figure in figures
    ]
    pd.DataFrame(manifest_rows, columns=["figure", "producer"]).to_csv(
        out / "mix_species_figure_manifest.csv", index=False,
    )
    return [row["figure"] for row in manifest_rows]


# ---------------------------------------------------------------------------
# Comparison / DEG / enrichment workflow
# ---------------------------------------------------------------------------

import json
import re
import warnings
import anndata as ad

from .differential import compute_multi_de, summarise_comparison
from .enrichment import run_enrichment
from .normalization import normalize_counts
from .filtering import filter_samples, filter_genes
from .plots import (
    plot_comparison_volcano, plot_comparison_heatmap, plot_deg_counts,
    plot_gene_boxplot, plot_gene_violin, plot_enrichment_dotplot,
    plot_enrichment_barplot,
)

DEFAULT_FIGURES = ("volcano", "heatmap", "bar", "box", "violin", "enrichment")


def _sanitize_stem(name) -> str:
    return re.sub(r"[^\w\-]+", "_", str(name)).strip("_") or "unnamed"


def _backend_versions() -> dict:
    versions = {}
    for pkg in ("drugseqpy", "anndata", "scanpy", "numpy", "pandas",
                "scipy", "statsmodels", "pydeseq2", "gseapy", "rpy2"):
        try:
            from importlib.metadata import version
            versions[pkg] = version(pkg)
        except Exception:
            versions[pkg] = None
    return versions


def _json_safe(value):
    """Convert workflow configuration values into JSON-native structures."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _default_boxplot_genes(dsd, n: int = 6) -> list[str]:
    """Union of top significant genes (by min padj) across contrasts."""
    tables = dsd.adata.uns.get("comparison_results") or \
        dsd.adata.uns.get("de_results", {})
    genes, seen = [], set()
    for df in tables.values():
        sig = df[df["significant"].fillna(False).astype(bool)]
        for g in sig.sort_values("padj")["gene"].tolist():
            if g not in seen:
                seen.add(g)
                genes.append(g)
    return genes[:n]


def run_comparison_workflow(
    dsd,
    output_dir: str | Path | None = None,
    *,
    comparisons: dict | list | None = None,
    group_col: str = "compound",
    reference: str = "DMSO",
    exclude_levels: list[str] | tuple[str, ...] | None = None,
    methods: list[str] | tuple[str, ...] = ("limma_voom",),
    batch_cols: list[str] | None = None,
    within_plate: bool = True,
    min_replicates: int = 2,
    fdr_threshold: float = 0.05,
    lfc_threshold: float = 0.5,
    n_jobs: int = 1,
    normalize: str | None = None,
    normalize_kwargs: dict | None = None,
    filter_samples_kwargs: dict | None = None,
    filter_genes_kwargs: dict | None = None,
    enrichment: bool = True,
    libraries: list[str] | tuple[str, ...] = ("hallmark", "go_bp", "reactome", "kegg"),
    gene_set_paths: dict | None = None,
    enrichment_mode: list[str] | tuple[str, ...] = ("gsea", "ora"),
    n_perm: int = 1000,
    min_size: int = 15,
    max_size: int = 500,
    species: str = "Human",
    background: str = "all_tested",
    seed: int = 42,
    cache_dir: str | None = None,
    allow_network: bool = False,
    figures: list[str] | tuple[str, ...] = DEFAULT_FIGURES,
    n_top_genes: int = 60,
    boxplot_genes: list[str] | None = None,
    formats: list[str] | tuple[str, ...] = ("png", "pdf"),
    inplace: bool = True,
) -> dict:
    """
    End-to-end comparison workflow on a DrugSeqData object (or plain AnnData).

    1. Runs DE for every contrast with each requested method
       (``limma_voom``, ``edgeR``, ``deseq``).
    2. Optionally runs GSEA + directional ORA for Hallmark / GO BP /
       Reactome / KEGG.
    3. Generates volcano, heatmap, DEG-count bar, box/violin and
       enrichment figures.
    4. When ``output_dir`` is given, writes DE/enrichment tables, figures,
       a manifest and the run configuration.

    Returns a dict with keys ``summary``, ``manifest``, ``figures``,
    ``files``, ``config`` (and ``dsd`` when ``inplace=False``).
    """
    source_dsd = dsd if isinstance(dsd, DrugSeqData) else None
    if not isinstance(dsd, DrugSeqData):
        dsd = DrugSeqData(dsd if isinstance(dsd, ad.AnnData) else
                          ad.AnnData(dsd))
    if not inplace:
        dsd = DrugSeqData(dsd.adata.copy())

    # Optional preprocessing is deliberately opt-in.  Raw counts remain in
    # ``layers['counts']``; normalization only replaces ``adata.X`` and
    # filtering returns a subset object.  For an in-place wrapper workflow,
    # propagate a filtered subset back to the caller's DrugSeqData instance.
    if filter_samples_kwargs is not None:
        qc_columns = {
            "total_umi", "n_genes_det", "pct_mito", "pct_ribo",
            "hk_mean_log", "hk_cv", "outlier_score",
        }
        if not qc_columns.issubset(dsd.obs.columns):
            compute_qc_metrics(dsd, inplace=True)
        dsd = filter_samples(dsd, **dict(filter_samples_kwargs))
    if filter_genes_kwargs is not None:
        dsd = filter_genes(dsd, **dict(filter_genes_kwargs))
    if normalize is not None:
        norm_kwargs = dict(normalize_kwargs or {})
        # Workflow ownership of the working object is explicit; prevent a
        # caller-provided ``inplace`` value from colliding with the argument
        # used to preserve raw counts and propagate subsets.
        norm_kwargs.pop("inplace", None)
        normalize_counts(
            dsd, method=normalize,
            inplace=True, **norm_kwargs,
        )
    if inplace and source_dsd is not None and dsd is not source_dsd:
        source_dsd._adata = dsd.adata
        dsd = source_dsd

    # A workflow invocation is a fresh analysis configuration.  Remove
    # enrichment artifacts from an earlier run so disabling enrichment (or
    # changing contrasts/libraries) cannot leave report figures or tables
    # backed by stale results.
    dsd.adata.uns.pop("enrichment_results", None)
    dsd.adata.uns.pop("enrichment_summary", None)

    methods = ([methods] if isinstance(methods, str)
               else (list(methods) if methods is not None else ["limma_voom"]))
    if len(methods) == 1 and str(methods[0]).strip().lower() == "all":
        methods = ["limma_voom", "edgeR", "deseq"]
    batch_cols = ([batch_cols] if isinstance(batch_cols, str)
                  else (list(batch_cols) if batch_cols is not None else None))
    libraries = ([libraries] if isinstance(libraries, (str, Path, dict))
                 else (list(libraries) if libraries is not None else
                       ["hallmark", "go_bp", "reactome", "kegg"]))
    if len(libraries) == 1 and str(libraries[0]).strip().lower() == "all":
        libraries = ["hallmark", "go_bp", "reactome", "kegg"]
    enrichment_mode = ([enrichment_mode] if isinstance(enrichment_mode, str)
                       else (list(enrichment_mode) if enrichment_mode is not None
                             else ["gsea", "ora"]))
    figures = [figures] if isinstance(figures, str) else list(figures)
    formats = [formats] if isinstance(formats, str) else list(formats)

    config = {
        "comparisons": _json_safe(comparisons),
        "group_col": group_col,
        "reference": reference,
        "exclude_levels": _json_safe(exclude_levels),
        "methods": list(methods),
        "normalize": normalize,
        "normalize_kwargs": dict(normalize_kwargs or {}),
        "filter_samples_kwargs": dict(filter_samples_kwargs or {}),
        "filter_genes_kwargs": dict(filter_genes_kwargs or {}),
        "batch_cols": batch_cols,
        "fdr_threshold": fdr_threshold,
        "lfc_threshold": lfc_threshold,
        "enrichment": enrichment,
        "libraries": _json_safe(libraries),
        "gene_set_paths": {
            str(key): str(value) for key, value in (gene_set_paths or {}).items()
        },
        "enrichment_mode": _json_safe(enrichment_mode),
        "formats": _json_safe(formats),
        "within_plate": within_plate,
        "min_replicates": min_replicates,
        "n_jobs": n_jobs,
        "n_perm": n_perm,
        "min_size": min_size,
        "max_size": max_size,
        "species": species,
        "background": background,
        "seed": seed,
        "cache_dir": cache_dir,
        "allow_network": allow_network,
        "n_top_genes": n_top_genes,
        "boxplot_genes": ([boxplot_genes] if isinstance(boxplot_genes, str)
                          else list(boxplot_genes or [])),
        "backend_versions": _backend_versions(),
    }
    dsd.adata.uns["comparison_config"] = config

    manifest_rows = []

    # 1. Differential expression
    dsd.adata.uns["comparison_results"] = {}
    dsd.adata.uns["comparison_failures"] = {}
    dsd.adata.uns["de_results"] = {}
    try:
        compute_multi_de(
            dsd, comparisons=comparisons, group_col=group_col,
            reference=reference, exclude_levels=exclude_levels,
            methods=methods, batch_cols=batch_cols,
            within_plate=within_plate, min_replicates=min_replicates,
            fdr_threshold=fdr_threshold, lfc_threshold=lfc_threshold,
            n_jobs=n_jobs, inplace=True,
        )
        de_failures = dsd.adata.uns.get("comparison_failures", {})
        if de_failures:
            note = "; ".join(
                f"{key}: {message}" for key, message in de_failures.items()
            )
            has_results = bool(dsd.adata.uns.get("comparison_results"))
            manifest_rows.append({
                "type": "de", "name": "all",
                "status": "partial" if has_results else "failed",
                "note": note,
            })
        else:
            manifest_rows.append({"type": "de", "name": "all",
                                  "status": "done", "note": ""})
    except Exception as e:
        warnings.warn(f"DE step failed: {e}")
        manifest_rows.append({"type": "de", "name": "all",
                              "status": "failed", "note": str(e)})

    summary = summarise_comparison(dsd, fdr_threshold, lfc_threshold) \
        if (dsd.adata.uns.get("comparison_results") or
            dsd.adata.uns.get("de_results")) else pd.DataFrame()

    # 2. Enrichment (optional)
    if enrichment and not summary.empty:
        try:
            run_enrichment(
                dsd, libraries=libraries, gene_set_paths=gene_set_paths,
                mode=enrichment_mode, fdr_threshold=fdr_threshold,
                lfc_threshold=lfc_threshold, n_perm=n_perm,
                min_size=min_size, max_size=max_size, species=species,
                background=background, seed=seed, cache_dir=cache_dir,
                allow_network=allow_network, inplace=True,
            )
            manifest_rows.append({"type": "enrichment", "name": "all",
                                  "status": "done", "note": ""})
        except Exception as e:
            warnings.warn(f"Enrichment step failed: {e}")
            manifest_rows.append({"type": "enrichment", "name": "all",
                                  "status": "failed", "note": str(e)})

    # 3. Figures
    figure_objs: dict[str, plt.Figure] = {}
    tables = dsd.adata.uns.get("comparison_results") or \
        dsd.adata.uns.get("de_results", {})

    def _record(kind, name, fn, stem=None):
        # Report artifacts follow the reference-script convention
        # ``<contrast>_<figure>``.  ``name`` remains the manifest/object label
        # when callers do not need a custom stem.
        stem = stem or f"{name}_{kind}"
        try:
            figure_objs[stem] = fn()
            manifest_rows.append({"type": "figure", "name": stem,
                                  "status": "done", "note": ""})
        except Exception as e:
            manifest_rows.append({"type": "figure", "name": stem,
                                  "status": "skipped", "note": str(e)})

    if tables:
        if "volcano" in figures:
            for key in tables:
                _record("volcano", _sanitize_stem(key),
                        lambda k=key: plot_comparison_volcano(
                            dsd, contrast=k, fdr_threshold=fdr_threshold,
                            lfc_threshold=lfc_threshold),
                        stem=f"{_sanitize_stem(key)}_volcano")
        if "heatmap" in figures:
            _record("heatmap", "deg_logfc",
                    lambda: plot_comparison_heatmap(
                        dsd, n_top_genes=n_top_genes,
                        fdr_threshold=fdr_threshold,
                        lfc_threshold=lfc_threshold),
                    stem="de_logfc_heatmap")
        if "bar" in figures:
            _record("bar", "deg_counts",
                    lambda: plot_deg_counts(
                        dsd, fdr_threshold=fdr_threshold,
                        lfc_threshold=lfc_threshold),
                    stem="deg_counts")
        if "box" in figures or "violin" in figures:
            genes = ([boxplot_genes] if isinstance(boxplot_genes, str)
                     else list(boxplot_genes)) if boxplot_genes else \
                _default_boxplot_genes(dsd)
            if not genes:
                manifest_rows.append({
                    "type": "figure", "name": "box_violin",
                    "status": "skipped",
                    "note": "no significant genes found"})
            else:
                if "box" in figures:
                    _record("box", "gene_expression",
                            lambda: plot_gene_boxplot(
                                dsd, genes, group_col=group_col),
                            stem="gene_expression_boxplots")
                    for key in tables:
                        _record(
                            "box", f"{_sanitize_stem(key)}_boxplot",
                            lambda k=key: plot_gene_boxplot(
                                dsd, genes, group_col=group_col,
                                contrast=k),
                            stem=f"{_sanitize_stem(key)}_boxplot",
                        )
                if "violin" in figures:
                    _record("violin", "gene_expression",
                            lambda: plot_gene_violin(
                                dsd, genes, group_col=group_col),
                            stem="gene_expression_violins")
                    for key in tables:
                        _record(
                            "violin", f"{_sanitize_stem(key)}_violin",
                            lambda k=key: plot_gene_violin(
                                dsd, genes, group_col=group_col,
                                contrast=k),
                            stem=f"{_sanitize_stem(key)}_violin",
                        )

    if "enrichment" in figures and dsd.adata.uns.get("enrichment_results"):
        enr = dsd.adata.uns["enrichment_results"]
        for contrast in enr:
            contrast_stem = _sanitize_stem(contrast)
            if "ora" in enrichment_mode:
                for direc in ("up", "down"):
                    _record(
                        "enrichment", f"{contrast_stem}_{direc}_dotplot",
                        lambda c=contrast, d=direc: plot_enrichment_dotplot(
                            dsd, contrast=c, direction=d, mode="ora",
                            n_terms=20),
                        stem=f"{contrast_stem}_{direc}_enrichment_dotplot",
                    )
                for lib in enr[contrast]:
                    for direc in ("up", "down"):
                        _record(
                            "enrichment", f"{contrast_stem}_{lib}_{direc}",
                            lambda c=contrast, l=lib, d=direc:
                                plot_enrichment_barplot(
                                    dsd, contrast=c, library=l,
                                    direction=d, mode="ora", n_terms=20),
                            stem=(f"{contrast_stem}_{direc}_"
                                  f"{_sanitize_stem(lib)}_pathways"),
                        )
            elif "gsea" in enrichment_mode:
                _record(
                    "enrichment", f"{contrast_stem}_dotplot",
                    lambda c=contrast: plot_enrichment_dotplot(
                        dsd, contrast=c, direction="up", mode="gsea",
                        n_terms=20),
                    stem=f"{contrast_stem}_enrichment_dotplot",
                )
                for lib in enr[contrast]:
                    _record(
                        "enrichment", f"{contrast_stem}_{lib}_gsea",
                        lambda c=contrast, l=lib: plot_enrichment_barplot(
                            dsd, contrast=c, library=l, direction="up",
                            mode="gsea", n_terms=20),
                        stem=(f"{contrast_stem}_gsea_"
                              f"{_sanitize_stem(lib)}_pathways"),
                    )

    manifest = pd.DataFrame(manifest_rows)

    # 4. File export
    files: list[str] = []
    if output_dir is not None:
        out = Path(output_dir)
        de_dir = out / "results" / "de"
        enr_dir = out / "results" / "enrichment"
        fig_dir = out / "results" / "figures"
        man_dir = out / "results" / "manifests"
        for d in (de_dir, enr_dir, fig_dir, man_dir):
            d.mkdir(parents=True, exist_ok=True)

        for key, df in tables.items():
            key_stem = _sanitize_stem(key)
            method = ""
            if "method" in df.columns and not df.empty:
                method = _sanitize_stem(df["method"].iloc[0])
            # The method-qualified name is the canonical report artifact.
            # Keep the historical <contrast>.tsv alias for one-method runs so
            # existing consumers do not break.
            primary_stem = key_stem
            if method and not key_stem.endswith(f"_{method}"):
                primary_stem = f"{key_stem}_{method}"
            path = de_dir / f"{primary_stem}.tsv"
            df.to_csv(path, sep="\t", index=False)
            files.append(str(path))
            if "::" not in str(key) and primary_stem != key_stem:
                legacy = de_dir / f"{key_stem}.tsv"
                df.to_csv(legacy, sep="\t", index=False)
                files.append(str(legacy))

        # Combined DEG summary is useful for report builders and mirrors the
        # reference scripts' aggregate ``deg_summary.tsv`` artifact.
        summary_path = de_dir / "comparison_summary.tsv"
        summary.to_csv(summary_path, sep="\t", index=False)
        files.append(str(summary_path))

        enr = dsd.adata.uns.get("enrichment_results", {})
        for contrast, libs in enr.items():
            for lib, slots in libs.items():
                for slot, df in slots.items():
                    if df is None:
                        continue
                    mode_name = "gsea" if slot == "gsea" else "ora"
                    direction = "" if slot == "gsea" else slot.split("_")[-1]
                    stem = "_".join(filter(None, [
                        _sanitize_stem(contrast), _sanitize_stem(lib),
                        mode_name, direction]))
                    path = enr_dir / f"{stem}.tsv"
                    df.to_csv(path, sep="\t", index=False)
                    files.append(str(path))
        enrichment_table = dsd.adata.uns.get("enrichment_summary")
        if isinstance(enrichment_table, pd.DataFrame):
            pathway_path = enr_dir / "pathway_summary.tsv"
            enrichment_table.to_csv(pathway_path, sep="\t", index=False)
            files.append(str(pathway_path))

        for name, fig in figure_objs.items():
            for fmt in formats:
                path = fig_dir / f"{name}.{fmt}"
                fig.savefig(path, bbox_inches="tight",
                            dpi=180 if fmt == "png" else 300)
                files.append(str(path))
        for fig in figure_objs.values():
            plt.close(fig)

        manifest.to_csv(man_dir / "comparison_manifest.tsv",
                        sep="\t", index=False)
        files.append(str(man_dir / "comparison_manifest.tsv"))
        with open(man_dir / "comparison_config.json", "w") as fh:
            json.dump(config, fh, indent=2, default=str)
        files.append(str(man_dir / "comparison_config.json"))

    result = {
        "summary": summary,
        "manifest": manifest,
        "figures": figure_objs,
        "files": files,
        "config": config,
    }
    if not inplace or source_dsd is None:
        result["dsd"] = dsd
    return result
