"""Tidy plate-coordinate and layout helpers."""
from __future__ import annotations

import re
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import is_color_like
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as path_effects
import seaborn as sns


def _wrap_label(text: str, width: int) -> str:
    """Wrap a well label to ``width`` chars so it fits inside its marker."""
    if width <= 0 or len(text) <= width:
        return text
    wrapped = textwrap.wrap(text, width=width)
    return "\n".join(wrapped) if wrapped else text


def _parse_well_ids(wells: pd.Series, n_rows: int = 8, n_cols: int = 12) -> tuple[pd.Series, pd.Series]:
    rows, cols = [], []
    for well in wells.astype(str):
        numeric = re.search(r"(?:^|[-_])(?:Well[-_]?dT[-_]?)?0*(\d+)$", well, flags=re.IGNORECASE)
        alpha = re.search(r"^([A-Za-z]+)[-_]?0*(\d+)$", well)
        if re.search(r"Well[-_]?dT", well, flags=re.IGNORECASE) and numeric:
            number = int(numeric.group(1)) - 1
            rows.append(number % n_rows + 1)
            cols.append(number // n_rows + 1)
        elif alpha:
            row = 0
            for char in alpha.group(1).upper():
                row = row * 26 + ord(char) - 64
            rows.append(row)
            cols.append(int(alpha.group(2)))
        elif numeric:
            number = int(numeric.group(1)) - 1
            rows.append(number % n_rows + 1)
            cols.append(number // n_rows + 1)
        else:
            rows.append(np.nan)
            cols.append(np.nan)
    return pd.Series(rows, index=wells.index, dtype="float"), pd.Series(cols, index=wells.index, dtype="float")


def _coordinate_match(a: pd.Series, b: pd.Series) -> float:
    mask = a.notna() & b.notna()
    if not mask.any():
        return 0.0
    return float((a[mask].round().astype(int) == b[mask].round().astype(int)).mean())


def _well_id_from_position(row: int, col: int, n_rows: int = 8) -> str:
    return f"Well-dT-{(col - 1) * n_rows + row}"


def _row_label(row: int) -> str:
    """Return spreadsheet-style row labels (A..Z, AA..AZ, ...)."""
    label = ""
    while row > 0:
        row, remainder = divmod(row - 1, 26)
        label = chr(65 + remainder) + label
    return label


def _validate_plate_shape(nrow: int, ncol: int) -> tuple[int, int]:
    if not isinstance(nrow, (int, np.integer)) or nrow < 1:
        raise ValueError("nrow must be a positive integer")
    if not isinstance(ncol, (int, np.integer)) or ncol < 1:
        raise ValueError("ncol must be a positive integer")
    return int(nrow), int(ncol)


def _coordinates(
    obs: pd.DataFrame,
    *,
    nrow: int = 8,
    ncol: int = 12,
) -> tuple[pd.Series, pd.Series]:
    well_rows, well_cols = _parse_well_ids(
        obs.get("well_id", pd.Series(index=obs.index, dtype=str)).astype(str),
        n_rows=nrow,
        n_cols=ncol,
    )
    well_mask = well_rows.notna() & well_cols.notna()
    if well_mask.any():
        return well_rows, well_cols
    if {"x", "y"}.issubset(obs.columns):
        x = pd.to_numeric(obs["x"], errors="coerce")
        y = pd.to_numeric(obs["y"], errors="coerce")
        xy_score = _coordinate_match(x, well_rows) + _coordinate_match(y, well_cols)
        yx_score = _coordinate_match(y, well_rows) + _coordinate_match(x, well_cols)
        if max(xy_score, yx_score) >= 1.0:
            return (x, y) if xy_score >= yx_score else (y, x)
        # If the metadata has numeric coordinates but no parseable well_id,
        # treat the dimension with fewer unique values as plate rows.
        if x.nunique(dropna=True) <= y.nunique(dropna=True):
            return x, y
        return y, x
    return well_rows, well_cols


def complete_plate_layout(
    layout: pd.DataFrame,
    *,
    nrow: int = 8,
    ncol: int = 12,
) -> pd.DataFrame:
    """Complete every plate to an exact ``nrow`` by ``ncol`` grid.

    Placeholder wells are marked by ``is_missing_well=True`` and are drawn as
    white circles with black borders.  Numbered ``Well-dT`` identifiers are
    column-major: on a 96-well plate, wells 1--8 form the first column.
    """
    nrow, ncol = _validate_plate_shape(nrow, ncol)
    if "plate_id" not in layout.columns:
        return layout
    plated = layout.dropna(subset=["row", "column"])
    if plated.empty:
        return layout
    rows = []
    for plate, sub in layout.groupby(layout["plate_id"].astype(str), sort=False):
        present = {
            (int(row), int(col))
            for row, col in sub[["row", "column"]].dropna().itertuples(index=False)
        }
        for col in range(1, ncol + 1):
            for row in range(1, nrow + 1):
                if (row, col) not in present:
                    well_id = _well_id_from_position(row, col, n_rows=nrow)
                    rows.append({
                        "plate_id": plate,
                        "well_id": well_id,
                        "row": float(row),
                        "column": float(col),
                        "value": np.nan,
                        "row_label": _row_label(row),
                        "is_missing_well": True,
                    })
    out = layout.copy()
    out["is_missing_well"] = False
    if rows:
        out = pd.concat([out, pd.DataFrame(rows)], ignore_index=True, sort=False)
    return out


def build_plate_layout(
    obs: pd.DataFrame,
    value_col: str | None = None,
    *,
    nrow: int = 8,
    ncol: int = 12,
) -> pd.DataFrame:
    """Build a tidy plate table for an exact plate shape."""
    nrow, ncol = _validate_plate_shape(nrow, ncol)
    row, col = _coordinates(obs, nrow=nrow, ncol=ncol)
    invalid = row.notna() & col.notna() & (
        (row < 1) | (row > nrow) | (col < 1) | (col > ncol)
    )
    if invalid.any():
        wells = obs.get("well_id", pd.Series(obs.index, index=obs.index))
        examples = ", ".join(wells.loc[invalid].astype(str).head(5))
        raise ValueError(
            f"Well coordinates exceed the requested {nrow}x{ncol} plate "
            f"shape (examples: {examples}). Set nrow and ncol explicitly."
        )
    out = obs.copy()
    out["row"], out["column"] = row, col
    out["value"] = out[value_col] if value_col and value_col in out else np.nan
    out["well_id"] = out.get("well_id", pd.Series(out.index, index=out.index)).astype(str)
    out["row_label"] = out["row"].map(
        lambda x: _row_label(int(x)) if pd.notna(x) else ""
    )
    out["is_missing_well"] = False
    if "sample_id" in out.columns:
        return out.reset_index(names="metadata_index")
    return out.reset_index(names="sample_id")


def plot_plate_layout(
    obs: pd.DataFrame,
    value_col: str | None = None,
    *,
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
    figsize: tuple[float, float] | None = None,
    well_size: float = 0.76,
    well_border: bool = True,
    well_border_color: str = "#111827",
    well_border_width: float = 0.8,
    palette: str | list[str] | tuple[str, ...] | dict[str, str] | None = None,
    cmap: str = "RdYlBu_r",
    label_color: str = "#111827",
    label_size: float | None = None,
    label_style: str = "normal",
    label_weight: str | int = "normal",
):
    """Plot one plate-layout view, faceted when multiple plates are present.

    Exactly one of ``value_col`` or ``well_color`` must be supplied.
    ``value_col`` is categorical unless ``continuous=True``. ``well_color``
    is a literal Matplotlib color and intentionally has no legend. Supplying
    ``label_col`` writes that metadata field inside each occupied well.
    Continuous values use ``log1p`` by default; set ``log_values=False`` for
    the original scale. ``plate_id`` names the metadata column that identifies
    plates. ``split_by`` optionally creates separate physical-layout panels
    for each plate-by-split combination, while ``nrow_plate`` controls how
    many panels appear in one figure row.
    """
    nrow, ncol = _validate_plate_shape(nrow, ncol)
    if (value_col is None) == (well_color is None):
        raise ValueError("Exactly one of value_col or well_color must be supplied")
    if value_col is not None and value_col not in obs.columns:
        raise KeyError(f"value_col={value_col!r} is not present in obs")
    if well_color is not None and not is_color_like(well_color):
        raise ValueError(f"well_color={well_color!r} is not a valid color")
    if plate_id not in obs.columns:
        raise KeyError(f"plate_id={plate_id!r} is not present in obs")
    if split_by is not None and split_by not in obs.columns:
        raise KeyError(f"split_by={split_by!r} is not present in obs")
    if label_col is not None and label_col not in obs.columns:
        raise KeyError(f"label_col={label_col!r} is not present in obs")
    if nrow_plate is not None and (
        not isinstance(nrow_plate, (int, np.integer)) or nrow_plate < 1
    ):
        raise ValueError("nrow_plate must be None or a positive integer")
    if not 0 < float(well_size) <= 1.25:
        raise ValueError("well_size must be greater than 0 and at most 1.25")
    if well_border and not is_color_like(well_border_color):
        raise ValueError(
            f"well_border_color={well_border_color!r} is not a valid color"
        )
    if well_border_width < 0:
        raise ValueError("well_border_width must be non-negative")
    if label_size is not None and label_size <= 0:
        raise ValueError("label_size must be positive")

    plot_obs = obs.copy()
    if plate_id != "plate_id":
        plot_obs["plate_id"] = plot_obs[plate_id].astype(str)
    if split_by is not None:
        plot_obs["physical_plate_id"] = plot_obs["plate_id"].astype(str)
        plot_obs["plate_id"] = (
            plot_obs["physical_plate_id"]
            + " · " + split_by + ": "
            + plot_obs[split_by].fillna("unknown").astype(str)
        )
    layout = complete_plate_layout(
        build_plate_layout(plot_obs, value_col=value_col, nrow=nrow, ncol=ncol),
        nrow=nrow,
        ncol=ncol,
    )
    plates = list(pd.unique(layout["plate_id"].astype(str))) if "plate_id" in layout else ["plate"]
    has_labels = label_col is not None
    cell_target_in = 0.50 if not has_labels else 0.62
    panel_width = max(ncol * cell_target_in + 1.55, 5.8)
    panel_height = max(nrow * cell_target_in + 1.45, 4.6)
    cell_in_w = (panel_width - 1.55) / ncol
    cell_in_h = (panel_height - 1.45) / nrow
    cell_in = min(cell_in_w, cell_in_h)
    if nrow_plate is None:
        panel_cols = min(2, max(1, len(plates)))
        panel_rows = int(np.ceil(len(plates) / panel_cols))
    else:
        panel_cols = min(int(nrow_plate), max(1, len(plates)))
        panel_rows = int(np.ceil(len(plates) / panel_cols))

    categorical = value_col is not None and not continuous
    if continuous:
        numeric = pd.to_numeric(layout["value"], errors="coerce")
        nonmissing_input = (
            ~layout["is_missing_well"].to_numpy(dtype=bool)
            & layout["value"].notna().to_numpy()
        )
        if nonmissing_input.any() and np.isnan(
            numeric.to_numpy()[nonmissing_input]
        ).any():
            raise TypeError(
                f"continuous=True requires numeric values in {value_col!r}"
            )
        if log_values:
            negative = numeric.dropna().lt(0)
            if negative.any():
                raise ValueError(
                    "log_values=True requires non-negative continuous values; "
                    "set log_values=False for signed metrics"
                )
            layout["plot_value"] = np.log1p(numeric.to_numpy())
        else:
            layout["plot_value"] = numeric.to_numpy()
    categories = (
        [str(x) for x in pd.unique(layout["value"].dropna().astype(str))]
        if categorical else []
    )
    legend_ncol = min(8, max(1, len(categories)))
    legend_rows = int(np.ceil(len(categories) / legend_ncol)) if categories else 0
    legend_extra = (
        0.72 + 0.28 * legend_rows
        if categorical and show_legend and categories else 0.0
    )
    if figsize is None:
        figsize = (panel_width * panel_cols, panel_height * panel_rows + legend_extra)
    fig, axes = plt.subplots(panel_rows, panel_cols, figsize=figsize, squeeze=False)
    default_palette = "colorblind" if len(categories) <= 10 else "husl"
    if categorical and isinstance(palette, dict):
        fallback = sns.color_palette(default_palette, max(1, len(categories))).as_hex()
        colors = {
            category: palette.get(category, fallback[index])
            for index, category in enumerate(categories)
        }
    elif categorical:
        selected_palette = default_palette if palette is None else palette
        colors = dict(zip(
            categories,
            sns.color_palette(selected_palette, max(1, len(categories))).as_hex(),
        ))
    else:
        colors = {}
    marker_diameter_in = float(well_size) * cell_in
    marker_size = float(np.pi * (marker_diameter_in * 36.0) ** 2)
    numeric_values = layout["plot_value"] if continuous else None
    vmin = float(np.nanmin(numeric_values)) if numeric_values is not None and numeric_values.notna().any() else None
    vmax = float(np.nanmax(numeric_values)) if numeric_values is not None and numeric_values.notna().any() else None
    scatter_for_colorbar = None
    used_axes = []
    full_col_ticks = list(range(1, ncol + 1))
    full_row_ticks = list(range(1, nrow + 1))
    full_row_labels = [_row_label(r) for r in full_row_ticks]
    edge_color = well_border_color if well_border else "none"
    edge_width = float(well_border_width) if well_border else 0.0
    for ax, plate in zip(axes.flat, plates):
        used_axes.append(ax)
        plate_body = FancyBboxPatch(
            (0.18, 0.18),
            ncol + 0.64,
            nrow + 0.64,
            boxstyle="round,pad=0.03,rounding_size=0.38",
            facecolor="#F3F4F6",
            edgecolor="#94A3B8",
            linewidth=1.2,
            zorder=0,
        )
        ax.add_patch(plate_body)
        sub = layout if plate == "plate" else layout[layout["plate_id"] == plate]
        sub = sub.dropna(subset=["row", "column"])
        missing = sub[sub.get("is_missing_well", False).astype(bool)]
        sub = sub[~sub.get("is_missing_well", False).astype(bool)]
        if len(missing):
            ax.scatter(
                missing["column"],
                missing["row"],
                facecolors="white",
                edgecolors=edge_color,
                linewidth=edge_width,
                s=marker_size,
                zorder=2,
            )
        if well_color is not None:
            sc = ax.scatter(
                sub["column"],
                sub["row"],
                facecolors=well_color,
                edgecolors=edge_color,
                s=marker_size,
                linewidth=edge_width,
                zorder=3,
            )
        elif categorical:
            point_colors = sub["value"].astype(str).map(colors)
            point_colors = point_colors.where(sub["value"].notna(), "#D1D5DB")
            sc = ax.scatter(sub["column"], sub["row"], c=point_colors, s=marker_size, edgecolor=edge_color, linewidth=edge_width, zorder=3)
        else:
            sc = ax.scatter(sub["column"], sub["row"], c=sub["plot_value"], cmap=cmap, s=marker_size, edgecolor=edge_color, linewidth=edge_width, vmin=vmin, vmax=vmax, zorder=3)
            scatter_for_colorbar = sc
        if label_col is not None:
            diameter_pt = marker_diameter_in * 72.0
            wrap_width = max(4, min(12, int(diameter_pt / 4.2)))
            fontsize = (
                float(label_size) if label_size is not None
                else max(3.4, min(7.5, diameter_pt / 7.0))
            )
            for _, row in sub.iterrows():
                ax.text(
                    row["column"],
                    row["row"],
                    _wrap_label(str(row[label_col]), wrap_width),
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    clip_on=True,
                    zorder=4,
                    color=label_color,
                    fontstyle=label_style,
                    fontweight=label_weight,
                    path_effects=[path_effects.withStroke(linewidth=.8, foreground="white")],
                )
        ax.set_xticks(full_col_ticks)
        ax.set_xticklabels([f"{x:02d}" for x in full_col_ticks], fontsize=7)
        ax.set_yticks(full_row_ticks)
        ax.set_yticklabels(full_row_labels, fontsize=7)
        ax.set_xlim(0, ncol + 1)
        ax.set_ylim(nrow + 1, 0)
        ax.set_title(str(plate), fontsize=12, pad=6)
        ax.set_xlabel("Column", fontsize=9)
        ax.set_ylabel("Row", fontsize=9)
        ax.set_aspect("equal")
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax in axes.flat[len(plates):]:
        ax.set_visible(False)
    has_colorbar = scatter_for_colorbar is not None and show_legend
    has_bottom_legend = bool(categorical and show_legend and categories)
    bottom = (
        max(.13, legend_extra / figsize[1])
        if has_bottom_legend else .07
    )
    right = .91 if has_colorbar else .98
    top = .92
    fig.subplots_adjust(left=.055, right=right, bottom=bottom, top=top, wspace=.16, hspace=.25)
    if has_colorbar:
        cax = fig.add_axes([right + .015, bottom + .07, .012, .78 - bottom])
        cbar = fig.colorbar(scatter_for_colorbar, cax=cax)
        colorbar_label = f"log1p({value_col})" if log_values else str(value_col)
        cbar.set_label(colorbar_label, fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    if has_bottom_legend:
        fontsize = 7 if len(colors) <= 12 else 5
        fig.legend([Line2D([0], [0], marker="o", color="w", markerfacecolor=c, label=k, markersize=6) for k, c in colors.items()], list(colors), title=value_col, loc="lower center", bbox_to_anchor=(.5, .015), ncol=legend_ncol, frameon=False, fontsize=fontsize, title_fontsize=8)
    return fig, layout


def adaptive_figsize(n_panels: int = 1, n_items: int = 1, base: tuple[float, float] = (5.5, 4.2)) -> tuple[float, float]:
    cols = min(4, max(1, int(np.ceil(np.sqrt(max(n_panels, 1))))))
    rows = int(np.ceil(n_panels / cols))
    return (max(base[0], cols * base[0] * (1 + min(n_items, 200) / 1000)), max(base[1], rows * base[1]))
