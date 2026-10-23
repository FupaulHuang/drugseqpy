"""Scientific invariants of the combined-library QC figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import drugseqpy as ds
from drugseqpy.combined import READ_METRICS


@pytest.fixture
def combined_input():
    # b changes dominant species in sample_2; c is eligible only there. Baseline
    # labels must still come from sample_1, and c must be excluded from both groups.
    x = np.array([
        [8, 0, 0, 2, 0, 0, 0], [0, 0, 0, 8, 2, 0, 0], [1, 0, 0, 0, 0, 0, 0],
        [2, 2, 0, 6, 0, 0, 0], [8, 0, 0, 0, 2, 0, 0], [900, 0, 0, 0, 0, 0, 0],
    ])
    obs = pd.DataFrame({
        "plate_id": ["sample_1"] * 3 + ["sample_2"] * 3, "barcode": ["a", "b", "c"] * 2,
        "well_id": ["A1", "A2", "A3"] * 2, "reads_per_well": [100, 90, 1, 200, 180, 1000],
        "filtered": [True, True, False, True, True, True],
        "expected_cells": ["231", "3T3", "water"] * 2,
    }, index=[f"w{i}" for i in range(6)])
    var = pd.DataFrame({"gene_symbol": ["ACTB", "MT-A", "RPL1", "Actb", "Rpl1", "MRPL1", "ACTB"]},
                       index=["ENSG1", "ENSG2", "ENSG3", "ENSMUSG1", "ENSMUSG2", "ENSG4", "ENSG5"])
    a = ad.AnnData(x.astype(float) / 2, obs=obs, var=var)
    a.layers["counts"] = sparse.csr_matrix(x)
    kwargs = dict(baseline_group="sample_1", filtered_col="filtered",
                  housekeeping_sets={"human": ["ACTB", "RPL1"], "mouse": ["Actb", "Rpl1"]},
                  marker_sets={"Human": ["ACTB", "RPL1"], "Mouse": ["Actb", "Rpl1"]})
    return a, kwargs


def test_shared_selection_fixed_baseline_and_denominators(combined_input):
    a, kwargs = combined_input
    original = a.obs.copy(deep=True)
    t = ds.compute_combined_qc(a, **kwargs)
    w = t["wells"].set_index(["group", "barcode"])
    assert len(w) == 4
    assert set(t["excluded_wells"]["barcode"]) == {"c"}
    assert w.loc[("sample_2", "b"), "baseline_species"] == "mouse"
    assert w.loc[("sample_2", "b"), "mouse_fraction"] == .2
    assert w.loc[("sample_2", "a"), "pct_mito"] == 20
    assert w.loc[("sample_2", "b"), "pct_ribo"] == 20
    assert w.loc[("sample_1", "a"), "human_housekeeping_mean_umi"] == 4  # includes zero RPL1
    assert len(t["composition_values"]) == 12  # 4 + 4 + 2 + 2
    assert set(t["gene_sets"].query("gene_set == 'ribosomal'")["gene_symbol"]) == {"RPL1", "Rpl1"}
    markers = t["markers"].query("group == 'sample_1' and gene == 'RPL1'")
    assert len(markers) == 2 and markers["mean_umi"].eq(0).all()
    assert t["gene_totals"].query("group == 'sample_2'")["total_umi"].sum() == 20
    pd.testing.assert_frame_equal(original, a.obs)
    np.testing.assert_array_equal(a.X, a.layers["counts"].toarray() / 2)


def test_sparse_dense_wrapper_and_duplicate_symbols(combined_input):
    a, kwargs = combined_input
    # Duplicate symbols remain independent feature rows in totals/detection;
    # expression panels deliberately use the first matching feature.
    dense = a.layers["counts"].toarray()
    dense[0, -1] = 10
    a.layers["counts"] = dense
    t = ds.compute_combined_qc(a, **kwargs)
    assert len(t["gene_totals"].query("group == 'sample_1' and gene_symbol == 'ACTB'")) == 2
    assert t["markers"].query("group == 'sample_1' and gene == 'ACTB' and baseline_species == 'human'")["mean_umi"].iloc[0] == 8
    a.layers["counts"] = sparse.csc_matrix(dense)
    other = ds.compute_combined_qc(ds.DrugSeqData(a), **kwargs)
    for key in t:
        pd.testing.assert_frame_equal(t[key], other[key])


@pytest.mark.parametrize("problem,match", [
    ("duplicate", "unique"), ("no_shared", "No shared"), ("string_filter", "boolean"),
    ("zero_baseline", "zero-count"), ("missing_layer", "Raw count layer"),
    ("fractional", "integer"), ("missing_hk", "housekeeping"),
])
def test_invalid_inputs(combined_input, problem, match):
    a, kwargs = combined_input
    if problem == "duplicate":
        a.obs.loc["w1", "barcode"] = "a"
    elif problem == "no_shared":
        a.obs.loc[a.obs.plate_id == "sample_1", "filtered"] = False
    elif problem == "string_filter":
        a.obs["filtered"] = "False"
    elif problem == "zero_baseline":
        a.layers["counts"] = a.layers["counts"].toarray()
        a.layers["counts"][0] = 0
    elif problem == "missing_layer":
        del a.layers["counts"]
    elif problem == "fractional":
        a.layers["counts"] = a.X
        a.layers["counts"][0, 0] = .5
    elif problem == "missing_hk":
        kwargs["housekeeping_sets"]["human"] = ["absent"]
    with pytest.raises(ValueError, match=match):
        ds.compute_combined_qc(a, **kwargs)


def test_star_summary_and_doughnut(tmp_path):
    path = tmp_path / "Summary.csv"
    path.write_text("\n".join(f"{key},{v}" for key, v in zip(READ_METRICS.values(), [0, .2, .5, .75, 1])))
    metrics = ds.read_star_summary_metrics({"sample_1": path})
    np.testing.assert_allclose(metrics["percent"], [0, 20, 50, 75, 100])
    fig = ds.plot_combined_read_doughnut(metrics)
    # Third foreground ring is 50%, regardless of the largest metric.
    assert fig.axes[0].patches[4].theta2 - fig.axes[0].patches[4].theta1 == pytest.approx(180)
    plt.close(fig)
    path.write_text("\n".join(f"{key},95" for key in READ_METRICS.values()))
    with pytest.raises(ValueError, match="fraction"):
        ds.read_star_summary_metrics({"sample_1": path})


def test_full_report_and_plate_scales(combined_input, tmp_path):
    a, kwargs = combined_input
    # h5ad round trips store repeated metadata strings as categories.
    a.obs["expected_cells"] = a.obs["expected_cells"].astype("category")
    metrics = pd.DataFrame([
        {"group": g, "metric": m, "percent": 75} for g in ["sample_1", "sample_2"] for m in READ_METRICS
    ])
    result = ds.generate_combined_qc_report(a, tmp_path, read_metrics=metrics, formats=("pdf",),
                                           nrow=1, ncol=3, **kwargs)
    from pathlib import Path
    pdfs = [Path(p) for p in result["files"] if p.endswith(".pdf")]
    assert len(pdfs) == 20 and all(p.stat().st_size > 1000 for p in pdfs)
    assert not result["skipped"]
    layout = result["tables"]["plate_map_Read_Count"]
    assert layout["is_missing_well"].sum() == 2
    assert layout.groupby("plate_id")["color_vmax"].first().to_dict() == pytest.approx(
        {"sample_1": np.log1p(100), "sample_2": np.log1p(200)}
    )
    identity = result["tables"]["plate_map_Human_Mouse"]
    assert identity["value"].eq("Excluded").sum() == 2
    assert not plt.get_fignums()


def test_plate_legend_option_preserves_default():
    obs = pd.DataFrame({"plate_id": ["a", "a", "b", "b"], "well_id": ["A1", "A2"] * 2,
                        "value": [1, 2, 10, 20]})
    fig, shared = ds.plot_plate_layout(obs, "value", continuous=True, log_values=False, nrow=1, ncol=2)
    assert shared["color_vmin"].eq(1).all() and shared["color_vmax"].eq(20).all()
    plt.close(fig)
    fig, separate = ds.plot_plate_layout(obs, "value", continuous=True, log_values=False,
                                         nrow=1, ncol=2, legend_per_plate=True)
    assert len(fig.axes) == 4
    assert separate.groupby("plate_id")["color_vmin"].first().to_dict() == {"a": 1, "b": 10}
    plt.close(fig)
