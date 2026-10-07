"""Focused control-audit and cross-plate reproducibility tests."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from drugseqpy import (
    audit_control_outliers,
    compute_cross_plate_reproducibility,
    compute_qc_metrics,
    compute_within_plate_repeatability,
    create_drugseq_object,
    normalize_counts,
    plot_control_outlier_audit,
    plot_cross_plate_reproducibility,
    plot_plate_logfc_concordance,
    plot_single_plate_de_heatmap,
    plot_single_plate_de_layout,
    plot_single_plate_de_volcano,
    plot_within_plate_correlation_heatmap,
    plot_within_plate_replicate_scatter,
    plot_within_plate_repeatability,
    summarise_within_plate_repeatability,
)
from drugseqpy.utils import make_dummy_screen


def _screen():
    counts, obs = make_dummy_screen(
        n_genes=80,
        n_plates=2,
        n_compounds=2,
        n_reps=4,
        n_dmso_per_plate=5,
        seed=51,
    )
    dsd = create_drugseq_object(counts, obs)
    compute_qc_metrics(dsd, inplace=True)
    normalize_counts(dsd, method="limma_voom", inplace=True)
    return dsd


def test_control_audit_table_and_plot_are_independent():
    dsd = _screen()
    threshold = float(dsd.obs.loc[
        dsd.obs["compound"].eq("DMSO"), "outlier_score"
    ].median())
    audit = audit_control_outliers(dsd, threshold=threshold)
    assert {"sample_id", "excluded", "exclusion_rule"}.issubset(audit.columns)
    assert audit["excluded"].any()
    figure = plot_control_outlier_audit(audit)
    assert hasattr(figure, "savefig")
    plt.close(figure)


def test_repeatability_compute_and_plot_are_independent():
    dsd = _screen()
    table = compute_within_plate_repeatability(dsd, method="spearman")
    assert {
        "plate_id", "compound", "sample_a", "sample_b",
        "method", "correlation",
    }.issubset(table.columns)
    assert table["method"].eq("spearman").all()
    summary = summarise_within_plate_repeatability(table)
    assert len(summary) == 6
    assert {
        "plate_id", "compound", "median_correlation", "minimum_correlation",
    }.issubset(summary.columns)
    figure = plot_within_plate_repeatability(table)
    assert hasattr(figure, "savefig")
    plt.close(figure)


def test_repeatability_supports_pearson_scatter_and_heatmap():
    dsd = _screen()
    table = compute_within_plate_repeatability(dsd)
    assert table["method"].eq("pearson").all()
    assert table["expression_source"].eq("log10_raw_umi").all()
    assert table.groupby(["plate_id", "compound"])["n_replicates"].nunique().eq(1).all()

    scatter = plot_within_plate_replicate_scatter(
        dsd,
        plate="Plate01",
        compound="Cmpd01",
        max_pairs=3,
    )
    heatmap = plot_within_plate_correlation_heatmap(
        dsd,
        plate="Plate01",
        compound="Cmpd01",
    )
    normalized = plot_within_plate_replicate_scatter(
        dsd,
        plate="Plate01",
        compound="Cmpd01",
        use_norm=True,
        max_pairs=1,
    )
    assert hasattr(scatter, "savefig")
    assert hasattr(heatmap, "savefig")
    assert hasattr(normalized, "savefig")
    assert len(scatter.axes) == 4
    assert len(heatmap.axes) == 2
    assert "Pearson r" in scatter.axes[0].get_title()
    assert "log10(raw UMI + 1)" in scatter.axes[0].get_xlabel()
    assert "Normalized expression" in normalized.axes[0].get_xlabel()
    plt.close("all")


def test_cross_plate_compute_and_each_plot_are_independent():
    dsd = _screen()
    result = compute_cross_plate_reproducibility(dsd)
    summary = result["summary"]
    assert len(summary) == 2
    assert {"spearman_logfc", "significant_jaccard"}.issubset(summary.columns)
    gene_table = next(iter(result["gene_results"].values()))
    assert any(column.startswith("log10_mean_umi_") for column in gene_table)
    assert any(
        column.startswith("mean_normalized_expression_")
        for column in gene_table
    )

    overview = plot_cross_plate_reproducibility(summary)
    concordance = plot_plate_logfc_concordance(result, max_panels=2)
    raw_expression = plot_plate_logfc_concordance(
        result, max_panels=2, value_type="log10_umi"
    )
    normalized = plot_plate_logfc_concordance(
        result, max_panels=2, value_type="normalized"
    )
    volcano = plot_single_plate_de_volcano(
        result, plate="Plate01", compound="Cmpd01"
    )
    layout = plot_single_plate_de_layout(
        dsd, result, plate="Plate01", compound="Cmpd01", n_top_genes=10
    )
    heatmap = plot_single_plate_de_heatmap(
        dsd, result, plate="Plate01", compound="Cmpd01", n_top_genes=10
    )
    assert hasattr(overview, "savefig")
    assert hasattr(concordance, "savefig")
    assert hasattr(raw_expression, "savefig")
    assert hasattr(normalized, "savefig")
    assert hasattr(volcano, "savefig")
    assert hasattr(layout, "savefig")
    assert hasattr(heatmap, "savefig")
    assert "Pearson" in overview.axes[0].get_xlabel()
    assert "log10(mean UMI + 1)" in raw_expression.axes[0].get_xlabel()
    assert "Mean normalized expression" in normalized.axes[0].get_xlabel()
    assert {text.get_text() for text in concordance.axes[0].get_legend().texts} <= {
        "Neither", "Plate01 only", "Plate02 only", "Shared",
    }
    plt.close("all")
