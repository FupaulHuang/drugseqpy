"""tests/test_comparison_plots.py — comparison and enrichment figures."""

import importlib.util

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from drugseqpy import (
    compute_multi_de,
    create_drugseq_object,
    plot_comparison_heatmap,
    plot_comparison_volcano,
    plot_deg_counts,
    plot_enrichment_barplot,
    plot_enrichment_dotplot,
    plot_gene_boxplot,
    plot_gene_violin,
    run_enrichment,
)
from drugseqpy.enrichment import _save_gmt
from drugseqpy.utils import make_dummy_screen

HAS_GSEAPY = importlib.util.find_spec("gseapy") is not None


@pytest.fixture(scope="module")
def dsd_de():
    counts, obs = make_dummy_screen(
        n_genes=100, n_plates=2, n_dmso_per_plate=6,
        n_compounds=3, n_reps=4, seed=21,
    )
    dsd = create_drugseq_object(counts, obs)
    compute_multi_de(dsd, reference="DMSO", methods=["limma_voom"],
                     within_plate=True, n_jobs=1, inplace=True)
    return dsd


@pytest.fixture(scope="module")
def dsd_enriched(dsd_de, tmp_path_factory):
    if not HAS_GSEAPY:
        pytest.skip("gseapy not installed")
    gene_sets = {
        "SET_UP": [f"Gene{i:04d}" for i in range(1, 19)],
        "SET_DOWN": [f"Gene{i:04d}" for i in range(21, 39)],
    }
    gmt = tmp_path_factory.mktemp("gmt") / "lib.gmt"
    _save_gmt(gmt, gene_sets)
    run_enrichment(dsd_de, libraries=["lib"], gene_set_paths={"lib": gmt},
                   mode=("gsea", "ora"), min_size=5, max_size=50,
                   n_perm=200, inplace=True)
    return dsd_de


class TestComparisonPlots:
    def test_volcano_returns_figure(self, dsd_de):
        fig = plot_comparison_volcano(dsd_de)
        assert hasattr(fig, "savefig")
        plt.close("all")

    def test_volcano_specific_contrast(self, dsd_de):
        fig = plot_comparison_volcano(dsd_de, contrast="Cmpd02")
        assert hasattr(fig, "savefig")
        plt.close("all")

    def test_plain_anndata_is_accepted(self, dsd_de):
        fig = plot_comparison_volcano(dsd_de.adata.copy(), contrast="Cmpd01")
        assert hasattr(fig, "savefig")
        plt.close("all")

    def test_volcano_unknown_contrast_raises(self, dsd_de):
        with pytest.raises(KeyError, match="Nope"):
            plot_comparison_volcano(dsd_de, contrast="Nope")

    def test_heatmap_returns_figure(self, dsd_de):
        fig = plot_comparison_heatmap(dsd_de, n_top_genes=30)
        assert hasattr(fig, "savefig")
        plt.close("all")

    def test_heatmap_subset_contrasts(self, dsd_de):
        fig = plot_comparison_heatmap(dsd_de, contrasts=["Cmpd01"])
        assert hasattr(fig, "savefig")
        plt.close("all")

    def test_deg_counts_returns_figure(self, dsd_de):
        fig = plot_deg_counts(dsd_de)
        assert hasattr(fig, "savefig")
        plt.close("all")

    def test_boxplot_returns_figure(self, dsd_de):
        fig = plot_gene_boxplot(dsd_de, genes=["Gene0001", "Gene0013"])
        assert hasattr(fig, "savefig")
        plt.close("all")

    def test_boxplot_contrast_restricted(self, dsd_de):
        fig = plot_gene_boxplot(dsd_de, genes=["Gene0001"],
                                contrast="Cmpd01")
        assert hasattr(fig, "savefig")
        plt.close("all")

    def test_boxplot_accepts_pooled_contrast(self, dsd_de):
        from drugseqpy import DrugSeqData, compute_multi_de
        dsd = DrugSeqData(dsd_de.adata.copy())
        compute_multi_de(
            dsd,
            comparisons=[{"name": "pooled", "case": "Cmpd01",
                          "control": ["Cmpd02", "Cmpd03"]}],
            methods=["limma_voom"], within_plate=False, n_jobs=1,
            inplace=True,
        )
        fig = plot_gene_boxplot(dsd, genes="Gene0001", contrast="pooled")
        assert hasattr(fig, "savefig")
        plt.close("all")

    def test_boxplot_missing_gene_raises(self, dsd_de):
        with pytest.raises(ValueError, match="None of the requested"):
            plot_gene_boxplot(dsd_de, genes=["NotAGene"])

    def test_violin_returns_figure(self, dsd_de):
        fig = plot_gene_violin(dsd_de, genes=["Gene0001"])
        assert hasattr(fig, "savefig")
        plt.close("all")

    def test_no_results_raises(self):
        counts, obs = make_dummy_screen(n_genes=40, n_compounds=1, seed=22)
        dsd = create_drugseq_object(counts, obs)
        with pytest.raises(ValueError, match="comparison results"):
            plot_comparison_volcano(dsd)

    def test_no_significant_genes_heatmap_raises(self):
        counts, obs = make_dummy_screen(n_genes=40, n_compounds=1, seed=23)
        dsd = create_drugseq_object(counts, obs)
        compute_multi_de(dsd, reference="DMSO", methods=["limma_voom"],
                         fdr_threshold=1e-300, lfc_threshold=1e9,
                         within_plate=False, n_jobs=1, inplace=True)
        with pytest.raises(ValueError, match="No significant genes"):
            plot_comparison_heatmap(dsd)


class TestEnrichmentPlots:
    def test_dotplot_returns_figure(self, dsd_enriched):
        fig = plot_enrichment_dotplot(dsd_enriched, direction="both",
                                      mode="ora")
        assert hasattr(fig, "savefig")
        plt.close("all")

    def test_dotplot_has_overlap_size_legend(self, dsd_enriched):
        fig = plot_enrichment_dotplot(dsd_enriched, direction="up", mode="ora")
        legends = [axis.get_legend() for axis in fig.axes if axis.get_legend()]
        assert legends
        assert all(legend.get_title().get_text() == "Overlap genes"
                   for legend in legends)
        plt.close("all")

    def test_dotplot_gsea_mode(self, dsd_enriched):
        fig = plot_enrichment_dotplot(dsd_enriched, mode="gsea")
        assert hasattr(fig, "savefig")
        plt.close("all")

    def test_barplot_returns_figure(self, dsd_enriched):
        fig = plot_enrichment_barplot(dsd_enriched, library="lib",
                                      direction="up", mode="ora")
        assert hasattr(fig, "savefig")
        plt.close("all")

    def test_barplot_empty_results_returns_figure(self):
        if not HAS_GSEAPY:
            pytest.skip("gseapy not installed")
        counts, obs = make_dummy_screen(n_genes=60, n_compounds=1, seed=24)
        dsd = create_drugseq_object(counts, obs)
        compute_multi_de(dsd, reference="DMSO", methods=["limma_voom"],
                         fdr_threshold=1e-300, lfc_threshold=1e9,
                         within_plate=False, n_jobs=1, inplace=True)
        gs = {"S": [f"Gene{i:04d}" for i in range(1, 19)]}
        with _tmp_gmt(gs) as gmt:
            run_enrichment(dsd, libraries=["lib"],
                           gene_set_paths={"lib": gmt}, mode=("ora",),
                           min_size=5, max_size=50, inplace=True)
            fig = plot_enrichment_barplot(dsd, library="lib",
                                          direction="up", mode="ora")
            assert hasattr(fig, "savefig")
            plt.close("all")

    def test_no_enrichment_results_shows_empty_panel(self):
        # fresh object without enrichment results (not the shared fixture)
        counts, obs = make_dummy_screen(n_genes=40, n_compounds=1, seed=25)
        dsd = create_drugseq_object(counts, obs)
        compute_multi_de(dsd, reference="DMSO", methods=["limma_voom"],
                         within_plate=False, n_jobs=1, inplace=True)
        fig = plot_enrichment_dotplot(dsd)
        assert hasattr(fig, "savefig")
        plt.close("all")


from contextlib import contextmanager


@contextmanager
def _tmp_gmt(gene_sets):
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "lib.gmt")
        _save_gmt(path, gene_sets)
        yield path
