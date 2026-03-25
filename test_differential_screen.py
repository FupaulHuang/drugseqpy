"""tests/test_differential_screen.py"""

import numpy as np
import pandas as pd
import pytest

from drugseqpy import (
    compute_multi_de,
    summarise_de,
)
from drugseqpy.screen import (
    aggregate_by_de,
    compute_compound_fingerprint,
    compute_compound_similarity_network,
    plot_screen_overview,
    plot_screen_heatmap,
)


class TestComputeMultiDE:
    def test_de_results_populated(self, dsd_de):
        assert len(dsd_de.adata.uns["de_results"]) > 0

    def test_de_results_compounds_are_non_dmso(self, dsd_de):
        for cmpd in dsd_de.adata.uns["de_results"]:
            assert cmpd != "DMSO"

    def test_de_result_columns(self, dsd_de):
        for cmpd, df in dsd_de.adata.uns["de_results"].items():
            for col in ["gene","logFC","pvalue","padj","significant"]:
                assert col in df.columns, f"Missing '{col}' for {cmpd}"

    def test_significant_is_boolean(self, dsd_de):
        for df in dsd_de.adata.uns["de_results"].values():
            assert df["significant"].dtype == bool

    def test_padj_between_0_and_1(self, dsd_de):
        for df in dsd_de.adata.uns["de_results"].values():
            valid = df["padj"].dropna()
            assert (valid >= 0).all() and (valid <= 1).all()

    def test_n_genes_matches_var(self, dsd_de):
        n_var = dsd_de.n_vars
        for cmpd, df in dsd_de.adata.uns["de_results"].items():
            assert len(df) == n_var, f"Row count mismatch for {cmpd}"


class TestSummariseDE:
    def test_returns_dataframe(self, dsd_de):
        df = summarise_de(dsd_de)
        assert isinstance(df, pd.DataFrame)

    def test_one_row_per_compound(self, dsd_de):
        df = summarise_de(dsd_de)
        n_cpds = len(dsd_de.adata.uns["de_results"])
        assert len(df) == n_cpds

    def test_n_sig_total_equals_sum_up_down(self, dsd_de):
        df = summarise_de(dsd_de)
        assert (df["n_sig_total"] == df["n_sig_up"] + df["n_sig_down"]).all()


class TestAggregateByDE:
    def test_shape(self, dsd_de):
        mat = aggregate_by_de(dsd_de)
        n_cpds = len(dsd_de.adata.uns["de_results"])
        assert mat.shape == (dsd_de.n_vars, n_cpds)

    def test_columns_are_compound_names(self, dsd_de):
        mat = aggregate_by_de(dsd_de)
        assert set(mat.columns) == set(dsd_de.adata.uns["de_results"].keys())

    def test_no_inf_or_nan(self, dsd_de):
        mat = aggregate_by_de(dsd_de)
        assert not mat.isnull().any().any()
        assert not np.isinf(mat.values).any()


class TestCompoundFingerprint:
    def test_returns_dataframe(self, dsd_de):
        fp = compute_compound_fingerprint(dsd_de, fdr_threshold=1.0, lfc_threshold=0.0)
        assert isinstance(fp, pd.DataFrame)

    def test_values_in_minus1_0_1(self, dsd_de):
        fp = compute_compound_fingerprint(dsd_de, fdr_threshold=1.0, lfc_threshold=0.0)
        unique = set(np.unique(fp.values))
        assert unique.issubset({-1, 0, 1})

    def test_columns_are_compounds(self, dsd_de):
        fp = compute_compound_fingerprint(dsd_de, fdr_threshold=1.0, lfc_threshold=0.0)
        assert set(fp.columns) == set(dsd_de.adata.uns["de_results"].keys())


class TestConnectivityScore:
    def test_cosine_similarity_shape(self, dsd_de):
        from drugseqpy import connectivity_score
        sim = connectivity_score(dsd_de, method="cosine")
        n = len(dsd_de.adata.uns["de_results"])
        assert sim.shape == (n, n)

    def test_diagonal_near_1(self, dsd_de):
        from drugseqpy import connectivity_score
        sim = connectivity_score(dsd_de, method="cosine")
        diag = np.diag(sim)
        assert (diag > 0.99).all()

    def test_similarity_network_returns_dict(self, dsd_de):
        net = compute_compound_similarity_network(dsd_de, method="cosine", threshold=-1)
        assert "similarity" in net
        assert isinstance(net["similarity"], pd.DataFrame)


class TestScreenPlots:
    def test_screen_overview_returns_figure(self, dsd_de):
        import matplotlib.pyplot as plt
        fig = plot_screen_overview(dsd_de, use_pert_score=False, top_n_label=0)
        assert hasattr(fig, "savefig")
        plt.close("all")

    def test_screen_heatmap_returns_figure(self, dsd_de):
        import matplotlib.pyplot as plt
        fig = plot_screen_heatmap(dsd_de, n_top=20, fdr_threshold=1.0)
        assert hasattr(fig, "savefig")
        plt.close("all")
