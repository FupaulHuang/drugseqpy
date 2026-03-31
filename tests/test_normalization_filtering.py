"""tests/test_normalization_filtering.py"""

import numpy as np
import pandas as pd
import pytest
import anndata as ad

from drugseqpy import (
    normalize_counts, compare_normalizations, export_matrix,
    filter_samples, filter_genes, compute_qc_metrics,
)
from drugseqpy.utils import make_dummy_screen


class TestNormalizeCounts:
    @pytest.mark.parametrize("method", ["log1p", "CPM", "TMM", "limma_voom"])
    def test_method_runs_and_sets_uns(self, dsd_qc, method):
        adata = dsd_qc.copy()
        normalize_counts(adata, method=method, inplace=True)
        assert adata.X is not None
        assert adata.uns["norm_method"] == method

    def test_shape_preserved(self, dsd_qc):
        adata = dsd_qc.copy()
        normalize_counts(adata, method="limma_voom", inplace=True)
        assert adata.X.shape == (adata.n_obs, adata.n_vars)

    def test_counts_layer_unchanged(self, dsd_qc):
        adata = dsd_qc.copy()
        import scipy.sparse as sp
        raw_before = adata.layers["counts"].toarray().copy()
        normalize_counts(adata, method="TMM", inplace=True)
        raw_after = adata.layers["counts"].toarray()
        np.testing.assert_array_equal(raw_before, raw_after)

    def test_invalid_method_raises(self, dsd_qc):
        with pytest.raises(ValueError, match="method must be one of"):
            normalize_counts(dsd_qc.copy(), method="bad_method", inplace=True)

    def test_scanpy_hvg_works_after_normalize(self, dsd_qc):
        """sc.pp.highly_variable_genes must run directly after normalize_counts."""
        import scanpy as sc
        adata = dsd_qc.copy()
        normalize_counts(adata, method="CPM", inplace=True)
        sc.pp.highly_variable_genes(adata, n_top_genes=50, flavor="seurat")
        assert "highly_variable" in adata.var.columns

    def test_limma_voom_lower_cv_than_raw(self):
        from drugseqpy.normalization import _limma_voom
        adata = make_dummy_screen(n_genes=200, n_dmso_per_plate=10,
                                   n_compounds=4, seed=333)
        compute_qc_metrics(adata, inplace=True)
        dmso = (adata.obs["sample_type"] == "DMSO").values
        raw  = adata.layers["counts"][dmso].toarray().astype(float)
        lib  = raw.sum(axis=1, keepdims=True)
        raw_log = np.log2(raw / (lib + 1e-8) * 1e6 + 1)
        raw_cv  = np.mean(raw_log.std(0) / (np.abs(raw_log.mean(0)) + 1e-8))
        norm_cv = np.mean(_limma_voom(raw).std(0) /
                           (np.abs(_limma_voom(raw).mean(0)) + 1e-8))
        assert norm_cv < raw_cv


class TestCompareNormalizations:
    def test_columns_present(self, dsd_qc):
        df = compare_normalizations(dsd_qc, methods=["raw","CPM","TMM","limma_voom"])
        assert set(df["method"]) == {"raw","CPM","TMM","limma_voom"}
        for col in ["median_rle_center","median_rle_iqr","mean_cv"]:
            assert col in df.columns

    def test_values_non_negative(self, dsd_qc):
        df = compare_normalizations(dsd_qc, methods=["raw","TMM"])
        assert (df[["median_rle_center","median_rle_iqr","mean_cv"]] >= 0).all().all()

    def test_raw_higher_rle_than_cpm(self, dsd_qc):
        df = compare_normalizations(dsd_qc, methods=["raw","CPM"], subset_type=None)
        raw = df.loc[df["method"]=="raw",  "median_rle_center"].values[0]
        cpm = df.loc[df["method"]=="CPM",  "median_rle_center"].values[0]
        assert raw >= cpm


class TestExportMatrix:
    @pytest.mark.parametrize("method", [
        "raw","CPM","TMM","limma_voom","log1p","size_factors"
    ])
    def test_all_methods_return_dataframe(self, dsd_qc, method):
        df = export_matrix(dsd_qc, method=method, path=None)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (dsd_qc.n_obs, dsd_qc.n_vars)

    def test_index_and_columns(self, dsd_qc):
        df = export_matrix(dsd_qc, method="TMM")
        assert list(df.index)   == dsd_qc.obs_names.tolist()
        assert list(df.columns) == dsd_qc.var_names.tolist()

    def test_counts_layer_non_negative(self, dsd_qc):
        df = export_matrix(dsd_qc, layer="counts")
        assert (df.values >= 0).all()

    def test_X_layer_shape(self, dsd_normed):
        df = export_matrix(dsd_normed, layer="X")
        assert df.shape == (dsd_normed.n_obs, dsd_normed.n_vars)

    def test_gene_subset(self, dsd_qc):
        genes = dsd_qc.var_names[:10].tolist()
        df = export_matrix(dsd_qc, method="CPM", genes=genes)
        assert list(df.columns) == genes

    def test_sample_subset(self, dsd_qc):
        sids = dsd_qc.obs_names[:5].tolist()
        df = export_matrix(dsd_qc, method="TMM", samples=sids)
        assert list(df.index) == sids

    def test_combined_subset_shape(self, dsd_qc):
        genes = dsd_qc.var_names[:8].tolist()
        sids  = dsd_qc.obs_names[:4].tolist()
        df = export_matrix(dsd_qc, method="TMM", genes=genes, samples=sids)
        assert df.shape == (4, 8)

    def test_obs_cols_prepended(self, dsd_qc):
        df = export_matrix(dsd_qc, method="CPM", obs_cols=["compound","plate_id"])
        assert list(df.columns[:2]) == ["compound","plate_id"]

    def test_log_transform_false_non_negative(self, dsd_qc):
        df = export_matrix(dsd_qc, method="CPM", log_transform=False)
        assert (df.values >= 0).all()

    def test_round_decimals(self, dsd_qc):
        df = export_matrix(dsd_qc, method="TMM", round_decimals=2)
        assert np.allclose(df.values, np.round(df.values, 2), atol=1e-9)

    def test_write_csv(self, tmp_path, dsd_qc):
        p = str(tmp_path / "out.csv")
        export_matrix(dsd_qc, method="CPM", path=p)
        assert pd.read_csv(p, index_col=0).shape == (dsd_qc.n_obs, dsd_qc.n_vars)

    def test_write_tsv(self, tmp_path, dsd_qc):
        p = str(tmp_path / "out.tsv")
        export_matrix(dsd_qc, method="TMM", path=p)
        assert pd.read_csv(p, sep="\t", index_col=0).shape == (dsd_qc.n_obs, dsd_qc.n_vars)

    def test_fmt_inferred_from_extension(self, tmp_path, dsd_qc):
        p = str(tmp_path / "out.tsv")
        df = export_matrix(dsd_qc, method="CPM", path=p, return_df=True)
        loaded = pd.read_csv(p, sep="\t", index_col=0)
        assert loaded.shape == df.shape

    def test_csv_roundtrip(self, tmp_path, dsd_qc):
        p = str(tmp_path / "rt.csv")
        df_out = export_matrix(dsd_qc, method="TMM", path=p,
                                round_decimals=4, return_df=True)
        df_in  = pd.read_csv(p, index_col=0)
        np.testing.assert_allclose(df_out.values, df_in.values, atol=1e-3)

    def test_invalid_method_raises(self, dsd_qc):
        with pytest.raises(ValueError, match="method must be one of"):
            export_matrix(dsd_qc, method="bad")

    def test_invalid_format_raises(self, dsd_qc):
        with pytest.raises(ValueError, match="Unsupported format"):
            export_matrix(dsd_qc, method="CPM", path=None, fmt="docx")

    def test_invalid_layer_raises(self, dsd_qc):
        with pytest.raises(KeyError, match="Layer"):
            export_matrix(dsd_qc, layer="nonexistent")

    def test_return_false_with_path_gives_none(self, tmp_path, dsd_qc):
        p = str(tmp_path / "out.csv")
        assert export_matrix(dsd_qc, method="CPM", path=p, return_df=False) is None

    def test_path_none_always_returns_df(self, dsd_qc):
        assert isinstance(export_matrix(dsd_qc, method="TMM",
                                          path=None, return_df=False), pd.DataFrame)

    def test_tmm_differs_from_cpm(self, dsd_qc):
        tmm = export_matrix(dsd_qc, method="TMM")
        cpm = export_matrix(dsd_qc, method="CPM")
        assert not np.allclose(tmm.values, cpm.values, atol=1e-6)

    def test_h5ad_roundtrip(self, tmp_path, dsd_qc):
        import anndata as ad
        p = str(tmp_path / "out.h5ad")
        export_matrix(dsd_qc, method="TMM", path=p)
        loaded = ad.read_h5ad(p)
        assert loaded.n_obs == dsd_qc.n_obs
        assert loaded.uns.get("export_norm_method") == "TMM"


class TestFilterSamples:
    def test_protected_dmso_not_removed(self, dsd_qc):
        filtered = filter_samples(dsd_qc, min_umi=999_999,
                                   keep_sample_types=["DMSO"])
        n_dmso = (dsd_qc.obs["sample_type"] == "DMSO").sum()
        assert filtered.n_obs == n_dmso

    def test_loose_threshold_removes_nothing(self, dsd_qc):
        assert filter_samples(dsd_qc, min_umi=0, max_pct_mito=100).n_obs == dsd_qc.n_obs

    def test_raises_if_all_removed(self, dsd_qc):
        with pytest.raises(ValueError, match="All samples"):
            filter_samples(dsd_qc, min_umi=999_999_999, keep_sample_types=[])

    def test_returns_anndata(self, dsd_qc):
        assert isinstance(filter_samples(dsd_qc, min_umi=0), ad.AnnData)


class TestFilterGenes:
    def test_strict_threshold_reduces_genes(self, dsd_qc):
        assert filter_genes(dsd_qc, min_count=999, min_samples=1,
                             group_aware=False, verbose=False).n_vars < dsd_qc.n_vars

    def test_zero_threshold_keeps_all(self, dsd_qc):
        assert filter_genes(dsd_qc, min_count=0, min_samples=0,
                             group_aware=False, verbose=False).n_vars == dsd_qc.n_vars

    def test_group_aware_keeps_more(self, dsd_qc):
        ga   = filter_genes(dsd_qc, min_count=5, min_samples=3,
                             group_aware=True,  verbose=False)
        glob = filter_genes(dsd_qc, min_count=5, min_samples=3,
                             group_aware=False, verbose=False)
        assert ga.n_vars >= glob.n_vars

    def test_mito_removal(self, dsd_qc):
        f = filter_genes(dsd_qc, remove_mito=True, mito_pattern="^MT-", verbose=False)
        assert f.var_names.str.startswith("MT-").sum() == 0

    def test_raises_if_all_removed(self, dsd_qc):
        with pytest.raises(ValueError, match="All genes"):
            filter_genes(dsd_qc, min_count=999_999, min_samples=1, verbose=False)

    def test_returns_anndata(self, dsd_qc):
        assert isinstance(filter_genes(dsd_qc, min_count=0,
                                        min_samples=0, verbose=False), ad.AnnData)
