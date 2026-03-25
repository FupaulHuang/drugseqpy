"""tests/test_normalization_filtering.py"""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from drugseqpy import (
    create_drugseq_object,
    normalize_counts,
    compare_normalizations,
    export_matrix,
    filter_samples,
    filter_genes,
    compute_qc_metrics,
)
from drugseqpy.utils import make_dummy_screen


class TestNormalizeCounts:
    @pytest.mark.parametrize("method", ["log1p","CPM","TMM","limma_voom"])
    def test_method_runs(self, dsd_qc, method):
        from drugseqpy.core import DrugSeqData
        dsd = DrugSeqData(dsd_qc.adata.copy())
        normalize_counts(dsd, method=method, inplace=True)
        assert dsd.adata.X is not None
        assert dsd.adata.uns["norm_method"] == method

    def test_output_shape_matches_input(self, dsd_qc):
        from drugseqpy.core import DrugSeqData
        dsd = DrugSeqData(dsd_qc.adata.copy())
        normalize_counts(dsd, method="limma_voom", inplace=True)
        assert dsd.adata.X.shape == (dsd.n_obs, dsd.n_vars)

    def test_counts_layer_unchanged(self, dsd_qc):
        from drugseqpy.core import DrugSeqData
        dsd = DrugSeqData(dsd_qc.adata.copy())
        raw_before = dsd.adata.layers["counts"].toarray().copy()
        normalize_counts(dsd, method="TMM", inplace=True)
        raw_after = dsd.adata.layers["counts"].toarray()
        np.testing.assert_array_equal(raw_before, raw_after)

    def test_invalid_method_raises(self, dsd_qc):
        from drugseqpy.core import DrugSeqData
        dsd = DrugSeqData(dsd_qc.adata.copy())
        with pytest.raises(ValueError, match="method must be"):
            normalize_counts(dsd, method="invalid_method", inplace=True)

    def test_limma_voom_lower_cv_than_raw(self):
        from drugseqpy.normalization import _limma_voom
        counts, obs = make_dummy_screen(n_genes=200, n_dmso_per_plate=10, n_compounds=4, seed=333)
        dsd = create_drugseq_object(counts, obs)
        compute_qc_metrics(dsd, inplace=True)
        dmso_mask = dsd.obs["sample_type"] == "DMSO"
        raw = dsd.adata.layers["counts"][dmso_mask].toarray().astype(float)
        lib = raw.sum(axis=1, keepdims=True)
        raw_log = np.log2(raw / (lib + 1e-8) * 1e6 + 1)
        raw_cv  = np.mean(raw_log.std(axis=0) / (np.abs(raw_log.mean(axis=0)) + 1e-8))
        norm_log = _limma_voom(raw)
        norm_cv  = np.mean(norm_log.std(axis=0) / (np.abs(norm_log.mean(axis=0)) + 1e-8))
        assert norm_cv < raw_cv


class TestCompareNormalizations:
    def test_returns_dataframe_with_all_methods(self, dsd_qc):
        df = compare_normalizations(dsd_qc, methods=["raw","CPM","TMM","limma_voom"])
        assert set(df["method"]) == {"raw","CPM","TMM","limma_voom"}
        assert "median_rle_center" in df.columns
        assert "median_rle_iqr"    in df.columns
        assert "mean_cv"           in df.columns

    def test_values_are_non_negative(self, dsd_qc):
        df = compare_normalizations(dsd_qc, methods=["raw","TMM","limma_voom"])
        assert (df["median_rle_center"] >= 0).all()
        assert (df["median_rle_iqr"]    >= 0).all()
        assert (df["mean_cv"]           >= 0).all()

    def test_raw_has_higher_rle_center_than_normalized(self, dsd_qc):
        df = compare_normalizations(dsd_qc, methods=["raw","CPM"], subset_type=None)
        raw_rle = df.loc[df["method"]=="raw", "median_rle_center"].values[0]
        cpm_rle = df.loc[df["method"]=="CPM", "median_rle_center"].values[0]
        assert raw_rle >= cpm_rle


class TestExportMatrix:

    @pytest.mark.parametrize("method", [
        "raw", "CPM", "TMM", "limma_voom", "log1p", "size_factors"
    ])
    def test_all_methods_return_dataframe(self, dsd_qc, method):
        df = export_matrix(dsd_qc, method=method, path=None)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (dsd_qc.n_obs, dsd_qc.n_vars)

    def test_index_matches_sample_ids(self, dsd_qc):
        df = export_matrix(dsd_qc, method="CPM")
        assert list(df.index) == dsd_qc.obs_names.tolist()

    def test_columns_match_gene_ids(self, dsd_qc):
        df = export_matrix(dsd_qc, method="TMM")
        assert list(df.columns) == dsd_qc.var_names.tolist()

    def test_raw_layer_non_negative(self, dsd_qc):
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
        samples = dsd_qc.obs_names[:5].tolist()
        df = export_matrix(dsd_qc, method="TMM", samples=samples)
        assert list(df.index) == samples

    def test_gene_and_sample_subset_combined(self, dsd_qc):
        genes   = dsd_qc.var_names[:8].tolist()
        samples = dsd_qc.obs_names[:4].tolist()
        df = export_matrix(dsd_qc, method="TMM", genes=genes, samples=samples)
        assert df.shape == (4, 8)

    def test_obs_cols_prepended(self, dsd_qc):
        df = export_matrix(dsd_qc, method="CPM", obs_cols=["compound","plate_id"])
        assert list(df.columns[:2]) == ["compound","plate_id"]

    def test_log_transform_false_non_negative(self, dsd_qc):
        df = export_matrix(dsd_qc, method="CPM", log_transform=False)
        assert (df.values >= 0).all()

    def test_size_factors_linear_non_negative(self, dsd_qc):
        df = export_matrix(dsd_qc, method="size_factors", log_transform=False)
        assert (df.values >= 0).all()

    def test_round_decimals(self, dsd_qc):
        df = export_matrix(dsd_qc, method="TMM", round_decimals=2)
        assert np.allclose(df.values, np.round(df.values, 2), atol=1e-9)

    def test_write_csv(self, tmp_path, dsd_qc):
        p = str(tmp_path / "out.csv")
        export_matrix(dsd_qc, method="CPM", path=p)
        loaded = pd.read_csv(p, index_col=0)
        assert loaded.shape == (dsd_qc.n_obs, dsd_qc.n_vars)

    def test_write_tsv(self, tmp_path, dsd_qc):
        p = str(tmp_path / "out.tsv")
        export_matrix(dsd_qc, method="TMM", path=p)
        loaded = pd.read_csv(p, sep="\t", index_col=0)
        assert loaded.shape == (dsd_qc.n_obs, dsd_qc.n_vars)

    def test_fmt_inferred_from_extension(self, tmp_path, dsd_qc):
        p = str(tmp_path / "matrix.tsv")
        df = export_matrix(dsd_qc, method="CPM", path=p, return_df=True)
        loaded = pd.read_csv(p, sep="\t", index_col=0)
        assert loaded.shape == df.shape

    def test_csv_roundtrip_values_close(self, tmp_path, dsd_qc):
        p = str(tmp_path / "roundtrip.csv")
        df_out = export_matrix(dsd_qc, method="TMM", path=p,
                                round_decimals=4, return_df=True)
        df_in = pd.read_csv(p, index_col=0)
        np.testing.assert_allclose(df_out.values, df_in.values, atol=1e-3)

    def test_invalid_method_raises(self, dsd_qc):
        with pytest.raises(ValueError, match="method must be one of"):
            export_matrix(dsd_qc, method="scran_invalid")

    def test_invalid_format_raises(self, dsd_qc):
        with pytest.raises(ValueError, match="Unsupported format"):
            export_matrix(dsd_qc, method="CPM", path=None, fmt="docx")

    def test_invalid_layer_raises(self, dsd_qc):
        with pytest.raises(KeyError, match="Layer"):
            export_matrix(dsd_qc, layer="nonexistent_layer")

    def test_return_df_false_returns_none_when_path_given(self, tmp_path, dsd_qc):
        p = str(tmp_path / "out.csv")
        result = export_matrix(dsd_qc, method="CPM", path=p, return_df=False)
        assert result is None

    def test_path_none_always_returns_df(self, dsd_qc):
        df = export_matrix(dsd_qc, method="TMM", path=None, return_df=False)
        assert isinstance(df, pd.DataFrame)

    def test_tmm_differs_from_cpm(self, dsd_qc):
        tmm = export_matrix(dsd_qc, method="TMM")
        cpm = export_matrix(dsd_qc, method="CPM")
        assert not np.allclose(tmm.values, cpm.values, atol=1e-6)

    def test_h5ad_export(self, tmp_path, dsd_qc):
        pytest.importorskip("anndata")
        import anndata as ad
        p = str(tmp_path / "out.h5ad")
        export_matrix(dsd_qc, method="TMM", path=p, fmt="h5ad")
        loaded = ad.read_h5ad(p)
        assert loaded.n_obs == dsd_qc.n_obs
        assert loaded.n_vars == dsd_qc.n_vars
        assert loaded.uns.get("export_norm_method") == "TMM"


class TestFilterSamples:
    def test_low_umi_samples_removed(self, dsd_qc):
        filtered = filter_samples(dsd_qc, min_umi=999_999, keep_sample_types=["DMSO"])
        dmso_n = (dsd_qc.obs["sample_type"] == "DMSO").sum()
        assert filtered.n_obs == dmso_n

    def test_no_samples_removed_with_loose_thresholds(self, dsd_qc):
        filtered = filter_samples(dsd_qc, min_umi=0, max_pct_mito=100)
        assert filtered.n_obs == dsd_qc.n_obs

    def test_raises_if_all_removed(self, dsd_qc):
        with pytest.raises(ValueError, match="All samples"):
            filter_samples(dsd_qc, min_umi=999_999_999, keep_sample_types=[])


class TestFilterGenes:
    def test_genes_reduced_with_strict_threshold(self, dsd_qc):
        filtered = filter_genes(dsd_qc, min_count=999, min_samples=1,
                                 group_aware=False, verbose=False)
        assert filtered.n_vars < dsd_qc.n_vars

    def test_no_genes_removed_with_min_count_0(self, dsd_qc):
        filtered = filter_genes(dsd_qc, min_count=0, min_samples=0,
                                 group_aware=False, verbose=False)
        assert filtered.n_vars == dsd_qc.n_vars

    def test_group_aware_keeps_more_genes_than_global(self, dsd_qc):
        ga   = filter_genes(dsd_qc, min_count=5, min_samples=3, group_aware=True,  verbose=False)
        glob = filter_genes(dsd_qc, min_count=5, min_samples=3, group_aware=False, verbose=False)
        assert ga.n_vars >= glob.n_vars

    def test_mito_removal(self, dsd_qc):
        filtered = filter_genes(dsd_qc, remove_mito=True, mito_pattern="^MT-", verbose=False)
        assert filtered.var_names.str.startswith("MT-").sum() == 0

    def test_raises_if_all_removed(self, dsd_qc):
        with pytest.raises(ValueError, match="All genes"):
            filter_genes(dsd_qc, min_count=999_999, min_samples=1, verbose=False)
