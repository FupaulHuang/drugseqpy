"""tests/test_comparison_api.py — contrast normalization and DE methods."""

import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from drugseqpy import (
    create_drugseq_object,
    compute_multi_de,
    normalize_contrasts,
    run_comparison,
    run_de,
    summarise_comparison,
)
from drugseqpy.utils import make_dummy_screen

HAS_PYDESEQ2 = importlib.util.find_spec("pydeseq2") is not None
HAS_RPY2 = importlib.util.find_spec("rpy2") is not None

RESULT_COLS = ["gene", "logFC", "base_mean", "stat", "pvalue", "padj",
               "significant", "direction", "contrast", "case", "control",
               "method"]


@pytest.fixture(scope="module")
def screen():
    counts, obs = make_dummy_screen(
        n_genes=80, n_plates=2, n_dmso_per_plate=6,
        n_compounds=3, n_reps=4, seed=7,
    )
    return create_drugseq_object(counts, obs)


class TestNormalizeContrasts:
    def test_default_generates_all_vs_control(self, screen):
        cts = normalize_contrasts(None, screen.obs, "compound", "DMSO")
        assert [c["case"] for c in cts] == ["Cmpd01", "Cmpd02", "Cmpd03"]
        assert all(c["control"] == "DMSO" for c in cts)
        assert all(c["group_col"] == "compound" for c in cts)

    def test_dict_case_to_control(self, screen):
        cts = normalize_contrasts({"Cmpd01": "DMSO"}, screen.obs)
        assert cts == [{"name": "Cmpd01", "group_col": "compound",
                        "case": "Cmpd01", "control": "DMSO"}]

    def test_dict_named_tuple_form(self, screen):
        cts = normalize_contrasts({"A vs B": ("Cmpd01", "Cmpd02")}, screen.obs)
        assert cts[0]["name"] == "A vs B"
        assert cts[0]["case"] == "Cmpd01"
        assert cts[0]["control"] == "Cmpd02"

    def test_sequence_of_tuples(self, screen):
        cts = normalize_contrasts([("Cmpd01", "Cmpd02")], screen.obs)
        assert cts[0]["name"] == "Cmpd01 vs Cmpd02"

    def test_sequence_of_dicts_with_name(self, screen):
        cts = normalize_contrasts(
            [{"name": "one_vs_rest", "case": "Cmpd01", "control": "Cmpd02"}],
            screen.obs)
        assert cts[0]["name"] == "one_vs_rest"

    def test_single_contrast_mapping_form(self, screen):
        cts = normalize_contrasts(
            {"name": "one_vs_two", "case": "Cmpd01",
             "control": "Cmpd02"}, screen.obs
        )
        assert cts == [{"name": "one_vs_two", "group_col": "compound",
                        "case": "Cmpd01", "control": "Cmpd02"}]

    def test_pooled_control_levels(self, screen):
        cts = normalize_contrasts(
            [{"name": "one_vs_rest", "case": "Cmpd01",
              "control": ["Cmpd02", "Cmpd03"]}], screen.obs
        )
        assert cts[0]["control"] == ["Cmpd02", "Cmpd03"]

    def test_compact_mapping_supports_pooled_control(self, screen):
        cts = normalize_contrasts(
            {"Cmpd01": ["Cmpd02", "Cmpd03"]}, screen.obs
        )
        assert cts[0]["case"] == "Cmpd01"
        assert cts[0]["control"] == ["Cmpd02", "Cmpd03"]

    def test_single_pair_tuple_is_accepted(self, screen):
        cts = normalize_contrasts(("Cmpd01", "Cmpd02"), screen.obs)
        assert cts == [{"name": "Cmpd01 vs Cmpd02", "group_col": "compound",
                        "case": "Cmpd01", "control": "Cmpd02"}]

    def test_default_without_sample_type_keeps_nonreference_levels(self, screen):
        obs = screen.obs.drop(columns=["sample_type"]).copy()
        obs.loc[obs.index[0], "compound"] = "water"
        cts = normalize_contrasts(None, obs, "compound", "DMSO")
        assert "water" in {c["case"] for c in cts}

    def test_invalid_level_raises(self, screen):
        with pytest.raises(ValueError, match="not found"):
            normalize_contrasts({"Nope": "DMSO"}, screen.obs)

    def test_missing_obs_raises(self):
        with pytest.raises(ValueError, match="obs"):
            normalize_contrasts(None, None)


class TestRunComparison:
    def test_schema_and_direction(self, screen):
        res = run_comparison(screen, case="Cmpd01", control="DMSO",
                             method="limma_voom", within_plate=False)
        for col in RESULT_COLS:
            assert col in res.columns, f"missing {col}"
        assert len(res) == screen.n_vars
        assert res["significant"].dtype == bool
        assert set(res["direction"].unique()) <= {"up", "down", "ns"}
        assert (res["logFC"] == res["logFC"]).all()  # no NaN logFC

    def test_one_group_vs_other_group(self, screen):
        res = run_comparison(screen, case="Cmpd01", control="Cmpd02",
                             method="limma_voom", within_plate=False)
        assert res["case"].iloc[0] == "Cmpd01"
        assert res["control"].iloc[0] == "Cmpd02"

    def test_one_group_vs_pooled_other_groups(self, screen):
        res = run_comparison(
            screen, case="Cmpd01", control=["Cmpd02", "Cmpd03"],
            method="limma_voom", within_plate=False,
        )
        assert res["case"].iloc[0] == "Cmpd01"
        assert res["control"].iloc[0] == "Cmpd02|Cmpd03"

    def test_one_group_vs_rest_selector(self, screen):
        res = run_comparison(screen, case="Cmpd01", control="rest",
                             method="limma_voom", within_plate=False)
        assert res["control"].iloc[0] == "Cmpd02|Cmpd03|DMSO"

    def test_batch_covariates(self, screen):
        res = run_comparison(screen, case="Cmpd01", control="DMSO",
                             method="limma_voom", batch_cols=["plate_id"],
                             within_plate=False)
        assert len(res) == screen.n_vars

    def test_method_aliases(self, screen):
        a = run_comparison(screen, "Cmpd01", "DMSO", method="ols_voom",
                           within_plate=False)
        b = run_comparison(screen, "Cmpd01", "DMSO", method="limma_voom",
                           within_plate=False)
        np.testing.assert_allclose(a["logFC"], b["logFC"])
        assert a["method"].iloc[0] == "limma_voom"

    def test_method_names_are_case_and_alias_tolerant(self, screen):
        result = run_comparison(screen, "Cmpd01", "DMSO",
                                method="DESeq2", within_plate=False)
        assert result["method"].iloc[0] == "deseq"

    def test_invalid_method_raises(self, screen):
        with pytest.raises(ValueError, match="method"):
            run_comparison(screen, "Cmpd01", "DMSO", method="bogus")

    def test_insufficient_replicates_raises(self, screen):
        with pytest.raises(ValueError, match="replicate"):
            run_comparison(screen, "Cmpd01", "DMSO", method="limma_voom",
                           min_replicates=100)

    def test_missing_case_raises(self, screen):
        with pytest.raises(ValueError, match="not found"):
            run_comparison(screen, "Nope", "DMSO")

    def test_plain_anndata_is_accepted(self, screen):
        res = run_comparison(screen.adata.copy(), "Cmpd01", "DMSO",
                             within_plate=False)
        assert len(res) == screen.n_vars

    @pytest.mark.skipif(not HAS_PYDESEQ2, reason="pydeseq2 not installed")
    def test_deseq_method(self, screen):
        res = run_comparison(screen, "Cmpd01", "DMSO", method="deseq",
                             within_plate=False)
        assert res["method"].iloc[0] == "deseq"
        assert len(res) == screen.n_vars

    @pytest.mark.skipif(HAS_RPY2, reason="rpy2 present — real R backend")
    def test_edger_missing_dependency_message(self, screen):
        with pytest.raises(ImportError, match="edgeR.*rpy2"):
            run_comparison(screen, "Cmpd01", "DMSO", method="edgeR")

    def test_run_de_legacy_delegates(self, screen):
        res = run_de(screen, "Cmpd01", reference="DMSO",
                     method="ols_voom", within_plate=False)
        assert "direction" in res.columns
        assert res["control"].iloc[0] == "DMSO"


class TestComputeMultiDe:
    def test_single_method_stores_both_keys(self, screen):
        compute_multi_de(screen, reference="DMSO", methods=["limma_voom"],
                         within_plate=False, n_jobs=1, inplace=True)
        assert set(screen.adata.uns["comparison_results"].keys()) == \
            {"Cmpd01", "Cmpd02", "Cmpd03"}
        assert set(screen.adata.uns["de_results"].keys()) == \
            {"Cmpd01", "Cmpd02", "Cmpd03"}

    def test_multiple_methods_keyed_by_method(self, screen):
        compute_multi_de(screen, reference="DMSO",
                         methods=["limma_voom", "t_test"],
                         within_plate=False, n_jobs=1, inplace=True)
        keys = screen.adata.uns["comparison_results"].keys()
        assert "Cmpd01::limma_voom" in keys
        assert "Cmpd01::t_test" in keys
        # legacy mirror still populated from first method
        assert set(screen.adata.uns["de_results"]) == \
            {"Cmpd01", "Cmpd02", "Cmpd03"}

    def test_custom_comparisons_only(self, screen):
        compute_multi_de(
            screen,
            comparisons={"Cmpd01": "Cmpd02"},
            methods=["limma_voom"], within_plate=False, n_jobs=1,
            inplace=True,
        )
        res = screen.adata.uns["comparison_results"]["Cmpd01"]
        assert res["case"].iloc[0] == "Cmpd01"
        assert res["control"].iloc[0] == "Cmpd02"

    def test_summarise_comparison(self, screen):
        summary = summarise_comparison(screen)
        assert {"contrast", "method", "n_sig_up", "n_sig_down",
                "n_sig_total"}.issubset(summary.columns)
        assert (summary["n_sig_total"] ==
                summary["n_sig_up"] + summary["n_sig_down"]).all()

    def test_h5ad_roundtrip(self, screen, tmp_path):
        path = tmp_path / "screen.h5ad"
        screen.write(path)
        from drugseqpy import DrugSeqData
        dsd2 = DrugSeqData.read_h5ad(path)
        assert set(dsd2.adata.uns["comparison_results"]) == \
            set(screen.adata.uns["comparison_results"])
        df = dsd2.adata.uns["comparison_results"]["Cmpd01"]
        assert df["significant"].dtype == bool

    def test_parallel_jobs_store_results(self, screen):
        screen.adata.uns["comparison_results"] = {}
        compute_multi_de(screen, reference="DMSO", methods=["limma_voom"],
                         within_plate=False, n_jobs=2, inplace=True)
        assert set(screen.adata.uns["comparison_results"]) == {
            "Cmpd01", "Cmpd02", "Cmpd03"
        }

    @pytest.mark.skipif(HAS_RPY2, reason="rpy2 present — real R backend")
    def test_failed_optional_backend_is_recorded(self, screen):
        screen.adata.uns["comparison_results"] = {}
        compute_multi_de(screen, reference="DMSO", methods=["edgeR"],
                         within_plate=False, n_jobs=1, inplace=True)
        failures = screen.adata.uns.get("comparison_failures", {})
        assert "Cmpd01" in failures
        assert "edgeR" in failures["Cmpd01"]
