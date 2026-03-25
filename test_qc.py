"""tests/test_qc.py — QC metric correctness tests."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from drugseqpy import (
    create_drugseq_object,
    compute_qc_metrics,
    compute_plate_qc,
    plate_qc_summary,
    compute_group_qc,
    compute_replicate_icc,
    select_robust_controls,
    check_zero_inflation,
    validate_metadata,
)
from drugseqpy.utils import make_dummy_screen


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_simple_object(n_genes=100, n_dmso=6, n_treat=8, seed=0):
    rng = np.random.default_rng(seed)
    gene_names = (["MT-G1","MT-G2","RPL1","RPS1"] +
                  [f"Gene{i:03d}" for i in range(n_genes - 4)])
    n_samp = n_dmso + n_treat
    mat    = rng.negative_binomial(5, 5/405, size=(n_genes, n_samp)).astype(float)
    obs = pd.DataFrame({
        "plate_id":    "P01",
        "well_id":     [f"A{i+1:02d}" for i in range(n_samp)],
        "compound":    ["DMSO"] * n_dmso + ["CmpdA"] * n_treat,
        "dose":        [0.0] * n_dmso + [10.0] * n_treat,
        "dose_unit":   "uM",
        "sample_type": ["DMSO"] * n_dmso + ["treatment"] * n_treat,
    }, index=[f"S{i:04d}" for i in range(n_samp)])
    counts = pd.DataFrame(mat, index=gene_names,
                           columns=[f"S{i:04d}" for i in range(n_samp)])
    return create_drugseq_object(counts, obs)


# ---------------------------------------------------------------------------
# Per-sample QC
# ---------------------------------------------------------------------------

class TestComputeQCMetrics:
    def test_metrics_added_to_obs(self):
        dsd = _make_simple_object()
        compute_qc_metrics(dsd, inplace=True)
        expected = ["total_umi","n_genes_det","pct_mito","pct_ribo",
                    "gini_index","outlier_score"]
        for col in expected:
            assert col in dsd.obs.columns, f"Missing: {col}"

    def test_n_obs_rows_match(self):
        dsd = _make_simple_object()
        compute_qc_metrics(dsd, inplace=True)
        assert len(dsd.obs) == dsd.n_obs

    def test_pct_mito_100_for_mito_only_sample(self):
        """A sample with only MT genes should have pct_mito ≈ 100."""
        rng = np.random.default_rng(99)
        gene_names = ["MT-G1","MT-G2","Gene001","Gene002","Gene003"]
        mat = np.zeros((len(gene_names), 4), dtype=float)
        mat[0:2, 0] = rng.integers(100, 500, 2)   # S0: only mito
        mat[2:,  1] = rng.integers(100, 500, 3)   # S1: only non-mito
        mat[:,   2] = rng.integers(50, 200, len(gene_names))
        mat[:,   3] = rng.integers(50, 200, len(gene_names))
        obs = pd.DataFrame({
            "plate_id":"P01","well_id":["A1","A2","A3","A4"],
            "compound":"DMSO","dose":0,"dose_unit":"uM",
            "sample_type":"DMSO",
        }, index=[f"S{i}" for i in range(4)])
        counts = pd.DataFrame(mat, index=gene_names,
                               columns=[f"S{i}" for i in range(4)])
        dsd = create_drugseq_object(counts, obs)
        compute_qc_metrics(dsd, mito_pattern="^MT-", ribo_pattern="^XXX", inplace=True)
        assert round(dsd.obs.loc["S0","pct_mito"]) == 100
        assert round(dsd.obs.loc["S1","pct_mito"]) == 0

    def test_gini_near_zero_for_uniform_counts(self):
        """Uniform counts across genes → Gini ≈ 0."""
        rng = np.random.default_rng(7)
        n_genes, n_samp = 80, 4
        mat = np.full((n_genes, n_samp), 100.0)
        obs = pd.DataFrame({
            "plate_id":"P01","well_id":[f"A{i}" for i in range(n_samp)],
            "compound":"DMSO","dose":0,"dose_unit":"uM","sample_type":"DMSO",
        }, index=[f"S{i}" for i in range(n_samp)])
        counts = pd.DataFrame(mat, index=[f"G{i}" for i in range(n_genes)],
                               columns=[f"S{i}" for i in range(n_samp)])
        dsd = create_drugseq_object(counts, obs)
        compute_qc_metrics(dsd, inplace=True)
        assert (dsd.obs["gini_index"] < 0.02).all()

    def test_outlier_score_is_non_negative(self):
        dsd = _make_simple_object()
        compute_qc_metrics(dsd, inplace=True)
        assert (dsd.obs["outlier_score"] >= 0).all()

    def test_inplace_false_returns_dsd(self):
        dsd = _make_simple_object()
        result = compute_qc_metrics(dsd, inplace=False)
        assert isinstance(result, type(dsd))


# ---------------------------------------------------------------------------
# Plate QC
# ---------------------------------------------------------------------------

class TestComputePlateQC:
    def test_plate_qc_stored_in_uns(self, dsd_qc):
        assert "plate_qc" in dsd_qc.adata.uns
        assert len(dsd_qc.adata.uns["plate_qc"]) > 0

    def test_plate_keys_match_obs_plates(self, dsd_qc):
        plates_meta = set(dsd_qc.obs["plate_id"].unique())
        plates_qc   = set(dsd_qc.adata.uns["plate_qc"].keys())
        assert plates_qc == plates_meta

    def test_plate_qc_summary_returns_dataframe(self, dsd_qc):
        df = plate_qc_summary(dsd_qc)
        assert isinstance(df, pd.DataFrame)
        assert "zprime" in df.columns

    def test_zprime_perfect_separation(self):
        """Z'-factor = 1 when pos and neg controls have zero variance and
        non-zero separation."""
        pos = np.array([100.0, 100.0, 100.0])
        neg = np.array([10.0,  10.0,  10.0])
        from drugseqpy.qc import _zprime
        assert _zprime(pos, neg) == pytest.approx(1.0)

    def test_flag_is_valid_value(self, dsd_qc):
        for pid, d in dsd_qc.adata.uns["plate_qc"].items():
            assert d["flag"] in ("pass","warn","fail"), \
                f"Unexpected flag for plate {pid}: {d['flag']}"


# ---------------------------------------------------------------------------
# Group QC and ICC
# ---------------------------------------------------------------------------

class TestGroupQC:
    def test_group_qc_returns_dataframe(self, dsd_qc):
        df = compute_group_qc(dsd_qc, group_by="compound")
        assert isinstance(df, pd.DataFrame)
        for col in ["sd_value","mad_value","z_score","cv_pct"]:
            assert col in df.columns

    def test_group_qc_one_row_per_group(self, dsd_qc):
        df = compute_group_qc(dsd_qc, group_by="compound")
        n_groups = dsd_qc.obs["compound"].nunique()
        assert len(df) == n_groups

    def test_icc_between_0_and_1(self, dsd_qc):
        res = compute_replicate_icc(dsd_qc, group_by="compound", metric="total_umi")
        assert 0.0 <= res["icc"] <= 1.0


# ---------------------------------------------------------------------------
# Robust controls & zero inflation
# ---------------------------------------------------------------------------

class TestRobustControls:
    def test_returns_list(self, dsd_qc):
        result = select_robust_controls(dsd_qc, min_cor=0.0)
        assert isinstance(result, list)

    def test_all_ids_in_obs(self, dsd_qc):
        result = select_robust_controls(dsd_qc, min_cor=0.0)
        assert all(sid in dsd_qc.obs_names for sid in result)


class TestZeroInflation:
    def test_returns_dataframe(self, dsd_qc):
        df = check_zero_inflation(dsd_qc, max_genes=50)
        assert isinstance(df, pd.DataFrame)
        assert "zero_inflated" in df.columns

    def test_zero_inflated_column_is_bool(self, dsd_qc):
        df = check_zero_inflation(dsd_qc, max_genes=50)
        assert df["zero_inflated"].dtype == bool


# ---------------------------------------------------------------------------
# Metadata validation
# ---------------------------------------------------------------------------

class TestValidateMetadata:
    def test_valid_object_passes(self, mini_screen):
        result = validate_metadata(mini_screen, verbose=False)
        assert result["ok"]

    def test_missing_column_flagged(self):
        counts, obs = make_dummy_screen(n_genes=40, n_compounds=2, seed=99)
        obs2 = obs.drop(columns=["compound"])
        dsd = create_drugseq_object(counts, obs2)
        result = validate_metadata(dsd, verbose=False)
        assert not result["ok"]
        assert any("compound" in iss for iss in result["issues"])
