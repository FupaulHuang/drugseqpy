"""tests/test_integration.py — End-to-end pipeline integration tests."""

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

import drugseqpy as ds
from drugseqpy.utils import make_dummy_screen


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def test_full_pipeline_end_to_end():
    """Run the complete DRUGseqPy workflow and assert data consistency."""
    counts, obs = make_dummy_screen(
        n_genes=200, n_plates=2,
        n_dmso_per_plate=8, n_compounds=4, n_reps=4, seed=2024
    )

    # 1. Construct
    dsd = ds.create_drugseq_object(counts, obs)
    assert dsd.n_obs == len(obs)
    assert dsd.n_vars == len(counts)

    # 2. Validate
    result = ds.validate_metadata(dsd, verbose=False)
    assert result["ok"]

    # 3. QC
    ds.compute_qc_metrics(dsd, inplace=True)
    assert "total_umi" in dsd.obs.columns
    assert "outlier_score" in dsd.obs.columns

    ds.compute_plate_qc(dsd, inplace=True)
    assert len(dsd.adata.uns["plate_qc"]) == 2

    # 4. Group QC
    grp = ds.compute_group_qc(dsd)
    assert "sd_value" in grp.columns

    # 5. Filter
    dsd_f = ds.filter_samples(dsd, min_umi=50, max_pct_mito=100)
    dsd_f = ds.filter_genes(dsd_f, min_count=1, min_samples=2,
                             group_aware=True, verbose=False)
    assert dsd_f.n_obs <= dsd.n_obs
    assert dsd_f.n_vars <= dsd.n_vars

    # 6. Normalize
    ds.normalize_counts(dsd_f, method="limma_voom", inplace=True)
    assert dsd_f.adata.X is not None
    assert dsd_f.adata.uns["norm_method"] == "limma_voom"
    assert dsd_f.adata.X.shape == (dsd_f.n_obs, dsd_f.n_vars)

    # 7. PCA
    ds.run_pca(dsd_f, n_pcs=20, n_variable_genes=100, inplace=True)
    assert "X_pca" in dsd_f.adata.obsm
    assert dsd_f.adata.obsm["X_pca"].shape[0] == dsd_f.n_obs

    # 8. DMSO embedding
    ds.embed_dmso(dsd_f, n_pcs=10, n_variable_genes=100, inplace=True)
    assert "X_dmso_pca" in dsd_f.adata.obsm
    assert "pert_score" in dsd_f.obs.columns
    assert len(dsd_f.obs["pert_score"]) == dsd_f.n_obs

    # 9. Differential expression
    ds.compute_multi_de(
        dsd_f, reference="DMSO", method="ols_voom",
        within_plate=True, n_jobs=1, inplace=True
    )
    de = dsd_f.adata.uns["de_results"]
    assert len(de) > 0
    for cmpd, df in de.items():
        assert "logFC" in df.columns
        assert "padj" in df.columns
        assert "significant" in df.columns

    # 10. Summarise DE
    de_sum = ds.summarise_de(dsd_f)
    assert len(de_sum) == len(de)
    assert (de_sum["n_sig_total"] == de_sum["n_sig_up"] + de_sum["n_sig_down"]).all()

    # 11. aggregate_by_de
    from drugseqpy.screen import aggregate_by_de
    de_mat = aggregate_by_de(dsd_f)
    assert de_mat.shape == (dsd_f.n_vars, len(de))
    assert not de_mat.isnull().any().any()

    # 12. Fingerprint
    from drugseqpy.screen import compute_compound_fingerprint
    fp = compute_compound_fingerprint(dsd_f, fdr_threshold=1.0, lfc_threshold=0.0)
    assert set(np.unique(fp.values)).issubset({-1, 0, 1})

    # 13. Connectivity
    sim = ds.connectivity_score(dsd_f, method="cosine")
    n   = len(de)
    assert sim.shape == (n, n)
    assert np.all(np.diag(sim) > 0.99)

    # 14. Screen overview plot
    fig = ds.plot_screen_overview(dsd_f, use_pert_score=True, top_n_label=0)
    assert hasattr(fig, "savefig")
    plt.close("all")

    print("\n✓ Full pipeline integration test passed.")


# ---------------------------------------------------------------------------
# Data consistency
# ---------------------------------------------------------------------------

def test_filter_then_normalize_shapes_consistent():
    counts, obs = make_dummy_screen(n_genes=100, n_compounds=3, seed=42)
    dsd = ds.create_drugseq_object(counts, obs)
    ds.compute_qc_metrics(dsd, inplace=True)
    dsd_f = ds.filter_genes(dsd, min_count=1, min_samples=2, verbose=False)
    ds.normalize_counts(dsd_f, method="TMM", inplace=True)
    assert dsd_f.adata.X.shape == (dsd_f.n_obs, dsd_f.n_vars)


def test_pca_embedding_rows_match_n_obs():
    counts, obs = make_dummy_screen(n_genes=120, n_compounds=3, seed=43)
    dsd = ds.create_drugseq_object(counts, obs)
    ds.normalize_counts(dsd, method="log1p", inplace=True)
    ds.run_pca(dsd, n_pcs=10, n_variable_genes=80, inplace=True)
    assert dsd.adata.obsm["X_pca"].shape[0] == dsd.n_obs


def test_de_compound_names_subset_of_meta():
    counts, obs = make_dummy_screen(n_genes=100, n_compounds=3, seed=44)
    dsd = ds.create_drugseq_object(counts, obs)
    ds.normalize_counts(dsd, method="log1p", inplace=True)
    ds.compute_multi_de(dsd, reference="DMSO", method="t_test",
                         within_plate=False, n_jobs=1, inplace=True)
    meta_cmpds = set(obs["compound"].unique()) - {"DMSO"}
    de_cmpds   = set(dsd.adata.uns["de_results"].keys())
    assert de_cmpds.issubset(meta_cmpds)


def test_merge_then_de_succeeds():
    counts1, obs1 = make_dummy_screen(n_genes=80, n_compounds=2, seed=50)
    counts2, obs2 = make_dummy_screen(n_genes=80, n_compounds=2, seed=51)
    n2 = len(obs2)
    obs2.index   = [f"T{i:04d}" for i in range(n2)]
    counts2.columns = obs2.index

    dsd1 = ds.create_drugseq_object(counts1, obs1)
    dsd2 = ds.create_drugseq_object(counts2, obs2)
    merged = ds.merge_drugseq_objects(dsd1, dsd2)
    ds.normalize_counts(merged, method="log1p", inplace=True)
    ds.compute_multi_de(merged, reference="DMSO", method="t_test",
                         within_plate=False, inplace=True)
    assert len(merged.adata.uns["de_results"]) > 0


def test_dose_response_fit_runs():
    from drugseqpy.dose_response import fit_dose_response

    # Build a tiny dose-response dataset
    rng = np.random.default_rng(99)
    doses = [0, 0.1, 1, 10, 100]
    n_reps = 3
    n_genes = 30

    obs_rows = []
    for dose in doses:
        for r in range(n_reps):
            obs_rows.append({
                "compound": "DMSO" if dose == 0 else "TestCmpd",
                "dose": dose, "dose_unit": "uM",
                "plate_id": "P01",
                "well_id": f"A{len(obs_rows)+1:02d}",
                "sample_type": "DMSO" if dose == 0 else "treatment",
            })

    obs = pd.DataFrame(obs_rows,
                        index=[f"S{i:04d}" for i in range(len(obs_rows))])
    n_samp = len(obs)
    mat = rng.negative_binomial(5, 0.01, size=(n_genes, n_samp)).astype(float)
    counts = pd.DataFrame(mat,
                           index=[f"G{i:03d}" for i in range(n_genes)],
                           columns=obs.index)

    dsd = ds.create_drugseq_object(counts, obs)
    ds.normalize_counts(dsd, method="log1p", inplace=True)

    dr = fit_dose_response(dsd, compound="TestCmpd", n_genes_fit=10, min_r2=0.0)
    assert isinstance(dr, pd.DataFrame)
    assert "EC50" in dr.columns
    assert "r_squared" in dr.columns
    assert len(dr) == 10
