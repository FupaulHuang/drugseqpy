"""Batch-correction and embedding regression tests."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.sparse as sp

from drugseqpy import (
    compute_batch_effect_metrics,
    correct_batch,
    create_drugseq_object,
    plot_mds,
    plot_pca,
    plot_tsne,
    plot_umap,
    run_pca,
    run_tsne,
    run_umap,
)
from drugseqpy.utils import make_dummy_screen


def _screen():
    counts, obs = make_dummy_screen(
        n_genes=100,
        n_plates=2,
        n_compounds=2,
        n_reps=3,
        n_dmso_per_plate=4,
        seed=2,
    )
    return create_drugseq_object(counts, obs)


@pytest.mark.parametrize("method", ["limma_voom", "edgeR"])
def test_batch_correction_preserves_raw_counts_and_writes_expression_layers(method):
    dsd = _screen()
    raw = dsd.layers["counts"].copy()

    correct_batch(
        dsd,
        method=method,
        covariate_cols=["compound"],
        inplace=True,
    )

    raw_after = dsd.layers["counts"]
    difference = raw_after - raw if sp.issparse(raw) else raw_after - raw
    assert difference.nnz == 0 if sp.issparse(difference) else not difference.any()
    assert dsd.layers["batch_uncorrected"].shape == dsd.shape
    assert dsd.layers["batch_corrected"].shape == dsd.shape
    assert np.allclose(dsd.X, dsd.layers["batch_corrected"])
    assert dsd.uns["batch_method"] == method
    assert dsd.uns["batch_covariates"] == ["compound"]


def test_batch_correction_removes_plate_mean_and_preserves_balanced_compound_effect():
    dsd = _screen()
    correct_batch(
        dsd,
        method="limma_voom",
        covariate_cols=["compound"],
        inplace=True,
    )
    obs = dsd.obs
    uncorrected = np.asarray(dsd.layers["batch_uncorrected"])
    corrected = np.asarray(dsd.X)
    plate_levels = list(obs["plate_id"].astype(str).unique())
    plate_means = [
        corrected[obs["plate_id"].astype(str).eq(plate).to_numpy()].mean(axis=0)
        for plate in plate_levels
    ]
    assert np.allclose(plate_means[0], plate_means[1], atol=2e-5)

    dmso = obs["compound"].astype(str).eq("DMSO").to_numpy()
    drug = obs["compound"].astype(str).eq("Cmpd01").to_numpy()
    before_effect = uncorrected[drug].mean(axis=0) - uncorrected[dmso].mean(axis=0)
    after_effect = corrected[drug].mean(axis=0) - corrected[dmso].mean(axis=0)
    assert np.allclose(before_effect, after_effect, atol=2e-5)


def test_batch_correction_rejects_confounding():
    dsd = _screen()
    dsd.obs["confounded"] = dsd.obs["plate_id"].astype(str)
    with pytest.raises(ValueError, match="rank deficient"):
        correct_batch(dsd, covariate_cols=["confounded"], inplace=True)


def test_named_embedding_functions_create_each_requested_figure():
    dsd = _screen()
    correct_batch(dsd, covariate_cols=["compound"], inplace=True)
    run_pca(
        dsd,
        n_pcs=10,
        n_variable_genes=80,
        use_highly_variable=False,
        inplace=True,
    )
    run_umap(dsd, dims=8, n_neighbors=5, inplace=True)
    run_tsne(dsd, dims=8, perplexity=5, inplace=True)

    figures = [
        plot_pca(dsd, color_by="plate_id"),
        plot_umap(dsd, color_by="plate_id"),
        plot_tsne(dsd, color_by="plate_id"),
        plot_mds(dsd, group_by="plate_id", label_by=None, n_top_genes=50),
    ]
    assert all(hasattr(figure, "savefig") for figure in figures)
    assert dsd.obsm["X_tsne"].shape == (dsd.n_obs, 2)
    plt.close("all")


def test_named_embedding_functions_split_by_plate_and_color_by_compound():
    dsd = _screen()
    run_pca(
        dsd,
        n_pcs=10,
        n_variable_genes=80,
        use_highly_variable=False,
        inplace=True,
    )
    run_umap(dsd, dims=8, n_neighbors=5, inplace=True)
    run_tsne(dsd, dims=8, perplexity=5, inplace=True)

    figures = [
        plot_pca(dsd, color_by="compound", split_by="plate_id"),
        plot_umap(dsd, color_by="compound", split_by="plate_id"),
        plot_tsne(dsd, color_by="compound", split_by="plate_id"),
        plot_mds(
            dsd,
            group_by="compound",
            split_by="plate_id",
            label_by=None,
            n_top_genes=50,
        ),
    ]
    assert all(len(figure.axes) == 2 for figure in figures)
    assert all(len(figure.legends) == 1 for figure in figures)
    assert all(
        [axis.get_title() for axis in figure.axes] == [
            "plate_id = Plate01",
            "plate_id = Plate02",
        ]
        for figure in figures
    )
    plt.close("all")


def test_batch_effect_metrics_returns_one_row_per_diagnostic():
    before = _screen()
    corrected = before.copy()
    correct_batch(corrected, covariate_cols=["compound"], inplace=True)
    for dsd in (before, corrected):
        run_pca(
            dsd, n_pcs=10, n_variable_genes=80,
            use_highly_variable=False, inplace=True,
        )
    metrics = compute_batch_effect_metrics(before, corrected, n_dims=8)
    assert set(metrics.columns) == {"metric", "before", "after"}
    assert {
        "plate_silhouette",
        "plate_variance_fraction",
        "plate_classifier_balanced_accuracy",
        "median_matched_compound_centroid_distance",
        "cross_plate_compound_retrieval_accuracy",
    } == set(metrics["metric"])
