"""tests/test_core.py — DrugSeqData construction, subsetting, merging."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from drugseqpy import create_drugseq_object, merge_drugseq_objects, DrugSeqData
from drugseqpy.utils import make_dummy_screen


def test_create_basic():
    counts, obs = make_dummy_screen(n_genes=50, n_plates=1,
                                    n_dmso_per_plate=4, n_compounds=2,
                                    n_reps=3, seed=1)
    dsd = create_drugseq_object(counts, obs)
    assert isinstance(dsd, DrugSeqData)
    assert dsd.n_vars == 50
    assert dsd.n_obs == len(obs)


def test_counts_layer_preserved():
    counts, obs = make_dummy_screen(n_genes=40, n_compounds=2, seed=2)
    dsd = create_drugseq_object(counts, obs)
    raw = dsd.adata.layers["counts"]
    assert sp.issparse(raw) or isinstance(raw, np.ndarray)
    assert raw.shape == (dsd.n_obs, dsd.n_vars)


def test_shape_property():
    counts, obs = make_dummy_screen(n_genes=60, n_compounds=3, seed=3)
    dsd = create_drugseq_object(counts, obs)
    assert dsd.shape == (dsd.n_obs, dsd.n_vars)


def test_subset_by_boolean_mask():
    counts, obs = make_dummy_screen(n_genes=80, n_compounds=3, seed=4)
    dsd = create_drugseq_object(counts, obs)
    mask = obs["sample_type"] == "DMSO"
    sub  = dsd[mask.values]
    assert sub.n_obs == mask.sum()
    assert sub.n_vars == dsd.n_vars


def test_subset_preserves_counts_layer():
    counts, obs = make_dummy_screen(n_genes=60, n_compounds=2, seed=5)
    dsd = create_drugseq_object(counts, obs)
    sub = dsd[:5]
    assert "counts" in sub.adata.layers
    assert sub.adata.layers["counts"].shape[0] == 5


def test_repr_contains_key_info():
    counts, obs = make_dummy_screen(n_genes=40, n_compounds=2, seed=6)
    dsd = create_drugseq_object(counts, obs)
    r = repr(dsd)
    assert "DrugSeqData" in r
    assert "Samples" in r


def test_mismatched_ids_warns():
    counts, obs = make_dummy_screen(n_genes=40, n_compounds=2, seed=7)
    obs_bad = obs.copy()
    obs_bad.index = ["X" + i for i in obs_bad.index]  # mismatch
    with pytest.raises(ValueError, match="Only"):
        create_drugseq_object(counts, obs_bad, min_overlap=0.9)


def test_merge_two_objects():
    counts1, obs1 = make_dummy_screen(n_genes=80, n_compounds=2, seed=10)
    counts2, obs2 = make_dummy_screen(n_genes=80, n_compounds=2, seed=11)
    n2 = len(obs2)
    obs2.index = [f"T{i:04d}" for i in range(n2)]
    counts2.columns = obs2.index

    dsd1 = create_drugseq_object(counts1, obs1)
    dsd2 = create_drugseq_object(counts2, obs2)
    merged = merge_drugseq_objects(dsd1, dsd2)

    assert merged.n_obs == dsd1.n_obs + dsd2.n_obs
    assert merged.n_vars == dsd1.n_vars  # same genes


def test_merge_raises_on_duplicate_ids():
    counts, obs = make_dummy_screen(n_genes=50, n_compounds=2, seed=12)
    dsd1 = create_drugseq_object(counts, obs)
    dsd2 = create_drugseq_object(counts, obs)   # same IDs
    with pytest.raises(ValueError, match="Duplicate"):
        merge_drugseq_objects(dsd1, dsd2)
