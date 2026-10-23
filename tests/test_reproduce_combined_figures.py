"""The documentation runner must not export original sample identifiers."""
from pathlib import Path
import runpy

import anndata as ad
import numpy as np
import pandas as pd


def test_sample_labels_and_metadata_are_anonymized_without_changing_counts():
    runner = runpy.run_path(str(Path(__file__).parents[1] / "scripts/reproduce_combined_figures.py"))
    counts = np.array([[1, 2], [3, 4], [5, 6]])
    obs = pd.DataFrame({
        "plate_id": pd.Categorical(["private_batch_b", "private_batch_a", "private_batch_b"]),
        "sample_label": ["private_batch_b", "private_batch_a", "private_batch_b"],
        "sample_id": ["original_1", "original_2", "original_3"],
        "well_id": ["A1", "A1", "A2"],
        "barcode": ["AAA", "AAA", "CCC"],
        "reads_per_well": [10, 20, 30],
        "in_filtered_matrix": [True, True, False],
        "expected_cells": ["human", "human", "mouse"],
        "source_path": ["/private/source"] * 3,
    }, index=["private_batch_b_A1", "private_batch_a_A1", "private_batch_b_A2"])
    obj = ad.AnnData(counts.copy(), obs=obs)
    labels = runner["anonymize_metadata"](obj)
    assert labels == {"private_batch_b": "sample_1", "private_batch_a": "sample_2"}
    assert obj.obs["plate_id"].tolist() == ["sample_1", "sample_2", "sample_1"]
    assert obj.obs_names.tolist() == ["sample_1_well_1", "sample_2_well_1", "sample_1_well_2"]
    assert not {"sample_label", "sample_id", "source_path"} & set(obj.obs.columns)
    assert "private" not in obj.obs.to_csv()
    assert obj.obs["barcode"].tolist() == ["AAA", "AAA", "CCC"]
    np.testing.assert_array_equal(obj.X, counts)
