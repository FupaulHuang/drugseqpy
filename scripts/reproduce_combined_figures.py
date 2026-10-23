#!/usr/bin/env python3
"""Reproduce and validate combined figures using neutral sample labels.

Usage: python scripts/reproduce_combined_figures.py SOURCE_FOLDER OUTPUT_FOLDER
The source folder is read-only; all generated files go to OUTPUT_FOLDER.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import anndata as ad
import numpy as np
import pandas as pd

import drugseqpy as ds


def validate(tables, source, sample_labels):
    """Compare independently recomputed values against the original R exports."""
    checks = []

    def compare(name, actual, expected, keys, columns):
        expected = expected.copy()
        if "group" in expected:
            expected["group"] = expected["group"].astype(str).map(sample_labels)
            if expected["group"].isna().any():
                raise ValueError(f"{name}: reference groups do not match the input metadata.")
        actual = actual.set_index(keys).sort_index()
        expected = expected.set_index(keys).sort_index()
        pd.testing.assert_index_equal(actual.index, expected.index, exact=False)
        np.testing.assert_allclose(actual[columns].to_numpy(dtype=float),
                                   expected[columns].to_numpy(dtype=float), rtol=1e-10, atol=1e-8)
        checks.append({"table": name, "rows": len(actual), "columns": columns, "status": "PASS"})

    reference = source / "results" / "tables"
    composition = pd.read_csv(reference / "mt_ribo_housekeeping_per_well.csv").rename(columns={
        "mt.pct": "pct_mito", "ribo.pct": "pct_ribo", "cytosolic_ribosomal_umi": "ribosomal_umi",
    })
    compare("composition", tables["wells"], composition, ["group", "barcode"], [
        "total_umi", "mitochondrial_umi", "ribosomal_umi", "pct_mito", "pct_ribo",
        "human_housekeeping_mean_umi", "mouse_housekeeping_mean_umi",
    ])
    saturation = pd.read_csv(reference / "saturation_per_well.csv").rename(columns={
        "cbMatch": "reads_per_well", "symbol_genes": "detected_gene_symbol",
        "entrez_genes": "detected_ensembl_id",
    })
    compare("saturation", tables["wells"], saturation, ["group", "barcode"],
            ["reads_per_well", "detected_gene_symbol", "detected_ensembl_id"])
    totals = pd.read_csv(reference / "total_umi_per_gene.csv").rename(columns={
        "Gene_ID": "gene_id", "Total_UMI": "total_umi",
    })
    compare("gene_totals", tables["gene_totals"], totals, ["group", "gene_id"], ["total_umi"])
    for panel in ("Human_MDA-MB-231_Markers", "Mouse_3T3_Markers"):
        marker_file = unique_file(reference, f"*_{panel}.csv")
        markers = pd.read_csv(marker_file).rename(columns={
            "cell_type": "baseline_species", "Gene": "gene", "Mean_Expression": "mean_umi",
        })
        compare(panel, tables["markers"].loc[tables["markers"].panel.eq(panel)], markers,
                ["group", "baseline_species", "gene"], ["mean_umi"])
    purity = pd.read_csv(reference / "barnyard_species_purity_statistics.csv").rename(columns={
        "cell_type": "baseline_species", "n": "n_wells", "min_r": "min", "max_r": "max",
        "med_r": "median", "mean_r": "mean",
    })
    compare("purity", tables["purity_statistics"], purity, ["group", "baseline_species"],
            ["n_wells", "min", "max", "median", "mean"])
    read_metrics = pd.read_csv(reference / "read_level_metrics.csv").rename(columns={
        "Ring": "metric", "Pct": "percent",
    })
    compare("read_metrics", tables["read_metrics"], read_metrics, ["group", "metric"], ["percent"])
    expected_wells = composition[["group", "barcode", "baseline_species"]].copy()
    expected_wells["group"] = expected_wells["group"].astype(str).map(sample_labels)
    pd.testing.assert_frame_equal(
        tables["wells"][expected_wells.columns].sort_values(["group", "barcode"]).reset_index(drop=True),
        expected_wells.sort_values(["group", "barcode"]).reset_index(drop=True),
        check_dtype=False,
    )
    assert len(tables["composition_values"]) == 3 * len(expected_wells)
    for name, table in tables.items():
        if name.startswith("plate_map_"):
            continuous = name not in {"plate_map_Human_Mouse", "plate_map_231_3T3_Water"}
            observed = table.loc[~table["is_missing_well"]].groupby("plate_id").size()
            expected = tables["wells" if continuous else "all_wells"].groupby("group").size()
            assert observed.to_dict() == expected.to_dict()
            checks.append({"table": name, "rows": len(table), "status": "PASS"})
    return checks


def unique_file(directory, pattern):
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {pattern!r} file in the reference directory.")
    return matches[0]


def anonymize_metadata(adata):
    """Keep required plotting fields and replace sample IDs before export.

    No mapping to the original labels is written to output files. This is
    sample-label anonymization; experimental counts and well barcodes remain.
    """
    groups = list(pd.unique(adata.obs["plate_id"].astype(str)))
    labels = {group: f"sample_{i + 1}" for i, group in enumerate(groups)}
    columns = ["plate_id", "well_id", "barcode", "reads_per_well", "in_filtered_matrix",
               "expected_cells", "expected_row", "expected_column"]
    obs = adata.obs[[col for col in columns if col in adata.obs]].copy()
    obs["plate_id"] = obs["plate_id"].astype(str).map(labels)
    ordinal = obs.groupby("plate_id", sort=False).cumcount() + 1
    obs.index = pd.Index(obs["plate_id"] + "_well_" + ordinal.astype(str), name="sample_id")
    adata.obs = obs
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline-group", help="Input group to use as baseline; defaults to the first group.")
    args = parser.parse_args()
    source, out = args.source.resolve(), args.output.resolve()
    if out == source or source in out.parents:
        parser.error("Use an output folder outside the source folder.")
    adata = ad.read_h5ad(unique_file(source / "results/plate_maps", "*.h5ad"))
    sample_labels = anonymize_metadata(adata)
    baseline = args.baseline_group or next(iter(sample_labels))
    if baseline not in sample_labels:
        parser.error("The requested baseline group is absent from the input metadata.")
    summary_files = {
        label: source / "inputs/star_results_UMIstart" / group / "library_seq_Solo.out/Gene/Summary.csv"
        for group, label in sample_labels.items()
    }
    result = ds.generate_combined_qc_report(
        adata, out, baseline_group=sample_labels[baseline], filtered_col="in_filtered_matrix",
        summary_files=summary_files, formats=("pdf", "png"),
    )
    checks = validate(result["tables"], source, sample_labels)
    figures = [p for p in result["files"] if p.endswith(".pdf")]
    assert len(figures) == 20
    report = {"status": "PASS", "samples": list(sample_labels.values()), "pdf_figures": len(figures),
              "checks": checks, "skipped": result["skipped"]}
    (out / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
