#!/usr/bin/env python3
"""Reproduce and validate the S6/S7 combined figure folder using drugseqpy.

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


def validate(tables, source):
    """Compare independently recomputed values against the original R exports."""
    checks = []

    def compare(name, actual, expected, keys, columns):
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
        markers = pd.read_csv(reference / f"Combined_S6_S7_{panel}.csv").rename(columns={
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
    assert tables["wells"].groupby("group").size().to_dict() == {"S6": 68, "S7": 68}
    assert tables["wells"].groupby(["group", "baseline_species"]).size().to_dict() == {
        ("S6", "human"): 33, ("S6", "mouse"): 35, ("S7", "human"): 33, ("S7", "mouse"): 35,
    }
    assert len(tables["composition_values"]) == 408
    for name, table in tables.items():
        if name.startswith("plate_map_"):
            continuous = name not in {"plate_map_Human_Mouse", "plate_map_231_3T3_Water"}
            assert len(table) == 192
            assert table["is_missing_well"].sum() == (56 if continuous else 0)
            checks.append({"table": name, "rows": len(table), "status": "PASS"})
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source, out = args.source.resolve(), args.output.resolve()
    if out == source or source in out.parents:
        parser.error("Use an output folder outside the source folder.")
    adata = ad.read_h5ad(source / "results/plate_maps/S6_S7_raw_96_wells.h5ad")
    summary_files = {
        group: source / "inputs/star_results_UMIstart" / group / "library_seq_Solo.out/Gene/Summary.csv"
        for group in ("S6", "S7")
    }
    result = ds.generate_combined_qc_report(
        adata, out, baseline_group="S6", filtered_col="in_filtered_matrix",
        summary_files=summary_files, formats=("pdf", "png"),
    )
    checks = validate(result["tables"], source)
    figures = [p for p in result["files"] if p.endswith(".pdf")]
    assert len(figures) == 20
    report = {"status": "PASS", "source": str(source), "pdf_figures": len(figures),
              "checks": checks, "skipped": result["skipped"]}
    (out / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
