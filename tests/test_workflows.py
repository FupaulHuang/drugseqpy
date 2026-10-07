import gzip
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

from drugseqpy.io import load_metadata, load_expression_matrix, create_drugseq_from_files
from drugseqpy.plate import build_plate_layout, complete_plate_layout, plot_plate_layout
from drugseqpy.mix_species import assign_well_species, compute_mix_species_qc
from drugseqpy.workflows import (
    generate_detected_gene_density_figure,
    generate_gene_umi_distribution_figures,
    generate_gene_set_violin_figure,
    generate_marker_gene_dotplot_figure,
    generate_marker_gene_figure,
    generate_marker_gene_violin_figure,
    generate_gene_summary_figures,
    generate_mix_species_barnyard_figure,
    generate_mix_species_contamination_figure,
    generate_mix_species_concordance_figure,
    generate_mix_species_fraction_figure,
    generate_mix_species_purity_figure,
    generate_mix_species_qc_metrics_figure,
    generate_mix_species_report,
    generate_outlier_score_scatter_figure,
    generate_plate_layout_figures,
    generate_qc_distribution_figures,
    generate_qc_metric_violin_figure,
    generate_qc_report,
    generate_star_read_qc_figure,
    generate_star_read_count_violin_figure,
    generate_star_saturation_figures,
    generate_star_qc_figures,
    generate_umi_density_figure,
)


def _metadata(ids):
    return pd.DataFrame({
        "plate_id": "F1",
        "barcode": ids,
        "well_id": ["A01", "A02", "B01", "B02"][:len(ids)],
        "compound": ["Drug1", "Drug1", "water", "Drug2"][:len(ids)],
        "dose": ["5 uM", "5 uM", np.nan, "10 uM"][:len(ids)],
        "time": ["24h", "24h", "24h", "48h"][:len(ids)],
        "x": [1, 2, 1, 2][:len(ids)],
        "y": [1, 1, 2, 2][:len(ids)],
    }, index=ids)


def _counts(ids):
    return pd.DataFrame(
        [[10, 20, 1, 5], [2, 3, 20, 1], [4, 4, 2, 6], [0, 1, 3, 0]],
        index=["ENSG1", "ENSMUSG1", "MT-G1", "G4"],
        columns=ids,
    )


def test_loaders_accept_csv_and_normalize_barcode_suffix(tmp_path):
    ids = ["S1", "S2", "S3", "S4"]
    matrix_path = tmp_path / "counts.tsv"
    metadata_path = tmp_path / "metadata.csv"
    _counts(ids).to_csv(matrix_path, sep="\t")
    _metadata(ids).to_csv(metadata_path)

    dsd = create_drugseq_from_files(matrix_path, metadata_path)
    assert dsd.n_obs == 4
    assert dsd.adata.layers["counts"].shape == (4, 4)
    assert dsd.obs["barcode"].tolist()[0] == "S1"
    assert dsd.obs["sample_id"].tolist()[0] == "F1_S1"
    assert dsd.obs["treatment"].tolist()[0] == "Drug1_5 uM_24h"


def test_metadata_txt_delimiter_is_auto_detected(tmp_path):
    metadata_path = tmp_path / "metadata.txt"
    _metadata(["S1", "S2", "S3", "S4"]).to_csv(
        metadata_path, sep=";", index=False,
    )

    metadata = load_metadata(metadata_path)

    assert metadata.index.tolist() == ["S1", "S2", "S3", "S4"]
    assert {"plate_id", "well_id", "compound"}.issubset(metadata.columns)


def test_explicit_metadata_delimiter_overrides_file_extension(tmp_path):
    ids = ["S1", "S2", "S3", "S4"]
    matrix_path = tmp_path / "counts.tsv"
    metadata_path = tmp_path / "metadata.csv"
    _counts(ids).to_csv(matrix_path, sep="\t")
    _metadata(ids).to_csv(metadata_path, sep="|", index=False)

    dsd = create_drugseq_from_files(
        matrix_path,
        metadata_path,
        delimiter="|",
    )

    assert dsd.n_obs == 4
    assert dsd.obs["sample_id"].tolist()[0] == "F1_S1"


def test_expression_directory_accepts_three_file_matrix_bundle(tmp_path):
    from scipy import sparse
    from scipy.io import mmwrite

    ids = ["S1", "S2", "S3", "S4"]
    bundle = tmp_path / "filtered"
    bundle.mkdir()
    counts = _counts(ids)
    matrix_plain = bundle / "matrix.mtx"
    mmwrite(matrix_plain, sparse.coo_matrix(counts.to_numpy()))
    with matrix_plain.open("rb") as source, gzip.open(
        bundle / "matrix.mtx.gz", "wb",
    ) as destination:
        destination.write(source.read())
    matrix_plain.unlink()
    pd.DataFrame({0: counts.index, 1: ["H1", "M1", "MT1", "G4"]}).to_csv(
        bundle / "features.tsv.gz",
        sep="\t",
        header=False,
        index=False,
        compression="gzip",
    )
    pd.DataFrame({0: [f"{sample}-1" for sample in ids]}).to_csv(
        bundle / "barcodes.tsv.gz",
        sep="\t",
        header=False,
        index=False,
        compression="gzip",
    )
    metadata_path = tmp_path / "metadata.txt"
    _metadata(ids).to_csv(metadata_path, sep=";", index=False)

    dsd = create_drugseq_from_files(bundle, metadata_path)

    assert dsd.shape == (4, 4)
    assert dsd.var["gene_symbol"].tolist() == ["H1", "M1", "MT1", "G4"]
    assert dsd.obs["barcode"].tolist() == ids


def test_plate_layout_uses_xy_and_returns_tidy_table(tmp_path):
    dsd = create_drugseq_from_files(
        tmp_path / "counts.tsv", tmp_path / "metadata.csv"
    ) if False else None
    obs = _metadata(["S1", "S2", "S3", "S4"])
    obs["metric"] = [1, 2, 3, 4]
    layout = build_plate_layout(obs, value_col="metric")
    assert set(["row", "column", "value", "well_id"]).issubset(layout.columns)
    assert layout["row"].max() == 2
    assert layout["column"].max() == 2


def test_plate_layout_maps_well_dt_column_major():
    obs = pd.DataFrame({
        "plate_id": "F1",
        "well_id": ["Well-dT-1", "Well-dT-2", "Well-dT-8",
                    "Well-dT-9", "Well-dT-16", "Well-dT-96"],
        "x": [1, 1, 1, 2, 2, 12],
        "y": [1, 2, 8, 1, 8, 8],
    })
    layout = build_plate_layout(obs)
    got = {
        row["well_id"]: (int(row["row"]), int(row["column"]))
        for _, row in layout.iterrows()
    }
    assert got["Well-dT-1"] == (1, 1)
    assert got["Well-dT-2"] == (2, 1)
    assert got["Well-dT-8"] == (8, 1)
    assert got["Well-dT-9"] == (1, 2)
    assert got["Well-dT-16"] == (8, 2)
    assert got["Well-dT-96"] == (8, 12)


def test_plate_layout_completes_missing_wells():
    obs = pd.DataFrame({
        "plate_id": "F1",
        "well_id": ["Well-dT-2", "Well-dT-9"],
        "total_well": [16, 16],
    })
    layout = complete_plate_layout(
        build_plate_layout(obs, nrow=8, ncol=2), nrow=8, ncol=2,
    )
    missing = layout[layout["is_missing_well"]]
    assert len(layout) == 16
    assert {"Well-dT-1", "Well-dT-16"}.issubset(set(missing["well_id"]))
    assert not bool(layout.loc[layout["well_id"] == "Well-dT-2", "is_missing_well"].iloc[0])


def test_mix_species_workflow_is_independent_and_calls_species(tmp_path):
    dsd = create_drugseq_from_files(
        tmp_path / "counts.tsv", tmp_path / "metadata.csv"
    ) if False else None
    from drugseqpy import create_drugseq_object
    dsd = create_drugseq_object(_counts(["S1", "S2", "S3", "S4"]), _metadata(["S1", "S2", "S3", "S4"]))
    result = compute_mix_species_qc(dsd, expected_col="compound")
    assert {"human_umi", "mouse_umi", "human_fraction", "observed_species"}.issubset(result.columns)
    assert result.loc["S1", "observed_species"] == "human"
    # S2 has 20 human-prefix counts versus 3 mouse-prefix counts in the
    # fixture, so the dominant-species classifier correctly calls it human.
    assert result.loc["S2", "observed_species"] == "human"
    assert "mix_species" not in dsd.obs.columns


def test_report_generators_write_figures_and_tables(tmp_path):
    from drugseqpy import create_drugseq_object
    dsd = create_drugseq_object(_counts(["S1", "S2", "S3", "S4"]), _metadata(["S1", "S2", "S3", "S4"]))
    qc_out = tmp_path / "qc"
    mix_out = tmp_path / "mix"
    qc_files = generate_qc_report(dsd, qc_out, formats=("png",))
    mix_files = generate_mix_species_report(dsd, mix_out, formats=("png",))
    assert any(Path(p).suffix == ".png" for p in qc_files)
    assert (qc_out / "sample_qc_metrics.csv").exists()
    assert any(Path(p).suffix == ".png" for p in mix_files)
    assert (mix_out / "per_sample_species_qc.csv").exists()
    mix_stems = {Path(p).stem for p in mix_files}
    for expected in ["species_barnyard", "species_purity",
                     "species_concordance_counts", "mix_species_qc_metrics",
                     "species_fraction_by_expected",
                     "species_cross_contamination_by_base"]:
        assert expected in mix_stems, f"Missing figure '{expected}' in mix-species report"
    for expected in ["species_summary.csv", "species_concordance.csv",
                     "mix_species_outlier_audit.csv",
                     "species_barnyard_metric_summary.csv",
                     "species_fraction_plot_table.csv",
                     "species_cross_contamination_by_base_plot_table.csv",
                     "mix_species_figure_manifest.csv"]:
        assert (mix_out / expected).exists(), f"Missing table '{expected}' in mix-species report"
    mix_manifest = pd.read_csv(mix_out / "mix_species_figure_manifest.csv")
    assert set(mix_manifest.columns) == {"figure", "producer"}
    assert set(mix_manifest["producer"]) == {
        "generate_mix_species_barnyard_figure",
        "generate_mix_species_purity_figure",
        "generate_mix_species_concordance_figure",
        "generate_mix_species_qc_metrics_figure",
        "generate_mix_species_fraction_figure",
        "generate_mix_species_contamination_figure",
    }
    qc_stems = {Path(p).stem for p in qc_files}
    # Default plate set: unlabeled metric/compound layouts plus the labeled
    # well_id layout and the QC violin/density figures.
    for expected in ["plate_layout_detected_umi", "plate_layout_detected_genes",
                     "plate_layout_compound", "plate_layout_well_id",
                     "F1_QC_Boxplots", "F1_Outlier_Score_Scatter",
                     "F1_UMI_Density"]:
        assert expected in qc_stems, f"Missing figure '{expected}' in QC report"
    manifest = pd.read_csv(qc_out / "qc_figure_manifest.csv")
    assert set(manifest.columns) == {"figure", "producer"}
    assert set(manifest["producer"]) == {
        "generate_plate_layout_figures",
        "generate_qc_metric_violin_figure",
        "generate_outlier_score_scatter_figure",
        "generate_umi_density_figure",
        "generate_detected_gene_density_figure",
        "generate_gene_umi_distribution_figures",
    }


def test_mix_species_figure_types_are_independently_callable(tmp_path):
    from drugseqpy import create_drugseq_object

    dsd = create_drugseq_object(
        _counts(["S1", "S2", "S3", "S4"]),
        _metadata(["S1", "S2", "S3", "S4"]),
    )
    generators = [
        (generate_mix_species_barnyard_figure, "species_barnyard"),
        (generate_mix_species_purity_figure, "species_purity"),
        (generate_mix_species_concordance_figure, "species_concordance_counts"),
        (generate_mix_species_qc_metrics_figure, "mix_species_qc_metrics"),
        (generate_mix_species_fraction_figure, "species_fraction_by_expected"),
        (
            generate_mix_species_contamination_figure,
            "species_cross_contamination_by_base",
        ),
    ]
    for generator, expected_stem in generators:
        files = generator(dsd, tmp_path / expected_stem, formats=("png",))
        assert {Path(path).stem for path in files} == {expected_stem}
        assert all(Path(path).exists() for path in files)


def test_two_species_configuration_and_explicit_well_assignment():
    from drugseqpy import create_drugseq_object

    ids = ["S1", "S2", "S3", "S4"]
    dsd = create_drugseq_object(_counts(ids), _metadata(ids))
    copied = assign_well_species(
        dsd,
        source_col="compound",
        mapping={"Drug1": "primate", "Drug2": "rodent", "water": "control"},
        output_col="designed_species",
        inplace=False,
    )
    assert "designed_species" not in dsd.obs
    assert copied.obs["designed_species"].tolist() == [
        "primate", "primate", "control", "rodent",
    ]

    result = compute_mix_species_qc(
        copied,
        species_a_prefix="ENSG",
        species_b_prefix="ENSMUSG",
        species_a_name="primate",
        species_b_name="rodent",
        expected_col="designed_species",
    )
    assert result.attrs["species_config"]["species_a_name"] == "primate"
    assert result.loc["S1", "observed_species"] == "primate"
    assert result.loc["S1", "base_species"] == "primate"
    assert result.loc["S1", "cross_contamination_fraction"] == pytest.approx(2 / 12)
    assert result.loc["S4", "base_species"] == "rodent"
    assert result.loc["S4", "cross_contamination_fraction"] == pytest.approx(5 / 6)
    assert "human_umi" not in result
    assert "mouse_umi" not in result


def test_contamination_figure_shows_both_base_species(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    import drugseqpy.workflows as workflows
    from drugseqpy import create_drugseq_object

    ids = ["S1", "S2", "S3", "S4"]
    dsd = create_drugseq_object(_counts(ids), _metadata(ids))
    assign_well_species(
        dsd,
        source_col="compound",
        mapping={"Drug1": "human", "Drug2": "mouse", "water": "control"},
    )
    captured = {}

    def capture_figure(fig, *_args, **_kwargs):
        captured["figure"] = fig
        return []

    monkeypatch.setattr(workflows, "_save", capture_figure)
    workflows.generate_mix_species_contamination_figure(
        dsd, tmp_path, formats=("png",), expected_col="expected_species",
    )
    titles = [ax.get_title() for ax in captured["figure"].axes]
    assert any("Base species: human" in title for title in titles)
    assert any("Base species: mouse" in title for title in titles)
    plt.close(captured["figure"])


def test_mix_species_qc_medians_are_above_each_violin(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    import drugseqpy.workflows as workflows
    from drugseqpy import create_drugseq_object

    ids = ["S1", "S2", "S3", "S4"]
    obs = _metadata(ids)
    obs.loc[["S3", "S4"], "plate_id"] = "F2"
    dsd = create_drugseq_object(_counts(ids), obs)
    assign_well_species(
        dsd,
        source_col="compound",
        mapping={"Drug1": "human", "Drug2": "mouse", "water": "control"},
    )
    captured = {}

    def capture_figure(fig, *_args, **_kwargs):
        captured["figure"] = fig
        return []

    monkeypatch.setattr(workflows, "_save", capture_figure)
    workflows.generate_mix_species_qc_metrics_figure(
        dsd,
        tmp_path,
        formats=("png",),
        expected_col="expected_species",
    )
    plot_table = pd.read_csv(
        tmp_path / "mix_species_qc_metrics_plot_table.csv"
    )
    for ax in captured["figure"].axes:
        metric = ax.get_title()
        if not ax.get_visible() or metric not in plot_table:
            continue
        maxima = (
            plot_table[["plate_id", metric]]
            .dropna()
            .groupby("plate_id")[metric]
            .max()
            .sort_index()
        )
        labels = [
            text for text in ax.texts
            if text.get_text().startswith("Median:")
        ]
        assert len(labels) == len(maxima)
        for text, maximum in zip(labels, maxima):
            assert text.get_position()[1] > maximum
    plt.close(captured["figure"])


def test_mix_species_barnyard_panels_are_square(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    import drugseqpy.workflows as workflows
    from drugseqpy import create_drugseq_object

    ids = ["S1", "S2", "S3", "S4"]
    obs = _metadata(ids)
    obs.loc[["S3", "S4"], "plate_id"] = "F2"
    dsd = create_drugseq_object(_counts(ids), obs)
    captured = {}

    def capture_figure(fig, *_args, **_kwargs):
        captured["figure"] = fig
        return []

    monkeypatch.setattr(workflows, "_save", capture_figure)
    workflows.generate_mix_species_barnyard_figure(
        dsd, tmp_path, formats=("png",), nrow_plate=2,
    )
    visible_axes = [ax for ax in captured["figure"].axes if ax.get_visible()]
    assert visible_axes
    assert all(ax.get_box_aspect() == pytest.approx(1.0) for ax in visible_axes)
    assert all(ax.get_xlim() == pytest.approx(ax.get_ylim()) for ax in visible_axes)
    plt.close(captured["figure"])


def test_mix_species_purity_panels_are_square(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    import drugseqpy.workflows as workflows
    from drugseqpy import create_drugseq_object

    ids = ["S1", "S2", "S3", "S4"]
    obs = _metadata(ids)
    obs.loc[["S3", "S4"], "plate_id"] = "F2"
    dsd = create_drugseq_object(_counts(ids), obs)
    captured = {}

    def capture_figure(fig, *_args, **_kwargs):
        captured["figure"] = fig
        return []

    monkeypatch.setattr(workflows, "_save", capture_figure)
    workflows.generate_mix_species_purity_figure(
        dsd, tmp_path, formats=("png",), nrow_plate=2,
    )
    visible_axes = [ax for ax in captured["figure"].axes if ax.get_visible()]
    assert visible_axes
    assert all(ax.get_box_aspect() == pytest.approx(1.0) for ax in visible_axes)
    assert all(ax.get_xlim() == pytest.approx((0, 100)) for ax in visible_axes)
    assert all(ax.get_ylim() == pytest.approx((0, 100)) for ax in visible_axes)
    plt.close(captured["figure"])


def test_barnyard_metric_species_selects_one_both_or_none(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    import drugseqpy.workflows as workflows
    from drugseqpy import create_drugseq_object

    ids = ["S1", "S2", "S3", "S4"]
    dsd = create_drugseq_object(_counts(ids), _metadata(ids))
    assign_well_species(
        dsd,
        source_col="compound",
        mapping={"Drug1": "human", "Drug2": "mouse", "water": "control"},
    )
    captured = {}

    def capture_figure(fig, *_args, **_kwargs):
        captured["figure"] = fig
        return []

    monkeypatch.setattr(workflows, "_save", capture_figure)

    workflows.generate_mix_species_barnyard_figure(
        dsd, tmp_path / "base", formats=("png",),
        expected_col="expected_species",
    )
    annotation = "\n".join(
        text.get_text() for text in captured["figure"].axes[0].texts
    )
    assert "Base human purity" in annotation
    assert "Base mouse purity" in annotation
    assert all(name in annotation for name in ["min", "mean", "median", "max"])
    summary = pd.read_csv(
        tmp_path / "base" / "species_barnyard_metric_summary.csv"
    )
    assert set(summary["base_species"]) == {"human", "mouse"}
    assert {"minimum", "mean", "median", "maximum"}.issubset(summary.columns)
    plt.close(captured["figure"])

    workflows.generate_mix_species_barnyard_figure(
        dsd, tmp_path / "one", formats=("png",),
        expected_col="expected_species", metric_species="human",
    )
    annotation = "\n".join(
        text.get_text() for text in captured["figure"].axes[0].texts
    )
    assert "Base human purity" in annotation
    assert "Base mouse purity" not in annotation
    plt.close(captured["figure"])

    workflows.generate_mix_species_barnyard_figure(
        dsd, tmp_path / "none", formats=("png",),
        expected_col="expected_species", metric_species=None,
    )
    assert not any(
        "Base " in text.get_text()
        for text in captured["figure"].axes[0].texts
    )
    plt.close(captured["figure"])


def test_gene_set_violin_uses_plate_gene_resolution_and_writes_sources(
    tmp_path, monkeypatch,
):
    import matplotlib.pyplot as plt
    import drugseqpy.workflows as workflows
    from drugseqpy import create_drugseq_object

    ids = ["S1", "S2", "S3", "S4"]
    dsd = create_drugseq_object(_counts(ids), _metadata(ids))
    captured = {}

    def capture_figure(fig, *_args, **_kwargs):
        captured["figure"] = fig
        return [str(tmp_path / "species_gene_sets.png")]

    monkeypatch.setattr(workflows, "_save", capture_figure)
    files = generate_gene_set_violin_figure(
        dsd,
        tmp_path,
        gene_sets={
            "Human-specific set": ["ENSG1"],
            "Mouse-specific set": ["ENSMUSG1"],
        },
        gene_aggregation="mean",
        formats=("png",),
        groupby="compound",
        log_values=False,
        output_name="species_gene_sets",
    )
    assert files == [str(tmp_path / "species_gene_sets.png")]
    plot_table = pd.read_csv(tmp_path / "species_gene_sets_plot_table.csv")
    resolution = pd.read_csv(
        tmp_path / "species_gene_sets_gene_resolution.csv"
    )
    assert len(plot_table) == 2 * dsd.n_obs
    assert set(plot_table["gene_set"]) == {
        "Human-specific set", "Mouse-specific set",
    }
    assert set(resolution["requested_gene"]) == {"ENSG1", "ENSMUSG1"}
    assert set(resolution["matched_by"]) == {"both"}
    assert (tmp_path / "species_gene_sets_per_gene_expression.csv").exists()

    for ax in captured["figure"].axes:
        if not ax.get_visible():
            continue
        subset = plot_table[plot_table["gene_set"] == ax.get_title()]
        maxima = subset.groupby("compound")["plot_expression"].max().sort_index()
        labels = [
            text for text in ax.texts
            if text.get_text().startswith("Median:")
        ]
        assert len(labels) == len(maxima)
        for text, maximum in zip(labels, maxima):
            assert text.get_position()[1] > maximum
    plt.close(captured["figure"])


def test_qc_figure_groups_are_independently_callable(tmp_path):
    from drugseqpy import create_drugseq_object
    dsd = create_drugseq_object(
        _counts(["S1", "S2", "S3", "S4"]),
        _metadata(["S1", "S2", "S3", "S4"]),
    )

    plate_files = []
    plate_files += generate_plate_layout_figures(
        dsd, tmp_path / "plate", formats=("png",),
        plate_value="total_umi", continuous=True,
        output_name="plate_layout_detected_umi",
    )
    plate_files += generate_plate_layout_figures(
        dsd, tmp_path / "plate", formats=("png",),
        plate_value="n_genes_det", continuous=True,
        output_name="plate_layout_detected_genes",
    )
    plate_files += generate_plate_layout_figures(
        dsd, tmp_path / "plate", formats=("png",),
        plate_value="compound", output_name="plate_layout_compound",
    )
    plate_files += generate_plate_layout_figures(
        dsd, tmp_path / "plate", formats=("png",),
        well_color="white", label_col="well_id", show_legend=False,
        output_name="plate_layout_well_id",
    )
    distribution_files = generate_qc_distribution_figures(
        dsd, tmp_path / "distribution", formats=("png",),
    )
    violin_files = generate_qc_metric_violin_figure(
        dsd, tmp_path / "focused", formats=("png",),
    )
    outlier_files = generate_outlier_score_scatter_figure(
        dsd, tmp_path / "focused", formats=("png",),
    )
    umi_density_files = generate_umi_density_figure(
        dsd, tmp_path / "focused", formats=("png",),
    )
    gene_density_files = generate_detected_gene_density_figure(
        dsd, tmp_path / "focused", formats=("png",),
    )
    gene_files = generate_gene_summary_figures(
        dsd, tmp_path / "gene", formats=("png",), marker_genes=[],
    )

    assert {Path(path).stem for path in plate_files} == {
        "plate_layout_detected_umi",
        "plate_layout_detected_genes",
        "plate_layout_compound",
        "plate_layout_well_id",
    }
    assert {Path(path).stem for path in distribution_files} == {
        "F1_QC_Boxplots", "F1_Outlier_Score_Scatter",
        "F1_UMI_Density", "F1_Gene_Density",
    }
    assert {Path(path).stem for path in violin_files} == {"F1_QC_Boxplots"}
    assert {Path(path).stem for path in outlier_files} == {
        "F1_Outlier_Score_Scatter"
    }
    assert {Path(path).stem for path in umi_density_files} == {"F1_UMI_Density"}
    assert {Path(path).stem for path in gene_density_files} == {"F1_Gene_Density"}
    assert {Path(path).stem for path in gene_files} == {
        "F1_UMI_per_Entrez_ID", "F1_UMI_per_Gene_Symbol",
    }
    assert (tmp_path / "plate" / "plate_layout_coordinates.csv").exists()
    assert (tmp_path / "gene" / "gene_umi_summary.csv").exists()
    assert (tmp_path / "focused" / "outlier_score_plot_table.csv").exists()


def test_plate_layout_generator_requires_one_color_source(tmp_path):
    from drugseqpy import create_drugseq_object

    dsd = create_drugseq_object(
        _counts(["S1", "S2", "S3", "S4"]),
        _metadata(["S1", "S2", "S3", "S4"]),
    )
    with pytest.raises(ValueError, match="Exactly one"):
        generate_plate_layout_figures(dsd, tmp_path)
    with pytest.raises(ValueError, match="Exactly one"):
        generate_plate_layout_figures(
            dsd, tmp_path, plate_value="compound", well_color="white",
        )


def test_plate_layout_supports_384_well_dimensions():
    obs = pd.DataFrame({
        "plate_id": ["P384", "P384", "P384"],
        "well_id": ["Well-dT-1", "Well-dT-16", "Well-dT-384"],
    })
    layout = build_plate_layout(obs, nrow=16, ncol=24)
    got = {
        row["well_id"]: (int(row["row"]), int(row["column"]))
        for _, row in layout.iterrows()
    }
    assert got["Well-dT-1"] == (1, 1)
    assert got["Well-dT-16"] == (16, 1)
    assert got["Well-dT-384"] == (16, 24)
    completed = complete_plate_layout(layout, nrow=16, ncol=24)
    assert len(completed) == 384


def test_plate_layout_label_style_and_default_are_configurable():
    import inspect
    import matplotlib.pyplot as plt

    assert inspect.signature(generate_plate_layout_figures).parameters[
        "label_col"
    ].default is None
    obs = pd.DataFrame({
        "plate_id": ["P1", "P1"],
        "well_id": ["A1", "B2"],
    })
    fig, layout = plot_plate_layout(
        obs,
        well_color="white",
        label_col="well_id",
        nrow=2,
        ncol=2,
        label_color="#DC2626",
        label_size=9,
        label_style="italic",
        label_weight="bold",
    )
    labels = fig.axes[0].texts
    assert {label.get_text() for label in labels} == {"A1", "B2"}
    assert all(label.get_color() == "#DC2626" for label in labels)
    assert all(label.get_fontsize() == 9 for label in labels)
    assert all(label.get_fontstyle() == "italic" for label in labels)
    assert layout["is_missing_well"].sum() == 2
    plt.close(fig)


def test_plate_layout_log_border_panel_rows_and_plate_id_column():
    import inspect
    import matplotlib.pyplot as plt

    signature = inspect.signature(generate_plate_layout_figures)
    assert signature.parameters["log_values"].default is True
    assert signature.parameters["well_border"].default is True
    assert signature.parameters["cmap"].default == "RdYlBu_r"
    assert signature.parameters["plate_id"].default == "plate_id"
    assert signature.parameters["split_by"].default is None

    obs = pd.DataFrame({
        "source_plate": ["P1", "P1", "P2", "P2"],
        "well_id": ["A1", "B2", "A1", "B2"],
        "metric": [0.0, 9.0, 99.0, 999.0],
        "split_group": ["A", "B", "A", "B"],
    })
    fig, layout = plot_plate_layout(
        obs,
        value_col="metric",
        plate_id="source_plate",
        continuous=True,
        nrow=2,
        ncol=2,
        nrow_plate=1,
    )
    observed = layout.loc[~layout["is_missing_well"]]
    np.testing.assert_allclose(
        observed["plot_value"], np.log1p(observed["value"].astype(float)),
    )
    assert set(observed["plate_id"]) == {"P1", "P2"}
    visible_axes = [axis for axis in fig.axes if axis.get_visible()]
    assert visible_axes[0].get_position().y0 > visible_axes[1].get_position().y0
    plt.close(fig)

    split_fig, split_layout = plot_plate_layout(
        obs,
        value_col="metric",
        plate_id="source_plate",
        split_by="split_group",
        continuous=True,
        nrow=2,
        ncol=2,
        nrow_plate=2,
    )
    split_observed = split_layout.loc[~split_layout["is_missing_well"]]
    assert split_observed["plate_id"].nunique() == 4
    assert split_observed["physical_plate_id"].nunique() == 2
    plt.close(split_fig)


def test_outlier_scatter_draws_requested_labels(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    import drugseqpy.workflows as workflows
    from drugseqpy import create_drugseq_object

    dsd = create_drugseq_object(
        _counts(["S1", "S2", "S3", "S4"]),
        _metadata(["S1", "S2", "S3", "S4"]),
    )
    captured = {}

    def capture_figure(fig, *_args, **_kwargs):
        captured["figure"] = fig
        return []

    monkeypatch.setattr(workflows, "_save", capture_figure)
    workflows.generate_outlier_score_scatter_figure(
        dsd,
        tmp_path,
        formats=("png",),
        label_outliers=True,
        label_col="well_id",
        max_labels=2,
    )
    labels = {
        text.get_text() for text in captured["figure"].axes[0].texts
        if text.get_text() in set(dsd.obs["well_id"].astype(str))
    }
    assert len(labels) == 2
    plt.close(captured["figure"])


@pytest.mark.parametrize(
    ("aggregation", "expected"),
    [
        ("mean", [16 / 3, 9.0, 23 / 3, 4.0]),
        ("sum", [16.0, 27.0, 23.0, 12.0]),
        ("median", [4.0, 4.0, 2.0, 5.0]),
    ],
)
def test_plate_gene_set_expression_aggregation(tmp_path, aggregation, expected):
    from drugseqpy import create_drugseq_object

    dsd = create_drugseq_object(
        _counts(["S1", "S2", "S3", "S4"]),
        _metadata(["S1", "S2", "S3", "S4"]),
    )
    stem = f"geneset_{aggregation}"
    files = generate_plate_layout_figures(
        dsd,
        tmp_path,
        formats=("png",),
        plate_gene=["ENSG1", "ENSMUSG1", "MT-G1"],
        gene_aggregation=aggregation,
        continuous=True,
        log_values=False,
        nrow=2,
        ncol=2,
        output_name=stem,
    )
    assert {Path(path).stem for path in files} == {stem}
    table = pd.read_csv(tmp_path / f"{stem}_plot_table.csv")
    index_col = "metadata_index" if "metadata_index" in table else "sample_id"
    observed = table.loc[~table["is_missing_well"]].set_index(index_col)
    np.testing.assert_allclose(
        observed.loc[["S1", "S2", "S3", "S4"], "value"], expected,
    )
    resolution = pd.read_csv(tmp_path / f"{stem}_gene_resolution.csv")
    assert set(resolution["requested_gene"]) == {
        "ENSG1", "ENSMUSG1", "MT-G1",
    }
    per_gene = pd.read_csv(tmp_path / f"{stem}_per_gene_expression.csv")
    assert {"ENSG1", "ENSMUSG1", "MT-G1", "aggregate_expression"}.issubset(
        per_gene.columns
    )


def test_plate_single_gene_expression_and_missing_gene_validation(tmp_path):
    from drugseqpy import create_drugseq_object

    dsd = create_drugseq_object(
        _counts(["S1", "S2", "S3", "S4"]),
        _metadata(["S1", "S2", "S3", "S4"]),
    )
    generate_plate_layout_figures(
        dsd,
        tmp_path,
        formats=("png",),
        plate_gene="ENSG1",
        continuous=True,
        log_values=False,
        nrow=2,
        ncol=2,
        output_name="single_gene",
    )
    table = pd.read_csv(tmp_path / "single_gene_plot_table.csv")
    index_col = "metadata_index" if "metadata_index" in table else "sample_id"
    observed = table.loc[~table["is_missing_well"]].set_index(index_col)
    np.testing.assert_allclose(
        observed.loc[["S1", "S2", "S3", "S4"], "value"],
        [10.0, 20.0, 1.0, 5.0],
    )
    with pytest.raises(KeyError, match="NOT_A_GENE"):
        generate_plate_layout_figures(
            dsd,
            tmp_path,
            formats=("png",),
            plate_gene=["ENSG1", "NOT_A_GENE"],
            continuous=True,
            nrow=2,
            ncol=2,
        )


def test_qc_workflow_functions_are_exported_from_package_root():
    import drugseqpy as ds

    for name in [
        "merge_drugseq_objects",
        "compute_qc_metrics",
        "generate_plate_layout_figures",
        "generate_qc_metric_violin_figure",
        "generate_outlier_score_scatter_figure",
        "generate_umi_density_figure",
        "generate_detected_gene_density_figure",
        "generate_gene_umi_distribution_figures",
        "generate_marker_gene_figure",
        "generate_marker_gene_dotplot_figure",
        "generate_marker_gene_violin_figure",
        "generate_gene_summary_figures",
        "generate_star_qc_figures",
        "generate_star_read_qc_figure",
        "generate_star_read_count_violin_figure",
        "generate_star_saturation_figures",
        "generate_qc_report",
    ]:
        assert callable(getattr(ds, name))


def test_focused_gene_summary_functions_and_identifier_selection(tmp_path):
    from drugseqpy import create_drugseq_object

    dsd = create_drugseq_object(
        _counts(["S1", "S2", "S3", "S4"]),
        _metadata(["S1", "S2", "S3", "S4"]),
    )
    dsd.var["gene_symbol"] = ["VIM", "CD44", "MT-G1", "G4"]

    ensembl_files = generate_gene_umi_distribution_figures(
        dsd,
        tmp_path / "gene",
        formats=("png",),
        identifier_types="ensembl_id",
    )
    assert {Path(path).stem for path in ensembl_files} == {
        "F1_UMI_per_Entrez_ID",
    }
    both_files = generate_gene_umi_distribution_figures(
        dsd,
        tmp_path / "gene_both",
        formats=("png",),
    )
    assert {Path(path).stem for path in both_files} == {
        "F1_UMI_per_Entrez_ID", "F1_UMI_per_Gene_Symbol",
    }
    grouped_gene_table = pd.read_csv(
        tmp_path / "gene_both" / "gene_umi_distribution_plot_table.csv"
    )
    assert {"plate_id", "identifier_type", "log2_total_umi"}.issubset(
        grouped_gene_table.columns
    )
    assert grouped_gene_table["plate_id"].eq("F1").all()
    marker_files = generate_marker_gene_figure(
        dsd,
        tmp_path / "marker",
        marker_genes=["VIM", "CD44"],
        formats=("png",),
        groupby="compound",
    )
    assert {Path(path).stem for path in marker_files} == {
        "F1_Human_MDA-MB-231_Markers",
    }
    availability = pd.read_csv(tmp_path / "marker" / "marker_gene_summary.csv")
    assert availability["requested"].tolist() == ["VIM", "CD44"]
    assert availability["available"].all()
    marker_table = pd.read_csv(
        tmp_path / "marker" / "marker_expression_plot_table.csv"
    )
    assert "compound" in marker_table.columns


def test_marker_dotplot_and_violin_group_treatments_and_split_plates(tmp_path):
    from drugseqpy import create_drugseq_object

    ids = [f"S{index}" for index in range(1, 9)]
    counts = pd.DataFrame(
        [
            [8, 1, 7, 2, 9, 1, 8, 2],
            [1, 6, 2, 7, 1, 8, 2, 6],
            [4, 2, 5, 1, 6, 2, 5, 1],
            [0, 1, 0, 2, 0, 1, 0, 2],
        ],
        index=["ENSG1", "ENSG2", "ENSG3", "ENSG4"],
        columns=ids,
    )
    obs = pd.DataFrame({
        "plate_id": ["F3"] * 4 + ["F4"] * 4,
        "barcode": ids,
        "well_id": [f"A{index:02d}" for index in range(1, 9)],
        "compound": ["231", "3T3", "231", "3T3"] * 2,
    }, index=ids)
    dsd = create_drugseq_object(counts, obs)
    dsd.var["gene_symbol"] = ["VIM", "FN1", "CDH2", "CD44"]

    dot_files = generate_marker_gene_dotplot_figure(
        dsd,
        tmp_path,
        marker_genes=["VIM", "FN1", "CDH2", "CD44"],
        formats=("png",),
        groupby="compound",
        split_by="plate_id",
        nrow_plate=2,
    )
    violin_files = generate_marker_gene_violin_figure(
        dsd,
        tmp_path,
        marker_genes=["VIM", "FN1", "CDH2", "CD44"],
        formats=("png",),
        groupby="compound",
        split_by="plate_id",
        nrow_plate=2,
    )

    assert {Path(path).stem for path in dot_files} == {
        "Combined_F3_F4_Human_MDA-MB-231_Markers_Dotplot_split_by_plate_id"
    }
    assert {Path(path).stem for path in violin_files} == {
        "Combined_F3_F4_Human_MDA-MB-231_Markers_Violin_split_by_plate_id"
    }
    dot_table = pd.read_csv(tmp_path / "marker_dotplot_table.csv")
    assert set(dot_table["compound"]) == {"231", "3T3"}
    assert set(dot_table["plate_id"]) == {"F3", "F4"}
    assert {"mean_expression", "pct_expressing", "n_samples"}.issubset(
        dot_table.columns
    )
    violin_table = pd.read_csv(tmp_path / "marker_violin_plot_table.csv")
    assert set(violin_table["compound"]) == {"231", "3T3"}
    assert set(violin_table["plate_id"]) == {"F3", "F4"}
    assert {"gene", "expression", "plot_expression"}.issubset(
        violin_table.columns
    )


def test_plate_figure_does_not_embed_empty_well_note():
    import matplotlib.pyplot as plt

    obs = pd.DataFrame({
        "plate_id": ["P1"],
        "well_id": ["A1"],
        "metric": [1.0],
    })
    fig, _ = plot_plate_layout(
        obs,
        value_col="metric",
        continuous=True,
        nrow=2,
        ncol=2,
    )
    assert "White wells represent empty/missing wells." not in {
        text.get_text() for text in fig.texts
    }
    plt.close(fig)


def test_qc_figures_accept_custom_groupby_and_palette(tmp_path):
    from drugseqpy import create_drugseq_object

    dsd = create_drugseq_object(
        _counts(["S1", "S2", "S3", "S4"]),
        _metadata(["S1", "S2", "S3", "S4"]),
    )
    palette = {
        "Drug1": "#3B82F6",
        "Drug2": "#F97316",
        "water": "#94A3B8",
        "F1": "#14B8A6",
    }

    violin_files = generate_qc_metric_violin_figure(
        dsd,
        tmp_path / "custom",
        formats=("png",),
        metrics=["total_umi", "n_genes_det"],
        groupby="compound",
        palette=palette,
    )
    outlier_files = generate_outlier_score_scatter_figure(
        dsd,
        tmp_path / "custom",
        formats=("png",),
        groupby="compound",
        palette=palette,
        threshold=5,
        label_outliers=True,
        max_labels=2,
        output_name="outlier_by_compound",
    )
    umi_files = generate_umi_density_figure(
        dsd,
        tmp_path / "custom",
        formats=("png",),
        groupby="compound",
        palette="Set2",
    )
    gene_files = generate_detected_gene_density_figure(
        dsd,
        tmp_path / "custom",
        formats=("png",),
        groupby="compound",
        palette=["#3B82F6", "#F97316", "#94A3B8"],
    )
    assert all(
        Path(path).exists()
        for path in violin_files + outlier_files + umi_files + gene_files
    )
    assert {Path(path).stem for path in outlier_files} == {
        "outlier_by_compound"
    }
    outlier_table = pd.read_csv(
        tmp_path / "custom" / "outlier_by_compound_plot_table.csv"
    )
    assert {"sample_order", "compound", "outlier_score", "above_threshold"}.issubset(
        outlier_table.columns
    )

    with pytest.raises(KeyError, match="groupby='not_a_column'"):
        generate_qc_metric_violin_figure(
            dsd,
            tmp_path / "invalid",
            formats=("png",),
            groupby="not_a_column",
        )
    with pytest.raises(ValueError, match="at least one"):
        generate_qc_metric_violin_figure(
            dsd, tmp_path / "invalid", formats=("png",), metrics=[],
        )


def test_qc_violin_median_labels_are_above_violin_data(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    import drugseqpy.workflows as workflows
    from drugseqpy import create_drugseq_object

    dsd = create_drugseq_object(
        _counts(["S1", "S2", "S3", "S4"]),
        _metadata(["S1", "S2", "S3", "S4"]),
    )
    captured = {}

    def capture_figure(fig, *_args, **_kwargs):
        captured["figure"] = fig
        return []

    monkeypatch.setattr(workflows, "_save", capture_figure)
    workflows.generate_qc_metric_violin_figure(
        dsd,
        tmp_path,
        formats=("png",),
        metrics=["n_genes_det"],
        groupby="compound",
    )

    ax = captured["figure"].axes[0]
    order = sorted(dsd.obs["compound"].astype(str).unique())
    maxima = dsd.obs.groupby("compound")["n_genes_det"].max()
    median_labels = [text for text in ax.texts if text.get_text().startswith("Median:")]
    assert len(median_labels) == len(order)
    for text, group in zip(median_labels, order):
        assert text.get_position()[1] > maxima[group]
    plt.close(captured["figure"])


def test_report_level_palette_and_cmap_are_configurable(tmp_path):
    from drugseqpy import create_drugseq_object

    dsd = create_drugseq_object(
        _counts(["S1", "S2", "S3", "S4"]),
        _metadata(["S1", "S2", "S3", "S4"]),
    )
    files = generate_qc_report(
        dsd,
        tmp_path / "qc",
        formats=("png",),
        marker_genes=[],
        violin_metrics=["total_umi", "n_genes_det"],
        groupby="compound",
        palette="Set2",
        cmap="viridis",
        outlier_threshold=5,
        label_outliers=True,
        max_outlier_labels=2,
    )
    assert files and all(Path(path).exists() for path in files)

    mix_files = generate_mix_species_report(
        dsd,
        tmp_path / "mix",
        formats=("png",),
        expected_palette={
            "Drug1": "#16B4D1", "Drug2": "#1B2A40", "water": "#64748B",
        },
        observed_palette={
            "human": "#16B4D1", "mouse": "#1B2A40",
            "mixed": "#F59E0B", "undetermined": "#CBD5E1",
        },
        plate_palette={"F1": "#16B4D1"},
        reference_line_color="#777777",
        guide_line_color="#CCCCCC",
        annotation_color="#C62828",
        edge_color="#FFFFFF",
        grid_color="#DCE6F2",
        panel_border_color="#334155",
        point_color="#253746",
        label_color="#FFFFFF",
        point_alpha=.7,
        fill_alpha=.9,
        show_observed_style=False,
    )
    assert mix_files and all(Path(path).exists() for path in mix_files)


def test_qc_report_uses_observed_plate_labels(tmp_path):
    from drugseqpy import create_drugseq_object
    ids = ["S1", "S2", "S3", "S4"]
    obs = _metadata(ids)
    obs.loc[["S3", "S4"], "plate_id"] = "F2"
    dsd = create_drugseq_object(_counts(ids), obs)

    qc_files = generate_qc_report(
        dsd, tmp_path / "qc", formats=("png",), marker_genes=[],
    )

    qc_stems = {Path(path).stem for path in qc_files}
    assert "Combined_F1_F2_QC_Boxplots" in qc_stems
    assert "Combined_F1_F2_UMI_Density" in qc_stems


def test_qc_report_star_summaries_use_observed_plates(tmp_path):
    from drugseqpy import create_drugseq_object
    ids = ["S1", "S2", "S3", "S4"]
    obs = _metadata(ids)
    obs.loc[["S3", "S4"], "plate_id"] = "F2"
    dsd = create_drugseq_object(_counts(ids), obs)
    star_root = tmp_path / "star"
    for plate in ("F1", "F2"):
        gene_dir = star_root / plate / "library_seq_Solo.out" / "Gene"
        gene_dir.mkdir(parents=True)
        (gene_dir / "Summary.csv").write_text(
            "Reads With Valid Barcodes,0.90\n"
            "Q30 Bases in CB+UMI,0.91\n"
            "Q30 Bases in RNA read,0.92\n"
            "Reads Mapped to Genome: Unique,0.75\n"
            "Reads Mapped to Gene: Unique Gene,0.65\n",
            encoding="utf-8",
        )
        # Deliberately omit S4 from STAR records: the read-count violin must
        # still retain that analyzed well as zero with a false match flag.
        barcodes = ("S1", "S2") if plate == "F1" else ("S3",)
        (gene_dir / "CellReads.stats").write_text(
            "CB\tcbMatch\tnGenesUnique\n"
            + "".join(
                f"{barcode}\t{1000 + idx * 100}\t99\n"
                for idx, barcode in enumerate(barcodes)
            ),
            encoding="utf-8",
        )

    qc_files = generate_qc_report(
        dsd, tmp_path / "qc", formats=("png",), star_root=star_root,
        marker_genes=[],
    )

    qc_stems = {Path(path).stem for path in qc_files}
    assert "Combined_F1_F2_Multilayered_Doughnut" in qc_stems
    assert "Combined_F1_F2_STAR_Read_Count_Violin" in qc_stems

    star_files = generate_star_qc_figures(
        dsd, star_root, tmp_path / "star_figures", formats=("png",),
    )
    assert {Path(path).stem for path in star_files} == {
        "Combined_F1_F2_Multilayered_Doughnut",
        "Combined_F1_F2_STAR_Read_Count_Violin",
        "Combined_F1_F2_Saturation_Entrez_ID",
        "Combined_F1_F2_Saturation_Gene_Symbol",
    }
    read_files = generate_star_read_qc_figure(
        dsd,
        star_root,
        tmp_path / "star_read",
        formats=("png",),
        groupby="plate_id",
        nrow_plate=1,
    )
    assert {Path(path).stem for path in read_files} == {
        "Combined_F1_F2_Multilayered_Doughnut"
    }
    read_count_files = generate_star_read_count_violin_figure(
        dsd,
        star_root,
        tmp_path / "star_read_count",
        formats=("png",),
        groupby="plate_id",
        nrow_plate=1,
    )
    assert {Path(path).stem for path in read_count_files} == {
        "Combined_F1_F2_STAR_Read_Count_Violin"
    }
    read_counts = pd.read_csv(
        tmp_path / "star_read_count" / "star_read_count_per_well.csv"
    )
    assert len(read_counts) == len(dsd.obs)
    assert read_counts["matched_star_record"].sum() == len(dsd.obs) - 1
    missing_s4 = read_counts.loc[read_counts["barcode"] == "S4"].iloc[0]
    assert not missing_s4["matched_star_record"]
    assert missing_s4["read_count"] == 0
    read_summary = pd.read_csv(
        tmp_path / "star_read_count" / "star_read_count_summary.csv"
    )
    assert set(read_summary["plate_id"]) == {"F1", "F2"}
    assert read_summary["n_wells"].eq(2).all()
    assert {
        "mean_read_count", "median_read_count", "min_read_count",
        "max_read_count",
    }.issubset(read_summary.columns)
    saturation_files = generate_star_saturation_figures(
        dsd,
        star_root,
        tmp_path / "star_saturation",
        formats=("png",),
        groupby="plate_id",
        nrow_plate=1,
    )
    assert {Path(path).stem for path in saturation_files} == {
        "Combined_F1_F2_Saturation_Entrez_ID",
        "Combined_F1_F2_Saturation_Gene_Symbol",
    }
    saturation = pd.read_csv(
        tmp_path / "star_saturation" / "star_saturation_metrics.csv"
    )
    assert saturation["nGenesUnique"].eq(99).all()
    assert saturation["symbol_genes"].lt(99).all()
    split_files = generate_star_saturation_figures(
        dsd,
        star_root,
        tmp_path / "star_saturation_split",
        formats=("png",),
        groupby="plate_id",
        split_by="plate_id",
        nrow_plate=2,
    )
    assert {Path(path).stem for path in split_files} == {
        "Combined_F1_F2_Saturation_Entrez_ID_split_by_plate_id",
        "Combined_F1_F2_Saturation_Gene_Symbol_split_by_plate_id",
    }


# ---------------------------------------------------------------------------
# Comparison workflow
# ---------------------------------------------------------------------------

import importlib.util

import pytest

from drugseqpy import create_drugseq_object, run_comparison_workflow
from drugseqpy.enrichment import _save_gmt
from drugseqpy.utils import make_dummy_screen

HAS_GSEAPY = importlib.util.find_spec("gseapy") is not None


@pytest.fixture(scope="module")
def cmp_screen():
    counts, obs = make_dummy_screen(
        n_genes=80, n_plates=1, n_dmso_per_plate=6,
        n_compounds=2, n_reps=4, seed=31,
    )
    return create_drugseq_object(counts, obs)


def test_comparison_workflow_object_only(cmp_screen):
    result = run_comparison_workflow(
        cmp_screen, methods=["limma_voom"], enrichment=False,
        figures=("volcano", "heatmap", "bar", "box", "violin"),
    )
    assert "summary" in result and "manifest" in result
    assert not result["summary"].empty
    assert set(result["summary"]["method"]) == {"limma_voom"}
    assert cmp_screen.adata.uns["comparison_config"]["methods"] == \
        ["limma_voom"]
    manifest = result["manifest"]
    assert (manifest["status"] == "done").all()
    assert len(result["figures"]) >= 5


def test_comparison_workflow_accepts_anndata(cmp_screen):
    import anndata as ad
    adata = cmp_screen.adata.copy()
    result = run_comparison_workflow(
        adata, methods=["limma_voom"], enrichment=False,
        figures=("volcano",),
    )
    assert not result["summary"].empty


def test_comparison_workflow_returns_wrapped_plain_anndata(cmp_screen):
    adata = cmp_screen.adata.copy()
    result = run_comparison_workflow(
        adata, methods=["limma_voom"], enrichment=False,
        filter_samples_kwargs={"min_umi": 0, "min_genes": 0,
                               "max_pct_mito": 100},
        figures=(),
    )
    assert "dsd" in result
    assert result["dsd"].n_obs == adata.n_obs


def test_comparison_workflow_inplace_false_returns_copy(cmp_screen):
    result = run_comparison_workflow(
        cmp_screen, methods=["limma_voom"], enrichment=False,
        figures=(), inplace=False,
    )
    assert "dsd" in result
    assert result["dsd"] is not cmp_screen


def test_comparison_workflow_writes_report(tmp_path, cmp_screen):
    out = tmp_path / "comparison"
    result = run_comparison_workflow(
        cmp_screen, out,
        methods=["limma_voom"], enrichment=False,
        figures=("volcano", "heatmap", "bar"),
        formats=("png",),
    )
    files = result["files"]
    assert any("/results/de/" in f for f in files)
    assert any(f.endswith("comparison_manifest.tsv") for f in files)
    assert any(f.endswith("comparison_config.json") for f in files)
    assert any(f.endswith(".png") for f in files)
    assert (out / "results" / "de" / "Cmpd01.tsv").exists()
    assert (out / "results" / "de" / "Cmpd01_limma_voom.tsv").exists()
    assert (out / "results" / "de" / "comparison_summary.tsv").exists()
    manifest = pd.read_csv(out / "results" / "manifests" /
                           "comparison_manifest.tsv", sep="\t")
    assert {"type", "name", "status", "note"}.issubset(manifest.columns)


def test_comparison_workflow_optional_normalization_and_filtering(cmp_screen):
    result = run_comparison_workflow(
        cmp_screen, methods=["limma_voom"], enrichment=False, figures=(),
        normalize="CPM", filter_genes_kwargs={"min_count": 0, "min_samples": 0},
        filter_samples_kwargs={"min_umi": 0, "min_genes": 0,
                               "max_pct_mito": 100},
    )
    assert result["config"]["normalize"] == "CPM"
    assert cmp_screen.adata.uns["norm_method"] == "CPM"


@pytest.mark.skipif(not HAS_GSEAPY, reason="gseapy not installed")
def test_comparison_workflow_with_enrichment(tmp_path, cmp_screen):
    gs = {
        "SET_UP": [f"Gene{i:04d}" for i in range(1, 19)],
        "SET_DOWN": [f"Gene{i:04d}" for i in range(21, 39)],
    }
    gmt = tmp_path / "lib.gmt"
    _save_gmt(gmt, gs)
    result = run_comparison_workflow(
        cmp_screen, tmp_path / "out",
        methods=["limma_voom"], libraries=["lib"],
        gene_set_paths={"lib": gmt}, enrichment_mode=("ora",),
        figures=("enrichment",), formats=("png",),
    )
    assert "enrichment_results" in cmp_screen.adata.uns
    assert any("enrichment" in f for f in result["files"])
    assert any("/results/enrichment/" in f for f in result["files"])
    figure_names = {Path(path).name for path in result["files"]}
    assert any(name.endswith("_up_enrichment_dotplot.png") for name in figure_names)
    assert any(name.endswith("_down_enrichment_dotplot.png") for name in figure_names)


def test_comparison_workflow_missing_library_recorded(cmp_screen):
    result = run_comparison_workflow(
        cmp_screen, methods=["limma_voom"],
        libraries=["no_such_lib"], allow_network=False,
        figures=(),
    )
    enr_rows = result["manifest"][result["manifest"]["type"] == "enrichment"]
    assert len(enr_rows) == 1
    assert enr_rows.iloc[0]["status"] == "failed"
    assert "no_such_lib" in enr_rows.iloc[0]["note"]


def test_comparison_workflow_de_backend_failure_recorded(cmp_screen):
    result = run_comparison_workflow(
        cmp_screen, methods=["edgeR"], enrichment=False, figures=(),
    )
    de_rows = result["manifest"][result["manifest"]["type"] == "de"]
    assert (de_rows["status"] != "done").any()
    assert de_rows["note"].astype(str).str.contains("edgeR").any()
