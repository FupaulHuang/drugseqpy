"""
drugseqpy
=========
Quality control and analysis of DRUG-seq / HTTr high-throughput
transcriptomics data.  Built on AnnData / scanpy.

All functions operate on plain anndata.AnnData objects — no wrapper class.
Every scanpy function (sc.pl.*, sc.pp.*, sc.tl.*) works directly on the
AnnData returned by create_drugseq_object() or any drugseqpy function.

Quick start
-----------
>>> import drugseqpy as ds
>>> import scanpy as sc
>>> adata = ds.create_drugseq_object(counts, obs)
>>> ds.compute_qc_metrics(adata)
>>> sc.pl.violin(adata, keys=['total_counts','pct_counts_mt'], groupby='plate_id')
>>> ds.normalize_counts(adata, method='limma_voom')
>>> sc.pp.pca(adata)
>>> sc.pl.pca(adata, color='compound')
"""

from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("drugseqpy")
except PackageNotFoundError:
    __version__ = "0.1.0"

from .core import (
    DrugSeqData,
    activate_gene_names,
    create_drugseq_object,
    merge_drugseq_objects,
)
from .qc import (
    compute_qc_metrics,
    compute_plate_qc,
    plate_qc_summary,
    compute_group_qc,
    compute_replicate_icc,
    audit_control_outliers,
    plot_control_outlier_audit,
    select_robust_controls,
    check_zero_inflation,
    validate_metadata,
)
from .normalization import (
    normalize_counts,
    compare_normalizations,
    export_matrix,
)
from .filtering import (
    filter_samples,
    filter_genes,
)
from .batch import (
    correct_batch,
    compute_batch_effect_metrics,
)
from .reduction import (
    run_pca,
    run_umap,
    run_tsne,
    embed_dmso,
    cluster_compounds,
)
from .differential import (
    run_de,
    compute_multi_de,
    summarise_de,
    run_comparison,
    normalize_contrasts,
    summarise_comparison,
)
from .screen import (
    aggregate_by_de,
    compute_compound_umap,
    compute_compound_fingerprint,
    compute_compound_similarity_network,
    compute_screen_profile,
    plot_screen_overview,
    plot_screen_heatmap,
    plot_gene_counts,
)
from .dose_response import (
    fit_dose_response,
    plot_dose_gene_counts,
    plot_signature_dose_response,
    compute_multi_dr,
    plot_dr_panel,
)
from .enrichment import (
    run_gsea,
    run_go_enrichment,
    run_ora,
    connectivity_score,
    run_enrichment,
    resolve_gene_sets,
    enrichment_summary,
    CANONICAL_LIBRARIES,
)
from .plots import (
    plot_qc_summary,
    plot_plate_heatmap,
    plot_zprime,
    plot_hk_genes,
    plot_embedding,
    plot_pca,
    plot_umap,
    plot_tsne,
    plot_mds,
    plot_rle,
    plot_volcano,
    plot_ma,
    plot_pc_elbow,
    plot_qc_scatter,
    plot_replicate_distance,
    plot_group_qc_heatmap,
    plot_norm_comparison,
    plot_compound_umap,
    plot_comparison_volcano,
    plot_comparison_heatmap,
    plot_deg_counts,
    plot_gene_boxplot,
    plot_gene_violin,
    plot_enrichment_dotplot,
    plot_enrichment_barplot,
)
from .io import load_metadata, load_expression_matrix, create_drugseq_from_files
from .plate import build_plate_layout, plot_plate_layout
from .mix_species import assign_well_species, compute_mix_species_qc
from .reproducibility import (
    compute_within_plate_repeatability,
    summarise_within_plate_repeatability,
    plot_within_plate_repeatability,
    plot_within_plate_replicate_scatter,
    plot_within_plate_correlation_heatmap,
    compute_cross_plate_reproducibility,
    plot_single_plate_de_volcano,
    plot_single_plate_de_layout,
    plot_single_plate_de_heatmap,
    plot_cross_plate_reproducibility,
    plot_plate_logfc_concordance,
)
from .workflows import (
    generate_plate_layout_figures,
    generate_gene_set_violin_figure,
    generate_qc_metric_violin_figure,
    generate_outlier_score_scatter_figure,
    generate_umi_density_figure,
    generate_detected_gene_density_figure,
    generate_qc_distribution_figures,
    generate_gene_umi_distribution_figures,
    generate_marker_gene_figure,
    generate_marker_gene_dotplot_figure,
    generate_marker_gene_violin_figure,
    generate_gene_summary_figures,
    generate_star_read_qc_figure,
    generate_star_read_count_violin_figure,
    generate_star_saturation_figures,
    generate_star_qc_figures,
    generate_qc_report,
    generate_mix_species_barnyard_figure,
    generate_mix_species_purity_figure,
    generate_mix_species_concordance_figure,
    generate_mix_species_qc_metrics_figure,
    generate_mix_species_fraction_figure,
    generate_mix_species_contamination_figure,
    generate_mix_species_report,
    run_comparison_workflow,
    MDA_MB_231_MARKERS,
)

__all__ = [
    # core
    "DrugSeqData", "activate_gene_names", "create_drugseq_object",
    "merge_drugseq_objects",
    # qc
    "compute_qc_metrics", "compute_plate_qc", "plate_qc_summary",
    "compute_group_qc", "compute_replicate_icc", "select_robust_controls",
    "audit_control_outliers", "plot_control_outlier_audit",
    "check_zero_inflation", "validate_metadata",
    # normalization
    "normalize_counts", "compare_normalizations", "export_matrix",
    # filtering
    "filter_samples", "filter_genes",
    # batch
    "correct_batch", "compute_batch_effect_metrics",
    # reduction
    "run_pca", "run_umap", "run_tsne", "embed_dmso", "cluster_compounds",
    # differential
    "run_de", "compute_multi_de", "summarise_de",
    "run_comparison", "normalize_contrasts", "summarise_comparison",
    # screen
    "aggregate_by_de", "compute_compound_umap",
    "compute_compound_fingerprint", "compute_compound_similarity_network",
    "compute_screen_profile", "plot_screen_overview", "plot_screen_heatmap",
    "plot_gene_counts",
    # dose-response
    "fit_dose_response", "compute_multi_dr", "plot_dr_panel","plot_dose_gene_counts","plot_signature_dose_response",
    # enrichment
    "run_gsea", "run_ora", "connectivity_score","run_go_enrichment",
    "run_enrichment", "resolve_gene_sets", "enrichment_summary",
    "CANONICAL_LIBRARIES",
    # plots
    "plot_qc_summary", "plot_plate_heatmap", "plot_zprime", "plot_hk_genes",
    "plot_embedding", "plot_pca", "plot_umap", "plot_tsne", "plot_mds",
    "plot_rle", "plot_volcano", "plot_ma",
    "plot_pc_elbow", "plot_qc_scatter", "plot_replicate_distance",
    "plot_group_qc_heatmap", "plot_norm_comparison", "plot_compound_umap",
    "plot_comparison_volcano", "plot_comparison_heatmap", "plot_deg_counts",
    "plot_gene_boxplot", "plot_gene_violin", "plot_enrichment_dotplot",
    "plot_enrichment_barplot",
    "load_metadata", "load_expression_matrix", "create_drugseq_from_files",
    "build_plate_layout", "plot_plate_layout", "assign_well_species",
    "compute_mix_species_qc",
    "compute_within_plate_repeatability", "plot_within_plate_repeatability",
    "summarise_within_plate_repeatability",
    "plot_within_plate_replicate_scatter",
    "plot_within_plate_correlation_heatmap",
    "compute_cross_plate_reproducibility",
    "plot_single_plate_de_volcano", "plot_single_plate_de_layout",
    "plot_single_plate_de_heatmap", "plot_cross_plate_reproducibility",
    "plot_plate_logfc_concordance",
    "generate_plate_layout_figures", "generate_gene_set_violin_figure",
    "generate_qc_metric_violin_figure",
    "generate_outlier_score_scatter_figure",
    "generate_umi_density_figure", "generate_detected_gene_density_figure",
    "generate_qc_distribution_figures",
    "generate_gene_umi_distribution_figures", "generate_marker_gene_figure",
    "generate_marker_gene_dotplot_figure",
    "generate_marker_gene_violin_figure",
    "generate_gene_summary_figures", "MDA_MB_231_MARKERS",
    "generate_star_read_qc_figure",
    "generate_star_read_count_violin_figure",
    "generate_star_saturation_figures",
    "generate_star_qc_figures", "generate_qc_report",
    "generate_mix_species_barnyard_figure", "generate_mix_species_purity_figure",
    "generate_mix_species_concordance_figure", "generate_mix_species_qc_metrics_figure",
    "generate_mix_species_fraction_figure", "generate_mix_species_contamination_figure",
    "generate_mix_species_report", "run_comparison_workflow",
]
