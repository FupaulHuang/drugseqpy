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
    create_drugseq_object,
    merge_drugseq_objects,
)
from .qc import (
    compute_qc_metrics,
    compute_plate_qc,
    plate_qc_summary,
    compute_group_qc,
    compute_replicate_icc,
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
)
from .reduction import (
    run_pca,
    run_umap,
    embed_dmso,
    cluster_compounds,
)
from .differential import (
    run_de,
    compute_multi_de,
    summarise_de,
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
)
from .plots import (
    plot_qc_summary,
    plot_plate_heatmap,
    plot_zprime,
    plot_hk_genes,
    plot_embedding,
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
)

__all__ = [
    # core
    "create_drugseq_object", "merge_drugseq_objects",
    # qc
    "compute_qc_metrics", "compute_plate_qc", "plate_qc_summary",
    "compute_group_qc", "compute_replicate_icc", "select_robust_controls",
    "check_zero_inflation", "validate_metadata",
    # normalization
    "normalize_counts", "compare_normalizations", "export_matrix",
    # filtering
    "filter_samples", "filter_genes",
    # batch
    "correct_batch",
    # reduction
    "run_pca", "run_umap", "embed_dmso", "cluster_compounds",
    # differential
    "run_de", "compute_multi_de", "summarise_de",
    # screen
    "aggregate_by_de", "compute_compound_umap",
    "compute_compound_fingerprint", "compute_compound_similarity_network",
    "compute_screen_profile", "plot_screen_overview", "plot_screen_heatmap",
    "plot_gene_counts",
    # dose-response
    "fit_dose_response", "compute_multi_dr", "plot_dr_panel","plot_dose_gene_counts","plot_signature_dose_response",
    # enrichment
    "run_gsea", "run_ora", "connectivity_score","run_go_enrichment",
    # plots
    "plot_qc_summary", "plot_plate_heatmap", "plot_zprime", "plot_hk_genes",
    "plot_embedding", "plot_mds", "plot_rle", "plot_volcano", "plot_ma",
    "plot_pc_elbow", "plot_qc_scatter", "plot_replicate_distance",
    "plot_group_qc_heatmap", "plot_norm_comparison", "plot_compound_umap",
]
