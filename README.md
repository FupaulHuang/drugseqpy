# DRUGseqPy

Python reimplementation of the DRUGseqR package for quality control and analysis of Drug-seq / MAC-seq / HTTr high-throughput transcriptomics data. Built on [AnnData](https://anndata.readthedocs.io) and [scanpy](https://scanpy.readthedocs.io).

## Overview

DRUGseqPy follows scanpy conventions throughout: the central data structure is a standard `AnnData` object where **rows are samples** and **columns are genes**, wrapped in a thin `DrugSeqData` class that enforces Drug-seq metadata conventions. The wrapper exposes `.adata` and forwards common AnnData attributes (`X`, `layers`, `obs`, `var`, and `uns`); the comparison workflow also accepts plain `AnnData` directly.

## AnnData layout

| Slot | Contents |
|---|---|
| `adata.X` | Normalized expression (log-CPM or voom-weighted), float32 |
| `adata.layers['counts']` | Raw integer counts — never modified |
| `adata.obs` | Sample metadata: `plate_id`, `well_id`, `compound`, `dose`, `sample_type` |
| `adata.var` | Gene metadata: `is_mito`, `is_ribo`, `highly_variable` |
| `adata.uns['plate_qc']` | Per-plate Z', SSMD, DMSO CV, spatial well matrix |
| `adata.uns['de_results']` | Dict of compound → DE DataFrame |
| `adata.uns['comparison_results']` | Dict of contrast (or contrast::method) → DE DataFrame |
| `adata.uns['enrichment_results']` | Contrast × library × GSEA/ORA pathway tables |
| `adata.uns['enrichment_summary']` | Flattened pathway summary table |
| `adata.uns['gsea']` | Dict of compound → gseapy results |
| `adata.obsm['X_pca']` | PCA embedding |
| `adata.obsm['X_umap']` | UMAP embedding |
| `adata.obsm['X_dmso_pca']` | DMSO-anchored perturbation embedding |
| `adata.obs['pert_score']` | Z-scored distance from DMSO centroid |

## Installation

```bash
conda create -n drugseqpy python==3.10.0
source activate drugseqpy
conda install git

# Core
#pip install drugseqpy
pip install git+https://github.com/FupaulHuang/drugseqpy.git

# Full (includes rdkit via conda, igraph, decoupler)
#pip install "drugseqpy[full]"

# Development
#pip install "drugseqpy[dev]"
```

For rdkit (cheminformatics):
```bash
conda install -c conda-forge rdkit
```

## Quickstart

```python
import drugseqpy as ds
from drugseqpy.utils import make_dummy_screen

# Build object
counts, obs = make_dummy_screen(n_genes=500, n_compounds=8)
dsd = ds.create_drugseq_object(counts=counts, obs=obs)

# QC
ds.compute_qc_metrics(dsd, inplace=True)
ds.compute_plate_qc(dsd, inplace=True)
ds.plot_qc_summary(dsd)
ds.plot_zprime(dsd)

# Filter
dsd = ds.filter_samples(dsd, min_umi=5000, max_pct_mito=20)
dsd = ds.filter_genes(dsd, min_count=5, group_aware=True)

# Normalize (limma_voom recommended for Drug-seq)
ds.normalize_counts(dsd, method='limma_voom', inplace=True)

# Batch correction + dimensionality reduction
ds.correct_batch(
    dsd, batch_col='plate_id', method='limma_voom',
    covariate_cols=['compound'], inplace=True,
)
ds.run_pca(dsd, inplace=True)
ds.run_umap(dsd, inplace=True)
ds.run_tsne(dsd, inplace=True)
ds.embed_dmso(dsd, inplace=True)   # perturbation score

# Differential expression (all compounds vs DMSO; limma_voom is the default)
ds.compute_multi_de(dsd, method='limma_voom', n_jobs=4, inplace=True)
de_summary = ds.summarise_de(dsd)

# Screen-level analysis
ds.plot_screen_overview(dsd)
ds.plot_screen_heatmap(dsd, n_top=50)
compound_umap = ds.compute_compound_umap(dsd)
network = ds.compute_compound_similarity_network(dsd)

# Dose-response (lmfit, drc equivalent)
dr = ds.fit_dose_response(dsd, compound='CmpdA')
ds.plot_dr_panel(dr, dsd, compound='CmpdA')

# GSEA (local/cache by default; set allow_network=True for a download)
ds.run_gsea(dsd, gene_sets='MSigDB_Hallmark_2020', allow_network=True,
            inplace=True)

# Comparison workflow: DE (limma_voom/edgeR/deseq) + optional enrichment
# (Hallmark / GO BP / Reactome / KEGG) + DEG figures. Local GMT paths are
# recommended for reproducible offline enrichment.
result = ds.run_comparison_workflow(dsd, 'comparison_output')
```

See [docs/comparison_workflow.md](docs/comparison_workflow.md) for the
full comparison API: flexible contrasts ("all samples vs one control" or
"one group vs another"), optional enrichment libraries, volcano / heatmap
/ bar / box / violin figures, output trees, and the `drugseqpy compare`
CLI.

Two universal comparison tutorials separate the workflows that have different
statistical requirements:

- [Single-plate comparison](examples/single_plate_comparison_workflow_tutorial.html)
  covers focused QC, audited DMSO exclusion, within-plate differential
  expression, and offline directional enrichment. F1 supplies the real worked
  output but is not part of the workflow definition.
- [Multi-plate comparison](examples/multi_plate_comparison_workflow_tutorial.html)
  adds limma/edgeR batch-corrected exploratory embeddings, plate-adjusted DE,
  within-plate repeatability, and cross-plate treatment-effect reproducibility.
  F1/F2 supply the real worked output.

The workflow documentation includes three self-contained tutorials. Example
plate names identify the worked data and are not part of the API definition:

- [Single-plate comparison workflow](examples/single_plate_comparison_workflow_tutorial.html)
  covers one-batch analysis without inappropriate batch correction.
- [Multi-plate comparison workflow](examples/multi_plate_comparison_workflow_tutorial.html)
  covers two or more plates and validates both batch removal and biological
  reproducibility.
- [Quality-control workflow](examples/qc_workflow_tutorial.html) covers matrix
  and metadata QC, plate layouts, optional STARsolo QC, related functions,
  saved source tables, and terminology.
- [Mix-species workflow](examples/mix_species_workflow_tutorial.html)
  configures any two species, assigns the intended base species of each well,
  measures cross-well contamination in both directions, and uses human/mouse
  F3/F4 data only as the worked example.
- [API reference](examples/api_reference.html) groups the public functions by
  task and documents their live signatures, parameters, defaults, return
  values, side effects, and workflow notes.

The two tutorial pages embed figures generated by the public functions using
real F3/F4 example data. To regenerate the outputs and rebuild all three HTML
pages, run:

```bash
python examples/build_f3_f4_workflow_tutorials.py
```

QC visualization APIs use `groupby="plate_id"` by default to color or
categorize groups within one axes. Set `split_by` only when separate panels
are desired; its default is `None`.

## Notebooks

| Notebook | Contents |
|---|---|
| `01_QC_Workflow.ipynb` | Metadata validation, per-sample QC, plate QC (Z'), MDS, normalization comparison (RLE), group-aware gene filtering |
| `02_Screen_Analysis.ipynb` | Batch correction, PCA/UMAP, DMSO-anchored perturbation scoring, multi-compound DE, volcano/MA plots, screen heatmap, compound similarity networks, GSEA |
| `03_Dose_Response_Cheminformatics.ipynb` | LL4/Weibull fitting (lmfit), AIC model selection, EC50 CIs, multi-compound DR, SMILES retrieval from PubChem, RDKit descriptors, SAR modelling |

## Module overview

| Module | Key functions |
|---|---|
| `core` | `create_drugseq_object`, `activate_gene_names`, `merge_drugseq_objects`, `DrugSeqData` |
| `mix_species` | `assign_well_species`, `compute_mix_species_qc` |
| `qc` | `compute_qc_metrics`, `compute_plate_qc`, `compute_group_qc`, `compute_replicate_icc`, `select_robust_controls`, `check_zero_inflation`, `validate_metadata` |
| `normalization` | `normalize_counts` (log1p/CPM/TMM/limma_voom), `compare_normalizations` |
| `filtering` | `filter_samples`, `filter_genes` (group-aware) |
| `batch` | `correct_batch` (limma-voom / edgeR-TMM expression correction) |
| `reduction` | `run_pca`, `run_umap`, `run_tsne`, `embed_dmso`, `cluster_compounds` |
| `differential` | `run_de`, `compute_multi_de`, `summarise_de`, `run_comparison`, `normalize_contrasts`, `summarise_comparison` |
| `screen` | `aggregate_by_de`, `compute_compound_umap`, `compute_compound_fingerprint`, `compute_compound_similarity_network`, `plot_screen_overview`, `plot_screen_heatmap` |
| `dose_response` | `fit_dose_response`, `compute_multi_dr`, `plot_dr_panel` |
| `enrichment` | `run_gsea`, `run_ora`, `connectivity_score`, `run_enrichment`, `resolve_gene_sets`, `enrichment_summary` |
| `workflows` | `generate_plate_layout_figures`, `generate_gene_set_violin_figure`, `generate_qc_metric_violin_figure`, `generate_outlier_score_scatter_figure`, `generate_umi_density_figure`, `generate_detected_gene_density_figure`, `generate_qc_distribution_figures`, `generate_gene_umi_distribution_figures`, `generate_marker_gene_figure`, `generate_gene_summary_figures`, `generate_star_read_qc_figure`, `generate_star_saturation_figures`, `generate_star_qc_figures`, `generate_qc_report`, `generate_mix_species_barnyard_figure`, `generate_mix_species_purity_figure`, `generate_mix_species_concordance_figure`, `generate_mix_species_qc_metrics_figure`, `generate_mix_species_fraction_figure`, `generate_mix_species_contamination_figure`, `generate_mix_species_report`, `run_comparison_workflow` |
| `plots` | `plot_qc_summary`, `plot_plate_heatmap`, `plot_zprime`, `plot_pca`, `plot_umap`, `plot_tsne`, `plot_mds`, `plot_rle`, `plot_embedding`, `plot_volcano`, `plot_ma`, `plot_norm_comparison`, `plot_compound_umap` |

## macpie / DRUGseqR mapping

| R / macpie function | Python equivalent |
|---|---|
| `computeQCMetrics()` | `compute_qc_metrics()` |
| `computePlateQC()` | `compute_plate_qc()` |
| `macpie::compute_qc_metrics()` group-level | `compute_group_qc()` |
| `macpie::plot_mds()` | `plot_mds()` |
| `macpie::plot_rle()` | `plot_rle()` |
| `macpie::filter_genes_by_expression()` | `filter_genes(group_aware=True)` |
| `macpie::select_robust_controls()` | `select_robust_controls()` |
| `macpie::aggregate_by_de()` | `aggregate_by_de()` |
| `macpie::plot_multi_de()` | `plot_screen_heatmap()` |
| `normalizeCounts(method="limma_voom")` | `normalize_counts(method='limma_voom')` |
| `runDE(method="limma")` | `run_de(method='ols_voom')` |
| `fitDoseResponse()` (drc) | `fit_dose_response()` (lmfit) |
| `connectivityScore()` | `connectivity_score()` |
| `toSeurat()` / `fromSeurat()` | native AnnData (no bridge needed) |
| `computeReplicateICC()` | `compute_replicate_icc()` (novel) |
| `plotNormComparison()` | `plot_norm_comparison()` (novel) |
| `computeCompoundFingerprint()` | `compute_compound_fingerprint()` (novel) |
| `computeCompoundSimilarityNetwork()` | `compute_compound_similarity_network()` (novel) |
| `plotScreenOverview()` | `plot_screen_overview()` (novel) |

## Running tests

```bash
pip install "drugseqpy[dev]"
pytest tests/ -v
```

## Dependencies

**Required**: anndata, scanpy, numpy, pandas, scipy, matplotlib, seaborn, plotly, pydeseq2, statsmodels, scikit-learn, umap-learn, gseapy, lmfit, tqdm, joblib, requests

**Optional (full)**: rdkit (cheminformatics), igraph (compound networks), decoupler (pathway activity)

## License

MIT
