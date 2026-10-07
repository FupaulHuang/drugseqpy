# Comparison Workflow

End-to-end differential expression, enrichment analysis, and DEG
visualization for `DrugSeqData` objects (or plain `AnnData`).

## Input contract

The workflow expects the standard drugseqpy object layout:

- `adata.layers["counts"]` — raw integer counts (never modified)
- `adata.X` — normalized expression (used by box/violin plots)
- `adata.obs` — sample metadata with a grouping column (`compound` by
  default) and, when available, `plate_id` for within-plate pairing

## Quick start

```python
import drugseqpy as ds

# All compounds vs DMSO with the default method (Python limma_voom),
# all four enrichment libraries, and the full figure set:
result = ds.run_comparison_workflow(dsd, "comparison_output")

# One group versus another group:
result = ds.run_comparison_workflow(
    dsd,
    comparisons={"DrugA": "DrugB"},
    methods=["limma_voom", "deseq"],
)

# All three DE methods, no enrichment, figures only:
result = ds.run_comparison_workflow(
    dsd,
    methods=["limma_voom", "edgeR", "deseq"],
    enrichment=False,
    figures=("volcano", "heatmap", "bar", "box", "violin"),
)
```

The repository includes two self-contained, universal tutorials. F1 is used
only to render real output for the single-plate workflow; F1/F2 are used only
to render real output for the multi-plate workflow. STARsolo bundles live
under `data/F1` and `data/F2`; local Hallmark, GO BP, and Reactome GMT files
live under `data/gene_sets`:

```bash
PYTHONPATH=src python examples/run_universal_comparison_tutorials.py
python examples/build_universal_comparison_tutorials.py
```

Open [`single_plate_comparison_workflow_tutorial.html`](../examples/single_plate_comparison_workflow_tutorial.html)
for one-plate analyses. Open
[`multi_plate_comparison_workflow_tutorial.html`](../examples/multi_plate_comparison_workflow_tutorial.html)
for two or more plates.

Both worked examples deliberately use only the QC checks needed to support the
comparison decision. They compute sample metrics and plate layouts, then
remove only DMSO wells with `outlier_score > 5`. Water is excluded from the
automatic treatment contrasts. Genes must have at least five counts in at
least three wells before the Drug1-Drug16 versus DMSO models are fit.

The multi-plate workflow treats batch correction and inference as separate
operations. `correct_batch()` uses limma-voom or edgeR-style TMM log expression,
then subtracts the fitted plate component while preserving named biological
covariates. PCA, UMAP, t-SNE, and MDS are recomputed from that exploratory
expression view, while raw counts remain unchanged. Differential expression
continues to use raw counts and includes `plate_id` as a covariate. The report
evaluates batch removal with pre/post plate separation and matched-treatment
centroids, within-plate repeatability with replicate-profile correlations, and
cross-plate reproducibility with independently estimated log-fold-change
signatures, DEG overlap, and sign concordance.

Enrichment uses significant up- and downregulated DEGs separately with all
tested genes as the ORA background. When DE tables use Ensembl feature IDs,
`run_enrichment` maps them through `adata.var["gene_symbol"]` before matching
symbol-based GMT libraries.

## Comparison modes

- **All samples vs one control** (default): `comparisons=None` generates
  one contrast per non-control level of `group_col` versus `reference`
  (default `"DMSO"`). Only the configured reference and explicit
  `exclude_levels=[...]` are omitted; labels such as water/media are retained
  unless the caller excludes them.
- **One group vs other groups**: pass explicit contrasts —
  `{"DrugA": "DrugB"}` (case → control), `{"name": ("case", "control")}`,
  a list of `(case, control)` tuples, or dicts with
  `{"name", "case", "control", "group_col"}`. The `case` and `control`
  values may also be lists of levels to pool; use `control="rest"` for one
  group versus every other level in the grouping column.

```python
from drugseqpy import run_comparison, normalize_contrasts

# One-off contrast without the workflow:
de_table = ds.run_comparison(dsd, case="DrugA", control="DMSO",
                             method="limma_voom", batch_cols=["plate_id"])
```

## DE methods

| method        | backend                                             | dependencies               |
|---------------|-----------------------------------------------------|----------------------------|
| `limma_voom`  | Python voom log-CPM OLS + eBayes (default)          | built-in (statsmodels)     |
| `deseq`       | PyDESeq2 negative-binomial GLM                      | `pip install pydeseq2`     |
| `edgeR`       | R edgeR via rpy2 (exact/QLF test)                   | `conda install -c conda-forge rpy2 r-edger` |

Legacy names `ols_voom` → `limma_voom`, `pydeseq2`/`DESeq2` → `deseq`, and
`edger` → `edgeR` remain accepted. `t_test` is kept for fast screening/back-compat. Missing
optional backends raise an `ImportError` naming the installation command.
Passing `methods="all"` to the workflow or `compute_multi_de` expands to the
three requested backends; unavailable optional backends are recorded as
failures.

## Enrichment

Optional, default **on**, runs all four libraries by default:

| key        | library                      |
|------------|------------------------------|
| `hallmark` | MSigDB Hallmark 2020         |
| `go_bp`    | GO Biological Process 2023   |
| `reactome` | Reactome 2022                |
| `kegg`     | KEGG 2021 Human              |

For each contrast × library the workflow runs preranked **GSEA** and
directional **ORA** (up and down genes separately) via `gseapy`.
Use `libraries="all"` as a shorthand for the four canonical libraries.

Library resolution order per library (network access is opt-in; the workflow
defaults to `allow_network=False`):

1. local GMT file given via `gene_set_paths={"hallmark": "/path/x.gmt"}`
   (fully offline);
2. inline gene-set dicts passed in `libraries`;
3. an existing cached GMT under `~/.cache/drugseqpy/gene_sets/`;
4. optionally, a gseapy network download when `allow_network=True`, cached as
   GMT for later offline runs.

```python
result = ds.run_comparison_workflow(
    dsd,
    gene_set_paths={"hallmark": "gmt/hallmark.gmt", "kegg": "gmt/kegg.gmt"},
    enrichment_mode=("ora",),   # ORA only
)
```

Optional preprocessing is explicit. Pass `normalize="CPM"` (or
`"limma_voom"`) and/or keyword dictionaries such as
`filter_samples_kwargs={"min_umi": 5000}` and
`filter_genes_kwargs={"min_count": 5}`. If sample filtering is requested and
QC metrics are absent, the workflow computes them first.

## Figures

`figures=("volcano", "heatmap", "bar", "box", "violin", "enrichment")`
(default: all):

- **volcano** — one per contrast × method (up red / down blue / NS grey)
- **heatmap** — union of top `n_top_genes` significant DEGs by |logFC|
  across contrasts
- **bar** — mirrored up/down significant DEG counts per contrast
- **box** / **violin** — global expression panels plus one panel per
  contrast for the top significant genes (override with `boxplot_genes=[...]`;
  pass `contrast=` to `ds.plot_gene_boxplot`/`ds.plot_gene_violin` to restrict
  to one comparison)
- **enrichment** — pathway dot plot (direction-aware) and bar plots of
  `-log10(adjusted p-value)` per library and direction

Every plot returns a `matplotlib` Figure; empty results render an
explicit "No significant …" panel instead of raising.

The workflow result is a dictionary containing the summary, manifest, figure
objects, written files, and configuration. It also includes a wrapped `dsd`
entry when `inplace=False` or when a plain AnnData input had to be subsetted
during optional preprocessing.

## Storage

- `adata.uns["comparison_results"]` — dict keyed by contrast name, or
  `"<contrast>::<method>"` when several methods run; each table has
  `gene, logFC, base_mean, stat, pvalue, padj, significant, direction,
  contrast, case, control, method`
- `adata.uns["comparison_failures"]` — per-contrast/method error messages
  for optional backends or comparisons that could not run
- `adata.uns["de_results"]` — legacy mirror, populated for default
  compound-vs-reference runs so `summarise_de`, `aggregate_by_de` and
  screen helpers keep working
- `adata.uns["enrichment_results"][contrast][library]` — `"gsea"`,
  `"ora_up"`, `"ora_down"` frames (empty frames keep the schema)
- `adata.uns["comparison_config"]` — full run configuration and backend
  versions
- `adata.uns["enrichment_summary"]` — flattened pathway table for report
  builders and downstream filtering

## Output tree (when `output_dir` is given)

```
comparison_output/
└── results/
    ├── de/<contrast>_<method>.tsv
    ├── de/comparison_summary.tsv
    ├── enrichment/<contrast>_<library>_<mode>_<direction>.tsv
    ├── enrichment/pathway_summary.tsv
    ├── figures/<contrast>_<figure>.(png|pdf)
    └── manifests/
        ├── comparison_manifest.tsv      # per-step status + failure notes
        └── comparison_config.json
```

Failed optional steps are recorded in the manifest with the error
message rather than silently discarded.

## CLI

```bash
drugseqpy compare counts.tsv metadata.csv output_dir \
    --methods limma_voom,deseq \
    --comparisons "DrugA=DMSO;DrugB=DMSO" \
    --batch-cols plate_id \
    --libraries hallmark,go_bp,reactome,kegg \
    --gmt-dir gmt/          # offline libraries
    --no-enrichment         # add --allow-network only if downloads are desired
```

## API reference

| function | purpose |
|---|---|
| `run_comparison_workflow(dsd, ...)` | end-to-end workflow |
| `run_comparison(dsd, case, control, ...)` | one contrast, one method |
| `normalize_contrasts(comparisons, obs, ...)` | contrast normalization |
| `compute_multi_de(dsd, methods=..., comparisons=...)` | multi-contrast DE |
| `summarise_comparison(dsd)` | one row per contrast × method |
| `run_enrichment(dsd, libraries=..., gene_set_paths=...)` | GSEA + directional ORA |
| `resolve_gene_sets(libraries, ...)` | canonical libraries → gene-set dicts |
| `enrichment_summary(dsd)` | flat pathway summary table |
| `plot_comparison_volcano / _heatmap` | DEG figures |
| `plot_deg_counts` | mirrored DEG count bars |
| `plot_gene_boxplot / plot_gene_violin` | per-gene expression |
| `plot_enrichment_dotplot / _barplot` | pathway figures |
