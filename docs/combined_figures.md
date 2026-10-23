# Combined-library QC figures

`generate_combined_qc_report()` implements the 14 QC figures and six plate maps
in the combined-sample workflow. It accepts `AnnData` or `DrugSeqData`,
recomputes values from raw counts, and writes figures plus auditable CSV tables.
Individual `plot_*` functions return Matplotlib figures for notebook use.

## Generate a report

```python
import drugseqpy as ds

result = ds.generate_combined_qc_report(
    adata,
    "combined_qc",
    baseline_group="sample_1",
    groupby="plate_id",
    barcode_col="barcode",
    filtered_col="in_filtered_matrix",
    reads_col="reads_per_well",  # STARsolo CellReads.stats cbMatch
    summary_files={
        "sample_1": "sample_1/library_seq_Solo.out/Gene/Summary.csv",
        "sample_2": "sample_2/library_seq_Solo.out/Gene/Summary.csv",
    },
    formats=("pdf", "png"),
)
print(result["files"])
print(result["skipped"])
```

Required input fields:

| Location | Content |
| --- | --- |
| `layers['counts']` | Raw, nonnegative integer UMIs; observations × features. Sparse matrices remain sparse. Use `layer=None` explicitly if raw counts are in `X`. |
| `obs['plate_id']` | Library/group identity, configurable with `groupby`. |
| `obs['barcode']` | Barcode identity shared between libraries, unique within each group. Normalize barcode suffixes before combining libraries. |
| `obs['in_filtered_matrix']` | Optional boolean flag identifying filtered wells. Pass its name via `filtered_col`; without it, all supplied wells are eligible. |
| `obs['reads_per_well']` | Barcode-matching reads (`cbMatch`), configurable with `reads_col`. These are not restricted to gene-mapped reads. |
| `var['gene_symbol']` | Feature symbols, configurable with `symbol_col`. |
| `var['gene_id']` or `var_names` | Human `ENSG…` and mouse `ENSMUSG…` feature IDs. `gene_id_col` overrides automatic selection. |
| `obs['well_id']` | Physical plate positions such as `A01`, for plate maps. In the report, explicit `expected_row` and `expected_column` override numbered well IDs. |
| `obs['expected_cells']` | Planned plate-map labels (for example `231`, `3T3`, `water`); configurable with `planned_col`. |

Supply the full raw-well object if you want full-layout identity maps and
baseline species inferred from unfiltered wells. For QC-only output, use
`include_plate_maps=False`. `nrow` and `ncol` set the physical plate dimensions.
The report never modifies the input object or normalizes counts.

The doughnut requires either `summary_files` or a DataFrame `read_metrics` with
`group`, `metric`, and `percent` columns. `read_star_summary_metrics()` converts
STAR fractions to percentages. If neither input is supplied, the report skips
the doughnut and records the reason in `result['skipped']`.

## Individual calculations and figures

```python
tables = ds.compute_combined_qc(
    adata, baseline_group="sample_1", filtered_col="in_filtered_matrix",
)
fig = ds.plot_combined_barnyard(tables["wells"])
fig.savefig("barnyard.pdf")

fig = ds.plot_mt_ribo_housekeeping(tables["composition_values"])
fig.savefig("composition.pdf")
```

| Reference figure | Function and input table |
| --- | --- |
| Barnyard | `plot_combined_barnyard(tables['wells'])` |
| Species purity | `plot_combined_species_purity(tables['wells'])` |
| UMI / detected-gene density | `plot_combined_density(tables['wells'], 'total_umi' / 'n_genes_det')` |
| Human / mouse marker dashboard | `plot_combined_markers(tables['markers'], 'Human_MDA-MB-231_Markers' / 'Mouse_3T3_Markers')` |
| All-feature / symbol / Ensembl UMI distribution | `plot_combined_gene_umi(tables['gene_totals'], 'all' / 'gene_symbol' / 'ensembl_id')` |
| Symbol / Ensembl saturation | `plot_combined_saturation(tables['wells'], 'gene_symbol' / 'ensembl_id')` |
| Read / UMI / detected-feature boxplots | `plot_combined_qc_boxplots(tables['qc_values'])` |
| Mitochondrial / ribosomal / housekeeping boxplots | `plot_mt_ribo_housekeeping(tables['composition_values'])` |
| Full-library doughnut | `plot_combined_read_doughnut(read_metrics)` |
| Four quantitative and two identity maps | `generate_combined_qc_report()` or existing `plot_plate_layout()` / `generate_plate_layout_figures()` |

`marker_sets` accepts a mapping of panel names to symbols.
`housekeeping_sets` accepts `{'human': [...], 'mouse': [...]}`. Defaults reproduce
the reference human and mouse panels, including mouse **Hprt1**. Missing
housekeeping genes raise an error so means cannot silently change their
denominators; missing markers warn and are omitted. Zero-expression markers
remain in the panels. Each matched housekeeping feature contributes to the
mean, including zeros.

## Population and metric definitions

- Shared wells are the intersection of eligible barcodes across **all** groups.
  Eligibility and inclusion are exported for every input observation.
- Species identity is inferred once from `baseline_group` and carried to the
  other libraries: human if human UMIs / (human + mouse UMIs) ≥ 0.5, otherwise
  mouse. Shared wells with an undefined baseline are rejected. Planned cell
  labels are kept separate from inferred species.
- Purity uses human + mouse UMIs as its denominator. The barnyard reports
  count, minimum, maximum, median, and mean separately for both species.
- Mitochondrial percentage uses symbols matching `^(MT-|mt-)`. Cytosolic
  ribosomal percentage uses `RPL/RPS/RPLP/RPSA` and the corresponding mouse
  symbols, following the exact reference regex. MRPL/MRPS are excluded. Both
  percentages use total matrix UMIs as the denominator; zero totals give NaN.
- Housekeeping values are arithmetic means of raw UMIs. Human means appear
  only in human-baseline wells and mouse means only in mouse-baseline wells.
- Marker and housekeeping symbol lookup uses the first matching feature,
  matching the R source. Feature distributions and detected-feature counts
  retain duplicate symbols as independent feature rows.
- Gene distributions omit zero-total features. Totals are calculated per
  library before applying `log2(UMI + 1)`. Pooled plots retain separate
  library-feature observations; they do not sum libraries together. KDE is
  evaluated in transformed space and multiplied by the number of features.
- “Saturation” plots are reads-versus-detected-features scatterplots, not
  FASTQ downsampling curves. Read coordinates remain linear, with tick labels
  in millions. Legacy `Entrez_ID` filenames refer to **Ensembl IDs**.
- STAR summary percentages describe the **full library**, not just shared
  wells. Q30 measures bases; barcode and mapping metrics measure reads.
- Quantitative plate maps include shared wells and empty circles at other
  positions. Identity maps retain all supplied wells, marking nonshared
  positions `Excluded` on the species map. Planned labels remain available.

## Rendering and reproducibility

The Python figures preserve the reference metrics, major layouts, colors for
species/composition, and PDF page sizes; they are not pixel-identical ggplot2
renderings. SciPy's Silverman KDE can differ from R's density estimator.
Doughnut arcs deliberately use a fixed 0–100% circle rather than the R plot's
automatically scaled polar axis. Printed percentages remain unchanged.

Jitter is reproducible with a local random generator (default seed `20261013`).
`tables/` contains the actual well-level values, marker means, feature totals,
gene sets, purity statistics, boxplot medians, and plate plotting tables.

The existing plate functions now accept `legend_per_plate=True`, which gives
each panel a separate legend/colorbar and an independent continuous range.
Their default shared scale remains available. Independent ranges help examine
variation within each plate; compare colorbar values when comparing plates.

To reproduce a local combined-figure dataset and compare computations against
its exported reference tables:

```bash
PYTHONPATH=src python scripts/reproduce_combined_figures.py \
  /path/to/combined_reference \
  validation/combined_samples
```

The reference folder contains one raw AnnData file under `results/plate_maps/`,
reference CSVs under `results/tables/`, and per-sample STAR summaries under
`inputs/star_results_UMIstart/<sample>/library_seq_Solo.out/Gene/Summary.csv`.
The script discovers sample identifiers from the input metadata and assigns
`sample_1`, `sample_2`, and so on in first-observed order. The first sample is
the baseline unless `--baseline-group` is supplied. It creates 20 PDFs plus
PNGs for the default panels and checks the values against the reference tables.
The output manifest omits the input path. Original input files are not modified.

Use the same neutral labels in public metadata, filenames, plot legends, and
captions; keep any mapping to original identifiers outside published materials.
