# F1 standard QC vignette

This vignette uses the F1 STARsolo filtered matrix and F1 metadata as the standard
QC case. The independent mix-species workflow is intentionally not called here.

```python
from drugseqpy.io import create_drugseq_from_files
from drugseqpy.workflows import generate_qc_report

dsd = create_drugseq_from_files(
    expression_path="/path/to/F1/library_seq_Solo.out/Gene/filtered",
    metadata_path="/path/to/F1_metadata.csv",
)

# The file loader adds:
#   sample_id = plate_id + barcode
#   treatment = compound + dose + time
generate_qc_report(dsd, "results/f1_qc", formats=("pdf", "png"))
```

The output directory contains sample and plate QC tables, plate coordinates,
plate-layout plots, and adaptive QC boxplots. To run the independent species
workflow, use `generate_mix_species_report()` or the `drugseqpy mix-species`
command separately.
