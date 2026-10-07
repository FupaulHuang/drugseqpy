"""File loaders for expression matrices and Drug-seq metadata."""
from __future__ import annotations

import gzip
import re
from pathlib import Path

import pandas as pd
from scipy.io import mmread

from .core import DrugSeqData, create_drugseq_object


def load_metadata(
    path: str | Path,
    index_col: str | int | None = None,
    delimiter: str | None = "auto",
) -> pd.DataFrame:
    """Load sample metadata with optional delimiter auto-detection.

    ``delimiter="auto"`` uses comma for ``.csv``, tab for ``.tsv``, and
    pandas' Python-engine delimiter detection for ``.txt``. Supplying a
    delimiter explicitly overrides the extension-based choice for any of
    those text formats. Excel files do not accept a delimiter.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    auto_delimiter = delimiter is None or (
        isinstance(delimiter, str) and delimiter.casefold() == "auto"
    )
    if not auto_delimiter and (not isinstance(delimiter, str) or delimiter == ""):
        raise ValueError("delimiter must be 'auto', None, or a non-empty string")

    if suffix in {".csv", ".tsv", ".txt"}:
        if auto_delimiter:
            if suffix == ".csv":
                df = pd.read_csv(path, sep=",")
            elif suffix == ".tsv":
                df = pd.read_csv(path, sep="\t")
            else:
                df = pd.read_csv(path, sep=None, engine="python")
        else:
            df = pd.read_csv(path, sep=delimiter, engine="python")
    elif suffix in {".xlsx", ".xls"}:
        if not auto_delimiter:
            raise ValueError("delimiter cannot be specified for Excel metadata")
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported metadata format: {path}")
    if index_col is not None:
        if isinstance(index_col, int):
            try:
                index_col = df.columns[index_col]
            except IndexError as exc:
                raise IndexError(
                    f"metadata index_col position {index_col} is out of range"
                ) from exc
        df = df.set_index(index_col)
    elif "sample_id" in df.columns:
        df = df.set_index("sample_id")
    elif "barcode" in df.columns:
        df = df.set_index("barcode")
    if df.index.has_duplicates:
        raise ValueError("Metadata contains duplicate sample identifiers.")
    return df


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t" if path.suffix.lower() in {".tsv", ".txt"} else ",", index_col=0)


def load_expression_matrix(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Load a genes-by-samples table or a three-file STARsolo bundle.

    A directory input must contain ``barcodes.tsv.gz``, ``features.tsv.gz``,
    and ``matrix.mtx.gz`` directly inside it.
    """
    path = Path(path)
    if path.is_dir():
        required = [path / "matrix.mtx.gz", path / "features.tsv.gz", path / "barcodes.tsv.gz"]
        if not all(p.exists() for p in required):
            raise FileNotFoundError("STARsolo directory must contain matrix.mtx.gz, features.tsv.gz, and barcodes.tsv.gz")
        with gzip.open(required[1], "rt") as handle:
            features = pd.read_csv(handle, sep="\t", header=None)
        with gzip.open(required[2], "rt") as handle:
            barcodes = [re.sub(r"-1$", "", x.strip()) for x in handle if x.strip()]
        matrix = mmread(required[0]).tocsr()
        if matrix.shape != (len(features), len(barcodes)):
            raise ValueError("STARsolo matrix dimensions do not match features/barcodes.")
        gene_ids = features.iloc[:, 0].astype(str).tolist()
        var = pd.DataFrame(index=gene_ids)
        if features.shape[1] > 1:
            var["gene_symbol"] = features.iloc[:, 1].astype(str).values
        return pd.DataFrame.sparse.from_spmatrix(matrix, index=gene_ids, columns=barcodes), var
    if not path.exists():
        raise FileNotFoundError(path)
    return _read_table(path), None


def _normalized_ids(index: pd.Index) -> pd.Index:
    return pd.Index([re.sub(r"-1$", "", str(x)) for x in index])


def _merged_metadata_label(frame: pd.DataFrame, columns: list[str], fallback: pd.Series) -> pd.Series:
    """Join non-empty metadata components with underscores."""
    parts = []
    for column in columns:
        if column in frame.columns:
            values = frame[column].where(frame[column].notna(), "").astype(str).str.strip()
            parts.append(values.replace({"nan": "", "None": ""}))
        else:
            parts.append(pd.Series("", index=frame.index))
    merged = pd.DataFrame(parts).T.apply(
        lambda row: "_".join(value for value in row if value), axis=1
    )
    return merged.where(merged.ne(""), fallback.astype(str))


def create_drugseq_from_files(
    expression_path: str | Path,
    metadata_path: str | Path,
    *,
    metadata_key: str | int | None = None,
    delimiter: str | None = "auto",
    min_overlap: float = 0.5,
) -> DrugSeqData:
    """Create DrugSeqData from expression and metadata files.

    ``expression_path`` may be a delimited gene-by-sample table or a folder
    containing ``barcodes.tsv.gz``, ``features.tsv.gz``, and
    ``matrix.mtx.gz``. ``delimiter`` applies only to delimited metadata and
    is forwarded to :func:`load_metadata`.
    """
    counts, var = load_expression_matrix(expression_path)
    obs = load_metadata(
        metadata_path,
        index_col=metadata_key,
        delimiter=delimiter,
    )
    count_ids = _normalized_ids(counts.columns)
    meta_ids = _normalized_ids(obs.index)
    obs = obs.copy()
    obs.index = meta_ids
    counts = counts.copy()
    counts.columns = count_ids
    if counts.columns.has_duplicates or obs.index.has_duplicates:
        raise ValueError("Sample identifiers become duplicated after barcode normalization.")
    shared = counts.columns.intersection(obs.index)
    if len(counts.columns) == 0 or len(shared) / len(counts.columns) < min_overlap:
        raise ValueError(f"Only {len(shared) / max(len(counts.columns), 1):.0%} of expression samples match metadata.")
    counts = counts.loc[:, shared]
    obs = obs.loc[shared]
    if "sample_type" not in obs.columns and "compound" in obs.columns:
        labels = obs["compound"].astype(str).str.lower()
        obs["sample_type"] = "treatment"
        obs.loc[labels.isin({"water", "blank", "media"}), "sample_type"] = "media"
        obs.loc[labels.isin({"dmso", "vehicle"}), "sample_type"] = obs.loc[labels.isin({"dmso", "vehicle"}), "compound"].astype(str)
    barcode = (
        obs["barcode"].astype(str).str.replace(r"-1$", "", regex=True)
        if "barcode" in obs.columns
        else pd.Series(obs.index, index=obs.index, dtype=str)
    )
    # Preserve the normalized join key even when load_metadata() used the
    # original barcode column as the DataFrame index.
    obs["barcode"] = barcode.values
    obs["sample_id"] = _merged_metadata_label(
        pd.DataFrame({"plate_id": obs.get("plate_id", ""), "barcode": barcode}, index=obs.index),
        ["plate_id", "barcode"],
        fallback=pd.Series(obs.index, index=obs.index),
    )
    obs["treatment"] = _merged_metadata_label(
        obs,
        ["compound", "dose", "time"],
        fallback=pd.Series("", index=obs.index),
    )
    return create_drugseq_object(counts, obs, var=var, min_overlap=1.0)
