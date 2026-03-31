"""
core.py
-------
DrugSeqData: thin wrapper around AnnData that enforces Drug-seq metadata
conventions and provides a repr, validation, and subsetting interface.

AnnData row-orientation: obs = samples, var = genes  (scanpy convention).
"""

from __future__ import annotations

import warnings
from typing import Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Required / recommended metadata columns
# ---------------------------------------------------------------------------

REQUIRED_OBS_COLS = ["plate_id", "well_id", "compound", "dose", "sample_type"]
RECOMMENDED_OBS_COLS = REQUIRED_OBS_COLS + ["dose_unit"]
VALID_SAMPLE_TYPES = {"treatment", "DMSO", "vehicle", "positive_ctrl",
                      "negative_ctrl", "media"}


# ---------------------------------------------------------------------------
# DrugSeqData
# ---------------------------------------------------------------------------

class DrugSeqData:
    """
    Thin wrapper around :class:`anndata.AnnData` that enforces Drug-seq
    metadata conventions and adds convenience properties.

    The underlying AnnData is always accessible as :attr:`adata` and all
    standard scanpy functions accept it directly.

    Parameters
    ----------
    adata : AnnData
        An AnnData with obs = samples (rows) and var = genes (columns).
        Must contain a ``counts`` layer with raw integer counts.
    """

    def __init__(self, adata: ad.AnnData) -> None:
        self._adata = adata

    # -- pass-through to AnnData -------------------------------------------

    @property
    def adata(self) -> ad.AnnData:
        """The underlying AnnData object."""
        return self._adata

    @property
    def obs(self) -> pd.DataFrame:
        return self._adata.obs

    @property
    def var(self) -> pd.DataFrame:
        return self._adata.var

    @property
    def uns(self) -> dict:
        return self._adata.uns

    @property
    def obsm(self) -> dict:
        return self._adata.obsm

    @property
    def n_obs(self) -> int:
        return self._adata.n_obs

    @property
    def n_vars(self) -> int:
        return self._adata.n_vars

    @property
    def shape(self) -> tuple[int, int]:
        return self._adata.shape

    @property
    def obs_names(self) -> pd.Index:
        return self._adata.obs_names

    @property
    def var_names(self) -> pd.Index:
        return self._adata.var_names

    @property
    def counts(self) -> sp.csr_matrix:
        """Raw count matrix (samples × genes)."""
        if "counts" not in self._adata.layers:
            raise KeyError("'counts' layer not found. Use normalize_counts() first.")
        return self._adata.layers["counts"]

    @property
    def norm_counts(self) -> np.ndarray | sp.csr_matrix:
        """Normalized expression matrix stored in adata.X."""
        return self._adata.X

    # -- subsetting --------------------------------------------------------

    def __getitem__(self, idx) -> "DrugSeqData":
        """Subset by sample (obs) indices or boolean mask."""
        return DrugSeqData(self._adata[idx])

    def subset(self,
               obs_mask: np.ndarray | None = None,
               var_mask: np.ndarray | None = None) -> "DrugSeqData":
        """Return a view with obs and/or var filtered."""
        sub = self._adata
        if obs_mask is not None:
            sub = sub[obs_mask]
        if var_mask is not None:
            sub = sub[:, var_mask]
        return DrugSeqData(sub.copy())

    # -- repr --------------------------------------------------------------

    def __repr__(self) -> str:
        n_plates = self._adata.obs["plate_id"].nunique() \
            if "plate_id" in self._adata.obs else "?"
        n_compounds = self._adata.obs["compound"].nunique() \
            if "compound" in self._adata.obs else "?"
        qc_done = "total_umi" in self._adata.obs
        norm_done = self._adata.uns.get("norm_method", None)
        de_done = len(self._adata.uns.get("de_results", {}))
        reductions = list(self._adata.obsm.keys())

        lines = [
            "DrugSeqData",
            f"  Samples   : {self.n_obs}",
            f"  Genes     : {self.n_vars}",
            f"  Plates    : {n_plates}",
            f"  Compounds : {n_compounds}",
            f"  QC done   : {qc_done}",
            f"  Normalized: {norm_done or 'no'}",
            f"  DE results: {de_done} compound(s)",
        ]
        if reductions:
            lines.append(f"  Reductions: {', '.join(reductions)}")
        return "\n".join(lines)

    def write(self, filename: str | Path, compression: str | None = "gzip", **kwargs) -> None:
            """
            Write the object to an .h5ad file.
            
            Parameters
            ----------
            filename : str or Path
                File path to save the data (e.g., "data.h5ad").
            compression : str or None, default="gzip"
                Compression method for the HDF5 file.
            **kwargs
                Additional arguments passed to `anndata.AnnData.write_h5ad`.
            """
            # (可选) 在保存前打个思想钢印，记录这原本是一个 DrugSeqData 对象
            self._adata.uns["__is_drugseq_data__"] = True 
            
            # 直接调用底层的保存方法
            self._adata.write_h5ad(filename, compression=compression, **kwargs)

    @classmethod
    def read_h5ad(cls, filename: str | Path, backed: str | None = None, **kwargs) -> "DrugSeqData":
        """
        Read an .h5ad file and initialize a DrugSeqData object directly.
        
        Parameters
        ----------
        filename : str or Path
            File path to the .h5ad file.
        backed : str or None, default=None
            If 'r', load AnnData in backed mode (memory efficient).
        **kwargs
            Additional arguments passed to `anndata.read_h5ad`.
            
        Returns
        -------
        DrugSeqData
        """
        # 1. 使用 anndata 原生函数读取文件
        adata = ad.read_h5ad(filename, backed=backed, **kwargs)
        
        # 2. 将读取到的 AnnData 包装成当前的类 (DrugSeqData) 并返回
        return cls(adata)

# ---------------------------------------------------------------------------
# create_drugseq_object
# ---------------------------------------------------------------------------

def create_drugseq_object(
    counts: np.ndarray | sp.spmatrix | pd.DataFrame,
    obs: pd.DataFrame,
    var: pd.DataFrame | None = None,
    min_overlap: float = 0.5,
) -> DrugSeqData:
    """
    Create a :class:`DrugSeqData` from a count matrix and sample metadata.

    Parameters
    ----------
    counts : array-like, shape (n_genes, n_samples) or (n_samples, n_genes)
        Raw integer count matrix.  If a DataFrame, rows are treated as genes
        and columns as samples (matching the R convention).  A numpy/scipy
        matrix may be either orientation — the function auto-detects based
        on index alignment with *obs*.
    obs : pd.DataFrame
        Sample metadata.  Index must match sample IDs in *counts*.
        Recommended columns: plate_id, well_id, compound, dose, dose_unit,
        sample_type.
    var : pd.DataFrame, optional
        Gene metadata.  If None a minimal table with gene IDs is created.
    min_overlap : float
        Minimum fraction of count-column IDs that must appear in obs.index.

    Returns
    -------
    DrugSeqData
    """
    # -- coerce counts to DataFrame (genes × samples) -----------------------
    if isinstance(counts, pd.DataFrame):
        counts_df = counts
    else:
        counts_arr = np.asarray(counts)
        raise TypeError(
            "counts must be a pd.DataFrame with gene row-names and sample "
            "column-names, or provide index/columns explicitly."
        )

    # -- align samples -------------------------------------------------------
    shared = counts_df.columns.intersection(obs.index)
    frac = len(shared) / len(counts_df.columns)
    if frac < min_overlap:
        raise ValueError(
            f"Only {frac:.0%} of count-column IDs appear in obs.index. "
            "Check that colnames(counts) and obs.index use the same format."
        )
    if len(shared) < len(counts_df.columns):
        warnings.warn(
            f"{len(counts_df.columns) - len(shared)} samples in counts have "
            "no matching row in obs and will be dropped."
        )
    counts_df = counts_df[shared]
    obs = obs.loc[shared]

    # -- validate required columns ------------------------------------------
    missing = [c for c in REQUIRED_OBS_COLS if c not in obs.columns]
    if missing:
        warnings.warn(
            f"obs is missing recommended columns: {missing}. "
            "Some functions may not work correctly."
        )

    if "sample_type" in obs.columns:
        bad = set(obs["sample_type"].dropna()) - VALID_SAMPLE_TYPES
        if bad:
            warnings.warn(f"Unexpected sample_type values: {bad}")

    # -- build AnnData (obs = samples, var = genes) -------------------------
    # Transpose: AnnData wants (n_obs, n_vars) = (samples, genes)
    X = sp.csr_matrix(counts_df.values.T.astype(np.float32))

    if var is None:
        var = pd.DataFrame(index=counts_df.index)
        var.index.name = "gene_id"

    adata = ad.AnnData(
        X=X,
        obs=obs.copy(),
        var=var.copy(),
    )
    adata.layers["counts"] = X.copy()
    adata.uns["norm_method"] = None
    adata.uns["de_results"] = {}
    adata.uns["plate_qc"] = {}
    adata.uns["gsea"] = {}

    dsd = DrugSeqData(adata)
    print(
        f"DrugSeqData created: {dsd.n_vars} genes × {dsd.n_obs} samples "
        f"across {obs.get('plate_id', pd.Series()).nunique()} plate(s)."
    )
    return dsd


# ---------------------------------------------------------------------------
# merge_drugseq_objects
# ---------------------------------------------------------------------------

def merge_drugseq_objects(
    *objects: DrugSeqData,
    fill_missing: float = 0.0,
    batch_key: str = "batch",
) -> DrugSeqData:
    """
    Concatenate two or more :class:`DrugSeqData` objects along the obs axis.

    Parameters
    ----------
    *objects : DrugSeqData
        Objects to merge.  Sample IDs must be unique across all objects.
    fill_missing : float
        Value used for genes absent in some objects (default 0).
    batch_key : str
        Key added to obs to track the source object index (default 'batch').

    Returns
    -------
    DrugSeqData
    """
    if len(objects) < 2:
        raise ValueError("Provide at least two DrugSeqData objects.")

    adatas = [o.adata for o in objects]

    # check for duplicate obs_names
    all_ids = [a.obs_names for a in adatas]
    flat = np.concatenate(all_ids)
    if len(flat) != len(set(flat)):
        raise ValueError(
            "Duplicate sample IDs found across objects. "
            "Make sample names unique before merging."
        )

    merged = ad.concat(
        adatas,
        join="outer",
        fill_value=fill_missing,
        label=batch_key,
        keys=[str(i) for i in range(len(adatas))],
    )
    merged.uns = adatas[0].uns.copy()
    merged.uns["de_results"] = {}
    merged.uns["plate_qc"] = {}
    return DrugSeqData(merged)
