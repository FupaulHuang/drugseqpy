"""
enrichment.py
-------------
Gene set enrichment analysis and connectivity scoring.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

from .core import DrugSeqData


# ---------------------------------------------------------------------------
# run_gsea
# ---------------------------------------------------------------------------

def run_gsea(
    dsd: DrugSeqData,
    gene_sets: str | dict | list = "MSigDB_Hallmark_2020",
    compounds: list[str] | None = None,
    rank_by: str = "stat",
    n_perm: int = 1000,
    min_size: int = 15,
    max_size: int = 500,
    species: str = "Human",
    fdr_threshold: float = 0.25,
    inplace: bool = True,
) -> DrugSeqData | None:
    """
    Run GSEA for each compound using gseapy.prerank.

    Parameters
    ----------
    gene_sets : MSigDB shorthand string, dict of gene sets, or path to GMT file
    rank_by   : DE result column used to rank genes (default 'stat')
    inplace   : store results in adata.uns['gsea']
    """
    try:
        import gseapy as gp
    except ImportError:
        raise ImportError("gseapy required. Install: pip install gseapy")

    if not inplace:
        dsd = DrugSeqData(dsd.adata.copy())

    de = dsd.adata.uns.get("de_results", {})
    if not de:
        raise ValueError("de_results empty. Run compute_multi_de() first.")

    if compounds is None:
        compounds = list(de.keys())

    # resolve gene sets
    if isinstance(gene_sets, str):
        gs_dict = gp.get_library(gene_sets, organism=species)
    elif isinstance(gene_sets, dict):
        gs_dict = gene_sets
    else:
        gs_dict = gene_sets  # assume already resolved

    print(f"Running GSEA for {len(compounds)} compound(s), "
          f"{len(gs_dict)} gene sets.")

    gsea_results = {}
    for cmpd in compounds:
        df = de[cmpd]
        col = rank_by if rank_by in df.columns else "logFC"
        ranked = df.dropna(subset=[col]).set_index("gene")[col]\
                   .sort_values(ascending=False)
        try:
            res = gp.prerank(
                rnk=ranked,
                gene_sets=gs_dict,
                permutation_num=n_perm,
                min_size=min_size,
                max_size=max_size,
                outdir=None,
                verbose=False,
            )
            gsea_results[cmpd] = res.res2d
        except Exception as e:
            warnings.warn(f"  GSEA failed for '{cmpd}': {e}")

    dsd.adata.uns["gsea"] = gsea_results
    print(f"GSEA complete for {len(gsea_results)} compound(s).")
    return dsd if not inplace else None


# ---------------------------------------------------------------------------
# run_ora
# ---------------------------------------------------------------------------

def run_ora(
    dsd: DrugSeqData,
    compound: str,
    gene_sets: str | dict = "MSigDB_Hallmark_2020",
    fdr_threshold: float = 0.05,
    lfc_threshold: float = 0.5,
    species: str = "Human",
) -> pd.DataFrame:
    """
    Over-representation analysis (Fisher's exact test) for significant DE genes.
    """
    try:
        import gseapy as gp
    except ImportError:
        raise ImportError("gseapy required. Install: pip install gseapy")

    de = dsd.adata.uns.get("de_results", {})
    if compound not in de:
        raise ValueError(f"'{compound}' not in de_results.")

    df = de[compound]
    sig_genes = df.loc[
        df["padj"].notna() &
        (df["padj"] < fdr_threshold) &
        (df["logFC"].abs() >= lfc_threshold),
        "gene"
    ].tolist()

    if not sig_genes:
        warnings.warn(f"No significant genes for '{compound}' at given thresholds.")
        return pd.DataFrame()

    if isinstance(gene_sets, str):
        gs_dict = gp.get_library(gene_sets, organism=species)
    else:
        gs_dict = gene_sets

    res = gp.enrichr(
        gene_list=sig_genes,
        gene_sets=gs_dict,
        outdir=None,
        verbose=False,
    )
    return res.res2d


# ---------------------------------------------------------------------------
# connectivity_score
# ---------------------------------------------------------------------------

def connectivity_score(
    dsd: DrugSeqData,
    compounds: list[str] | None = None,
    method: str = "cosine",
    rank_by: str = "logFC",
    n_genes: int = 250,
    reference: np.ndarray | pd.DataFrame | None = None,
) -> np.ndarray:
    """
    Compute pairwise compound-compound connectivity scores.

    Parameters
    ----------
    method : 'cosine' or 'pearson'
    n_genes : number of up + down landmark genes per compound
    reference : optional external signature matrix (genes × n_signatures)

    Returns
    -------
    Symmetric matrix (n_compounds × n_compounds) or
    (n_compounds × n_signatures) if reference is supplied.
    """
    de = dsd.adata.uns.get("de_results", {})
    if not de:
        raise ValueError("de_results empty.")

    if compounds is None:
        compounds = list(de.keys())

    # align to common gene set
    all_genes = None
    for c in compounds:
        genes_c = set(de[c]["gene"])
        all_genes = genes_c if all_genes is None else all_genes & genes_c
    all_genes = sorted(all_genes)

    # optionally reduce to landmark genes
    if np.isfinite(n_genes) and len(all_genes) > n_genes * 2:
        mean_abs_lfc = pd.Series(
            {g: np.mean([abs(de[c].set_index("gene").get(rank_by, pd.Series())[g])
                         for c in compounds if g in de[c]["gene"].values])
             for g in all_genes}
        )
        all_genes = mean_abs_lfc.nlargest(n_genes * 2).index.tolist()

    sig_mat = np.column_stack([
        de[c].set_index("gene").reindex(all_genes)[rank_by].fillna(0).values
        for c in compounds
    ])  # (n_genes, n_compounds)

    if reference is not None:
        if isinstance(reference, pd.DataFrame):
            ref_genes = reference.index.intersection(all_genes)
            reference = reference.loc[ref_genes].values
            sig_mat   = sig_mat[[all_genes.index(g) for g in ref_genes], :]
        B = reference
    else:
        B = sig_mat

    if method == "cosine":
        norm_A = sig_mat / (np.linalg.norm(sig_mat, axis=0, keepdims=True) + 1e-10)
        norm_B = B       / (np.linalg.norm(B,       axis=0, keepdims=True) + 1e-10)
        return norm_A.T @ norm_B
    else:
        return np.corrcoef(sig_mat.T, B.T)[:len(compounds), len(compounds):]
