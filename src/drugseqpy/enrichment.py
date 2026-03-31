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

def _sanitize_df_for_h5ad(df: pd.DataFrame) -> pd.DataFrame:
    """
    清洗 DataFrame，使其格式能够被 anndata.write_h5ad() 安全保存。
    参考 Scanpy 处理复杂 metadata 的逻辑。
    """
    if df is None or df.empty:
        return df
        
    df = df.copy()
    for col in df.columns:
        # 1. 检查并处理列表/元组（例如 GSEA 结果中的 leading_edge 基因集）
        has_list = df[col].apply(lambda x: isinstance(x, (list, tuple, np.ndarray))).any()
        if has_list:
            df[col] = df[col].apply(
                lambda x: ";".join(map(str, x)) if isinstance(x, (list, tuple, np.ndarray)) else x
            )
            
        # 2. 将 object 类型的列（通常包含混合类型的字符串和 NaN）强制转换为纯字符串
        if df[col].dtype == "object":
            df[col] = df[col].fillna("").astype(str)
            
    return df


# ---------------------------------------------------------------------------
# run_gsea
# ---------------------------------------------------------------------------

def run_gsea(
    dsd: DrugSeqData, # 这里假设已经引入了相关的类型提示
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
            # 🚀 在存入前清洗数据
            clean_df = _sanitize_df_for_h5ad(res.res2d)
            gsea_results[cmpd] = clean_df
            
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
    
    # 🚀 在返回前清洗数据
    clean_df = _sanitize_df_for_h5ad(res.res2d)
    return clean_df


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
    
import warnings
import pandas as pd
import numpy as np

# 确保 _sanitize_df_for_h5ad 已经在这个文件顶部定义过了
# from .utils import _sanitize_df_for_h5ad 

def run_go_enrichment(
    dsd: "DrugSeqData",
    compounds: list[str] | str | None = None,
    ontologies: list[str] | str = [
        "GO_Biological_Process_2023",
        "GO_Molecular_Function_2023",
        "GO_Cellular_Component_2023"
    ],
    fdr_threshold: float = 0.05,
    lfc_threshold: float = 0.5,
    species: str = "Human",
    inplace: bool = True,
) -> "DrugSeqData" | None:
    """
    Run Gene Ontology (GO) enrichment analysis for significantly differentially expressed genes.

    Parameters
    ----------
    dsd : DrugSeqData object containing DE results.
    compounds : List of compound names to analyze. If None, runs all compounds in de_results.
    ontologies : List of GO databases to query (from Enrichr).
    fdr_threshold : Adjusted p-value cutoff for defining significant genes.
    lfc_threshold : Absolute log2 fold-change cutoff for defining significant genes.
    species : Organism name ('Human', 'Mouse', etc.).
    inplace : If True, stores results in dsd.adata.uns['go']. If False, returns a copied object.

    Returns
    -------
    DrugSeqData (if inplace=False) or None (if inplace=True).
    """
    try:
        import gseapy as gp
    except ImportError:
        raise ImportError("gseapy required. Install: pip install gseapy")

    # 处理 inplace 逻辑
    if not inplace:
        # 注意：这里假设你的 DrugSeqData 已经 import 或者在同一个文件中
        dsd = type(dsd)(dsd.adata.copy())

    de = dsd.adata.uns.get("de_results", {})
    if not de:
        raise ValueError("de_results empty. Run DE analysis first.")

    # 格式化输入参数
    if compounds is None:
        compounds = list(de.keys())
    elif isinstance(compounds, str):
        compounds = [compounds]

    if isinstance(ontologies, str):
        ontologies = [ontologies]

    print(f"Running GO enrichment for {len(compounds)} compound(s)...")

    # 获取或初始化 uns 中的 go 字典
    go_results = dsd.adata.uns.get("go", {})

    for cmpd in compounds:
        if cmpd not in de:
            warnings.warn(f"'{cmpd}' not found in de_results. Skipping.")
            continue

        df = de[cmpd]
        
        # 提取显著差异表达基因
        sig_genes = df.loc[
            df["padj"].notna() &
            (df["padj"] < fdr_threshold) &
            (df["logFC"].abs() >= lfc_threshold),
            "gene"
        ].tolist()

        if not sig_genes:
            warnings.warn(f"  No significant genes for '{cmpd}' at given thresholds. Skipping.")
            continue

        try:
            # 调用 gseapy 的 Enrichr 接口
            res = gp.enrichr(
                gene_list=sig_genes,
                gene_sets=ontologies,
                organism=species,
                outdir=None,  # 不在本地生成冗余的文件夹
                verbose=False,
            )
            
            # 🚀 核心步骤：清洗 DataFrame，把 list 转成字符串，防止 HDF5 报错
            clean_df = _sanitize_df_for_h5ad(res.res2d)
            
            go_results[cmpd] = clean_df
            
        except Exception as e:
            warnings.warn(f"  GO enrichment failed for '{cmpd}': {e}")

    # 将结果保存回 AnnData 对象
    dsd.adata.uns["go"] = go_results
    print(f"GO enrichment complete. Results stored in adata.uns['go'].")
    
    return dsd if not inplace else None
