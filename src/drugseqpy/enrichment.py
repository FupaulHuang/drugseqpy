"""
enrichment.py
-------------
Gene set enrichment analysis and connectivity scoring.
"""

from __future__ import annotations

import warnings
from pathlib import Path

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
            df[col] = ["" if (v is None or
                             (isinstance(v, float) and np.isnan(v)))
                       else str(v) for v in df[col]]

    return df


# ---------------------------------------------------------------------------
# Canonical pathway libraries and GMT resolution
# ---------------------------------------------------------------------------

# Canonical library keys -> Enrichr/MSigDB library names (as used in the
# reference drug-treatment scripts).
CANONICAL_LIBRARIES = {
    "hallmark": "MSigDB_Hallmark_2020",
    "go_bp": "GO_Biological_Process_2023",
    "reactome": "Reactome_2022",
    "kegg": "KEGG_2021_Human",
}

DEFAULT_CACHE_DIR = "~/.cache/drugseqpy/gene_sets"


def _filter_gene_sets(gs_dict: dict, min_size: int, max_size: int) -> dict:
    """Keep gene sets whose size falls inside [min_size, max_size]."""
    normalized = {}
    for name, genes in gs_dict.items():
        # GMT and inline dictionaries often contain blank/duplicated entries;
        # normalize them once at the resolver boundary while preserving the
        # original identifier spelling for downstream matching/reporting.
        clean_name = str(name).strip()
        if not clean_name:
            continue
        if isinstance(genes, str) or not hasattr(genes, "__iter__"):
            genes = [genes]
        seen = set()
        clean_genes = []
        for gene in genes:
            if gene is None:
                continue
            clean_gene = str(gene).strip()
            dedup_key = clean_gene.upper()
            if clean_gene and dedup_key not in seen:
                seen.add(dedup_key)
                clean_genes.append(clean_gene)
        if min_size <= len(clean_genes) <= max_size:
            normalized[clean_name] = clean_genes
    return normalized


def _analysis_gene_sets(gs_dict: dict) -> dict[str, list[str]]:
    """Return case-insensitive gene identifiers for enrichment matching."""
    out = {}
    for name, genes in gs_dict.items():
        if isinstance(genes, str) or not hasattr(genes, "__iter__"):
            genes = [genes]
        seen = set()
        normalized = []
        for gene in genes:
            key = str(gene).strip().upper()
            if key and key not in seen:
                seen.add(key)
                normalized.append(key)
        out[str(name).strip()] = normalized
    return out


def _analysis_ranking(
    df: pd.DataFrame,
    column: str,
    gene_column: str = "gene",
) -> pd.Series:
    """Build a deterministic, case-insensitive ranked gene vector."""
    ranked = df.dropna(subset=[column])[[gene_column, column]].copy()
    ranked[gene_column] = (
        ranked[gene_column].astype(str).str.strip().str.upper()
    )
    ranked[column] = pd.to_numeric(ranked[column], errors="coerce")
    ranked = ranked.dropna(subset=[column])
    # Duplicate gene identifiers are collapsed deterministically before GSEA.
    ranked = ranked.groupby(gene_column, sort=True)[column].mean()
    return ranked.sort_values(ascending=False)


def _enrichment_gene_ids(dsd: DrugSeqData, genes: pd.Series) -> pd.Series:
    """Map DE feature IDs to gene symbols when the object provides them.

    Differential-expression tables intentionally retain stable feature IDs,
    which are commonly Ensembl IDs in STARsolo output.  Public pathway GMT
    files generally use gene symbols.  Resolve that boundary here while
    falling back to the original feature ID for missing/blank annotations.
    """
    gene_ids = genes.astype(str).str.strip()
    if "gene_symbol" not in dsd.var.columns:
        return gene_ids

    symbols = dsd.var["gene_symbol"].astype("string")
    symbols.index = dsd.var_names.astype(str)
    mapped = gene_ids.map(symbols)
    valid = (
        mapped.notna()
        & mapped.astype("string").str.strip().ne("")
        & ~mapped.astype("string").str.lower().isin({"nan", "none"})
    )
    return mapped.astype("string").str.strip().where(valid, gene_ids)


def _load_gmt(path) -> dict[str, list[str]]:
    """Pure-python GMT parser (set name, description, gene1, gene2, ...)."""
    gs = {}
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            gs[parts[0]] = [g for g in parts[2:] if g]
    return gs


def _save_gmt(path, gs_dict: dict) -> None:
    """Write a gene-set dict as a GMT file for offline reuse."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for name, genes in gs_dict.items():
            if isinstance(genes, str):
                genes = [genes]
            fh.write("\t".join([str(name), str(name),
                                 *(str(gene) for gene in genes)]) + "\n")


def resolve_gene_sets(
    libraries: str | Path | list[str | Path] | tuple[str | Path, ...] | dict,
    gene_set_paths: dict | None = None,
    species: str = "Human",
    cache_dir: str | None = None,
    min_size: int = 15,
    max_size: int = 500,
    allow_network: bool = False,
) -> dict[str, dict[str, list[str]]]:
    """
    Resolve pathway libraries into gene-set dicts.

    *libraries* entries may be canonical keys (``'hallmark'``, ``'go_bp'``,
    ``'reactome'``, ``'kegg'``), gseapy/Enrichr library names, or ready-made
    gene-set dicts passed inline.

    Resolution order per library:
    1. a local GMT path provided via ``gene_set_paths`` (offline);
    2. a gene-set dict passed inline;
    3. gseapy ``get_library`` (network), cached as GMT in ``cache_dir`` so
       later runs are offline.  Requires ``allow_network=True``.

    Missing libraries raise a ``ValueError`` listing them and how to fix.
    """
    gene_set_paths = gene_set_paths or {}
    resolved: dict[str, dict[str, list[str]]] = {}
    missing: list[str] = []

    if libraries is None:
        raise ValueError("At least one pathway library is required")
    if min_size < 1 or max_size < min_size:
        raise ValueError("gene-set sizes must satisfy 1 <= min_size <= max_size")
    if isinstance(libraries, (str, Path)):
        entries = list(CANONICAL_LIBRARIES) if (
            isinstance(libraries, str) and libraries.strip().lower() == "all"
        ) else [libraries]
    elif isinstance(libraries, dict):
        # A mapping is normally the labelled form {"label": {set: genes}}.
        # Also accept a bare gene-set dictionary {set: [genes]} as one custom
        # library, which is convenient for programmatic callers.
        def _is_gene_collection(value):
            if isinstance(value, (list, tuple, set, frozenset, np.ndarray,
                                  pd.Index, pd.Series)):
                return True
            if isinstance(value, str):
                path_value = Path(value)
                return not (path_value.suffix.lower() == ".gmt" or
                            path_value.exists())
            return False

        if libraries and all(_is_gene_collection(value)
                             for value in libraries.values()):
            entries = [{"inline": libraries}]
        else:
            entries = [{key: value} for key, value in libraries.items()]
    else:
        entries = list(libraries)
    if not entries:
        raise ValueError("At least one pathway library is required")
    if len(entries) == 1 and isinstance(entries[0], str) \
            and entries[0].strip().lower() == "all":
        entries = list(CANONICAL_LIBRARIES)

    canonical_by_lower = {key.lower(): value
                          for key, value in CANONICAL_LIBRARIES.items()}

    for entry in entries:
        if isinstance(entry, dict):
            for inline_key, inline_value in entry.items():
                if isinstance(inline_value, dict):
                    resolved[str(inline_key)] = _filter_gene_sets(
                        dict(inline_value),
                        min_size, max_size,
                    )
                    continue
                if isinstance(inline_value, (str, Path)):
                    key = str(inline_key)
                    try:
                        gs = _load_gmt(inline_value)
                    except (OSError, FileNotFoundError, TypeError, ValueError):
                        gs = None
                    if gs is None:
                        missing.append(key)
                    else:
                        resolved[key] = _filter_gene_sets(
                            gs, min_size, max_size,
                        )
                    continue
                raise ValueError(
                    f"Unsupported library entry: {entry!r}. Inline dicts "
                    "must map {label: gene_sets_dict} or {label: gmt_path}."
                )
            continue

        if not isinstance(entry, (str, Path)):
            raise ValueError(f"Unsupported library entry: {entry!r}")

        entry_path = Path(entry)
        if entry_path.is_file():
            key = entry_path.stem
            label = key
            path = entry_path
        else:
            raw_key = str(entry)
            key = raw_key
            label = canonical_by_lower.get(raw_key.lower(), raw_key)
            # Canonical keys should remain stable and lower-case in storage.
            if raw_key.lower() in canonical_by_lower:
                key = raw_key.lower()
            path_lookup = {str(k).lower(): value
                           for k, value in gene_set_paths.items()}
            path = next(
                (path_lookup[candidate.lower()] for candidate in
                 (raw_key, label, key) if candidate.lower() in path_lookup),
                None,
            )

        if path is not None:
            try:
                gs = _load_gmt(path)
            except (OSError, FileNotFoundError, TypeError, ValueError):
                gs = None
        else:
            gs = _fetch_library(
                label, species=species, cache_dir=cache_dir,
                allow_network=allow_network,
            )
        if gs is None:
            missing.append(key)
            continue
        resolved[key] = _filter_gene_sets(gs, min_size, max_size)

    if missing:
        raise ValueError(
            f"Could not resolve pathway libraries: {missing}. "
            "Provide local GMT files via gene_set_paths={library: path} "
            "or allow network access (allow_network=True) with gseapy "
            "installed."
        )
    return resolved


def _fetch_library(library_name: str, species: str, cache_dir: str | None,
                   allow_network: bool) -> dict | None:
    """Fetch a library via gseapy, caching the GMT locally."""
    cache = Path(cache_dir or DEFAULT_CACHE_DIR).expanduser()
    gmt_path = cache / f"{library_name}.gmt"

    if gmt_path.exists():
        gs = _load_gmt(gmt_path)
        if gs:
            return gs

    if not allow_network:
        return None

    try:
        import gseapy as gp
    except ImportError:
        raise ImportError(
            "gseapy is required to resolve pathway libraries over the "
            "network. Install: pip install gseapy (or pass gene_set_paths)."
        )

    cache.mkdir(parents=True, exist_ok=True)

    try:
        gs = gp.get_library(name=library_name, organism=species)
        if not gs:
            return None
        _save_gmt(gmt_path, gs)
        return gs
    except Exception as e:
        warnings.warn(f"  Library '{library_name}' fetch failed: {e}")
        return None


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
    allow_network: bool = False,
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

    if not isinstance(dsd, DrugSeqData):
        dsd = DrugSeqData(dsd)
    if not inplace:
        dsd = DrugSeqData(dsd.adata.copy())

    de = _comparison_tables(dsd)

    if compounds is None:
        compounds = list(de.keys())
    elif isinstance(compounds, str):
        compounds = [compounds]

    # resolve gene sets (canonical keys, gseapy names, GMT paths, or dicts)
    if isinstance(gene_sets, (str, Path)):
        if Path(gene_sets).is_file():
            gs_dict = _filter_gene_sets(
                _load_gmt(gene_sets), min_size=min_size, max_size=max_size,
            )
        else:
            gs_dict = resolve_gene_sets(
                [gene_sets], species=species, min_size=min_size,
                max_size=max_size, allow_network=allow_network,
            )
            gs_dict = next(iter(gs_dict.values()))
    elif isinstance(gene_sets, dict):
        gs_dict = _filter_gene_sets(
            gene_sets, min_size=min_size, max_size=max_size,
        )
    else:
        resolved = resolve_gene_sets(
            gene_sets, species=species, min_size=min_size,
            max_size=max_size, allow_network=allow_network,
        )
        gs_dict = next(iter(resolved.values()))
    gs_dict = _analysis_gene_sets(gs_dict)

    print(f"Running GSEA for {len(compounds)} compound(s), "
          f"{len(gs_dict)} gene sets.")

    gsea_results = {}
    for cmpd in compounds:
        key = _comparison_key(de, cmpd)
        if key is None:
            warnings.warn(f"  '{cmpd}' not found in comparison results. Skipping.")
            continue
        df = de[key]
        col = rank_by if rank_by in df.columns else "logFC"
        ranked = _analysis_ranking(df, col)
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
    allow_network: bool = False,
) -> pd.DataFrame:
    """
    Over-representation analysis (Fisher's exact test) for significant DE genes.
    """
    if not isinstance(dsd, DrugSeqData):
        dsd = DrugSeqData(dsd)
    try:
        import gseapy as gp
    except ImportError:
        raise ImportError("gseapy required. Install: pip install gseapy")

    de = _comparison_tables(dsd)
    key = _comparison_key(de, compound)
    if key is None:
        raise ValueError(f"'{compound}' not in comparison results.")

    df = de[key]
    sig_genes = df.loc[
        df["padj"].notna() &
        (df["padj"] < fdr_threshold) &
        (df["logFC"].abs() >= lfc_threshold),
        "gene"
    ].astype(str).str.strip().str.upper().drop_duplicates().tolist()

    if not sig_genes:
        warnings.warn(f"No significant genes for '{compound}' at given thresholds.")
        return pd.DataFrame()

    if isinstance(gene_sets, (str, Path)):
        if Path(gene_sets).is_file():
            gs_dict = _load_gmt(gene_sets)
        else:
            resolved = resolve_gene_sets(
                [gene_sets], species=species, allow_network=allow_network,
            )
            gs_dict = next(iter(resolved.values()))
    else:
        gs_dict = _filter_gene_sets(
            gene_sets, min_size=1, max_size=max(len(sig_genes), 1_000_000),
        )
    gs_dict = _analysis_gene_sets(gs_dict)

    universe = df["gene"].astype(str).str.strip().str.upper()\
               .drop_duplicates().tolist()
    res = gp.enrich(
        gene_list=sig_genes,
        gene_sets=gs_dict,
        background=universe,
        outdir=None,
        no_plot=True,
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
    if not isinstance(dsd, DrugSeqData):
        dsd = DrugSeqData(dsd)
    de = dsd.adata.uns.get("comparison_results") or \
         dsd.adata.uns.get("de_results", {})
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
    ontologies: list[str] | str = (
        "GO_Biological_Process_2023",
        "GO_Molecular_Function_2023",
        "GO_Cellular_Component_2023"
    ),
    fdr_threshold: float = 0.05,
    lfc_threshold: float = 0.5,
    species: str = "Human",
    allow_network: bool = False,
    gene_set_paths: dict | None = None,
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
    allow_network : permit gseapy/Enrichr library downloads when no local
        GMT or cache entry is available (default ``False``).
    gene_set_paths : optional mapping from ontology/library name to local GMT.
    inplace : If True, stores results in dsd.adata.uns['go']. If False, returns a copied object.

    Returns
    -------
    DrugSeqData (if inplace=False) or None (if inplace=True).
    """
    try:
        import gseapy as gp
    except ImportError:
        raise ImportError("gseapy required. Install: pip install gseapy")

    if not isinstance(dsd, DrugSeqData):
        dsd = DrugSeqData(dsd)

    # 处理 inplace 逻辑
    if not inplace:
        # 注意：这里假设你的 DrugSeqData 已经 import 或者在同一个文件中
        dsd = type(dsd)(dsd.adata.copy())

    de = _comparison_tables(dsd)

    # 格式化输入参数
    if compounds is None:
        compounds = list(de.keys())
    elif isinstance(compounds, str):
        compounds = [compounds]

    if isinstance(ontologies, str):
        ontologies = [ontologies]

    resolved = resolve_gene_sets(
        ontologies, gene_set_paths=gene_set_paths, species=species,
        min_size=1, max_size=1_000_000, allow_network=allow_network,
    )
    gene_sets = {}
    for library_sets in resolved.values():
        gene_sets.update(_analysis_gene_sets(library_sets))

    print(f"Running GO enrichment for {len(compounds)} compound(s)...")

    # 获取或初始化 uns 中的 go 字典
    go_results = dsd.adata.uns.get("go", {})

    for cmpd in compounds:
        key = _comparison_key(de, cmpd)
        if key is None:
            warnings.warn(f"'{cmpd}' not found in comparison results. Skipping.")
            continue

        df = de[key]
        
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
            universe = (
                df["gene"].astype(str).str.strip().str.upper()
                .drop_duplicates().tolist()
            )
            res = gp.enrich(
                gene_list=(pd.Series(sig_genes).astype(str).str.upper()
                           .drop_duplicates().tolist()),
                gene_sets=gene_sets,
                background=universe,
                outdir=None,
                no_plot=True,
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


# ---------------------------------------------------------------------------
# run_enrichment  (canonical workflow over comparison results)
# ---------------------------------------------------------------------------

ORA_COLUMNS = ["Term", "Overlap", "P-value", "Adjusted P-value",
               "Odds Ratio", "Combined Score", "Genes"]
GSEA_COLUMNS = ["Name", "Term", "ES", "NES", "NOM p-val", "FDR q-val",
                "FWER p-val", "Tag %", "Gene %", "Lead_genes"]
ENRICHMENT_SUMMARY_COLUMNS = [
    "contrast", "library", "mode", "direction", "pathway", "overlap",
    "genes", "effect", "pvalue", "padj",
]


def _empty_enrichment(schema: list[str] | None = None) -> pd.DataFrame:
    """Empty result frame, optionally matching a known column schema."""
    cols = schema or ORA_COLUMNS
    return pd.DataFrame(columns=cols)


def _comparison_tables(dsd) -> dict[str, pd.DataFrame]:
    """Comparison tables, preferring the new comparison_results storage."""
    tables = dsd.adata.uns.get("comparison_results") or \
        dsd.adata.uns.get("de_results", {})
    if not tables:
        raise ValueError(
            "comparison_results empty. Run compute_multi_de() first."
        )
    return tables


def _comparison_key(tables: dict, requested: str) -> str | None:
    """Resolve an exact or base contrast key from comparison storage."""
    if requested in tables:
        return requested
    matches = [key for key in tables
               if str(key).startswith(f"{requested}::")]
    if not matches:
        return None
    return matches[0]


def run_enrichment(
    dsd: DrugSeqData,
    contrasts: list[str] | None = None,
    libraries: list[str] | tuple[str, ...] = ("hallmark", "go_bp", "reactome", "kegg"),
    gene_set_paths: dict | None = None,
    mode: tuple[str, ...] | list[str] = ("gsea", "ora"),
    rank_by: str = "stat",
    fdr_threshold: float = 0.05,
    lfc_threshold: float = 0.5,
    n_perm: int = 1000,
    min_size: int = 15,
    max_size: int = 500,
    species: str = "Human",
    background: str = "all_tested",
    seed: int = 42,
    cache_dir: str | None = None,
    allow_network: bool = False,
    inplace: bool = True,
) -> DrugSeqData | None:
    """
    Run preranked GSEA and/or directional ORA for each contrast and each
    requested pathway library (canonical keys: ``'hallmark'``, ``'go_bp'``,
    ``'reactome'``, ``'kegg'``).

    Results are stored under
    ``adata.uns['enrichment_results'][contrast][library]`` with keys
    ``'gsea'``, ``'ora_up'`` and ``'ora_down'``.  Empty DataFrames (same
    schema) are stored when a direction has no significant genes.

    Parameters
    ----------
    dsd : DrugSeqData with comparison results.
    contrasts : contrast keys to analyze (None = all stored contrasts).
    libraries : canonical keys, gseapy library names, or inline gene-set dicts.
    gene_set_paths : dict mapping library -> local GMT path (offline mode).
    mode : subset of ``('gsea', 'ora')``.
    rank_by : DE column used to rank genes for GSEA.
    background : ``'all_tested'`` (default) or ``'library_genes'`` — gene
        universe used for ORA.
    seed : deterministic seed passed to gseapy.
    """
    if not isinstance(dsd, DrugSeqData):
        dsd = DrugSeqData(dsd)
    try:
        import gseapy as gp
    except ImportError:
        raise ImportError("gseapy required. Install: pip install gseapy")

    if not inplace:
        dsd = DrugSeqData(dsd.adata.copy())

    tables = _comparison_tables(dsd)
    if contrasts is None:
        selected_contrasts = list(tables.keys())
    else:
        requested_contrasts = ([contrasts] if isinstance(contrasts, str)
                               else list(contrasts))
        selected_contrasts = []
        for requested in requested_contrasts:
            if requested in tables:
                selected_contrasts.append(requested)
                continue
            selected_contrasts.extend(
                key for key in tables
                if str(key).startswith(f"{requested}::")
            )

    mode = (mode,) if isinstance(mode, str) else tuple(mode)
    if not mode:
        raise ValueError("mode must include 'gsea', 'ora', or both")
    invalid_modes = set(mode) - {"gsea", "ora"}
    if invalid_modes:
        raise ValueError(
            f"mode must contain only 'gsea' and/or 'ora'; got {sorted(invalid_modes)}"
        )

    if libraries is None:
        library_spec = list(CANONICAL_LIBRARIES)
    elif isinstance(libraries, (str, Path, dict)):
        library_spec = libraries
    else:
        library_spec = list(libraries)
    resolved = resolve_gene_sets(
        library_spec, gene_set_paths=gene_set_paths, species=species,
        cache_dir=cache_dir, min_size=min_size, max_size=max_size,
        allow_network=allow_network,
    )
    if not resolved:
        raise ValueError("At least one pathway library is required")

    if background not in {"all_tested", "library_genes"}:
        raise ValueError(
            "background must be 'all_tested' or 'library_genes'"
        )
    enr_results = dsd.adata.uns.get("enrichment_results", {})

    for key in selected_contrasts:
        if key not in tables:
            warnings.warn(f"  Contrast '{key}' not in comparison results. Skipping.")
            continue
        de = tables[key]
        analysis_de = de.copy()
        analysis_de["_enrichment_gene"] = _enrichment_gene_ids(
            dsd, de["gene"]
        )
        if "significant" in de.columns:
            # DE backends record the cutoffs used to define significance;
            # retain that explicit classification for downstream enrichment.
            sig_mask = de["significant"].fillna(False).astype(bool)
        elif {"padj", "logFC"}.issubset(de.columns):
            sig_mask = de["padj"].notna() & (de["padj"] < fdr_threshold) & \
                (de["logFC"].abs() >= lfc_threshold)
        else:
            sig_mask = pd.Series(False, index=de.index)
        up_genes = analysis_de.loc[
            sig_mask & (de["logFC"] > 0), "_enrichment_gene"
        ]\
                    .astype(str).str.strip().str.upper().drop_duplicates().tolist()
        down_genes = analysis_de.loc[
            sig_mask & (de["logFC"] < 0), "_enrichment_gene"
        ]\
                      .astype(str).str.strip().str.upper().drop_duplicates().tolist()
        tested_genes = analysis_de["_enrichment_gene"]\
            .astype(str).str.strip().str.upper().drop_duplicates().tolist()

        contrast_res = enr_results.setdefault(key, {})
        # Remove libraries from a previous run that are not part of this
        # requested configuration, while retaining results for unselected
        # contrasts.
        for old_library in list(contrast_res):
            if old_library not in resolved:
                del contrast_res[old_library]
        for lib_key, gs_dict in resolved.items():
            lib_res = contrast_res.setdefault(lib_key, {})
            analysis_sets = _analysis_gene_sets(gs_dict)
            if "gsea" not in mode:
                lib_res.pop("gsea", None)
            if "ora" not in mode:
                lib_res.pop("ora_up", None)
                lib_res.pop("ora_down", None)

            if "gsea" in mode:
                col = rank_by if rank_by in de.columns else "logFC"
                ranked = _analysis_ranking(
                    analysis_de, col, gene_column="_enrichment_gene"
                )
                try:
                    gsea = gp.prerank(
                        rnk=ranked, gene_sets=analysis_sets,
                        permutation_num=n_perm,
                        min_size=min_size, max_size=max_size, outdir=None,
                        verbose=False, seed=seed,
                    )
                    lib_res["gsea"] = _sanitize_df_for_h5ad(gsea.res2d)
                except Exception as e:
                    warnings.warn(
                        f"  GSEA failed for '{key}' [{lib_key}]: {e}"
                    )
                    lib_res["gsea"] = _empty_enrichment(GSEA_COLUMNS)

            if "ora" in mode:
                universe = tested_genes if background == "all_tested" else \
                    sorted({g for genes in analysis_sets.values() for g in genes})
                for direction, genes in (("up", up_genes), ("down", down_genes)):
                    slot = f"ora_{direction}"
                    if not genes:
                        lib_res[slot] = _empty_enrichment()
                        continue
                    try:
                        ora = gp.enrich(
                            gene_list=genes, gene_sets=analysis_sets,
                            background=universe, outdir=None, no_plot=True,
                            verbose=False,
                        )
                        lib_res[slot] = _sanitize_df_for_h5ad(ora.res2d)
                    except Exception as e:
                        warnings.warn(
                            f"  ORA({direction}) failed for '{key}' "
                            f"[{lib_key}]: {e}"
                        )
                        lib_res[slot] = _empty_enrichment()

    dsd.adata.uns["enrichment_results"] = enr_results
    # Persist the flattened table alongside the nested result structure so
    # report builders and H5AD consumers can read one tidy pathway table
    # without re-traversing the nested dictionaries.
    dsd.adata.uns["enrichment_summary"] = _sanitize_df_for_h5ad(
        enrichment_summary(dsd)
    )
    print(f"Enrichment complete for {len(selected_contrasts)} contrast(s) × "
          f"{len(resolved)} librar{'y' if len(resolved) == 1 else 'ies'}.")
    return dsd if not inplace else None


def enrichment_summary(dsd: DrugSeqData) -> pd.DataFrame:
    """
    Flatten ``uns['enrichment_results']`` into a tidy pathway summary table.

    Columns: contrast, library, mode, direction, pathway, overlap, genes,
    effect (NES for GSEA / odds ratio for ORA), pvalue, padj.
    """
    if not isinstance(dsd, DrugSeqData):
        dsd = DrugSeqData(dsd)
    enr = dsd.adata.uns.get("enrichment_results", {})
    rows = []
    for contrast, libs in enr.items():
        for library, slots in libs.items():
            for slot, df in slots.items():
                if df is None or df.empty:
                    continue
                mode_name, direction = ("gsea", "") if slot == "gsea" else \
                    (slot[:3], slot.split("_")[-1])
                for _, r in df.iterrows():
                    rows.append({
                        "contrast": contrast,
                        "library": library,
                        "mode": mode_name,
                        "direction": direction,
                        "pathway": r.get("Term", r.get("Name", "")),
                        "overlap": r.get("Overlap", r.get("Tag %", "")),
                        "genes": r.get("Genes", r.get("Lead_genes", "")),
                        "effect": r.get("Odds Ratio",
                                        r.get("NES", np.nan)),
                        "pvalue": r.get("P-value", r.get("NOM p-val", np.nan)),
                        "padj": r.get("Adjusted P-value",
                                      r.get("FDR q-val", np.nan)),
                    })
    return pd.DataFrame(rows, columns=ENRICHMENT_SUMMARY_COLUMNS)
