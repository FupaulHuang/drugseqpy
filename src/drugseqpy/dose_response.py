"""
dose_response.py
----------------
Dose-response curve fitting using lmfit (Python's equivalent of drc).

Models supported
----------------
LL4   : 4-parameter log-logistic (Hill equation)
W14   : 4-parameter Weibull type 1
W24   : 4-parameter Weibull type 2

For each gene the best model is selected by AIC when model_selection=True.
EC50 confidence intervals are computed via lmfit's built-in CI propagation.
"""

from __future__ import annotations

import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from lmfit import Model, Parameters
import seaborn as sns
from typing import List, Union, Optional

from .core import DrugSeqData



# ---------------------------------------------------------------------------
# Model functions
# ---------------------------------------------------------------------------

def _ll4(dose, b, c, d, e):
    """4-parameter log-logistic (Hill equation).
    b=slope, c=lower, d=upper, e=EC50"""
    return c + (d - c) / (1 + (e / (dose + 1e-12)) ** b)


def _w14(dose, b, c, d, e):
    """Weibull type 1."""
    return c + (d - c) * np.exp(-np.exp(b * (np.log(dose + 1e-12) - np.log(e + 1e-12))))


def _w24(dose, b, c, d, e):
    """Weibull type 2."""
    return c + (d - c) * (1 - np.exp(-np.exp(b * (np.log(dose + 1e-12) - np.log(e + 1e-12)))))


_MODEL_FUNCS = {"LL4": _ll4, "W14": _w14, "W24": _w24}


# ---------------------------------------------------------------------------
# fit_dose_response
# ---------------------------------------------------------------------------

def fit_dose_response(
    dsd: DrugSeqData,
    compound: str,
    compound_col: str = "compound",
    dose_col: str = "dose",
    reference: str = "DMSO",
    use_norm: bool = True,
    n_genes_fit: int = 200,
    model_selection: bool = False,
    min_r2: float = 0.5,
) -> pd.DataFrame:
    """
    Fit dose-response curves for the top *n_genes_fit* genes using lmfit.

    This replaces the custom ``stats::nls`` Hill fitter from the R package
    with a more robust implementation that provides:
    - Proper confidence intervals on EC50 via lmfit (analogous to drc's Delta
      method CI)
    - AIC-based model selection across LL4, W1.4, W2.4 families
    - Convergence diagnostics

    Parameters
    ----------
    dsd : DrugSeqData with normalized counts
    compound : compound name to fit
    n_genes_fit : maximum genes to fit (ranked by |logFC| from DE if available)
    model_selection : if True, compare LL4/W14/W24 by AIC; else use LL4 only
    min_r2 : minimum pseudo-R² to flag a fit as converged

    Returns
    -------
    pd.DataFrame  gene × (model, EC50, EC50_ci_lower, EC50_ci_upper,
                           slope, Emax, E0, r_squared, aic, converged)
    """
    obs = dsd.obs
    mask = obs[compound_col].isin([compound, reference])
    doses = obs.loc[mask, dose_col].values.astype(float)
    doses[obs.loc[mask, compound_col].values == reference] = 0.0

    if use_norm and dsd.adata.X is not None:
        X = dsd.adata.X
        if sp.issparse(X):
            X = X.toarray()
        mat = X[mask].astype(float)
    else:
        mat = dsd.adata.layers["counts"][mask].toarray().astype(float)

    # gene ranking
    de = dsd.adata.uns.get("de_results", {})
    if compound in de:
        de_df = de[compound]
        ranked_genes = de_df.set_index("gene")["logFC"].abs()\
                            .sort_values(ascending=False).index.tolist()
    else:
        var = mat.var(axis=0)
        ranked_genes = [dsd.var_names[i] for i in np.argsort(var)[::-1]]

    ranked_genes = [g for g in ranked_genes if g in dsd.var_names][:n_genes_fit]
    gene_indices = [dsd.var_names.get_loc(g) for g in ranked_genes]

    print(f"fit_dose_response (lmfit): fitting {len(ranked_genes)} genes for '{compound}'.")

    models_to_try = ["LL4", "W14", "W24"] if model_selection else ["LL4"]

    rows = []
    for g, gi in zip(ranked_genes, gene_indices):
        y = mat[:, gi]
        rows.append(_fit_one_gene(doses, y, g, models_to_try, min_r2))

    return pd.DataFrame(rows)


def _fit_one_gene(doses, y, gene_name, models, min_r2):
    """Fit all requested models and return the best by AIC."""
    empty = {
        "gene": gene_name, "model": None,
        "EC50": np.nan, "EC50_ci_lower": np.nan, "EC50_ci_upper": np.nan,
        "slope": np.nan, "Emax": np.nan, "E0": np.nan,
        "r_squared": np.nan, "aic": np.nan, "converged": False,
    }

    # replace zero doses with small positive value for log models
    dose_pos = doses.copy()
    pos_min  = dose_pos[dose_pos > 0].min() if (dose_pos > 0).any() else 0.01
    dose_pos[dose_pos == 0] = pos_min / 100.0

    E0_init   = float(np.mean(y[doses == 0])) if (doses == 0).any() else float(y.min())
    Emax_init = float(np.mean(y[dose_pos == dose_pos.max()])) - E0_init
    EC50_init = float(np.median(dose_pos[dose_pos > pos_min / 10]))

    best_result = None
    best_aic    = np.inf
    best_name   = None

    for mname in models:
        func = _MODEL_FUNCS[mname]
        mod  = Model(func)
        params = Parameters()
        params.add("b",  value=1.0,  min=0.01, max=20.0)
        params.add("c",  value=E0_init)
        params.add("d",  value=E0_init + Emax_init)
        params.add("e",  value=EC50_init, min=1e-9)

        try:
            result = mod.fit(y, dose=dose_pos, params=params,
                             method="least_squares", nan_policy="omit",
                             max_nfev=1000)
            aic = result.aic
            if np.isfinite(aic) and aic < best_aic:
                best_aic    = aic
                best_result = result
                best_name   = mname
        except Exception:
            continue

    if best_result is None:
        return empty

    p   = best_result.params
    ec50 = float(p["e"].value)
    ci   = {}
    try:
        ci = best_result.conf_interval(sigmas=[1])
        ec50_lo = ci["e"][0][1] if ci else np.nan
        ec50_hi = ci["e"][-1][1] if ci else np.nan
    except Exception:
        ec50_lo = ec50_hi = np.nan

    pred   = best_result.best_fit
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2     = 1 - ss_res / (ss_tot + 1e-10)

    return {
        "gene":          gene_name,
        "model":         best_name,
        "EC50":          ec50,
        "EC50_ci_lower": float(ec50_lo) if not np.isnan(ec50_lo) else np.nan,
        "EC50_ci_upper": float(ec50_hi) if not np.isnan(ec50_hi) else np.nan,
        "slope":         float(p["b"].value),
        "Emax":          float(p["d"].value),
        "E0":            float(p["c"].value),
        "r_squared":     r2,
        "aic":           best_aic,
        "converged":     r2 >= min_r2 and np.isfinite(ec50),
    }


# ---------------------------------------------------------------------------
# compute_multi_dr
# ---------------------------------------------------------------------------

def compute_multi_dr(
    dsd: DrugSeqData,
    compounds: list[str] | None = None,
    n_jobs: int = 1,
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """
    Fit dose-response curves for multiple compounds in parallel.

    Returns
    -------
    dict  compound → DataFrame from fit_dose_response
    """
    de = dsd.adata.uns.get("de_results", {})
    if compounds is None:
        compounds = list(de.keys())

    print(f"compute_multi_dr: fitting {len(compounds)} compound(s).")

    def _run(c):
        try:
            return c, fit_dose_response(dsd, c, **kwargs)
        except Exception as e:
            warnings.warn(f"  Failed for '{c}': {e}")
            return c, None

    if n_jobs == 1:
        results = [_run(c) for c in compounds]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futs = {ex.submit(_run, c): c for c in compounds}
            results = [f.result() for f in as_completed(futs)]

    return {c: df for c, df in results if df is not None}


# ---------------------------------------------------------------------------
# plot_dr_panel
# ---------------------------------------------------------------------------

def plot_dr_panel(
    dr_result: pd.DataFrame,
    dsd: DrugSeqData,
    compound: str,
    n_genes: int = 9,
    dose_col: str = "dose",
    ncol: int = 3,
    figsize: tuple | None = None,
) -> plt.Figure:
    """
    Multi-gene dose-response panel plot.

    Overlays fitted lmfit curves on observed data points.
    """
    conv = dr_result[dr_result["converged"].fillna(False)]\
               .sort_values("r_squared", ascending=False)
    if conv.empty:
        warnings.warn("No converged dose-response fits to plot.")
        return plt.figure()

    top_genes = conv["gene"].head(n_genes).tolist()
    nrow = int(np.ceil(len(top_genes) / ncol))
    if figsize is None:
        figsize = (ncol * 4, nrow * 3.5)

    obs = dsd.obs
    mask = obs["compound"].isin([compound, "DMSO"])
    doses = obs.loc[mask, dose_col].values.astype(float)
    doses[obs.loc[mask, "compound"].values == "DMSO"] = 0.0

    X = dsd.adata.X
    if sp.issparse(X):
        X = X.toarray()
    mat = X[mask].astype(float)

    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)

    for idx, gene in enumerate(top_genes):
        ax   = axes[idx // ncol][idx % ncol]
        row  = conv[conv["gene"] == gene].iloc[0]
        gi   = dsd.var_names.get_loc(gene)
        y    = mat[:, gi]

        dose_pos = doses.copy()
        pos_min  = dose_pos[dose_pos > 0].min() if (dose_pos > 0).any() else 0.01
        dose_pos[dose_pos == 0] = pos_min / 100.0

        ax.scatter(dose_pos, y, s=20, alpha=0.6, color="#2980B9", zorder=3)
        ax.set_xscale("log")

        # overlay fitted curve
        mname = row.get("model", "LL4")
        func  = _MODEL_FUNCS.get(mname, _ll4)
        d_seq = np.logspace(np.log10(dose_pos.min()),
                             np.log10(dose_pos.max()), 100)
        try:
            y_fit = func(d_seq, b=row["slope"], c=row["E0"],
                         d=row["Emax"], e=row["EC50"])
            ax.plot(d_seq, y_fit, color="#C0392B", linewidth=1.5)
        except Exception:
            pass

        ec50_str = f"EC50={row['EC50']:.2g}" if np.isfinite(row["EC50"]) else ""
        r2_str   = f"R²={row['r_squared']:.2f}"
        ax.set_title(f"{gene}\n{ec50_str}  {r2_str}", fontsize=8)
        ax.set_xlabel("Dose (log)", fontsize=7)
        ax.set_ylabel("Expression", fontsize=7)

    for idx in range(len(top_genes), nrow * ncol):
        axes[idx // ncol][idx % ncol].set_visible(False)

    fig.suptitle(f"Dose-response curves: {compound}", fontsize=10)
    fig.tight_layout()
    return fig

def plot_dose_gene_counts(
    dsd: "DrugSeqData",
    ctrl_label: str = "DMSO",
    cmpd_col: str = "compound",
    dose_col: str = "dose",
    lfc_threshold: float = 0.5,
    direction: str = "both",
    compounds: Optional[Union[str, List[str]]] = None,
    combine: bool = True,
    figsize: tuple = (8, 5)
):
    """
    实时计算每个 dose 与对照组的 LogFC，并可视化受影响基因数随剂量增加的变化趋势。
    
    Parameters
    ----------
    dsd : DrugSeqData 对象 (假设 dsd.adata.X 已进行 log1p 标准化)
    ctrl_label : 对照组的名称 (如 'DMSO')
    cmpd_col : obs 中存放化合物名称的列
    dose_col : obs 中存放剂量信息的列
    lfc_threshold : Log Fold Change 绝对值阈值 (用于筛选基因)
    direction : 'up' (上调), 'down' (下调), 或 'both' (两者都算)
    compounds : 指定分析的化合物名称。如果为 None 则分析所有。
    """
    obs = dsd.adata.obs
    X = dsd.adata.X
    
    # 1. 获取全局对照组 (DMSO) 的平均表达量
    ctrl_idx = np.where(obs[cmpd_col] == ctrl_label)[0]
    if len(ctrl_idx) == 0:
        raise ValueError(f"未在 obs['{cmpd_col}'] 中找到对照组 '{ctrl_label}'。")
        
    # 计算均值 (兼容稀疏矩阵)
    if sp.issparse(X):
        ctrl_mean = np.asarray(X[ctrl_idx].mean(axis=0)).flatten()
    else:
        ctrl_mean = X[ctrl_idx].mean(axis=0)

    # 2. 确定要分析的化合物
    if compounds is None:
        compounds = [c for c in obs[cmpd_col].unique() if c != ctrl_label]
    elif isinstance(compounds, str):
        compounds = [compounds]

    stats_list = []
    
    print(f"开始计算 {len(compounds)} 个化合物的 dose-response...")
    
    # 3. 遍历化合物和剂量，实时计算 LogFC
    for cmpd in compounds:
        cmpd_idx = obs[cmpd_col] == cmpd
        if not cmpd_idx.any():
            print(f"⚠️ 警告: 未找到化合物 '{cmpd}'，已跳过。")
            continue
            
        # 获取该化合物的所有有效 dose
        doses = obs.loc[cmpd_idx, dose_col].unique()
        
        for dose in doses:
            # 尝试将 dose 转为数值，方便后续画图排序
            try:
                dose_val = float(dose)
            except (ValueError, TypeError):
                dose_val = dose
                
            dose_idx = np.where((obs[cmpd_col] == cmpd) & (obs[dose_col] == dose))[0]
            if len(dose_idx) == 0: 
                continue
                
            # 计算该 dose 的平均表达量
            if sp.issparse(X):
                dose_mean = np.asarray(X[dose_idx].mean(axis=0)).flatten()
            else:
                dose_mean = X[dose_idx].mean(axis=0)
                
            # 核心：计算 LogFC (假设 X 是 log-normalized 数据，均值差即为 LogFC)
            lfc = dose_mean - ctrl_mean
            
            # 统计符合阈值的基因数
            if direction == "up":
                count = np.sum(lfc > lfc_threshold)
            elif direction == "down":
                count = np.sum(lfc < -lfc_threshold)
            else:
                count = np.sum(np.abs(lfc) > lfc_threshold)
                
            stats_list.append({
                "compound": cmpd,
                "dose": dose_val,
                "gene_count": count
            })

    if not stats_list:
        print("计算完毕，但没有任何数据符合绘图条件。")
        return

    # 4. 数据整理与排序
    stats_df = pd.DataFrame(stats_list)
    stats_df = stats_df.sort_values(["compound", "dose"])

    # 5. 绘图逻辑
    sns.set_style("whitegrid")
    
    if combine:
        plt.figure(figsize=figsize)
        # 用对数坐标轴 (log scale) 画 Dose 往往更符合药理学直觉
        ax = sns.lineplot(data=stats_df, x="dose", y="gene_count", hue="compound", marker="o")
        # plt.xscale('log') # 如果你的 dose 跨度很大(比如 0.1 到 10000)，建议取消这行的注释
        
        plt.title(f"Dose-Response: Number of {direction.capitalize()}regulated Genes (LFC > {lfc_threshold})")
        plt.xlabel("Dose")
        plt.ylabel("Number of Genes")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
    else:
        unique_cmpds = stats_df["compound"].unique()
        n = len(unique_cmpds)
        fig, axes = plt.subplots(n, 1, figsize=(figsize[0], figsize[1]*n), sharex=False)
        if n == 1: axes = [axes]
        
        for i, cmpd in enumerate(unique_cmpds):
            data_sub = stats_df[stats_df["compound"] == cmpd]
            sns.lineplot(data=data_sub, x="dose", y="gene_count", ax=axes[i], marker="o", color="black")
            # axes[i].set_xscale('log') # 同理，视需要开启
            axes[i].set_title(f"Compound: {cmpd}")
            axes[i].set_ylabel("Gene Count")
            axes[i].set_xlabel("Dose")
            
        plt.tight_layout()

    plt.show()

def plot_signature_dose_response(
    dsd: "DrugSeqData",
    ctrl_label: str = "DMSO",
    cmpd_col: str = "compound",
    dose_col: str = "dose",
    lfc_threshold: float = 0.5,
    padj_threshold: float = 0.05,
    direction: str = "up",
    top_n: int = 50,
    compounds: Optional[Union[str, List[str]]] = None,
    run_de_if_missing: bool = True,  # 🚀 新增：默认开启实时 DE 分析
    combine: bool = True,
    log_x: bool = True,
    figsize: tuple = (8, 5)
):
    """
    基于整体 DE 结果提取 Signature，计算 Module Score，并绘制 Dose-Response S型曲线。
    如果缺少 DE 结果，将自动执行实时的 Wilcoxon 差异分析。
    """
    try:
        import scanpy as sc
    except ImportError:
        raise ImportError("此功能依赖 scanpy。请运行: pip install scanpy")

    obs = dsd.adata.obs.copy()
    
    # 确定要分析的化合物列表
    if compounds is None:
        compounds = [c for c in obs[cmpd_col].unique() if c != ctrl_label]
    elif isinstance(compounds, str):
        compounds = [compounds]

    # 1. 获取或实时计算 DE 结果
    de_dict = dsd.adata.uns.get("de_results", {})
    
    if not de_dict:
        if run_de_if_missing:
            print("💡 未在 uns['de_results'] 中找到差异表达记录，正在启动实时 DE 分析 (Wilcoxon)...")
            de_dict = {}
            for cmpd in compounds:
                # 提取当前化合物与对照组的数据
                mask = obs[cmpd_col].isin([cmpd, ctrl_label])
                if sum(obs[cmpd_col] == cmpd) == 0:
                    print(f"⚠️ 实时 DE 跳过 '{cmpd}'：表达矩阵中未找到该化合物的样本。")
                    continue
                    
                sub_adata = dsd.adata[mask].copy()
                try:
                    # 使用 scanpy 进行快速差异分析
                    sc.tl.rank_genes_groups(
                        sub_adata, 
                        groupby=cmpd_col, 
                        groups=[cmpd], 
                        reference=ctrl_label, 
                        method='wilcoxon',
                        use_raw=False
                    )
                    # 提取并格式化为标准 DataFrame
                    df = sc.get.rank_genes_groups_df(sub_adata, group=cmpd)
                    df = df.rename(columns={'names': 'gene', 'logfoldchanges': 'logFC', 'pvals_adj': 'padj'})
                    de_dict[cmpd] = df
                except Exception as e:
                    print(f"⚠️ 实时计算 '{cmpd}' DE 失败: {e}")
        else:
            raise ValueError("未找到 de_results，且 run_de_if_missing=False。请先运行 DE 分析。")

    plot_data_list = []
    print(f"\n开始分析 {len(compounds)} 个化合物的 Signature Score...")

    for cmpd in compounds:
        if cmpd not in de_dict:
            print(f"⚠️ 跳过 '{cmpd}'：未能获取该化合物的差异表达结果。")
            continue

        # 2. 提取 Signature 基因集并显式打印过滤原因
        df = de_dict[cmpd]
        if direction == "up":
            sig_df = df[(df["padj"] < padj_threshold) & (df["logFC"] > lfc_threshold)]
            sig_df = sig_df.sort_values("logFC", ascending=False)
        elif direction == "down":
            sig_df = df[(df["padj"] < padj_threshold) & (df["logFC"] < -lfc_threshold)]
            sig_df = sig_df.sort_values("logFC", ascending=True) 
        else:
            raise ValueError("direction 必须是 'up' 或 'down'。")

        sig_genes = sig_df.head(top_n)["gene"].tolist()
        
        # 🚀 显式的过滤提示
        if len(sig_genes) < 3:
            print(f"⚠️ 剔除 '{cmpd}'：满足阈值 (LFC > {lfc_threshold}, padj < {padj_threshold}) 的显著 {direction} 调基因仅有 {len(sig_genes)} 个 (绘图要求至少 3 个)。")
            continue

        # 3. 使用 Scanpy 计算 Module Score
        score_name = f"sig_score_{cmpd}_{direction}"
        sc.tl.score_genes(dsd.adata, gene_list=sig_genes, score_name=score_name, use_raw=False)
        
        # 4. 提取用于绘图的 Dose 和 Score 数据
        sample_mask = (obs[cmpd_col] == cmpd) | (obs[cmpd_col] == ctrl_label)
        sub_obs = dsd.adata.obs.loc[sample_mask, [cmpd_col, dose_col, score_name]].copy()
        
        def parse_dose(d, c):
            if c == ctrl_label: return 0.0
            try: return float(d)
            except: return np.nan
            
        sub_obs["numeric_dose"] = [parse_dose(d, c) for d, c in zip(sub_obs[dose_col], sub_obs[cmpd_col])]
        
        # 🚀 检查 Dose 是否解析失败
        before_drop = len(sub_obs)
        sub_obs = sub_obs.dropna(subset=["numeric_dose"])
        if len(sub_obs) < before_drop:
            print(f"ℹ️ 提示: '{cmpd}' 中有 {before_drop - len(sub_obs)} 个样本的 dose 无法转换为纯数字，已被丢弃。")
            
        if len(sub_obs[sub_obs[cmpd_col] == cmpd]) == 0:
            print(f"⚠️ 剔除 '{cmpd}'：提取后没有有效的剂量数据进行绘图。")
            continue

        sub_obs = sub_obs.rename(columns={score_name: "signature_score"})
        sub_obs["target_compound"] = cmpd
        plot_data_list.append(sub_obs)

    if not plot_data_list:
        print("\n❌ 最终没有可供绘图的有效数据。")
        return

    # 合并所有绘图数据
    plot_df = pd.concat(plot_data_list, ignore_index=True)

    # 5. 处理 Log-X 轴下的 DMSO (0 浓度) 锚点
    if log_x:
        min_nonzero_dose = plot_df[plot_df["numeric_dose"] > 0]["numeric_dose"].min()
        pseudo_zero = min_nonzero_dose / 5.0  
        plot_df["plot_dose"] = plot_df["numeric_dose"].replace(0.0, pseudo_zero)
    else:
        plot_df["plot_dose"] = plot_df["numeric_dose"]

    # 6. 可视化绘图
    sns.set_style("ticks")
    
    if combine:
        plt.figure(figsize=figsize)
        ax = sns.lineplot(data=plot_df, x="plot_dose", y="signature_score", 
                          hue="target_compound", marker="o", err_style="bars", errorbar="se")
        
        if log_x:
            ax.set_xscale("log")
            xticks = ax.get_xticks()
            ax.set_xticks([pseudo_zero] + [x for x in xticks if x >= min_nonzero_dose])
            ax.set_xticklabels(["0\n(DMSO)"] + [f"{x:g}" for x in xticks if x >= min_nonzero_dose])
            
        plt.title(f"Signature Score Dose-Response (Top {top_n} {direction.capitalize()} Genes)")
        plt.xlabel("Dose")
        plt.ylabel("Signature Score")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        sns.despine()
        plt.tight_layout()
        
    else:
        unique_cmpds = plot_df["target_compound"].unique()
        n = len(unique_cmpds)
        fig, axes = plt.subplots(n, 1, figsize=(figsize[0], figsize[1]*n), sharex=False)
        if n == 1: axes = [axes]
        
        for i, cmpd in enumerate(unique_cmpds):
            data_sub = plot_df[plot_df["target_compound"] == cmpd]
            ax = sns.lineplot(data=data_sub, x="plot_dose", y="signature_score", 
                              ax=axes[i], marker="o", color="darkred" if direction=="up" else "darkblue",
                              err_style="bars", errorbar="se")
            
            if log_x:
                ax.set_xscale("log")
                xticks = ax.get_xticks()
                ax.set_xticks([pseudo_zero] + [x for x in xticks if x >= min_nonzero_dose])
                ax.set_xticklabels(["0"] + [f"{x:g}" for x in xticks if x >= min_nonzero_dose])
                
            axes[i].set_title(f"{cmpd} Signature Score")
            axes[i].set_ylabel("Score")
            axes[i].set_xlabel("Dose")
            sns.despine(ax=axes[i])
            
        plt.tight_layout()

    plt.show()
