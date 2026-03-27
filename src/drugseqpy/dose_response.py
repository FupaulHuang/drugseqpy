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
