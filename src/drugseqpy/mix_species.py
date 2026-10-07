"""Independent two-species mixture and cross-contamination workflow."""
from __future__ import annotations

from collections.abc import Mapping
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .core import DrugSeqData


def assign_well_species(
    dsd: DrugSeqData,
    source_col: str,
    mapping: Mapping[str, str],
    *,
    output_col: str = "expected_species",
    default: str = "unknown",
    inplace: bool = True,
) -> DrugSeqData | None:
    """Assign each well's designed species from sample metadata.

    Matching is case-insensitive after converting both metadata values and
    mapping keys to strings. Mapping values should be the configured species
    names or ``"control"``. Unmapped values receive ``default``.
    """
    if not isinstance(dsd, DrugSeqData):
        raise TypeError("dsd must be a DrugSeqData object")
    if source_col not in dsd.obs:
        raise KeyError(f"dsd.obs does not contain {source_col!r}")
    if not mapping:
        raise ValueError("mapping must contain at least one metadata-to-species assignment")
    if not isinstance(output_col, str) or not output_col.strip():
        raise ValueError("output_col must be a non-empty string")

    target = dsd if inplace else dsd.copy()
    normalized_mapping = {
        str(key).strip().casefold(): str(value).strip()
        for key, value in mapping.items()
    }
    source = target.obs[source_col].astype("string").str.strip().str.casefold()
    target.obs[output_col] = source.map(normalized_mapping).fillna(default).astype(str)
    target.uns["well_species_assignment"] = {
        "source_col": source_col,
        "output_col": output_col,
        "mapping": {str(key): str(value) for key, value in mapping.items()},
        "default": str(default),
    }
    if not inplace:
        return target
    return None


def _species_configuration(
    *,
    species_a_prefix: str,
    species_b_prefix: str,
    species_a_name: str,
    species_b_name: str,
    assignment_threshold: float,
    min_species_umi: float,
    human_prefix: str | None,
    mouse_prefix: str | None,
    human_threshold: float | None,
    min_total_umi: float | None,
) -> tuple[str, str, str, str, float, float]:
    """Resolve generic settings and backward-compatible human/mouse aliases."""
    if human_prefix is not None:
        species_a_prefix = human_prefix
    if mouse_prefix is not None:
        species_b_prefix = mouse_prefix
    if human_threshold is not None:
        assignment_threshold = human_threshold
    if min_total_umi is not None:
        min_species_umi = min_total_umi

    names = [str(species_a_name).strip(), str(species_b_name).strip()]
    prefixes = [str(species_a_prefix), str(species_b_prefix)]
    if not all(names):
        raise ValueError("species_a_name and species_b_name must be non-empty")
    if names[0].casefold() == names[1].casefold():
        raise ValueError("species_a_name and species_b_name must be different")
    if not all(prefixes):
        raise ValueError("species_a_prefix and species_b_prefix must be non-empty")
    if prefixes[0] == prefixes[1]:
        raise ValueError("species_a_prefix and species_b_prefix must be different")
    if not 0 <= assignment_threshold <= 1:
        raise ValueError("assignment_threshold must be between 0 and 1")
    if min_species_umi < 0:
        raise ValueError("min_species_umi must be non-negative")
    return (
        prefixes[0], prefixes[1], names[0], names[1],
        float(assignment_threshold), float(min_species_umi),
    )


def _expected_species(
    values: pd.Series,
    *,
    species_a_name: str,
    species_b_name: str,
    mapping: Mapping[str, str] | None,
) -> np.ndarray:
    """Resolve metadata labels to the two configured species or control."""
    labels = values.astype("string").fillna("").str.strip().str.casefold()
    canonical = {
        species_a_name.casefold(): species_a_name,
        species_b_name.casefold(): species_b_name,
        "control": "control",
        "unknown": "unknown",
    }
    if mapping is not None:
        normalized_mapping = {
            str(key).strip().casefold(): canonical.get(
                str(value).strip().casefold(), str(value).strip()
            )
            for key, value in mapping.items()
        }
        resolved = labels.map(normalized_mapping).fillna("unknown").astype(str)
        allowed = {species_a_name, species_b_name, "control", "unknown"}
        invalid = sorted(set(resolved) - allowed)
        if invalid:
            raise ValueError(
                "expected_species_map values must be either configured species, "
                f"'control', or 'unknown'; found {invalid}"
            )
        return resolved.to_numpy(dtype=object)

    species_a_pattern = re.escape(species_a_name.casefold())
    species_b_pattern = re.escape(species_b_name.casefold())
    if species_a_name.casefold() == "human":
        species_a_pattern = "human|231"
    if species_b_name.casefold() == "mouse":
        species_b_pattern = "mouse|3t3"
    return np.select(
        [
            labels.str.contains(species_a_pattern, regex=True),
            labels.str.contains(species_b_pattern, regex=True),
            labels.str.contains("water|blank|control", regex=True),
        ],
        [species_a_name, species_b_name, "control"],
        default="unknown",
    )


def _add_legacy_human_mouse_columns(
    result: pd.DataFrame,
    *,
    species_a_name: str,
    species_b_name: str,
) -> None:
    """Retain historical columns when human or mouse is configured."""
    for slot, name in (("a", species_a_name), ("b", species_b_name)):
        normalized = name.casefold()
        if normalized not in {"human", "mouse"}:
            continue
        result[f"{normalized}_umi"] = result[f"species_{slot}_umi"]
        result[f"{normalized}_fraction"] = result[f"species_{slot}_fraction"]


def compute_mix_species_qc(
    dsd: DrugSeqData,
    *,
    species_a_prefix: str = "ENSG",
    species_b_prefix: str = "ENSMUSG",
    species_a_name: str = "human",
    species_b_name: str = "mouse",
    expected_col: str | None = None,
    expected_species_map: Mapping[str, str] | None = None,
    assignment_threshold: float = 0.5,
    min_species_umi: float = 0,
    mitochondrial_pattern: str = r"^(?:MT-|mt-)",
    human_prefix: str | None = None,
    mouse_prefix: str | None = None,
    human_threshold: float | None = None,
    min_total_umi: float | None = None,
) -> pd.DataFrame:
    """Quantify any two species and evaluate cross-well contamination.

    The generic ``species_a_*`` and ``species_b_*`` columns are always
    returned. Historical ``human_*`` and ``mouse_*`` aliases are additionally
    emitted when those names are configured. ``human_prefix``,
    ``mouse_prefix``, ``human_threshold``, and ``min_total_umi`` remain as
    backward-compatible aliases for the corresponding generic parameters.
    """
    (
        species_a_prefix,
        species_b_prefix,
        species_a_name,
        species_b_name,
        assignment_threshold,
        min_species_umi,
    ) = _species_configuration(
        species_a_prefix=species_a_prefix,
        species_b_prefix=species_b_prefix,
        species_a_name=species_a_name,
        species_b_name=species_b_name,
        assignment_threshold=assignment_threshold,
        min_species_umi=min_species_umi,
        human_prefix=human_prefix,
        mouse_prefix=mouse_prefix,
        human_threshold=human_threshold,
        min_total_umi=min_total_umi,
    )

    matrix = dsd.adata.layers.get("counts", dsd.adata.X)
    if sp.issparse(matrix):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=float)
    genes = np.asarray(dsd.var_names.astype(str).to_numpy(), dtype="U")
    symbols = (
        np.asarray(dsd.var["gene_symbol"].astype(str).to_numpy(), dtype="U")
        if "gene_symbol" in dsd.var else genes
    )
    species_a_mask = np.char.startswith(genes, species_a_prefix)
    species_b_mask = np.char.startswith(genes, species_b_prefix)
    if np.any(species_a_mask & species_b_mask):
        raise ValueError("species prefixes select overlapping feature sets")
    mitochondrial_mask = np.array(
        pd.Series(symbols).str.contains(mitochondrial_pattern, regex=True)
    )

    species_a_umi = matrix[:, species_a_mask].sum(axis=1)
    species_b_umi = matrix[:, species_b_mask].sum(axis=1)
    species_umi = species_a_umi + species_b_umi
    total_umi = matrix.sum(axis=1)
    species_a_fraction = np.divide(
        species_a_umi,
        species_umi,
        out=np.full(len(species_umi), np.nan),
        where=species_umi > 0,
    )
    species_b_fraction = np.divide(
        species_b_umi,
        species_umi,
        out=np.full(len(species_umi), np.nan),
        where=species_umi > 0,
    )
    observed = np.where(
        species_umi < min_species_umi,
        "low_umi",
        np.where(
            np.isnan(species_a_fraction),
            "undetermined",
            np.where(
                species_a_fraction >= assignment_threshold,
                species_a_name,
                np.where(
                    species_b_fraction >= assignment_threshold,
                    species_b_name,
                    "mixed",
                ),
            ),
        ),
    )

    result = dsd.obs.copy()
    result["species_a_name"] = species_a_name
    result["species_b_name"] = species_b_name
    result["species_a_umi"] = species_a_umi
    result["species_b_umi"] = species_b_umi
    result["species_umi"] = species_umi
    result["species_a_fraction"] = species_a_fraction
    result["species_b_fraction"] = species_b_fraction
    result["total_umi"] = total_umi
    result["n_genes_det"] = (matrix > 0).sum(axis=1)
    result["pct_mito"] = (
        100 * matrix[:, mitochondrial_mask].sum(axis=1) / (total_umi + 1e-8)
    )
    result["observed_species"] = observed
    _add_legacy_human_mouse_columns(
        result,
        species_a_name=species_a_name,
        species_b_name=species_b_name,
    )

    if expected_col and expected_col in result:
        expected = _expected_species(
            result[expected_col],
            species_a_name=species_a_name,
            species_b_name=species_b_name,
            mapping=expected_species_map,
        )
        result["expected_species"] = expected
        is_species_a = expected == species_a_name
        is_species_b = expected == species_b_name
        is_control = expected == "control"
        result["base_species"] = np.where(
            is_species_a | is_species_b, expected, "unassigned"
        )
        result["on_target_umi"] = np.where(
            is_species_a,
            species_a_umi,
            np.where(is_species_b, species_b_umi, np.nan),
        )
        result["off_target_umi"] = np.where(
            is_species_a,
            species_b_umi,
            np.where(is_species_b, species_a_umi, np.nan),
        )
        result["on_target_fraction"] = np.where(
            is_species_a,
            species_a_fraction,
            np.where(is_species_b, species_b_fraction, np.nan),
        )
        result["cross_contamination_fraction"] = np.where(
            is_species_a,
            species_b_fraction,
            np.where(is_species_b, species_a_fraction, np.nan),
        )
        result["concordance"] = np.select(
            [
                is_control & (species_umi > 0),
                is_control & (species_umi == 0),
                (is_species_a | is_species_b) & (observed == expected),
                is_species_a | is_species_b,
            ],
            ["signal_control", "absent", "concordant", "discordant"],
            default="not_assessed",
        )

    result.attrs["species_config"] = {
        "species_a_name": species_a_name,
        "species_b_name": species_b_name,
        "species_a_prefix": species_a_prefix,
        "species_b_prefix": species_b_prefix,
        "assignment_threshold": assignment_threshold,
        "min_species_umi": min_species_umi,
    }
    return result
