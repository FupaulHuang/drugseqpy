"""tests/test_enrichment_workflow.py — library resolution and enrichment."""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from drugseqpy import (
    CANONICAL_LIBRARIES,
    create_drugseq_object,
    compute_multi_de,
    enrichment_summary,
    resolve_gene_sets,
    run_enrichment,
)
from drugseqpy.enrichment import _load_gmt, _save_gmt
from drugseqpy.utils import make_dummy_screen

HAS_GSEAPY = importlib.util.find_spec("gseapy") is not None
pytestmark = pytest.mark.skipif(not HAS_GSEAPY, reason="gseapy not installed")


@pytest.fixture(scope="module")
def dsd_de():
    counts, obs = make_dummy_screen(
        n_genes=100, n_plates=1, n_dmso_per_plate=6,
        n_compounds=2, n_reps=4, seed=11,
    )
    dsd = create_drugseq_object(counts, obs)
    compute_multi_de(dsd, reference="DMSO", methods=["limma_voom"],
                     within_plate=True, n_jobs=1, inplace=True)
    return dsd


@pytest.fixture(scope="module")
def gene_sets():
    """Three small gene sets drawn from the synthetic gene space."""
    return {
        "SET_UP": [f"Gene{i:04d}" for i in range(1, 19)],
        "SET_DOWN": [f"Gene{i:04d}" for i in range(21, 39)],
        "SET_NEUTRAL": [f"Gene{i:04d}" for i in range(41, 59)],
    }


@pytest.fixture()
def gmt_path(tmp_path, gene_sets):
    path = tmp_path / "test_lib.gmt"
    _save_gmt(path, gene_sets)
    return path


class TestCanonicalLibraries:
    def test_keys_present(self):
        assert set(CANONICAL_LIBRARIES) == {"hallmark", "go_bp",
                                            "reactome", "kegg"}


class TestResolveGeneSets:
    def test_gmt_roundtrip(self, gmt_path, gene_sets):
        assert _load_gmt(gmt_path) == gene_sets

    def test_resolve_from_gmt(self, gmt_path, gene_sets):
        resolved = resolve_gene_sets(
            ["testlib"], gene_set_paths={"testlib": gmt_path},
            min_size=5, max_size=50,
        )
        assert set(resolved["testlib"]) == set(gene_sets)

    def test_size_filtering(self, gmt_path):
        # all three fixture gene sets contain 18 genes
        too_strict = resolve_gene_sets(
            ["testlib"], gene_set_paths={"testlib": gmt_path},
            min_size=19, max_size=50,
        )
        assert len(too_strict["testlib"]) == 0
        kept = resolve_gene_sets(
            ["testlib"], gene_set_paths={"testlib": gmt_path},
            min_size=5, max_size=18,
        )
        assert len(kept["testlib"]) == 3

    def test_inline_dict(self, gene_sets):
        resolved = resolve_gene_sets([{"inline": gene_sets}])
        assert any("SET_UP" in gs for gs in resolved.values())

    def test_inline_mapping_and_single_string(self, gene_sets, gmt_path):
        resolved = resolve_gene_sets({"inline": gene_sets}, min_size=5,
                                     max_size=50)
        assert "inline" in resolved
        resolved = resolve_gene_sets("testlib",
                                     gene_set_paths={"testlib": gmt_path},
                                     min_size=5, max_size=50)
        assert "testlib" in resolved

    def test_path_and_bare_gene_set_mapping_forms(self, gmt_path, gene_sets):
        resolved = resolve_gene_sets(gmt_path, min_size=5, max_size=50)
        assert gmt_path.stem in resolved
        resolved = resolve_gene_sets(gene_sets, min_size=5, max_size=50)
        assert "inline" in resolved

    def test_missing_library_error_offline(self):
        with pytest.raises(ValueError, match="Could not resolve"):
            resolve_gene_sets(["no_such_lib"], allow_network=False)

    def test_missing_gmt_path_error(self):
        with pytest.raises(ValueError, match="Could not resolve"):
            resolve_gene_sets(
                ["testlib"],
                gene_set_paths={"testlib": "/nonexistent/file.gmt"},
                allow_network=False,
            )

    def test_invalid_enrichment_options_raise(self, dsd_de, gene_sets):
        with pytest.raises(ValueError, match="mode"):
            run_enrichment(dsd_de, libraries={"inline": gene_sets},
                           mode=("bad",), inplace=True)
        with pytest.raises(ValueError, match="library"):
            run_enrichment(dsd_de, libraries=[], mode=("ora",), inplace=True)

    def test_network_is_opt_in(self):
        import inspect
        assert inspect.signature(resolve_gene_sets).parameters[
            "allow_network"
        ].default is False


class TestRunEnrichment:
    def test_maps_feature_ids_to_gene_symbols(self):
        genes = [f"ENSG{i:05d}" for i in range(8)]
        symbols = [f"SYMBOL{i}" for i in range(8)]
        counts = pd.DataFrame(
            np.ones((8, 4), dtype=int),
            index=genes,
            columns=["S1", "S2", "S3", "S4"],
        )
        obs = pd.DataFrame(
            {
                "plate_id": ["P1"] * 4,
                "well_id": ["A1", "A2", "A3", "A4"],
                "compound": ["Drug", "Drug", "DMSO", "DMSO"],
                "sample_type": ["treatment", "treatment", "DMSO", "DMSO"],
            },
            index=counts.columns,
        )
        var = pd.DataFrame({"gene_symbol": symbols}, index=genes)
        dsd = create_drugseq_object(counts, obs, var=var)
        dsd.adata.uns["comparison_results"] = {
            "Drug": pd.DataFrame(
                {
                    "gene": genes,
                    "logFC": [2, 1.5, 1, .2, 0, -.2, -.5, -1],
                    "stat": [8, 7, 6, 1, 0, -1, -2, -3],
                    "padj": [.001, .002, .003, .8, .9, .8, .4, .2],
                    "significant": [True, True, True, False, False,
                                    False, False, False],
                }
            )
        }

        run_enrichment(
            dsd,
            libraries={"symbols": {"SYMBOL_PATHWAY": symbols[:3]}},
            mode=("ora",),
            min_size=1,
            max_size=20,
            inplace=True,
        )

        result = dsd.adata.uns["enrichment_results"]["Drug"]["symbols"][
            "ora_up"
        ]
        assert not result.empty
        assert result.iloc[0]["Term"] == "SYMBOL_PATHWAY"

    def test_storage_structure(self, dsd_de, gmt_path):
        run_enrichment(dsd_de, libraries=["testlib"],
                       gene_set_paths={"testlib": gmt_path},
                       mode=("gsea", "ora"), min_size=5, max_size=50,
                       n_perm=200, inplace=True)
        enr = dsd_de.adata.uns["enrichment_results"]
        assert set(enr) == {"Cmpd01", "Cmpd02"}
        for libs in enr.values():
            assert set(libs) == {"testlib"}
            assert set(libs["testlib"]) == {"gsea", "ora_up", "ora_down"}
            for df in libs["testlib"].values():
                assert isinstance(df, pd.DataFrame)

    def test_ora_up_nonempty_down_empty_schema(self, dsd_de, gmt_path):
        enr = dsd_de.adata.uns["enrichment_results"]
        for libs in enr.values():
            up, down = libs["testlib"]["ora_up"], libs["testlib"]["ora_down"]
            assert len(up) > 0
            assert len(down) == 0
            assert {"Term", "Overlap", "P-value",
                    "Adjusted P-value"}.issubset(down.columns)

    def test_deterministic_gsea(self, dsd_de, gmt_path):
        from drugseqpy import DrugSeqData
        dsd2 = DrugSeqData(dsd_de.adata.copy())
        run_enrichment(dsd2, libraries=["testlib"],
                       gene_set_paths={"testlib": gmt_path},
                       mode=("gsea",), min_size=5, max_size=50,
                       n_perm=200, inplace=True)
        a = dsd_de.adata.uns["enrichment_results"]["Cmpd01"]["testlib"]["gsea"]
        b = dsd2.adata.uns["enrichment_results"]["Cmpd01"]["testlib"]["gsea"]
        pd.testing.assert_frame_equal(a, b)

    def test_no_significant_genes_gives_empty_frames(self, gmt_path):
        counts, obs = make_dummy_screen(n_genes=60, n_compounds=2, seed=12)
        dsd = create_drugseq_object(counts, obs)
        # fdr_threshold=0 guarantees no significant genes (padj < 0 never holds)
        compute_multi_de(dsd, reference="DMSO", methods=["limma_voom"],
                         fdr_threshold=0.0, lfc_threshold=1e9,
                         within_plate=False, n_jobs=1, inplace=True)
        run_enrichment(dsd, libraries=["testlib"],
                       gene_set_paths={"testlib": gmt_path},
                       mode=("ora",), min_size=5, max_size=50, inplace=True)
        slots = dsd.adata.uns["enrichment_results"]["Cmpd01"]["testlib"]
        assert set(slots) == {"ora_up", "ora_down"}
        assert all(len(df) == 0 for df in slots.values())
        assert {"contrast", "library", "mode", "direction", "pathway",
                "pvalue", "padj"}.issubset(
            dsd.adata.uns["enrichment_summary"].columns
        )

    def test_empty_gsea_keeps_gsea_schema(self, tmp_path):
        counts, obs = make_dummy_screen(n_genes=60, n_compounds=1, seed=13)
        dsd = create_drugseq_object(counts, obs)
        compute_multi_de(dsd, reference="DMSO", methods=["limma_voom"],
                         fdr_threshold=0.0, lfc_threshold=1e9,
                         within_plate=False, n_jobs=1, inplace=True)
        unseen = tmp_path / "unseen.gmt"
        _save_gmt(unseen, {"UNSEEN": [f"Other{i}" for i in range(18)]})
        run_enrichment(dsd, libraries=["testlib"],
                       gene_set_paths={"testlib": unseen}, mode=("gsea",),
                       min_size=5, max_size=50, n_perm=2, inplace=True)
        empty = dsd.adata.uns["enrichment_results"]["Cmpd01"]["testlib"]["gsea"]
        assert len(empty) == 0
        assert {"Term", "NES", "FDR q-val"}.issubset(empty.columns)

    def test_summary_table(self, dsd_de):
        summary = enrichment_summary(dsd_de)
        assert {"contrast", "library", "mode", "direction", "pathway",
                "pvalue", "padj"}.issubset(summary.columns)
        assert len(summary) > 0
        assert set(summary["mode"].unique()) <= {"gsea", "ora"}

    def test_summary_is_persisted(self, dsd_de, gmt_path):
        run_enrichment(dsd_de, libraries=["testlib"],
                       gene_set_paths={"testlib": gmt_path}, mode=("ora",),
                       min_size=5, max_size=50, inplace=True)
        assert "enrichment_summary" in dsd_de.adata.uns
        assert isinstance(dsd_de.adata.uns["enrichment_summary"], pd.DataFrame)

    def test_h5ad_safe_serialization(self, dsd_de, gmt_path, tmp_path):
        run_enrichment(dsd_de, libraries=["testlib"],
                       gene_set_paths={"testlib": gmt_path}, mode=("ora",),
                       min_size=5, max_size=50, inplace=True)
        path = tmp_path / "enriched.h5ad"
        dsd_de.write(path)
        from drugseqpy import DrugSeqData
        dsd2 = DrugSeqData.read_h5ad(path)
        enr = dsd2.adata.uns["enrichment_results"]
        for libs in enr.values():
            for df in libs["testlib"].values():
                for col in df.columns:
                    if df[col].dtype == object:
                        assert all(isinstance(v, str) for v in df[col])
