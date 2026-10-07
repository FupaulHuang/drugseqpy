"""Command-line entry points for reproducible QC and analysis reports."""
from __future__ import annotations

import argparse
from pathlib import Path

from .io import create_drugseq_from_files
from .workflows import (
    generate_qc_report,
    generate_mix_species_report,
    run_comparison_workflow,
)
from .enrichment import CANONICAL_LIBRARIES


def _csv(value: str | None) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()] if value else []


def _parse_comparisons(value: str | None) -> dict | None:
    """Parse 'CaseA=CtrlA;CaseB=CtrlB' into {case: control}."""
    if not value:
        return None
    out = {}
    for part in value.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(
                f"Invalid --comparisons entry '{part}'. "
                "Use 'Case=Control;Case2=Control2'."
            )
        case, control = part.split("=", 1)
        out[case.strip()] = control.strip()
    return out


def _parse_gmt_dir(gmt_dir: str | None, libraries: list[str]) -> dict | None:
    """Map library keys to '<gmt_dir>/<library>.gmt' when the file exists."""
    if not gmt_dir:
        return None
    root = Path(gmt_dir)
    paths = {}
    for lib in libraries:
        for candidate in (lib, CANONICAL_LIBRARIES.get(lib, lib)):
            path = root / f"{candidate}.gmt"
            if path.exists():
                paths[lib] = str(path)
                break
    if not paths:
        raise SystemExit(
            f"No .gmt files found in '{gmt_dir}' for libraries {libraries}."
        )
    return paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="drugseqpy")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("qc", "mix-species"):
        cmd = sub.add_parser(name)
        cmd.add_argument("expression")
        cmd.add_argument("metadata")
        cmd.add_argument("output")
        cmd.add_argument(
            "--metadata-delimiter",
            default="auto",
            help="metadata delimiter; default 'auto' detects .txt delimiters",
        )
        cmd.set_defaults(kind=name)

    cmp_ = sub.add_parser(
        "compare", help="Run the comparison/DEG/enrichment workflow")
    cmp_.add_argument("expression")
    cmp_.add_argument("metadata")
    cmp_.add_argument("output")
    cmp_.add_argument(
        "--metadata-delimiter",
        default="auto",
        help="metadata delimiter; default 'auto' detects .txt delimiters",
    )
    cmp_.add_argument("--group-col", default="compound")
    cmp_.add_argument("--reference", default="DMSO")
    cmp_.add_argument("--methods", default="limma_voom",
                      help="comma-separated: limma_voom,edgeR,deseq")
    cmp_.add_argument("--comparisons", default=None,
                      help="'Case=Control;Case2=Control2' "
                           "(default: every group vs reference)")
    cmp_.add_argument("--batch-cols", default=None,
                      help="comma-separated covariates, e.g. plate_id")
    cmp_.add_argument("--libraries", default="hallmark,go_bp,reactome,kegg",
                      help="hallmark,go_bp,reactome,kegg or gseapy names")
    cmp_.add_argument("--no-enrichment", action="store_true")
    cmp_.add_argument("--gmt-dir", default=None,
                      help="directory with <library>.gmt files (offline)")
    network = cmp_.add_mutually_exclusive_group()
    network.add_argument("--allow-network", action="store_true",
                         help="allow gseapy to download missing libraries")
    # Keep the old spelling as a no-op-compatible explicit safeguard.
    network.add_argument("--no-network", action="store_true",
                         help="forbid downloading gene-set libraries (default)")
    cmp_.add_argument("--figures",
                      default="volcano,heatmap,bar,box,violin,enrichment")
    cmp_.add_argument("--formats", default="png,pdf")
    cmp_.add_argument("--fdr", type=float, default=0.05)
    cmp_.add_argument("--lfc", type=float, default=0.5)
    cmp_.set_defaults(kind="compare")
    return parser


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    dsd = create_drugseq_from_files(
        args.expression,
        args.metadata,
        delimiter=args.metadata_delimiter,
    )

    if args.kind in ("qc", "mix-species"):
        handler = (generate_qc_report if args.kind == "qc"
                   else generate_mix_species_report)
        handler(dsd, args.output)
        return 0

    libraries = _csv(args.libraries)
    result = run_comparison_workflow(
        dsd, args.output,
        comparisons=_parse_comparisons(args.comparisons),
        group_col=args.group_col,
        reference=args.reference,
        methods=_csv(args.methods),
        batch_cols=_csv(args.batch_cols),
        fdr_threshold=args.fdr,
        lfc_threshold=args.lfc,
        enrichment=not args.no_enrichment,
        libraries=libraries,
        gene_set_paths=_parse_gmt_dir(args.gmt_dir, libraries),
        allow_network=bool(args.allow_network and not args.no_network),
        figures=tuple(_csv(args.figures)),
        formats=tuple(_csv(args.formats)),
    )
    print(result["summary"].to_string(index=False))
    failed = result["manifest"][result["manifest"]["status"] != "done"]
    if not failed.empty:
        print(f"\n{len(failed)} step(s) skipped/failed — see "
              f"{args.output}/results/manifests/comparison_manifest.tsv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
