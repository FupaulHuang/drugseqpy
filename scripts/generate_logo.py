#!/usr/bin/env python3
"""Generate the drugseqpy vector logo and transparent PNG exports.

Run from the repository root with Python and matplotlib installed.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D


OUT = Path(__file__).resolve().parents[1] / "docs/assets/branding"
NAVY = "#17384A"
TEAL = "#149E96"
GOLD = "#E8AF35"


def canvas(width, height):
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, width), ylim=(height, 0), aspect="equal")
    ax.axis("off")
    return fig, ax


def plate(ax, x=0, y=0, scale=1, *, ink=NAVY, mono=False):
    # A clipped upper-left corner makes the silhouette a laboratory plate.
    vertices = [
        (65, 26), (207, 26), (230, 26), (230, 49), (230, 207),
        (230, 230), (207, 230), (49, 230), (26, 230), (26, 207),
        (26, 65), (65, 26), (65, 26),
    ]
    codes = [
        MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3,
        MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO,
        MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO, MplPath.LINETO,
        MplPath.CLOSEPOLY,
    ]
    transform = Affine2D().scale(scale).translate(x, y) + ax.transData
    ax.add_patch(PathPatch(MplPath(vertices, codes), transform=transform,
                           facecolor="none", edgecolor=ink, linewidth=7.5 * scale,
                           joinstyle="round", capstyle="round"))
    for cx, cy, accent in [(86, 86, TEAL), (170, 86, TEAL),
                            (86, 170, TEAL), (170, 170, GOLD)]:
        ax.add_patch(Circle((cx, cy), 27, transform=transform,
                            facecolor="none", edgecolor=ink if mono else accent,
                            linewidth=7 * scale))


def wordmark(ax, x, baseline, *, size=78, ink=NAVY, mono=False):
    font = FontProperties(family="DejaVu Sans", weight="bold")
    left = TextPath((0, 0), "drugseq", size=size, prop=font)
    right = TextPath((0, 0), "py", size=size, prop=font)
    transform = Affine2D().scale(1, -1).translate(x, baseline) + ax.transData
    ax.add_patch(PathPatch(left, transform=transform, color=ink, linewidth=0))
    offset = left.get_extents().xmax + size * .05
    transform = Affine2D().scale(1, -1).translate(x + offset, baseline) + ax.transData
    ax.add_patch(PathPatch(right, transform=transform, color=ink if mono else TEAL,
                           linewidth=0))


def export(name, *, icon=False, ink=NAVY, mono=False):
    fig, ax = canvas(256 if icon else 800, 256)
    plate(ax, ink=ink, mono=mono)
    if not icon:
        wordmark(ax, 286, 151, ink=ink, mono=mono)
    for fmt in ("svg", "png"):
        path = OUT / f"{name}.{fmt}"
        fig.savefig(path, transparent=True, dpi=300,
                    metadata={"Creator": "drugseqpy", "Date": None} if fmt == "svg" else None)
        if fmt == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context({"svg.fonttype": "path", "svg.hashsalt": "drugseqpy-logo"}):
        export("drugseqpy-logo")
        export("drugseqpy-icon", icon=True)
        export("drugseqpy-logo-monochrome", ink="#111111", mono=True)
        export("drugseqpy-logo-dark", ink="#F4F8FA")
        fig, ax = canvas(1200, 740)
        fig.patch.set_facecolor("#F6F8FA")
        ax.text(60, 56, "drugseqpy", fontsize=15, weight="bold", color=NAVY)
        ax.text(1140, 56, "FOUR-WELL PLATE", ha="right", fontsize=9, color="#637783")
        plate(ax, x=140, y=150)
        wordmark(ax, 426, 301)
        ax.text(600, 468, "A simple plate. Four wells. One distinct response.",
                ha="center", fontsize=14, color="#637783")
        for x, color, label in [(420, NAVY, "INK"), (600, TEAL, "CULTURE"),
                                 (780, GOLD, "RESPONSE")]:
            ax.add_patch(Circle((x, 567), 13, color=color, linewidth=0))
            ax.text(x, 606, label, ha="center", fontsize=8, color="#637783")
            ax.text(x, 630, color, ha="center", fontsize=9, color=NAVY)
        fig.savefig(OUT / "preview.png", dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
