# -*- coding: utf-8 -*-
"""Estilo gráfico común para las figuras de WALO."""
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

COLORS = {
    "Micrófono 1": "#008766",
    "Micrófono 2": "#E47732",
    "mic1": "#008766",       # Micrófono 1 / PDoA / WALO
    "mic2": "#E47732",       # Micrófono 2 / TDoA / Tracker
    "phase": "#008766",
    "tdoa": "#E47732",
    "reference": "#555555",
    "grid": "#D9DEE3",
    "mock": "#9A3D3D",
}

FIGSIZE_WIDE = (10.8, 4.6)
FIGSIZE_TALL = (10.8, 5.8)
DPI = 300


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "serif",
        "font.size": 9.5,
        "axes.titlesize": 10,
        "axes.labelsize": 9.5,
        "legend.fontsize": 8,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.55,
        "grid.alpha": 0.75,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        "figure.dpi": 120,
        "savefig.dpi": DPI,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def style_axis(ax, minor=True):
    ax.grid(True, which="major", color=COLORS["grid"], linewidth=0.55, alpha=0.75)
    if minor:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(True, which="minor", color=COLORS["grid"], linewidth=0.3, alpha=0.35)
    ax.tick_params(which="both", direction="in", top=True, right=True)


def save_figure(fig, directory, stem):
    directory.mkdir(parents=True, exist_ok=True)
    png = directory / f"{stem}.png"
    pdf = directory / f"{stem}.pdf"
    fig.savefig(png, dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Guardado: {png}")
    print(f"  Guardado: {pdf}")
