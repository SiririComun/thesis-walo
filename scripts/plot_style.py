# -*- coding: utf-8 -*-
"""Estilo gráfico común para las figuras de WALO.

Los tamaños se eligen para conservar legibilidad al insertar las figuras
como gráficos de página completa en el manuscrito o proyectarlas.
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

COLORS = {
    "Micrófono 1": "#008766",
    "Micrófono 2": "#E47732",
    "mic1": "#008766",
    "mic2": "#E47732",
    "phase": "#008766",
    "tdoa": "#E47732",
    "reference": "#555555",
    "grid": "#D9DEE3",
    "mock": "#9A3D3D",
}

# Tamaños en pulgadas: más altura para evitar paneles excesivamente achatados.
FIGSIZE_WIDE = (11.2, 6.2)
FIGSIZE_TALL = (10.8, 7.6)
FIGSIZE_SINGLE = (10.8, 6.4)
DPI = 300


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "legend.fontsize": 10.5,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.65,
        "grid.alpha": 0.75,
        "lines.linewidth": 2.0,
        "lines.markersize": 6.0,
        "figure.dpi": 120,
        "savefig.dpi": DPI,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "legend.framealpha": 0.96,
    })


def style_axis(ax, minor=True):
    ax.grid(True, which="major", color=COLORS["grid"], linewidth=0.65, alpha=0.75)
    if minor:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(True, which="minor", color=COLORS["grid"], linewidth=0.35, alpha=0.35)
    ax.tick_params(which="both", direction="in", top=True, right=True, length=4)
    ax.tick_params(which="minor", length=2.5)


def save_figure(fig, directory, stem):
    directory.mkdir(parents=True, exist_ok=True)
    png = directory / f"{stem}.png"
    pdf = directory / f"{stem}.pdf"
    fig.savefig(png, dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Guardado: {png}")
    print(f"  Guardado: {pdf}")
