#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figura 4 - Experimento 01
Evaluación de la estimación de distancia mediante ToA.

Entradas:
    data/distance_characterization_raw_mic1.csv
    data/distance_characterization_raw_mic2.csv

Columnas utilizadas:
    r_tape_cm : distancia de referencia medida con cinta [cm]
    r_toa_cm  : distancia estimada mediante ToA [cm]

Salidas:
    figures/fig_exp01_toa.png
    figures/fig_exp01_toa.pdf
    figures/fig_exp01_toa.svg

Los residuos se definen como:
    e_i = r_toa_i - r_ref_i

Las barras de error representan una desviación estándar
de las mediciones individuales en cada posición.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.ticker import AutoMinorLocator


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "figures"

FILES = {
    "Micrófono 1": DATA_DIR / "distance_characterization_raw_mic1.csv",
    "Micrófono 2": DATA_DIR / "distance_characterization_raw_mic2.csv",
}

COL_REFERENCE = "r_tape_cm"
COL_ESTIMATED = "r_toa_cm"

# Colores coherentes con las figuras anteriores.
COLORS = {
    "Micrófono 1": "#008766",
    "Micrófono 2": "#E47732",
}

# Tipografía y presentación.
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.9,
    "lines.linewidth": 1.8,
    "savefig.dpi": 400,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# 2. LECTURA Y VALIDACIÓN DE LOS DATOS
# ============================================================

def load_data(path, microphone_name):

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo:\n{path}"
        )

    df = pd.read_csv(path)

    required_columns = [
        COL_REFERENCE,
        COL_ESTIMATED,
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Faltan columnas en {path.name}: {missing}\n"
            f"Columnas disponibles: {list(df.columns)}"
        )

    df = df[required_columns].copy()

    # Conversión numérica.
    for col in required_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Eliminar registros incompletos.
    df = df.dropna()

    # Eliminar valores físicamente no válidos.
    df = df[
        (df[COL_REFERENCE] > 0)
        & (df[COL_ESTIMATED] > 0)
    ]

    if df.empty:
        raise ValueError(
            f"No hay mediciones válidas para {microphone_name}."
        )

    df["residual_cm"] = (
        df[COL_ESTIMATED] - df[COL_REFERENCE]
    )

    df["residual_mm"] = (
        10.0 * df["residual_cm"]
    )

    return df.reset_index(drop=True)


data = {
    name: load_data(path, name)
    for name, path in FILES.items()
}


# ============================================================
# 3. ESTADÍSTICAS POR POSICIÓN
# ============================================================

def calculate_statistics(df):

    grouped = (
        df.groupby(COL_REFERENCE)[COL_ESTIMATED]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    grouped = grouped.sort_values(COL_REFERENCE)

    # Si existe una sola medición en una posición,
    # la desviación estándar no está definida.
    grouped["std"] = grouped["std"].fillna(0.0)

    return grouped


statistics = {
    name: calculate_statistics(df)
    for name, df in data.items()
}


# ============================================================
# 4. MÉTRICAS DE ERROR
# ============================================================

def calculate_metrics(df):

    errors = df["residual_cm"].to_numpy()

    bias = np.mean(errors)
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))

    return {
        "bias": bias,
        "mae": mae,
        "rmse": rmse,
        "n": len(errors),
    }


metrics = {
    name: calculate_metrics(df)
    for name, df in data.items()
}


# ============================================================
# 5. CREACIÓN DE LA FIGURA
# ============================================================

fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(8.2, 7.8),
    sharex=True,
    gridspec_kw={
        "height_ratios": [1.35, 1.0],
        "hspace": 0.08,
    }
)

fig.subplots_adjust(
    left=0.14,
    right=0.97,
    top=0.94,
    bottom=0.11,
)


# ============================================================
# PANEL SUPERIOR: DISTANCIA ESTIMADA VS REFERENCIA
# ============================================================

all_reference = np.concatenate([
    df[COL_REFERENCE].to_numpy()
    for df in data.values()
])

all_estimated = np.concatenate([
    df[COL_ESTIMATED].to_numpy()
    for df in data.values()
])

xmin = np.floor(np.min(all_reference) / 10) * 10
xmax = np.ceil(np.max(all_reference) / 10) * 10

ymin = min(xmin, np.floor(np.min(all_estimated) / 10) * 10)
ymax = max(xmax, np.ceil(np.max(all_estimated) / 10) * 10)

# Referencia ideal y = x.
ideal_x = np.linspace(xmin, xmax, 300)

ax1.plot(
    ideal_x,
    ideal_x,
    linestyle="--",
    color="#555555",
    linewidth=1.5,
    label="Referencia ideal",
    zorder=1,
)

for name, df in data.items():

    color = COLORS[name]
    stats = statistics[name]

    x = stats[COL_REFERENCE].to_numpy()
    y = stats["mean"].to_numpy()
    yerr = stats["std"].to_numpy()

    ax1.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="o-",
        color=color,
        ecolor=color,
        markersize=5.5,
        markeredgecolor="white",
        markeredgewidth=0.6,
        capsize=3.5,
        elinewidth=1.25,
        linewidth=1.7,
        label=name,
        zorder=3,
    )


ax1.set_ylabel(
    r"Distancia estimada $r_{\mathrm{ToA}}$ [cm]"
)

ax1.set_title(
    "(a) Estimación de distancia",
    pad=9,
)

ax1.grid(
    True,
    which="major",
    linewidth=0.65,
    alpha=0.55,
    color="#C8CDD2",
)

ax1.grid(
    True,
    which="minor",
    linewidth=0.35,
    alpha=0.25,
)

ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
ax1.yaxis.set_minor_locator(AutoMinorLocator(2))

ax1.legend(
    loc="upper left",
    frameon=True,
    framealpha=0.95,
)


# ============================================================
# PANEL INFERIOR: RESIDUOS
# ============================================================

ax2.axhline(
    0,
    color="#555555",
    linestyle="--",
    linewidth=1.5,
    label="Error nulo",
    zorder=1,
)

# Desplazamiento horizontal pequeño y reproducible
# para distinguir los puntos de ambos micrófonos.
offsets = {
    "Micrófono 1": -0.7,
    "Micrófono 2": +0.7,
}

for name, df in data.items():

    color = COLORS[name]
    offset = offsets[name]

    # Mediciones individuales.
    x_individual = (
        df[COL_REFERENCE].to_numpy() + offset
    )

    residuals = df["residual_mm"].to_numpy()

    ax2.scatter(
        x_individual,
        residuals,
        s=13,
        color=color,
        alpha=0.23,
        edgecolors="none",
        label=f"{name}: mediciones",
        zorder=2,
    )

    # Estadísticas de residuos por posición.
    residual_stats = (
        df.groupby(COL_REFERENCE)["residual_mm"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(COL_REFERENCE)
    )

    residual_stats["std"] = (
        residual_stats["std"].fillna(0.0)
    )

    x = (
        residual_stats[COL_REFERENCE].to_numpy()
        + offset
    )

    mean = residual_stats["mean"].to_numpy()
    std = residual_stats["std"].to_numpy()

    ax2.errorbar(
        x,
        mean,
        yerr=std,
        fmt="o",
        color=color,
        ecolor=color,
        markersize=5.5,
        markeredgecolor="white",
        markeredgewidth=0.6,
        capsize=3.5,
        elinewidth=1.25,
        label=f"{name}: media ± DE",
        zorder=4,
    )


ax2.set_xlabel(
    r"Distancia de referencia $r_{\mathrm{ref}}$ [cm]"
)

ax2.set_ylabel(
    r"Residuo $e_i$ [mm]"
)

ax2.set_title(
    "(b) Residuos de estimación",
    pad=9,
)

ax2.grid(
    True,
    which="major",
    linewidth=0.65,
    alpha=0.55,
    color="#C8CDD2",
)

ax2.grid(
    True,
    which="minor",
    linewidth=0.35,
    alpha=0.25,
)

ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
ax2.yaxis.set_minor_locator(AutoMinorLocator(2))


# ============================================================
# 6. LÍMITES Y FORMATO DE LOS EJES
# ============================================================

ax1.set_xlim(xmin - 3, xmax + 3)

# Límites verticales superiores.
upper_min = min(
    np.min(all_reference),
    np.min(all_estimated)
)

upper_max = max(
    np.max(all_reference),
    np.max(all_estimated)
)

margin = 0.08 * (upper_max - upper_min)

ax1.set_ylim(
    upper_min - margin,
    upper_max + margin
)

# Límites simétricos para los residuos.
all_residuals = np.concatenate([
    df["residual_mm"].to_numpy()
    for df in data.values()
])

residual_limit = np.max(np.abs(all_residuals))

if residual_limit == 0:
    residual_limit = 1.0

residual_limit *= 1.18

ax2.set_ylim(
    -residual_limit,
    residual_limit
)


# ============================================================
# 7. MÉTRICAS EN CONSOLA
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENTO 01 - EVALUACIÓN DE DISTANCIA MEDIANTE ToA")
print("=" * 60)

for name, result in metrics.items():

    print(f"\n{name}")
    print("-" * 40)

    print(f"Número de mediciones : {result['n']}")
    print(f"Sesgo medio          : {result['bias']:+.3f} cm")
    print(f"MAE                  : {result['mae']:.3f} cm")
    print(f"RMSE                 : {result['rmse']:.3f} cm")

print("\n" + "=" * 60)


# ============================================================
# 8. EXPORTACIÓN
# ============================================================

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)

outputs = [
    FIGURES_DIR / "fig_exp01_toa.png",
    FIGURES_DIR / "fig_exp01_toa.pdf",
    FIGURES_DIR / "fig_exp01_toa.svg",
]

for output in outputs:

    fig.savefig(
        output,
        dpi=400,
        bbox_inches="tight",
        facecolor="white",
    )

    print(f"Figura guardada: {output}")


plt.show()