
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generación de figuras del Experimento 01: caracterización radial.

Entradas:
    data/distance_characterization_raw_mic1.csv
    data/distance_characterization_raw_mic2.csv

Salidas:
    figures/exp01_fig1_amplitude_calibration.png
    figures/exp01_fig2_energy_calibration.png
    figures/exp01_fig3_distance_prediction_MOCK.png
    figures/exp01_fig4_toa_range.png

IMPORTANTE:
    La Figura 3 utiliza datos sintéticos de demostración.
    No representa resultados experimentales.

Uso:
    python scripts/generate_exp01_figures.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


# ============================================================
# 1. CONFIGURACIÓN GENERAL
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "figures"

CSV_FILES = {
    "Micrófono 1": DATA_DIR / "distance_characterization_raw_mic1.csv",
    "Micrófono 2": DATA_DIR / "distance_characterization_raw_mic2.csv",
}

COLORS = {
    "Micrófono 1": "#008766",
    "Micrófono 2": "#E47732",
    "reference": "#555555",
    "grid": "#D9DEE3",
}

REQUIRED_COLUMNS = {
    "r_tape_cm",
    "amp_direct_mv",
    "energy_direct",
    "r_toa_cm",
    "pulse_num",
}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",

    "font.family": "serif",
    "font.size": 10,

    "axes.titlesize": 11,
    "axes.labelsize": 10,

    "legend.fontsize": 8.5,

    "xtick.labelsize": 9,
    "ytick.labelsize": 9,

    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,

    "axes.grid": True,
    "grid.color": COLORS["grid"],
    "grid.linewidth": 0.65,
    "grid.alpha": 0.8,

    "lines.linewidth": 1.7,

    "savefig.dpi": 400,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# 2. LECTURA Y VALIDACIÓN DE DATOS
# ============================================================

def load_data():

    data = {}

    for name, path in CSV_FILES.items():

        if not path.exists():
            raise FileNotFoundError(
                f"No se encontró el archivo:\n{path}"
            )

        df = pd.read_csv(path)

        missing = REQUIRED_COLUMNS.difference(df.columns)

        if missing:
            raise ValueError(
                f"Faltan columnas en {path.name}: "
                f"{sorted(missing)}\n"
                f"Columnas disponibles: {list(df.columns)}"
            )

        df = df.copy()

        numeric_columns = [
            "r_tape_cm",
            "amp_direct_mv",
            "energy_direct",
            "r_toa_cm",
            "pulse_num",
        ]

        for column in numeric_columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        # La distancia de referencia debe ser válida.
        df = df.dropna(subset=["r_tape_cm"])

        df = df[df["r_tape_cm"] > 0]

        # Los observables pueden contener valores no disponibles.
        # No se eliminan filas completas por un observable faltante:
        # cada análisis utiliza únicamente sus registros válidos.

        data[name] = df.reset_index(drop=True)

        if data[name].empty:
            raise ValueError(
                f"No hay registros válidos en {path.name}."
            )

    return data


# ============================================================
# 3. FUNCIONES ESTADÍSTICAS
# ============================================================

def grouped_stats(df, column):
    """
    Calcula media, desviación estándar y número de observaciones
    para cada distancia de referencia.
    """

    valid = df[["r_tape_cm", column]].dropna()

    return (
        valid.groupby("r_tape_cm")[column]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("r_tape_cm")
    )


def fit_power_inverse(stats, power):
    """
    Ajuste del modelo:

        y = k / r^p

    Se realiza una regresión lineal forzada al origen en el
    dominio transformado x = 1/r^p.

    Las unidades de r utilizadas en el ajuste son metros.
    """

    r_m = stats["r_tape_cm"].to_numpy(dtype=float) / 100.0
    y = stats["mean"].to_numpy(dtype=float)

    x = 1.0 / r_m**power

    denominator = np.dot(x, x)

    if denominator == 0:
        raise ValueError("No es posible realizar el ajuste.")

    k = np.dot(x, y) / denominator

    y_fit = k * x

    ss_res = np.sum((y - y_fit)**2)
    ss_tot = np.sum((y - np.mean(y))**2)

    r2 = (
        1.0 - ss_res / ss_tot
        if ss_tot > 0
        else np.nan
    )

    return k, r2


def calculate_metrics(df):
    """
    Calcula métricas sobre todas las mediciones individuales.

    Residuo:
        e_i = r_ToA,i - r_ref,i
    """

    valid = df[["r_tape_cm", "r_toa_cm"]].dropna()

    errors = (
        valid["r_toa_cm"].to_numpy()
        - valid["r_tape_cm"].to_numpy()
    )

    if errors.size == 0:
        raise ValueError(
            "No hay mediciones ToA válidas para calcular métricas."
        )

    return {
        "bias": np.mean(errors),
        "mae": np.mean(np.abs(errors)),
        "rmse": np.sqrt(np.mean(errors**2)),
        "n": errors.size,
    }


# ============================================================
# 4. FORMATO Y EXPORTACIÓN
# ============================================================

def style_axis(ax):

    ax.grid(
        True,
        which="major",
        color=COLORS["grid"],
        linewidth=0.65,
        alpha=0.8,
    )

    ax.grid(
        True,
        which="minor",
        color=COLORS["grid"],
        linewidth=0.35,
        alpha=0.4,
    )

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.tick_params(
        which="both",
        direction="in",
        top=True,
        right=True,
    )


def save_figure(fig, stem):
    """
    Exporta únicamente PNG.

    Los archivos PDF de las figuras individuales no se generan,
    de acuerdo con la política de conservación del repositorio.
    """

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output = FIGURES_DIR / f"{stem}.png"

    fig.savefig(
        output,
        dpi=400,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print(f"  Guardado: {output.relative_to(ROOT)}")


# ============================================================
# 5. FIGURA 1 Y FIGURA 2: CALIBRACIONES
# ============================================================

def plot_calibration(
    data,
    observable,
    power,
    ylabel,
    title,
    stem,
    unit_label,
):

    fig, (ax_lin, ax_phys) = plt.subplots(
        1,
        2,
        figsize=(10.8, 4.15),
    )

    fit_results = {}

    for name, df in data.items():

        color = COLORS[name]

        stats = grouped_stats(df, observable)

        if stats.empty:
            raise ValueError(
                f"No hay datos válidos para {name}, "
                f"observable {observable}."
            )

        k, r2 = fit_power_inverse(stats, power)

        fit_results[name] = (k, r2)

        r_m = (
            stats["r_tape_cm"].to_numpy(dtype=float)
            / 100.0
        )

        y = stats["mean"].to_numpy(dtype=float)

        yerr = (
            stats["std"]
            .fillna(0)
            .to_numpy(dtype=float)
        )

        x_lin = 1.0 / r_m**power

        # Dominio linealizado.
        ax_lin.errorbar(
            x_lin,
            y,
            yerr=yerr,
            fmt="o",
            ms=4.5,
            capsize=2.5,
            color=color,
            ecolor=color,
            label=f"{name}: datos",
        )

        xfit = np.linspace(
            0,
            x_lin.max() * 1.04,
            250,
        )

        ax_lin.plot(
            xfit,
            k * xfit,
            color=color,
            label=(
                fr"{name}: $k={k:.4g}$, "
                fr"$R^2={r2:.4f}$"
            ),
        )

        # Dominio físico.
        ax_phys.errorbar(
            stats["r_tape_cm"],
            y,
            yerr=yerr,
            fmt="o",
            ms=4.5,
            capsize=2.5,
            color=color,
            ecolor=color,
            label=f"{name}: datos",
        )

        r_curve_cm = np.linspace(
            stats["r_tape_cm"].min() * 0.92,
            stats["r_tape_cm"].max() * 1.04,
            300,
        )

        r_curve_m = r_curve_cm / 100.0

        ax_phys.plot(
            r_curve_cm,
            k / r_curve_m**power,
            "--",
            color=color,
            label=fr"{name}: $k/r^{power}$",
        )

    ax_lin.set_title("(a) Dominio linealizado")

    ax_lin.set_xlabel(
        fr"$1/r^{power}$ [{unit_label}]"
    )

    ax_lin.set_ylabel(ylabel)

    ax_lin.legend(
        frameon=True,
        loc="upper left",
    )

    style_axis(ax_lin)

    ax_phys.set_title("(b) Dominio físico")

    ax_phys.set_xlabel(
        r"Distancia de referencia $r$ [cm]"
    )

    ax_phys.set_ylabel(ylabel)

    ax_phys.legend(
        frameon=True,
        loc="upper right",
    )

    style_axis(ax_phys)

    fig.suptitle(
        title,
        y=1.02,
        fontsize=12,
    )

    fig.tight_layout()

    save_figure(fig, stem)

    return fit_results


# ============================================================
# 6. FIGURA 3: ESTIMACIÓN DE DISTANCIA (MOCK)
# ============================================================

def plot_mock_distance(data, amp_fits, energy_fits):
    """
    Figura provisional construida con datos sintéticos.

    Los observables se generan a partir de los coeficientes
    de calibración y se invierten para ilustrar la estimación
    de distancia.

    Esta figura NO constituye una validación experimental.
    """

    rng = np.random.default_rng(20261020)

    r_ref_cm = np.array(
        [15, 20, 25, 30, 40, 50, 60, 80, 100],
        dtype=float,
    )

    r_ref_m = r_ref_cm / 100.0

    fig, (ax_amp, ax_energy) = plt.subplots(
        1,
        2,
        figsize=(10.8, 4.2),
    )

    for name in CSV_FILES:

        color = COLORS[name]

        k_a = amp_fits[name][0]
        k_e = energy_fits[name][0]

        # Ruido sintético ilustrativo.
        amp_noise = rng.normal(
            0.0,
            0.035,
            size=r_ref_cm.size,
        )

        energy_noise = rng.normal(
            0.0,
            0.055,
            size=r_ref_cm.size,
        )

        amp_synthetic = (
            k_a / r_ref_m
        ) * (1.0 + amp_noise)

        energy_synthetic = (
            k_e / r_ref_m**2
        ) * (1.0 + energy_noise)

        r_amp_est_cm = (
            k_a / amp_synthetic
        ) * 100.0

        r_energy_est_cm = (
            np.sqrt(k_e / energy_synthetic)
        ) * 100.0

        ax_amp.plot(
            r_ref_cm,
            r_amp_est_cm,
            "o-",
            ms=4,
            color=color,
            label=name,
        )

        ax_energy.plot(
            r_ref_cm,
            r_energy_est_cm,
            "o-",
            ms=4,
            color=color,
            label=name,
        )

    lo, hi = 10, 105

    for ax in (ax_amp, ax_energy):

        ax.plot(
            [lo, hi],
            [lo, hi],
            "--",
            color=COLORS["reference"],
            lw=1.3,
            label="Estimación ideal",
        )

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

        ax.set_xlabel(
            r"Distancia de referencia $r_{\mathrm{ref}}$ [cm]"
        )

        ax.set_ylabel(
            r"Distancia estimada $\hat r$ [cm]"
        )

        ax.legend(frameon=True)

        style_axis(ax)

    ax_amp.set_title(
        r"(a) Estimación mediante $A_{\mathrm{RMS}}$"
    )

    ax_energy.set_title(
        r"(b) Estimación mediante $E_{\mathrm{dig}}$"
    )

    fig.suptitle(
        "FIGURA PROVISIONAL — DATOS SINTÉTICOS, NO EXPERIMENTALES",
        color="#A32626",
        fontsize=11,
        fontweight="bold",
        y=1.04,
    )

    fig.text(
        0.5,
        0.005,
        "Mock reproducible. No representa mediciones experimentales.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#A32626",
    )

    fig.tight_layout(
        rect=(0, 0.04, 1, 0.96)
    )

    save_figure(
        fig,
        "exp01_fig3_distance_prediction_MOCK",
    )


# ============================================================
# 7. FIGURA 4: ESTIMACIÓN DE DISTANCIA MEDIANTE ToA
# ============================================================

def plot_toa(data):
    """
    Evaluación experimental de la estimación de distancia mediante ToA.

    Panel superior:
        Media de la distancia estimada por posición ± desviación estándar.

    Panel inferior:
        Mediciones individuales de los residuos y media ± desviación
        estándar por posición.

    Residuo:
        e_i = r_ToA,i - r_ref,i

    Las métricas se calculan sobre las mediciones individuales,
    no sobre las medias agrupadas.
    """

    fig, (ax_top, ax_res) = plt.subplots(
        2,
        1,
        figsize=(8.2, 7.8),
        sharex=True,
        gridspec_kw={
            "height_ratios": [1.35, 1.0],
            "hspace": 0.08,
        },
    )

    metrics = {}

    all_reference = []
    all_estimated = []
    all_residuals = []

    offsets = {
        "Micrófono 1": -0.7,
        "Micrófono 2": +0.7,
    }

    # --------------------------------------------------------
    # Procesamiento y representación de cada micrófono
    # --------------------------------------------------------

    for name, df in data.items():

        color = COLORS[name]
        offset = offsets[name]

        valid = df[
            ["r_tape_cm", "r_toa_cm"]
        ].dropna()

        valid = valid[
            (valid["r_tape_cm"] > 0)
            & (valid["r_toa_cm"] > 0)
        ].copy()

        if valid.empty:
            raise ValueError(
                f"No hay mediciones ToA válidas para {name}."
            )

        valid["residual_cm"] = (
            valid["r_toa_cm"]
            - valid["r_tape_cm"]
        )

        valid["residual_mm"] = (
            10.0 * valid["residual_cm"]
        )

        metrics[name] = calculate_metrics(valid)

        stats = (
            valid.groupby("r_tape_cm")["r_toa_cm"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values("r_tape_cm")
        )

        stats["std"] = stats["std"].fillna(0.0)

        x = stats["r_tape_cm"].to_numpy()
        y = stats["mean"].to_numpy()
        yerr = stats["std"].to_numpy()

        # Panel superior: media por posición ± DE.
        ax_top.errorbar(
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
            zorder=4,
        )

        # Panel inferior: mediciones individuales.
        x_individual = (
            valid["r_tape_cm"].to_numpy()
            + offset
        )

        residuals = valid["residual_mm"].to_numpy()

        ax_res.scatter(
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
            valid.groupby("r_tape_cm")["residual_mm"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values("r_tape_cm")
        )

        residual_stats["std"] = (
            residual_stats["std"].fillna(0.0)
        )

        x_res = (
            residual_stats["r_tape_cm"].to_numpy()
            + offset
        )

        mean_res = residual_stats["mean"].to_numpy()
        std_res = residual_stats["std"].to_numpy()

        ax_res.errorbar(
            x_res,
            mean_res,
            yerr=std_res,
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

        all_reference.extend(
            valid["r_tape_cm"].to_numpy().tolist()
        )

        all_estimated.extend(
            valid["r_toa_cm"].to_numpy().tolist()
        )

        all_residuals.extend(residuals.tolist())

    # --------------------------------------------------------
    # Panel superior
    # --------------------------------------------------------

    xmin = min(all_reference)
    xmax = max(all_reference)

    ideal_x = np.linspace(
        xmin,
        xmax,
        300,
    )

    ax_top.plot(
        ideal_x,
        ideal_x,
        linestyle="--",
        color=COLORS["reference"],
        linewidth=1.5,
        label="Referencia ideal",
        zorder=1,
    )

    ax_top.set_ylabel(
        r"Distancia estimada $r_{\mathrm{ToA}}$ [cm]"
    )

    ax_top.set_title(
        "(a) Estimación de distancia",
        pad=9,
    )

    ax_top.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.95,
    )

    style_axis(ax_top)

    # --------------------------------------------------------
    # Panel inferior
    # --------------------------------------------------------

    ax_res.axhline(
        0,
        color=COLORS["reference"],
        linestyle="--",
        linewidth=1.5,
        label="Error nulo",
        zorder=1,
    )

    ax_res.set_xlabel(
        r"Distancia de referencia $r_{\mathrm{ref}}$ [cm]"
    )

    ax_res.set_ylabel(
        r"Residuo $e_i$ [mm]"
    )

    ax_res.set_title(
        "(b) Residuos de estimación",
        pad=9,
    )

    style_axis(ax_res)

    # --------------------------------------------------------
    # Límites de los ejes
    # --------------------------------------------------------

    ax_top.set_xlim(
        xmin - 3,
        xmax + 3,
    )

    upper_min = min(
        min(all_reference),
        min(all_estimated),
    )

    upper_max = max(
        max(all_reference),
        max(all_estimated),
    )

    margin = 0.08 * (upper_max - upper_min)

    if margin == 0:
        margin = 1.0

    ax_top.set_ylim(
        upper_min - margin,
        upper_max + margin,
    )

    residual_limit = max(
        abs(np.min(all_residuals)),
        abs(np.max(all_residuals)),
    )

    if residual_limit == 0:
        residual_limit = 1.0

    residual_limit *= 1.18

    ax_res.set_ylim(
        -residual_limit,
        residual_limit,
    )

    # --------------------------------------------------------
    # Exportación
    # --------------------------------------------------------

    fig.subplots_adjust(
        left=0.14,
        right=0.97,
        top=0.94,
        bottom=0.11,
        hspace=0.08,
    )

    save_figure(
        fig,
        "exp01_fig4_toa_range",
    )

    return metrics


# ============================================================
# 8. FUNCIÓN PRINCIPAL
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("EXPERIMENTO 01 - CARACTERIZACIÓN RADIAL")
    print("=" * 60)

    print("\nLeyendo datos experimentales...")

    data = load_data()

    # --------------------------------------------------------
    # Figura 1
    # --------------------------------------------------------

    print("\nGenerando Figura 1: calibración de amplitud...")

    amp_fits = plot_calibration(
        data=data,
        observable="amp_direct_mv",
        power=1,
        ylabel=r"Amplitud RMS $A_{\mathrm{RMS}}$ [mV]",
        title="Calibración radial de amplitud",
        stem="exp01_fig1_amplitude_calibration",
        unit_label=r"m$^{-1}$",
    )

    # --------------------------------------------------------
    # Figura 2
    # --------------------------------------------------------

    print("\nGenerando Figura 2: calibración de energía digital...")

    energy_fits = plot_calibration(
        data=data,
        observable="energy_direct",
        power=2,
        ylabel=r"Energía digital $E_{\mathrm{dig}}$ [unid. arb.]",
        title="Calibración radial de energía digital",
        stem="exp01_fig2_energy_calibration",
        unit_label=r"m$^{-2}$",
    )

    # --------------------------------------------------------
    # Figura 3
    # --------------------------------------------------------

    print("\nGenerando Figura 3: estimación de distancia (MOCK)...")

    plot_mock_distance(
        data,
        amp_fits,
        energy_fits,
    )

    # --------------------------------------------------------
    # Figura 4
    # --------------------------------------------------------

    print("\nGenerando Figura 4: estimación mediante ToA...")

    toa_metrics = plot_toa(data)

    # --------------------------------------------------------
    # Resultados de calibración
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("COEFICIENTES DE CALIBRACIÓN")
    print("=" * 60)

    for name in CSV_FILES:

        ka, r2a = amp_fits[name]
        ke, r2e = energy_fits[name]

        print(f"\n{name}")

        print(
            f"  Amplitud: kA = {ka:.6g} mV·m, "
            f"R² = {r2a:.5f}"
        )

        print(
            f"  Energía:  kE = {ke:.6g}, "
            f"R² = {r2e:.5f}"
        )

    # --------------------------------------------------------
    # Métricas ToA
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("EVALUACIÓN DE DISTANCIA MEDIANTE ToA")
    print("=" * 60)

    for name, result in toa_metrics.items():

        print(f"\n{name}")
        print("-" * 40)

        print(
            f"Número de mediciones : {result['n']}"
        )

        print(
            f"Sesgo medio          : "
            f"{result['bias']:+.3f} cm"
        )

        print(
            f"MAE                  : "
            f"{result['mae']:.3f} cm"
        )

        print(
            f"RMSE                 : "
            f"{result['rmse']:.3f} cm"
        )

    print("\n" + "=" * 60)

    print("\nProceso terminado.")
    print("Revisa las figuras en figures/.")

    print(
        "\nADVERTENCIA: la Figura 3 contiene datos sintéticos "
        "de demostración y no debe presentarse como resultado "
        "experimental."
    )


# ============================================================
# 9. EJECUCIÓN
# ============================================================

if __name__ == "__main__":

    try:
        main()

    except Exception as exc:

        print(
            f"\nERROR: {exc}",
            file=sys.stderr,
        )

        raise
