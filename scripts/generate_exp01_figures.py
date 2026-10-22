
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generación de figuras del Experimento 01: caracterización radial.

Entradas:
    data/distance_characterization_raw_mic1.csv
    data/distance_characterization_raw_mic2.csv

Salidas:
    figures/exp01_fig1_amplitude_calibration.png
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
from plot_style import COLORS, FIGSIZE_WIDE, FIGSIZE_TALL, apply_style, style_axis as common_style_axis, save_figure as common_save_figure


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

REQUIRED_COLUMNS = {
    "r_tape_cm", "amp_direct_mv", "r_toa_cm", "pulse_num",
}

apply_style()



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
    common_style_axis(ax)


def save_figure(fig, stem):
    common_save_figure(fig, FIGURES_DIR, stem)


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
        figsize=FIGSIZE_WIDE,
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




def plot_combined_calibration(data):
    """
    Figura compuesta 2x2 que conserva la organización de las dos
    calibraciones originales:
      fila superior: amplitud RMS (1/r a la izquierda; r a la derecha)
      fila inferior: energía digital (1/r² a la izquierda; r a la derecha)

    En cada panel aparecen simultáneamente los dos micrófonos.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 11.0))
    ax_amp_lin, ax_amp_phys = axes[0]
    ax_energy_lin, ax_energy_phys = axes[1]

    observables = [
        ("amp_direct_mv", 1, "amplitude", r"Amplitud RMS [mV]",
         r"$1/r$ [m$^{-1}$]", r"$k/r$", "Calibración radial de amplitud"),
        ("energy_direct", 2, "energy", r"Energía digital [u.a.]",
         r"$1/r^2$ [m$^{-2}$]", r"$k/r^2$", "Calibración radial de energía digital"),
    ]
    fits = {"amplitude": {}, "energy": {}}

    for row, (column, power, fit_key, ylabel, xlabel_lin, model_label, _) in enumerate(observables):
        ax_lin = axes[row, 0]
        ax_phys = axes[row, 1]
        all_y = []
        all_xlin = []

        for name, df in data.items():
            stats = grouped_stats(df, column)
            if stats.empty:
                raise ValueError(f"Sin datos válidos para {column}, {name}")

            k, r2 = fit_power_inverse(stats, power)
            fits[fit_key][name] = (k, r2)

            r_cm = stats["r_tape_cm"].to_numpy(dtype=float)
            r_m = r_cm / 100.0
            mean = stats["mean"].to_numpy(dtype=float)
            std = stats["std"].fillna(0).to_numpy(dtype=float)
            xlin = 1.0 / (r_m ** power)
            color = COLORS[name]
            all_y.extend((mean - std).tolist())
            all_y.extend((mean + std).tolist())
            all_xlin.extend(xlin.tolist())

            label_data = f"{name}: datos"
            label_fit = fr"{name}: $k={k:.3g}$, $R^2={r2:.4f}$"

            ax_lin.errorbar(
                xlin, mean, yerr=std, fmt="o", ms=5, capsize=3,
                color=color, ecolor=color, label=label_data, zorder=3
            )
            xfit_lin = np.linspace(0, max(xlin) * 1.04, 250)
            ax_lin.plot(
                xfit_lin, k * xfit_lin, color=color, lw=2.2,
                label=label_fit, zorder=2
            )

            ax_phys.errorbar(
                r_cm, mean, yerr=std, fmt="o", ms=5, capsize=3,
                color=color, ecolor=color, label=label_data, zorder=3
            )
            rfit_cm = np.linspace(max(1.0, min(r_cm) * 0.92), max(r_cm) * 1.04, 300)
            rfit_m = rfit_cm / 100.0
            ax_phys.plot(
                rfit_cm, k / (rfit_m ** power), color=color, lw=2.2,
                label=fr"{name}: {model_label}", zorder=2
            )

        ax_lin.set_xlabel(xlabel_lin)
        ax_lin.set_ylabel(ylabel)
        ax_lin.set_title("(a) Dominio linealizado" if row == 0 else "(c) Dominio linealizado")
        ax_lin.legend(loc="best", framealpha=0.95, fontsize=9)
        style_axis(ax_lin)

        ax_phys.set_xlabel(r"Distancia de referencia $r$ [cm]")
        ax_phys.set_ylabel(ylabel)
        ax_phys.set_title("(b) Dominio físico" if row == 0 else "(d) Dominio físico")
        ax_phys.legend(loc="best", framealpha=0.95, fontsize=9)
        style_axis(ax_phys)

        # Escalas comparables entre los micrófonos dentro de cada observable.
        if all_y:
            ymin = min(0.0, min(all_y))
            ymax = max(all_y)
            pad = max((ymax - ymin) * 0.08, 0.5)
            ax_lin.set_ylim(ymin - pad, ymax + pad)
            ax_phys.set_ylim(ymin - pad, ymax + pad)

    # Mantener una misma extensión horizontal para los dos paneles de cada dominio.
    ax_amp_lin.set_xlim(left=0)
    ax_energy_lin.set_xlim(left=0)
    ax_amp_phys.set_xlim(10, 105)
    ax_energy_phys.set_xlim(10, 105)

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.94,
                        wspace=0.27, hspace=0.30)
    save_figure(fig, "exp01_fig1_calibration_four_panel")
    return fits

# ============================================================
# 6. FIGURA 3: ESTIMACIÓN DE DISTANCIA (MOCK)
# ============================================================

def plot_mock_distance(amp_fits):
    """Predicción sintética RMS con la misma composición que la figura ToA.

    Panel superior: media estimada por distancia de referencia ± 1σ.
    Panel inferior: residuos individuales y media ± 1σ por posición.

    Se generan repeticiones sintéticas únicamente para visualizar la
    estructura gráfica; no son observaciones ni estimaciones medidas.
    """
    rng = np.random.default_rng(20261020)
    r_ref_cm = np.array([15, 20, 25, 30, 40, 50, 60, 80, 100], dtype=float)
    n_repeats = 10

    fig, (ax_top, ax_res) = plt.subplots(
        2, 1, figsize=FIGSIZE_TALL, sharex=True,
        gridspec_kw={"height_ratios": [1.35, 1.0], "hspace": 0.10},
    )

    offsets = {"Micrófono 1": -0.65, "Micrófono 2": 0.65}
    all_estimates, all_residuals = [], []

    for name in CSV_FILES:
        color = COLORS[name]
        k_a = amp_fits[name][0]
        offset = offsets[name]
        r_ref_m = r_ref_cm / 100.0

        # Variación relativa sintética alrededor de la amplitud ideal.
        relative_noise = rng.normal(0.0, 0.035, size=(n_repeats, r_ref_cm.size))
        amp_synthetic = (k_a / r_ref_m)[None, :] * (1.0 + relative_noise)
        estimates = (k_a / amp_synthetic) * 100.0
        residuals_mm = (estimates - r_ref_cm[None, :]) * 10.0

        mean_est = np.mean(estimates, axis=0)
        std_est = np.std(estimates, axis=0, ddof=1)
        mean_res = np.mean(residuals_mm, axis=0)
        std_res = np.std(residuals_mm, axis=0, ddof=1)

        ax_top.errorbar(
            r_ref_cm, mean_est, yerr=std_est, fmt="o-", color=color,
            ecolor=color, markersize=6, capsize=4, elinewidth=1.5,
            linewidth=2.0, markeredgecolor="white", markeredgewidth=0.7,
            label=name, zorder=4,
        )

        xj = r_ref_cm + offset
        for row in residuals_mm:
            ax_res.scatter(xj, row, s=20, color=color, alpha=0.22,
                           edgecolors="none", zorder=2)
        ax_res.errorbar(
            xj, mean_res, yerr=std_res, fmt="o", color=color,
            ecolor=color, markersize=5.5, capsize=4, elinewidth=1.5,
            markeredgecolor="white", markeredgewidth=0.6,
            label=f"{name}: media ± 1σ", zorder=4,
        )
        all_estimates.extend(estimates.ravel().tolist())
        all_residuals.extend(residuals_mm.ravel().tolist())

    ideal = np.linspace(r_ref_cm.min(), r_ref_cm.max(), 250)
    ax_top.plot(ideal, ideal, "--", color=COLORS["reference"],
                linewidth=1.6, label="Referencia ideal", zorder=1)
    ax_top.set_ylabel(r"Distancia estimada $\hat r$ [cm]")
    ax_top.set_title("(a) Estimación de distancia", pad=10)
    ax_top.legend(loc="upper left", ncol=2)
    style_axis(ax_top)

    ax_res.axhline(0, color=COLORS["reference"], linestyle="--",
                   linewidth=1.5, label="Residuo nulo", zorder=1)
    ax_res.set_xlabel(r"Distancia de referencia $r_{\mathrm{ref}}$ [cm]")
    ax_res.set_ylabel(r"Residuo $e_i$ [mm]")
    ax_res.set_title("(b) Residuos de estimación", pad=10)
    ax_res.legend(loc="upper left", ncol=2)
    style_axis(ax_res)

    ax_top.set_xlim(r_ref_cm.min() - 4, r_ref_cm.max() + 4)
    low = min(r_ref_cm.min(), np.min(all_estimates))
    high = max(r_ref_cm.max(), np.max(all_estimates))
    margin = 0.08 * (high - low)
    ax_top.set_ylim(low - margin, high + margin)
    lim = max(abs(np.min(all_residuals)), abs(np.max(all_residuals)), 1.0) * 1.18
    ax_res.set_ylim(-1100, 1100)

    fig.text(0.5, 0.015,
             "MOCK — datos sintéticos; no son mediciones experimentales",
             ha="center", va="bottom", fontsize=10, color=COLORS["mock"])
    fig.subplots_adjust(left=0.14, right=0.97, top=0.96, bottom=0.10, hspace=0.28)
    save_figure(fig, "exp01_fig3_distance_prediction_MOCK")


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
        figsize=FIGSIZE_TALL,
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

    # Escala común con la figura MOCK de distancia: no se autoajusta por figura.
    ax_res.set_ylim(-1100, 1100)

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
    data = load_data()

    print("\nGenerando calibración combinada de amplitud y energía...")
    calibration_fits = plot_combined_calibration(data)
    amp_fits = calibration_fits["amplitude"]

    print("\nGenerando Figura 3: estimación de distancia (MOCK)...")
    plot_mock_distance(amp_fits)

    print("\nGenerando Figura 4: estimación mediante ToA...")
    toa_metrics = plot_toa(data)

    print("\nCOEFICIENTES DE CALIBRACIÓN")
    for observable, results in calibration_fits.items():
        print(f"\n{observable.upper()}")
        for name, (k, r2) in results.items():
            print(f"  {name}: k={k:.6g}, R²={r2:.5f}")

    print("\nMÉTRICAS ToA")
    for name, result in toa_metrics.items():
        print(f"\n{name}: N={result['n']}, sesgo={result['bias']:+.3f} cm, "
              f"MAE={result['mae']:.3f} cm, RMSE={result['rmse']:.3f} cm")
    print("\nProceso terminado. Revisa las figuras en figures/.")
    print("La figura de predicción de distancia es sintética (MOCK).")


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
