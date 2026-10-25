
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generación de figuras del Experimento 01: caracterización radial.

Entradas:
    data/distance_characterization_raw_mic1.csv
    data/distance_characterization_raw_mic2.csv
    data/exp01_rssi_distance_prediction.csv

Salidas:
    figures/exp01_fig1_calibration_four_panel.png
    figures/exp01_fig2_method_distance_comparison.png
    figures/exp01_fig3_method_residual_comparison.png
    data/exp01_rssi_validation_metrics.csv

La validación RSSI utiliza coeficientes k fijados previamente.
No reajusta k con los datos de validación.

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

# Coeficientes obtenidos durante la calibración original.
# Se mantienen congelados: NO se estiman a partir del CSV de validación.
RSSI_VALIDATION_FILE = DATA_DIR / "exp01_rssi_distance_prediction.csv"
FIXED_AMPLITUDE_K = {
    "Micrófono 1": 20.6934438197599,
    "Micrófono 2": 18.8589498970794,
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
# 6. VALIDACIÓN INDEPENDIENTE RSSI (DATOS REALES)
# ============================================================

def load_rssi_validation():
    """Carga datos nuevos y aplica los coeficientes k congelados."""
    if not RSSI_VALIDATION_FILE.exists():
        raise FileNotFoundError(f"No se encontró {RSSI_VALIDATION_FILE}")
    df = pd.read_csv(RSSI_VALIDATION_FILE)
    required = {"microphone", "repeat", "r_ref_cm", "k_amplitude", "amp_mv"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en CSV RSSI: {sorted(missing)}")
    for col in ["repeat", "r_ref_cm", "k_amplitude", "amp_mv"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=list(required)).copy()
    df = df[(df.r_ref_cm > 0) & (df.amp_mv > 0)].copy()

    unknown = set(df.microphone.unique()) - set(FIXED_AMPLITUDE_K)
    if unknown:
        raise ValueError(f"Micrófonos desconocidos: {sorted(unknown)}")

    for mic, group in df.groupby("microphone"):
        if not np.allclose(group.k_amplitude, FIXED_AMPLITUDE_K[mic],
                           rtol=0, atol=1e-8):
            raise ValueError(f"k de {mic} no coincide con el k congelado.")

    # k [mV*m] / amplitud [mV] = distancia [m]; convertir a cm.
    df["r_est_cm_recomputed"] = df.apply(
        lambda row: 100 * FIXED_AMPLITUDE_K[row.microphone] / row.amp_mv,
        axis=1,
    )
    if "r_est_cm" in df:
        recorded = pd.to_numeric(df.r_est_cm, errors="coerce")
        valid = recorded.notna()
        if valid.any() and not np.allclose(
            recorded[valid], df.loc[valid, "r_est_cm_recomputed"],
            rtol=1e-7, atol=1e-7
        ):
            raise ValueError("r_est_cm no coincide con k fijo / amp_mv.")

    df["error_cm"] = df.r_est_cm_recomputed - df.r_ref_cm
    df["abs_error_cm"] = df.error_cm.abs()
    df["abs_percentage_error"] = 100 * df.abs_error_cm / df.r_ref_cm
    return df.sort_values(["microphone", "r_ref_cm", "repeat"]).reset_index(drop=True)


def rssi_metrics(group):
    e = group.error_cm.to_numpy(dtype=float)
    return {
        "n": len(e),
        "bias_cm": np.mean(e),
        "mae_cm": np.mean(np.abs(e)),
        "rmse_cm": np.sqrt(np.mean(e**2)),
        "mape_pct": group.abs_percentage_error.mean(),
        "max_abs_error_cm": np.max(np.abs(e)),
    }


def summarize_rssi_validation(df):
    rows = []
    for mic in list(FIXED_AMPLITUDE_K) + ["Global"]:
        g = df if mic == "Global" else df[df.microphone == mic]
        if not g.empty:
            rows.append({"microphone": mic, **rssi_metrics(g)})
    return pd.DataFrame(rows)


def _group_mean_std(df, x_col, y_col):
    """Media, desviación estándar y N por distancia de referencia."""
    return (df.groupby(x_col)[y_col].agg(["mean", "std", "count"])
            .reset_index().sort_values(x_col))


def _plot_method_prediction(ax, datasets, x_col, y_col, panel_title,
                            x_label, y_label):
    """Panel de predicción: medias ± DE para ambos micrófonos."""
    for mic, frame in datasets.items():
        stats = _group_mean_std(frame, x_col, y_col)
        stats["std"] = stats["std"].fillna(0.0)
        n_total = int(stats["count"].sum())
        n_per_distance = int(stats["count"].min()) if len(stats) else 0
        ax.errorbar(
            stats[x_col], stats["mean"], yerr=stats["std"], fmt="o-",
            color=COLORS[mic], ecolor=COLORS[mic], markersize=5.2,
            markeredgecolor="white", markeredgewidth=0.6, capsize=3.2,
            elinewidth=1.2, linewidth=1.7,
            label=f"{mic} (n={n_total}; {n_per_distance}/dist.)", zorder=4,
        )
    ideal = np.linspace(10, 105, 300)
    ax.plot(ideal, ideal, "--", color=COLORS["reference"],
            linewidth=1.5, label="Referencia ideal", zorder=1)
    ax.set(xlim=(10, 105), ylim=(10, 110), xlabel=x_label,
           ylabel=y_label, title=panel_title)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=8.5)
    style_axis(ax)


def _plot_residual_panel(ax, datasets, x_col, error_col, panel_title,
                         ylabel, offsets, show_xlabel=False, show_legend=True):
    """Residuos: mediciones individuales y media ± DE."""
    for mic, frame in datasets.items():
        offset = offsets[mic]
        ax.scatter(frame[x_col].to_numpy(dtype=float) + offset,
                   frame[error_col].to_numpy(dtype=float), s=17, alpha=0.28,
                   color=COLORS[mic], edgecolors="none",
                   label=f"{mic}: mediciones", zorder=2)
        stats = _group_mean_std(frame, x_col, error_col)
        stats["std"] = stats["std"].fillna(0.0)
        ax.errorbar(stats[x_col] + offset, stats["mean"], yerr=stats["std"],
                    fmt="o", color=COLORS[mic], ecolor=COLORS[mic],
                    markersize=5.0, markeredgecolor="white", markeredgewidth=0.6,
                    capsize=3.0, elinewidth=1.2,
                    label=f"{mic}: media ± DE", zorder=4)
    ax.axhline(0, color=COLORS["reference"], linestyle="--",
               linewidth=1.5, label="Error nulo", zorder=1)
    ax.set_title(panel_title)
    ax.set_ylabel(ylabel)
    if show_xlabel:
        ax.set_xlabel(r"Distancia de referencia $r_{\mathrm{ref}}$ [cm]")
    else:
        ax.set_xlabel("")
    ax.set_xlim(10, 105)
    if show_legend:
        ax.legend(loc="best", framealpha=0.95, fontsize=8.0, ncol=2)
    style_axis(ax)


def plot_method_comparisons(rssi_df, toa_data, metrics):
    """Dos figuras comparables: predicción RSSI/ToA y residuos RSSI/ToA."""
    rssi, toa = {}, {}
    for mic in FIXED_AMPLITUDE_K:
        rssi[mic] = rssi_df[rssi_df.microphone == mic].copy()
        raw = toa_data[mic][["r_tape_cm", "r_toa_cm"]].dropna().copy()
        raw = raw[(raw.r_tape_cm > 0) & (raw.r_toa_cm > 0)]
        raw["error_cm"] = raw.r_toa_cm - raw.r_tape_cm
        toa[mic] = raw

    # Figura 2: predicción lado a lado, con escalas idénticas.
    fig, (ax_rssi, ax_toa) = plt.subplots(
        1, 2, figsize=(15.0, 6.0), sharex=True, sharey=True)
    _plot_method_prediction(
        ax_rssi, rssi, "r_ref_cm", "r_est_cm_recomputed", "(a) RSSI",
        r"Distancia de referencia $r_{\mathrm{ref}}$ [cm]",
        r"Distancia estimada $\hat r$ [cm]",
    )
    _plot_method_prediction(
        ax_toa, toa, "r_tape_cm", "r_toa_cm", "(b) ToA",
        r"Distancia de referencia $r_{\mathrm{ref}}$ [cm]",
        r"Distancia estimada $r_{\mathrm{ToA}}$ [cm]",
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.13, top=0.92, wspace=0.22)
    save_figure(fig, "exp01_fig2_method_distance_comparison")

    # Figura 3: residuos apilados, ambos en cm y con límites comunes.
    fig, (ax_rssi_res, ax_toa_res) = plt.subplots(
        2, 1, figsize=(11.5, 9.0), sharex=True, sharey=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.16})
    offsets = {"Micrófono 1": -0.7, "Micrófono 2": 0.7}
    _plot_residual_panel(
        ax_rssi_res, rssi, "r_ref_cm", "error_cm", "(a) RSSI",
        r"Residuo $e=\hat r-r_{\mathrm{ref}}$ [cm]", offsets)
    _plot_residual_panel(
        ax_toa_res, toa, "r_tape_cm", "error_cm", "(b) ToA",
        r"Residuo $e=r_{\mathrm{ToA}}-r_{\mathrm{ref}}$ [cm]", offsets,
        show_xlabel=True, show_legend=False)
    all_errors = np.concatenate([
        frame.error_cm.to_numpy(dtype=float)
        for frame in list(rssi.values()) + list(toa.values()) if not frame.empty
    ])
    max_abs = float(np.nanmax(np.abs(all_errors)))
    y_limit = max(2.0, float(np.ceil(max_abs * 1.10)))
    ax_rssi_res.set_ylim(-y_limit, y_limit)
    ax_toa_res.set_ylim(-y_limit, y_limit)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.09, top=0.95, hspace=0.20)
    save_figure(fig, "exp01_fig3_method_residual_comparison")

    metrics_path = DATA_DIR / "exp01_rssi_validation_metrics.csv"
    metrics.to_csv(metrics_path, index=False, float_format="%.8f")
    print(f"  [+] Métricas RSSI guardadas en: {metrics_path}")


def main():
    print("\n" + "=" * 60)
    print("EXPERIMENTO 01 - CARACTERIZACIÓN RADIAL")
    print("=" * 60)
    data = load_data()

    print("\nGenerando calibración combinada de amplitud y energía...")
    calibration_fits = plot_combined_calibration(data)
    amp_fits = calibration_fits["amplitude"]

    print("\nValidando el estimador RSSI con datos nuevos y k congelados...")
    validation = load_rssi_validation()
    validation_metrics = summarize_rssi_validation(validation)
    plot_method_comparisons(validation, data, validation_metrics)
    print(validation_metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nCOEFICIENTES DE CALIBRACIÓN")
    for observable, results in calibration_fits.items():
        print(f"\n{observable.upper()}")
        for name, (k, r2) in results.items():
            print(f"  {name}: k={k:.6g}, R²={r2:.5f}")

    print("\nProceso terminado. Revisa las figuras en figures/.")
    print("RSSI usa datos experimentales nuevos y k congelados; RSSI y ToA se comparan con escalas comunes.")


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
