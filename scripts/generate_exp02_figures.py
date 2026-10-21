#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experimento 02 — caracterización angular.

Genera dos figuras a partir de:
    data/angle_benchmark_patched_prototype.csv

Salidas (PNG y PDF):
    figures/exp02_fig1_angle_benchmark
    figures/exp02_fig3_angular_residuals

Se utilizan exclusivamente:
    station_id, target_deg, pulse_num,
    theta_tdoa_deg, theta_phase_deg, distance_cm

Se descartan deliberadamente:
    theta_energy_deg, delta_phi_deg, delta_r_cm

Ejecución desde la raíz del repositorio:
    python scripts/generate_exp02_figures.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_style import COLORS, FIGSIZE_WIDE, FIGSIZE_TALL, apply_style, style_axis, save_figure as common_save_figure


# ============================================================
# CONFIGURACIÓN
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "angle_benchmark_patched_prototype.csv"
FIGURES_DIR = ROOT / "figures"

BASELINE_CM = 5.0
DPI = 300
SAVE_PDF = True
SHOW_FIGURES = True

COLOR_PHASE = COLORS["phase"]
COLOR_TDOA = COLORS["tdoa"]
COLOR_IDEAL = COLORS["reference"]
COLOR_ZERO = COLORS["reference"]
apply_style()



# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def wrap_angle(angle_deg):
    """Envuelve un ángulo al intervalo [-180°, 180°)."""
    a = np.asarray(angle_deg, dtype=float)
    return (a + 180.0) % 360.0 - 180.0


def angular_error(estimated, reference):
    """Residuo angular envuelto: estimación menos referencia."""
    return wrap_angle(
        np.asarray(estimated, dtype=float)
        - np.asarray(reference, dtype=float)
    )


def circular_mean_deg(angles_deg):
    """Media circular de ángulos en grados."""
    a = np.asarray(angles_deg, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan

    rad = np.deg2rad(a)
    return float(np.rad2deg(np.arctan2(
        np.mean(np.sin(rad)),
        np.mean(np.cos(rad))
    )))


def circular_std_deg(angles_deg):
    """Desviación estándar circular en grados."""
    a = np.asarray(angles_deg, dtype=float)
    a = a[np.isfinite(a)]
    if a.size < 2:
        return 0.0

    rad = np.deg2rad(a)
    resultant = np.hypot(np.mean(np.sin(rad)), np.mean(np.cos(rad)))
    resultant = np.clip(resultant, 1e-12, 1.0)
    return float(np.rad2deg(np.sqrt(-2.0 * np.log(resultant))))


def polar_to_cartesian(radius, angle_deg):
    """Convención: 0° = +y; positivo hacia +x."""
    angle = np.deg2rad(angle_deg)
    return radius * np.sin(angle), radius * np.cos(angle)


def save_figure(fig, stem):
    common_save_figure(fig, FIGURES_DIR, stem)


# ============================================================
# CARGA Y VALIDACIÓN
# ============================================================

def load_data():
    print("\nLeyendo datos experimentales...")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de datos:\n{DATA_FILE}"
        )

    # El separador coma corresponde al CSV entregado.
    df = pd.read_csv(DATA_FILE)

    required = [
        "station_id",
        "target_deg",
        "pulse_num",
        "theta_tdoa_deg",
        "theta_phase_deg",
        "distance_cm",
    ]

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas requeridas: {missing}\n"
            f"Columnas disponibles: {list(df.columns)}"
        )

    numeric = [
        "target_deg",
        "pulse_num",
        "theta_tdoa_deg",
        "theta_phase_deg",
        "distance_cm",
    ]
    for column in numeric:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # La distancia solo es necesaria para la reconstrucción geométrica.
    # No se eliminan registros por distancia ausente si sus ángulos son válidos.
    df = df.dropna(
        subset=["target_deg", "theta_tdoa_deg", "theta_phase_deg"]
    ).copy()

    if df.empty:
        raise ValueError("No hay mediciones angulares válidas.")

    # Las columnas de energía y diferencias auxiliares no se utilizan.
    print(f"  Mediciones angulares válidas: {len(df)}")
    print(f"  Posiciones: {sorted(df['target_deg'].unique())}")
    print(f"  Estaciones: {df['station_id'].nunique()}")

    return df


# ============================================================
# RESUMEN POR POSICIÓN Y MÉTRICAS
# ============================================================

def summarize_by_target(df):
    """Resume estimaciones por ángulo objetivo usando estadística circular."""
    rows = []

    for target, group in df.groupby("target_deg", sort=True):
        phase = group["theta_phase_deg"].dropna().to_numpy()
        tdoa = group["theta_tdoa_deg"].dropna().to_numpy()
        distances = group["distance_cm"].dropna().to_numpy()

        rows.append({
            "target_deg": float(target),
            "phase_mean": circular_mean_deg(phase),
            "phase_std": circular_std_deg(phase),
            "phase_n": len(phase),
            "tdoa_mean": circular_mean_deg(tdoa),
            "tdoa_std": circular_std_deg(tdoa),
            "tdoa_n": len(tdoa),
            "distance_cm": float(np.median(distances)) if len(distances) else np.nan,
            "n_measurements": len(group),
        })

    return pd.DataFrame(rows)


def calculate_metrics(df):
    """Métricas globales calculadas sobre todos los residuos individuales."""
    phase_errors = angular_error(
        df["theta_phase_deg"].to_numpy(),
        df["target_deg"].to_numpy()
    )
    tdoa_errors = angular_error(
        df["theta_tdoa_deg"].to_numpy(),
        df["target_deg"].to_numpy()
    )

    def metrics(errors):
        errors = np.asarray(errors, dtype=float)
        errors = errors[np.isfinite(errors)]
        if errors.size == 0:
            return {"mae": np.nan, "sigma": np.nan, "bias": np.nan, "rmse": np.nan}
        return {
            "mae": float(np.mean(np.abs(errors))),
            "sigma": float(np.std(errors, ddof=1)) if errors.size > 1 else 0.0,
            "bias": float(np.mean(errors)),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
        }

    return {
        "phase_errors": phase_errors,
        "tdoa_errors": tdoa_errors,
        "phase": metrics(phase_errors),
        "tdoa": metrics(tdoa_errors),
    }


# ============================================================
# FIGURA 1 — BENCHMARK ANGULAR
# ============================================================

def generate_figure_1(summary, metrics, radius_cm):
    print("\nGenerando Figura 1: benchmark angular...")

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    targets = summary["target_deg"].to_numpy()

    x = np.linspace(-70, 70, 300)
    ax.fill_between(
        x, x - 4.0, x + 4.0,
        color=COLOR_TDOA, alpha=0.10,
        label=r"Corredor de aceptación ($\pm4^\circ$)",
        zorder=1
    )
    ax.plot(
        x, x, "--", color=COLOR_IDEAL, linewidth=1.8,
        label="Referencia ideal", zorder=2
    )

    ax.errorbar(
        targets,
        summary["phase_mean"],
        yerr=summary["phase_std"],
        fmt="o-", color=COLOR_PHASE, ecolor=COLOR_PHASE,
        capsize=4, elinewidth=1.4, markersize=7,
        label=(
            f"PDoA (MAE = {metrics['phase']['mae']:.1f}°, "
            f"σ = {metrics['phase']['sigma']:.1f}°)"
        ),
        zorder=4
    )

    ax.errorbar(
        targets,
        summary["tdoa_mean"],
        yerr=summary["tdoa_std"],
        fmt="s-", color=COLOR_TDOA, ecolor=COLOR_TDOA,
        capsize=4, elinewidth=1.4, markersize=7,
        label=(
            f"TDoA (MAE = {metrics['tdoa']['mae']:.1f}°, "
            f"σ = {metrics['tdoa']['sigma']:.1f}°)"
        ),
        zorder=3
    )

    ax.set_title(
        "Benchmark angular: PDoA y TDoA\n"
        f"(r ≈ {radius_cm:.1f} cm, separación d = {BASELINE_CM:.1f} cm)",
        pad=12
    )
    ax.set_xlabel(
        r"Ángulo de referencia $\theta_{\mathrm{ref}}$ [°]"
    )
    ax.set_ylabel(r"Ángulo estimado $\hat{\theta}$ [°]")
    ax.set_xlim(-70, 70)
    ax.set_ylim(-100, 100)
    ax.set_xticks(np.arange(-60, 61, 20))
    ax.set_yticks(np.arange(-100, 101, 25))
    ax.legend(loc="upper left", framealpha=0.95)
    fig.tight_layout()

    save_figure(fig, "exp02_fig1_angle_benchmark")
    return fig


# ============================================================
# FIGURA 3 — RESIDUOS ANGULARES
# ============================================================

def generate_figure_3(df):
    print("\nGenerando figura de residuos angulares...")

    work = df.copy()
    work["residual_phase_deg"] = angular_error(
        work["theta_phase_deg"], work["target_deg"]
    )
    work["residual_tdoa_deg"] = angular_error(
        work["theta_tdoa_deg"], work["target_deg"]
    )

    stats = []
    for target, group in work.groupby("target_deg", sort=True):
        for method, column in [
            ("phase", "residual_phase_deg"),
            ("tdoa", "residual_tdoa_deg")
        ]:
            values = group[column].dropna().to_numpy()
            stats.append({
                "target_deg": target,
                "method": method,
                "bias": float(np.mean(values)) if len(values) else np.nan,
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            })

    stats = pd.DataFrame(stats)
    phase_stats = stats[stats["method"] == "phase"]
    tdoa_stats = stats[stats["method"] == "tdoa"]

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    rng = np.random.default_rng(42)
    jitter = 1.5

    ax.axhline(
        0, color=COLOR_ZERO, linestyle="--", linewidth=1.5,
        label="Residuo nulo", zorder=1
    )

    for target, group in work.groupby("target_deg", sort=True):
        phase = group["residual_phase_deg"].dropna().to_numpy()
        tdoa = group["residual_tdoa_deg"].dropna().to_numpy()

        phase_x = target - jitter + rng.uniform(-0.6, 0.6, len(phase))
        tdoa_x = target + jitter + rng.uniform(-0.6, 0.6, len(tdoa))

        ax.scatter(
            phase_x, phase, color=COLOR_PHASE, alpha=0.28,
            s=24, edgecolors="none", zorder=2
        )
        ax.scatter(
            tdoa_x, tdoa, color=COLOR_TDOA, alpha=0.28,
            s=24, edgecolors="none", zorder=2
        )

    ax.errorbar(
        phase_stats["target_deg"] - jitter,
        phase_stats["bias"],
        yerr=phase_stats["std"],
        fmt="o-", color=COLOR_PHASE, markersize=7,
        linewidth=1.8, capsize=4, elinewidth=1.5,
        label="PDoA: media ± 1σ", zorder=4
    )
    ax.errorbar(
        tdoa_stats["target_deg"] + jitter,
        tdoa_stats["bias"],
        yerr=tdoa_stats["std"],
        fmt="s-", color=COLOR_TDOA, markersize=7,
        linewidth=1.8, capsize=4, elinewidth=1.5,
        label="TDoA: media ± 1σ", zorder=4
    )

    ax.set_xlabel(r"Ángulo de referencia $\theta_{\mathrm{target}}$ [°]")
    ax.set_ylabel(r"Residuo angular $e_\theta$ [°]")
    ax.set_title("Residuos angulares: PDoA y TDoA", pad=12)
    targets = np.sort(work["target_deg"].unique())
    ax.set_xticks(targets)
    ax.minorticks_on()
    ax.grid(True, which="major", linestyle="-", alpha=0.25)
    ax.grid(True, which="minor", linestyle=":", alpha=0.15)
    ax.legend(loc="best", framealpha=0.96)

    all_residuals = np.concatenate([
        work["residual_phase_deg"].dropna().to_numpy(),
        work["residual_tdoa_deg"].dropna().to_numpy()
    ])
    max_abs = float(np.max(np.abs(all_residuals))) if all_residuals.size else 10.0
    limit = max(10.0, np.ceil(max_abs / 10.0) * 10.0)
    ax.set_ylim(-limit * 1.15, limit * 1.15)

    style_axis(ax)
    fig.tight_layout()
    save_figure(fig, "exp02_fig3_angular_residuals")

    return work


# ============================================================
# INFORME ESTADÍSTICO
# ============================================================

def print_results(summary, metrics, residual_data):
    print("\n" + "=" * 62)
    print("RESULTADOS — EXPERIMENTO 02")
    print("=" * 62)
    print(f"Posiciones angulares: {len(summary)}")
    print(f"Mediciones válidas:   {len(residual_data)}")

    for name, key in [("PDoA", "phase"), ("TDoA", "tdoa")]:
        m = metrics[key]
        print(f"\n{name}")
        print("-" * 42)
        print(f"Sesgo medio         : {m['bias']:.3f}°")
        print(f"MAE                 : {m['mae']:.3f}°")
        print(f"RMSE                : {m['rmse']:.3f}°")
        print(f"Desviación estándar : {m['sigma']:.3f}°")

    print("\nResumen por posición (medias y desviaciones circulares):")
    print(
        f"{'Target':>9} {'PDoA':>10} {'σ PDoA':>10} "
        f"{'TDoA':>10} {'σ TDoA':>10} {'N':>6}"
    )
    for _, row in summary.iterrows():
        print(
            f"{row['target_deg']:>8.1f}° "
            f"{row['phase_mean']:>9.2f}° "
            f"{row['phase_std']:>9.2f}° "
            f"{row['tdoa_mean']:>9.2f}° "
            f"{row['tdoa_std']:>9.2f}° "
            f"{int(row['n_measurements']):>6}"
        )


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    print("=" * 62)
    print("EXPERIMENTO 02 — CARACTERIZACIÓN ANGULAR")
    print("=" * 62)

    df = load_data()
    summary = summarize_by_target(df)
    metrics = calculate_metrics(df)

    valid_distances = df["distance_cm"].dropna()
    if valid_distances.empty:
        raise ValueError(
            "La columna distance_cm no contiene datos válidos; "
            "no se puede generar la reconstrucción 2D a escala."
        )
    radius_cm = float(valid_distances.median())

    generate_figure_1(summary, metrics, radius_cm)
    residual_data = generate_figure_3(df)

    print_results(summary, metrics, residual_data)

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close("all")

    print("\nProceso terminado. Se generaron el benchmark y los residuos angulares.")


if __name__ == "__main__":
    main()
