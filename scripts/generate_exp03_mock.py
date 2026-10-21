#!/usr/bin/env python3
"""
MOCK SINTÉTICO — Experimento 03: cinemática Doppler y comparación con Tracker.

Genera:
  figures/mock_exp03_fig1_doppler_frequencies.{png,pdf}
  figures/mock_exp03_fig2_velocity_comparison.{png,pdf}
  figures/mock_exp03_fig3_position_comparison.{png,pdf}
  data/mock_exp03/mock_exp03_synthetic_data.csv

IMPORTANTE:
Los datos son completamente sintéticos. No son mediciones de WALO ni de Tracker.
Las frecuencias representan estimaciones procesadas, no muestras acústicas crudas.
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_style import COLORS, FIGSIZE_WIDE, FIGSIZE_TALL, apply_style, style_axis, save_figure as common_save_figure


# ---------------------------------------------------------------------
# Estilo y constantes del montaje
# ---------------------------------------------------------------------

apply_style()

RAIL_LENGTH_CM = 200.0
X_START_CM = 10.0
X_EQ_CM = 100.0
X_END_CM = 190.0
N_TRIALS = 10
DURATION_S = 10.0
F0_HZ = 2620.0
TEMPERATURE_C = 20.0
SOUND_SPEED = 331.3 + 0.606 * TEMPERATURE_C  # m/s


def moving_average(values, window):
    """Suavizado sencillo, sin dependencia de SciPy."""
    window = max(1, int(window))
    if window == 1:
        return values.copy()
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(values, (window // 2, window - 1 - window // 2),
                    mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def damped_trajectory(t, amplitude, gamma, omega):
    """
    Oscilador amortiguado iniciado en un extremo de retorno:
    x(0) = x_eq - amplitude y v(0) = 0.

    La forma incluye el término gamma/omega para imponer v(0)=0.
    """
    decay = np.exp(-gamma * t)
    phase = omega * t
    shape = np.cos(phase) + (gamma / omega) * np.sin(phase)

    x = X_EQ_CM - amplitude * decay * shape
    v = amplitude * decay * (omega + gamma**2 / omega) * np.sin(phase)
    return x, v


def simulate_trial(trial_index, t_walo, rng, tracker_rate):
    """Simula un ensayo independiente y sus dos vías de observación."""

    # Variabilidad física moderada entre liberaciones.
    amplitude = rng.normal(90.0, 1.8)
    gamma = rng.normal(0.105, 0.008)
    omega = rng.normal(2.75, 0.035)

    x_true, v_true_cm_s = damped_trajectory(
        t_walo, amplitude, gamma, omega
    )

    # Variaciones suaves de trayectoria entre ensayos.
    phase_jitter = rng.normal(0.0, 0.012)
    x_true, v_true_cm_s = damped_trajectory(
        t_walo + phase_jitter, amplitude, gamma, omega
    )

    # Limitar únicamente por seguridad geométrica del dibujo.
    # No se simula un modelo de impacto con los topes.
    x_true = np.clip(x_true, 5.0, 195.0)

    # -------------------------------------------------------------
    # Canal acústico: frecuencias Doppler en micrófonos opuestos.
    # Micrófono 1: la fuente se aleja al moverse hacia +x.
    # Micrófono 2: la fuente se acerca al moverse hacia +x.
    # -------------------------------------------------------------
    v_m_s = v_true_cm_s / 100.0

    # Deriva común lenta de la frecuencia del buzzer.
    drift = (
        2.0 * np.sin(2 * np.pi * 0.13 * t_walo + rng.uniform(0, 2*np.pi))
        + 0.9 * np.sin(2 * np.pi * 0.035 * t_walo)
        + np.cumsum(rng.normal(0.0, 0.018, len(t_walo)))
    )
    f_common_true = F0_HZ + drift

    f1_ideal = f_common_true * SOUND_SPEED / (SOUND_SPEED + v_m_s)
    f2_ideal = f_common_true * SOUND_SPEED / (SOUND_SPEED - v_m_s)

    # Ruido de estimación de frecuencia: blanco + componente correlacionada
    # + algunos valores atípicos pequeños, como puede ocurrir en seguimiento.
    noise1 = rng.normal(0.0, 0.85, len(t_walo))
    noise2 = rng.normal(0.0, 0.85, len(t_walo))
    colored1 = moving_average(rng.normal(0.0, 1.0, len(t_walo)), 5) * 1.0
    colored2 = moving_average(rng.normal(0.0, 1.0, len(t_walo)), 5) * 1.0

    outlier_mask1 = rng.random(len(t_walo)) < 0.004
    outlier_mask2 = rng.random(len(t_walo)) < 0.004
    outliers1 = outlier_mask1 * rng.normal(0.0, 5.0, len(t_walo))
    outliers2 = outlier_mask2 * rng.normal(0.0, 5.0, len(t_walo))

    f1 = f1_ideal + noise1 + colored1 + outliers1
    f2 = f2_ideal + noise2 + colored2 + outliers2
    f_common_measured = 0.5 * (f1 + f2)

    # Inversión Doppler diferencial prevista en el protocolo.
    v_walo_cm_s = (
        SOUND_SPEED * (f2 - f1) / (f1 + f2) * 100.0
    )

    # Integración trapezoidal desde la posición inicial conocida.
    x_walo_cm = np.empty_like(t_walo)
    x_walo_cm[0] = X_START_CM
    dt = np.diff(t_walo)
    x_walo_cm[1:] = X_START_CM + np.cumsum(
        0.5 * (v_walo_cm_s[1:] + v_walo_cm_s[:-1]) * dt
    )

    # -------------------------------------------------------------
    # Tracker: muestreo óptico independiente a menor frecuencia.
    # Simula una escala espacial, ruido de seguimiento y desfase temporal.
    # -------------------------------------------------------------
    dt_tracker = 1.0 / tracker_rate
    t_tracker = np.arange(0.0, DURATION_S + 0.5 * dt_tracker, dt_tracker)

    sync_offset = rng.normal(0.0, 0.012)
    t_tracker_shifted = np.clip(t_tracker + sync_offset, 0.0, DURATION_S)

    x_tracker_true, _ = damped_trajectory(
        t_tracker_shifted + phase_jitter, amplitude, gamma, omega
    )
    scale_error = rng.normal(1.0, 0.0025)
    zero_error = rng.normal(0.0, 0.25)
    tracker_noise = rng.normal(0.0, 0.55, len(t_tracker))
    x_tracker_raw = (
        X_START_CM
        + (x_tracker_true - X_START_CM) * scale_error
        + zero_error
        + tracker_noise
    )

    # Suavizado moderado antes de derivar; evita amplificar el ruido de píxel.
    smooth_window = max(5, int(round(tracker_rate * 0.20)) | 1)
    x_tracker_smooth = moving_average(x_tracker_raw, smooth_window)
    v_tracker_cm_s = np.gradient(x_tracker_smooth, t_tracker)

    # Interpolar Tracker sobre la malla temporal WALO para la comparación.
    x_tracker_on_walo = np.interp(t_walo, t_tracker, x_tracker_smooth)
    v_tracker_on_walo = np.interp(t_walo, t_tracker, v_tracker_cm_s)

    return {
        "trial": trial_index + 1,
        "t_walo": t_walo,
        "f1": f1,
        "f2": f2,
        "f_common": f_common_measured,
        "v_walo": v_walo_cm_s,
        "x_walo": x_walo_cm,
        "t_tracker": t_tracker,
        "x_tracker_raw": x_tracker_raw,
        "x_tracker": x_tracker_smooth,
        "v_tracker": v_tracker_cm_s,
        "x_tracker_on_walo": x_tracker_on_walo,
        "v_tracker_on_walo": v_tracker_on_walo,
        "x_true": x_true,
        "v_true": v_true_cm_s,
    }


def save_figure(fig, output_base):
    common_save_figure(fig, output_base.parent, output_base.name)


def plot_frequencies(trials, output_dir):
    fig, axes = plt.subplots(2, 1, figsize=FIGSIZE_TALL, sharex=True, gridspec_kw={"height_ratios": [1.35, 1.0], "hspace": 0.28})

    colors = {"f1": COLORS["mic1"], "f2": COLORS["mic2"], "common": COLORS["reference"]}

    # Diez ensayos en transparencia; el primer ensayo se destaca.
    for i, trial in enumerate(trials):
        alpha = 0.22 if i != 0 else 0.75
        lw = 0.8 if i != 0 else 1.15
        axes[0].plot(trial["t_walo"], trial["f1"], color=colors["f1"],
                     alpha=alpha, lw=lw)
        axes[0].plot(trial["t_walo"], trial["f2"], color=colors["f2"],
                     alpha=alpha, lw=lw)
        axes[1].plot(trial["t_walo"], trial["f_common"],
                     color=colors["common"], alpha=0.20, lw=0.8)

    mean_f1 = np.mean([tr["f1"] for tr in trials], axis=0)
    mean_f2 = np.mean([tr["f2"] for tr in trials], axis=0)
    mean_fc = np.mean([tr["f_common"] for tr in trials], axis=0)

    axes[0].plot(trials[0]["t_walo"], mean_f1, color=colors["f1"],
                 lw=1.8, label=r"Micrófono 1 ($f_1$), media")
    axes[0].plot(trials[0]["t_walo"], mean_f2, color=colors["f2"],
                 lw=1.8, label=r"Micrófono 2 ($f_2$), media")
    axes[0].set_title("Frecuencias recibidas en los dos micrófonos")
    axes[0].set_ylabel("Frecuencia [Hz]")
    axes[0].legend(loc="upper right", ncol=2)

    axes[1].plot(trials[0]["t_walo"], mean_fc, color=colors["common"],
                 lw=1.8, label=r"Frecuencia común $(f_1+f_2)/2$")
    axes[1].axhline(F0_HZ, color="0.35", ls="--", lw=1.0,
                    label=fr"Portadora nominal ({F0_HZ:.0f} Hz)")
    axes[1].set_title("Componente común y deriva de la fuente")
    axes[1].set_ylabel("Frecuencia [Hz]")
    axes[1].set_xlabel("Tiempo [s]")
    axes[1].legend(loc="upper right")

    fig.text(0.5, 0.005, "MOCK — datos sintéticos; no son mediciones experimentales",
             ha="center", va="bottom", fontsize=8, color=COLORS["mock"])
    for ax in axes:
        style_axis(ax)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    save_figure(fig, output_dir / "mock_exp03_fig1_doppler_frequencies")


def plot_velocity(trials, output_dir):
    t = trials[0]["t_walo"]
    vw = np.array([tr["v_walo"] for tr in trials])
    vt = np.array([tr["v_tracker_on_walo"] for tr in trials])

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE, sharey=True)

    panels = [
        (axes[0], vw, "WALO — Doppler diferencial", "WALO"),
        (axes[1], vt, "Tracker — referencia óptica", "Tracker"),
    ]

    for ax, values, title, label in panels:
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0, ddof=1)

        for i, row in enumerate(values):
            ax.plot(t, row, color=COLORS["mic1"] if label == "WALO" else COLORS["mic2"],
                    alpha=0.17, lw=0.8)

        color = "#0072B2" if label == "WALO" else "#D55E00"
        ax.fill_between(t, mean - std, mean + std, color=color,
                        alpha=0.18, label=r"Dispersión entre ensayos ($\pm1\sigma$)")
        ax.plot(t, mean, color=color, lw=2.0, label=r"Media $\bar{v}(t)$")
        ax.axhline(0, color="0.25", ls="--", lw=1.0)
        ax.set_title(title)
        ax.set_xlabel("Tiempo [s]")
        ax.legend(loc="upper right")

    axes[0].set_ylabel("Velocidad [cm/s]")
    fig.text(0.5, 0.005, "MOCK — datos sintéticos; no son mediciones experimentales",
             ha="center", va="bottom", fontsize=8, color=COLORS["mock"])
    for ax in axes:
        style_axis(ax)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    save_figure(fig, output_dir / "mock_exp03_fig2_velocity_comparison")


def plot_position(trials, output_dir):
    t = trials[0]["t_walo"]
    xw = np.array([tr["x_walo"] for tr in trials])
    xt = np.array([tr["x_tracker_on_walo"] for tr in trials])

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE, sharey=True)

    panels = [
        (axes[0], xw, "WALO — Posición integrada", COLORS["mic1"]),
        (axes[1], xt, "Tracker — Posición óptica", COLORS["mic2"]),
    ]

    for ax, values, title, color in panels:
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0, ddof=1)

        for row in values:
            ax.plot(t, row, color=color, alpha=0.17, lw=0.8)

        ax.fill_between(t, mean - std, mean + std, color=color,
                        alpha=0.18, label=r"Dispersión entre ensayos ($\pm1\sigma$)")
        ax.plot(t, mean, color=color, lw=2.1, label=r"Media $\bar{x}(t)$")
        ax.axhline(X_EQ_CM, color="0.2", ls="--", lw=1.0,
                   label="Centro del recorrido (100 cm)")
        ax.axhline(X_START_CM, color=COLORS["reference"], ls=":", lw=1.1,
                   label="Posición inicial (10 cm)")
        ax.axhline(X_END_CM, color=COLORS["reference"], ls=":", lw=1.1,
                   label="Extremo útil opuesto (190 cm)")
        ax.set_title(title)
        ax.set_xlabel("Tiempo [s]")
        ax.set_ylim(-5, 205)
        ax.legend(loc="best", fontsize=7.5, framealpha=0.95)

    axes[0].set_ylabel("Posición [cm]")
    fig.text(0.5, 0.005, "MOCK — datos sintéticos; no son mediciones experimentales",
             ha="center", va="bottom", fontsize=8, color=COLORS["mock"])
    for ax in axes:
        style_axis(ax)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    save_figure(fig, output_dir / "mock_exp03_fig3_position_comparison")


def export_csv(trials, output_path):
    """Exporta una tabla larga, con una fila por instante y ensayo."""
    rows = []
    for tr in trials:
        for i, time_s in enumerate(tr["t_walo"]):
            rows.append({
                "trial": tr["trial"],
                "time_s": time_s,
                "f_mic1_hz": tr["f1"][i],
                "f_mic2_hz": tr["f2"][i],
                "f_common_hz": tr["f_common"][i],
                "velocity_walo_cm_s": tr["v_walo"][i],
                "position_walo_cm": tr["x_walo"][i],
                "velocity_tracker_cm_s": tr["v_tracker_on_walo"][i],
                "position_tracker_cm": tr["x_tracker_on_walo"][i],
                "position_synthetic_reference_cm": tr["x_true"][i],
                "velocity_synthetic_reference_cm_s": tr["v_true"][i],
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"  CSV: {output_path}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Genera figuras mock sintéticas para el Experimento 03."
    )
    parser.add_argument("--duration", type=float, default=DURATION_S,
                        help="Duración de cada ensayo en segundos (default: 10).")
    parser.add_argument("--trials", type=int, default=N_TRIALS,
                        help="Número de ensayos (default: 10).")
    parser.add_argument("--tracker-fps", type=float, default=30.0,
                        help="Frecuencia de cuadros de Tracker (default: 30).")
    parser.add_argument("--walo-rate", type=float, default=100.0,
                        help="Tasa temporal de estimaciones procesadas WALO (default: 100 Hz).")
    parser.add_argument("--seed", type=int, default=20261020,
                        help="Semilla para reproducibilidad.")
    args = parser.parse_args()

    if args.trials < 2:
        parser.error("--trials debe ser al menos 2.")
    if args.duration <= 0 or args.tracker_fps <= 0 or args.walo_rate <= 0:
        parser.error("Duración y tasas deben ser positivas.")

    root = Path(__file__).resolve().parents[1]
    figures_dir = root / "figures"
    data_dir = root / "data" / "mock_exp03"
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    dt_walo = 1.0 / args.walo_rate
    t_walo = np.arange(0.0, args.duration + 0.5 * dt_walo, dt_walo)

    print("=" * 66)
    print("MOCK SINTÉTICO — EXPERIMENTO 03")
    print("=" * 66)
    print(f"Ensayos:              {args.trials}")
    print(f"Duración por ensayo:  {args.duration:.2f} s")
    print(f"Inicio:               x = {X_START_CM:.1f} cm (extremo Mic. 1)")
    print(f"Extremos útiles:      {X_START_CM:.1f}–{X_END_CM:.1f} cm")
    print(f"Tracker:              {args.tracker_fps:.1f} fps")
    print(f"Estimación WALO:      {args.walo_rate:.1f} Hz")
    print(f"Temperatura simulada: {TEMPERATURE_C:.1f} °C")
    print()
    print("Generando ensayos sintéticos...")

    trials = [
        simulate_trial(i, t_walo, rng, args.tracker_fps)
        for i in range(args.trials)
    ]

    print("Generando Figura 1: frecuencias Doppler...")
    plot_frequencies(trials, figures_dir)

    print("Generando Figura 2: comparación de velocidad...")
    plot_velocity(trials, figures_dir)

    print("Generando Figura 3: comparación de posición...")
    plot_position(trials, figures_dir)

    print("Exportando datos sintéticos...")
    df = export_csv(trials, data_dir / "mock_exp03_synthetic_data.csv")

    print()
    print("=" * 66)
    print("RESUMEN DEL MOCK — NO SON RESULTADOS EXPERIMENTALES")
    print("=" * 66)
    for method, col in [
        ("WALO", "velocity_walo_cm_s"),
        ("Tracker", "velocity_tracker_cm_s"),
    ]:
        values = df[col].to_numpy()
        print(f"{method:8s}: media global = {np.mean(values):8.2f} cm/s, "
              f"desviación = {np.std(values):7.2f} cm/s")

    print()
    print("Proceso terminado.")
    print("Se generaron tres figuras y un CSV sintético.")
    print("No utilizar estos datos como evidencia experimental.")


if __name__ == "__main__":
    main()
