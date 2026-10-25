#!/usr/bin/env python3
"""
Genera las tres figuras finales planificadas para Exp03 y sus métricas reproducibles.

Entradas:
  data/experiments/exp03_walo_doppler_10trials_10s.xlsx
  data/experiments/exp03_tracker_walo_aligned.csv

Salidas:
  figures/experiments/exp03_fig1_frequencies_with_rest_reference.png
  figures/experiments/exp03_fig2_individual_mic_velocities.png
  figures/experiments/exp03_fig3_joint_velocity_and_position.png
  data/analysis/exp03_doppler/manuscript_figures/exp03_final_metrics.csv
  data/analysis/exp03_doppler/manuscript_figures/exp03_velocity_metrics_by_sensor.csv
  data/analysis/exp03_doppler/manuscript_figures/exp03_frequency_statistics.csv
  data/analysis/exp03_doppler/manuscript_figures/exp03_sensor_baseline_diagnostics.csv
  data/analysis/exp03_doppler/manuscript_figures/exp03_real_figures_manifest.csv

Las métricas se calculan con datos sin suavizar. El suavizado solo se usa
para visualizar. MIC1 y MIC2 se recalculan desde las frecuencias medidas, usando una referencia independiente por canal y ensayo, estimada en un intervalo de reposo configurable. No se ajustan offsets con el Tracker.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import pearsonr
from scipy.signal import medfilt

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

def require_columns(df: pd.DataFrame, columns: set[str], source: str) -> None:
    missing = columns.difference(df.columns)
    if missing:
        raise ValueError(f"{source}: faltan columnas {sorted(missing)}")

def smooth(values, window: int, polyorder: int = 3) -> np.ndarray:
    """Savitzky-Golay para visualización; nunca modifica los datos fuente."""
    y = np.asarray(values, dtype=float).copy()
    if y.size < polyorder + 2:
        return y
    y = pd.Series(y).interpolate(limit_direction="both").to_numpy(float)
    w = min(window, y.size if y.size % 2 else y.size - 1)
    if w % 2 == 0:
        w -= 1
    if w <= polyorder:
        return y
    return savgol_filter(y, window_length=w, polyorder=polyorder, mode="interp")

def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(pearsonr(x, y).statistic)

def safe_ratio(a: float, b: float) -> float:
    return float(a / b) if np.isfinite(b) and b != 0 else float("nan")

def velocity_metrics(walo: np.ndarray, tracker: np.ndarray,
                     movement_valid: np.ndarray | None = None) -> dict:
    mask = (
        np.isfinite(walo) & np.isfinite(tracker)
        & (np.abs(walo) <= 3.0) & (np.abs(tracker) <= 3.0)
    )
    if movement_valid is not None:
        mask &= np.asarray(movement_valid, dtype=bool)
    w, t = walo[mask], tracker[mask]
    if not len(w):
        return {k: np.nan for k in (
            "n", "r", "mae_m_s", "rmse_m_s", "bias_m_s",
            "rms_walo_m_s", "rms_tracker_m_s", "rms_ratio"
        )}
    d = w - t
    rms_w = float(np.sqrt(np.mean(w ** 2)))
    rms_t = float(np.sqrt(np.mean(t ** 2)))
    return {
        "n": int(len(w)),
        "r": pearson(w, t),
        "mae_m_s": float(np.mean(np.abs(d))),
        "rmse_m_s": float(np.sqrt(np.mean(d ** 2))),
        "bias_m_s": float(np.mean(d)),
        "rms_walo_m_s": rms_w,
        "rms_tracker_m_s": rms_t,
        "rms_ratio": safe_ratio(rms_w, rms_t),
    }

def add_metric_header(fig, legend_handles, legend_labels, metrics_text):
    """Franja superior independiente: evita solapamiento con las curvas."""
    header = fig.add_axes([0.075, 0.865, 0.85, 0.105])
    header.set_axis_off()
    header.legend(
        legend_handles, legend_labels, frameon=False,
        loc="center left", bbox_to_anchor=(0.0, 0.5),
        ncol=1, borderaxespad=0.0, handlelength=2.2,
    )
    header.text(
        1.0, 0.5, metrics_text, transform=header.transAxes,
        ha="right", va="center", fontsize=8.0, linespacing=1.45,
        bbox=dict(boxstyle="round,pad=0.42", facecolor="white",
                  edgecolor="0.80", alpha=1.0),
    )

def mean_sd_by_time(df, time_col, columns):
    agg = {}
    for source, label in columns.items():
        agg[f"{label}_mean"] = (source, "mean")
        agg[f"{label}_sd"] = (source, "std")
    return df.groupby(time_col, sort=True).agg(**agg).reset_index()

def load_and_merge(xlsx: Path, aligned_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    aligned = pd.read_csv(aligned_path)
    require_columns(aligned, {
        "trial", "tau_s", "t_walo_original_s", "v_walo_m_s",
        "vx_tracker_interp_m_s", "position_walo_relative_m",
        "x_tracker_relative_m"
    }, aligned_path.name)
    raw = pd.read_excel(xlsx, sheet_name="WALO_TimeSeries")
    require_columns(raw, {
        "trial", "time_s", "f_mic1_hz", "f_mic2_hz", "f_common_hz"
    }, "WALO_TimeSeries")
    metadata = pd.read_excel(xlsx, sheet_name="Session_Metadata")
    require_columns(metadata, {"speed_of_sound_mps"}, "Session_Metadata")
    c_sound = float(metadata["speed_of_sound_mps"].iloc[0])
    # El CSV alineado ya conserva las frecuencias de ambos canales en la misma
    # fila temporal que el Tracker. Se usan directamente para evitar un segundo
    # merge que duplicaría columnas y podría alterar la correspondencia.
    require_columns(aligned, {"f_mic1_hz", "f_mic2_hz", "f_common_hz"}, aligned_path.name)
    missing = aligned[["f_mic1_hz", "f_mic2_hz"]].isna().any(axis=1).mean()
    if missing > 0.01:
        warnings.warn(f"{missing:.1%} de filas alineadas no tienen frecuencia individual.")
    return raw, aligned, c_sound

def load_movement_onsets(events_path: Path) -> pd.DataFrame:
    """Carga el inicio de movimiento detectado independientemente en el Tracker.

    tracker_onset_t_s está expresado en el reloj original de WALO, por lo que
    se compara directamente con t_walo_original_s. No se estima el inicio a
    partir de la señal Doppler (evita que el artefacto determine el recorte).
    """
    events = pd.read_csv(events_path)
    require_columns(events, {"trial", "tracker_onset_t_s"}, events_path.name)
    onsets = events[["trial", "tracker_onset_t_s"]].copy()
    onsets["trial"] = pd.to_numeric(onsets["trial"], errors="raise").astype(int)
    onsets["tracker_onset_t_s"] = pd.to_numeric(
        onsets["tracker_onset_t_s"], errors="coerce"
    )
    if onsets["tracker_onset_t_s"].isna().any():
        bad = onsets.loc[onsets["tracker_onset_t_s"].isna(), "trial"].tolist()
        raise ValueError(f"Inicios de movimiento ausentes para ensayos: {bad}")
    if onsets["trial"].duplicated().any():
        raise ValueError("Hay ensayos duplicados en el archivo de eventos.")
    return onsets

def apply_movement_onset_mask(df: pd.DataFrame, onsets: pd.DataFrame) -> pd.DataFrame:
    """Marca como no válidas todas las muestras anteriores al movimiento real."""
    out = df.merge(onsets, on="trial", how="left", validate="many_to_one")
    if out["tracker_onset_t_s"].isna().any():
        missing = sorted(out.loc[out["tracker_onset_t_s"].isna(), "trial"].unique())
        raise ValueError(f"No hay inicio de movimiento para los ensayos: {missing}")
    out["movement_valid"] = out["t_walo_original_s"] >= out["tracker_onset_t_s"]
    out["tau_movement_onset_s"] = out["tracker_onset_t_s"] - (
        out["t_walo_original_s"] - out["tau_s"]
    )
    return out

def estimate_individual_velocities(
    merged: pd.DataFrame, c_sound: float, baseline_start: float,
    baseline_end: float, median_kernel: int = 9,
    movement_onsets: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula v1/v2 con f0 independiente por canal y ensayo.

    MIC1: v1 = -c (f1-f01)/f01; MIC2: v2 = +c (f2-f02)/f02.
    Los signos opuestos corresponden a la orientación acústica de cada canal.
    """
    if baseline_end <= baseline_start:
        raise ValueError("El final del intervalo de reposo debe ser mayor que el inicio.")
    if median_kernel < 1 or median_kernel % 2 == 0:
        raise ValueError("median_kernel debe ser un entero impar positivo.")

    out = merged.copy()
    baseline_rows = []
    for trial, g in out.groupby("trial", sort=True):
        g = g.copy()
        f1_series = pd.to_numeric(g["f_mic1_hz"], errors="coerce").interpolate(limit_direction="both").to_numpy(float)
        f2_series = pd.to_numeric(g["f_mic2_hz"], errors="coerce").interpolate(limit_direction="both").to_numpy(float)
        if median_kernel > 1:
            f1_series = medfilt(f1_series, kernel_size=median_kernel)
            f2_series = medfilt(f2_series, kernel_size=median_kernel)
        g["_f1_filtered"] = f1_series
        g["_f2_filtered"] = f2_series
        stationary_mask = g["t_walo_original_s"].between(
            baseline_start, baseline_end, inclusive="both")
        # El intervalo de referencia nunca puede incluir muestras posteriores
        # al inicio físico, incluso si onset cae dentro del intervalo nominal.
        if movement_onsets is not None:
            onset_row = movement_onsets.loc[movement_onsets["trial"] == int(trial)]
            if onset_row.empty:
                raise ValueError(f"No hay onset del Tracker para ensayo {trial}.")
            onset_t = float(onset_row["tracker_onset_t_s"].iloc[0])
            stationary_mask &= g["t_walo_original_s"] < onset_t
        stationary = g[stationary_mask]
        if len(stationary) < 10:
            raise ValueError(
                f"Ensayo {trial}: solo {len(stationary)} muestras en el intervalo "
                f"de reposo [{baseline_start}, {baseline_end}] s."
            )
        refs = {}
        for col, label in [("_f1_filtered", "mic1"), ("_f2_filtered", "mic2")]:
            valid = stationary[["t_walo_original_s", col]].dropna()
            vals = valid[col].to_numpy(float)
            if len(vals) < 10:
                raise ValueError(f"Ensayo {trial}: frecuencia insuficiente para {label}.")
            refs[label] = float(np.median(vals))
            q25, q75 = np.percentile(vals, [25, 75])
            slope_hz_s = float(np.polyfit(valid["t_walo_original_s"], vals, 1)[0])
            baseline_rows.append({
                "trial": int(trial), "sensor": label,
                "baseline_start_s": baseline_start,
                "baseline_end_s": baseline_end,
                "n_baseline": int(len(vals)),
                "f0_reference_hz": refs[label],
                "baseline_iqr_hz": float(q75 - q25),
                "baseline_p05_hz": float(np.percentile(vals, 5)),
                "baseline_p95_hz": float(np.percentile(vals, 95)),
                "baseline_linear_drift_hz_per_s": slope_hz_s,
            })

        idx = g.index
        f1 = g["_f1_filtered"].to_numpy(float)
        f2 = g["_f2_filtered"].to_numpy(float)
        v1 = -c_sound * (f1 - refs["mic1"]) / refs["mic1"]
        v2 = +c_sound * (f2 - refs["mic2"]) / refs["mic2"]
        out.loc[idx, "v_mic1_corrected_m_s"] = v1
        out.loc[idx, "v_mic2_corrected_m_s"] = v2
        out.loc[idx, "f0_mic1_reference_hz"] = refs["mic1"]
        out.loc[idx, "f0_mic2_reference_hz"] = refs["mic2"]

    diagnostics = pd.DataFrame(baseline_rows)
    unstable = diagnostics[
        (diagnostics["baseline_iqr_hz"] > 3.0)
        | (diagnostics["baseline_linear_drift_hz_per_s"].abs() > 3.0)
    ]
    if not unstable.empty:
        warnings.warn(
            "La frecuencia no parece estacionaria en parte del intervalo de reposo. "
            "Revise exp03_sensor_baseline_diagnostics.csv antes de interpretar "
            "las métricas individuales. No aplique offsets verticales manuales."
        )
    return out, diagnostics

def plot_frequencies(raw, output, window, polyorder, tmin, tmax, baseline_diagnostics):
    raw = raw[(raw.time_s >= tmin) & (raw.time_s <= tmax)].copy()
    chunks = []
    for _, g in raw.groupby("trial", sort=True):
        g = g.sort_values("time_s").copy()
        for src, dst in [
            ("f_mic1_hz", "mic1"), ("f_mic2_hz", "mic2"),
            ("f_common_hz", "common")
        ]:
            g[dst] = smooth(g[src].to_numpy(float), window, polyorder)
        chunks.append(g)
    s = pd.concat(chunks, ignore_index=True)

    avg = mean_sd_by_time(s, "time_s", {
        "mic1": "mic1", "mic2": "mic2", "common": "common"
    })
    fig, axes = plt.subplots(
        2, 1, figsize=(10.5, 7.0), sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1.0], "hspace": 0.16}
    )
    t = avg.time_s.to_numpy()

    # Panel (a): solo mediciones individuales.
    ax = axes[0]
    for key, label, color in [
        ("mic1", "MIC1", "tab:blue"), ("mic2", "MIC2", "tab:orange")
    ]:
        m = avg[f"{key}_mean"].to_numpy()
        sd = avg[f"{key}_sd"].fillna(0).to_numpy()
        ax.plot(t, m, color=color, lw=2.0, label=f"{label}: media")
        ax.fill_between(t, m - sd, m + sd, color=color, alpha=0.14, linewidth=0)
    # Referencias de reposo: media entre ensayos de las medianas f0
    # estimadas individualmente para cada micrófono.
    f0_means = baseline_diagnostics.groupby("sensor")["f0_reference_hz"].mean()
    f0_mic1 = float(f0_means.loc["mic1"])
    f0_mic2 = float(f0_means.loc["mic2"])
    ax.axhline(f0_mic1, color="tab:blue", ls="--", lw=1.15, alpha=0.9,
               label=f"MIC1 reposo: {f0_mic1:.2f} Hz")
    ax.axhline(f0_mic2, color="tab:orange", ls="--", lw=1.15, alpha=0.9,
               label=f"MIC2 reposo: {f0_mic2:.2f} Hz")
    ax.set_ylabel("Frecuencia (Hz)")
    ax.set_title("(a) Mediciones independientes: MIC1 y MIC2", loc="left", pad=7)
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, ncol=2, loc="upper right")

    # Panel (b): componente común separada, con escala vertical propia.
    ax = axes[1]
    m = avg["common_mean"].to_numpy()
    sd = avg["common_sd"].fillna(0).to_numpy()
    ax.plot(t, m, color="tab:green", lw=2.15, label="Componente común: media")
    ax.fill_between(t, m - sd, m + sd, color="tab:green", alpha=0.16, linewidth=0)
    f0_common = 0.5 * (f0_mic1 + f0_mic2)
    ax.axhline(f0_common, color="tab:green", ls="--", lw=1.15, alpha=0.9,
               label=f"Reposo común: {f0_common:.2f} Hz")
    ax.set_ylabel("Frecuencia (Hz)")
    ax.set_xlabel("Tiempo (s)")
    ax.set_title("(b) Componente común", loc="left", pad=7)
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, loc="upper right")
    ax.set_xlim(tmin, tmax)

    fig.suptitle("Exp03 — Evolución temporal de las frecuencias Doppler", y=0.985)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.10)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Estadística descriptiva de frecuencias por canal, sobre datos originales.
    rows = []
    for col, label in [
        ("f_mic1_hz", "MIC1"), ("f_mic2_hz", "MIC2"),
        ("f_common_hz", "componente_comun")
    ]:
        vals = pd.to_numeric(raw[col], errors="coerce").dropna()
        rows.append({
            "signal": label, "n": int(vals.size),
            "mean_hz": float(vals.mean()) if vals.size else np.nan,
            "std_hz": float(vals.std()) if vals.size > 1 else np.nan,
            "min_hz": float(vals.min()) if vals.size else np.nan,
            "max_hz": float(vals.max()) if vals.size else np.nan,
        })
    return pd.DataFrame(rows)

def smooth_after_onset(values: np.ndarray, valid: np.ndarray,
                       window: int, polyorder: int) -> np.ndarray:
    """Suaviza solo desde el inicio de movimiento; nunca rellena el reposo."""
    y = np.asarray(values, dtype=float)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(y)
    out = np.full(y.shape, np.nan, dtype=float)
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return out
    # El tramo válido empieza en el onset; conserva huecos internos para que
    # smooth() los interpole solo dentro del tramo móvil.
    first = int(idx[0])
    segment = y[first:].copy()
    segment[~valid[first:]] = np.nan
    out[first:] = smooth(segment, window, polyorder)
    return out

def plot_velocity_comparison(df, output, metrics, estimator_col, estimator_label,
                             title, window, polyorder, tmin, tmax):
    raw = df[(df.tau_s >= tmin) & (df.tau_s <= tmax)].copy()
    raw = raw.sort_values(["trial", "tau_s"])
    chunks = []
    for _, g in raw.groupby("trial", sort=True):
        g = g.copy()
        valid = g["movement_valid"].to_numpy(dtype=bool)
        for col, dest in [
            (estimator_col, "walo_plot"),
            ("vx_tracker_interp_m_s", "tracker_plot")
        ]:
            y = g[col].to_numpy(dtype=float, copy=True)
            y[np.abs(y) > 3.0] = np.nan
            g[dest] = smooth_after_onset(y, valid, window, polyorder)
        chunks.append(g)
    s = pd.concat(chunks, ignore_index=True)
    avg = mean_sd_by_time(s, "tau_s", {
        "walo_plot": "walo", "tracker_plot": "tracker"
    })

    fig = plt.figure(figsize=(10.5, 6.0))
    ax = fig.add_axes([0.10, 0.13, 0.87, 0.70])

    for _, g in s.groupby("trial", sort=True):
        ax.plot(g.tau_s, g.walo_plot, color="tab:blue", lw=0.65, alpha=0.13)
        ax.plot(g.tau_s, g.tracker_plot, color="tab:orange", lw=0.65, alpha=0.13)

    t = avg.tau_s.to_numpy()
    for key, label, color in [
        ("walo", estimator_label, "tab:blue"),
        ("tracker", "Tracker manual", "tab:orange")
    ]:
        m = avg[f"{key}_mean"].to_numpy()
        sd = avg[f"{key}_sd"].fillna(0).to_numpy()
        ax.fill_between(t, m - sd, m + sd, color=color, alpha=0.14, linewidth=0)
        ax.plot(t, m, color=color, lw=2.35, label=f"{label}: media ± 1 DE")

    ax.axhline(0, color="0.35", lw=0.8, ls="--")
    ax.set(
        xlim=(tmin, tmax),
        xlabel="Tiempo relativo al evento de alineación (s)",
        ylabel="Velocidad (m/s)"
    )
    ax.grid(True, alpha=0.22)

    metric_text = (
        f"n = {metrics['n']:,} pares   |   r = {metrics['r']:.3f}   "
        f"MAE = {metrics['mae_m_s']:.3f} m/s   RMSE = {metrics['rmse_m_s']:.3f} m/s\n"
        f"Sesgo WALO−Tracker = {metrics['bias_m_s']:+.3f} m/s   "
        f"Razón RMS = {metrics['rms_ratio']:.2f}"
    )
    handles, labels = ax.get_legend_handles_labels()
    add_metric_header(fig, handles, labels, metric_text)
    fig.suptitle(title, fontsize=12, y=0.975)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_individual_sensor_comparison(df, output, metrics_by_sensor,
                                      window, polyorder, tmin, tmax):
    """Una figura de dos paneles: MIC1 y MIC2, cada uno frente a Tracker."""
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.0), sharex=True)
    specs = [
        ("v_mic1_corrected_m_s", "MIC1", metrics_by_sensor["mic1"]),
        ("v_mic2_corrected_m_s", "MIC2", metrics_by_sensor["mic2"]),
    ]
    for ax, (column, sensor, metrics) in zip(axes, specs):
        raw = df[(df.tau_s >= tmin) & (df.tau_s <= tmax)].sort_values(
            ["trial", "tau_s"]
        ).copy()
        chunks = []
        for _, g in raw.groupby("trial", sort=True):
            g = g.copy()
            valid = g["movement_valid"].to_numpy(bool)
            for col, dest in [
                (column, "walo_plot"),
                ("vx_tracker_interp_m_s", "tracker_plot"),
            ]:
                y = g[col].to_numpy(float, copy=True)
                y[np.abs(y) > 3.0] = np.nan
                g[dest] = smooth_after_onset(y, valid, window, polyorder)
            chunks.append(g)
        s = pd.concat(chunks, ignore_index=True)
        avg = mean_sd_by_time(s, "tau_s", {
            "walo_plot": "walo", "tracker_plot": "tracker"
        })
        for _, g in s.groupby("trial", sort=True):
            ax.plot(g.tau_s, g.walo_plot, color="tab:blue", lw=0.55, alpha=0.10)
            ax.plot(g.tau_s, g.tracker_plot, color="tab:orange", lw=0.55, alpha=0.10)
        t = avg.tau_s.to_numpy()
        for key, label, color in [
            ("walo", f"WALO {sensor}", "tab:blue"),
            ("tracker", "Tracker", "tab:orange"),
        ]:
            m = avg[f"{key}_mean"].to_numpy()
            sd = avg[f"{key}_sd"].fillna(0).to_numpy()
            ax.fill_between(t, m - sd, m + sd, color=color, alpha=0.14, linewidth=0)
            ax.plot(t, m, color=color, lw=2.1, label=f"{label}: media ± 1 DE")
        ax.axhline(0, color="0.35", lw=0.8, ls="--")
        ax.set_ylabel("Velocidad (m/s)")
        ax.set_title(f"({ 'a' if sensor == 'MIC1' else 'b' }) {sensor}", loc="left")
        ax.grid(True, alpha=0.22)
        ax.legend(frameon=False, loc="upper right")
        ax.text(
            0.015, 0.04,
            f"n={metrics['n']:,} | r={metrics['r']:.3f} | "
            f"MAE={metrics['mae_m_s']:.3f} m/s | "
            f"RMSE={metrics['rmse_m_s']:.3f} m/s | "
            f"sesgo={metrics['bias_m_s']:+.3f} m/s",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8.2,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="0.82", alpha=0.95)
        )
    axes[-1].set_xlim(tmin, tmax)
    axes[-1].set_xlabel("Tiempo relativo al evento de alineación (s)")
    fig.suptitle("Exp03 — Reconstrucción individual de velocidad por micrófono",
                 fontsize=12, y=0.99)
    fig.tight_layout(rect=[0.04, 0.03, 0.98, 0.96])
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_common_velocity_and_position(df, pos, output, velocity_metrics_row,
                                      position_metrics_row, window, position_window,
                                      polyorder, tmin, tmax):
    """Dos paneles con leyendas y métricas en bandas externas, sin tapar los datos."""
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(14.2, 7.0))
    gs = GridSpec(
        3, 2, figure=fig,
        height_ratios=[0.13, 1.0, 0.22],
        hspace=0.30, wspace=0.16,
        left=0.075, right=0.985, top=0.88, bottom=0.10
    )
    legend_axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    axes = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    metric_axes = [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]
    for lax, maxis in zip(legend_axes, metric_axes):
        lax.set_axis_off()
        maxis.set_axis_off()

    # Panel izquierdo: velocidad diferencial conjunta.
    ax = axes[0]
    raw = df[(df.tau_s >= tmin) & (df.tau_s <= tmax)].sort_values(
        ["trial", "tau_s"]
    ).copy()
    chunks = []
    for _, g in raw.groupby("trial", sort=True):
        g = g.copy()
        valid = g["movement_valid"].to_numpy(bool)
        for col, dest in [
            ("v_walo_m_s", "walo_plot"),
            ("vx_tracker_interp_m_s", "tracker_plot"),
        ]:
            y = g[col].to_numpy(float, copy=True)
            y[np.abs(y) > 3.0] = np.nan
            g[dest] = smooth_after_onset(y, valid, window, polyorder)
        chunks.append(g)
    s = pd.concat(chunks, ignore_index=True)
    avg = mean_sd_by_time(s, "tau_s", {
        "walo_plot": "walo", "tracker_plot": "tracker"
    })
    for _, g in s.groupby("trial", sort=True):
        ax.plot(g.tau_s, g.walo_plot, color="tab:blue", lw=0.5, alpha=0.07)
        ax.plot(g.tau_s, g.tracker_plot, color="tab:orange", lw=0.5, alpha=0.07)
    t = avg.tau_s.to_numpy()
    for key, label, color in [
        ("walo", "WALO: Doppler conjunto", "tab:blue"),
        ("tracker", "Tracker", "tab:orange"),
    ]:
        m = avg[f"{key}_mean"].to_numpy()
        sd = avg[f"{key}_sd"].fillna(0).to_numpy()
        ax.fill_between(t, m - sd, m + sd, color=color, alpha=0.14, linewidth=0)
        ax.plot(t, m, color=color, lw=2.2, label=f"{label}: media ± 1 DE")
    ax.axhline(0, color="0.35", lw=0.8, ls="--")
    ax.set(xlim=(tmin, tmax), xlabel="Tiempo relativo al evento (s)",
           ylabel="Velocidad (m/s)", title="(a) Velocidad reconstruida")
    ax.grid(True, alpha=0.22)

    # Panel derecho: desplazamiento integrado.
    ax = axes[1]
    p = pos[(pos.tau_s >= tmin) & (pos.tau_s <= tmax)].copy()
    chunks = []
    for _, g in p.groupby("trial", sort=True):
        g = g.sort_values("tau_s").copy()
        g["xw_plot"] = smooth(g.x_walo_rel.to_numpy(float), position_window, polyorder)
        g["xt_plot"] = smooth(g.x_tracker_rel.to_numpy(float), position_window, polyorder)
        chunks.append(g)
    ps = pd.concat(chunks, ignore_index=True)
    pavg = mean_sd_by_time(ps, "tau_s", {"xw_plot": "walo", "xt_plot": "tracker"})
    for _, g in ps.groupby("trial", sort=True):
        ax.plot(g.tau_s, g.xw_plot, color="tab:blue", lw=0.5, alpha=0.07)
        ax.plot(g.tau_s, g.xt_plot, color="tab:orange", lw=0.5, alpha=0.07)
    t = pavg.tau_s.to_numpy()
    for key, label, color in [
        ("walo", "WALO: trayectoria Doppler", "tab:blue"),
        ("tracker", "Tracker", "tab:orange"),
    ]:
        m = pavg[f"{key}_mean"].to_numpy()
        sd = pavg[f"{key}_sd"].fillna(0).to_numpy()
        ax.fill_between(t, m - sd, m + sd, color=color, alpha=0.14, linewidth=0)
        ax.plot(t, m, color=color, lw=2.2, label=f"{label}: media ± 1 DE")
    ax.set(xlim=(tmin, tmax), xlabel="Tiempo relativo al evento (s)",
           ylabel="Desplazamiento relativo (m)", title="(b) Trayectoria reconstruida")
    ax.grid(True, alpha=0.22)

    # Las leyendas ocupan una fila dedicada fuera de las zonas de datos.
    for lax, plot_ax in zip(legend_axes, axes):
        handles, labels = plot_ax.get_legend_handles_labels()
        lax.legend(handles, labels, loc="center", ncol=1, frameon=False,
                   fontsize=8.5, handlelength=2.5)

    # Métricas en tarjetas independientes bajo cada panel.
    metric_axes[0].text(
        0.5, 0.5,
        f"n = {velocity_metrics_row['n']:,}    |    r = {velocity_metrics_row['r']:.3f}\n"
        f"MAE = {velocity_metrics_row['mae_m_s']:.3f} m/s    |    "
        f"RMSE = {velocity_metrics_row['rmse_m_s']:.3f} m/s\n"
        f"Sesgo = {velocity_metrics_row['bias_m_s']:+.3f} m/s    |    "
        f"Razón RMS = {velocity_metrics_row['rms_ratio']:.2f}",
        transform=metric_axes[0].transAxes, ha="center", va="center", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                  edgecolor="0.78", alpha=1.0)
    )
    metric_axes[1].text(
        0.5, 0.5,
        f"n = {position_metrics_row['n']:,}    |    r = {position_metrics_row['r']:.3f}\n"
        f"MAE = {position_metrics_row['mae_m']:.3f} m    |    "
        f"RMSE = {position_metrics_row['rmse_m']:.3f} m\n"
        f"Sesgo = {position_metrics_row['bias_m']:+.3f} m    |    "
        f"Razón de picos = {position_metrics_row['mean_peak_ratio']:.2f}",
        transform=metric_axes[1].transAxes, ha="center", va="center", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                  edgecolor="0.78", alpha=1.0)
    )

    fig.suptitle("Exp03 — Estimación conjunta Doppler: velocidad y trayectoria",
                 fontsize=13, y=0.97)
    fig.savefig(output, dpi=300)
    plt.close(fig)

def position_metrics_and_data(aligned):
    chunks = []
    for trial, group in aligned.groupby("trial", sort=True):
        z = group.sort_values("tau_s").copy()
        z["x_walo_rel"] = z["position_walo_relative_m"] - z[
            "position_walo_relative_m"
        ].iloc[0]
        z["x_tracker_rel"] = z["x_tracker_relative_m"] - z[
            "x_tracker_relative_m"
        ].iloc[0]
        chunks.append(z)
    pos = pd.concat(chunks, ignore_index=True)

    mask = np.isfinite(pos.x_walo_rel) & np.isfinite(pos.x_tracker_rel)
    xw = pos.loc[mask, "x_walo_rel"].to_numpy(float)
    xt = pos.loc[mask, "x_tracker_rel"].to_numpy(float)
    d = xw - xt
    peaks = pos.groupby("trial").agg(
        walo_peak=("x_walo_rel", "max"),
        tracker_peak=("x_tracker_rel", "max")
    )
    mw, mt = float(peaks.walo_peak.mean()), float(peaks.tracker_peak.mean())
    metrics = {
        "n": int(mask.sum()), "r": pearson(xw, xt),
        "mae_m": float(np.mean(np.abs(d))),
        "rmse_m": float(np.sqrt(np.mean(d ** 2))),
        "bias_m": float(np.mean(d)),
        "mean_peak_walo_m": mw, "mean_peak_tracker_m": mt,
        "mean_peak_ratio": safe_ratio(mw, mt)
    }
    return pos, metrics

def plot_position(pos, output, metrics, window, polyorder, tmin, tmax):
    raw = pos[(pos.tau_s >= tmin) & (pos.tau_s <= tmax)].copy()
    chunks = []
    for _, g in raw.groupby("trial", sort=True):
        g = g.sort_values("tau_s").copy()
        g["xw_plot"] = smooth(g.x_walo_rel.to_numpy(float), position_window, polyorder)
        g["xt_plot"] = smooth(g.x_tracker_rel.to_numpy(float), position_window, polyorder)
        chunks.append(g)
    s = pd.concat(chunks, ignore_index=True)
    avg = mean_sd_by_time(s, "tau_s", {"xw_plot": "walo", "xt_plot": "tracker"})

    fig = plt.figure(figsize=(10.5, 6.0))
    ax = fig.add_axes([0.10, 0.13, 0.87, 0.70])
    for _, g in s.groupby("trial", sort=True):
        ax.plot(g.tau_s, g.xw_plot, color="tab:blue", lw=0.65, alpha=0.13)
        ax.plot(g.tau_s, g.xt_plot, color="tab:orange", lw=0.65, alpha=0.13)

    t = avg.tau_s.to_numpy()
    for key, label, color in [
        ("walo", "WALO Doppler", "tab:blue"),
        ("tracker", "Tracker manual", "tab:orange")
    ]:
        m = avg[f"{key}_mean"].to_numpy()
        sd = avg[f"{key}_sd"].fillna(0).to_numpy()
        ax.fill_between(t, m - sd, m + sd, color=color, alpha=0.14, linewidth=0)
        ax.plot(t, m, color=color, lw=2.35, label=f"{label}: media ± 1 DE")

    ax.set(
        xlim=(tmin, tmax),
        xlabel="Tiempo relativo al evento de alineación (s)",
        ylabel="Desplazamiento relativo (m)"
    )
    ax.grid(True, alpha=0.22)
    metric_text = (
        f"n = {metrics['n']:,} pares   |   r = {metrics['r']:.3f}   "
        f"MAE = {metrics['mae_m']:.3f} m   RMSE = {metrics['rmse_m']:.3f} m\n"
        f"Sesgo WALO−Tracker = {metrics['bias_m']:+.3f} m   "
        f"Razón de picos medios = {metrics['mean_peak_ratio']:.2f}"
    )
    handles, labels = ax.get_legend_handles_labels()
    add_metric_header(fig, handles, labels, metric_text)
    fig.suptitle("Exp03 — Comparación de desplazamiento reconstruido", fontsize=12, y=0.975)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Genera las tres figuras finales planificadas para Exp03."
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--frequency-window", type=int, default=51)
    parser.add_argument("--kinematic-window", type=int, default=31)
    parser.add_argument("--position-window", type=int, default=51)
    parser.add_argument("--polyorder", type=int, default=3)
    parser.add_argument("--frequency-tmin", type=float, default=0.5)
    parser.add_argument("--frequency-tmax", type=float, default=9.5)
    parser.add_argument("--comparison-tmin", type=float, default=-1.8)
    parser.add_argument("--comparison-tmax", type=float, default=7.5)
    parser.add_argument("--baseline-start", type=float, default=0.50)
    parser.add_argument("--baseline-end", type=float, default=0.80)
    parser.add_argument("--baseline-median-kernel", type=int, default=9)
    args = parser.parse_args()

    root = args.root.resolve()
    experiments_dir = root / "data" / "experiments"
    xlsx = experiments_dir / "exp03_walo_doppler_10trials_10s.xlsx"
    aligned_path = experiments_dir / "exp03_tracker_walo_aligned.csv"
    events_path = experiments_dir / "exp03_synchronization_events.csv"
    figures = root / "figures" / "experiments"
    out_dir = (root / "data" / "analysis" / "exp03_doppler"
              / "manuscript_figures")
    figures.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in (xlsx, aligned_path, events_path):
        if not p.is_file():
            raise FileNotFoundError(f"No se encontró el archivo requerido: {p}")

    raw, aligned, c_sound = load_and_merge(xlsx, aligned_path)
    movement_onsets = load_movement_onsets(events_path)
    aligned = apply_movement_onset_mask(aligned, movement_onsets)
    aligned, baseline_diagnostics = estimate_individual_velocities(
        aligned, c_sound, args.baseline_start, args.baseline_end,
        args.baseline_median_kernel, movement_onsets
    )

    outputs = {
        "frequency": figures / "exp03_fig1_frequencies_with_rest_reference.png",
        "individual": figures / "exp03_fig2_individual_mic_velocities.png",
        "joint": figures / "exp03_fig3_joint_velocity_and_position.png",
    }

    freq_stats = plot_frequencies(
        raw, outputs["frequency"], args.frequency_window, args.polyorder,
        args.frequency_tmin, args.frequency_tmax, baseline_diagnostics
    )

    # Comparación individual: referencias f0 independientes por sensor y ensayo.
    mic_metrics = {}
    for key, col in [
        ("mic1", "v_mic1_corrected_m_s"),
        ("mic2", "v_mic2_corrected_m_s"),
    ]:
        mic_metrics[key] = velocity_metrics(
            aligned[col].to_numpy(float),
            aligned["vx_tracker_interp_m_s"].to_numpy(float),
            aligned["movement_valid"].to_numpy(bool)
        )
    plot_individual_sensor_comparison(
        aligned, outputs["individual"], mic_metrics,
        args.kinematic_window, args.polyorder,
        args.comparison_tmin, args.comparison_tmax
    )

    # La estimación conjunta es la velocidad diferencial ya calculada por WALO:
    # v = c (f2-f1)/(f1+f2) = c (f2-f1)/(2 f_common).
    # No se obtiene convirtiendo f_common por sí sola en una velocidad.
    common_velocity_metrics = velocity_metrics(
        aligned["v_walo_m_s"].to_numpy(float),
        aligned["vx_tracker_interp_m_s"].to_numpy(float),
        aligned["movement_valid"].to_numpy(bool)
    )
    pos, pos_metrics = position_metrics_and_data(aligned)
    plot_common_velocity_and_position(
        aligned, pos, outputs["joint"], common_velocity_metrics, pos_metrics,
        args.kinematic_window, args.position_window, args.polyorder,
        args.comparison_tmin, args.comparison_tmax
    )

    metric_rows = []
    for sensor, metrics in mic_metrics.items():
        for metric, value in metrics.items():
            metric_rows.append({"comparison": sensor, "metric": metric, "value": value})
    for metric, value in common_velocity_metrics.items():
        metric_rows.append({"comparison": "joint_velocity", "metric": metric, "value": value})
    for metric, value in pos_metrics.items():
        metric_rows.append({"comparison": "position", "metric": metric, "value": value})

    pd.DataFrame(metric_rows).to_csv(out_dir / "exp03_final_metrics.csv", index=False)
    pd.DataFrame([
        {"figure": 1, "file": str(outputs["frequency"].relative_to(root)),
         "description": "Frecuencias MIC1/MIC2 y componente común con referencias medias de reposo"},
        {"figure": 2, "file": str(outputs["individual"].relative_to(root)),
         "description": "Velocidad reconstruida individualmente por MIC1 y MIC2 frente a Tracker"},
        {"figure": 3, "file": str(outputs["joint"].relative_to(root)),
         "description": "Velocidad Doppler conjunta y trayectoria reconstruida frente a Tracker"},
    ]).to_csv(out_dir / "exp03_final_figures_manifest.csv", index=False)
    freq_stats.to_csv(out_dir / "exp03_frequency_statistics.csv", index=False)
    baseline_diagnostics.to_csv(out_dir / "exp03_sensor_baseline_diagnostics.csv", index=False)
    movement_onsets.to_csv(out_dir / "exp03_movement_onsets_used.csv", index=False)

    print("\\nFIGURAS GENERADAS")
    for path in outputs.values():
        print(f"  {path.relative_to(root)}")
    print("\\nMÉTRICAS (datos sin suavizar)")
    for name, metrics in [("MIC1", mic_metrics["mic1"]),
                          ("MIC2", mic_metrics["mic2"]),
                          ("Doppler conjunto", common_velocity_metrics)]:
        print(f"{name}: n={metrics['n']}, r={metrics['r']:.3f}, "
              f"MAE={metrics['mae_m_s']:.3f} m/s, "
              f"RMSE={metrics['rmse_m_s']:.3f} m/s, "
              f"sesgo={metrics['bias_m_s']:+.3f} m/s, "
              f"razón RMS={metrics['rms_ratio']:.2f}")
    print(f"Posición: n={pos_metrics['n']}, r={pos_metrics['r']:.3f}, "
          f"MAE={pos_metrics['mae_m']:.3f} m, RMSE={pos_metrics['rmse_m']:.3f} m")
    print("\\nReferencias medias de reposo (Hz):")
    print(baseline_diagnostics.groupby("sensor")["f0_reference_hz"].agg(["mean", "std"]).round(3))
    print("\\nRevise especialmente el diagnóstico de estabilidad del intervalo de reposo.")

if __name__ == "__main__":
    main()