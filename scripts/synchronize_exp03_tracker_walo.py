#!/usr/bin/env python3
"""
Exp03 — prepara Tracker y WALO para comparación por primer punto de retorno.

Genera:
  data/experiments/exp03_synchronization_events.csv
  data/experiments/exp03_tracker_walo_aligned.csv
  data/experiments/exp03_tracker_walo_comparison.csv
  figures/experiments/exp03_tracker_walo_alignment_check.png

Y archiva (NO elimina) los productos intermedios de los ensayos anteriores en:
  ensayos/exp03_tracker_preprocessing/

Conserva intactos los datos originales:
  data/experiments/exp03_tracker/*.csv
  data/experiments/exp03_raw_dual_channel.npz
  data/experiments/exp03_walo_doppler_10trials_10s.csv (si existe)
  data/experiments/exp03.mp4

Uso desde la raíz del repositorio:
  python scripts/synchronize_exp03_tracker_walo.py

Opciones útiles:
  python scripts/synchronize_exp03_tracker_walo.py --no-archive
  python scripts/synchronize_exp03_tracker_walo.py --min-walo-event-time 1.0
  python scripts/synchronize_exp03_tracker_walo.py --min-walo-event-time 1.3

Dependencias: numpy, pandas, matplotlib, scipy, openpyxl (solo si se lee XLSX).

NOTAS METODOLÓGICAS:
- Tracker: el primer máximo se detecta en x(t), con prominencia mínima configurable.
- WALO: se busca el primer cambio de signo sostenido de velocity_diff después del
  tiempo mínimo configurado. Para detectar el evento se aplica mediana móvil; los
  datos exportados originales NO se modifican.
- La diferencia entre tiempos de evento de dos archivos con relojes independientes
  NO es, por sí sola, una latencia instrumental. Se informa como diferencia entre
  referencias temporales de archivo.
- La comparación de velocidades conserva el signo exportado por WALO. No se invierte
  automáticamente: verificar primero la convención de signo y la geometría.
- Tracker se interpola linealmente en la malla temporal WALO solo cuando los dos
  puntos que delimitan la interpolación están separados por <= MAX_TRACKER_GAP_S.
  No se extrapola ni se rellenan huecos grandes.
- No se eliminan outliers de los datos fuente.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS_DIR = ROOT / "data" / "experiments"
TRACKER_DIR = EXPERIMENTS_DIR / "exp03_tracker"
WALO_CSV_CANDIDATES = [
    EXPERIMENTS_DIR / "exp03_walo_doppler_10trials_10s.csv",
]
WALO_XLSX_CANDIDATES = [
    EXPERIMENTS_DIR / "exp03_walo_doppler_10trials_10s.xlsx",
]

EVENTS_CSV = EXPERIMENTS_DIR / "exp03_synchronization_events.csv"
ALIGNED_CSV = EXPERIMENTS_DIR / "exp03_tracker_walo_aligned.csv"
COMPARISON_CSV = EXPERIMENTS_DIR / "exp03_tracker_walo_comparison.csv"
FIGURES_DIR = ROOT / "figures" / "experiments"
CHECK_FIGURE = FIGURES_DIR / "exp03_tracker_walo_alignment_check.png"

ARCHIVE_ROOT = (ROOT / "data" / "analysis" / "exp03_doppler"
                / "preprocessing_archive")

# Tracker: criterio coherente con el procesamiento preliminar ya utilizado.
TRACKER_PEAK_PROMINENCE_M = 0.05
TRACKER_ONSET_THRESHOLD_M = 0.005
TRACKER_ONSET_SUSTAINED_SAMPLES = 3
TRACKER_BASELINE_S = 0.50

# WALO: parámetros explícitos y modificables para detectar una inversión robusta.
WALO_MIN_EVENT_TIME_S = 1.00
WALO_MEDIAN_WINDOW_SAMPLES = 7
WALO_STABLE_SIGN_SAMPLES = 5
WALO_MIN_FLANK_SPEED_M_S = 0.10

# Comparación: no interpolar a través de zonas de oclusión grandes.
MAX_TRACKER_GAP_S = 0.10


def parse_number(value):
    """Convierte números exportados con coma o punto decimal."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    s = str(value).strip().replace("\u2212", "-").replace(",", ".")
    if not s:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_tracker_mixed_unit(value):
    """
    Tracker exportó enteros en mm/ms y valores con punto decimal en m/s.
    Se conserva el criterio empleado en el preprocesamiento preliminar:
    enteros sin punto decimal y |valor| >= 10 se convierten dividiendo por 1000.
    """
    number = parse_number(value)
    if not np.isfinite(number):
        return np.nan
    raw = str(value).strip()
    if "." not in raw and abs(number) >= 10:
        return number / 1000.0
    return number


def get_trial_number(path: Path) -> int:
    m = re.search(r"try\s*(\d+)", path.stem, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"No se reconoce el número de ensayo: {path.name}")
    return int(m.group(1))


def find_col(df, candidates):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    raise ValueError(f"No encuentro columnas {candidates}. Disponibles: {list(df.columns)}")


def read_tracker_trials():
    files = sorted(TRACKER_DIR.glob("Try*.csv"), key=get_trial_number)
    if len(files) != 10:
        raise RuntimeError(
            f"Se esperaban 10 archivos Try*.csv en {TRACKER_DIR}; encontrados: {len(files)}"
        )

    all_data = []
    event_rows = []

    for path in files:
        trial = get_trial_number(path)
        raw = pd.read_csv(path, header=1, dtype=str, keep_default_na=False)
        raw.columns = [str(c).strip() for c in raw.columns]

        tc = find_col(raw, ["t", "time"])
        xc = find_col(raw, ["x", "position"])
        vxc = find_col(raw, ["vx", "v_x"])
        fc = find_col(raw, ["frame", "step"])

        d = pd.DataFrame({
            "trial": trial,
            "source_file": path.name,
            "source_row": np.arange(3, len(raw) + 3),
            "t_tracker_s": raw[tc].map(parse_tracker_mixed_unit),
            "x_tracker_m": raw[xc].map(parse_tracker_mixed_unit),
            "vx_tracker_m_s": raw[vxc].map(parse_tracker_mixed_unit),
            "tracker_frame": raw[fc].map(parse_number),
        })
        d = d[np.isfinite(d["t_tracker_s"])].copy()
        d = d[d["t_tracker_s"].between(0.0, 10.0, inclusive="both")].copy()
        d = d.sort_values(["t_tracker_s", "source_row"], kind="stable").reset_index(drop=True)

        t = d["t_tracker_s"].to_numpy(float)
        x = d["x_tracker_m"].to_numpy(float)
        valid = np.isfinite(t) & np.isfinite(x)
        if valid.sum() < 5:
            raise RuntimeError(f"Try {trial}: insuficientes posiciones válidas.")

        # Detectar inicio de movimiento para excluir máximos espurios de la zona en reposo.
        t_first = float(np.nanmin(t[valid]))
        baseline_mask = valid & (t >= t_first) & (t <= t_first + TRACKER_BASELINE_S)
        if baseline_mask.sum() < 3:
            raise RuntimeError(f"Try {trial}: no hay suficientes muestras para estimar reposo.")
        x_rest = float(np.nanmedian(x[baseline_mask]))
        moved = valid & (np.abs(x - x_rest) >= TRACKER_ONSET_THRESHOLD_M)

        onset_idx = None
        for k in range(0, len(d) - TRACKER_ONSET_SUSTAINED_SAMPLES + 1):
            if moved[k:k + TRACKER_ONSET_SUSTAINED_SAMPLES].all():
                onset_idx = k
                break
        if onset_idx is None:
            raise RuntimeError(f"Try {trial}: no se pudo detectar el inicio del movimiento.")

        # Primer máximo local posterior al inicio. Se usa prominencia de 5 cm.
        post_idx = np.flatnonzero(valid & (np.arange(len(d)) >= onset_idx))
        x_post = x[post_idx]
        peaks, props = find_peaks(x_post, prominence=TRACKER_PEAK_PROMINENCE_M)
        if len(peaks) == 0:
            raise RuntimeError(f"Try {trial}: no se detectó primer máximo con prominencia "
                               f"{TRACKER_PEAK_PROMINENCE_M:.3f} m.")
        peak_idx = int(post_idx[int(peaks[0])])
        t_peak = float(d.loc[peak_idx, "t_tracker_s"])
        x_peak = float(d.loc[peak_idx, "x_tracker_m"])
        onset_t = float(d.loc[onset_idx, "t_tracker_s"])

        d["tau_tracker_s"] = d["t_tracker_s"] - t_peak
        d["x_tracker_relative_m"] = d["x_tracker_m"] - x_peak
        d["tracker_first_peak_t_s"] = t_peak
        d["tracker_first_peak_x_m"] = x_peak
        all_data.append(d)

        event_rows.append({
            "trial": trial,
            "tracker_onset_t_s": onset_t,
            "tracker_first_peak_t_s": t_peak,
            "tracker_first_peak_x_m": x_peak,
            "tracker_first_peak_frame": d.loc[peak_idx, "tracker_frame"],
            "tracker_peak_prominence_m": float(props["prominences"][0]),
            "tracker_samples_0_10s": len(d),
            "tracker_valid_position_samples": int(valid.sum()),
        })

    return pd.concat(all_data, ignore_index=True), pd.DataFrame(event_rows)


def read_walo():
    # Primero buscar el CSV consolidado en las ubicaciones habituales.
    csv_path = next((p for p in WALO_CSV_CANDIDATES if p.is_file()), None)

    # XLSX original almacenado junto a los demás datos experimentales.
    # Acepta sufijos automáticos como "(1)".
    xlsx_candidates = sorted(
        EXPERIMENTS_DIR.glob("exp03_walo_doppler_10trials_10s*.xlsx")
    ) if EXPERIMENTS_DIR.is_dir() else []
    xlsx_path = next((p for p in xlsx_candidates if p.is_file()), None)

    if csv_path is not None:
        # CSV consolidado esperado: una fila por instante y ensayo.
        try:
            w = pd.read_csv(csv_path)
        except Exception:
            w = pd.read_csv(csv_path, sep=None, engine="python")
        source = str(csv_path.relative_to(ROOT))
    elif xlsx_path is not None:
        w = pd.read_excel(xlsx_path, sheet_name="WALO_TimeSeries")
        source = str(xlsx_path.relative_to(ROOT)) + " [WALO_TimeSeries]"
    else:
        raise FileNotFoundError(
            "No encuentro los datos procesados de WALO. Busqué:\\n"
            "  data/experiments/exp03_walo_doppler_10trials_10s.csv\\n"
            "  exp03_walo_doppler_10trials_10s.csv\\n"
            "  data/exp03_walo_doppler_10trials_10s*.xlsx\\n"
            "  exp03_walo_doppler_10trials_10s*.xlsx\\n"
            "Copia el XLSX original (con la hoja WALO_TimeSeries) o el CSV "
            "consolidado a data/. El NPZ contiene audio crudo y no sustituye "
            "esta tabla procesada."
        )

    w.columns = [str(c).strip() for c in w.columns]
    trial_c = find_col(w, ["trial", "try", "ensayo"])
    time_c = find_col(w, ["time_s", "t_s", "time"])
    vel_c = find_col(w, ["velocity_diff_cm_s", "v_diff_cm_s", "velocity_diff_cmps",
                         "v_diff_cmps", "v_walo_cm_s"])
    f1_c = next((c for c in w.columns if c.lower() in ["f_mic1_hz", "f1_hz"]), None)
    f2_c = next((c for c in w.columns if c.lower() in ["f_mic2_hz", "f2_hz"]), None)
    fc_c = next((c for c in w.columns if c.lower() in ["f_common_hz", "fc_hz"]), None)
    disp_c = next((c for c in w.columns if c.lower() in ["displacement_cm", "displacement"]), None)
    pos_c = next((c for c in w.columns if c.lower() in ["position_cm", "position"]), None)

    out = pd.DataFrame({
        "trial": pd.to_numeric(w[trial_c], errors="coerce"),
        "t_walo_s": pd.to_numeric(w[time_c], errors="coerce"),
        "v_walo_m_s": pd.to_numeric(w[vel_c], errors="coerce") / 100.0,
    })
    out["f_mic1_hz"] = pd.to_numeric(w[f1_c], errors="coerce") if f1_c else np.nan
    out["f_mic2_hz"] = pd.to_numeric(w[f2_c], errors="coerce") if f2_c else np.nan
    out["f_common_hz"] = pd.to_numeric(w[fc_c], errors="coerce") if fc_c else np.nan
    out["displacement_walo_m"] = (
        pd.to_numeric(w[disp_c], errors="coerce") / 100.0 if disp_c else np.nan
    )
    out["position_walo_m"] = (
        pd.to_numeric(w[pos_c], errors="coerce") / 100.0 if pos_c else np.nan
    )
    out = out.dropna(subset=["trial", "t_walo_s"]).copy()
    out["trial"] = out["trial"].astype(int)
    out = out.sort_values(["trial", "t_walo_s"]).reset_index(drop=True)
    out.attrs["source"] = source
    return out


def rolling_median(values, window):
    return pd.Series(values).rolling(window=window, center=True, min_periods=1).median().to_numpy()


def detect_walo_turn(g, min_time_s):
    """Primer cambio de signo sostenido, tras mediana móvil; no altera los datos exportados."""
    g = g.sort_values("t_walo_s").reset_index(drop=True)
    t = g["t_walo_s"].to_numpy(float)
    v = g["v_walo_m_s"].to_numpy(float)
    valid = np.isfinite(t) & np.isfinite(v) & (t >= min_time_s)
    if valid.sum() < 2 * WALO_STABLE_SIGN_SAMPLES + 1:
        return None

    smooth = rolling_median(v, WALO_MEDIAN_WINDOW_SAMPLES)
    n = WALO_STABLE_SIGN_SAMPLES
    candidates = []
    for j in range(n, len(g) - n):
        if not valid[j]:
            continue
        left = smooth[j-n:j]
        right = smooth[j:j+n]
        if not (np.isfinite(left).all() and np.isfinite(right).all()):
            continue
        left_sign = np.sign(np.median(left))
        right_sign = np.sign(np.median(right))
        if left_sign == 0 or right_sign == 0 or left_sign == right_sign:
            continue
        if np.median(np.abs(left)) < WALO_MIN_FLANK_SPEED_M_S:
            continue
        if np.median(np.abs(right)) < WALO_MIN_FLANK_SPEED_M_S:
            continue
        # Candidato en el punto de menor |v| dentro de la transición local.
        lo, hi = max(0, j-2), min(len(g), j+3)
        local = np.arange(lo, hi)
        local = local[valid[local]]
        if len(local) == 0:
            continue
        k = int(local[np.nanargmin(np.abs(smooth[local]))])
        candidates.append((k, left_sign, right_sign,
                           float(np.median(np.abs(left))),
                           float(np.median(np.abs(right)))))
        break

    if not candidates:
        return None

    k, s_before, s_after, v_before, v_after = candidates[0]
    return {
        "index": k,
        "time_s": float(t[k]),
        "velocity_smoothed_m_s": float(smooth[k]),
        "sign_before": int(s_before),
        "sign_after": int(s_after),
        "median_abs_velocity_before_m_s": v_before,
        "median_abs_velocity_after_m_s": v_after,
    }


def interpolate_tracker_at(t_query, t_source, y_source, max_gap):
    """Interpolación lineal solo dentro de intervalos observados y suficientemente densos."""
    result = np.full(len(t_query), np.nan, dtype=float)
    good = np.isfinite(t_source) & np.isfinite(y_source)
    ts, ys = t_source[good], y_source[good]
    if len(ts) < 2:
        return result
    order = np.argsort(ts)
    ts, ys = ts[order], ys[order]
    # Eliminar tiempos duplicados para np.interp, conservando la primera observación.
    unique_t, unique_idx = np.unique(ts, return_index=True)
    ts, ys = unique_t, ys[unique_idx]
    if len(ts) < 2:
        return result

    inside = (t_query >= ts[0]) & (t_query <= ts[-1])
    qidx = np.flatnonzero(inside)
    if len(qidx) == 0:
        return result
    q = t_query[qidx]
    right = np.searchsorted(ts, q, side="left")
    right = np.clip(right, 1, len(ts)-1)
    left = right - 1
    gaps = ts[right] - ts[left]
    acceptable = gaps <= max_gap
    vals = np.full(len(q), np.nan)
    vals[acceptable] = (
        ys[left[acceptable]]
        + (ys[right[acceptable]] - ys[left[acceptable]])
        * (q[acceptable] - ts[left[acceptable]]) / gaps[acceptable]
    )
    # Si coincide exactamente con una muestra, conservarla incluso en bordes.
    exact = np.searchsorted(ts, q, side="left")
    exact_ok = (exact < len(ts)) & np.isclose(ts[np.minimum(exact, len(ts)-1)], q, atol=1e-9)
    vals[exact_ok] = ys[exact[exact_ok]]
    result[qidx] = vals
    return result


def calculate_metrics(g):
    d = g.dropna(subset=["vx_tracker_interp_m_s", "v_walo_m_s"]).copy()
    if len(d) < 3:
        return {
            "n_comparison_samples": len(d), "velocity_bias_walo_minus_tracker_m_s": np.nan,
            "velocity_mae_m_s": np.nan, "velocity_rmse_m_s": np.nan,
            "velocity_correlation": np.nan,
        }
    err = d["v_walo_m_s"].to_numpy() - d["vx_tracker_interp_m_s"].to_numpy()
    corr = np.nan
    if d["v_walo_m_s"].std() > 0 and d["vx_tracker_interp_m_s"].std() > 0:
        corr = d["v_walo_m_s"].corr(d["vx_tracker_interp_m_s"])
    return {
        "n_comparison_samples": len(d),
        "velocity_bias_walo_minus_tracker_m_s": float(np.mean(err)),
        "velocity_mae_m_s": float(np.mean(np.abs(err))),
        "velocity_rmse_m_s": float(np.sqrt(np.mean(err**2))),
        "velocity_correlation": corr,
    }


def archive_intermediate_files():
    """Mueve productos de prueba a ensayos/ sin tocar fuentes originales."""
    candidates = [
        "data/exp03_tracker_processed.csv",
        "data/exp03_tracker_aligned_onset.csv",
        "data/exp03_tracker_onset_summary.csv",
        "data/exp03_tracker_aligned_first_maximum.csv",
        "data/exp03_tracker_first_maximum_summary.csv",
        "data/exp03_tracker_first_maximum_mean_std.csv",
        "figures/experiments/exp03_tracker_overview_initial.png",
        "figures/experiments/exp03_tracker_aligned_onset.png",
        "figures/experiments/exp03_tracker_aligned_first_maximum.png",
        "figures/experiments/exp03_tracker_first_maximum_mean_std.png",
        "scripts/preprocess_exp03_tracker.py",
        "scripts/align_exp03_tracker_onset.py",
        "scripts/align_exp03_tracker_first_maximum.py",
        "scripts/plot_exp03_tracker_mean_std.py",
        "fotograma_referencia.png",
    ]
    moved = []
    for rel in candidates:
        src = ROOT / rel
        if not src.exists():
            continue
        dest = ARCHIVE_ROOT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            stem, suffix = dest.stem, dest.suffix
            n = 2
            while dest.with_name(f"{stem}_{n}{suffix}").exists():
                n += 1
            dest = dest.with_name(f"{stem}_{n}{suffix}")
        shutil.move(str(src), str(dest))
        moved.append((rel, str(dest.relative_to(ROOT))))
    return moved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-walo-event-time", type=float, default=WALO_MIN_EVENT_TIME_S,
                        help="No buscar inversiones Doppler antes de este tiempo (s).")
    parser.add_argument("--no-archive", action="store_true",
                        help="No mover los archivos intermedios a ensayos/.")
    parser.add_argument("--max-tracker-gap", type=float, default=MAX_TRACKER_GAP_S,
                        help="Separación máxima entre muestras Tracker para interpolar (s).")
    args = parser.parse_args()

    if not TRACKER_DIR.is_dir():
        raise FileNotFoundError(f"No se encontró el directorio de Tracker: {TRACKER_DIR}")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    tracker, events = read_tracker_trials()
    walo = read_walo()
    event_rows = []

    walo_turns = {}
    for trial, g in walo.groupby("trial", sort=True):
        turn = detect_walo_turn(g, args.min_walo_event_time)
        walo_turns[int(trial)] = turn

    for _, row in events.iterrows():
        trial = int(row["trial"])
        turn = walo_turns.get(trial)
        if turn is None:
            event_rows.append({
                **row.to_dict(),
                "walo_turn_t_s": np.nan,
                "walo_turn_velocity_smoothed_m_s": np.nan,
                "walo_sign_before": np.nan,
                "walo_sign_after": np.nan,
                "event_time_difference_walo_minus_tracker_s": np.nan,
                "walo_event_status": "NO_DETECTED_REVIEW_REQUIRED",
            })
        else:
            event_rows.append({
                **row.to_dict(),
                "walo_turn_t_s": turn["time_s"],
                "walo_turn_velocity_smoothed_m_s": turn["velocity_smoothed_m_s"],
                "walo_sign_before": turn["sign_before"],
                "walo_sign_after": turn["sign_after"],
                "walo_flank_speed_before_m_s": turn["median_abs_velocity_before_m_s"],
                "walo_flank_speed_after_m_s": turn["median_abs_velocity_after_m_s"],
                "event_time_difference_walo_minus_tracker_s":
                    turn["time_s"] - float(row["tracker_first_peak_t_s"]),
                "walo_event_status": "CANDIDATE_REVIEW_REQUIRED",
            })

    events_out = pd.DataFrame(event_rows).sort_values("trial")
    aligned_frames = []
    comparison_rows = []

    for trial in sorted(set(tracker["trial"]) & set(walo["trial"])):
        tg = tracker[tracker["trial"] == trial].copy()
        wg = walo[walo["trial"] == trial].copy()
        er = events_out[events_out["trial"] == trial].iloc[0]

        if not np.isfinite(er["tracker_first_peak_t_s"]) or not np.isfinite(er["walo_turn_t_s"]):
            continue

        # Cada fuente se alinea independientemente en su evento detectado.
        tg["tau_s"] = tg["t_tracker_s"] - float(er["tracker_first_peak_t_s"])
        wg["tau_s"] = wg["t_walo_s"] - float(er["walo_turn_t_s"])

        # WALO aporta la malla temporal de 100 Hz; Tracker se interpola solo
        # donde hay soporte observado suficientemente próximo.
        tq = wg["tau_s"].to_numpy(float)
        t_tracker = tg["tau_s"].to_numpy(float)
        aligned = pd.DataFrame({
            "trial": trial,
            "tau_s": tq,
            "t_tracker_original_s": tq + float(er["tracker_first_peak_t_s"]),
            "t_walo_original_s": wg["t_walo_s"].to_numpy(float),
            "x_tracker_relative_m": interpolate_tracker_at(
                tq, t_tracker, tg["x_tracker_relative_m"].to_numpy(float), args.max_tracker_gap
            ),
            "vx_tracker_interp_m_s": interpolate_tracker_at(
                tq, t_tracker, tg["vx_tracker_m_s"].to_numpy(float), args.max_tracker_gap
            ),
            "v_walo_m_s": wg["v_walo_m_s"].to_numpy(float),
            "f_mic1_hz": wg["f_mic1_hz"].to_numpy(float),
            "f_mic2_hz": wg["f_mic2_hz"].to_numpy(float),
            "f_common_hz": wg["f_common_hz"].to_numpy(float),
            "displacement_walo_relative_m":
                wg["displacement_walo_m"].to_numpy(float)
                - float(wg["displacement_walo_m"].iloc[int(np.argmin(np.abs(wg["tau_s"].to_numpy())))])
                if wg["displacement_walo_m"].notna().any() else np.nan,
            "position_walo_relative_m":
                wg["position_walo_m"].to_numpy(float)
                - float(wg["position_walo_m"].iloc[int(np.argmin(np.abs(wg["tau_s"].to_numpy())))])
                if wg["position_walo_m"].notna().any() else np.nan,
        })
        aligned["velocity_error_walo_minus_tracker_m_s"] = (
            aligned["v_walo_m_s"] - aligned["vx_tracker_interp_m_s"]
        )
        aligned_frames.append(aligned)

        metrics = calculate_metrics(aligned)
        comparison_rows.append({
            "trial": trial,
            "tracker_first_peak_t_s": float(er["tracker_first_peak_t_s"]),
            "walo_turn_t_s": float(er["walo_turn_t_s"]),
            "event_time_difference_walo_minus_tracker_s":
                float(er["event_time_difference_walo_minus_tracker_s"]),
            "comparison_tau_min_s": float(np.nanmin(tq)),
            "comparison_tau_max_s": float(np.nanmax(tq)),
            **metrics,
        })

    if len(aligned_frames) == 0:
        raise RuntimeError("No se pudo construir ninguna comparación. Revisa los eventos WALO.")

    aligned_out = pd.concat(aligned_frames, ignore_index=True)
    comparison = pd.DataFrame(comparison_rows).sort_values("trial")

    # Resumen pooled descriptivo, calculado solo con pares disponibles.
    pooled = calculate_metrics(aligned_out)
    pooled_row = {"trial": "ALL_POOLED", **pooled}
    comparison = pd.concat([comparison, pd.DataFrame([pooled_row])], ignore_index=True)

    EVENTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    CHECK_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    events_out.to_csv(EVENTS_CSV, index=False, float_format="%.9g")
    aligned_out.to_csv(ALIGNED_CSV, index=False, float_format="%.9g")
    comparison.to_csv(COMPARISON_CSV, index=False, float_format="%.9g")

    # Gráfica de control: no es una figura final de tesis; sirve para validar
    # visualmente la detección de eventos y el sentido de las velocidades.
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True, constrained_layout=True)
    for trial, g in aligned_out.groupby("trial", sort=True):
        g = g.sort_values("tau_s")
        axes[0].plot(g["tau_s"], g["x_tracker_relative_m"], lw=0.9, alpha=0.8,
                     label=f"Try {trial} Tracker")
        axes[1].plot(g["tau_s"], g["vx_tracker_interp_m_s"], lw=0.8, alpha=0.7)
        axes[1].plot(g["tau_s"], g["v_walo_m_s"], lw=0.8, alpha=0.7, linestyle="--")
    axes[0].axvline(0, color="black", linestyle=":", lw=1)
    axes[0].set_ylabel("x Tracker relativa (m)")
    axes[0].set_title("Control de alineación por primer punto de retorno")
    axes[0].grid(True, alpha=0.25)
    axes[1].axvline(0, color="black", linestyle=":", lw=1, label="Evento")
    axes[1].set_xlabel("Tiempo relativo al evento de cada sistema (s)")
    axes[1].set_ylabel("Velocidad (m/s)")
    axes[1].set_title("Tracker (línea continua) y WALO (línea discontinua)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(["Tracker", "WALO", "Evento"], fontsize=8, ncol=3)
    fig.savefig(CHECK_FIGURE, dpi=190)
    plt.close(fig)

    moved = []
    if not args.no_archive:
        moved = archive_intermediate_files()

    print("\n=== EVENTOS DE SINCRONIZACIÓN ===")
    print(events_out.to_string(index=False))
    print("\n=== COMPARACIÓN DE VELOCIDAD ===")
    print(comparison.to_string(index=False))
    print("\nArchivos generados:")
    print(f"  {EVENTS_CSV.relative_to(ROOT)}")
    print(f"  {ALIGNED_CSV.relative_to(ROOT)}")
    print(f"  {COMPARISON_CSV.relative_to(ROOT)}")
    print(f"  {CHECK_FIGURE.relative_to(ROOT)}")
    print(f"\nFuente WALO: {walo.attrs.get('source', 'desconocida')}")
    print(f"Archivos intermedios archivados: {len(moved)}")
    if moved:
        print(f"Archivo de ensayos: {ARCHIVE_ROOT.relative_to(ROOT)}")
        for src, dst in moved:
            print(f"  {src} -> {dst}")
    print("\nIMPORTANTE: todos los eventos WALO quedan marcados como candidatos para "
          "revisión visual. event_time_difference_walo_minus_tracker_s no es una "
          "latencia instrumental si los relojes de ambos sistemas no comparten origen.")
    print("Verifica también la convención de signo antes de interpretar MAE/RMSE.")


if __name__ == "__main__":
    main()
