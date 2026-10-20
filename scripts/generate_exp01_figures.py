#!/usr/bin/env python3
"""
Genera las cuatro figuras del Experimento 01 (caracterización radial).

Estructura esperada:
thesis-walo/
├── data/
│   ├── distance_characterization_raw_mic1.csv
│   └── distance_characterization_raw_mic2.csv
├── figures/
└── scripts/
    └── generate_exp01_figures.py

Uso:
    python scripts/generate_exp01_figures.py

La Figura 3 se genera con datos SINTÉTICOS de demostración. No son mediciones
y debe reemplazarse cuando se repita la estimación experimental de distancia.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"

CSV_FILES = {
    "Micrófono 1": DATA_DIR / "distance_characterization_raw_mic1.csv",
    "Micrófono 2": DATA_DIR / "distance_characterization_raw_mic2.csv",
}

# Estética sobria para el manuscrito: fondo claro y acento institucional.
COLORS = {
    "Micrófono 1": "#007F5F",
    "Micrófono 2": "#D97736",
    "reference": "#555555",
    "grid": "#D9DEE3",
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
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def load_data():
    data = {}
    for name, path in CSV_FILES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"No se encontró {path}\n"
                "Comprueba que los CSV estén en la carpeta data/ y tengan "
                "los nombres indicados en el encabezado del script."
            )
        df = pd.read_csv(path)
        required = {
            "r_tape_cm", "amp_direct_mv", "energy_direct", "r_toa_cm",
            "pulse_num"
        }
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{path.name}: faltan columnas {sorted(missing)}")
        data[name] = df
    return data


def grouped_stats(df, column):
    """Media, desviación estándar y número de pulsos por distancia."""
    return (
        df.groupby("r_tape_cm")[column]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("r_tape_cm")
    )


def fit_power_inverse(stats, value_column_power):
    """
    Ajuste forzado por el origen en el dominio linealizado:
        y = k / r**p
    Se ajustan las medias por estación, con igual peso por posición.
    """
    r_m = stats["r_tape_cm"].to_numpy(dtype=float) / 100.0
    y = stats["mean"].to_numpy(dtype=float)
    x = 1.0 / r_m**value_column_power
    k = np.dot(x, y) / np.dot(x, x)
    y_fit = k * x
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return k, r2


def style_axis(ax):
    ax.grid(True, which="major")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True)


def save_figure(fig, stem):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Guardado: figures/{stem}.pdf y .png")


def plot_calibration(data, observable, power, ylabel, title, stem, unit_label):
    fig, (ax_lin, ax_phys) = plt.subplots(1, 2, figsize=(10.8, 4.15))
    fit_results = {}

    for name, df in data.items():
        color = COLORS[name]
        stats = grouped_stats(df, observable)
        k, r2 = fit_power_inverse(stats, power)
        fit_results[name] = (k, r2)

        r_m = stats["r_tape_cm"].to_numpy() / 100.0
        y = stats["mean"].to_numpy()
        yerr = stats["std"].fillna(0).to_numpy()
        x_lin = 1.0 / r_m**power

        # Dominio linealizado: datos medios ± desviación estándar y ajuste.
        ax_lin.errorbar(
            x_lin, y, yerr=yerr, fmt="o", ms=4.5, capsize=2.5,
            color=color, ecolor=color, alpha=0.9, label=f"{name}: datos"
        )
        xfit = np.linspace(0, x_lin.max() * 1.04, 250)
        ax_lin.plot(
            xfit, k * xfit, color=color,
            label=fr"{name}: $k={k:.4g}$, $R^2={r2:.4f}$"
        )

        # Dominio físico: curva de calibración y datos.
        ax_phys.errorbar(
            stats["r_tape_cm"], y, yerr=yerr, fmt="o", ms=4.5,
            capsize=2.5, color=color, ecolor=color, label=f"{name}: datos"
        )
        r_curve_cm = np.linspace(stats["r_tape_cm"].min() * 0.92,
                                 stats["r_tape_cm"].max() * 1.04, 300)
        r_curve_m = r_curve_cm / 100.0
        ax_phys.plot(
            r_curve_cm, k / r_curve_m**power, "--",
            color=color, label=fr"{name}: $k/r^{power}$"
        )

    ax_lin.set_title(f"(a) Dominio linealizado")
    ax_lin.set_xlabel(fr"$1/r^{power}$ [{unit_label}]")
    ax_lin.set_ylabel(ylabel)
    ax_lin.legend(frameon=True, loc="upper left")
    style_axis(ax_lin)

    ax_phys.set_title("(b) Dominio físico")
    ax_phys.set_xlabel(r"Distancia de referencia $r$ [cm]")
    ax_phys.set_ylabel(ylabel)
    ax_phys.legend(frameon=True, loc="upper right")
    style_axis(ax_phys)

    fig.suptitle(title, y=1.02, fontsize=12)
    fig.tight_layout()
    save_figure(fig, stem)
    return fit_results


def plot_mock_distance(data, amp_fits, energy_fits):
    """
    Figura provisional. Genera observables sintéticos alrededor de los modelos
    calibrados y los invierte para mostrar el formato de la figura final.
    La semilla fija hace que el mock sea reproducible.
    """
    rng = np.random.default_rng(20261020)
    r_ref_cm = np.array([15, 20, 25, 30, 40, 50, 60, 80, 100], dtype=float)
    r_ref_m = r_ref_cm / 100.0

    fig, (ax_amp, ax_energy) = plt.subplots(1, 2, figsize=(10.8, 4.2))

    for name in CSV_FILES:
        color = COLORS[name]
        k_a = amp_fits[name][0]
        k_e = energy_fits[name][0]

        # Ruido ilustrativo, no estimado de la incertidumbre experimental.
        amp_noise = rng.normal(0.0, 0.035, size=r_ref_cm.size)
        energy_noise = rng.normal(0.0, 0.055, size=r_ref_cm.size)

        amp_synthetic = (k_a / r_ref_m) * (1.0 + amp_noise)
        energy_synthetic = (k_e / r_ref_m**2) * (1.0 + energy_noise)

        r_amp_est_cm = (k_a / amp_synthetic) * 100.0
        r_energy_est_cm = np.sqrt(k_e / energy_synthetic) * 100.0

        ax_amp.plot(r_ref_cm, r_amp_est_cm, "o-", ms=4, color=color,
                    label=name)
        ax_energy.plot(r_ref_cm, r_energy_est_cm, "o-", ms=4, color=color,
                       label=name)

    lo, hi = 10, 105
    for ax in (ax_amp, ax_energy):
        ax.plot([lo, hi], [lo, hi], "--", color=COLORS["reference"],
                lw=1.3, label="Estimación ideal")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(r"Distancia de referencia $r_{\mathrm{ref}}$ [cm]")
        ax.set_ylabel(r"Distancia estimada $\hat r$ [cm]")
        ax.legend(frameon=True)
        style_axis(ax)

    ax_amp.set_title(r"(a) Estimación mediante $A_{\mathrm{RMS}}$")
    ax_energy.set_title(r"(b) Estimación mediante $E_{\mathrm{dig}}$")

    fig.suptitle(
        "FIGURA PROVISIONAL — DATOS SINTÉTICOS, NO EXPERIMENTALES",
        color="#A32626", fontsize=11, fontweight="bold", y=1.04
    )
    fig.text(
        0.5, 0.005,
        "Mock reproducible para definir la composición. Sustituir por mediciones reales.",
        ha="center", va="bottom", fontsize=8.5, color="#A32626"
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    save_figure(fig, "exp01_fig3_distance_prediction_MOCK")


def plot_toa(data):
    fig, (ax_top, ax_res) = plt.subplots(
        2, 1, figsize=(7.6, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.0], "hspace": 0.08}
    )

    all_distances = []
    for name, df in data.items():
        color = COLORS[name]
        stats = grouped_stats(df, "r_toa_cm")
        x = stats["r_tape_cm"].to_numpy()
        y = stats["mean"].to_numpy()
        yerr = stats["std"].fillna(0).to_numpy()
        residual = y - x

        ax_top.errorbar(
            x, y, yerr=yerr, fmt="o", ms=4.5, capsize=2.8,
            color=color, ecolor=color, label=name
        )
        ax_res.errorbar(
            x, residual * 10.0, yerr=yerr * 10.0, fmt="o", ms=4.5,
            capsize=2.8, color=color, ecolor=color, label=name
        )
        all_distances.extend(x.tolist())

    lo = min(all_distances) - 3
    hi = max(all_distances) + 3
    ax_top.plot([lo, hi], [lo, hi], "--", color=COLORS["reference"],
                lw=1.3, label="Referencia ideal")
    ax_top.set_ylabel(r"Distancia estimada $r_{\mathrm{ToA}}$ [cm]")
    ax_top.set_title("(a) Estimación de distancia")
    ax_top.legend(frameon=True, loc="upper left")
    style_axis(ax_top)

    ax_res.axhline(0, color=COLORS["reference"], ls="--", lw=1.2)
    ax_res.set_xlabel(r"Distancia de referencia $r_{\mathrm{ref}}$ [cm]")
    ax_res.set_ylabel("Residuo [mm]")
    ax_res.set_title("(b) Residuos de estimación")
    style_axis(ax_res)

    fig.suptitle("Estimación de distancia mediante ToA", y=0.99, fontsize=12)
    fig.tight_layout()
    save_figure(fig, "exp01_fig4_toa_range")


def main():
    print("Leyendo datos del Experimento 01...")
    data = load_data()

    print("Generando Figura 1: calibración de amplitud...")
    amp_fits = plot_calibration(
        data=data,
        observable="amp_direct_mv",
        power=1,
        ylabel=r"Amplitud RMS $A_{\mathrm{RMS}}$ [mV]",
        title="Calibración radial de amplitud",
        stem="exp01_fig1_amplitude_calibration",
        unit_label=r"m$^{-1}$",
    )

    print("Generando Figura 2: calibración de energía digital...")
    energy_fits = plot_calibration(
        data=data,
        observable="energy_direct",
        power=2,
        ylabel=r"Energía digital $E_{\mathrm{dig}}$ [unid. arb.]",
        title="Calibración radial de energía digital",
        stem="exp01_fig2_energy_calibration",
        unit_label=r"m$^{-2}$",
    )

    print("Generando Figura 3: mock de estimación de distancia...")
    plot_mock_distance(data, amp_fits, energy_fits)

    print("Generando Figura 4: estimación mediante ToA...")
    plot_toa(data)

    print("\nCoeficientes obtenidos a partir de las medias por posición:")
    for name in CSV_FILES:
        ka, r2a = amp_fits[name]
        ke, r2e = energy_fits[name]
        print(f"{name}: kA={ka:.6g} mV·m, R²={r2a:.5f}; "
              f"kE={ke:.6g}, R²={r2e:.5f}")

    print("\nProceso terminado. Revisa las figuras en la carpeta figures/.")
    print("IMPORTANTE: la Figura 3 contiene datos sintéticos de demostración.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
