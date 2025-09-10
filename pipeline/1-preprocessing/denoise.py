"""
Improved Savitzky–Golay denoising for MoS₂ Raman spectra
with publication-quality plotting (three panels, all overlays).

Produces:
  • results/mos2/mos2_denoised.csv   – Wavenumber, Denoised
  • results/mos2/denoised_spectra_mos2.png – 3-panel figure
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from io import StringIO


DATA_PATH   = Path("data/mos2_data.txt")
OUTPUT_DIR  = Path("results/mos2")
PNG_FILE    = OUTPUT_DIR / "denoised_spectra_mos2.png"
CSV_FILE    = OUTPUT_DIR / "mos2_denoised.csv"

WN_MIN, WN_MAX = 200, 1400         
ZOOM_RANGE     = (350, 450)  
WINDOW_SIZE    = 5       
POLY_ORDER     = 2


def load_spectrum(path: Path) -> pd.DataFrame:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    return pd.read_csv(
        StringIO(txt),
        sep=r"\s+",
        names=["Wavenumber", "Intensity"],
        header=None,
        engine="python",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = (
        load_spectrum(DATA_PATH)
        .query(f"{WN_MIN} <= Wavenumber <= {WN_MAX}")
        .sort_values("Wavenumber")
        .reset_index(drop=True)
    )


    df["Denoised"] = savgol_filter(
        df["Intensity"].to_numpy(), WINDOW_SIZE, POLY_ORDER, mode="nearest"
    )
    df[["Wavenumber", "Denoised"]].to_csv(CSV_FILE, index=False)
    print(f"✅  Denoised data → {CSV_FILE}")


    plt.rcParams.update({
        "figure.dpi": 100,
        "axes.linewidth": 1.2,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "grid.alpha": 0.4,
        "grid.linewidth": 0.5,
    })

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), constrained_layout=True)


    axes[0].plot(df["Wavenumber"], df["Intensity"],
                 color="#1f77b4", linewidth=1, label="Raw")
    axes[0].plot(df["Wavenumber"], df["Denoised"],
                 color="#ff7f0e", linewidth=1.5, label="Denoised")
    axes[0].set_title("Denoised Raman Spectrum")
    axes[0].set_xlabel("Wavenumber (cm$^{-1}$)")
    axes[0].set_ylabel("Intensity (a.u.)")
    axes[0].set_xlim(WN_MIN, WN_MAX)
    axes[0].margins(x=0)
    axes[0].legend(frameon=False)
    axes[0].grid(True)


    axes[1].plot(df["Wavenumber"], df["Intensity"],
                 color="#1f77b4", linewidth=1, label="Raw")
    axes[1].plot(df["Wavenumber"], df["Denoised"],
                 color="#ff7f0e", linewidth=1.5, label="Denoised")
    axes[1].set_title("Raw vs. Denoised")
    axes[1].set_xlabel("Wavenumber (cm$^{-1}$)")
    axes[1].set_ylabel("Intensity (a.u.)")
    axes[1].set_xlim(WN_MIN, WN_MAX)
    axes[1].margins(x=0)
    axes[1].legend(frameon=False)
    axes[1].grid(True)


    zoom_df = df.query(f"{ZOOM_RANGE[0]} <= Wavenumber <= {ZOOM_RANGE[1]}")
    axes[2].plot(zoom_df["Wavenumber"], zoom_df["Intensity"],
                 color="#1f77b4", linewidth=1, label="Raw")
    axes[2].plot(zoom_df["Wavenumber"], zoom_df["Denoised"],
                 color="#ff7f0e", linewidth=1.5, label="Denoised")
    axes[2].set_title(
        f"Zoom {ZOOM_RANGE[0]}–{ZOOM_RANGE[1]} cm$^{{-1}}$ (Raw vs Denoised)"
    )
    axes[2].set_xlabel("Wavenumber (cm$^{-1}$)")
    axes[2].set_ylabel("Intensity (a.u.)")
    axes[2].set_xlim(*ZOOM_RANGE)
    axes[2].margins(x=0)
    axes[2].legend(frameon=False)
    axes[2].grid(True)

    fig.savefig(PNG_FILE, dpi=300, bbox_inches="tight")
    print(f"✅  Plot saved → {PNG_FILE}")
    print(f"📊  SG params → window={WINDOW_SIZE}, poly={POLY_ORDER}")

if __name__ == "__main__":
    main()