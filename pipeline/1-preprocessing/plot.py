"""
plot basic raman spectra optimized for mos2
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re
from io import StringIO

def main():

    input_file = "data/mos2_data.txt"
    output_dir = Path("results/mos2")
    xmin, xmax = 200, 1400

    output_dir.mkdir(parents=True, exist_ok=True)

    txt = Path(input_file).read_text(encoding="utf-8", errors="ignore")
    if re.search(r"\d+,\d+", txt):
        txt = re.sub(",", ".", txt)

    df = pd.read_csv(
        StringIO(txt),
        sep=r"\s+",
        names=["Wavenumber", "Intensity"],
        header=None,
        engine="python",
    ).query("@xmin <= Wavenumber <= @xmax")

    fig, ax = plt.subplots(figsize=(10, 7)) 
    ax.plot(df["Wavenumber"], df["Intensity"], linewidth=1.5)


    ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=16)
    ax.set_ylabel("Intensity (a.u.)",   fontsize=16)
    ax.set_title("Raw Raman Spectrum (Graphene)", fontsize=20, pad=15)
    ax.set_xlim(xmin, xmax)
    ax.margins(x=0)
    ax.tick_params(axis="both", which="major", labelsize=14)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()

    out = output_dir / "raw_spectrum_mos2.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Plot saved → {out}")
    plt.show()

if __name__ == "__main__":
    main()