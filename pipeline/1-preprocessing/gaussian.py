from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths, savgol_filter
from scipy import interpolate

RAW_PATH      = Path("data/graphene_1.txt")
SG_PATH       = Path("results/graphene/mos2_denoised.csv")
OUT_DIR       = Path("results/mos2")
PNG_FILE      = OUT_DIR / "mos2_gaussian_improved.png"
PEAKS_JSON    = OUT_DIR / "mos2_peaks.json"
RANGE_EXPR    = "200 <= Wavenumber <= 1400"

def load_raw() -> pd.DataFrame:
    df = pd.read_csv(
        RAW_PATH,
        sep=r"\s+",
        header=None,
        names=["Wavenumber", "Intensity"],
        decimal=".",
    )
    
    df["Wavenumber"] = pd.to_numeric(df["Wavenumber"], errors='coerce')
    df["Intensity"] = pd.to_numeric(df["Intensity"], errors='coerce')
    df = df.dropna()
    
    return (
        df.query(RANGE_EXPR)
        .sort_values("Wavenumber")
        .reset_index(drop=True)
    )

def load_sg() -> pd.DataFrame:
    df = pd.read_csv(SG_PATH)
    
    if "Denoised" in df.columns:
        df["Denoised"] = pd.to_numeric(df["Denoised"], errors='coerce')
    
    if "Wavenumber" in df.columns:
        df["Wavenumber"] = pd.to_numeric(df["Wavenumber"], errors='coerce')
    
    df = df.dropna()
    return df.query(RANGE_EXPR).reset_index(drop=True)

def gaussian_smooth_adaptive(y: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Gentle Gaussian smoothing that preserves peak structure"""
    y = np.asarray(y, dtype=np.float64)
    return gaussian_filter1d(y, sigma)

def gaussian_smooth_variable(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Variable sigma based on local data density and peak detection"""
    y = np.asarray(y, dtype=np.float64)
    

    baseline_threshold = 0.1 * np.max(y)
    

    sigma_array = np.where(y > baseline_threshold, 0.8, 2.0)
    

    result = np.zeros_like(y)
    for i in range(len(y)):
        local_sigma = sigma_array[i]

        window_size = int(6 * local_sigma)
        start_idx = max(0, i - window_size)
        end_idx = min(len(y), i + window_size + 1)
        

        local_y = y[start_idx:end_idx]
        local_smooth = gaussian_filter1d(local_y, local_sigma)
        result[i] = local_smooth[i - start_idx] if i - start_idx < len(local_smooth) else y[i]
    
    return result

def savitzky_golay_custom(y: np.ndarray, window_length: int = 15, polyorder: int = 3) -> np.ndarray:
    """Custom Savitzky-Golay filter as alternative to Gaussian"""
    y = np.asarray(y, dtype=np.float64)
    

    if window_length % 2 == 0:
        window_length += 1
    if window_length >= len(y):
        window_length = len(y) - 1 if len(y) % 2 == 0 else len(y) - 2
    
    return savgol_filter(y, window_length, polyorder)

def moving_average_weighted(y: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Weighted moving average that preserves peaks better than Gaussian"""
    y = np.asarray(y, dtype=np.float64)
    

    weights = np.bartlett(window_size)
    weights = weights / weights.sum()
    

    pad_width = window_size // 2
    y_padded = np.pad(y, pad_width, mode='edge')
    

    result = np.convolve(y_padded, weights, mode='same')
    

    return result[pad_width:-pad_width]

def detect_peaks_improved(x: np.ndarray, y: np.ndarray, method: str = "adaptive") -> list[dict]:
    """Improved peak detection with multiple prominence thresholds"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    

    if method == "adaptive":

        noise_level = np.std(y[y < 0.1 * np.max(y)])
        min_prominence = max(0.02 * y.max(), 3 * noise_level)
    else:
        min_prominence = 0.05 * y.max()
    

    idx, props = find_peaks(y, 
                          prominence=min_prominence,
                          distance=5, 
                          height=0.01 * y.max()) 
    
    if len(idx) == 0:
        idx, props = find_peaks(y, prominence=0.01 * y.max())
    
    widths, _, left, right = peak_widths(y, idx, rel_height=0.5)
    step = np.mean(np.diff(x))
    peaks = []
    
    for i, w, l, r in zip(idx, widths, left, right):

        peak_height = y[i]
        local_background = np.mean([y[max(0, i-10)], y[min(len(y)-1, i+10)]])
        signal_to_noise = peak_height / (local_background + 1e-10)
        
        peaks.append({
            "position": float(x[i]),
            "intensity": float(y[i]),
            "prominence": float(props['prominences'][list(idx).index(i)]) if 'prominences' in props else 0.0,
            "whm": float(w * step),
            "left_half": float(x[int(l)]) if int(l) < len(x) else float(x[0]),
            "right_half": float(x[int(r)]) if int(r) < len(x) else float(x[-1]),
            "signal_to_noise": float(signal_to_noise),
        })
    

    peaks.sort(key=lambda p: p["intensity"], reverse=True)
    
    return peaks

def save_json(peaks: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PEAKS_JSON, "w") as fh:
        json.dump(peaks, fh, indent=2)

def make_comparison_plots(x, y_raw, y_sg, smoothing_results, peaks_dict):
    """Create comprehensive comparison plots"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    methods = list(smoothing_results.keys())
    colors = ['green', 'purple', 'red', 'brown']
    
    axes[0,0].plot(x, y_raw, label="Raw", alpha=0.3, linewidth=2, color='lightblue')
    axes[0,0].plot(x, y_sg, label="Original SG", color="orange", linewidth=2)
    
    for i, (method, y_smooth) in enumerate(smoothing_results.items()):
        axes[0,0].plot(x, y_smooth, label=method, color=colors[i], linewidth=2)
        
        if method in peaks_dict:
            peaks = peaks_dict[method]
            p_x = [p["position"] for p in peaks]
            p_int = [p["intensity"] for p in peaks]
            axes[0,0].scatter(p_x, p_int, marker="x", color=colors[i], zorder=3, s=50)
    
    axes[0,0].set_title("Smoothing Methods Comparison")
    axes[0,0].set_xlabel("Wavenumber (cm⁻¹)")
    axes[0,0].set_ylabel("Intensity (a.u.)")
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    main_peak_mask = (x >= 350) & (x <= 450)
    axes[0,1].plot(x[main_peak_mask], y_raw[main_peak_mask], alpha=0.3, linewidth=3, color='lightblue', label="Raw")
    axes[0,1].plot(x[main_peak_mask], y_sg[main_peak_mask], color="orange", linewidth=2, label="Original SG")
    
    for i, (method, y_smooth) in enumerate(smoothing_results.items()):
        axes[0,1].plot(x[main_peak_mask], y_smooth[main_peak_mask], color=colors[i], linewidth=2, label=method)
        
        if method in peaks_dict:
            peaks = peaks_dict[method]
            region_peaks_x = [p["position"] for p in peaks if 350 <= p["position"] <= 450]
            region_peaks_y = [p["intensity"] for p in peaks if 350 <= p["position"] <= 450]
            axes[0,1].scatter(region_peaks_x, region_peaks_y, marker="x", color=colors[i], zorder=3, s=80)
    
    axes[0,1].set_title("Main Peak Region (350-450 cm⁻¹)")
    axes[0,1].set_xlabel("Wavenumber (cm⁻¹)")
    axes[0,1].set_ylabel("Intensity (a.u.)")
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    

    best_method = list(smoothing_results.keys())[0]
    y_best = smoothing_results[best_method]
    best_peaks = peaks_dict.get(best_method, [])
    
    axes[1,0].plot(x, y_raw, alpha=0.2, linewidth=1, color='lightgray', label="Raw")
    axes[1,0].plot(x, y_best, color=colors[0], linewidth=2, label=f"{best_method}")
    
    if best_peaks:
        p_x = [p["position"] for p in best_peaks]
        p_int = [p["intensity"] for p in best_peaks]
        axes[1,0].scatter(p_x, p_int, marker="o", color="red", zorder=3, s=60, label="Detected Peaks")
        

        for px, py in zip(p_x[:5], p_int[:5]):
            axes[1,0].annotate(f'{px:.0f}', (px, py), 
                             xytext=(5, 5), textcoords='offset points', 
                             fontsize=8, alpha=0.8)
    
    axes[1,0].set_title(f"Best Result: {best_method}")
    axes[1,0].set_xlabel("Wavenumber (cm⁻¹)")
    axes[1,0].set_ylabel("Intensity (a.u.)")
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    if peaks_dict:
        methods_list = list(peaks_dict.keys())
        peak_counts = [len(peaks_dict[method]) for method in methods_list]
        
        bars = axes[1,1].bar(methods_list, peak_counts, color=colors[:len(methods_list)], alpha=0.7)
        axes[1,1].set_title("Number of Detected Peaks by Method")
        axes[1,1].set_ylabel("Peak Count")
        axes[1,1].tick_params(axis='x', rotation=45)
        

        for bar, count in zip(bars, peak_counts):
            axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                          str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(PNG_FILE, dpi=300, bbox_inches="tight")

def main():
    try:
        raw_df = load_raw()
        sg_df = load_sg()

        x = raw_df["Wavenumber"].to_numpy().astype(np.float64)
        y_raw = raw_df["Intensity"].to_numpy().astype(np.float64)
        y_sg = sg_df["Denoised"].to_numpy().astype(np.float64)
        
        print(f"Data loaded successfully:")
        print(f"  Data range: {x.min():.1f} - {x.max():.1f} cm⁻¹")
        print(f"  Raw intensity range: {y_raw.min():.3f} - {y_raw.max():.3f}")
        
        smoothing_results = {
            "Gentle Gaussian (σ=1.0)": gaussian_smooth_adaptive(y_raw, sigma=1.0),
            "Variable Gaussian": gaussian_smooth_variable(x, y_raw),
            "Savitzky-Golay": savitzky_golay_custom(y_raw, window_length=11, polyorder=3),
            "Weighted Moving Avg": moving_average_weighted(y_raw, window_size=7)
        }
        
        peaks_dict = {}
        for method, y_smooth in smoothing_results.items():
            peaks = detect_peaks_improved(x, y_smooth, method="adaptive")
            peaks_dict[method] = peaks
            print(f"{method}: Found {len(peaks)} peaks")
        
        best_method = "Gentle Gaussian (σ=1.0)"
        best_peaks = peaks_dict[best_method]
        save_json(best_peaks)
        print(f"Best peaks ({best_method}) saved to {PEAKS_JSON}")
        
        make_comparison_plots(x, y_raw, y_sg, smoothing_results, peaks_dict)
        print(f"Comparison plots saved to {PNG_FILE}")
        
        print(f"\nTop peaks using {best_method}:")
        for i, peak in enumerate(best_peaks[:5]):
            print(f"  {i+1}. {peak['position']:.1f} cm⁻¹ (intensity: {peak['intensity']:.3f}, "
                  f"prominence: {peak['prominence']:.3f})")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()