"""
Borehole Televiewer Image Analysis

Analyses BH01 and BH02 televiewer log images to:
  1. Separate optical and acoustic image channels
  2. Detect fractures / planar features via sinusoidal detection
     (dark-pixel thresholding - Hough-based sinusoid voting)
  3. Extract fracture parameters (depth, amplitude, phase - dip angle proxy)
  4. Produce summary statistics and visualisations

Author: Arnav Bhat
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from scipy.ndimage import median_filter
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks
import warnings
warnings.filterwarnings("ignore")


# CONFIGURATION

IMAGE_PATHS = {
    "BH01": r"C:\Users\ARNAV BHAT\OneDrive\Documents\Arduino\microint\BH01_TeleviewerLog_With Picks.bmp",
    "BH02": r"C:\Users\ARNAV BHAT\OneDrive\Documents\Arduino\microint\BH02_TeleviewerLog_With Picks.bmp",
}
OUTPUT_DIR = r"C:\Users\ARNAV BHAT\OneDrive\Documents\Arduino\microint"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Column layout (x-pixel bounds) shared across both images.
# Determined by visual inspection of the log header:
#   Depth label | Optical channel | Acoustic channel | Dips | … rest
COL_OPTICAL  = (103, 253)   # x start, x end
COL_ACOUSTIC = (255, 420)   # x start, x end
HEADER_ROWS  = 100          # rows to skip (header / scale bar)

# Fracture-detection thresholds
DARK_PIXEL_FRACTION = 0.20   # keep darkest 20 % of pixels
MIN_FRACTURE_VOTES  = 8      # minimum column votes to keep a sinusoid candidate
PHASE_STEP_DEG      = 10     # degrees: step for phase search
AMP_STEPS           = 15     # number of amplitude steps to try

# IMAGE LOADING & CHANNEL EXTRACTION

def load_image(path: str) -> np.ndarray:
    """Load a BMP televiewer log and return as uint8 RGB array."""
    img = Image.open(path).convert("RGB")
    return np.array(img)


def extract_channel(arr: np.ndarray, col_bounds: tuple, header_rows: int) -> np.ndarray:
    """
    Crop a single log channel (optical or acoustic) from the full log image.
 
    Parameters
    ----------
    arr        : full RGB image array  (H, W, 3)
    col_bounds : (x_start, x_end) pixel columns of the channel
    header_rows: rows to skip at the top

    Returns
    -------
    channel_rgb : np.ndarray  (H', W', 3)
    """
    x0, x1 = col_bounds
    channel = arr[header_rows:, x0:x1, :]
    return channel

# PRE-PROCESSING

def preprocess_channel(channel_rgb: np.ndarray) -> np.ndarray:
    """
    Convert channel to grayscale, apply median filter, and threshold to
    keep only the darkest fraction of pixels (likely fractures / dark features).

    Returns a binary mask (True = dark feature pixel).
    """
    gray = np.mean(channel_rgb, axis=2).astype(np.float32)

    # Remove salt-and-pepper noise
    gray_filt = median_filter(gray, size=3)

    # Threshold: keep darkest DARK_PIXEL_FRACTION of pixels
    threshold = np.percentile(gray_filt, DARK_PIXEL_FRACTION * 100)
    binary = gray_filt <= threshold

    return binary, gray_filt

# SINUSOID/ FRACTURE DETECTION

def detect_sinusoids(binary: np.ndarray, gray: np.ndarray):
    """
    Detect sinusoidal fracture traces in a pre-processed binary channel image.

    Strategy (following Hall 1996 / Cruz 2017 approach):
      For each candidate row position y0 (stepped every 2 pixels):
        For each amplitude A and phase θ:
          Count dark pixels that fall on y = y0 + A*sin(x*2π/W + θ)
      Keep the (y0, A, θ) triplet with the most votes.

    Returns a list of dicts, one per detected fracture:
        {depth_row, amplitude_px, phase_deg, votes, dip_angle_approx}
    """
    H, W = binary.shape
    fractures = []

    # Amplitude range: 2 px to 15 % of channel width
    amp_min, amp_max = 2, max(3, int(0.15 * W))
    amplitudes = np.linspace(amp_min, amp_max, AMP_STEPS).astype(int)
    phases_deg = np.arange(0, 360, PHASE_STEP_DEG)
    x_coords   = np.arange(W)

    # Step through candidate baseline rows (every 4 px for speed)
    row_step = 4
    already_used = set()   # avoid duplicate detections within 8 px

    for y0 in range(0, H, row_step):
        best_votes, best_A, best_theta = 0, 0, 0

        for A in amplitudes:
            for theta in phases_deg:
                theta_rad = np.deg2rad(theta)
                y_curve = (y0 + A * np.sin(2 * np.pi * x_coords / W + theta_rad)).astype(int)
                # Clip to image bounds
                valid = (y_curve >= 0) & (y_curve < H)
                if valid.sum() < W * 0.6:
                    continue
                votes = binary[y_curve[valid], x_coords[valid]].sum()
                if votes > best_votes:
                    best_votes, best_A, best_theta = votes, A, theta

        if best_votes >= MIN_FRACTURE_VOTES:
            # Suppress duplicates within ±8 rows
            nearby = any(abs(y0 - u) <= 8 for u in already_used)
            if not nearby:
                # Dip angle proxy: larger amplitude relative to width → steeper dip
                dip_approx = np.degrees(np.arctan2(best_A, W / (2 * np.pi)))
                fractures.append({
                    "depth_row"       : y0,
                    "amplitude_px"    : int(best_A),
                    "phase_deg"       : int(best_theta),
                    "votes"           : int(best_votes),
                    "dip_angle_approx": round(dip_approx, 1),
                })
                already_used.add(y0)

    return fractures

# SUMMARY STATISTICS

def compute_statistics(fractures: list, channel_name: str, borehole_id: str) -> dict:
    """Compute summary statistics for detected fractures."""
    n = len(fractures)
    if n == 0:
        return {
            "borehole": borehole_id, "channel": channel_name,
            "count": 0,
            "mean_amplitude_px": None, "std_amplitude_px": None,
            "mean_dip_deg": None, "std_dip_deg": None,
            "mean_phase_deg": None,
        }

    amps  = [f["amplitude_px"]     for f in fractures]
    dips  = [f["dip_angle_approx"] for f in fractures]
    phases= [f["phase_deg"]        for f in fractures]

    return {
        "borehole"          : borehole_id,
        "channel"           : channel_name,
        "count"             : n,
        "mean_amplitude_px" : round(float(np.mean(amps)),  2),
        "std_amplitude_px"  : round(float(np.std(amps)),   2),
        "mean_dip_deg"      : round(float(np.mean(dips)),  2),
        "std_dip_deg"       : round(float(np.std(dips)),   2),
        "mean_phase_deg"    : round(float(np.mean(phases)), 2),
    }

# VISUALISATION

def plot_results(channel_rgb, binary, gray, fractures, title, save_path):
    """
    4-panel figure:
      (a) original channel strip
      (b) binary (dark-pixel) mask
      (c) grayscale with detected sinusoids overlaid in cyan
      (d) dip-angle histogram
    """
    H, W = binary.shape
    x_coords = np.arange(W)

    fig, axes = plt.subplots(1, 4, figsize=(18, max(6, H // 80)),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.5]})
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Panel (a): original channel
    axes[0].imshow(channel_rgb, aspect="auto")
    axes[0].set_title("(a) Original channel")
    axes[0].set_xlabel("Azimuth (px)")
    axes[0].set_ylabel("Depth (px)")

    # Panel (b): binary mask
    axes[1].imshow(binary, cmap="gray", aspect="auto")
    axes[1].set_title("(b) Dark-pixel mask (20 %)")
    axes[1].set_xlabel("Azimuth (px)")

    # Panel (c): grayscale + detected sinusoids
    axes[2].imshow(gray, cmap="gray", aspect="auto", vmin=0, vmax=255)
    axes[2].set_title(f"(c) Detected fractures (n={len(fractures)})")
    axes[2].set_xlabel("Azimuth (px)")

    colors = plt.cm.hsv(np.linspace(0, 1, max(1, len(fractures))))
    for i, frac in enumerate(fractures):
        y0    = frac["depth_row"]
        A     = frac["amplitude_px"]
        theta = np.deg2rad(frac["phase_deg"])
        y_curve = (y0 + A * np.sin(2 * np.pi * x_coords / W + theta)).astype(int)
        valid = (y_curve >= 0) & (y_curve < H)
        axes[2].plot(x_coords[valid], y_curve[valid],
                     color=colors[i], linewidth=1.2, alpha=0.85)

    # Panel (d): dip-angle histogram
    if fractures:
        dips = [f["dip_angle_approx"] for f in fractures]
        axes[3].hist(dips, bins=max(5, len(fractures)//2 + 1),
                     color="steelblue", edgecolor="white", linewidth=0.6)
        axes[3].axvline(np.mean(dips), color="red", linestyle="--",
                        linewidth=1.5, label=f"Mean = {np.mean(dips):.1f}°")
        axes[3].set_xlabel("Estimated dip angle (°)")
        axes[3].set_ylabel("Count")
        axes[3].set_title("(d) Dip angle distribution")
        axes[3].legend(fontsize=9)
    else:
        axes[3].text(0.5, 0.5, "No fractures\ndetected",
                     ha="center", va="center", transform=axes[3].transAxes,
                     fontsize=12, color="gray")
        axes[3].set_title("(d) Dip angle distribution")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_path}")


def plot_combined_summary(all_stats, save_path):
    """Bar chart comparing fracture counts and mean dip across boreholes/channels."""
    labels = [f"{s['borehole']}\n{s['channel']}" for s in all_stats]
    counts  = [s["count"] for s in all_stats]
    dips    = [s["mean_dip_deg"] if s["mean_dip_deg"] is not None else 0
               for s in all_stats]

    x = np.arange(len(labels))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Cross-borehole comparison summary", fontsize=13, fontweight="bold")

    bars1 = ax1.bar(x, counts, width, color=["#4C72B0","#55A868","#C44E52","#8172B2"])
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_ylabel("Number of fractures detected")
    ax1.set_title("Fracture counts by borehole & channel")
    for bar, val in zip(bars1, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 str(val), ha="center", va="bottom", fontsize=10)

    bars2 = ax2.bar(x, dips, width, color=["#4C72B0","#55A868","#C44E52","#8172B2"])
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_ylabel("Mean estimated dip angle (°)")
    ax2.set_title("Mean dip angle by borehole & channel")
    for bar, val in zip(bars2, dips):
        if val:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                     f"{val:.1f}°", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_path}")

# MAIN PIPELINE

def analyse_borehole(bh_id: str, image_path: str) -> list:
    """Full pipeline for one borehole image. Returns list of stats dicts."""
    print(f"\n{'='*55}")
    print(f"  Processing {bh_id}  ({os.path.basename(image_path)})")
    print(f"{'='*55}")

    arr = load_image(image_path)
    print(f"  Image size: {arr.shape[1]} x {arr.shape[0]} px")

    results = []

    for chan_name, col_bounds in [("Optical",  COL_OPTICAL),
                                   ("Acoustic", COL_ACOUSTIC)]:
        print(f"\n  ── Channel: {chan_name} (x={col_bounds[0]}–{col_bounds[1]}) ──")

        channel_rgb = extract_channel(arr, col_bounds, HEADER_ROWS)
        print(f"     Channel crop: {channel_rgb.shape[1]} x {channel_rgb.shape[0]} px")

        binary, gray = preprocess_channel(channel_rgb)
        dark_frac = binary.mean()
        print(f"     Dark-pixel fraction: {dark_frac:.3f}")

        print(f"     Running sinusoid detection …")
        fractures = detect_sinusoids(binary, gray)
        print(f"     Detected fractures: {len(fractures)}")

        if fractures:
            amps = [f["amplitude_px"]     for f in fractures]
            dips = [f["dip_angle_approx"] for f in fractures]
            print(f"     Amplitude (px): mean={np.mean(amps):.1f}, std={np.std(amps):.1f}")
            print(f"     Dip angle (°) : mean={np.mean(dips):.1f}, std={np.std(dips):.1f}")

        # Save channel visualisation
        fig_path = os.path.join(OUTPUT_DIR,
                                f"{bh_id}_{chan_name.lower()}_analysis.png")
        plot_results(
            channel_rgb, binary, gray, fractures,
            title=f"{bh_id} – {chan_name} channel analysis",
            save_path=fig_path,
        )

        stats = compute_statistics(fractures, chan_name, bh_id)
        results.append(stats)

    return results


def print_report(all_stats: list):
    """Print a formatted text report to stdout."""
    print(f"\n{'='*55}")
    print("  SUMMARY REPORT")
    print(f"{'='*55}")
    header = f"{'Borehole':<8} {'Channel':<10} {'Count':>6} {'Mean Amp(px)':>14} {'Mean Dip(°)':>12} {'Std Dip(°)':>11}"
    print(header)
    print("-" * len(header))
    for s in all_stats:
        count = s["count"]
        ma    = f"{s['mean_amplitude_px']:.1f}" if s['mean_amplitude_px'] is not None else "—"
        md    = f"{s['mean_dip_deg']:.1f}"       if s['mean_dip_deg']       is not None else "—"
        sd    = f"{s['std_dip_deg']:.1f}"        if s['std_dip_deg']        is not None else "—"
        print(f"{s['borehole']:<8} {s['channel']:<10} {count:>6} {ma:>14} {md:>12} {sd:>11}")
    print(f"{'='*55}\n")


def main():
    all_stats = []

    for bh_id, img_path in IMAGE_PATHS.items():
        if not os.path.exists(img_path):
            print(f"[WARN] Image not found, skipping: {img_path}")
            continue
        stats_list = analyse_borehole(bh_id, img_path)
        all_stats.extend(stats_list)

    # Cross-borehole comparison chart
    summary_path = os.path.join(OUTPUT_DIR, "borehole_comparison_summary.png")
    plot_combined_summary(all_stats, summary_path)

    print_report(all_stats)
    print("Analysis complete. Output files saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()


 