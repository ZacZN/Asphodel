"""
sum_spectra.py
--------------
Load all .npz spectra from an input directory and produce two summed spectra:

  1. Raw sum          -- intensities added bin-by-bin with no correction
  2. Drift-corrected  -- each spectrum is integer-shifted so its peak aligns
                         with the peak of the first spectrum before summing

Each .npz file is expected to have:
    arr_0  : frequency axis (1-D array)
    arr_1  : intensity array (1-D array, same length as arr_0)

Usage
-----
    python sum_spectra.py <input_dir> <output_dir>

Output files (written to <output_dir>):
    <folder_name>_summed.npz          -- raw sum
    <folder_name>_summed_aligned.npz  -- drift-corrected sum
    <folder_name>_comparison.png      -- side-by-side comparison plot
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import logcapture

# output also to data/logs/ and this tool's Output window (core/logcapture.py)
if __name__ == "__main__":
    logcapture.start("sum_spectra")

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def integer_shift(arr: np.ndarray, shift: int) -> np.ndarray:
    """
    Shift *arr* by *shift* bins (positive = shift right).
    Vacated bins are filled with zero; no wrap-around.
    """
    if shift == 0:
        return arr.copy()
    out = np.roll(arr, shift)
    if shift > 0:
        out[:shift] = 0.0
    else:
        out[shift:] = 0.0
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sum 1-D spectra from a directory of .npz files."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing the .npz spectrum files.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where output .npz files and plot will be saved.",
    )
    args = parser.parse_args()

    input_dir: Path = args.input_dir.expanduser().resolve()
    output_dir: Path = args.output_dir.expanduser().resolve()

    if not input_dir.is_dir():
        sys.exit(f"Error: '{input_dir}' is not a directory.")

    npz_files = sorted(input_dir.glob("*.npz"))
    if not npz_files:
        sys.exit(f"Error: no .npz files found in '{input_dir}'.")

    n_files = len(npz_files)
    print(f"Found {n_files} .npz file(s) in '{input_dir}'.")

    # ------------------------------------------------------------------
    # Load the first file — determines array size and reference peak
    # ------------------------------------------------------------------
    first = np.load(npz_files[0])
    freq: np.ndarray = first["arr_0"]
    first_intensity: np.ndarray = first["arr_1"].astype(float)
    n_bins = len(first_intensity)

    ref_peak_idx: int = int(np.argmax(first_intensity))
    print(
        f"Reference peak (file 1/{n_files}): bin {ref_peak_idx}, "
        f"frequency {freq[ref_peak_idx]:.6g}"
    )

    # ------------------------------------------------------------------
    # Accumulate sums
    # ------------------------------------------------------------------
    sum_raw = np.zeros(n_bins)
    sum_aligned = np.zeros(n_bins)

    for file_idx, npz_file in enumerate(npz_files):
        data = np.load(npz_file)
        intensity = data["arr_1"].astype(float)

        if len(intensity) != n_bins:
            print(
                f"  WARNING: '{npz_file.name}' has {len(intensity)} bins "
                f"(expected {n_bins}) — skipping."
            )
            continue

        # Raw accumulation
        sum_raw += intensity

        # Peak alignment
        peak_idx = int(np.argmax(intensity))
        shift = ref_peak_idx - peak_idx
        sum_aligned += integer_shift(intensity, shift)

        print(
            f"  [{file_idx + 1:>{len(str(n_files))}}/{n_files}] "
            f"{npz_file.name}  peak_bin={peak_idx}  shift={shift:+d}"
        )

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    folder_name = input_dir.name

    out_raw = output_dir / f"{folder_name}_summed.npz"
    out_aligned = output_dir / f"{folder_name}_summed_aligned.npz"

    np.savez(out_raw, arr_0=freq, arr_1=sum_raw)
    np.savez(out_aligned, arr_0=freq, arr_1=sum_aligned)

    print(f"\nSaved raw sum      → {out_raw}")
    print(f"Saved aligned sum  → {out_aligned}")

    # ------------------------------------------------------------------
    # Side-by-side comparison plot
    # ------------------------------------------------------------------
    fig, (ax_raw, ax_al) = plt.subplots(
        1, 2, figsize=(14, 5), sharey=False
    )

    ax_raw.plot(freq, sum_raw, lw=0.8, color="steelblue")
    ax_raw.set_title(f"Raw sum  ({n_files} spectra)", fontsize=12)
    ax_raw.set_xlabel("Frequency")
    ax_raw.set_ylabel("Summed intensity")

    ax_al.plot(freq, sum_aligned, lw=0.8, color="darkorange")
    ax_al.set_title(f"Drift-corrected sum  ({n_files} spectra)", fontsize=12)
    ax_al.set_xlabel("Frequency")
    ax_al.set_ylabel("Summed intensity")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
