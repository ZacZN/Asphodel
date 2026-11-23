"""
sum_spectra.py
--------------
Load all .npz spectra from an input directory and produce two summed spectra:

  1. Raw sum          -- intensities added bin-by-bin with no correction
  2. Drift-corrected  -- each spectrum is shifted by whole bins so its peak
                         lines up with the first spectrum's peak before summing

The peak is tracked from spectrum to spectrum (core/summing.py): the first
spectrum's strongest bin, then in every next spectrum the strongest bin within
±window (--window, default 500 Hz) of the previous spectrum's peak; a
spectrum whose peak is not found inside the window is left out (listed).  Spectra
are taken in natural file-name order (…_2 before …_10), i.e. time order.
Power values are never interpolated, and only the bins every shifted spectrum
covers are kept, so every bin of a sum holds the same number of spectra.

Each .npz file is expected to have:
    arr_0  : frequency axis (1-D array)
    arr_1  : intensity array (1-D array, same length as arr_0)

Usage
-----
    python sum_spectra.py <input_dir> <output_dir> [--window HZ]

Output files (written to <output_dir>):
    <folder_name>_summed.npz          -- raw sum
    <folder_name>_summed_aligned.npz  -- drift-corrected sum
A side-by-side comparison plot is shown at the end.
"""

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import logcapture, summing

# output also to data/logs/ and this tool's Output window (core/logcapture.py)
if __name__ == "__main__":
    logcapture.start("sum_spectra")

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def natural_key(path: Path):
    """'run_2.npz' before 'run_10.npz' (time order for numbered files)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def load_npz(path):
    """(freq, intensity) of one spectrum file; the file is closed afterwards."""
    with np.load(path) as data:
        return np.asarray(data["arr_0"], dtype=float), np.asarray(data["arr_1"], dtype=float)


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
    parser.add_argument(
        "--window", type=float, default=summing.DEFAULT_WINDOW_HZ,
        help="± window (Hz) around the previous spectrum's peak in which the next "
             f"spectrum's peak is looked for (default {summing.DEFAULT_WINDOW_HZ:g}).",
    )
    args = parser.parse_args()

    input_dir: Path = args.input_dir.expanduser().resolve()
    output_dir: Path = args.output_dir.expanduser().resolve()

    if not input_dir.is_dir():
        sys.exit(f"Error: '{input_dir}' is not a directory.")

    npz_files = sorted(input_dir.glob("*.npz"), key=natural_key)
    if not npz_files:
        sys.exit(f"Error: no .npz files found in '{input_dir}'.")

    n_files = len(npz_files)
    print(f"Found {n_files} .npz file(s) in '{input_dir}'.")

    # read from disk when needed, so a large folder is never all in memory
    spectra = summing.LazySpectra(npz_files, load_npz)

    # ------------------------------------------------------------------
    # Raw sum (no shift)
    # ------------------------------------------------------------------
    try:
        freq_raw, sum_raw, info_raw = summing.sum_whole_bins(spectra)
    except ValueError as exc:
        sys.exit(f"Error: {exc}")

    # ------------------------------------------------------------------
    # Drift-corrected sum: tracked peak, whole-bin shifts
    # ------------------------------------------------------------------
    print(f"Tracking the strongest peak (±{args.window:g} Hz from one spectrum to the next):")
    positions, lost = summing.track_strongest(spectra, args.window)
    try:
        freq_al, sum_aligned, info = summing.sum_whole_bins(spectra, positions)
    except ValueError as exc:
        sys.exit(f"Error: {exc}")
    shifts = dict(zip(info["used"], info["shift_bins"]))
    resid = dict(zip(info["used"], info["residual_hz"]))
    w = len(str(n_files))
    why = dict(lost)
    for i, path in enumerate(npz_files):
        if positions[i] is None:
            print(f"  [{i + 1:>{w}}/{n_files}] {path.name}  {why[i]} — left out")
        else:
            print(f"  [{i + 1:>{w}}/{n_files}] {path.name}  peak={positions[i]:.1f} Hz  "
                  f"shift={shifts[i]:+d} bins  (left over {resid[i]:+.1f} Hz)")
    if lost:
        print(f"WARNING: {len(lost)} spectrum/spectra left out of the drift-corrected sum "
              f"(peak not found near the tracked position; a larger --window may help).")
    for name, inf in (("raw", info_raw), ("drift-corrected", info)):
        if inf["cropped_low"] or inf["cropped_high"]:
            print(f"Bins not covered by every spectrum dropped from the {name} sum: "
                  f"{inf['cropped_low']} at the low end, {inf['cropped_high']} at the high end.")

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    folder_name = input_dir.name

    out_raw = output_dir / f"{folder_name}_summed.npz"
    out_aligned = output_dir / f"{folder_name}_summed_aligned.npz"

    np.savez(out_raw, arr_0=freq_raw, arr_1=sum_raw)
    np.savez(out_aligned, arr_0=freq_al, arr_1=sum_aligned)

    print(f"\nSaved raw sum      → {out_raw}")
    print(f"Saved aligned sum  → {out_aligned}  ({info['n']} spectra)")

    # ------------------------------------------------------------------
    # Side-by-side comparison plot
    # ------------------------------------------------------------------
    fig, (ax_raw, ax_al) = plt.subplots(
        1, 2, figsize=(14, 5), sharey=False
    )

    ax_raw.plot(freq_raw, sum_raw, lw=0.8, color="steelblue")
    ax_raw.set_title(f"Raw Sum  ({info_raw['n']} spectra)", fontsize=12)
    ax_raw.set_xlabel("Frequency")
    ax_raw.set_ylabel("Summed Intensity")

    ax_al.plot(freq_al, sum_aligned, lw=0.8, color="darkorange")
    ax_al.set_title(f"Drift-Corrected Sum  ({info['n']} spectra)", fontsize=12)
    ax_al.set_xlabel("Frequency")
    ax_al.set_ylabel("Summed Intensity")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
