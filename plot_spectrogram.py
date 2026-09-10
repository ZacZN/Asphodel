#!/usr/bin/env python3
"""
Plot a spectrogram from a single ESR injection .npz file.

Usage
-----
  python plot_spectrogram.py <file> <freq_min> <freq_max> [--linear] [--plot3d] [--time-avg N]

Arguments
---------
  file          Path to a .npz file (freq, time, power arrays)
  freq_min      Lower frequency bound (same units as the freq array)
  freq_max      Upper frequency bound
  --linear      Plot raw power instead of 10·log10(power)
  --plot3d      Interactive 3-D surface plot (uses plotly; opens in browser)
  --time-avg N  Average every N consecutive time frames (e.g. 5 → T/5 frames)

Example
-------
  python plot_spectrogram.py 1081.5.npz 245000 246000
  python plot_spectrogram.py 1081.5+1082.0.npz 245000 246000 --plot3d
  python plot_spectrogram.py 1081.5.npz 245000 246000 --linear --plot3d
  python plot_spectrogram.py 1081.5.npz 245000 246000 --time-avg 5
"""

import sys
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def parse_args():
    parser = argparse.ArgumentParser(description="Plot ESR injection spectrogram")
    parser.add_argument("file",     help="Path to .npz file")
    parser.add_argument("freq_min", type=float, help="Lower frequency bound")
    parser.add_argument("freq_max", type=float, help="Upper frequency bound")
    parser.add_argument("--linear", action="store_true",
                        help="Plot linear power instead of dB (log scale)")
    parser.add_argument("--plot3d", action="store_true",
                        help="Interactive 3-D surface plot via plotly (opens in browser)")
    parser.add_argument("--time-avg", type=int, metavar="N", default=None,
                        help="Average every N consecutive time frames (reduces time axis by factor N)")
    return parser.parse_args()


def load(path):
    """Load and return freq_1d (F,), time_1d (T,), power_2d (T, F)."""
    data = np.load(path)
    freq  = data['freq'].ravel()   # (1, F) → (F,)
    time  = data['time'].ravel()   # (T, 1) → (T,)
    power = data['power']          # (T, F)
    data.close()

    # Normalise time to start at 0
    time = time - time[0]

    return freq, time, power


def slice_freq(freq, power, freq_min, freq_max):
    """Return (freq_slice, power_slice) restricted to [freq_min, freq_max]."""
    mask = (freq >= freq_min) & (freq <= freq_max)
    if not mask.any():
        raise ValueError(
            f"No frequency bins found in [{freq_min}, {freq_max}]. "
            f"File covers [{freq.min():.1f}, {freq.max():.1f}]."
        )
    return freq[mask], power[:, mask]


def to_db(power):
    """Convert power to dB, guarding against zeros/negatives."""
    eps = np.finfo(float).tiny
    return 10.0 * np.log10(np.maximum(power, eps))


# ---------------------------------------------------------------------------
# 2-D matplotlib plot (original)
# ---------------------------------------------------------------------------

def plot_2d(freq, time, power, title, log_scale=True):
    fig, ax = plt.subplots(figsize=(12, 5))

    z = to_db(power) if log_scale else power

    # pcolormesh expects edges; build them from centres
    def edges(arr):
        d = np.diff(arr)
        return np.concatenate([[arr[0] - d[0] / 2],
                                arr[:-1] + d / 2,
                                [arr[-1] + d[-1] / 2]])

    t_edges = edges(time)
    f_edges = edges(freq)

    mesh = ax.pcolormesh(
        t_edges, f_edges, z.T,          # z.T → (F, T) for (y, x) orientation
        cmap="viridis",
        shading="flat",
        rasterized=True,
    )

    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("Power (dB)" if log_scale else "Power (linear)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    ax.invert_yaxis()

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Frequency (Hz)", fontsize=11)
    ax.set_title(title, fontsize=12, pad=8)

    # Frequency axis: show full integers, no scientific notation
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.tick_params(labelsize=9)

    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Time rebinning
# ---------------------------------------------------------------------------

def rebin_time(time, power, n):
    """Average every N consecutive time frames.

    Trailing frames that don't fill a complete group are discarded.
    Averaging is done in linear power space.

    Returns (time_rebinned, power_rebinned) with shape (T//N, F).
    """
    T, F = power.shape
    T_trim = (T // n) * n
    power_r = power[:T_trim].reshape(T // n, n, F).mean(axis=1)
    time_r  = time[:T_trim].reshape(T // n, n).mean(axis=1)
    return time_r, power_r


# ---------------------------------------------------------------------------
# 3-D interactive plotly plot
# ---------------------------------------------------------------------------

def plot_3d(freq, time, power, title, log_scale=True):
    try:
        import plotly.graph_objects as go
    except ImportError:
        sys.exit(
            "plotly is required for --plot3d.  Install it with:\n"
            "  pip install plotly"
        )

    z = to_db(power) if log_scale else power
    z_label = "Power (dB)" if log_scale else "Power (linear)"

    # Downsample if very large to keep the browser responsive
    # Target ≤ 500 points per axis
    MAX_PTS = 500
    t_step = max(1, len(time) // MAX_PTS)
    f_step = max(1, len(freq) // MAX_PTS)
    t_ds   = time[::t_step]
    f_ds   = freq[::f_step]
    z_ds   = z[::t_step, ::f_step]

    if t_step > 1 or f_step > 1:
        print(
            f"Downsampled for 3-D plot: {len(t_ds)} time pts × {len(f_ds)} freq pts "
            f"(original: {len(time)} × {len(freq)})"
        )

    fig = go.Figure(data=[go.Surface(
        x=t_ds,                  # time along X
        y=f_ds[::-1],            # frequency along Y (high→low)
        z=z_ds[:, ::-1].T,       # flip freq axis, transpose to (F, T)
        colorscale="Inferno",
        colorbar=dict(
            title=dict(text=z_label, side="right"),
            thickness=18,
        ),
        hovertemplate=(
            "Time: %{x:.2f} s<br>"
            "Freq: %{y:,.0f} Hz<br>"
            + z_label + ": %{z:.2f}<extra></extra>"
        ),
    )])

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font_size=14),
        scene=dict(
            xaxis=dict(title="Time (s)"),
            yaxis=dict(title="Frequency (Hz)", tickformat=",d", autorange="reversed"),
            zaxis=dict(title=z_label),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.8)),
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        height=700,
    )

    fig.show()   # opens in the default browser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not os.path.isfile(args.file):
        sys.exit(f"Error: file not found — {args.file}")

    freq, time, power = load(args.file)
    freq_s, power_s = slice_freq(freq, power, args.freq_min, args.freq_max)

    scale_label = "linear" if args.linear else "dB"
    title = (
        f"{os.path.basename(args.file)}   "
        f"[{args.freq_min:,.0f} – {args.freq_max:,.0f} Hz]   "
        f"({scale_label})"
    )

    n_avg = args.time_avg  # None or an integer

    if n_avg is not None:
        if n_avg < 2:
            sys.exit("Error: --time-avg N requires N ≥ 2.")
        T = power_s.shape[0]
        time, power_s = rebin_time(time, power_s, n_avg)
        print(
            f"Time rebinned: {T} frames → {len(time)} frames "
            f"(averaged every {n_avg}, {T % n_avg} trailing frame(s) discarded)"
        )
        title = title.rstrip(")") + f", time-avg {n_avg})"

    print(f"Plotting {power_s.shape[1]} freq bins × {power_s.shape[0]} time bins")

    if args.plot3d:
        plot_3d(freq_s, time, power_s, title, log_scale=not args.linear)
    else:
        plot_2d(freq_s, time, power_s, title, log_scale=not args.linear)


if __name__ == "__main__":
    main()
