#!/usr/bin/env python3
"""
Integrate power over a frequency range in each time frame of an ESR .npz file,
then fit and plot an exponential decay.

Usage
-----
  python decay_fit.py <file> <freq_min> <freq_max> [--no-offset] [--tmin T] [--tmax T] [--gamma G]

Arguments
---------
  file         Path to a .npz file (freq, time, power arrays)
  freq_min     Lower frequency bound (same units as the freq array, Hz)
  freq_max     Upper frequency bound
  --no-offset  Force the decay to zero (fit A·exp(-t/τ) with no floor)
  --tmin T     Only use time frames at or after T seconds (default: 0)
  --tmax T     Only use time frames at or before T seconds (default: end)
  --gamma G    Lorentz factor γ; if given, also shows the centre-of-mass
               half-life t½(c.o.m.) = t½(lab) / γ
  --title TEXT Extra text appended to the auto-generated plot title
  --skip X Y   Skip the first X and last Y time bins

Example
-------
  python decay_fit.py 1081.5.npz 245000 246000
  python decay_fit.py 1081.5.npz 245000 246000 --gamma 1.43
  python decay_fit.py 1081.5.npz 245000 246000 --no-offset --tmax 60 --gamma 1.43
  python decay_fit.py 1081.5.npz 245000 246000 --title "Cd-102, injection 3"
"""

import sys
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Integrate ESR power over a frequency band and fit exponential decay"
    )
    parser.add_argument("file",     help="Path to .npz file")
    parser.add_argument("freq_min", type=float, help="Lower frequency bound (Hz)")
    parser.add_argument("freq_max", type=float, help="Upper frequency bound (Hz)")
    parser.add_argument(
        "--no-offset", action="store_true",
        help="Fit A·exp(-t/τ) without a constant background offset"
    )
    parser.add_argument("--tmin", type=float, default=None,
                        help="Start of time window to fit (s)")
    parser.add_argument("--tmax", type=float, default=None,
                        help="End of time window to fit (s)")
    parser.add_argument("--gamma", type=float, default=None,
                        help="Lorentz factor γ; used to compute t½(c.o.m.) = t½(lab) / γ")
    parser.add_argument("--title", type=str, default=None,
                        help="Extra text appended to the auto-generated plot title")
    parser.add_argument("--skip", type=int, nargs=2, default=[0, 0], metavar=("X", "Y"),
                        help="Skip the first X and last Y time bins (applied before --tmin/--tmax)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading (mirrors plot_spectrogram.py)
# ---------------------------------------------------------------------------

def load(path):
    """Return freq (F,), time (T,), power (T, F) from an ESR .npz file."""
    data  = np.load(path)
    freq  = data["freq"].ravel()   # (1, F) → (F,)
    time  = data["time"].ravel()   # (T, 1) → (T,)
    power = data["power"]          # (T, F)
    data.close()

    time = time - time[0]          # normalise to start at 0
    return freq, time, power


def slice_freq(freq, power, freq_min, freq_max):
    """Restrict arrays to [freq_min, freq_max]."""
    mask = (freq >= freq_min) & (freq <= freq_max)
    if not mask.any():
        raise ValueError(
            f"No frequency bins found in [{freq_min:.1f}, {freq_max:.1f}] Hz. "
            f"File covers [{freq.min():.1f}, {freq.max():.1f}] Hz."
        )
    return freq[mask], power[:, mask]


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

def integrate_band(freq, power):
    """
    Sum (integrate) linear power across the frequency axis for every time frame.

    Uses the trapezoid rule so the result is independent of bin width.

    Parameters
    ----------
    freq  : (F,)  frequency values in Hz
    power : (T, F) linear power

    Returns
    -------
    integrated : (T,) total power in the band per time frame
    """
    return np.trapz(power, x=freq, axis=1)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def model_with_offset(t, A, tau, C):
    """Exponential decay with a constant background: A·exp(-t/τ) + C."""
    return A * np.exp(-t / tau) + C


def model_no_offset(t, A, tau):
    """Pure exponential decay: A·exp(-t/τ)."""
    return A * np.exp(-t / tau)


def fit_decay(time, intensity, with_offset=True):
    """
    Fit exponential decay to (time, intensity) data.

    Returns
    -------
    popt  : optimal parameters  (A, tau[, C])
    perr  : 1-σ uncertainties   (same shape)
    model : callable model function
    """
    # Initial guesses
    A0   = intensity[0] - (intensity[-1] if with_offset else 0)
    tau0 = (time[-1] - time[0]) / 3.0   # rough 1/3 of the time range
    C0   = intensity[-1]

    try:
        if with_offset:
            popt, pcov = curve_fit(
                model_with_offset, time, intensity,
                p0=[A0, tau0, C0],
                bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf]),
                maxfev=10_000,
            )
        else:
            popt, pcov = curve_fit(
                model_no_offset, time, intensity,
                p0=[A0, tau0],
                bounds=([0, 0], [np.inf, np.inf]),
                maxfev=10_000,
            )
        perr = np.sqrt(np.diag(pcov))
    except RuntimeError as exc:
        sys.exit(f"Curve fit failed: {exc}")

    model = model_with_offset if with_offset else model_no_offset
    return popt, perr, model


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def format_value(v, err=None, unit=""):
    """Format a value (± error) with a suitable SI prefix."""
    prefixes = [
        (1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k"),
        (1,    ""), (1e-3, "m"), (1e-6, "µ"), (1e-9, "n"),
    ]
    for scale, prefix in prefixes:
        if abs(v) >= scale * 0.9:
            val_str = f"{v / scale:.4g}"
            if err is not None:
                val_str += f" ± {err / scale:.2g}"
            return f"{val_str} {prefix}{unit}".strip()
    return f"{v:.4g}{(' ± ' + f'{err:.2g}') if err else ''} {unit}".strip()


def format_time(v, err=None):
    """Format a time value in seconds, switching to minutes if >= 60 s."""
    if abs(v) >= 60:
        scale, unit = 60.0, "min"
    else:
        scale, unit = 1.0, "s"
    val_str = f"{v / scale:.4g}"
    if err is not None:
        val_str += f" ± {err / scale:.2g}"
    return f"{val_str} {unit}"


def make_equation_text(popt, perr, with_offset, half_life, half_life_err, gamma=None):
    """Build the annotation string shown on the plot."""
    A, tau = popt[0], popt[1]
    dA, dtau = perr[0], perr[1]

    lines = [
        r"$I(t) = A \cdot e^{-t/\tau}$" + ("$ + C$" if with_offset else ""),
        f"$A$   = {format_value(A, dA)}",
        f"$\\tau$ = {format_time(tau, dtau)}",
    ]
    if with_offset:
        C, dC = popt[2], perr[2]
        lines.append(f"$C$   = {format_value(C, dC)}")

    lines.append("")   # blank separator
    lines.append(
        f"$t_{{1/2}}$ (lab)   = {format_time(half_life, half_life_err)}"
    )
    if gamma is not None:
        com_half_life     = half_life     / gamma
        com_half_life_err = half_life_err / gamma
        lines.append(
            f"$t_{{1/2}}$ (c.o.m.) = {format_time(com_half_life, com_half_life_err)}"
        )
    return "\n".join(lines)


def plot_decay(time, intensity, time_fit, popt, perr, model,
               with_offset, freq_min, freq_max, filename, gamma=None, title_suffix=None):
    """Draw the data, the fit, and the annotation box."""

    # Double the default matplotlib font size (default is 10 → 20)
    plt.rcParams.update({"font.size": 20})

    A, tau = popt[0], popt[1]
    dA, dtau = perr[0], perr[1]

    half_life     = tau  * np.log(2)
    half_life_err = dtau * np.log(2)

    com_half_life     = half_life     / gamma if gamma is not None else None
    com_half_life_err = half_life_err / gamma if gamma is not None else None

    t_dense = np.linspace(time_fit[0], time_fit[-1], 2000)
    I_fit   = model(t_dense, *popt)

    fig, ax = plt.subplots(figsize=(16, 6))

    # Data
    ax.plot(time, intensity, "o", ms=3, alpha=0.6,
            color="#4c72b0", label="Data (integrated power)")

    # Fit — only over the fitted window, but highlight it clearly
    ax.plot(t_dense, I_fit, "-", lw=2,
            color="#dd4444", label="Exponential fit")

    # --- Annotation box ---
    eq_text = make_equation_text(popt, perr, with_offset, half_life, half_life_err, gamma)
    ax.text(
        0.97, 0.97, eq_text,
        transform=ax.transAxes,
        va="top", ha="right",
        fontsize=18,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor="0.7", alpha=0.9),
    )

    # Half-life marker (vertical dashed line at t½(lab) from start of fit window)
    t_half = time_fit[0] + half_life
    if time_fit[0] <= t_half <= time_fit[-1]:
        ax.axvline(t_half, ls="--", lw=1, color="0.5",
                   label=f"$t_{{1/2}}$ (lab) = {format_time(half_life)}")

    title = (
        f"{os.path.basename(filename)}   "
        f"[{freq_min:,.0f} – {freq_max:,.0f} Hz]   "
        f"exponential decay"
    )
    if title_suffix:
        title += f"   {title_suffix}"

    ax.set_xlabel("Time (s)", fontsize=22)
    ax.set_ylabel("Integrated power (arb. units)", fontsize=22)
    ax.set_title(title, fontsize=22, pad=8)
    ax.tick_params(labelsize=20)
    ax.legend(fontsize=18)
    ax.set_xlim(left=0)
    fig.tight_layout(pad=0)
    fig.subplots_adjust(right=0.98)   # small right margin so legend/box aren't clipped
    plt.show()

    # Console summary
    print(f"\n{'─'*50}")
    print(f"  Fit:  I(t) = A·exp(-t/τ)" + (" + C" if with_offset else ""))
    print(f"  A    = {format_value(A, dA)}")
    print(f"  τ    = {format_time(tau, dtau)}")
    if with_offset:
        C, dC = popt[2], perr[2]
        print(f"  C    = {format_value(C, dC)}")
    print(f"  t½ (lab)   = {format_time(half_life, half_life_err)}")
    if gamma is not None:
        print(f"  γ          = {gamma}")
        print(f"  t½ (c.o.m.)= {format_time(com_half_life, com_half_life_err)}")
    print(f"{'─'*50}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not os.path.isfile(args.file):
        sys.exit(f"Error: file not found — {args.file}")

    freq, time, power = load(args.file)
    freq_s, power_s   = slice_freq(freq, power, args.freq_min, args.freq_max)

    print(
        f"Loaded {power_s.shape[1]} freq bins × {power_s.shape[0]} time frames  "
        f"(Δf ≈ {np.diff(freq_s).mean():.1f} Hz per bin)"
    )

    # Integrate power over the frequency band for each time frame
    intensity = integrate_band(freq_s, power_s)

    # Skip bins from start and/or end before anything else
    skip_start, skip_end = args.skip
    if skip_start + skip_end >= len(time):
        sys.exit(
            f"--skip {skip_start} {skip_end} would remove all {len(time)} time bins."
        )
    if skip_start > 0:
        print(f"Skipping first {skip_start} time bins (up to t={time[skip_start - 1]:.3f} s)")
    if skip_end > 0:
        print(f"Skipping last {skip_end} time bins (from t={time[len(time) - skip_end]:.3f} s)")
    end_idx   = len(time) - skip_end if skip_end > 0 else len(time)
    time      = time[skip_start:end_idx]
    intensity = intensity[skip_start:end_idx]

    # Optional time window for the fit
    t_mask = np.ones(len(time), dtype=bool)
    if args.tmin is not None:
        t_mask &= time >= args.tmin
    if args.tmax is not None:
        t_mask &= time <= args.tmax

    time_fit      = time[t_mask]
    intensity_fit = intensity[t_mask]

    if len(time_fit) < 4:
        sys.exit(
            f"Only {len(time_fit)} points in the requested time window — "
            "too few to fit.  Adjust --tmin / --tmax."
        )

    print(f"Fitting {len(time_fit)} points from t={time_fit[0]:.2f} s "
          f"to t={time_fit[-1]:.2f} s")

    with_offset = not args.no_offset
    popt, perr, model = fit_decay(time_fit, intensity_fit, with_offset)

    plot_decay(
        time, intensity,
        time_fit, popt, perr, model,
        with_offset,
        args.freq_min, args.freq_max,
        args.file,
        gamma=args.gamma,
        title_suffix=args.title,
    )


if __name__ == "__main__":
    main()
