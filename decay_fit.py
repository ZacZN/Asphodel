#!/usr/bin/env python3
"""
Integrate power over a frequency range in each time frame of an ESR .npz file,
then fit and plot an exponential decay or Bateman two-member chain.

Usage
-----
  python decay_fit.py <file> <freq_min> <freq_max> [options]

Arguments
---------
  file              Path to a .npz file (freq, time, power arrays)
  freq_min          Lower frequency bound (same units as the freq array, Hz)
  freq_max          Upper frequency bound

Decay options (default mode)
  --no-offset       Fit A·exp(-t/τ) with no constant background floor

Bateman options (--bateman mode)
  --bateman         Fit Bateman two-member chain A→B→C.
                    Model: I(t) = P·(e^{-t/τ_A} − e^{-t/τ_B}) + B·e^{-t/τ_B} + C
                    B is fixed to the first time frame of the fit window.
                    Free parameters by default: P, τ_A, τ_B, C.
  --halflife-isomer T
                    Known half-life of the isomer in the c.o.m. frame (seconds).
                    Converted to lab frame via T_lab = T × γ, then fixes τ_A.
                    Requires --gamma.
  --halflife-groundstate T
                    Known half-life of the ground state in the c.o.m. frame (seconds).
                    Converted to lab frame via T_lab = T × γ, then fixes τ_B.
                    Requires --gamma.

Common options
  --tmin T          Only use time frames at or after T seconds
  --tmax T          Only use time frames at or before T seconds
  --gamma G         Lorentz factor γ; converts c.o.m. half-lives to lab frame,
                    and shows t½(c.o.m.) = t½(lab) / γ in the plot annotation.
  --title TEXT      Extra text appended to the auto-generated plot title
  --skip X Y        Skip the first X and last Y time bins

Example
-------
  # Simple decay
  python decay_fit.py 1081.5.npz 245000 246000 --no-offset --gamma 1.43

  # Bateman: fit P, τ_A, τ_B, C; B from first frame
  python decay_fit.py 1081.5.npz 245000 246000 --bateman --gamma 1.43

  # Bateman: both half-lives known in c.o.m. frame; fit only P and C
  python decay_fit.py 1081.5.npz 245000 246000 --bateman --gamma 1.43 \\
      --halflife-isomer 1.4 --halflife-groundstate 107.4
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
        description="Integrate ESR power over a frequency band and fit exponential decay or Bateman chain"
    )
    parser.add_argument("file",     help="Path to .npz file")
    parser.add_argument("freq_min", type=float, help="Lower frequency bound (Hz)")
    parser.add_argument("freq_max", type=float, help="Upper frequency bound (Hz)")
    parser.add_argument(
        "--no-offset", action="store_true",
        help="Decay mode: fit without constant background floor."
    )
    parser.add_argument(
        "--bateman", action="store_true",
        help="Fit Bateman two-member chain instead of simple decay."
    )
    parser.add_argument(
        "--halflife-isomer", type=float, default=None, metavar="T",
        help="(Bateman) Half-life of isomer in c.o.m. frame (s). Requires --gamma. Fixes τ_A."
    )
    parser.add_argument(
        "--halflife-groundstate", type=float, default=None, metavar="T",
        help="(Bateman) Half-life of ground state in c.o.m. frame (s). Requires --gamma. Fixes τ_B."
    )
    parser.add_argument("--tmin", type=float, default=None,
                        help="Start of time window to fit (s)")
    parser.add_argument("--tmax", type=float, default=None,
                        help="End of time window to fit (s)")
    parser.add_argument("--gamma", type=float, default=None,
                        help="Lorentz factor γ; converts c.o.m. half-lives to lab, "
                             "and shows t½(c.o.m.) in annotation.")
    parser.add_argument("--title", type=str, default=None,
                        help="Extra text appended to the auto-generated plot title")
    parser.add_argument("--skip", type=int, nargs=2, default=[0, 0], metavar=("X", "Y"),
                        help="Skip the first X and last Y time bins (applied before --tmin/--tmax)")
    parser.add_argument("--time-avg", type=int, metavar="N", default=None,
                        help="Average every N consecutive time frames before fitting")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load(path):
    """Return freq (F,), time (T,), power (T, F) from an ESR .npz file."""
    data  = np.load(path)
    freq  = data["freq"].ravel()
    time  = data["time"].ravel()
    power = data["power"]
    data.close()
    time = time - time[0]
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
    """Trapezoid-rule integral of linear power across the frequency band."""
    _trapz = getattr(np, "trapezoid", np.trapz)   # np.trapezoid added in NumPy 2.0
    return _trapz(power, x=freq, axis=1)


# ---------------------------------------------------------------------------
# Model functions
# ---------------------------------------------------------------------------

def model_with_offset(t, A, tau, C):
    return A * np.exp(-t / tau) + C

def model_no_offset(t, A, tau):
    return A * np.exp(-t / tau)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def _bateman_guesses(time, intensity, B_fixed):
    """Heuristic initial guesses for Bateman parameters from data shape."""
    peak_idx = np.argmax(intensity)
    t_peak   = time[peak_idx]
    span     = time[-1] - time[0]
    tau_A_0  = max(t_peak - time[0], span / 4)
    tau_B_0  = max(span - t_peak,    span / 8)
    P_0      = max(intensity[peak_idx] - B_fixed, 1e-10)
    return P_0, tau_A_0, tau_B_0


def fit_decay(time, intensity, with_offset=True, bateman=False,
              halflife_isomer_lab=None, halflife_groundstate_lab=None):
    """
    Fit exponential decay or Bateman chain.

    For Bateman:
      - B is always fixed to intensity[0] (first frame of the fit window).
      - C is always a free parameter (constant background offset).
      - halflife_isomer_lab    → fixes τ_A = T_lab/ln(2)
      - halflife_groundstate_lab → fixes τ_B = T_lab/ln(2)
      - Remaining free parameters: P, C, and any τ not fixed.

    Returns
    -------
    popt     : fitted parameter array (P first, C last; τ values in between)
    perr     : 1-σ uncertainties on popt
    model    : callable, model(t, *popt)
    resolved : dict  {'P','tau_A','tau_B','B','C'} → (value, err, label)
               label is 'fitted' or 'fixed'. None for simple decay.
    """
    if not bateman:
        # ---- Simple exponential decay ----
        # Use data range for A0 so the guess is always positive (avoids bound violation)
        C0   = np.min(intensity)
        A0   = max(np.max(intensity) - (C0 if with_offset else 0), 1e-30)
        tau0 = (time[-1] - time[0]) / 3.0
        try:
            if with_offset:
                popt, pcov = curve_fit(
                    model_with_offset, time, intensity,
                    p0=[A0, tau0, C0],
                    bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf]),
                    maxfev=10_000,
                )
                model = model_with_offset
            else:
                popt, pcov = curve_fit(
                    model_no_offset, time, intensity,
                    p0=[A0, tau0],
                    bounds=([0, 0], [np.inf, np.inf]),
                    maxfev=10_000,
                )
                model = model_no_offset
        except RuntimeError as exc:
            sys.exit(f"Curve fit failed: {exc}")
        perr = np.sqrt(np.diag(pcov))
        return popt, perr, model, None

    # ---- Bateman chain ----
    # B is fixed to the first frame of the fit window
    B_fixed = intensity[0]

    # Convert known lab half-lives to lifetimes
    tau_A_fixed = halflife_isomer_lab      / np.log(2) if halflife_isomer_lab      is not None else None
    tau_B_fixed = halflife_groundstate_lab / np.log(2) if halflife_groundstate_lab is not None else None

    A_known = tau_A_fixed is not None
    B_known = tau_B_fixed is not None

    # Heuristic initial guesses
    P_0, tau_A_0, tau_B_0 = _bateman_guesses(time, intensity, B_fixed)
    if tau_A_fixed is not None:
        tau_A_0 = tau_A_fixed
    if tau_B_fixed is not None:
        tau_B_0 = tau_B_fixed
    C_0 = 0.0

    # Inner model — B_fixed is captured from enclosing scope.
    # Parameter order always: P first, C last, any free τ in between.
    def _bateman(t, P, tau_A, tau_B, C):
        return (P * (np.exp(-t / tau_A) - np.exp(-t / tau_B))
                + B_fixed * np.exp(-t / tau_B)
                + C)

    try:
        if not A_known and not B_known:
            # Free: P, τ_A, τ_B, C
            fn = lambda t, P, tau_A, tau_B, C: _bateman(t, P, tau_A, tau_B, C)
            popt, pcov = curve_fit(
                fn, time, intensity,
                p0=[P_0, tau_A_0, tau_B_0, C_0],
                bounds=([-np.inf, 0, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf]),
                maxfev=20_000,
            )
            perr = np.sqrt(np.diag(pcov))
            P, tau_A, tau_B, C      = popt
            dP, dtau_A, dtau_B, dC  = perr

        elif A_known and not B_known:
            # Free: P, τ_B, C  (τ_A fixed)
            fn = lambda t, P, tau_B, C: _bateman(t, P, tau_A_fixed, tau_B, C)
            popt, pcov = curve_fit(
                fn, time, intensity,
                p0=[P_0, tau_B_0, C_0],
                bounds=([-np.inf, 0, -np.inf], [np.inf, np.inf, np.inf]),
                maxfev=20_000,
            )
            perr = np.sqrt(np.diag(pcov))
            P, tau_B, C        = popt
            dP, dtau_B, dC     = perr
            tau_A, dtau_A      = tau_A_fixed, 0.0

        elif not A_known and B_known:
            # Free: P, τ_A, C  (τ_B fixed)
            fn = lambda t, P, tau_A, C: _bateman(t, P, tau_A, tau_B_fixed, C)
            popt, pcov = curve_fit(
                fn, time, intensity,
                p0=[P_0, tau_A_0, C_0],
                bounds=([-np.inf, 0, -np.inf], [np.inf, np.inf, np.inf]),
                maxfev=20_000,
            )
            perr = np.sqrt(np.diag(pcov))
            P, tau_A, C        = popt
            dP, dtau_A, dC     = perr
            tau_B, dtau_B      = tau_B_fixed, 0.0

        else:
            # Free: P, C  (both τ fixed)
            fn = lambda t, P, C: _bateman(t, P, tau_A_fixed, tau_B_fixed, C)
            popt, pcov = curve_fit(
                fn, time, intensity,
                p0=[P_0, C_0],
                bounds=([-np.inf, -np.inf], [np.inf, np.inf]),
                maxfev=20_000,
            )
            perr = np.sqrt(np.diag(pcov))
            P, C               = popt
            dP, dC             = perr
            tau_A, dtau_A      = tau_A_fixed, 0.0
            tau_B, dtau_B      = tau_B_fixed, 0.0

    except RuntimeError as exc:
        sys.exit(f"Curve fit failed: {exc}")

    resolved = {
        'P':     (P,       dP,     'fitted'),
        'tau_A': (tau_A,   dtau_A, 'fixed'  if A_known else 'fitted'),
        'tau_B': (tau_B,   dtau_B, 'fixed'  if B_known else 'fitted'),
        'B':     (B_fixed, 0.0,    'fixed'),
        'C':     (C,       dC,     'fitted'),
    }

    # Return a closure that plot_decay can always call as model(t, *popt).
    # P is always popt[0], C is always popt[-1]; τ values are baked in.
    _tau_A_f, _tau_B_f = tau_A, tau_B
    model = lambda t, *args: _bateman(t, args[0], _tau_A_f, _tau_B_f, args[-1])

    return popt, perr, model, resolved


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_value(v, err=None, unit=""):
    prefixes = [
        (1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k"),
        (1,    ""), (1e-3, "m"), (1e-6, "µ"), (1e-9, "n"),
    ]
    for scale, prefix in prefixes:
        if abs(v) >= scale * 0.9:
            val_str = f"{v / scale:.4g}"
            if err is not None and err > 0:
                val_str += f" ± {err / scale:.2g}"
            return f"{val_str} {prefix}{unit}".strip()
    return f"{v:.4g}{(' ± ' + f'{err:.2g}') if (err is not None and err > 0) else ''} {unit}".strip()


def format_time(v, err=None):
    if abs(v) >= 60:
        scale, unit = 60.0, "min"
    else:
        scale, unit = 1.0, "s"
    val_str = f"{v / scale:.4g}"
    if err is not None and err > 0:
        val_str += f" ± {err / scale:.2g}"
    return f"{val_str} {unit}"


def _label_suffix(label):
    """Return a short annotation tag for fixed parameters."""
    if label == 'fixed':
        return "  [fixed]"
    return ""


# ---------------------------------------------------------------------------
# Annotation text
# ---------------------------------------------------------------------------

def make_equation_text_decay(popt, perr, with_offset, half_life, half_life_err, gamma=None):
    A, tau   = popt[0], popt[1]
    dA, dtau = perr[0], perr[1]
    lines = [
        r"$I(t) = A \cdot e^{-t/\tau}$" + ("$ + C$" if with_offset else ""),
        f"$A$   = {format_value(A, dA)}",
        f"$\\tau$ = {format_time(tau, dtau)}",
    ]
    if with_offset:
        C, dC = popt[2], perr[2]
        lines.append(f"$C$   = {format_value(C, dC)}")
    lines.append("")
    lines.append(f"$t_{{1/2}}$ (lab)   = {format_time(half_life, half_life_err)}")
    if gamma is not None:
        lines.append(f"$t_{{1/2}}$ (c.o.m.) = {format_time(half_life / gamma, half_life_err / gamma)}")
    return "\n".join(lines)


def make_equation_text_bateman(resolved, gamma=None):
    P,  dP,  lP  = resolved['P']
    tA, dtA, ltA = resolved['tau_A']
    tB, dtB, ltB = resolved['tau_B']
    B,  dB,  lB  = resolved['B']
    C,  dC,  lC  = resolved['C']

    hl_A  = tA * np.log(2);  dhl_A = dtA * np.log(2)
    hl_B  = tB * np.log(2);  dhl_B = dtB * np.log(2)

    lines = [
        r"$I(t) = P \cdot (e^{-t/\tau_A} - e^{-t/\tau_B}) + B \cdot e^{-t/\tau_B} + C$",
        f"$P$      = {format_value(P, dP if lP == 'fitted' else None)}{_label_suffix(lP)}",
        f"$\\tau_A$ = {format_time(tA, dtA if ltA == 'fitted' else None)}   (isomer){_label_suffix(ltA)}",
        f"$\\tau_B$ = {format_time(tB, dtB if ltB == 'fitted' else None)}   (g.s.){_label_suffix(ltB)}",
        f"$B$      = {format_value(B)}{_label_suffix(lB)}",
        f"$C$      = {format_value(C, dC)}",
        "",
        f"$t_{{1/2,A}}$ (lab)   = {format_time(hl_A, dhl_A if ltA == 'fitted' else None)}",
    ]
    if gamma is not None:
        lines.append(f"$t_{{1/2,A}}$ (c.o.m.) = {format_time(hl_A / gamma, (dhl_A / gamma) if ltA == 'fitted' else None)}")
    lines.append(f"$t_{{1/2,B}}$ (lab)   = {format_time(hl_B, dhl_B if ltB == 'fitted' else None)}")
    if gamma is not None:
        lines.append(f"$t_{{1/2,B}}$ (c.o.m.) = {format_time(hl_B / gamma, (dhl_B / gamma) if ltB == 'fitted' else None)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_decay(time, intensity, time_fit, popt, perr, model, resolved,
               with_offset, bateman, freq_min, freq_max, filename,
               gamma=None, title_suffix=None):

    plt.rcParams.update({"font.size": 20})

    t_dense = np.linspace(time_fit[0], time_fit[-1], 2000)
    I_fit   = model(t_dense, *popt)

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(time, intensity, "o", ms=3, alpha=0.6,
            color="#4c72b0", label="Data (integrated power)")
    ax.plot(t_dense, I_fit, "-", lw=2, color="#dd4444",
            label="Bateman fit" if bateman else "Exponential fit")

    # Annotation box
    if bateman:
        eq_text = make_equation_text_bateman(resolved, gamma)
    else:
        tau          = popt[1];  dtau         = perr[1]
        half_life    = tau * np.log(2);  half_life_err = dtau * np.log(2)
        eq_text = make_equation_text_decay(popt, perr, with_offset,
                                           half_life, half_life_err, gamma)

    ax.text(
        0.97, 0.97, eq_text,
        transform=ax.transAxes,
        va="top", ha="right",
        fontsize=16,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor="0.7", alpha=0.9),
    )

    # Half-life marker — τ_A for Bateman, τ for decay
    if bateman:
        tau_m  = resolved['tau_A'][0]
        marker_label = r"$t_{1/2,A}$"
    else:
        tau_m  = popt[1]
        marker_label = r"$t_{1/2}$"

    hl_m = tau_m * np.log(2)
    t_half = time_fit[0] + hl_m
    if time_fit[0] <= t_half <= time_fit[-1]:
        ax.axvline(t_half, ls="--", lw=1, color="0.5",
                   label=f"{marker_label} (lab) = {format_time(hl_m)}")

    mode_str = "Bateman chain" if bateman else "exponential decay"
    title = (
        f"{os.path.basename(filename)}   "
        f"[{freq_min:,.0f} – {freq_max:,.0f} Hz]   {mode_str}"
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
    fig.subplots_adjust(right=0.98)
    plt.show()

    # Console summary
    print(f"\n{'─'*55}")
    if bateman:
        P,  dP,  lP  = resolved['P']
        tA, dtA, ltA = resolved['tau_A']
        tB, dtB, ltB = resolved['tau_B']
        B,  dB,  lB  = resolved['B']
        C,  dC,  lC  = resolved['C']
        hl_A = tA * np.log(2);  dhl_A = dtA * np.log(2)
        hl_B = tB * np.log(2);  dhl_B = dtB * np.log(2)
        print(f"  Fit:  I(t) = P·(e^(-t/τ_A) - e^(-t/τ_B)) + B·e^(-t/τ_B) + C")
        print(f"  P      = {format_value(P, dP if lP == 'fitted' else None)}  ({lP})")
        print(f"  τ_A    = {format_time(tA, dtA if ltA == 'fitted' else None)}  (isomer, {ltA})")
        print(f"  τ_B    = {format_time(tB, dtB if ltB == 'fitted' else None)}  (ground state, {ltB})")
        print(f"  B      = {format_value(B)}  ({lB}, from first frame)")
        print(f"  C      = {format_value(C, dC)}  ({lC})")
        print(f"  t½_A (lab)   = {format_time(hl_A, dhl_A if ltA == 'fitted' else None)}")
        if gamma is not None:
            print(f"  γ            = {gamma}")
            print(f"  t½_A (c.o.m.)= {format_time(hl_A / gamma, (dhl_A / gamma) if ltA == 'fitted' else None)}")
        print(f"  t½_B (lab)   = {format_time(hl_B, dhl_B if ltB == 'fitted' else None)}")
        if gamma is not None:
            print(f"  t½_B (c.o.m.)= {format_time(hl_B / gamma, (dhl_B / gamma) if ltB == 'fitted' else None)}")
    else:
        A, tau   = popt[0], popt[1]
        dA, dtau = perr[0], perr[1]
        hl = tau * np.log(2);  dhl = dtau * np.log(2)
        print(f"  Fit:  I(t) = A·exp(-t/τ)" + (" + C" if with_offset else ""))
        print(f"  A    = {format_value(A, dA)}")
        print(f"  τ    = {format_time(tau, dtau)}")
        if with_offset:
            C, dC = popt[2], perr[2]
            print(f"  C    = {format_value(C, dC)}")
        print(f"  t½ (lab)   = {format_time(hl, dhl)}")
        if gamma is not None:
            print(f"  γ          = {gamma}")
            print(f"  t½ (c.o.m.)= {format_time(hl / gamma, dhl / gamma)}")
    print(f"{'─'*55}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not os.path.isfile(args.file):
        sys.exit(f"Error: file not found — {args.file}")

    if (args.halflife_isomer is not None or args.halflife_groundstate is not None):
        if not args.bateman:
            print("Warning: --halflife-isomer/--halflife-groundstate are only used with --bateman; ignoring.")
            halflife_isomer_lab      = None
            halflife_groundstate_lab = None
        elif args.gamma is None:
            sys.exit(
                "Error: --gamma is required when supplying half-lives "
                "(values are in the c.o.m. frame and must be converted to lab)."
            )
        else:
            halflife_isomer_lab = (
                args.halflife_isomer * args.gamma if args.halflife_isomer is not None else None
            )
            halflife_groundstate_lab = (
                args.halflife_groundstate * args.gamma if args.halflife_groundstate is not None else None
            )
    else:
        halflife_isomer_lab      = None
        halflife_groundstate_lab = None

    freq, time, power = load(args.file)
    freq_s, power_s   = slice_freq(freq, power, args.freq_min, args.freq_max)

    print(
        f"Loaded {power_s.shape[1]} freq bins × {power_s.shape[0]} time frames  "
        f"(Δf ≈ {np.diff(freq_s).mean():.1f} Hz per bin)"
    )

    if args.time_avg is not None and args.time_avg > 1:
        n = args.time_avg
        T = power_s.shape[0]
        T_trim = (T // n) * n
        if T_trim == 0:
            sys.exit(f"--time-avg {n}: not enough time frames ({T}) to form a single group.")
        discarded = T - T_trim
        power_s = power_s[:T_trim].reshape(T // n, n, power_s.shape[1]).mean(axis=1)
        time    = time[:T_trim].reshape(T // n, n).mean(axis=1)
        print(f"Time rebinned: {T} frames → {len(time)} "
              f"(averaged {n}" + (f", {discarded} discarded" if discarded else "") + ")")

    intensity = integrate_band(freq_s, power_s)

    skip_start, skip_end = args.skip
    if skip_start + skip_end >= len(time):
        sys.exit(f"--skip {skip_start} {skip_end} would remove all {len(time)} time bins.")
    if skip_start > 0:
        print(f"Skipping first {skip_start} time bins (up to t={time[skip_start-1]:.3f} s)")
    if skip_end > 0:
        print(f"Skipping last {skip_end} time bins (from t={time[len(time)-skip_end]:.3f} s)")
    end_idx   = len(time) - skip_end if skip_end > 0 else len(time)
    time      = time[skip_start:end_idx]
    intensity = intensity[skip_start:end_idx]

    t_mask = np.ones(len(time), dtype=bool)
    if args.tmin is not None:
        t_mask &= time >= args.tmin
    if args.tmax is not None:
        t_mask &= time <= args.tmax

    time_fit      = time[t_mask]
    intensity_fit = intensity[t_mask]

    if args.bateman:
        n_fixed_tau = (1 if halflife_isomer_lab      is not None else 0) + \
                      (1 if halflife_groundstate_lab is not None else 0)
        n_free  = 4 - n_fixed_tau   # P, τ_A, τ_B, C minus fixed τ
        min_pts = n_free + 1
    else:
        min_pts = 3 if args.no_offset else 4

    if len(time_fit) < min_pts:
        sys.exit(
            f"Only {len(time_fit)} points in the requested time window — "
            f"too few to fit (need ≥ {min_pts}).  Adjust --tmin / --tmax."
        )

    print(f"Fitting {len(time_fit)} points from t={time_fit[0]:.2f} s "
          f"to t={time_fit[-1]:.2f} s")

    if args.bateman:
        print(f"Bateman: B fixed to first frame = {format_value(intensity_fit[0])}; "
              f"C free  ({n_free} free parameter{'s' if n_free != 1 else ''})")
        if halflife_isomer_lab is not None:
            print(f"  τ_A fixed: t½ = {args.halflife_isomer:.4g} s (c.o.m.) "
                  f"→ {halflife_isomer_lab:.4g} s (lab)")
        if halflife_groundstate_lab is not None:
            print(f"  τ_B fixed: t½ = {args.halflife_groundstate:.4g} s (c.o.m.) "
                  f"→ {halflife_groundstate_lab:.4g} s (lab)")

    with_offset = not args.no_offset
    popt, perr, model, resolved = fit_decay(
        time_fit, intensity_fit,
        with_offset              = with_offset,
        bateman                  = args.bateman,
        halflife_isomer_lab      = halflife_isomer_lab,
        halflife_groundstate_lab = halflife_groundstate_lab,
    )

    plot_decay(
        time, intensity,
        time_fit, popt, perr, model, resolved,
        with_offset, args.bateman,
        args.freq_min, args.freq_max,
        args.file,
        gamma=args.gamma,
        title_suffix=args.title,
    )


if __name__ == "__main__":
    main()
