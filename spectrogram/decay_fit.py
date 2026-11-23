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

Joint parent + daughter mode (--parent-window)
  --parent-window FMIN FMAX
                    Frequency window of the parent peak; freq_min/freq_max are
                    the daughter peak. Both windows are fitted simultaneously:
                      parent   I_P(t) = A·e^{-t/τ_A} + C_A          (offset)
                      daughter I_D(t) = P·(e^{-t/τ_A} − e^{-t/τ_B}) + B·e^{-t/τ_B}
                                                                   (no offset)
                    with τ_A shared. Implies --bateman; the fixed half-life
                    options still apply. Error bars are estimated per window.

Common options
  --tmin T          Only use time frames at or after T seconds
  --tmax T          Only use time frames at or before T seconds
  --gamma G         Lorentz factor γ; converts c.o.m. half-lives to lab frame,
                    and shows t½(c.o.m.) = t½(lab) / γ in the plot annotation.
  --title TEXT      Extra text appended to the auto-generated plot title
  --title-full TEXT Use TEXT as the whole plot title instead
  --log-plot        Add a semi-log panel (data − C, fit − C) to the right
  --residuals       Add residual panels underneath ((data − fit)/σ)
  --skip X Y        Skip the first X and last Y time bins
  --err-coeffs A B  Manual per-point error bars σ = A·I + B (single-frame
                    calibration). With --time-avg N, σ is divided by √N.
                    Without this option the error model is estimated from the
                    data (two-pass fit, see below).
  --unweighted      Skip error bars entirely and do a single unweighted fit.

Error bars (default: automatic two-pass fit)
  1. Unweighted fit → model f(t).
  2. Residuals r_i = I_i − f(t_i). Fit |r_i|·√(π/2)·√(n/(n−p)) = a·f(t_i) + b
     (√(π/2) converts mean |r| to a Gaussian σ; √(n/(n−p)) corrects for the
     p fitted parameters). If that gives σ ≤ 0 anywhere, a constant σ is used.
  3. Refit with σ_i = a·f(t_i) + b, absolute_sigma=True; parameter and
     half-life errors come from this weighted fit.
  If the weighted refit (3) fails, the unweighted fit from (1) is shown
  instead, with a warning in the plot and "unweighted (weighted refit
  failed)" as the saved error model.

Fit failures
------------
  fit_decay and fit_bateman_joint raise FitError when the fit does not
  converge; run as a script, the message is printed and the exit code is 1
  (the spectrogram page shows it).

Saving a fit
------------
  The plot window has a "Save fit parameters" button (top left). It writes
  one JSON record per fit (results only - use the window's own save icon
  for the figure) to data/<experiment>/outputs/fits/ (see core/paths.py):
  fit_<YYYYmmdd_HHMMSS>_<file>[_<title>].json with the input file, the
  windows and options used, every parameter (value, error, fitted/fixed),
  the half-lives (lab and c.o.m.), chi^2/ndf and the error model.

  "Save to log book" (next to it) opens a popout to send the half-lives and
  the figure to the experiment's results log book (one tab per ion; with a
  Bateman / joint fit the parent and daughter half-lives each go to their own
  ion): core/logbook_dialog.py, core/results_logbook.py.

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
import datetime
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.optimize import curve_fit

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import baseline, logcapture, logpanel, paths

# output also to data/logs/ and this tool's Output window (core/logcapture.py)
if __name__ == "__main__":
    logcapture.start("decay_fit")

# Set by plot_spectrogram.py when it starts a fit: this file is created once
# the fit window has been drawn, so the page can tell "window open" from a
# fit that crashed or never showed a window.
READY_FILE = os.environ.get("ASPHODEL_READY_FILE")
_NON_GUI_BACKENDS = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}


def show_window():
    """plt.show(), reporting to the spectrogram page once the window is drawn.
    Every window gets an "Output" button (what this fit printed: core/logpanel.py),
    next to "Save fit parameters" where there is one."""
    logpanel.add_mpl_buttons(lambda f: [0.30, 0.945, 0.07, 0.045]
                             if hasattr(f, "_logbook_button") else
                             [0.16, 0.945, 0.07, 0.045]
                             if hasattr(f, "_save_fit_button") else [0.005, 0.945, 0.07, 0.045],
                             "Decay fit")
    if READY_FILE:
        backend = plt.get_backend().lower()
        if backend in _NON_GUI_BACKENDS:
            sys.exit(f"Error: matplotlib has no GUI backend (it is using '{backend}'), "
                     "so the fit window cannot open.")
        cids = []

        def _drawn(_evt):
            try:
                open(READY_FILE, "w").close()
            except OSError:
                pass
            for f, c in cids:
                f.canvas.mpl_disconnect(c)

        for n in plt.get_fignums():
            f = plt.figure(n)
            cids.append((f, f.canvas.mpl_connect("draw_event", _drawn)))
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Title formatting: ion names and harmonic labels
# ---------------------------------------------------------------------------
import re as _re

# NUBASE-style ion name: optional isomer letter, mass number, element, charge,
# e.g. m207Tl80+ -> ^{207m}Tl^{80+};  harmonic label h213 -> "h = 213"
_ION_RE  = _re.compile(r"(?<![\w$])([mnpqrx]?)(\d{1,3})([A-Z][a-z]?)(\d{1,3})\+")
_HARM_RE = _re.compile(r"(?<!\S)h(\d+)(?!\S)")
# "m207Tl80+ h213" -> "m207Tl80+, h213" (adds the comma if it is missing)
_ION_HARM_RE = _re.compile(r"((?<![\w$])[mnpqrx]?\d{1,3}[A-Z][a-z]?\d{1,3}\+)(\s+)(h\d+)(?!\S)")


def format_title_mathtext(text):
    """Plain title text -> matplotlib mathtext with formatted ions/harmonics."""
    if not text:
        return text
    text = text.replace("$", r"\$")            # literal $ in user text
    text = _ION_HARM_RE.sub(r"\1,\2\3", text)
    # "{+}" keeps mathtext from spacing the + like an operator ("80 +")
    text = _ION_RE.sub(lambda m: (r"$^{%s%s}\mathrm{%s}^{%s{+}}$"
                                  % (m.group(2), r"\mathrm{%s}" % m.group(1) if m.group(1) else "",
                                     m.group(3), m.group(4))), text)
    text = _HARM_RE.sub(lambda m: r"$h$ = %s" % m.group(1), text)
    return text


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
    parser.add_argument("--title-full", type=str, default=None, metavar="TEXT",
                        help="Use TEXT as the whole plot title (replaces the auto-generated "
                             "title; --title is then ignored)")
    parser.add_argument("--skip", type=int, nargs=2, default=[0, 0], metavar=("X", "Y"),
                        help="Skip the first X and last Y time bins (applied before --tmin/--tmax)")
    parser.add_argument("--err-coeffs", type=float, nargs=2, default=None, metavar=("A", "B"),
                        help="Error bars σ = A·I + B on each intensity point; used as "
                             "absolute 1σ weights in the fit")
    parser.add_argument("--parent-window", type=float, nargs=2, default=None, metavar=("FMIN", "FMAX"),
                        help="Joint Bateman fit: parent peak frequency window. The positional "
                             "freq_min/freq_max are then the daughter window. Parent fitted as "
                             "A·e^(-t/τ_A) + C_A, daughter as Bateman growth with no offset, τ_A shared.")
    parser.add_argument("--unweighted", action="store_true",
                        help="Single unweighted fit, no error bars")
    parser.add_argument("--log-plot", action="store_true",
                        help="Add a semi-log panel to the right: data − C and fit − C on a log "
                             "axis (C = fitted constant background), so an exponential decay "
                             "is a straight line. Single-window fits only.")
    parser.add_argument("--residuals", action="store_true",
                        help="Add residual panels under the plot(s): (data − fit)/σ, or "
                             "data − fit for an unweighted fit. Single-window fits only.")
    parser.add_argument("--time-avg", type=int, metavar="N", default=None,
                        help="Average every N consecutive time frames before fitting")
    parser.add_argument("--baseline", type=str, default=None, metavar="JSON",
                        help="Remove a BrPLS baseline (pybaselines) from every frame first; "
                             "JSON of its parameters, e.g. '{\"lam\": 1e5}' (missing = default). "
                             "Needs --baseline-window.")
    parser.add_argument("--baseline-window", type=float, nargs=2, default=None,
                        metavar=("FMIN", "FMAX"),
                        help="Frequency window (Hz) the baseline is fitted over (the spectrogram "
                             "page's loaded crop); the fit windows must lie inside it")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load(path):
    """Return freq (F,), time (T,), power (T, F) from an ESR .npz file."""
    with np.load(path) as data:    # closed even if a key is missing
        freq  = data["freq"].ravel()
        time  = data["time"].ravel()
        power = data["power"]
    time = time - time[0]
    return freq, time, power


def load_data(args):
    """load(args.file), with --baseline applied: frames restricted to
    --baseline-window and each one's BrPLS baseline subtracted."""
    freq, time, power = load(args.file)
    if args.baseline is None:
        return freq, time, power
    lo, hi = args.baseline_window
    freq, power = slice_freq(freq, power, lo, hi)
    power = baseline.remove_cached(args.file, freq, power, (lo, hi), args.baseline_params)
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
    _trapz = getattr(np, "trapezoid", None) or np.trapz   # np.trapezoid added in NumPy 2.0; trapz removed later
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

class FitError(RuntimeError):
    """A fit did not converge or could not be started (message for the user)."""



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
              halflife_isomer_lab=None, halflife_groundstate_lab=None,
              sigma=None):
    """
    Fit exponential decay or Bateman chain.

    For Bateman:
      - B is always fixed to intensity[0] (first frame of the fit window).
      - C is always a free parameter (constant background offset).
      - halflife_isomer_lab    → fixes τ_A = T_lab/ln(2)
      - halflife_groundstate_lab → fixes τ_B = T_lab/ln(2)
      - Remaining free parameters: P, C, and any τ not fixed.

    sigma : optional per-point 1-σ errors on intensity. When given, the fit is
            weighted by 1/σ² and absolute_sigma=True, so parameter errors
            (and hence half-life errors) come directly from these error bars.

    Returns
    -------
    popt     : fitted parameter array (P first, C last; τ values in between)
    perr     : 1-σ uncertainties on popt
    model    : callable, model(t, *popt)
    resolved : dict  {'P','tau_A','tau_B','B','C'} → (value, err, label)
               label is 'fitted' or 'fixed'. None for simple decay.

    Raises FitError if curve_fit fails.
    """
    _w = dict(sigma=sigma, absolute_sigma=True) if sigma is not None else {}

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
                    maxfev=10_000, **_w,
                )
                model = model_with_offset
            else:
                popt, pcov = curve_fit(
                    model_no_offset, time, intensity,
                    p0=[A0, tau0],
                    bounds=([0, 0], [np.inf, np.inf]),
                    maxfev=10_000, **_w,
                )
                model = model_no_offset
        except (RuntimeError, ValueError) as exc:
            raise FitError(f"Curve fit failed: {exc}") from exc
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
                maxfev=20_000, **_w,
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
                maxfev=20_000, **_w,
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
                maxfev=20_000, **_w,
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
                maxfev=20_000, **_w,
            )
            perr = np.sqrt(np.diag(pcov))
            P, C               = popt
            dP, dC             = perr
            tau_A, dtau_A      = tau_A_fixed, 0.0
            tau_B, dtau_B      = tau_B_fixed, 0.0

    except (RuntimeError, ValueError) as exc:
        raise FitError(f"Curve fit failed: {exc}") from exc

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


def estimate_error_model(time, intensity, popt, model):
    """
    Estimate σ(I) = a·I + b from the residuals of a first-pass fit.

    Uses the model intensity f(t_i) (not the noisy data) as the regressor, so
    low-fluctuating points aren't given artificially small errors.

    Returns
    -------
    sigma : per-point σ_i evaluated at the model intensity
    a, b  : coefficients (a = 0 when falling back to a constant σ)
    kind  : 'linear' or 'constant'
    """
    f      = model(time, *popt)
    r      = intensity - f
    n, p   = len(r), len(popt)
    dof_corr = np.sqrt(n / max(n - p, 1))
    y      = np.abs(r) * np.sqrt(np.pi / 2) * dof_corr

    if np.ptp(f) > 0 and n >= 3:
        a, b  = np.polyfit(f, y, 1)
        sigma = a * f + b
        if np.all(sigma > 0):
            return sigma, a, b, "linear"

    # Fallback: constant σ = rms residual (dof-corrected)
    s0 = np.sqrt(np.sum(r ** 2) / max(n - p, 1))
    return np.full(n, s0), 0.0, s0, "constant"


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

def _box_extent(artist, renderer):
    """Window extent of a legend, or of a text's background box if it has one."""
    patch = artist.get_bbox_patch() if hasattr(artist, "get_bbox_patch") else None
    return (patch or artist).get_window_extent(renderer)


def place_boxes(fig, ax, xy_sets, fixed=(), legend_kw=None, locs=None, log=False,
                max_iter=8, pad_px=6, min_fontsize=10, prefer=None):
    """Wrapper: if the legend can't be placed clear of everything with at most
    `max_iter` headroom steps, retry with a smaller legend font.

    prefer: preferred legend position - "top_right" (top-right corner of the
    panel) or "below_fixed_right" (right-aligned just under the first fixed
    box). The clear spot nearest to it is used; None = first clear spot from
    the top-left."""
    fig.canvas.draw()          # one full draw; later steps reuse its layout
    if legend_kw is None:
        return _place_boxes_once(fig, ax, xy_sets, fixed, None, log, max_iter, pad_px, prefer)
    y0 = ax.get_ylim()
    fs0 = legend_kw.get("fontsize", 18)
    for fs in sorted({fs0, max(min_fontsize, fs0 - 3), max(min_fontsize, fs0 - 6)}, reverse=True):
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        ax.set_ylim(y0)
        if _place_boxes_once(fig, ax, xy_sets, fixed, dict(legend_kw, fontsize=fs),
                             log, max_iter, pad_px, prefer):
            return True
    # no room inside the panel: one row of legend entries just above it
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    n = len(ax.get_legend_handles_labels()[0])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=max(n, 1),
              borderaxespad=0.3, **dict(legend_kw, fontsize=max(min_fontsize, fs0 - 6)))
    return False


def _place_boxes_once(fig, ax, xy_sets, fixed, legend_kw, log, max_iter, pad_px, prefer=None):
    """Keep legend / text boxes from covering data on `ax`.

    xy_sets : list of (x, y) arrays in data coordinates (points, error-bar
              ends, fit-curve samples) that must stay visible.
    fixed   : text artists with fixed positions (e.g. the parameter box).
    The legend is put at the first clear spot found by scanning the panel
    (top rows first, left to right) - clear of the data and the fixed boxes.
    Only if no spot exists (or a fixed box covers data) is headroom added
    above the data, a little at a time. `locs` is unused (kept for callers).
    """
    xs = np.concatenate([np.asarray(x, float).ravel() for x, _ in xy_sets]) if xy_sets else np.array([])
    ys = np.concatenate([np.asarray(y, float).ravel() for _, y in xy_sets]) if xy_sets else np.array([])
    good = np.isfinite(xs) & np.isfinite(ys) & ((ys > 0) if log else True)
    xy = np.column_stack([xs[good], ys[good]])

    def covers(bb, pts, pad=0):
        return bool(np.any((pts[:, 0] >= bb.x0 - pad) & (pts[:, 0] <= bb.x1 + pad) &
                           (pts[:, 1] >= bb.y0 - pad) & (pts[:, 1] <= bb.y1 + pad)))

    leg = None
    if legend_kw is not None:
        leg = ax.legend(loc="upper left", **legend_kw)
    r = fig.canvas.get_renderer()
    for _ in range(max_iter + 1):
        pts = ax.transData.transform(xy) if len(xy) else np.empty((0, 2))
        fixed_bb = [_box_extent(t, r) for t in fixed]
        if not any(covers(b, pts, pad_px) for b in fixed_bb):
            if leg is None:
                return True
            # scan candidate positions for the legend's upper-left corner
            axb = ax.get_window_extent(r)
            lb  = _box_extent(leg, r)
            w, h = lb.width + 2 * pad_px, lb.height + 2 * pad_px
            fx = np.linspace(0.0, 1.0, 51)
            cx0 = axb.x0 + fx * axb.width                    # candidate left edges
            cy1 = axb.y0 + fx[::-1] * axb.height             # candidate top edges, top first
            X0, Y1 = np.meshgrid(cx0, cy1)                   # rows: top -> bottom
            X0, Y1 = X0.ravel(), Y1.ravel()
            # preferred spot (candidate rectangle includes pad_px all round)
            target = None
            if prefer == "top_right":
                target = (axb.x1 - w, axb.y1)
            elif prefer == "below_fixed_right" and fixed_bb:
                b0 = fixed_bb[0]
                target = (b0.x1 - w + pad_px, b0.y0)
            if target is not None:
                X0 = np.concatenate([[target[0]], X0])
                Y1 = np.concatenate([[target[1]], Y1])
            inside = (X0 + w <= axb.x1) & (Y1 - h >= axb.y0)
            free = inside.copy()
            for b in fixed_bb:                               # not on a fixed box
                free &= ~((X0 < b.x1) & (X0 + w > b.x0) & (Y1 > b.y0) & (Y1 - h < b.y1))
            idx = np.nonzero(free)[0]
            if len(idx) and len(pts):
                # drop candidates that contain any data point (chunked)
                keep = np.ones(len(idx), bool)
                for k in range(0, len(idx), 400):
                    c = idx[k:k + 400]
                    hit = ((pts[None, :, 0] >= X0[c, None]) & (pts[None, :, 0] <= X0[c, None] + w) &
                           (pts[None, :, 1] <= Y1[c, None]) & (pts[None, :, 1] >= Y1[c, None] - h))
                    keep[k:k + 400] = ~hit.any(axis=1)
                idx = idx[keep]
            if len(idx):
                if target is not None:   # clear spot nearest the preferred one
                    c = idx[np.argmin((X0[idx] - target[0]) ** 2 + (Y1[idx] - target[1]) ** 2)]
                else:
                    c = idx[0]
                # place the legend frame (pad_px inside the candidate rectangle)
                ax_xy = ax.transAxes.inverted().transform((X0[c] + pad_px, Y1[c] - pad_px))
                leg.remove()
                leg = ax.legend(loc="upper left", bbox_to_anchor=tuple(ax_xy),
                                borderaxespad=0.0, **legend_kw)
                return True
        if _ == max_iter:
            break
        # nothing fits: add headroom above the data (transforms update
        # immediately; box sizes in pixels don't change)
        lo, hi = ax.get_ylim()
        if log:
            ax.set_ylim(lo, hi * (hi / lo) ** 0.08)
        else:
            ax.set_ylim(lo, hi + 0.08 * (hi - lo))
    return leg is None   # fixed boxes alone may still overlap after max_iter


# ---------------------------------------------------------------------------
# Fit records (JSON, saved with the "Save fit parameters" button in the plot window)
# ---------------------------------------------------------------------------

FIT_RECORD_FORMAT = "asphodel-decay-fit"
FIT_RECORD_VERSION = 1


def _num(v):
    """JSON-safe number: numpy -> float, NaN/inf/None -> None."""
    if v is None:
        return None
    v = float(v)
    return v if np.isfinite(v) else None


def _param(value, error, status):
    """One parameter entry; the error is left out (None) for fixed parameters."""
    return {"value": _num(value), "error": _num(error) if status == "fitted" else None,
            "status": status}


def _half_life(tau, dtau, fitted, gamma):
    """Half-life entry from τ ± dτ (lab frame), with the c.o.m. value if γ is known."""
    ln2 = np.log(2)
    hl, dhl = tau * ln2, (dtau * ln2 if fitted else None)
    out = {"lab_s": _num(hl), "lab_err_s": _num(dhl), "status": "fitted" if fitted else "fixed"}
    if gamma:
        out["com_s"] = _num(hl / gamma)
        out["com_err_s"] = _num(dhl / gamma) if dhl is not None else None
    return out


def _settings_of(args):
    """The inputs and options of this run, as given on the command line."""
    return {
        "freq_window_hz":         [_num(args.freq_min), _num(args.freq_max)],
        "parent_window_hz":       ([_num(x) for x in args.parent_window]
                                   if args.parent_window is not None else None),
        "tmin_s":                 _num(args.tmin),
        "tmax_s":                 _num(args.tmax),
        "skip_bins":              [int(x) for x in args.skip],
        "time_avg":               args.time_avg,
        "baseline":               ({"method": baseline.METHOD, "params": args.baseline_params,
                                    "window_hz": [_num(x) for x in args.baseline_window]}
                                   if args.baseline is not None else None),
        "gamma":                  _num(args.gamma),
        "bateman":                bool(args.bateman),
        "with_offset":            not args.no_offset,
        "halflife_isomer_com_s":      _num(args.halflife_isomer),
        "halflife_groundstate_com_s": _num(args.halflife_groundstate),
        "err_coeffs":             ([_num(x) for x in args.err_coeffs]
                                   if args.err_coeffs is not None else None),
        "unweighted":             bool(args.unweighted),
        "title":                  args.title,
        "title_full":             args.title_full,
    }


def _record_base(args, mode, model_text):
    return {
        "format":     FIT_RECORD_FORMAT,
        "version":    FIT_RECORD_VERSION,
        "saved":      None,                       # filled in when saved
        "experiment": paths.experiment(),
        "file":       os.path.abspath(args.file),
        "label":      args.title_full or args.title or "",
        "mode":       mode,                       # "decay" | "bateman" | "joint"
        "model":      model_text,
        "settings":   _settings_of(args),
    }


def _chi2_part(chi2, ndf):
    return {"chi2": _num(chi2), "ndf": ndf,
            "chi2_ndf": _num(chi2 / ndf) if (chi2 is not None and ndf) else None}


def single_fit_record(args, time_fit, intensity_fit, popt, perr, model, resolved,
                      with_offset, sigma_fit, err_note):
    """Record of a single-window fit (exponential decay or Bateman)."""
    residuals = intensity_fit - model(time_fit, *popt)
    chi2 = ndf = None
    if sigma_fit is not None:
        chi2 = float(np.sum((residuals / sigma_fit) ** 2))
        ndf = len(time_fit) - len(popt)
    gamma = args.gamma
    if args.bateman:
        rec = _record_base(args, "bateman",
                           "I(t) = P·(e^(-t/τ_A) - e^(-t/τ_B)) + B·e^(-t/τ_B) + C")
        params = {k: _param(*resolved[k]) for k in ("P", "tau_A", "tau_B", "B", "C")}
        half_lives = {
            "A": _half_life(resolved["tau_A"][0], resolved["tau_A"][1],
                            resolved["tau_A"][2] == "fitted", gamma),
            "B": _half_life(resolved["tau_B"][0], resolved["tau_B"][1],
                            resolved["tau_B"][2] == "fitted", gamma),
        }
    else:
        rec = _record_base(args, "decay",
                           "I(t) = A·exp(-t/τ)" + (" + C" if with_offset else ""))
        params = {"A": _param(popt[0], perr[0], "fitted"),
                  "tau": _param(popt[1], perr[1], "fitted")}
        if with_offset:
            params["C"] = _param(popt[2], perr[2], "fitted")
        half_lives = {"decay": _half_life(popt[1], perr[1], True, gamma)}
    rec["result"] = {
        "n_points":       int(len(time_fit)),
        "t_range_s":      [_num(time_fit[0]), _num(time_fit[-1])],
        "parameters":     params,
        "half_lives":     half_lives,
        **_chi2_part(chi2, ndf),
        "error_model":    err_note or ("unweighted" if sigma_fit is None else ""),
        "mean_intensity": _num(np.mean(intensity_fit)),
        "std_residual":   _num(np.std(residuals)),
    }
    return rec


def joint_fit_record(args, tP, vals, errs, free, chi2, ndf, corr, err_note):
    """Record of a joint parent + daughter Bateman fit."""
    rec = _record_base(args, "joint",
                       "I_P(t) = A·e^(-t/τ_A) + C_A;  "
                       "I_D(t) = P·(e^(-t/τ_A) - e^(-t/τ_B)) + B·e^(-t/τ_B)")
    params = {n: _param(vals[n], errs[n], "fitted" if n in free else "fixed")
              for n in _JOINT_NAMES}
    rec["result"] = {
        "n_points":    int(len(tP)),              # per window
        "t_range_s":   [_num(tP[0]), _num(tP[-1])],
        "parameters":  params,
        "half_lives":  {"A": _half_life(vals["tau_A"], errs["tau_A"], "tau_A" in free, args.gamma),
                        "B": _half_life(vals["tau_B"], errs["tau_B"], "tau_B" in free, args.gamma)},
        **_chi2_part(chi2, ndf),
        "corr_tau_A_tau_B": _num(corr),
        "error_model": err_note or ("unweighted" if chi2 is None else ""),
    }
    return rec


def save_fit_record(record):
    """Write the record to outputs/fits/ and return its path."""
    rec = dict(record, saved=datetime.datetime.now().isoformat(timespec="seconds"))
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = os.path.splitext(os.path.basename(rec["file"]))[0]
    label = _re.sub(r"[^A-Za-z0-9+.-]+", "_", rec.get("label") or "").strip("_")
    name = f"fit_{stamp}_{stem}" + (f"_{label[:40]}" if label else "") + ".json"
    path = os.path.join(paths.outputs("fits"), name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=2, ensure_ascii=False)
    return path


def add_save_button(fig, record):
    """'Save fit parameters' button in the top-left corner of the figure. One click saves
    the record; later clicks only say where it was saved."""
    ax_btn = fig.add_axes([0.005, 0.945, 0.15, 0.045])
    try:
        ax_btn.set_in_layout(False)               # not moved by constrained layout
    except AttributeError:
        pass
    btn = Button(ax_btn, "Save fit parameters", color="#fff4ec", hovercolor="#f57d15")
    btn.label.set_fontsize(13)
    state = {"path": None}

    def on_click(_event):
        if state["path"] is None:
            try:
                state["path"] = save_fit_record(record)
            except Exception as exc:
                print(f"Could not save the fit: {exc}")
                btn.label.set_text("Save failed")
                fig.canvas.draw_idle()
                return
            print(f"Fit saved to {state['path']}")
            btn.label.set_text("Parameters saved ✓")
        else:
            print(f"Already saved: {state['path']}")
        fig.canvas.draw_idle()

    btn.on_clicked(on_click)
    fig._save_fit_button = btn                    # keep a reference (else it stops responding)
    return btn


# ---------------------------------------------------------------------------
# "Save to log book" (results log book, core/results_logbook.py)
# ---------------------------------------------------------------------------

def _hl_items(hl, slot, who=""):
    """Value rows (lab and c.o.m. half-life) of one half-life entry of a record."""
    from core.results_logbook import fmt_time_value
    out = []
    fixed = hl.get("status") == "fixed"
    tag = (f" {who}" if who else "") + (" (fixed)" if fixed else "")
    for frame, k in (("c.o.m.", "com"), ("lab", "lab")):
        v = hl.get(f"{k}_s")
        if v is None:
            continue
        item = fmt_time_value(v, hl.get(f"{k}_err_s"))
        # fixed half-lives were inputs, not results: shown but not ticked
        out.append(dict(item, label=f"T½{tag} ({frame})", slot=slot, checked=not fixed))
    return out


def logbook_spec(record):
    """What the "Save to log book" popout (core/logbook_dialog.py) offers for a fit."""
    label = record.get("label") or ""
    ions = ["".join(m.groups()[:3]) + m.group(4) + "+" for m in _ION_RE.finditer(label)]
    harm = _HARM_RE.search(label)
    res = record.get("result", {})
    hls = res.get("half_lives", {})
    st = record.get("settings", {})
    two = record.get("mode") in ("bateman", "joint")
    if two:
        slots = [{"label": "Parent (A)", "value": ions[0] if ions else ""},
                 {"label": "Daughter (B)", "value": ions[1] if len(ions) > 1 else ""}]
        items = _hl_items(hls.get("A", {}), 0) + _hl_items(hls.get("B", {}), 1)
    else:
        slots = [{"label": "Ion", "value": ions[0] if ions else ""}]
        items = _hl_items(hls.get("decay", {}), None)
    if harm:
        items.append({"label": "Harmonic h", "value": int(harm.group(1)), "slot": None})
    if res.get("chi2_ndf") is not None:
        items.append({"label": "χ²/ndf", "value": round(res["chi2_ndf"], 3), "slot": None})
    fw = st.get("freq_window_hz") or []
    if len(fw) == 2 and None not in fw:
        items.append({"label": "Daughter window" if record.get("mode") == "joint" else "Frequency window",
                      "value": f"{fw[0]:.1f} – {fw[1]:.1f}", "unit": "Hz", "slot": None,
                      "checked": False})
    pw = st.get("parent_window_hz")
    if pw and None not in pw:
        items.append({"label": "Parent window", "value": f"{pw[0]:.1f} – {pw[1]:.1f}", "unit": "Hz",
                      "slot": None, "checked": False})
    tr = res.get("t_range_s") or []
    if len(tr) == 2 and None not in tr:
        items.append({"label": "Fit time range", "value": f"{tr[0]:.4g} – {tr[1]:.4g}", "unit": "s",
                      "slot": None, "checked": False})
    mode = {"decay": "exponential decay", "bateman": "Bateman (A → B)",
            "joint": "joint parent + daughter (Bateman)"}.get(record.get("mode"), record.get("mode"))
    context = {"file": record.get("file"), "fit": mode, "model": record.get("model"),
               "error model": res.get("error_model") or "", "γ": st.get("gamma"),
               "fit title": label}
    return {"source": "Decay fit",
            "title": f"Half-li{'ves' if two else 'fe'}: {mode}",
            "slots": slots, "items": items,
            "context": {k: v for k, v in context.items() if v not in (None, "")}}


def _figure_png(fig):
    """The figure as a PNG file (temporary), without the buttons."""
    import tempfile
    hidden = []
    for attr in ("_save_fit_button", "_logbook_button", "_asphodel_output_button"):
        b = getattr(fig, attr, None)
        if b is not None and b.ax.get_visible():
            b.ax.set_visible(False)
            hidden.append(b.ax)
    fd, path = tempfile.mkstemp(prefix="asphodel_logbook_", suffix=".png")
    os.close(fd)
    try:
        fig.savefig(path, format="png", dpi=110)
    finally:
        for ax in hidden:
            ax.set_visible(True)
        fig.canvas.draw_idle()
    return path


def open_logbook_dialog(fig, record):
    """Open the "Save to log book" popout (own process: Tk) for this fit."""
    import subprocess
    import tempfile
    spec = logbook_spec(record)
    spec["plots"] = [{"path": _figure_png(fig), "caption": "Decay fit", "checked": True}]
    fd, path = tempfile.mkstemp(prefix="asphodel_logbook_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(spec, fh, ensure_ascii=False)
    script = os.path.join(paths.REPO_ROOT, "core", "logbook_dialog.py")
    env = dict(os.environ, **{paths.ENV_EXPERIMENT: paths.experiment()})
    env.pop("ASPHODEL_READY_FILE", None)
    subprocess.Popen([sys.executable, script, path], env=env)


def add_logbook_button(fig, record):
    """'Save to log book' button next to 'Save fit parameters': opens a popout
    to send the half-lives and the figure to the results log book."""
    ax_btn = fig.add_axes([0.16, 0.945, 0.13, 0.045])
    try:
        ax_btn.set_in_layout(False)
    except AttributeError:
        pass
    btn = Button(ax_btn, "Save to log book", color="#fff4ec", hovercolor="#f57d15")
    btn.label.set_fontsize(13)

    def on_click(_event):
        try:
            open_logbook_dialog(fig, record)
        except Exception as exc:
            print(f"Could not open the log book popout: {exc}")

    btn.on_clicked(on_click)
    fig._logbook_button = btn                     # keep a reference (else it stops responding)
    return btn


def plot_decay(time, intensity, time_fit, intensity_fit, popt, perr, model, resolved,
               with_offset, bateman, freq_min, freq_max, filename,
               gamma=None, title_suffix=None, sigma_fit=None, err_note=None,
               title_full=None, log_plot=False, show_residuals=False, record=None,
               warning=None):

    plt.rcParams.update({"font.size": 20})

    t_dense = np.linspace(time_fit[0], time_fit[-1], 2000)
    I_fit   = model(t_dense, *popt)

    # Mean intensity and residual std dev over the fitted window
    residuals      = intensity_fit - model(time_fit, *popt)
    std_resid      = np.std(residuals)
    mean_intensity = np.mean(intensity_fit)

    # χ²/ndf when error bars are supplied
    chi2 = ndf = None
    if sigma_fit is not None:
        chi2 = float(np.sum((residuals / sigma_fit) ** 2))
        ndf  = len(time_fit) - len(popt)

    # Layout: linear panel (+ semi-log panel to the right), each with an
    # optional residual panel underneath sharing its time axis
    ncols = 2 if log_plot else 1
    nrows = 2 if show_residuals else 1
    multi = ncols > 1 or nrows > 1
    figsize = (16 if ncols == 1 else 20, 6 if nrows == 1 else 9.5)
    if multi:
        # constrained layout keeps axis labels/titles inside the figure,
        # also when the window is smaller than figsize or is resized
        try:
            fig = plt.figure(figsize=figsize, layout="constrained")
        except TypeError:                                   # matplotlib < 3.5
            fig = plt.figure(figsize=figsize, constrained_layout=True)
        try:
            fig.get_layout_engine().set(h_pad=0.2, w_pad=0.25, hspace=0.0, wspace=0.04)
        except AttributeError:
            fig.set_constrained_layout_pads(h_pad=0.2, w_pad=0.25, hspace=0.0, wspace=0.04)
    else:
        fig = plt.figure(figsize=figsize)
    gs  = fig.add_gridspec(nrows, ncols, height_ratios=[3, 1] if nrows == 2 else [1])
    ax      = fig.add_subplot(gs[0, 0])
    ax_log  = fig.add_subplot(gs[0, 1]) if log_plot else None
    ax_res  = fig.add_subplot(gs[1, 0], sharex=ax) if show_residuals else None
    ax_lres = fig.add_subplot(gs[1, 1], sharex=ax_log) if (show_residuals and log_plot) else None

    if sigma_fit is not None:
        ax.errorbar(time_fit, intensity_fit, yerr=sigma_fit, fmt="o", ms=3,
                    color="black", ecolor="black", elinewidth=0.8,
                    capsize=2.5, capthick=0.8, label="Data (integrated power)")
    else:
        ax.plot(time_fit, intensity_fit, "o", ms=3,
                color="black", label="Data (integrated power)")
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

    # (⟨I⟩ and the σ error-model note are printed to the terminal, not shown in the box)
    if chi2 is not None:
        chi2_red = chi2 / ndf if ndf > 0 else float("nan")
        eq_text += f"\n\n$\\chi^2/ndf$ = {chi2:.4g}/{ndf} = {chi2_red:.3g}"
    if warning:
        eq_text += f"\n\n{warning}"

    par_box = ax.text(
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
    if title_full:
        title = title_full

    ax.set_ylabel("Integrated power (arb. units)", fontsize=22)
    ax.tick_params(labelsize=20)
    # narrower panel next to the log plot: keep the legend clear of the
    # parameter box (lower left is empty for a decay)
    # legend is placed at the end (place_boxes) so it avoids data and the box
    ax.set_xlim(left=0)

    # ── Semi-log panel: data − C and fit − C, so an exponential is a straight line
    log_pts = []
    if ax_log is not None:
        if bateman:
            C_bg = resolved['C'][0]
        else:
            C_bg = popt[2] if with_offset else 0.0
        sub   = f" − C" if C_bg != 0 else ""
        y_dat = intensity_fit - C_bg
        y_fit = I_fit - C_bg
        ok    = y_dat > 0
        n_bad = int(np.sum(~ok))
        if n_bad:
            print(f"Log plot: {n_bad} point(s) with data{sub} ≤ 0 not shown")
        if sigma_fit is not None:
            ax_log.errorbar(time_fit[ok], y_dat[ok], yerr=sigma_fit[ok], fmt="o", ms=3,
                            color="black", ecolor="black", elinewidth=0.8,
                            capsize=2.5, capthick=0.8, label=f"Data{sub}")
        else:
            ax_log.plot(time_fit[ok], y_dat[ok], "o", ms=3, color="black", label=f"Data{sub}")
        okf = y_fit > 0
        ax_log.plot(t_dense[okf], y_fit[okf], "-", lw=2, color="#dd4444",
                    label=("Bateman fit" if bateman else "Exponential fit") + sub)
        if time_fit[0] <= t_half <= time_fit[-1]:
            ax_log.axvline(t_half, ls="--", lw=1, color="0.5")
        ax_log.set_yscale("log")
        # keep error bars that reach ≤ 0 from stretching the log axis
        if ok.any():
            lo = np.min(y_dat[ok]); hi = max(np.max(y_dat[ok]), np.max(y_fit[okf]) if okf.any() else 0)
            ax_log.set_ylim(lo / 2, hi * 2)
        ax_log.set_ylabel(f"Integrated power{sub} (log)", fontsize=22)
        ax_log.tick_params(labelsize=20)
        log_pts = [(time_fit[ok], y_dat[ok]), (t_dense[okf], y_fit[okf])]
        if sigma_fit is not None:
            up = (y_dat + sigma_fit)[ok]
            lo_e = (y_dat - sigma_fit)[ok]
            log_pts += [(time_fit[ok], up), (time_fit[ok][lo_e > 0], lo_e[lo_e > 0])]
        ax_log.set_xlim(left=0)

    # ── Residual panels: (data − fit)/σ, or data − fit for an unweighted fit
    res_boxes = []
    if ax_res is not None:
        if sigma_fit is not None:
            r_plot, r_label = residuals / sigma_fit, r"(data − fit)/$\sigma$"
            band, band_txt  = 1.0, "±1σ"
        else:
            # unweighted: no per-point σ, so use the rms residual as 1σ
            r_plot, r_label = residuals, "data − fit"
            band, band_txt  = float(np.sqrt(np.mean(residuals ** 2))), "±1σ (rms)"
        n_out   = int(np.sum(np.abs(r_plot) > band))
        pct_out = 100.0 * n_out / len(r_plot) if len(r_plot) else float("nan")
        stat_txt = (f"outside {band_txt}: {pct_out:.1f}%  ({n_out}/{len(r_plot)})"
                    f"   Gaussian ≈ 31.7%")
        print(f"  Residuals: {pct_out:.1f}% of points outside {band_txt} "
              f"({n_out}/{len(r_plot)}; 31.7% expected for Gaussian errors)")
        for a_r in (ax_res, ax_lres):
            if a_r is None:
                continue
            a_r.axhspan(-band, band, color="0.85", zorder=0)
            res_boxes.append((a_r, a_r.text(0.995, 0.95, stat_txt, transform=a_r.transAxes, ha="right", va="top",
                     fontsize=13, bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                                            edgecolor="0.7", alpha=0.85))))
            a_r.axhline(0, color="#dd4444", lw=1.5)
            a_r.plot(time_fit, r_plot, "o", ms=3, color="black")
            a_r.set_ylabel(r_label, fontsize=18)
            a_r.set_xlabel("Time (s)", fontsize=22)
            a_r.tick_params(labelsize=18)
            a_r.set_xlim(left=0)
        plt.setp(ax.get_xticklabels(), visible=False)
        if ax_log is not None and ax_lres is not None:
            plt.setp(ax_log.get_xticklabels(), visible=False)
    else:
        ax.set_xlabel("Time (s)", fontsize=22)
        if ax_log is not None:
            ax_log.set_xlabel("Time (s)", fontsize=22)

    # One title: over the single panel, or centred over both columns
    if ncols == 1:
        ax.set_title(format_title_mathtext(title), fontsize=22, pad=8)
    else:
        fig.suptitle(format_title_mathtext(title), fontsize=22)
    if not multi:
        fig.tight_layout(pad=0.6)

    # ── Keep legends / text boxes off the data: try legend positions, and add
    #    headroom above the data until nothing overlaps
    lin_pts = [(time_fit, intensity_fit), (t_dense, I_fit)]
    if sigma_fit is not None:
        lin_pts += [(time_fit, intensity_fit + sigma_fit), (time_fit, intensity_fit - sigma_fit)]
    jobs = [dict(ax=ax, xy_sets=lin_pts, fixed=[par_box], legend_kw=dict(fontsize=18),
                 prefer="below_fixed_right",
                 locs=("upper center", "upper left", "center right", "lower left",
                       "lower center", "center"))]
    if ax_log is not None:
        jobs.append(dict(ax=ax_log, xy_sets=log_pts, legend_kw=dict(fontsize=18), log=True,
                         prefer="top_right",
                         locs=("upper right", "lower left", "upper center", "lower center",
                               "center right")))
    for a_r, t_box in res_boxes:
        jobs.append(dict(ax=a_r, xy_sets=[(time_fit, r_plot)], fixed=[t_box]))
    orig_ylim = [j["ax"].get_ylim() for j in jobs]

    def relayout_boxes(*_):
        # positions depend on the on-screen size: redo from the original
        # y-ranges whenever the window is (re)sized
        for j, yl in zip(jobs, orig_ylim):
            if j["ax"].get_legend() is not None:
                j["ax"].get_legend().remove()
            j["ax"].set_ylim(yl)
        for j in jobs:
            place_boxes(fig, **j)
        fig.canvas.draw_idle()

    relayout_boxes()
    fig._relayout_boxes = relayout_boxes
    fig.canvas.mpl_connect("resize_event", relayout_boxes)
    if record is not None:
        add_save_button(fig, record)
        add_logbook_button(fig, record)
    show_window()

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
        print(f"  ⟨I⟩          = {format_value(mean_intensity)} ± {format_value(std_resid)}")
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
    print(f"  ⟨I⟩        = {format_value(mean_intensity)} ± {format_value(std_resid)}")
    if chi2 is not None:
        chi2_red = chi2 / ndf if ndf > 0 else float("nan")
        print(f"  χ²/ndf     = {chi2:.4g}/{ndf} = {chi2_red:.3g}  (errors: absolute σ)")
    print(f"{'─'*55}\n")


# ---------------------------------------------------------------------------
# Joint parent + daughter Bateman fit (two frequency windows)
# ---------------------------------------------------------------------------
#
#   Parent window   : I_P(t) = A·e^{-t/τ_A} + C_A                 (with offset)
#   Daughter window : I_D(t) = P·(e^{-t/τ_A} − e^{-t/τ_B}) + B·e^{-t/τ_B}
#                                                                 (no offset)
#
# τ_A is shared, so it is constrained by both the parent's decay and the
# daughter's growth. The daughter has no offset because for τ_B ≫ the
# measurement, B·e^{-t/τ_B} is nearly flat and would be degenerate with one.
# For τ_A < τ_B the growth term needs P < 0.

_JOINT_NAMES = ["A", "tau_A", "C_A", "P", "tau_B", "B"]


def _joint_curves(tP, tD, A, tau_A, C_A, P, tau_B, B):
    IP = A * np.exp(-tP / tau_A) + C_A
    ID = P * (np.exp(-tD / tau_A) - np.exp(-tD / tau_B)) + B * np.exp(-tD / tau_B)
    return IP, ID


def fit_bateman_joint(tP, IP, tD, ID, tau_A_fixed=None, tau_B_fixed=None,
                      sigmaP=None, sigmaD=None):
    """
    Simultaneous fit of the parent and daughter windows.

    Returns
    -------
    vals  : dict name → value (all six parameters)
    errs  : dict name → 1σ error (0 for fixed)
    free  : list of fitted parameter names
    pcov  : covariance of the free parameters (order = free)
    """
    nP   = len(tP)
    span = max(tP[-1], tD[-1]) - min(tP[0], tD[0])

    # Initial guesses
    C_A0 = float(np.min(IP))
    A0   = max(float(np.max(IP)) - C_A0, 1e-30)
    above = np.nonzero(IP - C_A0 < A0 / np.e)[0]
    tau_A0 = float(tP[above[0]] - tP[0]) if len(above) and tP[above[0]] > tP[0] else span / 5
    tau_A0 = max(tau_A0, span / 200)
    tau_B0 = 10 * span
    B0   = float(ID[0])
    P0   = -max(float(np.max(ID)) - B0, 1e-30)

    guess  = dict(A=A0, tau_A=tau_A0, C_A=C_A0, P=P0, tau_B=tau_B0, B=B0)
    lower  = dict(A=0, tau_A=0, C_A=-np.inf, P=-np.inf, tau_B=0, B=-np.inf)
    fixed  = {}
    if tau_A_fixed is not None:
        fixed["tau_A"] = tau_A_fixed
    if tau_B_fixed is not None:
        fixed["tau_B"] = tau_B_fixed
    free = [n for n in _JOINT_NAMES if n not in fixed]

    def unpack(args):
        v = dict(fixed); v.update(zip(free, args)); return v

    def f(x, *args):
        v = unpack(args)
        IPm, IDm = _joint_curves(x[:nP], x[nP:], **v)
        return np.concatenate([IPm, IDm])

    x = np.concatenate([tP, tD])
    y = np.concatenate([IP, ID])
    kw = {}
    if sigmaP is not None:
        kw = dict(sigma=np.concatenate([sigmaP, sigmaD]), absolute_sigma=True)
    try:
        popt, pcov = curve_fit(
            f, x, y,
            p0=[guess[n] for n in free],
            bounds=([lower[n] for n in free], [np.inf] * len(free)),
            maxfev=50_000, **kw,
        )
    except (RuntimeError, ValueError) as exc:
        raise FitError(f"Joint Bateman fit failed: {exc}") from exc

    vals = unpack(popt)
    perr = np.sqrt(np.diag(pcov))
    errs = {n: 0.0 for n in _JOINT_NAMES}
    errs.update(zip(free, perr))
    return vals, errs, free, pcov


def _window_intensity(freq, power, time, fmin, fmax, n_avg, skip):
    """Integrate a frequency window, time-average, then apply --skip."""
    fs, ps = slice_freq(freq, power, fmin, fmax)
    I = integrate_band(fs, ps)
    t = time
    if n_avg > 1:
        T_trim = (len(I) // n_avg) * n_avg
        I = I[:T_trim].reshape(-1, n_avg).mean(axis=1)
        t = t[:T_trim].reshape(-1, n_avg).mean(axis=1)
    s0, s1 = skip
    end = len(t) - s1 if s1 > 0 else len(t)
    return t[s0:end], I[s0:end], ps.shape[1]


def run_joint(args, halflife_isomer_lab, halflife_groundstate_lab):
    pmin, pmax = args.parent_window
    dmin, dmax = args.freq_min, args.freq_max
    n_avg = args.time_avg if (args.time_avg is not None and args.time_avg > 1) else 1

    freq, time, power = load_data(args)
    tP_all, IP_all, nbP = _window_intensity(freq, power, time, pmin, pmax, n_avg, args.skip)
    tD_all, ID_all, nbD = _window_intensity(freq, power, time, dmin, dmax, n_avg, args.skip)
    print(f"Parent window   [{pmin:,.0f} – {pmax:,.0f} Hz]: {nbP} bins")
    print(f"Daughter window [{dmin:,.0f} – {dmax:,.0f} Hz]: {nbD} bins")
    if n_avg > 1:
        print(f"Time rebinned: averaged {n_avg} frames")

    mask = np.ones(len(tP_all), dtype=bool)
    if args.tmin is not None:
        mask &= tP_all >= args.tmin
    if args.tmax is not None:
        mask &= tP_all <= args.tmax
    tP, IP = tP_all[mask], IP_all[mask]
    tD, ID = tD_all[mask], ID_all[mask]

    tau_A_fixed = halflife_isomer_lab      / np.log(2) if halflife_isomer_lab      is not None else None
    tau_B_fixed = halflife_groundstate_lab / np.log(2) if halflife_groundstate_lab is not None else None
    n_free = 6 - (tau_A_fixed is not None) - (tau_B_fixed is not None)
    if 2 * len(tP) <= n_free:
        sys.exit(f"Only {len(tP)} points per window — too few for {n_free} free parameters.")
    print(f"Joint Bateman fit: {len(tP)} points per window from t={tP[0]:.2f} s "
          f"to t={tP[-1]:.2f} s  ({n_free} free parameters)")

    # ---- Error bars
    sigmaP = sigmaD = None
    err_note = None
    if args.err_coeffs is not None:
        a, b = args.err_coeffs
        sigmaP = (a * IP + b) / np.sqrt(n_avg)
        sigmaD = (a * ID + b) / np.sqrt(n_avg)
        if np.any(sigmaP <= 0) or np.any(sigmaD <= 0):
            sys.exit(f"Error bars σ = {a:g}·I + {b:g} are ≤ 0 for some points. Adjust the coefficients.")
        err_note = f"σ = {a:g}·I + {b:g}" + (f" (÷√{n_avg})" if n_avg > 1 else "") + "  [manual, both windows]"
        vals, errs, free, pcov = fit_bateman_joint(tP, IP, tD, ID, tau_A_fixed, tau_B_fixed, sigmaP, sigmaD)
    else:
        vals, errs, free, pcov = fit_bateman_joint(tP, IP, tD, ID, tau_A_fixed, tau_B_fixed)
        if not args.unweighted:
            # Two-pass: separate error model per window from the pass-1 residuals
            v1 = vals
            nP_par = sum(n in free for n in ("A", "tau_A", "C_A"))
            nD_par = sum(n in free for n in ("P", "tau_A", "tau_B", "B"))
            mP = lambda t, *_: _joint_curves(t, t, **v1)[0]
            mD = lambda t, *_: _joint_curves(t, t, **v1)[1]
            sigmaP, aP, bP, kP = estimate_error_model(tP, IP, np.zeros(nP_par), mP)
            sigmaD, aD, bD, kD = estimate_error_model(tD, ID, np.zeros(nD_par), mD)
            def _desc(a, b, k):
                return (f"{format_value(a)}·I + {format_value(b)}" if k == "linear"
                        else f"{format_value(b)} (const)")
            print(f"Error model from residuals — parent: σ = {_desc(aP, bP, kP)};  "
                  f"daughter: σ = {_desc(aD, bD, kD)}")
            print(f"  Pass 1 (unweighted): τ_A = {format_time(vals['tau_A'], errs['tau_A'])}, "
                  f"τ_B = {format_time(vals['tau_B'], errs['tau_B'])}")
            err_note = "σ from residuals (per window)"
            # if the weighted refit fails, keep the unweighted fit of pass 1
            fit1 = (vals, errs, free, pcov)
            try:
                vals, errs, free, pcov = fit_bateman_joint(tP, IP, tD, ID, tau_A_fixed,
                                                           tau_B_fixed, sigmaP, sigmaD)
            except FitError as exc:
                print(f"WARNING: weighted refit failed ({exc}); showing the unweighted fit of "
                      f"pass 1")
                vals, errs, free, pcov = fit1
                sigmaP = sigmaD = None
                err_note = "unweighted (weighted refit failed)"

    # ---- χ² and τ_A–τ_B correlation
    IPm, IDm = _joint_curves(tP, tD, **vals)
    chi2 = ndf = None
    if sigmaP is not None:
        chi2 = float(np.sum(((IP - IPm) / sigmaP) ** 2) + np.sum(((ID - IDm) / sigmaD) ** 2))
        ndf  = 2 * len(tP) - len(free)
    corr = None
    if "tau_A" in free and "tau_B" in free:
        i, j = free.index("tau_A"), free.index("tau_B")
        corr = pcov[i, j] / np.sqrt(pcov[i, i] * pcov[j, j])

    record = joint_fit_record(args, tP, vals, errs, free, chi2, ndf, corr, err_note)
    plot_joint(args, tP, IP, tD, ID, sigmaP, sigmaD, vals, errs,
               tau_A_fixed is not None, tau_B_fixed is not None,
               chi2, ndf, corr, err_note, (pmin, pmax), (dmin, dmax), record=record)


def plot_joint(args, tP, IP, tD, ID, sigmaP, sigmaD, vals, errs,
               fixA, fixB, chi2, ndf, corr, err_note, pwin, dwin, record=None):
    gamma = args.gamma
    ln2   = np.log(2)
    hlA, dhlA = vals["tau_A"] * ln2, (None if fixA else errs["tau_A"] * ln2)
    hlB, dhlB = vals["tau_B"] * ln2, (None if fixB else errs["tau_B"] * ln2)

    plt.rcParams.update({"font.size": 20})
    fig, (axP, axD) = plt.subplots(2, 1, figsize=(16, 11), sharex=True)
    t0 = min(tP[0], tD[0]); t1 = max(tP[-1], tD[-1])
    td = np.linspace(t0, t1, 2000)
    IPd, IDd = _joint_curves(td, td, **vals)

    for ax, t, I, s, Id, lab, loc in ((axP, tP, IP, sigmaP, IPd, "Parent",   "upper center"),
                                      (axD, tD, ID, sigmaD, IDd, "Daughter", "center right")):
        if s is not None:
            ax.errorbar(t, I, yerr=s, fmt="o", ms=3, color="black", ecolor="black",
                        elinewidth=0.8, capsize=2.5, capthick=0.8, label=f"{lab} data")
        else:
            ax.plot(t, I, "o", ms=3, color="black", label=f"{lab} data")
        ax.plot(td, Id, "-", lw=2, color="#dd4444", label="Joint fit")
        ax.set_ylabel("Integrated power\n(arb. units)", fontsize=20)
        ax.tick_params(labelsize=18)
        ax.legend(fontsize=16, loc=loc)

    fix = lambda f: "  [fixed]" if f else ""
    com = lambda k, hl, dhl: (f"\n$t_{{1/2,{k}}}$ (c.o.m.) = {format_time(hl / gamma, dhl / gamma if dhl else None)}"
                              if gamma is not None else "")
    txtP = (r"$I_P(t) = A \cdot e^{-t/\tau_A} + C_A$" +
            f"\n$A$     = {format_value(vals['A'], errs['A'])}"
            f"\n$\\tau_A$ = {format_time(vals['tau_A'], None if fixA else errs['tau_A'])}{fix(fixA)}"
            f"\n$C_A$   = {format_value(vals['C_A'], errs['C_A'])}"
            f"\n\n$t_{{1/2,A}}$ (lab)   = {format_time(hlA, dhlA)}" + com("A", hlA, dhlA))
    txtD = (r"$I_D(t) = P \cdot (e^{-t/\tau_A} - e^{-t/\tau_B}) + B \cdot e^{-t/\tau_B}$" +
            f"\n$P$     = {format_value(vals['P'], errs['P'])}"
            f"\n$\\tau_B$ = {format_time(vals['tau_B'], None if fixB else errs['tau_B'])}{fix(fixB)}"
            f"\n$B$     = {format_value(vals['B'], errs['B'])}"
            f"\n\n$t_{{1/2,B}}$ (lab)   = {format_time(hlB, dhlB)}" + com("B", hlB, dhlB))
    if chi2 is not None:
        txtD += f"\n\n$\\chi^2/ndf$ = {chi2:.4g}/{ndf} = {chi2 / ndf if ndf > 0 else float('nan'):.3g}"
    if corr is not None:
        txtD += f"\n$\\rho(\\tau_A, \\tau_B)$ = {corr:+.2f}"
    if err_note:
        txtD += f"\n{err_note}"
    box = dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="0.7", alpha=0.9)
    axP.text(0.97, 0.95, txtP, transform=axP.transAxes, va="top", ha="right",
             fontsize=14, family="monospace", bbox=box)
    axD.text(0.60, 0.05, txtD, transform=axD.transAxes, va="bottom", ha="right",
             fontsize=14, family="monospace", bbox=box)

    title = (f"{os.path.basename(args.file)}   joint Bateman   "
             f"parent [{pwin[0]:,.0f} – {pwin[1]:,.0f} Hz]   "
             f"daughter [{dwin[0]:,.0f} – {dwin[1]:,.0f} Hz]")
    if args.title:
        title += f"   {args.title}"
    if args.title_full:
        title = args.title_full
    axP.set_title(format_title_mathtext(title), fontsize=18, pad=8)
    axD.set_xlabel("Time (s)", fontsize=22)
    axD.set_xlim(left=0)
    fig.tight_layout()
    if record is not None:
        add_save_button(fig, record)
        add_logbook_button(fig, record)
    show_window()

    # Console summary
    print(f"\n{'─'*60}")
    print("  Joint fit:  I_P = A·e^(-t/τ_A) + C_A")
    print("              I_D = P·(e^(-t/τ_A) - e^(-t/τ_B)) + B·e^(-t/τ_B)")
    for n, lab in (("A", "A"), ("C_A", "C_A"), ("P", "P"), ("B", "B")):
        print(f"  {lab:<6} = {format_value(vals[n], errs[n])}")
    print(f"  τ_A    = {format_time(vals['tau_A'], None if fixA else errs['tau_A'])}{fix(fixA)}")
    print(f"  τ_B    = {format_time(vals['tau_B'], None if fixB else errs['tau_B'])}{fix(fixB)}")
    print(f"  t½_A (lab)    = {format_time(hlA, dhlA)}")
    if gamma is not None:
        print(f"  t½_A (c.o.m.) = {format_time(hlA / gamma, dhlA / gamma if dhlA else None)}")
    print(f"  t½_B (lab)    = {format_time(hlB, dhlB)}")
    if gamma is not None:
        print(f"  t½_B (c.o.m.) = {format_time(hlB / gamma, dhlB / gamma if dhlB else None)}")
    if chi2 is not None:
        print(f"  χ²/ndf        = {chi2:.4g}/{ndf} = {chi2 / ndf if ndf > 0 else float('nan'):.3g}")
    if corr is not None:
        print(f"  ρ(τ_A, τ_B)   = {corr:+.2f}")
    print(f"{'─'*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not os.path.isfile(args.file):
        sys.exit(f"Error: file not found — {args.file}")

    if args.parent_window is not None:
        args.bateman = True          # joint mode is always a Bateman fit

    args.baseline_params = None
    if args.baseline is not None:
        if args.baseline_window is None:
            sys.exit("Error: --baseline needs --baseline-window FMIN FMAX.")
        try:
            args.baseline_params = baseline.parse_params(json.loads(args.baseline))
        except (ValueError, TypeError) as exc:
            sys.exit(f"Error: --baseline: {exc}")
        print(f"Baseline: {baseline.describe(args.baseline_params)} over "
              f"[{args.baseline_window[0]:,.0f} – {args.baseline_window[1]:,.0f} Hz]")

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

    if args.parent_window is not None:
        if args.log_plot or args.residuals:
            print("Note: --log-plot / --residuals apply to single-window fits only; "
                  "ignored for the joint parent/daughter fit.")
        run_joint(args, halflife_isomer_lab, halflife_groundstate_lab)
        return

    freq, time, power = load_data(args)
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

    sigma_fit = None
    if args.err_coeffs is not None:
        err_a, err_b = args.err_coeffs
        sigma_fit = err_a * intensity_fit + err_b
        # a, b describe a single frame; averaging N independent frames
        # reduces the error on the mean by √N
        n_avg = args.time_avg if (args.time_avg is not None and args.time_avg > 1) else 1
        sigma_fit = sigma_fit / np.sqrt(n_avg)
        if np.any(sigma_fit <= 0):
            n_bad = int(np.sum(sigma_fit <= 0))
            sys.exit(
                f"Error bars σ = {err_a:g}·I + {err_b:g} are ≤ 0 for {n_bad} point(s) "
                f"(min σ = {sigma_fit.min():.4g}). Adjust the coefficients."
            )
        avg_note = f", ÷√{n_avg} for time averaging" if n_avg > 1 else ""
        sig_str = f"({err_a:g}·I + {err_b:g})/√{n_avg}" if n_avg > 1 else f"{err_a:g}·I + {err_b:g}"
        print(f"Error bars: σ = {sig_str}  "
              f"(range {format_value(sigma_fit.min())} – {format_value(sigma_fit.max())}{avg_note})")

    with_offset = not args.no_offset
    fit_kw = dict(
        with_offset              = with_offset,
        bateman                  = args.bateman,
        halflife_isomer_lab      = halflife_isomer_lab,
        halflife_groundstate_lab = halflife_groundstate_lab,
    )

    err_note = None
    fit_warning = None
    if sigma_fit is not None:
        # Manual error coefficients
        err_note = f"σ = {err_a:g}·I + {err_b:g}" + (f" (÷√{n_avg})" if n_avg > 1 else "") + "  [manual]"
        popt, perr, model, resolved = fit_decay(time_fit, intensity_fit, sigma=sigma_fit, **fit_kw)

    elif args.unweighted:
        popt, perr, model, resolved = fit_decay(time_fit, intensity_fit, **fit_kw)

    else:
        # Pass 1: unweighted fit
        popt1, perr1, model1, resolved1 = fit_decay(time_fit, intensity_fit, **fit_kw)
        # Error model from residuals
        sigma_fit, err_a, err_b, kind = estimate_error_model(time_fit, intensity_fit, popt1, model1)
        if kind == "linear":
            print(f"Error model from residuals: σ = {format_value(err_a)}·I + {format_value(err_b)}  "
                  f"(range {format_value(sigma_fit.min())} – {format_value(sigma_fit.max())})")
            err_note = f"σ = {format_value(err_a)}·I + {format_value(err_b)}  [from residuals]"
        else:
            print(f"Error model from residuals: linear σ(I) went ≤ 0 or was ill-defined; "
                  f"using constant σ = {format_value(err_b)}")
            err_note = f"σ = {format_value(err_b)}  [constant, from residuals]"
        # Pass 2: weighted fit; if it fails, keep the unweighted fit of pass 1
        try:
            popt, perr, model, resolved = fit_decay(time_fit, intensity_fit, sigma=sigma_fit,
                                                    **fit_kw)
        except FitError as exc:
            print(f"WARNING: weighted refit failed ({exc}); showing the unweighted fit of pass 1")
            popt, perr, model, resolved = popt1, perr1, model1, resolved1
            sigma_fit = None
            err_note = "unweighted (weighted refit failed)"
            fit_warning = "Weighted refit failed: unweighted fit shown"
        else:
            if not args.bateman:
                print(f"  Pass 1 (unweighted): τ = {format_time(popt1[1], perr1[1])}")
                print(f"  Pass 2 (weighted):   τ = {format_time(popt[1], perr[1])}")

    record = single_fit_record(args, time_fit, intensity_fit, popt, perr, model, resolved,
                               with_offset, sigma_fit, err_note)
    plot_decay(
        time, intensity,
        time_fit, intensity_fit, popt, perr, model, resolved,
        with_offset, args.bateman,
        args.freq_min, args.freq_max,
        args.file,
        gamma=args.gamma,
        title_suffix=args.title,
        title_full=args.title_full,
        log_plot=args.log_plot,
        show_residuals=args.residuals,
        sigma_fit=sigma_fit,
        err_note=err_note,
        record=record,
        warning=fit_warning,
    )


if __name__ == "__main__":
    try:
        main()
    except FitError as exc:
        sys.exit(str(exc))
