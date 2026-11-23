"""baseline.py — per-frame baseline removal for ESR spectrograms (core).

Each time frame (row of power, linear scale) gets its own baseline from the
BrPLS method (Bayesian reweighted penalized least squares, Wang et al. 2022)
in pybaselines, fitted over the frequency window it is given; the baseline is
subtracted, so the result can go below zero.

Used by the spectrogram page (over its loaded frequency crop) and by
decay_fit.py (--baseline, over the same window), so the decay fit sees the
same corrected frames as the page.

Results are cached in the temp folder (key: file, its mtime, window,
parameters), so reloading the same view doesn't fit every frame again.
"""
import hashlib
import json
import os
import tempfile
import time

import numpy as np

METHOD = "brpls"
# pybaselines' own defaults for Baseline.brpls
DEFAULTS = {"lam": 1e5, "diff_order": 2, "max_iter": 50, "tol": 1e-3,
            "max_iter_2": 50, "tol_2": 1e-3}
# label, help text: used for the page's settings panel
FIELDS = {
    "lam":        ("λ (smoothness)",
                   "Larger = stiffer, smoother baseline. Scales with the number of "
                   "frequency bins in the window: try 1e4 – 1e8."),
    "diff_order": ("Difference Order",
                   "Order of the difference penalty (2 is usual; 1 = flatter, 3 = curvier)."),
    "max_iter":   ("Max Iterations (inner)", "Maximum reweighting iterations per fit."),
    "tol":        ("Tolerance (inner)", "Stop the inner loop when the change is below this."),
    "max_iter_2": ("Max Iterations (outer)", "Maximum iterations of the outer β update."),
    "tol_2":      ("Tolerance (outer)", "Stop the outer loop when β changes by less than this."),
}
_INTS = ("diff_order", "max_iter", "max_iter_2")
CACHE_DIR = os.path.join(tempfile.gettempdir(), "asphodel_baseline")


def parse_params(raw):
    """Validated BrPLS parameters from a dict (missing keys -> defaults).
    Raises ValueError with a readable message."""
    raw = raw or {}
    out = {}
    for k, default in DEFAULTS.items():
        v = raw.get(k, default)
        if v is None or (isinstance(v, str) and not v.strip()):
            v = default
        try:
            v = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"{FIELDS[k][0]}: not a number ({v!r})")
        if not np.isfinite(v) or v <= 0:
            raise ValueError(f"{FIELDS[k][0]} must be > 0")
        if k in _INTS:
            if v != int(v):
                raise ValueError(f"{FIELDS[k][0]} must be a whole number")
            v = int(v)
        out[k] = v
    if out["diff_order"] > 5:
        raise ValueError("Difference order must be 1 – 5")
    return out


def _fitter(freq):
    try:
        from pybaselines import Baseline
    except ImportError:
        raise ImportError("baseline removal needs pybaselines: pip install pybaselines")
    return Baseline(x_data=np.asarray(freq, dtype=float)).brpls


def remove(freq, power, params, progress=True):
    """power (T, F) linear -> power minus each frame's BrPLS baseline (float64).
    freq (F,) is the window the baseline is fitted over."""
    p = parse_params(params)
    fit = _fitter(freq)
    power = np.asarray(power, dtype=float)
    out = np.empty_like(power)
    n = power.shape[0]
    t0 = time.time()
    for i in range(n):
        bl, _ = fit(power[i], **p)
        out[i] = power[i] - bl
        if progress and (i + 1) % 100 == 0:
            print(f"  BrPLS baseline: {i + 1}/{n} frames ({time.time() - t0:.0f} s)", flush=True)
    if progress:
        print(f"BrPLS baseline removed from {n} frames × {power.shape[1]} bins "
              f"in {time.time() - t0:.1f} s  ({describe(p)})", flush=True)
    return out


def _cache_path(file_path, window, p):
    st = os.stat(file_path)
    key = json.dumps([METHOD, os.path.abspath(file_path), st.st_mtime_ns, st.st_size,
                      [float(window[0]), float(window[1])], p], sort_keys=True)
    return os.path.join(CACHE_DIR, hashlib.sha1(key.encode()).hexdigest()[:20] + ".npy")


def remove_cached(file_path, freq, power, window, params, progress=True):
    """remove(), reusing a cached result for the same file, window and
    parameters. window = (fmin, fmax) of freq, as requested."""
    p = parse_params(params)
    cp = None
    try:
        cp = _cache_path(file_path, window, p)
        arr = np.load(cp)
        if arr.shape == np.shape(power):
            if progress:
                print(f"BrPLS baseline: reused cached result ({describe(p)})", flush=True)
            return arr
    except (OSError, ValueError):
        pass
    out = remove(freq, power, p, progress)
    if cp:
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            tmp = cp + f".{os.getpid()}.tmp.npy"
            np.save(tmp, out)
            os.replace(tmp, cp)
        except OSError:
            pass
    return out


def describe(params):
    p = parse_params(params)
    return (f"BrPLS λ={p['lam']:g}, order {p['diff_order']}, "
            f"iter {p['max_iter']}/{p['max_iter_2']}, tol {p['tol']:g}/{p['tol_2']:g}")
