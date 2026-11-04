"""
Shared physics for the Asphodel suite.

1. Ion-frequency simulation (moved here from Theia, theia/theia.py; plot_spectrogram.py
   had a port of the same code):  build_ion_database, simulate_frequencies,
   delta_f, apply_correction, half-life helpers and the hover labels.
   Every harmonic is placed from the reference ion's own peak in that
   harmonic (harmonic_references, window ref_search_hz) and gets its own
   2nd-order correction if the profile has one (simulate_spectrum,
   resolve_correction).

2. Shared physics settings per experiment, in data/<exp>/settings/physics.json.
   They are the defaults for every tool; a tool's own settings file keeps only
   the values that differ from them (its overrides):

       cfg = physics.with_shared(TOOL_DEFAULTS, saved)     # when loading
       saved = physics.strip_shared(cfg)                   # before saving

   A value set back to the shared default stops being an override, and a
   change to physics.json reaches every tool that has not overridden it.
"""

import json
import math
import os
import re
from functools import lru_cache

import numpy as np
import pandas as pd

from . import paths
from .nubase_reader import df_nubase, rect_restrict, diag_restrict, index_by_name


# ===========================================================================
# Simulation
# ===========================================================================

def delta_f(ref_frequency, ref_mq, delta_mq, gamma_t):
    return (-1 / (gamma_t**2)) * (delta_mq / ref_mq) * ref_frequency


def apply_correction(freq, a, b, c):
    """Return freq + a*freq**2 + b*freq + c (second-order polynomial correction)."""
    return freq + a * freq**2 + b * freq + c


def ion_base_name(ion_name: str) -> str:
    """'204Hg79+' -> '204Hg',  'm186Hf72+' -> 'm186Hf'"""
    return re.sub(r"\d+\+$", "", ion_name)


def build_hover_label(ion_name: str, row: "pd.Series") -> str:
    """Build a rich hover label for an ion frequency line.

    Always shows the ion name and its lifetime (T ± T_unc T_unit).
    For isomers (name begins with a letter, e.g. ``m186Hf72+``) also shows
    the excitation energy (E_x ± E_x_unc keV).
    Values containing '#' are theoretical and the symbol is preserved as-is.
    """
    def _val(key: str) -> str:
        """Return a stripped string for a cell, or '' if missing/NaN."""
        v = row.get(key, None)
        if v is None:
            return ""
        try:
            if pd.isna(v):
                return ""
        except (TypeError, ValueError):
            pass
        return str(v).strip()

    def _fmt_float(v) -> str:
        """Format a float without a spurious trailing '.0'."""
        try:
            if v == int(v):
                return str(int(v))
            return f"{v:.3f}".rstrip("0").rstrip(".")
        except (ValueError, OverflowError):
            return str(v)

    lines = [ion_name]

    # --- Mass excess ---
    # nubase.py strips '#' from m.e./m.e._unc and converts to float, but
    # records whether the original value was theoretical in 'm.e. sys?'.
    me_sys     = bool(row.get("m.e. sys?", False))
    me_raw     = row.get("m.e.", None)
    me_unc_raw = row.get("m.e._unc", None)
    me_ok      = (me_raw is not None and not pd.isna(me_raw))
    if me_ok:
        prefix  = "#" if me_sys else ""
        me_line = f"m.e. = {_fmt_float(me_raw)}{prefix}"
        me_unc_ok = (me_unc_raw is not None and not pd.isna(me_unc_raw))
        if me_unc_ok:
            me_line += f" ± {_fmt_float(me_unc_raw)}{prefix}"
        me_line += " keV"
        lines.append(me_line)

    # --- Lifetime ---
    t_val  = _val("T")
    t_unc  = _val("T_unc")
    t_unit = _val("T_unit")
    if t_val:
        t_line = f"T = {t_val}"
        if t_unc:
            t_line += f" ± {t_unc}"
        if t_unit:
            t_line += f" {t_unit}"
        lines.append(t_line)

    # --- Excitation energy (isomers only) ---
    # An isomer's index name begins with a letter before the mass number
    if ion_name and ion_name[0].isalpha():
        ex_val = _val("E_x")
        ex_unc = _val("E_x_unc")
        if ex_val:
            ex_line = f"E_x = {ex_val}"
            if ex_unc:
                ex_line += f" ± {ex_unc}"
            ex_line += " keV"
            lines.append(ex_line)

    # --- Decay modes / branching ratios ---
    br = _val("BR")
    if br:
        lines.append(f"Decay: {br}")

    return "<br>".join(lines)


# NUBASE half-life units -> seconds
HL_UNITS = {"ys": 1e-24, "zs": 1e-21, "as": 1e-18, "fs": 1e-15, "ps": 1e-12, "ns": 1e-9,
            "us": 1e-6, "ms": 1e-3, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0,
            "y": 3.15576e7, "ky": 3.15576e10, "My": 3.15576e13, "Gy": 3.15576e16,
            "Ty": 3.15576e19, "Py": 3.15576e22, "Ey": 3.15576e25, "Zy": 3.15576e28,
            "Yy": 3.15576e31}
# units offered for the half-life cut in the GUI
HL_CUT_UNITS = ["ns", "us", "ms", "s", "m", "h", "d", "y"]


def half_life_seconds(t, unit):
    """NUBASE half-life (T, T_unit) -> seconds.  'stbl' -> inf.  Estimates
    ('#') and limits ('>', '<', '~') use the number given.  Unknown (blank,
    'p-unst', or an unrecognised unit) -> NaN."""
    t = "" if t is None or (isinstance(t, float) and np.isnan(t)) else str(t).strip()
    if t.startswith("stbl"):
        return np.inf
    num = re.sub(r"[#<>~=\s]", "", t)
    unit = "" if unit is None or (isinstance(unit, float) and np.isnan(unit)) else str(unit).strip()
    try:
        return float(num) * HL_UNITS[unit]
    except (ValueError, KeyError):
        return np.nan


def hl_allowed(ion_db, s):
    """Bool array over ion_db rows: True = the ion may be identified.

    Ions whose half-life is below the cut (min_half_life x unit) are excluded;
    ions with an unknown half-life are excluded only if exclude_unknown_hl.
    Their simulated lines are still drawn, in a separate style."""
    thr = float(s.get("min_half_life") or 0)
    if thr <= 0 or "half_life_s" not in ion_db:
        return np.ones(len(ion_db), dtype=bool)
    thr *= HL_UNITS.get(s.get("min_half_life_unit") or "s", 1.0)
    hl = ion_db["half_life_s"].to_numpy(dtype=float)
    known = ~np.isnan(hl)
    ok = np.zeros(len(hl), dtype=bool)
    ok[known] = hl[known] >= thr
    if not s.get("exclude_unknown_hl"):
        ok[~known] = True
    return ok


def hl_cut_text(s):
    """Short description of the half-life cut, '' if off."""
    thr = float(s.get("min_half_life") or 0)
    if thr <= 0:
        return ""
    return (f"T\u00bd < {thr:g} {s.get('min_half_life_unit') or 's'}"
            + (" or unknown" if s.get("exclude_unknown_hl") else ""))


@lru_cache(maxsize=8)
def build_ion_database(z_lims, n_lims, el1, el2, ref_name, mq_width):
    """Bare, H-, He- and Li-like ions inside the selection, restricted to the
    m/q window around the reference ion (as in viewer.py).  Cached."""
    frames = []
    for electrons in (0, 1, 2, 3):
        d = df_nubase(electrons=electrons)
        d = rect_restrict(df=d, z_lims=list(z_lims), n_lims=list(n_lims))
        d = diag_restrict(df=d, el1=list(el1), el2=list(el2))
        d = index_by_name(df=d, electrons=electrons)
        d["electrons"] = electrons
        d["charge"] = d["ZZZi"] - electrons
        frames.append(d)
    db = pd.concat(frames).sort_values("m/q", ascending=False)

    if ref_name not in db.index:
        raise ValueError(f"Reference ion '{ref_name}' is not in the ion selection "
                         f"(check Z/N limits, el1/el2).")
    mq_ref = db.loc[ref_name, "m/q"]
    if isinstance(mq_ref, pd.Series):
        mq_ref = mq_ref.iloc[0]
    db = db[(db["m/q"] >= mq_ref - mq_width) & (db["m/q"] <= mq_ref + mq_width)].copy()
    db["ion"] = db.index.astype(str)
    db["ion_base"] = db["ion"].map(ion_base_name)
    db["half_life_s"] = [half_life_seconds(t, u) for t, u in zip(db["T"], db["T_unit"])]
    db["hover"] = [build_hover_label(name, row) for name, (_, row) in zip(db["ion"], db.iterrows())]
    return db.reset_index(drop=True), float(mq_ref)


def _corr_for(h, corr_by_h, general):
    """(a, b, c) for harmonic h: its own entry in corr_by_h (keys int or str), else general."""
    if corr_by_h:
        v = corr_by_h.get(int(h), corr_by_h.get(str(int(h))))
        if v is not None:
            return tuple(float(x) for x in v)
    return general


def simulate_frequencies(ion_db, ref_mq, ref_obs_freq, ref_harmonic, gamma_t, harmonics,
                         correction_enabled=False, corr_a=0.0, corr_b=0.0, corr_c=0.0,
                         ref_by_h=None, corr_by_h=None):
    """Vectorised version of viewer.compute_frequencies_multi.

    ref_by_h: {harmonic: observed frequency of the reference ion at that
    harmonic} (from harmonic_references); a harmonic not in it is placed from
    ref_obs_freq / ref_harmonic, as before.  Each harmonic's lines are
    simulated from its own reference revolution frequency.
    corr_by_h: {harmonic: (a, b, c)}; harmonics not in it use corr_a, corr_b, corr_c.

    Returns a DataFrame with one row per (ion, harmonic): ion_row, harmonic, f_sim.
    """
    mq = ion_db["m/q"].to_numpy(dtype=float)
    rows = np.arange(len(ion_db))
    general = (corr_a, corr_b, corr_c)
    out = []
    for h in harmonics:
        h = int(h)
        f_h = (ref_by_h or {}).get(h)
        ref_rev = (float(f_h) / h) if f_h is not None else ref_obs_freq / ref_harmonic
        f_true = ref_rev + delta_f(ref_rev, ref_mq, mq - ref_mq, gamma_t)
        if correction_enabled:
            f_true = apply_correction(f_true, *_corr_for(h, corr_by_h, general))
        out.append(pd.DataFrame({"ion_row": rows, "harmonic": h, "f_sim": f_true * h}))
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------
# Reference peak of every harmonic
# ---------------------------------------------------------------------------
# The reference ion's line at harmonic h is looked for in the spectrum near
# f_ref / h_ref * h: the strongest bin within ±window, refined with a 3-point
# parabola.  It counts as found only if that bin is a local maximum inside
# the window (a maximum on the window edge is the flank of something else).
# Not found -> the predicted position is used, with a warning.

def parabolic_peak(ff, pp, i):
    """Sub-bin position of the maximum at bin i (3-point parabola); ff[i] at an edge."""
    if i <= 0 or i >= len(pp) - 1:
        return float(ff[i])
    y0, y1, y2 = float(pp[i - 1]), float(pp[i]), float(pp[i + 1])
    den = y0 - 2 * y1 + y2
    if not np.isfinite(den) or den >= 0:
        return float(ff[i])
    d = 0.5 * (y0 - y2) / den                     # in bins, |d| <= 0.5
    step = 0.5 * (float(ff[i + 1]) - float(ff[i - 1]))
    return float(ff[i]) + float(np.clip(d, -0.5, 0.5)) * step


def peak_near(ff, pp, centre, window_hz, climb=False, max_travel_hz=None):
    """(frequency, bin) of a peak near centre, or (None, None).  ff must be sorted.

    climb=False: the strongest bin within centre ± window_hz (the window is
    widened to at least ±3 bins); a maximum on the window edge does not
    count (it is the flank of something outside the window).
    climb=True: the local maximum reached by walking uphill bin by bin from
    the bin at centre, i.e. the top of the peak or bump the position is on
    (a small bump on the flank of a big peak is kept, not passed over).
    Gives up beyond max_travel_hz from centre (default 10 windows)."""
    if ff is None or pp is None or len(ff) < 3 or window_hz <= 0:
        return None, None
    pp = np.asarray(pp, dtype=float)
    n = len(pp)
    k = int(np.clip(np.searchsorted(ff, centre), 1, n - 1))
    if climb:
        i = k if abs(ff[k] - centre) <= abs(ff[k - 1] - centre) else k - 1
        if not np.isfinite(pp[i]):
            return None, None
        limit = float(max_travel_hz) if max_travel_hz is not None else 10.0 * float(window_hz)
        while True:
            left = pp[i - 1] if i > 0 and np.isfinite(pp[i - 1]) else -np.inf
            right = pp[i + 1] if i < n - 1 and np.isfinite(pp[i + 1]) else -np.inf
            if left <= pp[i] and right <= pp[i]:
                break                            # local maximum
            i = i - 1 if left > right else i + 1
            if abs(float(ff[i]) - centre) > limit:
                return None, None
        if i == 0 or i == n - 1:                 # spectrum edge, not a peak
            return None, None
        return parabolic_peak(ff, pp, i), i
    # at least ±3 bins: with 2^17 bins over 10 MHz a bin is 76 Hz, so ±100 Hz
    # alone holds only 2-3 bins and the maximum is nearly always on the edge
    window_hz = max(float(window_hz), 3.0 * abs(float(ff[k]) - float(ff[k - 1])))
    lo = int(np.searchsorted(ff, centre - window_hz, side="left"))
    hi = int(np.searchsorted(ff, centre + window_hz, side="right"))
    if hi - lo < 3:
        return None, None
    seg = pp[lo:hi]
    if not np.isfinite(seg).any():
        return None, None
    i = lo + int(np.nanargmax(seg))
    if i == lo or i == hi - 1:                   # on the window edge: no peak here
        return None, None
    return parabolic_peak(ff, pp, i), i


def peak_search_explain(ff, pp, centre, window_hz):
    """Why peak_near(..., climb=False) did or did not find a peak at centre.

    Returns {"lo_hz", "hi_hz" (window actually searched, after widening to
    >= 3 bins), "window_hz", "bin_hz", "n_bins", "f_max", "p_max" (strongest
    bin in the window, or None), "edge" ("low" / "high" / None), "found"
    (frequency or None), "reason" (text)}.  ff must be sorted."""
    out = {"lo_hz": None, "hi_hz": None, "window_hz": float(window_hz or 0), "bin_hz": None,
           "n_bins": 0, "f_max": None, "p_max": None, "edge": None, "found": None, "reason": ""}
    if ff is None or pp is None or len(ff) < 3:
        out["reason"] = "no spectrum"
        return out
    if not window_hz or window_hz <= 0:
        out["reason"] = "search off (window 0)"
        return out
    ff = np.asarray(ff, dtype=float)
    pp = np.asarray(pp, dtype=float)
    n = len(pp)
    k = int(np.clip(np.searchsorted(ff, centre), 1, n - 1))
    bin_hz = abs(float(ff[k]) - float(ff[k - 1]))
    w = max(float(window_hz), 3.0 * bin_hz)
    lo = int(np.searchsorted(ff, centre - w, side="left"))
    hi = int(np.searchsorted(ff, centre + w, side="right"))
    out.update(lo_hz=float(centre - w), hi_hz=float(centre + w), window_hz=w, bin_hz=bin_hz,
               n_bins=max(hi - lo, 0))
    if centre - w > ff[-1] or centre + w < ff[0]:
        out["reason"] = (f"the search window ({centre - w:,.1f} – {centre + w:,.1f} Hz) is outside "
                         f"the spectrum ({ff[0]:,.1f} – {ff[-1]:,.1f} Hz)")
        return out
    if hi - lo < 3:
        out["reason"] = f"only {hi - lo} bin(s) of the spectrum in the search window (need 3)"
        return out
    seg = pp[lo:hi]
    if not np.isfinite(seg).any():
        out["reason"] = "no valid (finite) power values in the search window"
        return out
    i = lo + int(np.nanargmax(seg))
    out.update(f_max=float(ff[i]), p_max=float(pp[i]))
    if i == lo or i == hi - 1:
        side = "low" if i == lo else "high"
        out["edge"] = side
        out["reason"] = (f"the strongest bin in ±{w:.1f} Hz is on the {side}-frequency edge of the "
                         f"window ({ff[i]:,.1f} Hz, {ff[i] - centre:+.1f} Hz from expected): the power "
                         f"is still rising {'below' if side == 'low' else 'above'} the window, so it is "
                         f"the flank of something outside it, not a peak inside")
        if lo == 0 or hi == n:
            out["reason"] += " (the window reaches the end of the spectrum)"
        return out
    out["found"] = parabolic_peak(ff, pp, i)
    out["reason"] = (f"peak found at {out['found']:,.1f} Hz ({out['found'] - centre:+.1f} Hz from "
                     f"expected)")
    return out


def harmonic_references(ff, pp, ref_obs_freq, ref_harmonic, harmonics, window_hz):
    """Observed frequency of the reference ion at every harmonic.

    Returns ({h: frequency}, {h: info}).  The reference harmonic is
    ref_obs_freq itself.  Other harmonics: peak_near(f_ref/h_ref*h, window).
    info: {"predicted", "f", "found" (True / False / None = search off)}.
    window_hz <= 0 or no spectrum: search off, everything predicted (old
    behaviour)."""
    refs, info = {}, {}
    ref_harmonic = int(ref_harmonic)
    search = window_hz is not None and float(window_hz) > 0 and ff is not None and pp is not None
    if search:
        order = None
        if np.any(np.diff(ff) < 0):
            order = np.argsort(ff)
            ff, pp = np.asarray(ff)[order], np.asarray(pp)[order]
    for h in harmonics:
        h = int(h)
        pred = ref_obs_freq / ref_harmonic * h
        if h == ref_harmonic:
            refs[h] = float(ref_obs_freq)
            info[h] = {"predicted": float(pred), "f": float(ref_obs_freq), "found": True}
            continue
        if not search:
            refs[h] = float(pred)
            info[h] = {"predicted": float(pred), "f": float(pred), "found": None}
            continue
        f, _ = peak_near(ff, pp, pred, float(window_hz))
        refs[h] = float(f) if f is not None else float(pred)
        info[h] = {"predicted": float(pred), "f": refs[h], "found": f is not None}
    return refs, info


def reference_warnings(info, ref_name, window_hz):
    """Warnings for the harmonics whose reference peak was not found."""
    return [f"h{h}: no {ref_name} peak within ±{float(window_hz):g} Hz of {d['predicted']:.1f} Hz; "
            f"the predicted position is used" for h, d in sorted(info.items()) if d["found"] is False]


REF_STATUS_LABEL = {"reference": "reference", "found": "found", "not_found": "not found",
                    "off": "search off"}


def reference_rows(info, ref_harmonic):
    """Rows for a reference-peak table, one per harmonic (sorted):
    {"h", "status" (reference / found / not_found / off), "label", "predicted",
    "f", "offset" (f - predicted, None unless found)}."""
    rows = []
    for h, d in sorted(info.items(), key=lambda kv: int(kv[0])):
        h = int(h)
        if h == int(ref_harmonic):
            st = "reference"
        elif d.get("found") is True:
            st = "found"
        elif d.get("found") is False:
            st = "not_found"
        else:
            st = "off"
        off = float(d["f"]) - float(d["predicted"]) if st == "found" else None
        rows.append({"h": h, "status": st, "label": REF_STATUS_LABEL[st],
                     "predicted": float(d["predicted"]), "f": float(d["f"]), "offset": off})
    return rows


def simulate_spectrum(ion_db, ref_mq, ref_obs_freq, s, ff=None, pp=None):
    """simulate_frequencies for a tool's parsed settings s, with the
    reference searched in every harmonic of the spectrum (ff, pp) and the
    per-harmonic correction.

    s: ref_name, ref_harmonic, gamma_t, harmonics, correction_enabled,
    corr_a/b/c (general), corr_by_h ({h: (a, b, c)}, optional),
    ref_search_hz (optional; 0 / missing = search off).
    Returns (sims, info, warnings)."""
    window = float(s.get("ref_search_hz") or 0)
    refs, info = harmonic_references(ff, pp, ref_obs_freq, s["ref_harmonic"], s["harmonics"],
                                     window)
    sims = simulate_frequencies(ion_db, ref_mq, ref_obs_freq, s["ref_harmonic"], s["gamma_t"],
                                s["harmonics"], s["correction_enabled"],
                                s["corr_a"], s["corr_b"], s["corr_c"],
                                ref_by_h=refs, corr_by_h=s.get("corr_by_h"))
    return sims, info, reference_warnings(info, s.get("ref_name", "reference"), window)


def resolve_correction(s, exp=None):
    """Fill s["corr_a/b/c"] (general) and s["corr_by_h"] ({"h": [a, b, c]},
    JSON-safe) from s["corr_profile"] and the a, b, c boxes.  Returns s and a
    list of warnings (e.g. the stored profile no longer exists)."""
    from . import corrections                # corrections imports physics lazily
    r = corrections.resolve(s.get("corr_profile"), s.get("corr_a"), s.get("corr_b"),
                            s.get("corr_c"), exp=exp)
    s["corr_profile"] = r["name"]
    s["corr_a"], s["corr_b"], s["corr_c"] = (float(x) for x in r["general"])
    s["corr_by_h"] = {str(h): [float(x) for x in v] for h, v in r["by_h"].items()}
    warn = []
    if r["missing"]:
        warn.append("the stored correction profile no longer exists; the a, b, c boxes are used")
    return s, warn


# ===========================================================================
# Shared physics settings (data/<exp>/settings/physics.json)
# ===========================================================================

PHYSICS_FILE = "physics.json"

# Canonical form of every shared setting, used when physics.json is missing
# or lacks a key (values of the E0203 208Pb80+ setting).
BUILTIN_PHYSICS = {
    "ref_name":           "208Pb80+",
    "ref_frequency":      412391232.0,
    "ref_harmonic":       214,
    "gamma_t":            1.399,
    "harmonics":          [212, 213, 214],
    "z_lims":             [60, 83],
    "n_lims":             [100, 127],
    "el1":                [156, 66],
    "el2":                [200, 82],
    "mq_width":           0.2,
    "correction_enabled": True,
    "corr_a":             2.8498393862845746e-07,
    "corr_b":             -1.0910358215490124,
    "corr_c":             1044186.4522976408,
    # correction profile name (core/corrections.py); "" = the profile matching
    # corr_a/b/c (settings from before names were stored)
    "corr_profile":       "",
    # ± window (Hz) for the reference ion's peak in the other harmonics
    # (harmonic_references); 0 = off (one reference for all harmonics)
    "ref_search_hz":      100.0,
    "min_half_life":      1.0,
    "min_half_life_unit": "ms",
    "exclude_unknown_hl": False,
}
PHYSICS_KEYS = list(BUILTIN_PHYSICS)

_INT_LIST_KEYS = {"harmonics", "z_lims", "n_lims", "el1", "el2"}
_BOOL_KEYS = {"correction_enabled", "exclude_unknown_hl"}
_INT_KEYS = {"ref_harmonic"}
_FLOAT_KEYS = {"ref_frequency", "gamma_t", "mq_width", "corr_a", "corr_b", "corr_c",
               "min_half_life", "ref_search_hz"}


def canonical(key, value):
    """Any tool's representation of a shared setting -> its canonical form.

    "60, 83" / [60, 83] -> [60, 83];  ["on"] / [] / True / "1" -> bool;
    "0.2" -> 0.2;  "214" -> 214.  Raises ValueError if it cannot be read.
    """
    if key in _BOOL_KEYS:
        if isinstance(value, (list, tuple)):
            return len(value) > 0
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "on", "yes")
        return bool(value)
    if key in _INT_LIST_KEYS:
        items = value if isinstance(value, (list, tuple)) else re.split(r"[,\s]+", str(value).strip())
        return [int(float(x)) for x in items if str(x).strip() != ""]
    if key in _INT_KEYS:
        return int(float(value))
    if key in _FLOAT_KEYS:
        return float(value)
    return str(value).strip()


def _same(key, a, b):
    """True if two values of a shared setting mean the same (in any representation)."""
    try:
        ca, cb = canonical(key, a), canonical(key, b)
    except (TypeError, ValueError):
        return False
    if key in _FLOAT_KEYS:
        return math.isclose(ca, cb, rel_tol=1e-12, abs_tol=0.0) or ca == cb
    return ca == cb


def _fmt_num(v):
    """Number as text, without a spurious '.0' (412391232.0 -> '412391232')."""
    if isinstance(v, float) and v.is_integer() and abs(v) < 1e16:
        return str(int(v))
    return repr(v)


def as_like(key, value, like):
    """Canonical `value` written in the same representation as `like` (a tool's
    own default for that key): checklist ['on']/[], bool, text or number."""
    if key in _BOOL_KEYS:
        if isinstance(like, list):
            return ["on"] if value else []
        if isinstance(like, str):
            return "1" if value else ""
        return bool(value)
    if isinstance(like, str):
        if key in _INT_LIST_KEYS:
            return ", ".join(str(int(x)) for x in value)
        return _fmt_num(value) if isinstance(value, (int, float)) else str(value)
    if isinstance(like, list) and key in _INT_LIST_KEYS:
        return [int(x) for x in value]
    if isinstance(like, bool):
        return bool(value)
    if isinstance(like, int) and isinstance(value, (int, float)):
        return int(value) if float(value).is_integer() else float(value)
    return value


def shared_path(exp=None):
    return paths.settings_file(PHYSICS_FILE, exp)


def load_shared(exp=None):
    """Shared physics settings (canonical form) for the experiment."""
    out = dict(BUILTIN_PHYSICS)
    try:
        with open(shared_path(exp), encoding="utf-8") as fh:
            saved = json.load(fh)
    except (OSError, ValueError):
        saved = {}
    if isinstance(saved, dict):
        for k, v in saved.items():
            if k in BUILTIN_PHYSICS:
                try:
                    out[k] = canonical(k, v)
                except (TypeError, ValueError):
                    pass                      # unreadable entry: keep the built-in value
    return out


def save_shared(values, exp=None):
    """Write the shared physics settings (any representation is accepted)."""
    out = load_shared(exp)
    for k, v in values.items():
        if k in BUILTIN_PHYSICS:
            out[k] = canonical(k, v)
    path = shared_path(exp)
    tmp = path + ".tmp"
    text = json.dumps(out, indent=2)
    # short lists on one line ("harmonics": [212, 213, 214]) for hand editing
    text = re.sub(r"\[\s*([^\[\]{}]*?)\s*\]",
                  lambda m: "[" + ", ".join(x.strip() for x in m.group(1).split(",")) + "]", text)
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    os.replace(tmp, path)


def with_shared(tool_defaults, saved, exp=None):
    """A tool's full settings: its own defaults, then the shared physics values
    (for the shared keys the tool has), then its saved overrides."""
    shared = load_shared(exp)
    out = dict(tool_defaults)
    for k in tool_defaults:
        if k in shared:
            out[k] = as_like(k, shared[k], tool_defaults[k])
    out.update(saved or {})
    return out


def strip_shared(cfg, exp=None):
    """Copy of `cfg` without the shared keys whose value equals the shared
    default, i.e. only the tool's own overrides of the shared physics remain."""
    shared = load_shared(exp)
    return {k: v for k, v in cfg.items() if not (k in shared and _same(k, v, shared[k]))}
