"""
Shared physics for the Asphodel suite.

1. Ion-frequency simulation (moved here from Theia, theia/theia.py; plot_spectrogram.py
   had a port of the same code):  build_ion_database, simulate_frequencies,
   delta_f, apply_correction, half-life helpers and the hover labels.

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


def simulate_frequencies(ion_db, ref_mq, ref_obs_freq, ref_harmonic, gamma_t, harmonics,
                         correction_enabled=False, corr_a=0.0, corr_b=0.0, corr_c=0.0):
    """Vectorised version of viewer.compute_frequencies_multi.

    Returns a DataFrame with one row per (ion, harmonic): ion_row, harmonic, f_sim.
    """
    ref_rev = ref_obs_freq / ref_harmonic
    mq = ion_db["m/q"].to_numpy(dtype=float)
    f_true = ref_rev + delta_f(ref_rev, ref_mq, mq - ref_mq, gamma_t)
    if correction_enabled:
        f_true = apply_correction(f_true, corr_a, corr_b, corr_c)
    rows = np.arange(len(ion_db))
    out = [pd.DataFrame({"ion_row": rows, "harmonic": int(h), "f_sim": f_true * h})
           for h in harmonics]
    return pd.concat(out, ignore_index=True)


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
    "min_half_life":      1.0,
    "min_half_life_unit": "ms",
    "exclude_unknown_hl": False,
}
PHYSICS_KEYS = list(BUILTIN_PHYSICS)

_INT_LIST_KEYS = {"harmonics", "z_lims", "n_lims", "el1", "el2"}
_BOOL_KEYS = {"correction_enabled", "exclude_unknown_hl"}
_INT_KEYS = {"ref_harmonic"}
_FLOAT_KEYS = {"ref_frequency", "gamma_t", "mq_width", "corr_a", "corr_b", "corr_c",
               "min_half_life"}


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
