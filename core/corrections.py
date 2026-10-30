"""
2nd-order frequency correction profiles, per experiment.

Each fit of the 2nd-order correction (tartarus/new_2nd_order_correction_plot.py)
can be saved as a named profile with its a, b, c and the setup it was fitted
with.  Tools offer the profiles in a dropdown next to the a/b/c boxes; picking
one fills the boxes.  The profile in use is not stored separately: it is the
profile whose a, b, c equal the boxes (otherwise "custom"), so physics.json and
the tool settings keep only a, b, c as before.

File: data/<experiment>/settings/correction_profiles.json

    {"format": "asphodel-correction-profiles", "version": 1,
     "profiles": [{"name", "a", "b", "c", "a_unc", "b_unc", "c_unc", "created",
                   "fit": {ref_ion, ref_f_exp, ref_harmonic, f_ref_rev, gamma_t,
                           mass_model, weighted, n_points, rms_rev_hz, ions,
                           source_file, sheet}, "notes"}]}

If the file does not exist yet, it is created with one profile holding the
a, b, c currently in physics.json (setup not recorded).
"""
import datetime
import json
import math
import os

from . import paths
from .nubase_reader import MASS_MODEL

PROFILES_FILE = "correction_profiles.json"
FORMAT = "asphodel-correction-profiles"
CUSTOM = "__custom__"            # dropdown value when a, b, c match no profile
CUSTOM_LABEL = "custom (values typed in)"
SEED_NAME = "physics.json values (before profiles)"
REF_SHIFT_WARN_HZ = 1.0          # warn if the correction moves the reference line more


def profiles_path(exp=None):
    return paths.settings_file(PROFILES_FILE, exp)


def _write(profiles, exp=None):
    path = profiles_path(exp)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"format": FORMAT, "version": 1, "profiles": profiles}, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


def _seed(exp=None):
    """First use: one profile with the a, b, c of physics.json."""
    from . import physics                    # physics imports nothing from here
    sh = physics.load_shared(exp)
    prof = {"name": SEED_NAME, "a": float(sh["corr_a"]), "b": float(sh["corr_b"]),
            "c": float(sh["corr_c"]), "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "fit": {}, "notes": "Saved automatically when correction profiles were introduced; "
                                "fit setup not recorded."}
    try:
        _write([prof], exp)
    except OSError:
        pass
    return [prof]


def load_profiles(exp=None):
    """List of profiles (oldest first).  Creates the file on first use."""
    path = profiles_path(exp)
    if not os.path.exists(path):
        return _seed(exp)
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return []
    out = []
    for p in (data.get("profiles") or []) if isinstance(data, dict) else []:
        try:
            out.append(dict(p, name=str(p["name"]), a=float(p["a"]), b=float(p["b"]),
                            c=float(p["c"]), fit=p.get("fit") or {}))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def get_profile(name, exp=None):
    return next((p for p in load_profiles(exp) if p["name"] == name), None)


def save_profile(profile, exp=None, overwrite=False):
    """Add a profile (dict with at least name, a, b, c).  Raises ValueError if
    the name is empty or already used (unless overwrite)."""
    name = str(profile.get("name") or "").strip()
    if not name or name == CUSTOM:
        raise ValueError("the profile needs a name")
    profiles = load_profiles(exp)
    if any(p["name"] == name for p in profiles):
        if not overwrite:
            raise ValueError(f"a profile called '{name}' already exists")
        profiles = [p for p in profiles if p["name"] != name]
    prof = dict(profile, name=name, a=float(profile["a"]), b=float(profile["b"]),
                c=float(profile["c"]))
    prof.setdefault("created", datetime.datetime.now().isoformat(timespec="seconds"))
    profiles.append(prof)
    _write(profiles, exp)
    return prof


def _same(x, y):
    try:
        return math.isclose(float(x), float(y), rel_tol=1e-12, abs_tol=0.0)
    except (TypeError, ValueError):
        return False


def match(a, b, c, exp=None, profiles=None):
    """Name of the (first) profile with these a, b, c, or None."""
    for p in (load_profiles(exp) if profiles is None else profiles):
        if _same(a, p["a"]) and _same(b, p["b"]) and _same(c, p["c"]):
            return p["name"]
    return None


def options(profiles=None, exp=None):
    """Dropdown options: every profile, then 'custom'."""
    profiles = load_profiles(exp) if profiles is None else profiles
    return ([{"label": p["name"], "value": p["name"]} for p in profiles]
            + [{"label": CUSTOM_LABEL, "value": CUSTOM}])


def describe(p):
    """One-line summary of a profile's fit (for tooltips)."""
    f = p.get("fit") or {}
    if not f:
        return p.get("notes") or "fit setup not recorded"
    parts = [f"ref {f.get('ref_ion')} h{f.get('ref_harmonic')} f_exp {f.get('ref_f_exp')} Hz",
             f"γ_t {f.get('gamma_t')}",
             "weighted" if f.get("weighted") else "unweighted",
             f"{f.get('n_points')} ions"]
    if f.get("rms_rev_hz") is not None:
        parts.append(f"RMS {float(f['rms_rev_hz']):.3g} Hz (rev)")
    parts.append(f"masses: {f.get('mass_model')}")
    return ", ".join(parts)


def _num(v):
    try:
        v = float(v)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def warnings(profile, ref_name=None, ref_harmonic=None, gamma_t=None, ref_frequency=None,
             a=None, b=None, c=None):
    """List of warnings for using `profile` with a tool's settings (any of the
    settings may be None = not known in that tool).

    Checks the reference ion, its harmonic, γ_t and the mass model against the
    fit, and (if ref_frequency and a, b, c are known) how far the correction
    moves the tool's reference line: the fit is anchored at its own reference
    frequency, so a different f_ref gets a non-zero correction."""
    if not profile:
        return []
    f = profile.get("fit") or {}
    if not f:
        return []                    # describe() already says the setup is not recorded
    out = []
    if ref_name and f.get("ref_ion") and str(ref_name).strip() != f["ref_ion"]:
        out.append(f"fitted with reference {f['ref_ion']}, not {str(ref_name).strip()}")
    h = _num(ref_harmonic)
    if h is not None and f.get("ref_harmonic") is not None and int(h) != int(f["ref_harmonic"]):
        out.append(f"fitted at reference harmonic {f['ref_harmonic']}, not {int(h)}")
    g = _num(gamma_t)
    if g is not None and f.get("gamma_t") is not None and not math.isclose(g, float(f["gamma_t"]),
                                                                          rel_tol=1e-9):
        out.append(f"fitted with γ_t = {f['gamma_t']}, not {g:g}")
    if f.get("mass_model") and f["mass_model"] != MASS_MODEL:
        out.append(f"fitted with other masses ({f['mass_model']}); the tools now use {MASS_MODEL}")
    fr = _num(ref_frequency)
    vals = [_num(x) for x in (a, b, c)]
    if fr and h and None not in vals:
        x = fr / h
        shift = (vals[0] * x * x + vals[1] * x + vals[2]) * h
        if abs(shift) > REF_SHIFT_WARN_HZ:
            out.append(f"the correction moves the reference line by {shift:+.1f} Hz "
                       f"(fitted at f_ref = {f.get('ref_f_exp')} Hz, here {fr:.0f} Hz)")
    return out
