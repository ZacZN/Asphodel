"""
2nd-order frequency correction profiles, per experiment.

Each fit of the 2nd-order correction (tartarus/new_2nd_order_correction_plot.py)
can be saved as a named profile with its a, b, c and the setup it was fitted
with.  Tools offer the profiles in a dropdown next to the a/b/c boxes; picking
one fills the boxes.

File: data/<experiment>/settings/correction_profiles.json

    {"format": "asphodel-correction-profiles", "version": 2,
     "profiles": [{"name", "a", "b", "c", "a_unc", "b_unc", "c_unc", "created",
                   "fit": {ref_ion, ref_f_exp, ref_harmonic, f_ref_rev, gamma_t,
                           mass_model, weighted, n_points, rms_rev_hz, ions,
                           source_file, sheet},
                   "per_harmonic": {"212": {"a", "b", "c", "a_unc", "b_unc", "c_unc",
                                            "fit": {... as above, plus ref_f_h}}, ...},
                   "notes"}]}

Per-harmonic profiles (version 2, from the GUI correction tool
tartarus/gui_2nd_order_correction.py): every harmonic in "per_harmonic" has
its own a, b, c, fitted with the simulated lines of that harmonic placed from
that harmonic's own reference peak (physics.harmonic_references).  The
profile's top-level a, b, c ("general") are a copy of the reference
harmonic's fit and are used for every harmonic the profile has no own entry
for.  Version-1 profiles (no "per_harmonic") use their a, b, c for every
harmonic, as before.

Tools store the profile *name* (setting "corr_profile"; physics.json and the
tool settings).  "" = old settings: the profile is the one whose a, b, c
equal the a, b, c boxes.  CUSTOM = the a, b, c typed in, for every harmonic.

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
        json.dump({"format": FORMAT, "version": 2, "profiles": profiles}, fh, indent=2)
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
                            c=float(p["c"]), fit=p.get("fit") or {},
                            per_harmonic=_clean_per_h(p.get("per_harmonic"))))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _clean_per_h(d):
    """{"212": {...}} with float a, b, c; entries that cannot be read are dropped."""
    out = {}
    for h, v in (d or {}).items() if isinstance(d, dict) else []:
        try:
            out[str(int(h))] = dict(v, a=float(v["a"]), b=float(v["b"]), c=float(v["c"]))
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
    if not prof.get("per_harmonic"):
        prof.pop("per_harmonic", None)
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


def by_harmonic(profile):
    """{harmonic (int): (a, b, c)} of a per-harmonic profile ({} for version 1)."""
    if not profile:
        return {}
    return {int(h): (v["a"], v["b"], v["c"])
            for h, v in (profile.get("per_harmonic") or {}).items()}


def resolve(name, a=None, b=None, c=None, exp=None, profiles=None):
    """The correction a tool applies, from its stored profile name and its
    a, b, c boxes.  Returns {"name": profile name or CUSTOM, "profile": the
    profile or None, "general": (a, b, c), "by_h": {h: (a, b, c)},
    "missing": True if the stored name is not a profile (any more)}.

    name '' (settings from before profile names were stored): the profile
    whose a, b, c match the boxes, else custom.  CUSTOM: the boxes, for every
    harmonic.  A profile: its own a, b, c (general) and per-harmonic values."""
    profiles = load_profiles(exp) if profiles is None else profiles
    name = "" if name is None else str(name).strip()
    p, missing = None, False
    if name and name != CUSTOM:
        p = next((q for q in profiles if q["name"] == name), None)
        missing = p is None
    elif not name:
        m = match(a, b, c, profiles=profiles)
        p = next((q for q in profiles if q["name"] == m), None) if m else None
    if p is None:
        vals = tuple(_num(x) or 0.0 for x in (a, b, c))
        return {"name": CUSTOM, "profile": None, "general": vals, "by_h": {}, "missing": missing}
    return {"name": p["name"], "profile": p, "general": (p["a"], p["b"], p["c"]),
            "by_h": by_harmonic(p), "missing": False}


def coefficient_lines(p):
    """Short per-harmonic lines for the info boxes: 'h212: a = …, b = …, c = …'."""
    if not p:
        return []
    out = [f"h{h}: a = {v['a']:.6e}, b = {v['b']:.6e}, c = {v['c']:.6e}"
           + (f" (RMS {float(v['fit']['rms_rev_hz']):.3g} Hz, {v['fit'].get('n_points')} ions)"
              if (v.get("fit") or {}).get("rms_rev_hz") is not None else "")
           for h, v in sorted((p.get("per_harmonic") or {}).items(), key=lambda kv: int(kv[0]))]
    gen = f"general (other harmonics): a = {p['a']:.6e}, b = {p['b']:.6e}, c = {p['c']:.6e}"
    return out + [gen] if out else [f"all harmonics: a = {p['a']:.6e}, b = {p['b']:.6e}, "
                                    f"c = {p['c']:.6e}"]


def describe(p):
    """One-line summary of a profile's fit (for tooltips)."""
    f = p.get("fit") or {}
    per_h = sorted(int(h) for h in (p.get("per_harmonic") or {}))
    if per_h:
        hs = ", ".join(f"h{h}" for h in per_h)
        rh = f.get("ref_harmonic")
        other = (f"the h{rh} fit" if rh is not None and int(rh) in per_h else
                 "the general a, b, c" if any(p.get(k) for k in ("a", "b", "c")) else
                 "no correction (general a, b, c = 0)")
        return (f"per-harmonic profile ({hs}; other harmonics use {other}) · "
                f"ref {f.get('ref_ion')} · γ_t {f.get('gamma_t')} · "
                + ("weighted" if f.get("weighted") else "unweighted")
                + f" · masses: {f.get('mass_model')}")
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


# ---------------------------------------------------------------------------
# The fit (shared by tartarus/new_2nd_order_correction_plot.py and the GUI
# correction tool tartarus/gui_2nd_order_correction.py)
# ---------------------------------------------------------------------------

def anchored_quadratic_fit(x, y, x0, sigma=None):
    """Least squares for y = s1*dx + s2*dx^2 (no intercept), dx = x - x0.
    With sigma: weighted by 1/sigma^2.  The covariance is scaled by the
    residual variance (chi^2/ndf) in both cases."""
    import numpy as np
    x, y = np.asarray(x, float), np.asarray(y, float)
    dx = x - x0
    X = np.column_stack([dx, dx ** 2])
    w = np.ones_like(y) if sigma is None else 1.0 / np.asarray(sigma, float)
    Xw, yw = X * w[:, None], y * w
    coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid = y - X @ coef
    dof = len(y) - X.shape[1]
    s2 = np.sum((resid * w) ** 2) / dof if dof > 0 else np.nan
    cov = s2 * np.linalg.inv(Xw.T @ Xw)
    return coef, cov, resid, s2


def expanded(s1, s2, x0, cov):
    """Shifted (s1, s2, x0) -> expanded a, b, c of f_diff = a x^2 + b x + c and
    their propagated uncertainties.  Returns ((a, b, c), (da, db, dc))."""
    import numpy as np
    a, b, c = s2, s1 - 2 * s2 * x0, s2 * x0 ** 2 - s1 * x0
    J = np.array([[0.0, 1.0],            # da/d(s1,s2)
                  [1.0, -2 * x0],        # db/d(s1,s2)
                  [-x0, x0 ** 2]])       # dc/d(s1,s2)
    cov_c = J @ cov @ J.T
    return (float(a), float(b), float(c)), tuple(float(v) for v in np.sqrt(np.diag(cov_c)))


def fit_correction(x, y, x0, sigma=None):
    """Anchored 2nd-order fit of f_diff (y) vs f_sim (x), revolution
    frequencies, through (x0, 0).  The anchor itself must not be in x, y.
    Returns a dict: s1, s2, ds1, ds2, x0, abc, abc_unc, resid, rms, chi2_ndf, n."""
    import numpy as np
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2:
        raise ValueError(f"needs at least 2 points besides the anchor (has {len(x)})")
    (s1, s2), cov, resid, chi2 = anchored_quadratic_fit(x, y, x0, sigma)
    ds1, ds2 = (float(v) for v in np.sqrt(np.abs(np.diag(cov))))
    abc, dabc = expanded(s1, s2, x0, cov)
    return {"s1": float(s1), "s2": float(s2), "ds1": ds1, "ds2": ds2, "x0": float(x0),
            "abc": abc, "abc_unc": dabc, "resid": resid, "n": int(len(x)),
            "rms": float(np.sqrt(np.mean(resid ** 2))), "chi2_ndf": float(chi2)}


def _num(v):
    try:
        v = float(v)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def warnings(profile, ref_name=None, ref_harmonic=None, gamma_t=None, ref_frequency=None,
             a=None, b=None, c=None, harmonics=None):
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
    per_h = profile.get("per_harmonic") or {}
    if per_h and harmonics:
        try:
            other = [int(x) for x in harmonics if str(int(x)) not in per_h]
        except (TypeError, ValueError):
            other = []
        if other:
            out.append("no own correction for " + ", ".join(f"h{x}" for x in other)
                       + ": the profile's general a, b, c is used")
    fr = _num(ref_frequency)
    vals = [_num(x) for x in (a, b, c)]
    if fr and h and None not in vals:
        x = fr / h
        shift = (vals[0] * x * x + vals[1] * x + vals[2]) * h
        if abs(shift) > REF_SHIFT_WARN_HZ:
            out.append(f"the correction moves the reference line by {shift:+.1f} Hz "
                       f"(fitted at f_ref = {f.get('ref_f_exp')} Hz, here {fr:.0f} Hz)")
    return out
