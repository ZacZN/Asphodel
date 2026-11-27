"""
Energy calculation from frequency differences (shared by Theia's "Energy
Calculation" box and the spectrogram's Energy Calculation panel).

No GUI code here.  Moved out of theia/gui/energy.py in Nov 2026.

1. Calibration: pick two peaks (a fitted peak's centre is used if you click
   on one, otherwise the peak is fitted on its own in a small window), name
   them (NUBASE names, e.g. 207Tl81+ or m207Tl), press Calibrate:
       k = (M2 - M1) / (f2 - f1)          [keV per Hz]
   M = ion mass (mass excess - q m_e c^2 + binding energy of the q removed
   electrons, NIST) when the names carry charges; atomic mass excess otherwise.
   For an isomer / ground-state pair of the same nuclide the nuclear part is
   the excitation-energy difference (with its own, smaller uncertainty).
2. Then pick peaks in pairs A, B:  dE = E(B) - E(A) = k (f_B - f_A).

A "pick" is a dict {f, df, fwhm, dfwhm, src} (Hz); the energy state is
{calib: [pick, pick], k, dk, pending: pick | None, pairs: [...], msg, ...}.
"""

import re
from functools import lru_cache

import numpy as np
import pandas as pd

from core.nubase_reader import get_base_dataframe, ion_electron_energy_kev
from core.peakfit import _region_mask, run_fit, _snap_peak


@lru_cache(maxsize=1)
def _nubase_table():
    d = get_base_dataframe()
    d = d.copy()
    for c in ("E_x", "E_x_unc"):
        d[c] = pd.to_numeric(d[c].astype(str).str.replace("#", "", regex=False), errors="coerce")
    return d


def lookup_mass_excess(name):
    """NUBASE entry for an ion/nuclide name ('207Tl81+', 'm207Tl', '208Pb').
    Returns dict(name, A, Z, iso, me, dme, ex, dex, sys) or raises ValueError."""
    m = re.fullmatch(r"\s*([a-z]?)(\d+)([A-Z][a-z]?)(\d+\+)?\s*", name or "")
    if not m:
        raise ValueError(f"'{name}' is not a name like 207Tl81+, m207Tl or 208Pb")
    iso, A, sym = m.group(1), m.group(2), m.group(3)
    d = _nubase_table()
    rows = d[d["Element"].astype(str).str.strip() == f"{A}{sym}"]
    rows = rows[rows["Isomer_Flag"] == iso] if iso else rows[rows["Isomer_Flag"].isna()]
    if not len(rows):
        raise ValueError(f"'{iso}{A}{sym}' is not in NUBASE")
    r = rows.iloc[0]
    if pd.isna(r["m.e."]):
        raise ValueError(f"no mass excess for '{iso}{A}{sym}' in NUBASE")
    ex = 0.0 if not iso else (float(r["E_x"]) if pd.notna(r["E_x"]) else np.nan)
    dex = 0.0 if not iso else (float(r["E_x_unc"]) if pd.notna(r["E_x_unc"]) else np.nan)
    return dict(name=f"{iso}{A}{sym}", A=int(A), Z=int(r["ZZZi"]), iso=iso,
                me=float(r["m.e."]), dme=float(r["m.e._unc"]) if pd.notna(r["m.e._unc"]) else np.nan,
                ex=ex, dex=dex, sys=bool(r["m.e. sys?"]))


def _name_charge(name):
    """Charge from an ion name ('207Tl81+' -> 81), None if no charge is given."""
    m = re.fullmatch(r"\s*[a-z]?\d+[A-Z][a-z]?(\d+)\+\s*", name or "")
    return int(m.group(1)) if m else None


def mass_excess_difference(n1, n2):
    """Mass difference E(n2) - E(n1) in keV with its uncertainty.

    Nuclear part: NUBASE mass excesses, or for two states of the same nuclide
    the excitation energies (correlated ME errors).  If both names carry a
    charge (207Tl81+), the ion masses are used: each side adds
    -q m_e c^2 + S(Z, q), S = binding energy of the q removed electrons (NIST
    ionization energies).  Without charges the atomic difference is returned."""
    a, b = lookup_mass_excess(n1), lookup_mass_excess(n2)
    same = (a["A"], a["Z"]) == (b["A"], b["Z"]) and np.isfinite(a["ex"]) and np.isfinite(b["ex"])
    if same:
        de, dde, how = b["ex"] - a["ex"], float(np.hypot(a["dex"], b["dex"])), "excitation energies"
    else:
        de, dde, how = b["me"] - a["me"], float(np.hypot(a["dme"], b["dme"])), "mass excesses"
    qa, qb = _name_charge(n1), _name_charge(n2)
    if qa is None or qb is None:
        return de, dde, a, b, how + " (atomic: give both charges to include the electrons)"
    for r, q in ((a, qa), (b, qb)):
        if not 1 <= q <= r["Z"]:
            raise ValueError(f"charge {q}+ is not possible for {r['name']} (Z = {r['Z']})")
    ea, dea = ion_electron_energy_kev(a["Z"], qa)
    eb, deb = ion_electron_energy_kev(b["Z"], qb)
    if not (np.isfinite(ea) and np.isfinite(eb)):
        raise ValueError("no NIST ionization energies for one of the ions")
    a = dict(a, name=f"{a['name']}{qa}+")
    b = dict(b, name=f"{b['name']}{qb}+")
    return (de + float(eb - ea), float(np.sqrt(dde ** 2 + dea ** 2 + deb ** 2)), a, b,
            "ion masses (" + how + ", electron masses, NIST electron binding)")


def local_peak_fit(ff, pp, sigma, x, shape):
    """Fit one peak (with a constant background) in a small window around x,
    independently of the main fit.  Returns (centre, uncertainty, fwhm, fwhm unc)."""
    pk = _snap_peak(ff, pp, x, [0.0, 0.0], "none")
    bin_hz = float(np.median(np.diff(ff)))
    # window of +-1 FWHM around the maximum: wide enough for the peak, narrow
    # enough to keep neighbouring peaks out
    half = max(1.0 * pk["fwhm"], 6 * bin_hz)
    region = [pk["centre"] - half, pk["centre"] + half]
    m = _region_mask(ff, region)
    base = float(np.nanmin(pp[m])) if m.any() else 0.0
    pk = _snap_peak(ff, pp, x, [base, 0.0], "const")
    pk["shape"] = shape
    # the centre may only move within half a FWHM of the clicked maximum, so
    # the fit cannot slide onto a bigger neighbouring peak
    w = max(0.5 * pk["fwhm"], 2 * bin_hz)
    st = {"peaks": [pk], "bg": "const", "bgp": [base, 0.0], "region": region,
          "xref": pk["centre"], "shape": shape,
          "centre_bounds": [[pk["centre"] - w, pk["centre"] + w]],
          "fwhm_bounds": [[0.3 * pk["fwhm"], 3.0 * pk["fwhm"]]]}
    res = run_fit(ff, pp, sigma, st)
    c = res["peaks"][0]["centre"]
    dc = res["unc"]["peaks"][0].get("centre", np.nan)
    dw = res["unc"]["peaks"][0].get("fwhm", np.nan)
    return float(c), float(dc), float(res["peaks"][0]["fwhm"]), float(dw)


def pick_peak(ff, pp, sigma, x, fit_state, shape):
    """Frequency for a click at x: the centre of a peak in the main fit if the
    click is on it (within half its FWHM), otherwise a local single-peak fit.
    (fit_state None: always the local fit - the spectrogram has no main fit.)"""
    bin_hz = float(np.median(np.diff(ff)))
    if fit_state and fit_state.get("peaks"):
        unc = (fit_state.get("unc") or {}).get("peaks") or [{} for _ in fit_state["peaks"]]
        best = None
        for i, pk in enumerate(fit_state["peaks"]):
            d = abs(pk["centre"] - x)
            if d <= max(0.5 * pk["fwhm"], 2 * bin_hz) and (best is None or d < best[0]):
                best = (d, i)
        if best is not None:
            i = best[1]
            pk = fit_state["peaks"][i]
            du = unc[i].get("centre") if i < len(unc) else None
            src = f"main fit, peak {i + 1}" + ("" if fit_state.get("fitted") else " (not fitted)")
            dw = unc[i].get("fwhm") if i < len(unc) else None
            return {"f": float(pk["centre"]), "df": float(du) if du is not None else np.nan,
                    "fwhm": float(pk["fwhm"]), "dfwhm": float(dw) if dw is not None else np.nan,
                    "src": src}
    c, dc, w, dw = local_peak_fit(ff, pp, sigma, x, shape)
    return {"f": c, "df": dc, "fwhm": w, "dfwhm": dw, "src": "local fit"}


def fmt_pm(v, dv, fmt=".1f"):
    """'v ± dv' ('–' if v is missing; no ± part if dv is missing)."""
    if v is None or not np.isfinite(v):
        return "–"
    return format(v, fmt) + ("" if dv is None or not np.isfinite(dv) else " ± " + format(dv, fmt))


def empty_state():
    return {"calib": [], "k": None, "dk": None, "pending": None, "pairs": [], "msg": ""}


def calibrate(est, name1, name2):
    """Step 1: k (keV/Hz) from the two picked calibration peaks and their
    names.  Returns the new state; problems go in est["msg"]."""
    est = dict(est)
    if len(est.get("calib") or []) < 2:
        est["msg"] = "Pick two calibration peaks first."
        return est
    try:
        dme, ddme, a, b, how = mass_excess_difference(name1, name2)
        if a["name"] == b["name"]:
            raise ValueError("the two peaks have the same name (no energy difference)")
    except Exception as exc:
        est["msg"] = f"Calibration: {exc}"
        return est
    p1, p2 = est["calib"]
    dfreq = p2["f"] - p1["f"]
    if abs(dfreq) < 1e-9:
        est["msg"] = "The two calibration peaks have the same frequency."
        return est
    s1, s2 = np.nan_to_num(p1["df"]), np.nan_to_num(p2["df"])
    k = dme / dfreq
    dk = float(np.sqrt((ddme / dfreq) ** 2 + (dme * np.hypot(s1, s2) / dfreq ** 2) ** 2))
    est.update(k=float(k), dk=dk, names=[a["name"], b["name"]], dme=float(dme),
               ddme=float(ddme), how=how, sys=bool(a["sys"] or b["sys"]))
    est["pairs"], est["pending"] = [], None
    return est


def add_pick(est, pk, suggest=None):
    """A picked peak: a calibration peak until calibrated, then peak A or B
    of a pair (dE = k (f_B - f_A)).  Returns the new state."""
    est = dict(est)
    if est.get("k") is None:                      # calibration step
        if len(est.get("calib") or []) >= 2:
            est["msg"] = ("Two calibration peaks are already picked: name them and press "
                          "Calibrate, or Reset to pick again.")
            return est
        if suggest is not None:
            pk = dict(pk, suggest=suggest)
        est["calib"] = list(est.get("calib") or []) + [pk]
        return est
    if est.get("pending") is None:                # first peak (A) of a pair
        est["pending"] = pk
        return est
    a, b = est["pending"], pk
    dfreq = b["f"] - a["f"]
    ddf = float(np.hypot(np.nan_to_num(a["df"]), np.nan_to_num(b["df"])))
    de = est["k"] * dfreq
    dde = float(np.hypot(est["k"] * ddf, dfreq * est["dk"]))
    est["pairs"] = list(est.get("pairs") or []) + [{"a": a, "b": b, "dfreq": dfreq, "ddf": ddf,
                                                    "de": de, "dde": dde}]
    est["pending"] = None
    return est


def calibration_text(est):
    """One-line description of the calibration ('' if not calibrated)."""
    if est.get("k") is None:
        return ""
    return (f"k = {est['k']:.6g} ± {est['dk']:.2g} keV/Hz   "
            f"(ΔE = {est['dme']:.2f} ± {est['ddme']:.2f} keV from {est['how']} of "
            f"{est['names'][1]} − {est['names'][0]})"
            + ("  · includes estimated (#) NUBASE values" if est.get("sys") else ""))


def status_text(est, on=True):
    """What to do next (or the last error)."""
    calib = est.get("calib") or []
    if est.get("msg"):
        return est["msg"]
    if not on:
        return "Turn on Energy calculation to pick peaks on the plot."
    if est.get("k") is None:
        return ("Step 1 – calibration: click two peaks, enter their names (or click a "
                "suggestion), then press Calibrate." if len(calib) < 2 else
                "Step 1 – enter the names of the two peaks and press Calibrate.")
    if est.get("pending") is None:
        return "Step 2 – click peak A of a pair (the energy difference is E(B) − E(A))."
    return f"Step 2 – peak A at {est['pending']['f']:.1f} Hz picked; now click peak B."
