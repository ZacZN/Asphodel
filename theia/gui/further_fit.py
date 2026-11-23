"""
Further output: peak fitting on the summed spectrum, Search ions.

Part of Theia (theia/theia.py); split out of theia.py in Nov 2026.
"""

import json
import re
import time
import traceback

import numpy as np
import plotly.graph_objects as go
import dash
from dash import ALL, dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate

from core.scan import load_config, save_config

from .common import (ACCENT, ACCENT_PALE, BOX_BORDER, HELP_STYLE, ISO_MODAL_SHOWN, LABEL_STYLE,
                     RESID_BOX_HIDDEN, RESID_BOX_SHOWN, RESULTS_PAGE_GRID, RESULTS_PAGE_HIDDEN)
from .app import app
from .figures import empty_fig, EXCLUDED_TRACE
from .further_files import (FIT_TRACE_STYLES, _RESTORE_FIT, _SUM_CACHE, _SUM_CACHE_LOCK,
                            _SUM_META, _SUM_SIMS)


# Peak-fit parameter table (one row per peak).  Value columns are editable;
# uncertainties and derived quantities are not.
FIT_PANEL_SHOWN = {"display": "block"}
FIT_PANEL_HIDDEN = {**FIT_PANEL_SHOWN, "display": "none"}

# line shapes available for each fitted peak
SHAPE_LABELS = {"gauss": "Gaussian", "lorentz": "Lorentzian", "pvoigt": "Pseudo-Voigt"}


# fix toggles: click the cell to switch between free (☐) and fixed (☑)
FIX_OFF, FIX_ON = "☐", "☑"
FIX_PARAMS = ("centre", "amp", "fwhm", "eta")
# per-peak limits for the fit (blank = default: centre within the fit region,
# FWHM between 0.05 bins and the region width)
LIMIT_KEYS = ("centre_min", "centre_max", "fwhm_min", "fwhm_max")

FIT_COLUMNS = [
    {"name": "Peak", "id": "peak", "editable": False},
    # ion name (e.g. 207Tl81+) and harmonic: needed to send the peak to the
    # isochronicity curve
    {"name": "Ion", "id": "ion", "editable": True},
    {"name": "h", "id": "harmonic", "type": "numeric", "editable": True},
    {"name": "Shape", "id": "shape", "presentation": "dropdown", "editable": True},
    {"name": "Centre (Hz)", "id": "centre", "type": "numeric", "editable": True},
    {"name": "± Centre", "id": "d_centre", "editable": False},
    {"name": "Fix", "id": "fix_centre", "editable": False},
    {"name": "Centre Min", "id": "centre_min", "type": "numeric", "editable": True},
    {"name": "Centre Max", "id": "centre_max", "type": "numeric", "editable": True},
    {"name": "Amplitude", "id": "amp", "type": "numeric", "editable": True},
    {"name": "± Amp", "id": "d_amp", "editable": False},
    {"name": "Fix", "id": "fix_amp", "editable": False},
    {"name": "FWHM (Hz)", "id": "fwhm", "type": "numeric", "editable": True},
    {"name": "± FWHM", "id": "d_fwhm", "editable": False},
    {"name": "Fix", "id": "fix_fwhm", "editable": False},
    {"name": "FWHM Min", "id": "fwhm_min", "type": "numeric", "editable": True},
    {"name": "FWHM Max", "id": "fwhm_max", "type": "numeric", "editable": True},
    {"name": "η (Lorentz fraction)", "id": "eta", "type": "numeric", "editable": True},
    {"name": "± η", "id": "d_eta", "editable": False},
    {"name": "Fix", "id": "fix_eta", "editable": False},
    {"name": "Area", "id": "area", "editable": False},
    {"name": "± Area", "id": "d_area", "editable": False},
]


def fit_columns(pv, limits):
    """Table columns: η only with a pseudo-Voigt peak, limits only when shown."""
    drop = set()
    if not pv:
        drop |= {"eta", "d_eta", "fix_eta"}
    if not limits:
        drop |= set(LIMIT_KEYS)
    return [c for c in FIT_COLUMNS if c["id"] not in drop]


# Further output: reads files saved by confirmation mode (confirmed/*.csv) for
# further processing.  Sub-tabs: "Aligning and Fitting" (file list, summing,
# peak fitting) and "Isochronicity Curve" (to come).
CALIB_BORDER = "1px solid #999999"
CALIB_TH = {"fontSize": "11px", "fontWeight": "700", "color": "#666", "textTransform": "uppercase",
            "letterSpacing": "0.04em", "padding": "4px 10px", "textAlign": "left",
            "border": CALIB_BORDER, "background": "#f9f9f9"}
CALIB_TD = {"fontSize": "13px", "padding": "5px 10px", "verticalAlign": "middle",
            "border": CALIB_BORDER}
SUGG_BOX_STYLE = {"marginLeft": "0", "marginTop": "4px", "width": "620px", "maxWidth": "90%",
                  "border": BOX_BORDER, "borderRadius": "6px", "background": "white",
                  "padding": "4px 0", "boxShadow": "0 4px 12px rgba(0,0,0,0.12)"}


SRCH_MODAL_SHOWN = {**ISO_MODAL_SHOWN, "alignItems": "flex-start", "paddingTop": "8vh"}
SRCH_MODAL_HIDDEN = {**SRCH_MODAL_SHOWN, "display": "none"}
SRCH_GRID = {"display": "grid", "gridTemplateColumns": "1.4fr 0.8fr 1.4fr 1fr"}
SRCH_DEFAULT_KHZ = 10.0

# Search ions (as in the spectrogram): an ion's simulated lines on the plotted
# spectrum; a click zooms the plot to f_sim ± width
srch_modal = html.Div([
    html.Div([
        html.Div([
            html.Span("Search Ions", style={"fontWeight": "700", "fontSize": "15px"}),
            html.Button("✕", id="srch-close", className="btn btn-outline-secondary btn-sm"),
        ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center",
                  "marginBottom": "8px"}),
        html.Div([
            html.Span("Ion", style={**LABEL_STYLE, "margin": "0 6px 0 0"}),
            dcc.Input(id="srch-ion", type="text", debounce=False, value="",
                      placeholder="e.g. 186Hf or m186Hf72+",
                      className="form-control form-control-sm",
                      style={"width": "220px", "marginRight": "12px"}),
            html.Span("±", style={"marginRight": "4px"},
                      title="Half-width of the window shown around the chosen line"),
            dcc.Input(id="srch-width", type="number", min=0, step="any", debounce=True,
                      value=SRCH_DEFAULT_KHZ, placeholder="10",
                      className="form-control form-control-sm",
                      style={"width": "80px", "marginRight": "4px"}),
            html.Span("kHz", style={"marginRight": "12px", "fontSize": "13px"}),
            html.Span(id="srch-status", style={"fontSize": "13px"}),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "6px"}),
        html.Small("Simulated lines across the whole plotted spectrum (current PID settings). "
                   "Click one to zoom to ± the width above around it (default ±10 kHz).  Grey: "
                   "below the half-life cut.", style={**HELP_STYLE, "marginBottom": "6px"}),
        html.Div(id="srch-list", style={"maxHeight": "55vh", "overflowY": "auto",
                                        "border": "1px solid #dddddd", "borderRadius": "4px"}),
    ], className="theia-box",
       style={"border": BOX_BORDER, "borderRadius": "6px", "background": "white",
              "width": "640px", "maxWidth": "94vw", "padding": "14px 16px",
              "boxShadow": "0 8px 30px rgba(0,0,0,0.3)"}),
], id="srch-modal", style=SRCH_MODAL_HIDDEN)


# ===========================================================================
# Further output: peak fitting on the summed spectrum
# ===========================================================================
#
# Parameters per peak: centre (Hz), amplitude (peak height above background),
# FWHM (Hz) and, for pseudo-Voigt, eta (Lorentzian fraction, 0..1).  All
# shapes are written in terms of FWHM so switching shape keeps the width.
# Fitting is done in offset frequency u = f - xref (xref = middle of the fit
# region) to keep the numbers well conditioned.

_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def _peak_shape(u, c, a, w, eta, shape):
    w = max(abs(w), 1e-9)
    g = np.exp(-0.5 * ((u - c) / (w * _FWHM_TO_SIGMA)) ** 2)
    if shape == "gauss":
        return a * g
    lor = 1.0 / (1.0 + ((u - c) / (w / 2.0)) ** 2)
    if shape == "lorentz":
        return a * lor
    eta = min(max(eta, 0.0), 1.0)
    return a * (eta * lor + (1.0 - eta) * g)


def _peak_area(a, w, eta, shape):
    ga = a * w * _FWHM_TO_SIGMA * np.sqrt(2 * np.pi)
    la = a * np.pi * w / 2.0
    if shape == "gauss":
        return ga
    if shape == "lorentz":
        return la
    return eta * la + (1 - eta) * ga


def _n_bg(bg):
    return {"none": 0, "const": 1, "linear": 2}.get(bg, 0)


def _shapes(state):
    """Line shape of each peak (each peak has its own)."""
    default = state.get("shape") or "gauss"
    return [pk.get("shape") or default for pk in state["peaks"]]


def _pack(state):
    """state -> (p0, names) for the parameters; centres as offsets from xref.
    eta is a parameter only for pseudo-Voigt peaks."""
    xref = state["xref"]
    p, names = [], []
    for i, (pk, shape) in enumerate(zip(state["peaks"], _shapes(state))):
        p += [pk["centre"] - xref, pk["amp"], pk["fwhm"]]
        names += [(i, "centre"), (i, "amp"), (i, "fwhm")]
        if shape == "pvoigt":
            p.append(pk.get("eta", 0.5))
            names.append((i, "eta"))
    for j in range(_n_bg(state["bg"])):
        p.append(state["bgp"][j])
        names.append((None, f"b{j}"))
    return np.array(p, dtype=float), names


def _model_from_vector(u, p, shapes, bg):
    """Model for the parameter vector laid out as in _pack."""
    y = np.zeros_like(u, dtype=float)
    q = 0
    for shape in shapes:
        c, a, w = p[q:q + 3]
        q += 3
        eta = 0.0
        if shape == "pvoigt":
            eta = p[q]
            q += 1
        y += _peak_shape(u, c, a, w, eta, shape)
    nb = _n_bg(bg)
    if nb >= 1:
        y += p[q]
    if nb == 2:
        y += p[q + 1] * u
    return y


def fit_overlay(ff, state):
    """(x, y) for the four overlay traces: total, components (each on top of
    the background), background, peak markers.  Evaluated on a fine grid
    over the fit region; all empty if there are no peaks."""
    if not state or not state.get("peaks") or not state.get("region"):
        return [([], [])] * len(FIT_TRACE_STYLES)
    lo, hi = state["region"]
    bin_hz = float(np.median(np.diff(ff))) if len(ff) > 1 else 1.0
    n = int(min(6000, max(200, 6 * (hi - lo) / bin_hz)))
    x = np.linspace(lo, hi, n)
    u = x - state["xref"]
    nb = _n_bg(state["bg"])
    bgp = state.get("bgp") or [0.0, 0.0]
    back = np.zeros_like(u)
    if nb >= 1:
        back += bgp[0]
    if nb == 2:
        back += bgp[1] * u
    total = back.copy()
    cx, cy = [], []
    for pk, shape in zip(state["peaks"], _shapes(state)):
        comp = _peak_shape(u, pk["centre"] - state["xref"], pk["amp"], pk["fwhm"],
                           pk.get("eta", 0.5), shape)
        total += comp
        cx += x.tolist() + [None]
        cy += (comp + back).tolist() + [None]
    mx = [pk["centre"] for pk in state["peaks"]]
    my = [float(np.interp(c, x, total)) for c in mx]
    # FWHM handles at the half-height points of each component (on its curve)
    hx, hy = [], []
    for pk in state["peaks"]:
        for side in (-1, 1):
            xh = pk["centre"] + side * 0.5 * pk["fwhm"]
            hx.append(xh)
            hy.append(float(np.interp(xh, x, back)) + 0.5 * pk["amp"])
    return [(x.tolist(), total.tolist()),
            (cx, cy),
            (x.tolist(), back.tolist()) if nb else ([], []),
            (mx, my),
            (hx, hy)]


def _region_mask(ff, region):
    lo, hi = region
    return (ff >= lo) & (ff <= hi)


def run_fit(ff, pp, sigma, state):
    """Least-squares fit over the region, starting from the current values.
    Returns the updated state (values, uncertainties, chi2/ndf)."""
    from scipy.optimize import curve_fit
    m = _region_mask(ff, state["region"]) & np.isfinite(pp)
    u, y = ff[m] - state["xref"], pp[m]
    p0, names = _pack(state)
    lo_u, hi_u = float(u.min()), float(u.max())
    bin_hz = float(np.median(np.diff(ff[m])))
    xref = state["xref"]
    lb, ub = [], []
    cb = state.get("centre_bounds")             # optional per-peak centre limits (Hz)
    fb = state.get("fwhm_bounds")

    def _lim(pk, key):
        v = (pk.get("lim") or {}).get(key)
        return None if v is None or not np.isfinite(v) else float(v)
    for (i, what) in names:
        pk = state["peaks"][i] if i is not None else {}
        if what == "centre":
            lo, hi = lo_u, hi_u
            if cb and cb[i]:
                lo, hi = max(lo, cb[i][0] - xref), min(hi, cb[i][1] - xref)
            if _lim(pk, "centre_min") is not None:
                lo = max(lo, _lim(pk, "centre_min") - xref)
            if _lim(pk, "centre_max") is not None:
                hi = min(hi, _lim(pk, "centre_max") - xref)
            if not lo < hi:
                raise ValueError(f"peak {i + 1}: the centre limits do not overlap the fit region")
            lb.append(lo); ub.append(hi)
        elif what == "amp":
            lb.append(0.0); ub.append(np.inf)
        elif what == "fwhm":
            lo, hi = 0.05 * bin_hz, max(hi_u - lo_u, bin_hz)
            if fb and fb[i]:
                lo, hi = fb[i]
            if _lim(pk, "fwhm_min") is not None:
                lo = max(_lim(pk, "fwhm_min"), 1e-9)
            if _lim(pk, "fwhm_max") is not None:
                hi = _lim(pk, "fwhm_max")
            if not lo < hi:
                raise ValueError(f"peak {i + 1}: FWHM min must be below FWHM max")
            lb.append(lo); ub.append(hi)
        elif what == "eta":
            lb.append(0.0); ub.append(1.0)
        else:
            lb.append(-np.inf); ub.append(np.inf)
    lb, ub = np.array(lb), np.array(ub)

    # fixed parameters are left out of the fit (held at their current value)
    bg_fix = set(state.get("bg_fix") or [])
    free = np.array([(what not in bg_fix) if i is None
                     else (what not in (state["peaks"][i].get("fix") or []))
                     for (i, what) in names], dtype=bool)
    n_free = int(free.sum())
    if n_free == 0:
        raise ValueError("every parameter is fixed: free at least one")
    if len(u) <= n_free:
        raise ValueError(f"Only {len(u)} bins in the fit region for {n_free} free parameters")
    for q in np.flatnonzero(free):                # start strictly inside the bounds
        if not (lb[q] < p0[q] < ub[q]):
            span = (ub[q] - lb[q]) if np.isfinite(ub[q] - lb[q]) else max(abs(p0[q]), 1.0)
            p0[q] = min(max(p0[q], lb[q] + 1e-6 * span), ub[q] - 1e-6 * span)
    shapes, bg = _shapes(state), state["bg"]
    n_peaks = len(shapes)
    p_all = p0.copy()

    def full(pf):
        p = p_all.copy()
        p[free] = pf
        return p

    def f_free(uu, *pf):
        return _model_from_vector(uu, full(np.array(pf)), shapes, bg)

    # absolute_sigma=False: the covariance is scaled by chi2/ndf, i.e. the
    # uncertainties are scaled by sqrt(chi2/ndf).  The noise sigma is only a
    # rough estimate (and real peaks are not perfect line shapes), so this
    # lets the scatter of the data about the fit set the uncertainties.
    popt_f, pcov_f = curve_fit(f_free, u, y, p0=p0[free], sigma=np.full(len(u), sigma),
                               absolute_sigma=False, bounds=(lb[free], ub[free]), maxfev=20000)
    popt = full(popt_f)
    pcov = np.zeros((len(p0), len(p0)))
    pcov[np.ix_(free, free)] = pcov_f
    ok_cov = np.all(np.isfinite(pcov_f))
    perr = np.sqrt(np.clip(np.diag(pcov), 0, None)) if ok_cov else np.full(len(popt), np.nan)
    perr[~free] = 0.0
    resid = y - _model_from_vector(u, popt, shapes, bg)
    ndf = max(len(u) - n_free, 1)

    new = dict(state)
    new["peaks"] = [dict(pk) for pk in state["peaks"]]
    new["bgp"] = list(state.get("bgp") or [0.0, 0.0])
    unc = {"peaks": [{} for _ in state["peaks"]], "bgp": [None, None]}
    for (i, what), v, e in zip(names, popt, perr):
        if i is None:
            j = int(what[1])
            new["bgp"][j], unc["bgp"][j] = float(v), float(e)
        else:
            new["peaks"][i][what] = float(v + state["xref"]) if what == "centre" else float(v)
            unc["peaks"][i][what] = float(e)
    # area uncertainty: linear error propagation with a numerical gradient
    for i in range(n_peaks):
        idx = [k for k, (j, _) in enumerate(names) if j == i]
        keys = [names[k][1] for k in idx]

        def area_of(vec, shape=shapes[i]):
            d = dict(zip(keys, vec))
            return _peak_area(d["amp"], d["fwhm"], d.get("eta", 0.5), shape)
        v0 = popt[idx].astype(float)
        grad = np.zeros(len(idx))
        for q in range(len(idx)):
            h = 1e-6 * max(abs(v0[q]), 1.0)
            vp = v0.copy()
            vp[q] += h
            grad[q] = (area_of(vp) - area_of(v0)) / h
        sub = pcov[np.ix_(idx, idx)]
        unc["peaks"][i]["area"] = (float(np.sqrt(max(grad @ sub @ grad, 0.0)))
                                   if ok_cov else float("nan"))
    new.update(unc=unc, chi2=float(np.sum((resid / sigma) ** 2) / ndf), ndf=int(ndf),
               sigma=float(sigma), fitted=True)
    return new


def _snap_peak(ff, pp, x, bgp, bg):
    """Start values for a peak clicked at x: centre and amplitude from the
    highest bin within +-3 bins, FWHM from where the data fall to half height."""
    k0 = int(np.clip(np.searchsorted(ff, x), 0, len(ff) - 1))
    lo, hi = max(k0 - 3, 0), min(k0 + 4, len(ff))
    k = lo + int(np.nanargmax(pp[lo:hi]))
    base = bgp[0] if _n_bg(bg) >= 1 else 0.0
    amp = max(float(pp[k] - base), 0.0)
    half = base + amp / 2.0
    left = k
    while left > 0 and pp[left - 1] > half and k - left < 200:
        left -= 1
    right = k
    while right < len(pp) - 1 and pp[right + 1] > half and right - k < 200:
        right += 1
    bin_hz = float(np.median(np.diff(ff[max(k - 5, 0):k + 6])))
    fwhm = max((right - left + 1) * bin_hz, 1.5 * bin_hz)
    return {"centre": float(ff[k]), "amp": amp, "fwhm": fwhm, "eta": 0.5}


def _fmt_num(v, fmt):
    try:
        return "–" if v is None or not np.isfinite(v) else format(v, fmt)
    except TypeError:
        return "–"


def fit_table_rows(state):
    if not state or not state.get("peaks"):
        return []
    unc = (state.get("unc") or {}).get("peaks") or [{} for _ in state["peaks"]]
    rows = []
    for i, (pk, e, shape) in enumerate(zip(state["peaks"], unc, _shapes(state)), start=1):
        pv = shape == "pvoigt"
        fix = set(pk.get("fix") or [])
        lim = pk.get("lim") or {}

        def err(key, fmt):
            return "fixed" if key in fix else _fmt_num(e.get(key), fmt)
        area = _peak_area(pk["amp"], pk["fwhm"], pk.get("eta", 0.5), shape)
        rows.append({
            "peak": i, "ion": pk.get("ion") or "", "harmonic": pk.get("harmonic"),
            "shape": shape,
            "centre": round(pk["centre"], 2), "d_centre": err("centre", ".2f"),
            "amp": float(f"{pk['amp']:.6g}"), "d_amp": err("amp", ".3g"),
            "fwhm": round(pk["fwhm"], 2), "d_fwhm": err("fwhm", ".2f"),
            "eta": round(pk.get("eta", 0.5), 4) if pv else None,
            "d_eta": err("eta", ".3f") if pv else "",
            "area": _fmt_num(area, ".5g"), "d_area": _fmt_num(e.get("area"), ".3g"),
            **{f"fix_{k}": (FIX_ON if k in fix else FIX_OFF) if (k != "eta" or pv) else ""
               for k in FIX_PARAMS},
            **{k: lim.get(k) for k in LIMIT_KEYS},
        })
    return rows


def _set_region(state, region):
    """New fit region [lo, hi]; xref moves to its middle (a linear background
    keeps the same line)."""
    lo, hi = sorted(float(v) for v in region)
    if not hi > lo:
        raise ValueError("the fit region needs min < max")
    old_ref = state.get("xref") or 0.5 * (lo + hi)
    state["region"], state["xref"] = [lo, hi], 0.5 * (lo + hi)
    if _n_bg(state["bg"]) == 2:
        state["bgp"][0] += state["bgp"][1] * (state["xref"] - old_ref)
    return state


def _num_or_none(v):
    try:
        v = float(v)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _harmonic_or_none(v):
    """Harmonic typed in the table: a positive integer, else None."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return int(round(f)) if np.isfinite(f) and f >= 1 and abs(f - round(f)) < 1e-6 else None


def _only_names_changed(state, table):
    """True if the table differs from the state only in the ion / harmonic
    columns (same rows, same values otherwise)."""
    if not state or not state.get("peaks") or table is None:
        return False
    old = fit_table_rows(state)
    if len(old) != len(table):
        return False
    skip = {"ion", "harmonic"}
    for a, b in zip(old, table):
        for k, v in a.items():
            if k in skip:
                continue
            w = b.get(k)
            if v == w:
                continue
            try:                                   # numbers may come back as str / float
                if float(v) == float(w):
                    continue
            except (TypeError, ValueError):
                pass
            if (v in (None, "") and w in (None, "")):
                continue
            return False
    return True


def _fit_status(state):
    if not state or not state.get("peaks"):
        return ""
    lo, hi = state["region"]
    reg = f"region {lo:.1f} – {hi:.1f} Hz ({hi - lo:.0f} Hz wide)"
    if state.get("fitted"):
        return html.Span([html.B("Fitted: "),
                          f"{len(state['peaks'])} peak(s), {reg}  ·  χ²/ndf = "
                          f"{state['chi2']:.3g} (ndf = {state['ndf']}, noise σ = "
                          f"{state['sigma']:.3g} per bin; uncertainties scaled by √(χ²/ndf))"])
    return html.Span([html.B("Start values, not fitted: "),
                      f"{len(state['peaks'])} peak(s), {reg}  ·  edit the table or press Fit"],
                     style={"color": "#888"})


@app.callback(
    Output("fit-mode", "data"),
    Output("btn-fit-mode", "children"),
    Output("btn-fit-mode", "style"),
    Output("energy-mode", "data"),
    Output("btn-energy-mode", "children"),
    Output("btn-energy-mode", "style"),
    Input("btn-fit-mode", "n_clicks"),
    Input("btn-energy-mode", "n_clicks"),
    State("fit-mode", "data"),
    State("energy-mode", "data"),
    prevent_initial_call=True,
)
def toggle_modes(_f, _e, fit_on, energy_on):
    """Peak fitting and energy calculation are click modes for the summed
    plot; turning one on turns the other off."""
    trig = dash.callback_context.triggered[0]["prop_id"] if dash.callback_context.triggered else ""
    if trig.startswith("btn-fit-mode"):
        fit_on, energy_on = not fit_on, False
    else:
        energy_on, fit_on = not energy_on, False
    base = {"marginRight": "12px", "fontWeight": "600"}
    on_style = {**base, "backgroundColor": ACCENT, "color": "white", "borderColor": ACCENT}
    return (fit_on, "Peak Fitting: ON (click peaks)" if fit_on else "Peak Fitting",
            on_style if fit_on else base,
            energy_on, "Energy Calculation: ON (click peaks)" if energy_on else "Energy Calculation",
            on_style if energy_on else base)


# In peak-fitting mode a click (without dragging) on the summed plot adds a
# peak there: the browser converts the pixel position to Hz and hands it to
# the server through the fit-click store (with the current view range, which
# becomes the fit region).
app.clientside_callback(
    """
    function(on, energyOn, figure) {
        window._theiaFitMode = !!on;
        window._theiaEnergyMode = !!energyOn;
        var box = document.getElementById('sum-graph');
        if (box) box.classList.toggle('fit-mode', !!on || !!energyOn);
        setTimeout(function() {
            var gd = document.querySelector('#sum-graph .js-plotly-plot');
            if (!gd || gd._theiaFitClick) return;
            gd._theiaFitClick = true;
            var down = null;
            gd.addEventListener('mousedown', function(e) {
                down = [e.clientX, e.clientY]; window._theiaSkipClick = false; }, true);

            // ---- drag on the fit overlay ---------------------------------
            //  * x marker of a peak: move centre + amplitude (cursor: move)
            //  * below the marker (the peak's column): centre only (cursor: <->)
            //  * diamond handle at half height: FWHM (cursor: <->)
            //  * orange region line: fit region edge (cursor: col-resize)
            // The curve is redrawn in the browser while dragging (from the peak
            // parameters in the marker trace's customdata); on release the new
            // values go to the server (fit-drag / fit-region-set stores).
            function traceIdx(name) {
                for (var i = 0; i < gd.data.length; i++) if (gd.data[i].name === name) return i;
                return -1;
            }
            function shapeIdx(name) {
                var S = (gd.layout && gd.layout.shapes) || [];
                for (var i = 0; i < S.length; i++) if (S[i] && S[i].name === name) return i;
                return -1;
            }
            function profile(code, u, w, eta) {
                w = Math.max(Math.abs(w), 1e-9);
                var s = w / (2 * Math.sqrt(2 * Math.LN2));
                var g = Math.exp(-0.5 * (u / s) * (u / s));
                var l = 1 / (1 + (2 * u / w) * (2 * u / w));
                if (code === 0) return g;
                if (code === 1) return l;
                eta = Math.min(Math.max(eta, 0), 1);
                return eta * l + (1 - eta) * g;
            }
            function interp(X, Y, x) {
                var n = X.length;
                if (!n) return 0;
                if (x <= X[0]) return Y[0];
                if (x >= X[n - 1]) return Y[n - 1];
                var f = (x - X[0]) / (X[n - 1] - X[0]) * (n - 1), i = Math.floor(f), w = f - i;
                return Y[i] * (1 - w) + Y[Math.min(i + 1, n - 1)] * w;
            }
            function pos(e) {
                var L = gd._fullLayout;
                if (!L || !L.xaxis || !L.yaxis) return null;
                var r = gd.getBoundingClientRect();
                var px = e.clientX - r.left - L.xaxis._offset;
                var py = e.clientY - r.top - L.yaxis._offset;
                if (px < 0 || px > L.xaxis._length || py < 0 || py > L.yaxis._length) return null;
                return {px: px, py: py, L: L};
            }
            function hit(e) {
                var p = pos(e);
                if (!p) return null;
                var L = p.L, ih = traceIdx('fit width handles'), ip = traceIdx('fit peaks');
                if (ih >= 0) {
                    var hx = gd.data[ih].x || [], hy = gd.data[ih].y || [];
                    for (var j = 0; j < hx.length; j++)
                        if (Math.abs(L.xaxis.d2p(hx[j]) - p.px) < 7 &&
                            Math.abs(L.yaxis.d2p(hy[j]) - p.py) < 7)
                            return {kind: 'width', k: Math.floor(j / 2), side: (j % 2) ? 1 : -1};
                }
                if (ip >= 0) {
                    var mx = gd.data[ip].x || [], my = gd.data[ip].y || [];
                    var best = -1, bestd = 9, onTop = false;
                    for (var k = 0; k < mx.length; k++) {
                        var d = Math.abs(L.xaxis.d2p(mx[k]) - p.px), top = L.yaxis.d2p(my[k]);
                        if (d < bestd && p.py >= top - 15) {
                            best = k; bestd = d; onTop = Math.abs(p.py - top) < 11; }
                    }
                    if (best >= 0) return {kind: onTop ? 'move' : 'centre', k: best};
                }
                var names = ['fit-region-lo', 'fit-region-hi'];
                for (var q = 0; q < 2; q++) {
                    var si = shapeIdx(names[q]);
                    if (si >= 0 && Math.abs(L.xaxis.d2p(gd.layout.shapes[si].x0) - p.px) < 6)
                        return {kind: 'region', which: q};
                }
                return null;
            }
            var drag = null;
            gd.addEventListener('mousedown', function(e) {
                if (e.button !== 0) return;
                var h = hit(e);
                if (!h) return;
                e.stopPropagation(); e.preventDefault();      // no pan
                window._theiaSkipClick = true;
                if (h.kind === 'region') {
                    drag = {kind: 'region', which: h.which, x0: e.clientX, y0: e.clientY,
                            lo: gd.layout.shapes[shapeIdx('fit-region-lo')].x0,
                            hi: gd.layout.shapes[shapeIdx('fit-region-hi')].x0};
                    return;
                }
                var it = traceIdx('fit'), ic = traceIdx('fit components'),
                    ib = traceIdx('fit background'), ip = traceIdx('fit peaks'),
                    ih = traceIdx('fit width handles');
                var X = (gd.data[it].x || []).map(Number), T = (gd.data[it].y || []).map(Number);
                var n = X.length, k = h.k;
                var cy = gd.data[ic].y, seg = k * (n + 1);       // segments split by null
                var B = (gd.data[ib].x && gd.data[ib].x.length) ? gd.data[ib].y.map(Number)
                                                               : X.map(function() { return 0; });
                var C = [];
                for (var j = 0; j < n; j++) C.push(Number(cy[seg + j]) - B[j]);
                var par = (gd.data[ip].customdata || [])[k];
                if (!par) { drag = null; return; }
                drag = {kind: h.kind, side: h.side, k: k, x0: e.clientX, y0: e.clientY,
                        X: X, T: T, C: C, B: B, seg: seg, n: n, it: it, ic: ic, ip: ip, ih: ih,
                        cx: gd.data[ic].x.slice(), cy: cy.slice(),
                        mx: gd.data[ip].x.slice(), my: gd.data[ip].y.slice(),
                        hx: gd.data[ih].x.slice(), hy: gd.data[ih].y.slice(),
                        c0: Number(par[0]), a0: Number(par[1]), w0: Number(par[2]),
                        eta: Number(par[3]), code: Number(par[4])};
                drag.c = drag.c0; drag.a = drag.a0; drag.w = drag.w0;
            }, true);
            // redraw at most once per animation frame while dragging
            var pending = null;
            document.addEventListener('mousemove', function(e) {
                if (!drag) return;
                var first = (pending === null);
                pending = [e.clientX, e.clientY];
                if (first) requestAnimationFrame(function() {
                    var q = pending; pending = null;
                    if (drag) dragTo(drag, q[0], q[1]);
                });
            });
            function deltas(d, clientX, clientY, xRef, yRef) {
                // pixel offsets -> data offsets, measured from a reference point
                var L = gd._fullLayout;
                return [L.xaxis.p2d(L.xaxis.d2p(xRef) + (clientX - d.x0)) - xRef,
                        L.yaxis.p2d(L.yaxis.d2p(yRef) + (clientY - d.y0)) - yRef];
            }
            function dragTo(d, clientX, clientY) {
                if (d.kind === 'region') {
                    var L = gd._fullLayout, r = gd.getBoundingClientRect();
                    var x = L.xaxis.p2d(clientX - r.left - L.xaxis._offset);
                    if (d.which === 0) d.lo = x; else d.hi = x;
                    var upd = {}, ilo = shapeIdx('fit-region-lo'), ihi = shapeIdx('fit-region-hi'),
                        it = shapeIdx('fit-region-tint');
                    upd['shapes[' + ilo + '].x0'] = d.lo; upd['shapes[' + ilo + '].x1'] = d.lo;
                    upd['shapes[' + ihi + '].x0'] = d.hi; upd['shapes[' + ihi + '].x1'] = d.hi;
                    if (it >= 0) { upd['shapes[' + it + '].x0'] = Math.min(d.lo, d.hi);
                                   upd['shapes[' + it + '].x1'] = Math.max(d.lo, d.hi); }
                    Plotly.relayout(gd, upd);
                    return;
                }
                var k = d.k;
                if (d.kind === 'width') {
                    var xh = d.c0 + d.side * d.w0 / 2;
                    var dx = deltas(d, clientX, clientY, xh, 0)[0];
                    d.w = Math.max(2 * d.side * (xh + dx - d.c0), 1e-3);
                } else {
                    var dd = deltas(d, clientX, clientY, d.mx[k], d.my[k]);
                    d.c = d.c0 + dd[0];
                    if (d.kind === 'move') d.a = Math.max(d.a0 + dd[1], 0);
                }
                var X = d.X, n = d.n, newT = [], cy = d.cy.slice();
                for (var j = 0; j < n; j++) {
                    var c = d.a * profile(d.code, X[j] - d.c, d.w, d.eta);
                    newT.push(d.T[j] - d.C[j] + c);
                    cy[d.seg + j] = c + d.B[j];
                }
                var mx = d.mx.slice(), my = d.my.slice(), hx = d.hx.slice(), hy = d.hy.slice();
                mx[k] = d.c; my[k] = interp(X, newT, d.c);
                hx[2 * k] = d.c - d.w / 2; hx[2 * k + 1] = d.c + d.w / 2;
                hy[2 * k] = interp(X, d.B, hx[2 * k]) + d.a / 2;
                hy[2 * k + 1] = interp(X, d.B, hx[2 * k + 1]) + d.a / 2;
                Plotly.restyle(gd, {x: [X, d.cx, mx, hx], y: [newT, cy, my, hy]},
                               [d.it, d.ic, d.ip, d.ih]);
            }
            document.addEventListener('mouseup', function(e) {
                if (!drag) return;
                var d = drag; drag = null;
                var moved = Math.abs(e.clientX - d.x0) >= 2 || Math.abs(e.clientY - d.y0) >= 2;
                if (!moved) {
                    // energy mode: a tap on a fitted peak picks that peak
                    if (window._theiaEnergyMode && (d.kind === 'move' || d.kind === 'centre')) {
                        var L0 = gd._fullLayout;
                        window.dash_clientside.set_props('energy-click', {data: {
                            x: d.mx[d.k], xrange: [Number(L0.xaxis.range[0]), Number(L0.xaxis.range[1])],
                            t: Date.now()}});
                    }
                    return;
                }
                dragTo(d, e.clientX, e.clientY);                 // final position
                if (d.kind === 'region') {
                    window.dash_clientside.set_props('fit-region-set', {data: {
                        xrange: [Math.min(d.lo, d.hi), Math.max(d.lo, d.hi)], t: Date.now()}});
                    return;
                }
                var out = {peak: d.k, t: Date.now()};
                if (d.kind === 'width') out.fwhm = d.w;
                else out.centre = d.c;
                if (d.kind === 'move') out.amp = d.a;
                window.dash_clientside.set_props('fit-drag', {data: out});
            });
            // cursor over a draggable element
            gd.addEventListener('mousemove', function(e) {
                if (drag) return;
                var box = document.getElementById('sum-graph');
                if (!box) return;
                var h = hit(e);
                box.classList.toggle('fit-grab', !!h && (h.kind === 'centre' || h.kind === 'width'));
                box.classList.toggle('fit-grab-move', !!h && h.kind === 'move');
                box.classList.toggle('fit-grab-region', !!h && h.kind === 'region');
            });

            gd.addEventListener('click', function(e) {
                if (window._theiaSkipClick) { window._theiaSkipClick = false; return; }
                if (!(window._theiaFitMode || window._theiaEnergyMode) || !down) return;
                if (Math.abs(e.clientX - down[0]) > 4 || Math.abs(e.clientY - down[1]) > 4) return;
                var L = gd._fullLayout;
                if (!L || !L.xaxis || !L.yaxis) return;
                var r = gd.getBoundingClientRect();
                var px = e.clientX - r.left - L.xaxis._offset;
                var py = e.clientY - r.top - L.yaxis._offset;
                if (px < 0 || px > L.xaxis._length || py < 0 || py > L.yaxis._length) return;
                var x = L.xaxis.p2d(px);
                if (!isFinite(x)) return;
                window.dash_clientside.set_props(
                    window._theiaFitMode ? 'fit-click' : 'energy-click', {data: {
                    x: x, xrange: [Number(L.xaxis.range[0]), Number(L.xaxis.range[1])],
                    t: Date.now()}});
            });
        }, 200);
        return window.dash_clientside.no_update;
    }
    """,
    Output("_fit-mode-hook", "children"),
    Input("fit-mode", "data"),
    Input("energy-mode", "data"),
    Input("sum-graph", "figure"),
)

# Current zoom of the summed plot (saved with the Further output state)
# (read shortly after a zoom/pan or a new figure, once Plotly has drawn it)
app.clientside_callback(
    """
    function(rd, fig) {
        setTimeout(function() {
            var gd = document.querySelector('#sum-graph .js-plotly-plot');
            if (!gd || !gd._fullLayout || !gd._fullLayout.xaxis || !gd.data || !gd.data.length)
                return;
            var L = gd._fullLayout;
            var v = {x: [Number(L.xaxis.range[0]), Number(L.xaxis.range[1])],
                     y: [Number(L.yaxis.range[0]), Number(L.yaxis.range[1])]};
            var key = JSON.stringify(v);
            if (key === window._theiaLastView) return;
            window._theiaLastView = key;
            window.dash_clientside.set_props('sum-view', {data: v});
        }, 400);
        return window.dash_clientside.no_update;
    }
    """,
    Output("_sum-view-hook", "children"),
    Input("sum-graph", "relayoutData"),
    Input("sum-graph", "figure"),
    prevent_initial_call=True,
)


@app.callback(
    Output("_further-saved", "children"),
    Input("further-files", "data"),
    Input("sum-align", "value"),
    Input("sum-ion", "value"),
    Input("sum-unsure", "value"),
    Input("sum-window", "value"),
    Input("sum-key", "data"),
    Input("sum-view", "data"),
    Input("fit-state", "data"),
    Input("energy-state", "data"),
    Input("energy-name-1", "value"),
    Input("energy-name-2", "value"),
    Input("pid-settings", "data"),
    Input("further-source", "value"),
    Input("fsrc-dir", "value"),
    Input("fsrc-name", "value"),
    prevent_initial_call=True,
)
def save_further_state(files, align, ion, unsure, window, key, view, fit, energy, name1, name2,
                       pid, source, f_dir, f_name):
    """Save the Further output tab after every change, so it is restored the
    next time Theia runs (scanner_config.json -> further_state)."""
    trig = dash.callback_context.triggered[0]["prop_id"] if dash.callback_context.triggered else ""
    old = load_config().get("further_state") or {}
    st = {"files": files or [], "align": align, "ion": ion, "unsure": unsure or [],
          "window": window,
          "summed": key is not None, "view": view or old.get("view"), "fit": old.get("fit"),
          "energy": old.get("energy"), "pid": pid,
          # chosen data source and single file; "plot" = what the plot shows
          "source": source or "sum", "file_dir": f_dir, "file_name": f_name,
          "plot": _SUM_META.get(key) if key is not None else old.get("plot")}
    if trig.startswith("sum-key"):
        st["view"] = view if view else old.get("view")
    if trig.startswith("fit-state"):
        st["fit"] = ({k: v for k, v in fit.items() if k != "sum_key"}
                     if fit and fit.get("peaks") else None)
    if trig.startswith("energy-state") or trig.startswith("energy-name"):
        has = energy and (energy.get("calib") or energy.get("pairs") or energy.get("k") is not None)
        st["energy"] = ({"state": {k: v for k, v in energy.items() if k != "msg"},
                         "names": [name1 or "", name2 or ""]}
                        if has or name1 or name2 else None)
    if key is None and not trig.startswith("sum-key"):
        # page just loaded and the restored sum is not back yet: keep it
        st["summed"] = old.get("summed", False)
        st["plot"] = old.get("plot")
    save_config({"further_state": st})
    return dash.no_update


# Show / hide the simulated lines on the summed plot without a redraw
app.clientside_callback(
    """
    function(value) {
        var gd = document.querySelector('#sum-graph .js-plotly-plot');
        if (!gd || !gd.data) return window.dash_clientside.no_update;
        var idx = [];
        gd.data.forEach(function(t, i) { if (t.meta === 'sim') idx.push(i); });
        if (idx.length) Plotly.restyle(gd, {visible: (value && value.length) ? true : false}, idx);
        return window.dash_clientside.no_update;
    }
    """,
    Output("_sum-sims-hook", "children"),
    Input("sum-show-sims", "value"),
    prevent_initial_call=True,
)


# "Fit": pass the current view range (the fit region) to the server
app.clientside_callback(
    """
    function(n) {
        var gd = document.querySelector('#sum-graph .js-plotly-plot');
        if (!n || !gd || !gd._fullLayout || !gd._fullLayout.xaxis)
            return window.dash_clientside.no_update;
        var R = gd._fullLayout.xaxis.range;
        return {xrange: [Number(R[0]), Number(R[1])], t: Date.now()};
    }
    """,
    Output("fit-run", "data"),
    Input("btn-fit-run", "n_clicks"),
    prevent_initial_call=True,
)


# "Region = View": the current zoom range becomes the fit region
app.clientside_callback(
    """
    function(n) {
        var gd = document.querySelector('#sum-graph .js-plotly-plot');
        if (!n || !gd || !gd._fullLayout || !gd._fullLayout.xaxis)
            return window.dash_clientside.no_update;
        var R = gd._fullLayout.xaxis.range;
        return {xrange: [Number(R[0]), Number(R[1])], t: Date.now()};
    }
    """,
    Output("fit-region-set", "data"),
    Input("btn-fit-region-view", "n_clicks"),
    prevent_initial_call=True,
)


@app.callback(
    Output("fit-state", "data"),
    Output("fit-table", "data"),
    Output("fit-table", "columns"),
    Output("fit-b0", "value"),
    Output("fit-b1", "value"),
    Output("fit-b0-unc", "children"),
    Output("fit-b1-unc", "children"),
    Output("fit-status", "children"),
    Output("fit-panel", "style"),
    Output("fit-table", "active_cell"),
    Output("fit-table", "selected_cells"),
    Output("fit-region-lo", "value"),
    Output("fit-region-hi", "value"),
    Output("fit-bg-fix", "value"),
    Input("fit-click", "data"),
    Input("fit-drag", "data"),
    Input("fit-run", "data"),
    Input("fit-table", "data_timestamp"),
    Input("fit-table", "active_cell"),
    Input("fit-shape", "value"),
    Input("fit-bg", "value"),
    Input("fit-b0", "value"),
    Input("fit-b1", "value"),
    Input("fit-bg-fix", "value"),
    Input("fit-region-set", "data"),
    Input("fit-region-lo", "value"),
    Input("fit-region-hi", "value"),
    Input("fit-show-limits", "value"),
    Input("btn-fit-clear", "n_clicks"),
    Input("sum-key", "data"),
    Input("iso-names-apply", "data"),
    State("fit-state", "data"),
    State("fit-table", "data"),
    prevent_initial_call=True,
)
def update_fit(click, dragged, run, _ts, active, shape, bg, b0, b1, bg_fix, region_set,
               region_lo, region_hi, show_limits, _clear, key, names_apply, state, table):
    """Every change to the fit goes through here (one callback, because these
    outputs also feed its inputs): clicks add peaks, table / background edits
    change the curve by hand, Fit runs the least-squares fit."""
    trig = dash.callback_context.triggered[0]["prop_id"] if dash.callback_context.triggered else ""
    if trig.startswith("fit-table.active_cell"):
        # a click on a "fix" cell toggles that parameter; any other cell: nothing
        col = (active or {}).get("column_id") or ""
        if not col.startswith("fix_") or not state or not state.get("peaks"):
            raise PreventUpdate
    with _SUM_CACHE_LOCK:
        cached = _SUM_CACHE.get(key)
    error = None
    restored = _RESTORE_FIT.pop(key, None) if trig.startswith("sum-key") else None
    if cached is None:
        state = None
    elif restored:
        state = dict(restored, sum_key=key)              # fit saved last session
    elif (not state or state.get("sum_key") != key or trig.startswith("btn-fit-clear")
          or trig.startswith("sum-key")):
        state = {"sum_key": key, "peaks": [], "bgp": [0.0, 0.0], "region": None,
                 "xref": 0.0, "fitted": False}
    if state is not None:
        ff, pp, sigma = cached
        state = dict(state, shape=shape, bg=bg)
        state["bgp"] = list(state.get("bgp") or [0.0, 0.0])
        try:
            if trig.startswith("fit-click") and click:
                if not state["peaks"]:
                    if not state.get("region"):           # region: the view, unless set
                        _set_region(state, click["xrange"])
                    # background start value: median of the data in the region
                    m = _region_mask(ff, state["region"])
                    state["bgp"] = [float(np.nanmedian(pp[m])) if m.any() else 0.0, 0.0]
                new_pk = _snap_peak(ff, pp, click["x"], state["bgp"], bg)
                new_pk["shape"] = shape              # shape chosen for new peaks
                state["peaks"] = state["peaks"] + [new_pk]
                state["fitted"], state["unc"] = False, None
            elif trig.startswith("fit-drag") and dragged:
                k = int(dragged["peak"])
                if 0 <= k < len(state["peaks"]):
                    state["peaks"] = [dict(pk) for pk in state["peaks"]]
                    pk = state["peaks"][k]
                    for what in ("centre", "amp", "fwhm"):     # whichever were dragged
                        if dragged.get(what) is not None:
                            pk[what] = float(dragged[what])
                    pk["amp"], pk["fwhm"] = max(pk["amp"], 0.0), max(abs(pk["fwhm"]), 1e-6)
                    state["fitted"], state["unc"] = False, None
            elif trig.startswith("fit-run") and run:
                if not state["peaks"]:
                    raise ValueError("add at least one peak first (turn on Peak fitting, then "
                                     "click on a peak)")
                if not state.get("region"):
                    _set_region(state, run["xrange"])
                state = run_fit(ff, pp, sigma, state)
            elif trig.startswith("fit-region-set") and region_set:
                _set_region(state, region_set["xrange"])     # dragged lines / "Region = View"
                state["fitted"], state["unc"] = False, None
            elif trig.startswith("fit-region-lo") or trig.startswith("fit-region-hi"):
                if region_lo is not None and region_hi is not None:
                    _set_region(state, [region_lo, region_hi])
                    state["fitted"], state["unc"] = False, None
            elif trig.startswith("fit-table.active_cell"):
                k, what = int(active["row"]), active["column_id"][4:]
                if 0 <= k < len(state["peaks"]) and (
                        what != "eta" or _shapes(state)[k] == "pvoigt"):
                    state["peaks"] = [dict(pk) for pk in state["peaks"]]
                    fix = set(state["peaks"][k].get("fix") or [])
                    fix ^= {what}
                    state["peaks"][k]["fix"] = sorted(fix)
            elif trig.startswith("fit-bg-fix"):
                state["bg_fix"] = list(bg_fix or [])
            elif trig.startswith("iso-names-apply"):
                # names / harmonics given in the "Send to Isocurve" popup
                ok = bool(names_apply) and names_apply.get("sum_key") == key
                state["peaks"] = [dict(pk) for pk in state["peaks"]]
                for k, nm in ((names_apply.get("names") or {}) if ok else {}).items():
                    k = int(k)
                    if 0 <= k < len(state["peaks"]):
                        state["peaks"][k]["ion"] = nm["ion"]
                        state["peaks"][k]["harmonic"] = nm["harmonic"]
            elif trig.startswith("fit-table") and _only_names_changed(state, table):
                # ion / harmonic typed in the table: keep the fit and its
                # uncertainties (nothing else changed)
                state["peaks"] = [dict(pk) for pk in state["peaks"]]
                for pk, r in zip(state["peaks"], table):
                    pk["ion"] = str(r.get("ion") or "").strip()
                    pk["harmonic"] = _harmonic_or_none(r.get("harmonic"))
            elif trig.startswith("fit-table"):
                # hand edits (or deleted rows): rebuild the peaks from the table
                peaks = []
                for r in table or []:
                    n = int(r.get("peak") or 0)
                    old = state["peaks"][n - 1] if 0 < n <= len(state["peaks"]) else {}
                    try:
                        eta = r.get("eta")
                        shp = r.get("shape") if r.get("shape") in SHAPE_LABELS else \
                            old.get("shape", shape)
                        pk = {"centre": float(r["centre"]), "amp": float(r["amp"]),
                              "fwhm": abs(float(r["fwhm"])), "shape": shp,
                              "eta": float(eta) if eta not in (None, "") else old.get("eta", 0.5),
                              "fix": list(old.get("fix") or []),
                              "ion": str(r.get("ion") or "").strip(),
                              "harmonic": _harmonic_or_none(r.get("harmonic")),
                              "lim": {k: (_num_or_none(r[k]) if k in r
                                          else (old.get("lim") or {}).get(k))
                                      for k in LIMIT_KEYS}}
                        pk["eta"] = min(max(pk["eta"], 0.0), 1.0)
                    except (TypeError, ValueError, KeyError):
                        pk = dict(old)
                    if pk:
                        peaks.append(pk)
                state["peaks"], state["fitted"], state["unc"] = peaks, False, None
            elif trig.startswith("fit-b0") or trig.startswith("fit-b1"):
                if b0 is not None:
                    state["bgp"][0] = float(b0)
                if b1 is not None:
                    state["bgp"][1] = float(b1)
                state["fitted"], state["unc"] = False, None
            elif trig.startswith("fit-bg"):
                state["fitted"], state["unc"] = False, None
            # (fit-shape only sets the shape for peaks added from now on)
        except Exception as exc:
            traceback.print_exc()
            error = str(exc)

    # eta columns only when at least one peak is pseudo-Voigt
    pv = bool(state) and "pvoigt" in _shapes(state)
    cols = fit_columns(pv, bool(show_limits))
    nu = dash.no_update
    if state is None:
        msg = ("Sum the spectra first, then turn on peak fitting."
               if trig.startswith(("fit-click", "fit-run")) else "")
        return None, [], cols, None, None, "", "", msg, FIT_PANEL_HIDDEN, None, [], None, None, []
    # fix-cell clicks: deselect the cell so the next click toggles again
    act, sel = (None, []) if trig.startswith("fit-table.active_cell") else (nu, nu)
    reg = state.get("region")
    rlo = round(reg[0], 1) if reg else None
    rhi = round(reg[1], 1) if reg else None
    nb = _n_bg(state["bg"])
    bunc = (state.get("unc") or {}).get("bgp") or [None, None]
    b0v = float(f"{state['bgp'][0]:.6g}") if nb >= 1 else None
    b1v = float(f"{state['bgp'][1]:.6g}") if nb == 2 else None
    status = _fit_status(state)
    if error:
        status = html.Span(f"Fit error: {error}", style={"color": "#c0392b"})
    panel = FIT_PANEL_SHOWN if state["peaks"] else FIT_PANEL_HIDDEN
    bfix = set(state.get("bg_fix") or [])

    def bunc_txt(j):
        if j == 0 and nb < 1 or j == 1 and nb < 2:
            return ""
        if f"b{j}" in bfix:
            return "fixed"
        return ("± " + _fmt_num(bunc[j], ".3g")) if bunc[j] is not None else ""
    return (state, fit_table_rows(state), cols, b0v, b1v, bunc_txt(0), bunc_txt(1),
            status, panel, act, sel, rlo, rhi, sorted(bfix))


@app.callback(
    Output("fit-resid", "figure"),
    Output("fit-resid-box", "style"),
    Input("fit-state", "data"),
    State("sum-key", "data"),
)
def show_residuals(state, key):
    hidden = RESID_BOX_HIDDEN
    with _SUM_CACHE_LOCK:
        cached = _SUM_CACHE.get(key)
    if cached is None or not state or not state.get("peaks") or not state.get("region"):
        return empty_fig(), hidden
    ff, pp, sigma = cached
    m = _region_mask(ff, state["region"])
    x, y = ff[m], pp[m]
    p, _ = _pack(state)
    model = _model_from_vector(x - state["xref"], p, _shapes(state), state["bg"])
    fig = go.Figure()
    fig.add_hrect(y0=-sigma, y1=sigma, fillcolor=ACCENT_PALE, opacity=0.9, line_width=0,
                  layer="below")
    fig.add_hline(y=0, line=dict(color="#999", width=1))
    fig.add_trace(go.Scatter(x=x, y=y - model, mode="lines", name="data − fit",
                             line=dict(shape="hvh", color="black", width=1),
                             hovertemplate="f = %{x:.1f} Hz<br>data − fit = %{y:.4g}<extra></extra>"))
    fig.update_layout(template="plotly_white", margin=dict(l=60, r=20, t=30, b=40),
                      title=dict(text="Residuals (data − fit); Shaded Band: ±1σ Noise",
                                 font=dict(size=13)),
                      xaxis_title="Frequency (Hz)", yaxis_title="Data − Fit", dragmode="pan",
                      showlegend=False, font=dict(size=12), uirevision=str(state.get("region")))
    return fig, RESID_BOX_SHOWN


# Snippet: show/hide the half-life-excluded lines in the browser, so the
# current zoom/pan is kept (the next snippet drawn uses the same setting)
app.clientside_callback(
    """
    function(value) {
        var gd = document.querySelector('#snippet-graph .js-plotly-plot');
        if (!gd || !gd.data) return window.dash_clientside.no_update;
        var idx = [];
        gd.data.forEach(function(t, i) { if (t.name === '%s') idx.push(i); });
        if (idx.length) Plotly.restyle(gd, {visible: (value && value.length) ? true : false}, idx);
        return window.dash_clientside.no_update;
    }
    """ % EXCLUDED_TRACE,
    Output("_snippet-excluded-hook", "children"),
    Input("snippet-show-excluded", "value"),
    prevent_initial_call=True,
)


@app.callback(
    Output("results-page-ions", "style"),
    Output("results-page-ref", "style"),
    Input("results-subtabs", "value"),
)
def show_results_page(page):
    return (RESULTS_PAGE_GRID if page == "ions" else RESULTS_PAGE_HIDDEN,
            RESULTS_PAGE_GRID if page == "ref" else RESULTS_PAGE_HIDDEN)


@app.callback(
    Output("further-page-align", "style"),
    Output("further-page-iso", "style"),
    Input("further-subtabs", "value"),
)
def show_further_page(page):
    base = {"padding": "14px 18px", "overflowY": "auto", "height": "calc(100vh - 82px)"}
    return ({**base, "display": "block" if page == "align" else "none"},
            {**base, "display": "block" if page == "iso" else "none"})


# ---------------------------------------------------------------------------
# Search ions (as in the spectrogram) and the h±1 / q±1 / A±1 buttons
# ---------------------------------------------------------------------------

def _srch_half_width_hz(width_khz):
    try:
        v = float(width_khz)
    except (TypeError, ValueError):
        return SRCH_DEFAULT_KHZ * 1000.0
    return v * 1000.0 if np.isfinite(v) and v > 0 else SRCH_DEFAULT_KHZ * 1000.0


def _srch_lines(key):
    sims = _SUM_SIMS.get(key)
    if sims is None or not len(sims):
        return None
    if "t_half" not in sims:                 # (older cache entry)
        sims = sims.assign(t_half="", excluded=False, is_ref=False)
    return sims


@app.callback(
    Output("srch-modal", "style"),
    Input("btn-srch", "n_clicks"),
    Input("srch-close", "n_clicks"),
    Input("srch-zoom", "data"),
    prevent_initial_call=True,
)
def toggle_search_modal(_open, _close, zoom):
    trig = dash.callback_context.triggered[0]["prop_id"] if dash.callback_context.triggered else ""
    if trig.startswith("btn-srch"):
        return SRCH_MODAL_SHOWN
    if trig.startswith("srch-zoom") and not (zoom or {}).get("from_list"):
        raise PreventUpdate                       # h±1 buttons: modal stays as it is
    return SRCH_MODAL_HIDDEN


@app.callback(
    Output("srch-list", "children"),
    Output("srch-status", "children"),
    Output("srch-hits", "data"),
    Output("srch-width", "style"),
    Input("srch-ion", "value"),
    Input("srch-width", "value"),
    Input("sum-key", "data"),
)
def search_ions(q, width, key):
    """Simulated lines whose ion name contains the query (case-insensitive),
    sorted by frequency (first 300 shown)."""
    w_style = {"width": "80px", "marginRight": "4px"}
    try:
        wv = float(width) if width not in (None, "") else None
    except (TypeError, ValueError):
        wv = -1
    if wv is not None and not (np.isfinite(wv) and wv > 0):
        w_style["borderColor"] = "#c0392b"
    trig = dash.callback_context.triggered[0]["prop_id"] if dash.callback_context.triggered else ""
    if trig.startswith("srch-width") and wv is not None and wv > 0:
        save_config({"search_width_khz": wv})
    q = (q or "").strip().lower()
    if not q:
        return [], "", [], w_style
    sims = _srch_lines(key)
    if sims is None:
        return [], html.Span("✗ no simulated lines: sum the spectra / load a spectrum first",
                             style={"color": "#c0392b"}), [], w_style
    hit = sims[sims["ion"].astype(str).str.lower().str.contains(q, regex=False)]
    hit = hit.sort_values("f_sim")
    n = len(hit)
    if not n:
        return [], html.Span(f"No simulated lines for \"{q}\" in this spectrum",
                             style={"color": "#c0392b"}), [], w_style
    hits = [{"ion": str(r.ion), "h": int(r.harmonic), "f": float(r.f_sim),
             "t_half": r.t_half or "", "excluded": bool(r.excluded), "is_ref": bool(r.is_ref)}
            for r in hit.head(300).itertuples()]
    hw = _srch_half_width_hz(width)
    cell = {"padding": "3px 12px", "fontSize": "13px"}
    head = {**cell, "fontSize": "11px", "fontWeight": "700", "color": "#888",
            "textTransform": "uppercase", "letterSpacing": "0.04em"}
    out = [html.Div([html.Span(t, style=head) for t in ("Ion", "Harmonic", "f_sim (Hz)", "T½")],
                    style={**SRCH_GRID, "borderBottom": "1px solid #dddddd", "position": "sticky",
                           "top": "0", "background": "white"}, className="theia-box")]
    for i, l in enumerate(hits):
        col = {"color": "#aaa"} if l["excluded"] else {}
        out.append(html.Div([
            html.Span(l["ion"] + ("  (ref)" if l["is_ref"] else ""),
                      style={**cell, **col, "fontWeight": "600"}),
            html.Span(f"h{l['h']}", style={**cell, **col}),
            html.Span(f"{l['f']:,.1f}", style={**cell, **col, "fontFamily": "monospace"}),
            html.Span(l["t_half"] or "?", style={**cell, **col}),
        ], id={"type": "srch-line", "i": i}, n_clicks=0, className="theia-fb-row",
           title=f"Zoom to {l['f'] - hw:,.0f} – {l['f'] + hw:,.0f} Hz"
                 + ("  (below the half-life cut)" if l["excluded"] else ""),
           style={**SRCH_GRID, "cursor": "pointer"}))
    status = html.Span(f"{n} line{'' if n == 1 else 's'}"
                       + (f" (first {len(hits)} shown)" if n > len(hits) else ""),
                       style={"color": "#2e7d32"})
    return out, status, hits, w_style


# step buttons: (kind, step) -> button id
SRCH_STEP_BTNS = {("h", -1): "btn-harm-dn", ("h", 1): "btn-harm-up",
                  ("q", -1): "btn-q-dn", ("q", 1): "btn-q-up",
                  ("A", -1): "btn-a-dn", ("A", 1): "btn-a-up"}


def _srch_step_target(cur, kind, step):
    """(ion name, harmonic) that a step button goes to from the current line:
    'h' same ion at h + step; 'q' same nuclide (isomer kept) with charge
    q + step, same harmonic; 'A' isotope A + step (ground state), same charge
    and harmonic.  None if the name can't be parsed or q would be 0."""
    # imported here: .isocurve imports this module
    from .isocurve import ION_NAME_RE
    if kind == "h":
        return cur["ion"], cur["h"] + step
    m = ION_NAME_RE.fullmatch(cur["ion"])
    if not m:
        return None
    iso, A, sym, q = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
    if kind == "q":
        return (f"{iso}{A}{sym}{q + step}+", cur["h"]) if q + step >= 1 else None
    return f"{A + step}{sym}{q}+", cur["h"]


def _srch_neighbour(key, cur, step, kind="h"):
    """Simulated line that the h/q/A step button (kind, step) loads, or None
    (with the target name) if it isn't among the simulated lines."""
    sims = _srch_lines(key)
    if sims is None or not cur:
        return None
    tgt = _srch_step_target(cur, kind, step)
    if tgt is None:
        return None
    m = sims[(sims["ion"].astype(str) == tgt[0]) & (sims["harmonic"] == tgt[1])]
    if not len(m):
        return None
    r = m.iloc[0]
    return {"ion": tgt[0], "h": int(r["harmonic"]), "f": float(r["f_sim"]),
            "excluded": bool(r["excluded"]) if "excluded" in m.columns else False}


def _step_labels(ion, h):
    """Labels between the step buttons: ('h212', '77+', '198Ir') for
    '198Ir77+', h212; q and A are '' if the ion name can't be parsed."""
    m = re.match(r"^([a-z]?)(\d+)([A-Z][a-z]?)(\d+)\+$", str(ion).strip())
    if not m:
        return f"h{int(h)}", "", ""
    iso, a, sym, q = m.groups()
    return f"h{int(h)}", f"{q}+", f"{iso}{a}{sym}"


@app.callback(
    Output("srch-zoom", "data"),
    Output("srch-current", "data"),
    Output("harm-cur", "children"),
    Output("q-cur", "children"),
    Output("a-cur", "children"),
    Input({"type": "srch-line", "i": ALL}, "n_clicks"),
    *[Input(b, "n_clicks") for b in SRCH_STEP_BTNS.values()],
    State("srch-hits", "data"),
    State("srch-current", "data"),
    State("srch-width", "value"),
    State("sum-key", "data"),
    prevent_initial_call=True,
)
def search_pick(_rows, *args):
    """A line from the list, or an h±1 / q±1 / A±1 button: zoom to f ± width."""
    hits, cur, width, key = args[len(SRCH_STEP_BTNS):]
    trig = dash.callback_context.triggered[0] if dash.callback_context.triggered else None
    if not trig or not trig["value"]:
        raise PreventUpdate
    pid = trig["prop_id"]
    from_list = '"srch-line"' in pid
    if from_list:
        i = json.loads(pid.rsplit(".", 1)[0])["i"]
        if not hits or i >= len(hits):
            raise PreventUpdate
        line = hits[i]
    else:
        bid = pid.rsplit(".", 1)[0]
        kind, step = next(k for k, b in SRCH_STEP_BTNS.items() if b == bid)
        line = _srch_neighbour(key, cur, step, kind)
        if line is None:
            raise PreventUpdate
    hw = _srch_half_width_hz(width)
    zoom = {"x": [line["f"] - hw, line["f"] + hw], "from_list": from_list, "t": time.time()}
    # the label is set here, in the same reply as the zoom, so it never lags
    return zoom, {"ion": line["ion"], "h": int(line["h"])}, *_step_labels(line["ion"], line["h"])


@app.callback(
    Output("harm-grp", "style"),
    Output("harm-cur", "title"),
    *[o for b in SRCH_STEP_BTNS.values()
      for o in (Output(b, "disabled"), Output(b, "title"))],
    Input("srch-current", "data"),
    Input("sum-key", "data"),
    Input("srch-width", "value"),
)
def show_harmonic_step(cur, key, width):
    if not cur:
        return {"display": "none"}, "", *([True, ""] * len(SRCH_STEP_BTNS))
    hw = _srch_half_width_hz(width)
    out = []
    for kind, step in SRCH_STEP_BTNS:
        l = _srch_neighbour(key, cur, step, kind)
        if l is None:
            tgt = _srch_step_target(cur, kind, step)
            out += [True, (f"{tgt[0]}, h{tgt[1]} is not among the simulated lines of this "
                           f"spectrum") if tgt else "No charge state below 1+"]
        else:
            out += [False, f"Zoom to {l['ion']}, h{l['h']}: {l['f'] - hw:,.0f} – "
                           f"{l['f'] + hw:,.0f} Hz"
                           + ("  (below the half-life cut)" if l["excluded"] else "")]
    return ({"display": "inline-flex", "alignItems": "center", "gap": "10px",
             "padding": "3px 8px", "border": "1px solid #adb5bd", "borderRadius": "6px",
             "marginRight": "8px"},
            "Harmonic of the ion chosen in Search ions", *out)


# Zoom the summed plot to the chosen window; y fitted to the data inside it
app.clientside_callback(
    """
    function(z) {
        var gd = document.querySelector('#sum-graph .js-plotly-plot');
        if (!z || !gd || !gd._fullData || !gd._fullData.length)
            return window.dash_clientside.no_update;
        var lo = z.x[0], hi = z.x[1], X = gd._fullData[0].x, Y = gd._fullData[0].y;
        var ymin = Infinity, ymax = -Infinity;
        for (var i = 0; i < X.length; i++) {
            if (X[i] < lo || X[i] > hi) continue;
            var y = Y[i];
            if (isFinite(y)) { if (y < ymin) ymin = y; if (y > ymax) ymax = y; }
        }
        var upd = {'xaxis.range': [lo, hi]};
        if (isFinite(ymin) && isFinite(ymax)) {
            var pad = 0.05 * ((ymax - ymin) || Math.abs(ymax) || 1);
            upd['yaxis.range'] = [ymin - pad, ymax + pad];
        } else {
            upd['yaxis.autorange'] = true;
        }
        Plotly.relayout(gd, upd);
        return window.dash_clientside.no_update;
    }
    """,
    Output("_srch-zoom-hook", "children"),
    Input("srch-zoom", "data"),
    prevent_initial_call=True,
)
