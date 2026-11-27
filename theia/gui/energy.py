"""
Further output: the energy calculation.

Part of Theia (theia/theia.py); split out of theia.py in Nov 2026.
"""

import json
import re
import traceback

import numpy as np
import dash
from dash import ALL, html, Input, Output, State

# the calculation itself is shared with the spectrogram (core/energy.py)
from core.energy import (add_pick, calibrate, calibration_text, empty_state, fmt_pm,  # noqa: F401
                         local_peak_fit, lookup_mass_excess, mass_excess_difference, pick_peak,
                         status_text)
from core.physics import build_ion_database

from .common import ACCENT, ACCENT_PALE
from .app import app
from .info import INFO_TEXTS
from .further_files import (_fit_into_patch, FIT_TRACE_STYLES, pid_base, _pid_display, PID_KEYS,
                            pid_overrides, pid_settings, _RESTORE_ENERGY, _SUM_ANNOTATIONS,
                            _SUM_CACHE, _SUM_CACHE_LOCK, _SUM_META, _SUM_SHAPES, sum_sim_traces,
                            _SUM_SIMS)
from .further_fit import fit_overlay, SUGG_BOX_STYLE


# ===========================================================================
# Aligning and fitting: energy calculation
# ===========================================================================
#
# 1. Calibration: pick two peaks (a fitted peak's centre is used if you click
#    on one, otherwise the peak is fitted on its own in a small window), name
#    them (NUBASE names, e.g. 207Tl81+ or m207Tl), press Calibrate:
#        k = (M2 - M1) / (f2 - f1)          [keV per Hz]
#    M = ion mass (mass excess - q m_e c^2 + binding energy of the q removed
#    electrons, NIST) when the names carry charges; atomic mass excess otherwise.
#    For an isomer / ground-state pair of the same nuclide the nuclear part is
#    the excitation-energy difference (with its own, smaller uncertainty).
# 2. Then pick peaks in pairs A, B:  dE = E(B) - E(A) = k (f_B - f_A).

ENERGY_TRACE = "energy picks"
ENERGY_PAIR_TRACE = "energy pairs"
# index of the first simulated-line trace in the summed figure: spectrum,
# 4 fit traces, energy picks, energy pairs, then the 3 simulated-line traces
SIM_TRACE_INDEX = 1 + len(FIT_TRACE_STYLES) + 2


_fmt_pm = fmt_pm


def suggest_ions(key, f, n=6):
    """Simulated ions nearest to frequency f (from the summed plot's lines)."""
    sims = _SUM_SIMS.get(key)
    if sims is None or not len(sims):
        return []
    d = (sims["f_sim"] - f).abs()
    near = sims.assign(d=d).sort_values("d").head(n)
    return [(r.ion, int(r.harmonic), float(r.f_sim - f)) for r in near.itertuples()]


_empty_energy = empty_state


@app.callback(
    Output("energy-state", "data"),
    Output("energy-name-1", "value"),
    Output("energy-name-2", "value"),
    Input("energy-click", "data"),
    Input("energy-name-1", "value"),
    Input("energy-name-2", "value"),
    Input({"type": "energy-sugg", "peak": ALL, "name": ALL, "h": ALL}, "n_clicks"),
    Input("btn-energy-calibrate", "n_clicks"),
    Input("btn-energy-reset", "n_clicks"),
    Input("btn-energy-clear-pairs", "n_clicks"),
    Input("sum-key", "data"),
    State("energy-state", "data"),
    State("fit-state", "data"),
    State("fit-shape", "value"),
    prevent_initial_call=True,
)
def update_energy(click, name1, name2, _sugg, _cal, _reset, _clear, key, est, fit_state, shape):
    trig = dash.callback_context.triggered[0] if dash.callback_context.triggered else None
    nu = dash.no_update
    if trig is None:
        return nu, nu, nu
    pid, val = trig["prop_id"], trig["value"]
    est = dict(est or _empty_energy())
    est["msg"] = ""
    if pid.startswith("sum-key"):
        saved = _RESTORE_ENERGY.pop(key, None)          # saved last session
        if saved:
            names = saved.get("names") or ["", ""]
            return dict(_empty_energy(), **(saved.get("state") or {})), names[0], names[1]
        return _empty_energy(), "", ""
    if pid.startswith("btn-energy-reset"):
        return _empty_energy(), "", ""
    if pid.startswith("btn-energy-clear-pairs"):
        est["pairs"], est["pending"] = [], None
        return est, nu, nu
    if '"energy-sugg"' in pid:
        if not val:
            return nu, nu, nu
        idd = json.loads(pid.rsplit(".", 1)[0])
        if idd["peak"] == 1:
            return nu, idd["name"], nu
        return nu, nu, idd["name"]
    if pid.startswith("energy-name"):
        est["k"], est["dk"] = None, None              # names changed: recalibrate
        return est, nu, nu

    with _SUM_CACHE_LOCK:
        cached = _SUM_CACHE.get(key)
    if cached is None:
        est["msg"] = "Sum the spectra first."
        return est, nu, nu
    ff, pp, sigma = cached

    if pid.startswith("energy-click") and click:
        try:
            pk = pick_peak(ff, pp, sigma, float(click["x"]), fit_state, shape or "gauss")
        except Exception as exc:
            traceback.print_exc()
            est["msg"] = f"Could not fit the peak there: {exc}"
            return est, nu, nu
        sugg = suggest_ions(key, pk["f"]) if est.get("k") is None else None
        return add_pick(est, pk, sugg), nu, nu

    if pid.startswith("btn-energy-calibrate"):
        return calibrate(est, name1, name2), nu, nu
    return nu, nu, nu


@app.callback(
    Output("energy-sugg-1", "style"),
    Output("energy-sugg-2", "style"),
    Input("energy-suggest-btn-1", "n_clicks"),
    Input("energy-suggest-btn-2", "n_clicks"),
    Input({"type": "energy-sugg", "peak": ALL, "name": ALL, "h": ALL}, "n_clicks"),
    State("energy-sugg-1", "style"),
    State("energy-sugg-2", "style"),
    prevent_initial_call=True,
)
def toggle_suggestions(_b1, _b2, _rows, st1, st2):
    """'Suggest' opens/closes that peak's list; choosing an ion closes it."""
    trig = dash.callback_context.triggered[0] if dash.callback_context.triggered else None
    if not trig:
        return dash.no_update, dash.no_update
    pid = trig["prop_id"]
    open_ = lambda st: (st or {}).get("display") != "none"
    show = lambda on: {**SUGG_BOX_STYLE, "display": "block" if on else "none"}
    if pid.startswith("energy-suggest-btn-1"):
        return show(not open_(st1)), show(False)
    if pid.startswith("energy-suggest-btn-2"):
        return show(False), show(not open_(st2))
    if '"energy-sugg"' in pid and trig["value"]:
        return show(False), show(False)
    return dash.no_update, dash.no_update


@app.callback(
    Output("energy-calib-info", "children"),
    Output("energy-sugg-1", "children"),
    Output("energy-sugg-2", "children"),
    Output("energy-name-1-check", "children"),
    Output("energy-name-2-check", "children"),
    Output("energy-f-1", "children"),
    Output("energy-f-2", "children"),
    Output("energy-fwhm-1", "children"),
    Output("energy-fwhm-2", "children"),
    Output("energy-status", "children"),
    Output("energy-table", "data"),
    Output("energy-panel", "style"),
    Input("energy-state", "data"),
    Input("energy-name-1", "value"),
    Input("energy-name-2", "value"),
    Input("energy-mode", "data"),
)
def render_energy(est, name1, name2, on):
    est = est or _empty_energy()
    calib = est.get("calib") or []

    def sugg(i):
        """Table of nearby simulated ions (nearest first); rows are clickable."""
        if i >= len(calib):
            return html.Div("Click the peak on the plot first.",
                            style={"fontSize": "12px", "color": "#888", "padding": "4px 12px"})
        rows = calib[i].get("suggest") or []
        if not rows:
            return html.Div("No simulated ions near this peak (sum the spectra first?).",
                            style={"fontSize": "12px", "color": "#888", "padding": "4px 12px"})
        cell = {"padding": "3px 12px", "fontSize": "13px"}
        head = {**cell, "fontSize": "11px", "fontWeight": "700", "color": "#888",
                "textTransform": "uppercase", "letterSpacing": "0.04em"}
        grid = {"display": "grid", "gridTemplateColumns": "1.3fr 0.9fr 1.4fr 1fr"}
        f0 = calib[i]["f"]
        out = [html.Div(f"{len(rows)} nearby, nearest centre first, click one",
                        style={"fontSize": "12px", "color": "#2e7d32", "padding": "2px 12px 4px"}),
               html.Div([html.Span("Ion", style=head), html.Span("Harmonic", style=head),
                         html.Span("f_sim (Hz)", style=head), html.Span("From Centre", style=head)],
                        style={**grid, "borderBottom": "1px solid #dddddd"})]
        for ion, h, d in rows:
            out.append(html.Div([
                html.Span(ion, style={**cell, "fontWeight": "600"}),
                html.Span(f"h{h}", style=cell),
                html.Span(f"{f0 + d:,.1f}", style={**cell, "fontFamily": "monospace"}),
                html.Span(f"{d:+.0f} Hz", style={**cell, "fontFamily": "monospace"}),
            ], id={"type": "energy-sugg", "peak": i + 1, "name": ion, "h": h}, n_clicks=0,
               className="theia-fb-row", style={**grid, "cursor": "pointer"}))
        return out

    def check(name):
        """Mass excess column (NUBASE), or why the name is not valid."""
        if not name:
            return html.Span("–", style={"color": "#888"})
        try:
            r = lookup_mass_excess(name)
            out = [html.Span(f"{r['me']:.1f} ± {r['dme']:.1f}" + (" #" if r["sys"] else ""),
                             style={"fontFamily": "monospace"})]
            if r["iso"]:
                out.append(html.Div(f"E_x = {r['ex']:.2f} ± {r['dex']:.2f}",
                                    style={"fontSize": "12px", "color": "#666",
                                           "fontFamily": "monospace"}))
            return out
        except ValueError as exc:
            return html.Span(f"✗ {exc}", style={"color": "#c0392b", "fontSize": "12px"})

    def freq(i):
        if i >= len(calib):
            return html.Span("click the peak on the plot", style={"color": "#888",
                                                               "fontFamily": "inherit"})
        p = calib[i]
        src = re.sub(r"\s*\(FWHM [^)]*\)", "", p["src"])     # FWHM goes in its own column
        return [html.Span(_fmt_pm(p["f"], p["df"])),
                html.Div(src, style={"fontSize": "11px", "color": "#888", "fontFamily": "inherit"})]

    def fwhm(i):
        if i >= len(calib):
            return html.Span("–", style={"color": "#888"})
        p = calib[i]
        if p_get(p, "fwhm") is None:
            # peaks picked by an older version only have it in the source text
            m = re.search(r"FWHM ([\d.]+) Hz", p.get("src", ""))
            return m.group(1) if m else html.Span("–", style={"color": "#888"})
        return _fmt_pm(p["fwhm"], p.get("dfwhm"))

    def p_get(p, k):
        v = p.get(k)
        return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else v

    info = []
    if est.get("k") is not None:
        info.append(html.Div([html.B("Calibration: "), calibration_text(est)],
                             style={"marginTop": "4px"}))

    if est.get("msg"):
        status = html.Span(est["msg"], style={"color": "#c0392b"})
    else:
        status = status_text(est, on)

    rows = [{"pair": i, "f_A": round(p["a"]["f"], 1), "f_B": round(p["b"]["f"], 1),
             "df": _fmt_pm(p["dfreq"], p["ddf"]), "de": _fmt_pm(p["de"], p["dde"], ".2f"),
             "src": f"A: {p['a']['src']};  B: {p['b']['src']}"}
            for i, p in enumerate(est.get("pairs") or [], start=1)]
    # panel shown while the mode is on, or whenever there is something in it
    has = bool(calib or est.get("pairs") or est.get("k") is not None)
    panel = {"display": "block" if (on or has) else "none"}
    return (info, sugg(0), sugg(1), check(name1), check(name2), freq(0), freq(1), fwhm(0),
            fwhm(1), status, rows, panel)


@app.callback(
    Output("sum-graph", "figure", allow_duplicate=True),
    Input("fit-state", "data"),
    Input("energy-state", "data"),
    State("sum-view", "data"),
    State("sum-key", "data"),
    prevent_initial_call=True,
)
def draw_overlays(fit_state, est, view, key):
    """Update the fit and energy overlays of the summed plot in ONE Patch.

    (Two callbacks patching the figure at the same moment - as happens when
    both are restored on start-up - can lose one of the updates.)  Runs only
    when the fit or the energy picks change, never on zoom/pan: patching the
    131k-point figure during a zoom makes the plot fight the zoom."""
    with _SUM_CACHE_LOCK:
        cached = _SUM_CACHE.get(key)
    if cached is None:
        return dash.no_update
    trig = dash.callback_context.triggered[0]["prop_id"] if dash.callback_context.triggered else ""
    patch = dash.Patch()
    if not fit_state or fit_state.get("sum_key") == key:
        _fit_into_patch(patch, fit_state, cached)
    _energy_into_patch(patch, est, fit_state, view, key, cached, trig)
    return patch


def _fit_shapes(fit_state, key):
    """Fit region (two draggable orange lines, named fit-region-lo/hi, with a
    pale tint between them) and each peak's centre limits (grey band)."""
    if not fit_state or fit_state.get("sum_key") != key or not fit_state.get("region"):
        return []
    lo, hi = fit_state["region"]
    line = dict(color=ACCENT, width=2)
    out = [dict(type="rect", xref="x", yref="paper", x0=lo, x1=hi, y0=0, y1=1,
                fillcolor=ACCENT_PALE, opacity=0.5, line_width=0, layer="below",
                name="fit-region-tint"),
           dict(type="line", xref="x", yref="paper", x0=lo, x1=lo, y0=0, y1=1, line=line,
                layer="above", name="fit-region-lo"),
           dict(type="line", xref="x", yref="paper", x0=hi, x1=hi, y0=0, y1=1, line=line,
                layer="above", name="fit-region-hi")]
    for pk in fit_state.get("peaks") or []:
        lim = pk.get("lim") or {}
        cmin, cmax = lim.get("centre_min"), lim.get("centre_max")
        if cmin is None and cmax is None:
            continue
        out.append(dict(type="rect", xref="x", yref="paper",
                        x0=lo if cmin is None else max(cmin, lo),
                        x1=hi if cmax is None else min(cmax, hi), y0=0, y1=0.04,
                        fillcolor="#999999", opacity=0.5, line_width=0, layer="above",
                        name="fit-centre-limits"))
    return out


def _energy_into_patch(patch, est, fit_state, view, key, cached, trig):
    """Picked-peak labels (C1, C2, A1/B1, ...) and the pair brackets + labels."""
    ff, pp, _ = cached
    est = est or _empty_energy()
    pts = [(p["f"], f"C{i}") for i, p in enumerate(est.get("calib") or [], start=1)]
    for i, pr in enumerate(est.get("pairs") or [], start=1):
        pts += [(pr["a"]["f"], f"A{i}"), (pr["b"]["f"], f"B{i}")]
    if est.get("pending"):
        pts.append((est["pending"]["f"], f"A{len(est.get('pairs') or []) + 1}"))
    merged = {}                                   # one label per position ("C1, A1")
    for f, lab in pts:
        merged.setdefault(round(f, 1), []).append(lab)
    pts = [(f, ", ".join(labs)) for f, labs in merged.items()]
    idx = 1 + len(FIT_TRACE_STYLES)
    patch["data"][idx]["x"] = [p[0] for p in pts]
    patch["data"][idx]["y"] = [float(np.interp(p[0], ff, pp)) for p in pts]
    patch["data"][idx]["text"] = [p[1] for p in pts]

    # a bracket just above each pair, from centre A to centre B, with dE +- sigma.
    # Drawn as layout shapes whose vertical size is in PIXELS above the
    # highest data / fit point between the peaks (ysizemode="pixel"), so it
    # sits the same distance above the peaks at any zoom without redrawing.
    fx, fy = fit_overlay(ff, fit_state)[0] if fit_state else ([], [])
    fx, fy = np.asarray(fx, dtype=float), np.asarray(fy, dtype=float)
    shapes, labels = [], []
    line = dict(color="black", width=1.5)
    for i, pr in enumerate(est.get("pairs") or []):
        fa, fb = pr["a"]["f"], pr["b"]["f"]
        lo, hi = min(fa, fb), max(fa, fb)
        ya, yb = float(np.interp(fa, ff, pp)), float(np.interp(fb, ff, pp))
        m = (ff >= lo) & (ff <= hi)
        top = max(ya, yb, float(np.nanmax(pp[m])) if m.any() else ya)   # clear the data
        mf = (fx >= lo) & (fx <= hi)
        if mf.any():                                                    # ... and the fit
            top = max(top, float(np.nanmax(fy[mf])))
        # pixels above `top`: clear of the peak numbers / pick labels; later
        # pairs are stacked higher
        h = 34 + 46 * i
        mid = 0.5 * (fa + fb)
        common = dict(type="line", xref="x", yref="y", ysizemode="pixel", yanchor=top,
                      line=line, layer="above")
        shapes += [dict(common, x0=fa, x1=fb, y0=h, y1=h),              # horizontal
                   dict(common, x0=fa, x1=fa, y0=h - 7, y1=h),          # tick down at A
                   dict(common, x0=fb, x1=fb, y0=h - 7, y1=h),          # tick down at B
                   dict(common, x0=mid, x1=mid, y0=h, y1=h + 7)]        # tick up, middle
        # label: drawn on top of everything, black text on an opaque white
        # box with a black border
        labels.append(dict(x=mid, y=top, xref="x", yref="y", yshift=h + 8, yanchor="bottom",
                           text=f"ΔE = {_fmt_pm(pr['de'], pr['dde'], '.2f')} keV",
                           showarrow=False, font=dict(color="black", size=26),
                           bgcolor="white", bordercolor="black", borderwidth=1.5, borderpad=4,
                           opacity=1))
    patch["data"][idx + 1]["x"] = []                 # (old bracket trace, no longer used)
    patch["data"][idx + 1]["y"] = []
    patch["layout"]["shapes"] = (list(_SUM_SHAPES.get(key, [])) + _fit_shapes(fit_state, key)
                                 + shapes)
    patch["layout"]["annotations"] = list(_SUM_ANNOTATIONS.get(key, [])) + labels
    return patch


@app.callback(
    Output("pid-modal", "style"),
    *[Output(f"pid-{k}", "value") for k in PID_KEYS],
    Output("pid-error", "children"),
    Output("pid-settings", "data"),
    Input("btn-pid", "n_clicks"),
    Input("pid-apply", "n_clicks"),
    Input("pid-reset", "n_clicks"),
    Input("pid-cancel", "n_clicks"),
    *[State(f"pid-{k}", "value") for k in PID_KEYS],
    State("further-files", "data"),
    State("pid-settings", "data"),
    State("further-source", "value"),
    prevent_initial_call=True,
)
def pid_popup(_open, _apply, _reset, _cancel, *args):
    """PID Settings popup: opens filled with the settings in use for the
    chosen data source (file / Viewer settings + edits); Apply validates and
    stores the edits; Reset to file drops them."""
    values, (files, store, source) = list(args[:len(PID_KEYS)]), args[len(PID_KEYS):]
    source = source or "sum"
    overrides = pid_overrides(store, source)

    def stored(over):
        new = {"sum": pid_overrides(store, "sum"), "file": pid_overrides(store, "file")}
        new[source] = over
        return new
    shown = {"display": "flex", "position": "fixed", "inset": "0", "zIndex": "1000",
             "background": "rgba(0,0,0,0.35)", "alignItems": "center", "justifyContent": "center"}
    hidden = {**shown, "display": "none"}
    nu = dash.no_update
    keep = [nu] * len(PID_KEYS)
    trig = dash.callback_context.triggered[0]["prop_id"] if dash.callback_context.triggered else ""

    def filled(over):
        raw = {**pid_base(source, files), **(over or {})}
        return [_pid_display(raw, k) for k in PID_KEYS]

    if trig.startswith("pid-cancel"):
        return (hidden, *keep, "", nu)
    try:
        if trig.startswith("btn-pid"):
            return (shown, *filled(overrides), "", nu)
        if trig.startswith("pid-reset"):
            msg = ("Reset to the Viewer settings." if source == "file"
                   else "Reset to the settings of the loaded file.")
            return (shown, *filled(None), msg, stored(None))
        if trig.startswith("pid-apply"):
            new = dict(zip(PID_KEYS, values))
            s = pid_settings(files, new, source)             # validates the numbers
            build_ion_database(*s["ion_key"])                # ...and the reference ion
            return (hidden, *keep, "", stored(new))
    except Exception as exc:
        msg = html.Span(f"⚠ {exc}", style={"color": "#c0392b"})
        if trig.startswith("btn-pid"):
            return (shown, *keep, msg, nu)
        return (nu, *keep, msg, nu)
    return (nu, *keep, nu, nu)


@app.callback(
    Output("sum-graph", "figure", allow_duplicate=True),
    Output("sum-sim-info", "children", allow_duplicate=True),
    Input("pid-settings", "data"),
    State("sum-key", "data"),
    State("further-files", "data"),
    State("sum-unsure", "value"),
    State("sum-show-sims", "value"),
    prevent_initial_call=True,
)
def apply_pid(store, key, files, unsure, show_sims):
    """Redraw only the simulated lines with the new PID settings of the
    plotted data source (no re-sum; the zoom is kept)."""
    with _SUM_CACHE_LOCK:
        cached = _SUM_CACHE.get(key)
    if cached is None:
        return dash.no_update, dash.no_update
    ff, pp, _ = cached
    source = (_SUM_META.get(key) or {}).get("source", "sum")
    try:
        traces, text = sum_sim_traces(ff, pp, files, bool(unsure), pid_overrides(store, source),
                                      key=key, source=source)
    except Exception as exc:
        traceback.print_exc()
        return dash.no_update, f"simulated lines not drawn: {exc}"
    patch = dash.Patch()
    base = SIM_TRACE_INDEX
    for i, (x, y, h) in enumerate(traces):
        patch["data"][base + i]["x"] = x
        patch["data"][base + i]["y"] = y
        patch["data"][base + i]["hovertext"] = h
        patch["data"][base + i]["visible"] = bool(show_sims)
    return patch, text


@app.callback(
    Output("info-modal", "style"),
    Output("info-title", "children"),
    Output("info-body", "children"),
    Input({"type": "info-btn", "key": ALL}, "n_clicks"),
    Input("info-close", "n_clicks"),
    prevent_initial_call=True,
)
def show_info(_clicks, _close):
    """ⓘ buttons: show that section's help in a popup; ✕ closes it."""
    shown = {"display": "flex", "position": "fixed", "inset": "0", "zIndex": "1000",
             "background": "rgba(0,0,0,0.35)", "alignItems": "center", "justifyContent": "center"}
    trig = dash.callback_context.triggered[0] if dash.callback_context.triggered else None
    if not trig or not trig["value"]:
        return dash.no_update, dash.no_update, dash.no_update
    if trig["prop_id"].startswith("info-close"):
        return {**shown, "display": "none"}, dash.no_update, dash.no_update
    key = json.loads(trig["prop_id"].rsplit(".", 1)[0])["key"]
    title, lines = INFO_TEXTS.get(key, ("Information", []))
    body = html.Ul([html.Li(t, style={"marginBottom": "4px"}) for t in lines],
                   style={"paddingLeft": "18px", "margin": "0"})
    return shown, title, body
