"""
Further output: the isochronicity curve.

Part of Theia (theia/theia.py); split out of theia.py in Nov 2026.
"""

import datetime
import json
import os
import re
import time
import traceback

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import ALL, dash_table, dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate

from core import jsonfile, paths
from core.scan import _load_json

from .common import (ACCENT, ACCENT_PALE, BOX_BORDER, BOX_STYLE, HELP_STYLE, ISO_MODAL_HIDDEN,
                     ISO_MODAL_SHOWN, LABEL_STYLE, TABLE_CELL, TABLE_HEADER, TABLE_ODD)
from .app import app
from .figures import empty_fig
from .info import info_button
from .further_files import _SUM_META
from .further_fit import (CALIB_BORDER, CALIB_TD, CALIB_TH, FIX_OFF, FIX_ON, _harmonic_or_none,
                          _peak_area, _shapes)


# "Send to Isocurve": asks for the ion names / harmonics that are missing and
# confirms replacing points already on the curve
iso_modal = html.Div([
    html.Div([
        html.Div([
            html.Span("Send to Isochronicity Curve", style={"fontWeight": "700", "fontSize": "15px"}),
            html.Button("✕", id="iso-cancel-x", className="btn btn-outline-secondary btn-sm"),
        ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center",
                  "marginBottom": "8px"}),
        html.Div(id="iso-modal-body", style={"fontSize": "13px"}),
        html.Div([
            html.Button("Cancel", id="iso-cancel", className="btn btn-outline-secondary btn-sm",
                        style={"marginRight": "8px"}),
            html.Button("Send", id="iso-confirm", className="btn btn-sm",
                        style={"backgroundColor": ACCENT, "color": "white", "fontWeight": "600",
                               "minWidth": "80px"}),
        ], style={"display": "flex", "justifyContent": "flex-end", "marginTop": "10px"}),
    ], className="theia-box",
       style={"border": BOX_BORDER, "borderRadius": "6px", "background": "white",
              "width": "720px", "maxWidth": "94vw", "maxHeight": "92vh", "overflowY": "auto",
              "padding": "14px 16px", "boxShadow": "0 8px 30px rgba(0,0,0,0.3)"}),
], id="iso-modal",
   style={"display": "none", "position": "fixed", "inset": "0", "zIndex": "1000",
          "background": "rgba(0,0,0,0.35)", "alignItems": "center", "justifyContent": "center"})


# Points sent from the peak fitting ("Send to Isocurve"), one per ion and
# harmonic, kept in outputs/isocurve/isocurve_points.csv
ISO_TABLE_COLUMNS = [
    ("ion", "Ion"), ("harmonic", "h"), ("A", "A"), ("Z", "Z"), ("q", "q"),
    ("centre_hz", "Centre (Hz)"), ("d_centre_hz", "± Centre"),
    ("f_rev_hz", "f_rev (Hz)"), ("d_f_rev_hz", "± f_rev"),
    ("fwhm_hz", "FWHM (Hz)"), ("d_fwhm_hz", "± FWHM"),
    ("amp", "Amplitude"), ("d_amp", "± Amp"), ("area", "Area"), ("d_area", "± Area"),
    ("shape", "Shape"), ("eta", "η"), ("d_eta", "± η"), ("fixed", "Fixed"),
    ("chi2_ndf", "χ²/ndf"), ("source", "Data Source"), ("sent", "Sent"),
]

# derived values for the curve (calculated from the points, not saved)
ISO_CURVE_COLUMNS = [
    ("include", "In Fit"), ("ion", "Ion"), ("harmonic", "h"), ("T_ns", "T (ns)"),
    ("fwhm_T_ps", "FWHM(T) (ps)"), ("sigma_T_ps", "σ(T) (ps)"),
    ("d_sigma_T_ps", "Err σ(T) (ps)"),
]

# curve-fit settings (saved per experiment in settings/isocurve.json)
ISO_FIT_DEFAULTS = {"circumference_m": 108.36, "c_m_s": 299792458.0,
                    "gt_min": 1.0, "gt_max": 3.0, "sp_min_pct": 0.0, "sp_max_pct": 0.2,
                    "ssys_min_ps": 0.0, "ssys_max_ps": 0.4}
ISO_BIN_DEFAULTS = {"rec_freq": "10e6", "n_bins": "2^17"}   # bin-width error inputs
ISO_SETTING_INPUTS = [   # first key, second key (range) or None, label, width
    ("circumference_m", None, "Circumference (m)", "100px"),
    ("c_m_s", None, "c (m/s)", "170px"),
    ("gt_min", "gt_max", "γ_t  min – max", "70px"),
    ("sp_min_pct", "sp_max_pct", "σp/p (%)  min – max", "70px"),
    ("ssys_min_ps", "ssys_max_ps", "σ_sys (ps)  min – max", "70px"),
]
ISO_SETTING_KEYS = [k for k, k2, _, _ in ISO_SETTING_INPUTS for k in (k, k2) if k]

# heading of a fold-away section (html.Details); the ▸/▾ marker is styled in _INPUT_CSS
FOLD_SUMMARY = {"fontWeight": "700", "fontSize": "14px", "cursor": "pointer", "marginBottom": "4px"}

isochronicity_tab = html.Div([
    html.Div([
        html.H5("Isochronicity Curve", style={"fontWeight": "700", "margin": "0 8px 0 0"}),
        info_button("iso"),
    ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}),
    html.Div([
        html.Div(id="iso-info", style={"fontSize": "13px", "marginBottom": "6px"}),
        # table folds away (closed when the page loads)
        html.Details([
            html.Summary("Isochronicity Points", style=FOLD_SUMMARY),
            dash_table.DataTable(
                id="iso-table", data=[], row_deletable=True, editable=False,
                sort_action="native",
                columns=[{"name": n, "id": c} for c, n in ISO_TABLE_COLUMNS],
                style_header={**TABLE_HEADER, "whiteSpace": "normal", "height": "auto"},
                style_cell={**TABLE_CELL, "fontSize": "12px"},
                style_data_conditional=TABLE_ODD,
                style_cell_conditional=[
                    {"if": {"column_id": c}, "textAlign": "left", "fontFamily": "inherit"}
                    for c in ("ion", "shape", "fixed", "source", "sent")] + [
                    {"if": {"column_id": "ion"}, "fontWeight": "600"},
                    {"if": {"column_id": "source"}, "color": "#666", "maxWidth": "360px",
                     "overflow": "hidden", "textOverflow": "ellipsis"}],
                tooltip_duration=None,
                style_table={"overflowX": "auto"}),
        ], open=False, className="theia-fold"),
    ], className="theia-box", style={**BOX_STYLE, "padding": "8px 10px"}),
    # values for the isochronicity curve, calculated from the points above
    html.Details([
        html.Summary("Isochronicity Curve Parameters", style=FOLD_SUMMARY),
        html.Small("T = 10⁹ · h / centre  ·  FWHM(T) = (FWHM / centre) · T · 1000  ·  "
                   "σ(T) = FWHM(T) / 2√(2 ln 2)  ·  err σ(T) = (± FWHM / FWHM) · FWHM(T) / "
                   "2√(2 ln 2)   (2√(2 ln 2) ≈ 2.3548)",
                   style={**HELP_STYLE, "display": "block", "marginBottom": "6px"}),
        dash_table.DataTable(
            id="iso-curve-table", data=[], sort_action="native",
            columns=[{"name": n, "id": c} for c, n in ISO_CURVE_COLUMNS],
            style_header={**TABLE_HEADER, "whiteSpace": "normal", "height": "auto"},
            style_cell={**TABLE_CELL, "fontSize": "12px"},
            style_data_conditional=TABLE_ODD + [
                {"if": {"column_id": "include"}, "cursor": "pointer", "textAlign": "center",
                 "fontSize": "13px", "color": "#222"},
                {"if": {"filter_query": f'{{include}} = "{FIX_OFF}"'}, "color": "#aaa"},
                {"if": {"filter_query": '{include} = "–"'}, "color": "#aaa"}],
            style_cell_conditional=[{"if": {"column_id": "ion"}, "textAlign": "left",
                                     "fontFamily": "inherit", "fontWeight": "600"}],
            style_table={"overflowX": "auto", "width": "fit-content", "maxWidth": "100%"}),
    ], open=False, className="theia-box theia-fold",
       style={**BOX_STYLE, "padding": "8px 10px", "marginTop": "10px"}),
    # fit of the curve  σ_T(T) = sqrt( ((1/γ² − 1/γ_t²) · σp/p · T)² + σ_sys² )
    html.Div([
        html.Div([
            html.Span("Isochronicity Curve Fit", style={"fontWeight": "700", "fontSize": "14px",
                                                        "marginRight": "8px"}),
            info_button("isofit", {"marginRight": "14px"}),
            html.Button("Save Figure", id="btn-iso-save-fig",
                        className="btn btn-outline-secondary btn-sm",
                        title="Save the plot (matplotlib) as PNG and PDF in outputs/figures/",
                        style={"marginRight": "10px"}),
            html.Span(id="iso-save-status", style={"fontSize": "12px"}),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "6px"}),
        html.Div([
            *[html.Div([
                html.Label(lab, style=LABEL_STYLE),
                html.Div([
                    dcc.Input(id=f"iso-set-{k}", type="number", step="any", debounce=True,
                              className="form-control form-control-sm",
                              style={"width": w, "fontSize": "13px"}),
                    *([html.Span("–", style={"margin": "0 4px"}),
                       dcc.Input(id=f"iso-set-{k2}", type="number", step="any", debounce=True,
                                 className="form-control form-control-sm",
                                 style={"width": w, "fontSize": "13px"})] if k2 else []),
                ], style={"display": "flex", "alignItems": "center"}),
            ], style={"marginRight": "18px"}) for k, k2, lab, w in ISO_SETTING_INPUTS],
            dcc.Checklist(id="iso-set-floor", value=[],
                          options=[{"label": " Systematic floor: auto (χ²/ndf = 1)",
                                    "value": "auto"}],
                          inputStyle={"marginRight": "4px"},
                          labelStyle={"fontSize": "13px"},
                          style={"paddingBottom": "6px"}),
        ], style={"display": "flex", "alignItems": "flex-end", "flexWrap": "wrap",
                  "rowGap": "6px", "marginBottom": "8px"}),
        html.Div([
            dcc.Checklist(id="iso-set-binerr", value=[],
                          options=[{"label": " Bin-width error: bin/√12 (1σ) on every FWHM",
                                    "value": "on"}],
                          inputStyle={"marginRight": "4px"}, labelStyle={"fontSize": "13px"},
                          style={"marginRight": "14px"}),
            html.Span("Recording Frequency (Hz)", style={**LABEL_STYLE, "margin": "0 6px 0 0"}),
            dcc.Input(id="iso-set-recfreq", type="text", debounce=True,
                      value=ISO_BIN_DEFAULTS["rec_freq"], placeholder="10e6",
                      className="form-control form-control-sm",
                      style={"width": "100px", "marginRight": "12px", "fontSize": "13px"}),
            html.Span("Frequency Bins", style={**LABEL_STYLE, "margin": "0 6px 0 0"}),
            dcc.Input(id="iso-set-nbins", type="text", debounce=True,
                      value=ISO_BIN_DEFAULTS["n_bins"], placeholder="2^17",
                      className="form-control form-control-sm",
                      style={"width": "90px", "marginRight": "12px", "fontSize": "13px"}),
            html.Span(id="iso-bin-info", style={"fontSize": "13px", "fontFamily": "monospace"}),
        ], style={"display": "flex", "alignItems": "center", "flexWrap": "wrap",
                  "rowGap": "6px", "marginBottom": "8px"}),
        html.Div(id="iso-fit-result", style={"fontSize": "13px", "marginBottom": "6px"}),
        dcc.Graph(id="iso-graph", figure=empty_fig("No points yet"),
                  config={"scrollZoom": True, "displaylogo": False},
                  style={"height": "750px"}),
    ], className="theia-box", style={**BOX_STYLE, "padding": "8px 10px", "marginTop": "10px"}),
    # residuals of the curve fit, in a box of their own
    html.Div(
        dcc.Graph(id="iso-resid", figure=empty_fig(),
                  config={"scrollZoom": True, "displaylogo": False},
                  style={"height": "340px"}),
        className="theia-box", style={**BOX_STYLE, "padding": "8px 10px", "marginTop": "10px"}),
    dcc.Store(id="iso-version", data=0),
    dcc.Store(id="iso-points-ts", data=None),
    html.Div(id="_iso-label-hook", style={"display": "none"}),
    html.Div(id="_iso-resid-label-hook", style={"display": "none"}),
    dcc.Store(id="iso-settings-init", data=0),
], id="further-page-iso", style={"padding": "14px 18px", "overflowY": "auto",
                                 "height": "calc(100vh - 82px)", "display": "none"})


# ---------------------------------------------------------------------------
# Isochronicity curve: points sent from the peak fitting
# ---------------------------------------------------------------------------
# One row per (ion, harmonic) in outputs/isocurve/isocurve_points.csv of the
# active experiment.  Sending the same ion and harmonic again replaces the row.

ISO_POINTS_NAME = "isocurve_points.csv"
ISO_FILE_COLUMNS = ["ion", "harmonic", "A", "Z", "q", "isomer",
                    "centre_hz", "d_centre_hz", "f_rev_hz", "d_f_rev_hz",
                    "fwhm_hz", "d_fwhm_hz", "amp", "d_amp", "area", "d_area",
                    "shape", "eta", "d_eta", "fixed", "chi2_ndf", "ndf",
                    "region_lo_hz", "region_hi_hz", "background", "source", "sent",
                    "include"]
ION_NAME_RE = re.compile(r"\s*([a-z]?)(\d+)([A-Z][a-z]?)(\d+)\+\s*")


def iso_points_path():
    return os.path.join(paths.outputs("isocurve"), ISO_POINTS_NAME)


def load_iso_points():
    path = iso_points_path()
    if not os.path.exists(path):
        return pd.DataFrame(columns=ISO_FILE_COLUMNS)
    df = pd.read_csv(path)
    for c in ISO_FILE_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    return df


def save_iso_points(df):
    """Write the points file (via a temporary file, so a crash never leaves
    half a file), sorted by f_rev."""
    path = iso_points_path()
    df = df.sort_values("f_rev_hz", kind="stable") if len(df) else df
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def parse_ion_name(name):
    """Ion name with charge ('207Tl81+', 'm207Tl81+'), checked against NUBASE.
    Returns dict(ion, A, Z, q, iso) or raises ValueError."""
    # imported here: .energy imports this module
    from .energy import lookup_mass_excess
    m = ION_NAME_RE.fullmatch(name or "")
    if not m:
        raise ValueError("give the ion with its charge, e.g. 207Tl81+ or m207Tl81+")
    iso, A, sym, q = m.group(1), m.group(2), m.group(3), int(m.group(4))
    r = lookup_mass_excess(f"{iso}{A}{sym}")
    if not 1 <= q <= r["Z"]:
        raise ValueError(f"charge {q}+ is not possible for {sym} (Z = {r['Z']})")
    return dict(ion=f"{iso}{A}{sym}{q}+", A=r["A"], Z=r["Z"], q=q, iso=iso)


def _check_peak_name(ion, harmonic):
    """(parsed dict, None) if the ion name and harmonic are valid, else
    (None, reason)."""
    if not (ion or "").strip():
        return None, "no ion name"
    try:
        r = parse_ion_name(ion)
    except ValueError as exc:
        return None, str(exc)
    h = _harmonic_or_none(harmonic)
    if h is None:
        return None, "no harmonic (a whole number ≥ 1)"
    return dict(r, harmonic=h), None


def _iso_source(key, files, align, ion_key):
    """Short description of the spectrum the fit was made on."""
    meta = _SUM_META.get(key) or {}
    if meta.get("source") == "file":
        return f"file {os.path.join(meta.get('dir') or '', meta.get('name') or '')}"
    names = [os.path.basename(f.get("path", f) if isinstance(f, dict) else str(f))
             for f in (files or [])]
    how = {"none": "not aligned", "strongest": "aligned on strongest peak"}.get(align)
    if align == "identified" and ion_key:
        ion, harm = str(ion_key).rsplit("|", 1)
        how = f"aligned on {ion} h={harm}"
    return f"sum of {', '.join(names) or 'confirmation files'} ({how or align})"


def iso_points_from_fit(state, named, source):
    """One point per fitted peak.  named: peak index -> parsed name dict."""
    unc = (state.get("unc") or {}).get("peaks") or [{} for _ in state["peaks"]]
    shapes = _shapes(state)
    now = datetime.datetime.now().isoformat(timespec="seconds")
    rows = []
    for i, (pk, e, shape) in enumerate(zip(state["peaks"], unc, shapes)):
        nm = named[i]
        fix = set(pk.get("fix") or [])

        def u(k):
            v = e.get(k)
            return 0.0 if k in fix else (float(v) if v is not None else np.nan)
        h = nm["harmonic"]
        pv = shape == "pvoigt"
        rows.append({
            "ion": nm["ion"], "harmonic": h, "A": nm["A"], "Z": nm["Z"], "q": nm["q"],
            "isomer": nm["iso"] or "",
            "centre_hz": pk["centre"], "d_centre_hz": u("centre"),
            "f_rev_hz": pk["centre"] / h, "d_f_rev_hz": u("centre") / h,
            "fwhm_hz": pk["fwhm"], "d_fwhm_hz": u("fwhm"),
            "amp": pk["amp"], "d_amp": u("amp"),
            "area": _peak_area(pk["amp"], pk["fwhm"], pk.get("eta", 0.5), shape),
            "d_area": float(e.get("area")) if e.get("area") is not None else np.nan,
            "shape": shape, "eta": pk.get("eta", 0.5) if pv else np.nan,
            "d_eta": u("eta") if pv else np.nan,
            "fixed": " ".join(sorted(k for k in fix if k != "eta" or pv)),
            "chi2_ndf": state.get("chi2"), "ndf": state.get("ndf"),
            "region_lo_hz": state["region"][0], "region_hi_hz": state["region"][1],
            "background": state.get("bg"), "source": source, "sent": now,
            "include": True,
        })
    return pd.DataFrame(rows, columns=ISO_FILE_COLUMNS)


def _iso_modal_body(state, key, missing, typed, errors, replaces):
    """Popup content: an input row (name, harmonic, suggestions) for each peak
    without a valid name, and the points that would be replaced."""
    # imported here: .energy imports this module
    from .energy import suggest_ions
    out = []
    if missing:
        out.append(html.Div("Each peak sent to the isochronicity curve needs an ion name with "
                            "its charge (e.g. 207Tl81+ or m207Tl81+) and a harmonic.  Click a "
                            "suggestion (nearby simulated ions) or type them.",
                            style={"marginBottom": "8px"}))
        th = {**CALIB_TH}
        rows = []
        for i in missing:
            pk = state["peaks"][i]
            name, h = typed.get(i, (pk.get("ion") or "", pk.get("harmonic")))
            sugg = suggest_ions(key, pk["centre"], n=5)
            chips = [html.Button(f"{ion} h{hh} ({d:+.0f} Hz)",
                                 id={"type": "iso-sugg", "peak": i, "name": ion, "h": hh},
                                 n_clicks=0, className="btn btn-outline-secondary btn-sm",
                                 style={"margin": "0 4px 4px 0", "fontSize": "12px",
                                        "padding": "1px 6px"})
                     for ion, hh, d in sugg]
            err = errors.get(i)
            rows.append(html.Tr([
                html.Td(f"{i + 1}", style={**CALIB_TD, "fontWeight": "700", "color": ACCENT}),
                html.Td(f"{pk['centre']:,.1f}", style={**CALIB_TD, "fontFamily": "monospace"}),
                html.Td([
                    html.Div([
                        dcc.Input(id={"type": "iso-name", "peak": i}, type="text", value=name,
                                  placeholder="e.g. 207Tl81+", debounce=False,
                                  className="form-control form-control-sm",
                                  style={"width": "150px", "marginRight": "6px"}),
                        html.Span("h", style={"marginRight": "4px"}),
                        dcc.Input(id={"type": "iso-h", "peak": i}, type="number", value=h,
                                  min=1, step=1, debounce=False,
                                  className="form-control form-control-sm",
                                  style={"width": "70px"}),
                    ], style={"display": "flex", "alignItems": "center"}),
                    html.Div(f"✗ {err}", style={"color": "#c0392b", "fontSize": "12px",
                                                "marginTop": "3px"}) if err else None,
                ], style=CALIB_TD),
                html.Td(chips or html.Span("no simulated ions nearby",
                                           style={"color": "#888", "fontSize": "12px"}),
                        style=CALIB_TD),
            ]))
        out.append(html.Table([
            html.Thead(html.Tr([html.Th(t, style=th) for t in
                                ("Peak", "Centre (Hz)", "Ion and Harmonic", "Suggestions")])),
            html.Tbody(rows)],
            style={"borderCollapse": "collapse", "border": CALIB_BORDER, "width": "100%"}))
    if replaces:
        out.append(html.Div([
            html.B("Already on the curve, will be replaced: ", style={"color": ACCENT}),
            ", ".join(replaces)], style={"marginTop": "10px"}))
    return out


@app.callback(
    Output("iso-modal", "style"),
    Output("iso-modal-body", "children"),
    Output("iso-send-status", "children"),
    Output("iso-names-apply", "data"),
    Output("iso-version", "data"),
    Input("btn-iso-send", "n_clicks"),
    Input("iso-confirm", "n_clicks"),
    Input("iso-cancel", "n_clicks"),
    Input("iso-cancel-x", "n_clicks"),
    State({"type": "iso-name", "peak": ALL}, "value"),
    State({"type": "iso-name", "peak": ALL}, "id"),
    State({"type": "iso-h", "peak": ALL}, "value"),
    State({"type": "iso-h", "peak": ALL}, "id"),
    State("fit-state", "data"),
    State("sum-key", "data"),
    State("further-files", "data"),
    State("sum-align", "value"),
    State("sum-ion", "value"),
    State("iso-version", "data"),
    prevent_initial_call=True,
)
def send_to_isocurve(_send, _confirm, _cancel, _x, name_vals, name_ids, h_vals, h_ids,
                     state, key, files, align, ion_key, version):
    """'Send to Isocurve': every fitted peak needs a valid ion name and
    harmonic.  Missing ones (and replacing existing points) go through the
    popup; otherwise the points are written straight away."""
    trig = dash.callback_context.triggered[0]["prop_id"] if dash.callback_context.triggered else ""
    nu = dash.no_update
    if trig.startswith("iso-cancel"):
        return ISO_MODAL_HIDDEN, nu, "", nu, nu

    def fail(msg):
        return ISO_MODAL_HIDDEN, nu, html.Span(msg, style={"color": "#c0392b"}), nu, nu
    if not state or not state.get("peaks"):
        return fail("No peaks to send: add peaks and press Fit first.")
    if not state.get("fitted"):
        return fail("The peaks are not fitted (values changed since the last Fit): "
                    "press Fit first.")

    # names typed in the popup (only present when it is open)
    typed = {}
    if trig.startswith("iso-confirm"):
        hs = {d["peak"]: v for d, v in zip(h_ids or [], h_vals or [])}
        for d, v in zip(name_ids or [], name_vals or []):
            typed[d["peak"]] = ((v or "").strip(), hs.get(d["peak"]))

    named, errors = {}, {}
    for i, pk in enumerate(state["peaks"]):
        ion, h = typed.get(i, (pk.get("ion") or "", pk.get("harmonic")))
        r, err = _check_peak_name(ion, h)
        if r is None:
            errors[i] = err
        else:
            named[i] = r
    keys = [(r["ion"], r["harmonic"]) for r in named.values()]
    dup = sorted({f"{a} h{b}" for a, b in keys if keys.count((a, b)) > 1})
    try:
        old = load_iso_points()
    except Exception as exc:
        traceback.print_exc()
        return fail(f"Could not read {iso_points_path()}: {exc}")
    have = set(zip(old["ion"].astype(str), pd.to_numeric(old["harmonic"]).astype("Int64")))
    replaces = sorted({f"{a} h{b}" for a, b in keys if (a, b) in have})

    missing = sorted(set(errors) | set(typed))
    if errors or dup or (replaces and not trig.startswith("iso-confirm")):
        # on the first press a plain "no ion name" needs no error text
        shown = errors if trig.startswith("iso-confirm") else {
            i: e for i, e in errors.items() if e != "no ion name"}
        body = _iso_modal_body(state, key, missing, typed, shown, replaces)
        if dup:
            body.append(html.Div(f"✗ The same ion and harmonic is given to more than one peak: "
                                 f"{', '.join(dup)}", style={"color": "#c0392b",
                                                             "marginTop": "8px"}))
        return ISO_MODAL_SHOWN, body, "", nu, nu

    # all named: write the points (replacing the same ion + harmonic)
    new = iso_points_from_fit(state, named, _iso_source(key, files, align, ion_key))
    if len(old):
        drop = set(zip(new["ion"], new["harmonic"].astype(int)))
        old_keys = list(zip(old["ion"].astype(str), pd.to_numeric(old["harmonic"]).astype(int)))
        old = old[[k not in drop for k in old_keys]]
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new
    try:
        save_iso_points(out)
    except Exception as exc:
        traceback.print_exc()
        return fail(f"Could not write {iso_points_path()}: {exc}")
    sent = ", ".join(f"{r['ion']} h{r['harmonic']}" for r in named.values())
    msg = html.Span([html.B("Sent to isocurve: ", style={"color": "#2e7d32"}), sent]
                    + ([f"  (replaced {', '.join(replaces)})"] if replaces else []))
    apply = {"sum_key": key, "t": time.time(),
             "names": {str(i): {"ion": r["ion"], "harmonic": r["harmonic"]}
                       for i, r in named.items()}}
    return ISO_MODAL_HIDDEN, [], msg, apply, (version or 0) + 1


@app.callback(
    Output({"type": "iso-name", "peak": ALL}, "value"),
    Output({"type": "iso-h", "peak": ALL}, "value"),
    Input({"type": "iso-sugg", "peak": ALL, "name": ALL, "h": ALL}, "n_clicks"),
    State({"type": "iso-name", "peak": ALL}, "value"),
    State({"type": "iso-name", "peak": ALL}, "id"),
    State({"type": "iso-h", "peak": ALL}, "value"),
    prevent_initial_call=True,
)
def pick_iso_suggestion(_clicks, names, ids, hs):
    """A suggestion in the popup fills that peak's name and harmonic."""
    trig = dash.callback_context.triggered[0] if dash.callback_context.triggered else None
    if not trig or not trig["value"] or '"iso-sugg"' not in trig["prop_id"]:
        raise PreventUpdate
    idd = json.loads(trig["prop_id"].rsplit(".", 1)[0])
    names, hs = list(names), list(hs)
    for j, d in enumerate(ids):
        if d["peak"] == idd["peak"]:
            names[j], hs[j] = idd["name"], int(idd["h"])
    return names, hs


def _iso_cell(v, fmt):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
    return "–" if not np.isfinite(f) else format(f, fmt)


FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))     # ≈ 2.3548


def iso_curve_values(df):
    """Revolution time and its spread for each point:
        f_rev    = centre / h
        T (ns)   = 1e9 / f_rev
        FWHM(T)  = (FWHM / centre) * T * 1000        (ps; same relative width as f)
        σ(T)     = FWHM(T) / 2√(2 ln 2)
        err σ(T) = (dFWHM / FWHM) * FWHM(T) / 2√(2 ln 2)"""
    centre = pd.to_numeric(df["centre_hz"], errors="coerce")
    h = pd.to_numeric(df["harmonic"], errors="coerce")
    fwhm = pd.to_numeric(df["fwhm_hz"], errors="coerce")
    d_fwhm = pd.to_numeric(df["d_fwhm_hz"], errors="coerce")
    t_ns = 1e9 / (centre / h)
    fwhm_t_ps = (fwhm / centre) * t_ns * 1000.0
    return pd.DataFrame({
        "ion": df["ion"].astype(str), "harmonic": h, "T_ns": t_ns, "fwhm_T_ps": fwhm_t_ps,
        "sigma_T_ps": fwhm_t_ps / FWHM_PER_SIGMA,
        "d_sigma_T_ps": (d_fwhm / fwhm) * fwhm_t_ps / FWHM_PER_SIGMA,
    })


ISO_CURVE_FORMATS = {"harmonic": ".0f", "T_ns": ".6f", "fwhm_T_ps": ".4f",
                     "sigma_T_ps": ".4f", "d_sigma_T_ps": ".4f"}


ISO_FORMATS = {"centre_hz": ",.2f", "d_centre_hz": ".2f", "f_rev_hz": ",.4f",
               "d_f_rev_hz": ".4f", "fwhm_hz": ".2f", "d_fwhm_hz": ".2f", "amp": ".5g",
               "d_amp": ".3g", "area": ".5g", "d_area": ".3g", "eta": ".3f", "d_eta": ".3f",
               "chi2_ndf": ".3g", "harmonic": ".0f", "A": ".0f", "Z": ".0f", "q": ".0f"}


@app.callback(
    Output("iso-table", "data"),
    Output("iso-table", "tooltip_data"),
    Output("iso-info", "children"),
    Output("iso-points-ts", "data"),
    Input("iso-version", "data"),
    Input("further-subtabs", "value"),
    Input("iso-table", "data_timestamp"),
    State("iso-table", "data"),
)
def show_iso_points(_version, page, _ts, table):
    """Table of the points on the curve.  Deleting a row removes the point
    from the file."""
    trig = dash.callback_context.triggered[0]["prop_id"] if dash.callback_context.triggered else ""
    path = iso_points_path()
    try:
        df = load_iso_points()
        if trig.startswith("iso-table") and table is not None:
            keep = {(r["ion"], int(float(r["harmonic"]))) for r in table}
            keys = list(zip(df["ion"].astype(str), pd.to_numeric(df["harmonic"]).astype(int)))
            df = df[[k in keep for k in keys]]
            save_iso_points(df)
    except Exception as exc:
        traceback.print_exc()
        return [], [], html.Span(f"Could not read/write {path}: {exc}",
                                 style={"color": "#c0392b"}), time.time()
    if len(df):
        df = df.sort_values("f_rev_hz", kind="stable")
    rows = [{c: _iso_cell(r.get(c), ISO_FORMATS[c]) if c in ISO_FORMATS
             else ("" if pd.isna(r.get(c)) else str(r.get(c)))
             for c, _ in ISO_TABLE_COLUMNS} for r in df.to_dict("records")]
    tips = [{"source": {"value": row["source"], "type": "text"}} for row in rows]
    info = [html.B(f"{len(rows)} point(s)"), "  ·  ",
            html.Span(path, style={"fontFamily": "monospace", "fontSize": "12px",
                                   "color": "#666"})]
    if not rows:
        info.append(html.Div("No points yet: fit peaks in Aligning and fitting and press "
                             "'Send to Isocurve'.", style={"color": "#888", "marginTop": "4px"}))
    return rows, tips, info, time.time()


def _iso_curve_table(vals, floor_on):
    """Rows and columns of the curve-parameters table (built with the fit,
    so the total error with the systematic floor is known)."""
    cols = list(ISO_CURVE_COLUMNS)
    if floor_on:
        cols.append(("d_sigma_T_total_ps", "err σ(T) total (ps)"))
    columns = [{"name": n, "id": c} for c, n in cols]
    if vals is None or not len(vals):
        return [], columns
    rows = []
    for r in vals.sort_values("T_ns", ascending=False, kind="stable").to_dict("records"):
        row = {c: _iso_cell(r.get(c), ISO_CURVE_FORMATS.get(c, ".4f"))
               if c in ISO_CURVE_FORMATS or c == "d_sigma_T_total_ps" else str(r.get(c))
               for c, _ in cols if c != "include"}
        row["include"] = (FIX_ON if r["include"] else FIX_OFF) if r["fittable"] else "–"
        row["id"] = r["id"]
        rows.append(row)
    return rows, columns


# ---- fit of the isochronicity curve ---------------------------------------

def _iso_point_id(ion, h):
    return f"{ion}|{int(float(h))}"


def _iso_included(df):
    """'include' flags of the points (missing / blank = included)."""
    if "include" not in df.columns:
        return np.ones(len(df), dtype=bool)
    def one(v):
        if isinstance(v, str):
            return v.strip().lower() not in ("false", "0", "no")
        try:
            return True if pd.isna(v) else bool(v)
        except (TypeError, ValueError):
            return True
    return np.array([one(v) for v in df["include"]], dtype=bool)


def _iso_fittable(vals):
    """Points that can be fitted: finite T, σ(T) and err σ(T) > 0."""
    return (np.isfinite(vals["T_ns"]) & np.isfinite(vals["sigma_T_ps"])
            & np.isfinite(vals["d_sigma_T_ps"]) & (vals["d_sigma_T_ps"] > 0)).to_numpy()


def iso_sigma_t(t_ns, gamma_t, sp, ssys_ps, circ_m, c_m_s):
    """σ_T (ps) at revolution time T (ns):
    sqrt(((1/γ² − 1/γ_t²)·σp/p·T)² + σ_sys²), 1/γ² = 1 − (C/(c·T))²."""
    t_ns = np.asarray(t_ns, dtype=float)
    inv_g2 = 1.0 - (circ_m / (c_m_s * t_ns * 1e-9)) ** 2
    return np.sqrt(((inv_g2 - 1.0 / gamma_t ** 2) * sp * t_ns * 1000.0) ** 2 + ssys_ps ** 2)


def load_iso_fit_settings():
    saved = _load_json(paths.settings_file("isocurve.json"))
    st = {k: float(saved.get(k, v)) if isinstance(saved.get(k, v), (int, float))
          else v for k, v in ISO_FIT_DEFAULTS.items()}
    st["auto_floor"] = bool(saved.get("auto_floor", False))
    st["bin_err"] = bool(saved.get("bin_err", False))
    st["rec_freq"] = str(saved.get("rec_freq", ISO_BIN_DEFAULTS["rec_freq"]))
    st["n_bins"] = str(saved.get("n_bins", ISO_BIN_DEFAULTS["n_bins"]))
    return st


def parse_number_expr(text):
    """A number typed as 10e6, 131072, 2^17, 2**17 or 10*1e6 (+ − * / ^ only)."""
    import ast
    import operator as op
    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
           ast.Pow: op.pow, ast.USub: op.neg, ast.UAdd: op.pos}

    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.BinOp) and type(n.op) in ops:
            return ops[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in ops:
            return ops[type(n.op)](ev(n.operand))
        raise ValueError
    try:
        v = ev(ast.parse(str(text).strip().replace("^", "**"), mode="eval").body)
    except Exception:
        raise ValueError(f"'{text}' is not a number (e.g. 10e6 or 2^17)")
    if not np.isfinite(v) or v <= 0:
        raise ValueError(f"'{text}' must be a positive number")
    return v


def iso_bin_width(st):
    """Bin width (Hz) = recording frequency / number of bins."""
    return parse_number_expr(st["rec_freq"]) / parse_number_expr(st["n_bins"])


def save_iso_fit_settings(st):
    path = paths.settings_file("isocurve.json")
    try:
        jsonfile.write(path, st)
    except Exception as exc:
        jsonfile.warn(f"could not save the isochronicity fit settings to {path}: "
                      f"{type(exc).__name__}: {exc}")


def check_iso_fit_settings(st):
    if not (st["circumference_m"] > 0 and st["c_m_s"] > 0):
        raise ValueError("circumference and c must be > 0")
    for a, b, what in (("gt_min", "gt_max", "γ_t"), ("sp_min_pct", "sp_max_pct", "σp/p"),
                       ("ssys_min_ps", "ssys_max_ps", "σ_sys")):
        if not st[a] < st[b]:
            raise ValueError(f"{what}: min must be below max")
    if st["gt_min"] <= 0:
        raise ValueError("γ_t min must be > 0")


def fit_iso_curve(t_ns, sig_ps, err_ps, st):
    """Weighted fit of γ_t, σp/p, σ_sys.  Returns dict(popt, perr, chi2, ndf)."""
    from scipy.optimize import curve_fit
    circ, c = st["circumference_m"], st["c_m_s"]
    lb = [st["gt_min"], st["sp_min_pct"] / 100.0, st["ssys_min_ps"]]
    ub = [st["gt_max"], st["sp_max_pct"] / 100.0, st["ssys_max_ps"]]

    def f(t, gt, sp, ss):
        return iso_sigma_t(t, gt, sp, ss, circ, c)
    best = None
    # several γ_t starts: the curve has a minimum near γ = γ_t, and a single
    # start can end in the wrong one
    for gt0 in np.linspace(lb[0], ub[0], 9)[1:-1]:
        p0 = [gt0, 0.5 * (lb[1] + ub[1]), 0.5 * (lb[2] + ub[2])]
        try:
            popt, pcov = curve_fit(f, t_ns, sig_ps, p0=p0, sigma=err_ps, absolute_sigma=True,
                                   bounds=(lb, ub), maxfev=20000)
        except (RuntimeError, ValueError):
            continue
        chi2 = float(np.sum(((sig_ps - f(t_ns, *popt)) / err_ps) ** 2))
        if best is None or chi2 < best[2]:
            best = (popt, pcov, chi2)
    if best is None:
        raise ValueError("the fit did not converge")
    popt, pcov, chi2 = best
    perr = np.sqrt(np.clip(np.diag(pcov), 0, None)) if np.all(np.isfinite(pcov)) \
        else np.full(3, np.nan)
    ndf = len(t_ns) - 3
    return {"popt": [float(v) for v in popt], "perr": [float(v) for v in perr],
            "chi2": chi2, "ndf": ndf, "chi2_ndf": chi2 / ndf if ndf > 0 else np.nan}


def iso_fit_data(st):
    """Points (all, with flags) and the fit result (None if not possible, with
    the reason)."""
    df = load_iso_points()
    if not len(df):
        return None, None, "no points yet"
    vals = iso_curve_values(df)
    vals["include"] = _iso_included(df)
    vals["fittable"] = _iso_fittable(vals)
    vals["id"] = [_iso_point_id(i, h) for i, h in zip(vals["ion"], vals["harmonic"])]
    used = vals[vals["include"] & vals["fittable"]]
    if len(used) < 3:
        return vals, None, f"{len(used)} point(s) in the fit: at least 3 are needed"
    # bin-width error: 1σ of a value spread evenly across one bin, bin/√12
    # (±½ bin is the maximum, not 1σ), on every FWHM, as a σ(T) error (same
    # conversion as err σ(T)): (bin/√12 / centre) · T · 1000 / 2√(2 ln 2)
    bin_sigma = None
    vals["d_sigma_T_bin_ps"] = 0.0
    if st.get("bin_err"):
        try:
            bin_sigma = iso_bin_width(st) / np.sqrt(12.0)
        except ValueError as exc:
            return vals, None, f"bin-width error: {exc}"
        centre = vals["harmonic"] / (vals["T_ns"] * 1e-9)
        vals["d_sigma_T_bin_ps"] = (bin_sigma / centre) * vals["T_ns"] * 1000.0 / FWHM_PER_SIGMA
    used = vals[vals["include"] & vals["fittable"]]
    t, sg = used["T_ns"].to_numpy(), used["sigma_T_ps"].to_numpy()
    err = np.hypot(used["d_sigma_T_ps"].to_numpy(), used["d_sigma_T_bin_ps"].to_numpy())
    floor = 0.0
    try:
        res = fit_iso_curve(t, sg, err, st)
        chi2_stat = res["chi2_ndf"]
        if st.get("auto_floor") and res["ndf"] > 0 and res["chi2_ndf"] > 1:
            floor = iso_auto_floor(t, sg, err, st)
            res = fit_iso_curve(t, sg, np.hypot(err, floor), st)
    except Exception as exc:
        traceback.print_exc()
        return vals, None, f"fit failed: {exc}"
    res["n"] = len(used)
    res["floor"] = floor if st.get("auto_floor") else None
    res["chi2_ndf_stat"] = chi2_stat
    res["bin_sigma_hz"] = bin_sigma
    # total error (statistical ⊕ bin width ⊕ floor) for every point
    vals["d_sigma_T_total_ps"] = np.hypot(np.hypot(vals["d_sigma_T_ps"],
                                                   vals["d_sigma_T_bin_ps"]), floor)
    return vals, res, ""


def iso_auto_floor(t, sg, err, st):
    """Systematic floor f (ps) for which the fit with errors √(err² + f²)
    has χ²/ndf = 1 (bisection on f; χ²/ndf falls as f grows)."""
    def excess(f):
        return fit_iso_curve(t, sg, np.hypot(err, f), st)["chi2_ndf"] - 1.0
    lo, hi = 0.0, max(float(np.nanmax(sg)), 1e-3)
    while excess(hi) > 0 and hi < 1e3:          # make sure the root is bracketed
        lo, hi = hi, hi * 4
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if excess(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-6 * max(hi, 1e-9):
            break
    return 0.5 * (lo + hi)


def _iso_param_lines(res, extras=True):
    """Fit result lines; extras=False leaves out the bin-width / floor lines
    (the parameter box on the plots shows only the fit)."""
    (gt, sp, ss), (dgt, dsp, dss) = res["popt"], res["perr"]
    chi = "–" if not np.isfinite(res["chi2_ndf"]) else f"{res['chi2_ndf']:.3f}"
    lines = [f"γ_t = {gt:.4f} ± {dgt:.6f}",
             f"σp/p = {sp * 100:.4f} ± {dsp * 100:.4f} %",
             f"σ_sys = {ss:.4f} ± {dss:.4f} ps",
             f"χ²/ndf = {chi}  (ndf = {res['ndf']}, {res['n']} points)"]
    if extras and res.get("bin_sigma_hz") is not None:
        lines.append(f"bin-width error = {res['bin_sigma_hz']:.2f} Hz (1σ) on FWHM "
                     f"(bin/√12, {res['bin_sigma_hz'] * np.sqrt(12.0):.3f} Hz bins)")
    if extras and res.get("floor") is not None:
        c0 = res.get("chi2_ndf_stat")
        c0 = "–" if c0 is None or not np.isfinite(c0) else f"{c0:.3f}"
        lines.append(f"systematic floor = {res['floor']:.4f} ps  (auto; χ²/ndf without it = {c0})")
    return lines


def _iso_pull_values(df, res, st, ecol):
    """(σ(T) − fit) / error for the rows of df (NaN where there is no error)."""
    fit = iso_sigma_t(df["T_ns"].to_numpy(), *res["popt"], st["circumference_m"], st["c_m_s"])
    err = df[ecol].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(err > 0, (df["sigma_T_ps"].to_numpy() - fit) / err, np.nan)


def _iso_pulls(used, res, st, ecol):
    """(σ(T) − fit) / error for the fitted points, and the 'within ±1σ' text
    (as in the decay-fit residual panel)."""
    pull = _iso_pull_values(used, res, st, ecol)
    n = len(pull)
    n_in = int(np.sum(np.abs(pull) <= 1.0))
    pct = 100.0 * n_in / n if n else float("nan")
    return pull, f"within ±1σ: {pct:.1f}%  ({n_in}/{n})   Gaussian ≈ 68.3%"


def _iso_curve_x(vals):
    t = vals["T_ns"][np.isfinite(vals["T_ns"])]
    lo, hi = float(t.min()), float(t.max())
    pad = max(0.05 * (hi - lo), 0.05)
    return np.linspace(lo - pad, hi + pad, 2000)


@app.callback(
    Output("iso-set-circumference_m", "value"),
    Output("iso-set-c_m_s", "value"),
    Output("iso-set-gt_min", "value"),
    Output("iso-set-gt_max", "value"),
    Output("iso-set-sp_min_pct", "value"),
    Output("iso-set-sp_max_pct", "value"),
    Output("iso-set-ssys_min_ps", "value"),
    Output("iso-set-ssys_max_ps", "value"),
    Output("iso-set-floor", "value"),
    Output("iso-set-binerr", "value"),
    Output("iso-set-recfreq", "value"),
    Output("iso-set-nbins", "value"),
    Input("iso-settings-init", "data"),
)
def init_iso_settings(_):
    st = load_iso_fit_settings()
    return tuple(st[k] for k in ISO_SETTING_KEYS) + (
        ["auto"] if st["auto_floor"] else [], ["on"] if st["bin_err"] else [],
        st["rec_freq"], st["n_bins"])


@app.callback(
    Output("iso-graph", "figure"),
    Output("iso-resid", "figure"),
    Output("iso-fit-result", "children"),
    Output("iso-curve-table", "data"),
    Output("iso-curve-table", "columns"),
    Input("iso-points-ts", "data"),
    *[Input(f"iso-set-{k}", "value") for k in ISO_SETTING_KEYS],
    Input("iso-set-floor", "value"),
    Input("iso-set-binerr", "value"),
    Input("iso-set-recfreq", "value"),
    Input("iso-set-nbins", "value"),
    prevent_initial_call=True,
)
def fit_and_plot_iso(_ts, *args):
    """Fit the curve to the included points and draw it (with residuals)."""
    setting_vals, (floor_val, bin_val, rec_freq, n_bins) = args[:-4], args[-4:]
    trig = dash.callback_context.triggered[0]["prop_id"] if dash.callback_context.triggered else ""
    st = load_iso_fit_settings()
    err = ""
    if all(v is not None for v in setting_vals):
        new = {k: float(v) for k, v in zip(ISO_SETTING_KEYS, setting_vals)}
        try:
            check_iso_fit_settings(new)
            st = new
        except ValueError as exc:
            err = f"Settings not used: {exc}"
    else:
        err = "Settings not used: fill in every box"
    if floor_val is not None:
        st["auto_floor"] = "auto" in floor_val
    if bin_val is not None:
        st["bin_err"] = "on" in bin_val
    st["rec_freq"] = rec_freq if rec_freq not in (None, "") else st.get(
        "rec_freq", ISO_BIN_DEFAULTS["rec_freq"])
    st["n_bins"] = n_bins if n_bins not in (None, "") else st.get(
        "n_bins", ISO_BIN_DEFAULTS["n_bins"])
    if trig.startswith("iso-set-") and not err:
        save_iso_fit_settings(st)
    floor_on = bool(st.get("auto_floor") or st.get("bin_err"))   # total-error column
    try:
        vals, res, why = iso_fit_data(st)
    except Exception as exc:
        traceback.print_exc()
        return (empty_fig(str(exc)), empty_fig(), html.Span(str(exc), style={"color": "#c0392b"}),
                *_iso_curve_table(None, floor_on))
    if vals is None:
        return empty_fig("No points yet"), empty_fig(), "", *_iso_curve_table(None, floor_on)
    # error bars: the combined error (statistical ⊕ bin width ⊕ floor) used in the fit
    ecol = "d_sigma_T_total_ps" if "d_sigma_T_total_ps" in vals else "d_sigma_T_ps"

    fig = go.Figure()
    used = vals[vals["include"] & vals["fittable"]]
    out = vals[~(vals["include"] & vals["fittable"])]
    if res:
        x = _iso_curve_x(vals)
        fig.add_trace(go.Scatter(x=x, y=iso_sigma_t(x, *res["popt"], st["circumference_m"],
                                                    st["c_m_s"]),
                                 mode="lines", name="Isochronicity curve", meta="iso-curve",
                                 line=dict(color=ACCENT, width=2), hoverinfo="skip"))
    # labels: drawn in the browser as annotations with arrows (ISO_LABEL_JS)
    fig.add_trace(go.Scatter(
        x=used["T_ns"], y=used["sigma_T_ps"], mode="markers", name="In fit", meta="iso-pts",
        text=used["ion"], customdata=used["id"],
        marker=dict(symbol="x", size=9, color="black"),
        error_y=dict(type="data", array=used[ecol], color="#999", thickness=1, width=4),
        hovertemplate="%{text}<br>T = %{x:.4f} ns<br>σ(T) = %{y:.4f} ps"
                      "<br>click: leave out<extra></extra>"))
    if len(out):
        fig.add_trace(go.Scatter(
            x=out["T_ns"], y=out["sigma_T_ps"], mode="markers", name="Left out", meta="iso-out",
            text=out["ion"], customdata=out["id"],
            marker=dict(symbol="circle-open", size=9, color="#aaa"),
            error_y=dict(type="data", array=out["d_sigma_T_ps"].fillna(0), color="#ddd",
                         thickness=1, width=4),
            hovertemplate="%{text} (left out)<br>T = %{x:.4f} ns<br>σ(T) = %{y:.4f} ps"
                          "<br>click: put back<extra></extra>"))
    if res:
        fig.add_annotation(xref="paper", yref="paper", x=0.01, y=0.99, xanchor="left",
                           yanchor="top", align="left", showarrow=False,
                           text="<br>".join(_iso_param_lines(res, extras=False)),
                           bgcolor=ACCENT_PALE, bordercolor=ACCENT, borderwidth=1,
                           font=dict(size=13), name="isoparam")
    fig.update_layout(template="plotly_white", margin=dict(l=70, r=20, t=30, b=50),
                      xaxis_title="Revolution Time T (ns)", yaxis_title="σ_T (ps)",
                      legend=dict(x=0.99, y=0.02, xanchor="right", yanchor="bottom",
                                  bgcolor="rgba(255,255,255,0.85)", bordercolor="#dddddd",
                                  borderwidth=1),
                      font=dict(size=13), uirevision="iso", clickmode="event")

    # residuals in units of each point's error, as in the decay-fit panel:
    # grey ±1σ band, red zero line, black dots, 'within ±1σ' count
    # styled like the main plot: × markers, zero line in the curve's colour,
    # left-out points as grey circles, boxed ion labels (ISO_LABEL_JS)
    rfig = go.Figure()
    if res:
        pull, pull_txt = _iso_pulls(used, res, st, ecol)
        rfig.add_hrect(y0=-1, y1=1, fillcolor="#d9d9d9", opacity=1, line_width=0, layer="below")
        xr = _iso_curve_x(vals)
        rfig.add_trace(go.Scatter(x=[xr[0], xr[-1]], y=[0, 0], mode="lines", meta="iso-curve",
                                  line=dict(color=ACCENT, width=2), hoverinfo="skip"))
        rfig.add_trace(go.Scatter(
            x=used["T_ns"], y=pull, mode="markers", text=used["ion"], customdata=used["id"],
            meta="iso-pts", marker=dict(symbol="x", size=9, color="black"),
            hovertemplate="%{text}<br>T = %{x:.4f} ns<br>(σ(T) − fit)/err = %{y:.2f}"
                          "<br>click: leave out<extra></extra>"))
        if len(out):
            opull = _iso_pull_values(out, res, st, ecol)
            ok = np.isfinite(opull)
            if ok.any():
                o2 = out[ok]
                rfig.add_trace(go.Scatter(
                    x=o2["T_ns"], y=opull[ok], mode="markers", text=o2["ion"],
                    customdata=o2["id"], meta="iso-out",
                    marker=dict(symbol="circle-open", size=9, color="#aaa"),
                    hovertemplate="%{text} (left out)<br>T = %{x:.4f} ns<br>"
                                  "(σ(T) − fit)/err = %{y:.2f}<br>click: put back<extra></extra>"))
        rfig.add_annotation(xref="paper", yref="paper", x=0.995, y=0.97, xanchor="right",
                            yanchor="top", showarrow=False, text=pull_txt,
                            bgcolor="rgba(255,255,255,0.85)", bordercolor="#b3b3b3",
                            borderwidth=1, borderpad=4, font=dict(size=13))
    rfig.update_layout(template="plotly_white", margin=dict(l=70, r=20, t=30, b=50),
                       title=dict(text="Residuals", font=dict(size=13)),
                       xaxis_title="Revolution Time T (ns)", yaxis_title="(σ(T) − fit)/σ",
                       showlegend=False, font=dict(size=13), uirevision="iso",
                       clickmode="event")

    parts = []
    if res:
        parts.append(html.Div([html.B("Fit: "), "   ·   ".join(_iso_param_lines(res))]))
    if why:
        parts.append(html.Div(f"No fit: {why}", style={"color": "#888"}))
    if err:
        parts.append(html.Div(err, style={"color": "#c0392b"}))
    return fig, rfig, parts, *_iso_curve_table(vals, floor_on)


# Point labels of the isochronicity plot: placed in the browser with the same
# method as place_labels(), and placed again after every zoom / pan / resize /
# legend click, so they never overlap.  Drawn as annotations with arrows.
ISO_LABEL_JS = """
function(fig) {
    var ANG = [45, 135, -45, -135, 90, -90, 0, 180,
               67.5, 112.5, -67.5, -112.5, 22.5, 157.5, -22.5, -157.5,
               33.75, 56.25, 123.75, 146.25, -33.75, -56.25, -123.75, -146.25];
    var RAD = [24, 34, 46, 60, 76, 94, 115, 140, 170, 210, 260, 320];
    var FONT_PX = 12;
    function ov(a, b) {
        var w = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
        var h = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
        return (w > 0 && h > 0) ? w * h : 0;
    }
    // 2-px steps of the arrow (from 8 px off the point to the label box) on an error bar
    function leaderInBars(px, py, cx, cy, box, bars) {
        var d = Math.hypot(cx - px, cy - py);
        if (d <= 8 || !bars.length) return 0;
        var ux = (cx - px) / d, uy = (cy - py) / d, hits = 0;
        for (var s = 8; s < d; s += 2) {
            var x = px + s * ux, y = py + s * uy;
            if (x >= box[0] && x <= box[2] && y >= box[1] && y <= box[3]) break;
            for (var q = 0; q < bars.length; q++) {
                var b = bars[q];
                if (x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3]) { hits++; break; }
            }
        }
        return hits;
    }
    // visible arrow: 5 px off the point to where it enters the label box
    function leader(px, py, cx, cy, w, h) {
        var dx = cx - px, dy = cy - py, hw = w / 2 + 2, hh = h / 2 + 1;
        if (Math.abs(dx) <= hw && Math.abs(dy) <= hh) return null;
        var s = Math.min(dx ? hw / Math.abs(dx) : Infinity, dy ? hh / Math.abs(dy) : Infinity);
        var t0 = Math.min(5 / Math.hypot(dx, dy), 1 - s);
        return [px + t0 * dx, py + t0 * dy, px + (1 - s) * dx, py + (1 - s) * dy];
    }
    // point → label centre (arrow-crossing test; conservative, as the drawn
    // arrow stops at the box edge)
    function toCentre(px, py, cx, cy) {
        var d = Math.hypot(cx - px, cy - py);
        if (d <= 5) return null;
        return [px + 5 * (cx - px) / d, py + 5 * (cy - py) / d, cx, cy];
    }
    function segRect(s, r) {          // segment through rectangle (Liang–Barsky)
        if (!s) return false;
        var dx = s[2] - s[0], dy = s[3] - s[1], t0 = 0, t1 = 1;
        var P = [-dx, dx, -dy, dy], Q = [s[0] - r[0], r[2] - s[0], s[1] - r[1], r[3] - s[1]];
        for (var k = 0; k < 4; k++) {
            if (P[k] === 0) { if (Q[k] < 0) return false; }
            else {
                var t = Q[k] / P[k];
                if (P[k] < 0) t0 = Math.max(t0, t); else t1 = Math.min(t1, t);
                if (t0 > t1) return false;
            }
        }
        return true;
    }
    function segsCross(a, b) {
        if (!a || !b) return false;
        function o(ax, ay, bx, by, cx, cy) { return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax); }
        var d1 = o(b[0], b[1], b[2], b[3], a[0], a[1]), d2 = o(b[0], b[1], b[2], b[3], a[2], a[3]);
        var d3 = o(a[0], a[1], a[2], a[3], b[0], b[1]), d4 = o(a[0], a[1], a[2], a[3], b[2], b[3]);
        return d1 * d2 < 0 && d3 * d4 < 0;
    }
    function nearSeg(qx, qy, s) {
        if (!s) return false;
        var vx = s[2] - s[0], vy = s[3] - s[1], L2 = vx * vx + vy * vy;
        var t = L2 === 0 ? 0 : Math.min(Math.max(((qx - s[0]) * vx + (qy - s[1]) * vy) / L2, 0), 1);
        return Math.hypot(qx - (s[0] + t * vx), qy - (s[1] + t * vy)) < 7;
    }
    // same method as place_labels() in Python: candidate spots per label,
    // own cost + pair costs, simulated annealing with a fixed seed, polish
    function place(pts, sizes, obst, rect, bars) {
        var n = pts.length;
        if (!n) return [];
        var cands = [];
        for (var i = 0; i < n; i++) {
            var px = pts[i][0], py = pts[i][1], w = sizes[i][0], h = sizes[i][1], lst = [];
            RAD.forEach(function(r) {
                ANG.forEach(function(a) {
                    var cx = px + r * Math.cos(a * Math.PI / 180), cy = py - r * Math.sin(a * Math.PI / 180);
                    var box = [cx - w / 2 - 2, cy - h / 2 - 1, cx + w / 2 + 2, cy + h / 2 + 1];
                    var area = (box[2] - box[0]) * (box[3] - box[1]);
                    var u = 0.6 * r + 60 * (area - ov(box, rect));
                    for (var o = 0; o < obst.length; o++)
                        u += (obst[o].length > 4 ? obst[o][4] : 4) * ov(box, obst[o]);
                    u += 150 * leaderInBars(px, py, cx, cy, box, bars);
                    var lead = leader(px, py, cx, cy, w, h);
                    for (var k = 0; k < n; k++)   // (points under this point's own marker are unavoidable)
                        if (k !== i && Math.hypot(pts[k][0] - px, pts[k][1] - py) > 8 &&
                            nearSeg(pts[k][0], pts[k][1], lead)) u += 300;
                    var env = [Math.min(box[0], px), Math.min(box[1], py), Math.max(box[2], px), Math.max(box[3], py)];
                    lst.push([u, [cx, cy], box, lead, toCentre(px, py, cx, cy), env]);
                });
            });
            lst.sort(function(p, q) { return p[0] - q[0]; });
            cands.push(lst.slice(0, 120));
        }
        var cache = new Map();
        function pair(i, a, j, b) {
            var key = i < j ? ((i * 1000 + a) * 100000 + j * 1000 + b) : ((j * 1000 + b) * 100000 + i * 1000 + a);
            var v = cache.get(key);
            if (v !== undefined) return v;
            var A = cands[i][a], B = cands[j][b], ea = A[5], eb = B[5];
            v = 0;
            if (!(ea[2] < eb[0] || eb[2] < ea[0] || ea[3] < eb[1] || eb[3] < ea[1])) {
                v += 20 * ov(A[2], B[2]);
                if (segRect(A[3], B[2]) || segRect(B[3], A[2])) v += 6000;
                if (segsCross(A[4], B[4])) v += 6000;
            }
            cache.set(key, v);
            return v;
        }
        var env = cands.map(function(L) {
            var e = [Infinity, Infinity, -Infinity, -Infinity];
            L.forEach(function(c) { e[0] = Math.min(e[0], c[5][0]); e[1] = Math.min(e[1], c[5][1]);
                                    e[2] = Math.max(e[2], c[5][2]); e[3] = Math.max(e[3], c[5][3]); });
            return e;
        });
        var nb = [];
        for (var i = 0; i < n; i++) {
            nb.push([]);
            for (var j = 0; j < n; j++)
                if (j !== i && !(env[i][2] < env[j][0] || env[j][2] < env[i][0] ||
                                 env[i][3] < env[j][1] || env[j][3] < env[i][1])) nb[i].push(j);
        }
        var cur = new Array(n).fill(0);
        function local(i, a) {
            var c = cands[i][a][0];
            for (var q = 0; q < nb[i].length; q++) c += pair(i, a, nb[i][q], cur[nb[i][q]]);
            return c;
        }
        function total() {
            var t = 0;
            for (var i = 0; i < n; i++) {
                t += cands[i][cur[i]][0];
                for (var q = 0; q < nb[i].length; q++) t += 0.5 * pair(i, cur[i], nb[i][q], cur[nb[i][q]]);
            }
            return t;
        }
        var seed;                             // fixed seeds: same view, same layout
        function rnd() {                      // mulberry32
            seed |= 0; seed = seed + 0x6D2B79F5 | 0;
            var t = Math.imul(seed ^ seed >>> 15, 1 | seed);
            t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
            return ((t ^ t >>> 14) >>> 0) / 4294967296;
        }
        // several annealing runs, keep the layout with the lowest cost
        var iters = Math.min(25000, 700 * n), T0 = 3000, T1 = 5, bestLayout = null, bestCost = Infinity;
        for (var run = 0; run < 4; run++) {
            seed = 12345 + run;
            cur = new Array(n).fill(0);
            var tot = 0, bestTot = 0, best = cur.slice();
            for (var it = 0; it < iters; it++) {
                var T = T0 * Math.pow(T1 / T0, it / iters);
                var i = Math.floor(rnd() * n), a = Math.floor(rnd() * cands[i].length);
                if (a === cur[i]) continue;
                var d = local(i, a) - local(i, cur[i]);
                if (d < 0 || rnd() < Math.exp(-d / T)) {
                    cur[i] = a; tot += d;
                    if (tot < bestTot - 1e-9) { bestTot = tot; best = cur.slice(); }
                }
            }
            cur = best;
            for (var r5 = 0; r5 < 5; r5++) {  // polish: best spot for each label
                var changed = false;
                for (var i = 0; i < n; i++) {
                    var bi = cur[i], bc = local(i, cur[i]);
                    for (var a = 0; a < cands[i].length; a++) { var c = local(i, a); if (c < bc - 1e-9) { bc = c; bi = a; } }
                    if (bi !== cur[i]) { cur[i] = bi; changed = true; }
                }
                if (!changed) break;
            }
            var tc = total();
            if (tc < bestCost) { bestCost = tc; bestLayout = cur.slice(); }
        }
        cur = bestLayout;
        return cur.map(function(a, i) { return [cands[i][a][1][0] - pts[i][0], cands[i][a][1][1] - pts[i][1]]; });
    }
    var ctx2d = document.createElement('canvas').getContext('2d');
    function textW(t, font) { ctx2d.font = font; return ctx2d.measureText(t).width; }

    function layoutLabels(gd) {
        var L = gd._fullLayout;
        if (!L || !L.xaxis || !L.yaxis || !gd.data) return;
        var xa = L.xaxis, ya = L.yaxis;
        var rect = [xa._offset, ya._offset, xa._offset + xa._length, ya._offset + ya._length];
        var gr = gd.getBoundingClientRect();
        var font = FONT_PX + 'px "Open Sans", verdana, arial, sans-serif';
        var pts = [], sizes = [], obst = [], lab = [], bars = [];
        gd.data.forEach(function(tr, ti) {
            var full = gd._fullData[ti];
            if (!full || full.visible !== true) return;
            if (tr.meta === 'iso-curve') {
                var step = Math.max(1, Math.floor(full.x.length / 150));
                for (var k = 0; k < full.x.length; k += step) {
                    var cx = xa.l2p(full.x[k]) + xa._offset, cy = ya.l2p(full.y[k]) + ya._offset;
                    obst.push([cx - 3, cy - 3, cx + 3, cy + 3]);
                }
                return;
            }
            if (tr.meta !== 'iso-pts' && tr.meta !== 'iso-out') return;
            var err = (full.error_y && full.error_y.array) || [];
            for (var k = 0; k < full.x.length; k++) {
                var X = full.x[k], Y = full.y[k];
                var px = xa.l2p(X) + xa._offset, py = ya.l2p(Y) + ya._offset;
                if (!isFinite(px) || !isFinite(py)) continue;
                obst.push([px - 7, py - 7, px + 7, py + 7]);
                var e = err[k];
                if (isFinite(e) && e > 0) {
                    var y1 = ya.l2p(Y + e) + ya._offset, y2 = ya.l2p(Y - e) + ya._offset;
                    obst.push([px - 4, Math.min(y1, y2), px + 4, Math.max(y1, y2)]);
                    bars.push([px - 3, Math.min(y1, y2), px + 3, Math.max(y1, y2)]);
                }
                if (px < rect[0] || px > rect[2] || py < rect[1] || py > rect[3]) continue;
                var t = String(full.text[k]);
                pts.push([px, py]);
                // text + white box with a black border (borderpad 3 + border 1 each side)
                sizes.push([textW(t, font) + 2 + 8, Math.round(FONT_PX * 1.35) + 8]);
                lab.push({x: X, y: Y, text: t, out: tr.meta === 'iso-out'});
            }
        });
        // never under the parameter box or the legend
        gd.querySelectorAll('.annotation, .legend').forEach(function(el) {
            if (el.classList.contains('annotation') && el.getAttribute('data-index') !== null) {
                var a = (gd.layout.annotations || [])[+el.getAttribute('data-index')];
                if (a && a.name === 'isolab') return;
            }
            var r = el.getBoundingClientRect();
            if (r.width && r.height)
                obst.push([r.left - gr.left, r.top - gr.top, r.right - gr.left, r.bottom - gr.top, 20]);
        });
        var offs = place(pts, sizes, obst, rect, bars);
        var keep = (gd.layout.annotations || []).filter(function(a) { return a.name !== 'isolab'; });
        var anns = keep.concat(lab.map(function(l, i) {
            return {x: l.x, y: l.y, xref: 'x', yref: 'y', text: l.text, name: 'isolab',
                    showarrow: true, ax: offs[i][0], ay: offs[i][1], arrowhead: 2,
                    arrowsize: 0.8, arrowwidth: 1, arrowcolor: '#888888', standoff: 5,
                    font: {size: FONT_PX, color: l.out ? '#aaaaaa' : '#222222'},
                    bgcolor: '#ffffff', bordercolor: '#000000', borderwidth: 1, borderpad: 3,
                    captureevents: false};
        }));
        gd._isoLabelBusy = true;
        Plotly.relayout(gd, {annotations: anns}).then(function() { gd._isoLabelBusy = false; },
                                                      function() { gd._isoLabelBusy = false; });
    }

    setTimeout(function() {
        var gd = document.querySelector('#__GRAPH__ .js-plotly-plot');
        if (!gd) return;
        if (!gd._isoLabelHook) {
            gd._isoLabelHook = true;
            // Wait until the view is still: a scroll-wheel zoom is a preview
            // that Plotly commits later, and redrawing the labels in between
            // throws the preview away (the plot pulses in and out).  So
            // re-place only 0.4 s after the last wheel step / view change,
            // and never while a mouse button is held (drag-zoom / pan).
            var timer = null, lastWheel = 0, mouseDown = false, QUIET = 400;
            var soon = function() {
                clearTimeout(timer);
                timer = setTimeout(function run() {
                    if (mouseDown || Date.now() - lastWheel < QUIET) {
                        timer = setTimeout(run, QUIET);
                        return;
                    }
                    layoutLabels(gd);
                }, QUIET);
            };
            gd.addEventListener('wheel', function() { lastWheel = Date.now(); soon(); },
                                {capture: true, passive: true});
            gd.addEventListener('mousedown', function() { mouseDown = true; }, true);
            window.addEventListener('mouseup', function() { if (mouseDown) { mouseDown = false; soon(); } });
            gd.on('plotly_relayout', function(ev) {
                if (gd._isoLabelBusy) return;
                var keys = Object.keys(ev || {});
                if (keys.length && keys.every(function(k) { return k.indexOf('annotations') === 0; })) return;
                soon();
            });
            gd.on('plotly_restyle', soon);            // legend clicks (hide / show)
            window.addEventListener('resize', soon);
            gd._isoSoon = soon;
        }
        gd._isoSoon();
    }, 150);
    return window.dash_clientside.no_update;
}
"""

app.clientside_callback(
    ISO_LABEL_JS.replace("__GRAPH__", "iso-graph"),
    Output("_iso-label-hook", "children"),
    Input("iso-graph", "figure"),
    prevent_initial_call=True,
)
# the same labels on the residual plot
app.clientside_callback(
    ISO_LABEL_JS.replace("__GRAPH__", "iso-resid"),
    Output("_iso-resid-label-hook", "children"),
    Input("iso-resid", "figure"),
    prevent_initial_call=True,
)


@app.callback(
    Output("iso-version", "data", allow_duplicate=True),
    Output("iso-curve-table", "active_cell"),
    Output("iso-curve-table", "selected_cells"),
    Input("iso-curve-table", "active_cell"),
    Input("iso-graph", "clickData"),
    Input("iso-resid", "clickData"),
    State("iso-version", "data"),
    prevent_initial_call=True,
)
def toggle_iso_point(active, click, rclick, version):
    """'in fit' cell or a click on a point: leave the point out / put it back."""
    trig = dash.callback_context.triggered[0]["prop_id"] if dash.callback_context.triggered else ""
    pid = None
    if trig.startswith("iso-curve-table"):
        if not active or active.get("column_id") != "include":
            raise PreventUpdate
        pid = active.get("row_id")
    elif trig.startswith("iso-graph") and click and click.get("points"):
        pid = click["points"][0].get("customdata")
    elif trig.startswith("iso-resid") and rclick and rclick.get("points"):
        pid = rclick["points"][0].get("customdata")
    if not pid:
        raise PreventUpdate
    ion, h = pid.rsplit("|", 1)
    df = load_iso_points()
    keys = [_iso_point_id(i, hh) for i, hh in zip(df["ion"].astype(str), df["harmonic"])]
    if pid not in keys:
        raise PreventUpdate
    inc = _iso_included(df)
    vals = iso_curve_values(df)
    k = keys.index(pid)
    if not _iso_fittable(vals)[k]:
        raise PreventUpdate                       # no error bar: cannot be fitted anyway
    inc[k] = not inc[k]
    df["include"] = inc
    save_iso_points(df)
    return (version or 0) + 1, None, []


@app.callback(
    Output("iso-bin-info", "children"),
    Input("iso-set-recfreq", "value"),
    Input("iso-set-nbins", "value"),
)
def show_bin_width(rec_freq, n_bins):
    try:
        w = iso_bin_width({"rec_freq": rec_freq or "", "n_bins": n_bins or ""})
    except ValueError as exc:
        return html.Span(f"✗ {exc}", style={"color": "#c0392b", "fontFamily": "inherit"})
    return f"bin = {w:.3f} Hz  →  1σ = bin/√12 = {w / np.sqrt(12.0):.2f} Hz"


@app.callback(
    Output("iso-save-status", "children"),
    Input("btn-iso-save-fig", "n_clicks"),
    *[State(f"iso-set-{k}", "value") for k in ISO_SETTING_KEYS],
    State("iso-set-floor", "value"),
    State("iso-set-binerr", "value"),
    State("iso-set-recfreq", "value"),
    State("iso-set-nbins", "value"),
    prevent_initial_call=True,
)
def save_iso_figure(_n, *args):
    setting_vals, (floor_val, bin_val, rec_freq, n_bins) = args[:-4], args[-4:]
    try:
        st = {k: float(v) for k, v in zip(ISO_SETTING_KEYS, setting_vals)}
        check_iso_fit_settings(st)
    except (TypeError, ValueError):
        st = load_iso_fit_settings()
    st["auto_floor"] = "auto" in (floor_val or [])
    st["bin_err"] = "on" in (bin_val or [])
    st["rec_freq"] = rec_freq or ISO_BIN_DEFAULTS["rec_freq"]
    st["n_bins"] = n_bins or ISO_BIN_DEFAULTS["n_bins"]
    try:
        vals, res, why = iso_fit_data(st)
        if vals is None:
            raise ValueError("no points yet")
        files = draw_iso_figure(vals, res, st)
    except Exception as exc:
        traceback.print_exc()
        return html.Span(f"Could not save: {exc}", style={"color": "#c0392b"})
    return html.Span([html.B("Saved: ", style={"color": "#2e7d32"}),
                      "  ".join(os.path.basename(f) for f in files),
                      f"  (in {os.path.dirname(files[0])})"])


# ---- point labels without overlaps (isochronicity plot) ---------------------
# Greedy placement in pixel space: each label tries 16 directions × 6
# distances around its point and takes the lowest-cost spot.  Cost: overlap
# with labels already placed (heavy), with obstacles (points, error bars, the
# curve, parameter box, legend), area outside the plot, the leader line
# crossing a placed label, and the distance.  Crowded points go first.
# The same algorithm runs in the browser (ISO_LABEL_JS) after every zoom/pan.
LABEL_ANGLES = [45, 135, -45, -135, 90, -90, 0, 180,
                67.5, 112.5, -67.5, -112.5, 22.5, 157.5, -22.5, -157.5,
                33.75, 56.25, 123.75, 146.25, -33.75, -56.25, -123.75, -146.25]
LABEL_RADII = [24, 34, 46, 60, 76, 94, 115, 140, 170, 210, 260, 320]
LABEL_KEEP = 120             # cheapest candidate spots kept per label
LABEL_ANNEAL_PER_LABEL = 700    # annealing steps per label …
LABEL_ANNEAL_MAX = 25000        # … at most, per run
LABEL_RESTARTS = 4              # annealing runs (different fixed seeds); the cheapest is kept
ISO_LABEL_PT = 12          # label text size: saved figure (pt) and page (px)
ISO_LABEL_BOX_PAD = 0.2    # label box padding in the saved figure (× font size)
ISO_ARROW_BAR_COST = 150   # per 2 px of arrow lying on an error bar (same in ISO_LABEL_JS)
ISO_ARROW_BOX_COST = 6000   # an arrow through a label box, either way (same in ISO_LABEL_JS)
ISO_ARROW_CROSS_COST = 6000  # arrow crossing another label's arrow (same in ISO_LABEL_JS)
ISO_ARROW_POINT_COST = 300  # arrow passing over another point (same in ISO_LABEL_JS)


def _overlap(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def _leader_in_bars(px, py, cx, cy, box, bars):
    """Number of 2-px steps of the arrow (point → label, from 8 px off the
    point to where it enters the label box) that lie on an error bar."""
    d = float(np.hypot(cx - px, cy - py))
    if d <= 8 or not bars:
        return 0
    ux, uy = (cx - px) / d, (cy - py) / d
    hits, s = 0, 8.0
    while s < d:
        x, y = px + s * ux, py + s * uy
        if box[0] <= x <= box[2] and box[1] <= y <= box[3]:
            break                                  # arrow ends at the label box
        for b in bars:
            if b[0] <= x <= b[2] and b[1] <= y <= b[3]:
                hits += 1
                break
        s += 2.0
    return hits


def _leader(px, py, cx, cy, w, h):
    """Visible arrow of a label: from 5 px off the point to where the line
    point → label centre enters the label box.  None if the point is inside."""
    dx, dy = cx - px, cy - py
    hw, hh = w / 2 + 2, h / 2 + 1
    if abs(dx) <= hw and abs(dy) <= hh:
        return None
    s = min(hw / abs(dx) if dx else np.inf, hh / abs(dy) if dy else np.inf)
    d = float(np.hypot(dx, dy))
    t0 = min(5.0 / d, 1.0 - s)
    return (px + t0 * dx, py + t0 * dy, px + (1 - s) * dx, py + (1 - s) * dy)


def _to_centre(px, py, cx, cy):
    """Line from 5 px off the point to the label centre (used for the
    arrow-crossing test: conservative, as the drawn arrow stops at the box
    edge, whose exact size is only known once rendered)."""
    d = float(np.hypot(cx - px, cy - py))
    if d <= 5:
        return None
    return (px + 5 * (cx - px) / d, py + 5 * (cy - py) / d, cx, cy)


def _seg_rect(seg, r):
    """Does segment seg (x0, y0, x1, y1) pass through rectangle r (Liang–Barsky)?"""
    if seg is None:
        return False
    x0, y0, x1, y1 = seg
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - r[0]), (dx, r[2] - x0), (-dy, y0 - r[1]), (dy, r[3] - y0)):
        if p == 0:
            if q < 0:
                return False
        else:
            t = q / p
            if p < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                return False
    return True


def _segs_cross(a, b):
    """Do segments a and b (x0, y0, x1, y1) cross?"""
    if a is None or b is None:
        return False

    def orient(ax, ay, bx, by, cx, cy):
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    d1 = orient(b[0], b[1], b[2], b[3], a[0], a[1])
    d2 = orient(b[0], b[1], b[2], b[3], a[2], a[3])
    d3 = orient(a[0], a[1], a[2], a[3], b[0], b[1])
    d4 = orient(a[0], a[1], a[2], a[3], b[2], b[3])
    return d1 * d2 < 0 and d3 * d4 < 0


def _near_seg(qx, qy, seg, tol=7.0):
    """Is point q within tol px of segment seg?"""
    if seg is None:
        return False
    x0, y0, x1, y1 = seg
    vx, vy = x1 - x0, y1 - y0
    L2 = vx * vx + vy * vy
    t = 0.0 if L2 == 0 else min(max(((qx - x0) * vx + (qy - y0) * vy) / L2, 0.0), 1.0)
    return float(np.hypot(qx - (x0 + t * vx), qy - (y0 + t * vy))) < tol


def place_labels(pts, sizes, obstacles, rect, y_down=True, bars=None, seed=0):
    """Pixel offsets (dx, dy) of each label centre from its point.
    pts: [(x, y)], sizes: [(w, h)] (label box incl. padding), obstacles:
    [(x0, y0, x1, y1[, weight])] (default weight 4), rect: plot area,
    y_down: screen y grows downwards, bars: error-bar rectangles.

    Each label gets candidate spots (LABEL_ANGLES × LABEL_RADII around its
    point).  A spot's own cost: distance, area outside the plot, obstacles,
    its arrow on error bars, its arrow over other points.  Pair costs between
    two labels: box overlap, an arrow through the other's box (either way),
    crossing arrows.  The layout is found by simulated annealing (fixed seed,
    so the same view always gives the same layout) and then polished.  The
    same algorithm runs in the browser (ISO_LABEL_JS)."""
    n = len(pts)
    if n == 0:
        return []
    bars = bars or []
    up = -1.0 if y_down else 1.0
    cands = []
    for i in range(n):
        px, py = pts[i]
        w, h = sizes[i]
        lst = []
        for r in LABEL_RADII:
            for a in LABEL_ANGLES:
                cx = px + r * np.cos(np.radians(a))
                cy = py + up * r * np.sin(np.radians(a))
                box = (cx - w / 2 - 2, cy - h / 2 - 1, cx + w / 2 + 2, cy + h / 2 + 1)
                area = (box[2] - box[0]) * (box[3] - box[1])
                u = 0.6 * r + 60.0 * (area - _overlap(box, rect))
                for o in obstacles:
                    u += (o[4] if len(o) > 4 else 4.0) * _overlap(box, o)
                u += ISO_ARROW_BAR_COST * _leader_in_bars(px, py, cx, cy, box, bars)
                lead = _leader(px, py, cx, cy, w, h)
                for k, (qx, qy) in enumerate(pts):
                    # (points under this point's own marker cannot be avoided)
                    if k != i and np.hypot(qx - px, qy - py) > 8 and _near_seg(qx, qy, lead):
                        u += ISO_ARROW_POINT_COST
                env = (min(box[0], px), min(box[1], py), max(box[2], px), max(box[3], py))
                lst.append((u, (cx, cy), box, lead, _to_centre(px, py, cx, cy), env))
        lst.sort(key=lambda c: c[0])
        cands.append(lst[:LABEL_KEEP])

    cache = {}

    def pair(i, a, j, b):
        key = (i, a, j, b) if i < j else (j, b, i, a)
        v = cache.get(key)
        if v is None:
            A, B = cands[i][a], cands[j][b]
            ea, eb = A[5], B[5]
            v = 0.0
            if not (ea[2] < eb[0] or eb[2] < ea[0] or ea[3] < eb[1] or eb[3] < ea[1]):
                v += 20.0 * _overlap(A[2], B[2])
                if _seg_rect(A[3], B[2]) or _seg_rect(B[3], A[2]):
                    v += ISO_ARROW_BOX_COST
                if _segs_cross(A[4], B[4]):
                    v += ISO_ARROW_CROSS_COST
            cache[key] = v
        return v

    # labels that can interact at all (their candidate areas touch)
    env = [(min(c[5][0] for c in L), min(c[5][1] for c in L),
            max(c[5][2] for c in L), max(c[5][3] for c in L)) for L in cands]
    nb = [[j for j in range(n) if j != i and not (env[i][2] < env[j][0] or env[j][2] < env[i][0]
                                                   or env[i][3] < env[j][1] or env[j][3] < env[i][1])]
          for i in range(n)]
    cur = [0] * n

    def local(i, a):
        return cands[i][a][0] + sum(pair(i, a, j, cur[j]) for j in nb[i])

    def total():
        return (sum(cands[i][cur[i]][0] for i in range(n))
                + 0.5 * sum(pair(i, cur[i], j, cur[j]) for i in range(n) for j in nb[i]))

    # several annealing runs (fixed seeds), keep the layout with the lowest cost
    iters = min(LABEL_ANNEAL_MAX, LABEL_ANNEAL_PER_LABEL * n)
    T0, T1 = 3000.0, 5.0
    best_layout, best_cost = None, None
    for run in range(LABEL_RESTARTS):
        rng = np.random.default_rng(seed + run)
        cur[:] = [0] * n                            # start: each label's cheapest own spot
        tot = best_tot = 0.0
        best = list(cur)
        for it in range(iters):
            T = T0 * (T1 / T0) ** (it / iters)
            i = int(rng.integers(n))
            a = int(rng.integers(len(cands[i])))
            if a == cur[i]:
                continue
            d = local(i, a) - local(i, cur[i])
            if d < 0 or rng.random() < np.exp(-d / T):
                cur[i] = a
                tot += d
                if tot < best_tot - 1e-9:
                    best_tot, best = tot, list(cur)
        cur[:] = best
        for _ in range(5):                          # polish: best spot for each label
            changed = False
            for i in range(n):
                b = min(range(len(cands[i])), key=lambda a: local(i, a))
                if local(i, b) < local(i, cur[i]) - 1e-9:
                    cur[i], changed = b, True
            if not changed:
                break
        c = total()
        if best_cost is None or c < best_cost:
            best_cost, best_layout = c, list(cur)
    cur[:] = best_layout
    return [(cands[i][cur[i]][1][0] - pts[i][0], cands[i][cur[i]][1][1] - pts[i][1])
            for i in range(n)]


def _mpl_point_labels(fig, ax, used, out, boxes, ycol="sigma_T_ps", errors=True):
    """Ion labels with arrows, placed by place_labels (display pixels, y up).
    ycol: column with the plotted y values; errors: avoid the error bars."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    ecol = "d_sigma_T_total_ps" if "d_sigma_T_total_ps" in used else "d_sigma_T_ps"
    rows = [(r["ion"], r["T_ns"], r[ycol], r[ecol] if errors else np.nan, "black")
            for r in used.to_dict("records")] + \
           [(r["ion"], r["T_ns"], r[ycol], r["d_sigma_T_ps"] if errors else np.nan, "#999999")
            for r in out.to_dict("records")]
    if not rows:
        return
    bb = ax.get_window_extent(rend)
    rect = (bb.x0, bb.y0, bb.x1, bb.y1)
    pts, sizes, obst, bars = [], [], [], []
    for ion, t, sg, e, _ in rows:
        x, y = ax.transData.transform((t, sg))
        pts.append((x, y))
        tt = ax.text(0, 0, ion, fontsize=ISO_LABEL_PT)
        ext = tt.get_window_extent(rend)
        tt.remove()
        pad = 2 * (ISO_LABEL_BOX_PAD * ISO_LABEL_PT * fig.dpi / 72.0 + 1)   # box around it
        sizes.append((ext.width + pad, ext.height + pad))
        obst.append((x - 7, y - 7, x + 7, y + 7))
        if np.isfinite(e) and e > 0:
            _, ylo = ax.transData.transform((t, sg - e))
            _, yhi = ax.transData.transform((t, sg + e))
            obst.append((x - 4, ylo, x + 4, yhi))
            bars.append((x - 3, min(ylo, yhi), x + 3, max(ylo, yhi)))
    for line in ax.get_lines():                      # the curve
        xy = ax.transData.transform(np.column_stack(line.get_data()))
        for k in range(0, len(xy), max(1, len(xy) // 150)):
            obst.append((xy[k, 0] - 3, xy[k, 1] - 3, xy[k, 0] + 3, xy[k, 1] + 3))
    for b in boxes:
        if b is not None:
            e = b.get_window_extent(rend)
            obst.append((e.x0, e.y0, e.x1, e.y1, 20.0))      # never under these
    offs = place_labels(pts, sizes, obst, rect, y_down=False, bars=bars)
    k = 72.0 / fig.dpi                                 # pixels -> points
    for (ion, t, sg, _, col), (dx, dy) in zip(rows, offs):
        ax.annotate(ion, xy=(t, sg), xytext=(dx * k, dy * k), textcoords="offset points",
                    ha="center", va="center", fontsize=ISO_LABEL_PT, color=col,
                    bbox=dict(boxstyle=f"square,pad={ISO_LABEL_BOX_PAD}", facecolor="white",
                              edgecolor="black", linewidth=0.8),
                    arrowprops=dict(arrowstyle="-|>", color="#888888", lw=0.8,
                                    shrinkA=1, shrinkB=4, mutation_scale=8))


def draw_iso_figure(vals, res, st):
    """Matplotlib figure (curve + residuals), saved as PNG and PDF in
    outputs/figures/.  Returns the file paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    used = vals[vals["include"] & vals["fittable"]]
    out = vals[~(vals["include"] & vals["fittable"])]
    ecol = "d_sigma_T_total_ps" if "d_sigma_T_total_ps" in vals else "d_sigma_T_ps"
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(16, 19), sharex=True,
                                  gridspec_kw={"height_ratios": [13, 6], "hspace": 0.06})
    ax.errorbar(used["T_ns"], used["sigma_T_ps"], yerr=used[ecol], fmt="none",
                color="grey", capsize=4, zorder=2)
    ax.scatter(used["T_ns"], used["sigma_T_ps"], color="red", marker="x", s=50,
               label="Identified ions", zorder=3)
    if len(out):
        ax.scatter(out["T_ns"], out["sigma_T_ps"], facecolors="none", edgecolors="#aaaaaa",
                   s=50, label="Left out of the fit", zorder=2)
    if res:
        x = _iso_curve_x(vals)
        ax.plot(x, iso_sigma_t(x, *res["popt"], st["circumference_m"], st["c_m_s"]),
                color="black", label="Isochronicity curve")
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ptext = ax.text(0.02, 0.97, "\n".join(_iso_param_lines(res, extras=False)),
                        transform=ax.transAxes,
                        fontsize=16, verticalalignment="top", bbox=props)
        # residuals in units of each point's error (decay-fit panel), styled
        # like the main plot: red ×, black zero line, grey left-out circles, labels
        pull, pull_txt = _iso_pulls(used, res, st, ecol)
        axr.axhspan(-1.0, 1.0, color="0.85", zorder=0)
        xr = _iso_curve_x(vals)
        axr.plot([xr[0], xr[-1]], [0, 0], color="black")
        axr.scatter(used["T_ns"], pull, color="red", marker="x", s=50, zorder=3)
        r_used = used.assign(pull=pull)
        r_out = out.iloc[0:0].assign(pull=[])
        if len(out):
            opull = _iso_pull_values(out, res, st, ecol)
            ok = np.isfinite(opull)
            r_out = out[ok].assign(pull=opull[ok])
            axr.scatter(r_out["T_ns"], r_out["pull"], facecolors="none", edgecolors="#aaaaaa",
                        s=50, zorder=2)
        rtext = axr.text(0.995, 0.95, pull_txt, transform=axr.transAxes, ha="right", va="top",
                         fontsize=13, bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                                                edgecolor="0.7", alpha=0.85))
    else:
        ptext = None
    ax.set_ylabel(r"$\sigma_{T}$ (ps)", size=22)
    axr.set_ylabel(r"(σ(T) − fit)/$\sigma$", size=16)
    axr.set_xlabel("Revolution Time (ns)", size=22)
    ax.tick_params(labelsize=16)
    axr.tick_params(labelsize=16)
    leg = ax.legend(fontsize=16, loc="lower right")
    _mpl_point_labels(fig, ax, used, out, [ptext, leg])
    if res:
        _mpl_point_labels(fig, axr, r_used, r_out, [rtext], ycol="pull", errors=False)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(paths.outputs("figures"), f"isocurve_{stamp}")
    files = [base + ".png", base + ".pdf"]
    fig.savefig(files[0], dpi=150, bbox_inches="tight")
    fig.savefig(files[1], bbox_inches="tight")
    plt.close(fig)
    return files
