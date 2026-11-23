"""
Scan tab: the scan settings sidebar, running a scan and its progress.

Part of Theia (theia/theia.py); split out of theia.py in Nov 2026.
"""

import os
import threading
import time
import traceback

import numpy as np
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State

from core.physics import hl_cut_text
from core.scan import (analyse_spectrum, list_spectra, load_spectrum, noise_cutoff,
                       parse_settings, run_scan, save_config, SCAN_LOCK, SCAN_STATE, _set_state)

from .common import (ACCENT, BOX_STYLE, BTN, CFG, check, HELP_STYLE, hl_cut_inputs, inp,
                     LABEL_STYLE, radio, section, SETTING_FIELDS, _table)
from .app import app
from .figures import empty_fig, spectrum_figure
from .corr_profiles import corr_profile_ui
from .folder_browser import with_browse


sidebar = html.Div([
    html.H5("Scan Settings", style={"fontWeight": "700", "marginBottom": "4px"}),

    *section("Run"),
    inp("Run Label (optional)", "run_label", help="Free text stored with the database"),
    with_browse(inp("Spectra Directory", "spectra_dir"), "btn-browse-spectra_dir"),
    inp("File Pattern", "file_pattern", help="Files are processed in natural (numeric) order"),

    *section("Reference Ion"),
    inp("Ion Name", "ref_name"),
    inp("Frequency (Hz) — in First Spectrum", "ref_frequency", True, step=1),
    inp("Harmonic", "ref_harmonic", True, step=1),
    inp("Search Window ± (Hz)", "ref_window_hz", True),
    inp("Min Reference Height (a.u.)", "ref_min_height", True,
        help="Only peaks at least this tall (raw power) can be the reference. 0 = off"),
    radio("Pick in Window", "ref_mode", [
        {"label": "Strongest peak", "value": "strongest"},
        {"label": "Nearest to prediction", "value": "nearest"},
    ]),
    check("Track: centre next search on last found ref", "ref_tracking"),
    check("If not found, use predicted position", "use_predicted",
          help="Otherwise the spectrum is stored without identifications"),
    inp("Ref. Peak Search in the Other Harmonics ± (Hz)", "ref_search_hz", True,
        help="Each harmonic's lines are placed from the reference ion's own peak in that "
             "harmonic (nearest to f_ref/h_ref·h). 0 = off"),

    *section("Ion Selection"),
    inp("Z Limits  [min, max]", "z_lims"),
    inp("N Limits  [min, max]", "n_lims"),
    inp("el1  [A, Z]", "el1"),
    inp("el2  [A, Z]", "el2"),
    inp("m/q window (± value)", "mq_width", True),

    *section("Ring / Harmonics"),
    inp("γ_t", "gamma_t", True),
    inp("Harmonics (comma-separated)", "harmonics"),

    *section("Frequency Correction"),
    check("Second order correction", "correction_enabled"),
    corr_profile_ui("s-", CFG["corr_profile"], CFG["corr_a"], CFG["corr_b"], CFG["corr_c"],
                    ref_name=CFG["ref_name"], ref_h=CFG["ref_harmonic"], gamma_t=CFG["gamma_t"],
                    ref_f=CFG["ref_frequency"], harmonics=CFG["harmonics"]),
    inp("a  (f² coefficient)", "corr_a", True),
    inp("b  (f coefficient)", "corr_b", True),
    inp("c  (constant)", "corr_c", True),

    *section("Peak Detection (raw data)"),
    inp("Noise Block (Hz)", "noise_block_hz", True,
        help="Block size for the local median / MAD noise estimate. 0 = global"),
    inp("Noise Cutoff Factor X  [0 = off]", "noise_cut_factor", True,
        help="Bins above −min(spectrum)·X are excluded from the noise estimate"),
    inp("Prominence Threshold (σ)", "k_prom", True,
        help="Keep maxima with prominence ≥ k·σ"),
    inp("Height Threshold (σ)", "k_height", True,
        help="Keep peaks rising ≥ k·σ above the local baseline"),
    inp("Relative Noise r", "rel_noise", True,
        help="σ = √(σ_bg² + (r·height)²). Suppresses noise spikes on top of big peaks. 0 = off"),
    inp("Min Width FWHM (bins)  [0 = off]", "min_width_bins", True,
        help="Width at half prominence, in frequency bins (can be fractional)"),
    inp("Min Separation (Hz)  [0 = off]", "min_distance_hz", True),
    check("Detect shoulders (merged peaks)", "shoulders"),
    inp("Shoulder Scale s (Hz)", "shoulder_scale_hz", True,
        help="Step of the 2nd difference, roughly half the width of a peak"),
    inp("Shoulder Threshold (σ)", "k_shoulder", True),

    *section("Identification"),
    inp("Match Window ± x (Hz)", "match_tol_hz", True,
        help="Any peak within ±x of a simulated ion is identified as it"),
    hl_cut_inputs(),
    check("Also exclude ions with unknown half-life", "exclude_unknown_hl",
          help="Blank or 'p-unst' in NUBASE"),
    html.Div(style={"height": "30px"}),
], className="theia-sidebar",
   style={"width": "300px", "minWidth": "300px", "padding": "14px 18px", "background": "#f9f9f9",
          "overflowY": "auto", "borderRight": "1px solid #dddddd", "height": "calc(100vh - 42px)"})


scan_main = html.Div([
    html.Div([
        html.Div([
            html.Label("Preview Spectrum", style=LABEL_STYLE),
            dcc.Dropdown(id="preview-file", options=[], placeholder="Select a spectrum…",
                         style={"fontSize": "13px", "width": "380px"}),
        ], style={"marginRight": "12px"}),
        html.Button("Preview", id="btn-preview", className="btn btn-outline-secondary btn-sm", style=BTN),
        html.Button("Run Scan", id="btn-run", className="btn btn-sm",
                    style={**BTN, "backgroundColor": ACCENT, "color": "white"}),
        html.Button("Cancel", id="btn-cancel", className="btn btn-outline-danger btn-sm", style=BTN),
    ], style={"display": "flex", "alignItems": "flex-end", "marginBottom": "8px"}),
    html.Small("Preview uses the nominal reference frequency as the prediction. During a scan the "
               "prediction follows the reference found in the previous spectrum.", style=HELP_STYLE),
    html.Div([
        html.Div(id="progress-bar", style={"height": "100%", "width": "0%", "background": ACCENT,
                                           "transition": "width 0.4s"}),
    ], className="theia-progress-track",
       style={"height": "8px", "background": "#eeeeee", "borderRadius": "4px", "overflow": "hidden",
              "margin": "8px 0 4px 0"}),
    html.Div(id="scan-status", style={"fontSize": "13px", "color": "#333"}),
    html.Div([html.Span(id="action-msg"), html.Span(id="cancel-msg", style={"marginLeft": "8px"})],
             style={"fontSize": "12px", "color": "#888", "marginBottom": "6px"}),
    html.Div(id="run-error", style={"color": "#b00", "fontSize": "13px", "whiteSpace": "pre-wrap"}),
    html.Div(id="preview-info", style={"fontSize": "13px", "color": "#333", "whiteSpace": "pre-wrap"}),
    html.Div(id="settings-error", style={"color": "#b00", "fontSize": "13px", "whiteSpace": "pre-wrap"}),
    # plot in a box with its title (the spectrum's file name) as the heading,
    # as in the Viewer tab
    html.Div([
        html.H6("Preview", id="preview-title", style={"fontWeight": "600", "marginBottom": "6px"}),
        dcc.Graph(id="preview-graph", figure=empty_fig("Choose a spectrum and press Preview"),
                  style={"height": "58vh"}, config={"scrollZoom": True}),
    ], className="theia-box", style={**BOX_STYLE, "marginTop": "6px"}),
    html.H6("Identified in Preview", style={"fontWeight": "600", "marginTop": "6px"}),
    _table("preview-table", ["ion", "harmonic", "peak_freq", "sim_freq", "delta_hz", "power", "snr",
                             "kind", "closest_for_ion", "closest_for_peak"], page_size=15),
], style={"flex": "1", "minWidth": "0", "padding": "14px 18px", "overflowY": "auto",
          "height": "calc(100vh - 42px)"})


SETTING_FIELDS["corr_profile"] = ("s-corr_profile", "value")     # dropdown (corr_profile_ui)
SETTING_KEYS = list(SETTING_FIELDS)
SETTING_STATES = [State(cid, prop) for cid, prop in SETTING_FIELDS.values()]


def _raw_from_states(values):
    return dict(zip(SETTING_KEYS, values))


# ===========================================================================
# Callbacks — Scan tab
# ===========================================================================

@app.callback(
    Output("preview-file", "options"),
    Output("preview-file", "value"),
    Input("s-spectra_dir", "value"),
    Input("s-file_pattern", "value"),
    State("preview-file", "value"),
)
def refresh_preview_files(directory, pattern, current):
    names = list_spectra((directory or "").strip(), (pattern or "*.npz").strip())
    opts = [{"label": n, "value": n} for n in names]
    return opts, (current if current in names else (names[0] if names else None))


@app.callback(
    Output("preview-graph", "figure"),
    Output("preview-table", "data"),
    Output("preview-info", "children"),
    Output("settings-error", "children"),
    Output("preview-title", "children"),
    Input("btn-preview", "n_clicks"),
    State("preview-file", "value"),
    *SETTING_STATES,
    prevent_initial_call=True,
)
def do_preview(_, fname, *values):
    raw = _raw_from_states(values)
    save_config(raw)
    try:
        s = parse_settings(raw)
    except Exception as exc:
        return dash.no_update, [], "", f"Settings error: {exc}", dash.no_update
    if not fname:
        return empty_fig("No spectrum selected"), [], "", "", "Preview"
    try:
        t0 = time.time()
        ff, pp = load_spectrum(s["spectra_dir"], fname)
        r = analyse_spectrum(ff, pp, s, s["ref_frequency"])
        r["predicted"] = s["ref_frequency"]
        dt = time.time() - t0
    except Exception as exc:
        traceback.print_exc()
        return empty_fig(str(exc)), [], "", f"Error: {exc}", "Preview"

    peaks, m = r["peaks"], r["matches"]
    n_max = int((peaks["kind"] == "max").sum()) if len(peaks) else 0
    n_sh = int((peaks["kind"] == "shoulder").sum()) if len(peaks) else 0
    n_id_peaks = 0 if m is None or not len(m) else m["peak_row"].nunique()
    cut = noise_cutoff(pp, s["noise_cut_factor"])
    cut_txt = "off" if cut is None else f"{cut:.3g}"
    ref_txt = (f"{r['ref_used']:.1f} Hz (shift {r['ref_used'] - s['ref_frequency']:+.1f} Hz, {r['status']})"
               if r["ref_used"] is not None else "NOT FOUND in search window")
    info = (f"{fname}:  {n_max} maxima + {n_sh} shoulders  |  reference {ref_txt}  |  "
            f"{0 if m is None else len(m)} identifications on {n_id_peaks} peaks  |  "
            + (f"{int(r['excluded'].sum())} of {len(r['excluded'])} ions excluded "
               f"({hl_cut_text(s)})  |  " if hl_cut_text(s) else "")
            + f"median σ = {np.median(r['sig']):.3g}  |  noise cutoff = {cut_txt}  |  {dt:.2f} s")
    # the file name is the heading of the plot's box, not a plot title
    # (title=fname still keys the zoom memory, uirevision)
    fig = spectrum_figure(ff, pp, peaks, r["base"], r["sig"], s, r, title=fname)
    fig.update_layout(title=None)
    rows = [] if m is None else _fmt_records(m.sort_values("peak_freq"))
    return fig, rows, info, "", fname


def _fmt_records(df):
    out = df.copy()
    for c in ("peak_freq", "sim_freq", "ref_found"):
        if c in out:
            out[c] = out[c].map(lambda v: round(float(v), 2))
    for c in ("delta_hz",):
        if c in out:
            out[c] = out[c].map(lambda v: round(float(v), 2))
    for c in ("power", "snr", "mean_delta_hz", "std_delta_hz"):
        if c in out:
            out[c] = out[c].map(lambda v: None if pd.isna(v) else float(f"{v:.4g}"))
    return out.to_dict("records")


@app.callback(
    Output("action-msg", "children"),
    Output("run-error", "children"),
    Input("btn-run", "n_clicks"),
    *SETTING_STATES,
    prevent_initial_call=True,
)
def start_scan(_, *values):
    raw = _raw_from_states(values)
    save_config(raw)
    if SCAN_STATE["running"]:
        return "A scan is already running.", ""
    try:
        s = parse_settings(raw)
        if not os.path.isdir(s["spectra_dir"]):
            raise ValueError(f"Directory not found: {s['spectra_dir']}")
    except Exception as exc:
        return "", f"Settings error: {exc}"
    _set_state(running=True, cancel=False, done=0, total=0, msg="Starting…", db_path=None, error=None)
    threading.Thread(target=run_scan, args=(s, raw.get("run_label") or ""), daemon=True).start()
    return "Starting…", ""


@app.callback(
    Output("cancel-msg", "children"),
    Input("btn-cancel", "n_clicks"),
    prevent_initial_call=True,
)
def cancel_scan(_):
    if SCAN_STATE["running"]:
        _set_state(cancel=True)
        return "Cancelling… (the partial database is kept and marked 'cancelled')"
    return "No scan running."


@app.callback(
    Output("progress-bar", "style"),
    Output("scan-status", "children"),
    Output("scan-finished", "data"),
    Output("poll", "disabled"),
    Input("poll", "n_intervals"),
    Input("action-msg", "children"),   # fires when a scan is started -> enables polling
    State("progress-bar", "style"),
    State("scan-finished", "data"),
)
def poll_scan(_, _started, style, finished):
    with SCAN_LOCK:
        st = dict(SCAN_STATE)
    pct = 100.0 * st["done"] / st["total"] if st["total"] else (100.0 if not st["running"] and st["db_path"] else 0)
    style = {**(style or {}), "width": f"{pct:.1f}%"}
    new_finished = st["db_path"] if (not st["running"] and st["db_path"]) else None
    return (style, st["msg"], (new_finished if new_finished != finished else dash.no_update),
            not st["running"])
