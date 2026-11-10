"""
Scan Results tab: hits table, snippet viewer, confirmation mode, reference tracking.

Part of Theia (theia/theia.py); split out of theia.py in Nov 2026.
"""

import datetime
import json
import os
import re
import sqlite3
import traceback

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dash_table, dcc, html, Input, Output, State

from core import physics, segre_ids
from core.physics import build_ion_database, hl_allowed
from core.scan import (db_query, DEFAULTS, list_databases, load_config, load_spectrum,
                       natural_key, parse_settings, read_kv, read_run_info, save_config)

from .common import (ACCENT, ACCENT_PALE, ACCEPT_BTN, CFG, CONFIRM_BAR, CONFIRMED_DIR, FILL_BOX,
                     FILTER_HEAD_PX, HELP_STYLE, KV_WRAP, LABEL_STYLE, REF_FOLD_SUMMARY,
                     ref_peak_table, _ref_table_records, REJECT_BTN, RESULTS_PAGE_GRID,
                     RESULTS_PAGE_HIDDEN, RESULTS_PAGES, SUBTAB_SEL, SUBTAB_STYLE, _table,
                     TABLE_CELL, TABLE_HEADER, TABLE_ODD, TABLE_PAGE_SIZE, TABLE_ROW_PX,
                     UNSURE_BTN)
from .app import app
from .figures import add_sim_lines, empty_fig, spectrum_figure
from .scan_tab import _fmt_records


results_tab = html.Div([
    html.Div([
        html.Div([
            html.Label("Database", style=LABEL_STYLE),
            dcc.Dropdown(id="db-select", options=[], placeholder="Select a scan database…",
                         style={"fontSize": "13px", "width": "720px"}),
        ], style={"marginRight": "10px"}),
        html.Button("Refresh list", id="btn-refresh", className="btn btn-outline-secondary btn-sm",
                    style={"marginRight": "10px"}),
        # toggles the run info / settings tables, which open below this row
        html.Button("▸ Run info and settings used", id="btn-runinfo",
                    className="btn btn-outline-secondary btn-sm"),
    ], style={"display": "flex", "alignItems": "flex-end", "marginBottom": "10px"}),

    html.Div([
        html.Div([
            html.Div(_table("runinfo-table", ["key", "value"], page_size=50, height="320px",
                            **KV_WRAP),
                     style={"width": "40%", "marginRight": "16px", "minWidth": "0"}),
            html.Div(_table("settings-table", ["key", "value"], page_size=50, height="320px",
                            **KV_WRAP),
                     style={"width": "55%", "minWidth": "0"}),
        ], style={"display": "flex"}),
    ], id="runinfo-panel", style={"display": "none", "marginBottom": "12px"}),

    # Sub-tabs.  The bar only switches which page is visible: both pages stay
    # mounted, so plots keep their zoom (see RESULTS_PAGES).
    dcc.Tabs(id="results-subtabs", value="ions", children=[
        dcc.Tab(label="Ion search", value="ions", style=SUBTAB_STYLE, selected_style=SUBTAB_SEL),
        dcc.Tab(label="Reference tracking", value="ref", style=SUBTAB_STYLE,
                selected_style=SUBTAB_SEL),
    ], style={"height": "40px", "marginBottom": "10px"}),

    html.Div([
        # Ion search: filters + hits table (left) | snippet viewer (right)
        html.Div([
            # left: filters + hits table
            html.Div([
                # filters
                html.Div([
                    html.Div([
                        html.Div([
                            html.Label("Ion filter", style=LABEL_STYLE),
                            dcc.Dropdown(id="ion-filter", options=[], multi=True,
                                         placeholder="Type an ion, e.g. 197Os (all charge states) or 197Os76+",
                                         style={"fontSize": "13px"}),
                        ], style={"flex": "1", "marginRight": "10px"}),
                        html.Div([
                            html.Label("Spectrum filter", style=LABEL_STYLE),
                            dcc.Dropdown(id="spec-filter", options=[], multi=True, placeholder="All spectra",
                                         style={"fontSize": "13px"}),
                        ], style={"flex": "1"}),
                    ], style={"display": "flex", "marginBottom": "6px"}),
                    dcc.Checklist(id="closest-filter", options=[
                        {"label": " Only the closest peak for each ion", "value": "ion"},
                        {"label": " Only the closest ion for each peak", "value": "peak"},
                    ], value=[], inline=True, inputStyle={"marginRight": "4px"},
                        labelStyle={"marginRight": "16px"}, style={"fontSize": "13px", "marginBottom": "6px"}),
                    html.Div(id="hits-info", style={"fontSize": "13px", "color": "#333", "marginBottom": "4px"}),
                ], style={"height": f"{FILTER_HEAD_PX}px", "minWidth": "0", "flex": "0 0 auto"}),

                # hits table (Export button, always-visible horizontal scroll bar); its
                # page size is set in the browser to fill the space (see _HITS_FILL_JS)
                html.Div(_table("hits-table", ["spectrum", "ion", "harmonic", "peak_freq", "sim_freq",
                                               "delta_hz", "power", "snr", "kind"],
                                page_size=TABLE_PAGE_SIZE, export_format="csv", overflow_x="scroll"),
                         id="hits-table-wrap", style={"minWidth": "0", "flex": "1 1 0", "minHeight": "0",
                                                      "overflow": "hidden"}),
            ], style={"minWidth": "0", "minHeight": "0", "display": "flex", "flexDirection": "column"}),
            # right: peak-by-peak snippet viewer
            html.Div([
                # Peak-by-peak viewer: steps through the rows of the hits table as
                # currently filtered (and sorted) on the left
                html.Div([
                    html.Button("◀ Prev", id="btn-snip-prev", className="btn btn-outline-secondary btn-sm",
                                style={"marginRight": "6px"}),
                    dcc.Input(id="snippet-idx-input", type="number", min=1, step=1, value=1, debounce=True,
                              className="form-control form-control-sm",
                              style={"width": "90px", "marginRight": "6px"}),
                    html.Span(id="snippet-count", style={"fontSize": "13px", "marginRight": "6px"}),
                    html.Button("Next ▶", id="btn-snip-next", className="btn btn-outline-secondary btn-sm",
                                style={"marginRight": "16px"}),
                    html.Label("Snippet half-width (Hz)", style={**LABEL_STYLE, "margin": "0 8px 0 0"}),
                    dcc.Input(id="snippet-width", type="number", value=CFG["snippet_half_width"],
                              debounce=True,
                              className="form-control form-control-sm",
                              style={"width": "140px", "marginRight": "16px"}),
                    html.Button("Confirmation mode", id="btn-confirm-mode",
                                className="btn btn-outline-secondary btn-sm",
                                style={"marginRight": "16px"}),
                    html.Button("Send to PID plot", id="btn-send-pid",
                                title="Add the ion shown (charge state and harmonic) to the Segrè "
                                      "chart's identified ions, as confident",
                                className="btn btn-outline-secondary btn-sm",
                                style={"marginRight": "8px"}),
                    html.Span(id="send-pid-msg", style={"fontSize": "12px", "marginRight": "16px"}),
                    dcc.Checklist(id="snippet-show-excluded", value=["on"],
                                  options=[{"label": " Show ions excluded by half-life cut",
                                            "value": "on"}],
                                  style={"fontSize": "13px"}),
                ], style={"display": "flex", "alignItems": "center", "flexWrap": "wrap"}),
                html.Small("Steps through every identification in the table on the left, in table order; "
                           "the filters and table sorting decide which peaks are shown. "
                           "With an ion filter set, the closest to the prediction (smallest |Δ|) come first",
                           style=HELP_STYLE),
                # Confirmation mode: accept / reject the identification shown, then
                # move to the next one.  Decisions are kept for this session only,
                # until "Save accepted" writes them to confirmed/.
                html.Div([
                    html.Button("✗", id="btn-reject", title="Reject this identification",
                                className="btn btn-sm", style=REJECT_BTN),
                    html.Button("?", id="btn-unsure", title="Unsure about this identification",
                                className="btn btn-sm", style=UNSURE_BTN),
                    html.Button("✓", id="btn-accept", title="Accept this identification",
                                className="btn btn-sm", style={**ACCEPT_BTN, "marginRight": "8px"}),
                    html.Button("✓ Accept all", id="btn-accept-all",
                                title="Accept every identification in the current filtered list "
                                      "(overwrites rejects / unsures)",
                                className="btn btn-sm",
                                style={"backgroundColor": "#2e7d32", "color": "white", "fontWeight": "600",
                                       "height": "42px", "marginRight": "16px"}),
                    html.Div([
                        html.Div(id="confirm-status", style={"fontSize": "14px", "fontWeight": "600"}),
                        html.Div(id="confirm-counts", style={"fontSize": "12px"}),
                    ], style={"marginRight": "16px"}),
                    html.Button("Save accepted + unsure", id="btn-save-confirmed",
                                className="btn btn-sm",
                                style={"backgroundColor": ACCENT, "color": "white", "marginRight": "10px"}),
                    html.Span(id="confirm-save-msg", style={"fontSize": "12px"}),
                ], id="confirm-bar", style={"display": "none"}),
                dcc.Store(id="confirm-mode", data=False),
                dcc.Store(id="confirm-store", data={}),
                dcc.Store(id="snippet-idx", data=0),
                # Grows downwards to fill the box (the box fills the page height)
                dcc.Graph(id="snippet-graph", figure=empty_fig(), responsive=True,
                          style={"flex": "1 1 0", "minHeight": "300px"},
                          config={"scrollZoom": True}),
            ], className="theia-box", style=FILL_BOX),
        ], id="results-page-ions", style=RESULTS_PAGE_GRID),

        # Reference tracking: reference over the run (left) | reference in each spectrum (right).
        # Both boxes fill the page height; their plots take whatever the rest leaves.
        html.Div([
            html.Div([
                html.H6("Reference tracking", style={"fontWeight": "600", "marginBottom": "6px"}),
                dcc.Graph(id="ref-graph", figure=empty_fig(), responsive=True,
                          style={"flex": "1 1 0", "minHeight": "220px"}),
                html.Small("Click a point to open that spectrum on the right", style=HELP_STYLE),
                # per-harmonic table over the whole run
                html.Div("Reference peak in each harmonic (whole run)", id="refh-summary-title",
                         style={**REF_FOLD_SUMMARY, "marginTop": "8px", "cursor": "default"}),
                html.Div(id="refh-summary", style={"marginTop": "4px", "flex": "0 0 auto"}),
            ], className="theia-box", style=FILL_BOX),

            # Reference viewer: flip through spectra and see where the ref was found
            html.Div([
                html.H6("Reference in each spectrum", style={"fontWeight": "600", "marginBottom": "6px"}),
                html.Div([
                    html.Button("◀ Prev", id="btn-ref-prev", className="btn btn-outline-secondary btn-sm",
                                style={"marginRight": "6px"}),
                    dcc.Input(id="refview-seq-input", type="number", min=0, step=1, value=0,
                              debounce=True, className="form-control form-control-sm",
                              style={"width": "90px", "marginRight": "6px"}),
                    html.Button("Next ▶", id="btn-ref-next", className="btn btn-outline-secondary btn-sm",
                                style={"marginRight": "12px"}),
                    html.Label("Initial view ± (Hz)", style={**LABEL_STYLE, "margin": "0 6px 0 0"}),
                    dcc.Input(id="refview-width", type="number", value=2000, min=100, debounce=True,
                              className="form-control form-control-sm", style={"width": "100px"}),
                ], style={"display": "flex", "alignItems": "center", "flexWrap": "wrap"}),
                # step mode (left) beside this spectrum's per-harmonic table (right)
                html.Div([
                    html.Div([
                        dcc.RadioItems(id="refview-mode", options=[
                            {"label": " Step through all spectra", "value": "all"},
                            {"label": " Only spectra where the reference was not found", "value": "notfound"},
                            {"label": " Only harmonics where the lead peak was not found (reference found)",
                             "value": "harm_miss"},
                        ], value="all", inputStyle={"marginRight": "4px"},
                            labelStyle={"display": "block"}, style={"fontSize": "12px"}),
                        html.Div([
                            html.Span("harmonic", style={"fontSize": "12px", "marginRight": "6px"}),
                            dcc.Dropdown(id="refview-miss-h", options=[{"label": "all", "value": "all"}],
                                         value="all", clearable=False,
                                         style={"width": "170px", "fontSize": "12px"}),
                        ], title="Harmonic(s) to step through in the 'lead peak not found' mode",
                           style={"display": "flex", "alignItems": "center", "margin": "4px 0 0 18px"}),
                    ], style={"marginRight": "18px", "flex": "0 1 auto"}),
                    html.Div([
                        html.Div("Reference peak in each harmonic",
                                 style={**REF_FOLD_SUMMARY, "cursor": "default"}),
                        html.Div(id="refview-harm", style={"marginTop": "2px"}),
                    ], style={"flex": "0 1 auto", "minWidth": "0"}),
                ], style={"display": "flex", "alignItems": "flex-start", "flexWrap": "wrap",
                          "margin": "6px 0", "rowGap": "6px"}),
                html.Small("Double-click the plot to see the whole spectrum", style=HELP_STYLE),
                html.Div(id="refview-info", style={"fontSize": "12px", "color": "#333",
                                                   "whiteSpace": "pre-wrap", "marginBottom": "4px"}),
                dcc.Graph(id="refview-graph", figure=empty_fig("Load a database"), responsive=True,
                          style={"flex": "1 1 0", "minHeight": "260px"}, config={"scrollZoom": True}),
                dcc.Store(id="refview-seq", data={"seq": 0, "h": None}),
            ], className="theia-box", style=FILL_BOX),
        ], id="results-page-ref", style=RESULTS_PAGE_HIDDEN),
    ], style=RESULTS_PAGES),
], style={"padding": "14px 18px", "overflowY": "auto", "height": "calc(100vh - 42px)",
          "boxSizing": "border-box", "display": "flex", "flexDirection": "column"})


# The Scan Results plots stretch to fill their boxes, whose size changes with
# the window, the run-info panel and sub-tab switches (a hidden page has size
# 0).  Plotly only resizes on window resize, so watch each container and
# resize its plot.
_FILL_GRAPHS = ["snippet-graph", "ref-graph", "refview-graph"]
app.clientside_callback(
    """
    function(f1, f2, f3) {
        setTimeout(function() {
            if (!window.ResizeObserver) return;
            %s.forEach(function(id) {
                var box = document.getElementById(id);
                if (!box || box._theiaResize) return;
                box._theiaResize = true;
                new ResizeObserver(function() {
                    var gd = box.querySelector('.js-plotly-plot');
                    if (gd && window.Plotly && box.offsetHeight > 0) {
                        window.Plotly.Plots.resize(gd);
                    }
                }).observe(box);
            });
        }, 200);
        return window.dash_clientside.no_update;
    }
    """ % json.dumps(_FILL_GRAPHS),
    Output("_snippet-resize-hook", "children"),
    *[Input(g, "figure") for g in _FILL_GRAPHS],
)

# The hits table shows as many rows per page as fit in the space under the
# filters: watch that space and set the table's page size (keeping the first
# row shown on screen).
_HITS_FILL_JS = """
function(data) {
    setTimeout(function() {
        var wrap = document.getElementById('hits-table-wrap');
        if (!wrap || wrap._theiaFill || !window.ResizeObserver) return;
        wrap._theiaFill = true;
        var last = null;
        function fit() {
            var h = wrap.clientHeight;
            if (h <= 0) return;
            var tbl = wrap.querySelector('.dash-spreadsheet-container');
            var row = wrap.querySelector('tr td.dash-cell');
            var rowH = row ? row.parentElement.getBoundingClientRect().height : %d;
            if (!rowH) rowH = %d;
            // space taken by everything but the data rows: Export / paging bar,
            // header, filter row, horizontal scroll bar (measured when rows exist)
            var over = %d;
            if (tbl && row) {
                var rows = wrap.querySelectorAll('tr td.dash-cell:first-child').length;
                var content = wrap.firstElementChild || wrap;
                over = content.getBoundingClientRect().height - rows * rowH;
            }
            var n = Math.max(3, Math.floor((h - over - 2) / rowH));
            if (n === last) return;
            var cur = (window._theiaHitsPage || {});
            var size = cur.size || %d, page = cur.page || 0;
            var first = page * size;
            last = n;
            window._theiaHitsPage = {size: n, page: Math.floor(first / n)};
            window.dash_clientside.set_props('hits-table',
                {page_size: n, page_current: Math.floor(first / n)});
        }
        wrap._theiaFit = fit;
        new ResizeObserver(fit).observe(wrap);
        fit();
    }, 300);
    var wrap = document.getElementById('hits-table-wrap');
    if (wrap && wrap._theiaFit) { setTimeout(wrap._theiaFit, 100); }
    return window.dash_clientside.no_update;
}
""" % (TABLE_ROW_PX, TABLE_ROW_PX, 4 * TABLE_ROW_PX + 20, TABLE_PAGE_SIZE)
app.clientside_callback(
    _HITS_FILL_JS,
    Output("_hits-fill-hook", "children"),
    Input("hits-table", "data"),
)
# remember the page shown, so a resize keeps the same rows on screen
app.clientside_callback(
    """
    function(page, size) {
        window._theiaHitsPage = {size: size, page: page || 0};
        return window.dash_clientside.no_update;
    }
    """,
    Output("_hits-page-hook", "children"),
    Input("hits-table", "page_current"),
    State("hits-table", "page_size"),
)


# ===========================================================================
# Callbacks — Results tab
# ===========================================================================

def _db_label(path, info, st):
    lab = f" · {info['label']}" if info.get("label") else ""
    return (f"{info.get('created', '?')}{lab} · {info.get('status', '?')} · "
            f"{info.get('n_spectra_done', info.get('n_files', '?'))} spectra · "
            f"{info.get('n_identifications', '?')} ids · ref {st.get('ref_name', '?')} · "
            f"±{st.get('match_tol_hz', '?')} Hz · {os.path.basename(path)}")


@app.callback(
    Output("db-select", "options"),
    Output("db-select", "value"),
    Input("btn-refresh", "n_clicks"),
    Input("scan-finished", "data"),
    Input("tabs", "value"),
    State("db-select", "value"),
)
def refresh_db_list(_, finished, _tab, current):
    dbs = list_databases()
    opts = [{"label": _db_label(p, i, s), "value": p} for p, i, s in dbs]
    paths = [p for p, _, _ in dbs]
    if dash.callback_context.triggered and \
            dash.callback_context.triggered[0]["prop_id"].startswith("scan-finished") and finished in paths:
        return opts, finished
    return opts, (current if current in paths else (paths[0] if paths else None))


@app.callback(
    Output("runinfo-panel", "style"),
    Output("btn-runinfo", "children"),
    Input("btn-runinfo", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_runinfo(n):
    shown = bool(n) and n % 2 == 1
    return ({"display": "block" if shown else "none", "marginBottom": "12px"},
            ("▾ " if shown else "▸ ") + "Run info and settings used")


def _kv_value(k, v):
    # per-harmonic corrections: one line per harmonic, "214: [a, b, c]"
    if k == "corr_by_h" and isinstance(v, dict):
        def _hkey(h):
            try:
                return (0, int(h))
            except (TypeError, ValueError):
                return (1, str(h))
        return "\n".join(f"{h}: {json.dumps(v[h])}" for h in sorted(v, key=_hkey))
    return json.dumps(v) if isinstance(v, (list, dict, bool)) else str(v)


def _kv_rows(d):
    return [{"key": k, "value": _kv_value(k, v)} for k, v in d.items()]


@app.callback(
    Output("runinfo-table", "data"),
    Output("settings-table", "data"),
    Output("ion-filter", "options"),
    Output("spec-filter", "options"),
    Output("ref-graph", "figure"),
    Output("refh-summary", "children"),
    Output("refh-summary-title", "children"),
    Input("db-select", "value"),
)
def load_database(path):
    if not path or not os.path.exists(path):
        return [], [], [], [], empty_fig(), None, "Reference peak in each harmonic (whole run)"
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    info, st = read_run_info(con, path), read_kv(con, "settings")
    con.close()
    summary = db_query(path, """
        SELECT ion, COUNT(DISTINCT spectrum_id) AS n_spectra
        FROM identifications GROUP BY ion ORDER BY n_spectra DESC, ion""")
    bases = db_query(path, "SELECT ion_base, COUNT(DISTINCT spectrum_id) AS n FROM identifications "
                           "GROUP BY ion_base ORDER BY ion_base")
    ion_opts = ([{"label": f"{b}  (all charge states, {n} spectra)", "value": f"base:{b}"}
                 for b, n in zip(bases["ion_base"], bases["n"])] +
                [{"label": f"{i}  ({n} spectra)", "value": f"ion:{i}"}
                 for i, n in zip(summary["ion"], summary["n_spectra"])])
    spectra = db_query(path, "SELECT spectrum_id, seq, name, ref_status, ref_found, ref_predicted "
                             "FROM spectra ORDER BY seq")
    spec_opts = [{"label": f"{q}: {n}", "value": int(i)}
                 for i, q, n in zip(spectra["spectrum_id"], spectra["seq"], spectra["name"])]

    fig = go.Figure()
    for status, col, sym in [("found", ACCENT, "circle"), ("predicted", "#888", "x"),
                             ("not_found", "#000", "x-thin-open"), ("error", "#b00", "x")]:
        d = spectra[spectra["ref_status"] == status]
        if len(d):
            y = d["ref_found"] if status in ("found", "predicted") else d["ref_predicted"]
            fig.add_trace(go.Scattergl(x=d["seq"], y=y, mode="markers", name=status,
                                       marker=dict(color=col, size=5, symbol=sym),
                                       hovertext=d["name"], hoverinfo="text+y"))
    fig.update_layout(template="plotly_white", margin=dict(l=60, r=10, t=10, b=40),
                      xaxis_title="spectrum #", yaxis_title="ref freq (Hz)",
                      legend=dict(orientation="h", y=1.1), font=dict(size=11))
    return (_kv_rows(info), _kv_rows(st), ion_opts, spec_opts, fig,
            refh_summary(path, st),
            f"{st.get('ref_name', 'Reference')} peak in each harmonic (whole run)")


def _has_table(path, name):
    try:
        return not db_query(path, "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                            (name,)).empty
    except Exception:
        return False


def refh_summary(path, st):
    """Scan Results: the reference ion's peak in each harmonic over the run.
    (Its heading, refh-summary-title, is set in load_database.)"""
    if not _has_table(path, "ref_harmonics"):
        return html.Div([html.Small("Not recorded in this scan (made before the "
                                           "per-harmonic reference search was stored).",
                                           style=HELP_STYLE)])
    d = db_query(path, """
        SELECT harmonic,
               COUNT(*) AS n,
               SUM(status = 'found') AS n_found,
               SUM(status = 'not_found') AS n_not_found,
               SUM(status = 'off') AS n_off,
               SUM(status = 'predicted') AS n_pred,
               AVG(CASE WHEN status = 'found' THEN freq END) AS mean_f,
               AVG(offset_hz) AS mean_off,
               SQRT(MAX(AVG(offset_hz*offset_hz) - AVG(offset_hz)*AVG(offset_hz), 0)) AS std_off,
               MIN(offset_hz) AS min_off, MAX(offset_hz) AS max_off
        FROM ref_harmonics GROUP BY harmonic ORDER BY harmonic""")
    if d.empty:
        return html.Div([html.Small("No spectrum had a reference to search from.",
                                           style=HELP_STYLE)])
    h_ref = int(float(st.get("ref_harmonic") or 0))
    w = float(st.get("ref_search_hz") or 0)
    recs = []
    for r in d.itertuples():
        n = int(r.n)
        if int(r.harmonic) == h_ref:
            status = f"reference ({int(r.n_found or 0)} found, {int(r.n_pred or 0)} predicted)"
        elif int(r.n_off or 0) == n:
            status = "search off"
        else:
            status = f"{int(r.n_found or 0)} / {n} found"
        f = lambda v, fmt: "—" if v is None or pd.isna(v) else format(v, fmt)
        recs.append({"h": int(r.harmonic), "status": status,
                     "not_found": int(r.n_not_found or 0),
                     "mean_f": f(r.mean_f, ".1f"), "mean_off": f(r.mean_off, "+.1f"),
                     "std_off": f(r.std_off, ".1f"),
                     "range": "—" if r.min_off is None or pd.isna(r.min_off)
                     else f"{r.min_off:+.1f} … {r.max_off:+.1f}"})
    cols = [("h", "h"), ("status", "found"), ("not_found", "not found"),
            ("mean_f", "mean found at (Hz)"), ("mean_off", "mean offset (Hz)"),
            ("std_off", "spread (Hz)"), ("range", "offset range (Hz)")]
    tbl = dash_table.DataTable(
        id="refh-summary-dt", columns=[{"name": n, "id": c} for c, n in cols], data=recs, fill_width=False,
        style_table={"overflowX": "auto"}, style_header=TABLE_HEADER, style_cell=TABLE_CELL,
        style_data_conditional=TABLE_ODD + [
            {"if": {"filter_query": "{not_found} > 0", "column_id": "not_found"},
             "color": "#c0392b", "fontWeight": "600"},
            {"if": {"column_id": "status"}, "textAlign": "left"}])
    note = (f"Offset = found − f_ref / h_ref × h (other harmonics searched within ±{w:g} Hz). "
            if w > 0 else "Reference search ± was 0: other harmonics used predicted positions. ")
    return html.Div([tbl, html.Small(note + "Reference harmonic: the scan's own reference "
                                            "search, offset from its (tracked) prediction.",
                                            style=HELP_STYLE)])


HIT_COLUMNS = ["spectrum", "seq", "spectrum_id", "peak_id", "ion", "harmonic", "peak_freq", "sim_freq",
               "delta_hz", "power", "snr", "kind", "closest_for_ion", "closest_for_peak", "ref_found"]
MAX_ROWS = 20000


@app.callback(
    Output("hits-table", "data"),
    Output("hits-info", "children"),
    Input("db-select", "value"),
    Input("ion-filter", "value"),
    Input("spec-filter", "value"),
    Input("closest-filter", "value"),
)
def query_hits(path, ion_filter, spec_filter, closest):
    if not path or not os.path.exists(path):
        return [], ""
    where, params = [], []
    ions = [v[4:] for v in (ion_filter or []) if v.startswith("ion:")]
    bases = [v[5:] for v in (ion_filter or []) if v.startswith("base:")]
    ors = []
    if ions:
        ors.append(f"ion IN ({','.join('?' * len(ions))})")
        params += ions
    if bases:
        ors.append(f"ion_base IN ({','.join('?' * len(bases))})")
        params += bases
    if ors:
        where.append("(" + " OR ".join(ors) + ")")
    if spec_filter:
        where.append(f"spectrum_id IN ({','.join('?' * len(spec_filter))})")
        params += [int(x) for x in spec_filter]
    if "ion" in (closest or []):
        where.append("closest_for_ion = 1")
    if "peak" in (closest or []):
        where.append("closest_for_peak = 1")
    # With an ion filter, closest to the prediction first (|Δ| over all the
    # selected ions together), so the snippet viewer steps through the best
    # matches first; otherwise run order.  Sorting a column still overrides.
    order = "ABS(delta_hz), seq, peak_freq" if ors else "seq, peak_freq"
    sql = (f"SELECT {', '.join(HIT_COLUMNS)} FROM hits"
           + (" WHERE " + " AND ".join(where) if where else "")
           + f" ORDER BY {order} LIMIT {MAX_ROWS + 1}")
    df = db_query(path, sql, params)
    truncated = len(df) > MAX_ROWS
    df = df.iloc[:MAX_ROWS]
    n_spec = df["spectrum_id"].nunique() if len(df) else 0
    info = (f"{len(df)} identifications in {n_spec} spectra"
            + ("  (closest to prediction first)" if ors else "")
            + (f"  (showing first {MAX_ROWS}; narrow the filter)" if truncated else ""))

    return _fmt_records(df), info


@app.callback(
    Output("snippet-idx", "data"),
    Output("snippet-idx-input", "value"),
    Output("snippet-count", "children"),
    Input("btn-snip-prev", "n_clicks"),
    Input("btn-snip-next", "n_clicks"),
    Input("btn-accept", "n_clicks"),
    Input("btn-reject", "n_clicks"),
    Input("btn-unsure", "n_clicks"),
    Input("snippet-idx-input", "value"),
    Input("hits-table", "derived_virtual_data"),
    State("snippet-idx", "data"),
)
def snippet_step(_p, _n, _a, _r, _u, typed, rows, current):
    """Which row of the (filtered, sorted) hits table the snippet shows.
    Goes back to the first peak whenever the table contents change."""
    n = len(rows or [])
    if n == 0:
        return 0, 1, "of 0"
    trig = dash.callback_context.triggered[0]["prop_id"] if dash.callback_context.triggered else ""
    cur = int(current or 0)
    if trig.startswith(("btn-snip-next", "btn-accept", "btn-reject", "btn-unsure")):
        new = cur + 1   # accept / reject / unsure also move on to the next peak
    elif trig.startswith("btn-snip-prev"):
        new = cur - 1
    elif trig.startswith("snippet-idx-input") and typed is not None:
        new = int(typed) - 1
    else:
        new = 0
    new = min(max(new, 0), n - 1)
    return new, new + 1, f"of {n}"


@app.callback(
    Output("snippet-graph", "figure"),
    Input("snippet-idx", "data"),
    Input("snippet-width", "value"),
    State("snippet-show-excluded", "value"),
    State("hits-table", "derived_virtual_data"),
    State("db-select", "value"),
)
def show_snippet(idx, half_width, show_excluded, rows, path):
    if not rows or not path or idx is None or idx >= len(rows):
        return empty_fig("No identifications to show")
    row = rows[int(idx)]
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        st = read_kv(con, "settings")
        con.close()
        s = parse_settings(st)
        ff, pp = load_spectrum(s["spectra_dir"], row["spectrum"])
        peaks = db_query(path, "SELECT freq, power, prominence, snr, kind FROM peaks WHERE spectrum_id=?",
                         (int(row["spectrum_id"]),))
        ion_db, ref_mq = build_ion_database(*s["ion_key"])
        sims, _, _ = physics.simulate_spectrum(ion_db, ref_mq, float(row["ref_found"]), s, ff, pp)
        hits = db_query(path, "SELECT ion, harmonic FROM identifications WHERE spectrum_id=?",
                        (int(row["spectrum_id"]),))
        w = float(half_width or 1000)
        x0, x1 = row["peak_freq"] - w, row["peak_freq"] + w
        # All simulated lines in the spectrum (not just the view), so they are
        # there when you pan or zoom out
        sims = sims[(sims["f_sim"] >= ff.min()) & (sims["f_sim"] <= ff.max())].reset_index(drop=True)
        r = dict(sims=sims, ion_db=ion_db, matches=hits, ref_used=float(row["ref_found"]),
                 status="found", predicted=None, excluded=~hl_allowed(ion_db, s),
                 show_excluded=bool(show_excluded))
        fig = spectrum_figure(ff, pp, peaks, None, None, s, r,
                              title=f"{row['spectrum']}  –  {row['ion']} (h={row['harmonic']}), "
                                    f"Δ = {row['delta_hz']:+.1f} Hz",
                              xrange=[x0, x1], highlight_ion=row["ion"])
        fig.add_vrect(x0=row["sim_freq"] - s["match_tol_hz"], x1=row["sim_freq"] + s["match_tol_hz"],
                      fillcolor=ACCENT_PALE, opacity=0.8, line_width=0, layer="below")
        fig.add_trace(go.Scatter(x=[row["peak_freq"]], y=[row["power"]], mode="markers",
                                 name="this peak", hoverinfo="skip",
                                 marker=dict(symbol="circle-open", size=18, color=ACCENT,
                                             line=dict(width=2))))
        # Fix the vertical scale so this peak always reaches exactly half the
        # plot height: bottom = lowest point in view, top = bottom + 2 x height
        in_view = (ff >= x0) & (ff <= x1)
        if in_view.any():
            lo = float(pp[in_view].min())
            top = float(pp[int(np.argmin(np.abs(ff - float(row["peak_freq"]))))])
            if top > lo:
                fig.update_yaxes(range=[lo, lo + 2.0 * (top - lo)], autorange=False)

        # Peak properties (as stored by the scan) in a box at the top right
        pk = db_query(path, "SELECT power, prominence, snr, height_snr, width_hz, kind "
                            "FROM peaks WHERE peak_id = ?", (int(row["peak_id"]),))
        if len(pk):
            p = pk.iloc[0]
            bin_hz = (ff[-1] - ff[0]) / (len(ff) - 1)
            fmt = lambda v, f: "–" if pd.isna(v) else format(float(v), f)
            lines = [f"<b>Peak ({p['kind']})</b>",
                     f"Height: {fmt(p['power'], '.4g')}  ({fmt(p['height_snr'], '.1f')} σ)",
                     f"Prominence: {fmt(p['prominence'], '.4g')}  ({fmt(p['snr'], '.1f')} σ)"]
            if pd.notna(p["width_hz"]):
                lines.append(f"Width (FWHM): {p['width_hz']:.1f} Hz  ({p['width_hz'] / bin_hz:.2f} bins)")
            else:
                lines.append("Width: –")
            fig.add_annotation(text="<br>".join(lines), xref="paper", yref="paper", x=0.995, y=0.99,
                               xanchor="right", yanchor="top", showarrow=False, align="left",
                               font=dict(size=12, color="#222"), bgcolor="rgba(255,255,255,0.88)",
                               bordercolor=ACCENT, borderwidth=1, borderpad=6)
        return fig
    except Exception as exc:
        traceback.print_exc()
        return empty_fig(f"Could not load spectrum: {exc}")


@app.callback(
    Output("send-pid-msg", "children"),
    Input("btn-send-pid", "n_clicks"),
    Input("snippet-idx", "data"),
    State("hits-table", "derived_virtual_data"),
    State("db-select", "value"),
    prevent_initial_call=True,
)
def send_to_pid_plot(_n, idx, rows, path):
    """'Send to PID plot': add the ion in the snippet (its charge state and
    harmonic) to the Segrè chart as a confident identification of your own
    (core/segre_ids.py).  Already there -> left as it is.  The message is
    cleared when the snippet moves to another peak."""
    trig = dash.callback_context.triggered[0]["prop_id"] if dash.callback_context.triggered else ""
    if not trig.startswith("btn-send-pid"):
        return ""
    if not rows or not path or idx is None or int(idx) >= len(rows):
        return "No identification shown."
    row = rows[int(idx)]
    try:
        r = db_query(path, "SELECT Z, N FROM identifications WHERE spectrum_id = ? AND peak_id = ? "
                           "AND ion = ? AND harmonic = ? LIMIT 1",
                     (int(row["spectrum_id"]), int(row["peak_id"]), row["ion"], int(row["harmonic"])))
        if not len(r):
            return "✗ This identification is not in the database any more."
        note = (f"Sent from Theia: {row['spectrum']}, peak {float(row['peak_freq']):.1f} Hz, "
                f"Δ = {float(row['delta_hz']):+.1f} Hz ({os.path.basename(path)})")
        added, msg = segre_ids.add_identification(int(r.iloc[0]["Z"]), int(r.iloc[0]["N"]), row["ion"],
                                                   int(row["harmonic"]), "confident", note)
        print(f"Send to PID plot: {msg}", flush=True)
        return msg
    except Exception as exc:
        traceback.print_exc()
        return f"✗ Could not add it to the Segrè chart: {exc}"


@app.callback(
    Output("snippet-width", "value"),
    Input("tabs", "value"),
)
def restore_snippet_width(_tab):
    """Show the last saved half-width when the page (re)loads."""
    return load_config().get("snippet_half_width", DEFAULTS["snippet_half_width"])


@app.callback(
    Output("_snippet-width-saved", "children"),
    Input("snippet-width", "value"),
    prevent_initial_call=True,
)
def remember_snippet_width(value):
    """Save the half-width so it is restored next time Theia runs."""
    if value is not None and value > 0:
        save_config({"snippet_half_width": value})
    return dash.no_update


# ===========================================================================
# Callbacks — confirmation mode (Results tab, snippet viewer)
# ===========================================================================

def _ident_key(path, row):
    """Unique key for one identification (peak + ion + harmonic) in one database."""
    return f"{os.path.basename(path)}|{int(row['spectrum_id'])}|{int(row['peak_id'])}|" \
           f"{row['ion']}|{int(row['harmonic'])}"


def _filter_text(ion_filter, spec_filter, closest):
    ions = [v.split(":", 1)[1] + (" (all charge states)" if v.startswith("base:") else "")
            for v in (ion_filter or [])]
    parts = ["ions: " + (", ".join(ions) if ions else "all")]
    parts.append("spectra: " + (", ".join(map(str, spec_filter)) if spec_filter else "all"))
    if closest:
        parts.append("closest only: " + ", ".join(closest))
    return "; ".join(parts)


@app.callback(
    Output("confirm-mode", "data"),
    Output("confirm-bar", "style"),
    Output("btn-confirm-mode", "className"),
    Output("btn-confirm-mode", "style"),
    Input("btn-confirm-mode", "n_clicks"),
    State("confirm-mode", "data"),
)
def toggle_confirm_mode(n, active):
    on = (not active) if n else False
    style = CONFIRM_BAR if on else {"display": "none"}
    if on:
        return True, style, "btn btn-sm", {"backgroundColor": ACCENT, "color": "white"}
    return False, style, "btn btn-outline-secondary btn-sm", {}


@app.callback(
    Output("confirm-store", "data"),
    Input("btn-accept", "n_clicks"),
    Input("btn-reject", "n_clicks"),
    Input("btn-unsure", "n_clicks"),
    Input("btn-accept-all", "n_clicks"),
    State("snippet-idx", "data"),
    State("hits-table", "derived_virtual_data"),
    State("db-select", "value"),
    State("ion-filter", "value"),
    State("spec-filter", "value"),
    State("closest-filter", "value"),
    State("confirm-store", "data"),
    prevent_initial_call=True,
)
def record_decision(_a, _r, _u, _all, idx, rows, path, ion_filter, spec_filter, closest, store):
    """Remember accept / reject / unsure for the identification currently shown,
    or accept every identification in the filtered list ("Accept all")."""
    if not rows or not path:
        return dash.no_update
    trig = dash.callback_context.triggered[0]["prop_id"]
    store = dict(store or {})
    flt = _filter_text(ion_filter, spec_filter, closest)
    now = datetime.datetime.now().isoformat(timespec="seconds")

    def _rec(row, decision):
        store[_ident_key(path, row)] = dict(
            decision=decision, db=os.path.basename(path), spectrum=row["spectrum"],
            spectrum_id=int(row["spectrum_id"]), peak_id=int(row["peak_id"]),
            ion=row["ion"], harmonic=int(row["harmonic"]), filter=flt, time=now)

    if trig.startswith("btn-accept-all"):
        for row in rows:          # whole filtered list; overwrites earlier decisions
            _rec(row, "accept")
        return store
    if idx is None or idx >= len(rows):
        return dash.no_update
    _rec(rows[int(idx)], "accept" if trig.startswith("btn-accept") else
                         "unsure" if trig.startswith("btn-unsure") else "reject")
    return store


@app.callback(
    Output("confirm-status", "children"),
    Output("confirm-status", "style"),
    Output("confirm-counts", "children"),
    Input("confirm-store", "data"),
    Input("snippet-idx", "data"),
    Input("hits-table", "derived_virtual_data"),
    State("db-select", "value"),
)
def show_decision(store, idx, rows, path):
    base = {"fontSize": "14px", "fontWeight": "600"}
    if not path:
        return "", base, ""
    store = store or {}
    db = os.path.basename(path)
    mine = [v for v in store.values() if v["db"] == db]
    n_acc = sum(v["decision"] == "accept" for v in mine)
    n_uns = sum(v["decision"] == "unsure" for v in mine)
    n_rej = len(mine) - n_acc - n_uns
    counts = (f"Accepted {n_acc} · unsure {n_uns} · rejected {n_rej} "
              f"(this database, unsaved until you press Save)")
    if not rows or idx is None or idx >= len(rows):
        return "", base, counts
    row = rows[int(idx)]
    d = store.get(_ident_key(path, row))
    label = f"{row['ion']} (h={row['harmonic']}) in {row['spectrum']}: "
    if d is None:
        return label + "undecided", base, counts
    if d["decision"] == "accept":
        return label + "✓ accepted", {**base, "color": "#2e7d32"}, counts
    if d["decision"] == "unsure":
        return label + "? unsure", {**base, "color": ACCENT}, counts
    return label + "✗ rejected", {**base, "color": "#c62828"}, counts


@app.callback(
    Output("confirm-save-msg", "children"),
    Input("btn-save-confirmed", "n_clicks"),
    State("confirm-store", "data"),
    State("db-select", "value"),
    State("ion-filter", "value"),
    State("spec-filter", "value"),
    State("closest-filter", "value"),
    State("hits-table", "filter_query"),
    State("hits-table", "sort_by"),
    State("hits-table", "derived_virtual_data"),
    prevent_initial_call=True,
)
def save_confirmed(_n, store, path, ion_filter, spec_filter, closest, table_filter, sort_by, rows):
    """Write the accepted and unsure identifications for this database to
    confirmed/*.csv (with a 'decision' column), with the filters and the scan's
    run info and settings as '#' header lines."""
    if not path or not os.path.exists(path):
        return "No database loaded."
    db = os.path.basename(path)
    decisions = [v for v in (store or {}).values() if v["db"] == db]
    accepted = [v for v in decisions if v["decision"] == "accept"]
    unsure = [v for v in decisions if v["decision"] == "unsure"]
    to_save = accepted + unsure
    if not to_save:
        return "Nothing accepted or marked unsure yet - nothing saved."

    # Exact (unrounded) values straight from the database
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    info, st = read_run_info(con, path), read_kv(con, "settings")
    cols = ("s.name AS spectrum, s.seq, i.spectrum_id, i.peak_id, i.ion, i.ion_base, i.Z, i.N, i.A, "
            "i.charge, i.harmonic, i.peak_freq, i.sim_freq, i.delta_hz, i.power, i.snr, i.kind, "
            "i.closest_for_ion, i.closest_for_peak, s.ref_found")
    out = []
    for v in sorted(to_save, key=lambda d: (d["spectrum_id"], d["peak_id"], d["ion"])):
        r = pd.read_sql_query(
            f"SELECT {cols} FROM identifications i JOIN spectra s USING (spectrum_id) "
            "WHERE i.spectrum_id = ? AND i.peak_id = ? AND i.ion = ? AND i.harmonic = ?",
            con, params=(v["spectrum_id"], v["peak_id"], v["ion"], v["harmonic"]))
        if len(r):
            r = r.iloc[:1].copy()
            r.insert(0, "decision", "accepted" if v["decision"] == "accept" else "unsure")
            r["filter_when_decided"] = v["filter"]
            r["decided_at"] = v["time"]
            out.append(r)
    con.close()
    table = pd.concat(out, ignore_index=True)

    os.makedirs(CONFIRMED_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Ion names in the file name: the ions the list was filtered to, or, with
    # no ion filter, the ions that were saved (first 6, then "+N more")
    if ion_filter:
        name_ions = [v.split(":", 1)[1] for v in ion_filter]
    else:
        name_ions = sorted(table["ion"].unique(), key=natural_key)
    if len(name_ions) > 6:
        name_ions = name_ions[:6] + [f"+{len(name_ions) - 6}more"]
    ion_part = "_".join(re.sub(r"[^A-Za-z0-9+.-]", "", s) for s in name_ions) or "ions"
    # Harmonic in the file name if the list was filtered by harmonic (in the
    # hits table's filter row): use the harmonics left in the filtered list
    harmonics = []
    if table_filter and "{harmonic}" in table_filter and rows:
        harmonics = sorted({int(r["harmonic"]) for r in rows if r.get("harmonic") is not None})
    harm_part = ("_h" + "-".join(map(str, harmonics))) if harmonics else ""
    fname = os.path.join(CONFIRMED_DIR,
                         f"confirmed_{ion_part}{harm_part}_{stamp}_{os.path.splitext(db)[0]}.csv")
    n_list = len(rows or [])
    header = {
        "saved": datetime.datetime.now().isoformat(timespec="seconds"),
        "database": path,
        "filter": _filter_text(ion_filter, spec_filter, closest),
        "table_filter": table_filter or "",
        "table_sort": json.dumps(sort_by or []),
        "harmonic_filter": ", ".join(map(str, harmonics)) if harmonics else "none",
        "identifications_in_filtered_list": n_list,
        "accepted": len(accepted),
        "unsure": len(unsure),
        "rejected": len(decisions) - len(to_save),
        "run_info": json.dumps(info),
        "scan_settings": json.dumps(st),
    }
    with open(fname, "w", newline="") as fh:
        for k, v in header.items():
            fh.write(f"# {k}: {v}\n")
        table.to_csv(fh, index=False)
    return (f"Saved {len(table)} identifications ({len(accepted)} accepted, {len(unsure)} unsure) "
            f"to confirmed/{os.path.basename(fname)}")


# ===========================================================================
# Callbacks — reference viewer (Results tab)
# ===========================================================================

def _refview_cur(current):
    """refview-seq store -> (seq, h or None); older pages stored a bare seq."""
    if isinstance(current, dict):
        h = current.get("h")
        return int(current.get("seq") or 0), (None if h is None else int(h))
    return int(current or 0), None


def harm_miss_cases(path, h="all"):
    """(seq, harmonic) of every spectrum where the reference was found at the
    reference harmonic but the lead peak in harmonic h (or any) was not."""
    if not path or not _has_table(path, "ref_harmonics"):
        return []
    sql = ("SELECT s.seq, r.harmonic FROM ref_harmonics r JOIN spectra s USING (spectrum_id) "
           "WHERE r.status = 'not_found' AND s.ref_status = 'found'")
    params = ()
    if h not in (None, "all"):
        sql += " AND r.harmonic = ?"
        params = (int(h),)
    d = db_query(path, sql + " ORDER BY s.seq, r.harmonic", params)
    return [(int(a), int(b)) for a, b in zip(d["seq"], d["harmonic"])]


@app.callback(
    Output("refview-miss-h", "options"),
    Output("refview-miss-h", "value"),
    Input("db-select", "value"),
    State("refview-miss-h", "value"),
)
def refview_miss_options(path, current):
    cases = harm_miss_cases(path) if path and os.path.exists(path) else []
    counts = {}
    for _, h in cases:
        counts[h] = counts.get(h, 0) + 1
    opts = ([{"label": f"all ({len(cases)})", "value": "all"}] +
            [{"label": f"h{h} ({n})", "value": h} for h, n in sorted(counts.items())])
    return opts, (current if current in [o["value"] for o in opts] else "all")


@app.callback(
    Output("refview-seq", "data"),
    Output("refview-seq-input", "value"),
    Input("btn-ref-prev", "n_clicks"),
    Input("btn-ref-next", "n_clicks"),
    Input("refview-seq-input", "value"),
    Input("ref-graph", "clickData"),
    Input("db-select", "value"),
    Input("refview-mode", "value"),
    Input("refview-miss-h", "value"),
    State("refview-seq", "data"),
)
def refview_step(_p, _n, typed, click, path, mode, miss_h, current):
    """Work out which spectrum (seq number) the reference viewer shows, and
    in the 'lead peak not found' mode which harmonic."""
    if not path or not os.path.exists(path):
        return {"seq": 0, "h": None}, 0
    sp = db_query(path, "SELECT seq, ref_status FROM spectra ORDER BY seq")
    if sp.empty:
        return {"seq": 0, "h": None}, 0
    trig = dash.callback_context.triggered[0]["prop_id"] if dash.callback_context.triggered else ""
    cur, cur_h = _refview_cur(current)
    target = None                                # a seq asked for by typing / clicking
    if trig.startswith("ref-graph") and click and click.get("points"):
        target = int(round(click["points"][0]["x"]))
    elif trig.startswith("refview-seq-input") and typed is not None:
        target = int(typed)

    if mode == "harm_miss":
        cases = harm_miss_cases(path, miss_h)
        if not cases:                            # nothing to step through: show the note
            return {"seq": cur, "h": None, "miss_empty": True}, cur
        key = (cur, -1 if cur_h is None else cur_h)
        if trig.startswith("btn-ref-next"):
            later = [c for c in cases if c > key]
            new = later[0] if later else cases[-1]
        elif trig.startswith("btn-ref-prev"):
            earlier = [c for c in cases if c < key]
            new = earlier[-1] if earlier else cases[0]
        elif trig.startswith("db-select"):
            new = cases[0]
        else:
            # mode / harmonic changed, or a seq typed / clicked: first case at or after it
            want = target if target is not None else cur
            if cur_h is not None and target is None and key in cases:
                new = key
            else:
                after = [c for c in cases if c[0] >= want]
                new = after[0] if after else cases[-1]
        return {"seq": new[0], "h": new[1]}, new[0]

    allowed = sp[sp["ref_status"] != "found"]["seq"].to_numpy() if mode == "notfound" \
        else sp["seq"].to_numpy()
    if len(allowed) == 0:
        allowed = sp["seq"].to_numpy()
    if trig.startswith("db-select"):
        new = int(allowed[0])
    elif trig.startswith("btn-ref-next"):
        later = allowed[allowed > cur]
        new = int(later[0]) if len(later) else int(allowed[-1])
    elif trig.startswith("btn-ref-prev"):
        earlier = allowed[allowed < cur]
        new = int(earlier[-1]) if len(earlier) else int(allowed[0])
    elif target is not None:
        new = target
    else:
        new = cur
    new = int(min(max(new, int(sp["seq"].min())), int(sp["seq"].max())))
    return {"seq": new, "h": None}, new


@app.callback(
    Output("refview-graph", "figure"),
    Output("refview-info", "children"),
    Output("refview-harm", "children"),
    Input("refview-seq", "data"),
    Input("refview-width", "value"),
    Input("db-select", "value"),
    State("refview-miss-h", "value"),
)
def refview_plot(state, half_width, path, miss_h="all"):
    seq, h = _refview_cur(state)
    if h is not None:
        fig, info = _refview_miss_plot(seq, h, half_width, path, miss_h)
        return fig, info, refview_harm_table(path, seq, [], mark_h=h)
    rows = []
    fig, info = _refview_plot(seq, half_width, path, rows)
    if isinstance(state, dict) and state.get("miss_empty"):
        info = ("No spectra where the reference was found but a lead peak in another harmonic "
                "was not" + ("" if _has_table(path, "ref_harmonics") else
                             " (not recorded in this scan: it was made before the per-harmonic "
                             "reference search was stored)") + ".\n" + (info or ""))
    return fig, info, refview_harm_table(path, seq, rows)


def _refview_miss_plot(seq, h, half_width, path, miss_h="all"):
    """Reference viewer, 'lead peak not found' mode: harmonic h of spectrum
    seq around the expected position f_ref / h_ref * h, with the search
    window and why the search found no peak there."""
    if not path or not os.path.exists(path):
        return empty_fig("Load a database"), ""
    row = db_query(path, "SELECT s.spectrum_id, s.name, s.ref_found, r.predicted, r.status "
                         "FROM spectra s JOIN ref_harmonics r USING (spectrum_id) "
                         "WHERE s.seq = ? AND r.harmonic = ?", (int(seq), int(h)))
    if row.empty:
        return empty_fig("No such spectrum / harmonic"), ""
    row = row.iloc[0]
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    st = read_kv(con, "settings")
    con.close()
    h_ref = int(float(st.get("ref_harmonic") or 0))
    ref_name = st.get("ref_name", "reference")
    pred = float(row["predicted"])
    window = float(st.get("ref_search_hz") or 0)
    cases = harm_miss_cases(path, miss_h)
    i_case = cases.index((int(seq), int(h))) + 1 if (int(seq), int(h)) in cases else 0

    info = (f"Case {i_case} of {len(cases)}:  #{int(seq)}  {row['name']}  |  h{h}: {ref_name} "
            f"not found  |  h{h_ref} reference found at {float(row['ref_found']):,.1f} Hz\n"
            f"Expected at f_ref / {h_ref} × {h} = {pred:,.1f} Hz")
    try:
        ff, pp = load_spectrum(st["spectra_dir"], row["name"])
    except Exception as exc:
        return empty_fig(f"Could not load spectrum: {exc}"), info
    ff = np.asarray(ff, dtype=float)
    pp = np.asarray(pp, dtype=float)
    if np.any(np.diff(ff) < 0):
        o = np.argsort(ff)
        ff, pp = ff[o], pp[o]
    ex = physics.peak_search_explain(ff, pp, pred, window)
    if ex["lo_hz"] is not None:
        info += (f", searched {ex['lo_hz']:,.1f} – {ex['hi_hz']:,.1f} Hz (±{ex['window_hz']:.1f} Hz"
                 + (f", widened from ±{window:g} Hz to 3 bins" if ex["window_hz"] > window else "")
                 + f"; {ex['n_bins']} bins of {ex['bin_hz']:.1f} Hz)")
    if ex["found"] is not None:
        info += (f"\nRecalculated now: {ex['reason']}. The scan did not find it, so the data or "
                 f"settings must differ from when it was run.")
    else:
        info += f"\nWhy not found: {ex['reason']}."

    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=ff, y=pp, mode="lines", name="Spectrum", hoverinfo="skip",
                               line=dict(shape="hvh", color="black", width=1)))
    if ex["lo_hz"] is not None:
        fig.add_vrect(x0=ex["lo_hz"], x1=ex["hi_hz"], fillcolor=ACCENT_PALE, opacity=0.8,
                      line_width=0, layer="below")
    fig.add_vline(x=pred, line=dict(color="#999999", width=1, dash="dash"),
                  annotation_text=f"expected h{h}", annotation_font_color="#999999",
                  annotation_position="top left")
    if ex["f_max"] is not None:
        fig.add_trace(go.Scatter(x=[ex["f_max"]], y=[ex["p_max"]], mode="markers",
                                 name="strongest bin in window",
                                 marker=dict(symbol="x", color=ACCENT, size=11,
                                             line=dict(color=ACCENT, width=2)),
                                 hovertext=[f"strongest bin in the window<br>{ex['f_max']:,.1f} Hz"
                                            f"<br>P = {ex['p_max']:.4g}"], hoverinfo="text"))
    w = float(half_width or 2000)
    vm = (ff >= pred - w) & (ff <= pred + w)
    fig.update_layout(template="plotly_white", margin=dict(l=55, r=10, t=10, b=40),
                      xaxis_title="Frequency (Hz)", yaxis_title="Power (a.u.)",
                      dragmode="pan", hovermode="closest", font=dict(size=11), showlegend=False)
    fig.update_xaxes(range=[pred - w, pred + w], autorange=False)
    if vm.any():
        lo, hi = float(np.nanmin(pp[vm])), float(np.nanmax(pp[vm]))
        fig.update_yaxes(range=[lo - 0.05 * (hi - lo or 1), hi + 0.12 * (hi - lo or 1)])
    return fig, info


def refview_harm_table(path, seq, recalculated, mark_h=None):
    """Reference viewer: the reference peak in each harmonic for this
    spectrum, as stored by the scan (or recalculated for older scans)."""
    if not path or not os.path.exists(path):
        return None
    src = "stored by the scan"
    recs = []
    if _has_table(path, "ref_harmonics"):
        d = db_query(path, "SELECT r.harmonic, r.status, r.predicted, r.freq, r.offset_hz "
                           "FROM ref_harmonics r JOIN spectra s USING (spectrum_id) "
                           "WHERE s.seq = ? ORDER BY r.harmonic", (int(seq or 0),))
        for r in d.itertuples():
            shown = r.status in ("found", "predicted")
            recs.append({"h": int(r.harmonic),
                         "status": physics.REF_STATUS_LABEL.get(r.status, r.status),
                         "predicted": f"{r.predicted:.1f}",
                         "freq": f"{r.freq:.1f}" if shown else "—",
                         "offset": "—" if r.offset_hz is None or pd.isna(r.offset_hz)
                         else f"{r.offset_hz:+.1f}"})
    if not recs and recalculated:
        src = "recalculated (not stored in this scan)"
        recs = _ref_table_records(recalculated, ref_f_label="found")
    if not recs:
        return None
    return html.Div([
        html.Div(html.Span(src, style={"color": "#888", "fontSize": "11px"}),
                 style={"fontSize": "12px", "marginBottom": "3px"}),
        ref_peak_table("refview-harm-dt", recs, mark_h=mark_h)])


def _refview_plot(seq, half_width, path, rows_out):
    if not path or not os.path.exists(path):
        return empty_fig("Load a database"), ""
    row = db_query(path, "SELECT * FROM spectra WHERE seq = ?", (int(seq or 0),))
    if row.empty:
        return empty_fig("No such spectrum"), ""
    row = row.iloc[0]
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    st = read_kv(con, "settings")
    con.close()
    n_total = int(db_query(path, "SELECT COUNT(*) AS n FROM spectra")["n"].iloc[0])

    pred = row["ref_predicted"]
    found = row["ref_found"] if pd.notna(row["ref_found"]) else None
    window = float(st.get("ref_window_hz", 0))
    centre = found if found is not None else pred
    w = float(half_width or 2000)

    info = (f"#{int(row['seq'])} of {n_total - 1}:  {row['name']}  |  ref {st.get('ref_name', '?')} "
            f"(h={st.get('ref_harmonic', '?')})  |  status: {row['ref_status']}  |  "
            f"predicted {pred:.1f} Hz" if pd.notna(pred) else f"#{int(row['seq'])}: {row['name']}")
    if found is not None:
        info += f"  |  found {found:.1f} Hz ({found - pred:+.1f} Hz from prediction)"
    if isinstance(row.get("error"), str) and row["error"]:
        info += f"\nError: {row['error']}"

    try:
        ff, pp = load_spectrum(st["spectra_dir"], row["name"])
    except Exception as exc:
        return empty_fig(f"Could not load spectrum: {exc}"), info

    # The whole spectrum is sent so you can zoom/pan anywhere; the initial
    # view is zoomed on the reference (double-click to see everything)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=ff, y=pp, mode="lines", name="Spectrum", hoverinfo="skip",
                               line=dict(shape="hvh", color="black", width=1)))
    # Simulated lines, recalculated from this spectrum's reference exactly as in the scan
    if found is not None:
        try:
            s_db = parse_settings(st)
            ion_db, ref_mq = build_ion_database(*s_db["ion_key"])
            sims, rinfo, ref_warn = physics.simulate_spectrum(ion_db, ref_mq, float(found), s_db,
                                                              ff, pp)
            rows_out += physics.reference_rows(rinfo, s_db["ref_harmonic"])
            sims = sims[(sims["f_sim"] >= ff.min()) & (sims["f_sim"] <= ff.max())].reset_index(drop=True)
            ids = db_query(path, "SELECT ion, harmonic FROM identifications WHERE spectrum_id = ?",
                           (int(row["spectrum_id"]),))
            ylo, yhi = float(np.nanmin(pp)), float(np.nanmax(pp))
            pad = 0.05 * (yhi - ylo or 1)
            add_sim_lines(fig, sims, ion_db, ids, ylo - pad, yhi + pad,
                          excluded=~hl_allowed(ion_db, s_db))
            n_id = len(set(zip(ids["ion"], ids["harmonic"])))
            info += (f"\n{len(sims)} simulated lines from this reference "
                     f"({n_id} identified: dark dashed; no peak: light dashed)")
            info += "".join(f"\n⚠ {w}" for w in ref_warn)
        except Exception as exc:
            info += f"\nCould not calculate simulated lines: {exc}"
    else:
        info += "\nNo simulated lines: the reference was not found in this spectrum"

    pk = db_query(path, "SELECT freq, power, snr, kind FROM peaks WHERE spectrum_id = ?",
                  (int(row["spectrum_id"]),))
    pk = pk[pk["kind"] == "max"]
    if len(pk):
        fig.add_trace(go.Scatter(x=pk["freq"], y=pk["power"], mode="markers", name="Detected maxima",
                                 marker=dict(symbol="triangle-up", color=ACCENT, size=8,
                                             line=dict(color="black", width=0.6)),
                                 hovertext=[f"f = {f:.1f} Hz<br>P = {p:.4g}<br>SNR = {s:.1f}"
                                            for f, p, s in zip(pk["freq"], pk["power"], pk["snr"])],
                                 hoverinfo="text"))
    if pd.notna(pred):
        fig.add_vrect(x0=pred - window, x1=pred + window, fillcolor=ACCENT_PALE, opacity=0.8,
                      line_width=0, layer="below")
        hmin = float(st.get("ref_min_height") or 0)
        if hmin > 0:
            fig.add_shape(type="line", x0=pred - window, x1=pred + window, y0=hmin, y1=hmin,
                          line=dict(color=ACCENT, width=1, dash="dot"))
        fig.add_vline(x=pred, line=dict(color="#999999", width=1, dash="dash"),
                      annotation_text="predicted", annotation_font_color="#999999",
                      annotation_position="top left")
    if found is not None:
        fig.add_vline(x=found, line=dict(color=ACCENT, width=2),
                      annotation_text="found", annotation_font_color=ACCENT)

    vm = (ff >= centre - w) & (ff <= centre + w)
    fig.update_layout(template="plotly_white", margin=dict(l=55, r=10, t=10, b=40),
                      xaxis_title="Frequency (Hz)", yaxis_title="Power (a.u.)",
                      dragmode="pan", hovermode="x", font=dict(size=11), showlegend=False)
    fig.update_xaxes(range=[centre - w, centre + w], autorange=False)
    if vm.any():
        lo, hi = float(pp[vm].min()), float(pp[vm].max())
        fig.update_yaxes(range=[lo - 0.05 * (hi - lo or 1), hi + 0.12 * (hi - lo or 1)])
    return fig, info
