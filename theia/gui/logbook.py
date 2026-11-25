"""
"Save to Log Book": send what a Theia tab shows to the results log book.

Buttons in the Viewer tab (plot heading) and the Scan Results tab (snippet
controls, after "Send to PID Plot") open one popout: the ion (tab of the log
book; existing tabs offered, a new name makes a new tab), a title, the values
of what is shown (editable, ticked = sent), the plot as it is on screen
(snapshot taken in the browser when the popout opens) and notes.  Save adds
an entry with core/results_logbook.py (the log book page,
reference/results_logbook.py, shows it within a few seconds).

Part of Theia (theia/theia.py).
"""
import os
import sqlite3
import traceback

import dash
from dash import dash_table, dcc, html, Input, Output, State, ALL, ctx

from core import results_logbook as rlb
from core.results_logbook import settings_rows
from core.scan import DEFAULTS, db_query, read_kv

from .app import app
from .common import ACCENT, BOX_BORDER, HELP_STYLE, LABEL_STYLE

MODAL_SHOWN = {"display": "flex", "position": "fixed", "inset": "0", "zIndex": "1000",
               "background": "rgba(0,0,0,0.35)", "alignItems": "center", "justifyContent": "center"}
MODAL_HIDDEN = {**MODAL_SHOWN, "display": "none"}
SOURCES = {"viewer": ("Theia Viewer", "viewer-graph"),
           "results": ("Theia Scan Results", "snippet-graph")}


def logbook_button(src, **style):
    """The "Save to Log Book" button of a tab (src: 'viewer' or 'results')."""
    return html.Button("Save to Log Book", id={"type": "lb-open", "src": src}, n_clicks=0,
                       title="Send the ion, the values and the plot shown to the results log book",
                       className="btn btn-outline-secondary btn-sm", style=style)


lb_modal = html.Div([
    html.Div([
        html.Div([
            html.Span("Save to Log Book", style={"fontWeight": "700", "fontSize": "15px"}),
            html.Button("✕", id="lb-x", className="btn btn-outline-secondary btn-sm"),
        ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center",
                  "marginBottom": "4px"}),
        html.Small(id="lb-from", style={**HELP_STYLE, "marginBottom": "8px", "display": "block"}),
        html.Div([
            html.Div([html.Label("Ion (tab)", style=LABEL_STYLE),
                      dcc.Input(id="lb-ion", type="text", list="lb-ions", autoComplete="off",
                                placeholder="e.g. 198Ir, 198mIr, m186Hf72+",
                                className="form-control form-control-sm"),
                      html.Datalist(id="lb-ions")], style={"flex": "1"}),
            html.Div([html.Label("Title", style=LABEL_STYLE),
                      dcc.Input(id="lb-title", type="text", autoComplete="off",
                                className="form-control form-control-sm")], style={"flex": "2"}),
        ], style={"display": "flex", "gap": "12px"}),
        html.Small("A new ion name makes a new tab. Charge state and harmonic are stored with the entry.",
                   style=HELP_STYLE),
        html.Label("Values to Send (ticked; edit if needed)", style={**LABEL_STYLE, "marginTop": "10px", "display": "block"}),
        dash_table.DataTable(
            id="lb-values", editable=True, row_selectable="multi", selected_rows=[], data=[],
            columns=[{"name": "Quantity", "id": "label"}, {"name": "Value", "id": "value"},
                     {"name": "± Error", "id": "error"}, {"name": "Unit", "id": "unit"}],
            style_cell={"fontFamily": "monospace", "fontSize": "12px", "padding": "3px 6px",
                        "textAlign": "left", "border": "1px solid #dddddd"},
            style_header={"backgroundColor": "#fff4ec", "fontWeight": "600"},
            style_table={"maxHeight": "240px", "overflowY": "auto"}),
        html.Button("+ Add Value", id="lb-addrow", n_clicks=0, className="btn btn-outline-secondary btn-sm",
                    style={"marginTop": "6px"}),
        html.Label("Plot (as shown now)", style={**LABEL_STYLE, "marginTop": "10px", "display": "block"}),
        html.Div([
            dcc.Checklist(id="lb-plot-on", value=["on"], options=[{"label": " Send the plot", "value": "on"}],
                          style={"fontSize": "13px", "marginRight": "12px"}),
            html.Img(id="lb-thumb", style={"height": "110px", "border": "1px solid #dddddd",
                                           "borderRadius": "3px"}),
            dcc.Input(id="lb-caption", type="text", placeholder="caption", autoComplete="off",
                      className="form-control form-control-sm", style={"flex": "1", "marginLeft": "12px"}),
        ], style={"display": "flex", "alignItems": "center"}),
        html.Label("Notes", style={**LABEL_STYLE, "marginTop": "10px", "display": "block"}),
        dcc.Textarea(id="lb-notes", value="", className="form-control form-control-sm",
                     style={"width": "100%", "minHeight": "70px"}),
        html.Div([
            html.Button("Save", id="lb-save", n_clicks=0, className="btn btn-sm",
                        style={"backgroundColor": ACCENT, "color": "white", "fontWeight": "600",
                               "minWidth": "80px", "marginRight": "8px"}),
            html.Button("Cancel", id="lb-cancel", n_clicks=0, className="btn btn-outline-secondary btn-sm",
                        style={"marginRight": "12px"}),
            html.Span(id="lb-msg", style={"fontSize": "13px"}),
        ], style={"display": "flex", "alignItems": "center", "marginTop": "10px"}),
        dcc.Store(id="lb-prefill", data=None),    # {src, source, graph, context, n}
        dcc.Store(id="lb-png", data=None),        # snapshot of the plot (data URL)
    ], className="theia-box",
       style={"border": BOX_BORDER, "borderRadius": "6px", "background": "white",
              "width": "760px", "maxWidth": "96vw", "maxHeight": "94vh", "overflowY": "auto",
              "padding": "14px 16px", "boxShadow": "0 8px 30px rgba(0,0,0,0.3)"}),
], id="lb-modal", style=MODAL_HIDDEN)


# ---------------------------------------------------------------------------
# What each tab offers
# ---------------------------------------------------------------------------

def _num(v):
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def _results_prefill(idx, rows, db, half_width, extra):
    if not rows or not db or idx is None or int(idx) >= len(rows):
        raise ValueError("No identification shown in the snippet.")
    row = rows[int(idx)]
    values = [{"label": "Harmonic h", "value": row.get("harmonic")},
              {"label": "Peak Frequency", "value": _num(row.get("peak_freq")), "unit": "Hz"},
              {"label": "Simulated Frequency", "value": _num(row.get("sim_freq")), "unit": "Hz"},
              {"label": "Δ (peak − simulated)", "value": _num(row.get("delta_hz")), "unit": "Hz"},
              {"label": "Peak Power", "value": _num(row.get("power")), "checked": False},
              {"label": "SNR (prominence / σ)", "value": _num(row.get("snr"))},
              {"label": "Peak Kind", "value": row.get("kind"), "checked": False}]
    try:
        pk = db_query(db, "SELECT width_hz, height_snr FROM peaks WHERE peak_id = ?", (int(row["peak_id"]),))
        if len(pk) and _num(pk.iloc[0]["width_hz"]) is not None:
            values.append({"label": "Width (FWHM)", "value": round(float(pk.iloc[0]["width_hz"]), 1),
                           "unit": "Hz", "checked": False})
    except Exception:
        pass
    for v in values:
        if isinstance(v.get("value"), float):
            v["value"] = round(v["value"], 3)
    # the scan's own settings (stored in the database), then this page's options
    values.append({"label": "Scan Database", "value": os.path.basename(db), "group": "settings"})
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            st = read_kv(con, "settings")
        finally:
            con.close()
        values += settings_rows(st, [k for k in DEFAULTS if k in st] + [k for k in st if k not in DEFAULTS])
    except Exception as exc:
        print(f"Results log book: could not read the scan settings: {exc}", flush=True)
    ion_f, spec_f, closest, show_ex, dark = extra
    # the ion filter is a multi-select: values "ion:<ion>" or "base:<nuclide>" (all charge states)
    if isinstance(ion_f, (list, tuple)):
        ion_f = ", ".join(str(v).split(":", 1)[-1] + (" (all charge states)" if str(v).startswith("base:") else "")
                          for v in ion_f)
    values +=[dict(r, group="settings") for r in [{"label": "Snippet Half-Width", "value": _num(half_width), "unit": "Hz"},
               {"label": "Ion Filter", "value": str(ion_f or "").strip() or "none"},
               {"label": "Spectrum Filter", "value": ", ".join(map(str, spec_f)) if spec_f else "none"},
               {"label": "Only Closest Peak for Each Ion", "value": "on" if "ion" in (closest or []) else "off"},
               {"label": "Only Closest Ion for Each Peak", "value": "on" if "peak" in (closest or []) else "off"},
               {"label": "Show Ions Excluded by T½ Cut", "value": "on" if show_ex else "off"},
               {"label": "Dark Mode", "value": "on" if dark else "off"}]]
    return {"ion": row.get("ion") or "",
            "title": f"Peak in {row.get('spectrum')}",
            "values": values,
            "caption": f"{row.get('spectrum')} – {row.get('ion')} (h={row.get('harmonic')})",
            "context": {"tool": "Theia Scan Results", "scan database": db,
                        "spectrum": row.get("spectrum"), "peak id": row.get("peak_id"),
                        "snippet half-width (Hz)": half_width}}


def _viewer_prefill(vals, zoom, zoom_khz, harm_shown, dark):
    file_path, spectrum, highlight = vals.get("file_path"), vals.get("spectrum_name"), vals.get("highlight_ions")
    ion, values = "", []
    if isinstance(zoom, dict) and zoom.get("ion"):
        ion = zoom["ion"]
        values += [{"label": "Harmonic h", "value": zoom.get("h")},
                   {"label": "Simulated Frequency", "value": round(float(zoom["f"]), 3), "unit": "Hz"}]
    elif highlight:
        ion = str(highlight).split(",")[0].strip()
    values += settings_rows(vals, VIEWER_KEYS)
    values += [dict(r, group="settings") for r in [
               {"label": "Zoom Window ±", "value": _num(zoom_khz), "unit": "kHz"},
               {"label": "Zoomed To", "value": (f"{zoom['ion']} h{zoom.get('h')}" if isinstance(zoom, dict)
                                                and zoom.get("ion") else "off")},
               {"label": "Harmonics Shown", "value": ", ".join(str(h) for h in (harm_shown or [])) or "none"},
               {"label": "Dark Mode", "value": "on" if dark else "off"}]]
    return {"ion": ion,
            "title": f"Viewer: {spectrum}" if spectrum else "Viewer",
            "values": values,
            "caption": f"{spectrum}" + (f" – {ion}" if ion else ""),
            "context": {"tool": "Theia Viewer",
                        "file": os.path.join(str(file_path or ""), str(spectrum or "")),
                        "highlighted ions": highlight or ""}}



# the Viewer's inputs (ids v-<key>); same keys as viewer.VIEWER_DEFAULTS (not
# imported: viewer.py imports this module)
VIEWER_KEYS = ["file_path", "spectrum_name", "ref_name", "ref_frequency", "ref_harmonic", "z_lims",
               "n_lims", "el1", "el2", "mq_width", "gamma_t", "harmonics", "correction_enabled",
               "corr_a", "corr_b", "corr_c", "corr_profile", "ref_search_hz", "highlight_ions"]


@app.callback(
    Output("lb-prefill", "data"),
    Output("send-pid-msg", "children", allow_duplicate=True),
    Input({"type": "lb-open", "src": ALL}, "n_clicks"),
    State("snippet-idx", "data"), State("hits-table", "derived_virtual_data"),
    State("db-select", "value"), State("snippet-width", "value"),
    State("ion-filter", "value"), State("spec-filter", "value"), State("closest-filter", "value"),
    State("snippet-show-excluded", "value"),
    *[State(f"v-{k}", "value") for k in VIEWER_KEYS],
    State("v-zoom-idx", "data"), State("v-zoom-width", "value"), State("v-harm-toggle", "value"),
    State("dark-mode", "value"),
    State("lb-prefill", "data"),
    prevent_initial_call=True,
)
def lb_open(clicks, idx, rows, db, half_width, ion_f, spec_f, closest, show_ex, *rest):
    n = len(VIEWER_KEYS)
    v_vals, (v_zoom, zoom_khz, harm_shown, dark, old) = rest[:n], rest[n:]
    t = ctx.triggered_id
    if not isinstance(t, dict) or not any(clicks or []):
        return dash.no_update, dash.no_update
    src = t["src"]
    try:
        if src == "results":
            pre = _results_prefill(idx, rows, db, half_width, (ion_f, spec_f, closest, show_ex, dark))
        else:
            pre = _viewer_prefill(dict(zip(VIEWER_KEYS, v_vals)), v_zoom, zoom_khz, harm_shown, dark)
    except Exception as exc:
        # only the Scan Results tab can fail (nothing in the snippet): say so beside its buttons
        return dash.no_update, f"✗ {exc}"
    source, graph = SOURCES[src]
    return (dict(pre, src=src, source=source, graph=graph, n=((old or {}).get("n") or 0) + 1),
            dash.no_update)


@app.callback(
    Output("lb-modal", "style"), Output("lb-from", "children"),
    Output("lb-ion", "value"), Output("lb-ions", "children"), Output("lb-title", "value"),
    Output("lb-values", "data"), Output("lb-values", "selected_rows"),
    Output("lb-caption", "value"), Output("lb-plot-on", "value"), Output("lb-notes", "value"),
    Output("lb-msg", "children", allow_duplicate=True), Output("lb-save", "disabled"),
    Output("lb-cancel", "children"),
    Input("lb-prefill", "data"),
    prevent_initial_call=True,
)
def lb_fill(pre):
    if not pre:
        return (dash.no_update,) * 13
    try:
        names = rlb.ion_names()
    except Exception:
        names = []
    rows = [{"label": v.get("label", ""), "value": "" if v.get("value") is None else str(v["value"]),
             "error": "" if v.get("error") is None else str(v["error"]), "unit": v.get("unit", ""),
             "group": v.get("group", "")}
            for v in pre.get("values", [])]
    sel = [i for i, v in enumerate(pre.get("values", [])) if v.get("checked", True)]
    return (MODAL_SHOWN, f"From {pre['source']}", pre.get("ion", ""),
            [html.Option(value=k) for k in names], pre.get("title", ""), rows, sel,
            pre.get("caption", ""), ["on"], "", "", False, "Cancel")


# snapshot of the plot as it is on screen (light colours, also in dark mode)
app.clientside_callback(
    """
    async function(pre) {
        if (!pre || !pre.graph) { return [null, '']; }
        var box = document.getElementById(pre.graph);
        var gd = box && (box.classList.contains('js-plotly-plot') ? box
                         : box.querySelector('.js-plotly-plot'));
        if (!gd || !gd.data || !window.Plotly) { return [null, '']; }
        try {
            var w = gd.offsetWidth || 900, h = gd.offsetHeight || 500;
            var url = await Plotly.toImage(gd, {format: 'png', width: w, height: h, scale: 2});
            return [url, url];
        } catch (e) { return [null, '']; }
    }
    """,
    Output("lb-png", "data"), Output("lb-thumb", "src"),
    Input("lb-prefill", "data"),
    prevent_initial_call=True,
)


@app.callback(
    Output("lb-values", "data", allow_duplicate=True),
    Output("lb-values", "selected_rows", allow_duplicate=True),
    Input("lb-addrow", "n_clicks"),
    State("lb-values", "data"), State("lb-values", "selected_rows"),
    prevent_initial_call=True,
)
def lb_add_row(_n, rows, sel):
    rows = list(rows or [])
    rows.append({"label": "", "value": "", "error": "", "unit": ""})
    return rows, list(sel or []) + [len(rows) - 1]


@app.callback(
    Output("lb-modal", "style", allow_duplicate=True),
    Input("lb-cancel", "n_clicks"), Input("lb-x", "n_clicks"),
    prevent_initial_call=True,
)
def lb_close(_a, _b):
    return MODAL_HIDDEN


@app.callback(
    Output("lb-msg", "children", allow_duplicate=True),
    Output("lb-save", "disabled", allow_duplicate=True),
    Output("lb-cancel", "children", allow_duplicate=True),
    Input("lb-save", "n_clicks"),
    State("lb-prefill", "data"), State("lb-ion", "value"), State("lb-title", "value"),
    State("lb-values", "data"), State("lb-values", "selected_rows"),
    State("lb-plot-on", "value"), State("lb-png", "data"), State("lb-caption", "value"),
    State("lb-notes", "value"),
    prevent_initial_call=True,
)
def lb_save(_n, pre, ion, title, rows, sel, plot_on, png, caption, notes):
    if not pre:
        return dash.no_update, dash.no_update, dash.no_update
    if not (ion or "").strip():
        return "✗ Choose or type an ion.", False, dash.no_update
    sel = set(sel or [])
    values = [r for i, r in enumerate(rows or []) if i in sel]
    plots = [{"png": png, "caption": caption or ""}] if (plot_on and png) else []
    try:
        _, key = rlb.add_entry(ion, pre.get("source", "Theia"), title=title or "", notes=notes or "",
                               values=values, plots=plots, context=pre.get("context") or {})
    except Exception as exc:
        traceback.print_exc()
        return f"✗ {exc}", False, dash.no_update
    print(f"Results log book: saved to {key} from {pre.get('source')} ({rlb.logbook_path()})", flush=True)
    return f"✓ Saved to the log book: {key}", True, "Close"
