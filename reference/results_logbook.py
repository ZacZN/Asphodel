#!/usr/bin/env python3
"""
results_logbook.py  –  results log book: what you found out about each ion

A Dash page (http://127.0.0.1:8058/) with one vertical tab per ion (nuclide
+ energy state: 198Ir and 198mIr are separate tabs) down the left, and that
ion's entries on the right, newest first.  Started from the home page
(Reference section), which opens it in a new tab; it stops a few seconds after
its tab is closed, or with Stop on the home page.

Entries are sent from the other tools with "Save to log book" (decay-fit
window, spectrogram page, Theia Viewer and Scan Results): a popout where you
choose the ion and tick the values and the plot to send.  Each entry has a
title, the values (value ± error unit), plots, notes and where it came from;
Edit changes any of these (or moves it to another ion), Delete removes it
and its plot files.  The page notices entries saved from the other tools
within a few seconds.

Storage: core/results_logbook.py (outputs/logbook/ of the experiment).
Open at an ion: http://127.0.0.1:8058/?ion=198mIr

Usage
-----
    python reference/results_logbook.py      (normally started from the home page)
"""

import logging
import os
import socket
import sys
import threading
from urllib.parse import parse_qs

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import browser, icon, logcapture, logpanel, paths, tabwatch
from core import results_logbook as rlb

# output also to data/logs/ and the page's Log panel (core/logcapture.py)
if __name__ == "__main__":
    logcapture.start("results_logbook")

import dash
import flask
from dash import dcc, html, dash_table, Input, Output, State, ALL, ctx

PORT = 8058
ACCENT, PALE, SIDE, LINE = "#f57d15", "#fff4ec", "#f9f9f9", "#dddddd"

CSS = f"""
*{{box-sizing:border-box}}
body{{margin:0;font-family:-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;color:#222;
  background:#fff;font-size:14px;overflow:hidden}}
.page{{height:100vh;display:flex;flex-direction:column}}
header{{flex:none;height:52px;display:flex;align-items:center;gap:14px;padding:0 18px;
  border-bottom:3px solid {ACCENT}}}
header .logo{{font-weight:800;font-size:18px;letter-spacing:.5px}} header .logo span{{color:{ACCENT}}}
header .exp{{color:#777;font-family:monospace;font-size:12px}}
.main{{flex:1 1 auto;min-height:0;display:flex}}
/* vertical ion tabs */
.side{{flex:none;width:210px;border-right:1px solid {LINE};background:{SIDE};display:flex;flex-direction:column}}
.side .find{{padding:10px 10px 6px}}
.side .find input{{width:100%;padding:5px 7px;border:1px solid {LINE};border-radius:4px;font-size:13px}}
.side .list{{flex:1 1 auto;overflow-y:auto;padding:0 0 10px}}
.side .empty{{color:#777;font-size:12px;padding:8px 12px}}
.tab{{display:flex;width:100%;align-items:center;gap:6px;text-align:left;border:none;
  border-left:4px solid transparent;background:none;padding:7px 12px;cursor:pointer;font-size:14px;color:#222}}
.tab:hover{{background:{PALE}}}
.tab.on{{background:#fff;border-left-color:{ACCENT};font-weight:700}}
.tab .nm{{font-family:monospace;font-size:14px}}
.tab .ct{{margin-left:auto;font-size:11px;color:#777;font-weight:400;background:#fff;
  border:1px solid {LINE};border-radius:9px;padding:0 6px}}
.tab .z{{font-size:11px;color:#999;font-weight:400}}
/* entries */
.content{{flex:1 1 auto;min-width:0;overflow-y:auto;padding:16px 22px 40px}}
.ionhead{{display:flex;align-items:baseline;gap:12px;border-bottom:2px solid {ACCENT};
  padding-bottom:6px;margin-bottom:12px}}
.ionhead h2{{margin:0;font-size:22px;font-family:monospace}}
.ionhead .sub{{color:#777;font-size:13px}}
.ionhead .r{{margin-left:auto;display:flex;gap:8px;align-items:center}}
.hint{{color:#777;font-size:13px}}
.entry{{border:1px solid {LINE};border-left:5px solid {ACCENT};border-radius:5px;padding:10px 14px;
  margin:0 0 14px;background:#fff}}
.entry .top{{display:flex;align-items:baseline;gap:10px}}
.entry .ttl{{font-weight:700;font-size:15px}}
.entry .meta{{color:#777;font-size:12px}}
.entry .btns{{margin-left:auto;display:flex;gap:6px;flex:none}}
.entry table{{border-collapse:collapse;margin:8px 0 4px;font-size:13px}}
.entry th{{background:{PALE};text-align:left;padding:3px 10px;border-bottom:1px solid {LINE};
  font-weight:600;color:#555}}
.entry td{{padding:3px 10px;border-bottom:1px solid {LINE}}}
.entry td.v{{font-family:monospace}}
.entry .plots{{display:flex;flex-wrap:wrap;gap:12px;margin:8px 0}}
.entry figure{{margin:0;max-width:100%}}
.entry figure img{{max-width:min(720px,100%);max-height:460px;border:1px solid {LINE};border-radius:3px;
  cursor:zoom-in;display:block}}
.entry figcaption{{font-size:12px;color:#555;margin-top:3px}}
.entry .notes{{white-space:pre-wrap;margin:6px 0 2px;padding:6px 8px;background:{SIDE};border-radius:4px}}
.entry details{{margin-top:6px;font-size:12px;color:#555}}
.entry details summary{{cursor:pointer;color:#777}}
.entry details div{{font-family:monospace;word-break:break-all;margin:2px 0 0 12px}}
.btn{{border:1px solid {LINE};background:#fff;border-radius:4px;padding:4px 10px;cursor:pointer;font-size:12px}}
.btn:hover{{border-color:{ACCENT};color:{ACCENT}}}
.btn.primary{{background:{ACCENT};border-color:{ACCENT};color:#fff}} .btn.primary:hover{{color:#fff;opacity:.9}}
.btn.danger:hover{{border-color:#c62828;color:#c62828}}
.msg{{color:#8a4a10;font-size:12px}}
/* edit popout */
.modal-bg{{position:fixed;inset:0;background:rgba(0,0,0,.35);z-index:1000;display:flex;
  align-items:flex-start;justify-content:center;overflow-y:auto;padding:40px 10px}}
.modal{{background:#fff;border-radius:6px;border-top:4px solid {ACCENT};width:min(760px,96vw);
  padding:14px 18px;box-shadow:0 8px 30px rgba(0,0,0,.25)}}
.modal h3{{margin:0 0 8px}}
.modal label.f{{display:block;font-size:11px;color:#777;margin:10px 0 3px;text-transform:uppercase;letter-spacing:.5px}}
.modal input[type=text]{{width:100%;padding:5px 7px;border:1px solid {LINE};border-radius:4px;font-size:13px}}
.modal textarea{{width:100%;min-height:90px;padding:5px 7px;border:1px solid {LINE};border-radius:4px;
  font-size:13px;font-family:inherit}}
.modal .chk label{{display:flex;align-items:center;gap:8px;margin:4px 0}}
.modal .chk input{{accent-color:{ACCENT}}}
.modal .chk img{{height:70px;border:1px solid {LINE}}}
.modal .foot{{display:flex;gap:8px;align-items:center;margin-top:14px}}
"""

app = dash.Dash(__name__, title="Results log book", suppress_callback_exceptions=True)
app.index_string = app.index_string.replace("</head>", f"<style>{CSS}</style></head>")
app.index_string = app.index_string.replace(
    "{%scripts%}", "{%scripts%}\n<script>" + tabwatch.watch_js("Results log book") + "</script>")
icon.use_in_dash(app, "results_logbook")      # core/icons/results_logbook.png, else asphodel.png

PLOT_URL = "/logbook-plot/"


@app.server.route(PLOT_URL + "<path:rel>")
def _plot(rel):
    p = rlb.plot_path(rel)
    if not p or not os.path.isfile(p):
        flask.abort(404)
    resp = flask.send_file(p, mimetype="image/png")
    resp.headers["Cache-Control"] = "no-cache"
    return resp


# ===========================================================================
# Page
# ===========================================================================

def serve_layout():
    return html.Div([
        dcc.Location(id="url", refresh=False),
        html.Header([html.Div(["Results ", html.Span("log book")], className="logo"),
                     html.Div(f"experiment {paths.experiment()} · {rlb.logbook_path()}",
                              className="exp"),
                     logpanel.dash_button(className="btn", style={"marginLeft": "auto"})]),
        html.Div([
            html.Div([
                html.Div(dcc.Input(id="find", type="text", placeholder="Find ion (e.g. 198Ir, Hf)",
                                   debounce=False, autoComplete="off"), className="find"),
                html.Div(id="tabs", className="list"),
            ], className="side"),
            html.Div(id="content", className="content"),
        ], className="main"),
        edit_modal(),
        dcc.ConfirmDialog(id="del-confirm", message=""),
        dcc.Store(id="sel", data=None),           # ion key shown
        dcc.Store(id="ver", data=0),              # bumped after every change -> redraw
        dcc.Store(id="mtime", data=rlb.mtime()),  # file time last drawn
        dcc.Store(id="editing", data=None),       # entry id in the edit popout
        dcc.Store(id="deleting", data=None),      # entry id waiting for the delete confirmation
        dcc.Interval(id="tick", interval=3000),   # entries saved from the other tools
    ], className="page")


def edit_modal():
    lbl = lambda t: html.Label(t, className="f")
    return html.Div(html.Div([
        html.H3("Edit entry"),
        html.Div([
            html.Div([lbl("Ion (tab)"),
                      dcc.Input(id="ed-ion", type="text", list="ed-ions", autoComplete="off"),
                      html.Datalist(id="ed-ions")], style={"flex": "1"}),
            html.Div([lbl("Title"), dcc.Input(id="ed-title", type="text", autoComplete="off")],
                     style={"flex": "2"}),
        ], style={"display": "flex", "gap": "12px"}),
        lbl("Values"),
        dash_table.DataTable(
            id="ed-values", editable=True, row_deletable=True, data=[],
            columns=[{"name": "Quantity", "id": "label"}, {"name": "Value", "id": "value"},
                     {"name": "± Error", "id": "error"}, {"name": "Unit", "id": "unit"}],
            style_cell={"fontFamily": "monospace", "fontSize": "13px", "padding": "3px 6px",
                        "textAlign": "left", "border": f"1px solid {LINE}"},
            style_header={"backgroundColor": PALE, "fontWeight": "600", "fontFamily": "inherit"},
            style_table={"maxHeight": "260px", "overflowY": "auto"}),
        html.Button("+ Add value", id="ed-addrow", className="btn", n_clicks=0,
                    style={"marginTop": "6px"}),
        html.Div([lbl("Plots (untick to delete)"),
                  dcc.Checklist(id="ed-plots", options=[], value=[], className="chk")],
                 id="ed-plots-box"),
        lbl("Notes"),
        dcc.Textarea(id="ed-notes", value=""),
        html.Div([html.Button("Save", id="ed-save", className="btn primary", n_clicks=0),
                  html.Button("Cancel", id="ed-cancel", className="btn", n_clicks=0),
                  html.Span(id="ed-msg", className="msg")], className="foot"),
    ], className="modal"), id="ed-bg", className="modal-bg", style={"display": "none"})


app.layout = logpanel.dash_layout(serve_layout, "Results log book")
logpanel.register_dash(app)


# ===========================================================================
# Drawing
# ===========================================================================

def _when(iso):
    return (iso or "").replace("T", " ")[:16]


def _tab(key, ion, n, on):
    return html.Button([html.Span(key, className="nm"),
                        html.Span(f"Z={ion.get('Z')}", className="z"),
                        html.Span(str(n), className="ct")],
                       id={"type": "tab", "k": key}, n_clicks=0,
                       className="tab on" if on else "tab",
                       title=f"{key}: Z = {ion.get('Z')}, N = {ion.get('N')}, "
                             f"{'ground state' if ion.get('state', 'g') == 'g' else 'isomer ' + ion['state']}"
                             f" · {n} entr{'y' if n == 1 else 'ies'}")


def _matches(key, ion, q):
    q = (q or "").strip()
    if not q:
        return True
    try:
        p = rlb.parse_ion(q)
        return p["key"] == key or (p["state"] == "g" and p["A"] == ion.get("A")
                                   and p["sym"] == ion.get("sym"))
    except ValueError:
        ql = q.lower()
        return ql in key.lower() or ql == str(ion.get("sym", "")).lower()


def _value_rows(values):
    rows = []
    for v in values or []:
        rows.append(html.Tr([html.Td(v.get("label", "")), html.Td(rlb.fmt_value(v), className="v")]))
    return html.Table([html.Thead(html.Tr([html.Th("Quantity"), html.Th("Value")])),
                       html.Tbody(rows)]) if rows else None


def _entry_card(e):
    meta = f"{e.get('source') or 'manual'} · {_when(e.get('created'))}"
    if e.get("updated") and e.get("updated") != e.get("created"):
        meta += f" · edited {_when(e['updated'])}"
    plots = [html.Figure([html.A(html.Img(src=PLOT_URL + p["file"]), href=PLOT_URL + p["file"],
                                 target="_blank", title="Open full size"),
                          html.Figcaption(p.get("caption") or "")])
             for p in e.get("plots", []) if p.get("file")]
    ctxt = e.get("context") or {}
    return html.Div([
        html.Div([html.Span(e.get("title") or "(no title)", className="ttl"),
                  html.Span(meta, className="meta"),
                  html.Div([html.Button("Edit", id={"type": "edit", "id": e["id"]}, n_clicks=0,
                                        className="btn"),
                            html.Button("Delete", id={"type": "del", "id": e["id"]}, n_clicks=0,
                                        className="btn danger")], className="btns")],
                 className="top"),
        _value_rows(e.get("values")),
        html.Div(plots, className="plots") if plots else None,
        html.Div(e["notes"], className="notes") if e.get("notes") else None,
        html.Details([html.Summary("Where it came from"),
                      *[html.Div(f"{k}: {v}") for k, v in ctxt.items()]]) if ctxt else None,
    ], className="entry")


@app.callback(Output("sel", "data"),
              Input({"type": "tab", "k": ALL}, "n_clicks"), Input("url", "search"),
              State("sel", "data"))
def _select(clicks, search, sel):
    t = ctx.triggered_id
    if isinstance(t, dict) and t.get("type") == "tab":
        if any(clicks or []):
            return t["k"]
        return dash.no_update
    keys = rlb.ion_names()
    want = parse_qs((search or "").lstrip("?")).get("ion", [None])[0]
    if want:
        try:
            k = rlb.parse_ion(want)["key"]
            if k in keys:
                return k
        except ValueError:
            pass
    if sel in keys:
        return dash.no_update
    return keys[0] if keys else None


@app.callback(Output("tabs", "children"),
              Input("ver", "data"), Input("sel", "data"), Input("find", "value"))
def _draw_tabs(_ver, sel, q):
    lst = rlb.ions()
    if not lst:
        return html.Div("No ions yet. Use “Save to log book” in the decay-fit window, the "
                        "spectrogram page or Theia.", className="empty")
    shown = [_tab(k, ion, n, k == sel) for k, ion, n in lst if _matches(k, ion, q)]
    return shown or html.Div("No ion matches.", className="empty")


@app.callback(Output("content", "children"), Input("ver", "data"), Input("sel", "data"))
def _draw_content(_ver, sel):
    d = rlb.load()
    if not sel or sel not in d["ions"]:
        return html.Div([html.H2("Results log book"),
                         html.P("One tab per ion (nuclide + energy state). Entries are added with "
                                "“Save to log book” in the decay-fit window, on the "
                                "spectrogram page and in Theia (Viewer and Scan Results).",
                                className="hint")])
    ion = d["ions"][sel]
    entries = sorted([e for e in d["entries"] if e.get("ion") == sel],
                     key=lambda e: e.get("created", ""), reverse=True)
    state = "ground state" if ion.get("state", "g") == "g" else f"isomer ({ion['state']})"
    right = [html.Span(f"{len(entries)} entr{'y' if len(entries) == 1 else 'ies'}", className="hint")]
    if not entries:
        right.append(html.Button("Remove empty tab", id={"type": "rmtab", "k": sel}, n_clicks=0,
                                 className="btn danger"))
    return html.Div([
        html.Div([html.H2(sel), html.Span(f"Z = {ion.get('Z')}, N = {ion.get('N')}, {state}",
                                          className="sub"),
                  html.Div(right, className="r")], className="ionhead"),
        html.Div(id="rm-msg", className="msg"),
        *[_entry_card(e) for e in entries],
        html.P("No entries.", className="hint") if not entries else None,
    ])


@app.callback(Output("ver", "data", allow_duplicate=True), Output("mtime", "data"),
              Input("tick", "n_intervals"), State("mtime", "data"), State("ver", "data"),
              prevent_initial_call=True)
def _watch(_n, seen, ver):
    m = rlb.mtime()
    if m == seen:
        return dash.no_update, dash.no_update
    return (ver or 0) + 1, m


@app.callback(Output("ver", "data", allow_duplicate=True), Output("sel", "data", allow_duplicate=True),
              Input({"type": "rmtab", "k": ALL}, "n_clicks"), State("ver", "data"),
              prevent_initial_call=True)
def _remove_tab(clicks, ver):
    t = ctx.triggered_id
    if not isinstance(t, dict) or not any(clicks or []):
        return dash.no_update, dash.no_update
    try:
        rlb.delete_ion(t["k"])
    except Exception as exc:
        print(f"Results log book: {exc}")
        return dash.no_update, dash.no_update
    keys = rlb.ion_names()
    return (ver or 0) + 1, (keys[0] if keys else None)


# --------------------------------------------------------------------------- delete

@app.callback(Output("del-confirm", "displayed"), Output("del-confirm", "message"),
              Output("deleting", "data"),
              Input({"type": "del", "id": ALL}, "n_clicks"), prevent_initial_call=True)
def _ask_delete(clicks):
    t = ctx.triggered_id
    if not isinstance(t, dict) or not any(clicks or []):
        return dash.no_update, dash.no_update, dash.no_update
    e = rlb.get_entry(t["id"])
    if e is None:
        return dash.no_update, dash.no_update, dash.no_update
    n = len(e.get("plots", []))
    return True, (f"Delete the entry “{e.get('title') or '(no title)'}” of {e.get('ion')}"
                  + (f" and its {n} plot{'s' if n > 1 else ''}" if n else "")
                  + "? This can't be undone."), t["id"]


@app.callback(Output("ver", "data", allow_duplicate=True),
              Input("del-confirm", "submit_n_clicks"), State("deleting", "data"),
              State("ver", "data"), prevent_initial_call=True)
def _delete(_n, eid, ver):
    if not eid:
        return dash.no_update
    try:
        rlb.delete_entry(eid)
    except Exception as exc:
        print(f"Results log book: could not delete the entry: {exc}")
        return dash.no_update
    print(f"Results log book: deleted entry {eid}")
    return (ver or 0) + 1


# --------------------------------------------------------------------------- edit

@app.callback(Output("ed-bg", "style"), Output("editing", "data"),
              Output("ed-ion", "value"), Output("ed-ions", "children"), Output("ed-title", "value"),
              Output("ed-values", "data"), Output("ed-plots", "options"), Output("ed-plots", "value"),
              Output("ed-plots-box", "style"), Output("ed-notes", "value"),
              Output("ed-msg", "children", allow_duplicate=True),
              Input({"type": "edit", "id": ALL}, "n_clicks"), Input("ed-cancel", "n_clicks"),
              prevent_initial_call=True)
def _open_edit(clicks, _cancel):
    hide = ({"display": "none"}, None) + (dash.no_update,) * 8 + ("",)
    t = ctx.triggered_id
    if t == "ed-cancel":
        return hide
    if not isinstance(t, dict) or not any(clicks or []):
        return (dash.no_update,) * 11
    e = rlb.get_entry(t["id"])
    if e is None:
        return hide
    rows = [{"label": v.get("label", ""), "value": "" if v.get("value") is None else str(v["value"]),
             "error": "" if v.get("error") is None else str(v["error"]), "unit": v.get("unit", "")}
            for v in e.get("values", [])]
    opts = [{"label": html.Span([html.Img(src=PLOT_URL + p["file"]),
                                 html.Span(p.get("caption") or os.path.basename(p["file"]))]),
             "value": p["file"]} for p in e.get("plots", [])]
    return ({"display": "flex"}, e["id"], e.get("ion", ""),
            [html.Option(value=k) for k in rlb.ion_names()], e.get("title", ""), rows,
            opts, [o["value"] for o in opts], {"display": "block" if opts else "none"},
            e.get("notes", ""), "")


@app.callback(Output("ed-values", "data", allow_duplicate=True),
              Input("ed-addrow", "n_clicks"), State("ed-values", "data"), prevent_initial_call=True)
def _add_row(_n, rows):
    return (rows or []) + [{"label": "", "value": "", "error": "", "unit": ""}]


@app.callback(Output("ed-bg", "style", allow_duplicate=True),
              Output("ed-msg", "children", allow_duplicate=True),
              Output("ver", "data", allow_duplicate=True), Output("sel", "data", allow_duplicate=True),
              Input("ed-save", "n_clicks"),
              State("editing", "data"), State("ed-ion", "value"), State("ed-title", "value"),
              State("ed-values", "data"), State("ed-plots", "value"), State("ed-notes", "value"),
              State("ver", "data"), prevent_initial_call=True)
def _save_edit(_n, eid, ion, title, rows, keep, notes, ver):
    if not eid:
        return dash.no_update, "", dash.no_update, dash.no_update
    try:
        key = rlb.parse_ion(ion)["key"]
        rlb.update_entry(eid, title=title or "", notes=notes or "", values=rows or [],
                         ion=key, keep_plots=keep or [])
    except Exception as exc:
        return dash.no_update, f"✗ {exc}", dash.no_update, dash.no_update
    return {"display": "none"}, "", (ver or 0) + 1, key


# ===========================================================================

_TABS = tabwatch.TabTracker(on_last_closed=lambda: os._exit(0), name="Results log book")
tabwatch.install_flask(app.server, _TABS)


def _port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        return s.connect_ex(("127.0.0.1", port)) == 0


if __name__ == "__main__":
    url = f"http://127.0.0.1:{PORT}/"
    if _port_open(PORT):                # already running: just show it
        browser.open_url(url)
        sys.exit(0)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    print(f"Results log book at {url}")
    threading.Timer(1.0, lambda: browser.open_url(url)).start()
    app.run(debug=False, host="127.0.0.1", port=PORT)
