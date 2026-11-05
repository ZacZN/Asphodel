"""
Folder browser pop-up used by the Browse… buttons.

Part of Theia (theia/theia.py); split out of theia.py in Nov 2026.
"""

import json
import os

import dash
from dash import ALL, dcc, html, Input, Output, State

from core import paths
from core.scan import natural_key

from .common import ACCENT, BOX_BORDER
from .app import app


# ---------------------------------------------------------------------------
# Folder browser (popup inside the page) used by the "Browse…" buttons
# ---------------------------------------------------------------------------
# Browse buttons and the field each one fills in
BROWSE_TARGETS = {"btn-browse-spectra_dir": "s-spectra_dir",
                  "btn-browse-v-file_path": "v-file_path",
                  "btn-browse-fsrc-dir": "fsrc-dir"}


def with_browse(field_div, btn_id):
    """Put a 'Browse…' button to the right of the input in an inp()/v_input() block."""
    kids = list(field_div.children)
    box = kids[1]
    box.style = {**(box.style or {}), "flex": "1", "minWidth": "0", "width": "auto"}
    kids[1] = html.Div([
        box,
        html.Button("Browse…", id=btn_id, className="btn btn-outline-secondary btn-sm",
                    title="Choose the folder in a folder browser",
                    style={"marginLeft": "6px", "whiteSpace": "nowrap"}),
    ], style={"display": "flex", "alignItems": "center"})
    field_div.children = kids
    return field_div


def list_npz(directory):
    """Spectrum files (*.npz) in a folder, in natural order."""
    try:
        names = [n for n in os.listdir(directory)
                 if n.lower().endswith(".npz") and os.path.isfile(os.path.join(directory, n))]
    except OSError:
        return []
    return sorted(names, key=natural_key)


def _fb_start_dir(path):
    """Closest existing folder to `path` (else the experiment's data folder)."""
    p = os.path.expanduser((path or "").strip())
    while p and not os.path.isdir(p):
        parent = os.path.dirname(p.rstrip("/"))
        if parent == p:
            break
        p = parent
    if p and os.path.isdir(p):
        return os.path.abspath(p)
    return paths.experiment_dir() if os.path.isdir(paths.experiment_dir()) else os.path.expanduser("~")


folder_browser = html.Div([
    html.Div([
        html.Div("Choose a folder", style={"fontWeight": "700", "fontSize": "15px",
                                           "marginBottom": "8px"}),
        html.Div([
            html.Button("⬆ Up", id="fb-up", className="btn btn-outline-secondary btn-sm",
                        title="Parent folder", style={"marginRight": "6px", "whiteSpace": "nowrap"}),
            html.Button("⌂", id="fb-home", className="btn btn-outline-secondary btn-sm",
                        title="Home folder", style={"marginRight": "6px"}),
            dcc.Input(id="fb-path-input", type="text", debounce=True,
                      className="form-control form-control-sm",
                      style={"flex": "1", "fontSize": "13px", "fontFamily": "monospace"}),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "6px"}),
        html.Div(id="fb-info", style={"fontSize": "12px", "color": "#666", "marginBottom": "6px"}),
        # the experiment's connected data folders (home page → Folders), one click away
        html.Div(id="fb-conn", style={"display": "flex", "flexWrap": "wrap", "gap": "6px",
                                      "marginBottom": "6px"}),
        html.Div(id="fb-list", style={"height": "340px", "overflowY": "auto",
                                      "border": "1px solid #dddddd", "borderRadius": "4px",
                                      "padding": "4px"}),
        html.Div([
            html.Button("Cancel", id="fb-cancel", className="btn btn-outline-secondary btn-sm",
                        style={"marginRight": "8px"}),
            html.Button("Select this folder", id="fb-select", className="btn btn-sm",
                        style={"backgroundColor": ACCENT, "color": "white", "fontWeight": "600"}),
        ], style={"display": "flex", "justifyContent": "flex-end", "marginTop": "10px"}),
    ], className="theia-box",
       style={"border": BOX_BORDER, "borderRadius": "6px", "background": "white", "width": "640px", "maxWidth": "92vw",
              "padding": "14px 16px", "boxShadow": "0 8px 30px rgba(0,0,0,0.3)"}),
    dcc.Store(id="fb-path", data=None),
    dcc.Store(id="fb-target", data=None),
], id="fb-modal",
   style={"display": "none", "position": "fixed", "inset": "0", "zIndex": "1000",
          "background": "rgba(0,0,0,0.35)", "alignItems": "center", "justifyContent": "center"})

FB_SHOWN = {"display": "flex", "position": "fixed", "inset": "0", "zIndex": "1000",
            "background": "rgba(0,0,0,0.35)", "alignItems": "center", "justifyContent": "center"}
FB_HIDDEN = {**FB_SHOWN, "display": "none"}


# ===========================================================================
# Callbacks — folder browser ("Browse…" next to the path boxes)
# ===========================================================================

def _fb_listing(path):
    """(list children, info text) for the folder browser."""
    try:
        entries = sorted((e for e in os.scandir(path)
                          if e.is_dir() and not e.name.startswith(".")),
                         key=lambda e: natural_key(e.name))
    except OSError as exc:
        return [html.Div(f"Cannot read this folder: {exc}", style={"color": "#c0392b",
                                                                 "fontSize": "13px"})], ""
    n_npz = len(list_npz(path))
    rows = [html.Div("📁  " + e.name, id={"type": "fb-dir", "index": e.name}, n_clicks=0,
                     className="theia-fb-row",
                     style={"cursor": "pointer", "padding": "3px 8px", "fontSize": "13px",
                            "borderRadius": "3px"})
            for e in entries]
    if not rows:
        rows = [html.Div("(no sub-folders)", style={"color": "#888", "fontSize": "13px",
                                                    "padding": "3px 8px"})]
    info = f"{len(entries)} sub-folder(s)  ·  {n_npz} .npz spectrum file(s) in this folder"
    return rows, info


@app.callback(
    Output("fb-modal", "style"),
    Output("fb-path", "data"),
    Output("fb-target", "data"),
    Output("fb-list", "children"),
    Output("fb-path-input", "value"),
    Output("fb-info", "children"),
    Output("fb-conn", "children"),
    *[Output(f, "value", allow_duplicate=True) for f in BROWSE_TARGETS.values()],
    *[Input(b, "n_clicks") for b in BROWSE_TARGETS],
    Input({"type": "fb-dir", "index": ALL}, "n_clicks"),
    Input({"type": "fb-conn", "index": ALL}, "n_clicks"),
    Input("fb-up", "n_clicks"),
    Input("fb-home", "n_clicks"),
    Input("fb-path-input", "value"),
    Input("fb-select", "n_clicks"),
    Input("fb-cancel", "n_clicks"),
    State("fb-path", "data"),
    State("fb-target", "data"),
    *[State(f, "value") for f in BROWSE_TARGETS.values()],
    prevent_initial_call=True,
)
def folder_browser_cb(*args):
    """Open the browser at the field's current folder, navigate, and on
    'Select this folder' write the folder into the field that opened it."""
    fields = list(BROWSE_TARGETS.values())
    nf = len(fields)
    path, target = args[-nf - 2:-nf]
    current = dict(zip(fields, args[-nf:]))
    nu = dash.no_update
    keep = (nu,) * nf
    trig = dash.callback_context.triggered[0] if dash.callback_context.triggered else None
    if trig is None:
        return (nu,) * 7 + keep
    pid, val = trig["prop_id"], trig["value"]

    def show(p, tgt):
        rows, info = _fb_listing(p)
        return (FB_SHOWN, p, tgt, rows, p, info, _fb_connected(p)) + keep

    for btn, field in BROWSE_TARGETS.items():
        if pid.startswith(btn + "."):
            return show(_fb_start_dir(current[field]), field)
    if pid.startswith("fb-cancel"):
        return (FB_HIDDEN, nu, None, nu, nu, nu, nu) + keep
    if pid.startswith("fb-select"):
        chosen = os.path.join(path, "") if path else None      # with trailing '/'
        return (FB_HIDDEN, nu, None, nu, nu, nu, nu) + tuple(
            chosen if target == f and chosen else nu for f in fields)
    if '"fb-conn"' in pid and val:                    # a connected folder was clicked
        p = json.loads(pid.rsplit(".", 1)[0])["index"]
        if os.path.isdir(p):
            return show(p, target)
        return (nu,) * 5 + (html.Span("Missing folder: " + p, style={"color": "#c0392b"}), nu) + keep
    if not path:
        return (nu,) * 7 + keep
    if pid.startswith("fb-up"):
        return show(os.path.dirname(path.rstrip("/")) or "/", target)
    if pid.startswith("fb-home"):
        return show(os.path.expanduser("~"), target)
    if pid.startswith("fb-path-input"):
        p = os.path.expanduser((val or "").strip())
        if os.path.isdir(p):
            return show(os.path.abspath(p), target)
        return (nu, nu, nu, nu, nu, html.Span("Not a folder: " + (val or ""),
                                              style={"color": "#c0392b"}), nu) + keep
    if '"fb-dir"' in pid and val:                     # a sub-folder was clicked
        name = json.loads(pid.rsplit(".", 1)[0])["index"]
        return show(os.path.join(path, name), target)
    return (nu,) * 7 + keep


def _fb_connected(current):
    """Buttons for the experiment's connected data folders (home page → Folders)."""
    out = []
    try:
        folders = paths.connected("data")
    except Exception:
        folders = []
    cur = os.path.abspath(current or "").rstrip("/")
    for f in folders:
        here = f["path"].rstrip("/") == cur
        out.append(html.Button(
            f["name"] if f["name"] == paths.DATA_KINDS.get(f["kind"])
            else f"{f['name']} · {paths.DATA_KINDS.get(f['kind'], f['kind'])}",
            id={"type": "fb-conn", "index": f["path"]}, n_clicks=0, title=f["path"],
            className="btn btn-sm",
            style={"fontSize": "12px", "padding": "2px 8px", "borderRadius": "12px",
                   "border": f"1px solid {ACCENT}",
                   "background": ACCENT if here else "white", "color": "white" if here else "#333",
                   "opacity": 1 if os.path.isdir(f["path"]) else 0.5}))
    if out:
        out.insert(0, html.Span("Connected folders:", style={"fontSize": "12px", "color": "#666",
                                                              "alignSelf": "center"}))
    return out
