#!/usr/bin/env python3
"""
segre_chart.py  –  interactive Segrè chart (chart of nuclides) from NUBASE2020

A Dash page (http://127.0.0.1:8054/) showing every nuclide in
core/nubase_4.mas20 as a square at (N, Z).  Started from the home page
(Reference section), which opens it in a new tab; it stops a few seconds
after its tab is closed, or with Stop on the home page.

  * Colour by: half-life (log scale; stable black, unknown grey), main decay
    mode, mass measured vs. from systematics (#), or mass-excess uncertainty.
  * State: ground states, or one isomer level (m, n, p, ...) – each square
    is then that isomer of the nuclide.
  * Hover a square: its NUBASE data, in the same label as a simulated line
    in Theia's Viewer (core.physics.build_hover_label) plus Z and N.
  * Experiment region: outline of the nuclides the simulation includes
    (Z/N window and the el1–el2 diagonal cut from the experiment's
    physics.json) and the reference ion (★).  Reloading the page reads
    physics.json again.
  * Magic numbers: lines around Z, N = 8, 20, 28, 50, 82, 126.
  * Search: "198Ir", "Ir198", "Ir-198", "198Ir77+", "m186Hf" – zooms to it and
    marks it.  Zoomed in far enough, the squares are labelled A + symbol.

Usage
-----
    python reference/segre_chart.py      (normally started from the home page)
"""

import logging
import os
import re
import socket
import sys
import threading

import numpy as np
import pandas as pd

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import browser, icon, paths, physics, tabwatch
from core.nubase_reader import get_base_dataframe
from core.physics import build_hover_label, half_life_seconds

import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.graph_objects as go

PORT = 8054
ACCENT, PALE, SIDE, LINE = "#f57d15", "#fff4ec", "#f9f9f9", "#dddddd"
MAGIC = [8, 20, 28, 50, 82, 126]
LABEL_PX = 30               # squares get A + symbol labels when at least this many pixels high
ISOMER_LEVELS = [("m", "1st isomer (m)"), ("n", "2nd isomer (n)"), ("p", "p"), ("q", "q"),
                 ("r", "r"), ("x", "x")]


# ===========================================================================
# NUBASE table
# ===========================================================================

# decay-mode categories: (name, colour)
MODES = [("stable", "#000000"), ("β−", "#4a90d9"), ("β+ / EC", "#e05a5a"), ("α", "#f2d43a"),
         ("p", "#f5a04a"), ("n", "#9b59b6"), ("SF", "#3cab5a"), ("IT", "#9e9e9e"),
         ("other", "#7fd6d0"), ("unknown", "#e3e3e3")]
MODE_INDEX = {m: i for i, (m, _) in enumerate(MODES)}


def _mode_category(token):
    """NUBASE decay-mode token ('B-', 'B+p', '2EC', 'A', 'SF', '14C', ...) -> category."""
    t = token.strip()
    if t.startswith(("B-", "2B-")):
        return "β−"
    if t.startswith(("B+", "EC", "e+", "2B+", "2EC")):
        return "β+ / EC"
    if t == "A":
        return "α"
    if re.fullmatch(r"\d?p", t):
        return "p"
    if re.fullmatch(r"\d?n", t):
        return "n"
    if t == "SF":
        return "SF"
    if t == "IT":
        return "IT"
    return "other"


def main_decay_mode(br, t):
    """Category of the strongest decay branch ('IS=' abundances ignored);
    'stable' for T = stbl, 'unknown' if no branch is given."""
    if str(t).strip().startswith("stbl"):
        return "stable"
    best, best_v = None, -1.0
    for tok in str(br if isinstance(br, str) else "").split(";"):
        m = re.match(r"\s*([A-Za-z0-9+\-]+)\s*(?:[=~<>]+\s*([\d.]+))?", tok)
        if not m or m.group(1) == "IS":
            continue
        try:
            v = float(m.group(2)) if m.group(2) else 0.0
        except ValueError:
            v = 0.0
        if v > best_v:                      # ties: the first one listed
            best, best_v = m.group(1), v
    return _mode_category(best) if best else "unknown"


def load_table():
    """All NUBASE lines with Z, N, symbol, state ('g' or the isomer letter),
    half-life in s, main decay mode and the hover label."""
    d = get_base_dataframe()
    el = d["Element"].astype(str).str.strip()
    d.loc[el == "1n", "ZZZi"] = 0                       # free neutron (Z = 0)
    d = d[d["ZZZi"].notna() & d["AAA"].notna()].copy()
    d["Z"] = d["ZZZi"].astype(int)
    d["A"] = d["AAA"].astype(int)
    d["N"] = d["A"] - d["Z"]
    d["sym"] = el.loc[d.index].str.replace(r"^\d+", "", regex=True)
    d["state"] = d["Isomer_Flag"].fillna("g").astype(str).str.strip()
    # the suite's ion-name style: '198Ir', 'm186Hf'
    d["name"] = np.where(d["state"] == "g", "", d["state"]) + d["A"].astype(str) + d["sym"]
    d["hl_s"] = [half_life_seconds(t, u) for t, u in zip(d["T"], d["T_unit"])]
    d["mode"] = [main_decay_mode(b, t) for b, t in zip(d["BR"], d["T"])]
    isomers = (d[~d["state"].isin(["g", "i", "j"])].groupby(["Z", "N"])["state"]
               .apply(lambda s: ", ".join(s)))
    hover = []
    for _, row in d.iterrows():
        lines = build_hover_label(row["name"], row).split("<br>")
        lines.insert(1, f"Z = {row['Z']}, N = {row['N']}")
        if row["state"] == "g" and (row["Z"], row["N"]) in isomers.index:
            lines.append("Isomers: " + isomers.loc[(row["Z"], row["N"])])
        hover.append("<br>".join(lines))
    d["hover"] = hover
    return d.reset_index(drop=True)


TABLE = load_table()
Z_MAX, N_MAX = int(TABLE["Z"].max()), int(TABLE["N"].max())
SYMBOLS = (TABLE[TABLE["state"] == "g"].drop_duplicates("Z").set_index("Z")["sym"].to_dict())
Z_OF = {s.lower(): z for z, s in SYMBOLS.items() if z > 0}   # 'n' = nitrogen here


def z_of(sym):
    """Element symbol -> Z: exact case first ('n' neutron, 'N' nitrogen), then any case."""
    for z, s in SYMBOLS.items():
        if s == sym:
            return z
    return Z_OF.get(sym.lower())


# ===========================================================================
# Colourings
# ===========================================================================
HL_MIN, HL_MAX = -9.0, 18.0          # log10(T½ / s) range of the colour scale
HL_TICKS = [(-6, "1 µs"), (-3, "1 ms"), (0, "1 s"), (np.log10(60), "1 min"),
            (np.log10(3600), "1 h"), (np.log10(86400), "1 d"), (np.log10(3.15576e7), "1 y"),
            (np.log10(3.15576e10), "1 ky"), (np.log10(3.15576e13), "1 My"),
            (np.log10(3.15576e16), "1 Gy")]
VIRIDIS = ["#440154", "#482878", "#3e4989", "#31688e", "#26828e", "#1f9e89",
           "#35b779", "#6ece58", "#b5de2b", "#fde725"]


def _stepped(colours):
    """Discrete colour scale: colour i covers z in [i - 0.5, i + 0.5]."""
    n, out = len(colours), []
    for i, c in enumerate(colours):
        out += [[i / n, c], [(i + 1) / n, c]]
    return out


def _continuous_with_ends(colours, low, high, frac=0.04):
    """Colour scale with a flat band at each end (for 'unknown' and 'stable')."""
    out = [[0.0, low], [frac, low]]
    for i, c in enumerate(colours):
        # rounded: a position a hair above 1 - frac (float error) makes plotly drop the scale
        out.append([round(frac + (1 - 2 * frac) * i / (len(colours) - 1), 6), c])
    return out + [[1 - frac, high], [1.0, high]]


def colouring(rows, by):
    """z values, colour scale, zmin/zmax and colour-bar settings."""
    if by == "halflife":
        frac = 0.04
        span = (HL_MAX - HL_MIN) / (1 - 2 * frac)
        lo, hi = HL_MIN - frac * span, HL_MAX + frac * span
        hl = rows["hl_s"].to_numpy(dtype=float)
        z = np.full(len(rows), lo + frac * span / 2)                  # unknown
        known = np.isfinite(hl) & (hl > 0)
        z[known] = np.clip(np.log10(hl[known]), HL_MIN, HL_MAX)
        z[np.isinf(hl)] = hi - frac * span / 2                       # stable
        ticks = [(lo + frac * span / 2, "unknown")] + HL_TICKS + [(hi - frac * span / 2, "stable")]
        return z, _continuous_with_ends(VIRIDIS, "#d0d0d0", "#000000", frac), lo, hi, \
            dict(title="T½", tickvals=[t for t, _ in ticks], ticktext=[s for _, s in ticks])
    if by == "mode":
        z = rows["mode"].map(MODE_INDEX).to_numpy(dtype=float)
        n = len(MODES)
        return z, _stepped([c for _, c in MODES]), -0.5, n - 0.5, \
            dict(title="Main decay", tickvals=list(range(n)), ticktext=[m for m, _ in MODES])
    if by == "mass":
        known = rows["m.e."].notna().to_numpy()
        sys_ = rows["m.e. sys?"].astype(bool).to_numpy()
        z = np.where(~known, 2, np.where(sys_, 1, 0)).astype(float)
        labels = ["measured", "systematics (#)", "no value"]
        return z, _stepped([ACCENT, "#c8c8c8", "#f2f2f2"]), -0.5, 2.5, \
            dict(title="Mass excess", tickvals=[0, 1, 2], ticktext=labels)
    # mass-excess uncertainty, log10(keV); systematics values hatched by a grey end band
    unc = rows["m.e._unc"].to_numpy(dtype=float)
    lu = np.log10(np.where(unc > 0, unc, 1e-6))
    lo_v, hi_v = -6.0, 3.0
    frac = 0.04
    span = (hi_v - lo_v) / (1 - 2 * frac)
    lo, hi = lo_v - frac * span, hi_v + frac * span
    z = np.clip(lu, lo_v, hi_v)
    z[rows["m.e. sys?"].astype(bool).to_numpy()] = hi - frac * span / 2
    z[~np.isfinite(unc)] = lo + frac * span / 2
    ticks = ([(lo + frac * span / 2, "no value")]
             + [(v, f"{10.0 ** v:g} keV") for v in range(-5, 3)]
             + [(hi - frac * span / 2, "systematics (#)")])
    return z, _continuous_with_ends(VIRIDIS[::-1], "#f2f2f2", "#c8c8c8", frac), lo, hi, \
        dict(title="δ(m.e.)", tickvals=[t for t, _ in ticks], ticktext=[s for _, s in ticks])


# ===========================================================================
# Figure
# ===========================================================================

def state_rows(state):
    return TABLE[TABLE["state"] == state]


def region_info():
    """Experiment, its physics settings and the included (Z, N) cells."""
    exp = paths.experiment()
    s = physics.load_shared(exp)
    (z0, z1), (n0, n1) = s["z_lims"], s["n_lims"]
    (a1, zz1), (a2, zz2) = s["el1"], s["el2"]
    g = (zz2 - zz1) / ((a2 - zz2) - (a1 - zz1))
    c = zz2 - g * (a2 - zz2)
    cells = {(z, n) for z in range(z0, z1 + 1) for n in range(n0, n1 + 1) if z < g * n + c}
    # only cells that exist in NUBASE (as nuclear_reader's selection does)
    have = set(zip(TABLE.loc[TABLE["state"] == "g", "Z"], TABLE.loc[TABLE["state"] == "g", "N"]))
    cells &= have
    ref = re.match(r"(\d+)([A-Za-z]+)", s.get("ref_name", ""))
    ref_zn = None
    if ref and z_of(ref.group(2)) is not None:
        z = z_of(ref.group(2))
        ref_zn = (z, int(ref.group(1)) - z)
    return exp, s, cells, ref_zn


def outline(cells):
    """x, y (with None breaks) of the boundary of a set of (Z, N) cells."""
    xs, ys = [], []
    for z, n in cells:
        for dz, dn, seg in ((1, 0, ((n - .5, z + .5), (n + .5, z + .5))),
                            (-1, 0, ((n - .5, z - .5), (n + .5, z - .5))),
                            (0, 1, ((n + .5, z - .5), (n + .5, z + .5))),
                            (0, -1, ((n - .5, z - .5), (n - .5, z + .5)))):
            if (z + dz, n + dn) not in cells:
                xs += [seg[0][0], seg[1][0], None]
                ys += [seg[0][1], seg[1][1], None]
    return xs, ys


def magic_shapes(rows):
    """Lines along both sides of each magic proton row and neutron column."""
    shapes = []
    style = dict(type="line", line=dict(color="#555555", width=1), layer="above")
    for m in MAGIC:
        nz = rows.loc[rows["Z"] == m, "N"]
        if len(nz):
            for y in (m - .5, m + .5):
                shapes.append(dict(style, x0=nz.min() - 3, x1=nz.max() + 3, y0=y, y1=y))
        zn = rows.loc[rows["N"] == m, "Z"]
        if len(zn):
            for x in (m - .5, m + .5):
                shapes.append(dict(style, x0=x, x1=x, y0=zn.min() - 3, y1=zn.max() + 3))
    return shapes


def build_figure(by, state, overlays, mark=None, view=None, uirev="0"):
    rows = state_rows(state)
    z, scale, zmin, zmax, cbar = colouring(rows, by)
    grid = np.full((Z_MAX + 1, N_MAX + 1), np.nan)
    hov = np.full(grid.shape, "", dtype=object)
    txt = np.full(grid.shape, "", dtype=object)
    zi, ni = rows["Z"].to_numpy(), rows["N"].to_numpy()
    grid[zi, ni] = z
    hov[zi, ni] = rows["hover"].to_numpy()
    txt[zi, ni] = (rows["A"].astype(str) + rows["sym"]).to_numpy()

    fig = go.Figure(go.Heatmap(
        z=grid, x=np.arange(N_MAX + 1), y=np.arange(Z_MAX + 1),
        colorscale=scale, zmin=zmin, zmax=zmax, xgap=1, ygap=1,
        hovertext=hov, hoverinfo="text", hoverongaps=False,
        text=txt, texttemplate="", textfont=dict(size=9),
        colorbar=dict(thickness=14, len=0.9, **cbar)))

    shapes = magic_shapes(TABLE[TABLE["state"] == "g"]) if "magic" in overlays else []
    region_text = ""
    if "region" in overlays:
        try:
            exp, s, cells, ref_zn = region_info()
        except Exception as exc:                   # no experiment / unreadable physics.json
            exp, s, cells, ref_zn = None, None, set(), None
            region_text = f"⚠ experiment region not shown: {exc}"
    if "region" in overlays and s is not None:
        xs, ys = outline(cells)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=ACCENT, width=2.5),
                                 hoverinfo="skip", showlegend=False))
        if ref_zn:
            fig.add_trace(go.Scatter(x=[ref_zn[1]], y=[ref_zn[0]], mode="markers",
                                     marker=dict(symbol="star", size=13, color=ACCENT,
                                                 line=dict(color="#000", width=1)),
                                     hoverinfo="skip", showlegend=False))
        region_text = (f"{exp}: Z {s['z_lims'][0]}–{s['z_lims'][1]}, N {s['n_lims'][0]}–{s['n_lims'][1]}, "
                       f"below el1 {s['el1'][0]},{s['el1'][1]} – el2 {s['el2'][0]},{s['el2'][1]} "
                       f"(A,Z) · ★ {s['ref_name']} · {len(cells)} nuclides")
    if mark:
        zz, nn = mark
        shapes.append(dict(type="rect", x0=nn - .5, x1=nn + .5, y0=zz - .5, y1=zz + .5,
                           line=dict(color=ACCENT, width=4), layer="above"))

    xr, yr = (view or {}).get("x"), (view or {}).get("y")
    fig.update_layout(
        margin=dict(l=55, r=10, t=10, b=45), plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        shapes=shapes, dragmode="zoom", uirevision=uirev,
        xaxis=dict(title="N (neutrons)", range=xr or [-1, N_MAX + 1], showgrid=False,
                   zeroline=False, constrain="domain", ticks="outside"),
        yaxis=dict(title="Z (protons)", range=yr or [-1, Z_MAX + 1], showgrid=False,
                   zeroline=False, scaleanchor="x", scaleratio=1, ticks="outside"),
        font=dict(family='-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif', size=12),
    )
    return fig, region_text


# ===========================================================================
# Search
# ===========================================================================

def find_nuclide(q):
    """'198Ir', 'Ir198', 'Ir-198', '198Ir77+', 'm186Hf', '186Hfm', 'Z82 N126'
    -> ((Z, N), text) or (None, message)."""
    s = re.sub(r"[\s\-]+", "", q or "")
    m = re.fullmatch(r"[Zz]=?(\d+),?[Nn]=?(\d+)", s)
    if m:
        z, n = int(m.group(1)), int(m.group(2))
    else:
        s = re.sub(r"\d+\+$", "", s)
        m = (re.fullmatch(r"[a-z]?(\d{1,3})([A-Za-z]{1,2})[a-z]?", s)
             or re.fullmatch(r"([A-Za-z]{1,2})(\d{1,3})[a-z]?", s))
        if not m:
            return None, f"✗ can't read '{q.strip()}'"
        a, sym = (m.group(1), m.group(2)) if m.group(1).isdigit() else (m.group(2), m.group(1))
        z = z_of(sym)
        if z is None:
            return None, f"✗ no element '{sym}'"
        n = int(a) - z
    hit = TABLE[(TABLE["Z"] == z) & (TABLE["N"] == n) & (TABLE["state"] == "g")]
    if hit.empty:
        return None, f"✗ Z = {z}, N = {n} is not in NUBASE2020"
    return (z, n), f"{hit.iloc[0]['name']}: Z = {z}, N = {n}"


# ===========================================================================
# Page
# ===========================================================================
CSS = f"""
*{{box-sizing:border-box}}
body{{margin:0;font-family:-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;color:#222;
  background:#fff;font-size:14px;overflow:hidden}}
.page{{height:100vh;display:flex;flex-direction:column}}
header{{flex:none;height:52px;display:flex;align-items:center;gap:14px;padding:0 18px;
  border-bottom:3px solid {ACCENT}}}
header .logo{{font-weight:800;font-size:18px;letter-spacing:.5px}} header .logo span{{color:{ACCENT}}}
header .exp{{color:#777;font-family:monospace;font-size:12px}}
.bar{{flex:none;display:flex;flex-wrap:wrap;align-items:center;gap:8px 18px;padding:8px 18px;
  background:{SIDE};border-bottom:1px solid {LINE};font-size:13px}}
.bar .grp{{display:flex;align-items:center;gap:6px}} .bar .grp>*{{flex:none}}
.bar label.t{{font-size:11px;text-transform:uppercase;letter-spacing:.7px;color:#777;font-weight:700}}
.bar .dd{{width:190px}} .bar .dd-s{{width:150px}}
.bar .msg{{min-width:190px}}
.bar input[type=text]{{width:150px!important;flex:none;padding:4px 7px;border:1px solid {LINE};border-radius:4px;font-size:13px}}
.bar .chk label{{margin-right:12px}} .bar .chk input{{accent-color:{ACCENT};margin-right:4px}}
.bar .radio label{{margin-right:10px;color:#222!important}} .bar .radio input{{accent-color:{ACCENT};margin-right:3px}}
.btn{{border:1px solid {LINE};background:#fff;border-radius:4px;padding:4px 10px;cursor:pointer;font-size:12px}}
.btn:hover{{border-color:{ACCENT};color:{ACCENT}}}
.msg{{color:#8a4a10;font-size:12px}}
.info{{flex:none;padding:4px 18px;font-size:12px;color:#8a4a10;background:{PALE};
  border-bottom:1px solid {LINE};min-height:24px}}
.graph{{flex:1 1 auto;min-height:0}}
"""

app = dash.Dash(__name__, title="Segrè chart", suppress_callback_exceptions=True)
app.index_string = app.index_string.replace("</head>", f"<style>{CSS}</style></head>")
app.index_string = app.index_string.replace(
    "{%scripts%}", "{%scripts%}\n<script>" + tabwatch.watch_js("Segrè chart") + "</script>")
icon.use_in_dash(app, "segre")       # core/icons/segre.png, else asphodel.png


def serve_layout():
    levels = [{"label": lbl, "value": v} for v, lbl in ISOMER_LEVELS if (TABLE["state"] == v).any()]
    return html.Div([
        html.Header([html.Div(["Segrè ", html.Span("chart")], className="logo"),
                     html.Div(f"NUBASE2020 · core/nubase_4.mas20 · experiment {paths.experiment()}",
                              className="exp")]),
        html.Div([
            html.Div([html.Label("Colour by", className="t"),
                      dcc.Dropdown(id="by", clearable=False, className="dd", value="halflife",
                                   options=[{"label": "Half-life", "value": "halflife"},
                                            {"label": "Main decay mode", "value": "mode"},
                                            {"label": "Mass: measured / #", "value": "mass"},
                                            {"label": "Mass-excess uncertainty", "value": "massunc"}])],
                     className="grp"),
            html.Div([html.Label("State", className="t"),
                      dcc.RadioItems(id="state", value="g", className="radio", inline=True,
                                     options=[{"label": "Ground states", "value": "g"},
                                              {"label": "Isomers", "value": "iso"}]),
                      dcc.Dropdown(id="level", clearable=False, className="dd-s", value="m",
                                   options=levels, disabled=True)], className="grp"),
            html.Div(dcc.Checklist(id="overlays", value=["region", "magic"], className="chk", inline=True,
                                   options=[{"label": "Experiment region", "value": "region"},
                                            {"label": "Magic numbers", "value": "magic"}]),
                     className="grp"),
            html.Div([html.Label("Find", className="t"),
                      dcc.Input(id="q", type="text", placeholder="198Ir, Ir198, m186Hf",
                                debounce=False, n_submit=0, autoComplete="off",
                                style={"width": "150px"}),
                      html.Span(id="q-msg", className="msg")], className="grp"),
            html.Div([html.Button("Experiment region", id="zoom-region", className="btn", n_clicks=0,
                                  title="Zoom to the experiment's Z/N window"),
                      html.Button("Whole chart", id="zoom-all", className="btn", n_clicks=0)],
                     className="grp"),
        ], className="bar"),
        html.Div(id="region-info", className="info"),
        dcc.Graph(id="chart", className="graph", style={"height": "100%"},
                  config={"scrollZoom": True, "displaylogo": False,
                          "toImageButtonOptions": {"filename": "segre_chart", "scale": 3}}),
        html.Div(id="labels-dummy", style={"display": "none"}),
        dcc.Store(id="view", data={"rev": 0, "mark": None, "x": None, "y": None}),
    ], className="page")


app.layout = serve_layout


@app.callback(Output("level", "disabled"), Input("state", "value"))
def _level_enabled(state):
    return state != "iso"


@app.callback(
    Output("view", "data"), Output("q-msg", "children"),
    Input("q", "n_submit"), Input("zoom-region", "n_clicks"), Input("zoom-all", "n_clicks"),
    State("q", "value"), State("view", "data"),
    prevent_initial_call=True,
)
def _set_view(_s, _r, _a, q, view):
    """Search (Enter) and the zoom buttons: a new view (and a new uirevision,
    so the zoom is applied even if the chart was zoomed by hand)."""
    t = ctx.triggered_id
    rev = (view or {}).get("rev", 0) + 1
    if t == "q":
        zn, msg = find_nuclide(q)
        if not zn:
            return dash.no_update, msg
        z, n = zn
        return {"rev": rev, "mark": [z, n], "x": [n - 12, n + 12], "y": [z - 8, z + 8]}, msg
    if t == "zoom-region":
        s = physics.load_shared(paths.experiment())
        (z0, z1), (n0, n1) = s["z_lims"], s["n_lims"]
        return {"rev": rev, "mark": (view or {}).get("mark"), "x": [n0 - 3, n1 + 3],
                "y": [z0 - 3, z1 + 3]}, dash.no_update
    return {"rev": rev, "mark": None, "x": None, "y": None}, ""


@app.callback(
    Output("chart", "figure"), Output("region-info", "children"),
    Input("by", "value"), Input("state", "value"), Input("level", "value"),
    Input("overlays", "value"), Input("view", "data"),
)
def _draw(by, state, level, overlays, view):
    st = "g" if state != "iso" else (level or "m")
    view = view or {}
    mark = tuple(view["mark"]) if view.get("mark") else None
    fig, info = build_figure(by, st, overlays or [], mark=mark, view=view,
                             uirev=str(view.get("rev", 0)))
    n = len(state_rows(st))
    what = "ground states" if st == "g" else f"isomers '{st}'"
    return fig, (info + " · " if info else "") + f"{n} {what} shown"


# A + symbol in the squares once they are big enough on screen to read
# (done in the browser: it knows the size of a square in pixels)
app.clientside_callback(
    r"""
    function(_rl, _fig) {
        setTimeout(function() {
            var gd = document.querySelector('#chart .js-plotly-plot');
            if (!gd || !gd._fullLayout || !gd._fullLayout.yaxis || !gd.data || !gd.data.length) return;
            var ya = gd._fullLayout.yaxis, r = ya.range;
            var px = ya._length / Math.abs(r[1] - r[0]);
            var want = px >= %d ? '%%{text}' : '';
            if (gd.data[0].texttemplate !== want) Plotly.restyle(gd, {texttemplate: want}, [0]);
        }, 50);
        return window.dash_clientside.no_update;
    }
    """ % LABEL_PX,
    Output("labels-dummy", "children"),
    Input("chart", "relayoutData"), Input("chart", "figure"),
)


_TABS = tabwatch.TabTracker(on_last_closed=lambda: os._exit(0), name="Segrè chart")
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
    print(f"Segrè chart at {url}")
    threading.Timer(1.0, lambda: browser.open_url(url)).start()
    app.run(debug=False, host="127.0.0.1", port=PORT)
