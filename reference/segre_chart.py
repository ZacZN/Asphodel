#!/usr/bin/env python3
"""
segre_chart.py  –  interactive Segrè chart (chart of nuclides) from NUBASE2020

A Dash page (http://127.0.0.1:8054/) showing every nuclide in
core/nubase_4.mas20 as a square at (N, Z).  Started from the home page
(Reference section), which opens it in a new tab; it stops a few seconds
after its tab is closed, or with Stop on the home page.

  * Colour by: half-life (log scale; stable black, unknown grey), main decay
    mode, or identified ions (your own marks, with an editing panel).
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

import datetime
import glob
import json
import logging
import os
import re
import uuid
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
from dash import dcc, html, Input, Output, State, ALL, Patch, ctx
import plotly.graph_objects as go

PORT = 8054
ACCENT, PALE, SIDE, LINE = "#f57d15", "#fff4ec", "#f9f9f9", "#dddddd"
MAGIC = [8, 20, 28, 50, 82, 126]
# overlay lines around the measured nuclides: (checklist value, table column, legend, colour, dash)
LIMIT_LINES = [("mass_limit", "mass_measured", "Measured masses", "#000000", "solid"),
               ("hl_limit", "hl_measured", "Measured half-lives", "#000000", "6px,4px")]
# Overlay lines are drawn over a wider "casing" line of the opposite lightness,
# so they stand out on every square colour (the red / blue / yellow decay
# modes, viridis, the black stable squares): the limit lines black on white,
# the experiment region orange on black.
LIMIT_CASING = "#ffffff"
REGION_CASING = "#000000"


def add_cased_line(fig, xs, ys, colour, casing, width=2.5, dash="solid", name=None):
    """A line with a solid casing underneath (casing width = line width + 3 px)."""
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip", showlegend=False,
                             line=dict(color=casing, width=width + 3)))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip", name=name,
                             showlegend=name is not None,
                             line=dict(color=colour, width=width, dash=dash)))
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
    # measured (not from systematics) mass excess; measured half-life = a plain
    # number (no '#' estimate, no '<' / '>' / '~' limit) or stable
    d["mass_measured"] = d["m.e."].notna() & ~d["m.e. sys?"].astype(bool)
    t_str = d["T"].astype(str).str.strip()
    d["hl_measured"] = (t_str.str.fullmatch(r"\d*\.?\d+(?:[Ee][+-]?\d+)?") & d["T_unit"].notna()) \
        | t_str.str.startswith("stbl")
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


def colouring(rows, by, id_cells=None):
    """z values, colour scale, zmin/zmax and colour-bar settings."""
    if by == "identified":
        order = {name: i for i, (name, _) in enumerate(ID_COLOURS)}
        z = np.array([order[cell_status((id_cells or {}).get((zz, nn), []))]
                      for zz, nn in zip(rows["Z"], rows["N"])], dtype=float)
        n = len(ID_COLOURS)
        return z, _stepped([c for _, c in ID_COLOURS]), -0.5, n - 0.5, \
            dict(title="Identified", tickvals=list(range(n)), ticktext=[m for m, _ in ID_COLOURS])
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
    raise ValueError(f"unknown colouring {by!r}")


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


def _hex(c):
    c = c.lstrip("#")
    return [int(c[i:i + 2], 16) for i in (0, 2, 4)]


def scale_colours(z, scale, zmin, zmax):
    """Hex colour of each z on a plotly-style colour scale [[pos, '#rrggbb'], ...]
    (a step where two entries share a position, as in _stepped)."""
    pos = np.array([p for p, _ in scale], dtype=float)
    rgb = np.array([_hex(c) for _, c in scale], dtype=float)
    t = np.clip((np.asarray(z, dtype=float) - zmin) / (zmax - zmin), 0, 1)
    t = np.round(t * 80) / 80      # 81 shades at most: fewer traces to draw, no visible steps
    out = []
    for v in t:
        j = int(np.searchsorted(pos, v, side="right"))
        j = min(max(j, 1), len(pos) - 1)
        p0, p1 = pos[j - 1], pos[j]
        f = 0.0 if p1 <= p0 else (v - p0) / (p1 - p0)
        r, g, b = rgb[j - 1] + f * (rgb[j] - rgb[j - 1])
        out.append(f"#{int(round(r)):02x}{int(round(g)):02x}{int(round(b)):02x}")
    return out


def _text_colour(hexc):
    r, g, b = _hex(hexc)
    return "#000000" if 0.299 * r + 0.587 * g + 0.114 * b > 150 else "#ffffff"


HALF = 0.47                  # half side of a square (cells are 1 wide: a thin white gap)
STABLE_CORNER = 0.38         # Identified ions: side of the corner triangles (stable / isomer identified)
ISOMER_CORNER_COLOUR = "#8e44ad"   # bottom-left corner: an isomer of this nuclide is identified


SELECT_COLOUR = "#e0242c"     # red circle around the square selected for editing
SELECT_RADIUS = 0.72          # in squares: just outside the corners (half-diagonal 0.66)


def selection_trace(sel):
    """Red circle around the selected square (sel = [Z, N] or None).  Drawn
    in data units (the axes have equal scales), so it zooms with the chart."""
    xs, ys = [], []
    if sel:
        zz, nn = sel
        t = np.linspace(0, 2 * np.pi, 49)
        xs = (nn + SELECT_RADIUS * np.cos(t)).round(4).tolist()
        ys = (zz + SELECT_RADIUS * np.sin(t)).round(4).tolist()
    return go.Scatter(x=xs, y=ys, mode="lines", name="selection", hoverinfo="skip", showlegend=False,
                      line=dict(color=SELECT_COLOUR, width=3.5))


def build_figure(by, state, overlays, mark=None, view=None, uirev="0", sel=None):
    """The squares are SVG path shapes (one per colour), not a heatmap: a
    heatmap is a raster image that Plotly re-encodes after every zoom step,
    and the browser shows a blank frame while it decodes the new one (the
    flicker).  Paths are vectors, so a zoom just redraws them; as layout
    shapes they redraw about twice as fast as filled scatter traces.  On top:
    an invisible WebGL marker per nuclide for the hover label (the page sets
    hoverdistance to the square size after each zoom), the A + symbol labels
    around the view (copied by the page from the hidden "labels-all" trace),
    and a dummy trace that carries the colour bar."""
    rows = state_rows(state)
    id_cells = entries_by_cell(all_entries()) if by == "identified" else None
    z, scale, zmin, zmax, cbar = colouring(rows, by, id_cells)
    cols = scale_colours(z, scale, zmin, zmax)
    n_arr, z_arr = rows["N"].to_numpy(), rows["Z"].to_numpy()
    hover = rows["hover"].to_numpy()
    if id_cells:
        hover = np.array([h + "".join("<br>" + entry_hover_line(e) for e in id_cells.get((zz, nn), []))
                          for h, zz, nn in zip(hover, z_arr, n_arr)], dtype=object)

    fig = go.Figure()
    # trace 0: outline of the square selected for editing (Identified ions);
    # kept first so the selection callback can patch data[0]
    fig.add_trace(selection_trace(sel))
    by_colour = {}
    for n, zz, c in zip(n_arr, z_arr, cols):
        by_colour.setdefault(c, []).append((n, zz))
    cell_shapes = []
    for c, cells in by_colour.items():
        path = "".join(f"M{n - HALF:g},{zz - HALF:g}H{n + HALF:g}V{zz + HALF:g}H{n - HALF:g}Z"
                       for n, zz in cells)
        cell_shapes.append(dict(type="path", path=path, fillcolor=c, line=dict(width=0),
                                layer="below", xref="x", yref="y"))
    if by == "identified":
        # stable nuclides: the top-right corner of the square filled in black
        st_rows = rows[np.isinf(rows["hl_s"].to_numpy(dtype=float))]
        k = STABLE_CORNER
        path = "".join(f"M{n + HALF:g},{zz + HALF:g}H{n + HALF - k:g}L{n + HALF:g},{zz + HALF - k:g}Z"
                       for n, zz in zip(st_rows["N"], st_rows["Z"]))
        if path:
            cell_shapes.append(dict(type="path", path=path, fillcolor="#000000", line=dict(width=0),
                                    layer="below", xref="x", yref="y"))
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="stable",
                                     marker=dict(symbol="triangle-ne", size=11, color="#000000"),
                                     hoverinfo="skip", showlegend=True))
        # an isomer identified (any entry with isomer m, n, ...): bottom-left corner purple
        iso_cells = [c for c, es in (id_cells or {}).items() if any(e.get("isomer", "g") != "g" for e in es)]
        path = "".join(f"M{n - HALF:g},{zz - HALF:g}H{n - HALF + k:g}L{n - HALF:g},{zz - HALF + k:g}Z"
                       for zz, n in iso_cells)
        if path:
            cell_shapes.append(dict(type="path", path=path, fillcolor=ISOMER_CORNER_COLOUR,
                                    line=dict(width=0), layer="below", xref="x", yref="y"))
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="isomer identified",
                                     marker=dict(symbol="triangle-sw", size=11, color=ISOMER_CORNER_COLOUR),
                                     hoverinfo="skip", showlegend=True))
    # hover: one invisible square marker per nuclide, coloured like its square
    # (the hover label takes that colour, as the heatmap's did).  WebGL: 3500
    # SVG markers made every zoom step slow to redraw; WebGL ones cost nothing.
    fig.add_trace(go.Scattergl(
        x=n_arr, y=z_arr, mode="markers", name="hover",
        marker=dict(symbol="square", size=1, color=cols, opacity=0),
        hovertext=hover, hoverinfo="text", showlegend=False))
    # A + symbol labels: all of them in a hidden trace (data for the page), and
    # the ones around the view, copied by the page into "labels" once the
    # squares are big enough (a few hundred text elements instead of 3500)
    fig.add_trace(go.Scatter(
        # plain lists: numpy arrays reach the page base64-encoded, which the
        # page's own code can't read
        x=n_arr.tolist(), y=z_arr.tolist(), mode="text", name="labels-all", visible=False,
        text=(rows["A"].astype(str) + rows["sym"]).tolist(),
        textfont=dict(size=9, color=[_text_colour(c) for c in cols]),
        hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=[], y=[], text=[], mode="text", name="labels", textfont=dict(size=9),
        hoverinfo="skip", showlegend=False))
    # colour bar
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers", name="colourbar", hoverinfo="skip", showlegend=False,
        marker=dict(color=[zmin], colorscale=scale, cmin=zmin, cmax=zmax, showscale=True,
                    colorbar=dict(thickness=14, len=0.9, **cbar))))

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
        add_cased_line(fig, xs, ys, ACCENT, REGION_CASING, width=3, name="Experiment region")
        if ref_zn:
            fig.add_trace(go.Scatter(x=[ref_zn[1]], y=[ref_zn[0]], mode="markers",
                                     marker=dict(symbol="star", size=13, color=ACCENT,
                                                 line=dict(color="#000", width=1)),
                                     hoverinfo="skip", showlegend=False))
        region_text = (f"{exp}: Z {s['z_lims'][0]}–{s['z_lims'][1]}, N {s['n_lims'][0]}–{s['n_lims'][1]}, "
                       f"below el1 {s['el1'][0]},{s['el1'][1]} – el2 {s['el2'][0]},{s['el2'][1]} "
                       f"(A,Z) · ★ {s['ref_name']} · {len(cells)} nuclides")
    # limits of measured masses / half-lives: exact outline of the measured
    # nuclides of the states shown (gaps inside get their own boxes)
    for key, col, label, colour, dash in LIMIT_LINES:
        if key not in overlays:
            continue
        sel = rows[rows[col]]
        xs, ys = outline(set(zip(sel["Z"], sel["N"])))
        add_cased_line(fig, xs, ys, colour, LIMIT_CASING, width=2.5, dash=dash, name=label)
        extra = f"{label}: {len(sel)} of {len(rows)}"
        region_text = f"{region_text} · {extra}" if region_text else extra
    if mark:
        zz, nn = mark
        shapes.append(dict(type="rect", x0=nn - .5, x1=nn + .5, y0=zz - .5, y1=zz + .5,
                           line=dict(color=ACCENT, width=4), layer="above"))

    xr, yr = (view or {}).get("x"), (view or {}).get("y")
    fig.update_layout(
        margin=dict(l=55, r=10, t=10, b=45), plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        hovermode="closest", hoverdistance=1,
        legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top", bgcolor="rgba(255,255,255,0.85)",
                    bordercolor=LINE, borderwidth=1),
        shapes=cell_shapes + shapes, dragmode="pan", uirevision=uirev,
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
# Identified ions (Colour by -> Identified ions)
# ===========================================================================
# Your own marks, per experiment, in data/<exp>/outputs/identified/segre_identified.json:
#   {"format": "asphodel-segre-identified", "version": 1,
#    "entries": [{"id", "Z", "N", "A", "sym", "isomer": "g"/"m"/..., "charge",
#                 "harmonic", "status": "confident"/"unsure", "notes", "updated"}],
#    "confirmed_lists": ["confirmed_....csv", ...]}     <- Theia lists shown too
# Marks belong to the nuclide's ground-state square; the isomer is a field of
# the entry.  A square is green if any entry for it is confident, else orange.
IDS_FILE = "segre_identified.json"
ID_COLOURS = [("confident", "#2e9e4a"), ("unsure", "#f59a23"), ("not identified", "#ececec")]
_CONF_CACHE = {}                      # path -> (mtime, entries)


def ids_path():
    return os.path.join(paths.outputs("identified"), IDS_FILE)


def load_ids():
    d = {"format": "asphodel-segre-identified", "version": 1, "entries": [], "confirmed_lists": []}
    try:
        with open(ids_path(), encoding="utf-8") as fh:
            saved = json.load(fh)
        if isinstance(saved, dict):
            d["entries"] = [e for e in saved.get("entries", []) if isinstance(e, dict)]
            d["confirmed_lists"] = [str(x) for x in saved.get("confirmed_lists", [])]
    except (OSError, ValueError):
        pass
    return d


def save_ids(d):
    path = ids_path()
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    os.replace(tmp, path)


def parse_int_list(text, lo, hi):
    """'72', '70, 71 72', '70-72', '212, 214-215' -> (sorted unique ints, None)
    or ([], error text).  Required: an empty field is an error."""
    t = str(text or "").strip()
    if not t:
        return [], "required"
    out = set()
    for part in re.split(r"[,;\s]+", t):
        if not part:
            continue
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if not m:
            return [], f"can't read '{part}'"
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else a
        if b < a:
            a, b = b, a
        if b - a > 50:
            return [], f"range '{part}' is too long"
        for v in range(a, b + 1):
            if v < lo or (hi is not None and v > hi):
                return [], f"{v} is out of range"
            out.add(v)
    return sorted(out), None


NEW_ISOMER = "new"            # Energy State value of a newly identified isomer (not in NUBASE)
NEW_FIELDS = ["new_ex", "new_ex_unc", "new_t", "new_t_unc"]     # optional numbers of a new isomer
T_UNITS = ["ns", "us", "ms", "s", "m", "h", "d", "y"]


def _num(v):
    return f"{v:g}" if isinstance(v, (int, float)) else None


def new_isomer_text(e, html_=False):
    """'E_x = 1234 ± 5 keV, T½ = 3 ± 1 s' for a new isomer ('' if nothing is known yet)."""
    parts = []
    ex, exu = _num(e.get("new_ex")), _num(e.get("new_ex_unc"))
    if ex:
        parts.append(f"E_x = {ex}" + (f" ± {exu}" if exu else "") + " keV")
    t, tu = _num(e.get("new_t")), _num(e.get("new_t_unc"))
    if t:
        parts.append(f"T½ = {t}" + (f" ± {tu}" if tu else "") + f" {e.get('new_t_unit') or 's'}")
    return ", ".join(parts)


def ion_name(e):
    """'186Hf72+', 'm186Hf72+'; a new isomer by its energy: '186Hf(1234 keV)72+'
    ('186Hf(new)72+' while its energy is not known)."""
    if e.get("isomer") == NEW_ISOMER:
        ex = _num(e.get("new_ex"))
        return f"{e['A']}{e['sym']}({ex + ' keV' if ex else 'new'}){e['charge']}+"
    iso = "" if e.get("isomer", "g") == "g" else e["isomer"]
    return f"{iso}{e['A']}{e['sym']}{e['charge']}+"


def confirmed_files():
    """Theia's confirmed lists of the experiment: [(name, label)], newest first."""
    out = []
    try:
        files = sorted(glob.glob(os.path.join(paths.outputs("confirmed"), "*.csv")),
                       key=os.path.getmtime, reverse=True)
    except OSError:
        return out
    for f in files:
        head = {}
        try:
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    if not line.startswith("# "):
                        break
                    k, _, v = line[2:].partition(": ")
                    head[k.strip()] = v.strip()
        except OSError:
            continue
        label = (f"{head.get('filter', '').replace('ions: ', '') or os.path.basename(f)}"
                 f" · {head.get('saved', '')[:16].replace('T', ' ')}"
                 f" · {head.get('accepted', '?')} accepted, {head.get('unsure', '?')} unsure")
        out.append((os.path.basename(f), label))
    return out


def confirmed_entries(name):
    """Entries from one Theia confirmed list: one per (ion, harmonic);
    confident if any peak of it was accepted, unsure if only unsure ones
    (rejected peaks are left out).  Read-only, cached by file time."""
    path = os.path.join(paths.outputs("confirmed"), name)
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return []
    hit = _CONF_CACHE.get(path)
    if hit and hit[0] == mtime:
        return hit[1]
    out = []
    try:
        df = pd.read_csv(path, comment="#")
        df = df[df["decision"].isin(["accepted", "unsure"])]
        for (ion, harm), g in df.groupby(["ion", "harmonic"]):
            m = re.match(r"([a-z]?)(\d+)([A-Za-z]+)(\d+)\+$", str(ion))
            if not m:
                continue
            r0 = g.iloc[0]
            out.append({"id": f"theia:{name}:{ion}:{int(harm)}", "Z": int(r0["Z"]), "N": int(r0["N"]),
                        "A": int(m.group(2)), "sym": m.group(3), "isomer": m.group(1) or "g",
                        "charge": int(m.group(4)), "harmonic": int(harm),
                        "status": "confident" if (g["decision"] == "accepted").any() else "unsure",
                        "notes": f"{(g['decision'] == 'accepted').sum()} accepted, "
                                 f"{(g['decision'] == 'unsure').sum()} unsure peaks",
                        "source": name})
    except Exception as exc:                    # unreadable list: skip it, say so on the console
        print(f"Segrè chart: can't read {name}: {exc}", flush=True)
    _CONF_CACHE[path] = (mtime, out)
    return out


def all_entries(d=None):
    """Own entries + those of the selected Theia lists."""
    d = d or load_ids()
    out = [dict(e, source=None) for e in d["entries"]]
    for name in d["confirmed_lists"]:
        out += confirmed_entries(name)
    return out


def entries_by_cell(entries):
    cells = {}
    for e in entries:
        cells.setdefault((e["Z"], e["N"]), []).append(e)
    for v in cells.values():
        v.sort(key=lambda e: (e.get("isomer", "g") != "g", e.get("isomer", ""), -e["charge"], e["harmonic"]))
    return cells


def cell_status(es):
    if any(e["status"] == "confident" for e in es):
        return "confident"
    return "unsure" if es else "not identified"


def entry_hover_line(e):
    mark = "✓" if e["status"] == "confident" else "?"
    note = (e.get("notes") or "").strip().replace("<", "&lt;")
    if len(note) > 60:
        note = note[:57] + "…"
    src = " [Theia list]" if e.get("source") else ""
    new = f" [new isomer{': ' + new_isomer_text(e) if new_isomer_text(e) else ''}]" \
        if e.get("isomer") == NEW_ISOMER else ""
    return f"{mark} {ion_name(e)} h{e['harmonic']}{new}{src}" + (f" – {note}" if note else "")


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
.main{{flex:1 1 auto;min-height:0;display:flex}}
.graph{{flex:1 1 auto;min-width:0;min-height:0}}
/* Identified ions: editing panel on the right */
.panel{{flex:none;width:370px;border-left:1px solid {LINE};background:{SIDE};padding:12px 14px;
  overflow-y:auto;font-size:13px}}
.panel h4{{margin:0 0 8px;font-size:13px;text-transform:uppercase;letter-spacing:.7px;color:#555}}
.panel h4 span{{color:#777;text-transform:none;letter-spacing:0;font-weight:400;margin-left:6px}}
.panel .sub{{font-size:11px;text-transform:uppercase;letter-spacing:.7px;color:#777;font-weight:700;margin:14px 0 6px}}
.panel .selt{{font-weight:700;font-size:15px;margin:10px 0 6px}} .panel .selt span{{font-weight:400;color:#777;font-size:12px}}
.panel .hint{{color:#777;font-size:12px}}
.panel details{{background:#fff;border:1px solid {LINE};border-radius:5px;padding:6px 8px}}
.panel summary{{cursor:pointer;font-weight:600}}
.panel .lists label{{display:block;margin:4px 0;font-size:12px;word-break:break-all}}
.panel .lists input{{accent-color:{ACCENT};margin-right:5px}}
.entry{{background:#fff;border:1px solid {LINE};border-left:5px solid #ccc;border-radius:4px;padding:6px 8px;margin:6px 0}}
.entry.confident{{border-left-color:{ID_COLOURS[0][1]}}} .entry.unsure{{border-left-color:{ID_COLOURS[1][1]}}}
.entry .newiso{{color:{ISOMER_CORNER_COLOUR};font-weight:600;margin-top:3px}}
.entry .ion{{font-weight:700;font-family:monospace;font-size:13px}} .entry .st{{color:#555;margin-left:6px}}
.entry .notes{{white-space:pre-wrap;margin-top:3px;color:#333}} .entry .src{{color:#777;font-size:11px;margin-top:3px;word-break:break-all}}
.entry .btns{{margin-top:5px;display:flex;gap:6px}}
.form{{background:#fff;border:1px solid {LINE};border-radius:5px;padding:8px 10px;margin-top:8px}}
.form .row{{display:flex;gap:10px}} .form .row>div{{flex:1}}
.form label.f{{display:block;font-size:11px;color:#777;margin:6px 0 2px;text-transform:uppercase;letter-spacing:.5px}}
.form input[type=text]{{width:100%;padding:4px 6px;border:1px solid {LINE};border-radius:4px;font-size:13px}}
.form textarea{{width:100%;min-height:70px;padding:4px 6px;border:1px solid {LINE};border-radius:4px;font-size:13px;font-family:inherit}}
.form .radio label{{margin-right:12px}} .form .radio input{{accent-color:{ACCENT};margin-right:3px}}
.form .btns{{margin-top:8px;display:flex;gap:8px;align-items:center}}
.form .newbox{{border-left:4px solid {ISOMER_CORNER_COLOUR};padding:2px 0 4px 8px;margin-top:6px}}
.btn.primary{{background:{ACCENT};border-color:{ACCENT};color:#fff}} .btn.primary:hover{{color:#fff;opacity:.9}}
.btn.danger:hover{{border-color:#c62828;color:#c62828}}
.alllist{{max-height:260px;overflow-y:auto}}
.alllist button{{display:block;width:100%;text-align:left;margin:3px 0;font-family:monospace;font-size:12px}}
"""

app = dash.Dash(__name__, title="Segrè chart", suppress_callback_exceptions=True)
app.index_string = app.index_string.replace("</head>", f"<style>{CSS}</style></head>")
app.index_string = app.index_string.replace(
    "{%scripts%}", "{%scripts%}\n<script>" + tabwatch.watch_js("Segrè chart") + "</script>")
icon.use_in_dash(app, "segre")       # core/icons/segre.png, else asphodel.png


def id_panel():
    """Right-hand panel of the Identified ions colouring."""
    d = load_ids()
    files = confirmed_files()
    have = {n for n, _ in files}
    lbl = lambda t: html.Label(t, className="f")
    return html.Div([
        html.H4(["Identified ions", html.Span(id="id-count")]),
        html.Details([
            html.Summary("Theia confirmed lists"),
            html.Div("Lists whose ions are shown as identified too (read-only; "
                     "accepted → confident, only unsure → unsure).", className="hint"),
            dcc.Checklist(id="id-lists", className="lists",
                          options=[{"label": lab, "value": n} for n, lab in files],
                          value=[n for n in d["confirmed_lists"] if n in have]),
            html.Div("No confirmed lists in outputs/confirmed." if not files else "", className="hint"),
            html.Button("Apply", id="id-lists-apply", className="btn small", n_clicks=0),
        ]),
        html.Div(id="id-sel-title", className="selt"),
        html.Div(id="id-entries"),
        html.Div([
            lbl("Energy State"),
            dcc.Dropdown(id="id-iso", clearable=False, value="g",
                         options=[{"label": "ground state", "value": "g"}]),
            # a newly identified isomer: every number optional (fill in later by editing)
            html.Div([
                html.Div([html.Div([lbl("Energy E_x (keV)"),
                                    dcc.Input(id="id-new-ex", type="text", inputMode="decimal",
                                              placeholder="optional", autoComplete="off")]),
                          html.Div([lbl("± (keV)"),
                                    dcc.Input(id="id-new-ex-unc", type="text", inputMode="decimal",
                                              placeholder="optional", autoComplete="off")])],
                         className="row"),
                html.Div([html.Div([lbl("Half-life T½"),
                                    dcc.Input(id="id-new-t", type="text", inputMode="decimal",
                                              placeholder="optional", autoComplete="off")]),
                          html.Div([lbl("±"),
                                    dcc.Input(id="id-new-t-unc", type="text", inputMode="decimal",
                                              placeholder="optional", autoComplete="off")]),
                          html.Div([lbl("Unit"),
                                    dcc.Dropdown(id="id-new-t-unit", clearable=False, value="s",
                                                 options=T_UNITS)])],
                         className="row"),
                html.Div("All optional: rough values or ones measured later can be added by editing.",
                         className="hint"),
            ], id="id-new-box", className="newbox", style={"display": "none"}),
            html.Div([html.Div([lbl("Charge state(s) q *"),
                                dcc.Input(id="id-charge", type="text", inputMode="numeric", debounce=False,
                                          placeholder="72 or 70-72", autoComplete="off")]),
                      html.Div([lbl("Harmonic(s) *"),
                                dcc.Input(id="id-harm", type="text", inputMode="numeric", debounce=False,
                                          placeholder="213 or 212, 214", autoComplete="off")])],
                     className="row"),
            html.Div("Several values (commas, spaces or ranges like 212-214) add one entry per "
                     "charge state × harmonic, with the same status and notes.", className="hint"),
            lbl("Status"),
            dcc.RadioItems(id="id-status", value="confident", className="radio", inline=True,
                           options=[{"label": "Confident", "value": "confident"},
                                    {"label": "Unsure", "value": "unsure"}]),
            lbl("Notes"),
            dcc.Textarea(id="id-notes", value=""),
            html.Div([html.Button("Add", id="id-save", className="btn primary", n_clicks=0),
                      html.Button("Cancel edit", id="id-cancel", className="btn", n_clicks=0,
                                  style={"display": "none"})], className="btns"),
            html.Div(id="id-msg", className="msg"),
        ], id="id-form", className="form", style={"display": "none"}),
        html.Div("All identified", className="sub"),
        html.Div(id="id-all", className="alllist"),
    ], id="id-panel", className="panel", style={"display": "none"})


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
                                            {"label": "Identified ions", "value": "identified"}])],
                     className="grp"),
            html.Div([html.Label("State", className="t"),
                      dcc.RadioItems(id="state", value="g", className="radio", inline=True,
                                     options=[{"label": "Ground states", "value": "g"},
                                              {"label": "Isomers", "value": "iso"}]),
                      dcc.Dropdown(id="level", clearable=False, className="dd-s", value="m",
                                   options=levels, disabled=True)], className="grp"),
            html.Div(dcc.Checklist(id="overlays", value=["region", "magic"], className="chk", inline=True,
                                   options=[{"label": "Experiment region", "value": "region"},
                                            {"label": "Measured masses", "value": "mass_limit"},
                                            {"label": "Measured half-lives", "value": "hl_limit"},
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
        html.Div([
            dcc.Graph(id="chart", className="graph", style={"height": "100%"}, responsive=True,
                      config={"scrollZoom": True, "displaylogo": False,
                              "toImageButtonOptions": {"filename": "segre_chart", "scale": 3}}),
            id_panel(),
        ], className="main"),
        dcc.Store(id="id-sel", data=None),          # [Z, N] of the square being edited
        dcc.Store(id="id-editing", data=None),      # id of the entry loaded into the form
        dcc.Store(id="id-ver", data=0),             # bumped after every change -> redraw
        html.Div(id="labels-dummy", style={"display": "none"}),
        dcc.Store(id="view", data={"rev": 0, "mark": None, "x": None, "y": None}),
        # the view on screen (set by the page after every zoom / pan), so a
        # redraw for another option keeps it
        dcc.Store(id="cur-view", data=None),
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
    Input("overlays", "value"), Input("view", "data"), Input("id-ver", "data"),
    State("cur-view", "data"), State("id-sel", "data"),
)
def _draw(by, state, level, overlays, view, _ver, cur, sel):
    """Redraw for any option.  The view stays where it is (the ranges on
    screen are put into the new figure), except after a search or a zoom
    button, which set their own view."""
    st = "g" if state != "iso" else (level or "m")
    if by == "identified":
        st = "g"                       # identifications are marked on the ground-state squares
    view = view or {}
    mark = tuple(view["mark"]) if view.get("mark") else None
    shown = view
    if ctx.triggered_id != "view" and cur and cur.get("x") and cur.get("y"):
        shown = dict(view, x=cur["x"], y=cur["y"])
    fig, info = build_figure(by, st, overlays or [], mark=mark, view=shown,
                             uirev=str(view.get("rev", 0)),
                             sel=sel if by == "identified" else None)
    n = len(state_rows(st))
    what = "ground states" if st == "g" else f"isomers '{st}'"
    text = (info + " · " if info else "") + f"{n} {what} shown"
    if by == "identified":
        text += " · identified ions are marked on the ground-state squares: click a square to edit"
    return fig, text


# ---------------------------------------------------------------- Identified ions

@app.callback(Output("id-panel", "style"), Input("by", "value"))
def _id_panel_shown(by):
    return {} if by == "identified" else {"display": "none"}


@app.callback(
    Output("id-sel", "data"),
    Input("chart", "clickData"), Input({"type": "idgo", "z": ALL, "n": ALL}, "n_clicks"),
    State("by", "value"),
    prevent_initial_call=True,
)
def _id_select(click, _go, by):
    """Click a square (Identified ions only), or a nuclide in the 'All identified' list."""
    t = ctx.triggered_id
    if isinstance(t, dict) and t.get("type") == "idgo":
        if not (ctx.triggered and ctx.triggered[0]["value"]):
            return dash.no_update
        return [t["z"], t["n"]]
    if by != "identified" or not click or not click.get("points"):
        return dash.no_update
    pt = click["points"][0]
    try:
        nn, zz = int(round(pt["x"])), int(round(pt["y"]))
    except (KeyError, TypeError, ValueError):
        return dash.no_update
    if TABLE[(TABLE["Z"] == zz) & (TABLE["N"] == nn) & (TABLE["state"] == "g")].empty:
        return dash.no_update
    return [zz, nn]


@app.callback(Output("chart", "figure", allow_duplicate=True), Input("id-sel", "data"),
              State("by", "value"), prevent_initial_call=True)
def _id_outline(sel, by):
    """Move the selection circle (trace 0) without redrawing the chart."""
    t = selection_trace(sel if by == "identified" else None)
    p = Patch()
    p["data"][0]["x"] = list(t.x or [])
    p["data"][0]["y"] = list(t.y or [])
    return p


def _gs_row(zz, nn):
    r = TABLE[(TABLE["Z"] == zz) & (TABLE["N"] == nn) & (TABLE["state"] == "g")]
    return None if r.empty else r.iloc[0]


def _entry_card(e):
    st = "confident" if e["status"] == "confident" else "unsure"
    body = [html.Span(ion_name(e), className="ion"),
            html.Span(f"h = {e['harmonic']} · {st}", className="st")]
    out = [html.Div(body)]
    if e.get("isomer") == NEW_ISOMER:
        out.append(html.Div("New isomer" + (": " + new_isomer_text(e) if new_isomer_text(e) else
                                            " (energy and half-life not given yet)"), className="newiso"))
    if (e.get("notes") or "").strip():
        out.append(html.Div(e["notes"], className="notes"))
    if e.get("source"):
        out.append(html.Div(f"Theia list: {e['source']}", className="src"))
    else:
        if e.get("updated"):
            out.append(html.Div(f"updated {e['updated'][:16].replace('T', ' ')}", className="src"))
        out.append(html.Div([
            html.Button("Edit", id={"type": "idedit", "id": e["id"]}, className="btn", n_clicks=0),
            dcc.ConfirmDialogProvider(
                html.Button("Delete", className="btn danger"), id={"type": "iddel", "id": e["id"]},
                message=f"Delete the identification {ion_name(e)} h{e['harmonic']}?"),
        ], className="btns"))
    return html.Div(out, className=f"entry {st}")


@app.callback(
    Output("id-sel-title", "children"), Output("id-entries", "children"),
    Output("id-form", "style"), Output("id-iso", "options"),
    Output("id-all", "children"), Output("id-count", "children"),
    Input("id-sel", "data"), Input("id-ver", "data"),
)
def _id_panel(sel, _ver):
    entries = all_entries()
    cells = entries_by_cell(entries)
    own = sum(1 for e in entries if not e.get("source"))
    count = f"{len(cells)} nuclides · {own} own + {len(entries) - own} from Theia lists"
    # the list of all identified nuclides (click = select)
    allrows = []
    for (zz, nn) in sorted(cells):
        es = cells[(zz, nn)]
        r = _gs_row(zz, nn)
        nm = r["name"] if r is not None else f"Z{zz} N{nn}"
        txt = f"{'✓' if cell_status(es) == 'confident' else '?'} {nm}: " + \
            ", ".join(f"{ion_name(e)} h{e['harmonic']}" for e in es)
        allrows.append(html.Button(txt, id={"type": "idgo", "z": zz, "n": nn}, className="btn", n_clicks=0))
    if not allrows:
        allrows = [html.Div("Nothing identified yet.", className="hint")]
    if not sel:
        return ([html.Span("Click a square to select a nuclide", className="hint")], [],
                {"display": "none"}, [{"label": "ground state", "value": "g"}], allrows, count)
    zz, nn = sel
    r = _gs_row(zz, nn)
    title = [r["name"] if r is not None else f"Z = {zz}, N = {nn}",
             html.Span(f"  Z = {zz}, N = {nn}")]
    isos = [{"label": "ground state", "value": "g"}] + [
        {"label": f"{st} (E_x {x} keV)" if str(x) not in ("", "nan") else st, "value": st}
        for st, x in TABLE[(TABLE["Z"] == zz) & (TABLE["N"] == nn) & ~TABLE["state"].isin(["g", "i", "j"])]
        [["state", "E_x"]].itertuples(index=False)]
    isos.append({"label": "New isomer (not in NUBASE)", "value": NEW_ISOMER})
    es = cells.get((zz, nn), [])
    cards = [_entry_card(e) for e in es] or [html.Div("Not identified yet – add it below.", className="hint")]
    return title, cards, {}, isos, allrows, count


@app.callback(Output("id-new-box", "style"), Input("id-iso", "value"))
def _id_new_box(iso):
    return {} if iso == NEW_ISOMER else {"display": "none"}


EDIT_OUTPUTS = ["ver", "editing", "iso", "charge", "harm", "status", "notes",
                "new_ex", "new_ex_unc", "new_t", "new_t_unc", "new_t_unit", "msg", "save_label", "cancel_style"]


def _out(**kw):
    """The outputs of _id_edit in order; anything not given is left unchanged."""
    return tuple(kw.get(k, dash.no_update) for k in EDIT_OUTPUTS)


def _cleared(**kw):
    """Outputs that empty the form (the harmonic and status are kept: often the same)."""
    base = dict(editing=None, iso="g", charge="", notes="", new_ex="", new_ex_unc="", new_t="",
                new_t_unc="", new_t_unit="s", save_label="Add", cancel_style={"display": "none"})
    base.update(kw)
    return _out(**base)


def _opt_float(text, name):
    """Optional non-negative number from a text field: (value or None, error or None)."""
    t = str(text or "").strip()
    if not t:
        return None, None
    try:
        v = float(t)
    except ValueError:
        return None, f"⚠ {name}: can't read '{t}' (a number, or leave it empty)."
    if v < 0 or v != v:
        return None, f"⚠ {name}: must be ≥ 0."
    return v, None


def _str(v):
    return "" if v is None else f"{v:g}"


@app.callback(
    Output("id-ver", "data"), Output("id-editing", "data"),
    Output("id-iso", "value"), Output("id-charge", "value"), Output("id-harm", "value"),
    Output("id-status", "value"), Output("id-notes", "value"),
    Output("id-new-ex", "value"), Output("id-new-ex-unc", "value"),
    Output("id-new-t", "value"), Output("id-new-t-unc", "value"), Output("id-new-t-unit", "value"),
    Output("id-msg", "children"), Output("id-save", "children"), Output("id-cancel", "style"),
    Input("id-save", "n_clicks"), Input("id-cancel", "n_clicks"),
    Input({"type": "idedit", "id": ALL}, "n_clicks"),
    Input({"type": "iddel", "id": ALL}, "submit_n_clicks"),
    Input("id-sel", "data"),
    State("id-iso", "value"), State("id-charge", "value"), State("id-harm", "value"),
    State("id-status", "value"), State("id-notes", "value"),
    State("id-new-ex", "value"), State("id-new-ex-unc", "value"),
    State("id-new-t", "value"), State("id-new-t-unc", "value"), State("id-new-t-unit", "value"),
    State("id-editing", "data"), State("id-ver", "data"),
    prevent_initial_call=True,
)
def _id_edit(_save, _cancel, _edit, _del, sel, iso, charge, harm, status, notes,
             new_ex, new_ex_unc, new_t, new_t_unc, new_t_unit, editing, ver):
    """Add / save / edit / delete an identification of the selected nuclide."""
    t = ctx.triggered_id
    ver = ver or 0
    if t == "id-sel" or t == "id-cancel":
        return _cleared(msg="")
    if isinstance(t, dict):
        if not (ctx.triggered and ctx.triggered[0]["value"]):
            return _out()
        d = load_ids()
        e = next((x for x in d["entries"] if x.get("id") == t["id"]), None)
        if e is None:
            return _out(msg="⚠ that entry no longer exists")
        if t["type"] == "iddel":
            d["entries"] = [x for x in d["entries"] if x.get("id") != t["id"]]
            save_ids(d)
            return _cleared(ver=ver + 1, msg=f"Deleted {ion_name(e)} h{e['harmonic']}.")
        return _out(editing=e["id"], iso=e.get("isomer", "g"), charge=str(e["charge"]),
                    harm=str(e["harmonic"]), status=e["status"], notes=e.get("notes", ""),
                    new_ex=_str(e.get("new_ex")), new_ex_unc=_str(e.get("new_ex_unc")),
                    new_t=_str(e.get("new_t")), new_t_unc=_str(e.get("new_t_unc")),
                    new_t_unit=e.get("new_t_unit") or "s",
                    msg=f"Editing {ion_name(e)} h{e['harmonic']}", save_label="Save changes",
                    cancel_style={})
    # Add / Save changes
    if not sel:
        return _out(msg="Select a nuclide first (click its square).")
    zz, nn = sel
    r = _gs_row(zz, nn)
    qs, err = parse_int_list(charge, 1, zz)
    if err:
        return _out(msg=f"⚠ Charge state q: {err} (whole numbers from 1 to {zz}).")
    hs, err = parse_int_list(harm, 1, None)
    if err:
        return _out(msg=f"⚠ Harmonic: {err} (whole numbers ≥ 1).")
    if editing and (len(qs) > 1 or len(hs) > 1):
        return _out(msg="⚠ An edited entry has one charge state and one harmonic – "
                        "add the others as new entries.")
    iso = iso or "g"
    newvals = {}
    if iso == NEW_ISOMER:
        for key, text, name in [("new_ex", new_ex, "Energy"), ("new_ex_unc", new_ex_unc, "Energy ±"),
                                ("new_t", new_t, "Half-life"), ("new_t_unc", new_t_unc, "Half-life ±")]:
            v, err = _opt_float(text, name)
            if err:
                return _out(msg=err)
            newvals[key] = v
        newvals["new_t_unit"] = new_t_unit if new_t_unit in T_UNITS else "s"
    d = load_ids()
    now = datetime.datetime.now().isoformat(timespec="seconds")
    base = {"Z": zz, "N": nn, "A": zz + nn, "sym": r["sym"] if r is not None else SYMBOLS.get(zz, "?"),
            "isomer": iso, "status": "unsure" if status == "unsure" else "confident",
            "notes": (notes or "").strip(), "updated": now, **newvals}

    def clash(q, h):
        # a new isomer is told apart from other new isomers of the nuclide by its energy
        return next((x for x in d["entries"] if x.get("id") != editing and x["Z"] == zz and x["N"] == nn
                     and x.get("isomer", "g") == iso and x["charge"] == q and x["harmonic"] == h
                     and (iso != NEW_ISOMER or x.get("new_ex") == newvals.get("new_ex"))), None)

    old = next((x for x in d["entries"] if x.get("id") == editing), None) if editing else None
    if old is not None:
        q, h = qs[0], hs[0]
        c = clash(q, h)
        if c:
            return _out(msg=f"⚠ {ion_name(c)} h{h} is already in the list – edit that one.")
        for k in NEW_FIELDS + ["new_t_unit"]:          # no stale new-isomer values on other states
            old.pop(k, None)
        old.update(base, charge=q, harmonic=h)
        msg = f"Saved {ion_name(old)} h{h}."
    else:
        # one entry per (charge state, harmonic) combination; existing ones are skipped
        added, skipped = [], []
        for q in qs:
            for h in hs:
                e = dict(base, charge=q, harmonic=h)
                if clash(q, h):
                    skipped.append(f"{ion_name(e)} h{h}")
                    continue
                e["id"] = uuid.uuid4().hex[:10]
                d["entries"].append(e)
                added.append(f"{ion_name(e)} h{h}")
        if not added:
            return _out(msg="⚠ Already in the list: " + ", ".join(skipped) + " – edit those.")
        msg = f"Added {len(added)}: " + ", ".join(added) + "." + \
            (f" Skipped (already in the list): {', '.join(skipped)}." if skipped else "")
    save_ids(d)
    return _cleared(ver=ver + 1, msg=msg)


@app.callback(
    Output("id-ver", "data", allow_duplicate=True), Output("id-msg", "children", allow_duplicate=True),
    Input("id-lists-apply", "n_clicks"), State("id-lists", "value"), State("id-ver", "data"),
    prevent_initial_call=True,
)
def _id_lists(_n, chosen, ver):
    d = load_ids()
    d["confirmed_lists"] = list(chosen or [])
    save_ids(d)
    n = sum(len(confirmed_entries(x)) for x in d["confirmed_lists"])
    return (ver or 0) + 1, f"{len(d['confirmed_lists'])} Theia list(s) shown ({n} ion/harmonic entries)."


# A + symbol in the squares once they are big enough on screen to read
# (done in the browser: it knows the size of a square in pixels).
# Only when the view is still, as for Theia's isochronicity labels: a
# scroll-wheel zoom is a preview that Plotly commits later, and a restyle in
# between throws the preview away (the zoom pulses and the squares flicker).
# So the labels are switched 0.4 s after the last wheel step / view change,
# and never while a mouse button is held (drag-zoom / pan).
app.clientside_callback(
    r"""
    function(_fig) {
        setTimeout(function() {
            var gd = document.querySelector('#chart .js-plotly-plot');
            if (!gd) return;
            if (!gd._segreHook) {
                gd._segreHook = true;
                var timer = null, lastWheel = 0, mouseDown = false, QUIET = 400;
                var update = function() {
                    if (!gd._fullLayout || !gd._fullLayout.yaxis || !gd.data || !gd.data.length) return;
                    var ya = gd._fullLayout.yaxis, r = ya.range;
                    var px = ya._length / Math.abs(r[1] - r[0]);       // one square, in pixels
                    var iL = -1, iA = -1;
                    gd.data.forEach(function(t, i) {
                        if (t.name === 'labels') iL = i;
                        if (t.name === 'labels-all') iA = i;
                    });
                    var jobs = [], relayoutHD = null;
                    // WebGL hover finds the nearest marker centre within hoverdistance
                    // pixels; the nearest centre is the square under the mouse, so a
                    // radius of ~0.6 square covers the whole square
                    var hd = Math.max(2, Math.round(px * 0.6));
                    if (gd._fullLayout.hoverdistance !== hd) relayoutHD = hd;
                    // labels: the squares in and around the view (half a view width
                    // each side, so a short pan still shows them), only when big enough
                    if (iL >= 0 && iA >= 0) {
                        var xr = gd._fullLayout.xaxis.range.map(Number), yr = r.map(Number);
                        var x0 = Math.min(xr[0], xr[1]), x1 = Math.max(xr[0], xr[1]);
                        var y0 = Math.min(yr[0], yr[1]), y1 = Math.max(yr[0], yr[1]);
                        var cov = gd._segreCover;      // region the current labels cover
                        if (px < %d) {
                            if (gd.data[iL].x.length) jobs.push([{x: [[]], y: [[]], text: [[]]}, [iL]]);
                            gd._segreCover = null;
                        } else if (!cov || x0 < cov[0] || x1 > cov[1] || y0 < cov[2] || y1 > cov[3]) {
                            var w = x1 - x0, h = y1 - y0;
                            cov = [x0 - w / 2, x1 + w / 2, y0 - h / 2, y1 + h / 2];
                            var A = gd.data[iA], X = [], Y = [], T = [], C = [];
                            for (var k = 0; k < A.x.length; k++) {
                                if (A.x[k] < cov[0] || A.x[k] > cov[1] || A.y[k] < cov[2] || A.y[k] > cov[3]) continue;
                                X.push(A.x[k]); Y.push(A.y[k]); T.push(A.text[k]); C.push(A.textfont.color[k]);
                            }
                            gd._segreCover = cov;
                            jobs.push([{x: [X], y: [Y], text: [T], 'textfont.color': [C]}, [iL]]);
                        }
                        // font grows with the squares: "188Re" fills about 3/4 of a square
                        if (px >= %d) {
                            var fs = Math.round(Math.min(28, Math.max(9, px * 0.24)));
                            if ((gd.data[iL].textfont || {}).size !== fs)
                                jobs.push([{'textfont.size': fs}, [iL]]);
                        }
                    }
                    // remember the view for the next redraw (other colouring, state, overlays)
                    var cv = {x: gd._fullLayout.xaxis.range.map(Number), y: r.map(Number)};
                    var cvKey = JSON.stringify(cv);
                    if (cvKey !== gd._segreView) {
                        gd._segreView = cvKey;
                        window.dash_clientside.set_props('cur-view', {data: cv});
                    }
                    if (!jobs.length && relayoutHD === null) return;
                    gd._segreBusy = true;
                    var p = Promise.resolve();
                    if (relayoutHD !== null)
                        p = p.then(function() { return Plotly.relayout(gd, {hoverdistance: relayoutHD}); });
                    jobs.forEach(function(j) { p = p.then(function() { return Plotly.restyle(gd, j[0], j[1]); }); });
                    p.then(function() { gd._segreBusy = false; }, function() { gd._segreBusy = false; });
                };
                var soon = function() {
                    clearTimeout(timer);
                    timer = setTimeout(function run() {
                        if (mouseDown || Date.now() - lastWheel < QUIET) {
                            timer = setTimeout(run, QUIET);
                            return;
                        }
                        update();
                    }, QUIET);
                };
                gd.addEventListener('wheel', function() { lastWheel = Date.now(); soon(); },
                                    {capture: true, passive: true});
                gd.addEventListener('mousedown', function() { mouseDown = true; }, true);
                window.addEventListener('mouseup', function() { if (mouseDown) { mouseDown = false; soon(); } });
                gd.on('plotly_relayout', function() { if (!gd._segreBusy) soon(); });
                window.addEventListener('resize', soon);
                gd._segreSoon = soon;
            }
            gd._segreCover = null;           // a new figure has no labels yet
            gd._segreSoon();
        }, 150);
        return window.dash_clientside.no_update;
    }
    """ % (LABEL_PX, LABEL_PX),
    Output("labels-dummy", "children"),
    Input("chart", "figure"),
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
