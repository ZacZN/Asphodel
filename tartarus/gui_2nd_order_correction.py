#!/usr/bin/env python3
"""
gui_2nd_order_correction.py  –  GUI Second Order Correction

Fit the 2nd-order frequency correction by hand on a spectrum, as a Dash page
(http://127.0.0.1:8056/), started from the Asphodel home page (Tartarus).
One harmonic is worked on at a time.

1.  Open a 1-D or summed spectrum (.npz) from the connected spectra_1d /
    summed_1d folders and pick the harmonic.  The simulated lines of that
    harmonic are drawn as in Theia's Viewer: the strongest peak is the
    reference ion (208Pb80+) at the reference harmonic; another harmonic is
    placed from the reference ion's own peak in it (f_ref / h_ref * h, peak
    within ± ref_search_hz; core.physics.harmonic_references).  No correction
    is applied to the lines: they are what the correction is fitted against.
2.  Drag the lines into place (zoom in first: lines can be dragged when at
    most MAX_SHAPES are in view).  "Stretch all": dragging a line stretches
    all lines of the harmonic about its reference line, which stays fixed.
    "Move one line": the dragged line moves on its own; the reference line of
    a harmonic other than the reference harmonic can be dragged onto its peak.
3.  Assign peaks: click the ▼ marker above a line to select it, then click on
    its peak in the spectrum (or "Assign to nearest peak").  f_exp is the
    highest bin within ± the peak window, refined with a 3-point parabola.
4.  "Calculate": the anchored fit of the existing tool
    (tartarus/new_2nd_order_correction_plot.py, core.corrections.fit_correction)
    for the harmonic shown, in revolution frequencies:
        x = f_sim / h   (that harmonic's own reference, uncorrected)
        y = f_exp / h - x,  anchored at the reference line (x0, 0)
        y = a x^2 + b x + c
5.  "Save into profile": the harmonic's a, b, c go into a new or an existing
    correction profile (core/corrections.py, per_harmonic), replacing that
    harmonic if it is there already (after a second click).  The reference
    harmonic's fit is also the profile's general a, b, c (used for harmonics
    without their own).  "Also make it the default" puts the profile in the
    shared physics settings (physics.json).

Each harmonic keeps its own stretch, moved lines, reference line, assignments
and last fit, saved per spectrum file (data/<exp>/settings/
gui_correction_sessions.json) and restored when the file is opened again.
Settings: data/<exp>/settings/gui_correction.json (only the values that
differ from physics.json).

Plot speed (the rules used for Theia's and the Segrè chart's plots):
  * the spectrum (~131k points, WebGL) has no Plotly hover: a bottom-right
    readout shows f and P, and a click on the spectrum is caught in the page
    and sent as one value (hover on every point pushes hoverData through Dash
    on each mouse move);
  * the spectrum is sent once; zoom, drag, select and assign only patch the
    lines, markers and labels (dash.Patch);
  * the lines are a few WebGL traces; only the lines in view (at most
    MAX_SHAPES) are draggable shapes; the ▼ markers are WebGL;
  * the view is sent to the server only when it has been still for 0.4 s
    with no mouse button held (no patch during a wheel zoom or drag, which
    would throw the zoom preview away and make the plot pulse);
  * plain lists, not numpy arrays, for everything the page reads in JS.
"""

import datetime
import json
import logging
import os
import re
import socket
import sys
import threading
from functools import lru_cache

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import browser, corrections, icon, paths, physics, tabwatch
from core.nubase_reader import MASS_MODEL, unpack_npz

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, dash_table, Input, Output, State, ctx

APP_NAME = "GUI Second Order Correction"
PORT = 8056
ACCENT, PALE, SIDE, LINE = "#f57d15", "#fff4ec", "#f9f9f9", "#dddddd"
LINE_COL = "#2E91E5"        # simulated lines
ASSIGNED_COL = "#1CA71C"    # assigned lines
SETTINGS_FILE = "gui_correction.json"
SESSIONS_FILE = "gui_correction_sessions.json"
MIN_POINTS = 3              # assigned ions for a fit (2 parameters + 1 dof)
Y_TOP = 0.86                # top of the spectrum (paper); the marker strip is above it
MAX_SHAPES = 150            # draggable lines at once (zoom in for more)
MAX_LABELS = 60             # labels drawn (in view)
MAX_HOVER_LINES = 600       # lines in view that get a tooltip along their height
NEW_PROFILE = "__new__"

TOOL_DEFAULTS = {
    "ref_name": "", "ref_harmonic": 0, "gamma_t": 0.0, "harmonics": "",
    "z_lims": "", "n_lims": "", "el1": "", "el2": "", "mq_width": 0.0,
    "ref_search_hz": 0.0,
    "peak_window_hz": 100.0,     # ± window for the assigned peaks
    "file": "",                  # last spectrum opened
}
SETTING_KEYS = ["ref_name", "ref_harmonic", "gamma_t", "harmonics", "z_lims", "n_lims",
                "el1", "el2", "mq_width", "ref_search_hz", "peak_window_hz"]
LABELS = {
    "ref_name": "Reference ion (strongest peak)", "ref_harmonic": "Reference harmonic",
    "gamma_t": "γ_t", "harmonics": "Harmonics", "z_lims": "Z min, max", "n_lims": "N min, max",
    "el1": "el1 (A, Z)", "el2": "el2 (A, Z)", "mq_width": "m/q window (±)",
    "ref_search_hz": "Ref. peak search in other harmonics ± (Hz)",
    "peak_window_hz": "Peak window for assignments ± (Hz)",
}


# ===========================================================================
# Settings and sessions
# ===========================================================================

def _read_json(path):
    try:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


_LOCK = threading.Lock()


def load_settings():
    return physics.with_shared(TOOL_DEFAULTS, _read_json(paths.settings_file(SETTINGS_FILE)))


def save_settings(changes):
    with _LOCK:
        cfg = {**load_settings(), **changes}
        try:
            _write_json(paths.settings_file(SETTINGS_FILE), physics.strip_shared(cfg))
        except OSError:
            pass


def _session_key(path):
    return os.path.realpath(os.path.expanduser(path)) if path else ""


SESSION_KEYS = ("h", "k", "offsets", "ref_override", "assign", "fit")


def load_session(path):
    s = _read_json(paths.settings_file(SESSIONS_FILE)).get(_session_key(path)) or {}
    return new_state(path, s)


def save_session(state):
    if not state or not state.get("file"):
        return
    with _LOCK:
        p = paths.settings_file(SESSIONS_FILE)
        all_ = _read_json(p)
        d = {k: state.get(k) for k in SESSION_KEYS}
        d["saved"] = datetime.datetime.now().isoformat(timespec="seconds")
        all_[_session_key(state["file"])] = d
        try:
            _write_json(p, all_)
        except OSError:
            pass


def new_state(path, saved=None):
    saved = saved or {}
    fit = saved.get("fit") or {}
    if "harmonics" in fit:                   # first version: all harmonics in one fit
        fit = {h: dict(r, **{k: fit.get(k) for k in ("file", "f_ref", "ref_name", "ref_harmonic",
                                                      "gamma_t", "ref_search_hz",
                                                      "peak_window_hz", "time")}, h=int(h))
               for h, r in fit["harmonics"].items()}
    return {"file": path or "", "h": saved.get("h"), "k": saved.get("k") or {},
            "offsets": saved.get("offsets") or {}, "ref_override": saved.get("ref_override") or {},
            "assign": saved.get("assign") or {}, "fit": fit, "selected": None}


def _ints(v, n=None, what="value"):
    items = v if isinstance(v, (list, tuple)) else re.split(r"[,\s]+", str(v or "").strip())
    try:
        out = [int(float(x)) for x in items if str(x).strip() != ""]
    except ValueError:
        raise ValueError(f"{what}: expected integers, got {v!r}")
    if n is not None and len(out) != n:
        raise ValueError(f"{what}: expected {n} integers, got {v!r}")
    return out


def _f(v, what, default=None):
    if v is None or str(v).strip() == "":
        if default is not None:
            return float(default)
        raise ValueError(f"{what} is empty")
    try:
        return float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{what}: {v!r} is not a number")


def parse_settings(raw):
    s = {"ref_name": str(raw.get("ref_name") or "").strip()}
    if not s["ref_name"]:
        raise ValueError("reference ion is empty")
    s["ref_harmonic"] = int(_f(raw.get("ref_harmonic"), "reference harmonic"))
    s["gamma_t"] = _f(raw.get("gamma_t"), "γ_t")
    s["harmonics"] = sorted(set(_ints(raw.get("harmonics"), None, "harmonics")))
    if not s["harmonics"]:
        raise ValueError("harmonics list is empty")
    if s["ref_harmonic"] not in s["harmonics"]:
        s["harmonics"] = sorted(s["harmonics"] + [s["ref_harmonic"]])
    for k, n in (("z_lims", 2), ("n_lims", 2), ("el1", 2), ("el2", 2)):
        s[k] = _ints(raw.get(k), n, LABELS[k])
    s["mq_width"] = _f(raw.get("mq_width"), "m/q window")
    s["ref_search_hz"] = _f(raw.get("ref_search_hz"), "reference search window", 0)
    s["peak_window_hz"] = _f(raw.get("peak_window_hz"), "peak window", 100)
    if s["peak_window_hz"] <= 0:
        raise ValueError("the peak window must be > 0")
    return s


def current_h(s, state):
    h = (state or {}).get("h")
    try:
        h = int(h)
    except (TypeError, ValueError):
        h = None
    return h if h in s["harmonics"] else s["ref_harmonic"]


# ===========================================================================
# Spectrum, simulated lines and their positions on the plot
# ===========================================================================

@lru_cache(maxsize=4)
def _load(path, _mtime):
    d = unpack_npz(path)
    ff, pp = np.asarray(d["arr_0"], dtype=float), np.asarray(d["arr_1"], dtype=float)
    if ff.ndim != 1 or len(ff) != len(pp):
        raise ValueError(f"{os.path.basename(path)} is not a 1-D spectrum (arr_0, arr_1)")
    if np.any(np.diff(ff) < 0):
        o = np.argsort(ff)
        ff, pp = ff[o], pp[o]
    return ff, pp


def load_spectrum(path):
    path = paths.remap_legacy(os.path.expanduser(path))
    return _load(path, os.path.getmtime(path))


def spectrum_files():
    """Dropdown options: .npz files in the connected summed_1d and spectra_1d folders."""
    out, seen = [], set()
    folders = paths.connected("data")
    for kind in ("summed_1d", "spectra_1d"):
        for fo in folders:
            if fo["kind"] != kind or not os.path.isdir(fo["path"]):
                continue
            names = sorted(os.listdir(fo["path"]), key=lambda n: [int(t) if t.isdigit() else t
                                                                  for t in re.split(r"(\d+)", n)])
            for n in names:
                fp = os.path.join(fo["path"], n)
                if n.endswith(".npz") and fp not in seen:
                    seen.add(fp)
                    out.append({"label": f"{fo['name']} / {n}", "value": fp})
    return out


def line_key(ion, h):
    return f"{ion}|{int(h)}"


def split_key(key):
    ion, h = key.rsplit("|", 1)
    return ion, int(h)


@lru_cache(maxsize=16)
def _sim_lines(path, mtime, ion_key, f_ref, ref_h, gamma_t, h, f_ref_h):
    """Simulated lines of harmonic h (cached): DataFrame key, ion, h, f_sim, hover, is_ref."""
    ff, _ = _load(path, mtime)
    ion_db, ref_mq = physics.build_ion_database(*ion_key)
    sims = physics.simulate_frequencies(ion_db, ref_mq, f_ref, ref_h, gamma_t, [h], False,
                                        ref_by_h={h: f_ref_h})
    span = float(ff[-1] - ff[0])
    sims = sims[(sims["f_sim"] >= ff[0] - 0.05 * span)
                & (sims["f_sim"] <= ff[-1] + 0.05 * span)].reset_index(drop=True)
    rows = sims["ion_row"].to_numpy()
    names = ion_db["ion"].to_numpy()[rows].astype(str)
    lines = pd.DataFrame({"key": [line_key(n, h) for n in names], "ion": names, "h": int(h),
                          "f_sim": sims["f_sim"].to_numpy(float),
                          "hover": ion_db["hover"].to_numpy()[rows]})
    lines = lines.drop_duplicates("key").sort_values("f_sim").reset_index(drop=True)
    lines["is_ref"] = lines["ion"] == ion_key[4]
    return lines


def model(s, state):
    """Everything the plot and the fit need for the harmonic shown.

    Returns dict: h, ff, pp, f_ref (strongest bin), ref_h_f (reference line
    of h), ref_info, warnings, lines (DataFrame: key, ion, h, f_sim, is_ref,
    hover, f_show, assigned, f_exp, moved)."""
    h = current_h(s, state)
    path = paths.remap_legacy(os.path.expanduser(state["file"]))
    mtime = os.path.getmtime(path)
    ff, pp = _load(path, mtime)
    f_ref = float(ff[int(np.nanargmax(pp))])
    refs, info = physics.harmonic_references(ff, pp, f_ref, s["ref_harmonic"],
                                             sorted({h, s["ref_harmonic"]}), s["ref_search_hz"])
    warnings = physics.reference_warnings({h: info[h]}, s["ref_name"], s["ref_search_hz"])
    over = (state.get("ref_override") or {}).get(str(h))
    if over is not None and h != s["ref_harmonic"]:
        refs[h] = float(over)
        info[h] = dict(info[h], f=float(over), found="moved")
        warnings = []
    ion_key = (tuple(s["z_lims"]), tuple(s["n_lims"]), tuple(s["el1"]), tuple(s["el2"]),
               s["ref_name"], float(s["mq_width"]))
    lines = _sim_lines(path, mtime, ion_key, f_ref, s["ref_harmonic"], s["gamma_t"], h,
                       float(refs[h])).copy()
    k = float((state.get("k") or {}).get(str(h), 1.0))
    offsets = state.get("offsets") or {}
    assign = state.get("assign") or {}
    r = float(refs[h])
    f_sim = lines["f_sim"].to_numpy(float)
    off = np.array([float(offsets.get(q, 0.0)) for q in lines["key"]])
    show = np.where(lines["is_ref"], f_sim, r + k * (f_sim - r) + off)
    lines["assigned"] = [q in assign for q in lines["key"]]
    lines["f_exp"] = [float((assign.get(q) or {}).get("f_exp", np.nan)) for q in lines["key"]]
    lines["f_show"] = np.where(lines["assigned"], lines["f_exp"], show)
    lines["moved"] = [q in offsets for q in lines["key"]]
    return {"h": h, "ff": ff, "pp": pp, "f_ref": f_ref, "ref_h_f": r, "ref_info": info[h],
            "warnings": warnings, "lines": lines, "k": k}


def _vline_xy(xs):
    """Vertical lines 0 → 1 (on the hidden 0..1 axis) as one trace: plain lists."""
    X, Y = [], []
    for x in xs:
        X += [float(x), float(x), None]
        Y += [0.0, 1.0, None]
    return X, Y


LINE_GROUPS = [   # name, line style (WebGL traces 1..5)
    ("simulated", dict(color=LINE_COL, width=1, dash="dash")),
    ("moved", dict(color=LINE_COL, width=1.5, dash="dot")),
    ("assigned", dict(color=ASSIGNED_COL, width=2.5)),
    ("reference", dict(color=ACCENT, width=2.5)),
    ("selected", dict(color="black", width=3)),
]


def overlay(s, state, m, view, mode):
    """The parts of the plot that change: line traces (WebGL, one per group),
    the ▼ marker trace (WebGL), the draggable lines in view (shapes) and the
    labels.  Returns (traces, shapes, annotations, shape keys, note)."""
    lines = m["lines"]
    sel = state.get("selected")
    in_view = np.ones(len(lines), dtype=bool)
    if view:
        w = view[1] - view[0]
        in_view = ((lines["f_show"] >= view[0] - 0.05 * w)
                   & (lines["f_show"] <= view[1] + 0.05 * w)).to_numpy()
    movable = (~lines["assigned"] & (~lines["is_ref"] | ((mode == "one")
                                                        & (m["h"] != s["ref_harmonic"])))).to_numpy()
    cand = in_view & movable
    note = ""
    if cand.sum() > MAX_SHAPES:
        note = (f"{int(cand.sum())} lines in view – zoom in to drag them "
                f"(at most {MAX_SHAPES}; the ▼ markers always select).")
        cand[:] = False
    is_sel = (lines["key"] == sel).to_numpy()
    groups = {
        "simulated": ~cand & ~lines["assigned"] & ~lines["is_ref"] & ~lines["moved"] & ~is_sel,
        "moved": ~cand & ~lines["assigned"] & ~lines["is_ref"] & lines["moved"] & ~is_sel,
        "assigned": lines["assigned"] & ~is_sel,
        "reference": lines["is_ref"] & ~cand & ~is_sel,
        "selected": is_sel & ~cand,
    }
    traces = []
    for name, style in LINE_GROUPS:
        X, Y = _vline_xy(lines["f_show"][np.asarray(groups[name])])
        traces.append(go.Scattergl(x=X, y=Y, yaxis="y3", mode="lines", line=style, name=name,
                                   hoverinfo="skip", showlegend=False,
                                   connectgaps=False).to_plotly_json())
    # same tooltip as the simulated lines in Theia's Viewer / the spectrogram
    # (ion, m.e., T½, E_x, decay, harmonic, f_sim), then this tool's extras
    hov = [f"{hv}<br>Harmonic: {r.h}<br>f_sim = {r.f_sim:,.1f} Hz"
           + (f"<br>shown at {r.f_show:,.1f} Hz" if not r.is_ref and not r.assigned
              and abs(r.f_show - r.f_sim) > 0.05 else "")
           + (f"<br><b>assigned: f_exp = {r.f_exp:,.1f} Hz</b>" if r.assigned else "")
           + ("<br><b>reference line</b>" if r.is_ref else "")
           for hv, r in zip(lines["hover"], lines.itertuples(index=False))]
    colours = [("black" if q == sel else ACCENT if ref else ASSIGNED_COL if a else LINE_COL)
               for q, ref, a in zip(lines["key"], lines["is_ref"], lines["assigned"])]
    traces.append(go.Scattergl(
        x=lines["f_show"].tolist(), y=[0.5] * len(lines), yaxis="y2", mode="markers",
        name="lines", customdata=lines["key"].tolist(), hovertext=hov, hoverinfo="text",
        showlegend=False,
        marker=dict(symbol="triangle-down", color=colours,
                    size=[15 if q == sel else 10 for q in lines["key"]])).to_plotly_json())
    shapes, keys = [], []
    for r in lines[cand].itertuples(index=False):
        ln = (dict(color=ACCENT, width=2.5) if r.is_ref else
              dict(color=LINE_COL, width=1.5, dash="dot") if r.moved else
              dict(color=LINE_COL, width=1, dash="dash"))
        if r.key == sel:
            ln = dict(color="black", width=3)
        shapes.append(dict(type="line", xref="x", yref="paper", x0=r.f_show, x1=r.f_show,
                           y0=0, y1=Y_TOP, line=ln, layer="above"))
        keys.append(r.key)
    ann = []
    lab = lines[in_view & (lines["assigned"] | lines["is_ref"] | is_sel).to_numpy()]
    if len(lab) > MAX_LABELS:
        lab = lab[lab["key"] == sel] if sel else lab.iloc[:0]
    for r in lab.itertuples(index=False):
        ann.append(dict(x=r.f_show, y=Y_TOP - 0.005, xref="x", yref="paper", yanchor="top",
                        text=f"{r.ion} h{r.h}", showarrow=False, textangle=-90,
                        font=dict(size=11, color="black" if r.key == sel else
                                  (ACCENT if r.is_ref else ASSIGNED_COL)),
                        bgcolor="rgba(255,255,255,0.8)"))
    # tooltips along the whole height of the lines in view: one invisible bar
    # per line (hover anywhere on the bar; one label per line, not one per
    # point, so the patch stays small).  ~8 px wide at a 1000 px plot.
    span = (view[1] - view[0]) if view else float(m["ff"][-1] - m["ff"][0])
    hv_in = in_view if in_view.sum() <= MAX_HOVER_LINES else np.zeros(len(lines), dtype=bool)
    if not hv_in.any() and in_view.sum() > MAX_HOVER_LINES:
        note = (note + " " if note else "") + (f"Zoom in for the tooltips on the lines "
                                                f"(at most {MAX_HOVER_LINES} in view).")
    idx = np.where(hv_in)[0]
    traces.append(go.Bar(
        x=lines["f_show"].to_numpy()[idx].tolist(), y=[1.0] * len(idx),
        width=[span / 1000.0 * 8] * len(idx), yaxis="y3", name="line tooltips",
        hovertext=[hov[i] for i in idx], hoverinfo="text", showlegend=False,
        marker=dict(color="rgba(0,0,0,0)", line=dict(width=0))).to_plotly_json())
    return traces, shapes, ann, keys, note


def figure(s, state, m, view, mode):
    """Full plot: spectrum (trace 0, no hover) + overlay."""
    traces, shapes, ann, keys, note = overlay(s, state, m, view, mode)
    fig = go.Figure()
    # no Plotly hover on the spectrum: the readout shows f and P
    fig.add_trace(go.Scattergl(x=m["ff"], y=m["pp"], mode="lines", name="Spectrum",
                               hoverinfo="skip", line=dict(shape="hvh", color="black", width=1)))
    for t in traces:
        fig.add_trace(t)
    fig.update_layout(
        shapes=shapes, annotations=ann, template="plotly_white", dragmode="pan",
        hovermode="closest", hoverdistance=12, font=dict(size=16), height=660,
        barmode="overlay", bargap=0,
        showlegend=False, margin=dict(l=70, r=20, t=20, b=50),
        xaxis=dict(title="Frequency (Hz)", tickformat=".0f"),
        yaxis=dict(title="Schottky power (a.u.)", domain=[0, Y_TOP]),
        # the ▼ markers: a strip of their own above the spectrum, so a click on
        # a marker is not taken by the (draggable) line under it
        yaxis2=dict(domain=[Y_TOP + 0.01, 0.99], range=[0, 1], visible=False, fixedrange=True,
                    anchor="x"),
        # the lines: 0..1 over the spectrum, whatever its y zoom
        yaxis3=dict(overlaying="y", range=[0, 1], visible=False, fixedrange=True),
        uirevision=state.get("file") or "none")
    return fig, keys, note


# ===========================================================================
# The fit (one harmonic) and the profiles
# ===========================================================================

def fit_harmonic(s, state, m):
    """Anchored fit of the harmonic shown.  Returns a JSON-able dict; raises
    ValueError if there are not enough assigned ions."""
    h = m["h"]
    lines = m["lines"].set_index("key")
    pts, missing = [], []
    for key, a in (state.get("assign") or {}).items():
        if split_key(key)[1] != h:
            continue
        if key not in lines.index:
            missing.append(key)
            continue
        r = lines.loc[key]
        if not r["is_ref"]:
            pts.append((key, r["ion"], float(r["f_sim"]), float(a["f_exp"])))
    if len(pts) < MIN_POINTS:
        raise ValueError(f"h{h} has {len(pts)} assigned ion(s); the fit needs {MIN_POINTS} "
                         f"besides the reference")
    x = np.array([p[2] for p in pts]) / h
    y = np.array([p[3] for p in pts]) / h - x
    x0 = m["ref_h_f"] / h
    res = corrections.fit_correction(x, y, x0)
    return {"h": h, "n": res["n"], "abc": list(res["abc"]), "abc_unc": list(res["abc_unc"]),
            "s1": res["s1"], "s2": res["s2"], "ds1": res["ds1"], "ds2": res["ds2"],
            "x0": x0, "ref_f_h": m["ref_h_f"], "rms": res["rms"],
            "ref_found": m["ref_info"].get("found"), "stretch": m["k"], "missing": missing,
            "file": state["file"], "f_ref": m["f_ref"], "ref_name": s["ref_name"],
            "ref_harmonic": s["ref_harmonic"], "gamma_t": s["gamma_t"],
            "ref_search_hz": s["ref_search_hz"], "peak_window_hz": s["peak_window_hz"],
            "time": datetime.datetime.now().isoformat(timespec="seconds"),
            "points": [{"key": p[0], "ion": p[1], "f_sim": p[2], "f_exp": p[3], "x": float(xi),
                        "y": float(yi), "resid": float(ri)}
                       for p, xi, yi, ri in zip(pts, x, y, res["resid"])]}


REF_PEAK_TEXT = {True: "peak found", False: "predicted (no peak)", None: "f_ref/h_ref·h",
                 "moved": "moved by hand"}


def fit_figure(r):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
                        vertical_spacing=0.04)
    x = np.array([p["x"] for p in r["points"]])
    y = np.array([p["y"] for p in r["points"]])
    names = [p["ion"] for p in r["points"]]
    fig.add_trace(go.Scatter(x=x.tolist(), y=y.tolist(), mode="markers", text=names,
                             name="data", marker=dict(color=LINE_COL, size=9),
                             hovertemplate="%{text}<br>f_sim = %{x:.3f} Hz<br>"
                                           "f_diff = %{y:.3f} Hz<extra></extra>"), row=1, col=1)
    lo, hi = min(x.min(), r["x0"]), max(x.max(), r["x0"])
    pad = 0.05 * (hi - lo or 1)
    xx = np.linspace(lo - pad, hi + pad, 300)
    fig.add_trace(go.Scatter(x=xx.tolist(),
                             y=(r["s1"] * (xx - r["x0"]) + r["s2"] * (xx - r["x0"]) ** 2).tolist(),
                             mode="lines", line=dict(color=ACCENT, width=2), name="fit",
                             hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[r["x0"]], y=[0], mode="markers", name="anchor",
                             marker=dict(symbol="star", size=16, color=ACCENT,
                                         line=dict(color="black", width=1)),
                             hovertemplate="anchor (reference line)<br>x0 = %{x:.3f} Hz"
                                           "<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x.tolist(), y=[p["resid"] for p in r["points"]], mode="markers",
                             marker=dict(color=LINE_COL, size=8), showlegend=False, text=names,
                             hovertemplate="%{text}<br>residual = %{y:.3f} Hz<extra></extra>"),
                  row=2, col=1)
    fig.add_hline(y=0, line=dict(color="grey", width=1, dash="dash"), row=1, col=1)
    fig.add_hline(y=0, line=dict(color=ACCENT, width=1), row=2, col=1)
    fig.update_layout(template="plotly_white", height=600, font=dict(size=15),
                      margin=dict(l=70, r=20, t=30, b=50), legend=dict(orientation="v"),
                      title=dict(text=f"h{r['h']}", x=0.01, font=dict(size=16)))
    fig.update_yaxes(title_text="f_diff = f_exp,true − f_sim,true (Hz)", row=1, col=1)
    fig.update_yaxes(title_text="Residual (Hz)", row=2, col=1)
    fig.update_xaxes(title_text="f_sim,true (Hz, revolution)", tickformat=".0f", row=2, col=1)
    return fig


def fit_table(r, others=()):
    (a, b, c), (da, db, dc) = r["abc"], r["abc_unc"]
    head = ["Harmonic", "Ions", "a", "b", "c", "RMS (Hz, rev)", "Reference line"]
    row = [f"h{r['h']}" + (" (reference harmonic)" if r["h"] == r["ref_harmonic"] else ""), r["n"],
           f"{a:.6e} ± {da:.2e}", f"{b:.6e} ± {db:.2e}", f"{c:.6e} ± {dc:.2e}",
           f"{r['rms']:.3f}", f"{r['ref_f_h']:,.1f} Hz · {REF_PEAK_TEXT.get(r.get('ref_found'), '')}"]
    out = [html.Table([html.Thead(html.Tr([html.Th(x) for x in head])),
                       html.Tbody([html.Tr([html.Td(x) for x in row])])], className="tbl")]
    if r.get("missing"):
        out.append(html.Div("Not used (not in the simulation): " + ", ".join(
            q.replace("|", " h") for q in r["missing"]), className="warn"))
    if others:
        out.append(html.Div("Also calculated for this spectrum: " + ", ".join(others),
                            className="note"))
    return out


def default_new_name(h):
    return f"{datetime.datetime.now():%Y-%m-%d %H:%M} GUI per-harmonic"


def _fit_info(r):
    return {"ref_ion": r["ref_name"], "ref_f_exp": r["f_ref"], "ref_harmonic": r["ref_harmonic"],
            "harmonic": r["h"], "ref_f_h": r["ref_f_h"], "f_ref_rev": r["x0"],
            "gamma_t": r["gamma_t"], "mass_model": MASS_MODEL, "weighted": False,
            "n_points": r["n"], "rms_rev_hz": r["rms"], "chi2_ndf": None,
            "ions": [f"{p['ion']} h{r['h']}" for p in r["points"]],
            "ref_peak": REF_PEAK_TEXT.get(r.get("ref_found"), str(r.get("ref_found"))),
            "stretch": r.get("stretch"), "source_file": r["file"], "source": APP_NAME,
            "ref_search_hz": r["ref_search_hz"], "peak_window_hz": r["peak_window_hz"],
            "saved": datetime.datetime.now().isoformat(timespec="seconds")}


def save_harmonic(r, target, new_name, confirm_replace=False):
    """Put the fit r of one harmonic into a profile: a new one (target
    NEW_PROFILE, name new_name) or an existing one (target = its name).
    Returns (profile, replaced).  Raises ValueError (e.g. 'exists: confirm')."""
    h = str(r["h"])
    (a, b, c), (da, db, dc) = r["abc"], r["abc_unc"]
    entry = {"a": a, "b": b, "c": c, "a_unc": da, "b_unc": db, "c_unc": dc, "fit": _fit_info(r)}
    is_ref = r["h"] == r["ref_harmonic"]
    if target == NEW_PROFILE:
        name = (new_name or "").strip()
        if not name:
            raise ValueError("give the new profile a name")
        if corrections.get_profile(name) is not None:
            raise ValueError(f"a profile called '{name}' already exists (pick it in the list "
                             f"to add h{h} to it)")
        top = dict(_fit_info(r), per_harmonic=True)
        p = {"name": name, "a": a if is_ref else 0.0, "b": b if is_ref else 0.0,
             "c": c if is_ref else 0.0, "created": datetime.datetime.now().isoformat(
                 timespec="seconds"), "fit": top, "per_harmonic": {h: entry},
             "notes": "Made in the GUI correction tool, one harmonic at a time."
                      + ("" if is_ref else f" General a, b, c = 0 (no correction) until "
                                           f"h{r['ref_harmonic']} is saved.")}
        if is_ref:
            p.update(a_unc=da, b_unc=db, c_unc=dc)
        return corrections.save_profile(p), False
    p = corrections.get_profile(target)
    if p is None:
        raise ValueError(f"the profile '{target}' no longer exists")
    per_h = dict(p.get("per_harmonic") or {})
    replaced = h in per_h
    if replaced and not confirm_replace:
        raise ValueError("confirm")
    per_h[h] = entry
    p = dict(p, per_harmonic=per_h)
    if is_ref:                       # the reference harmonic is also the general a, b, c
        p.update(a=a, b=b, c=c, a_unc=da, b_unc=db, c_unc=dc,
                 fit=dict(_fit_info(r), per_harmonic=True))
        p["notes"] = (p.get("notes") or "").replace(
            f" General a, b, c = 0 (no correction) until h{r['ref_harmonic']} is saved.", "")
    else:
        p["fit"] = dict(p.get("fit") or {}, per_harmonic=True)
    return corrections.save_profile(p, overwrite=True), replaced


def profile_options():
    return ([{"label": "➕ New profile", "value": NEW_PROFILE}]
            + [{"label": q["name"] + ("  ·  " + ", ".join(
                f"h{x}" for x in sorted(q.get("per_harmonic") or {}, key=int))
                if q.get("per_harmonic") else "  ·  all harmonics"), "value": q["name"]}
               for q in corrections.load_profiles()])


# ===========================================================================
# Page
# ===========================================================================

CSS = f"""
body {{ font-family: system-ui, sans-serif; margin: 0; background: #fff; color: #222; }}
.top {{ display: flex; align-items: center; gap: 16px; padding: 10px 18px;
        border-bottom: 3px solid {ACCENT}; }}
.top h1 {{ font-size: 20px; margin: 0; }}
.top .exp {{ color: #666; font-size: 13px; }}
.wrap {{ display: grid; grid-template-columns: 300px minmax(0, 1fr); min-height: calc(100vh - 50px); }}
.side {{ background: {SIDE}; border-right: 1px solid {LINE}; padding: 12px 14px; font-size: 13px; }}
.side h3 {{ font-size: 13px; text-transform: uppercase; letter-spacing: .04em; color: {ACCENT};
           margin: 14px 0 6px; border-bottom: 1px solid {LINE}; padding-bottom: 3px; }}
.side label {{ display: block; margin: 6px 0 2px; color: #444; }}
.side input[type=text] {{ width: 100%; box-sizing: border-box; padding: 3px 6px;
                         border: 1px solid #bbb; border-radius: 3px; font-size: 13px; }}
.harm label {{ display: inline-block; margin: 2px 12px 2px 0; font-size: 15px; font-weight: 600; }}
.main {{ padding: 12px 18px; min-width: 0; }}
.box {{ border: 1px solid black; border-radius: 4px; margin-bottom: 14px; }}
.box > .head {{ background: {PALE}; border-bottom: 1px solid {LINE}; padding: 6px 10px;
               font-weight: 600; display: flex; justify-content: space-between; gap: 10px;
               flex-wrap: wrap; align-items: center; }}
.box > .body {{ padding: 8px 10px; }}
.btn {{ background: #fff; border: 1px solid #999; border-radius: 3px; padding: 4px 10px;
        font-size: 13px; cursor: pointer; margin: 2px 4px 2px 0; }}
.btn:hover {{ border-color: {ACCENT}; }}
.btn.primary {{ background: {ACCENT}; border-color: {ACCENT}; color: #fff; }}
.row {{ display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }}
.note {{ color: #555; font-size: 12px; }}
.warn {{ color: #b45309; font-size: 12px; }}
.ok {{ color: #15803d; font-size: 13px; }}
.err {{ color: #b91c1c; font-size: 13px; }}
.tbl {{ border-collapse: collapse; font-size: 13px; width: 100%; }}
.tbl th {{ background: {ACCENT}; color: #fff; text-align: left; padding: 4px 6px; }}
.tbl td {{ border-bottom: 1px solid {LINE}; padding: 3px 6px; }}
.tbl tr:nth-child(even) td {{ background: {PALE}; }}
.sel {{ font-size: 13px; padding: 4px 0; }}
#g-readout {{ position: fixed; bottom: 12px; right: 16px; background: rgba(30,30,30,0.80);
             color: #e8e8e8; padding: 5px 12px; border-radius: 5px; font-family: monospace;
             font-size: 15px; pointer-events: none; z-index: 9999; white-space: pre;
             display: none; }}
"""

app = dash.Dash(__name__, title=APP_NAME, suppress_callback_exceptions=True)
app.index_string = app.index_string.replace("</head>", f"<style>{CSS}</style></head>")
app.index_string = app.index_string.replace(
    "{%scripts%}", "{%scripts%}\n<script>" + tabwatch.watch_js(APP_NAME) + "</script>")
icon.use_in_dash(app, "gui_correction")      # core/icons/gui_correction.png, else asphodel.png


def _harm_options(cfg):
    try:
        s = parse_settings(cfg)
    except ValueError:
        return [], None
    return [{"label": f" h{h}" + (" (ref)" if h == s["ref_harmonic"] else ""), "value": h}
            for h in s["harmonics"]], s


def serve_layout():
    cfg = load_settings()
    files = spectrum_files()
    values = [o["value"] for o in files]
    f = cfg.get("file") if cfg.get("file") in values else (values[0] if values else "")
    state = load_session(f) if f else new_state("")
    hopts, s = _harm_options(cfg)
    h = current_h(s, state) if s else None
    state["h"] = h
    fields = []
    for k in SETTING_KEYS:
        v = cfg[k]
        v = ", ".join(str(x) for x in v) if isinstance(v, list) else physics._fmt_num(v) \
            if isinstance(v, float) else str(v)
        fields += [html.Label(LABELS[k]), dcc.Input(id=f"g-{k}", type="text", value=v, debounce=True)]
    return html.Div([
        dcc.Store(id="g-state", data=state),
        dcc.Store(id="g-view", data=None),       # set by the page when the view is still
        dcc.Store(id="g-click", data=None),      # click on the spectrum (set by the page)
        dcc.Store(id="g-drag", data=None),       # line (shape) drags (set by the page)
        dcc.Store(id="g-keys", data=[]),
        dcc.Store(id="g-sig", data=None),
        dcc.Store(id="g-fitstamp", data=0),
        dcc.Store(id="g-pconfirm", data=None),
        html.Div(id="_g-hook", style={"display": "none"}),
        html.Div(id="g-readout"),
        html.Div([html.H1(APP_NAME), html.Span(f"experiment {paths.experiment()}", className="exp")],
                 className="top"),
        html.Div([
            html.Div([
                html.H3("Spectrum"),
                dcc.Dropdown(id="g-file", options=files, value=f or None, clearable=False,
                             placeholder="No .npz in the connected spectra folders",
                             style={"fontSize": "12px"}),
                html.H3("Harmonic"),
                dcc.RadioItems(id="g-h", options=hopts, value=h, className="harm"),
                html.Div("Each harmonic keeps its own stretch, moved lines, assignments and fit.",
                         className="note"),
                html.H3("Dragging"),
                dcc.RadioItems(id="g-mode", value="stretch", options=[
                    {"label": " Stretch all about the reference", "value": "stretch"},
                    {"label": " Move one line", "value": "one"}],
                    labelStyle={"display": "block"}),
                html.Div(id="g-stretch", className="note"),
                html.Button("Reset stretch & moves (this harmonic)", id="g-reset", className="btn",
                            n_clicks=0),
                html.H3("Simulation"),
                *fields,
                html.Div("Shared defaults from physics.json; changes here are kept for this tool "
                         "only.  No correction is applied to the lines.", className="note"),
            ], className="side"),
            html.Div([
                html.Div([
                    html.Div([html.Span(id="g-title"), html.Span(id="g-info", className="note")],
                             className="head"),
                    html.Div([
                        html.Div(id="g-warn", className="warn"),
                        dcc.Graph(id="g-graph", clear_on_unhover=True,
                                  config={"edits": {"shapePosition": True}, "scrollZoom": True,
                                          "displaylogo": False, "doubleClick": "reset"}),
                        html.Div(id="g-sel", className="sel"),
                        html.Div([
                            html.Button("Assign to nearest peak", id="g-assign", className="btn primary",
                                        n_clicks=0),
                            html.Button("Unassign", id="g-unassign", className="btn", n_clicks=0),
                            html.Button("Deselect", id="g-deselect", className="btn", n_clicks=0),
                            html.Span(id="g-msg", className="note"),
                        ], className="row"),
                        html.Div("Click a ▼ marker to select its line, then click on the peak in "
                                 "the spectrum (or 'Assign to nearest peak').  Zoom in to drag "
                                 "lines; scroll to zoom, drag the background to pan, double-click "
                                 "for the whole spectrum.", className="note"),
                    ], className="body"),
                ], className="box"),
                html.Div([
                    html.Div([html.Span(id="g-assign-head", children="Assignments"),
                              html.Div([html.Button("Calculate correction", id="g-calc",
                                                    className="btn primary", n_clicks=0),
                                        html.Button("Clear assignments (this harmonic)",
                                                    id="g-clear", className="btn", n_clicks=0)])],
                             className="head"),
                    html.Div([
                        dash_table.DataTable(
                            id="g-table", row_deletable=True, page_size=15, data=[],
                            columns=[{"name": n, "id": i} for n, i in (
                                ("Ion", "ion"), ("h", "h"), ("f_sim (Hz)", "f_sim"),
                                ("f_exp (Hz)", "f_exp"), ("f_exp − f_sim (Hz)", "diff"),
                                ("f_diff rev (Hz)", "diff_rev"), ("", "note"))],
                            style_cell={"fontSize": "13px", "fontFamily": "system-ui",
                                        "padding": "3px 6px", "textAlign": "left"},
                            style_header={"backgroundColor": ACCENT, "color": "white",
                                          "fontWeight": "600"},
                            style_data_conditional=[{"if": {"row_index": "odd"},
                                                     "backgroundColor": PALE}]),
                        html.Div(id="g-calc-msg", className="note"),
                    ], className="body"),
                ], className="box"),
                html.Div([
                    html.Div([html.Span(id="g-fit-head", children="Second-order correction"),
                              html.Span(id="g-fit-when", className="note")], className="head"),
                    html.Div([
                        html.Div("Not calculated yet.", id="g-fit-table"),
                        dcc.Graph(id="g-fit-graph", figure=go.Figure(), style={"display": "none"}),
                        html.Div([
                            html.Label("Save into", style={"fontWeight": 600}),
                            dcc.Dropdown(id="g-ptarget", options=profile_options(),
                                         value=NEW_PROFILE, clearable=False,
                                         style={"width": "460px", "fontSize": "13px"}),
                            dcc.Input(id="g-pname", type="text", style={"width": "300px"},
                                      value=default_new_name(h), placeholder="new profile name"),
                        ], className="row"),
                        html.Div([
                            dcc.Checklist(id="g-pdefault", value=[], options=[
                                {"label": " also make the profile the default", "value": "on"}]),
                            html.Button(id="g-psave", className="btn primary", n_clicks=0,
                                        children=f"Save h{h} into profile"),
                        ], className="row"),
                        html.Div(id="g-pmsg"),
                    ], className="body"),
                ], className="box"),
            ], className="main"),
        ], className="wrap"),
    ])


app.layout = serve_layout
SETTING_INPUTS = [Input(f"g-{k}", "value") for k in SETTING_KEYS]
SETTING_STATES = [State(f"g-{k}", "value") for k in SETTING_KEYS]


def _settings(values):
    return parse_settings(dict(zip(SETTING_KEYS, values)))


@app.callback(Output("g-msg", "children", allow_duplicate=True), *SETTING_INPUTS,
              prevent_initial_call=True)
def persist_settings(*values):
    save_settings(dict(zip(SETTING_KEYS, values)))
    return dash.no_update


@app.callback(Output("g-h", "options"), *SETTING_INPUTS, prevent_initial_call=True)
def harmonic_options(*values):
    opts, s = _harm_options(dict(zip(SETTING_KEYS, values)))
    return opts if s else dash.no_update


# ---------------------------------------------------------------------------
# Page-side hooks: readout, view (when still), clicks on the spectrum
# ---------------------------------------------------------------------------

app.clientside_callback(
    """
    function(fig) {
        var tries = 0;
        function setup() {
            var gd = document.querySelector('#g-graph .js-plotly-plot');
            if (!gd || typeof gd.on !== 'function' || !gd._fullLayout) {
                if (tries++ < 50) { setTimeout(setup, 100); }
                return;
            }
            if (gd._gcHooks) return;
            gd._gcHooks = true;
            var readout = document.getElementById('g-readout');
            var lastWheel = 0, held = false, timer = null, lastView = null, down = null;

            function spectrum() {
                var fd = gd._fullData || [];
                for (var i = 0; i < fd.length; i++) if (fd[i].name === 'Spectrum') return fd[i];
                return null;
            }
            function dataX(evt) {
                var fl = gd._fullLayout, r = gd.getBoundingClientRect();
                return fl.xaxis.p2d(evt.clientX - r.left - fl.xaxis._offset);
            }
            function inMain(evt) {          // inside the spectrum area (not the marker strip)
                var fl = gd._fullLayout, r = gd.getBoundingClientRect();
                var px = evt.clientX - r.left, py = evt.clientY - r.top;
                return px >= fl.xaxis._offset && px <= fl.xaxis._offset + fl.xaxis._length &&
                       py >= fl.yaxis._offset && py <= fl.yaxis._offset + fl.yaxis._length;
            }
            // view -> server only when it has been still 0.4 s with no button held
            function check() {
                timer = null;
                if (held || Date.now() - lastWheel < 400) { soon(); return; }
                var xa = gd._fullLayout && gd._fullLayout.xaxis;
                if (!xa) return;
                var v = [Number(xa.range[0]), Number(xa.range[1])];
                var key = v[0].toFixed(1) + '|' + v[1].toFixed(1);
                if (key === lastView) return;
                lastView = key;
                dash_clientside.set_props('g-view', {data: v});
            }
            function soon() { if (timer) clearTimeout(timer); timer = setTimeout(check, 400); }
            gd.addEventListener('wheel', function() { lastWheel = Date.now(); soon(); },
                                {passive: true});
            gd.addEventListener('mousedown', function(evt) {
                held = true;
                down = {x: evt.clientX, y: evt.clientY,
                        main: inMain(evt)};
            }, true);          // capture: shapes and Plotly's drag layer stop the event
            window.addEventListener('mouseup', function(evt) {
                if (!held) return;
                held = false;
                var d = down; down = null;
                // a click (no movement) on the spectrum, also on a line drawn over it
                if (d && d.main && Math.abs(evt.clientX - d.x) < 4 &&
                        Math.abs(evt.clientY - d.y) < 4) {
                    var x = dataX(evt);
                    if (isFinite(x)) dash_clientside.set_props('g-click', {data: {x: x, t: Date.now()}});
                }
                soon();
            }, true);
            gd.on('plotly_relayout', function(ev) {
                // shape drags go to the server; zoom / pan relayouts do not (the view
                // is sent by check() once it is still)
                var mv = {}, any = false;
                for (var k in (ev || {})) {
                    if (/^shapes\\[\\d+\\]\\.x[01]$/.test(k)) { mv[k] = ev[k]; any = true; }
                }
                if (any) dash_clientside.set_props('g-drag', {data: {moves: mv, t: Date.now()}});
                soon();
            });
            // readout: f and P at the mouse (no Plotly hover on the spectrum)
            gd.addEventListener('mousemove', function(evt) {
                var sp = spectrum();
                if (!sp || !sp.x || !sp.x.length || !inMain(evt)) { readout.style.display = 'none'; return; }
                var x = dataX(evt);
                if (!isFinite(x)) return;
                var xs = sp.x, lo = 0, hi = xs.length - 1;
                while (lo < hi) { var mid = (lo + hi) >> 1; if (xs[mid] < x) lo = mid + 1; else hi = mid; }
                if (lo > 0 && Math.abs(xs[lo - 1] - x) < Math.abs(xs[lo] - x)) lo -= 1;
                readout.textContent = 'f = ' + xs[lo].toFixed(1) + ' Hz   P = ' +
                                      Number(sp.y[lo]).toPrecision(4);
                readout.style.display = 'block';
            });
            gd.addEventListener('mouseleave', function() { readout.style.display = 'none'; });
            // the draggable lines (shapes) take the mouse, so Plotly would not show
            // their tooltip: show the line's tooltip (invisible bar) by hand
            var lastShape = null;
            gd.addEventListener('mousemove', function(evt) {
                var el = evt.target.closest && evt.target.closest('.shapelayer path');
                if (!el) { lastShape = null; return; }
                var g = el.closest('[data-index]');
                if (!g) return;
                var i = Number(g.getAttribute('data-index'));
                var shp = (gd.layout.shapes || [])[i];
                if (!shp) return;
                var fd = gd._fullData || [], ti = -1;
                for (var k = 0; k < fd.length; k++) if (fd[k].name === 'line tooltips') { ti = k; break; }
                if (ti < 0 || !fd[ti].x || !fd[ti].x.length) return;
                var xs = fd[ti].x, best = 0, bd = Infinity;
                for (var j = 0; j < xs.length; j++) {
                    var d = Math.abs(xs[j] - shp.x0);
                    if (d < bd) { bd = d; best = j; }
                }
                if (lastShape === i) return;
                lastShape = i;
                try { Plotly.Fx.hover(gd, [{curveNumber: ti, pointNumber: best}], 'xy3'); } catch (e) {}
            });
            soon();
        }
        setup();
        return '';
    }
    """,
    Output("_g-hook", "children"),
    Input("g-graph", "figure"),
)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

@app.callback(
    Output("g-graph", "figure"), Output("g-keys", "data"), Output("g-title", "children"),
    Output("g-info", "children"), Output("g-warn", "children"), Output("g-sel", "children"),
    Output("g-table", "data"), Output("g-stretch", "children"), Output("g-sig", "data"),
    Output("g-assign-head", "children"),
    Input("g-state", "data"), Input("g-view", "data"), Input("g-mode", "value"), *SETTING_INPUTS,
    State("g-sig", "data"),
)
def draw(state, view, mode, *values):
    values, sig_old = values[:-1], values[-1]
    nu = dash.no_update
    if not state or not state.get("file"):
        return go.Figure(), [], "No spectrum", "", "", "", [], "", None, "Assignments"
    title = os.path.basename(state["file"])
    try:
        s = _settings(values)
        m = model(s, state)
    except Exception as exc:
        return go.Figure(), [], title, "", f"⚠ {exc}", "", nu, "", None, nu
    h = m["h"]
    sig = [state["file"]]
    if sig == sig_old:
        # same spectrum: only the lines, markers and labels are sent
        traces, shapes, ann, keys, note = overlay(s, state, m, view, mode)
        fig = dash.Patch()
        for j, t in enumerate(traces):
            fig["data"][1 + j] = t
        fig["layout"]["shapes"] = shapes
        fig["layout"]["annotations"] = ann
    else:
        fig, keys, note = figure(s, state, m, view, mode)
    lines = m["lines"].set_index("key")
    info = (f"h{h} · reference {s['ref_name']} h{s['ref_harmonic']} = strongest peak at "
            f"{m['f_ref']:,.1f} Hz · {len(lines)} lines")
    found = m["ref_info"].get("found")
    warn = [html.Div("⚠ " + w) for w in m["warnings"]] + ([html.Div(note)] if note else [])
    warn.append(html.Div(f"Reference line h{h}: {m['ref_h_f']:,.1f} Hz"
                         + {True: "" if h == s["ref_harmonic"] else " (peak found)",
                            False: " (predicted)", None: "", "moved": " (moved by hand)"}[found],
                         className="note"))
    sel = state.get("selected")
    if sel and sel in lines.index:
        r = lines.loc[sel]
        txt = [html.B(f"Selected: {r['ion']} h{r['h']}"), f" · f_sim {r['f_sim']:,.1f} Hz"]
        if r["is_ref"]:
            txt.append(" · reference line" + (" (fixed)" if h == s["ref_harmonic"] else ""))
        elif r["assigned"]:
            txt.append(f" · assigned to {r['f_exp']:,.1f} Hz "
                       f"(f_exp − f_sim = {r['f_exp'] - r['f_sim']:+.1f} Hz)")
        else:
            f, _ = physics.peak_near(m["ff"], m["pp"], r["f_show"], s["peak_window_hz"])
            txt.append(f" · shown at {r['f_show']:,.1f} Hz · nearest peak "
                       + (f"{f:,.1f} Hz" if f is not None else
                          f"none within ±{s['peak_window_hz']:g} Hz"))
    else:
        txt = "No line selected."
    rows = []
    mine = [(q, a) for q, a in (state.get("assign") or {}).items() if split_key(q)[1] == h]
    for key, a in sorted(mine, key=lambda kv: kv[1]["f_exp"]):
        ion, hh = split_key(key)
        if key in lines.index:
            fs = float(lines.loc[key, "f_sim"])
            rows.append({"key": key, "ion": ion, "h": hh, "f_sim": f"{fs:,.1f}",
                         "f_exp": f"{a['f_exp']:,.1f}", "diff": f"{a['f_exp'] - fs:+.1f}",
                         "diff_rev": f"{(a['f_exp'] - fs) / hh:+.3f}", "note": ""})
        else:
            rows.append({"key": key, "ion": ion, "h": hh, "f_sim": "", "f_exp": f"{a['f_exp']:,.1f}",
                         "diff": "", "diff_rev": "", "note": "not in the simulation (settings?)"})
    n_off = sum(1 for q in (state.get("offsets") or {}) if split_key(q)[1] == h)
    stretch = f"h{h} stretch ×{m['k']:.6f}" + (f" · {n_off} line(s) moved on their own"
                                                 if n_off else "")
    head = f"Assignments h{h} ({len(rows)})"
    return fig, keys, title, info, warn, txt, rows, stretch, sig, head


# ---------------------------------------------------------------------------
# State changes: dragging, clicks, buttons, table deletions, file / harmonic
# ---------------------------------------------------------------------------

def _shape_moves(relayout):
    moved = {}
    for k, v in (relayout or {}).items():
        mt = re.fullmatch(r"shapes\[(\d+)\]\.(x0|x1)", k)
        if mt:
            moved.setdefault(int(mt.group(1)), {})[mt.group(2)] = float(v)
    return moved


def _drag(moved, keys, state, s, m, mode):
    """Apply shape drags.  Returns a message."""
    lines = m["lines"].set_index("key")
    h = m["h"]
    msg = ""
    changed = [False]
    for idx, xs in moved.items():
        if idx >= len(keys) or keys[idx] not in lines.index:
            continue
        key = keys[idx]
        r = lines.loc[key]
        old = float(r["f_show"])
        x_new = max(xs.values(), key=lambda v: abs(v - old))
        if abs(x_new - old) < 1e-3:        # a click on the line, not a drag
            continue
        changed[0] = True
        ref_h = m["ref_h_f"]
        if r["is_ref"]:
            if h == s["ref_harmonic"]:
                msg = f"The reference line h{h} is the strongest peak and stays fixed."
            elif mode == "one":
                f, _ = physics.peak_near(m["ff"], m["pp"], x_new, s["peak_window_hz"])
                state.setdefault("ref_override", {})[str(h)] = float(f if f is not None else x_new)
                msg = (f"Reference line h{h} moved to " + (f"the peak at {f:,.1f} Hz" if f is not None
                                                           else f"{x_new:,.1f} Hz (no peak there)"))
            continue
        if r["assigned"]:
            continue
        if mode == "one":
            base = ref_h + m["k"] * (r["f_sim"] - ref_h)
            state.setdefault("offsets", {})[key] = float(x_new - base)
            continue
        d = float(r["f_sim"] - ref_h)
        if abs(d) < 1e-6:
            continue
        kn = (x_new - float((state.get("offsets") or {}).get(key, 0.0)) - ref_h) / d
        if not np.isfinite(kn) or kn <= 0:
            msg = "A line cannot be dragged across its reference line."
            continue
        state["k"] = dict(state.get("k") or {}, **{str(h): float(kn)})
        msg = f"h{h} stretch ×{kn:.6f}"
    return msg if changed[0] else None


def _assign(state, s, m, key, centre):
    lines = m["lines"].set_index("key")
    if key not in lines.index:
        return "The selected line is not in the simulation."
    r = lines.loc[key]
    f, _ = physics.peak_near(m["ff"], m["pp"], centre, s["peak_window_hz"])
    if f is None:
        return f"No peak within ±{s['peak_window_hz']:g} Hz of {centre:,.1f} Hz."
    if r["is_ref"]:
        if int(r["h"]) == s["ref_harmonic"]:
            return "The reference line of the reference harmonic is the anchor (strongest peak)."
        state.setdefault("ref_override", {})[str(int(r["h"]))] = float(f)
        return f"Reference line h{int(r['h'])} set to the peak at {f:,.1f} Hz."
    state.setdefault("assign", {})[key] = {"f_exp": float(f), "f_click": float(centre)}
    (state.get("offsets") or {}).pop(key, None)
    return f"Assigned {r['ion']} h{int(r['h'])} → {f:,.1f} Hz (f_exp − f_sim = {f - r['f_sim']:+.1f} Hz)."


@app.callback(
    Output("g-state", "data"), Output("g-msg", "children"),
    Input("g-drag", "data"), Input("g-graph", "clickData"), Input("g-click", "data"),
    Input("g-assign", "n_clicks"), Input("g-unassign", "n_clicks"), Input("g-deselect", "n_clicks"),
    Input("g-reset", "n_clicks"), Input("g-clear", "n_clicks"), Input("g-table", "data_timestamp"),
    Input("g-file", "value"), Input("g-h", "value"),
    State("g-table", "data"), State("g-state", "data"), State("g-keys", "data"),
    State("g-mode", "value"), *SETTING_STATES,
    prevent_initial_call=True,
)
def update_state(drag, click, sclick, _a, _u, _d, _r, _c, _ts, file, harm, table, state, keys,
                 mode, *values):
    nu = dash.no_update
    trig = ctx.triggered_id
    state = dict(state or new_state(file))
    if trig == "g-file":
        if file == state.get("file"):
            return nu, nu
        save_settings({"file": file})
        new = load_session(file)
        new["h"] = state.get("h") if new.get("h") is None else new["h"]
        return new, f"Opened {os.path.basename(file)}."
    if trig == "g-h":
        if harm is None or harm == state.get("h"):
            return nu, nu
        state["h"], state["selected"] = int(harm), None
        save_session(state)
        return state, ""
    if not state.get("file"):
        return nu, nu
    # cheap checks first: zoom / pan relayouts (no shape moved) do nothing here
    moved = {}
    if trig == "g-drag":
        moved = _shape_moves((drag or {}).get("moves"))
        if not moved:
            return nu, nu
    if trig == "g-click":
        if not sclick:
            return nu, nu
        if not state.get("selected"):
            return nu, "Select a line first (click its ▼ marker)."
    try:
        s = _settings(values)
        m = model(s, state)
    except Exception as exc:
        return nu, f"⚠ {exc}"
    h = m["h"]
    msg = ""
    if moved:
        msg = _drag(moved, keys or [], state, s, m, mode)
        if msg is None:                    # nothing moved (a click on a line)
            return nu, nu
    elif trig == "g-graph":
        pt = (click or {}).get("points", [{}])[0]
        key = pt.get("customdata")
        if isinstance(key, list):
            key = key[0] if key else None
        if not key or state.get("selected") == key:
            return nu, nu
        state["selected"] = key
    elif trig == "g-click":
        msg = _assign(state, s, m, state["selected"], float((sclick or {}).get("x")))
    elif trig == "g-assign":
        sel = state.get("selected")
        lines = m["lines"].set_index("key")
        if not sel or sel not in lines.index:
            return nu, "Select a line first (click its ▼ marker)."
        msg = _assign(state, s, m, sel, float(lines.loc[sel, "f_show"]))
    elif trig == "g-unassign":
        sel = state.get("selected")
        if sel and sel in (state.get("assign") or {}):
            state["assign"].pop(sel)
            msg = "Unassigned."
        elif sel and sel == line_key(s["ref_name"], h) and str(h) in (state.get("ref_override") or {}):
            state["ref_override"].pop(str(h))
            msg = "Reference line back to the automatic search."
        else:
            return nu, "The selected line is not assigned."
    elif trig == "g-deselect":
        state["selected"] = None
    elif trig == "g-reset":
        state["k"] = {k: v for k, v in (state.get("k") or {}).items() if k != str(h)}
        state["offsets"] = {q: v for q, v in (state.get("offsets") or {}).items()
                            if split_key(q)[1] != h}
        state["ref_override"] = {k: v for k, v in (state.get("ref_override") or {}).items()
                                 if k != str(h)}
        msg = f"h{h}: stretch, moved lines and moved reference line reset."
    elif trig == "g-clear":
        state["assign"] = {q: v for q, v in (state.get("assign") or {}).items()
                           if split_key(q)[1] != h}
        msg = f"h{h}: assignments cleared."
    elif trig == "g-table":
        left = {r["key"] for r in (table or [])}
        gone = [q for q in (state.get("assign") or {}) if split_key(q)[1] == h and q not in left]
        if not gone:
            return nu, nu
        for q in gone:
            state["assign"].pop(q, None)
        msg = "Removed " + ", ".join(q.replace("|", " h") for q in gone) + "."
    save_session(state)
    return state, msg


# ---------------------------------------------------------------------------
# Calculate (the harmonic shown), show its fit, save it into a profile
# ---------------------------------------------------------------------------

@app.callback(
    Output("g-state", "data", allow_duplicate=True), Output("g-calc-msg", "children"),
    Output("g-fitstamp", "data"),
    Input("g-calc", "n_clicks"), State("g-state", "data"), *SETTING_STATES,
    prevent_initial_call=True,
)
def calculate(n, state, *values):
    nu = dash.no_update
    if not state or not state.get("file"):
        return nu, "Open a spectrum first.", nu
    try:
        s = _settings(values)
        m = model(s, state)
        r = fit_harmonic(s, state, m)
    except Exception as exc:
        return nu, f"⚠ {exc}", nu
    state = dict(state, fit=dict(state.get("fit") or {}, **{str(m["h"]): r}))
    save_session(state)
    return state, f"✓ h{m['h']} calculated.", n


@app.callback(
    Output("g-fit-table", "children"), Output("g-fit-graph", "figure"),
    Output("g-fit-graph", "style"), Output("g-fit-when", "children"),
    Output("g-fit-head", "children"), Output("g-psave", "children"),
    Output("g-calc-msg", "children", allow_duplicate=True),
    Input("g-h", "value"), Input("g-fitstamp", "data"), Input("g-file", "value"),
    State("g-state", "data"),
    prevent_initial_call="initial_duplicate",
)
def show_fit(h, _stamp, _file, state):
    msg = dash.no_update if ctx.triggered_id == "g-fitstamp" else ""
    fits = (state or {}).get("fit") or {}
    head = f"Second-order correction h{h}"
    btn = f"Save h{h} into profile"
    r = fits.get(str(h))
    if not r:
        return (f"h{h} not calculated yet for this spectrum.", go.Figure(), {"display": "none"},
                "", head, btn, msg)
    others = [f"h{x}" for x in sorted(fits, key=int) if str(x) != str(h)]
    when = datetime.datetime.fromisoformat(r["time"]).strftime("calculated %d %b %H:%M")
    return fit_table(r, others), fit_figure(r), {"display": "block"}, when, head, btn, msg


@app.callback(Output("g-pname", "style"), Input("g-ptarget", "value"))
def show_name_box(target):
    return {"width": "300px", "display": "inline-block" if target == NEW_PROFILE else "none"}


@app.callback(
    Output("g-pmsg", "children"), Output("g-ptarget", "options"), Output("g-ptarget", "value"),
    Output("g-pconfirm", "data"),
    Input("g-psave", "n_clicks"), State("g-state", "data"), State("g-h", "value"),
    State("g-ptarget", "value"), State("g-pname", "value"), State("g-pdefault", "value"),
    State("g-pconfirm", "data"),
    prevent_initial_call=True,
)
def save_profile(_n, state, h, target, name, make_default, pending):
    nu = dash.no_update
    r = ((state or {}).get("fit") or {}).get(str(h))
    if not r:
        return html.Span(f"✗ Calculate h{h} first.", className="err"), nu, nu, None
    confirm = pending == {"target": target, "h": h, "time": r["time"]}
    try:
        prof, replaced = save_harmonic(r, target, name, confirm_replace=confirm)
    except ValueError as exc:
        if str(exc) == "confirm":
            return (html.Span(f"⚠ '{target}' already has h{h}. Click 'Save h{h} into profile' "
                              f"again to replace it.", className="warn"),
                    nu, nu, {"target": target, "h": h, "time": r["time"]})
        return html.Span(f"✗ Not saved: {exc}", className="err"), nu, nu, None
    except OSError as exc:
        return html.Span(f"✗ Not saved: {exc}", className="err"), nu, nu, None
    hs = ", ".join(f"h{x}" for x in sorted(prof.get("per_harmonic") or {}, key=int))
    text = (f"✓ {'Replaced' if replaced else 'Saved'} h{h} in '{prof['name']}' (now {hs}"
            + ("" if str(r["ref_harmonic"]) in (prof.get("per_harmonic") or {}) else
               f"; general a, b, c: {'0 until' if target == NEW_PROFILE else 'kept until'} "
               f"h{r['ref_harmonic']} is saved") + ")")
    if make_default:
        physics.save_shared({"corr_profile": prof["name"], "corr_a": prof["a"],
                             "corr_b": prof["b"], "corr_c": prof["c"]})
        text += " and made it the default (physics.json)"
    return (html.Span(text + ". Reload Theia / restart open tools to see it.", className="ok"),
            profile_options(), prof["name"], None)


# ===========================================================================

def _port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as so:
        so.settimeout(0.2)
        return so.connect_ex(("127.0.0.1", port)) == 0


_TABS = tabwatch.TabTracker(on_last_closed=lambda: os._exit(0), name=APP_NAME)
tabwatch.install_flask(app.server, _TABS)


if __name__ == "__main__":
    url = f"http://127.0.0.1:{PORT}/"
    if _port_open(PORT):                # already running: just show it
        browser.open_url(url)
        sys.exit(0)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    print(f"{APP_NAME} at {url}")
    threading.Timer(1.0, lambda: browser.open_url(url)).start()
    app.run(debug=False, host="127.0.0.1", port=PORT)
