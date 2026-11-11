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
    most MAX_SHAPES are in view): dragging a line stretches all lines of the
    harmonic about its reference line, which stays fixed.  The reference line
    of another harmonic is put on its peak by selecting it and clicking the
    peak (or "Assign to nearest peak").
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
from core import browser, corrections, icon, jsonfile, logcapture, logpanel, paths, physics, tabwatch, trackpad

# output also to data/logs/ and the page's Log panel (core/logcapture.py)
if __name__ == "__main__":
    logcapture.start("gui_correction")
from core.nubase_reader import MASS_MODEL, unpack_npz

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, dash_table, Input, Output, State, ALL, ctx

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
UNDO_MAX = 50               # drags remembered for "Undo last drag" (per spectrum)
MAX_HOVER_LINES = 600       # lines in view that get a tooltip along their height
NEW_PROFILE = "__new__"

TOOL_DEFAULTS = {
    "ref_name": "", "ref_harmonic": 0, "gamma_t": 0.0,
    "z_lims": "", "n_lims": "", "el1": "", "el2": "", "mq_width": 0.0,
    "ref_search_hz": 0.0,
    "peak_window_hz": 100.0,     # ± window for the assigned peaks
    "file": "",                  # last spectrum opened
}
SETTING_KEYS = ["ref_name", "ref_harmonic", "gamma_t", "z_lims", "n_lims",
                "el1", "el2", "mq_width", "ref_search_hz", "peak_window_hz"]
LABELS = {
    "ref_name": "Reference ion (strongest peak)", "ref_harmonic": "Reference harmonic",
    "gamma_t": "γ_t", "z_lims": "Z min, max", "n_lims": "N min, max",
    "el1": "el1 (A, Z)", "el2": "el2 (A, Z)", "mq_width": "m/q window (±)",
    "ref_search_hz": "Ref. peak search in other harmonics ± (Hz)",
    "peak_window_hz": "Peak window for assignments ± (Hz)",
}


# ===========================================================================
# Settings and sessions
# ===========================================================================

def _read_json(path):
    """A settings file as a dict; {} if missing.  A damaged file is set aside
    (its backup restored if there is one) with a warning: core/jsonfile.py."""
    return jsonfile.read(path, {})


def _write_json(path, data):
    jsonfile.write(path, data)          # atomic, previous version kept as .bak


_LOCK = threading.Lock()


def load_settings():
    return physics.with_shared(TOOL_DEFAULTS, _read_json(paths.settings_file(SETTINGS_FILE)))


def save_settings(changes):
    p = paths.settings_file(SETTINGS_FILE)
    with _LOCK, jsonfile.lock(p):
        cfg = {**load_settings(), **changes}
        jsonfile.try_write(p, physics.strip_shared(cfg), "the correction-tool settings")


def _session_key(path):
    return os.path.realpath(os.path.expanduser(path)) if path else ""


SESSION_KEYS = ("h", "harmonics", "k", "offsets", "ref_override", "assign", "fit", "undo",
                "corr_base")


def load_session(path):
    s = _read_json(paths.settings_file(SESSIONS_FILE)).get(_session_key(path)) or {}
    return new_state(path, s)


def save_session(state):
    if not state or not state.get("file"):
        return
    p = paths.settings_file(SESSIONS_FILE)
    with _LOCK, jsonfile.lock(p):
        all_ = _read_json(p)
        d = {k: state.get(k) for k in SESSION_KEYS}
        d["saved"] = datetime.datetime.now().isoformat(timespec="seconds")
        all_[_session_key(state["file"])] = d
        jsonfile.try_write(p, all_, "the correction-tool session")


def new_state(path, saved=None):
    saved = saved or {}
    fit = saved.get("fit") or {}
    if "harmonics" in fit:                   # first version: all harmonics in one fit
        fit = {h: dict(r, **{k: fit.get(k) for k in ("file", "f_ref", "ref_name", "ref_harmonic",
                                                      "gamma_t", "ref_search_hz",
                                                      "peak_window_hz", "time")}, h=int(h))
               for h, r in fit["harmonics"].items()}
    return {"file": path or "", "h": saved.get("h"), "harmonics": saved.get("harmonics"),
            "k": saved.get("k") or {},
            "offsets": saved.get("offsets") or {}, "ref_override": saved.get("ref_override") or {},
            "assign": saved.get("assign") or {}, "fit": fit, "undo": saved.get("undo") or [],
            "corr_base": saved.get("corr_base") or {},
            "selected": None}


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
    if s["ref_harmonic"] <= 0:
        raise ValueError("the reference harmonic must be > 0")
    s["harmonics"] = [s["ref_harmonic"]]       # the spectrum's list: state_harmonics()
    for k, n in (("z_lims", 2), ("n_lims", 2), ("el1", 2), ("el2", 2)):
        s[k] = _ints(raw.get(k), n, LABELS[k])
    s["mq_width"] = _f(raw.get("mq_width"), "m/q window")
    s["ref_search_hz"] = _f(raw.get("ref_search_hz"), "reference search window", 0)
    s["peak_window_hz"] = _f(raw.get("peak_window_hz"), "peak window", 100)
    if s["peak_window_hz"] <= 0:
        raise ValueError("the peak window must be > 0")
    return s


def state_harmonics(state, ref_h):
    """The harmonics listed for this spectrum: the reference harmonic plus the
    ones added with ＋ (saved in the session).  Sessions from before the ＋
    button: the reference harmonic plus every harmonic with work in it."""
    hs = (state or {}).get("harmonics")
    if hs is None:
        hs = {int(h) for d in ("k", "ref_override", "fit") for h in ((state or {}).get(d) or {})}
        hs |= {split_key(q)[1] for d in ("assign", "offsets") for q in ((state or {}).get(d) or {})}
    out = {int(h) for h in hs} | {int(ref_h)}
    return [int(ref_h)] + sorted(h for h in out if h != int(ref_h))


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
    # after Calculate the lines start from the corrected positions (that fit's
    # a, b, c on the revolution frequency); drags are applied on top of them
    cb = (state.get("corr_base") or {}).get(str(h))
    if cb:
        x = f_sim / h
        f_base = physics.apply_correction(x, *cb["abc"]) * h
    else:
        f_base = f_sim.copy()
    lines["f_base"] = f_base
    lines["f_corr"] = f_base if cb else np.nan
    off = np.array([float(offsets.get(q, 0.0)) for q in lines["key"]])
    show = np.where(lines["is_ref"], f_sim, r + k * (f_base - r) + off)
    lines["assigned"] = [q in assign for q in lines["key"]]
    lines["f_exp"] = [float((assign.get(q) or {}).get("f_exp", np.nan)) for q in lines["key"]]
    lines["f_show"] = np.where(lines["assigned"], lines["f_exp"], show)
    lines["moved"] = [q in offsets for q in lines["key"]]
    return {"h": h, "ff": ff, "pp": pp, "f_ref": f_ref, "ref_h_f": r, "ref_info": info[h],
            "warnings": warnings, "lines": lines, "k": k, "corr_base": cb}


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


def overlay(s, state, m, view):
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
    movable = (~lines["assigned"] & ~lines["is_ref"]).to_numpy()
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
           + (f"<br>f_sim corrected = {r.f_corr:,.1f} Hz ({r.f_corr - r.f_sim:+.1f} Hz)"
              if np.isfinite(r.f_corr) and not r.is_ref else "")
           + (f"<br>shown at {r.f_show:,.1f} Hz" if not r.is_ref and not r.assigned
              and abs(r.f_show - r.f_base) > 0.05 else "")
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


def figure(s, state, m, view):
    """Full plot: spectrum (trace 0, no hover) + overlay."""
    traces, shapes, ann, keys, note = overlay(s, state, m, view)
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
            "cov": res["cov"],
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


# matplotlib's default colours, as in tartarus/new_2nd_order_correction_plot.py
MPL_C0, MPL_C1, MPL_C3 = "#1f77b4", "#ff7f0e", "#d62728"


def pretty_ion_html(name):
    """'208Pb80+' -> <sup>208</sup>Pb<sup>80+</sup>, 'm170Tm66+' -> <sup>170m</sup>Tm<sup>66+</sup>
    (as pretty_ion in the old script)."""
    mt = re.fullmatch(r"([a-z]?)(\d+)([A-Za-z]+)(\d+[+-])", str(name))
    if not mt:
        return str(name)
    iso, A, el, q = mt.groups()
    return f"<sup>{A}{iso}</sup>{el}<sup>{q}</sup>"


# places for the coefficient box and the legend in the main panel:
# name -> (x, y, xanchor, yanchor) in axis-domain units
_SPOTS = {"tl": (0.01, 0.98, "left", "top"), "tr": (0.99, 0.98, "right", "top"),
          "bl": (0.01, 0.03, "left", "bottom"), "br": (0.99, 0.03, "right", "bottom"),
          "tc": (0.5, 0.98, "center", "top"), "bc": (0.5, 0.03, "center", "bottom"),
          "ml": (0.01, 0.5, "left", "middle"), "mr": (0.99, 0.5, "right", "middle")}
BOX_FRAC = (0.30, 0.36)      # rough size of the coefficient box (fraction of the panel)
LEGEND_FRAC = (0.34, 0.22)   # ... and of the legend


def _spot_rect(name, frac):
    x, y, xa, ya = _SPOTS[name]
    w, h = frac
    x0 = {"left": x, "right": x - w, "center": x - w / 2}[xa]
    y0 = {"bottom": y, "top": y - h, "middle": y - h / 2}[ya]
    return x0, y0, x0 + w, y0 + h


def _free_corners(x, y, xx, r):
    """Places for the coefficient box and the legend inside the main panel:
    the spots (corners, edge centres) whose area holds the fewest data points
    and fit curve, box first, legend not overlapping the box."""
    yc = r["s1"] * (xx - r["x0"]) + r["s2"] * (xx - r["x0"]) ** 2
    ally = np.concatenate([y, yc])
    lo, hi = float(ally.min()), float(ally.max())
    pad = 0.06 * (hi - lo or 1.0)
    lo, hi = lo - pad, hi + pad
    xlo, xhi = float(xx.min()), float(xx.max())
    px = (np.asarray(x) - xlo) / (xhi - xlo or 1)
    py = (np.asarray(y) - lo) / (hi - lo or 1)
    cx = (xx - xlo) / (xhi - xlo or 1)
    cy = (yc - lo) / (hi - lo or 1)

    def cost(rect):
        x0, y0, x1, y1 = rect
        m = 0.03                                   # a little room round each point
        pts = ((px >= x0 - m) & (px <= x1 + m) & (py >= y0 - m) & (py <= y1 + m)).sum()
        crv = ((cx >= x0) & (cx <= x1) & (cy >= y0) & (cy <= y1)).sum()
        return 50 * pts + crv

    order = list(_SPOTS)
    box = min(order, key=lambda c: (cost(_spot_rect(c, BOX_FRAC)), order.index(c)))
    br = _spot_rect(box, BOX_FRAC)

    def apart(c):
        a = _spot_rect(c, LEGEND_FRAC)
        return a[2] <= br[0] or br[2] <= a[0] or a[3] <= br[1] or br[3] <= a[1]

    pref = ["br", "bl", "tr", "tl", "bc", "tc", "mr", "ml"]
    leg = min((c for c in pref if apart(c)),
              key=lambda c: (cost(_spot_rect(c, LEGEND_FRAC)), pref.index(c)))
    lx, ly, lxa, lya = _SPOTS[leg]
    # the legend is placed in paper units: the main panel is the top 75 % of the plot
    ly = 0.25 + 0.75 * ly
    return _SPOTS[box], (lx, ly, lxa, lya)


def fit_figure(r):
    """The fit of one harmonic in the style of the old script's figure
    (new_2nd_order_correction_plot.draw_figure): data C0, anchored fit C1,
    red star anchor, labels with leader lines, coefficient box, residuals."""
    pts = r["points"]
    names = [r["ref_name"]] + [p["ion"] for p in pts]
    x = np.array([r["x0"]] + [p["x"] for p in pts], float)
    y = np.array([0.0] + [p["y"] for p in pts], float)
    resid = np.array([0.0] + [p["resid"] for p in pts], float)
    (a, b, c), (da, db, dc) = r["abc"], r["abc_unc"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25],
                        vertical_spacing=0.0)
    # fit curve, then data, then anchor (as the old figure's z-order)
    pad = 0.05 * (x.max() - x.min() or 1)
    xx = np.linspace(x.min() - pad, x.max() + pad, 500)
    fig.add_trace(go.Scatter(x=xx.tolist(), y=(r["s1"] * (xx - r["x0"])
                                               + r["s2"] * (xx - r["x0"]) ** 2).tolist(),
                             mode="lines", line=dict(color=MPL_C1, width=2), meta="fit-curve",
                             name="Anchored 2nd-order fit (unweighted)", hoverinfo="skip",
                             legendrank=3),
                  row=1, col=1)
    labels = [pretty_ion_html(n) for n in names]
    fig.add_trace(go.Scatter(x=x[1:].tolist(), y=y[1:].tolist(), mode="markers", name="Data", legendrank=1,
                             text=labels[1:], meta="fit-pts", marker=dict(color=MPL_C0, size=9),
                             hovertemplate="%{text}<br>f_sim = %{x:.3f} Hz<br>"
                                           "f_diff = %{y:.3f} Hz<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[r["x0"]], y=[0], mode="markers",
                             name=f"Anchor ({pretty_ion_html(r['ref_name'])}, "
                                  f"f<sub>exp</sub> = {r['ref_f_h']!r} Hz)",
                             marker=dict(symbol="star", size=20, color=MPL_C3), legendrank=2,
                             text=[labels[0]], meta="fit-pts",
                             hovertemplate="anchor (reference line)<br>x0 = %{x:.3f} Hz"
                                           "<extra></extra>"), row=1, col=1)
    # residuals
    fig.add_trace(go.Scatter(x=x[1:].tolist(), y=resid[1:].tolist(), mode="markers",
                             marker=dict(color=MPL_C0, size=9), showlegend=False, text=names[1:],
                             hovertemplate="%{text}<br>residual = %{y:.3f} Hz<extra></extra>"),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=[r["x0"]], y=[0], mode="markers", showlegend=False,
                             marker=dict(symbol="star", size=20, color=MPL_C3), hoverinfo="skip"),
                  row=2, col=1)
    fig.add_hline(y=0, line=dict(color="grey", width=0.8, dash="dash"), row=1, col=1)
    fig.add_hline(y=0, line=dict(color=MPL_C1, width=1.5), row=2, col=1)

    # ion labels: placed in the page so they never overlap (FIT_LABEL_JS, the
    # placement of Theia's isochronicity labels); here only the box
    ann = []
    # coefficient box, top left
    box = ("Expanded: f<sub>diff</sub> = a x² + b x + c<br>"
           f"  a = {a:.6e} ± {da:.2e} Hz⁻¹<br>"
           f"  b = {b:.6e} ± {db:.2e}<br>"
           f"  c = {c:.6e} ± {dc:.2e} Hz<br><br>"
           f"RMS residual = {r['rms']:.3f} Hz").replace("  ", "&nbsp;&nbsp;")
    # box and legend inside the plot, in the two corners with the fewest
    # points / curve (the labels are then placed around them by FIT_LABEL_JS)
    (bx, by, bxa, bya), (lx, ly, lxa, lya) = _free_corners(x, y, xx, r)
    ann.append(dict(xref="x domain", yref="y domain", x=bx, y=by, xanchor=bxa,
                    yanchor=bya, showarrow=False, align="left", text=box,
                    font=dict(family="DejaVu Sans Mono, monospace", size=14, color="black"),
                    bgcolor="rgba(255,255,255,0.9)", bordercolor="grey", borderwidth=1,
                    borderpad=6))
    grid = dict(showgrid=True, gridcolor="rgba(0,0,0,0.12)", zeroline=False, showline=True,
                linecolor="black", mirror=True, ticks="outside", tickformat=".0f")
    fig.update_layout(template="none", plot_bgcolor="white", paper_bgcolor="white",
                      height=700, font=dict(size=15, color="black"), annotations=ann,
                      margin=dict(l=90, r=30, t=60, b=60),
                      title=dict(text=f"Second-order Correction h{r['h']} "
                                      f"(masses incl. electron binding)", x=0.5,
                                 xanchor="center", font=dict(size=18)),
                      legend=dict(x=lx, y=ly, xanchor=lxa, yanchor=lya, yref="paper",
                                  bgcolor="rgba(255,255,255,0.85)", bordercolor="#cccccc",
                                  borderwidth=1))
    fig.update_xaxes(**grid)
    fig.update_yaxes(**{k: v for k, v in grid.items() if k != "tickformat"})
    fig.update_yaxes(title_text="f<sub>diff</sub> = f<sub>exp,true</sub> − f<sub>sim,true</sub> (Hz)",
                     row=1, col=1)
    fig.update_yaxes(title_text="Residual (Hz)", row=2, col=1)
    fig.update_xaxes(title_text="f<sub>sim,true</sub> (Hz)", row=2, col=1)
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
    # "shifted": s1, s2, their covariance and x0, so the correction's own
    # uncertainty can be worked out later (corrections.correction_sigma)
    shifted = (corrections.shifted_block(r["s1"], r["s2"], r["x0"], r["cov"])
               if r.get("cov") else None)
    entry = {"a": a, "b": b, "c": c, "a_unc": da, "b_unc": db, "c_unc": dc, "fit": _fit_info(r)}
    if shifted:
        entry["shifted"] = shifted
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
            if shifted:
                p["shifted"] = shifted
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
        if shifted:
            p["shifted"] = shifted
        else:
            p.pop("shifted", None)       # an older fit's block no longer matches a, b, c
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
.harm {{ display: flex; flex-wrap: wrap; gap: 6px; }}
.chip {{ display: inline-flex; border: 1px solid #999; border-radius: 14px; overflow: hidden;
         background: #fff; }}
.chip.on {{ border-color: {ACCENT}; }}
.chip button {{ border: none; background: none; cursor: pointer; font-size: 14px; padding: 3px 9px; }}
.chip-h.on {{ background: {ACCENT}; color: #fff; font-weight: 600; }}
.chip-x {{ color: #777; border-left: 1px solid #ddd !important; padding: 3px 7px !important; }}
.chip-x:hover {{ color: #b91c1c; }}
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
trackpad.install_dash(app)                   # touchpad zoom (core/trackpad.py)
icon.use_in_dash(app, "gui_correction")      # core/icons/gui_correction.png, else asphodel.png


def harmonic_bar(state, ref_h):
    """One chip per listed harmonic (click = show it; × = take it off the list,
    its work stays in the session), the reference harmonic first, without ×."""
    cur = (state or {}).get("h")
    chips = []
    for h in state_harmonics(state, ref_h):
        on = h == cur
        kids = [html.Button(f"h{h}" + (" (ref)" if h == ref_h else ""),
                            id={"type": "g-hsel", "h": h}, n_clicks=0,
                            className="chip-h" + (" on" if on else ""),
                            title=f"Show h{h}")]
        if h != ref_h:
            kids.append(html.Button("×", id={"type": "g-hdel", "h": h}, n_clicks=0,
                                    className="chip-x", title=f"Take h{h} off the list "
                                                              f"(its work is kept)"))
        chips.append(html.Span(kids, className="chip" + (" on" if on else "")))
    return chips


def serve_layout():
    cfg = load_settings()
    files = spectrum_files()
    values = [o["value"] for o in files]
    f = cfg.get("file") if cfg.get("file") in values else (values[0] if values else "")
    state = load_session(f) if f else new_state("")
    try:
        ref_h = int(float(cfg.get("ref_harmonic") or 0)) or None
    except (TypeError, ValueError):
        ref_h = None
    if ref_h:
        state["harmonics"] = state_harmonics(state, ref_h)
        h = state.get("h") if state.get("h") in state["harmonics"] else ref_h
    else:
        h = None
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
        dcc.Store(id="g-hcur", data=h),          # harmonic shown (for the fit panel)
        html.Div(id="_g-hook", style={"display": "none"}),
        html.Div(id="_g-fit-label-hook", style={"display": "none"}),
        html.Div(id="g-readout"),
        html.Div([html.H1(APP_NAME), html.Span(f"experiment {paths.experiment()}", className="exp"),
                  logpanel.dash_button(style={"marginLeft": "auto"})],
                 className="top"),
        html.Div([
            html.Div([
                html.H3("Spectrum"),
                dcc.Dropdown(id="g-file", options=files, value=f or None, clearable=False,
                             placeholder="No .npz in the connected spectra folders",
                             style={"fontSize": "12px"}),
                html.H3("Harmonic"),
                html.Div(harmonic_bar(state, ref_h) if ref_h else [], id="g-hbar", className="harm"),
                html.Div([
                    dcc.Input(id="g-hnew", type="number", min=1, step=1, placeholder="h",
                              debounce=False, n_submit=0, style={"width": "80px"}),
                    html.Button("＋", id="g-hadd", className="btn primary", n_clicks=0,
                                title="Add this harmonic to the list"),
                ], className="row", style={"marginTop": "6px"}),
                html.Div("Each harmonic keeps its own stretch, moved lines, assignments and fit.",
                         className="note"),
                html.H3("Dragging"),
                html.Div("Drag a line to stretch all lines of this harmonic about the "
                         "reference line.", className="note"),
                html.Div(id="g-stretch", className="note"),
                html.Button("Lines to corrected positions", id="g-tocorr", className="btn",
                            n_clicks=0, title="After Calculate: put this harmonic's lines back "
                                              "at the positions given by its last calculated "
                                              "correction (clears stretch and moved lines)"),
                html.Button("↶ Undo last drag", id="g-undo", className="btn", n_clicks=0,
                            title="Undo the last dragged line on this harmonic (stretch, "
                                  "moved line or moved reference line); click again for the "
                                  "one before"),
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
                                        html.Button("Plot (2nd-order correction script)",
                                                    id="g-mplplot", className="btn", n_clicks=0,
                                                    title="Open the figure of tartarus/new_2nd_order_"
                                                          "correction_plot.py for the assigned peaks of "
                                                          "this harmonic (its own window, with zoom "
                                                          "and save)"),
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
                html.Div([
                    html.Div([html.Span("Corrections by harmonic"),
                              html.Span(id="g-sum-note", className="note")], className="head"),
                    html.Div(id="g-summary", className="body"),
                ], className="box"),
            ], className="main"),
        ], className="wrap"),
    ])


app.layout = logpanel.dash_layout(serve_layout, "GUI Second Order Correction")
logpanel.register_dash(app)
SETTING_INPUTS = [Input(f"g-{k}", "value") for k in SETTING_KEYS]
SETTING_STATES = [State(f"g-{k}", "value") for k in SETTING_KEYS]


def _settings(values, state=None):
    s = parse_settings(dict(zip(SETTING_KEYS, values)))
    s["harmonics"] = state_harmonics(state, s["ref_harmonic"])
    return s


@app.callback(Output("g-msg", "children", allow_duplicate=True), *SETTING_INPUTS,
              prevent_initial_call=True)
def persist_settings(*values):
    save_settings(dict(zip(SETTING_KEYS, values)))
    return dash.no_update


@app.callback(Output("g-hbar", "children"), Input("g-state", "data"),
              Input("g-ref_harmonic", "value"), prevent_initial_call=True)
def draw_harmonic_bar(state, ref_h):
    try:
        ref_h = int(float(ref_h))
    except (TypeError, ValueError):
        return dash.no_update
    return harmonic_bar(state, ref_h)


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
    Input("g-state", "data"), Input("g-view", "data"), *SETTING_INPUTS,
    State("g-sig", "data"),
)
def draw(state, view, *values):
    values, sig_old = values[:-1], values[-1]
    nu = dash.no_update
    if not state or not state.get("file"):
        return go.Figure(), [], "No spectrum", "", "", "", [], "", None, "Assignments"
    title = os.path.basename(state["file"])
    try:
        s = _settings(values, state)
        m = model(s, state)
    except Exception as exc:
        return go.Figure(), [], title, "", f"⚠ {exc}", "", nu, "", None, nu
    h = m["h"]
    sig = [state["file"]]
    if sig == sig_old:
        # same spectrum: only the lines, markers and labels are sent
        traces, shapes, ann, keys, note = overlay(s, state, m, view)
        fig = dash.Patch()
        for j, t in enumerate(traces):
            fig["data"][1 + j] = t
        fig["layout"]["shapes"] = shapes
        fig["layout"]["annotations"] = ann
    else:
        fig, keys, note = figure(s, state, m, view)
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
            f, _ = physics.peak_near(m["ff"], m["pp"], r["f_show"], s["peak_window_hz"], climb=True)
            txt.append(f" · shown at {r['f_show']:,.1f} Hz · nearest peak "
                       + (f"{f:,.1f} Hz ({f - r['f_show']:+.1f} Hz)" if f is not None else
                          "none found"))
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
    if m.get("corr_base"):
        t = m["corr_base"].get("time") or ""
        stretch = (f"h{h} lines start from the corrected positions (fit of "
                   f"{t[11:16] or t}) · " + stretch)
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


def _drag(moved, keys, state, s, m):
    """Apply shape drags.  Returns a message."""
    lines = m["lines"].set_index("key")
    h = m["h"]
    msg = ""
    changed = [False]
    # what the drag changes, as it was before (for "Undo last drag")
    keys_moved = [keys[i] for i in moved if i < len(keys)]
    before = {"h": h, "k": (state.get("k") or {}).get(str(h)),
              "ref": (state.get("ref_override") or {}).get(str(h)),
              "offsets": {q: (state.get("offsets") or {}).get(q) for q in keys_moved}}
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
            msg = f"The reference line h{h} stays fixed."
            continue
        if r["assigned"]:
            continue
        d = float(r["f_base"] - ref_h)
        if abs(d) < 1e-6:
            continue
        kn = (x_new - float((state.get("offsets") or {}).get(key, 0.0)) - ref_h) / d
        if not np.isfinite(kn) or kn <= 0:
            msg = "A line cannot be dragged across its reference line."
            continue
        state["k"] = dict(state.get("k") or {}, **{str(h): float(kn)})
        msg = f"h{h} stretch ×{kn:.6f}"
    if changed[0]:
        state["undo"] = ((state.get("undo") or []) + [before])[-UNDO_MAX:]
    return msg if changed[0] else None


def _to_corrected(state, h, corr=None, set_base=False):
    """Lines of h back to the corrected positions: stretch and moved lines of h
    cleared (the moved reference line is kept); with set_base, corr (a fit
    result) becomes the corrected base.  Undoable."""
    offs = {q: v for q, v in (state.get("offsets") or {}).items() if split_key(q)[1] == h}
    entry = {"h": h, "k": (state.get("k") or {}).get(str(h)),
             "ref": (state.get("ref_override") or {}).get(str(h)), "offsets": offs}
    if set_base:
        entry["corr_base"] = (state.get("corr_base") or {}).get(str(h))
        state["corr_base"] = dict(state.get("corr_base") or {},
                                  **{str(h): {"abc": list(corr["abc"]), "time": corr["time"]}})
    state["undo"] = ((state.get("undo") or []) + [entry])[-UNDO_MAX:]
    state["k"] = {x: v for x, v in (state.get("k") or {}).items() if x != str(h)}
    state["offsets"] = {q: v for q, v in (state.get("offsets") or {}).items()
                        if split_key(q)[1] != h}


def _undo(state, h):
    """Undo the last drag on harmonic h.  Returns a message."""
    stack = list(state.get("undo") or [])
    i = next((j for j in range(len(stack) - 1, -1, -1) if stack[j].get("h") == h), None)
    if i is None:
        return f"Nothing to undo on h{h}."
    b = stack.pop(i)
    state["undo"] = stack
    k = dict(state.get("k") or {})
    if b.get("k") is None:
        k.pop(str(h), None)
    else:
        k[str(h)] = b["k"]
    state["k"] = k
    ro = dict(state.get("ref_override") or {})
    if b.get("ref") is None:
        ro.pop(str(h), None)
    else:
        ro[str(h)] = b["ref"]
    state["ref_override"] = ro
    off = dict(state.get("offsets") or {})
    for q, v in (b.get("offsets") or {}).items():
        if v is None:
            off.pop(q, None)
        else:
            off[q] = v
    state["offsets"] = off
    if "corr_base" in b:                   # undoing a Calculate / "to corrected positions"
        cbs = dict(state.get("corr_base") or {})
        if b["corr_base"] is None:
            cbs.pop(str(h), None)
        else:
            cbs[str(h)] = b["corr_base"]
        state["corr_base"] = cbs
    left = sum(1 for e in stack if e.get("h") == h)
    return f"Undid the last drag on h{h}" + (f" ({left} more to undo)." if left else ".")


def _assign(state, s, m, key, centre):
    lines = m["lines"].set_index("key")
    if key not in lines.index:
        return "The selected line is not in the simulation."
    r = lines.loc[key]
    f, _ = physics.peak_near(m["ff"], m["pp"], centre, s["peak_window_hz"], climb=True)
    if f is None:
        return (f"No peak found uphill from {centre:,.1f} Hz within "
                f"{10 * s['peak_window_hz']:g} Hz.")
    if r["is_ref"]:
        if int(r["h"]) == s["ref_harmonic"]:
            return "The reference line of the reference harmonic is the anchor (strongest peak)."
        state.setdefault("ref_override", {})[str(int(r["h"]))] = float(f)
        return f"Reference line h{int(r['h'])} set to the peak at {f:,.1f} Hz."
    state.setdefault("assign", {})[key] = {"f_exp": float(f), "f_click": float(centre)}
    (state.get("offsets") or {}).pop(key, None)
    return f"Assigned {r['ion']} h{int(r['h'])} → {f:,.1f} Hz (f_exp − f_sim = {f - r['f_sim']:+.1f} Hz)."


@app.callback(
    Output("g-state", "data"), Output("g-msg", "children"), Output("g-hcur", "data"),
    Input("g-drag", "data"), Input("g-graph", "clickData"), Input("g-click", "data"),
    Input("g-assign", "n_clicks"), Input("g-unassign", "n_clicks"), Input("g-deselect", "n_clicks"),
    Input("g-reset", "n_clicks"), Input("g-clear", "n_clicks"), Input("g-table", "data_timestamp"),
    Input("g-undo", "n_clicks"), Input("g-tocorr", "n_clicks"),
    Input("g-file", "value"),
    Input({"type": "g-hsel", "h": ALL}, "n_clicks"), Input({"type": "g-hdel", "h": ALL}, "n_clicks"),
    Input({"type": "g-hsum", "h": ALL}, "n_clicks"),
    Input("g-hadd", "n_clicks"), Input("g-hnew", "n_submit"),
    State("g-hnew", "value"),
    State("g-table", "data"), State("g-state", "data"), State("g-keys", "data"),
    *SETTING_STATES,
    prevent_initial_call=True,
)
def update_state(drag, click, sclick, _a, _u, _d, _r, _c, _ts, _undo_n, _tocorr, file, _hs, _hd, _hsum,
                 _ha, _hn, hnew,
                 table, state, keys, *values):
    nu = dash.no_update
    trig = ctx.triggered_id
    state = dict(state or new_state(file))
    try:
        ref_h = int(float(values[SETTING_KEYS.index("ref_harmonic")]))
    except (TypeError, ValueError):
        ref_h = None
    if trig == "g-file":
        if file == state.get("file"):
            return nu, nu, nu
        save_settings({"file": file})
        new = load_session(file)
        if ref_h:
            new["harmonics"] = state_harmonics(new, ref_h)
            if new.get("h") not in new["harmonics"]:
                new["h"] = ref_h
        return new, f"Opened {os.path.basename(file)}.", new.get("h")
    if isinstance(trig, dict) and trig.get("type") in ("g-hsel", "g-hdel", "g-hsum"):
        if not (ctx.triggered and ctx.triggered[0]["value"]) or not ref_h:
            return nu, nu, nu                # buttons just (re)drawn, not clicked
        h = int(trig["h"])
        hs = state_harmonics(state, ref_h)
        if trig["type"] == "g-hsum" and h not in hs:    # from the summary: back on the list
            state["harmonics"] = state_harmonics(dict(state, harmonics=hs + [h]), ref_h)
        if trig["type"] in ("g-hsel", "g-hsum"):
            if h == state.get("h"):
                return nu, nu, nu
            state["h"], state["selected"] = h, None
            msg = ""
        else:
            state["harmonics"] = [x for x in hs if x != h]
            msg = f"h{h} taken off the list (its assignments and fit are kept: ＋ it again)."
            if state.get("h") == h:
                state["h"], state["selected"] = ref_h, None
        save_session(state)
        return state, msg, state["h"]
    if trig in ("g-hadd", "g-hnew"):
        if not ref_h:
            return nu, "⚠ Set the reference harmonic first.", nu
        try:
            h = int(hnew)
            if h <= 0 or h != float(hnew):
                raise ValueError
        except (TypeError, ValueError):
            return nu, "Type a harmonic number (a positive integer) next to ＋.", nu
        hs = state_harmonics(state, ref_h)
        if h not in hs:
            state["harmonics"] = hs + [h]
        state["harmonics"] = state_harmonics(state, ref_h)
        state["h"], state["selected"] = h, None
        save_session(state)
        return state, (f"h{h} added." if h not in hs else f"h{h} is already listed."), h
    if not state.get("file"):
        return nu, nu, nu
    # cheap checks first: zoom / pan relayouts (no shape moved) do nothing here
    moved = {}
    if trig == "g-drag":
        moved = _shape_moves((drag or {}).get("moves"))
        if not moved:
            return nu, nu, nu
    if trig == "g-click":
        if not sclick:
            return nu, nu, nu
        if not state.get("selected"):
            return nu, "Select a line first (click its ▼ marker).", nu
    try:
        s = _settings(values, state)
        m = model(s, state)
    except Exception as exc:
        return nu, f"⚠ {exc}", nu
    h = m["h"]
    msg = ""
    if moved:
        msg = _drag(moved, keys or [], state, s, m)
        if msg is None:                    # nothing moved (a click on a line)
            return nu, nu, nu
    elif trig == "g-graph":
        pt = (click or {}).get("points", [{}])[0]
        key = pt.get("customdata")
        if isinstance(key, list):
            key = key[0] if key else None
        if not key or state.get("selected") == key:
            return nu, nu, nu
        state["selected"] = key
    elif trig == "g-click":
        msg = _assign(state, s, m, state["selected"], float((sclick or {}).get("x")))
    elif trig == "g-assign":
        sel = state.get("selected")
        lines = m["lines"].set_index("key")
        if not sel or sel not in lines.index:
            return nu, "Select a line first (click its ▼ marker).", nu
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
            return nu, "The selected line is not assigned.", nu
    elif trig == "g-deselect":
        state["selected"] = None
    elif trig == "g-reset":
        state["k"] = {k: v for k, v in (state.get("k") or {}).items() if k != str(h)}
        state["offsets"] = {q: v for q, v in (state.get("offsets") or {}).items()
                            if split_key(q)[1] != h}
        state["ref_override"] = {k: v for k, v in (state.get("ref_override") or {}).items()
                                 if k != str(h)}
        state["undo"] = [e for e in (state.get("undo") or []) if e.get("h") != h]
        msg = f"h{h}: stretch, moved lines and moved reference line reset."
    elif trig == "g-undo":
        msg = _undo(state, h)
    elif trig == "g-tocorr":
        if not (state.get("corr_base") or {}).get(str(h)):
            return nu, f"Calculate h{h} first: the corrected positions come from its fit.", nu
        _to_corrected(state, h)
        msg = f"h{h}: lines back at their corrected positions."
    elif trig == "g-clear":
        state["assign"] = {q: v for q, v in (state.get("assign") or {}).items()
                           if split_key(q)[1] != h}
        msg = f"h{h}: assignments cleared."
    elif trig == "g-table":
        left = {r["key"] for r in (table or [])}
        gone = [q for q in (state.get("assign") or {}) if split_key(q)[1] == h and q not in left]
        if not gone:
            return nu, nu, nu
        for q in gone:
            state["assign"].pop(q, None)
        msg = "Removed " + ", ".join(q.replace("|", " h") for q in gone) + "."
    save_session(state)
    return state, msg, nu


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
        s = _settings(values, state)
        m = model(s, state)
        r = fit_harmonic(s, state, m)
    except Exception as exc:
        return nu, f"⚠ {exc}", nu
    state = dict(state, fit=dict(state.get("fit") or {}, **{str(m["h"]): r}))
    _to_corrected(state, m["h"], r, set_base=True)
    save_session(state)
    return (state, f"✓ h{m['h']} calculated; the lines are now at their corrected positions "
                   f"(↶ Undo last drag goes back).", n)


def summary_table(state, target, cur_h):
    """Rows per harmonic: this spectrum's calculated fit, and what the profile
    chosen in 'Save into' holds for that harmonic."""
    fits = (state or {}).get("fit") or {}
    prof = corrections.get_profile(target) if target and target != NEW_PROFILE else None
    per_h = (prof or {}).get("per_harmonic") or {}
    hs = sorted({int(h) for h in fits} | {int(h) for h in per_h})
    pname = prof["name"] if prof else None
    head = ["Harmonic", "Ions", "a", "b", "c", "RMS (Hz, rev)", "Reference line", "Calculated",
            f"In profile '{pname}'" if pname else "In profile"]
    rows = []
    for h in hs:
        r = fits.get(str(h))
        e = per_h.get(str(h))
        btn = html.Button(f"h{h}", id={"type": "g-hsum", "h": h}, n_clicks=0,
                          className="chip-h" + (" on" if h == cur_h else ""),
                          title=f"Show h{h}", style={"border": "1px solid #999",
                                                     "borderRadius": "12px", "cursor": "pointer"})
        if r:
            (a, b, c), (da, db, dc) = r["abc"], r["abc_unc"]
            when = datetime.datetime.fromisoformat(r["time"]).strftime("%d %b %H:%M")
            cells = [r["n"], f"{a:.6e} ± {da:.2e}", f"{b:.6e} ± {db:.2e}", f"{c:.6e} ± {dc:.2e}",
                     f"{r['rms']:.3f}",
                     f"{r['ref_f_h']:,.1f} Hz · {REF_PEAK_TEXT.get(r.get('ref_found'), '')}", when]
        else:
            cells = ["–", "not calculated for this spectrum", "", "", "", "", ""]
        if not prof:
            inprof = "–" if target == NEW_PROFILE else "(profile not found)"
        elif e is None:
            inprof = "not in the profile (uses its general a, b, c)"
        elif r and all(corrections._same(x, y) for x, y in zip(r["abc"], (e["a"], e["b"], e["c"]))):
            inprof = "✓ same as this fit"
        else:
            src = os.path.basename((e.get("fit") or {}).get("source_file") or "") or "?"
            inprof = (f"a = {e['a']:.6e}, b = {e['b']:.6e}, c = {e['c']:.6e}"
                      + (" (differs from this fit)" if r else "") + f" · from {src}")
        rows.append([btn] + cells + [inprof])
    if prof:
        gen = ("general (other harmonics)", "", f"{prof['a']:.6e}", f"{prof['b']:.6e}",
               f"{prof['c']:.6e}", "", "", "", corrections.describe(prof))
        rows.append(list(gen))
    if not rows:
        return "No harmonic calculated for this spectrum yet."
    return html.Table([html.Thead(html.Tr([html.Th(x) for x in head])),
                       html.Tbody([html.Tr([html.Td(x) for x in r]) for r in rows])],
                      className="tbl")


@app.callback(
    Output("g-summary", "children"), Output("g-sum-note", "children"),
    Input("g-state", "data"), Input("g-ptarget", "value"), Input("g-pmsg", "children"),
)
def show_summary(state, target, _pmsg):
    h = (state or {}).get("h")
    note = ("this spectrum's fits; compare with the profile chosen under 'Save into'"
            if target and target != NEW_PROFILE else
            "this spectrum's fits; pick a profile under 'Save into' to compare")
    return summary_table(state, target, h), note


# Labels of the in-page fit plot: Theia's isochronicity label placement
# (candidate spots, overlap / arrow-crossing costs, simulated annealing with
# fixed seeds), re-run after every zoom / pan / resize once the view is still;
# styled like the matplotlib figure (black text and leader lines, no box).
FIT_LABEL_JS = r"""
function(fig) {
    var ANG = [45, 135, -45, -135, 90, -90, 0, 180,
               67.5, 112.5, -67.5, -112.5, 22.5, 157.5, -22.5, -157.5,
               33.75, 56.25, 123.75, 146.25, -33.75, -56.25, -123.75, -146.25];
    var RAD = [24, 34, 46, 60, 76, 94, 115, 140, 170, 210, 260, 320];
    var FONT_PX = 14;
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
    function textW(t, font) {
        // ion names are HTML (<sup>A</sup>El<sup>q+</sup>): superscripts at 70 %
        ctx2d.font = font;
        var w = 0, parts = String(t).split(/(<sup>.*?<\/sup>)/);
        parts.forEach(function(p) {
            if (!p) return;
            var m = p.match(/^<sup>(.*)<\/sup>$/);
            w += m ? 0.7 * ctx2d.measureText(m[1]).width
                   : ctx2d.measureText(p.replace(/<[^>]+>/g, '')).width;
        });
        return w;
    }

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
            if (tr.meta === 'fit-curve') {
                var step = Math.max(1, Math.floor(full.x.length / 150));
                for (var k = 0; k < full.x.length; k += step) {
                    var cx = xa.l2p(full.x[k]) + xa._offset, cy = ya.l2p(full.y[k]) + ya._offset;
                    obst.push([cx - 3, cy - 3, cx + 3, cy + 3]);
                }
                return;
            }
            if (tr.meta !== 'fit-pts') return;
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
                // text + a little white space round it (labels as in the matplotlib figure)
                sizes.push([textW(t, font) + 8, Math.round(FONT_PX * 1.5) + 6]);
                lab.push({x: X, y: Y, text: t});
            }
        });
        // never under the parameter box or the legend
        gd.querySelectorAll('.annotation, .legend').forEach(function(el) {
            if (el.classList.contains('annotation') && el.getAttribute('data-index') !== null) {
                var a = (gd.layout.annotations || [])[+el.getAttribute('data-index')];
                if (a && a.name === 'fitlab') return;
            }
            var r = el.getBoundingClientRect();
            if (r.width && r.height)
                obst.push([r.left - gr.left, r.top - gr.top, r.right - gr.left, r.bottom - gr.top, 20]);
        });
        var offs = place(pts, sizes, obst, rect, bars);
        var keep = (gd.layout.annotations || []).filter(function(a) { return a.name !== 'fitlab'; });
        var anns = keep.concat(lab.map(function(l, i) {
            return {x: l.x, y: l.y, xref: 'x', yref: 'y', text: l.text, name: 'fitlab',
                    showarrow: true, ax: offs[i][0], ay: offs[i][1], arrowhead: 0,
                    arrowwidth: 1.2, arrowcolor: '#000000', standoff: 3,
                    font: {size: FONT_PX, color: '#000000'},
                    bgcolor: 'rgba(255,255,255,0.75)', borderwidth: 0, borderpad: 2,
                    captureevents: false};
        }));
        gd._fitLabelBusy = true;
        Plotly.relayout(gd, {annotations: anns}).then(function() { gd._fitLabelBusy = false; },
                                                      function() { gd._fitLabelBusy = false; });
    }

    setTimeout(function() {
        var gd = document.querySelector('#g-fit-graph .js-plotly-plot');
        if (!gd) return;
        if (!gd._fitLabelHook) {
            gd._fitLabelHook = true;
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
                if (gd._fitLabelBusy) return;
                var keys = Object.keys(ev || {});
                if (keys.length && keys.every(function(k) { return k.indexOf('annotations') === 0; })) return;
                soon();
            });
            gd.on('plotly_restyle', soon);            // legend clicks (hide / show)
            window.addEventListener('resize', soon);
            gd._fitSoon = soon;
        }
        gd._fitSoon();
    }, 150);
    return window.dash_clientside.no_update;
}
"""

app.clientside_callback(
    FIT_LABEL_JS,
    Output("_g-fit-label-hook", "children"),
    Input("g-fit-graph", "figure"),
    prevent_initial_call=True,
)


CORR_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "new_2nd_order_correction_plot.py")


@app.callback(
    Output("g-calc-msg", "children", allow_duplicate=True),
    Input("g-mplplot", "n_clicks"), State("g-state", "data"), *SETTING_STATES,
    prevent_initial_call=True,
)
def matplotlib_plot(_n, state, *values):
    """The old script's figure for the assigned peaks of the shown harmonic, in
    its own window (a child process, so it closes with this tool)."""
    import subprocess
    import tempfile
    import time
    if not state or not state.get("file"):
        return "Open a spectrum first."
    try:
        s = _settings(values, state)
        m = model(s, state)
        r = fit_harmonic(s, state, m)
    except Exception as exc:
        return f"⚠ {exc}"
    fd, path = tempfile.mkstemp(prefix="gui_corr_points_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump({"points": [{"ion": p["ion"], "x": p["x"], "y": p["y"]} for p in r["points"]],
                   "x0": r["x0"], "ref_ion": s["ref_name"], "ref_f": r["ref_f_h"], "h": r["h"],
                   "file": state["file"]}, fh)
    err = tempfile.TemporaryFile()
    # stdout: this program's original one (the plot keeps its own log, core/logcapture.py)
    proc = subprocess.Popen([sys.executable, CORR_SCRIPT, "--points", path],
                            cwd=paths.REPO_ROOT, stderr=err,
                            stdout=logcapture.child_stdio().get("stdout"))
    time.sleep(1.5)
    if proc.poll() not in (None, 0):
        err.seek(0)
        tail = err.read().decode(errors="replace").strip().splitlines()[-3:]
        return "⚠ The plot window failed: " + " ".join(tail)
    return f"Opened the plot window for h{r['h']} ({r['n']} assigned ions)."


@app.callback(
    Output("g-fit-table", "children"), Output("g-fit-graph", "figure"),
    Output("g-fit-graph", "style"), Output("g-fit-when", "children"),
    Output("g-fit-head", "children"), Output("g-psave", "children"),
    Output("g-calc-msg", "children", allow_duplicate=True),
    Input("g-hcur", "data"), Input("g-fitstamp", "data"),
    State("g-state", "data"),
    prevent_initial_call="initial_duplicate",
)
def show_fit(h, _stamp, state):
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
    Input("g-psave", "n_clicks"), State("g-state", "data"), State("g-hcur", "data"),
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
