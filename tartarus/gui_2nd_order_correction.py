#!/usr/bin/env python3
"""
gui_2nd_order_correction.py  –  GUI Second Order Correction

Fit the 2nd-order frequency correction by hand on a spectrum, as a Dash page
(http://127.0.0.1:8056/), started from the Asphodel home page (Tartarus).

1.  Open a 1-D or summed spectrum (.npz) from the connected spectra_1d /
    summed_1d folders.  The simulated ion lines are drawn as in Theia's
    Viewer: the strongest peak is the reference ion (208Pb80+) at the
    reference harmonic.  Every other harmonic is placed from the reference
    ion's own peak in that harmonic (f_ref / h_ref * h, nearest peak within
    ± ref_search_hz; core.physics.harmonic_references).  No correction is
    applied to the lines: they are what the correction is fitted against.
2.  Drag the lines into place.  "Stretch all": dragging any line stretches
    the lines of its harmonic (or of all harmonics, if linked) about that
    harmonic's reference line, which stays fixed.  "Move one line": the
    dragged line moves on its own; dragging the reference line of a harmonic
    other than the reference harmonic moves that harmonic's reference to the
    peak where it is dropped.
3.  Assign peaks: click the ▼ marker at the top of a line to select it, then
    click on its peak in the spectrum (or "Assign to nearest peak").  The
    measured frequency f_exp is the highest bin within ± the peak window,
    refined with a 3-point parabola.  The line then sits on f_exp.
4.  "Calculate": for every harmonic with at least 3 assigned ions, the
    anchored fit of the existing tool (tartarus/new_2nd_order_correction_plot.py,
    core.corrections.fit_correction), in revolution frequencies:
        x = f_sim / h   (that harmonic's own reference, uncorrected)
        y = f_exp / h - x,  anchored at the reference line (x0, 0)
        y = a x^2 + b x + c
5.  "Save as profile": one correction profile with a, b, c for every fitted
    harmonic; the general a, b, c (used for harmonics without their own) are
    the reference harmonic's fit.  "Also make it the default" puts it in the
    shared physics settings (physics.json).

The stretch, moves, reference changes and assignments are saved per spectrum
file (data/<exp>/settings/gui_correction_sessions.json) and restored when the
file is opened again.  Settings: data/<exp>/settings/gui_correction.json
(only the values that differ from physics.json).
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
SETTINGS_FILE = "gui_correction.json"
SESSIONS_FILE = "gui_correction_sessions.json"
MIN_POINTS = 3              # assigned ions per harmonic for a fit (2 parameters + 1 dof)
Y_TOP = 0.86               # top of the spectrum (paper); the marker strip is above it
MAX_SHAPES = 500            # draggable lines drawn at once (zoom in for more)
# one colour per harmonic (Theia's highlight colours)
H_COLOURS = ["#2E91E5", "#E15F99", "#1CA71C", "#DA16FF", "#B68100", "#750D86", "#00A08B",
             "#FB0D0D", "#511CFB", "#6C7C32"]

# Settings shown in the sidebar.  The shared physics keys are placeholders:
# their values come from physics.json; gui_correction.json keeps overrides.
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


def load_session(path):
    s = _read_json(paths.settings_file(SESSIONS_FILE)).get(_session_key(path)) or {}
    return new_state(path, s)


def save_session(state):
    if not state or not state.get("file"):
        return
    with _LOCK:
        p = paths.settings_file(SESSIONS_FILE)
        all_ = _read_json(p)
        all_[_session_key(state["file"])] = {
            k: state.get(k) for k in ("k", "link", "offsets", "ref_override", "assign", "fit")}
        all_[_session_key(state["file"])]["saved"] = datetime.datetime.now().isoformat(
            timespec="seconds")
        try:
            _write_json(p, all_)
        except OSError:
            pass


def new_state(path, saved=None):
    saved = saved or {}
    return {"file": path or "", "k": saved.get("k") or {}, "link": saved.get("link", True),
            "offsets": saved.get("offsets") or {}, "ref_override": saved.get("ref_override") or {},
            "assign": saved.get("assign") or {}, "fit": saved.get("fit"), "selected": None}


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


def model(s, state):
    """Everything the plot and the fit need, for settings s and the state.

    Returns dict: ff, pp, f_ref (strongest bin), refs {h: reference line at h},
    ref_info, warnings, lines (DataFrame: key, ion, h, f_sim, is_ref, hover,
    f_show, assigned, moved)."""
    ff, pp = load_spectrum(state["file"])
    i = int(np.nanargmax(pp))
    f_ref = float(ff[i])
    ion_db, ref_mq = physics.build_ion_database(tuple(s["z_lims"]), tuple(s["n_lims"]),
                                                tuple(s["el1"]), tuple(s["el2"]),
                                                s["ref_name"], float(s["mq_width"]))
    refs, info = physics.harmonic_references(ff, pp, f_ref, s["ref_harmonic"], s["harmonics"],
                                             s["ref_search_hz"])
    warnings = physics.reference_warnings(info, s["ref_name"], s["ref_search_hz"])
    for h, f in (state.get("ref_override") or {}).items():
        if int(h) in refs and int(h) != s["ref_harmonic"]:
            refs[int(h)] = float(f)
            info[int(h)] = dict(info[int(h)], f=float(f), found="moved")
    warnings = [w for w in warnings
                if not any(w.startswith(f"h{h}:") for h in (state.get("ref_override") or {}))]
    sims = physics.simulate_frequencies(ion_db, ref_mq, f_ref, s["ref_harmonic"], s["gamma_t"],
                                        s["harmonics"], False, ref_by_h=refs)
    span = float(ff[-1] - ff[0])
    lo, hi = ff[0] - 0.05 * span, ff[-1] + 0.05 * span
    sims = sims[(sims["f_sim"] >= lo) & (sims["f_sim"] <= hi)].reset_index(drop=True)
    rows = sims["ion_row"].to_numpy()
    names = ion_db["ion"].to_numpy()[rows].astype(str)
    hs = sims["harmonic"].to_numpy().astype(int)
    f_sim = sims["f_sim"].to_numpy(float)
    lines = pd.DataFrame({"key": [line_key(n, h) for n, h in zip(names, hs)], "ion": names,
                          "h": hs, "f_sim": f_sim, "hover": ion_db["hover"].to_numpy()[rows],
                          "mq_unc": ion_db["m/q_unc"].to_numpy(float)[rows]
                          if "m/q_unc" in ion_db else np.nan})
    lines = lines.drop_duplicates("key").reset_index(drop=True)
    lines["is_ref"] = lines["ion"] == s["ref_name"]
    k = state.get("k") or {}
    offsets = state.get("offsets") or {}
    assign = state.get("assign") or {}
    r = lines["h"].map(refs).to_numpy(float)
    kk = lines["h"].map(lambda h: float(k.get(str(h), 1.0))).to_numpy(float)
    off = lines["key"].map(lambda q: float(offsets.get(q, 0.0))).to_numpy(float)
    show = r + kk * (lines["f_sim"].to_numpy(float) - r) + off
    show = np.where(lines["is_ref"], lines["f_sim"], show)
    lines["assigned"] = lines["key"].isin(list(assign))
    lines["f_exp"] = lines["key"].map(lambda q: (assign.get(q) or {}).get("f_exp", np.nan))
    lines["f_show"] = np.where(lines["assigned"], lines["f_exp"], show)
    lines["moved"] = lines["key"].isin(list(offsets))
    return {"ff": ff, "pp": pp, "f_ref": f_ref, "refs": refs, "ref_info": info,
            "warnings": warnings, "lines": lines}


def h_colour(h, harmonics):
    try:
        return H_COLOURS[list(harmonics).index(int(h)) % len(H_COLOURS)]
    except ValueError:
        return "#888888"


def overlay(s, state, m, shown, view):
    """The parts of the plot that change when lines move: one ▼ marker trace
    per harmonic of the settings (empty if hidden; at the top of the plot, for
    hover and click-to-select), the draggable lines (shapes) and the labels.
    Returns (marker traces, shapes, annotations, shape keys in order, note)."""
    lines = m["lines"][m["lines"]["h"].isin(shown)]
    sel = state.get("selected")
    traces = []
    for h in s["harmonics"]:
        d = lines[lines["h"] == h]
        hov = [f"{hv}<br>Harmonic: {h}<br>f_sim = {fs:,.1f} Hz (own reference, uncorrected)"
               + (f"<br>shown at {fsh:,.1f} Hz" if not ref and not a else "")
               + (f"<br><b>assigned: f_exp = {fe:,.1f} Hz</b>" if a else "")
               + ("<br><b>reference line</b>" if ref else "")
               for hv, fs, fsh, fe, a, ref in zip(d["hover"], d["f_sim"], d["f_show"], d["f_exp"],
                                                  d["assigned"], d["is_ref"])]
        col = h_colour(h, s["harmonics"])
        traces.append(go.Scatter(
            x=d["f_show"].tolist(), y=[0.5] * len(d), yaxis="y2", mode="markers",
            name=f"h={h}", customdata=d["key"].tolist(), hovertext=hov, hoverinfo="text",
            showlegend=h in shown,
            marker=dict(symbol="triangle-down", size=[14 if q == sel else 10 for q in d["key"]],
                        color=[ACCENT if r else col for r in d["is_ref"]],
                        line=dict(width=[2 if (q == sel or a) else 0
                                         for q, a in zip(d["key"], d["assigned"])],
                                  color="black"))).to_plotly_json())
    # draggable lines: only those in view (plus a margin), at most MAX_SHAPES
    vis = lines
    if view:
        w = view[1] - view[0]
        vis = lines[(lines["f_show"] >= view[0] - 0.1 * w) & (lines["f_show"] <= view[1] + 0.1 * w)]
    note = ""
    if len(vis) > MAX_SHAPES:
        vis = vis[vis["is_ref"] | vis["assigned"] | (vis["key"] == sel)]
        note = (f"{len(lines)} lines in view – too many to drag at once: zoom in to drag them "
                f"(the ▼ markers still select).")
    shapes, keys, ann = [], [], []
    for r in vis.itertuples(index=False):
        col = h_colour(r.h, s["harmonics"])
        if r.key == sel:
            ln = dict(color="black", width=3)
        elif r.is_ref:
            ln = dict(color=ACCENT, width=2.5)
        elif r.assigned:
            ln = dict(color=col, width=2.5)
        elif r.moved:
            ln = dict(color=col, width=1.5, dash="dot")
        else:
            ln = dict(color=col, width=1, dash="dash")
        shapes.append(dict(type="line", xref="x", yref="paper", x0=r.f_show, x1=r.f_show,
                           y0=0, y1=Y_TOP, line=ln, layer="above"))
        keys.append(r.key)
        if r.assigned or r.key == sel or r.is_ref:
            ann.append(dict(x=r.f_show, y=Y_TOP - 0.005, xref="x", yref="paper", yanchor="top",
                            text=f"{r.ion} h{r.h}", showarrow=False, textangle=-90,
                            font=dict(size=11, color="black" if r.key == sel else
                                      (ACCENT if r.is_ref else col)),
                            bgcolor="rgba(255,255,255,0.8)"))
    return traces, shapes, ann, keys, note


def figure(s, state, m, shown, view):
    """Full plot: spectrum (trace 0) + overlay.  Returns (figure, keys, note)."""
    traces, shapes, ann, keys, note = overlay(s, state, m, shown, view)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=m["ff"], y=m["pp"], mode="lines", name="Spectrum",
                               line=dict(shape="hvh", color="black", width=1),
                               hovertemplate="%{x:.1f} Hz<extra></extra>"))
    for t in traces:
        fig.add_trace(t)
    fig.update_layout(
        shapes=shapes, annotations=ann, template="plotly_white", dragmode="pan",
        hovermode="closest", font=dict(size=16), height=660,
        margin=dict(l=60, r=20, t=40, b=50),
        xaxis=dict(title="Frequency (Hz)", tickformat=".0f"),
        yaxis=dict(title="Schottky power (a.u.)", domain=[0, Y_TOP]),
        # the ▼ markers: a strip of their own above the spectrum, so a click on
        # a marker is not taken by the (draggable) line under it
        yaxis2=dict(domain=[Y_TOP + 0.01, 0.97], range=[0, 1], visible=False, fixedrange=True,
                    anchor="x"),
        legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0, xanchor="left"),
        uirevision=state.get("file") or "none")
    return fig, keys, note


# ===========================================================================
# The fit
# ===========================================================================

def fit_all(s, state, m):
    """Anchored fit per harmonic.  Returns a JSON-able dict."""
    lines = m["lines"].set_index("key")
    out = {"harmonics": {}, "skipped": {}, "file": state["file"], "f_ref": m["f_ref"],
           "ref_name": s["ref_name"], "ref_harmonic": s["ref_harmonic"], "gamma_t": s["gamma_t"],
           "ref_search_hz": s["ref_search_hz"], "peak_window_hz": s["peak_window_hz"],
           "missing": [], "time": datetime.datetime.now().isoformat(timespec="seconds")}
    by_h = {}
    for key, a in (state.get("assign") or {}).items():
        if key not in lines.index:
            out["missing"].append(key)
            continue
        r = lines.loc[key]
        if r["is_ref"]:
            continue
        by_h.setdefault(int(r["h"]), []).append((key, r["ion"], float(r["f_sim"]),
                                                 float(a["f_exp"])))
    for h in s["harmonics"]:
        pts = by_h.get(h, [])
        if len(pts) < MIN_POINTS:
            if pts or h == s["ref_harmonic"]:
                out["skipped"][str(h)] = f"{len(pts)} assigned ion(s), needs {MIN_POINTS}"
            continue
        x = np.array([p[2] for p in pts]) / h
        y = np.array([p[3] for p in pts]) / h - x
        x0 = m["refs"][h] / h
        res = corrections.fit_correction(x, y, x0)
        info = m["ref_info"].get(h, {})
        out["harmonics"][str(h)] = {
            "n": res["n"], "abc": list(res["abc"]), "abc_unc": list(res["abc_unc"]),
            "s1": res["s1"], "s2": res["s2"], "ds1": res["ds1"], "ds2": res["ds2"],
            "x0": x0, "ref_f_h": float(m["refs"][h]), "rms": res["rms"],
            "ref_found": info.get("found"), "stretch": float((state.get("k") or {}).get(str(h), 1)),
            "points": [{"key": p[0], "ion": p[1], "f_sim": p[2], "f_exp": p[3], "x": float(xi),
                        "y": float(yi), "resid": float(ri)}
                       for p, xi, yi, ri in zip(pts, x, y, res["resid"])]}
    return out


def fit_figure(fit, harmonics):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
                        vertical_spacing=0.04)
    for hs, r in sorted(fit["harmonics"].items()):
        h = int(hs)
        col = h_colour(h, harmonics)
        x = np.array([p["x"] for p in r["points"]])
        y = np.array([p["y"] for p in r["points"]])
        names = [p["ion"] for p in r["points"]]
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", text=names, name=f"h{h} data",
                                 legendgroup=hs, marker=dict(color=col, size=9),
                                 hovertemplate="%{text}<br>f_sim = %{x:.3f} Hz<br>"
                                               "f_diff = %{y:.3f} Hz<extra>h" + hs + "</extra>"),
                      row=1, col=1)
        xx = np.linspace(min(x.min(), r["x0"]), max(x.max(), r["x0"]), 300)
        pad = 0.05 * (xx[-1] - xx[0] or 1)
        xx = np.linspace(xx[0] - pad, xx[-1] + pad, 300)
        fig.add_trace(go.Scatter(x=xx, y=r["s1"] * (xx - r["x0"]) + r["s2"] * (xx - r["x0"]) ** 2,
                                 mode="lines", line=dict(color=col, width=2), name=f"h{h} fit",
                                 legendgroup=hs, hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=[r["x0"]], y=[0], mode="markers", name=f"h{h} anchor",
                                 legendgroup=hs, marker=dict(symbol="star", size=16, color=col,
                                                             line=dict(color="black", width=1)),
                                 hovertemplate=f"anchor (reference, h{h})<br>x0 = %{{x:.3f}} Hz"
                                               "<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=[p["resid"] for p in r["points"]], mode="markers",
                                 marker=dict(color=col, size=8), legendgroup=hs, showlegend=False,
                                 text=names, hovertemplate="%{text}<br>residual = %{y:.3f} Hz"
                                                           "<extra></extra>"), row=2, col=1)
    fig.add_hline(y=0, line=dict(color="grey", width=1, dash="dash"), row=1, col=1)
    fig.add_hline(y=0, line=dict(color="grey", width=1), row=2, col=1)
    fig.update_layout(template="plotly_white", height=620, font=dict(size=15),
                      margin=dict(l=70, r=20, t=30, b=50), legend=dict(orientation="v"))
    fig.update_yaxes(title_text="f_diff = f_exp,true − f_sim,true (Hz)", row=1, col=1)
    fig.update_yaxes(title_text="Residual (Hz)", row=2, col=1)
    fig.update_xaxes(title_text="f_sim,true (Hz, revolution)", tickformat=".0f", row=2, col=1)
    return fig


def default_profile_name(fit):
    hs = " ".join(f"h{h}" for h in sorted(fit["harmonics"], key=int))
    n = sum(r["n"] for r in fit["harmonics"].values())
    return f"{datetime.datetime.now():%Y-%m-%d %H:%M} GUI, {hs}, {n} ions"


def profile_from_fit(name, fit):
    """Correction profile (core/corrections.py, version 2) from a fit result."""
    href = str(fit["ref_harmonic"])
    if href not in fit["harmonics"]:
        raise ValueError(f"the reference harmonic h{href} is not fitted: its fit is the "
                         f"profile's general a, b, c")

    def fit_info(h, r):
        return {"ref_ion": fit["ref_name"], "ref_f_exp": fit["f_ref"],
                "ref_harmonic": fit["ref_harmonic"], "harmonic": int(h), "ref_f_h": r["ref_f_h"],
                "f_ref_rev": r["x0"], "gamma_t": fit["gamma_t"], "mass_model": MASS_MODEL,
                "weighted": False, "n_points": r["n"], "rms_rev_hz": r["rms"], "chi2_ndf": None,
                "ions": [f"{p['ion']} h{h}" for p in r["points"]],
                "ref_peak": {True: "found", False: "predicted", None: "search off",
                             "moved": "moved by hand"}.get(r.get("ref_found"), str(r.get("ref_found"))),
                "stretch": r.get("stretch")}

    per_h = {}
    for h, r in fit["harmonics"].items():
        (a, b, c), (da, db, dc) = r["abc"], r["abc_unc"]
        per_h[h] = {"a": a, "b": b, "c": c, "a_unc": da, "b_unc": db, "c_unc": dc,
                    "fit": fit_info(h, r)}
    g = fit["harmonics"][href]
    top = dict(fit_info(href, g), n_points=sum(r["n"] for r in fit["harmonics"].values()),
               ions=[i for h in sorted(per_h, key=int) for i in per_h[h]["fit"]["ions"]],
               source_file=fit["file"], sheet=None, source=APP_NAME,
               ref_search_hz=fit["ref_search_hz"], peak_window_hz=fit["peak_window_hz"],
               per_harmonic=True)
    (a, b, c), (da, db, dc) = g["abc"], g["abc_unc"]
    return {"name": name.strip(), "a": a, "b": b, "c": c, "a_unc": da, "b_unc": db, "c_unc": dc,
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "fit": top, "per_harmonic": per_h,
            "notes": f"Fitted in the GUI on {os.path.basename(fit['file'])}."}


def fit_table(fit):
    head = ["Harmonic", "Ions", "a", "b", "c", "RMS (Hz, rev)", "Reference line"]
    rows = []
    for h, r in sorted(fit["harmonics"].items(), key=lambda kv: int(kv[0])):
        (a, b, c), (da, db, dc) = r["abc"], r["abc_unc"]
        refp = {True: "peak found", False: "predicted (no peak)", None: "f_ref/h_ref·h",
                "moved": "moved by hand"}.get(r.get("ref_found"), "")
        rows.append([f"h{h}" + (" (general)" if int(h) == fit["ref_harmonic"] else ""), r["n"],
                     f"{a:.6e} ± {da:.2e}", f"{b:.6e} ± {db:.2e}", f"{c:.6e} ± {dc:.2e}",
                     f"{r['rms']:.3f}", f"{r['ref_f_h']:,.1f} Hz · {refp}"])
    for h, why in sorted(fit["skipped"].items(), key=lambda kv: int(kv[0])):
        rows.append([f"h{h}", "–", "not fitted: " + why, "", "", "", ""])
    return html.Table([html.Thead(html.Tr([html.Th(x) for x in head])),
                       html.Tbody([html.Tr([html.Td(x) for x in r]) for r in rows])],
                      className="tbl")


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
"""

app = dash.Dash(__name__, title=APP_NAME, suppress_callback_exceptions=True)
app.index_string = app.index_string.replace("</head>", f"<style>{CSS}</style></head>")
app.index_string = app.index_string.replace(
    "{%scripts%}", "{%scripts%}\n<script>" + tabwatch.watch_js(APP_NAME) + "</script>")
icon.use_in_dash(app, "gui_correction")      # core/icons/gui_correction.png, else asphodel.png


def serve_layout():
    cfg = load_settings()
    files = spectrum_files()
    values = [o["value"] for o in files]
    f = cfg.get("file") if cfg.get("file") in values else (values[0] if values else "")
    state = load_session(f) if f else new_state("")
    fields = []
    for k in SETTING_KEYS:
        v = cfg[k]
        v = ", ".join(str(x) for x in v) if isinstance(v, list) else physics._fmt_num(v) \
            if isinstance(v, float) else str(v)
        fields += [html.Label(LABELS[k]), dcc.Input(id=f"g-{k}", type="text", value=v, debounce=True)]
    fit = state.get("fit")
    return html.Div([
        dcc.Store(id="g-state", data=state),
        dcc.Store(id="g-view", data=None),
        dcc.Store(id="g-keys", data=[]),
        dcc.Store(id="g-sig", data=None),
        html.Div([html.H1(APP_NAME), html.Span(f"experiment {paths.experiment()}", className="exp")],
                 className="top"),
        html.Div([
            html.Div([
                html.H3("Spectrum"),
                dcc.Dropdown(id="g-file", options=files, value=f or None, clearable=False,
                             placeholder="No .npz in the connected spectra folders",
                             style={"fontSize": "12px"}),
                html.H3("Simulation"),
                *fields,
                html.Div("Shared defaults from physics.json; changes here are kept for this tool "
                         "only.  No correction is applied to the lines.", className="note"),
                html.H3("Dragging"),
                dcc.RadioItems(id="g-mode", value="stretch", options=[
                    {"label": " Stretch all about the reference", "value": "stretch"},
                    {"label": " Move one line", "value": "one"}],
                    labelStyle={"display": "block"}),
                dcc.Checklist(id="g-link", value=["on"] if state.get("link", True) else [],
                              options=[{"label": " Same stretch for all harmonics", "value": "on"}]),
                html.Div(id="g-stretch", className="note"),
                html.Button("Reset stretch & moves", id="g-reset", className="btn", n_clicks=0),
                html.H3("Harmonics shown"),
                dcc.Checklist(id="g-shown", value=None, options=[], inline=True),
            ], className="side"),
            html.Div([
                html.Div([
                    html.Div([html.Span(id="g-title"), html.Span(id="g-info", className="note")],
                             className="head"),
                    html.Div([
                        html.Div(id="g-warn", className="warn"),
                        dcc.Graph(id="g-graph", config={"edits": {"shapePosition": True},
                                                        "scrollZoom": True,
                                                        "displaylogo": False}),
                        html.Div(id="g-sel", className="sel"),
                        html.Div([
                            html.Button("Assign to nearest peak", id="g-assign", className="btn primary",
                                        n_clicks=0),
                            html.Button("Unassign", id="g-unassign", className="btn", n_clicks=0),
                            html.Button("Deselect", id="g-deselect", className="btn", n_clicks=0),
                            html.Span(id="g-msg", className="note"),
                        ], className="row"),
                        html.Div("Click a ▼ marker to select its line, then click on the peak in "
                                 "the spectrum (or 'Assign to nearest peak').  Drag lines with the "
                                 "mouse; scroll to zoom, drag the background to pan.",
                                 className="note"),
                    ], className="body"),
                ], className="box"),
                html.Div([
                    html.Div([html.Span("Assignments"),
                              html.Div([html.Button("Calculate correction", id="g-calc",
                                                    className="btn primary", n_clicks=0),
                                        html.Button("Clear all assignments", id="g-clear",
                                                    className="btn", n_clicks=0)])],
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
                    html.Div([html.Span("Second-order correction"),
                              html.Span(id="g-fit-when", className="note")], className="head"),
                    html.Div([
                        html.Div(fit_table(fit) if fit else "Not calculated yet.", id="g-fit-table"),
                        dcc.Graph(id="g-fit-graph", figure=fit_figure(fit, parse_harm(cfg))
                                  if fit and fit.get("harmonics") else go.Figure(),
                                  style={"display": "block" if fit and fit.get("harmonics")
                                         else "none"}),
                        html.Div([
                            html.Label("Profile name", style={"fontWeight": 600}),
                            dcc.Input(id="g-pname", type="text", style={"width": "420px"},
                                      value=default_profile_name(fit) if fit and fit.get("harmonics")
                                      else ""),
                            dcc.Checklist(id="g-pdefault", value=[], options=[
                                {"label": " also make it the default", "value": "on"}]),
                            html.Button("Save as profile", id="g-psave", className="btn primary",
                                        n_clicks=0),
                        ], className="row"),
                        html.Div(id="g-pmsg"),
                    ], className="body"),
                ], className="box"),
            ], className="main"),
        ], className="wrap"),
    ])


def parse_harm(cfg):
    try:
        return sorted(set(_ints(cfg.get("harmonics"))) | {int(float(cfg.get("ref_harmonic") or 0))})
    except ValueError:
        return []


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


@app.callback(Output("g-shown", "options"), Output("g-shown", "value"),
              *SETTING_INPUTS, State("g-shown", "value"), State("g-shown", "options"))
def shown_harmonics(*args):
    values, cur, old = args[:-2], args[-2], args[-1]
    try:
        hs = _settings(values)["harmonics"]
    except ValueError:
        return dash.no_update, dash.no_update
    before = {o["value"] for o in (old or [])}
    on = set(cur or [])
    return ([{"label": f" h{h} ", "value": h} for h in hs],
            [h for h in hs if cur is None or h in on or h not in before])


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

@app.callback(
    Output("g-graph", "figure"), Output("g-keys", "data"), Output("g-title", "children"),
    Output("g-info", "children"), Output("g-warn", "children"), Output("g-sel", "children"),
    Output("g-table", "data"), Output("g-stretch", "children"), Output("g-sig", "data"),
    Input("g-state", "data"), Input("g-view", "data"), Input("g-shown", "value"), *SETTING_INPUTS,
    State("g-sig", "data"),
)
def draw(state, view, shown, *values):
    values, sig_old = values[:-1], values[-1]
    nu = dash.no_update
    if not state or not state.get("file"):
        return go.Figure(), [], "No spectrum", "", "", "", [], "", None
    title = os.path.basename(state["file"])
    try:
        s = _settings(values)
        m = model(s, state)
    except Exception as exc:
        return go.Figure(), [], title, "", f"⚠ {exc}", "", nu, "", None
    shown = s["harmonics"] if shown is None else [int(h) for h in shown]
    sig = [state["file"], s["harmonics"]]
    if sig == sig_old:
        # same spectrum and harmonics: only the overlay is sent (not the spectrum)
        traces, shapes, ann, keys, note = overlay(s, state, m, shown, view)
        fig = dash.Patch()
        for j, t in enumerate(traces):
            fig["data"][1 + j] = t
        fig["layout"]["shapes"] = shapes
        fig["layout"]["annotations"] = ann
    else:
        fig, keys, note = figure(s, state, m, shown, view)
    lines = m["lines"].set_index("key")
    info = (f"reference {s['ref_name']} h{s['ref_harmonic']} = strongest peak at "
            f"{m['f_ref']:,.1f} Hz · {len(lines)} lines")
    warn = [html.Div("⚠ " + w) for w in m["warnings"]] + ([html.Div(note)] if note else [])
    refs = " · ".join(f"h{h}: {f:,.1f} Hz" + {True: "", False: " (predicted)", None: "",
                                              "moved": " (moved)"}[m["ref_info"][h]["found"]]
                      for h, f in sorted(m["refs"].items()))
    warn.append(html.Div("Reference lines: " + refs, className="note"))
    sel = state.get("selected")
    if sel and sel in lines.index:
        r = lines.loc[sel]
        txt = [html.B(f"Selected: {r['ion']} h{r['h']}"),
               f" · f_sim {r['f_sim']:,.1f} Hz"]
        if r["is_ref"]:
            txt.append(" · reference line" + ("" if r["h"] != s["ref_harmonic"] else
                                               " (fixed)"))
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
    for key, a in sorted((state.get("assign") or {}).items(),
                         key=lambda kv: (split_key(kv[0])[1], kv[1]["f_exp"])):
        ion, h = split_key(key)
        if key in lines.index:
            fs = float(lines.loc[key, "f_sim"])
            rows.append({"key": key, "ion": ion, "h": h, "f_sim": f"{fs:,.1f}",
                         "f_exp": f"{a['f_exp']:,.1f}", "diff": f"{a['f_exp'] - fs:+.1f}",
                         "diff_rev": f"{(a['f_exp'] - fs) / h:+.3f}", "note": ""})
        else:
            rows.append({"key": key, "ion": ion, "h": h, "f_sim": "", "f_exp": f"{a['f_exp']:,.1f}",
                         "diff": "", "diff_rev": "", "note": "not in the simulation (settings?)"})
    k = state.get("k") or {}
    stretch = "Stretch: " + ", ".join(f"h{h} ×{float(k.get(str(h), 1)):.6f}" for h in s["harmonics"])
    n_off = len(state.get("offsets") or {})
    if n_off:
        stretch += f" · {n_off} line(s) moved on their own"
    return fig, keys, title, info, warn, txt, rows, stretch, sig


# ---------------------------------------------------------------------------
# State changes: dragging, clicks, buttons, table deletions, file change
# ---------------------------------------------------------------------------

def _drag(relayout, keys, state, s, m, mode, link):
    """Apply a shape drag from relayoutData.  Returns a message ('' if none)."""
    moved = {}
    for k, v in relayout.items():
        mt = re.fullmatch(r"shapes\[(\d+)\]\.(x0|x1)", k)
        if mt:
            moved.setdefault(int(mt.group(1)), {})[mt.group(2)] = float(v)
    if not moved:
        return None
    lines = m["lines"].set_index("key")
    msg = ""
    for idx, xs in moved.items():
        if idx >= len(keys) or keys[idx] not in lines.index:
            continue
        key = keys[idx]
        r = lines.loc[key]
        old = float(r["f_show"])
        x_new = max(xs.values(), key=lambda v: abs(v - old))
        h = int(r["h"])
        ref_h = float(m["refs"][h])
        if r["is_ref"]:
            if h == s["ref_harmonic"]:
                msg = f"The reference line h{h} is the strongest peak and stays fixed."
            elif mode == "one":
                f, _ = physics.peak_near(m["ff"], m["pp"], x_new, s["peak_window_hz"])
                state.setdefault("ref_override", {})[str(h)] = float(f if f is not None else x_new)
                msg = (f"Reference line h{h} moved to " + (f"the peak at {f:,.1f} Hz" if f is not None
                                                           else f"{x_new:,.1f} Hz (no peak there)"))
            else:
                msg = "Reference lines stay fixed when stretching (use 'Move one line')."
            continue
        if r["assigned"]:
            msg = f"{r['ion']} h{h} is assigned; unassign it first to move it."
            continue
        if mode == "one":
            base = ref_h + float((state.get("k") or {}).get(str(h), 1.0)) * (r["f_sim"] - ref_h)
            state.setdefault("offsets", {})[key] = float(x_new - base)
            continue
        d = float(r["f_sim"] - ref_h)
        if abs(d) < 1e-6:
            continue
        kn = (x_new - float((state.get("offsets") or {}).get(key, 0.0)) - ref_h) / d
        if not np.isfinite(kn) or kn <= 0:
            msg = "A line cannot be dragged across its reference line."
            continue
        hs = s["harmonics"] if link else [h]
        state["k"] = dict(state.get("k") or {}, **{str(x): float(kn) for x in hs})
        msg = f"Stretch ×{kn:.6f}" + (" (all harmonics)" if link else f" (h{h})")
    return msg


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
    state.get("offsets", {}).pop(key, None)
    return f"Assigned {r['ion']} h{int(r['h'])} → {f:,.1f} Hz (f_exp − f_sim = {f - r['f_sim']:+.1f} Hz)."


@app.callback(
    Output("g-state", "data"), Output("g-view", "data"), Output("g-msg", "children"),
    Input("g-graph", "relayoutData"), Input("g-graph", "clickData"),
    Input("g-assign", "n_clicks"), Input("g-unassign", "n_clicks"), Input("g-deselect", "n_clicks"),
    Input("g-reset", "n_clicks"), Input("g-clear", "n_clicks"), Input("g-table", "data_timestamp"),
    Input("g-file", "value"), Input("g-link", "value"),
    State("g-table", "data"), State("g-state", "data"), State("g-view", "data"),
    State("g-keys", "data"), State("g-mode", "value"), *SETTING_STATES,
    prevent_initial_call=True,
)
def update_state(relayout, click, _a, _u, _d, _r, _c, _ts, file, link, table, state, view, keys,
                 mode, *values):
    nu = dash.no_update
    trig = ctx.triggered_id
    state = dict(state or new_state(file))
    if trig == "g-file":
        if file == state.get("file"):
            return nu, nu, nu
        save_settings({"file": file})
        return load_session(file), None, f"Opened {os.path.basename(file)}."
    if trig == "g-link":
        state["link"] = bool(link)
        save_session(state)
        return state, nu, nu
    if not state.get("file"):
        return nu, nu, nu
    try:
        s = _settings(values)
        m = model(s, state)
    except Exception as exc:
        return nu, nu, f"⚠ {exc}"
    msg = nu
    new_view = nu
    props = set(ctx.triggered_prop_ids)
    # a click can come together with a relayout (pan mode): the click wins
    if trig == "g-graph" and "g-graph.clickData" not in props:
        relayout = relayout or {}
        if "xaxis.range[0]" in relayout and "xaxis.range[1]" in relayout:
            new_view = [float(relayout["xaxis.range[0]"]), float(relayout["xaxis.range[1]"])]
        elif "xaxis.range" in relayout:
            new_view = [float(x) for x in relayout["xaxis.range"]]
        elif relayout.get("xaxis.autorange"):
            new_view = None
        out = _drag(relayout, keys or [], state, s, m, mode, bool(link))
        if out is None:
            return nu, new_view, nu
        msg = out
    elif trig == "g-graph":
        pt = (click or {}).get("points", [{}])[0]
        key = pt.get("customdata")
        if isinstance(key, list):
            key = key[0] if key else None
        if key:
            if state.get("selected") == key:
                return nu, nu, nu
            state["selected"] = key
            msg = ""
        elif state.get("selected") and pt.get("x") is not None:
            msg = _assign(state, s, m, state["selected"], float(pt["x"]))
        else:
            return nu, nu, "Select a line first (click its ▼ marker)."
    elif trig == "g-assign":
        sel = state.get("selected")
        if not sel:
            return nu, nu, "Select a line first (click its ▼ marker)."
        lines = m["lines"].set_index("key")
        if sel not in lines.index:
            return nu, nu, "The selected line is not in the simulation."
        msg = _assign(state, s, m, sel, float(lines.loc[sel, "f_show"]))
    elif trig == "g-unassign":
        sel = state.get("selected")
        if not sel or sel not in (state.get("assign") or {}):
            if sel and sel.rsplit("|", 1)[1] in (state.get("ref_override") or {}) \
                    and sel.startswith(s["ref_name"] + "|"):
                state["ref_override"].pop(sel.rsplit("|", 1)[1])
                msg = "Reference line back to the automatic search."
            else:
                return nu, nu, "The selected line is not assigned."
        else:
            state["assign"].pop(sel)
            msg = "Unassigned."
    elif trig == "g-deselect":
        state["selected"] = None
        msg = ""
    elif trig == "g-reset":
        state["k"], state["offsets"], state["ref_override"] = {}, {}, {}
        msg = "Stretch, moved lines and moved reference lines reset."
    elif trig == "g-clear":
        state["assign"] = {}
        msg = "All assignments cleared."
    elif trig == "g-table":
        left = {r["key"] for r in (table or [])}
        gone = [q for q in (state.get("assign") or {}) if q not in left]
        if not gone:
            return nu, nu, nu
        for q in gone:
            state["assign"].pop(q, None)
        msg = "Removed " + ", ".join(q.replace("|", " h") for q in gone) + "."
    save_session(state)
    return state, new_view, msg


# ---------------------------------------------------------------------------
# Calculate and save
# ---------------------------------------------------------------------------

@app.callback(
    Output("g-fit-table", "children"), Output("g-fit-graph", "figure"),
    Output("g-fit-graph", "style"), Output("g-pname", "value"), Output("g-calc-msg", "children"),
    Output("g-fit-when", "children"), Output("g-state", "data", allow_duplicate=True),
    Input("g-calc", "n_clicks"), State("g-state", "data"), *SETTING_STATES,
    prevent_initial_call=True,
)
def calculate(_n, state, *values):
    nu = dash.no_update
    if not state or not state.get("file"):
        return nu, nu, nu, nu, "Open a spectrum first.", nu, nu
    try:
        s = _settings(values)
        m = model(s, state)
        fit = fit_all(s, state, m)
    except Exception as exc:
        return nu, nu, nu, nu, f"⚠ {exc}", nu, nu
    msg = []
    if fit["missing"]:
        msg.append("Not used (not in the simulation): " + ", ".join(
            q.replace("|", " h") for q in fit["missing"]) + ". ")
    if not fit["harmonics"]:
        return (fit_table(fit), go.Figure(), {"display": "none"}, nu,
                "⚠ No harmonic has enough assigned ions (needs "
                f"{MIN_POINTS} besides the reference). " + "".join(msg), "", nu)
    state = dict(state, fit=fit)
    save_session(state)
    if str(s["ref_harmonic"]) not in fit["harmonics"]:
        msg.append(f"The reference harmonic h{s['ref_harmonic']} is not fitted, so the "
                   f"profile cannot be saved (its fit is the general a, b, c). ")
    return (fit_table(fit), fit_figure(fit, s["harmonics"]), {"display": "block"},
            default_profile_name(fit), "".join(msg) or "✓ Calculated.",
            f"calculated {datetime.datetime.now():%H:%M:%S}", state)


@app.callback(
    Output("g-pmsg", "children"),
    Input("g-psave", "n_clicks"), State("g-state", "data"), State("g-pname", "value"),
    State("g-pdefault", "value"),
    prevent_initial_call=True,
)
def save_profile(_n, state, name, make_default):
    fit = (state or {}).get("fit")
    if not fit or not fit.get("harmonics"):
        return html.Span("✗ Calculate the correction first.", className="err")
    try:
        prof = corrections.save_profile(profile_from_fit(name or "", fit))
    except (ValueError, OSError) as exc:
        return html.Span(f"✗ Not saved: {exc}", className="err")
    text = f"✓ Saved profile '{prof['name']}' ({', '.join('h' + h for h in sorted(prof['per_harmonic'], key=int))})"
    if make_default:
        physics.save_shared({"corr_profile": prof["name"], "corr_a": prof["a"],
                             "corr_b": prof["b"], "corr_c": prof["c"]})
        text += " and made it the default (physics.json)"
    return html.Span(text + ". Reload Theia / restart open tools to see it.", className="ok")


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
