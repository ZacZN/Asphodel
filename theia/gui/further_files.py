"""
Further output: confirmation files, summing, PID settings of the summed plot.

Part of Theia (theia/theia.py); split out of theia.py in Nov 2026.
"""

import glob
import json
import os
import threading
import time
import traceback

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import ALL, dcc, html, Input, Output, State

from core import corrections, paths, physics, summing
from core.physics import build_ion_database, hl_allowed, hl_cut_text, HL_CUT_UNITS, HL_UNITS
from core.scan import load_config, load_spectrum, natural_key, parse_settings

from .common import ACCENT, BOX_BORDER, BOX_STYLE, CONFIRMED_DIR, HELP_STYLE, LABEL_STYLE
from .app import app
from .figures import add_sim_lines, empty_fig, EXCLUDED_TRACE
from .corr_profiles import corr_profile_ui
from .folder_browser import list_npz
from .viewer import load_viewer_config, VIEWER_KEYS


PID_FIELDS = [   # key, label
    ("ref_name", "Reference Ion"), ("ref_harmonic", "Reference Harmonic"),
    ("z_lims", "Z Limits  [min, max]"), ("n_lims", "N Limits  [min, max]"),
    ("el1", "el1  [A, Z]"), ("el2", "el2  [A, Z]"),
    ("mq_width", "m/q window (± value)"), ("gamma_t", "γ_t"),
    ("harmonics", "Harmonics (comma-separated)"),
    ("ref_search_hz", "Ref. Peak Search, Other Harmonics ± (Hz)  [0 = off]"),
    ("corr_a", "a  (f² coefficient)"), ("corr_b", "b  (f coefficient)"),
    ("corr_c", "c  (constant)"),
    ("min_half_life", "Exclude Half-Life Below  [0 = off]"),
]
PID_KEYS = [k for k, _ in PID_FIELDS] + ["min_half_life_unit", "correction_enabled",
                                         "exclude_unknown_hl", "corr_profile"]

# Aligning and fitting: controls shown for the "confirmation files" source
# only, and the row of the "single spectrum file" source
SUM_CONTROLS_SHOWN = {"display": "flex", "alignItems": "center", "flexWrap": "wrap",
                      "rowGap": "6px"}
FILE_SRC_SHOWN = {"display": "flex", "alignItems": "center", "flexWrap": "wrap", "rowGap": "6px",
                  "marginBottom": "10px"}

def _pid_field(key, label):
    return html.Div([
        html.Label(label, style=LABEL_STYLE),
        dcc.Input(id=f"pid-{key}", type="text", debounce=False,
                  className="form-control form-control-sm", style={"fontSize": "13px"}),
    ], style={"marginBottom": "8px"})


pid_modal = html.Div([
    html.Div([
        html.Div([
            html.Span("PID Settings – Simulated Ion Lines",
                      style={"fontWeight": "700", "fontSize": "15px"}),
            html.Button("✕", id="pid-cancel", className="btn btn-outline-secondary btn-sm"),
        ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center",
                  "marginBottom": "4px"}),
        html.Small("The reference position is always the strongest peak of the plotted spectrum.  "
                   "Values start from the scan settings of the first loaded confirmation file, "
                   "or from the Viewer settings for a single spectrum file (each data source "
                   "keeps its own edits).",
                   style={**HELP_STYLE, "marginBottom": "8px"}),
        # shows / hides the lines straight away (no Apply needed)
        dcc.Checklist(id="sum-show-sims", value=["on"],
                      options=[{"label": " Show simulated ions on the plot", "value": "on"}],
                      style={"fontSize": "13px", "fontWeight": "600", "marginBottom": "10px"}),
        html.Div([
            html.Div([_pid_field(k, lab) for k, lab in PID_FIELDS[:7]]),
            html.Div([_pid_field(k, lab) for k, lab in PID_FIELDS[7:10]] + [
                dcc.Checklist(id="pid-correction_enabled",
                              options=[{"label": " Second order correction", "value": "on"}],
                              style={"fontSize": "13px", "marginBottom": "6px"}),
                corr_profile_ui("pid-"),
            ] + [_pid_field(k, lab) for k, lab in PID_FIELDS[10:13]] + [
                html.Label("Exclude Half-Life Below  [0 = off]", style=LABEL_STYLE),
                html.Div([
                    dcc.Input(id="pid-min_half_life", type="text",
                              className="form-control form-control-sm",
                              style={"flex": "1", "fontSize": "13px", "marginRight": "6px"}),
                    dcc.Dropdown(id="pid-min_half_life_unit", clearable=False,
                                 options=[{"label": u, "value": u} for u in HL_CUT_UNITS],
                                 style={"width": "80px", "fontSize": "13px"}),
                ], style={"display": "flex", "alignItems": "center", "marginBottom": "6px"}),
                dcc.Checklist(id="pid-exclude_unknown_hl",
                              options=[{"label": " Also exclude unknown half-life", "value": "on"}],
                              style={"fontSize": "13px"}),
            ]),
        ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "columnGap": "18px"}),
        html.Div(id="pid-error", style={"fontSize": "13px", "margin": "6px 0"}),
        html.Div([
            html.Button("Reset to File", id="pid-reset", className="btn btn-outline-secondary btn-sm",
                        title="Use the scan settings stored in the first loaded file",
                        style={"marginRight": "auto"}),
            html.Button("Apply", id="pid-apply", className="btn btn-sm",
                        style={"backgroundColor": ACCENT, "color": "white", "fontWeight": "600",
                               "minWidth": "80px"}),
        ], style={"display": "flex", "alignItems": "center", "marginTop": "6px"}),
    ], className="theia-box",
       style={"border": BOX_BORDER, "borderRadius": "6px", "background": "white",
              "width": "640px", "maxWidth": "94vw", "maxHeight": "92vh", "overflowY": "auto",
              "padding": "14px 16px", "boxShadow": "0 8px 30px rgba(0,0,0,0.3)"}),
], id="pid-modal",
   style={"display": "none", "position": "fixed", "inset": "0", "zIndex": "1000",
          "background": "rgba(0,0,0,0.35)", "alignItems": "center", "justifyContent": "center"})


# ===========================================================================
# Further output tab: confirmation-mode files
# ===========================================================================

def read_confirmed(path):
    """Read a confirmed/*.csv file -> (header dict, DataFrame).

    The '# key: value' lines at the top become the header dict (run_info and
    scan_settings are decoded from JSON); the rest is the table.
    """
    header = {}
    with open(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            key, _, value = line[1:].strip().partition(":")
            value = value.strip()
            if key in ("run_info", "scan_settings", "table_sort"):
                try:
                    value = json.loads(value)
                except ValueError:
                    pass
            header[key.strip()] = value
    table = pd.read_csv(path, comment="#")
    return header, table


def list_confirmed():
    """[(path, header)] for confirmed/*.csv, newest first."""
    out = []
    for path in sorted(glob.glob(os.path.join(CONFIRMED_DIR, "*.csv")),
                       key=os.path.getmtime, reverse=True):
        try:
            header = {}
            with open(path) as fh:
                for line in fh:
                    if not line.startswith("#"):
                        break
                    k, _, v = line[1:].strip().partition(":")
                    if k in ("saved", "filter", "accepted", "unsure"):
                        header[k] = v.strip()
            out.append((path, header))
        except OSError:
            continue
    return out


def load_further_files(paths):
    """Read the listed confirmation files and combine them into one table.
    Adds file_no (1, 2, ... as numbered in the list) and source_file columns."""
    frames = []
    for n, path in enumerate(paths, start=1):
        _, t = read_confirmed(path)
        t.insert(0, "file_no", n)
        t.insert(1, "source_file", os.path.basename(path))
        frames.append(t)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _confirmed_summary(path):
    """One-line description of a confirmation file for the numbered list."""
    header, table = read_confirmed(path)
    n_acc = int((table["decision"] == "accepted").sum()) if "decision" in table else len(table)
    n_uns = int((table["decision"] == "unsure").sum()) if "decision" in table else 0
    ions = ", ".join(sorted(table["ion"].unique(), key=natural_key)) if "ion" in table else "?"
    run = header.get("run_info") if isinstance(header.get("run_info"), dict) else {}
    scan = os.path.basename(str(header.get("database", "?")))
    return (f"{len(table)} identifications ({n_acc} accepted, {n_uns} unsure), "
            f"{table['spectrum'].nunique() if 'spectrum' in table else '?'} spectra  ·  ions: {ions}  ·  "
            f"scan: {scan}{' (' + run['label'] + ')' if run.get('label') else ''}  ·  "
            f"saved {header.get('saved', '?')}")


def _picker_options(files):
    opts = [{"label": f"{os.path.basename(p)}  ·  {h.get('accepted', '?')} accepted, "
                      f"{h.get('unsure', '0')} unsure  ·  saved {h.get('saved', '?')}",
             "value": p} for p, h in list_confirmed() if p not in files]
    return opts or [{"label": "No more confirmation files in confirmed/", "value": "",
                     "disabled": True}]


@app.callback(
    Output("further-files", "data"),
    Output("further-picker", "style"),
    Output("further-file", "options"),
    Output("further-file", "value"),
    Input("btn-further-add", "n_clicks"),
    Input("further-file", "value"),
    Input({"type": "further-remove", "index": ALL}, "n_clicks"),
    State("further-files", "data"),
    prevent_initial_call=True,
)
def edit_further_files(n_add, chosen, remove_clicks, files):
    """'+ Add File' shows the picker (files already listed are left out);
    choosing a file appends it to the list and hides the picker; a row's ✕
    removes that file.  (One callback, because these outputs feed each other.)"""
    files = list(files or [])
    hidden, nu = {"display": "none"}, dash.no_update
    trig = dash.callback_context.triggered[0] if dash.callback_context.triggered else None
    if trig is None:
        return nu, nu, nu, nu
    pid = trig["prop_id"]
    if pid.startswith("btn-further-add"):
        return nu, {"display": "block"}, _picker_options(files), None
    if pid.startswith("further-file."):
        if chosen and chosen not in files:
            files.append(chosen)
            return files, hidden, [], None
        return nu, nu, nu, nu
    if trig["value"]:                       # a ✕ was clicked (ignore list re-renders)
        idx = json.loads(pid.rsplit(".", 1)[0])["index"]
        if 0 <= idx < len(files):
            files.pop(idx)
            return files, nu, nu, nu
    return nu, nu, nu, nu


@app.callback(
    Output("further-list", "children"),
    Output("further-total", "children"),
    Input("further-files", "data"),
)
def render_further_list(files):
    files = files or []
    if not files:
        return html.Div("No files loaded yet. Click \u201c+ Add file\u201d to load a confirmation file.",
                        style={"fontSize": "13px", "color": "#888"}), ""
    rows = []
    for i, path in enumerate(files):
        try:
            desc = _confirmed_summary(path)
        except Exception as exc:
            desc = f"could not read: {exc}"
        rows.append(html.Div([
            html.Span(f"{i + 1}", style={"fontWeight": "700", "fontSize": "15px", "width": "28px",
                                         "color": ACCENT, "flex": "none"}),
            html.Div([
                html.Div(os.path.basename(path), style={"fontWeight": "600", "fontSize": "13px"}),
                html.Div(desc, style={"fontSize": "12px"}),
            ], style={"flex": "1", "minWidth": "0"}),
            html.Button("✕", id={"type": "further-remove", "index": i}, title="Remove from list",
                        className="btn btn-outline-secondary btn-sm", style={"marginLeft": "10px"}),
        ], className="theia-box",
           style={**BOX_STYLE, "display": "flex", "alignItems": "center", "padding": "6px 10px",
                  "marginBottom": "6px"}))
    try:
        combined = load_further_files(files)
        ions = ", ".join(sorted(combined["ion"].unique(), key=natural_key))
        total = (f"Combined: {len(combined)} identifications from {len(files)} file(s), "
                 f"{combined['spectrum'].nunique()} spectra  ·  ions: {ions}")
    except Exception as exc:
        total = f"Could not combine files: {exc}"
    return rows, total


SUM_ZOOM_HZ = 5000      # half-width of the initial view around the alignment peak


def _ion_harm_key(ion, harmonic):
    return f"{ion}|{int(harmonic)}"


@app.callback(
    Output("sum-ion", "options"),
    Output("sum-ion", "value"),
    Input("further-files", "data"),
    State("sum-ion", "value"),
)
def sum_ion_options(files, current):
    """Ion/harmonic choices for 'Identified peak' alignment (most common first)."""
    try:
        t = load_further_files(files or [])
    except Exception:
        t = pd.DataFrame()
    if not len(t):
        return [], None
    counts = (t.groupby(["ion", "harmonic"])["spectrum"].nunique()
                .reset_index().sort_values("spectrum", ascending=False))
    opts = [{"label": f"{r.ion}  h={int(r.harmonic)}  ({r.spectrum} spectra)",
             "value": _ion_harm_key(r.ion, r.harmonic)} for r in counts.itertuples()]
    values = [o["value"] for o in opts]
    return opts, current if current in values else values[0]


def sum_spectra(files, align, ion_key=None, include_unsure=True,
                window_hz=summing.DEFAULT_WINDOW_HZ):
    """Sum the spectra referenced by the confirmation files.

    align: 'none' | 'strongest' | 'identified'.  Returns (freq, summed power,
    info dict).  core/summing.py: each spectrum is shifted by whole bins (no
    interpolation) so its peak lands on the first spectrum's peak; only the
    bins every shifted spectrum covers are kept.  'strongest' tracks the
    strongest peak from spectrum to spectrum within ±window_hz.
    """
    frames = []
    for path in files:
        header, t = read_confirmed(path)
        scan = header.get("scan_settings") if isinstance(header.get("scan_settings"), dict) else {}
        t["spectra_dir"] = scan.get("spectra_dir", "")
        frames.append(t)
    t = pd.concat(frames, ignore_index=True)
    if not include_unsure and "decision" in t:
        t = t[t["decision"] != "unsure"]
    if not len(t):
        raise ValueError("No identifications left to sum.")

    # (dir, spectrum) -> peak_freq of the chosen ion/harmonic; used to align
    # ('identified') and to place the initial zoom (both aligned modes)
    ion_pos = {}
    if align != "none" and ion_key:
        ion, harm = ion_key.rsplit("|", 1)
        sel = t[(t["ion"] == ion) & (t["harmonic"] == int(harm))].copy()
        if len(sel):
            # several matches of that ion/harmonic in one spectrum: prefer the
            # one flagged closest_for_ion, otherwise the smallest |delta|
            closest = sel.get("closest_for_ion", pd.Series(False, index=sel.index))
            sel["_rank"] = (~closest.astype(str).str.lower().isin(["true", "1"])).astype(int)
            sel["_ad"] = sel["delta_hz"].abs()
            sel = sel.sort_values(["_rank", "_ad"]).drop_duplicates(["spectra_dir", "spectrum"])
            ion_pos = dict(zip(zip(sel["spectra_dir"], sel["spectrum"]),
                               sel["peak_freq"].astype(float)))

    peak_pos = {}                         # (dir, spectrum) -> frequency to align on
    if align == "identified":
        if not ion_key:
            raise ValueError("Choose the ion / harmonic to align on.")
        if not ion_pos:
            raise ValueError(f"{ion} h={harm} is not in the loaded files.")
        keys = list(ion_pos)
        peak_pos = dict(ion_pos)
    else:
        keys = list(dict.fromkeys(zip(t["spectra_dir"], t["spectrum"])))
    keys.sort(key=lambda k: (k[0], natural_key(k[1])))
    n_requested = len(t.drop_duplicates(["spectra_dir", "spectrum"]))

    spectra, failed = [], []
    for d, name in keys:
        try:
            ff, pp = load_spectrum(d, name)
        except Exception as exc:
            failed.append(f"{name}: {exc}")
            continue
        spectra.append(((d, name), ff, pp))
    if not spectra:
        raise ValueError("No spectra could be loaded" + (f" ({failed[0]})" if failed else "."))

    data = [(ff, pp) for _, ff, pp in spectra]
    if align == "strongest":
        # tracked from spectrum to spectrum (in file order), not the strongest
        # bin of each whole spectrum
        positions, lost = summing.track_strongest(data, window_hz)
        left_out = [f"{spectra[i][0][1]}: {why}" for i, why in lost]
    elif align == "identified":
        positions = [peak_pos[k] for k, _, _ in spectra]
        left_out = []
    else:
        positions, left_out = None, []
    freq, total, s_info = summing.sum_whole_bins(data, positions)
    used = [spectra[i] for i in s_info["used"]]
    shifts = s_info["shift_hz"]
    target = s_info["target"]
    # initial zoom: where the chosen ion ends up after shifting (median over
    # the summed spectra that contain it); falls back to the alignment target
    zoom, zoom_on_ion = target, False
    if target is not None:
        moved = [ion_pos[k] + s for (k, _, _), s in zip(used, shifts) if k in ion_pos]
        if moved:
            zoom, zoom_on_ion = float(np.median(moved)), True
    info = dict(n=len(used), n_requested=n_requested, failed=failed, target=target, zoom=zoom,
                zoom_on_ion=zoom_on_ion, shifts=shifts, shift_bins=s_info["shift_bins"],
                residual_hz=s_info["residual_hz"], bin_hz=s_info["bin_hz"], left_out=left_out,
                names=[k[1] for k, _, _ in used], n_dirs=len({k[0] for k, _, _ in used}))
    return freq, total, info


@app.callback(Output("sum-ion", "style"), Input("sum-align", "value"))
def toggle_sum_ion(align):
    return {"fontSize": "13px", "width": "240px", "marginRight": "14px",
            "display": "none" if align == "none" else "block"}


@app.callback(Output("sum-window-box", "style"), Input("sum-align", "value"))
def toggle_sum_window(align):
    return {"fontSize": "13px", "marginRight": "14px",
            "display": "inline-flex" if align == "strongest" else "none",
            "alignItems": "center"}


@app.callback(
    Output("further-sum-src", "style"),
    Output("further-file-src", "style"),
    Output("sum-only-controls", "style"),
    Output("btn-fsrc-load", "style"),
    Input("further-source", "value"),
)
def toggle_further_source(source):
    """Show the controls of the chosen data source (confirmation files / one file)."""
    hidden = {"display": "none"}
    load = {"backgroundColor": ACCENT, "color": "white", "marginRight": "14px"}
    if source == "file":
        return hidden, FILE_SRC_SHOWN, hidden, load
    return {}, hidden, SUM_CONTROLS_SHOWN, {**load, "display": "none"}


@app.callback(
    Output("fsrc-dir", "value"),
    Output("fsrc-name", "options"),
    Output("fsrc-name", "value"),
    Output("fsrc-load-req", "data"),
    Input("fsrc-dir", "value"),
    Input("btn-fsrc-viewer", "n_clicks"),
    State("fsrc-name", "value"),
    State("v-file_path", "value"),
    State("v-spectrum_name", "value"),
)
def fsrc_options(directory, _viewer, current, v_dir, v_name):
    """Single-file source: the .npz files in the folder.  'Use Viewer Spectrum'
    takes the Viewer's folder and file and loads it straight away."""
    trig = dash.callback_context.triggered[0]["prop_id"] if dash.callback_context.triggered else ""
    nu = dash.no_update
    req = nu
    if trig.startswith("btn-fsrc-viewer"):
        if not v_dir or not v_name:
            return nu, nu, nu, nu
        directory, current = v_dir, v_name
        req = {"t": time.time()}
    d = paths.remap_legacy(os.path.expanduser((directory or "").strip()))
    names = list_npz(d)
    opts = [{"label": n, "value": n} for n in names]
    value = current if current in names else (names[0] if names else None)
    return (directory if req is not nu else nu), opts, value, req


def load_single_spectrum(directory, name):
    """(freq, power) of one .npz spectrum file, sorted by frequency."""
    if not directory or not name:
        raise ValueError("Choose a folder and a spectrum file first.")
    ff, pp = load_spectrum(os.path.expanduser(directory.strip()), name)
    ff, pp = np.asarray(ff, dtype=float), np.asarray(pp, dtype=float)
    if len(ff) != len(pp) or not len(ff):
        raise ValueError(f"{name}: frequency and power arrays do not match")
    if np.any(np.diff(ff) < 0):
        order = np.argsort(ff)
        ff, pp = ff[order], pp[order]
    return ff, pp


_SUM_ANNOTATIONS = {}   # sum key -> the figure's own annotations (energy labels go after)
_SUM_SHAPES = {}        # ... and its own shapes (the energy brackets go after)
_SUM_SIMS = {}      # sum key -> simulated lines (ion, harmonic, f_sim), for name suggestions


# ---------------------------------------------------------------------------
# PID settings: the settings used for the simulated lines on the summed plot
# ---------------------------------------------------------------------------
# By default they are the scan settings stored in the first loaded
# confirmation file; anything changed in the "PID Settings" popup overrides
# them (saved with the tab).  The reference position is always the strongest
# peak of the summed spectrum.
SIM_TRACE_NAMES = ["Simulated (no peak)", "Simulated (identified)", EXCLUDED_TRACE]
SIM_TRACE_STYLES = [dict(color="#bbbbbb", width=1, dash="dash"),
                    dict(color="#555555", width=1.5, dash="dash"),
                    dict(color="#c8c8c8", width=1, dash="dot")]


def file_scan_settings(files):
    """Scan settings stored in the header of the first loaded file (raw dict)."""
    if not files:
        raise ValueError("no confirmation files loaded")
    header, _ = read_confirmed(files[0])
    st = header.get("scan_settings")
    if not isinstance(st, dict):
        raise ValueError("no scan settings in the file header")
    return st


def viewer_pid_settings():
    """Raw settings for the simulated lines of a single spectrum file: the
    scanner settings for anything the Viewer has not got, then the Viewer's
    settings (viewer_config.json + shared physics); no half-life cut, as in
    the Viewer."""
    raw = dict(load_config())
    v = load_viewer_config()
    raw.update({k: v[k] for k in VIEWER_KEYS if k in raw})
    raw["min_half_life"], raw["exclude_unknown_hl"] = 0, []
    return raw


def pid_base(source, files):
    """Raw settings the PID popup starts from, for the data source."""
    return viewer_pid_settings() if source == "file" else file_scan_settings(files)


def pid_overrides(store, source):
    """The PID edits stored for one data source.  The store holds
    {"sum": ..., "file": ...}; an older store (one flat dict) is the sum's."""
    if not store:
        return None
    if "sum" in store or "file" in store:
        return store.get(source)
    return store if source == "sum" else None


def pid_settings(files, overrides, source="sum"):
    """Parsed settings for the simulated lines: base settings + overrides.  If
    the correction is edited, the per-harmonic values stored with the base
    settings are dropped (they come from the profile chosen)."""
    raw = {**pid_base(source, files), **(overrides or {})}
    if overrides and any(k in overrides for k in ("corr_profile", "corr_a", "corr_b", "corr_c")):
        raw.pop("corr_by_h", None)
    return parse_settings(raw)


def _pid_display(raw, key):
    """A settings value as shown in the popup's fields."""
    v = raw.get(key)
    if key == "corr_profile":
        return corrections.resolve(v, raw.get("corr_a"), raw.get("corr_b"), raw.get("corr_c"))["name"]
    if key in ("correction_enabled", "exclude_unknown_hl"):
        return ["on"] if (v and v != []) else []
    if key == "min_half_life_unit":
        return v if v in HL_UNITS else "ms"
    if isinstance(v, (list, tuple)):
        return ", ".join(str(x) for x in v)
    if v is None:
        return "0" if key == "min_half_life" else ""
    return str(v)


def _hl_text(t, u):
    """Half-life as written in NUBASE ('3.2 m'); '' if unknown."""
    t = "" if t is None or (isinstance(t, float) and np.isnan(t)) else str(t).strip()
    u = "" if u is None or (isinstance(u, float) and np.isnan(u)) else str(u).strip()
    return (t + (" " + u if u else "")).strip()


def sum_sim_traces(ff, pp, files, include_unsure, overrides, key=None, source="sum"):
    """(x, y, hovertext) for the three simulated-line traces of the summed plot
    (no peak / identified / half-life excluded) and a one-line description.
    Same calculation as the scanner; the strongest peak is the reference.
    source 'file': a single spectrum, with the Viewer settings and no
    'identified' ions."""
    s = pid_settings(files, overrides, source)
    k = int(np.nanargmax(pp))
    ref_f = float(ff[k])
    ion_db, ref_mq = build_ion_database(*s["ion_key"])
    sims, _ref_h, ref_warn = physics.simulate_spectrum(ion_db, ref_mq, ref_f, s, ff, pp)
    sims = sims[(sims["f_sim"] >= ff.min()) & (sims["f_sim"] <= ff.max())].reset_index(drop=True)
    if key is not None:
        rows = sims["ion_row"].to_numpy()
        names = ion_db["ion"].to_numpy()[rows]
        _SUM_SIMS[key] = pd.DataFrame({"ion": names,
                                       "harmonic": sims["harmonic"].to_numpy(),
                                       "f_sim": sims["f_sim"].to_numpy(),
                                       # for Search ions
                                       "t_half": [_hl_text(t, u) for t, u in zip(
                                           ion_db["T"].to_numpy()[rows],
                                           ion_db["T_unit"].to_numpy()[rows])],
                                       "excluded": ~hl_allowed(ion_db, s)[rows],
                                       "is_ref": [(str(n) == s["ref_name"]
                                                   and int(h) == int(s["ref_harmonic"]))
                                                  for n, h in zip(names, sims["harmonic"])]})
        for k_old in list(_SUM_SIMS)[:-4]:
            _SUM_SIMS.pop(k_old, None)
    t = (pd.concat([read_confirmed(f)[1] for f in files], ignore_index=True)
         if source == "sum" and files else pd.DataFrame())
    if not include_unsure and "decision" in t:
        t = t[t["decision"] != "unsure"]
    listed = t[["ion", "harmonic"]].drop_duplicates() if len(t) else \
        pd.DataFrame(columns=["ion", "harmonic"])
    finite = pp[np.isfinite(pp)]
    ylo, yhi = float(finite.min()), float(finite.max())
    pad = 0.05 * (yhi - ylo or 1)
    tmp = go.Figure()
    add_sim_lines(tmp, sims, ion_db, listed, ylo - pad, yhi + pad,
                  excluded=~hl_allowed(ion_db, s))
    by_name = {tr.name: tr for tr in tmp.data}
    out = []
    for name in SIM_TRACE_NAMES:
        tr = by_name.get(name)
        out.append(([] if tr is None else list(tr.x), [] if tr is None else list(tr.y),
                    [] if tr is None else list(tr.hovertext)))
    cut = hl_cut_text(s)
    if overrides:
        src = "PID settings (edited)"
    elif source == "file":
        src = "Viewer settings"
    else:
        src = "settings of file 1" + (f" of {len(files)}" if len(files) > 1 else "")
    text = (f"Simulated lines: reference {s['ref_name']} (h={s['ref_harmonic']}) taken as the "
            f"strongest peak at {ref_f:.1f} Hz · harmonics {', '.join(map(str, s['harmonics']))}"
            f" · correction {'on' if s['correction_enabled'] else 'off'}"
            + (f" ({s['corr_profile']})" if s["correction_enabled"]
               and s.get("corr_profile") not in (None, "", corrections.CUSTOM) else "")
            + "".join(f" · ⚠ {w}" for w in ref_warn)
            + (f" · half-life cut: {cut}" if cut else "") + f" · from {src}"
            + (" · dark dashed: ions in the loaded files" if source == "sum" else ""))
    return out, text


# Summed spectra are kept on the server (keyed by sum-key) so the peak-fit
# callbacks can use the data without sending it back and forth.
_SUM_CACHE = {}
_SUM_CACHE_LOCK = threading.Lock()
# fit saved last session, waiting for its restored sum (sum key -> fit state)
_RESTORE_FIT = {}
_RESTORE_ENERGY = {}    # same for the energy calculation (state + peak names)
# sum key -> what the plot shows: {"source": "sum"} or
# {"source": "file", "dir": ..., "name": ...} (saved with the tab)
_SUM_META = {}


def _cache_sum(ff, pp):
    key = f"{time.time():.6f}"
    d = np.diff(pp[np.isfinite(pp)])
    # noise sigma per bin from the bin-to-bin differences (robust to peaks)
    sigma = float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)) if len(d) else 1.0
    with _SUM_CACHE_LOCK:
        _SUM_CACHE[key] = (ff, pp, sigma or 1.0)
        for k in list(_SUM_CACHE)[:-4]:          # keep the last few sums only
            _SUM_CACHE.pop(k, None)
            _SUM_META.pop(k, None)
    return key


# Fit-overlay traces appended after the spectrum (trace 0) in the summed plot;
# they are updated in place (Patch) so the view is not reset.
FIT_TRACE_STYLES = [
    dict(name="fit", mode="lines", line=dict(color=ACCENT, width=2.5)),
    dict(name="fit components", mode="lines", line=dict(color="#555555", width=1.2, dash="dash")),
    dict(name="fit background", mode="lines", line=dict(color="#999999", width=1.2, dash="dot")),
    dict(name="fit peaks", mode="markers+text", textposition="top center",
         textfont=dict(color=ACCENT, size=15, family="Arial Black, Arial, sans-serif"),
         marker=dict(symbol="x-thin", size=12, color=ACCENT, line=dict(width=2.5, color=ACCENT))),
    # FWHM handles: one at each half-height point of every peak (draggable)
    dict(name="fit width handles", mode="markers",
         marker=dict(symbol="diamond", size=8, color="white", line=dict(width=1.5, color=ACCENT))),
]
FIT_PEAKS_INDEX = 4          # trace index of "fit peaks" (trace 0 is the spectrum)
SHAPE_CODES = {"gauss": 0, "lorentz": 1, "pvoigt": 2}     # for the drag preview in the browser


def _fit_into_patch(patch, fit_state, cached):
    """Fit-overlay traces (total, components, background, peak markers)."""
    # imported here: .further_fit imports this module
    from .further_fit import fit_overlay, _shapes
    for i, (x, y) in enumerate(fit_overlay(cached[0], fit_state), start=1):
        patch["data"][i]["x"] = x
        patch["data"][i]["y"] = y
    # peak numbers (as in the table) next to the markers
    n = len((fit_state or {}).get("peaks") or [])
    patch["data"][FIT_PEAKS_INDEX]["text"] = [str(k) for k in range(1, n + 1)]
    # peak parameters for the drag preview: centre, amp, fwhm, eta, shape code
    peaks = (fit_state or {}).get("peaks") or []
    shapes = _shapes(fit_state) if peaks else []
    patch["data"][FIT_PEAKS_INDEX]["customdata"] = [
        [pk["centre"], pk["amp"], pk["fwhm"], pk.get("eta", 0.5), SHAPE_CODES.get(sh, 0)]
        for pk, sh in zip(peaks, shapes)]


@app.callback(
    Output("sum-graph", "figure"),
    Output("sum-status", "children"),
    Output("sum-key", "data"),
    Output("sum-sim-info", "children"),
    Input("btn-sum", "n_clicks"),
    Input("btn-fsrc-load", "n_clicks"),
    Input("fsrc-load-req", "data"),
    Input("further-restore", "data"),
    State("further-files", "data"),
    State("sum-align", "value"),
    State("sum-ion", "value"),
    State("sum-unsure", "value"),
    State("sum-show-sims", "value"),
    State("pid-settings", "data"),
    State("fsrc-dir", "value"),
    State("fsrc-name", "value"),
    State("sum-window", "value"),
)
def show_sum(_, _load, _req, restore, files, align, ion_key, unsure, show_sims, pid_store,
             f_dir, f_name, window):
    """'Sum Spectra' button, 'Load Spectrum' / 'Use Viewer Spectrum' (single
    spectrum file), or (on page load) rebuild the plot saved last time with
    its zoom; the saved fit is then picked up by update_fit."""
    # imported here: .energy imports this module
    from .energy import ENERGY_PAIR_TRACE, ENERGY_TRACE
    trig = dash.callback_context.triggered[0]["prop_id"] if dash.callback_context.triggered else ""
    nu = dash.no_update
    # anything but the buttons (e.g. the initial call on page load) = restore
    restoring = not (trig.startswith("btn-sum") or trig.startswith("btn-fsrc-load")
                     or trig.startswith("fsrc-load-req"))
    if trig.startswith("btn-sum"):
        source = "sum"
    elif not restoring:
        if not (_load if trig.startswith("btn-fsrc-load") else _req):
            return nu, nu, nu, nu
        source = "file"
    else:
        if not restore or not restore.get("summed"):
            return nu, nu, nu, nu
        plot = restore.get("plot") or {"source": "sum"}
        source = plot.get("source") or "sum"
        if source == "file":
            f_dir, f_name = plot.get("dir"), plot.get("name")
        else:
            files, align = restore.get("files") or files, restore.get("align") or align
            ion_key = restore.get("ion") or ion_key
            unsure = restore.get("unsure") if restore.get("unsure") is not None else unsure
            window = restore.get("window") or window
    pid_over = pid_overrides(pid_store, source)

    if source == "file":
        try:
            ff, pp = load_single_spectrum(f_dir, f_name)
        except Exception as exc:
            return (empty_fig(str(exc)), html.Span(f"Error: {exc}", style={"color": "#c0392b"}),
                    None, "")
        info = dict(n=1, n_requested=1, failed=[], target=None, zoom=None, zoom_on_ion=False,
                    shifts=np.array([]), names=[f_name], n_dirs=1)
        how, title = "", f_name
        meta = {"source": "file", "dir": f_dir, "name": f_name}
    else:
        if not files:
            return (empty_fig("No confirmation files loaded"), "Load at least one file first.",
                    None, "")
        try:
            try:
                window = float(window)
            except (TypeError, ValueError):
                window = 0.0
            if align == "strongest" and not window > 0:
                raise ValueError("the peak tracking window (± Hz) must be above 0")
            ff, pp, info = sum_spectra(files, align, ion_key, bool(unsure),
                                       window if window > 0 else summing.DEFAULT_WINDOW_HZ)
        except Exception as exc:
            return (empty_fig(str(exc)), html.Span(f"Error: {exc}", style={"color": "#c0392b"}),
                    None, "")
        if align == "identified":
            ion, harm = ion_key.rsplit("|", 1)
            how = f"aligned on {ion} h={harm}"
        elif align == "strongest":
            how = f"aligned on the strongest peak, tracked ±{window:g} Hz"
        else:
            how = "not aligned"
        title = f"Sum of {info['n']} Spectra ({how})"
        meta = {"source": "sum"}
    key = _cache_sum(ff, pp)
    _SUM_META[key] = meta
    if restoring and restore.get("fit"):
        _RESTORE_FIT[key] = restore["fit"]
    if restoring and restore.get("energy"):
        _RESTORE_ENERGY[key] = restore["energy"]

    fig = go.Figure()
    # name "Spectrum" + no Plotly hover, as in the other spectrum plots: the
    # bottom-right readout shows f and P instead.  Plotly hover on all ~131k
    # points made every mouse move push hoverData through Dash, which is slow.
    fig.add_trace(go.Scattergl(x=ff, y=pp, mode="lines", name="Spectrum", hoverinfo="skip",
                               line=dict(shape="hvh", color="black", width=1)))
    for st in FIT_TRACE_STYLES:                      # empty until peaks are added
        fig.add_trace(go.Scatter(x=[], y=[], hoverinfo="skip", **st))
    fig.add_trace(go.Scatter(x=[], y=[], hoverinfo="skip", name=ENERGY_TRACE,
                             mode="markers+text", textposition="top center",
                             textfont=dict(color="#2e7d32", size=13),
                             marker=dict(symbol="triangle-down", size=11, color="#2e7d32")))
    # bracket over each calculated pair, labelled with its energy difference
    fig.add_trace(go.Scatter(x=[], y=[], hoverinfo="skip", name=ENERGY_PAIR_TRACE,
                             mode="lines", line=dict(color="black", width=1.5)))
    # simulated ion lines: three fixed traces (no peak / identified / half-life
    # excluded) after the fit and energy traces, so "PID Settings" can redraw
    # them in place
    try:
        sim_traces, sim_text = sum_sim_traces(ff, pp, files, bool(unsure), pid_over, key=key,
                                              source=source)
    except Exception as exc:
        traceback.print_exc()
        sim_traces, sim_text = [([], [], [])] * 3, f"simulated lines not drawn: {exc}"
    for name, style, (x, y, h) in zip(SIM_TRACE_NAMES, SIM_TRACE_STYLES, sim_traces):
        fig.add_trace(go.Scattergl(x=x, y=y, mode="lines", name=name, line=style, meta="sim",
                                   hoverinfo="text", hovertext=h, connectgaps=False,
                                   visible=bool(show_sims)))
    if info["target"] is not None:
        fig.add_vline(x=info["target"], line=dict(color=ACCENT, width=1.5, dash="dash"),
                      annotation_text=f"aligned at {info['target']:.1f} Hz",
                      annotation_font_color=ACCENT)
    if info["zoom"] is not None:
        # initial view: zoom on the chosen ion from the list
        lo, hi = info["zoom"] - SUM_ZOOM_HZ, info["zoom"] + SUM_ZOOM_HZ
        fig.update_xaxes(range=[lo, hi], autorange=False)
        m = (ff >= lo) & (ff <= hi)
        if m.any():
            ylo, yhi = float(np.nanmin(pp[m])), float(np.nanmax(pp[m]))
            fig.update_yaxes(range=[ylo - 0.05 * (yhi - ylo or 1), yhi + 0.12 * (yhi - ylo or 1)])
    view = restore.get("view") if restoring else None
    if view and view.get("x") and view.get("y"):          # zoom saved last time
        fig.update_xaxes(range=view["x"], autorange=False)
        fig.update_yaxes(range=view["y"], autorange=False)
    fig.update_layout(title=title, xaxis_title="Frequency (Hz)",
                      yaxis_title=("Schottky Power (a.u.)" if source == "file"
                                   else "Summed Schottky Power (a.u.)"),
                      # hover only reaches the simulated lines (the spectrum
                      # and fit traces skip hover), as in the other plots
                      hovermode="x",
                      dragmode="pan", template="plotly_white", font=dict(size=14),
                      margin=dict(l=60, r=20, t=50, b=40), showlegend=False,
                      # keeps the user's zoom while the fit traces are patched
                      uirevision=key)
    # annotations of the plot itself; the energy labels are added after these
    _SUM_ANNOTATIONS[key] = [a.to_plotly_json() for a in fig.layout.annotations]
    _SUM_SHAPES[key] = [sh.to_plotly_json() for sh in fig.layout.shapes]

    if source == "file":
        status = [html.Span(f"Single spectrum {f_name}  ·  {len(ff)} bins, "
                            f"{ff.min():.1f} – {ff.max():.1f} Hz  ·  from {f_dir}")]
        return fig, status, key, sim_text
    sh = info["shifts"]
    parts = [f"Summed {info['n']} spectra ({how})"]
    if align == "identified":
        parts.append(f"{info['n_requested'] - info['n'] - len(info['failed'])} spectra without "
                     f"that ion/harmonic left out")
    if info["target"] is not None and len(sh):
        sb = info["shift_bins"]
        parts.append(f"whole-bin shifts {sb.min():+d} to {sb.max():+d} bins "
                     f"({sh.min():+.1f} to {sh.max():+.1f} Hz; left over at most "
                     f"{np.max(np.abs(info['residual_hz'])):.1f} Hz)")
    if align == "strongest" and ion_key:
        zi, zh = ion_key.rsplit("|", 1)
        parts.append(f"view centred on {zi} h={zh}" if info["zoom_on_ion"] else
                     f"{zi} h={zh} not in the summed spectra, view centred on the strongest peak")
    if info["n_dirs"] > 1:
        parts.append(f"from {info['n_dirs']} spectra directories")
    status = [html.Span("  ·  ".join(parts))]
    if info.get("left_out"):
        lo = info["left_out"]
        status.append(html.Div(f"Left out {len(lo)} (tracked peak not found; a larger ± window "
                               f"may help): " + "; ".join(lo[:3]) + (" …" if len(lo) > 3 else ""),
                               style={"color": "#c0392b"}))
    if info["failed"]:
        status.append(html.Div(f"Could not load {len(info['failed'])}: "
                               + "; ".join(info["failed"][:3])
                               + (" …" if len(info["failed"]) > 3 else ""),
                               style={"color": "#c0392b"}))
    return fig, status, key, sim_text
