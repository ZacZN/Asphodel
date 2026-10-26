"""
theia.py  –  Theia: spectrum viewer and batch scanner

Tabs: Viewer (one spectrum with the simulated ion frequencies overlaid; was
viewer.py), Scan, Scan Results and Further output.

The scanner loops over every 1-D Schottky spectrum (.npz, same format as viewer.py) in a
directory, detects peaks in the raw data, tracks the (drifting) reference peak
from spectrum to spectrum, simulates the NUBASE ion frequencies from that
reference and identifies every detected peak lying within ±x Hz of a simulated
ion.  All results go into a new SQLite database per run, which can be loaded
and searched in the "Scan Results" tab.

Usage
-----
    python theia/theia.py      (or start it from the Asphodel home page)
then open  http://127.0.0.1:8052/

Output  (in the active experiment's data folder, see core/paths.py)
------
    data/<experiment>/outputs/scans/
      index.csv                               one row per run (human readable)
      scan_<YYYYmmdd_HHMMSS>_<hash>.sqlite    one database per run
      scan_<YYYYmmdd_HHMMSS>_<hash>.json      settings used for that run
    data/<experiment>/outputs/confirmed/      confirmed identifications (*.csv)
    data/<experiment>/settings/               scanner_config.json, viewer_config.json

<hash> is a hash of the scan settings, so runs with identical settings share
the same hash.

Peak detection (no smoothing is ever applied to the data)
---------------------------------------------------------
1.  Noise: bins above -min(spectrum) * X (`noise_cut_factor`, 0 = off) are
    treated as signal and excluded.  The rest of the spectrum is split into
    blocks of `noise_block_hz`; the median (local baseline b) and the MAD-based
    sigma (σ_bg) of each block are interpolated across the spectrum.  Signal-dependent noise on top of large
    peaks is modelled as  σ = sqrt(σ_bg² + (r · (p − b))²), where r is the
    "relative noise" parameter.
2.  Maxima: every local maximum in the raw data is a candidate.  It is kept if
    prominence / σ ≥ k_prom  and  (p − b) / σ ≥ k_height.  Because the test is
    in units of the local noise, the same thresholds work for single-ion peaks
    and for peaks many orders of magnitude larger.
3.  Shoulders (optional): merged structures where a second peak produces no
    local maximum.  The raw second difference taken at a finite scale s,
    −(p[i−s] − 2p[i] + p[i+s]), is searched for local maxima exceeding
    k_shoulder · √6 · σ.  Candidates within ±s of an accepted maximum are
    discarded (they are that maximum's own curvature).  s is a finite-difference
    step, not a smoothing kernel: the reported position is a raw data bin.
"""

import os
import sys
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import paths, physics, tabwatch
from core.nubase_reader import (df_nubase, rect_restrict, diag_restrict, index_by_name, unpack_npz,
                                get_base_dataframe)
from core.physics import (delta_f, apply_correction, ion_base_name, build_hover_label, HL_UNITS,
                          HL_CUT_UNITS, half_life_seconds, hl_allowed, hl_cut_text,
                          build_ion_database, simulate_frequencies)

import re
import csv
import glob
import json
import time
import hashlib
import sqlite3
import datetime
import threading
import logging
import traceback
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths

import plotly.graph_objects as go

import dash
from dash.exceptions import PreventUpdate
from dash import dcc, html, Input, Output, State, ALL, dash_table
import dash_bootstrap_components as dbc


SCANS_DIR   = paths.outputs("scans")
INDEX_PATH  = os.path.join(SCANS_DIR, "index.csv")
CONFIG_PATH = paths.settings_file("scanner_config.json")
CONFIRMED_DIR = paths.outputs("confirmed")
VIEWER_CONFIG_PATH = paths.settings_file("viewer_config.json")
ACCENT      = "#f57d15"
ACCENT_PALE = "#fff4ec"


# ===========================================================================
# Physics (core/physics.py)
# ===========================================================================

# delta_f, apply_correction, the half-life cut, build_ion_database and
# simulate_frequencies live in core/physics.py (shared with plot_spectrogram.py).


# ===========================================================================
# Peak detection (raw data, no smoothing)
# ===========================================================================

def hz_to_bins(hz, bin_hz):
    if hz is None or hz <= 0 or bin_hz <= 0:
        return 0
    return max(1, int(round(hz / bin_hz)))


def noise_cutoff(pp, factor):
    """Upper cutoff for the noise estimate: -min(pp) * factor.

    In a baseline-subtracted spectrum the most negative value is set by the
    noise alone, so bins above (-min * factor) are taken to be signal and are
    excluded from the noise estimate.  Returns None (no cut) if factor <= 0 or
    the spectrum has no negative values.
    """
    pmin = float(np.min(pp))
    if factor is None or factor <= 0 or pmin >= 0:
        return None
    return -pmin * factor


def estimate_noise(pp, block_bins, cutoff=None):
    """Blockwise median (baseline) and MAD sigma, interpolated per bin.

    Bins above `cutoff` (peaks) are excluded from the statistics.  A block with
    too few bins left after the cut (i.e. lying inside a peak) is skipped and
    its values are interpolated from the neighbouring blocks.
    """
    n = len(pp)
    MIN_BINS = 10

    def _stats(seg):
        if cutoff is not None:
            seg = seg[seg <= cutoff]
        if len(seg) < MIN_BINS:
            return None
        med = np.median(seg)
        sig = 1.4826 * np.median(np.abs(seg - med))
        return med, sig

    g = _stats(pp)
    g_med, g_sig = g if g is not None else (0.0, 0.0)
    if g_sig <= 0:
        g_sig = np.std(pp) or 1.0
    if block_bins <= 0 or block_bins * 2 > n:
        return np.full(n, g_med), np.full(n, g_sig)

    nblocks = n // block_bins
    edges = np.linspace(0, n, nblocks + 1).astype(int)
    centres, meds, sigs = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        st = _stats(pp[a:b])
        if st is None:
            continue
        m, s = st
        centres.append(0.5 * (a + b - 1))
        meds.append(m)
        sigs.append(s if s > 0 else g_sig)
    if not centres:
        return np.full(n, g_med), np.full(n, g_sig)
    x = np.arange(n)
    return np.interp(x, centres, meds), np.interp(x, centres, sigs)


def detect_peaks(ff, pp, p):
    """Detect peaks.  `p` is the parsed settings dict.

    Returns (peaks DataFrame, baseline array, sigma array).
    Columns: idx, freq, power, prominence, snr, height_snr, width_hz, kind
    """
    ff = np.asarray(ff, dtype=float)
    pp = np.asarray(pp, dtype=float)
    n = len(pp)
    cols = ["idx", "freq", "power", "prominence", "snr", "height_snr", "width_hz", "kind"]
    if n < 5:
        return pd.DataFrame(columns=cols), np.zeros(n), np.ones(n)

    bin_hz = (ff[-1] - ff[0]) / (n - 1)
    cutoff = noise_cutoff(pp, p["noise_cut_factor"])
    base, sig_bg = estimate_noise(pp, hz_to_bins(p["noise_block_hz"], bin_hz), cutoff)
    r = p["rel_noise"]

    def sig_at(level, sbg):
        """Noise sigma at a given signal level above the baseline."""
        return np.sqrt(sbg**2 + (r * np.clip(level, 0, None)) ** 2)

    # --- local maxima -----------------------------------------------------
    idx, props = find_peaks(pp, prominence=0)
    prom = props["prominences"]
    sig_bg_i = sig_bg[idx]
    # Prominence is compared with the noise at the level of the saddle (the
    # higher of the two bases): a noise bump sitting on top of a large peak
    # has a high saddle and therefore a large sigma, an isolated peak does not.
    saddle = pp[idx] - prom - base[idx]
    prom_snr = prom / sig_at(saddle, sig_bg_i)
    # Height above the local baseline is always tested against the background noise
    h_snr = (pp[idx] - base[idx]) / sig_bg_i
    keep = (prom_snr >= p["k_prom"]) & (h_snr >= p["k_height"])
    idx, prom, prom_snr, h_snr = idx[keep], prom[keep], prom_snr[keep], h_snr[keep]
    lb, rb = props["left_bases"][keep], props["right_bases"][keep]

    if len(idx):
        widths = peak_widths(pp, idx, rel_height=0.5, prominence_data=(prom, lb, rb))[0] * bin_hz
    else:
        widths = np.array([])

    if p["min_width_bins"] > 0 and len(idx):
        k = widths / bin_hz >= p["min_width_bins"]
        idx, prom, prom_snr, h_snr, widths = idx[k], prom[k], prom_snr[k], h_snr[k], widths[k]

    dist = hz_to_bins(p["min_distance_hz"], bin_hz)
    if dist > 1 and len(idx) > 1:
        order = np.argsort(-prom)
        taken = np.zeros(n, dtype=bool)
        k = np.zeros(len(idx), dtype=bool)
        for j in order:
            i = idx[j]
            lo, hi = max(0, i - dist + 1), min(n, i + dist)
            if not taken[lo:hi].any():
                k[j] = True
                taken[i] = True
        idx, prom, prom_snr, h_snr, widths = idx[k], prom[k], prom_snr[k], h_snr[k], widths[k]

    maxima = pd.DataFrame({
        "idx": idx, "freq": ff[idx], "power": pp[idx], "prominence": prom,
        "snr": prom_snr, "height_snr": h_snr, "width_hz": widths, "kind": "max",
    })

    # --- shoulders --------------------------------------------------------
    frames = [maxima]
    s = hz_to_bins(p["shoulder_scale_hz"], bin_hz)
    if p["shoulders"] and s >= 1 and n > 2 * s + 2:
        c = np.zeros(n)
        c[s:-s] = -(pp[:-2 * s] - 2 * pp[s:-s] + pp[2 * s:])
        cidx, _ = find_peaks(c, distance=max(1, s))
        sig_bg_i = sig_bg[cidx]
        c_snr = c[cidx] / (np.sqrt(6.0) * sig_at(pp[cidx] - base[cidx], sig_bg_i))
        ch_snr = (pp[cidx] - base[cidx]) / sig_bg_i
        k = (c_snr >= p["k_shoulder"]) & (ch_snr >= p["k_height"])
        cidx, c_snr, ch_snr = cidx[k], c_snr[k], ch_snr[k]
        if len(cidx) and len(idx):
            ms = np.sort(idx)
            pos = np.searchsorted(ms, cidx)
            d_left = np.abs(cidx - ms[np.clip(pos - 1, 0, len(ms) - 1)])
            d_right = np.abs(ms[np.clip(pos, 0, len(ms) - 1)] - cidx)
            k = np.minimum(d_left, d_right) > s
            cidx, c_snr, ch_snr = cidx[k], c_snr[k], ch_snr[k]
        if len(cidx):
            frames.append(pd.DataFrame({
                "idx": cidx, "freq": ff[cidx], "power": pp[cidx], "prominence": np.nan,
                "snr": c_snr, "height_snr": ch_snr, "width_hz": np.nan, "kind": "shoulder",
            }))

    peaks = pd.concat(frames, ignore_index=True) if len(frames) > 1 else maxima
    peaks = peaks.sort_values("freq").reset_index(drop=True)
    return peaks, base, sig_bg


# ===========================================================================
# Reference tracking and identification
# ===========================================================================

def find_reference(peaks, predicted, window_hz, mode="strongest", min_height=0.0):
    """Pick the reference peak among detected maxima within ±window of the
    predicted position that are at least `min_height` tall (raw power).
    Returns the row (Series) or None."""
    if peaks is None or len(peaks) == 0:
        return None
    cand = peaks[(peaks["kind"] == "max") & ((peaks["freq"] - predicted).abs() <= window_hz)
                 & (peaks["power"] >= min_height)]
    if len(cand) == 0:
        return None
    if mode == "nearest":
        return cand.loc[(cand["freq"] - predicted).abs().idxmin()]
    return cand.loc[cand["power"].idxmax()]


def match_peaks(peaks, sims, ion_db, tol_hz):
    """All (peak, simulated ion) pairs with |f_peak − f_sim| ≤ tol.

    Adds flags closest_for_ion (the nearest peak for that ion/harmonic) and
    closest_for_peak (the nearest ion for that peak).
    """
    cols = ["peak_row", "ion", "ion_base", "Z", "N", "A", "charge", "harmonic",
            "sim_freq", "peak_freq", "delta_hz", "power", "snr", "kind",
            "closest_for_ion", "closest_for_peak"]
    if len(peaks) == 0 or len(sims) == 0:
        return pd.DataFrame(columns=cols)
    pf = peaks["freq"].to_numpy()          # already sorted
    fs = sims["f_sim"].to_numpy()
    lo = np.searchsorted(pf, fs - tol_hz, side="left")
    hi = np.searchsorted(pf, fs + tol_hz, side="right")
    counts = hi - lo
    if counts.sum() == 0:
        return pd.DataFrame(columns=cols)
    sim_rep = np.repeat(np.arange(len(fs)), counts)
    starts = np.repeat(lo, counts)
    offs = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    peak_rep = starts + offs

    s = sims.iloc[sim_rep].reset_index(drop=True)
    ions = ion_db.iloc[s["ion_row"].to_numpy()].reset_index(drop=True)
    pk = peaks.iloc[peak_rep].reset_index(drop=True)
    out = pd.DataFrame({
        "peak_row": peak_rep,
        "sim_row": sim_rep,
        "ion": ions["ion"],
        "ion_base": ions["ion_base"],
        "Z": ions["ZZZi"].astype(int),
        "N": ions["NNN"].astype(int),
        "A": ions["AAA"].astype(int),
        "charge": ions["charge"].astype(int),
        "harmonic": s["harmonic"].astype(int),
        "sim_freq": s["f_sim"],
        "peak_freq": pk["freq"],
        "power": pk["power"],
        "snr": pk["snr"],
        "kind": pk["kind"],
    })
    out["delta_hz"] = out["peak_freq"] - out["sim_freq"]
    absd = out["delta_hz"].abs()
    out["closest_for_ion"] = (absd == absd.groupby(out["sim_row"]).transform("min")).astype(int)
    out["closest_for_peak"] = (absd == absd.groupby(out["peak_row"]).transform("min")).astype(int)
    return out[cols]


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_spectra(directory, pattern="*.npz"):
    if not directory or not os.path.isdir(directory):
        return []
    names = [os.path.basename(f) for f in glob.glob(os.path.join(directory, pattern or "*.npz"))]
    return sorted(names, key=natural_key)


def load_spectrum(directory, name):
    data = unpack_npz(paths.remap_legacy(os.path.join(directory, name)))
    return np.asarray(data["arr_0"], dtype=float), np.asarray(data["arr_1"], dtype=float)


def analyse_spectrum(ff, pp, s, predicted_ref):
    """Full per-spectrum pipeline.  Returns dict with peaks, ref info, sims, matches."""
    peaks, base, sig = detect_peaks(ff, pp, s)
    ref = find_reference(peaks, predicted_ref, s["ref_window_hz"], s["ref_mode"],
                         s["ref_min_height"])
    if ref is not None:
        ref_used, status = float(ref["freq"]), "found"
    elif s["use_predicted"]:
        ref_used, status = float(predicted_ref), "predicted"
    else:
        ref_used, status = None, "not_found"

    ion_db, ref_mq = build_ion_database(*s["ion_key"])
    allowed = hl_allowed(ion_db, s)
    sims = matches = None
    if ref_used is not None:
        sims = simulate_frequencies(ion_db, ref_mq, ref_used, s["ref_harmonic"], s["gamma_t"],
                                    s["harmonics"], s["correction_enabled"],
                                    s["corr_a"], s["corr_b"], s["corr_c"])
        lo, hi = ff.min() - s["match_tol_hz"], ff.max() + s["match_tol_hz"]
        sims = sims[(sims["f_sim"] >= lo) & (sims["f_sim"] <= hi)].reset_index(drop=True)
        # ions excluded by the half-life cut are simulated (and drawn) but
        # never used for identification
        sims_id = sims[allowed[sims["ion_row"].to_numpy()]].reset_index(drop=True)
        matches = match_peaks(peaks, sims_id, ion_db, s["match_tol_hz"])
    return dict(peaks=peaks, base=base, sig=sig, ref=ref, ref_used=ref_used, status=status,
                sims=sims, matches=matches, ion_db=ion_db, excluded=~allowed)


# ===========================================================================
# Settings
# ===========================================================================

def _load_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return {}



DEFAULTS = dict(
    run_label          = "",
    spectra_dir        = os.path.join(paths.spectra_1d(), ""),
    file_pattern       = "*.npz",
    ref_name           = "",           # shared physics keys: filled from physics.json
    ref_frequency      = 0,
    ref_harmonic       = 0,
    ref_window_hz      = 500,
    ref_min_height     = 1000,
    ref_mode           = "strongest",
    ref_tracking       = ["on"],
    use_predicted      = [],
    z_lims             = "",
    n_lims             = "",
    el1                = "",
    el2                = "",
    mq_width           = 0.0,
    gamma_t            = 0.0,
    harmonics          = "",
    correction_enabled = [],
    corr_a             = 0.0,
    corr_b             = 0.0,
    corr_c             = 0.0,
    noise_block_hz     = 2000,
    noise_cut_factor   = 0,
    k_prom             = 5.0,
    k_height           = 5.0,
    rel_noise          = 0.05,
    min_width_bins     = 0,
    min_distance_hz    = 0,
    shoulders          = ["on"],
    shoulder_scale_hz  = 10,
    k_shoulder         = 5.0,
    match_tol_hz       = 100,
    min_half_life      = 0.0,    # 0 = no half-life cut
    min_half_life_unit = "ms",
    exclude_unknown_hl = [],
    snippet_half_width = 5000,   # Results tab, not a scan setting
    dark_mode          = [],     # display only, not a scan setting
)


def load_config():
    """Scanner settings: DEFAULTS, then the shared physics (physics.json), then
    the scanner's own saved values (which hold only its physics overrides)."""
    return physics.with_shared(DEFAULTS, _load_json(CONFIG_PATH))


_CONFIG_LOCK = threading.Lock()


def save_config(cfg):
    """Merge `cfg` into scanner_config.json (keys not in cfg are kept, e.g. the
    snippet half-width is saved separately from the scan settings).

    Several callbacks can save at the same moment (the server is threaded), so
    the read-merge-write is locked and the file is replaced atomically; this
    stops one save reading a half-written file and wiping the others.
    """
    try:
        with _CONFIG_LOCK:
            # physics values equal to the shared default are not stored, so
            # they follow physics.json; only real overrides are kept
            merged = physics.strip_shared({**_load_json(CONFIG_PATH), **cfg})
            tmp = CONFIG_PATH + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(merged, fh, indent=2)
            os.replace(tmp, CONFIG_PATH)
    except Exception:
        pass


def _ints(s, n=None, what="value"):
    if isinstance(s, (list, tuple)):
        s = ",".join(str(x) for x in s)
    try:
        vals = [int(x.strip()) for x in str(s).split(",") if x.strip()]
    except ValueError:
        raise ValueError(f"{what}: expected comma-separated integers, got '{s}'")
    if n is not None and len(vals) != n:
        raise ValueError(f"{what}: expected {n} integers, got '{s}'")
    return vals


def _f(v, what, default=None):
    if v is None or v == "":
        if default is not None:
            return float(default)
        raise ValueError(f"{what} is empty")
    try:
        return float(v)
    except ValueError:
        raise ValueError(f"{what}: '{v}' is not a number")


def parse_settings(raw):
    """Raw GUI values -> typed settings dict.  Raises ValueError with a readable message."""
    s = {}
    s["spectra_dir"] = (raw.get("spectra_dir") or "").strip()
    s["file_pattern"] = (raw.get("file_pattern") or "*.npz").strip()
    s["ref_name"] = (raw.get("ref_name") or "").strip()
    if not s["ref_name"]:
        raise ValueError("Reference ion name is empty")
    s["ref_frequency"] = _f(raw.get("ref_frequency"), "Reference frequency")
    s["ref_harmonic"] = int(_f(raw.get("ref_harmonic"), "Reference harmonic"))
    s["ref_window_hz"] = _f(raw.get("ref_window_hz"), "Reference search window")
    s["ref_min_height"] = _f(raw.get("ref_min_height"), "Reference min height", 0)
    s["ref_mode"] = raw.get("ref_mode") or "strongest"
    s["ref_tracking"] = bool(raw.get("ref_tracking"))
    s["use_predicted"] = bool(raw.get("use_predicted"))
    s["z_lims"] = _ints(raw.get("z_lims"), 2, "Z limits")
    s["n_lims"] = _ints(raw.get("n_lims"), 2, "N limits")
    s["el1"] = _ints(raw.get("el1"), 2, "el1")
    s["el2"] = _ints(raw.get("el2"), 2, "el2")
    s["mq_width"] = _f(raw.get("mq_width"), "m/q window")
    s["gamma_t"] = _f(raw.get("gamma_t"), "gamma_t")
    s["harmonics"] = _ints(raw.get("harmonics"), None, "Harmonics")
    if not s["harmonics"]:
        raise ValueError("Harmonics list is empty")
    s["correction_enabled"] = bool(raw.get("correction_enabled"))
    s["corr_a"] = _f(raw.get("corr_a"), "a", 0)
    s["corr_b"] = _f(raw.get("corr_b"), "b", 0)
    s["corr_c"] = _f(raw.get("corr_c"), "c", 0)
    s["noise_block_hz"] = _f(raw.get("noise_block_hz"), "Noise block", 0)
    s["noise_cut_factor"] = _f(raw.get("noise_cut_factor"), "Noise cutoff factor", 0)
    s["k_prom"] = _f(raw.get("k_prom"), "Prominence threshold", 0)
    s["k_height"] = _f(raw.get("k_height"), "Height threshold", 0)
    s["rel_noise"] = _f(raw.get("rel_noise"), "Relative noise", 0)
    s["min_width_bins"] = _f(raw.get("min_width_bins"), "Min width", 0)
    s["min_distance_hz"] = _f(raw.get("min_distance_hz"), "Min separation", 0)
    s["shoulders"] = bool(raw.get("shoulders"))
    s["shoulder_scale_hz"] = _f(raw.get("shoulder_scale_hz"), "Shoulder scale", 0)
    s["k_shoulder"] = _f(raw.get("k_shoulder"), "Shoulder threshold", 0)
    s["match_tol_hz"] = _f(raw.get("match_tol_hz"), "Match window")
    s["min_half_life"] = _f(raw.get("min_half_life"), "Min half-life", 0)
    s["min_half_life_unit"] = raw.get("min_half_life_unit") or "ms"
    if s["min_half_life_unit"] not in HL_UNITS:
        raise ValueError(f"Half-life unit '{s['min_half_life_unit']}' not recognised")
    s["exclude_unknown_hl"] = bool(raw.get("exclude_unknown_hl"))
    s["ion_key"] = (tuple(s["z_lims"]), tuple(s["n_lims"]), tuple(s["el1"]),
                    tuple(s["el2"]), s["ref_name"], s["mq_width"])
    return s


def settings_for_storage(s):
    """JSON-safe settings dict (drops the internal cache key)."""
    return {k: v for k, v in s.items() if k != "ion_key"}


def settings_hash(s):
    blob = json.dumps(settings_for_storage(s), sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:8]


# ===========================================================================
# Database
# ===========================================================================

SCHEMA = """
CREATE TABLE run_info (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE spectra (
    spectrum_id   INTEGER PRIMARY KEY,
    seq           INTEGER,
    name          TEXT,
    ref_status    TEXT,
    ref_predicted REAL,
    ref_found     REAL,
    ref_shift_hz  REAL,
    n_peaks       INTEGER,
    n_ids         INTEGER,
    f_min         REAL,
    f_max         REAL,
    noise_sigma   REAL,
    error         TEXT
);
CREATE TABLE peaks (
    peak_id     INTEGER PRIMARY KEY,
    spectrum_id INTEGER,
    freq        REAL,
    power       REAL,
    prominence  REAL,
    snr         REAL,
    height_snr  REAL,
    width_hz    REAL,
    kind        TEXT,
    identified  INTEGER
);
CREATE TABLE identifications (
    id               INTEGER PRIMARY KEY,
    spectrum_id      INTEGER,
    peak_id          INTEGER,
    ion              TEXT,
    ion_base         TEXT,
    Z INTEGER, N INTEGER, A INTEGER, charge INTEGER,
    harmonic         INTEGER,
    sim_freq         REAL,
    peak_freq        REAL,
    delta_hz         REAL,
    power            REAL,
    snr              REAL,
    kind             TEXT,
    closest_for_ion  INTEGER,
    closest_for_peak INTEGER
);
CREATE INDEX idx_id_ion      ON identifications(ion);
CREATE INDEX idx_id_ion_base ON identifications(ion_base);
CREATE INDEX idx_id_spec     ON identifications(spectrum_id);
CREATE INDEX idx_pk_spec     ON peaks(spectrum_id);
CREATE VIEW hits AS
    SELECT i.*, s.name AS spectrum, s.seq AS seq, s.ref_found AS ref_found
    FROM identifications i JOIN spectra s USING (spectrum_id);
"""


def _kv_insert(con, table, d):
    con.executemany(f"INSERT OR REPLACE INTO {table}(key, value) VALUES (?, ?)",
                    [(k, json.dumps(v)) for k, v in d.items()])


def read_kv(con, table):
    return {k: json.loads(v) for k, v in con.execute(f"SELECT key, value FROM {table}")}


# ===========================================================================
# Scan worker (runs in a background thread)
# ===========================================================================

SCAN_STATE = dict(running=False, cancel=False, done=0, total=0, msg="Idle", db_path=None,
                  error=None)
SCAN_LOCK = threading.Lock()


def _set_state(**kw):
    with SCAN_LOCK:
        SCAN_STATE.update(kw)


def run_scan(s, label):
    t0 = time.time()
    db_path = None
    try:
        files = list_spectra(s["spectra_dir"], s["file_pattern"])
        if not files:
            raise ValueError(f"No files matching '{s['file_pattern']}' in {s['spectra_dir']}")
        _set_state(total=len(files), done=0, msg="Building ion database…")
        build_ion_database(*s["ion_key"])  # validates the reference ion early

        os.makedirs(SCANS_DIR, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        h = settings_hash(s)
        stem = f"scan_{stamp}_{h}"
        db_path = os.path.join(SCANS_DIR, stem + ".sqlite")
        _set_state(db_path=db_path)

        con = sqlite3.connect(db_path)
        con.executescript(SCHEMA)
        stored = settings_for_storage(s)
        _kv_insert(con, "settings", stored)
        info = dict(created=datetime.datetime.now().isoformat(timespec="seconds"),
                    label=label, settings_hash=h, status="running",
                    n_files=len(files), scanner_version=1)
        _kv_insert(con, "run_info", info)
        con.commit()

        with open(os.path.join(SCANS_DIR, stem + ".json"), "w") as fh:
            json.dump({"run_info": info, "settings": stored}, fh, indent=2)

        predicted = s["ref_frequency"]
        n_ids_total = n_found = 0
        for seq, name in enumerate(files):
            if SCAN_STATE["cancel"]:
                break
            _set_state(msg=f"{seq + 1}/{len(files)}  {name}")
            err = None
            try:
                ff, pp = load_spectrum(s["spectra_dir"], name)
                r = analyse_spectrum(ff, pp, s, predicted)
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                con.execute("INSERT INTO spectra(seq, name, ref_status, ref_predicted, error) "
                            "VALUES (?,?,?,?,?)", (seq, name, "error", predicted, err))
                _set_state(done=seq + 1)
                continue

            peaks, matches = r["peaks"], r["matches"]
            n_ids = 0 if matches is None else len(matches)
            cur = con.execute(
                "INSERT INTO spectra(seq, name, ref_status, ref_predicted, ref_found, ref_shift_hz,"
                " n_peaks, n_ids, f_min, f_max, noise_sigma) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (seq, name, r["status"], predicted,
                 r["ref_used"], None if r["ref_used"] is None else r["ref_used"] - s["ref_frequency"],
                 len(peaks), n_ids, float(ff.min()), float(ff.max()), float(np.median(r["sig"]))))
            sid = cur.lastrowid

            identified = np.zeros(len(peaks), dtype=int)
            if matches is not None and len(matches):
                identified[matches["peak_row"].unique()] = 1
            con.executemany(
                "INSERT INTO peaks(spectrum_id, freq, power, prominence, snr, height_snr, width_hz,"
                " kind, identified) VALUES (?,?,?,?,?,?,?,?,?)",
                [(sid, float(a), float(b), None if pd.isna(c) else float(c), float(d), float(e),
                  None if pd.isna(f) else float(f), k, int(i))
                 for a, b, c, d, e, f, k, i in zip(peaks["freq"], peaks["power"], peaks["prominence"],
                                                   peaks["snr"], peaks["height_snr"],
                                                   peaks["width_hz"], peaks["kind"], identified)])
            # peak_id of row j = first id + j (rows inserted in order)
            first_pid = con.execute("SELECT MIN(peak_id) FROM peaks WHERE spectrum_id=?",
                                    (sid,)).fetchone()[0]
            if n_ids:
                m = matches
                con.executemany(
                    "INSERT INTO identifications(spectrum_id, peak_id, ion, ion_base, Z, N, A, charge,"
                    " harmonic, sim_freq, peak_freq, delta_hz, power, snr, kind, closest_for_ion,"
                    " closest_for_peak) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    [(sid, int(first_pid + pr), io, ib, int(z), int(nn), int(a), int(q), int(hh),
                      float(fs), float(fp), float(dd), float(pw), float(sn), kd, int(c1), int(c2))
                     for pr, io, ib, z, nn, a, q, hh, fs, fp, dd, pw, sn, kd, c1, c2 in zip(
                         m["peak_row"], m["ion"], m["ion_base"], m["Z"], m["N"], m["A"],
                         m["charge"], m["harmonic"], m["sim_freq"], m["peak_freq"],
                         m["delta_hz"], m["power"], m["snr"], m["kind"],
                         m["closest_for_ion"], m["closest_for_peak"])])
            n_ids_total += n_ids

            if r["status"] == "found":
                n_found += 1
                if s["ref_tracking"]:
                    predicted = r["ref_used"]
            if seq % 20 == 0:
                con.commit()
            _set_state(done=seq + 1)

        status = "cancelled" if SCAN_STATE["cancel"] else "completed"
        info.update(status=status, n_spectra_done=SCAN_STATE["done"], n_ref_found=n_found,
                    n_identifications=n_ids_total, duration_s=round(time.time() - t0, 1))
        _kv_insert(con, "run_info", info)
        con.commit()
        con.close()

        with open(os.path.join(SCANS_DIR, stem + ".json"), "w") as fh:
            json.dump({"run_info": info, "settings": stored}, fh, indent=2)
        _append_index(stem, info, stored)
        _set_state(running=False, msg=f"{status.capitalize()}: {SCAN_STATE['done']} spectra, "
                                      f"{n_ids_total} identifications, ref found in {n_found}. "
                                      f"Saved {os.path.basename(db_path)}")
    except Exception as exc:
        traceback.print_exc()
        _set_state(running=False, error=str(exc), msg=f"Scan failed: {exc}")


INDEX_FIELDS = ["file", "created", "label", "status", "settings_hash", "n_files",
                "n_spectra_done", "n_ref_found", "n_identifications", "spectra_dir", "ref_name",
                "ref_frequency", "ref_harmonic", "ref_window_hz", "harmonics", "gamma_t",
                "correction_enabled", "match_tol_hz", "k_prom", "k_height", "rel_noise",
                "shoulders", "settings_json"]


def _append_index(stem, info, stored):
    new = not os.path.exists(INDEX_PATH)
    row = {**{k: stored.get(k) for k in INDEX_FIELDS}, **{k: info.get(k) for k in INDEX_FIELDS if k in info}}
    row["file"] = stem + ".sqlite"
    row["settings_json"] = stem + ".json"
    with open(INDEX_PATH, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=INDEX_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def list_databases():
    """[(path, run_info, settings)] newest first."""
    out = []
    for path in sorted(glob.glob(os.path.join(SCANS_DIR, "*.sqlite")), reverse=True):
        try:
            con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            info, st = read_kv(con, "run_info"), read_kv(con, "settings")
            con.close()
            out.append((path, info, st))
        except Exception:
            continue
    return out


def db_query(path, sql, params=()):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return pd.read_sql_query(sql, con, params=params)
    finally:
        con.close()


# ===========================================================================
# Figures
# ===========================================================================

def _vlines(xs, y0, y1, hover, n_points=20):
    """Vertical lines as one trace.  Each line has n_points points (as in
    viewer.py) so its tooltip appears anywhere along it, not just at the ends."""
    ys = list(np.linspace(y0, y1, n_points))
    X, Y, H = [], [], []
    for x, h in zip(xs, hover):
        X += [x] * n_points + [None]
        Y += ys + [None]
        H += [h] * n_points + [None]
    return X, Y, H


def empty_fig(msg=""):
    fig = go.Figure()
    fig.update_layout(template="plotly_white", margin=dict(l=60, r=20, t=40, b=40))
    if msg:
        fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                           font=dict(size=15, color="#888"))
    return fig


EXCLUDED_TRACE = "Excluded (half-life cut)"


def add_sim_lines(fig, sims, ion_db, matches, y0, y1, highlight_ion=None,
                  excluded=None, show_excluded=True):
    """Draw simulated ion frequencies as vertical dashed lines with tooltips.

    Grey: no peak matched; dark grey: identified (a peak within the match
    window); orange: `highlight_ion`; light dotted: excluded from
    identification by the half-life cut (`excluded`, bool per ion_db row;
    hidden if not show_excluded).  `matches` needs columns ion, harmonic.
    """
    if sims is None or not len(sims):
        return
    ions = ion_db["ion"].to_numpy()
    names = ions[sims["ion_row"].to_numpy()]
    hover = ion_db["hover"].to_numpy()[sims["ion_row"].to_numpy()]
    labels = [f"{lab}<br>Harmonic: {h}<br>f_sim = {f:.1f} Hz" for lab, h, f in
              zip(hover, sims["harmonic"], sims["f_sim"])]
    matched = set() if matches is None else set(zip(matches["ion"], matches["harmonic"]))
    is_m = np.array([(n, h) in matched for n, h in zip(names, sims["harmonic"])])
    is_hi = np.array([highlight_ion is not None and n == highlight_ion for n in names])
    is_ex = (np.zeros(len(names), dtype=bool) if excluded is None
             else np.asarray(excluded, dtype=bool)[sims["ion_row"].to_numpy()])
    is_ex &= ~is_hi
    groups = [
        (~is_m & ~is_hi & ~is_ex, "Simulated (no peak)", dict(color="#bbbbbb", width=1, dash="dash")),
        (is_m & ~is_hi & ~is_ex, "Simulated (identified)", dict(color="#555555", width=1.5, dash="dash")),
        (is_hi, highlight_ion or "", dict(color=ACCENT, width=3, dash="solid")),
    ]
    # the excluded trace is always added (hidden if not show_excluded) so the
    # snippet's toggle can show/hide it in the browser without a redraw
    groups.append((is_ex, EXCLUDED_TRACE, dict(color="#c8c8c8", width=1, dash="dot")))
    for mask, name, style in groups:
        if mask.any():
            X, Y, H = _vlines(sims["f_sim"].to_numpy()[mask], y0, y1,
                              np.array(labels, dtype=object)[mask])
            fig.add_trace(go.Scattergl(x=X, y=Y, mode="lines", name=name, line=style,
                                       hoverinfo="text", hovertext=H, connectgaps=False,
                                       visible=True if (name != EXCLUDED_TRACE or show_excluded)
                                       else False))


def spectrum_figure(ff, pp, peaks, base, sig, s, r, title, xrange=None, highlight_ion=None):
    fig = go.Figure()
    # "hvh" centres each step on its frequency value (vertical edges halfway
    # between bins); "hv" would start each step at its value.
    fig.add_trace(go.Scattergl(x=ff, y=pp, mode="lines", name="Spectrum", hoverinfo="skip",
                               line=dict(shape="hvh", color="black", width=1)))
    if base is not None and sig is not None:
        fig.add_trace(go.Scattergl(x=ff, y=base + s["k_height"] * sig, mode="lines",
                                   name=f"baseline + {s['k_height']:g}σ", hoverinfo="skip",
                                   line=dict(color="#999", width=1, dash="dot")))
    ymin, ymax = float(np.nanmin(pp)), float(np.nanmax(pp))
    pad = 0.05 * (ymax - ymin or 1)

    # simulated ions
    add_sim_lines(fig, r.get("sims"), r.get("ion_db"), r.get("matches"),
                  ymin - pad, ymax + pad, highlight_ion,
                  excluded=r.get("excluded"), show_excluded=r.get("show_excluded", True))

    # reference window + found reference
    pred = r.get("predicted")
    if pred is not None:
        w = s["ref_window_hz"]
        fig.add_vrect(x0=pred - w, x1=pred + w, fillcolor=ACCENT_PALE, opacity=0.7,
                      line_width=0, layer="below")
        if s.get("ref_min_height", 0) > 0:
            fig.add_shape(type="line", x0=pred - w, x1=pred + w, y0=s["ref_min_height"],
                          y1=s["ref_min_height"], line=dict(color=ACCENT, width=1, dash="dot"))
    if r.get("ref_used") is not None:
        fig.add_vline(x=r["ref_used"], line=dict(color=ACCENT, width=2),
                      annotation_text=f"ref {s['ref_name']} ({r['status']})",
                      annotation_font_color=ACCENT)

    # peaks
    if peaks is not None and len(peaks):
        for kind, sym, col in [("max", "triangle-up", ACCENT), ("shoulder", "diamond", "#444")]:
            p = peaks[peaks["kind"] == kind]
            if len(p):
                hov = [f"{kind}<br>f = {f:.2f} Hz<br>P = {pw:.4g}<br>SNR = {sn:.1f}"
                       for f, pw, sn in zip(p["freq"], p["power"], p["snr"])]
                fig.add_trace(go.Scatter(x=p["freq"], y=p["power"], mode="markers",
                                         name=f"{'Maxima' if kind == 'max' else 'Shoulders'} ({len(p)})",
                                         marker=dict(symbol=sym, color=col, size=9,
                                                     line=dict(color="black", width=0.6)),
                                         hoverinfo="text", hovertext=hov))
    fig.update_layout(title=title, xaxis_title="Frequency (Hz)", yaxis_title="Schottky Power (a.u.)",
                      hovermode="x", dragmode="pan", template="plotly_white",
                      font=dict(size=14), margin=dict(l=60, r=20, t=50, b=40),
                      legend=dict(orientation="h", y=-0.15, x=0), uirevision=title)
    if xrange:
        fig.update_xaxes(range=xrange, autorange=False)
        m = (ff >= xrange[0]) & (ff <= xrange[1])
        if m.any():
            lo, hi = float(pp[m].min()), float(pp[m].max())
            fig.update_yaxes(range=[lo - 0.08 * (hi - lo or 1), hi + 0.12 * (hi - lo or 1)])
    return fig


# ===========================================================================
# Layout
# ===========================================================================

CFG = load_config()
LABEL_STYLE   = {"fontWeight": "600", "marginBottom": "2px", "fontSize": "13px"}
INPUT_STYLE   = {"width": "100%", "marginBottom": "10px"}
SECTION_STYLE = {"fontWeight": "700", "color": "#555", "marginBottom": "6px", "fontSize": "12px",
                 "textTransform": "uppercase", "letterSpacing": "0.04em"}
HELP_STYLE    = {"color": "#888", "fontSize": "11px", "display": "block", "marginTop": "2px"}
TABLE_HEADER  = {"backgroundColor": ACCENT, "color": "white", "fontWeight": "600", "fontSize": "12px"}
TABLE_CELL    = {"fontSize": "12px", "padding": "3px 8px", "textAlign": "right",
                 "fontFamily": "monospace"}
TABLE_ODD     = [{"if": {"row_index": "odd"}, "backgroundColor": ACCENT_PALE}]

# key -> (component id, property)
SETTING_FIELDS = {}


def inp(label, key, number=False, help=None, step="any"):
    cid = f"s-{key}"
    SETTING_FIELDS[key] = (cid, "value")
    kw = dict(id=cid, value=CFG[key], debounce=True, className="form-control form-control-sm",
              style={"width": "100%", "fontSize": "13px"})
    if number:
        try:
            kw["value"] = float(CFG[key])
        except (TypeError, ValueError):
            kw["value"] = None
        kw.update(type="number", step=step)
    else:
        kw["value"] = str(CFG[key])
    children = [html.Label(label, style=LABEL_STYLE), dcc.Input(**kw)]
    if help:
        children.append(html.Small(help, style=HELP_STYLE))
    return html.Div(children, style=INPUT_STYLE)


def check(label, key, help=None):
    cid = f"s-{key}"
    SETTING_FIELDS[key] = (cid, "value")
    children = [dcc.Checklist(id=cid, options=[{"label": " " + label, "value": "on"}],
                              value=CFG[key], style={"fontSize": "13px"})]
    if help:
        children.append(html.Small(help, style=HELP_STYLE))
    return html.Div(children, style=INPUT_STYLE)


def radio(label, key, options):
    cid = f"s-{key}"
    SETTING_FIELDS[key] = (cid, "value")
    return html.Div([
        html.Label(label, style=LABEL_STYLE),
        dcc.RadioItems(id=cid, options=options, value=CFG[key], style={"fontSize": "13px"},
                       inputStyle={"marginRight": "5px"},
                       labelStyle={"display": "block"}),
    ], style=INPUT_STYLE)


def hl_cut_inputs():
    """Half-life cut: number + unit dropdown on one row."""
    SETTING_FIELDS["min_half_life"] = ("s-min_half_life", "value")
    SETTING_FIELDS["min_half_life_unit"] = ("s-min_half_life_unit", "value")
    try:
        val = float(CFG["min_half_life"])
    except (TypeError, ValueError):
        val = 0.0
    unit = CFG["min_half_life_unit"] if CFG["min_half_life_unit"] in HL_UNITS else "ms"
    return html.Div([
        html.Label("Exclude ions with half-life below  [0 = off]", style=LABEL_STYLE),
        html.Div([
            dcc.Input(id="s-min_half_life", type="number", step="any", min=0, value=val,
                      debounce=True, className="form-control form-control-sm",
                      style={"flex": "1", "fontSize": "13px", "marginRight": "6px"}),
            dcc.Dropdown(id="s-min_half_life_unit", value=unit, clearable=False,
                         options=[{"label": u, "value": u} for u in HL_CUT_UNITS],
                         style={"width": "80px", "fontSize": "13px"}),
        ], style={"display": "flex", "alignItems": "center"}),
        html.Small("NUBASE ground-state / isomer half-life (neutral atom). Excluded ions are "
                   "not identified, but their simulated lines are still drawn (dotted)",
                   style=HELP_STYLE),
    ], style=INPUT_STYLE)


def section(title):
    return [html.Hr(style={"margin": "10px 0"}), html.P(title, style=SECTION_STYLE)]


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
       style={"border": "1px solid #dddddd", "borderRadius": "6px", "background": "white", "width": "640px", "maxWidth": "92vw",
              "padding": "14px 16px", "boxShadow": "0 8px 30px rgba(0,0,0,0.3)"}),
    dcc.Store(id="fb-path", data=None),
    dcc.Store(id="fb-target", data=None),
], id="fb-modal",
   style={"display": "none", "position": "fixed", "inset": "0", "zIndex": "1000",
          "background": "rgba(0,0,0,0.35)", "alignItems": "center", "justifyContent": "center"})

FB_SHOWN = {"display": "flex", "position": "fixed", "inset": "0", "zIndex": "1000",
            "background": "rgba(0,0,0,0.35)", "alignItems": "center", "justifyContent": "center"}
FB_HIDDEN = {**FB_SHOWN, "display": "none"}


sidebar = html.Div([
    html.H5("Scan settings", style={"fontWeight": "700", "marginBottom": "4px"}),

    *section("Run"),
    inp("Run label (optional)", "run_label", help="Free text stored with the database"),
    with_browse(inp("Spectra directory", "spectra_dir"), "btn-browse-spectra_dir"),
    inp("File pattern", "file_pattern", help="Files are processed in natural (numeric) order"),

    *section("Reference ion"),
    inp("Ion name", "ref_name"),
    inp("Frequency (Hz) — in first spectrum", "ref_frequency", True, step=1),
    inp("Harmonic", "ref_harmonic", True, step=1),
    inp("Search window ± (Hz)", "ref_window_hz", True),
    inp("Min reference height (a.u.)", "ref_min_height", True,
        help="Only peaks at least this tall (raw power) can be the reference. 0 = off"),
    radio("Pick in window", "ref_mode", [
        {"label": "Strongest peak", "value": "strongest"},
        {"label": "Nearest to prediction", "value": "nearest"},
    ]),
    check("Track: centre next search on last found ref", "ref_tracking"),
    check("If not found, use predicted position", "use_predicted",
          help="Otherwise the spectrum is stored without identifications"),

    *section("Ion selection"),
    inp("Z limits  [min, max]", "z_lims"),
    inp("N limits  [min, max]", "n_lims"),
    inp("el1  [A, Z]", "el1"),
    inp("el2  [A, Z]", "el2"),
    inp("m/q window (± value)", "mq_width", True),

    *section("Ring / harmonics"),
    inp("γ_t", "gamma_t", True),
    inp("Harmonics (comma-separated)", "harmonics"),

    *section("Frequency correction"),
    check("Second order correction", "correction_enabled"),
    inp("a  (f² coefficient)", "corr_a", True),
    inp("b  (f coefficient)", "corr_b", True),
    inp("c  (constant)", "corr_c", True),

    *section("Peak detection (raw data)"),
    inp("Noise block (Hz)", "noise_block_hz", True,
        help="Block size for the local median / MAD noise estimate. 0 = global"),
    inp("Noise cutoff factor X  [0 = off]", "noise_cut_factor", True,
        help="Bins above −min(spectrum)·X are excluded from the noise estimate"),
    inp("Prominence threshold (σ)", "k_prom", True,
        help="Keep maxima with prominence ≥ k·σ"),
    inp("Height threshold (σ)", "k_height", True,
        help="Keep peaks rising ≥ k·σ above the local baseline"),
    inp("Relative noise r", "rel_noise", True,
        help="σ = √(σ_bg² + (r·height)²). Suppresses noise spikes on top of big peaks. 0 = off"),
    inp("Min width FWHM (bins)  [0 = off]", "min_width_bins", True,
        help="Width at half prominence, in frequency bins (can be fractional)"),
    inp("Min separation (Hz)  [0 = off]", "min_distance_hz", True),
    check("Detect shoulders (merged peaks)", "shoulders"),
    inp("Shoulder scale s (Hz)", "shoulder_scale_hz", True,
        help="Step of the 2nd difference, roughly half the width of a peak"),
    inp("Shoulder threshold (σ)", "k_shoulder", True),

    *section("Identification"),
    inp("Match window ± x (Hz)", "match_tol_hz", True,
        help="Any peak within ±x of a simulated ion is identified as it"),
    hl_cut_inputs(),
    check("Also exclude ions with unknown half-life", "exclude_unknown_hl",
          help="Blank or 'p-unst' in NUBASE"),
    html.Div(style={"height": "30px"}),
], className="theia-sidebar",
   style={"width": "300px", "minWidth": "300px", "padding": "14px 18px", "background": "#f9f9f9",
          "overflowY": "auto", "borderRight": "1px solid #dddddd", "height": "calc(100vh - 42px)"})


def _table(id_, columns, page_size=20, height=None, overflow_x="auto", **kw):
    return dash_table.DataTable(
        id=id_, columns=[{"name": c, "id": c} for c in columns], data=[],
        page_size=page_size, sort_action="native", filter_action="native",
        style_table={"overflowX": overflow_x,
                     **({"maxHeight": height, "overflowY": "auto"} if height else {})},
        style_header=TABLE_HEADER, style_cell=TABLE_CELL, style_data_conditional=TABLE_ODD,
        style_filter={"backgroundColor": "#f9f9f9"}, **kw)


BTN = {"marginRight": "8px"}
ACCEPT_BTN = {"backgroundColor": "#2e7d32", "color": "white", "fontSize": "22px", "fontWeight": "700",
              "width": "56px", "height": "42px", "lineHeight": "1", "marginRight": "16px"}
REJECT_BTN = {**ACCEPT_BTN, "backgroundColor": "#c62828", "marginRight": "8px"}
UNSURE_BTN = {**ACCEPT_BTN, "backgroundColor": ACCENT, "marginRight": "8px"}
CONFIRM_BAR = {"display": "flex", "alignItems": "center", "flexWrap": "wrap", "margin": "8px 0 2px 0"}
# Outline used around the plots in the Results tab
BOX_STYLE = {"border": "1px solid #dddddd", "borderRadius": "6px", "padding": "10px",
             "background": "#ffffff"}
TABLE_PAGE_SIZE = 15    # rows per page of the hits table
TABLE_ROW_PX = 30       # DataTable row height
FILTER_HEAD_PX = 120    # height of the filter block above the hits table
SUMMARY_EXTRA_ROWS = 3  # extra rows in the ions-found table, which starts higher up
SUMMARY_HEAD_PX = FILTER_HEAD_PX - SUMMARY_EXTRA_ROWS * TABLE_ROW_PX   # its heading (30 px)

scan_main = html.Div([
    html.Div([
        html.Div([
            html.Label("Preview spectrum", style=LABEL_STYLE),
            dcc.Dropdown(id="preview-file", options=[], placeholder="Select a spectrum…",
                         style={"fontSize": "13px", "width": "380px"}),
        ], style={"marginRight": "12px"}),
        html.Button("Preview", id="btn-preview", className="btn btn-outline-secondary btn-sm", style=BTN),
        html.Button("Run scan", id="btn-run", className="btn btn-sm",
                    style={**BTN, "backgroundColor": ACCENT, "color": "white"}),
        html.Button("Cancel", id="btn-cancel", className="btn btn-outline-danger btn-sm", style=BTN),
    ], style={"display": "flex", "alignItems": "flex-end", "marginBottom": "8px"}),
    html.Small("Preview uses the nominal reference frequency as the prediction. During a scan the "
               "prediction follows the reference found in the previous spectrum.", style=HELP_STYLE),
    html.Div([
        html.Div(id="progress-bar", style={"height": "100%", "width": "0%", "background": ACCENT,
                                           "transition": "width 0.4s"}),
    ], className="theia-progress-track",
       style={"height": "8px", "background": "#eeeeee", "borderRadius": "4px", "overflow": "hidden",
              "margin": "8px 0 4px 0"}),
    html.Div(id="scan-status", style={"fontSize": "13px", "color": "#333"}),
    html.Div([html.Span(id="action-msg"), html.Span(id="cancel-msg", style={"marginLeft": "8px"})],
             style={"fontSize": "12px", "color": "#888", "marginBottom": "6px"}),
    html.Div(id="run-error", style={"color": "#b00", "fontSize": "13px", "whiteSpace": "pre-wrap"}),
    html.Div(id="preview-info", style={"fontSize": "13px", "color": "#333", "whiteSpace": "pre-wrap"}),
    html.Div(id="settings-error", style={"color": "#b00", "fontSize": "13px", "whiteSpace": "pre-wrap"}),
    dcc.Graph(id="preview-graph", figure=empty_fig("Choose a spectrum and press Preview"),
              style={"height": "58vh"}, config={"scrollZoom": True}),
    html.H6("Identified in preview", style={"fontWeight": "600", "marginTop": "6px"}),
    _table("preview-table", ["ion", "harmonic", "peak_freq", "sim_freq", "delta_hz", "power", "snr",
                             "kind", "closest_for_ion", "closest_for_peak"], page_size=15),
], style={"flex": "1", "minWidth": "0", "padding": "14px 18px", "overflowY": "auto",
          "height": "calc(100vh - 42px)"})

results_tab = html.Div([
    html.Div([
        html.Div([
            html.Label("Database", style=LABEL_STYLE),
            dcc.Dropdown(id="db-select", options=[], placeholder="Select a scan database…",
                         style={"fontSize": "13px", "width": "720px"}),
        ], style={"marginRight": "10px"}),
        html.Button("Refresh list", id="btn-refresh", className="btn btn-outline-secondary btn-sm"),
    ], style={"display": "flex", "alignItems": "flex-end", "marginBottom": "10px"}),

    html.Details([
        html.Summary("Run info and settings used", style={"fontWeight": "600", "cursor": "pointer"}),
        html.Div([
            html.Div(_table("runinfo-table", ["key", "value"], page_size=50, height="320px"),
                     style={"width": "40%", "marginRight": "16px"}),
            html.Div(_table("settings-table", ["key", "value"], page_size=50, height="320px"),
                     style={"width": "55%"}),
        ], style={"display": "flex", "marginTop": "6px"}),
    ], open=False, style={"marginBottom": "12px"}),

    # Grid layout.  The "Ions found" heading sits level with the filter labels
    # and its table starts straight under it; the hits table starts below the
    # filters.  Both header blocks have fixed heights, and the ions-found table
    # shows exactly enough extra rows to fill the difference, so the two
    # tables end level with each other (DataTable rows are 30 px).
    #   left  (rows 1-2): heading + ions-found table
    #   right row 1: filters        right row 2: hits table
    #   row 3: reference tracking (left) / snippet viewer (right)
    html.Div([
        # left, rows 1-2
        html.Div([
            html.Div(html.H6("Ions found (click a row to filter)",
                             style={"fontWeight": "600", "margin": "0", "paddingTop": "2px"}),
                     style={"height": f"{SUMMARY_HEAD_PX}px"}),
            _table("summary-table", ["ion", "n_spectra", "n_hits", "mean_delta_hz", "std_delta_hz"],
                   page_size=TABLE_PAGE_SIZE + SUMMARY_EXTRA_ROWS, export_format="csv",
                   overflow_x="scroll"),
        ], style={"gridColumn": "1", "gridRow": "1 / 3", "minWidth": "0"}),
        # right, row 1: filters
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
        ], style={"gridColumn": "2", "gridRow": "1", "height": f"{FILTER_HEAD_PX}px",
                  "minWidth": "0"}),

        # right, row 2: hits table (both tables have an Export button and an
        # always-visible horizontal scroll bar so their heights match)
        html.Div(_table("hits-table", ["spectrum", "seq", "ion", "harmonic", "peak_freq", "sim_freq",
                                       "delta_hz", "power", "snr", "kind", "closest_for_ion",
                                       "closest_for_peak"],
                        page_size=TABLE_PAGE_SIZE, export_format="csv", overflow_x="scroll"),
                 style={"gridColumn": "2", "gridRow": "2", "minWidth": "0"}),

        # row 3, left: reference tracking + per-spectrum reference viewer
        html.Div([
            html.Div([
                html.H6("Reference tracking", style={"fontWeight": "600", "marginBottom": "6px"}),
                dcc.Graph(id="ref-graph", figure=empty_fig(), style={"height": "280px"}),
                html.Small("Click a point to open that spectrum below", style=HELP_STYLE),
            ], className="theia-box", style={**BOX_STYLE, "marginTop": "10px"}),

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
                dcc.Checklist(id="refview-only-notfound",
                              options=[{"label": " Step only through spectra where the ref was not found",
                                        "value": "on"}],
                              value=[], inputStyle={"marginRight": "4px"},
                              style={"fontSize": "12px", "margin": "6px 0"}),
                html.Small("Double-click the plot to see the whole spectrum", style=HELP_STYLE),
                html.Div(id="refview-info", style={"fontSize": "12px", "color": "#333",
                                                   "whiteSpace": "pre-wrap"}),
                dcc.Graph(id="refview-graph", figure=empty_fig("Load a database"),
                          style={"height": "340px"}, config={"scrollZoom": True}),
                dcc.Store(id="refview-seq", data=0),
            ], className="theia-box", style={**BOX_STYLE, "marginTop": "10px"}),
        ], style={"gridColumn": "1", "gridRow": "3", "minWidth": "0"}),
        # row 3, right: peak-by-peak snippet viewer
        html.Div([
            # Peak-by-peak viewer: steps through the rows of the hits table as
            # currently filtered (and sorted) above
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
                dcc.Checklist(id="snippet-show-excluded", value=["on"],
                              options=[{"label": " Show ions excluded by half-life cut",
                                        "value": "on"}],
                              style={"fontSize": "13px"}),
            ], style={"display": "flex", "alignItems": "center", "flexWrap": "wrap"}),
            html.Small("Steps through every identification in the table above, in table order; "
                       "the filters and table sorting decide which peaks are shown",
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
            # Grows downwards to fill the row: this box and the reference-viewer
            # box on the left end at the same height and have the same padding,
            # so the bottoms of the two plots line up
            dcc.Graph(id="snippet-graph", figure=empty_fig(), responsive=True,
                      style={"flex": "1", "minHeight": "420px"},
                      config={"scrollZoom": True}),
        ], className="theia-box", style={**BOX_STYLE, "gridColumn": "2", "gridRow": "3", "minWidth": "0",
                  "alignSelf": "stretch", "marginTop": "10px",
                  "display": "flex", "flexDirection": "column"}),
    ], style={"display": "grid", "gridTemplateColumns": "34% minmax(0, 1fr)",
              "columnGap": "16px", "alignItems": "start"}),
], style={"padding": "14px 18px", "overflowY": "auto", "height": "calc(100vh - 42px)"})


# Peak-fit parameter table (one row per peak).  Value columns are editable;
# uncertainties and derived quantities are not.
FIT_PANEL_SHOWN = {"display": "block"}
FIT_PANEL_HIDDEN = {**FIT_PANEL_SHOWN, "display": "none"}

# line shapes available for each fitted peak
SHAPE_LABELS = {"gauss": "Gaussian", "lorentz": "Lorentzian", "pvoigt": "Pseudo-Voigt"}


# fix toggles: click the cell to switch between free (☐) and fixed (☑)
FIX_OFF, FIX_ON = "☐", "☑"
FIX_PARAMS = ("centre", "amp", "fwhm", "eta")
# per-peak limits for the fit (blank = default: centre within the fit region,
# FWHM between 0.05 bins and the region width)
LIMIT_KEYS = ("centre_min", "centre_max", "fwhm_min", "fwhm_max")

FIT_COLUMNS = [
    {"name": "peak", "id": "peak", "editable": False},
    # ion name (e.g. 207Tl81+) and harmonic: needed to send the peak to the
    # isochronicity curve
    {"name": "ion", "id": "ion", "editable": True},
    {"name": "h", "id": "harmonic", "type": "numeric", "editable": True},
    {"name": "shape", "id": "shape", "presentation": "dropdown", "editable": True},
    {"name": "centre (Hz)", "id": "centre", "type": "numeric", "editable": True},
    {"name": "± centre", "id": "d_centre", "editable": False},
    {"name": "fix", "id": "fix_centre", "editable": False},
    {"name": "centre min", "id": "centre_min", "type": "numeric", "editable": True},
    {"name": "centre max", "id": "centre_max", "type": "numeric", "editable": True},
    {"name": "amplitude", "id": "amp", "type": "numeric", "editable": True},
    {"name": "± amp", "id": "d_amp", "editable": False},
    {"name": "fix", "id": "fix_amp", "editable": False},
    {"name": "FWHM (Hz)", "id": "fwhm", "type": "numeric", "editable": True},
    {"name": "± FWHM", "id": "d_fwhm", "editable": False},
    {"name": "fix", "id": "fix_fwhm", "editable": False},
    {"name": "FWHM min", "id": "fwhm_min", "type": "numeric", "editable": True},
    {"name": "FWHM max", "id": "fwhm_max", "type": "numeric", "editable": True},
    {"name": "η (Lorentz fraction)", "id": "eta", "type": "numeric", "editable": True},
    {"name": "± η", "id": "d_eta", "editable": False},
    {"name": "fix", "id": "fix_eta", "editable": False},
    {"name": "area", "id": "area", "editable": False},
    {"name": "± area", "id": "d_area", "editable": False},
]


def fit_columns(pv, limits):
    """Table columns: η only with a pseudo-Voigt peak, limits only when shown."""
    drop = set()
    if not pv:
        drop |= {"eta", "d_eta", "fix_eta"}
    if not limits:
        drop |= set(LIMIT_KEYS)
    return [c for c in FIT_COLUMNS if c["id"] not in drop]


# Further output: reads files saved by confirmation mode (confirmed/*.csv) for
# further processing.  Sub-tabs: "Aligning and fitting" (file list, summing,
# peak fitting) and "Isochronicity Curve" (to come).
CALIB_BORDER = "1px solid #999999"
CALIB_TH = {"fontSize": "11px", "fontWeight": "700", "color": "#666", "textTransform": "uppercase",
            "letterSpacing": "0.04em", "padding": "4px 10px", "textAlign": "left",
            "border": CALIB_BORDER, "background": "#f9f9f9"}
CALIB_TD = {"fontSize": "13px", "padding": "5px 10px", "verticalAlign": "middle",
            "border": CALIB_BORDER}
SUGG_BOX_STYLE = {"marginLeft": "0", "marginTop": "4px", "width": "620px", "maxWidth": "90%",
                  "border": "1px solid #dddddd", "borderRadius": "6px", "background": "white",
                  "padding": "4px 0", "boxShadow": "0 4px 12px rgba(0,0,0,0.12)"}


# ---- ⓘ info buttons: the help text of a section, shown in a popup ----------
INFO_TEXTS = {
    "sum": ("Sum spectra", [
        "Each spectrum is added once, however many identifications or files it appears in.",
        "Aligned spectra are shifted (linear interpolation) so the chosen peak sits at the "
        "median of its positions.",
        "The plot opens zoomed to ±5 kHz around the ion/harmonic chosen in the dropdown.",
        "With an identified peak, spectra without that ion/harmonic are left out.",
        "Bins not covered by every shifted spectrum are dropped.",
        "Simulated ion lines: the strongest peak of the sum is taken as the reference ion (at its "
        "reference harmonic), with the scan settings stored in the first loaded file.  Ions in the "
        "loaded files are dark dashed, others light dashed, half-life-excluded dotted.",
        "Data source 'Single spectrum file': any .npz spectrum (e.g. a Tartarus sum in "
        "summed_1d, or the file open in the Viewer via 'Use Viewer spectrum') is shown as it is, "
        "for fitting and the energy calculation.  Its simulated lines use the Viewer settings "
        "(no half-life cut), with the strongest peak as the reference.",
        "Search ions…: type part of an ion name to list its simulated lines on the plotted "
        "spectrum; click one to zoom to f_sim ± the width (kHz, default ±10).  h−1 / h+1 then "
        "zoom to the same ion at the neighbouring harmonic.",
    ]),
    "fit": ("Peak fitting", [
        "Switch it on and click a peak to add it (the start values snap to the nearest maximum).",
        "Edit any value in the table to adjust the curve by hand, or press Fit.  Delete a row to "
        "remove a peak.",
        "The fit region is set by the two orange lines (set from the view when the first peak is "
        "added).  Drag a line, type the limits, or press 'Region = view'.",
        "Drag a peak's × marker to move its centre and amplitude together (cursor ✥); drag below "
        "the marker to move the centre only (cursor ↔); drag a ◆ handle at half height to change "
        "the FWHM.",
        "Each peak has its own line shape (shape column); 'Shape for new peaks' sets the shape of "
        "peaks added from then on.",
        "χ²/ndf uses a noise σ per bin estimated from the summed spectrum; uncertainties are "
        "scaled by √(χ²/ndf).",
        "'Send to isocurve' sends every fitted peak to the Isochronicity Curve tab.  Each peak "
        "needs an ion name with its charge (e.g. 207Tl81+, m207Tl81+) and a harmonic: type them "
        "in the ion / h columns, or you are asked for the missing ones (with nearby simulated "
        "ions as suggestions).  Press Fit first.",
    ]),
    "energy": ("Energy calculation", [
        "Step 1 – calibration: click two peaks, give their NUBASE names (or click a suggested "
        "nearby simulated ion) and press Calibrate.",
        "A click on a peak of the main fit (within half its FWHM) uses its fitted centre; anywhere "
        "else the peak is fitted on its own in a small window (shape for new peaks + constant "
        "background).",
        "Calibration: k = (ME₂ − ME₁) / (f₂ − f₁) from NUBASE (for two states of the same nuclide "
        "the excitation energies are used).",
        "Step 2: click peaks in pairs A, B:  ΔE = E(B) − E(A) = k·(f_B − f_A).",
        "A linear keV-per-Hz calibration is only accurate for pairs close in frequency to the "
        "calibration peaks (ideally the same A/q).",
    ]),
    "iso": ("Isochronicity curve", [
        "Points come from 'Send to isocurve' in the peak-fitting box of Aligning and fitting: "
        "one row per ion and harmonic, with the fitted centre, FWHM, amplitude, area (± 1σ) and "
        "the revolution frequency f_rev = centre / h.",
        "They are saved in outputs/isocurve/isocurve_points.csv of the active experiment, so "
        "they are kept between sessions.",
        "Sending the same ion and harmonic again replaces its point (you are asked first).",
        "Delete a row (×) to remove the point from the file.",
        "'fixed' lists parameters held fixed in the fit (their uncertainty is 0).",
    ]),
    "isofit": ("Isochronicity curve fit", [
        "σ_T(T) = √( ((1/γ² − 1/γ_t²) · σp/p · T)² + σ_sys² ),  with 1/γ² = 1 − (C / (c·T))², "
        "C the ring circumference.  Fitted parameters: γ_t, σp/p, σ_sys, within the min – max "
        "bounds.",
        "Weighted least squares with err σ(T) as absolute errors (as absolute_sigma=True); "
        "χ²/ndf = Σ((σ(T) − fit) / err σ(T))² / (N − 3).  Several γ_t start values are tried "
        "and the best fit kept.",
        "The fit is redone whenever the points, the 'in fit' ticks or the settings change.",
        "Leave a point out: click its ☑ in the 'in fit' column, or click the point on the plot "
        "(click again to put it back).  Points with err σ(T) = 0 cannot be fitted ('–').",
        "Systematic floor: auto (χ²/ndf = 1): a constant f is added in quadrature to every "
        "err σ(T), √(err² + f²), with f chosen so that the fit has χ²/ndf = 1 (f = 0 if it is "
        "already ≤ 1).  f covers what the peak-fit errors miss (alignment, background, line "
        "shape, close neighbours) and is shown with the result; treat it as an upper limit.",
        "Error bars show the combined error used in the fit: statistical ⊕ bin width ⊕ floor "
        "(whichever are ticked); the table lists the statistical and total errors.",
        "Bin-width error: bin = recording frequency / number of frequency bins (type e.g. "
        "10e6 and 2^17 → 76.294 Hz).  The true position is spread evenly across a bin, so the "
        "1σ error is bin/√12 (22.02 Hz for 76.294 Hz bins; ±½ bin is the maximum, not 1σ).  It "
        "is taken as an extra FWHM error and converted like err σ(T): (bin/√12 / centre) · T · "
        "1000 / 2√(2 ln 2), added in quadrature.  With "
        "the auto floor also ticked, the floor is fitted on top of it (what is left over).",
        "Settings are saved per experiment (settings/isocurve.json).",
        "'Save figure' draws the plot and residuals with matplotlib and saves PNG + PDF in "
        "outputs/figures/.",
    ]),
    "params": ("Fit parameters", [
        "centre, amplitude (height above the background), FWHM and – for pseudo-Voigt – η "
        "(Lorentzian fraction) can be edited; the curve is redrawn straight away.",
        "± columns are the 1σ uncertainties from the last Fit (cleared when you change anything "
        "by hand).",
        "Click a 'fix' cell (☐ / ☑) to hold that parameter at its value during the Fit; fixed "
        "parameters show 'fixed' instead of an uncertainty.  b0 / b1 have their own fix boxes.",
        "Tick 'limit columns' to set a min / max for each peak's centre and FWHM (blank = no "
        "extra limit).  Centre limits are drawn as a grey band.",
        "Background = b0 + b1·(f − f_mid), where f_mid is the centre of the fit region.",
    ]),
}


def info_button(key, style=None):
    return html.Button("ⓘ", id={"type": "info-btn", "key": key}, n_clicks=0,
                       className="btn btn-outline-secondary btn-sm", title="Information",
                       style={"padding": "0 7px", "borderRadius": "50%", "fontWeight": "700",
                              "lineHeight": "1.4", **(style or {})})


info_modal = html.Div([
    html.Div([
        html.Div([
            html.Span(id="info-title", style={"fontWeight": "700", "fontSize": "15px"}),
            html.Button("✕", id="info-close", className="btn btn-outline-secondary btn-sm"),
        ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center",
                  "marginBottom": "8px"}),
        html.Div(id="info-body", style={"fontSize": "13px"}),
    ], className="theia-box",
       style={"border": "1px solid #dddddd", "borderRadius": "6px", "background": "white",
              "width": "620px", "maxWidth": "92vw", "padding": "14px 16px",
              "boxShadow": "0 8px 30px rgba(0,0,0,0.3)"}),
], id="info-modal", n_clicks=0,
   style={"display": "none", "position": "fixed", "inset": "0", "zIndex": "1000",
          "background": "rgba(0,0,0,0.35)", "alignItems": "center", "justifyContent": "center"})


PID_FIELDS = [   # key, label
    ("ref_name", "Reference ion"), ("ref_harmonic", "Reference harmonic"),
    ("z_lims", "Z limits  [min, max]"), ("n_lims", "N limits  [min, max]"),
    ("el1", "el1  [A, Z]"), ("el2", "el2  [A, Z]"),
    ("mq_width", "m/q window (± value)"), ("gamma_t", "γ_t"),
    ("harmonics", "Harmonics (comma-separated)"),
    ("corr_a", "a  (f² coefficient)"), ("corr_b", "b  (f coefficient)"),
    ("corr_c", "c  (constant)"),
    ("min_half_life", "Exclude half-life below  [0 = off]"),
]
PID_KEYS = [k for k, _ in PID_FIELDS] + ["min_half_life_unit", "correction_enabled",
                                         "exclude_unknown_hl"]

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
            html.Span("PID Settings – simulated ion lines",
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
            html.Div([_pid_field(k, lab) for k, lab in PID_FIELDS[7:9]] + [
                dcc.Checklist(id="pid-correction_enabled",
                              options=[{"label": " Second order correction", "value": "on"}],
                              style={"fontSize": "13px", "marginBottom": "6px"}),
            ] + [_pid_field(k, lab) for k, lab in PID_FIELDS[9:12]] + [
                html.Label("Exclude half-life below  [0 = off]", style=LABEL_STYLE),
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
            html.Button("Reset to file", id="pid-reset", className="btn btn-outline-secondary btn-sm",
                        title="Use the scan settings stored in the first loaded file",
                        style={"marginRight": "auto"}),
            html.Button("Apply", id="pid-apply", className="btn btn-sm",
                        style={"backgroundColor": ACCENT, "color": "white", "fontWeight": "600",
                               "minWidth": "80px"}),
        ], style={"display": "flex", "alignItems": "center", "marginTop": "6px"}),
    ], className="theia-box",
       style={"border": "1px solid #dddddd", "borderRadius": "6px", "background": "white",
              "width": "640px", "maxWidth": "94vw", "maxHeight": "92vh", "overflowY": "auto",
              "padding": "14px 16px", "boxShadow": "0 8px 30px rgba(0,0,0,0.3)"}),
], id="pid-modal",
   style={"display": "none", "position": "fixed", "inset": "0", "zIndex": "1000",
          "background": "rgba(0,0,0,0.35)", "alignItems": "center", "justifyContent": "center"})


# "Send to isocurve": asks for the ion names / harmonics that are missing and
# confirms replacing points already on the curve
iso_modal = html.Div([
    html.Div([
        html.Div([
            html.Span("Send to isochronicity curve", style={"fontWeight": "700", "fontSize": "15px"}),
            html.Button("✕", id="iso-cancel-x", className="btn btn-outline-secondary btn-sm"),
        ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center",
                  "marginBottom": "8px"}),
        html.Div(id="iso-modal-body", style={"fontSize": "13px"}),
        html.Div([
            html.Button("Cancel", id="iso-cancel", className="btn btn-outline-secondary btn-sm",
                        style={"marginRight": "8px"}),
            html.Button("Send", id="iso-confirm", className="btn btn-sm",
                        style={"backgroundColor": ACCENT, "color": "white", "fontWeight": "600",
                               "minWidth": "80px"}),
        ], style={"display": "flex", "justifyContent": "flex-end", "marginTop": "10px"}),
    ], className="theia-box",
       style={"border": "1px solid #dddddd", "borderRadius": "6px", "background": "white",
              "width": "720px", "maxWidth": "94vw", "maxHeight": "92vh", "overflowY": "auto",
              "padding": "14px 16px", "boxShadow": "0 8px 30px rgba(0,0,0,0.3)"}),
], id="iso-modal",
   style={"display": "none", "position": "fixed", "inset": "0", "zIndex": "1000",
          "background": "rgba(0,0,0,0.35)", "alignItems": "center", "justifyContent": "center"})
ISO_MODAL_SHOWN = {"display": "flex", "position": "fixed", "inset": "0", "zIndex": "1000",
                   "background": "rgba(0,0,0,0.35)", "alignItems": "center",
                   "justifyContent": "center"}
ISO_MODAL_HIDDEN = {**ISO_MODAL_SHOWN, "display": "none"}


SRCH_MODAL_SHOWN = {**ISO_MODAL_SHOWN, "alignItems": "flex-start", "paddingTop": "8vh"}
SRCH_MODAL_HIDDEN = {**SRCH_MODAL_SHOWN, "display": "none"}
SRCH_GRID = {"display": "grid", "gridTemplateColumns": "1.4fr 0.8fr 1.4fr 1fr"}
SRCH_DEFAULT_KHZ = 10.0

# Search ions (as in the spectrogram): an ion's simulated lines on the plotted
# spectrum; a click zooms the plot to f_sim ± width
srch_modal = html.Div([
    html.Div([
        html.Div([
            html.Span("Search ions", style={"fontWeight": "700", "fontSize": "15px"}),
            html.Button("✕", id="srch-close", className="btn btn-outline-secondary btn-sm"),
        ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center",
                  "marginBottom": "8px"}),
        html.Div([
            html.Span("Ion", style={**LABEL_STYLE, "margin": "0 6px 0 0"}),
            dcc.Input(id="srch-ion", type="text", debounce=False, value="",
                      placeholder="e.g. 186Hf or m186Hf72+",
                      className="form-control form-control-sm",
                      style={"width": "220px", "marginRight": "12px"}),
            html.Span("±", style={"marginRight": "4px"},
                      title="Half-width of the window shown around the chosen line"),
            dcc.Input(id="srch-width", type="number", min=0, step="any", debounce=True,
                      value=SRCH_DEFAULT_KHZ, placeholder="10",
                      className="form-control form-control-sm",
                      style={"width": "80px", "marginRight": "4px"}),
            html.Span("kHz", style={"marginRight": "12px", "fontSize": "13px"}),
            html.Span(id="srch-status", style={"fontSize": "13px"}),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "6px"}),
        html.Small("Simulated lines across the whole plotted spectrum (current PID settings). "
                   "Click one to zoom to ± the width above around it (default ±10 kHz).  Grey: "
                   "below the half-life cut.", style={**HELP_STYLE, "marginBottom": "6px"}),
        html.Div(id="srch-list", style={"maxHeight": "55vh", "overflowY": "auto",
                                        "border": "1px solid #dddddd", "borderRadius": "4px"}),
    ], className="theia-box",
       style={"border": "1px solid #dddddd", "borderRadius": "6px", "background": "white",
              "width": "640px", "maxWidth": "94vw", "padding": "14px 16px",
              "boxShadow": "0 8px 30px rgba(0,0,0,0.3)"}),
], id="srch-modal", style=SRCH_MODAL_HIDDEN)


align_fit_tab = html.Div([
    html.H5("Aligning and fitting", style={"fontWeight": "700"}),
    # Data source: the sum of the spectra in the confirmation files, or one
    # spectrum file (e.g. a Tartarus sum, or the file open in the Viewer)
    html.Div([
        html.Span("Data source:", style={**LABEL_STYLE, "margin": "0 8px 0 0"}),
        dcc.RadioItems(id="further-source", value="sum", inline=True,
                       options=[{"label": "Confirmation files (summed)", "value": "sum"},
                                {"label": "Single spectrum file", "value": "file"}],
                       inputStyle={"marginRight": "4px"},
                       labelStyle={"marginRight": "14px", "fontSize": "13px"}),
    ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}),
    html.Div([
        # Numbered list of loaded confirmation files (any number); "+ Add file"
        # opens a picker, choosing a file appends it to the list.
        html.Div(id="further-list"),
        html.Div([
            html.Button("+ Add file", id="btn-further-add", className="btn btn-sm",
                        style={"backgroundColor": ACCENT, "color": "white", "marginRight": "10px"}),
            html.Div([
                dcc.Dropdown(id="further-file", options=[], placeholder="Choose a confirmation file…",
                             style={"fontSize": "13px", "width": "720px"}),
            ], id="further-picker", style={"display": "none"}),
        ], style={"display": "flex", "alignItems": "center", "margin": "8px 0 10px 0"}),
        html.Div(id="further-total", style={"fontSize": "13px", "fontWeight": "600",
                                            "marginBottom": "8px"}),
    ], id="further-sum-src"),
    # Single spectrum file: folder (Browse…) + .npz dropdown, or take the
    # file that is open in the Viewer tab
    html.Div([
        html.Span("Folder:", style={**LABEL_STYLE, "margin": "0 6px 0 0"}),
        dcc.Input(id="fsrc-dir", type="text", debounce=True, value=os.path.join(paths.summed_1d(), ""),
                  className="form-control form-control-sm",
                  style={"width": "460px", "fontSize": "13px", "fontFamily": "monospace"}),
        html.Button("Browse…", id="btn-browse-fsrc-dir", className="btn btn-outline-secondary btn-sm",
                    title="Choose the folder in a folder browser",
                    style={"marginLeft": "6px", "marginRight": "14px", "whiteSpace": "nowrap"}),
        html.Span("File:", style={**LABEL_STYLE, "margin": "0 6px 0 0"}),
        dcc.Dropdown(id="fsrc-name", options=[], placeholder="Choose a .npz spectrum…",
                     clearable=False, style={"fontSize": "13px", "width": "320px",
                                             "marginRight": "14px"}),
        html.Button("Use Viewer spectrum", id="btn-fsrc-viewer",
                    className="btn btn-outline-secondary btn-sm",
                    title="Load the spectrum that is open in the Viewer tab"),
    ], id="further-file-src", style={"display": "none"}),
    dcc.Store(id="fsrc-load-req", data=None),
    dcc.Store(id="further-files", data=[]),
    html.Div([
        # Sum the spectra referenced by the loaded files, optionally aligned.
        html.Div([
            html.Div([
                html.Span("Sum spectra", style={"fontWeight": "700", "marginRight": "4px"}),
                info_button("sum", {"marginRight": "14px"}),
                html.Span("Align on:", style={**LABEL_STYLE, "marginRight": "8px"}),
                dcc.RadioItems(id="sum-align", value="none", inline=True,
                               options=[{"label": "Not aligned", "value": "none"},
                                        {"label": "Strongest peak", "value": "strongest"},
                                        {"label": "Identified peak", "value": "identified"}],
                               inputStyle={"marginRight": "4px"},
                               labelStyle={"marginRight": "14px", "fontSize": "13px"}),
                dcc.Dropdown(id="sum-ion", options=[], placeholder="Ion / harmonic…",
                             clearable=False, style={"fontSize": "13px", "width": "240px",
                                                     "marginRight": "14px"}),
                dcc.Checklist(id="sum-unsure", value=["on"],
                              options=[{"label": " include unsure", "value": "on"}],
                              style={"fontSize": "13px", "marginRight": "14px"}),
                html.Button("Sum spectra", id="btn-sum", className="btn btn-sm",
                            style={"backgroundColor": ACCENT, "color": "white",
                                   "marginRight": "14px"}),
            ], id="sum-only-controls", style=SUM_CONTROLS_SHOWN),
            html.Button("Load spectrum", id="btn-fsrc-load", className="btn btn-sm",
                        title="Show the chosen spectrum file for fitting",
                        style={"backgroundColor": ACCENT, "color": "white", "marginRight": "14px",
                               "display": "none"}),
            html.Button("PID Settings", id="btn-pid", className="btn btn-outline-secondary btn-sm",
                        title="Settings used for the simulated ion lines",
                        style={"marginRight": "8px"}),
            html.Button("Search ions…", id="btn-srch", className="btn btn-outline-secondary btn-sm",
                        title="Find an ion's simulated lines and zoom to a window around one",
                        style={"marginRight": "8px"}),
            # h−1 / h+1 for the ion chosen in Search ions
            html.Span([
                html.Button("h−1", id="btn-harm-dn", className="btn btn-outline-secondary btn-sm",
                            disabled=True),
                html.Span(id="harm-cur", style={"fontSize": "13px", "fontWeight": "600",
                                                "margin": "0 6px"}),
                html.Button("h+1", id="btn-harm-up", className="btn btn-outline-secondary btn-sm",
                            disabled=True),
            ], id="harm-grp", style={"display": "none", "alignItems": "center"}),
        ], style={"display": "flex", "alignItems": "center", "flexWrap": "wrap", "rowGap": "6px",
                  "marginBottom": "6px"}),
        # spinner on the status line only, so fit updates of the plot do not
        # flash a loading overlay over the graph
        dcc.Loading(html.Div(id="sum-status", style={"fontSize": "13px", "margin": "6px 0",
                                                     "minHeight": "18px"}),
                    type="dot", color=ACCENT),
        # description of the simulated lines (kept hidden; updated by the callbacks)
        html.Div(id="sum-sim-info", style={"display": "none"}),
        dcc.Store(id="sum-key", data=None),
        dcc.Store(id="pid-settings", data=None),

        # Two columns: energy calculation (left), peak-fitting controls (right)
        html.Div([
        html.Div([
        # ---- Energy calculation (frequency -> energy from two known peaks) ----
        # outer box always shown (mode button); the content below it only when
        # the mode is on or there is a calibration / picks
        html.Div([
        html.Div([
            html.Button("Energy calculation", id="btn-energy-mode",
                        className="btn btn-outline-secondary btn-sm",
                        title="Calibrate keV per Hz on two known peaks, then get the energy "
                              "difference between any two peaks",
                        style={"marginRight": "12px", "fontWeight": "600"}),
            info_button("energy", {"marginRight": "12px"}),
            html.Span(id="energy-status", style={"fontSize": "13px"}),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}),
        html.Div([
            # calibration peaks: name (typed or suggested), NUBASE mass excess,
            # fitted frequency and FWHM
            html.Div("Calibrant Ions", style={"fontWeight": "700", "fontSize": "14px",
                                              "margin": "2px 0 6px 0"}),
            html.Table([
                html.Thead(html.Tr([html.Th("", style=CALIB_TH)] + [
                    html.Th(h, style=CALIB_TH) for h in
                    ("Name", "Mass excess (keV)", "Frequency (Hz)", "FWHM (Hz)")])),
                html.Tbody([
                    html.Tr([
                        html.Td(f"{i}", style={**CALIB_TD, "fontWeight": "700", "color": ACCENT}),
                        html.Td(html.Div([
                            dcc.Input(id=f"energy-name-{i}", type="text", debounce=True, value="",
                                      placeholder="type a name, e.g. 207Tl81+ or m207Tl",
                                      className="form-control form-control-sm",
                                      style={"width": "240px", "marginRight": "6px"}),
                            html.Button("Suggest ▾", id=f"energy-suggest-btn-{i}",
                                        className="btn btn-sm", title="Simulated ions near this peak",
                                        style={"backgroundColor": ACCENT, "color": "white",
                                               "fontWeight": "600", "whiteSpace": "nowrap"}),
                        ], style={"display": "flex", "alignItems": "center"}), style=CALIB_TD),
                        html.Td(id=f"energy-name-{i}-check", style=CALIB_TD),
                        html.Td(id=f"energy-f-{i}", style={**CALIB_TD, "fontFamily": "monospace"}),
                        html.Td(id=f"energy-fwhm-{i}", style={**CALIB_TD, "fontFamily": "monospace"}),
                    ]) for i in (1, 2)]),
            ], className="theia-calib-table",
               style={"borderCollapse": "collapse", "border": CALIB_BORDER, "marginBottom": "6px"}),
            # suggestion lists, opened by "Suggest"; a click on a row fills the name
            *[html.Div(id=f"energy-sugg-{i}", className="theia-box",
                       style={**SUGG_BOX_STYLE, "display": "none"}) for i in (1, 2)],
            html.Div(id="energy-calib-info", style={"fontSize": "13px", "marginBottom": "4px"}),
            html.Div([
                html.Button("Calibrate", id="btn-energy-calibrate", className="btn btn-sm",
                            style={"backgroundColor": ACCENT, "color": "white", "marginRight": "8px"}),
                html.Button("Reset calibration", id="btn-energy-reset",
                            className="btn btn-outline-secondary btn-sm", style={"marginRight": "8px"}),
                html.Button("Clear pairs", id="btn-energy-clear-pairs",
                            className="btn btn-outline-secondary btn-sm"),
            ], style={"margin": "6px 0"}),
            dash_table.DataTable(
                id="energy-table", data=[],
                columns=[{"name": "pair", "id": "pair"}, {"name": "f_A (Hz)", "id": "f_A"},
                         {"name": "f_B (Hz)", "id": "f_B"}, {"name": "f_B − f_A (Hz)", "id": "df"},
                         {"name": "E(B) − E(A) (keV)", "id": "de"},
                         {"name": "frequencies from", "id": "src"}],
                style_header=TABLE_HEADER, style_cell=TABLE_CELL, style_data_conditional=TABLE_ODD,
                style_cell_conditional=[{"if": {"column_id": "src"}, "textAlign": "left",
                                         "fontFamily": "inherit", "color": "#666"}],
                style_table={"overflowX": "auto"}),
        ], id="energy-panel", style={"display": "none"}),
        ], className="theia-box",
           style={**BOX_STYLE, "padding": "8px 10px", "margin": "4px 0 8px 0",
                  "width": "fit-content", "maxWidth": "100%"}),
        ], style={"flex": "0 0 auto", "maxWidth": "100%"}),
        # ---- Peak fitting (on the summed spectrum) ----------------------
        # controls on top, the parameter table underneath
        html.Div([
        html.Div([
            html.Div([
                html.Button("Peak fitting", id="btn-fit-mode",
                            className="btn btn-outline-secondary btn-sm",
                            title="Turn on, then click peaks in the plot to add them to the fit",
                            style={"marginRight": "12px", "fontWeight": "600"}),
                info_button("fit"),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}),
            html.Div([
                html.Span("Shape for new peaks:", style={**LABEL_STYLE, "margin": "0 6px 0 0",
                                                         "minWidth": "135px"}),
                dcc.Dropdown(id="fit-shape", value="gauss", clearable=False,
                             options=[{"label": "Gaussian", "value": "gauss"},
                                      {"label": "Lorentzian", "value": "lorentz"},
                                      {"label": "Pseudo-Voigt", "value": "pvoigt"}],
                             style={"width": "160px", "fontSize": "13px"}),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}),
            html.Div([
                html.Span("Background:", style={**LABEL_STYLE, "margin": "0 6px 0 0",
                                                "minWidth": "135px"}),
                dcc.RadioItems(id="fit-bg", value="const", inline=True,
                               options=[{"label": "none", "value": "none"},
                                        {"label": "constant", "value": "const"},
                                        {"label": "linear", "value": "linear"}],
                               inputStyle={"marginRight": "4px"},
                               labelStyle={"marginRight": "10px", "fontSize": "13px"}),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}),
            html.Div([
                html.Span("Fit region (Hz):", style={**LABEL_STYLE, "margin": "0 6px 0 0",
                                                     "minWidth": "135px"}),
                dcc.Input(id="fit-region-lo", type="number", step="any", debounce=True,
                          className="form-control form-control-sm",
                          style={"width": "170px", "marginRight": "4px"}),
                html.Span("–", style={"margin": "0 4px"}),
                dcc.Input(id="fit-region-hi", type="number", step="any", debounce=True,
                          className="form-control form-control-sm",
                          style={"width": "170px", "marginRight": "8px"}),
                html.Button("Region = view", id="btn-fit-region-view",
                            className="btn btn-outline-secondary btn-sm",
                            title="Set the fit region to the current zoom range (you can also "
                                  "drag the two orange lines in the plot)"),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}),
            html.Div([
                html.Button("Fit", id="btn-fit-run", className="btn btn-sm",
                            title="Least-squares fit over the fit region (orange lines), "
                                  "starting from the values in the table",
                            style={"backgroundColor": ACCENT, "color": "white",
                                   "marginRight": "8px", "minWidth": "70px"}),
                html.Button("Clear peaks", id="btn-fit-clear",
                            className="btn btn-outline-secondary btn-sm",
                            style={"marginRight": "8px"}),
                html.Button("Send to isocurve", id="btn-iso-send",
                            className="btn btn-outline-secondary btn-sm",
                            title="Send the fitted peaks (each needs an ion name and harmonic) "
                                  "to the Isochronicity Curve tab"),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "6px"}),
            html.Div(id="fit-status", style={"fontSize": "12px"}),
            html.Div(id="iso-send-status", style={"fontSize": "12px", "marginTop": "2px"}),
        ]),
            html.Div([
                html.Div(["Fit parameters", info_button("params", {"marginLeft": "6px"}),
                          dcc.Checklist(id="fit-show-limits", value=[],
                                        options=[{"label": "limit columns", "value": "on"}],
                                        inputStyle={"marginRight": "4px"},
                                        labelStyle={"fontSize": "12px", "fontWeight": "400"},
                                        style={"marginLeft": "16px"})],
                         style={"fontWeight": "700", "fontSize": "13px", "display": "flex",
                                "alignItems": "center", "margin": "6px 0 4px 0"}),
                dash_table.DataTable(
                    id="fit-table", data=[], editable=True, row_deletable=True,
                    columns=fit_columns(False, False),
                    dropdown={"shape": {"clearable": False, "options": [
                        {"label": v, "value": k} for k, v in SHAPE_LABELS.items()]}},
                    # compact, so the table fits beside the energy box
                    style_header={**TABLE_HEADER, "whiteSpace": "normal", "height": "auto",
                                  "fontSize": "10px", "padding": "2px 3px"},
                    style_cell={**TABLE_CELL, "padding": "2px 3px", "minWidth": "0",
                                "fontSize": "11px"},
                    style_data_conditional=TABLE_ODD + [
                        {"if": {"column_editable": False}, "color": "#666"},
                        # fix toggles: clickable, fixed values tinted
                        *[{"if": {"column_id": f"fix_{k}"}, "cursor": "pointer",
                           "textAlign": "center", "color": "#222", "fontSize": "13px"}
                          for k in FIX_PARAMS],
                        *[{"if": {"column_id": k, "filter_query": f'{{fix_{k}}} = "{FIX_ON}"'},
                           "backgroundColor": ACCENT_PALE, "fontWeight": "600"}
                          for k in FIX_PARAMS],
                        *[{"if": {"column_id": k}, "backgroundColor": "#f9f9f9"}
                          for k in LIMIT_KEYS],
                        # eta only applies to pseudo-Voigt peaks
                        {"if": {"column_id": "eta", "filter_query": '{shape} != "pvoigt"'},
                         "color": "transparent", "backgroundColor": "#f2f2f2"}],
                    style_cell_conditional=[{"if": {"column_id": "shape"}, "textAlign": "left",
                                             "minWidth": "88px", "width": "88px"},
                                            {"if": {"column_id": "ion"}, "textAlign": "left",
                                             "minWidth": "80px", "fontWeight": "600"},
                                            {"if": {"column_id": "harmonic"}, "minWidth": "34px"}],
                    # Bootstrap's CSS hides the DataTable dropdown menu
                    # (.Select-menu-outer gets display:none), so force it on; and
                    # let the menu hang below the table instead of being clipped
                    css=[{"selector": ".Select-menu-outer",
                          "rule": "display: block !important; background: white; z-index: 50; "
                                  "border: 1px solid #dddddd; box-shadow: 0 4px 10px rgba(0,0,0,0.15)"},
                         {"selector": ".Select-option", "rule": "color: #222 !important; padding: 4px 8px"},
                         {"selector": ".Select-option.is-focused", "rule": f"background: {ACCENT_PALE}"},
                         {"selector": ".dash-spreadsheet-container .dash-spreadsheet-inner",
                          "rule": "overflow: visible !important"}],
                    style_table={"overflow": "visible"}),
                html.Div([
                    html.Span("Background  b0", style={**LABEL_STYLE, "margin": "0 6px 0 0"}),
                    dcc.Input(id="fit-b0", type="number", step="any", debounce=True,
                              className="form-control form-control-sm",
                              style={"width": "120px", "marginRight": "4px"}),
                    html.Span(id="fit-b0-unc", style={"fontSize": "12px", "marginRight": "6px",
                                                      "fontFamily": "monospace"}),
                    html.Span("b1 (per Hz)", style={**LABEL_STYLE, "margin": "0 6px 0 0"}),
                    dcc.Input(id="fit-b1", type="number", step="any", debounce=True,
                              className="form-control form-control-sm",
                              style={"width": "120px", "marginRight": "4px"}),
                    html.Span(id="fit-b1-unc", style={"fontSize": "12px", "marginRight": "16px",
                                                      "fontFamily": "monospace"}),
                    dcc.Checklist(id="fit-bg-fix", value=[], inline=True,
                                  options=[{"label": "fix b0", "value": "b0"},
                                           {"label": "fix b1", "value": "b1"}],
                                  inputStyle={"marginRight": "4px"},
                                  labelStyle={"marginRight": "10px", "fontSize": "12px"}),
                ], style={"display": "flex", "alignItems": "center", "flexWrap": "wrap",
                          "marginTop": "6px"}),
            ], id="fit-panel", style=FIT_PANEL_HIDDEN),
        ], id="fit-bar", className="theia-box",
           style={**BOX_STYLE, "padding": "8px 10px", "margin": "4px 0 8px 0", "flex": "1 1 0",
                  # wide enough for the parameter table; if there is no room
                  # beside the energy box the whole box moves underneath it
                  "minWidth": "640px",
                  # controls on top, the parameter table underneath
                  "display": "flex", "flexDirection": "column", "rowGap": "8px"}),
        ], style={"display": "flex", "alignItems": "flex-start", "columnGap": "16px",
                  "flexWrap": "wrap"}),
        dcc.Store(id="energy-mode", data=False),
        dcc.Store(id="energy-click", data=None),
        dcc.Store(id="energy-state", data=None),
        dcc.Store(id="fit-mode", data=False),
        dcc.Store(id="fit-click", data=None),
        dcc.Store(id="fit-drag", data=None),
        dcc.Store(id="fit-run", data=None),
        dcc.Store(id="fit-region-set", data=None),
        dcc.Store(id="fit-state", data=None),
        dcc.Store(id="iso-names-apply", data=None),
        dcc.Store(id="srch-hits", data=[]),
        dcc.Store(id="srch-current", data=None),
        dcc.Store(id="srch-zoom", data=None),
        html.Div(id="_srch-zoom-hook", style={"display": "none"}),
        html.Div(id="_fit-mode-hook", style={"display": "none"}),
        html.Div(id="_sum-sims-hook", style={"display": "none"}),

        dcc.Graph(id="sum-graph", figure=empty_fig("Load files and click “Sum spectra”"),
                  # doubleClick off: a touchpad double-tap (to drag/pan)
                  # otherwise triggers Plotly's reset/autoscale, which
                  # redraws all ~131k points and stalls the page.
                  config={"scrollZoom": True, "displaylogo": False, "doubleClick": False},
                  style={"height": "560px"}),
        dcc.Graph(id="fit-resid", figure=empty_fig(),
                  config={"scrollZoom": True, "displaylogo": False, "doubleClick": False},
                  style={"height": "200px", "display": "none"}),
    ], id="further-content", className="theia-box", style={**BOX_STYLE, "padding": "8px 10px"}),
], id="further-page-align", style={"padding": "14px 18px", "overflowY": "auto",
                                   "height": "calc(100vh - 82px)"})

# Points sent from the peak fitting ("Send to isocurve"), one per ion and
# harmonic, kept in outputs/isocurve/isocurve_points.csv
ISO_TABLE_COLUMNS = [
    ("ion", "ion"), ("harmonic", "h"), ("A", "A"), ("Z", "Z"), ("q", "q"),
    ("centre_hz", "centre (Hz)"), ("d_centre_hz", "± centre"),
    ("f_rev_hz", "f_rev (Hz)"), ("d_f_rev_hz", "± f_rev"),
    ("fwhm_hz", "FWHM (Hz)"), ("d_fwhm_hz", "± FWHM"),
    ("amp", "amplitude"), ("d_amp", "± amp"), ("area", "area"), ("d_area", "± area"),
    ("shape", "shape"), ("eta", "η"), ("d_eta", "± η"), ("fixed", "fixed"),
    ("chi2_ndf", "χ²/ndf"), ("source", "data source"), ("sent", "sent"),
]

# derived values for the curve (calculated from the points, not saved)
ISO_CURVE_COLUMNS = [
    ("include", "in fit"), ("ion", "ion"), ("harmonic", "h"), ("T_ns", "T (ns)"),
    ("fwhm_T_ps", "FWHM(T) (ps)"), ("sigma_T_ps", "σ(T) (ps)"),
    ("d_sigma_T_ps", "err σ(T) (ps)"),
]

# curve-fit settings (saved per experiment in settings/isocurve.json)
ISO_FIT_DEFAULTS = {"circumference_m": 108.36, "c_m_s": 299792458.0,
                    "gt_min": 1.0, "gt_max": 3.0, "sp_min_pct": 0.0, "sp_max_pct": 0.2,
                    "ssys_min_ps": 0.0, "ssys_max_ps": 0.4}
ISO_BIN_DEFAULTS = {"rec_freq": "10e6", "n_bins": "2^17"}   # bin-width error inputs
ISO_SETTING_INPUTS = [   # first key, second key (range) or None, label, width
    ("circumference_m", None, "Circumference (m)", "100px"),
    ("c_m_s", None, "c (m/s)", "170px"),
    ("gt_min", "gt_max", "γ_t  min – max", "70px"),
    ("sp_min_pct", "sp_max_pct", "σp/p (%)  min – max", "70px"),
    ("ssys_min_ps", "ssys_max_ps", "σ_sys (ps)  min – max", "70px"),
]
ISO_SETTING_KEYS = [k for k, k2, _, _ in ISO_SETTING_INPUTS for k in (k, k2) if k]

isochronicity_tab = html.Div([
    html.Div([
        html.H5("Isochronicity Curve", style={"fontWeight": "700", "margin": "0 8px 0 0"}),
        info_button("iso"),
    ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}),
    html.Div([
        html.Div(id="iso-info", style={"fontSize": "13px", "marginBottom": "6px"}),
        dash_table.DataTable(
            id="iso-table", data=[], row_deletable=True, editable=False,
            sort_action="native",
            columns=[{"name": n, "id": c} for c, n in ISO_TABLE_COLUMNS],
            style_header={**TABLE_HEADER, "whiteSpace": "normal", "height": "auto"},
            style_cell={**TABLE_CELL, "fontSize": "12px"},
            style_data_conditional=TABLE_ODD,
            style_cell_conditional=[
                {"if": {"column_id": c}, "textAlign": "left", "fontFamily": "inherit"}
                for c in ("ion", "shape", "fixed", "source", "sent")] + [
                {"if": {"column_id": "ion"}, "fontWeight": "600"},
                {"if": {"column_id": "source"}, "color": "#666", "maxWidth": "360px",
                 "overflow": "hidden", "textOverflow": "ellipsis"}],
            tooltip_duration=None,
            style_table={"overflowX": "auto"}),
    ], className="theia-box", style={**BOX_STYLE, "padding": "8px 10px"}),
    # values for the isochronicity curve, calculated from the points above
    html.Div([
        html.Div("Isochronicity curve parameters", style={"fontWeight": "700", "fontSize": "14px",
                                                          "marginBottom": "2px"}),
        html.Small("T = 10⁹ · h / centre  ·  FWHM(T) = (FWHM / centre) · T · 1000  ·  "
                   "σ(T) = FWHM(T) / 2√(2 ln 2)  ·  err σ(T) = (± FWHM / FWHM) · FWHM(T) / "
                   "2√(2 ln 2)   (2√(2 ln 2) ≈ 2.3548)",
                   style={**HELP_STYLE, "display": "block", "marginBottom": "6px"}),
        dash_table.DataTable(
            id="iso-curve-table", data=[], sort_action="native",
            columns=[{"name": n, "id": c} for c, n in ISO_CURVE_COLUMNS],
            style_header={**TABLE_HEADER, "whiteSpace": "normal", "height": "auto"},
            style_cell={**TABLE_CELL, "fontSize": "12px"},
            style_data_conditional=TABLE_ODD + [
                {"if": {"column_id": "include"}, "cursor": "pointer", "textAlign": "center",
                 "fontSize": "13px", "color": "#222"},
                {"if": {"filter_query": f'{{include}} = "{FIX_OFF}"'}, "color": "#aaa"},
                {"if": {"filter_query": '{include} = "–"'}, "color": "#aaa"}],
            style_cell_conditional=[{"if": {"column_id": "ion"}, "textAlign": "left",
                                     "fontFamily": "inherit", "fontWeight": "600"}],
            style_table={"overflowX": "auto", "width": "fit-content", "maxWidth": "100%"}),
    ], className="theia-box", style={**BOX_STYLE, "padding": "8px 10px", "marginTop": "10px"}),
    # fit of the curve  σ_T(T) = sqrt( ((1/γ² − 1/γ_t²) · σp/p · T)² + σ_sys² )
    html.Div([
        html.Div([
            html.Span("Isochronicity curve fit", style={"fontWeight": "700", "fontSize": "14px",
                                                        "marginRight": "8px"}),
            info_button("isofit", {"marginRight": "14px"}),
            html.Button("Save figure", id="btn-iso-save-fig",
                        className="btn btn-outline-secondary btn-sm",
                        title="Save the plot (matplotlib) as PNG and PDF in outputs/figures/",
                        style={"marginRight": "10px"}),
            html.Span(id="iso-save-status", style={"fontSize": "12px"}),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "6px"}),
        html.Div([
            *[html.Div([
                html.Label(lab, style=LABEL_STYLE),
                html.Div([
                    dcc.Input(id=f"iso-set-{k}", type="number", step="any", debounce=True,
                              className="form-control form-control-sm",
                              style={"width": w, "fontSize": "13px"}),
                    *([html.Span("–", style={"margin": "0 4px"}),
                       dcc.Input(id=f"iso-set-{k2}", type="number", step="any", debounce=True,
                                 className="form-control form-control-sm",
                                 style={"width": w, "fontSize": "13px"})] if k2 else []),
                ], style={"display": "flex", "alignItems": "center"}),
            ], style={"marginRight": "18px"}) for k, k2, lab, w in ISO_SETTING_INPUTS],
            dcc.Checklist(id="iso-set-floor", value=[],
                          options=[{"label": " Systematic floor: auto (χ²/ndf = 1)",
                                    "value": "auto"}],
                          inputStyle={"marginRight": "4px"},
                          labelStyle={"fontSize": "13px"},
                          style={"paddingBottom": "6px"}),
        ], style={"display": "flex", "alignItems": "flex-end", "flexWrap": "wrap",
                  "rowGap": "6px", "marginBottom": "8px"}),
        html.Div([
            dcc.Checklist(id="iso-set-binerr", value=[],
                          options=[{"label": " Bin-width error: bin/√12 (1σ) on every FWHM",
                                    "value": "on"}],
                          inputStyle={"marginRight": "4px"}, labelStyle={"fontSize": "13px"},
                          style={"marginRight": "14px"}),
            html.Span("Recording frequency (Hz)", style={**LABEL_STYLE, "margin": "0 6px 0 0"}),
            dcc.Input(id="iso-set-recfreq", type="text", debounce=True,
                      value=ISO_BIN_DEFAULTS["rec_freq"], placeholder="10e6",
                      className="form-control form-control-sm",
                      style={"width": "100px", "marginRight": "12px", "fontSize": "13px"}),
            html.Span("Frequency bins", style={**LABEL_STYLE, "margin": "0 6px 0 0"}),
            dcc.Input(id="iso-set-nbins", type="text", debounce=True,
                      value=ISO_BIN_DEFAULTS["n_bins"], placeholder="2^17",
                      className="form-control form-control-sm",
                      style={"width": "90px", "marginRight": "12px", "fontSize": "13px"}),
            html.Span(id="iso-bin-info", style={"fontSize": "13px", "fontFamily": "monospace"}),
        ], style={"display": "flex", "alignItems": "center", "flexWrap": "wrap",
                  "rowGap": "6px", "marginBottom": "8px"}),
        html.Div(id="iso-fit-result", style={"fontSize": "13px", "marginBottom": "6px"}),
        dcc.Graph(id="iso-graph", figure=empty_fig("No points yet"),
                  config={"scrollZoom": True, "displaylogo": False},
                  style={"height": "750px"}),
        dcc.Graph(id="iso-resid", figure=empty_fig(),
                  config={"scrollZoom": True, "displaylogo": False},
                  style={"height": "340px"}),
    ], className="theia-box", style={**BOX_STYLE, "padding": "8px 10px", "marginTop": "10px"}),
    dcc.Store(id="iso-version", data=0),
    dcc.Store(id="iso-points-ts", data=None),
    html.Div(id="_iso-label-hook", style={"display": "none"}),
    html.Div(id="_iso-resid-label-hook", style={"display": "none"}),
    dcc.Store(id="iso-settings-init", data=0),
], id="further-page-iso", style={"padding": "14px 18px", "overflowY": "auto",
                                 "height": "calc(100vh - 82px)", "display": "none"})

# The sub-tab bar only switches which page is visible: both pages stay
# mounted, so plots keep their zoom and browser-side listeners.
FURTHER_PAGES = {"align": "further-page-align", "iso": "further-page-iso"}
SUBTAB_STYLE = {"padding": "6px 14px", "fontWeight": "600", "fontSize": "13px"}
SUBTAB_SEL = {**SUBTAB_STYLE, "borderTop": f"3px solid {ACCENT}"}
further_tab = html.Div([
    dcc.Tabs(id="further-subtabs", value="align", children=[
        dcc.Tab(label="Aligning and fitting", value="align", style=SUBTAB_STYLE,
                selected_style=SUBTAB_SEL),
        dcc.Tab(label="Isochronicity Curve", value="iso", style=SUBTAB_STYLE,
                selected_style=SUBTAB_SEL),
    ], style={"height": "40px"}),
    align_fit_tab,
    isochronicity_tab,
])


# ===========================================================================
# Viewer tab (formerly viewer.py): one spectrum with the simulated ion
# frequencies overlaid.  It has its own settings, saved to viewer_config.json.
# ===========================================================================

def v_get_chargestate(electrons, z_lims, n_lims, el1, el2):
    d = df_nubase(electrons=electrons)
    d = rect_restrict(df=d, z_lims=z_lims, n_lims=n_lims)
    d = diag_restrict(df=d, el1=el1, el2=el2)
    return index_by_name(df=d, electrons=electrons)


def v_compute_frequencies_multi(df, mq_col, ref_ion_name, ref_freq, gamma_t, harmonics,
                                correction_enabled=False, corr_a=0.0, corr_b=0.0, corr_c=0.0):
    result = df.sort_values(mq_col).copy()
    for h in harmonics:
        ref_mq = result.loc[ref_ion_name, mq_col]

        def _direct(row_mq, _h=h, _ref_mq=ref_mq, _corr=correction_enabled):
            if row_mq == _ref_mq:
                f_true = ref_freq
            else:
                f_true = ref_freq + delta_f(ref_freq, _ref_mq, row_mq - _ref_mq, gamma_t)
            # correction applied to the revolution frequency before harmonic scaling
            if _corr:
                f_true = apply_correction(f_true, corr_a, corr_b, corr_c)
            return f_true * _h

        result[f"frequency_direct_{h}"] = result[mq_col].apply(_direct)
        # same frequency without the 2nd-order correction (for the tooltip)
        result[f"frequency_uncorr_{h}"] = result[mq_col].apply(_direct, _corr=False)
    return result


def v_parse_highlight_ions(s):
    """Comma-separated ion names -> list of stripped, non-empty strings."""
    if not s or not s.strip():
        return []
    return [tok.strip() for tok in s.split(",") if tok.strip()]


def v_ion_matches_highlight(ion_name, highlight_list):
    """An entry with '+' must match the full ion name; without it, it matches
    every charge state of that isotope ('204Hg' matches '204Hg79+')."""
    base = ion_base_name(ion_name)
    for h in highlight_list:
        if ("+" in h and ion_name == h) or ("+" not in h and base == h):
            return True
    return False


def v_build_figure(file_path, spectrum_name, ref_name, ref_frequency, ref_harmonic,
                   z_lims, n_lims, el1, el2, mq_width, gamma_t, harmonics,
                   correction_enabled=False, corr_a=0.0, corr_b=0.0, corr_c=0.0,
                   highlight_ions=None):
    """Run the full computation and return (figure, error text)."""
    import plotly.express as px
    highlight_ions = highlight_ions or []
    try:
        full_database = pd.concat([v_get_chargestate(e, z_lims, n_lims, el1, el2)
                                   for e in (0, 1, 2, 3)])
        sorted_db = full_database.sort_values("m/q", ascending=False)

        mq_ref = sorted_db.loc[ref_name, "m/q"]
        db = sorted_db[(sorted_db["m/q"] >= mq_ref - mq_width) &
                       (sorted_db["m/q"] <= mq_ref + mq_width)]

        ref_freq = ref_frequency / ref_harmonic
        db = v_compute_frequencies_multi(db, "m/q", ref_name, ref_freq, gamma_t, harmonics,
                                         correction_enabled=correction_enabled,
                                         corr_a=corr_a, corr_b=corr_b, corr_c=corr_c)

        data = unpack_npz(file_path.rstrip("/") + "/" + spectrum_name)
        ff, pp = data["arr_0"], data["arr_1"]

        n_points = 20
        fig = go.Figure()
        # "hvh" centres each step on its frequency value
        fig.add_trace(go.Scattergl(x=ff, y=pp, mode="lines", name="Spectrum", hoverinfo="skip",
                                   line=dict(shape="hvh", color="black", width=1)))

        ymin, ymax = np.nanmin(pp), np.nanmax(pp)
        pad = (ymax - ymin) * 0.05
        y_range = np.linspace(ymin - pad, ymax + pad, n_points)
        freq_min, freq_max = np.nanmin(ff), np.nanmax(ff)

        colors = px.colors.qualitative.Plotly
        freq_cols = [f"frequency_direct_{h}" for h in harmonics]
        HIGHLIGHT_COLOR, HIGHLIGHT_WIDTH = "#FF8C00", 3
        multi_harmonic = len(harmonics) > 1
        highlighted_annotations = []

        for h, col in zip(harmonics, freq_cols):
            color = colors[freq_cols.index(col) % len(colors)]
            xs_norm, ys_norm, hover_norm = [], [], []
            xs_hi, ys_hi, hover_hi = [], [], []
            for ion_name, row in db.iterrows():
                f = row[col]
                if pd.isna(f) or f < freq_min or f > freq_max:
                    continue
                label = build_hover_label(str(ion_name), row) + f"<br>Harmonic: {h}"
                if correction_enabled:
                    label += (f"<br>f_sim (uncorrected) = {row[f'frequency_uncorr_{h}']:.1f} Hz"
                              f"<br>f_sim (corrected) = {f:.1f} Hz")
                else:
                    label += f"<br>f_sim = {f:.1f} Hz"
                seg_x = [f] * n_points + [None]
                seg_y = list(y_range) + [None]
                seg_h = [label] * n_points + [None]
                if v_ion_matches_highlight(str(ion_name), highlight_ions):
                    xs_hi += seg_x; ys_hi += seg_y; hover_hi += seg_h
                    highlighted_annotations.append((str(ion_name), f, h))
                else:
                    xs_norm += seg_x; ys_norm += seg_y; hover_norm += seg_h
            if xs_norm:
                fig.add_trace(go.Scattergl(x=xs_norm, y=ys_norm, mode="lines",
                                           line=dict(dash="dash", color=color, width=1), name=col,
                                           hoverinfo="text", hovertext=hover_norm,
                                           connectgaps=False))
            if xs_hi:
                fig.add_trace(go.Scattergl(x=xs_hi, y=ys_hi, mode="lines",
                                           line=dict(dash="solid", color=HIGHLIGHT_COLOR,
                                                     width=HIGHLIGHT_WIDTH),
                                           name=f"{col} ★", hoverinfo="text", hovertext=hover_hi,
                                           connectgaps=False))

        # labelled arrows for highlighted ions; three offsets so labels at
        # similar frequencies don't sit exactly on top of each other
        y_tip = ymax + pad
        AY_OFFSETS = [-38, -58, -78]
        for idx, (ion_name, freq, h) in enumerate(highlighted_annotations):
            text = ion_name if not multi_harmonic else \
                f"{ion_name}<br><span style='font-size:10px'>h={h}</span>"
            fig.add_annotation(x=freq, y=y_tip, text=text, showarrow=True, arrowhead=2,
                               arrowsize=1, arrowwidth=1.5, arrowcolor=HIGHLIGHT_COLOR, ax=0,
                               ay=AY_OFFSETS[idx % len(AY_OFFSETS)],
                               font=dict(size=11, color=HIGHLIGHT_COLOR),
                               bgcolor="rgba(255,255,255,0.88)", bordercolor=HIGHLIGHT_COLOR,
                               borderwidth=1, borderpad=3, align="center")

        fig.update_layout(title="Calculated Frequencies overlayed with data",
                          xaxis_title="Frequency (Hz)", yaxis_title="Schottky Power (a.u.)",
                          hovermode="x", dragmode="pan", template="plotly_white",
                          font=dict(size=18), margin=dict(l=60, r=20, t=50, b=50))
        return fig, ""
    except Exception as exc:
        fig = go.Figure()
        fig.update_layout(title=f"Error: {exc}", template="plotly_white",
                          annotations=[dict(text=str(exc), x=0.5, y=0.5, showarrow=False,
                                            font=dict(size=16, color="red"),
                                            xref="paper", yref="paper")])
        return fig, str(exc)


VIEWER_DEFAULTS = dict(
    file_path          = os.path.join(paths.spectra_1d(), ""),
    spectrum_name      = "1129.1.noBKG.npz",
    ref_name           = "",           # shared physics keys: filled from physics.json
    ref_frequency      = 0,
    ref_harmonic       = 0,
    z_lims             = "",
    n_lims             = "",
    el1                = "",
    el2                = "",
    mq_width           = 0.0,
    gamma_t            = 0.0,
    harmonics          = "",
    correction_enabled = [],
    corr_a             = 0.0,
    corr_b             = 0.0,
    corr_c             = 0.0,
    highlight_ions     = "",
)
VIEWER_KEYS = list(VIEWER_DEFAULTS)


def load_viewer_config():
    """Viewer settings: VIEWER_DEFAULTS, shared physics, then the viewer's overrides."""
    return physics.with_shared(VIEWER_DEFAULTS, _load_json(VIEWER_CONFIG_PATH))


def save_viewer_config(cfg):
    try:
        tmp = VIEWER_CONFIG_PATH + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(physics.strip_shared(cfg), fh, indent=2)
        os.replace(tmp, VIEWER_CONFIG_PATH)
    except Exception:
        pass


VCFG = load_viewer_config()


def v_input(label, key, number=False, step=None, help=None, placeholder=None):
    kw = dict(id=f"v-{key}", value=str(VCFG[key]), debounce=True,
              className="form-control form-control-sm", style={"width": "100%", "fontSize": "13px"})
    if number:
        kw["type"] = "number"
        if step is not None:
            kw["step"] = step
    if placeholder:
        kw["placeholder"] = placeholder
    children = [html.Label(label, style=LABEL_STYLE), dcc.Input(**kw)]
    if help:
        children.append(html.Small(help, style=HELP_STYLE))
    return html.Div(children, style=INPUT_STYLE)


viewer_sidebar = html.Div([
    html.H5("Viewer settings", style={"fontWeight": "700", "marginBottom": "4px"}),
    *section("Spectrum"),
    with_browse(v_input("File path", "file_path"), "btn-browse-v-file_path"),
    html.Div([
        html.Label("Spectrum", style=LABEL_STYLE),
        dcc.Dropdown(id="v-spectrum_name", value=str(VCFG["spectrum_name"]), clearable=False,
                     options=[{"label": n, "value": n} for n in list_npz(VCFG["file_path"])],
                     placeholder="No .npz files in this folder",
                     style={"fontSize": "13px"}),
    ], style=INPUT_STYLE),
    *section("Reference ion"),
    v_input("Ion name", "ref_name"),
    v_input("Frequency (Hz)", "ref_frequency", True, 1),
    v_input("Harmonic", "ref_harmonic", True, 1),
    *section("Ion selection"),
    v_input("Z limits  [min, max]", "z_lims"),
    v_input("N limits  [min, max]", "n_lims"),
    v_input("el1  [A, Z]", "el1"),
    v_input("el2  [A, Z]", "el2"),
    v_input("m/q window  (± value)", "mq_width", True, 0.001),
    *section("Ring / harmonics"),
    v_input("γ_t", "gamma_t", True, 0.001),
    v_input("Harmonics (comma-separated)", "harmonics"),
    *section("Frequency correction"),
    dcc.Checklist(id="v-correction_enabled",
                  options=[{"label": " Second order correction", "value": "on"}],
                  value=VCFG["correction_enabled"], style={"fontSize": "13px", "marginBottom": "10px"}),
    html.Div([
        v_input("a  (f² coefficient)", "corr_a", True, "any"),
        v_input("b  (f coefficient)", "corr_b", True, "any"),
        v_input("c  (constant)", "corr_c", True, "any"),
    ], id="v-correction-inputs",
       style={"display": "block" if VCFG["correction_enabled"] else "none"}),
    *section("Highlight ions"),
    v_input("Ion names (comma-separated)", "highlight_ions", placeholder="e.g. 204Hg, 208Tl79+",
            help="Omit charge state to highlight all charge states"),
    html.Hr(style={"margin": "10px 0"}),
    html.Div(id="v-error-box", style={"color": "#c0392b", "fontSize": "12px",
                                      "whiteSpace": "pre-wrap"}),
    html.Div(style={"height": "30px"}),
], className="theia-sidebar",
   style={"width": "300px", "minWidth": "300px", "padding": "14px 18px", "background": "#f9f9f9",
          "overflowY": "auto", "borderRight": "1px solid #dddddd", "height": "calc(100vh - 42px)"})

viewer_tab = html.Div([
    viewer_sidebar,
    html.Div([
        dcc.Graph(id="viewer-graph", style={"height": "calc(100vh - 42px)"},
                  config={"scrollZoom": True, "displayModeBar": True, "displaylogo": False}),
    ], style={"flex": "1", "minWidth": "0"}),
    html.Div(id="_viewer-saved", style={"display": "none"}),
], style={"display": "flex"})


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True,
                title="Theia")
app.title = "Theia"

# Results tables: move the page buttons from below the table to the top right,
# level with the Export button (both 30 px tall), so nothing is left below it.
_TABLE_CSS = """
#summary-table, #hits-table { position: relative; }
#summary-table .previous-next-container, #hits-table .previous-next-container {
    position: absolute; top: 0; right: 0; margin: 0; padding: 0;
    height: 30px; display: flex; align-items: center;
}
#summary-table .previous-next-container button, #hits-table .previous-next-container button {
    height: 30px; padding-top: 0; padding-bottom: 0;
}
"""

# Dark mode: a "dark" class on <body> (set by a clientside callback) switches
# these rules on.  Plots are darkened with an invert + hue-rotate filter, which
# turns the white background dark and black traces light while keeping the
# orange accents orange, so the figure code doesn't need a second colour scheme.
_FB_CSS = """
.theia-fb-row:hover { background: #fff4ec; }
body.dark .theia-fb-row:hover { background: #33220f !important; }
"""

_CALIB_CSS = """
body.dark .theia-calib-table th { background: #1c1c1c !important; color: #d0d0d0 !important; }
body.dark .theia-calib-table th, body.dark .theia-calib-table td,
body.dark .theia-calib-table { border-color: #6a6a6a !important; }
"""

_FIT_CSS = """
#sum-graph.fit-mode .nsewdrag { cursor: crosshair !important; }
#sum-graph.fit-grab .nsewdrag { cursor: ew-resize !important; }
#sum-graph.fit-grab-move .nsewdrag { cursor: move !important; }
#sum-graph.fit-grab-region .nsewdrag { cursor: col-resize !important; }
"""

_DARK_CSS = """
/* High-contrast dark theme: near-black backgrounds, near-white text */
body.dark { background: #0a0a0a !important; color: #f5f5f5 !important; }
body.dark .theia-sidebar { background: #141414 !important; border-color: #4a4a4a !important; }
body.dark .theia-box { background: #0a0a0a !important; border-color: #4a4a4a !important; }
body.dark hr { border-color: #6a6a6a !important; opacity: 1 !important; }
body.dark label, body.dark h5, body.dark h6, body.dark summary, body.dark span { color: #f5f5f5 !important; }
/* help text and sidebar section titles have grey colours set inline */
body.dark small { color: #c4c4c4 !important; }
body.dark .theia-sidebar p { color: #d6d6d6 !important; }
body.dark .form-control, body.dark input[type=number], body.dark input[type=text] {
    background: #000 !important; color: #fff !important; border-color: #6a6a6a !important; }
body.dark .dash-input-container { background: #000 !important; }
body.dark .dash-input-stepper { background: #1c1c1c !important; color: #f0f0f0 !important; border-color: #6a6a6a !important; }
body.dark .btn-outline-secondary { color: #f0f0f0 !important; border-color: #9a9a9a !important; }
body.dark .btn-outline-secondary:hover { background: #2e2e2e !important; }
body.dark #btn-settings { background: #1c1c1c !important; }
body.dark .btn-outline-danger { color: #ff9c94 !important; border-color: #e0574c !important; }
/* tabs */
body.dark .tab { background: #141414 !important; color: #d6d6d6 !important; border-color: #4a4a4a !important; }
body.dark .tab--selected { background: #0a0a0a !important; color: #fff !important; }
body.dark .tab-container, body.dark .tab-parent { background: #141414 !important; }
/* dropdowns: Dash 2 (react-select) and Dash 3+ */
body.dark .Select-control, body.dark .Select-menu-outer, body.dark .Select-menu,
body.dark .Select-option, body.dark .VirtualizedSelectOption,
body.dark .dash-dropdown, body.dark .dash-dropdown-content, body.dark .dash-dropdown-options,
body.dark .dash-dropdown-option, body.dark .dash-dropdown-search, body.dark .dash-dropdown-trigger {
    background: #000 !important; color: #fff !important; border-color: #6a6a6a !important; }
body.dark .Select-value-label, body.dark .Select-input input,
body.dark .Select--single > .Select-control .Select-value .Select-value-label,
body.dark .dash-dropdown-wrapper, body.dark .dash-dropdown-value-item { color: #fff !important; }
body.dark .Select-placeholder, body.dark .dash-dropdown-placeholder,
body.dark .dash-dropdown-action-button { color: #b8b8b8 !important; }
body.dark .Select-option.is-focused, body.dark .VirtualizedSelectFocusedOption,
body.dark .dash-dropdown-option:hover { background: #2e2e2e !important; }
body.dark .Select--multi .Select-value { background: #4a2a0c !important; color: #ffd9b8 !important;
    border-color: #a5642c !important; }
/* tables (orange header row is kept) */
body.dark .dash-spreadsheet td { background-color: #000 !important; color: #fff !important;
    border-color: #4a4a4a !important; }
body.dark .dash-spreadsheet tr:nth-child(even) td { background-color: #33220f !important; }
body.dark .dash-spreadsheet th.dash-filter { background-color: #141414 !important; border-color: #4a4a4a !important; }
body.dark .dash-spreadsheet th.dash-filter input { color: #fff !important; background: transparent !important; }
body.dark .dash-dropdown-clear, body.dark .dash-dropdown-trigger-icon, body.dark .Select-arrow-zone,
body.dark .Select-clear-zone { color: #d0d0d0 !important; }
body.dark .dash-spreadsheet-menu .export, body.dark .previous-next-container button {
    background: #1c1c1c !important; color: #fff !important; border-color: #8a8a8a !important; }
body.dark .previous-next-container, body.dark .page-number, body.dark .current-page,
body.dark .current-page-container input, body.dark .last-page { color: #fff !important;
    background: transparent !important; }
/* status / info lines that have a dark grey text colour set inline */
body.dark #scan-status, body.dark #preview-info, body.dark #hits-info, body.dark #refview-info,
body.dark #snippet-count { color: #fff !important; }
body.dark .theia-progress-track { background: #3a3a3a !important; }
/* the current page number is the box's placeholder until you type in it */
body.dark .current-page::placeholder { color: #fff !important; opacity: 1 !important; }
body.dark .current-page-shadow { color: transparent !important; }
/* plots: full inversion -> black background, white traces; orange stays orange */
body.dark .js-plotly-plot { filter: invert(1) hue-rotate(180deg) contrast(1.15); }
"""
app.index_string = app.index_string.replace(
    "{%css%}", "{%css%}\n<style>" + _TABLE_CSS + _FB_CSS + _CALIB_CSS + _FIT_CSS + _DARK_CSS + "</style>")
# close this tab when Theia stops (Quit / Stop on the home page, Quit here, crash)
app.index_string = app.index_string.replace(
    "{%scripts%}", "{%scripts%}\n<script>" + tabwatch.watch_js("Theia") + "</script>")
# Spectrum graphs that get the bottom-right coordinate readout
READOUT_GRAPHS = ["viewer-graph", "preview-graph", "snippet-graph", "refview-graph", "sum-graph"]
TAB_STYLE = {"padding": "8px 16px", "fontWeight": "600"}
TAB_SEL = {**TAB_STYLE, "borderTop": f"3px solid {ACCENT}"}

_BASE_LAYOUT = html.Div([
    # Settings button at the right-hand end of the tab bar (visible on every
    # page); opens a small popup menu holding display options such as dark
    # mode.  Opening/closing is done in the browser (clientside callback
    # below): the button toggles it, clicking anywhere outside closes it.
    # Quit: stops Theia and closes (or blanks) this browser tab
    html.Button("\u23FB Quit", id="btn-quit", className="btn btn-outline-danger btn-sm",
                title="Stop Theia and close this tab",
                style={"position": "absolute", "top": "7px", "right": "122px", "zIndex": "20",
                       "fontWeight": "600"}),
    html.Div(id="_quit-hook", style={"display": "none"}),
    html.Button("\u2699 Settings", id="btn-settings", className="btn btn-outline-secondary btn-sm",
                style={"position": "absolute", "top": "7px", "right": "14px", "zIndex": "20",
                       "fontWeight": "600"}),
    html.Div([
        html.Div("Settings", style={"fontWeight": "700", "fontSize": "14px", "marginBottom": "8px"}),
        dcc.Checklist(id="dark-mode", options=[{"label": " Dark mode", "value": "on"}],
                      value=CFG.get("dark_mode", []), inputStyle={"marginRight": "6px"},
                      style={"fontSize": "13px"}),
    ], id="settings-menu", className="theia-box theia-menu",
       style={**BOX_STYLE, "display": "none", "position": "absolute", "top": "42px",
              "right": "14px", "zIndex": "30", "minWidth": "200px", "background": "white",
              "boxShadow": "0 4px 14px rgba(0,0,0,0.18)"}),
    html.Div(id="_dark-mode-hook", style={"display": "none"}),
    html.Div(id="_settings-menu-hook", style={"display": "none"}),
    dcc.Tabs(id="tabs", value="viewer", children=[
        dcc.Tab(label="Viewer", value="viewer", style=TAB_STYLE, selected_style=TAB_SEL,
                children=[viewer_tab]),
        dcc.Tab(label="Scan", value="scan", style=TAB_STYLE, selected_style=TAB_SEL, children=[
            html.Div([sidebar, scan_main], style={"display": "flex"}),
        ]),
        dcc.Tab(label="Scan Results", value="results", style=TAB_STYLE, selected_style=TAB_SEL,
                children=[results_tab]),
        dcc.Tab(label="Further output", value="further", style=TAB_STYLE, selected_style=TAB_SEL,
                children=[further_tab]),
    # Tab bar stops short of the right edge to leave room for the Quit and
    # Settings buttons
    ], style={"height": "42px", "width": "calc(100% - 210px)"}),
    # Polls the background scan; only enabled while a scan is running
    dcc.Interval(id="poll", interval=700, disabled=True),
    dcc.Store(id="scan-finished"),
    # Hidden hooks used by the clientside callbacks that attach the readout
    *[html.Div(id=f"_readout-hook-{gid}", style={"display": "none"}) for gid in READOUT_GRAPHS],
    html.Div(id="_snippet-resize-hook", style={"display": "none"}),
    html.Div(id="_snippet-width-saved", style={"display": "none"}),
    html.Div(id="_snippet-excluded-hook", style={"display": "none"}),
    # Further output tab: saved state handed to the restore logic on page load
    folder_browser,
    info_modal,
    pid_modal,
    iso_modal,
    srch_modal,
    dcc.Store(id="further-restore", data=None),
    dcc.Store(id="sum-view", data=None),
    html.Div(id="_sum-view-hook", style={"display": "none"}),
    html.Div(id="_further-saved", style={"display": "none"}),
], style={"position": "relative"})


def _find_component(node, cid):
    """Depth-first search of the layout tree for the component with id cid."""
    if getattr(node, "id", None) == cid:
        return node
    kids = getattr(node, "children", None)
    if kids is None:
        return None
    for kid in (kids if isinstance(kids, (list, tuple)) else [kids]):
        if hasattr(kid, "to_plotly_json"):
            hit = _find_component(kid, cid)
            if hit is not None:
                return hit
    return None


def serve_layout():
    """Layout served on every page load, with the Further output tab's
    controls set to the state saved last time (scanner_config.json ->
    further_state).  The summed plot and the fit are rebuilt by show_sum /
    update_fit from the further-restore store."""
    # Viewer tab: last saved viewer settings
    vcfg = load_viewer_config()
    for k in VIEWER_KEYS:
        comp = _find_component(_BASE_LAYOUT, f"v-{k}")
        if comp is not None:
            comp.value = vcfg[k] if k == "correction_enabled" else str(vcfg[k])
    comp = _find_component(_BASE_LAYOUT, "v-correction-inputs")
    if comp is not None:
        comp.style = {"display": "block" if vcfg["correction_enabled"] else "none"}

    st = load_config().get("further_state") or {}
    files = [f for f in (st.get("files") or []) if os.path.isfile(f)]
    fit = st.get("fit") or {}
    plot = st.get("plot") or {"source": "sum"}
    if plot.get("source") == "file":
        try:
            restorable = os.path.isfile(paths.remap_legacy(os.path.join(
                os.path.expanduser(plot.get("dir") or ""), plot.get("name") or "")))
        except Exception:
            restorable = False
    else:
        restorable = bool(files)
    for cid, prop, value in [
        ("further-files", "data", files),
        ("sum-align", "value", st.get("align") or "none"),
        ("sum-ion", "value", st.get("ion")),
        ("sum-unsure", "value", st.get("unsure") if st.get("unsure") is not None else ["on"]),
        ("fit-shape", "value", fit.get("shape") or "gauss"),
        ("fit-bg", "value", fit.get("bg") or "const"),
        ("further-restore", "data", (dict(st, files=files) if st.get("summed") and restorable
                                     else None)),
        ("pid-settings", "data", st.get("pid")),
        ("further-source", "value", st.get("source") or "sum"),
        ("fsrc-dir", "value", st.get("file_dir") or os.path.join(paths.summed_1d(), "")),
        ("fsrc-name", "value", st.get("file_name")),
        ("srch-width", "value", load_config().get("search_width_khz") or SRCH_DEFAULT_KHZ),
    ]:
        comp = _find_component(_BASE_LAYOUT, cid)
        if comp is not None:
            setattr(comp, prop, value)
    return _BASE_LAYOUT


app.layout = serve_layout

# ---------------------------------------------------------------------------
# Coordinate readout (bottom-right) for the spectrum graphs
# ---------------------------------------------------------------------------
# Snaps to the spectrum bin under the cursor and shows that bin's frequency
# and power.  Pixel positions are converted using the axes' real offsets
# (xaxis._offset) rather than layout.margin, because automargin widens the
# margins to fit tick labels and margin.l would bias the readout.
_READOUT_JS = """
function(figure) {
    var graphId = '%s';
    setTimeout(function() {
        var gd = document.querySelector('#' + graphId + ' .js-plotly-plot');
        if (!gd || gd._theiaReadout) return;
        gd._theiaReadout = true;

        var readout = document.getElementById('_coord-readout');
        if (!readout) {
            readout = document.createElement('div');
            readout.id = '_coord-readout';
            readout.style.cssText = [
                'position:fixed', 'bottom:12px', 'right:16px',
                'background:rgba(30,30,30,0.80)', 'color:#e8e8e8',
                'padding:5px 12px', 'border-radius:5px',
                'font-family:monospace', 'font-size:16px',
                'pointer-events:none', 'z-index:9999',
                'letter-spacing:0.03em', 'white-space:pre', 'display:none'
            ].join(';');
            document.body.appendChild(readout);
        }

        gd.addEventListener('mousemove', function(evt) {
            var layout = gd._fullLayout;
            if (!layout || !layout.xaxis) return;
            var rect = gd.getBoundingClientRect();
            var x = layout.xaxis.p2d(evt.clientX - rect.left - layout.xaxis._offset);
            if (!isFinite(x)) return;

            var spec = null, fd = gd._fullData || [];
            for (var i = 0; i < fd.length; i++) {
                if (fd[i].name === 'Spectrum') { spec = fd[i]; break; }
            }
            if (!spec || !spec.x || spec.x.length === 0) return;
            var xs = spec.x, ys = spec.y, n = xs.length;
            var asc = xs[n - 1] >= xs[0];
            var lo = 0, hi = n - 1;
            while (lo < hi) {
                var mid = (lo + hi) >> 1;
                if (asc ? xs[mid] < x : xs[mid] > x) lo = mid + 1; else hi = mid;
            }
            var k = lo;
            if (k > 0 && Math.abs(xs[k - 1] - x) <= Math.abs(xs[k] - x)) k = k - 1;

            var bx = Number(xs[k]), by = Number(ys[k]);
            readout.style.display = 'block';
            readout.innerText = 'x: ' + bx.toFixed(1) + ' Hz\\ny: ' +
                (isFinite(by) ? by.toExponential(3) : '---');
        });
        gd.addEventListener('mouseleave', function() {
            readout.style.display = 'none';
        });
    }, 150);
    return window.dash_clientside.no_update;
}
"""

for _gid in READOUT_GRAPHS:
    app.clientside_callback(
        _READOUT_JS % _gid,
        Output(f"_readout-hook-{_gid}", "children"),
        Input(_gid, "figure"),
    )

# The snippet plot stretches to fill its grid row, whose height depends on the
# left column (e.g. when the reference-viewer info text wraps).  Plotly only
# resizes on window resize, so watch the container and resize the plot.
app.clientside_callback(
    """
    function(figure) {
        setTimeout(function() {
            var box = document.getElementById('snippet-graph');
            if (!box || box._theiaResize || !window.ResizeObserver) return;
            box._theiaResize = true;
            new ResizeObserver(function() {
                var gd = box.querySelector('.js-plotly-plot');
                if (gd && window.Plotly) { window.Plotly.Plots.resize(gd); }
            }).observe(box);
        }, 200);
        return window.dash_clientside.no_update;
    }
    """,
    Output("_snippet-resize-hook", "children"),
    Input("snippet-graph", "figure"),
)

SETTING_KEYS = list(SETTING_FIELDS)
SETTING_STATES = [State(cid, prop) for cid, prop in SETTING_FIELDS.values()]


def _raw_from_states(values):
    return dict(zip(SETTING_KEYS, values))


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
    *[Output(f, "value", allow_duplicate=True) for f in BROWSE_TARGETS.values()],
    *[Input(b, "n_clicks") for b in BROWSE_TARGETS],
    Input({"type": "fb-dir", "index": ALL}, "n_clicks"),
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
        return (nu,) * 6 + keep
    pid, val = trig["prop_id"], trig["value"]

    def show(p, tgt):
        rows, info = _fb_listing(p)
        return (FB_SHOWN, p, tgt, rows, p, info) + keep

    for btn, field in BROWSE_TARGETS.items():
        if pid.startswith(btn + "."):
            return show(_fb_start_dir(current[field]), field)
    if pid.startswith("fb-cancel"):
        return (FB_HIDDEN, nu, None, nu, nu, nu) + keep
    if pid.startswith("fb-select"):
        chosen = os.path.join(path, "") if path else None      # with trailing '/'
        return (FB_HIDDEN, nu, None, nu, nu, nu) + tuple(
            chosen if target == f and chosen else nu for f in fields)
    if not path:
        return (nu,) * 6 + keep
    if pid.startswith("fb-up"):
        return show(os.path.dirname(path.rstrip("/")) or "/", target)
    if pid.startswith("fb-home"):
        return show(os.path.expanduser("~"), target)
    if pid.startswith("fb-path-input"):
        p = os.path.expanduser((val or "").strip())
        if os.path.isdir(p):
            return show(os.path.abspath(p), target)
        return (nu, nu, nu, nu, nu, html.Span("Not a folder: " + (val or ""),
                                              style={"color": "#c0392b"})) + keep
    if '"fb-dir"' in pid and val:                     # a sub-folder was clicked
        name = json.loads(pid.rsplit(".", 1)[0])["index"]
        return show(os.path.join(path, name), target)
    return (nu,) * 6 + keep


@app.callback(
    Output("v-spectrum_name", "options"),
    Output("v-spectrum_name", "value"),
    Input("v-file_path", "value"),
    State("v-spectrum_name", "value"),
)
def v_spectrum_options(directory, current):
    """Viewer: the .npz files in the chosen folder, as dropdown options."""
    names = list_npz(os.path.expanduser((directory or "").strip()))
    opts = [{"label": n, "value": n} for n in names]
    if current in names:
        return opts, current
    return opts, (names[0] if names else None)


# ===========================================================================
# Callbacks — Viewer tab
# ===========================================================================

_V_INPUTS = [Input(f"v-{k}", "value") for k in VIEWER_KEYS]


@app.callback(Output("v-correction-inputs", "style"), Input("v-correction_enabled", "value"))
def v_toggle_correction_inputs(enabled):
    return {"display": "block" if enabled else "none"}


@app.callback(Output("_viewer-saved", "children"), *_V_INPUTS, prevent_initial_call=True)
def v_persist_config(*values):
    save_viewer_config(dict(zip(VIEWER_KEYS, values)))
    return dash.no_update


def _v_ints(s):
    return [int(p.strip()) for p in str(s).split(",") if p.strip()]


@app.callback(Output("viewer-graph", "figure"), Output("v-error-box", "children"), *_V_INPUTS)
def v_update_figure(*values):
    v = dict(zip(VIEWER_KEYS, values))
    if not v["spectrum_name"]:
        return empty_fig("No spectrum selected (no .npz files in this folder?)"), ""
    try:
        fig, err = v_build_figure(
            file_path=v["file_path"] or "", spectrum_name=v["spectrum_name"] or "",
            ref_name=v["ref_name"] or "", ref_frequency=float(v["ref_frequency"] or 0),
            ref_harmonic=int(float(v["ref_harmonic"] or 1)),
            z_lims=_v_ints(v["z_lims"]), n_lims=_v_ints(v["n_lims"]),
            el1=_v_ints(v["el1"]), el2=_v_ints(v["el2"]),
            mq_width=float(v["mq_width"] or 0.01), gamma_t=float(v["gamma_t"] or 1.399),
            harmonics=_v_ints(v["harmonics"]),
            correction_enabled=bool(v["correction_enabled"]),
            corr_a=float(v["corr_a"] or 0), corr_b=float(v["corr_b"] or 0),
            corr_c=float(v["corr_c"] or 0),
            highlight_ions=v_parse_highlight_ions(v["highlight_ions"] or ""))
        return fig, err
    except Exception as exc:
        return empty_fig(), f"Parse error: {exc}"


# ===========================================================================
# Callbacks — Scan tab
# ===========================================================================

@app.callback(
    Output("preview-file", "options"),
    Output("preview-file", "value"),
    Input("s-spectra_dir", "value"),
    Input("s-file_pattern", "value"),
    State("preview-file", "value"),
)
def refresh_preview_files(directory, pattern, current):
    names = list_spectra((directory or "").strip(), (pattern or "*.npz").strip())
    opts = [{"label": n, "value": n} for n in names]
    return opts, (current if current in names else (names[0] if names else None))


@app.callback(
    Output("preview-graph", "figure"),
    Output("preview-table", "data"),
    Output("preview-info", "children"),
    Output("settings-error", "children"),
    Input("btn-preview", "n_clicks"),
    State("preview-file", "value"),
    *SETTING_STATES,
    prevent_initial_call=True,
)
def do_preview(_, fname, *values):
    raw = _raw_from_states(values)
    save_config(raw)
    try:
        s = parse_settings(raw)
    except Exception as exc:
        return dash.no_update, [], "", f"Settings error: {exc}"
    if not fname:
        return empty_fig("No spectrum selected"), [], "", ""
    try:
        t0 = time.time()
        ff, pp = load_spectrum(s["spectra_dir"], fname)
        r = analyse_spectrum(ff, pp, s, s["ref_frequency"])
        r["predicted"] = s["ref_frequency"]
        dt = time.time() - t0
    except Exception as exc:
        traceback.print_exc()
        return empty_fig(str(exc)), [], "", f"Error: {exc}"

    peaks, m = r["peaks"], r["matches"]
    n_max = int((peaks["kind"] == "max").sum()) if len(peaks) else 0
    n_sh = int((peaks["kind"] == "shoulder").sum()) if len(peaks) else 0
    n_id_peaks = 0 if m is None or not len(m) else m["peak_row"].nunique()
    cut = noise_cutoff(pp, s["noise_cut_factor"])
    cut_txt = "off" if cut is None else f"{cut:.3g}"
    ref_txt = (f"{r['ref_used']:.1f} Hz (shift {r['ref_used'] - s['ref_frequency']:+.1f} Hz, {r['status']})"
               if r["ref_used"] is not None else "NOT FOUND in search window")
    info = (f"{fname}:  {n_max} maxima + {n_sh} shoulders  |  reference {ref_txt}  |  "
            f"{0 if m is None else len(m)} identifications on {n_id_peaks} peaks  |  "
            + (f"{int(r['excluded'].sum())} of {len(r['excluded'])} ions excluded "
               f"({hl_cut_text(s)})  |  " if hl_cut_text(s) else "")
            + f"median σ = {np.median(r['sig']):.3g}  |  noise cutoff = {cut_txt}  |  {dt:.2f} s")
    fig = spectrum_figure(ff, pp, peaks, r["base"], r["sig"], s, r, title=fname)
    rows = [] if m is None else _fmt_records(m.sort_values("peak_freq"))
    return fig, rows, info, ""


def _fmt_records(df):
    out = df.copy()
    for c in ("peak_freq", "sim_freq", "ref_found"):
        if c in out:
            out[c] = out[c].map(lambda v: round(float(v), 2))
    for c in ("delta_hz",):
        if c in out:
            out[c] = out[c].map(lambda v: round(float(v), 2))
    for c in ("power", "snr", "mean_delta_hz", "std_delta_hz"):
        if c in out:
            out[c] = out[c].map(lambda v: None if pd.isna(v) else float(f"{v:.4g}"))
    return out.to_dict("records")


@app.callback(
    Output("action-msg", "children"),
    Output("run-error", "children"),
    Input("btn-run", "n_clicks"),
    *SETTING_STATES,
    prevent_initial_call=True,
)
def start_scan(_, *values):
    raw = _raw_from_states(values)
    save_config(raw)
    if SCAN_STATE["running"]:
        return "A scan is already running.", ""
    try:
        s = parse_settings(raw)
        if not os.path.isdir(s["spectra_dir"]):
            raise ValueError(f"Directory not found: {s['spectra_dir']}")
    except Exception as exc:
        return "", f"Settings error: {exc}"
    _set_state(running=True, cancel=False, done=0, total=0, msg="Starting…", db_path=None, error=None)
    threading.Thread(target=run_scan, args=(s, raw.get("run_label") or ""), daemon=True).start()
    return "Starting…", ""


@app.callback(
    Output("cancel-msg", "children"),
    Input("btn-cancel", "n_clicks"),
    prevent_initial_call=True,
)
def cancel_scan(_):
    if SCAN_STATE["running"]:
        _set_state(cancel=True)
        return "Cancelling… (the partial database is kept and marked 'cancelled')"
    return "No scan running."


@app.callback(
    Output("progress-bar", "style"),
    Output("scan-status", "children"),
    Output("scan-finished", "data"),
    Output("poll", "disabled"),
    Input("poll", "n_intervals"),
    Input("action-msg", "children"),   # fires when a scan is started -> enables polling
    State("progress-bar", "style"),
    State("scan-finished", "data"),
)
def poll_scan(_, _started, style, finished):
    with SCAN_LOCK:
        st = dict(SCAN_STATE)
    pct = 100.0 * st["done"] / st["total"] if st["total"] else (100.0 if not st["running"] and st["db_path"] else 0)
    style = {**(style or {}), "width": f"{pct:.1f}%"}
    new_finished = st["db_path"] if (not st["running"] and st["db_path"]) else None
    return (style, st["msg"], (new_finished if new_finished != finished else dash.no_update),
            not st["running"])


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


def _kv_rows(d):
    return [{"key": k, "value": json.dumps(v) if isinstance(v, (list, dict, bool)) else str(v)}
            for k, v in d.items()]


@app.callback(
    Output("runinfo-table", "data"),
    Output("settings-table", "data"),
    Output("summary-table", "data"),
    Output("ion-filter", "options"),
    Output("spec-filter", "options"),
    Output("ref-graph", "figure"),
    Input("db-select", "value"),
)
def load_database(path):
    if not path or not os.path.exists(path):
        return [], [], [], [], [], empty_fig()
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    info, st = read_kv(con, "run_info"), read_kv(con, "settings")
    con.close()
    summary = db_query(path, """
        SELECT ion, COUNT(DISTINCT spectrum_id) AS n_spectra, COUNT(*) AS n_hits,
               AVG(delta_hz) AS mean_delta_hz,
               SQRT(MAX(AVG(delta_hz*delta_hz) - AVG(delta_hz)*AVG(delta_hz), 0)) AS std_delta_hz
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
    return (_kv_rows(info), _kv_rows(st), _fmt_records(summary), ion_opts, spec_opts, fig)


@app.callback(
    Output("ion-filter", "value"),
    Input("summary-table", "active_cell"),
    State("summary-table", "derived_viewport_data"),
    State("ion-filter", "value"),
    prevent_initial_call=True,
)
def summary_click(cell, rows, current):
    if not cell or not rows or cell["row"] >= len(rows):
        return dash.no_update
    return [f"ion:{rows[cell['row']]['ion']}"]


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
    sql = (f"SELECT {', '.join(HIT_COLUMNS)} FROM hits"
           + (" WHERE " + " AND ".join(where) if where else "")
           + f" ORDER BY seq, peak_freq LIMIT {MAX_ROWS + 1}")
    df = db_query(path, sql, params)
    truncated = len(df) > MAX_ROWS
    df = df.iloc[:MAX_ROWS]
    n_spec = df["spectrum_id"].nunique() if len(df) else 0
    info = (f"{len(df)} identifications in {n_spec} spectra"
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
        sims = simulate_frequencies(ion_db, ref_mq, float(row["ref_found"]), s["ref_harmonic"],
                                    s["gamma_t"], s["harmonics"], s["correction_enabled"],
                                    s["corr_a"], s["corr_b"], s["corr_c"])
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
    info, st = read_kv(con, "run_info"), read_kv(con, "settings")
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
    """'+ Add file' shows the picker (files already listed are left out);
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


def sum_spectra(files, align, ion_key=None, include_unsure=True):
    """Sum the spectra referenced by the confirmation files.

    align: 'none' | 'strongest' | 'identified'.  Returns (freq, summed power,
    info dict).  The first spectrum's frequency axis is the common grid; each
    spectrum is shifted by (target - its peak position) and linearly
    interpolated onto that grid, target = median of the peak positions.
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
        if align == "strongest":
            peak_pos[(d, name)] = float(ff[int(np.nanargmax(pp))])
        spectra.append(((d, name), ff, pp))
    if not spectra:
        raise ValueError("No spectra could be loaded" + (f" ({failed[0]})" if failed else "."))

    grid = spectra[0][1]
    target = float(np.median([peak_pos[k] for k, _, _ in spectra])) if align != "none" else None
    total = np.zeros_like(grid)
    covered = np.ones(grid.shape, dtype=bool)
    shifts = []
    for k, ff, pp in spectra:
        shift = (target - peak_pos[k]) if target is not None else 0.0
        shifts.append(shift)
        x = ff + shift
        if shift == 0.0 and len(ff) == len(grid) and np.array_equal(ff, grid):
            y = pp
        else:
            y = np.interp(grid, x, pp, left=np.nan, right=np.nan)
        covered &= np.isfinite(y)
        total += np.nan_to_num(y)
    total[~covered] = np.nan
    # initial zoom: where the chosen ion ends up after shifting (median over
    # the summed spectra that contain it); falls back to the alignment target
    zoom, zoom_on_ion = target, False
    if target is not None:
        moved = [ion_pos[k] + s for (k, _, _), s in zip(spectra, shifts) if k in ion_pos]
        if moved:
            zoom, zoom_on_ion = float(np.median(moved)), True
    info = dict(n=len(spectra), n_requested=n_requested, failed=failed, target=target, zoom=zoom,
                zoom_on_ion=zoom_on_ion,
                shifts=np.array(shifts), names=[k[1] for k, _, _ in spectra],
                n_dirs=len({k[0] for k, _, _ in spectra}))
    return grid[covered], total[covered], info


@app.callback(Output("sum-ion", "style"), Input("sum-align", "value"))
def toggle_sum_ion(align):
    return {"fontSize": "13px", "width": "240px", "marginRight": "14px",
            "display": "none" if align == "none" else "block"}


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
    """Single-file source: the .npz files in the folder.  'Use Viewer spectrum'
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
    """Parsed settings for the simulated lines: base settings + overrides."""
    return parse_settings({**pid_base(source, files), **(overrides or {})})


def _pid_display(raw, key):
    """A settings value as shown in the popup's fields."""
    v = raw.get(key)
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
    sims = simulate_frequencies(ion_db, ref_mq, ref_f, s["ref_harmonic"], s["gamma_t"],
                                s["harmonics"], s["correction_enabled"],
                                s["corr_a"], s["corr_b"], s["corr_c"])
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
)
def show_sum(_, _load, _req, restore, files, align, ion_key, unsure, show_sims, pid_store,
             f_dir, f_name):
    """'Sum spectra' button, 'Load spectrum' / 'Use Viewer spectrum' (single
    spectrum file), or (on page load) rebuild the plot saved last time with
    its zoom; the saved fit is then picked up by update_fit."""
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
            ff, pp, info = sum_spectra(files, align, ion_key, bool(unsure))
        except Exception as exc:
            return (empty_fig(str(exc)), html.Span(f"Error: {exc}", style={"color": "#c0392b"}),
                    None, "")
        if align == "identified":
            ion, harm = ion_key.rsplit("|", 1)
            how = f"aligned on {ion} h={harm}"
        elif align == "strongest":
            how = "aligned on the strongest peak"
        else:
            how = "not aligned"
        title = f"Sum of {info['n']} spectra ({how})"
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
        parts.append(f"shifts {sh.min():+.1f} to {sh.max():+.1f} Hz "
                     f"(median |shift| {np.median(np.abs(sh)):.1f} Hz)")
    if align == "strongest" and ion_key:
        zi, zh = ion_key.rsplit("|", 1)
        parts.append(f"view centred on {zi} h={zh}" if info["zoom_on_ion"] else
                     f"{zi} h={zh} not in the summed spectra, view centred on the strongest peak")
    if info["n_dirs"] > 1:
        parts.append(f"from {info['n_dirs']} spectra directories")
    status = [html.Span("  ·  ".join(parts))]
    if info["failed"]:
        status.append(html.Div(f"Could not load {len(info['failed'])}: "
                               + "; ".join(info["failed"][:3])
                               + (" …" if len(info["failed"]) > 3 else ""),
                               style={"color": "#c0392b"}))
    return fig, status, key, sim_text


# ===========================================================================
# Further output: peak fitting on the summed spectrum
# ===========================================================================
#
# Parameters per peak: centre (Hz), amplitude (peak height above background),
# FWHM (Hz) and, for pseudo-Voigt, eta (Lorentzian fraction, 0..1).  All
# shapes are written in terms of FWHM so switching shape keeps the width.
# Fitting is done in offset frequency u = f - xref (xref = middle of the fit
# region) to keep the numbers well conditioned.

_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def _peak_shape(u, c, a, w, eta, shape):
    w = max(abs(w), 1e-9)
    g = np.exp(-0.5 * ((u - c) / (w * _FWHM_TO_SIGMA)) ** 2)
    if shape == "gauss":
        return a * g
    lor = 1.0 / (1.0 + ((u - c) / (w / 2.0)) ** 2)
    if shape == "lorentz":
        return a * lor
    eta = min(max(eta, 0.0), 1.0)
    return a * (eta * lor + (1.0 - eta) * g)


def _peak_area(a, w, eta, shape):
    ga = a * w * _FWHM_TO_SIGMA * np.sqrt(2 * np.pi)
    la = a * np.pi * w / 2.0
    if shape == "gauss":
        return ga
    if shape == "lorentz":
        return la
    return eta * la + (1 - eta) * ga


def _n_bg(bg):
    return {"none": 0, "const": 1, "linear": 2}.get(bg, 0)


def _shapes(state):
    """Line shape of each peak (each peak has its own)."""
    default = state.get("shape") or "gauss"
    return [pk.get("shape") or default for pk in state["peaks"]]


def _pack(state):
    """state -> (p0, names) for the parameters; centres as offsets from xref.
    eta is a parameter only for pseudo-Voigt peaks."""
    xref = state["xref"]
    p, names = [], []
    for i, (pk, shape) in enumerate(zip(state["peaks"], _shapes(state))):
        p += [pk["centre"] - xref, pk["amp"], pk["fwhm"]]
        names += [(i, "centre"), (i, "amp"), (i, "fwhm")]
        if shape == "pvoigt":
            p.append(pk.get("eta", 0.5))
            names.append((i, "eta"))
    for j in range(_n_bg(state["bg"])):
        p.append(state["bgp"][j])
        names.append((None, f"b{j}"))
    return np.array(p, dtype=float), names


def _model_from_vector(u, p, shapes, bg):
    """Model for the parameter vector laid out as in _pack."""
    y = np.zeros_like(u, dtype=float)
    q = 0
    for shape in shapes:
        c, a, w = p[q:q + 3]
        q += 3
        eta = 0.0
        if shape == "pvoigt":
            eta = p[q]
            q += 1
        y += _peak_shape(u, c, a, w, eta, shape)
    nb = _n_bg(bg)
    if nb >= 1:
        y += p[q]
    if nb == 2:
        y += p[q + 1] * u
    return y


def fit_overlay(ff, state):
    """(x, y) for the four overlay traces: total, components (each on top of
    the background), background, peak markers.  Evaluated on a fine grid
    over the fit region; all empty if there are no peaks."""
    if not state or not state.get("peaks") or not state.get("region"):
        return [([], [])] * len(FIT_TRACE_STYLES)
    lo, hi = state["region"]
    bin_hz = float(np.median(np.diff(ff))) if len(ff) > 1 else 1.0
    n = int(min(6000, max(200, 6 * (hi - lo) / bin_hz)))
    x = np.linspace(lo, hi, n)
    u = x - state["xref"]
    nb = _n_bg(state["bg"])
    bgp = state.get("bgp") or [0.0, 0.0]
    back = np.zeros_like(u)
    if nb >= 1:
        back += bgp[0]
    if nb == 2:
        back += bgp[1] * u
    total = back.copy()
    cx, cy = [], []
    for pk, shape in zip(state["peaks"], _shapes(state)):
        comp = _peak_shape(u, pk["centre"] - state["xref"], pk["amp"], pk["fwhm"],
                           pk.get("eta", 0.5), shape)
        total += comp
        cx += x.tolist() + [None]
        cy += (comp + back).tolist() + [None]
    mx = [pk["centre"] for pk in state["peaks"]]
    my = [float(np.interp(c, x, total)) for c in mx]
    # FWHM handles at the half-height points of each component (on its curve)
    hx, hy = [], []
    for pk in state["peaks"]:
        for side in (-1, 1):
            xh = pk["centre"] + side * 0.5 * pk["fwhm"]
            hx.append(xh)
            hy.append(float(np.interp(xh, x, back)) + 0.5 * pk["amp"])
    return [(x.tolist(), total.tolist()),
            (cx, cy),
            (x.tolist(), back.tolist()) if nb else ([], []),
            (mx, my),
            (hx, hy)]


def _region_mask(ff, region):
    lo, hi = region
    return (ff >= lo) & (ff <= hi)


def run_fit(ff, pp, sigma, state):
    """Least-squares fit over the region, starting from the current values.
    Returns the updated state (values, uncertainties, chi2/ndf)."""
    from scipy.optimize import curve_fit
    m = _region_mask(ff, state["region"]) & np.isfinite(pp)
    u, y = ff[m] - state["xref"], pp[m]
    p0, names = _pack(state)
    lo_u, hi_u = float(u.min()), float(u.max())
    bin_hz = float(np.median(np.diff(ff[m])))
    xref = state["xref"]
    lb, ub = [], []
    cb = state.get("centre_bounds")             # optional per-peak centre limits (Hz)
    fb = state.get("fwhm_bounds")

    def _lim(pk, key):
        v = (pk.get("lim") or {}).get(key)
        return None if v is None or not np.isfinite(v) else float(v)
    for (i, what) in names:
        pk = state["peaks"][i] if i is not None else {}
        if what == "centre":
            lo, hi = lo_u, hi_u
            if cb and cb[i]:
                lo, hi = max(lo, cb[i][0] - xref), min(hi, cb[i][1] - xref)
            if _lim(pk, "centre_min") is not None:
                lo = max(lo, _lim(pk, "centre_min") - xref)
            if _lim(pk, "centre_max") is not None:
                hi = min(hi, _lim(pk, "centre_max") - xref)
            if not lo < hi:
                raise ValueError(f"peak {i + 1}: the centre limits do not overlap the fit region")
            lb.append(lo); ub.append(hi)
        elif what == "amp":
            lb.append(0.0); ub.append(np.inf)
        elif what == "fwhm":
            lo, hi = 0.05 * bin_hz, max(hi_u - lo_u, bin_hz)
            if fb and fb[i]:
                lo, hi = fb[i]
            if _lim(pk, "fwhm_min") is not None:
                lo = max(_lim(pk, "fwhm_min"), 1e-9)
            if _lim(pk, "fwhm_max") is not None:
                hi = _lim(pk, "fwhm_max")
            if not lo < hi:
                raise ValueError(f"peak {i + 1}: FWHM min must be below FWHM max")
            lb.append(lo); ub.append(hi)
        elif what == "eta":
            lb.append(0.0); ub.append(1.0)
        else:
            lb.append(-np.inf); ub.append(np.inf)
    lb, ub = np.array(lb), np.array(ub)

    # fixed parameters are left out of the fit (held at their current value)
    bg_fix = set(state.get("bg_fix") or [])
    free = np.array([(what not in bg_fix) if i is None
                     else (what not in (state["peaks"][i].get("fix") or []))
                     for (i, what) in names], dtype=bool)
    n_free = int(free.sum())
    if n_free == 0:
        raise ValueError("every parameter is fixed: free at least one")
    if len(u) <= n_free:
        raise ValueError(f"Only {len(u)} bins in the fit region for {n_free} free parameters")
    for q in np.flatnonzero(free):                # start strictly inside the bounds
        if not (lb[q] < p0[q] < ub[q]):
            span = (ub[q] - lb[q]) if np.isfinite(ub[q] - lb[q]) else max(abs(p0[q]), 1.0)
            p0[q] = min(max(p0[q], lb[q] + 1e-6 * span), ub[q] - 1e-6 * span)
    shapes, bg = _shapes(state), state["bg"]
    n_peaks = len(shapes)
    p_all = p0.copy()

    def full(pf):
        p = p_all.copy()
        p[free] = pf
        return p

    def f_free(uu, *pf):
        return _model_from_vector(uu, full(np.array(pf)), shapes, bg)

    # absolute_sigma=False: the covariance is scaled by chi2/ndf, i.e. the
    # uncertainties are scaled by sqrt(chi2/ndf).  The noise sigma is only a
    # rough estimate (and real peaks are not perfect line shapes), so this
    # lets the scatter of the data about the fit set the uncertainties.
    popt_f, pcov_f = curve_fit(f_free, u, y, p0=p0[free], sigma=np.full(len(u), sigma),
                               absolute_sigma=False, bounds=(lb[free], ub[free]), maxfev=20000)
    popt = full(popt_f)
    pcov = np.zeros((len(p0), len(p0)))
    pcov[np.ix_(free, free)] = pcov_f
    ok_cov = np.all(np.isfinite(pcov_f))
    perr = np.sqrt(np.clip(np.diag(pcov), 0, None)) if ok_cov else np.full(len(popt), np.nan)
    perr[~free] = 0.0
    resid = y - _model_from_vector(u, popt, shapes, bg)
    ndf = max(len(u) - n_free, 1)

    new = dict(state)
    new["peaks"] = [dict(pk) for pk in state["peaks"]]
    new["bgp"] = list(state.get("bgp") or [0.0, 0.0])
    unc = {"peaks": [{} for _ in state["peaks"]], "bgp": [None, None]}
    for (i, what), v, e in zip(names, popt, perr):
        if i is None:
            j = int(what[1])
            new["bgp"][j], unc["bgp"][j] = float(v), float(e)
        else:
            new["peaks"][i][what] = float(v + state["xref"]) if what == "centre" else float(v)
            unc["peaks"][i][what] = float(e)
    # area uncertainty: linear error propagation with a numerical gradient
    for i in range(n_peaks):
        idx = [k for k, (j, _) in enumerate(names) if j == i]
        keys = [names[k][1] for k in idx]

        def area_of(vec, shape=shapes[i]):
            d = dict(zip(keys, vec))
            return _peak_area(d["amp"], d["fwhm"], d.get("eta", 0.5), shape)
        v0 = popt[idx].astype(float)
        grad = np.zeros(len(idx))
        for q in range(len(idx)):
            h = 1e-6 * max(abs(v0[q]), 1.0)
            vp = v0.copy()
            vp[q] += h
            grad[q] = (area_of(vp) - area_of(v0)) / h
        sub = pcov[np.ix_(idx, idx)]
        unc["peaks"][i]["area"] = (float(np.sqrt(max(grad @ sub @ grad, 0.0)))
                                   if ok_cov else float("nan"))
    new.update(unc=unc, chi2=float(np.sum((resid / sigma) ** 2) / ndf), ndf=int(ndf),
               sigma=float(sigma), fitted=True)
    return new


def _snap_peak(ff, pp, x, bgp, bg):
    """Start values for a peak clicked at x: centre and amplitude from the
    highest bin within +-3 bins, FWHM from where the data fall to half height."""
    k0 = int(np.clip(np.searchsorted(ff, x), 0, len(ff) - 1))
    lo, hi = max(k0 - 3, 0), min(k0 + 4, len(ff))
    k = lo + int(np.nanargmax(pp[lo:hi]))
    base = bgp[0] if _n_bg(bg) >= 1 else 0.0
    amp = max(float(pp[k] - base), 0.0)
    half = base + amp / 2.0
    left = k
    while left > 0 and pp[left - 1] > half and k - left < 200:
        left -= 1
    right = k
    while right < len(pp) - 1 and pp[right + 1] > half and right - k < 200:
        right += 1
    bin_hz = float(np.median(np.diff(ff[max(k - 5, 0):k + 6])))
    fwhm = max((right - left + 1) * bin_hz, 1.5 * bin_hz)
    return {"centre": float(ff[k]), "amp": amp, "fwhm": fwhm, "eta": 0.5}


def _fmt_num(v, fmt):
    try:
        return "–" if v is None or not np.isfinite(v) else format(v, fmt)
    except TypeError:
        return "–"


def fit_table_rows(state):
    if not state or not state.get("peaks"):
        return []
    unc = (state.get("unc") or {}).get("peaks") or [{} for _ in state["peaks"]]
    rows = []
    for i, (pk, e, shape) in enumerate(zip(state["peaks"], unc, _shapes(state)), start=1):
        pv = shape == "pvoigt"
        fix = set(pk.get("fix") or [])
        lim = pk.get("lim") or {}

        def err(key, fmt):
            return "fixed" if key in fix else _fmt_num(e.get(key), fmt)
        area = _peak_area(pk["amp"], pk["fwhm"], pk.get("eta", 0.5), shape)
        rows.append({
            "peak": i, "ion": pk.get("ion") or "", "harmonic": pk.get("harmonic"),
            "shape": shape,
            "centre": round(pk["centre"], 2), "d_centre": err("centre", ".2f"),
            "amp": float(f"{pk['amp']:.6g}"), "d_amp": err("amp", ".3g"),
            "fwhm": round(pk["fwhm"], 2), "d_fwhm": err("fwhm", ".2f"),
            "eta": round(pk.get("eta", 0.5), 4) if pv else None,
            "d_eta": err("eta", ".3f") if pv else "",
            "area": _fmt_num(area, ".5g"), "d_area": _fmt_num(e.get("area"), ".3g"),
            **{f"fix_{k}": (FIX_ON if k in fix else FIX_OFF) if (k != "eta" or pv) else ""
               for k in FIX_PARAMS},
            **{k: lim.get(k) for k in LIMIT_KEYS},
        })
    return rows


def _set_region(state, region):
    """New fit region [lo, hi]; xref moves to its middle (a linear background
    keeps the same line)."""
    lo, hi = sorted(float(v) for v in region)
    if not hi > lo:
        raise ValueError("the fit region needs min < max")
    old_ref = state.get("xref") or 0.5 * (lo + hi)
    state["region"], state["xref"] = [lo, hi], 0.5 * (lo + hi)
    if _n_bg(state["bg"]) == 2:
        state["bgp"][0] += state["bgp"][1] * (state["xref"] - old_ref)
    return state


def _num_or_none(v):
    try:
        v = float(v)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _harmonic_or_none(v):
    """Harmonic typed in the table: a positive integer, else None."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return int(round(f)) if np.isfinite(f) and f >= 1 and abs(f - round(f)) < 1e-6 else None


def _only_names_changed(state, table):
    """True if the table differs from the state only in the ion / harmonic
    columns (same rows, same values otherwise)."""
    if not state or not state.get("peaks") or table is None:
        return False
    old = fit_table_rows(state)
    if len(old) != len(table):
        return False
    skip = {"ion", "harmonic"}
    for a, b in zip(old, table):
        for k, v in a.items():
            if k in skip:
                continue
            w = b.get(k)
            if v == w:
                continue
            try:                                   # numbers may come back as str / float
                if float(v) == float(w):
                    continue
            except (TypeError, ValueError):
                pass
            if (v in (None, "") and w in (None, "")):
                continue
            return False
    return True


def _fit_status(state):
    if not state or not state.get("peaks"):
        return ""
    lo, hi = state["region"]
    reg = f"region {lo:.1f} – {hi:.1f} Hz ({hi - lo:.0f} Hz wide)"
    if state.get("fitted"):
        return html.Span([html.B("Fitted: "),
                          f"{len(state['peaks'])} peak(s), {reg}  ·  χ²/ndf = "
                          f"{state['chi2']:.3g} (ndf = {state['ndf']}, noise σ = "
                          f"{state['sigma']:.3g} per bin; uncertainties scaled by √(χ²/ndf))"])
    return html.Span([html.B("Start values, not fitted: "),
                      f"{len(state['peaks'])} peak(s), {reg}  ·  edit the table or press Fit"],
                     style={"color": "#888"})


@app.callback(
    Output("fit-mode", "data"),
    Output("btn-fit-mode", "children"),
    Output("btn-fit-mode", "style"),
    Output("energy-mode", "data"),
    Output("btn-energy-mode", "children"),
    Output("btn-energy-mode", "style"),
    Input("btn-fit-mode", "n_clicks"),
    Input("btn-energy-mode", "n_clicks"),
    State("fit-mode", "data"),
    State("energy-mode", "data"),
    prevent_initial_call=True,
)
def toggle_modes(_f, _e, fit_on, energy_on):
    """Peak fitting and energy calculation are click modes for the summed
    plot; turning one on turns the other off."""
    trig = dash.callback_context.triggered[0]["prop_id"] if dash.callback_context.triggered else ""
    if trig.startswith("btn-fit-mode"):
        fit_on, energy_on = not fit_on, False
    else:
        energy_on, fit_on = not energy_on, False
    base = {"marginRight": "12px", "fontWeight": "600"}
    on_style = {**base, "backgroundColor": ACCENT, "color": "white", "borderColor": ACCENT}
    return (fit_on, "Peak fitting: ON (click peaks)" if fit_on else "Peak fitting",
            on_style if fit_on else base,
            energy_on, "Energy calculation: ON (click peaks)" if energy_on else "Energy calculation",
            on_style if energy_on else base)


# In peak-fitting mode a click (without dragging) on the summed plot adds a
# peak there: the browser converts the pixel position to Hz and hands it to
# the server through the fit-click store (with the current view range, which
# becomes the fit region).
app.clientside_callback(
    """
    function(on, energyOn, figure) {
        window._theiaFitMode = !!on;
        window._theiaEnergyMode = !!energyOn;
        var box = document.getElementById('sum-graph');
        if (box) box.classList.toggle('fit-mode', !!on || !!energyOn);
        setTimeout(function() {
            var gd = document.querySelector('#sum-graph .js-plotly-plot');
            if (!gd || gd._theiaFitClick) return;
            gd._theiaFitClick = true;
            var down = null;
            gd.addEventListener('mousedown', function(e) {
                down = [e.clientX, e.clientY]; window._theiaSkipClick = false; }, true);

            // ---- drag on the fit overlay ---------------------------------
            //  * x marker of a peak: move centre + amplitude (cursor: move)
            //  * below the marker (the peak's column): centre only (cursor: <->)
            //  * diamond handle at half height: FWHM (cursor: <->)
            //  * orange region line: fit region edge (cursor: col-resize)
            // The curve is redrawn in the browser while dragging (from the peak
            // parameters in the marker trace's customdata); on release the new
            // values go to the server (fit-drag / fit-region-set stores).
            function traceIdx(name) {
                for (var i = 0; i < gd.data.length; i++) if (gd.data[i].name === name) return i;
                return -1;
            }
            function shapeIdx(name) {
                var S = (gd.layout && gd.layout.shapes) || [];
                for (var i = 0; i < S.length; i++) if (S[i] && S[i].name === name) return i;
                return -1;
            }
            function profile(code, u, w, eta) {
                w = Math.max(Math.abs(w), 1e-9);
                var s = w / (2 * Math.sqrt(2 * Math.LN2));
                var g = Math.exp(-0.5 * (u / s) * (u / s));
                var l = 1 / (1 + (2 * u / w) * (2 * u / w));
                if (code === 0) return g;
                if (code === 1) return l;
                eta = Math.min(Math.max(eta, 0), 1);
                return eta * l + (1 - eta) * g;
            }
            function interp(X, Y, x) {
                var n = X.length;
                if (!n) return 0;
                if (x <= X[0]) return Y[0];
                if (x >= X[n - 1]) return Y[n - 1];
                var f = (x - X[0]) / (X[n - 1] - X[0]) * (n - 1), i = Math.floor(f), w = f - i;
                return Y[i] * (1 - w) + Y[Math.min(i + 1, n - 1)] * w;
            }
            function pos(e) {
                var L = gd._fullLayout;
                if (!L || !L.xaxis || !L.yaxis) return null;
                var r = gd.getBoundingClientRect();
                var px = e.clientX - r.left - L.xaxis._offset;
                var py = e.clientY - r.top - L.yaxis._offset;
                if (px < 0 || px > L.xaxis._length || py < 0 || py > L.yaxis._length) return null;
                return {px: px, py: py, L: L};
            }
            function hit(e) {
                var p = pos(e);
                if (!p) return null;
                var L = p.L, ih = traceIdx('fit width handles'), ip = traceIdx('fit peaks');
                if (ih >= 0) {
                    var hx = gd.data[ih].x || [], hy = gd.data[ih].y || [];
                    for (var j = 0; j < hx.length; j++)
                        if (Math.abs(L.xaxis.d2p(hx[j]) - p.px) < 7 &&
                            Math.abs(L.yaxis.d2p(hy[j]) - p.py) < 7)
                            return {kind: 'width', k: Math.floor(j / 2), side: (j % 2) ? 1 : -1};
                }
                if (ip >= 0) {
                    var mx = gd.data[ip].x || [], my = gd.data[ip].y || [];
                    var best = -1, bestd = 9, onTop = false;
                    for (var k = 0; k < mx.length; k++) {
                        var d = Math.abs(L.xaxis.d2p(mx[k]) - p.px), top = L.yaxis.d2p(my[k]);
                        if (d < bestd && p.py >= top - 15) {
                            best = k; bestd = d; onTop = Math.abs(p.py - top) < 11; }
                    }
                    if (best >= 0) return {kind: onTop ? 'move' : 'centre', k: best};
                }
                var names = ['fit-region-lo', 'fit-region-hi'];
                for (var q = 0; q < 2; q++) {
                    var si = shapeIdx(names[q]);
                    if (si >= 0 && Math.abs(L.xaxis.d2p(gd.layout.shapes[si].x0) - p.px) < 6)
                        return {kind: 'region', which: q};
                }
                return null;
            }
            var drag = null;
            gd.addEventListener('mousedown', function(e) {
                if (e.button !== 0) return;
                var h = hit(e);
                if (!h) return;
                e.stopPropagation(); e.preventDefault();      // no pan
                window._theiaSkipClick = true;
                if (h.kind === 'region') {
                    drag = {kind: 'region', which: h.which, x0: e.clientX, y0: e.clientY,
                            lo: gd.layout.shapes[shapeIdx('fit-region-lo')].x0,
                            hi: gd.layout.shapes[shapeIdx('fit-region-hi')].x0};
                    return;
                }
                var it = traceIdx('fit'), ic = traceIdx('fit components'),
                    ib = traceIdx('fit background'), ip = traceIdx('fit peaks'),
                    ih = traceIdx('fit width handles');
                var X = (gd.data[it].x || []).map(Number), T = (gd.data[it].y || []).map(Number);
                var n = X.length, k = h.k;
                var cy = gd.data[ic].y, seg = k * (n + 1);       // segments split by null
                var B = (gd.data[ib].x && gd.data[ib].x.length) ? gd.data[ib].y.map(Number)
                                                               : X.map(function() { return 0; });
                var C = [];
                for (var j = 0; j < n; j++) C.push(Number(cy[seg + j]) - B[j]);
                var par = (gd.data[ip].customdata || [])[k];
                if (!par) { drag = null; return; }
                drag = {kind: h.kind, side: h.side, k: k, x0: e.clientX, y0: e.clientY,
                        X: X, T: T, C: C, B: B, seg: seg, n: n, it: it, ic: ic, ip: ip, ih: ih,
                        cx: gd.data[ic].x.slice(), cy: cy.slice(),
                        mx: gd.data[ip].x.slice(), my: gd.data[ip].y.slice(),
                        hx: gd.data[ih].x.slice(), hy: gd.data[ih].y.slice(),
                        c0: Number(par[0]), a0: Number(par[1]), w0: Number(par[2]),
                        eta: Number(par[3]), code: Number(par[4])};
                drag.c = drag.c0; drag.a = drag.a0; drag.w = drag.w0;
            }, true);
            // redraw at most once per animation frame while dragging
            var pending = null;
            document.addEventListener('mousemove', function(e) {
                if (!drag) return;
                var first = (pending === null);
                pending = [e.clientX, e.clientY];
                if (first) requestAnimationFrame(function() {
                    var q = pending; pending = null;
                    if (drag) dragTo(drag, q[0], q[1]);
                });
            });
            function deltas(d, clientX, clientY, xRef, yRef) {
                // pixel offsets -> data offsets, measured from a reference point
                var L = gd._fullLayout;
                return [L.xaxis.p2d(L.xaxis.d2p(xRef) + (clientX - d.x0)) - xRef,
                        L.yaxis.p2d(L.yaxis.d2p(yRef) + (clientY - d.y0)) - yRef];
            }
            function dragTo(d, clientX, clientY) {
                if (d.kind === 'region') {
                    var L = gd._fullLayout, r = gd.getBoundingClientRect();
                    var x = L.xaxis.p2d(clientX - r.left - L.xaxis._offset);
                    if (d.which === 0) d.lo = x; else d.hi = x;
                    var upd = {}, ilo = shapeIdx('fit-region-lo'), ihi = shapeIdx('fit-region-hi'),
                        it = shapeIdx('fit-region-tint');
                    upd['shapes[' + ilo + '].x0'] = d.lo; upd['shapes[' + ilo + '].x1'] = d.lo;
                    upd['shapes[' + ihi + '].x0'] = d.hi; upd['shapes[' + ihi + '].x1'] = d.hi;
                    if (it >= 0) { upd['shapes[' + it + '].x0'] = Math.min(d.lo, d.hi);
                                   upd['shapes[' + it + '].x1'] = Math.max(d.lo, d.hi); }
                    Plotly.relayout(gd, upd);
                    return;
                }
                var k = d.k;
                if (d.kind === 'width') {
                    var xh = d.c0 + d.side * d.w0 / 2;
                    var dx = deltas(d, clientX, clientY, xh, 0)[0];
                    d.w = Math.max(2 * d.side * (xh + dx - d.c0), 1e-3);
                } else {
                    var dd = deltas(d, clientX, clientY, d.mx[k], d.my[k]);
                    d.c = d.c0 + dd[0];
                    if (d.kind === 'move') d.a = Math.max(d.a0 + dd[1], 0);
                }
                var X = d.X, n = d.n, newT = [], cy = d.cy.slice();
                for (var j = 0; j < n; j++) {
                    var c = d.a * profile(d.code, X[j] - d.c, d.w, d.eta);
                    newT.push(d.T[j] - d.C[j] + c);
                    cy[d.seg + j] = c + d.B[j];
                }
                var mx = d.mx.slice(), my = d.my.slice(), hx = d.hx.slice(), hy = d.hy.slice();
                mx[k] = d.c; my[k] = interp(X, newT, d.c);
                hx[2 * k] = d.c - d.w / 2; hx[2 * k + 1] = d.c + d.w / 2;
                hy[2 * k] = interp(X, d.B, hx[2 * k]) + d.a / 2;
                hy[2 * k + 1] = interp(X, d.B, hx[2 * k + 1]) + d.a / 2;
                Plotly.restyle(gd, {x: [X, d.cx, mx, hx], y: [newT, cy, my, hy]},
                               [d.it, d.ic, d.ip, d.ih]);
            }
            document.addEventListener('mouseup', function(e) {
                if (!drag) return;
                var d = drag; drag = null;
                var moved = Math.abs(e.clientX - d.x0) >= 2 || Math.abs(e.clientY - d.y0) >= 2;
                if (!moved) {
                    // energy mode: a tap on a fitted peak picks that peak
                    if (window._theiaEnergyMode && (d.kind === 'move' || d.kind === 'centre')) {
                        var L0 = gd._fullLayout;
                        window.dash_clientside.set_props('energy-click', {data: {
                            x: d.mx[d.k], xrange: [Number(L0.xaxis.range[0]), Number(L0.xaxis.range[1])],
                            t: Date.now()}});
                    }
                    return;
                }
                dragTo(d, e.clientX, e.clientY);                 // final position
                if (d.kind === 'region') {
                    window.dash_clientside.set_props('fit-region-set', {data: {
                        xrange: [Math.min(d.lo, d.hi), Math.max(d.lo, d.hi)], t: Date.now()}});
                    return;
                }
                var out = {peak: d.k, t: Date.now()};
                if (d.kind === 'width') out.fwhm = d.w;
                else out.centre = d.c;
                if (d.kind === 'move') out.amp = d.a;
                window.dash_clientside.set_props('fit-drag', {data: out});
            });
            // cursor over a draggable element
            gd.addEventListener('mousemove', function(e) {
                if (drag) return;
                var box = document.getElementById('sum-graph');
                if (!box) return;
                var h = hit(e);
                box.classList.toggle('fit-grab', !!h && (h.kind === 'centre' || h.kind === 'width'));
                box.classList.toggle('fit-grab-move', !!h && h.kind === 'move');
                box.classList.toggle('fit-grab-region', !!h && h.kind === 'region');
            });

            gd.addEventListener('click', function(e) {
                if (window._theiaSkipClick) { window._theiaSkipClick = false; return; }
                if (!(window._theiaFitMode || window._theiaEnergyMode) || !down) return;
                if (Math.abs(e.clientX - down[0]) > 4 || Math.abs(e.clientY - down[1]) > 4) return;
                var L = gd._fullLayout;
                if (!L || !L.xaxis || !L.yaxis) return;
                var r = gd.getBoundingClientRect();
                var px = e.clientX - r.left - L.xaxis._offset;
                var py = e.clientY - r.top - L.yaxis._offset;
                if (px < 0 || px > L.xaxis._length || py < 0 || py > L.yaxis._length) return;
                var x = L.xaxis.p2d(px);
                if (!isFinite(x)) return;
                window.dash_clientside.set_props(
                    window._theiaFitMode ? 'fit-click' : 'energy-click', {data: {
                    x: x, xrange: [Number(L.xaxis.range[0]), Number(L.xaxis.range[1])],
                    t: Date.now()}});
            });
        }, 200);
        return window.dash_clientside.no_update;
    }
    """,
    Output("_fit-mode-hook", "children"),
    Input("fit-mode", "data"),
    Input("energy-mode", "data"),
    Input("sum-graph", "figure"),
)

# Current zoom of the summed plot (saved with the Further output state)
# (read shortly after a zoom/pan or a new figure, once Plotly has drawn it)
app.clientside_callback(
    """
    function(rd, fig) {
        setTimeout(function() {
            var gd = document.querySelector('#sum-graph .js-plotly-plot');
            if (!gd || !gd._fullLayout || !gd._fullLayout.xaxis || !gd.data || !gd.data.length)
                return;
            var L = gd._fullLayout;
            var v = {x: [Number(L.xaxis.range[0]), Number(L.xaxis.range[1])],
                     y: [Number(L.yaxis.range[0]), Number(L.yaxis.range[1])]};
            var key = JSON.stringify(v);
            if (key === window._theiaLastView) return;
            window._theiaLastView = key;
            window.dash_clientside.set_props('sum-view', {data: v});
        }, 400);
        return window.dash_clientside.no_update;
    }
    """,
    Output("_sum-view-hook", "children"),
    Input("sum-graph", "relayoutData"),
    Input("sum-graph", "figure"),
    prevent_initial_call=True,
)


@app.callback(
    Output("_further-saved", "children"),
    Input("further-files", "data"),
    Input("sum-align", "value"),
    Input("sum-ion", "value"),
    Input("sum-unsure", "value"),
    Input("sum-key", "data"),
    Input("sum-view", "data"),
    Input("fit-state", "data"),
    Input("energy-state", "data"),
    Input("energy-name-1", "value"),
    Input("energy-name-2", "value"),
    Input("pid-settings", "data"),
    Input("further-source", "value"),
    Input("fsrc-dir", "value"),
    Input("fsrc-name", "value"),
    prevent_initial_call=True,
)
def save_further_state(files, align, ion, unsure, key, view, fit, energy, name1, name2, pid,
                       source, f_dir, f_name):
    """Save the Further output tab after every change, so it is restored the
    next time Theia runs (scanner_config.json -> further_state)."""
    trig = dash.callback_context.triggered[0]["prop_id"] if dash.callback_context.triggered else ""
    old = load_config().get("further_state") or {}
    st = {"files": files or [], "align": align, "ion": ion, "unsure": unsure or [],
          "summed": key is not None, "view": view or old.get("view"), "fit": old.get("fit"),
          "energy": old.get("energy"), "pid": pid,
          # chosen data source and single file; "plot" = what the plot shows
          "source": source or "sum", "file_dir": f_dir, "file_name": f_name,
          "plot": _SUM_META.get(key) if key is not None else old.get("plot")}
    if trig.startswith("sum-key"):
        st["view"] = view if view else old.get("view")
    if trig.startswith("fit-state"):
        st["fit"] = ({k: v for k, v in fit.items() if k != "sum_key"}
                     if fit and fit.get("peaks") else None)
    if trig.startswith("energy-state") or trig.startswith("energy-name"):
        has = energy and (energy.get("calib") or energy.get("pairs") or energy.get("k") is not None)
        st["energy"] = ({"state": {k: v for k, v in energy.items() if k != "msg"},
                         "names": [name1 or "", name2 or ""]}
                        if has or name1 or name2 else None)
    if key is None and not trig.startswith("sum-key"):
        # page just loaded and the restored sum is not back yet: keep it
        st["summed"] = old.get("summed", False)
        st["plot"] = old.get("plot")
    save_config({"further_state": st})
    return dash.no_update


# Show / hide the simulated lines on the summed plot without a redraw
app.clientside_callback(
    """
    function(value) {
        var gd = document.querySelector('#sum-graph .js-plotly-plot');
        if (!gd || !gd.data) return window.dash_clientside.no_update;
        var idx = [];
        gd.data.forEach(function(t, i) { if (t.meta === 'sim') idx.push(i); });
        if (idx.length) Plotly.restyle(gd, {visible: (value && value.length) ? true : false}, idx);
        return window.dash_clientside.no_update;
    }
    """,
    Output("_sum-sims-hook", "children"),
    Input("sum-show-sims", "value"),
    prevent_initial_call=True,
)


# "Fit": pass the current view range (the fit region) to the server
app.clientside_callback(
    """
    function(n) {
        var gd = document.querySelector('#sum-graph .js-plotly-plot');
        if (!n || !gd || !gd._fullLayout || !gd._fullLayout.xaxis)
            return window.dash_clientside.no_update;
        var R = gd._fullLayout.xaxis.range;
        return {xrange: [Number(R[0]), Number(R[1])], t: Date.now()};
    }
    """,
    Output("fit-run", "data"),
    Input("btn-fit-run", "n_clicks"),
    prevent_initial_call=True,
)


# "Region = view": the current zoom range becomes the fit region
app.clientside_callback(
    """
    function(n) {
        var gd = document.querySelector('#sum-graph .js-plotly-plot');
        if (!n || !gd || !gd._fullLayout || !gd._fullLayout.xaxis)
            return window.dash_clientside.no_update;
        var R = gd._fullLayout.xaxis.range;
        return {xrange: [Number(R[0]), Number(R[1])], t: Date.now()};
    }
    """,
    Output("fit-region-set", "data"),
    Input("btn-fit-region-view", "n_clicks"),
    prevent_initial_call=True,
)


@app.callback(
    Output("fit-state", "data"),
    Output("fit-table", "data"),
    Output("fit-table", "columns"),
    Output("fit-b0", "value"),
    Output("fit-b1", "value"),
    Output("fit-b0-unc", "children"),
    Output("fit-b1-unc", "children"),
    Output("fit-status", "children"),
    Output("fit-panel", "style"),
    Output("fit-table", "active_cell"),
    Output("fit-table", "selected_cells"),
    Output("fit-region-lo", "value"),
    Output("fit-region-hi", "value"),
    Output("fit-bg-fix", "value"),
    Input("fit-click", "data"),
    Input("fit-drag", "data"),
    Input("fit-run", "data"),
    Input("fit-table", "data_timestamp"),
    Input("fit-table", "active_cell"),
    Input("fit-shape", "value"),
    Input("fit-bg", "value"),
    Input("fit-b0", "value"),
    Input("fit-b1", "value"),
    Input("fit-bg-fix", "value"),
    Input("fit-region-set", "data"),
    Input("fit-region-lo", "value"),
    Input("fit-region-hi", "value"),
    Input("fit-show-limits", "value"),
    Input("btn-fit-clear", "n_clicks"),
    Input("sum-key", "data"),
    Input("iso-names-apply", "data"),
    State("fit-state", "data"),
    State("fit-table", "data"),
    prevent_initial_call=True,
)
def update_fit(click, dragged, run, _ts, active, shape, bg, b0, b1, bg_fix, region_set,
               region_lo, region_hi, show_limits, _clear, key, names_apply, state, table):
    """Every change to the fit goes through here (one callback, because these
    outputs also feed its inputs): clicks add peaks, table / background edits
    change the curve by hand, Fit runs the least-squares fit."""
    trig = dash.callback_context.triggered[0]["prop_id"] if dash.callback_context.triggered else ""
    if trig.startswith("fit-table.active_cell"):
        # a click on a "fix" cell toggles that parameter; any other cell: nothing
        col = (active or {}).get("column_id") or ""
        if not col.startswith("fix_") or not state or not state.get("peaks"):
            raise PreventUpdate
    with _SUM_CACHE_LOCK:
        cached = _SUM_CACHE.get(key)
    error = None
    restored = _RESTORE_FIT.pop(key, None) if trig.startswith("sum-key") else None
    if cached is None:
        state = None
    elif restored:
        state = dict(restored, sum_key=key)              # fit saved last session
    elif (not state or state.get("sum_key") != key or trig.startswith("btn-fit-clear")
          or trig.startswith("sum-key")):
        state = {"sum_key": key, "peaks": [], "bgp": [0.0, 0.0], "region": None,
                 "xref": 0.0, "fitted": False}
    if state is not None:
        ff, pp, sigma = cached
        state = dict(state, shape=shape, bg=bg)
        state["bgp"] = list(state.get("bgp") or [0.0, 0.0])
        try:
            if trig.startswith("fit-click") and click:
                if not state["peaks"]:
                    if not state.get("region"):           # region: the view, unless set
                        _set_region(state, click["xrange"])
                    # background start value: median of the data in the region
                    m = _region_mask(ff, state["region"])
                    state["bgp"] = [float(np.nanmedian(pp[m])) if m.any() else 0.0, 0.0]
                new_pk = _snap_peak(ff, pp, click["x"], state["bgp"], bg)
                new_pk["shape"] = shape              # shape chosen for new peaks
                state["peaks"] = state["peaks"] + [new_pk]
                state["fitted"], state["unc"] = False, None
            elif trig.startswith("fit-drag") and dragged:
                k = int(dragged["peak"])
                if 0 <= k < len(state["peaks"]):
                    state["peaks"] = [dict(pk) for pk in state["peaks"]]
                    pk = state["peaks"][k]
                    for what in ("centre", "amp", "fwhm"):     # whichever were dragged
                        if dragged.get(what) is not None:
                            pk[what] = float(dragged[what])
                    pk["amp"], pk["fwhm"] = max(pk["amp"], 0.0), max(abs(pk["fwhm"]), 1e-6)
                    state["fitted"], state["unc"] = False, None
            elif trig.startswith("fit-run") and run:
                if not state["peaks"]:
                    raise ValueError("add at least one peak first (turn on Peak fitting, then "
                                     "click on a peak)")
                if not state.get("region"):
                    _set_region(state, run["xrange"])
                state = run_fit(ff, pp, sigma, state)
            elif trig.startswith("fit-region-set") and region_set:
                _set_region(state, region_set["xrange"])     # dragged lines / "Region = view"
                state["fitted"], state["unc"] = False, None
            elif trig.startswith("fit-region-lo") or trig.startswith("fit-region-hi"):
                if region_lo is not None and region_hi is not None:
                    _set_region(state, [region_lo, region_hi])
                    state["fitted"], state["unc"] = False, None
            elif trig.startswith("fit-table.active_cell"):
                k, what = int(active["row"]), active["column_id"][4:]
                if 0 <= k < len(state["peaks"]) and (
                        what != "eta" or _shapes(state)[k] == "pvoigt"):
                    state["peaks"] = [dict(pk) for pk in state["peaks"]]
                    fix = set(state["peaks"][k].get("fix") or [])
                    fix ^= {what}
                    state["peaks"][k]["fix"] = sorted(fix)
            elif trig.startswith("fit-bg-fix"):
                state["bg_fix"] = list(bg_fix or [])
            elif trig.startswith("iso-names-apply"):
                # names / harmonics given in the "Send to isocurve" popup
                ok = bool(names_apply) and names_apply.get("sum_key") == key
                state["peaks"] = [dict(pk) for pk in state["peaks"]]
                for k, nm in ((names_apply.get("names") or {}) if ok else {}).items():
                    k = int(k)
                    if 0 <= k < len(state["peaks"]):
                        state["peaks"][k]["ion"] = nm["ion"]
                        state["peaks"][k]["harmonic"] = nm["harmonic"]
            elif trig.startswith("fit-table") and _only_names_changed(state, table):
                # ion / harmonic typed in the table: keep the fit and its
                # uncertainties (nothing else changed)
                state["peaks"] = [dict(pk) for pk in state["peaks"]]
                for pk, r in zip(state["peaks"], table):
                    pk["ion"] = str(r.get("ion") or "").strip()
                    pk["harmonic"] = _harmonic_or_none(r.get("harmonic"))
            elif trig.startswith("fit-table"):
                # hand edits (or deleted rows): rebuild the peaks from the table
                peaks = []
                for r in table or []:
                    n = int(r.get("peak") or 0)
                    old = state["peaks"][n - 1] if 0 < n <= len(state["peaks"]) else {}
                    try:
                        eta = r.get("eta")
                        shp = r.get("shape") if r.get("shape") in SHAPE_LABELS else \
                            old.get("shape", shape)
                        pk = {"centre": float(r["centre"]), "amp": float(r["amp"]),
                              "fwhm": abs(float(r["fwhm"])), "shape": shp,
                              "eta": float(eta) if eta not in (None, "") else old.get("eta", 0.5),
                              "fix": list(old.get("fix") or []),
                              "ion": str(r.get("ion") or "").strip(),
                              "harmonic": _harmonic_or_none(r.get("harmonic")),
                              "lim": {k: (_num_or_none(r[k]) if k in r
                                          else (old.get("lim") or {}).get(k))
                                      for k in LIMIT_KEYS}}
                        pk["eta"] = min(max(pk["eta"], 0.0), 1.0)
                    except (TypeError, ValueError, KeyError):
                        pk = dict(old)
                    if pk:
                        peaks.append(pk)
                state["peaks"], state["fitted"], state["unc"] = peaks, False, None
            elif trig.startswith("fit-b0") or trig.startswith("fit-b1"):
                if b0 is not None:
                    state["bgp"][0] = float(b0)
                if b1 is not None:
                    state["bgp"][1] = float(b1)
                state["fitted"], state["unc"] = False, None
            elif trig.startswith("fit-bg"):
                state["fitted"], state["unc"] = False, None
            # (fit-shape only sets the shape for peaks added from now on)
        except Exception as exc:
            traceback.print_exc()
            error = str(exc)

    # eta columns only when at least one peak is pseudo-Voigt
    pv = bool(state) and "pvoigt" in _shapes(state)
    cols = fit_columns(pv, bool(show_limits))
    nu = dash.no_update
    if state is None:
        msg = ("Sum the spectra first, then turn on peak fitting."
               if trig.startswith(("fit-click", "fit-run")) else "")
        return None, [], cols, None, None, "", "", msg, FIT_PANEL_HIDDEN, None, [], None, None, []
    # fix-cell clicks: deselect the cell so the next click toggles again
    act, sel = (None, []) if trig.startswith("fit-table.active_cell") else (nu, nu)
    reg = state.get("region")
    rlo = round(reg[0], 1) if reg else None
    rhi = round(reg[1], 1) if reg else None
    nb = _n_bg(state["bg"])
    bunc = (state.get("unc") or {}).get("bgp") or [None, None]
    b0v = float(f"{state['bgp'][0]:.6g}") if nb >= 1 else None
    b1v = float(f"{state['bgp'][1]:.6g}") if nb == 2 else None
    status = _fit_status(state)
    if error:
        status = html.Span(f"Fit error: {error}", style={"color": "#c0392b"})
    panel = FIT_PANEL_SHOWN if state["peaks"] else FIT_PANEL_HIDDEN
    bfix = set(state.get("bg_fix") or [])

    def bunc_txt(j):
        if j == 0 and nb < 1 or j == 1 and nb < 2:
            return ""
        if f"b{j}" in bfix:
            return "fixed"
        return ("± " + _fmt_num(bunc[j], ".3g")) if bunc[j] is not None else ""
    return (state, fit_table_rows(state), cols, b0v, b1v, bunc_txt(0), bunc_txt(1),
            status, panel, act, sel, rlo, rhi, sorted(bfix))


@app.callback(
    Output("fit-resid", "figure"),
    Output("fit-resid", "style"),
    Input("fit-state", "data"),
    State("sum-key", "data"),
)
def show_residuals(state, key):
    hidden = {"height": "200px", "display": "none"}
    with _SUM_CACHE_LOCK:
        cached = _SUM_CACHE.get(key)
    if cached is None or not state or not state.get("peaks") or not state.get("region"):
        return empty_fig(), hidden
    ff, pp, sigma = cached
    m = _region_mask(ff, state["region"])
    x, y = ff[m], pp[m]
    p, _ = _pack(state)
    model = _model_from_vector(x - state["xref"], p, _shapes(state), state["bg"])
    fig = go.Figure()
    fig.add_hrect(y0=-sigma, y1=sigma, fillcolor=ACCENT_PALE, opacity=0.9, line_width=0,
                  layer="below")
    fig.add_hline(y=0, line=dict(color="#999", width=1))
    fig.add_trace(go.Scatter(x=x, y=y - model, mode="lines", name="data − fit",
                             line=dict(shape="hvh", color="black", width=1),
                             hovertemplate="f = %{x:.1f} Hz<br>data − fit = %{y:.4g}<extra></extra>"))
    fig.update_layout(template="plotly_white", margin=dict(l=60, r=20, t=30, b=40),
                      title=dict(text="Residuals (data − fit); shaded band: ±1σ noise",
                                 font=dict(size=13)),
                      xaxis_title="Frequency (Hz)", yaxis_title="data − fit", dragmode="pan",
                      showlegend=False, font=dict(size=12), uirevision=str(state.get("region")))
    return fig, {"height": "200px", "display": "block"}


# Snippet: show/hide the half-life-excluded lines in the browser, so the
# current zoom/pan is kept (the next snippet drawn uses the same setting)
app.clientside_callback(
    """
    function(value) {
        var gd = document.querySelector('#snippet-graph .js-plotly-plot');
        if (!gd || !gd.data) return window.dash_clientside.no_update;
        var idx = [];
        gd.data.forEach(function(t, i) { if (t.name === '%s') idx.push(i); });
        if (idx.length) Plotly.restyle(gd, {visible: (value && value.length) ? true : false}, idx);
        return window.dash_clientside.no_update;
    }
    """ % EXCLUDED_TRACE,
    Output("_snippet-excluded-hook", "children"),
    Input("snippet-show-excluded", "value"),
    prevent_initial_call=True,
)


@app.callback(
    Output("further-page-align", "style"),
    Output("further-page-iso", "style"),
    Input("further-subtabs", "value"),
)
def show_further_page(page):
    base = {"padding": "14px 18px", "overflowY": "auto", "height": "calc(100vh - 82px)"}
    return ({**base, "display": "block" if page == "align" else "none"},
            {**base, "display": "block" if page == "iso" else "none"})


# ---------------------------------------------------------------------------
# Search ions (as in the spectrogram) and the h−1 / h+1 buttons
# ---------------------------------------------------------------------------

def _srch_half_width_hz(width_khz):
    try:
        v = float(width_khz)
    except (TypeError, ValueError):
        return SRCH_DEFAULT_KHZ * 1000.0
    return v * 1000.0 if np.isfinite(v) and v > 0 else SRCH_DEFAULT_KHZ * 1000.0


def _srch_lines(key):
    sims = _SUM_SIMS.get(key)
    if sims is None or not len(sims):
        return None
    if "t_half" not in sims:                 # (older cache entry)
        sims = sims.assign(t_half="", excluded=False, is_ref=False)
    return sims


@app.callback(
    Output("srch-modal", "style"),
    Input("btn-srch", "n_clicks"),
    Input("srch-close", "n_clicks"),
    Input("srch-zoom", "data"),
    prevent_initial_call=True,
)
def toggle_search_modal(_open, _close, zoom):
    trig = dash.callback_context.triggered[0]["prop_id"] if dash.callback_context.triggered else ""
    if trig.startswith("btn-srch"):
        return SRCH_MODAL_SHOWN
    if trig.startswith("srch-zoom") and not (zoom or {}).get("from_list"):
        raise PreventUpdate                       # h±1 buttons: modal stays as it is
    return SRCH_MODAL_HIDDEN


@app.callback(
    Output("srch-list", "children"),
    Output("srch-status", "children"),
    Output("srch-hits", "data"),
    Output("srch-width", "style"),
    Input("srch-ion", "value"),
    Input("srch-width", "value"),
    Input("sum-key", "data"),
)
def search_ions(q, width, key):
    """Simulated lines whose ion name contains the query (case-insensitive),
    sorted by frequency (first 300 shown)."""
    w_style = {"width": "80px", "marginRight": "4px"}
    try:
        wv = float(width) if width not in (None, "") else None
    except (TypeError, ValueError):
        wv = -1
    if wv is not None and not (np.isfinite(wv) and wv > 0):
        w_style["borderColor"] = "#c0392b"
    trig = dash.callback_context.triggered[0]["prop_id"] if dash.callback_context.triggered else ""
    if trig.startswith("srch-width") and wv is not None and wv > 0:
        save_config({"search_width_khz": wv})
    q = (q or "").strip().lower()
    if not q:
        return [], "", [], w_style
    sims = _srch_lines(key)
    if sims is None:
        return [], html.Span("✗ no simulated lines: sum the spectra / load a spectrum first",
                             style={"color": "#c0392b"}), [], w_style
    hit = sims[sims["ion"].astype(str).str.lower().str.contains(q, regex=False)]
    hit = hit.sort_values("f_sim")
    n = len(hit)
    if not n:
        return [], html.Span(f"No simulated lines for \"{q}\" in this spectrum",
                             style={"color": "#c0392b"}), [], w_style
    hits = [{"ion": str(r.ion), "h": int(r.harmonic), "f": float(r.f_sim),
             "t_half": r.t_half or "", "excluded": bool(r.excluded), "is_ref": bool(r.is_ref)}
            for r in hit.head(300).itertuples()]
    hw = _srch_half_width_hz(width)
    cell = {"padding": "3px 12px", "fontSize": "13px"}
    head = {**cell, "fontSize": "11px", "fontWeight": "700", "color": "#888",
            "textTransform": "uppercase", "letterSpacing": "0.04em"}
    out = [html.Div([html.Span(t, style=head) for t in ("Ion", "Harmonic", "f_sim (Hz)", "T½")],
                    style={**SRCH_GRID, "borderBottom": "1px solid #dddddd", "position": "sticky",
                           "top": "0", "background": "white"}, className="theia-box")]
    for i, l in enumerate(hits):
        col = {"color": "#aaa"} if l["excluded"] else {}
        out.append(html.Div([
            html.Span(l["ion"] + ("  (ref)" if l["is_ref"] else ""),
                      style={**cell, **col, "fontWeight": "600"}),
            html.Span(f"h{l['h']}", style={**cell, **col}),
            html.Span(f"{l['f']:,.1f}", style={**cell, **col, "fontFamily": "monospace"}),
            html.Span(l["t_half"] or "?", style={**cell, **col}),
        ], id={"type": "srch-line", "i": i}, n_clicks=0, className="theia-fb-row",
           title=f"Zoom to {l['f'] - hw:,.0f} – {l['f'] + hw:,.0f} Hz"
                 + ("  (below the half-life cut)" if l["excluded"] else ""),
           style={**SRCH_GRID, "cursor": "pointer"}))
    status = html.Span(f"{n} line{'' if n == 1 else 's'}"
                       + (f" (first {len(hits)} shown)" if n > len(hits) else ""),
                       style={"color": "#2e7d32"})
    return out, status, hits, w_style


def _srch_neighbour(key, cur, step):
    """Simulated line of the current ion at harmonic h + step, or None."""
    sims = _srch_lines(key)
    if sims is None or not cur:
        return None
    m = sims[(sims["ion"].astype(str) == cur["ion"]) & (sims["harmonic"] == cur["h"] + step)]
    if not len(m):
        return None
    r = m.iloc[0]
    return {"ion": cur["ion"], "h": int(r["harmonic"]), "f": float(r["f_sim"])}


@app.callback(
    Output("srch-zoom", "data"),
    Output("srch-current", "data"),
    Output("harm-cur", "children"),
    Input({"type": "srch-line", "i": ALL}, "n_clicks"),
    Input("btn-harm-dn", "n_clicks"),
    Input("btn-harm-up", "n_clicks"),
    State("srch-hits", "data"),
    State("srch-current", "data"),
    State("srch-width", "value"),
    State("sum-key", "data"),
    prevent_initial_call=True,
)
def search_pick(_rows, _dn, _up, hits, cur, width, key):
    """A line from the list, or h−1 / h+1: zoom to f ± width."""
    trig = dash.callback_context.triggered[0] if dash.callback_context.triggered else None
    if not trig or not trig["value"]:
        raise PreventUpdate
    pid = trig["prop_id"]
    from_list = '"srch-line"' in pid
    if from_list:
        i = json.loads(pid.rsplit(".", 1)[0])["i"]
        if not hits or i >= len(hits):
            raise PreventUpdate
        line = hits[i]
    else:
        line = _srch_neighbour(key, cur, -1 if pid.startswith("btn-harm-dn") else 1)
        if line is None:
            raise PreventUpdate
    hw = _srch_half_width_hz(width)
    zoom = {"x": [line["f"] - hw, line["f"] + hw], "from_list": from_list, "t": time.time()}
    # the label is set here, in the same reply as the zoom, so it never lags
    return zoom, {"ion": line["ion"], "h": int(line["h"])}, f"{line['ion']} h{int(line['h'])}"


@app.callback(
    Output("harm-grp", "style"),
    Output("harm-cur", "title"),
    Output("btn-harm-dn", "disabled"),
    Output("btn-harm-dn", "title"),
    Output("btn-harm-up", "disabled"),
    Output("btn-harm-up", "title"),
    Input("srch-current", "data"),
    Input("sum-key", "data"),
    Input("srch-width", "value"),
)
def show_harmonic_step(cur, key, width):
    if not cur:
        return {"display": "none"}, "", True, "", True, ""
    hw = _srch_half_width_hz(width)
    out = []
    for step in (-1, 1):
        l = _srch_neighbour(key, cur, step)
        if l is None:
            out += [True, f"h{cur['h'] + step} of {cur['ion']} is not among the simulated "
                          f"lines of this spectrum"]
        else:
            out += [False, f"Zoom to {cur['ion']}, h{l['h']}: {l['f'] - hw:,.0f} – "
                           f"{l['f'] + hw:,.0f} Hz"]
    return ({"display": "inline-flex", "alignItems": "center", "marginRight": "8px"},
            "Ion and harmonic chosen in Search ions", *out)


# Zoom the summed plot to the chosen window; y fitted to the data inside it
app.clientside_callback(
    """
    function(z) {
        var gd = document.querySelector('#sum-graph .js-plotly-plot');
        if (!z || !gd || !gd._fullData || !gd._fullData.length)
            return window.dash_clientside.no_update;
        var lo = z.x[0], hi = z.x[1], X = gd._fullData[0].x, Y = gd._fullData[0].y;
        var ymin = Infinity, ymax = -Infinity;
        for (var i = 0; i < X.length; i++) {
            if (X[i] < lo || X[i] > hi) continue;
            var y = Y[i];
            if (isFinite(y)) { if (y < ymin) ymin = y; if (y > ymax) ymax = y; }
        }
        var upd = {'xaxis.range': [lo, hi]};
        if (isFinite(ymin) && isFinite(ymax)) {
            var pad = 0.05 * ((ymax - ymin) || Math.abs(ymax) || 1);
            upd['yaxis.range'] = [ymin - pad, ymax + pad];
        } else {
            upd['yaxis.autorange'] = true;
        }
        Plotly.relayout(gd, upd);
        return window.dash_clientside.no_update;
    }
    """,
    Output("_srch-zoom-hook", "children"),
    Input("srch-zoom", "data"),
    prevent_initial_call=True,
)


# ---------------------------------------------------------------------------
# Isochronicity curve: points sent from the peak fitting
# ---------------------------------------------------------------------------
# One row per (ion, harmonic) in outputs/isocurve/isocurve_points.csv of the
# active experiment.  Sending the same ion and harmonic again replaces the row.

ISO_POINTS_NAME = "isocurve_points.csv"
ISO_FILE_COLUMNS = ["ion", "harmonic", "A", "Z", "q", "isomer",
                    "centre_hz", "d_centre_hz", "f_rev_hz", "d_f_rev_hz",
                    "fwhm_hz", "d_fwhm_hz", "amp", "d_amp", "area", "d_area",
                    "shape", "eta", "d_eta", "fixed", "chi2_ndf", "ndf",
                    "region_lo_hz", "region_hi_hz", "background", "source", "sent",
                    "include"]
ION_NAME_RE = re.compile(r"\s*([a-z]?)(\d+)([A-Z][a-z]?)(\d+)\+\s*")


def iso_points_path():
    return os.path.join(paths.outputs("isocurve"), ISO_POINTS_NAME)


def load_iso_points():
    path = iso_points_path()
    if not os.path.exists(path):
        return pd.DataFrame(columns=ISO_FILE_COLUMNS)
    df = pd.read_csv(path)
    for c in ISO_FILE_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    return df


def save_iso_points(df):
    """Write the points file (via a temporary file, so a crash never leaves
    half a file), sorted by f_rev."""
    path = iso_points_path()
    df = df.sort_values("f_rev_hz", kind="stable") if len(df) else df
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def parse_ion_name(name):
    """Ion name with charge ('207Tl81+', 'm207Tl81+'), checked against NUBASE.
    Returns dict(ion, A, Z, q, iso) or raises ValueError."""
    m = ION_NAME_RE.fullmatch(name or "")
    if not m:
        raise ValueError("give the ion with its charge, e.g. 207Tl81+ or m207Tl81+")
    iso, A, sym, q = m.group(1), m.group(2), m.group(3), int(m.group(4))
    r = lookup_mass_excess(f"{iso}{A}{sym}")
    if not 1 <= q <= r["Z"]:
        raise ValueError(f"charge {q}+ is not possible for {sym} (Z = {r['Z']})")
    return dict(ion=f"{iso}{A}{sym}{q}+", A=r["A"], Z=r["Z"], q=q, iso=iso)


def _check_peak_name(ion, harmonic):
    """(parsed dict, None) if the ion name and harmonic are valid, else
    (None, reason)."""
    if not (ion or "").strip():
        return None, "no ion name"
    try:
        r = parse_ion_name(ion)
    except ValueError as exc:
        return None, str(exc)
    h = _harmonic_or_none(harmonic)
    if h is None:
        return None, "no harmonic (a whole number ≥ 1)"
    return dict(r, harmonic=h), None


def _iso_source(key, files, align, ion_key):
    """Short description of the spectrum the fit was made on."""
    meta = _SUM_META.get(key) or {}
    if meta.get("source") == "file":
        return f"file {os.path.join(meta.get('dir') or '', meta.get('name') or '')}"
    names = [os.path.basename(f.get("path", f) if isinstance(f, dict) else str(f))
             for f in (files or [])]
    how = {"none": "not aligned", "strongest": "aligned on strongest peak"}.get(align)
    if align == "identified" and ion_key:
        ion, harm = str(ion_key).rsplit("|", 1)
        how = f"aligned on {ion} h={harm}"
    return f"sum of {', '.join(names) or 'confirmation files'} ({how or align})"


def iso_points_from_fit(state, named, source):
    """One point per fitted peak.  named: peak index -> parsed name dict."""
    unc = (state.get("unc") or {}).get("peaks") or [{} for _ in state["peaks"]]
    shapes = _shapes(state)
    now = datetime.datetime.now().isoformat(timespec="seconds")
    rows = []
    for i, (pk, e, shape) in enumerate(zip(state["peaks"], unc, shapes)):
        nm = named[i]
        fix = set(pk.get("fix") or [])

        def u(k):
            v = e.get(k)
            return 0.0 if k in fix else (float(v) if v is not None else np.nan)
        h = nm["harmonic"]
        pv = shape == "pvoigt"
        rows.append({
            "ion": nm["ion"], "harmonic": h, "A": nm["A"], "Z": nm["Z"], "q": nm["q"],
            "isomer": nm["iso"] or "",
            "centre_hz": pk["centre"], "d_centre_hz": u("centre"),
            "f_rev_hz": pk["centre"] / h, "d_f_rev_hz": u("centre") / h,
            "fwhm_hz": pk["fwhm"], "d_fwhm_hz": u("fwhm"),
            "amp": pk["amp"], "d_amp": u("amp"),
            "area": _peak_area(pk["amp"], pk["fwhm"], pk.get("eta", 0.5), shape),
            "d_area": float(e.get("area")) if e.get("area") is not None else np.nan,
            "shape": shape, "eta": pk.get("eta", 0.5) if pv else np.nan,
            "d_eta": u("eta") if pv else np.nan,
            "fixed": " ".join(sorted(k for k in fix if k != "eta" or pv)),
            "chi2_ndf": state.get("chi2"), "ndf": state.get("ndf"),
            "region_lo_hz": state["region"][0], "region_hi_hz": state["region"][1],
            "background": state.get("bg"), "source": source, "sent": now,
            "include": True,
        })
    return pd.DataFrame(rows, columns=ISO_FILE_COLUMNS)


def _iso_modal_body(state, key, missing, typed, errors, replaces):
    """Popup content: an input row (name, harmonic, suggestions) for each peak
    without a valid name, and the points that would be replaced."""
    out = []
    if missing:
        out.append(html.Div("Each peak sent to the isochronicity curve needs an ion name with "
                            "its charge (e.g. 207Tl81+ or m207Tl81+) and a harmonic.  Click a "
                            "suggestion (nearby simulated ions) or type them.",
                            style={"marginBottom": "8px"}))
        th = {**CALIB_TH}
        rows = []
        for i in missing:
            pk = state["peaks"][i]
            name, h = typed.get(i, (pk.get("ion") or "", pk.get("harmonic")))
            sugg = suggest_ions(key, pk["centre"], n=5)
            chips = [html.Button(f"{ion} h{hh} ({d:+.0f} Hz)",
                                 id={"type": "iso-sugg", "peak": i, "name": ion, "h": hh},
                                 n_clicks=0, className="btn btn-outline-secondary btn-sm",
                                 style={"margin": "0 4px 4px 0", "fontSize": "12px",
                                        "padding": "1px 6px"})
                     for ion, hh, d in sugg]
            err = errors.get(i)
            rows.append(html.Tr([
                html.Td(f"{i + 1}", style={**CALIB_TD, "fontWeight": "700", "color": ACCENT}),
                html.Td(f"{pk['centre']:,.1f}", style={**CALIB_TD, "fontFamily": "monospace"}),
                html.Td([
                    html.Div([
                        dcc.Input(id={"type": "iso-name", "peak": i}, type="text", value=name,
                                  placeholder="e.g. 207Tl81+", debounce=False,
                                  className="form-control form-control-sm",
                                  style={"width": "150px", "marginRight": "6px"}),
                        html.Span("h", style={"marginRight": "4px"}),
                        dcc.Input(id={"type": "iso-h", "peak": i}, type="number", value=h,
                                  min=1, step=1, debounce=False,
                                  className="form-control form-control-sm",
                                  style={"width": "70px"}),
                    ], style={"display": "flex", "alignItems": "center"}),
                    html.Div(f"✗ {err}", style={"color": "#c0392b", "fontSize": "12px",
                                                "marginTop": "3px"}) if err else None,
                ], style=CALIB_TD),
                html.Td(chips or html.Span("no simulated ions nearby",
                                           style={"color": "#888", "fontSize": "12px"}),
                        style=CALIB_TD),
            ]))
        out.append(html.Table([
            html.Thead(html.Tr([html.Th(t, style=th) for t in
                                ("Peak", "Centre (Hz)", "Ion and harmonic", "Suggestions")])),
            html.Tbody(rows)],
            style={"borderCollapse": "collapse", "border": CALIB_BORDER, "width": "100%"}))
    if replaces:
        out.append(html.Div([
            html.B("Already on the curve, will be replaced: ", style={"color": ACCENT}),
            ", ".join(replaces)], style={"marginTop": "10px"}))
    return out


@app.callback(
    Output("iso-modal", "style"),
    Output("iso-modal-body", "children"),
    Output("iso-send-status", "children"),
    Output("iso-names-apply", "data"),
    Output("iso-version", "data"),
    Input("btn-iso-send", "n_clicks"),
    Input("iso-confirm", "n_clicks"),
    Input("iso-cancel", "n_clicks"),
    Input("iso-cancel-x", "n_clicks"),
    State({"type": "iso-name", "peak": ALL}, "value"),
    State({"type": "iso-name", "peak": ALL}, "id"),
    State({"type": "iso-h", "peak": ALL}, "value"),
    State({"type": "iso-h", "peak": ALL}, "id"),
    State("fit-state", "data"),
    State("sum-key", "data"),
    State("further-files", "data"),
    State("sum-align", "value"),
    State("sum-ion", "value"),
    State("iso-version", "data"),
    prevent_initial_call=True,
)
def send_to_isocurve(_send, _confirm, _cancel, _x, name_vals, name_ids, h_vals, h_ids,
                     state, key, files, align, ion_key, version):
    """'Send to isocurve': every fitted peak needs a valid ion name and
    harmonic.  Missing ones (and replacing existing points) go through the
    popup; otherwise the points are written straight away."""
    trig = dash.callback_context.triggered[0]["prop_id"] if dash.callback_context.triggered else ""
    nu = dash.no_update
    if trig.startswith("iso-cancel"):
        return ISO_MODAL_HIDDEN, nu, "", nu, nu

    def fail(msg):
        return ISO_MODAL_HIDDEN, nu, html.Span(msg, style={"color": "#c0392b"}), nu, nu
    if not state or not state.get("peaks"):
        return fail("No peaks to send: add peaks and press Fit first.")
    if not state.get("fitted"):
        return fail("The peaks are not fitted (values changed since the last Fit): "
                    "press Fit first.")

    # names typed in the popup (only present when it is open)
    typed = {}
    if trig.startswith("iso-confirm"):
        hs = {d["peak"]: v for d, v in zip(h_ids or [], h_vals or [])}
        for d, v in zip(name_ids or [], name_vals or []):
            typed[d["peak"]] = ((v or "").strip(), hs.get(d["peak"]))

    named, errors = {}, {}
    for i, pk in enumerate(state["peaks"]):
        ion, h = typed.get(i, (pk.get("ion") or "", pk.get("harmonic")))
        r, err = _check_peak_name(ion, h)
        if r is None:
            errors[i] = err
        else:
            named[i] = r
    keys = [(r["ion"], r["harmonic"]) for r in named.values()]
    dup = sorted({f"{a} h{b}" for a, b in keys if keys.count((a, b)) > 1})
    try:
        old = load_iso_points()
    except Exception as exc:
        traceback.print_exc()
        return fail(f"Could not read {iso_points_path()}: {exc}")
    have = set(zip(old["ion"].astype(str), pd.to_numeric(old["harmonic"]).astype("Int64")))
    replaces = sorted({f"{a} h{b}" for a, b in keys if (a, b) in have})

    missing = sorted(set(errors) | set(typed))
    if errors or dup or (replaces and not trig.startswith("iso-confirm")):
        # on the first press a plain "no ion name" needs no error text
        shown = errors if trig.startswith("iso-confirm") else {
            i: e for i, e in errors.items() if e != "no ion name"}
        body = _iso_modal_body(state, key, missing, typed, shown, replaces)
        if dup:
            body.append(html.Div(f"✗ The same ion and harmonic is given to more than one peak: "
                                 f"{', '.join(dup)}", style={"color": "#c0392b",
                                                             "marginTop": "8px"}))
        return ISO_MODAL_SHOWN, body, "", nu, nu

    # all named: write the points (replacing the same ion + harmonic)
    new = iso_points_from_fit(state, named, _iso_source(key, files, align, ion_key))
    if len(old):
        drop = set(zip(new["ion"], new["harmonic"].astype(int)))
        old_keys = list(zip(old["ion"].astype(str), pd.to_numeric(old["harmonic"]).astype(int)))
        old = old[[k not in drop for k in old_keys]]
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new
    try:
        save_iso_points(out)
    except Exception as exc:
        traceback.print_exc()
        return fail(f"Could not write {iso_points_path()}: {exc}")
    sent = ", ".join(f"{r['ion']} h{r['harmonic']}" for r in named.values())
    msg = html.Span([html.B("Sent to isocurve: ", style={"color": "#2e7d32"}), sent]
                    + ([f"  (replaced {', '.join(replaces)})"] if replaces else []))
    apply = {"sum_key": key, "t": time.time(),
             "names": {str(i): {"ion": r["ion"], "harmonic": r["harmonic"]}
                       for i, r in named.items()}}
    return ISO_MODAL_HIDDEN, [], msg, apply, (version or 0) + 1


@app.callback(
    Output({"type": "iso-name", "peak": ALL}, "value"),
    Output({"type": "iso-h", "peak": ALL}, "value"),
    Input({"type": "iso-sugg", "peak": ALL, "name": ALL, "h": ALL}, "n_clicks"),
    State({"type": "iso-name", "peak": ALL}, "value"),
    State({"type": "iso-name", "peak": ALL}, "id"),
    State({"type": "iso-h", "peak": ALL}, "value"),
    prevent_initial_call=True,
)
def pick_iso_suggestion(_clicks, names, ids, hs):
    """A suggestion in the popup fills that peak's name and harmonic."""
    trig = dash.callback_context.triggered[0] if dash.callback_context.triggered else None
    if not trig or not trig["value"] or '"iso-sugg"' not in trig["prop_id"]:
        raise PreventUpdate
    idd = json.loads(trig["prop_id"].rsplit(".", 1)[0])
    names, hs = list(names), list(hs)
    for j, d in enumerate(ids):
        if d["peak"] == idd["peak"]:
            names[j], hs[j] = idd["name"], int(idd["h"])
    return names, hs


def _iso_cell(v, fmt):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
    return "–" if not np.isfinite(f) else format(f, fmt)


FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))     # ≈ 2.3548


def iso_curve_values(df):
    """Revolution time and its spread for each point:
        f_rev    = centre / h
        T (ns)   = 1e9 / f_rev
        FWHM(T)  = (FWHM / centre) * T * 1000        (ps; same relative width as f)
        σ(T)     = FWHM(T) / 2√(2 ln 2)
        err σ(T) = (dFWHM / FWHM) * FWHM(T) / 2√(2 ln 2)"""
    centre = pd.to_numeric(df["centre_hz"], errors="coerce")
    h = pd.to_numeric(df["harmonic"], errors="coerce")
    fwhm = pd.to_numeric(df["fwhm_hz"], errors="coerce")
    d_fwhm = pd.to_numeric(df["d_fwhm_hz"], errors="coerce")
    t_ns = 1e9 / (centre / h)
    fwhm_t_ps = (fwhm / centre) * t_ns * 1000.0
    return pd.DataFrame({
        "ion": df["ion"].astype(str), "harmonic": h, "T_ns": t_ns, "fwhm_T_ps": fwhm_t_ps,
        "sigma_T_ps": fwhm_t_ps / FWHM_PER_SIGMA,
        "d_sigma_T_ps": (d_fwhm / fwhm) * fwhm_t_ps / FWHM_PER_SIGMA,
    })


ISO_CURVE_FORMATS = {"harmonic": ".0f", "T_ns": ".6f", "fwhm_T_ps": ".4f",
                     "sigma_T_ps": ".4f", "d_sigma_T_ps": ".4f"}


ISO_FORMATS = {"centre_hz": ",.2f", "d_centre_hz": ".2f", "f_rev_hz": ",.4f",
               "d_f_rev_hz": ".4f", "fwhm_hz": ".2f", "d_fwhm_hz": ".2f", "amp": ".5g",
               "d_amp": ".3g", "area": ".5g", "d_area": ".3g", "eta": ".3f", "d_eta": ".3f",
               "chi2_ndf": ".3g", "harmonic": ".0f", "A": ".0f", "Z": ".0f", "q": ".0f"}


@app.callback(
    Output("iso-table", "data"),
    Output("iso-table", "tooltip_data"),
    Output("iso-info", "children"),
    Output("iso-points-ts", "data"),
    Input("iso-version", "data"),
    Input("further-subtabs", "value"),
    Input("iso-table", "data_timestamp"),
    State("iso-table", "data"),
)
def show_iso_points(_version, page, _ts, table):
    """Table of the points on the curve.  Deleting a row removes the point
    from the file."""
    trig = dash.callback_context.triggered[0]["prop_id"] if dash.callback_context.triggered else ""
    path = iso_points_path()
    try:
        df = load_iso_points()
        if trig.startswith("iso-table") and table is not None:
            keep = {(r["ion"], int(float(r["harmonic"]))) for r in table}
            keys = list(zip(df["ion"].astype(str), pd.to_numeric(df["harmonic"]).astype(int)))
            df = df[[k in keep for k in keys]]
            save_iso_points(df)
    except Exception as exc:
        traceback.print_exc()
        return [], [], html.Span(f"Could not read/write {path}: {exc}",
                                 style={"color": "#c0392b"}), time.time()
    if len(df):
        df = df.sort_values("f_rev_hz", kind="stable")
    rows = [{c: _iso_cell(r.get(c), ISO_FORMATS[c]) if c in ISO_FORMATS
             else ("" if pd.isna(r.get(c)) else str(r.get(c)))
             for c, _ in ISO_TABLE_COLUMNS} for r in df.to_dict("records")]
    tips = [{"source": {"value": row["source"], "type": "text"}} for row in rows]
    info = [html.B(f"{len(rows)} point(s)"), "  ·  ",
            html.Span(path, style={"fontFamily": "monospace", "fontSize": "12px",
                                   "color": "#666"})]
    if not rows:
        info.append(html.Div("No points yet: fit peaks in Aligning and fitting and press "
                             "'Send to isocurve'.", style={"color": "#888", "marginTop": "4px"}))
    return rows, tips, info, time.time()


def _iso_curve_table(vals, floor_on):
    """Rows and columns of the curve-parameters table (built with the fit,
    so the total error with the systematic floor is known)."""
    cols = list(ISO_CURVE_COLUMNS)
    if floor_on:
        cols.append(("d_sigma_T_total_ps", "err σ(T) total (ps)"))
    columns = [{"name": n, "id": c} for c, n in cols]
    if vals is None or not len(vals):
        return [], columns
    rows = []
    for r in vals.sort_values("T_ns", ascending=False, kind="stable").to_dict("records"):
        row = {c: _iso_cell(r.get(c), ISO_CURVE_FORMATS.get(c, ".4f"))
               if c in ISO_CURVE_FORMATS or c == "d_sigma_T_total_ps" else str(r.get(c))
               for c, _ in cols if c != "include"}
        row["include"] = (FIX_ON if r["include"] else FIX_OFF) if r["fittable"] else "–"
        row["id"] = r["id"]
        rows.append(row)
    return rows, columns


# ---- fit of the isochronicity curve ---------------------------------------

def _iso_point_id(ion, h):
    return f"{ion}|{int(float(h))}"


def _iso_included(df):
    """'include' flags of the points (missing / blank = included)."""
    if "include" not in df.columns:
        return np.ones(len(df), dtype=bool)
    def one(v):
        if isinstance(v, str):
            return v.strip().lower() not in ("false", "0", "no")
        try:
            return True if pd.isna(v) else bool(v)
        except (TypeError, ValueError):
            return True
    return np.array([one(v) for v in df["include"]], dtype=bool)


def _iso_fittable(vals):
    """Points that can be fitted: finite T, σ(T) and err σ(T) > 0."""
    return (np.isfinite(vals["T_ns"]) & np.isfinite(vals["sigma_T_ps"])
            & np.isfinite(vals["d_sigma_T_ps"]) & (vals["d_sigma_T_ps"] > 0)).to_numpy()


def iso_sigma_t(t_ns, gamma_t, sp, ssys_ps, circ_m, c_m_s):
    """σ_T (ps) at revolution time T (ns):
    sqrt(((1/γ² − 1/γ_t²)·σp/p·T)² + σ_sys²), 1/γ² = 1 − (C/(c·T))²."""
    t_ns = np.asarray(t_ns, dtype=float)
    inv_g2 = 1.0 - (circ_m / (c_m_s * t_ns * 1e-9)) ** 2
    return np.sqrt(((inv_g2 - 1.0 / gamma_t ** 2) * sp * t_ns * 1000.0) ** 2 + ssys_ps ** 2)


def load_iso_fit_settings():
    saved = _load_json(paths.settings_file("isocurve.json"))
    st = {k: float(saved.get(k, v)) if isinstance(saved.get(k, v), (int, float))
          else v for k, v in ISO_FIT_DEFAULTS.items()}
    st["auto_floor"] = bool(saved.get("auto_floor", False))
    st["bin_err"] = bool(saved.get("bin_err", False))
    st["rec_freq"] = str(saved.get("rec_freq", ISO_BIN_DEFAULTS["rec_freq"]))
    st["n_bins"] = str(saved.get("n_bins", ISO_BIN_DEFAULTS["n_bins"]))
    return st


def parse_number_expr(text):
    """A number typed as 10e6, 131072, 2^17, 2**17 or 10*1e6 (+ − * / ^ only)."""
    import ast
    import operator as op
    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
           ast.Pow: op.pow, ast.USub: op.neg, ast.UAdd: op.pos}

    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.BinOp) and type(n.op) in ops:
            return ops[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in ops:
            return ops[type(n.op)](ev(n.operand))
        raise ValueError
    try:
        v = ev(ast.parse(str(text).strip().replace("^", "**"), mode="eval").body)
    except Exception:
        raise ValueError(f"'{text}' is not a number (e.g. 10e6 or 2^17)")
    if not np.isfinite(v) or v <= 0:
        raise ValueError(f"'{text}' must be a positive number")
    return v


def iso_bin_width(st):
    """Bin width (Hz) = recording frequency / number of bins."""
    return parse_number_expr(st["rec_freq"]) / parse_number_expr(st["n_bins"])


def save_iso_fit_settings(st):
    path = paths.settings_file("isocurve.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(st, fh, indent=2)
    os.replace(tmp, path)


def check_iso_fit_settings(st):
    if not (st["circumference_m"] > 0 and st["c_m_s"] > 0):
        raise ValueError("circumference and c must be > 0")
    for a, b, what in (("gt_min", "gt_max", "γ_t"), ("sp_min_pct", "sp_max_pct", "σp/p"),
                       ("ssys_min_ps", "ssys_max_ps", "σ_sys")):
        if not st[a] < st[b]:
            raise ValueError(f"{what}: min must be below max")
    if st["gt_min"] <= 0:
        raise ValueError("γ_t min must be > 0")


def fit_iso_curve(t_ns, sig_ps, err_ps, st):
    """Weighted fit of γ_t, σp/p, σ_sys.  Returns dict(popt, perr, chi2, ndf)."""
    from scipy.optimize import curve_fit
    circ, c = st["circumference_m"], st["c_m_s"]
    lb = [st["gt_min"], st["sp_min_pct"] / 100.0, st["ssys_min_ps"]]
    ub = [st["gt_max"], st["sp_max_pct"] / 100.0, st["ssys_max_ps"]]

    def f(t, gt, sp, ss):
        return iso_sigma_t(t, gt, sp, ss, circ, c)
    best = None
    # several γ_t starts: the curve has a minimum near γ = γ_t, and a single
    # start can end in the wrong one
    for gt0 in np.linspace(lb[0], ub[0], 9)[1:-1]:
        p0 = [gt0, 0.5 * (lb[1] + ub[1]), 0.5 * (lb[2] + ub[2])]
        try:
            popt, pcov = curve_fit(f, t_ns, sig_ps, p0=p0, sigma=err_ps, absolute_sigma=True,
                                   bounds=(lb, ub), maxfev=20000)
        except (RuntimeError, ValueError):
            continue
        chi2 = float(np.sum(((sig_ps - f(t_ns, *popt)) / err_ps) ** 2))
        if best is None or chi2 < best[2]:
            best = (popt, pcov, chi2)
    if best is None:
        raise ValueError("the fit did not converge")
    popt, pcov, chi2 = best
    perr = np.sqrt(np.clip(np.diag(pcov), 0, None)) if np.all(np.isfinite(pcov)) \
        else np.full(3, np.nan)
    ndf = len(t_ns) - 3
    return {"popt": [float(v) for v in popt], "perr": [float(v) for v in perr],
            "chi2": chi2, "ndf": ndf, "chi2_ndf": chi2 / ndf if ndf > 0 else np.nan}


def iso_fit_data(st):
    """Points (all, with flags) and the fit result (None if not possible, with
    the reason)."""
    df = load_iso_points()
    if not len(df):
        return None, None, "no points yet"
    vals = iso_curve_values(df)
    vals["include"] = _iso_included(df)
    vals["fittable"] = _iso_fittable(vals)
    vals["id"] = [_iso_point_id(i, h) for i, h in zip(vals["ion"], vals["harmonic"])]
    used = vals[vals["include"] & vals["fittable"]]
    if len(used) < 3:
        return vals, None, f"{len(used)} point(s) in the fit: at least 3 are needed"
    # bin-width error: 1σ of a value spread evenly across one bin, bin/√12
    # (±½ bin is the maximum, not 1σ), on every FWHM, as a σ(T) error (same
    # conversion as err σ(T)): (bin/√12 / centre) · T · 1000 / 2√(2 ln 2)
    bin_sigma = None
    vals["d_sigma_T_bin_ps"] = 0.0
    if st.get("bin_err"):
        try:
            bin_sigma = iso_bin_width(st) / np.sqrt(12.0)
        except ValueError as exc:
            return vals, None, f"bin-width error: {exc}"
        centre = vals["harmonic"] / (vals["T_ns"] * 1e-9)
        vals["d_sigma_T_bin_ps"] = (bin_sigma / centre) * vals["T_ns"] * 1000.0 / FWHM_PER_SIGMA
    used = vals[vals["include"] & vals["fittable"]]
    t, sg = used["T_ns"].to_numpy(), used["sigma_T_ps"].to_numpy()
    err = np.hypot(used["d_sigma_T_ps"].to_numpy(), used["d_sigma_T_bin_ps"].to_numpy())
    floor = 0.0
    try:
        res = fit_iso_curve(t, sg, err, st)
        chi2_stat = res["chi2_ndf"]
        if st.get("auto_floor") and res["ndf"] > 0 and res["chi2_ndf"] > 1:
            floor = iso_auto_floor(t, sg, err, st)
            res = fit_iso_curve(t, sg, np.hypot(err, floor), st)
    except Exception as exc:
        traceback.print_exc()
        return vals, None, f"fit failed: {exc}"
    res["n"] = len(used)
    res["floor"] = floor if st.get("auto_floor") else None
    res["chi2_ndf_stat"] = chi2_stat
    res["bin_sigma_hz"] = bin_sigma
    # total error (statistical ⊕ bin width ⊕ floor) for every point
    vals["d_sigma_T_total_ps"] = np.hypot(np.hypot(vals["d_sigma_T_ps"],
                                                   vals["d_sigma_T_bin_ps"]), floor)
    return vals, res, ""


def iso_auto_floor(t, sg, err, st):
    """Systematic floor f (ps) for which the fit with errors √(err² + f²)
    has χ²/ndf = 1 (bisection on f; χ²/ndf falls as f grows)."""
    def excess(f):
        return fit_iso_curve(t, sg, np.hypot(err, f), st)["chi2_ndf"] - 1.0
    lo, hi = 0.0, max(float(np.nanmax(sg)), 1e-3)
    while excess(hi) > 0 and hi < 1e3:          # make sure the root is bracketed
        lo, hi = hi, hi * 4
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if excess(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-6 * max(hi, 1e-9):
            break
    return 0.5 * (lo + hi)


def _iso_param_lines(res, extras=True):
    """Fit result lines; extras=False leaves out the bin-width / floor lines
    (the parameter box on the plots shows only the fit)."""
    (gt, sp, ss), (dgt, dsp, dss) = res["popt"], res["perr"]
    chi = "–" if not np.isfinite(res["chi2_ndf"]) else f"{res['chi2_ndf']:.3f}"
    lines = [f"γ_t = {gt:.4f} ± {dgt:.6f}",
             f"σp/p = {sp * 100:.4f} ± {dsp * 100:.4f} %",
             f"σ_sys = {ss:.4f} ± {dss:.4f} ps",
             f"χ²/ndf = {chi}  (ndf = {res['ndf']}, {res['n']} points)"]
    if extras and res.get("bin_sigma_hz") is not None:
        lines.append(f"bin-width error = {res['bin_sigma_hz']:.2f} Hz (1σ) on FWHM "
                     f"(bin/√12, {res['bin_sigma_hz'] * np.sqrt(12.0):.3f} Hz bins)")
    if extras and res.get("floor") is not None:
        c0 = res.get("chi2_ndf_stat")
        c0 = "–" if c0 is None or not np.isfinite(c0) else f"{c0:.3f}"
        lines.append(f"systematic floor = {res['floor']:.4f} ps  (auto; χ²/ndf without it = {c0})")
    return lines


def _iso_pull_values(df, res, st, ecol):
    """(σ(T) − fit) / error for the rows of df (NaN where there is no error)."""
    fit = iso_sigma_t(df["T_ns"].to_numpy(), *res["popt"], st["circumference_m"], st["c_m_s"])
    err = df[ecol].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(err > 0, (df["sigma_T_ps"].to_numpy() - fit) / err, np.nan)


def _iso_pulls(used, res, st, ecol):
    """(σ(T) − fit) / error for the fitted points, and the 'within ±1σ' text
    (as in the decay-fit residual panel)."""
    pull = _iso_pull_values(used, res, st, ecol)
    n = len(pull)
    n_in = int(np.sum(np.abs(pull) <= 1.0))
    pct = 100.0 * n_in / n if n else float("nan")
    return pull, f"within ±1σ: {pct:.1f}%  ({n_in}/{n})   Gaussian ≈ 68.3%"


def _iso_curve_x(vals):
    t = vals["T_ns"][np.isfinite(vals["T_ns"])]
    lo, hi = float(t.min()), float(t.max())
    pad = max(0.05 * (hi - lo), 0.05)
    return np.linspace(lo - pad, hi + pad, 2000)


@app.callback(
    Output("iso-set-circumference_m", "value"),
    Output("iso-set-c_m_s", "value"),
    Output("iso-set-gt_min", "value"),
    Output("iso-set-gt_max", "value"),
    Output("iso-set-sp_min_pct", "value"),
    Output("iso-set-sp_max_pct", "value"),
    Output("iso-set-ssys_min_ps", "value"),
    Output("iso-set-ssys_max_ps", "value"),
    Output("iso-set-floor", "value"),
    Output("iso-set-binerr", "value"),
    Output("iso-set-recfreq", "value"),
    Output("iso-set-nbins", "value"),
    Input("iso-settings-init", "data"),
)
def init_iso_settings(_):
    st = load_iso_fit_settings()
    return tuple(st[k] for k in ISO_SETTING_KEYS) + (
        ["auto"] if st["auto_floor"] else [], ["on"] if st["bin_err"] else [],
        st["rec_freq"], st["n_bins"])


@app.callback(
    Output("iso-graph", "figure"),
    Output("iso-resid", "figure"),
    Output("iso-fit-result", "children"),
    Output("iso-curve-table", "data"),
    Output("iso-curve-table", "columns"),
    Input("iso-points-ts", "data"),
    *[Input(f"iso-set-{k}", "value") for k in ISO_SETTING_KEYS],
    Input("iso-set-floor", "value"),
    Input("iso-set-binerr", "value"),
    Input("iso-set-recfreq", "value"),
    Input("iso-set-nbins", "value"),
    prevent_initial_call=True,
)
def fit_and_plot_iso(_ts, *args):
    """Fit the curve to the included points and draw it (with residuals)."""
    setting_vals, (floor_val, bin_val, rec_freq, n_bins) = args[:-4], args[-4:]
    trig = dash.callback_context.triggered[0]["prop_id"] if dash.callback_context.triggered else ""
    st = load_iso_fit_settings()
    err = ""
    if all(v is not None for v in setting_vals):
        new = {k: float(v) for k, v in zip(ISO_SETTING_KEYS, setting_vals)}
        try:
            check_iso_fit_settings(new)
            st = new
        except ValueError as exc:
            err = f"Settings not used: {exc}"
    else:
        err = "Settings not used: fill in every box"
    if floor_val is not None:
        st["auto_floor"] = "auto" in floor_val
    if bin_val is not None:
        st["bin_err"] = "on" in bin_val
    st["rec_freq"] = rec_freq if rec_freq not in (None, "") else st.get(
        "rec_freq", ISO_BIN_DEFAULTS["rec_freq"])
    st["n_bins"] = n_bins if n_bins not in (None, "") else st.get(
        "n_bins", ISO_BIN_DEFAULTS["n_bins"])
    if trig.startswith("iso-set-") and not err:
        save_iso_fit_settings(st)
    floor_on = bool(st.get("auto_floor") or st.get("bin_err"))   # total-error column
    try:
        vals, res, why = iso_fit_data(st)
    except Exception as exc:
        traceback.print_exc()
        return (empty_fig(str(exc)), empty_fig(), html.Span(str(exc), style={"color": "#c0392b"}),
                *_iso_curve_table(None, floor_on))
    if vals is None:
        return empty_fig("No points yet"), empty_fig(), "", *_iso_curve_table(None, floor_on)
    # error bars: the combined error (statistical ⊕ bin width ⊕ floor) used in the fit
    ecol = "d_sigma_T_total_ps" if "d_sigma_T_total_ps" in vals else "d_sigma_T_ps"

    fig = go.Figure()
    used = vals[vals["include"] & vals["fittable"]]
    out = vals[~(vals["include"] & vals["fittable"])]
    if res:
        x = _iso_curve_x(vals)
        fig.add_trace(go.Scatter(x=x, y=iso_sigma_t(x, *res["popt"], st["circumference_m"],
                                                    st["c_m_s"]),
                                 mode="lines", name="Isochronicity curve", meta="iso-curve",
                                 line=dict(color=ACCENT, width=2), hoverinfo="skip"))
    # labels: drawn in the browser as annotations with arrows (ISO_LABEL_JS)
    fig.add_trace(go.Scatter(
        x=used["T_ns"], y=used["sigma_T_ps"], mode="markers", name="In fit", meta="iso-pts",
        text=used["ion"], customdata=used["id"],
        marker=dict(symbol="x", size=9, color="black"),
        error_y=dict(type="data", array=used[ecol], color="#999", thickness=1, width=4),
        hovertemplate="%{text}<br>T = %{x:.4f} ns<br>σ(T) = %{y:.4f} ps"
                      "<br>click: leave out<extra></extra>"))
    if len(out):
        fig.add_trace(go.Scatter(
            x=out["T_ns"], y=out["sigma_T_ps"], mode="markers", name="Left out", meta="iso-out",
            text=out["ion"], customdata=out["id"],
            marker=dict(symbol="circle-open", size=9, color="#aaa"),
            error_y=dict(type="data", array=out["d_sigma_T_ps"].fillna(0), color="#ddd",
                         thickness=1, width=4),
            hovertemplate="%{text} (left out)<br>T = %{x:.4f} ns<br>σ(T) = %{y:.4f} ps"
                          "<br>click: put back<extra></extra>"))
    if res:
        fig.add_annotation(xref="paper", yref="paper", x=0.01, y=0.99, xanchor="left",
                           yanchor="top", align="left", showarrow=False,
                           text="<br>".join(_iso_param_lines(res, extras=False)),
                           bgcolor=ACCENT_PALE, bordercolor=ACCENT, borderwidth=1,
                           font=dict(size=13), name="isoparam")
    fig.update_layout(template="plotly_white", margin=dict(l=70, r=20, t=30, b=50),
                      xaxis_title="Revolution time T (ns)", yaxis_title="σ_T (ps)",
                      legend=dict(x=0.99, y=0.02, xanchor="right", yanchor="bottom",
                                  bgcolor="rgba(255,255,255,0.85)", bordercolor="#dddddd",
                                  borderwidth=1),
                      font=dict(size=13), uirevision="iso", clickmode="event")

    # residuals in units of each point's error, as in the decay-fit panel:
    # grey ±1σ band, red zero line, black dots, 'within ±1σ' count
    # styled like the main plot: × markers, zero line in the curve's colour,
    # left-out points as grey circles, boxed ion labels (ISO_LABEL_JS)
    rfig = go.Figure()
    if res:
        pull, pull_txt = _iso_pulls(used, res, st, ecol)
        rfig.add_hrect(y0=-1, y1=1, fillcolor="#d9d9d9", opacity=1, line_width=0, layer="below")
        xr = _iso_curve_x(vals)
        rfig.add_trace(go.Scatter(x=[xr[0], xr[-1]], y=[0, 0], mode="lines", meta="iso-curve",
                                  line=dict(color=ACCENT, width=2), hoverinfo="skip"))
        rfig.add_trace(go.Scatter(
            x=used["T_ns"], y=pull, mode="markers", text=used["ion"], customdata=used["id"],
            meta="iso-pts", marker=dict(symbol="x", size=9, color="black"),
            hovertemplate="%{text}<br>T = %{x:.4f} ns<br>(σ(T) − fit)/err = %{y:.2f}"
                          "<br>click: leave out<extra></extra>"))
        if len(out):
            opull = _iso_pull_values(out, res, st, ecol)
            ok = np.isfinite(opull)
            if ok.any():
                o2 = out[ok]
                rfig.add_trace(go.Scatter(
                    x=o2["T_ns"], y=opull[ok], mode="markers", text=o2["ion"],
                    customdata=o2["id"], meta="iso-out",
                    marker=dict(symbol="circle-open", size=9, color="#aaa"),
                    hovertemplate="%{text} (left out)<br>T = %{x:.4f} ns<br>"
                                  "(σ(T) − fit)/err = %{y:.2f}<br>click: put back<extra></extra>"))
        rfig.add_annotation(xref="paper", yref="paper", x=0.995, y=0.97, xanchor="right",
                            yanchor="top", showarrow=False, text=pull_txt,
                            bgcolor="rgba(255,255,255,0.85)", bordercolor="#b3b3b3",
                            borderwidth=1, borderpad=4, font=dict(size=13))
    rfig.update_layout(template="plotly_white", margin=dict(l=70, r=20, t=30, b=50),
                       title=dict(text="Residuals", font=dict(size=13)),
                       xaxis_title="Revolution time T (ns)", yaxis_title="(σ(T) − fit)/σ",
                       showlegend=False, font=dict(size=13), uirevision="iso",
                       clickmode="event")

    parts = []
    if res:
        parts.append(html.Div([html.B("Fit: "), "   ·   ".join(_iso_param_lines(res))]))
    if why:
        parts.append(html.Div(f"No fit: {why}", style={"color": "#888"}))
    if err:
        parts.append(html.Div(err, style={"color": "#c0392b"}))
    return fig, rfig, parts, *_iso_curve_table(vals, floor_on)


# Point labels of the isochronicity plot: placed in the browser with the same
# method as place_labels(), and placed again after every zoom / pan / resize /
# legend click, so they never overlap.  Drawn as annotations with arrows.
ISO_LABEL_JS = """
function(fig) {
    var ANG = [45, 135, -45, -135, 90, -90, 0, 180,
               67.5, 112.5, -67.5, -112.5, 22.5, 157.5, -22.5, -157.5,
               33.75, 56.25, 123.75, 146.25, -33.75, -56.25, -123.75, -146.25];
    var RAD = [24, 34, 46, 60, 76, 94, 115, 140, 170, 210, 260, 320];
    var FONT_PX = 12;
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
    function textW(t, font) { ctx2d.font = font; return ctx2d.measureText(t).width; }

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
            if (tr.meta === 'iso-curve') {
                var step = Math.max(1, Math.floor(full.x.length / 150));
                for (var k = 0; k < full.x.length; k += step) {
                    var cx = xa.l2p(full.x[k]) + xa._offset, cy = ya.l2p(full.y[k]) + ya._offset;
                    obst.push([cx - 3, cy - 3, cx + 3, cy + 3]);
                }
                return;
            }
            if (tr.meta !== 'iso-pts' && tr.meta !== 'iso-out') return;
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
                // text + white box with a black border (borderpad 3 + border 1 each side)
                sizes.push([textW(t, font) + 2 + 8, Math.round(FONT_PX * 1.35) + 8]);
                lab.push({x: X, y: Y, text: t, out: tr.meta === 'iso-out'});
            }
        });
        // never under the parameter box or the legend
        gd.querySelectorAll('.annotation, .legend').forEach(function(el) {
            if (el.classList.contains('annotation') && el.getAttribute('data-index') !== null) {
                var a = (gd.layout.annotations || [])[+el.getAttribute('data-index')];
                if (a && a.name === 'isolab') return;
            }
            var r = el.getBoundingClientRect();
            if (r.width && r.height)
                obst.push([r.left - gr.left, r.top - gr.top, r.right - gr.left, r.bottom - gr.top, 20]);
        });
        var offs = place(pts, sizes, obst, rect, bars);
        var keep = (gd.layout.annotations || []).filter(function(a) { return a.name !== 'isolab'; });
        var anns = keep.concat(lab.map(function(l, i) {
            return {x: l.x, y: l.y, xref: 'x', yref: 'y', text: l.text, name: 'isolab',
                    showarrow: true, ax: offs[i][0], ay: offs[i][1], arrowhead: 2,
                    arrowsize: 0.8, arrowwidth: 1, arrowcolor: '#888888', standoff: 5,
                    font: {size: FONT_PX, color: l.out ? '#aaaaaa' : '#222222'},
                    bgcolor: '#ffffff', bordercolor: '#000000', borderwidth: 1, borderpad: 3,
                    captureevents: false};
        }));
        gd._isoLabelBusy = true;
        Plotly.relayout(gd, {annotations: anns}).then(function() { gd._isoLabelBusy = false; },
                                                      function() { gd._isoLabelBusy = false; });
    }

    setTimeout(function() {
        var gd = document.querySelector('#__GRAPH__ .js-plotly-plot');
        if (!gd) return;
        if (!gd._isoLabelHook) {
            gd._isoLabelHook = true;
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
                if (gd._isoLabelBusy) return;
                var keys = Object.keys(ev || {});
                if (keys.length && keys.every(function(k) { return k.indexOf('annotations') === 0; })) return;
                soon();
            });
            gd.on('plotly_restyle', soon);            // legend clicks (hide / show)
            window.addEventListener('resize', soon);
            gd._isoSoon = soon;
        }
        gd._isoSoon();
    }, 150);
    return window.dash_clientside.no_update;
}
"""

app.clientside_callback(
    ISO_LABEL_JS.replace("__GRAPH__", "iso-graph"),
    Output("_iso-label-hook", "children"),
    Input("iso-graph", "figure"),
    prevent_initial_call=True,
)
# the same labels on the residual plot
app.clientside_callback(
    ISO_LABEL_JS.replace("__GRAPH__", "iso-resid"),
    Output("_iso-resid-label-hook", "children"),
    Input("iso-resid", "figure"),
    prevent_initial_call=True,
)


@app.callback(
    Output("iso-version", "data", allow_duplicate=True),
    Output("iso-curve-table", "active_cell"),
    Output("iso-curve-table", "selected_cells"),
    Input("iso-curve-table", "active_cell"),
    Input("iso-graph", "clickData"),
    Input("iso-resid", "clickData"),
    State("iso-version", "data"),
    prevent_initial_call=True,
)
def toggle_iso_point(active, click, rclick, version):
    """'in fit' cell or a click on a point: leave the point out / put it back."""
    trig = dash.callback_context.triggered[0]["prop_id"] if dash.callback_context.triggered else ""
    pid = None
    if trig.startswith("iso-curve-table"):
        if not active or active.get("column_id") != "include":
            raise PreventUpdate
        pid = active.get("row_id")
    elif trig.startswith("iso-graph") and click and click.get("points"):
        pid = click["points"][0].get("customdata")
    elif trig.startswith("iso-resid") and rclick and rclick.get("points"):
        pid = rclick["points"][0].get("customdata")
    if not pid:
        raise PreventUpdate
    ion, h = pid.rsplit("|", 1)
    df = load_iso_points()
    keys = [_iso_point_id(i, hh) for i, hh in zip(df["ion"].astype(str), df["harmonic"])]
    if pid not in keys:
        raise PreventUpdate
    inc = _iso_included(df)
    vals = iso_curve_values(df)
    k = keys.index(pid)
    if not _iso_fittable(vals)[k]:
        raise PreventUpdate                       # no error bar: cannot be fitted anyway
    inc[k] = not inc[k]
    df["include"] = inc
    save_iso_points(df)
    return (version or 0) + 1, None, []


@app.callback(
    Output("iso-bin-info", "children"),
    Input("iso-set-recfreq", "value"),
    Input("iso-set-nbins", "value"),
)
def show_bin_width(rec_freq, n_bins):
    try:
        w = iso_bin_width({"rec_freq": rec_freq or "", "n_bins": n_bins or ""})
    except ValueError as exc:
        return html.Span(f"✗ {exc}", style={"color": "#c0392b", "fontFamily": "inherit"})
    return f"bin = {w:.3f} Hz  →  1σ = bin/√12 = {w / np.sqrt(12.0):.2f} Hz"


@app.callback(
    Output("iso-save-status", "children"),
    Input("btn-iso-save-fig", "n_clicks"),
    *[State(f"iso-set-{k}", "value") for k in ISO_SETTING_KEYS],
    State("iso-set-floor", "value"),
    State("iso-set-binerr", "value"),
    State("iso-set-recfreq", "value"),
    State("iso-set-nbins", "value"),
    prevent_initial_call=True,
)
def save_iso_figure(_n, *args):
    setting_vals, (floor_val, bin_val, rec_freq, n_bins) = args[:-4], args[-4:]
    try:
        st = {k: float(v) for k, v in zip(ISO_SETTING_KEYS, setting_vals)}
        check_iso_fit_settings(st)
    except (TypeError, ValueError):
        st = load_iso_fit_settings()
    st["auto_floor"] = "auto" in (floor_val or [])
    st["bin_err"] = "on" in (bin_val or [])
    st["rec_freq"] = rec_freq or ISO_BIN_DEFAULTS["rec_freq"]
    st["n_bins"] = n_bins or ISO_BIN_DEFAULTS["n_bins"]
    try:
        vals, res, why = iso_fit_data(st)
        if vals is None:
            raise ValueError("no points yet")
        files = draw_iso_figure(vals, res, st)
    except Exception as exc:
        traceback.print_exc()
        return html.Span(f"Could not save: {exc}", style={"color": "#c0392b"})
    return html.Span([html.B("Saved: ", style={"color": "#2e7d32"}),
                      "  ".join(os.path.basename(f) for f in files),
                      f"  (in {os.path.dirname(files[0])})"])


# ---- point labels without overlaps (isochronicity plot) ---------------------
# Greedy placement in pixel space: each label tries 16 directions × 6
# distances around its point and takes the lowest-cost spot.  Cost: overlap
# with labels already placed (heavy), with obstacles (points, error bars, the
# curve, parameter box, legend), area outside the plot, the leader line
# crossing a placed label, and the distance.  Crowded points go first.
# The same algorithm runs in the browser (ISO_LABEL_JS) after every zoom/pan.
LABEL_ANGLES = [45, 135, -45, -135, 90, -90, 0, 180,
                67.5, 112.5, -67.5, -112.5, 22.5, 157.5, -22.5, -157.5,
                33.75, 56.25, 123.75, 146.25, -33.75, -56.25, -123.75, -146.25]
LABEL_RADII = [24, 34, 46, 60, 76, 94, 115, 140, 170, 210, 260, 320]
LABEL_KEEP = 120             # cheapest candidate spots kept per label
LABEL_ANNEAL_PER_LABEL = 700    # annealing steps per label …
LABEL_ANNEAL_MAX = 25000        # … at most, per run
LABEL_RESTARTS = 4              # annealing runs (different fixed seeds); the cheapest is kept
ISO_LABEL_PT = 12          # label text size: saved figure (pt) and page (px)
ISO_LABEL_BOX_PAD = 0.2    # label box padding in the saved figure (× font size)
ISO_ARROW_BAR_COST = 150   # per 2 px of arrow lying on an error bar (same in ISO_LABEL_JS)
ISO_ARROW_BOX_COST = 6000   # an arrow through a label box, either way (same in ISO_LABEL_JS)
ISO_ARROW_CROSS_COST = 6000  # arrow crossing another label's arrow (same in ISO_LABEL_JS)
ISO_ARROW_POINT_COST = 300  # arrow passing over another point (same in ISO_LABEL_JS)


def _overlap(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def _leader_in_bars(px, py, cx, cy, box, bars):
    """Number of 2-px steps of the arrow (point → label, from 8 px off the
    point to where it enters the label box) that lie on an error bar."""
    d = float(np.hypot(cx - px, cy - py))
    if d <= 8 or not bars:
        return 0
    ux, uy = (cx - px) / d, (cy - py) / d
    hits, s = 0, 8.0
    while s < d:
        x, y = px + s * ux, py + s * uy
        if box[0] <= x <= box[2] and box[1] <= y <= box[3]:
            break                                  # arrow ends at the label box
        for b in bars:
            if b[0] <= x <= b[2] and b[1] <= y <= b[3]:
                hits += 1
                break
        s += 2.0
    return hits


def _leader(px, py, cx, cy, w, h):
    """Visible arrow of a label: from 5 px off the point to where the line
    point → label centre enters the label box.  None if the point is inside."""
    dx, dy = cx - px, cy - py
    hw, hh = w / 2 + 2, h / 2 + 1
    if abs(dx) <= hw and abs(dy) <= hh:
        return None
    s = min(hw / abs(dx) if dx else np.inf, hh / abs(dy) if dy else np.inf)
    d = float(np.hypot(dx, dy))
    t0 = min(5.0 / d, 1.0 - s)
    return (px + t0 * dx, py + t0 * dy, px + (1 - s) * dx, py + (1 - s) * dy)


def _to_centre(px, py, cx, cy):
    """Line from 5 px off the point to the label centre (used for the
    arrow-crossing test: conservative, as the drawn arrow stops at the box
    edge, whose exact size is only known once rendered)."""
    d = float(np.hypot(cx - px, cy - py))
    if d <= 5:
        return None
    return (px + 5 * (cx - px) / d, py + 5 * (cy - py) / d, cx, cy)


def _seg_rect(seg, r):
    """Does segment seg (x0, y0, x1, y1) pass through rectangle r (Liang–Barsky)?"""
    if seg is None:
        return False
    x0, y0, x1, y1 = seg
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - r[0]), (dx, r[2] - x0), (-dy, y0 - r[1]), (dy, r[3] - y0)):
        if p == 0:
            if q < 0:
                return False
        else:
            t = q / p
            if p < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                return False
    return True


def _segs_cross(a, b):
    """Do segments a and b (x0, y0, x1, y1) cross?"""
    if a is None or b is None:
        return False

    def orient(ax, ay, bx, by, cx, cy):
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    d1 = orient(b[0], b[1], b[2], b[3], a[0], a[1])
    d2 = orient(b[0], b[1], b[2], b[3], a[2], a[3])
    d3 = orient(a[0], a[1], a[2], a[3], b[0], b[1])
    d4 = orient(a[0], a[1], a[2], a[3], b[2], b[3])
    return d1 * d2 < 0 and d3 * d4 < 0


def _near_seg(qx, qy, seg, tol=7.0):
    """Is point q within tol px of segment seg?"""
    if seg is None:
        return False
    x0, y0, x1, y1 = seg
    vx, vy = x1 - x0, y1 - y0
    L2 = vx * vx + vy * vy
    t = 0.0 if L2 == 0 else min(max(((qx - x0) * vx + (qy - y0) * vy) / L2, 0.0), 1.0)
    return float(np.hypot(qx - (x0 + t * vx), qy - (y0 + t * vy))) < tol


def place_labels(pts, sizes, obstacles, rect, y_down=True, bars=None, seed=0):
    """Pixel offsets (dx, dy) of each label centre from its point.
    pts: [(x, y)], sizes: [(w, h)] (label box incl. padding), obstacles:
    [(x0, y0, x1, y1[, weight])] (default weight 4), rect: plot area,
    y_down: screen y grows downwards, bars: error-bar rectangles.

    Each label gets candidate spots (LABEL_ANGLES × LABEL_RADII around its
    point).  A spot's own cost: distance, area outside the plot, obstacles,
    its arrow on error bars, its arrow over other points.  Pair costs between
    two labels: box overlap, an arrow through the other's box (either way),
    crossing arrows.  The layout is found by simulated annealing (fixed seed,
    so the same view always gives the same layout) and then polished.  The
    same algorithm runs in the browser (ISO_LABEL_JS)."""
    n = len(pts)
    if n == 0:
        return []
    bars = bars or []
    up = -1.0 if y_down else 1.0
    cands = []
    for i in range(n):
        px, py = pts[i]
        w, h = sizes[i]
        lst = []
        for r in LABEL_RADII:
            for a in LABEL_ANGLES:
                cx = px + r * np.cos(np.radians(a))
                cy = py + up * r * np.sin(np.radians(a))
                box = (cx - w / 2 - 2, cy - h / 2 - 1, cx + w / 2 + 2, cy + h / 2 + 1)
                area = (box[2] - box[0]) * (box[3] - box[1])
                u = 0.6 * r + 60.0 * (area - _overlap(box, rect))
                for o in obstacles:
                    u += (o[4] if len(o) > 4 else 4.0) * _overlap(box, o)
                u += ISO_ARROW_BAR_COST * _leader_in_bars(px, py, cx, cy, box, bars)
                lead = _leader(px, py, cx, cy, w, h)
                for k, (qx, qy) in enumerate(pts):
                    # (points under this point's own marker cannot be avoided)
                    if k != i and np.hypot(qx - px, qy - py) > 8 and _near_seg(qx, qy, lead):
                        u += ISO_ARROW_POINT_COST
                env = (min(box[0], px), min(box[1], py), max(box[2], px), max(box[3], py))
                lst.append((u, (cx, cy), box, lead, _to_centre(px, py, cx, cy), env))
        lst.sort(key=lambda c: c[0])
        cands.append(lst[:LABEL_KEEP])

    cache = {}

    def pair(i, a, j, b):
        key = (i, a, j, b) if i < j else (j, b, i, a)
        v = cache.get(key)
        if v is None:
            A, B = cands[i][a], cands[j][b]
            ea, eb = A[5], B[5]
            v = 0.0
            if not (ea[2] < eb[0] or eb[2] < ea[0] or ea[3] < eb[1] or eb[3] < ea[1]):
                v += 20.0 * _overlap(A[2], B[2])
                if _seg_rect(A[3], B[2]) or _seg_rect(B[3], A[2]):
                    v += ISO_ARROW_BOX_COST
                if _segs_cross(A[4], B[4]):
                    v += ISO_ARROW_CROSS_COST
            cache[key] = v
        return v

    # labels that can interact at all (their candidate areas touch)
    env = [(min(c[5][0] for c in L), min(c[5][1] for c in L),
            max(c[5][2] for c in L), max(c[5][3] for c in L)) for L in cands]
    nb = [[j for j in range(n) if j != i and not (env[i][2] < env[j][0] or env[j][2] < env[i][0]
                                                   or env[i][3] < env[j][1] or env[j][3] < env[i][1])]
          for i in range(n)]
    cur = [0] * n

    def local(i, a):
        return cands[i][a][0] + sum(pair(i, a, j, cur[j]) for j in nb[i])

    def total():
        return (sum(cands[i][cur[i]][0] for i in range(n))
                + 0.5 * sum(pair(i, cur[i], j, cur[j]) for i in range(n) for j in nb[i]))

    # several annealing runs (fixed seeds), keep the layout with the lowest cost
    iters = min(LABEL_ANNEAL_MAX, LABEL_ANNEAL_PER_LABEL * n)
    T0, T1 = 3000.0, 5.0
    best_layout, best_cost = None, None
    for run in range(LABEL_RESTARTS):
        rng = np.random.default_rng(seed + run)
        cur[:] = [0] * n                            # start: each label's cheapest own spot
        tot = best_tot = 0.0
        best = list(cur)
        for it in range(iters):
            T = T0 * (T1 / T0) ** (it / iters)
            i = int(rng.integers(n))
            a = int(rng.integers(len(cands[i])))
            if a == cur[i]:
                continue
            d = local(i, a) - local(i, cur[i])
            if d < 0 or rng.random() < np.exp(-d / T):
                cur[i] = a
                tot += d
                if tot < best_tot - 1e-9:
                    best_tot, best = tot, list(cur)
        cur[:] = best
        for _ in range(5):                          # polish: best spot for each label
            changed = False
            for i in range(n):
                b = min(range(len(cands[i])), key=lambda a: local(i, a))
                if local(i, b) < local(i, cur[i]) - 1e-9:
                    cur[i], changed = b, True
            if not changed:
                break
        c = total()
        if best_cost is None or c < best_cost:
            best_cost, best_layout = c, list(cur)
    cur[:] = best_layout
    return [(cands[i][cur[i]][1][0] - pts[i][0], cands[i][cur[i]][1][1] - pts[i][1])
            for i in range(n)]


def _mpl_point_labels(fig, ax, used, out, boxes, ycol="sigma_T_ps", errors=True):
    """Ion labels with arrows, placed by place_labels (display pixels, y up).
    ycol: column with the plotted y values; errors: avoid the error bars."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    ecol = "d_sigma_T_total_ps" if "d_sigma_T_total_ps" in used else "d_sigma_T_ps"
    rows = [(r["ion"], r["T_ns"], r[ycol], r[ecol] if errors else np.nan, "black")
            for r in used.to_dict("records")] + \
           [(r["ion"], r["T_ns"], r[ycol], r["d_sigma_T_ps"] if errors else np.nan, "#999999")
            for r in out.to_dict("records")]
    if not rows:
        return
    bb = ax.get_window_extent(rend)
    rect = (bb.x0, bb.y0, bb.x1, bb.y1)
    pts, sizes, obst, bars = [], [], [], []
    for ion, t, sg, e, _ in rows:
        x, y = ax.transData.transform((t, sg))
        pts.append((x, y))
        tt = ax.text(0, 0, ion, fontsize=ISO_LABEL_PT)
        ext = tt.get_window_extent(rend)
        tt.remove()
        pad = 2 * (ISO_LABEL_BOX_PAD * ISO_LABEL_PT * fig.dpi / 72.0 + 1)   # box around it
        sizes.append((ext.width + pad, ext.height + pad))
        obst.append((x - 7, y - 7, x + 7, y + 7))
        if np.isfinite(e) and e > 0:
            _, ylo = ax.transData.transform((t, sg - e))
            _, yhi = ax.transData.transform((t, sg + e))
            obst.append((x - 4, ylo, x + 4, yhi))
            bars.append((x - 3, min(ylo, yhi), x + 3, max(ylo, yhi)))
    for line in ax.get_lines():                      # the curve
        xy = ax.transData.transform(np.column_stack(line.get_data()))
        for k in range(0, len(xy), max(1, len(xy) // 150)):
            obst.append((xy[k, 0] - 3, xy[k, 1] - 3, xy[k, 0] + 3, xy[k, 1] + 3))
    for b in boxes:
        if b is not None:
            e = b.get_window_extent(rend)
            obst.append((e.x0, e.y0, e.x1, e.y1, 20.0))      # never under these
    offs = place_labels(pts, sizes, obst, rect, y_down=False, bars=bars)
    k = 72.0 / fig.dpi                                 # pixels -> points
    for (ion, t, sg, _, col), (dx, dy) in zip(rows, offs):
        ax.annotate(ion, xy=(t, sg), xytext=(dx * k, dy * k), textcoords="offset points",
                    ha="center", va="center", fontsize=ISO_LABEL_PT, color=col,
                    bbox=dict(boxstyle=f"square,pad={ISO_LABEL_BOX_PAD}", facecolor="white",
                              edgecolor="black", linewidth=0.8),
                    arrowprops=dict(arrowstyle="-|>", color="#888888", lw=0.8,
                                    shrinkA=1, shrinkB=4, mutation_scale=8))


def draw_iso_figure(vals, res, st):
    """Matplotlib figure (curve + residuals), saved as PNG and PDF in
    outputs/figures/.  Returns the file paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    used = vals[vals["include"] & vals["fittable"]]
    out = vals[~(vals["include"] & vals["fittable"])]
    ecol = "d_sigma_T_total_ps" if "d_sigma_T_total_ps" in vals else "d_sigma_T_ps"
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(16, 19), sharex=True,
                                  gridspec_kw={"height_ratios": [13, 6], "hspace": 0.06})
    ax.errorbar(used["T_ns"], used["sigma_T_ps"], yerr=used[ecol], fmt="none",
                color="grey", capsize=4, zorder=2)
    ax.scatter(used["T_ns"], used["sigma_T_ps"], color="red", marker="x", s=50,
               label="Identified ions", zorder=3)
    if len(out):
        ax.scatter(out["T_ns"], out["sigma_T_ps"], facecolors="none", edgecolors="#aaaaaa",
                   s=50, label="Left out of the fit", zorder=2)
    if res:
        x = _iso_curve_x(vals)
        ax.plot(x, iso_sigma_t(x, *res["popt"], st["circumference_m"], st["c_m_s"]),
                color="black", label="Isochronicity curve")
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ptext = ax.text(0.02, 0.97, "\n".join(_iso_param_lines(res, extras=False)),
                        transform=ax.transAxes,
                        fontsize=16, verticalalignment="top", bbox=props)
        # residuals in units of each point's error (decay-fit panel), styled
        # like the main plot: red ×, black zero line, grey left-out circles, labels
        pull, pull_txt = _iso_pulls(used, res, st, ecol)
        axr.axhspan(-1.0, 1.0, color="0.85", zorder=0)
        xr = _iso_curve_x(vals)
        axr.plot([xr[0], xr[-1]], [0, 0], color="black")
        axr.scatter(used["T_ns"], pull, color="red", marker="x", s=50, zorder=3)
        r_used = used.assign(pull=pull)
        r_out = out.iloc[0:0].assign(pull=[])
        if len(out):
            opull = _iso_pull_values(out, res, st, ecol)
            ok = np.isfinite(opull)
            r_out = out[ok].assign(pull=opull[ok])
            axr.scatter(r_out["T_ns"], r_out["pull"], facecolors="none", edgecolors="#aaaaaa",
                        s=50, zorder=2)
        rtext = axr.text(0.995, 0.95, pull_txt, transform=axr.transAxes, ha="right", va="top",
                         fontsize=13, bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                                                edgecolor="0.7", alpha=0.85))
    else:
        ptext = None
    ax.set_ylabel(r"$\sigma_{T}$ (ps)", size=22)
    axr.set_ylabel(r"(σ(T) − fit)/$\sigma$", size=16)
    axr.set_xlabel("Revolution Time (ns)", size=22)
    ax.tick_params(labelsize=16)
    axr.tick_params(labelsize=16)
    leg = ax.legend(fontsize=16, loc="lower right")
    _mpl_point_labels(fig, ax, used, out, [ptext, leg])
    if res:
        _mpl_point_labels(fig, axr, r_used, r_out, [rtext], ycol="pull", errors=False)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(paths.outputs("figures"), f"isocurve_{stamp}")
    files = [base + ".png", base + ".pdf"]
    fig.savefig(files[0], dpi=150, bbox_inches="tight")
    fig.savefig(files[1], bbox_inches="tight")
    plt.close(fig)
    return files


# ---------------------------------------------------------------------------
# Quit
# ---------------------------------------------------------------------------

# The page checks in regularly: its tab closes when Theia stops, and Theia
# stops (a running scan too) a few seconds after its last tab is closed.
_TABS = tabwatch.TabTracker(on_last_closed=lambda: os._exit(0), name="Theia")
tabwatch.install_flask(app.server, _TABS)


@app.server.route("/_theia_quit", methods=["POST"])
def _theia_quit():
    """Stop the program shortly after answering (so the browser gets the
    reply).  The Further output state is already saved after every change."""
    print("Quit pressed in the browser: stopping Theia.")
    threading.Timer(0.5, lambda: os._exit(0)).start()
    return "stopping"


app.clientside_callback(
    """
    function(n) {
        if (!n) return window.dash_clientside.no_update;
        if (!window.confirm('Quit Theia?\\n\\nThis stops Theia ' +
                            '(a scan that is running is stopped too).')) {
            return window.dash_clientside.no_update;
        }
        fetch('_theia_quit', {method: 'POST'}).catch(function() {}).then(function() {
            window.close();
            // browsers only let a script close tabs it opened itself; if this
            // tab is still open, replace the page with a message
            setTimeout(function() {
                document.body.innerHTML =
                    '<div style="font-family:sans-serif;padding:60px;text-align:center;">' +
                    '<h3 style="color:#f57d15">Theia has stopped</h3>' +
                    '<p>You can close this tab.</p></div>';
            }, 300);
        });
        return window.dash_clientside.no_update;
    }
    """,
    Output("_quit-hook", "children"),
    Input("btn-quit", "n_clicks"),
    prevent_initial_call=True,
)


# ===========================================================================
# Aligning and fitting: energy calculation
# ===========================================================================
#
# 1. Calibration: pick two peaks (a fitted peak's centre is used if you click
#    on one, otherwise the peak is fitted on its own in a small window), name
#    them (NUBASE names, e.g. 207Tl81+ or m207Tl), press Calibrate:
#        k = (ME2 - ME1) / (f2 - f1)        [keV per Hz]
#    For an isomer / ground-state pair of the same nuclide, ME2 - ME1 is the
#    excitation-energy difference (with its own, smaller uncertainty).
# 2. Then pick peaks in pairs A, B:  dE = E(B) - E(A) = k (f_B - f_A).

ENERGY_TRACE = "energy picks"
ENERGY_PAIR_TRACE = "energy pairs"
# index of the first simulated-line trace in the summed figure: spectrum,
# 4 fit traces, energy picks, energy pairs, then the 3 simulated-line traces
SIM_TRACE_INDEX = 1 + len(FIT_TRACE_STYLES) + 2


@lru_cache(maxsize=1)
def _nubase_table():
    d = get_base_dataframe()
    d = d.copy()
    for c in ("E_x", "E_x_unc"):
        d[c] = pd.to_numeric(d[c].astype(str).str.replace("#", "", regex=False), errors="coerce")
    return d


def lookup_mass_excess(name):
    """NUBASE entry for an ion/nuclide name ('207Tl81+', 'm207Tl', '208Pb').
    Returns dict(name, A, Z, iso, me, dme, ex, dex, sys) or raises ValueError."""
    m = re.fullmatch(r"\s*([a-z]?)(\d+)([A-Z][a-z]?)(\d+\+)?\s*", name or "")
    if not m:
        raise ValueError(f"'{name}' is not a name like 207Tl81+, m207Tl or 208Pb")
    iso, A, sym = m.group(1), m.group(2), m.group(3)
    d = _nubase_table()
    rows = d[d["Element"].astype(str).str.strip() == f"{A}{sym}"]
    rows = rows[rows["Isomer_Flag"] == iso] if iso else rows[rows["Isomer_Flag"].isna()]
    if not len(rows):
        raise ValueError(f"'{iso}{A}{sym}' is not in NUBASE")
    r = rows.iloc[0]
    if pd.isna(r["m.e."]):
        raise ValueError(f"no mass excess for '{iso}{A}{sym}' in NUBASE")
    ex = 0.0 if not iso else (float(r["E_x"]) if pd.notna(r["E_x"]) else np.nan)
    dex = 0.0 if not iso else (float(r["E_x_unc"]) if pd.notna(r["E_x_unc"]) else np.nan)
    return dict(name=f"{iso}{A}{sym}", A=int(A), Z=int(r["ZZZi"]), iso=iso,
                me=float(r["m.e."]), dme=float(r["m.e._unc"]) if pd.notna(r["m.e._unc"]) else np.nan,
                ex=ex, dex=dex, sys=bool(r["m.e. sys?"]))


def mass_excess_difference(n1, n2):
    """ME(n2) - ME(n1) in keV with its uncertainty.  For two states of the
    same nuclide the excitation energies are used (correlated ME errors)."""
    a, b = lookup_mass_excess(n1), lookup_mass_excess(n2)
    if (a["A"], a["Z"]) == (b["A"], b["Z"]) and np.isfinite(a["ex"]) and np.isfinite(b["ex"]):
        return b["ex"] - a["ex"], float(np.hypot(a["dex"], b["dex"])), a, b, "excitation energies"
    return b["me"] - a["me"], float(np.hypot(a["dme"], b["dme"])), a, b, "mass excesses"


def local_peak_fit(ff, pp, sigma, x, shape):
    """Fit one peak (with a constant background) in a small window around x,
    independently of the main fit.  Returns (centre, uncertainty, fwhm)."""
    pk = _snap_peak(ff, pp, x, [0.0, 0.0], "none")
    bin_hz = float(np.median(np.diff(ff)))
    # window of +-1 FWHM around the maximum: wide enough for the peak, narrow
    # enough to keep neighbouring peaks out
    half = max(1.0 * pk["fwhm"], 6 * bin_hz)
    region = [pk["centre"] - half, pk["centre"] + half]
    m = _region_mask(ff, region)
    base = float(np.nanmin(pp[m])) if m.any() else 0.0
    pk = _snap_peak(ff, pp, x, [base, 0.0], "const")
    pk["shape"] = shape
    # the centre may only move within half a FWHM of the clicked maximum, so
    # the fit cannot slide onto a bigger neighbouring peak
    w = max(0.5 * pk["fwhm"], 2 * bin_hz)
    st = {"peaks": [pk], "bg": "const", "bgp": [base, 0.0], "region": region,
          "xref": pk["centre"], "shape": shape,
          "centre_bounds": [[pk["centre"] - w, pk["centre"] + w]],
          "fwhm_bounds": [[0.3 * pk["fwhm"], 3.0 * pk["fwhm"]]]}
    res = run_fit(ff, pp, sigma, st)
    c = res["peaks"][0]["centre"]
    dc = res["unc"]["peaks"][0].get("centre", np.nan)
    dw = res["unc"]["peaks"][0].get("fwhm", np.nan)
    return float(c), float(dc), float(res["peaks"][0]["fwhm"]), float(dw)


def pick_peak(ff, pp, sigma, x, fit_state, shape):
    """Frequency for a click at x: the centre of a peak in the main fit if the
    click is on it (within half its FWHM), otherwise a local single-peak fit."""
    bin_hz = float(np.median(np.diff(ff)))
    if fit_state and fit_state.get("peaks"):
        unc = (fit_state.get("unc") or {}).get("peaks") or [{} for _ in fit_state["peaks"]]
        best = None
        for i, pk in enumerate(fit_state["peaks"]):
            d = abs(pk["centre"] - x)
            if d <= max(0.5 * pk["fwhm"], 2 * bin_hz) and (best is None or d < best[0]):
                best = (d, i)
        if best is not None:
            i = best[1]
            pk = fit_state["peaks"][i]
            du = unc[i].get("centre") if i < len(unc) else None
            src = f"main fit, peak {i + 1}" + ("" if fit_state.get("fitted") else " (not fitted)")
            dw = unc[i].get("fwhm") if i < len(unc) else None
            return {"f": float(pk["centre"]), "df": float(du) if du is not None else np.nan,
                    "fwhm": float(pk["fwhm"]), "dfwhm": float(dw) if dw is not None else np.nan,
                    "src": src}
    c, dc, w, dw = local_peak_fit(ff, pp, sigma, x, shape)
    return {"f": c, "df": dc, "fwhm": w, "dfwhm": dw, "src": "local fit"}


def _fmt_pm(v, dv, fmt=".1f"):
    if v is None or not np.isfinite(v):
        return "–"
    return format(v, fmt) + ("" if dv is None or not np.isfinite(dv) else " ± " + format(dv, fmt))


def suggest_ions(key, f, n=6):
    """Simulated ions nearest to frequency f (from the summed plot's lines)."""
    sims = _SUM_SIMS.get(key)
    if sims is None or not len(sims):
        return []
    d = (sims["f_sim"] - f).abs()
    near = sims.assign(d=d).sort_values("d").head(n)
    return [(r.ion, int(r.harmonic), float(r.f_sim - f)) for r in near.itertuples()]


def _empty_energy():
    return {"calib": [], "k": None, "dk": None, "pending": None, "pairs": [], "msg": ""}


@app.callback(
    Output("energy-state", "data"),
    Output("energy-name-1", "value"),
    Output("energy-name-2", "value"),
    Input("energy-click", "data"),
    Input("energy-name-1", "value"),
    Input("energy-name-2", "value"),
    Input({"type": "energy-sugg", "peak": ALL, "name": ALL, "h": ALL}, "n_clicks"),
    Input("btn-energy-calibrate", "n_clicks"),
    Input("btn-energy-reset", "n_clicks"),
    Input("btn-energy-clear-pairs", "n_clicks"),
    Input("sum-key", "data"),
    State("energy-state", "data"),
    State("fit-state", "data"),
    State("fit-shape", "value"),
    prevent_initial_call=True,
)
def update_energy(click, name1, name2, _sugg, _cal, _reset, _clear, key, est, fit_state, shape):
    trig = dash.callback_context.triggered[0] if dash.callback_context.triggered else None
    nu = dash.no_update
    if trig is None:
        return nu, nu, nu
    pid, val = trig["prop_id"], trig["value"]
    est = dict(est or _empty_energy())
    est["msg"] = ""
    if pid.startswith("sum-key"):
        saved = _RESTORE_ENERGY.pop(key, None)          # saved last session
        if saved:
            names = saved.get("names") or ["", ""]
            return dict(_empty_energy(), **(saved.get("state") or {})), names[0], names[1]
        return _empty_energy(), "", ""
    if pid.startswith("btn-energy-reset"):
        return _empty_energy(), "", ""
    if pid.startswith("btn-energy-clear-pairs"):
        est["pairs"], est["pending"] = [], None
        return est, nu, nu
    if '"energy-sugg"' in pid:
        if not val:
            return nu, nu, nu
        idd = json.loads(pid.rsplit(".", 1)[0])
        if idd["peak"] == 1:
            return nu, idd["name"], nu
        return nu, nu, idd["name"]
    if pid.startswith("energy-name"):
        est["k"], est["dk"] = None, None              # names changed: recalibrate
        return est, nu, nu

    with _SUM_CACHE_LOCK:
        cached = _SUM_CACHE.get(key)
    if cached is None:
        est["msg"] = "Sum the spectra first."
        return est, nu, nu
    ff, pp, sigma = cached

    if pid.startswith("energy-click") and click:
        try:
            pk = pick_peak(ff, pp, sigma, float(click["x"]), fit_state, shape or "gauss")
        except Exception as exc:
            traceback.print_exc()
            est["msg"] = f"Could not fit the peak there: {exc}"
            return est, nu, nu
        if est.get("k") is None:                      # calibration step
            if len(est["calib"]) >= 2:
                est["msg"] = ("Two calibration peaks are already picked: name them and press "
                              "Calibrate, or Reset to pick again.")
                return est, nu, nu
            pk["suggest"] = suggest_ions(key, pk["f"])
            est["calib"] = est["calib"] + [pk]
            return est, nu, nu
        if est.get("pending") is None:                # first peak (A) of a pair
            est["pending"] = pk
            return est, nu, nu
        a, b = est["pending"], pk
        dfreq = b["f"] - a["f"]
        ddf = float(np.hypot(np.nan_to_num(a["df"]), np.nan_to_num(b["df"])))
        de = est["k"] * dfreq
        dde = float(np.hypot(est["k"] * ddf, dfreq * est["dk"]))
        est["pairs"] = est["pairs"] + [{"a": a, "b": b, "dfreq": dfreq, "ddf": ddf,
                                        "de": de, "dde": dde}]
        est["pending"] = None
        return est, nu, nu

    if pid.startswith("btn-energy-calibrate"):
        if len(est["calib"]) < 2:
            est["msg"] = "Pick two calibration peaks first."
            return est, nu, nu
        try:
            dme, ddme, a, b, how = mass_excess_difference(name1, name2)
            if a["name"] == b["name"]:
                raise ValueError("the two peaks have the same name (no energy difference)")
        except Exception as exc:
            est["msg"] = f"Calibration: {exc}"
            return est, nu, nu
        p1, p2 = est["calib"]
        dfreq = p2["f"] - p1["f"]
        if abs(dfreq) < 1e-9:
            est["msg"] = "The two calibration peaks have the same frequency."
            return est, nu, nu
        s1, s2 = np.nan_to_num(p1["df"]), np.nan_to_num(p2["df"])
        k = dme / dfreq
        dk = float(np.sqrt((ddme / dfreq) ** 2 + (dme * np.hypot(s1, s2) / dfreq ** 2) ** 2))
        est.update(k=float(k), dk=dk, names=[a["name"], b["name"]], dme=float(dme),
                   ddme=float(ddme), how=how, sys=bool(a["sys"] or b["sys"]))
        est["pairs"], est["pending"] = [], None
        return est, nu, nu
    return nu, nu, nu


@app.callback(
    Output("energy-sugg-1", "style"),
    Output("energy-sugg-2", "style"),
    Input("energy-suggest-btn-1", "n_clicks"),
    Input("energy-suggest-btn-2", "n_clicks"),
    Input({"type": "energy-sugg", "peak": ALL, "name": ALL, "h": ALL}, "n_clicks"),
    State("energy-sugg-1", "style"),
    State("energy-sugg-2", "style"),
    prevent_initial_call=True,
)
def toggle_suggestions(_b1, _b2, _rows, st1, st2):
    """'Suggest' opens/closes that peak's list; choosing an ion closes it."""
    trig = dash.callback_context.triggered[0] if dash.callback_context.triggered else None
    if not trig:
        return dash.no_update, dash.no_update
    pid = trig["prop_id"]
    open_ = lambda st: (st or {}).get("display") != "none"
    show = lambda on: {**SUGG_BOX_STYLE, "display": "block" if on else "none"}
    if pid.startswith("energy-suggest-btn-1"):
        return show(not open_(st1)), show(False)
    if pid.startswith("energy-suggest-btn-2"):
        return show(False), show(not open_(st2))
    if '"energy-sugg"' in pid and trig["value"]:
        return show(False), show(False)
    return dash.no_update, dash.no_update


@app.callback(
    Output("energy-calib-info", "children"),
    Output("energy-sugg-1", "children"),
    Output("energy-sugg-2", "children"),
    Output("energy-name-1-check", "children"),
    Output("energy-name-2-check", "children"),
    Output("energy-f-1", "children"),
    Output("energy-f-2", "children"),
    Output("energy-fwhm-1", "children"),
    Output("energy-fwhm-2", "children"),
    Output("energy-status", "children"),
    Output("energy-table", "data"),
    Output("energy-panel", "style"),
    Input("energy-state", "data"),
    Input("energy-name-1", "value"),
    Input("energy-name-2", "value"),
    Input("energy-mode", "data"),
)
def render_energy(est, name1, name2, on):
    est = est or _empty_energy()
    calib = est.get("calib") or []

    def sugg(i):
        """Table of nearby simulated ions (nearest first); rows are clickable."""
        if i >= len(calib):
            return html.Div("Click the peak on the plot first.",
                            style={"fontSize": "12px", "color": "#888", "padding": "4px 12px"})
        rows = calib[i].get("suggest") or []
        if not rows:
            return html.Div("No simulated ions near this peak (sum the spectra first?).",
                            style={"fontSize": "12px", "color": "#888", "padding": "4px 12px"})
        cell = {"padding": "3px 12px", "fontSize": "13px"}
        head = {**cell, "fontSize": "11px", "fontWeight": "700", "color": "#888",
                "textTransform": "uppercase", "letterSpacing": "0.04em"}
        grid = {"display": "grid", "gridTemplateColumns": "1.3fr 0.9fr 1.4fr 1fr"}
        f0 = calib[i]["f"]
        out = [html.Div(f"{len(rows)} nearby, nearest centre first, click one",
                        style={"fontSize": "12px", "color": "#2e7d32", "padding": "2px 12px 4px"}),
               html.Div([html.Span("Ion", style=head), html.Span("Harmonic", style=head),
                         html.Span("f_sim (Hz)", style=head), html.Span("From centre", style=head)],
                        style={**grid, "borderBottom": "1px solid #dddddd"})]
        for ion, h, d in rows:
            out.append(html.Div([
                html.Span(ion, style={**cell, "fontWeight": "600"}),
                html.Span(f"h{h}", style=cell),
                html.Span(f"{f0 + d:,.1f}", style={**cell, "fontFamily": "monospace"}),
                html.Span(f"{d:+.0f} Hz", style={**cell, "fontFamily": "monospace"}),
            ], id={"type": "energy-sugg", "peak": i + 1, "name": ion, "h": h}, n_clicks=0,
               className="theia-fb-row", style={**grid, "cursor": "pointer"}))
        return out

    def check(name):
        """Mass excess column (NUBASE), or why the name is not valid."""
        if not name:
            return html.Span("–", style={"color": "#888"})
        try:
            r = lookup_mass_excess(name)
            out = [html.Span(f"{r['me']:.1f} ± {r['dme']:.1f}" + (" #" if r["sys"] else ""),
                             style={"fontFamily": "monospace"})]
            if r["iso"]:
                out.append(html.Div(f"E_x = {r['ex']:.2f} ± {r['dex']:.2f}",
                                    style={"fontSize": "12px", "color": "#666",
                                           "fontFamily": "monospace"}))
            return out
        except ValueError as exc:
            return html.Span(f"✗ {exc}", style={"color": "#c0392b", "fontSize": "12px"})

    def freq(i):
        if i >= len(calib):
            return html.Span("click the peak on the plot", style={"color": "#888",
                                                               "fontFamily": "inherit"})
        p = calib[i]
        src = re.sub(r"\s*\(FWHM [^)]*\)", "", p["src"])     # FWHM goes in its own column
        return [html.Span(_fmt_pm(p["f"], p["df"])),
                html.Div(src, style={"fontSize": "11px", "color": "#888", "fontFamily": "inherit"})]

    def fwhm(i):
        if i >= len(calib):
            return html.Span("–", style={"color": "#888"})
        p = calib[i]
        if p_get(p, "fwhm") is None:
            # peaks picked by an older version only have it in the source text
            m = re.search(r"FWHM ([\d.]+) Hz", p.get("src", ""))
            return m.group(1) if m else html.Span("–", style={"color": "#888"})
        return _fmt_pm(p["fwhm"], p.get("dfwhm"))

    def p_get(p, k):
        v = p.get(k)
        return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else v

    info = []
    if est.get("k") is not None:
        info.append(html.Div([
            html.B("Calibration: "),
            f"k = {est['k']:.6g} ± {est['dk']:.2g} keV/Hz   "
            f"(ΔE = {est['dme']:.2f} ± {est['ddme']:.2f} keV from {est['how']} of "
            f"{est['names'][1]} − {est['names'][0]})"
            + ("  · includes estimated (#) NUBASE values" if est.get("sys") else "")],
            style={"marginTop": "4px"}))

    if est.get("msg"):
        status = html.Span(est["msg"], style={"color": "#c0392b"})
    elif not on:
        status = "Turn on Energy calculation to pick peaks on the plot."
    elif est.get("k") is None:
        status = ("Step 1 – calibration: click two peaks, enter their names (or click a "
                  "suggestion), then press Calibrate." if len(calib) < 2 else
                  "Step 1 – enter the names of the two peaks and press Calibrate.")
    elif est.get("pending") is None:
        status = "Step 2 – click peak A of a pair (the energy difference is E(B) − E(A))."
    else:
        status = f"Step 2 – peak A at {est['pending']['f']:.1f} Hz picked; now click peak B."

    rows = [{"pair": i, "f_A": round(p["a"]["f"], 1), "f_B": round(p["b"]["f"], 1),
             "df": _fmt_pm(p["dfreq"], p["ddf"]), "de": _fmt_pm(p["de"], p["dde"], ".2f"),
             "src": f"A: {p['a']['src']};  B: {p['b']['src']}"}
            for i, p in enumerate(est.get("pairs") or [], start=1)]
    # panel shown while the mode is on, or whenever there is something in it
    has = bool(calib or est.get("pairs") or est.get("k") is not None)
    panel = {"display": "block" if (on or has) else "none"}
    return (info, sugg(0), sugg(1), check(name1), check(name2), freq(0), freq(1), fwhm(0),
            fwhm(1), status, rows, panel)


@app.callback(
    Output("sum-graph", "figure", allow_duplicate=True),
    Input("fit-state", "data"),
    Input("energy-state", "data"),
    State("sum-view", "data"),
    State("sum-key", "data"),
    prevent_initial_call=True,
)
def draw_overlays(fit_state, est, view, key):
    """Update the fit and energy overlays of the summed plot in ONE Patch.

    (Two callbacks patching the figure at the same moment - as happens when
    both are restored on start-up - can lose one of the updates.)  Runs only
    when the fit or the energy picks change, never on zoom/pan: patching the
    131k-point figure during a zoom makes the plot fight the zoom."""
    with _SUM_CACHE_LOCK:
        cached = _SUM_CACHE.get(key)
    if cached is None:
        return dash.no_update
    trig = dash.callback_context.triggered[0]["prop_id"] if dash.callback_context.triggered else ""
    patch = dash.Patch()
    if not fit_state or fit_state.get("sum_key") == key:
        _fit_into_patch(patch, fit_state, cached)
    _energy_into_patch(patch, est, fit_state, view, key, cached, trig)
    return patch


def _fit_shapes(fit_state, key):
    """Fit region (two draggable orange lines, named fit-region-lo/hi, with a
    pale tint between them) and each peak's centre limits (grey band)."""
    if not fit_state or fit_state.get("sum_key") != key or not fit_state.get("region"):
        return []
    lo, hi = fit_state["region"]
    line = dict(color=ACCENT, width=2)
    out = [dict(type="rect", xref="x", yref="paper", x0=lo, x1=hi, y0=0, y1=1,
                fillcolor=ACCENT_PALE, opacity=0.5, line_width=0, layer="below",
                name="fit-region-tint"),
           dict(type="line", xref="x", yref="paper", x0=lo, x1=lo, y0=0, y1=1, line=line,
                layer="above", name="fit-region-lo"),
           dict(type="line", xref="x", yref="paper", x0=hi, x1=hi, y0=0, y1=1, line=line,
                layer="above", name="fit-region-hi")]
    for pk in fit_state.get("peaks") or []:
        lim = pk.get("lim") or {}
        cmin, cmax = lim.get("centre_min"), lim.get("centre_max")
        if cmin is None and cmax is None:
            continue
        out.append(dict(type="rect", xref="x", yref="paper",
                        x0=lo if cmin is None else max(cmin, lo),
                        x1=hi if cmax is None else min(cmax, hi), y0=0, y1=0.04,
                        fillcolor="#999999", opacity=0.5, line_width=0, layer="above",
                        name="fit-centre-limits"))
    return out


def _energy_into_patch(patch, est, fit_state, view, key, cached, trig):
    """Picked-peak labels (C1, C2, A1/B1, ...) and the pair brackets + labels."""
    ff, pp, _ = cached
    est = est or _empty_energy()
    pts = [(p["f"], f"C{i}") for i, p in enumerate(est.get("calib") or [], start=1)]
    for i, pr in enumerate(est.get("pairs") or [], start=1):
        pts += [(pr["a"]["f"], f"A{i}"), (pr["b"]["f"], f"B{i}")]
    if est.get("pending"):
        pts.append((est["pending"]["f"], f"A{len(est.get('pairs') or []) + 1}"))
    merged = {}                                   # one label per position ("C1, A1")
    for f, lab in pts:
        merged.setdefault(round(f, 1), []).append(lab)
    pts = [(f, ", ".join(labs)) for f, labs in merged.items()]
    idx = 1 + len(FIT_TRACE_STYLES)
    patch["data"][idx]["x"] = [p[0] for p in pts]
    patch["data"][idx]["y"] = [float(np.interp(p[0], ff, pp)) for p in pts]
    patch["data"][idx]["text"] = [p[1] for p in pts]

    # a bracket just above each pair, from centre A to centre B, with dE +- sigma.
    # Drawn as layout shapes whose vertical size is in PIXELS above the
    # highest data / fit point between the peaks (ysizemode="pixel"), so it
    # sits the same distance above the peaks at any zoom without redrawing.
    fx, fy = fit_overlay(ff, fit_state)[0] if fit_state else ([], [])
    fx, fy = np.asarray(fx, dtype=float), np.asarray(fy, dtype=float)
    shapes, labels = [], []
    line = dict(color="black", width=1.5)
    for i, pr in enumerate(est.get("pairs") or []):
        fa, fb = pr["a"]["f"], pr["b"]["f"]
        lo, hi = min(fa, fb), max(fa, fb)
        ya, yb = float(np.interp(fa, ff, pp)), float(np.interp(fb, ff, pp))
        m = (ff >= lo) & (ff <= hi)
        top = max(ya, yb, float(np.nanmax(pp[m])) if m.any() else ya)   # clear the data
        mf = (fx >= lo) & (fx <= hi)
        if mf.any():                                                    # ... and the fit
            top = max(top, float(np.nanmax(fy[mf])))
        # pixels above `top`: clear of the peak numbers / pick labels; later
        # pairs are stacked higher
        h = 34 + 46 * i
        mid = 0.5 * (fa + fb)
        common = dict(type="line", xref="x", yref="y", ysizemode="pixel", yanchor=top,
                      line=line, layer="above")
        shapes += [dict(common, x0=fa, x1=fb, y0=h, y1=h),              # horizontal
                   dict(common, x0=fa, x1=fa, y0=h - 7, y1=h),          # tick down at A
                   dict(common, x0=fb, x1=fb, y0=h - 7, y1=h),          # tick down at B
                   dict(common, x0=mid, x1=mid, y0=h, y1=h + 7)]        # tick up, middle
        # label: drawn on top of everything, black text on an opaque white
        # box with a black border
        labels.append(dict(x=mid, y=top, xref="x", yref="y", yshift=h + 8, yanchor="bottom",
                           text=f"ΔE = {_fmt_pm(pr['de'], pr['dde'], '.2f')} keV",
                           showarrow=False, font=dict(color="black", size=26),
                           bgcolor="white", bordercolor="black", borderwidth=1.5, borderpad=4,
                           opacity=1))
    patch["data"][idx + 1]["x"] = []                 # (old bracket trace, no longer used)
    patch["data"][idx + 1]["y"] = []
    patch["layout"]["shapes"] = (list(_SUM_SHAPES.get(key, [])) + _fit_shapes(fit_state, key)
                                 + shapes)
    patch["layout"]["annotations"] = list(_SUM_ANNOTATIONS.get(key, [])) + labels
    return patch


@app.callback(
    Output("pid-modal", "style"),
    *[Output(f"pid-{k}", "value") for k in PID_KEYS],
    Output("pid-error", "children"),
    Output("pid-settings", "data"),
    Input("btn-pid", "n_clicks"),
    Input("pid-apply", "n_clicks"),
    Input("pid-reset", "n_clicks"),
    Input("pid-cancel", "n_clicks"),
    *[State(f"pid-{k}", "value") for k in PID_KEYS],
    State("further-files", "data"),
    State("pid-settings", "data"),
    State("further-source", "value"),
    prevent_initial_call=True,
)
def pid_popup(_open, _apply, _reset, _cancel, *args):
    """PID Settings popup: opens filled with the settings in use for the
    chosen data source (file / Viewer settings + edits); Apply validates and
    stores the edits; Reset to file drops them."""
    values, (files, store, source) = list(args[:len(PID_KEYS)]), args[len(PID_KEYS):]
    source = source or "sum"
    overrides = pid_overrides(store, source)

    def stored(over):
        new = {"sum": pid_overrides(store, "sum"), "file": pid_overrides(store, "file")}
        new[source] = over
        return new
    shown = {"display": "flex", "position": "fixed", "inset": "0", "zIndex": "1000",
             "background": "rgba(0,0,0,0.35)", "alignItems": "center", "justifyContent": "center"}
    hidden = {**shown, "display": "none"}
    nu = dash.no_update
    keep = [nu] * len(PID_KEYS)
    trig = dash.callback_context.triggered[0]["prop_id"] if dash.callback_context.triggered else ""

    def filled(over):
        raw = {**pid_base(source, files), **(over or {})}
        return [_pid_display(raw, k) for k in PID_KEYS]

    if trig.startswith("pid-cancel"):
        return (hidden, *keep, "", nu)
    try:
        if trig.startswith("btn-pid"):
            return (shown, *filled(overrides), "", nu)
        if trig.startswith("pid-reset"):
            msg = ("Reset to the Viewer settings." if source == "file"
                   else "Reset to the settings of the loaded file.")
            return (shown, *filled(None), msg, stored(None))
        if trig.startswith("pid-apply"):
            new = dict(zip(PID_KEYS, values))
            s = pid_settings(files, new, source)             # validates the numbers
            build_ion_database(*s["ion_key"])                # ...and the reference ion
            return (hidden, *keep, "", stored(new))
    except Exception as exc:
        msg = html.Span(f"⚠ {exc}", style={"color": "#c0392b"})
        if trig.startswith("btn-pid"):
            return (shown, *keep, msg, nu)
        return (nu, *keep, msg, nu)
    return (nu, *keep, nu, nu)


@app.callback(
    Output("sum-graph", "figure", allow_duplicate=True),
    Output("sum-sim-info", "children", allow_duplicate=True),
    Input("pid-settings", "data"),
    State("sum-key", "data"),
    State("further-files", "data"),
    State("sum-unsure", "value"),
    State("sum-show-sims", "value"),
    prevent_initial_call=True,
)
def apply_pid(store, key, files, unsure, show_sims):
    """Redraw only the simulated lines with the new PID settings of the
    plotted data source (no re-sum; the zoom is kept)."""
    with _SUM_CACHE_LOCK:
        cached = _SUM_CACHE.get(key)
    if cached is None:
        return dash.no_update, dash.no_update
    ff, pp, _ = cached
    source = (_SUM_META.get(key) or {}).get("source", "sum")
    try:
        traces, text = sum_sim_traces(ff, pp, files, bool(unsure), pid_overrides(store, source),
                                      key=key, source=source)
    except Exception as exc:
        traceback.print_exc()
        return dash.no_update, f"simulated lines not drawn: {exc}"
    patch = dash.Patch()
    base = SIM_TRACE_INDEX
    for i, (x, y, h) in enumerate(traces):
        patch["data"][base + i]["x"] = x
        patch["data"][base + i]["y"] = y
        patch["data"][base + i]["hovertext"] = h
        patch["data"][base + i]["visible"] = bool(show_sims)
    return patch, text


@app.callback(
    Output("info-modal", "style"),
    Output("info-title", "children"),
    Output("info-body", "children"),
    Input({"type": "info-btn", "key": ALL}, "n_clicks"),
    Input("info-close", "n_clicks"),
    prevent_initial_call=True,
)
def show_info(_clicks, _close):
    """ⓘ buttons: show that section's help in a popup; ✕ closes it."""
    shown = {"display": "flex", "position": "fixed", "inset": "0", "zIndex": "1000",
             "background": "rgba(0,0,0,0.35)", "alignItems": "center", "justifyContent": "center"}
    trig = dash.callback_context.triggered[0] if dash.callback_context.triggered else None
    if not trig or not trig["value"]:
        return dash.no_update, dash.no_update, dash.no_update
    if trig["prop_id"].startswith("info-close"):
        return {**shown, "display": "none"}, dash.no_update, dash.no_update
    key = json.loads(trig["prop_id"].rsplit(".", 1)[0])["key"]
    title, lines = INFO_TEXTS.get(key, ("Information", []))
    body = html.Ul([html.Li(t, style={"marginBottom": "4px"}) for t in lines],
                   style={"paddingLeft": "18px", "margin": "0"})
    return shown, title, body


# ===========================================================================
# Callbacks — settings popup menu and dark mode (all pages)
# ===========================================================================

app.clientside_callback(
    """
    function(n) {
        var menu = document.getElementById('settings-menu');
        var btn = document.getElementById('btn-settings');
        if (!menu || !btn) return window.dash_clientside.no_update;
        if (!window._theiaMenuListener) {
            // close when clicking anywhere outside the menu and the button
            window._theiaMenuListener = true;
            document.addEventListener('mousedown', function(evt) {
                var m = document.getElementById('settings-menu');
                var b = document.getElementById('btn-settings');
                if (m && m.style.display !== 'none' && !m.contains(evt.target)
                        && !(b && b.contains(evt.target))) {
                    m.style.display = 'none';
                }
            });
            document.addEventListener('keydown', function(evt) {
                var m = document.getElementById('settings-menu');
                if (evt.key === 'Escape' && m) m.style.display = 'none';
            });
        }
        if (n) menu.style.display = (menu.style.display === 'none') ? 'block' : 'none';
        return window.dash_clientside.no_update;
    }
    """,
    Output("_settings-menu-hook", "children"),
    Input("btn-settings", "n_clicks"),
)

app.clientside_callback(
    """
    function(value) {
        document.body.classList.toggle('dark', !!(value && value.length));
        return window.dash_clientside.no_update;
    }
    """,
    Output("_dark-mode-hook", "children"),
    Input("dark-mode", "value"),
)


@app.callback(
    Output("dark-mode", "value"),
    Input("tabs", "value"),
)
def restore_dark_mode(_tab):
    """Show the last saved dark-mode choice when the page (re)loads."""
    return load_config().get("dark_mode", [])


@app.callback(
    Output("_dark-mode-hook", "title"),
    Input("dark-mode", "value"),
    prevent_initial_call=True,
)
def remember_dark_mode(value):
    """Save the choice so it is restored next time Theia runs."""
    save_config({"dark_mode": value or []})
    return dash.no_update



# ===========================================================================
# Callbacks — reference viewer (Results tab)
# ===========================================================================

@app.callback(
    Output("refview-seq", "data"),
    Output("refview-seq-input", "value"),
    Input("btn-ref-prev", "n_clicks"),
    Input("btn-ref-next", "n_clicks"),
    Input("refview-seq-input", "value"),
    Input("ref-graph", "clickData"),
    Input("db-select", "value"),
    State("refview-seq", "data"),
    State("refview-only-notfound", "value"),
)
def refview_step(_p, _n, typed, click, path, current, only_nf):
    """Work out which spectrum (seq number) the reference viewer shows."""
    if not path or not os.path.exists(path):
        return 0, 0
    sp = db_query(path, "SELECT seq, ref_status FROM spectra ORDER BY seq")
    if sp.empty:
        return 0, 0
    allowed = sp[sp["ref_status"] != "found"]["seq"].to_numpy() if only_nf else sp["seq"].to_numpy()
    if len(allowed) == 0:
        allowed = sp["seq"].to_numpy()
    trig = dash.callback_context.triggered[0]["prop_id"] if dash.callback_context.triggered else ""
    cur = int(current or 0)

    if trig.startswith("db-select"):
        new = int(allowed[0])
    elif trig.startswith("btn-ref-next"):
        later = allowed[allowed > cur]
        new = int(later[0]) if len(later) else int(allowed[-1])
    elif trig.startswith("btn-ref-prev"):
        earlier = allowed[allowed < cur]
        new = int(earlier[-1]) if len(earlier) else int(allowed[0])
    elif trig.startswith("ref-graph") and click and click.get("points"):
        new = int(round(click["points"][0]["x"]))
    elif trig.startswith("refview-seq-input") and typed is not None:
        new = int(typed)
    else:
        new = cur
    new = int(min(max(new, int(sp["seq"].min())), int(sp["seq"].max())))
    return new, new


@app.callback(
    Output("refview-graph", "figure"),
    Output("refview-info", "children"),
    Input("refview-seq", "data"),
    Input("refview-width", "value"),
    Input("db-select", "value"),
)
def refview_plot(seq, half_width, path):
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
            sims = simulate_frequencies(ion_db, ref_mq, float(found), s_db["ref_harmonic"],
                                        s_db["gamma_t"], s_db["harmonics"],
                                        s_db["correction_enabled"],
                                        s_db["corr_a"], s_db["corr_b"], s_db["corr_c"])
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



# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    import webbrowser
    # Hide the per-request access log lines; warnings and errors are still shown
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    url = "http://127.0.0.1:8052/"
    print(f"Starting Theia at {url}")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(debug=False, host="127.0.0.1", port=8052)
