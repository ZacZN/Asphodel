"""
scanner.py  –  Theia batch scanner

Loops over every 1-D Schottky spectrum (.npz, same format as viewer.py) in a
directory, detects peaks in the raw data, tracks the (drifting) reference peak
from spectrum to spectrum, simulates the NUBASE ion frequencies from that
reference and identifies every detected peak lying within ±x Hz of a simulated
ion.  All results go into a new SQLite database per run, which can be loaded
and searched in the "Results" tab.

Usage
-----
    python scanner.py
then open  http://127.0.0.1:8052/

Output
------
    scans/
      index.csv                               one row per run (human readable)
      scan_<YYYYmmdd_HHMMSS>_<hash>.sqlite    one database per run
      scan_<YYYYmmdd_HHMMSS>_<hash>.json      settings used for that run

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

from nubase_reader import df_nubase, rect_restrict, diag_restrict, index_by_name, unpack_npz

import os
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
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc


HERE        = os.path.dirname(os.path.abspath(__file__))
SCANS_DIR   = os.path.join(HERE, "scans")
INDEX_PATH  = os.path.join(SCANS_DIR, "index.csv")
CONFIG_PATH = os.path.join(HERE, "scanner_config.json")
ACCENT      = "#f57d15"
ACCENT_PALE = "#fff4ec"


# ===========================================================================
# Physics (identical to viewer.py)
# ===========================================================================

def delta_f(ref_frequency, ref_mq, delta_mq, gamma_t):
    return (-1 / (gamma_t**2)) * (delta_mq / ref_mq) * ref_frequency


def apply_correction(freq, a, b, c):
    """Return freq + a*freq**2 + b*freq + c (second-order polynomial correction)."""
    return freq + a * freq**2 + b * freq + c


def ion_base_name(ion_name: str) -> str:
    """'204Hg79+' -> '204Hg',  'm186Hf72+' -> 'm186Hf'"""
    return re.sub(r"\d+\+$", "", ion_name)


@lru_cache(maxsize=8)
def build_ion_database(z_lims, n_lims, el1, el2, ref_name, mq_width):
    """Bare, H-, He- and Li-like ions inside the selection, restricted to the
    m/q window around the reference ion (as in viewer.py).  Cached."""
    frames = []
    for electrons in (0, 1, 2, 3):
        d = df_nubase(electrons=electrons)
        d = rect_restrict(df=d, z_lims=list(z_lims), n_lims=list(n_lims))
        d = diag_restrict(df=d, el1=list(el1), el2=list(el2))
        d = index_by_name(df=d, electrons=electrons)
        d["electrons"] = electrons
        d["charge"] = d["ZZZi"] - electrons
        frames.append(d)
    db = pd.concat(frames).sort_values("m/q", ascending=False)

    if ref_name not in db.index:
        raise ValueError(f"Reference ion '{ref_name}' is not in the ion selection "
                         f"(check Z/N limits, el1/el2).")
    mq_ref = db.loc[ref_name, "m/q"]
    if isinstance(mq_ref, pd.Series):
        mq_ref = mq_ref.iloc[0]
    db = db[(db["m/q"] >= mq_ref - mq_width) & (db["m/q"] <= mq_ref + mq_width)].copy()
    db["ion"] = db.index.astype(str)
    db["ion_base"] = db["ion"].map(ion_base_name)
    return db.reset_index(drop=True), float(mq_ref)


def simulate_frequencies(ion_db, ref_mq, ref_obs_freq, ref_harmonic, gamma_t, harmonics,
                         correction_enabled=False, corr_a=0.0, corr_b=0.0, corr_c=0.0):
    """Vectorised version of viewer.compute_frequencies_multi.

    Returns a DataFrame with one row per (ion, harmonic): ion_row, harmonic, f_sim.
    """
    ref_rev = ref_obs_freq / ref_harmonic
    mq = ion_db["m/q"].to_numpy(dtype=float)
    f_true = ref_rev + delta_f(ref_rev, ref_mq, mq - ref_mq, gamma_t)
    if correction_enabled:
        f_true = apply_correction(f_true, corr_a, corr_b, corr_c)
    rows = np.arange(len(ion_db))
    out = [pd.DataFrame({"ion_row": rows, "harmonic": int(h), "f_sim": f_true * h})
           for h in harmonics]
    return pd.concat(out, ignore_index=True)


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

def find_reference(peaks, predicted, window_hz, mode="strongest"):
    """Pick the reference peak among detected maxima within ±window of the
    predicted position.  Returns the row (Series) or None."""
    if peaks is None or len(peaks) == 0:
        return None
    cand = peaks[(peaks["kind"] == "max") & ((peaks["freq"] - predicted).abs() <= window_hz)]
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
    data = unpack_npz(os.path.join(directory, name))
    return np.asarray(data["arr_0"], dtype=float), np.asarray(data["arr_1"], dtype=float)


def analyse_spectrum(ff, pp, s, predicted_ref):
    """Full per-spectrum pipeline.  Returns dict with peaks, ref info, sims, matches."""
    peaks, base, sig = detect_peaks(ff, pp, s)
    ref = find_reference(peaks, predicted_ref, s["ref_window_hz"], s["ref_mode"])
    if ref is not None:
        ref_used, status = float(ref["freq"]), "found"
    elif s["use_predicted"]:
        ref_used, status = float(predicted_ref), "predicted"
    else:
        ref_used, status = None, "not_found"

    ion_db, ref_mq = build_ion_database(*s["ion_key"])
    sims = matches = None
    if ref_used is not None:
        sims = simulate_frequencies(ion_db, ref_mq, ref_used, s["ref_harmonic"], s["gamma_t"],
                                    s["harmonics"], s["correction_enabled"],
                                    s["corr_a"], s["corr_b"], s["corr_c"])
        lo, hi = ff.min() - s["match_tol_hz"], ff.max() + s["match_tol_hz"]
        sims = sims[(sims["f_sim"] >= lo) & (sims["f_sim"] <= hi)].reset_index(drop=True)
        matches = match_peaks(peaks, sims, ion_db, s["match_tol_hz"])
    return dict(peaks=peaks, base=base, sig=sig, ref=ref, ref_used=ref_used, status=status,
                sims=sims, matches=matches, ion_db=ion_db)


# ===========================================================================
# Settings
# ===========================================================================

def _load_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return {}


_viewer_cfg = _load_json(os.path.join(HERE, "viewer_config.json"))
_pf_cfg = _load_json(os.path.join(HERE, "peak_finder_config.json"))

DEFAULTS = dict(
    run_label          = "",
    spectra_dir        = _pf_cfg.get("file_path",
                         "/home/zn00092/Desktop/1dSpectra/SplitTDMS_Output_2^17_nobaseline/"),
    file_pattern       = "*.npz",
    ref_name           = _viewer_cfg.get("ref_name", "204Hg79+"),
    ref_frequency      = _viewer_cfg.get("ref_frequency", 411910517),
    ref_harmonic       = _viewer_cfg.get("ref_harmonic", 213),
    ref_window_hz      = 500,
    ref_mode           = "strongest",
    ref_tracking       = ["on"],
    use_predicted      = [],
    z_lims             = _viewer_cfg.get("z_lims", "60, 83"),
    n_lims             = _viewer_cfg.get("n_lims", "100, 127"),
    el1                = _viewer_cfg.get("el1", "156, 66"),
    el2                = _viewer_cfg.get("el2", "200, 82"),
    mq_width           = _viewer_cfg.get("mq_width", 0.01),
    gamma_t            = _viewer_cfg.get("gamma_t", 1.399),
    harmonics          = _viewer_cfg.get("harmonics", "212, 213, 214"),
    correction_enabled = _viewer_cfg.get("correction_enabled", []),
    corr_a             = _viewer_cfg.get("corr_a", 0),
    corr_b             = _viewer_cfg.get("corr_b", 0),
    corr_c             = _viewer_cfg.get("corr_c", 0),
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
)


def load_config():
    return {**DEFAULTS, **_load_json(CONFIG_PATH)}


def save_config(cfg):
    try:
        with open(CONFIG_PATH, "w") as fh:
            json.dump(cfg, fh, indent=2)
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

def _vlines(xs, y0, y1, hover):
    X, Y, H = [], [], []
    for x, h in zip(xs, hover):
        X += [x, x, None]
        Y += [y0, y1, None]
        H += [h, h, None]
    return X, Y, H


def empty_fig(msg=""):
    fig = go.Figure()
    fig.update_layout(template="plotly_white", margin=dict(l=60, r=20, t=40, b=40))
    if msg:
        fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                           font=dict(size=15, color="#888"))
    return fig


def spectrum_figure(ff, pp, peaks, base, sig, s, r, title, xrange=None, highlight_ion=None):
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=ff, y=pp, mode="lines", name="Spectrum", hoverinfo="skip",
                               line=dict(shape="hv", color="black", width=1)))
    if base is not None and sig is not None:
        fig.add_trace(go.Scattergl(x=ff, y=base + s["k_height"] * sig, mode="lines",
                                   name=f"baseline + {s['k_height']:g}σ", hoverinfo="skip",
                                   line=dict(color="#999", width=1, dash="dot")))
    ymin, ymax = float(np.nanmin(pp)), float(np.nanmax(pp))
    pad = 0.05 * (ymax - ymin or 1)

    # simulated ions
    sims, ion_db, matches = r.get("sims"), r.get("ion_db"), r.get("matches")
    if sims is not None and len(sims):
        ions = ion_db["ion"].to_numpy()
        names = ions[sims["ion_row"].to_numpy()]
        labels = [f"{n}  (h={h})<br>f_sim = {f:.1f} Hz" for n, h, f in
                  zip(names, sims["harmonic"], sims["f_sim"])]
        matched = set() if matches is None else set(zip(matches["ion"], matches["harmonic"]))
        is_m = np.array([(n, h) in matched for n, h in zip(names, sims["harmonic"])])
        is_hi = np.array([highlight_ion is not None and n == highlight_ion for n in names])
        for mask, name, style in [
            (~is_m & ~is_hi, "Simulated (no peak)", dict(color="#bbbbbb", width=1, dash="dash")),
            (is_m & ~is_hi, "Simulated (identified)", dict(color="#555555", width=1.5, dash="dash")),
            (is_hi, highlight_ion or "", dict(color=ACCENT, width=3, dash="solid")),
        ]:
            if mask.any():
                X, Y, H = _vlines(sims["f_sim"].to_numpy()[mask], ymin - pad, ymax + pad,
                                  np.array(labels, dtype=object)[mask])
                fig.add_trace(go.Scattergl(x=X, y=Y, mode="lines", name=name, line=style,
                                           hoverinfo="text", hovertext=H, connectgaps=False))

    # reference window + found reference
    pred = r.get("predicted")
    if pred is not None:
        w = s["ref_window_hz"]
        fig.add_vrect(x0=pred - w, x1=pred + w, fillcolor=ACCENT_PALE, opacity=0.7,
                      line_width=0, layer="below")
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
                      hovermode="closest", dragmode="pan", template="plotly_white",
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


def section(title):
    return [html.Hr(style={"margin": "10px 0"}), html.P(title, style=SECTION_STYLE)]


sidebar = html.Div([
    html.H5("Scan settings", style={"fontWeight": "700", "marginBottom": "4px"}),

    *section("Run"),
    inp("Run label (optional)", "run_label", help="Free text stored with the database"),
    inp("Spectra directory", "spectra_dir"),
    inp("File pattern", "file_pattern", help="Files are processed in natural (numeric) order"),

    *section("Reference ion"),
    inp("Ion name", "ref_name"),
    inp("Frequency (Hz) — in first spectrum", "ref_frequency", True, step=1),
    inp("Harmonic", "ref_harmonic", True, step=1),
    inp("Search window ± (Hz)", "ref_window_hz", True),
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
    html.Div(style={"height": "30px"}),
], style={"width": "300px", "minWidth": "300px", "padding": "14px 18px", "background": "#f9f9f9",
          "overflowY": "auto", "borderRight": "1px solid #dddddd", "height": "calc(100vh - 42px)"})


def _table(id_, columns, page_size=20, height=None, **kw):
    return dash_table.DataTable(
        id=id_, columns=[{"name": c, "id": c} for c in columns], data=[],
        page_size=page_size, sort_action="native", filter_action="native",
        style_table={"overflowX": "auto", **({"maxHeight": height, "overflowY": "auto"} if height else {})},
        style_header=TABLE_HEADER, style_cell=TABLE_CELL, style_data_conditional=TABLE_ODD,
        style_filter={"backgroundColor": "#f9f9f9"}, **kw)


BTN = {"marginRight": "8px"}

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
    ], style={"height": "8px", "background": "#eeeeee", "borderRadius": "4px", "overflow": "hidden",
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

    html.Div([
        # left: ion summary
        html.Div([
            html.H6("Ions found (click a row to filter)", style={"fontWeight": "600"}),
            _table("summary-table", ["ion", "n_spectra", "n_hits", "mean_delta_hz", "std_delta_hz"],
                   page_size=25),
            html.H6("Reference tracking", style={"fontWeight": "600", "marginTop": "10px"}),
            dcc.Graph(id="ref-graph", figure=empty_fig(), style={"height": "280px"}),
        ], style={"width": "34%", "marginRight": "16px"}),

        # right: filtered hits
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
            _table("hits-table", ["spectrum", "seq", "ion", "harmonic", "peak_freq", "sim_freq",
                                  "delta_hz", "power", "snr", "kind", "closest_for_ion",
                                  "closest_for_peak"],
                   page_size=15, row_selectable="single", export_format="csv"),
            html.Div([
                dcc.Graph(id="hits-graph", figure=empty_fig(), style={"height": "300px", "flex": "1"}),
            ], style={"display": "flex"}),
            html.Div([
                html.Label("Snippet half-width (Hz)", style={**LABEL_STYLE, "marginRight": "8px"}),
                dcc.Input(id="snippet-width", type="number", value=1000, debounce=True,
                          className="form-control form-control-sm", style={"width": "140px"}),
                html.Small("  Select a row in the table to view that peak in its spectrum",
                           style={**HELP_STYLE, "marginLeft": "10px", "marginTop": "0"}),
            ], style={"display": "flex", "alignItems": "center", "marginTop": "6px"}),
            dcc.Graph(id="snippet-graph", figure=empty_fig(), style={"height": "420px"},
                      config={"scrollZoom": True}),
        ], style={"flex": "1", "minWidth": "0"}),
    ], style={"display": "flex"}),
], style={"padding": "14px 18px", "overflowY": "auto", "height": "calc(100vh - 42px)"})


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "Theia Scanner"
TAB_STYLE = {"padding": "8px 16px", "fontWeight": "600"}
TAB_SEL = {**TAB_STYLE, "borderTop": f"3px solid {ACCENT}"}

app.layout = html.Div([
    dcc.Tabs(id="tabs", value="scan", children=[
        dcc.Tab(label="Scan", value="scan", style=TAB_STYLE, selected_style=TAB_SEL, children=[
            html.Div([sidebar, scan_main], style={"display": "flex"}),
        ]),
        dcc.Tab(label="Results", value="results", style=TAB_STYLE, selected_style=TAB_SEL,
                children=[results_tab]),
    ], style={"height": "42px"}),
    # Polls the background scan; only enabled while a scan is running
    dcc.Interval(id="poll", interval=700, disabled=True),
    dcc.Store(id="scan-finished"),
])

SETTING_KEYS = list(SETTING_FIELDS)
SETTING_STATES = [State(cid, prop) for cid, prop in SETTING_FIELDS.values()]


def _raw_from_states(values):
    return dict(zip(SETTING_KEYS, values))


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
            f"median σ = {np.median(r['sig']):.3g}  |  noise cutoff = {cut_txt}  |  {dt:.2f} s")
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
    Output("hits-table", "selected_rows"),
    Output("hits-info", "children"),
    Output("hits-graph", "figure"),
    Input("db-select", "value"),
    Input("ion-filter", "value"),
    Input("spec-filter", "value"),
    Input("closest-filter", "value"),
)
def query_hits(path, ion_filter, spec_filter, closest):
    if not path or not os.path.exists(path):
        return [], [], "", empty_fig()
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

    fig = go.Figure()
    if len(df):
        groups = df.groupby(["ion", "harmonic"]) if ors else [(("all", ""), df)]
        for (ion, h), d in list(groups)[:40]:
            fig.add_trace(go.Scattergl(x=d["seq"], y=d["delta_hz"], mode="markers",
                                       name=f"{ion} h={h}" if ors else "all hits",
                                       marker=dict(size=6 if ors else 3),
                                       hovertext=d["spectrum"] + "<br>" + d["ion"], hoverinfo="text+y"))
    fig.update_layout(template="plotly_white", margin=dict(l=60, r=10, t=10, b=40),
                      xaxis_title="spectrum #", yaxis_title="f_peak − f_sim (Hz)", font=dict(size=11),
                      colorway=[ACCENT, "#333333", "#999999", "#c45a00", "#666666", "#ffb877"])
    return _fmt_records(df), [], info, fig


@app.callback(
    Output("snippet-graph", "figure"),
    Input("hits-table", "selected_rows"),
    Input("snippet-width", "value"),
    State("hits-table", "data"),
    State("db-select", "value"),
)
def show_snippet(selected, half_width, rows, path):
    if not selected or not rows or not path or selected[0] >= len(rows):
        return empty_fig("Select a row in the table above")
    row = rows[selected[0]]
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
        sims = sims[(sims["f_sim"] >= x0) & (sims["f_sim"] <= x1)].reset_index(drop=True)
        r = dict(sims=sims, ion_db=ion_db, matches=hits, ref_used=float(row["ref_found"]),
                 status="found", predicted=None)
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
        return fig
    except Exception as exc:
        traceback.print_exc()
        return empty_fig(f"Could not load spectrum: {exc}")


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    import webbrowser
    # Hide the per-request access log lines; warnings and errors are still shown
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    url = "http://127.0.0.1:8052/"
    print(f"Starting Theia scanner at {url}")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(debug=False, host="127.0.0.1", port=8052)
