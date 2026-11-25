"""
Theia's scan engine (no GUI): peak detection, reference tracking,
identification, scan settings, the scan databases and the scan worker.
Theia (theia/theia.py) uses it for its Scan and Scan Results tabs; it can also
be used on its own, e.g. from a notebook with the repo root on sys.path:

    from core import scan
    s = scan.parse_settings(scan.load_config())        # Theia's saved scan settings
    ff, pp = scan.load_spectrum(directory, name)
    r = scan.analyse_spectrum(ff, pp, s, s["ref_frequency"])
    r["peaks"], r["matches"]                           # DataFrames

    scan.list_databases()                              # [(path, run_info, settings)]
    scan.db_query(path, "SELECT * FROM hits WHERE ion = ?", ("208Pb80+",))

The scanner loops over every 1-D Schottky spectrum (.npz) in a directory,
detects peaks in the raw data, tracks the (drifting) reference peak from
spectrum to spectrum, simulates the NUBASE ion frequencies from that reference
and identifies every detected peak lying within ±x Hz of a simulated ion.  All
results go into a new SQLite database per run.

Output  (in the active experiment's data folder, see core/paths.py)
------
    data/<experiment>/outputs/scans/
      index.csv                               one row per run (human readable)
      scan_<YYYYmmdd_HHMMSS>_<hash>.sqlite    one database per run
      scan_<YYYYmmdd_HHMMSS>_<hash>.json      settings used for that run
    data/<experiment>/settings/scanner_config.json

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

import csv
import datetime
import glob
import hashlib
import json
import os
import re
import sqlite3
import sys
import threading
import time
import traceback

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths

from . import jsonfile, paths, physics
from .nubase_reader import unpack_npz
from .physics import HL_UNITS, build_ion_database, hl_allowed

# The active experiment's folders, read once when this module is imported
# (Theia is started per experiment by the home page)
SCANS_DIR   = paths.outputs("scans")
INDEX_PATH  = os.path.join(SCANS_DIR, "index.csv")
CONFIG_PATH = paths.settings_file("scanner_config.json")


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


def nearby_counts(peak_freq, sim_freq, all_sim_freq, tol_hz):
    """For each hit (peak matched to the simulated line at sim_freq): how many
    OTHER simulated lines (any ion, any harmonic, including ions excluded by
    the half-life cut) lie within ±tol_hz of the peak.  Used to step through
    the most isolated peaks first."""
    pf = np.asarray(peak_freq, dtype=float)
    sf = np.asarray(sim_freq, dtype=float)
    fs = np.sort(np.asarray(all_sim_freq, dtype=float))
    n = np.searchsorted(fs, pf + tol_hz, side="right") - np.searchsorted(fs, pf - tol_hz, side="left")
    own = (sf >= pf - tol_hz) & (sf <= pf + tol_hz)       # the matched line itself
    return np.maximum(n - own.astype(int), 0)


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
    sims = matches = ref_rows = None
    if ref_used is not None:
        # every harmonic from the reference ion's own peak in it, with its own correction
        sims, ref_info, _ = physics.simulate_spectrum(ion_db, ref_mq, ref_used, s, ff, pp)
        ref_rows = physics.reference_rows(ref_info, s["ref_harmonic"])
        lo, hi = ff.min() - s["match_tol_hz"], ff.max() + s["match_tol_hz"]
        sims = sims[(sims["f_sim"] >= lo) & (sims["f_sim"] <= hi)].reset_index(drop=True)
        # ions excluded by the half-life cut are simulated (and drawn) but
        # never used for identification
        sims_id = sims[allowed[sims["ion_row"].to_numpy()]].reset_index(drop=True)
        matches = match_peaks(peaks, sims_id, ion_db, s["match_tol_hz"])
        matches["n_nearby"] = (nearby_counts(matches["peak_freq"], matches["sim_freq"],
                                             sims["f_sim"], s["match_tol_hz"])
                               if len(matches) else [])
    return dict(peaks=peaks, base=base, sig=sig, ref=ref, ref_used=ref_used, status=status,
                sims=sims, matches=matches, ion_db=ion_db, excluded=~allowed, ref_rows=ref_rows)


# ===========================================================================
# Settings
# ===========================================================================

def _load_json(path):
    """A settings file as a dict; {} if missing.  A damaged file is set aside
    (its backup restored if there is one) with a warning: core/jsonfile.py."""
    return jsonfile.read(path, {})



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
    corr_profile       = "",
    ref_search_hz      = 0.0,
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


def save_config(cfg):
    """Merge `cfg` into scanner_config.json (keys not in cfg are kept, e.g. the
    snippet half-width is saved separately from the scan settings).

    Several callbacks (and the home page, another process) can save at the
    same moment, so the read-merge-write holds the file's lock; the file is
    replaced atomically and the previous version kept as .bak
    (core/jsonfile.py).  A failed save is reported, not ignored.
    """
    try:
        with jsonfile.lock(CONFIG_PATH):
            # physics values equal to the shared default are not stored, so
            # they follow physics.json; only real overrides are kept
            merged = physics.strip_shared({**_load_json(CONFIG_PATH), **cfg})
            jsonfile.write(CONFIG_PATH, merged)
    except Exception as exc:
        jsonfile.warn(f"could not save the scanner settings to {CONFIG_PATH}: "
                      f"{type(exc).__name__}: {exc}")


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
    # correction profile (name) -> general a, b, c and per-harmonic a, b, c.
    # Stored settings of a scan keep the per-harmonic values they were run with.
    s["corr_profile"] = str(raw.get("corr_profile") or "").strip()
    if isinstance(raw.get("corr_by_h"), dict):
        s["corr_by_h"] = {str(k): [float(x) for x in v] for k, v in raw["corr_by_h"].items()}
    else:
        physics.resolve_correction(s)
    # reference peak search in the other harmonics; missing (scans from before) = off
    s["ref_search_hz"] = _f(raw.get("ref_search_hz"), "Reference search in other harmonics", 0)
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
    closest_for_peak INTEGER,
    n_nearby         INTEGER       -- other simulated lines within ± match window of the peak
);
CREATE TABLE ref_harmonics (
    spectrum_id INTEGER,
    harmonic    INTEGER,
    status      TEXT,
    predicted   REAL,
    freq        REAL,
    offset_hz   REAL
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
    """Run a scan (background thread).  If it fails part-way, the run is
    marked "failed" (with the error) in the database, its .json file and
    index.csv, and the database is closed."""
    t0 = time.time()
    db_path = con = info = stored = stem = None
    n_ids_total = n_found = 0
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
                    n_files=len(files), scanner_version=1,
                    nearby_counted=True)
        _kv_insert(con, "run_info", info)
        con.commit()

        with open(os.path.join(SCANS_DIR, stem + ".json"), "w") as fh:
            json.dump({"run_info": info, "settings": stored}, fh, indent=2)

        predicted = s["ref_frequency"]
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
                    " closest_for_peak, n_nearby) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    [(sid, int(first_pid + pr), io, ib, int(z), int(nn), int(a), int(q), int(hh),
                      float(fs), float(fp), float(dd), float(pw), float(sn), kd, int(c1), int(c2),
                      int(nb))
                     for pr, io, ib, z, nn, a, q, hh, fs, fp, dd, pw, sn, kd, c1, c2, nb in zip(
                         m["peak_row"], m["ion"], m["ion_base"], m["Z"], m["N"], m["A"],
                         m["charge"], m["harmonic"], m["sim_freq"], m["peak_freq"],
                         m["delta_hz"], m["power"], m["snr"], m["kind"],
                         m["closest_for_ion"], m["closest_for_peak"], m["n_nearby"])])
            n_ids_total += n_ids
            # the reference ion's peak in every harmonic (reference harmonic: the scan's own search)
            if r["ref_rows"]:
                con.executemany(
                    "INSERT INTO ref_harmonics(spectrum_id, harmonic, status, predicted, freq,"
                    " offset_hz) VALUES (?,?,?,?,?,?)",
                    [(sid, rr["h"], r["status"], float(predicted), rr["f"],
                      rr["f"] - float(predicted)) if rr["status"] == "reference" else
                     (sid, rr["h"], rr["status"], rr["predicted"], rr["f"], rr["offset"])
                     for rr in r["ref_rows"]])

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
        con = None

        with open(os.path.join(SCANS_DIR, stem + ".json"), "w") as fh:
            json.dump({"run_info": info, "settings": stored}, fh, indent=2)
        _append_index(stem, info, stored)
        _set_state(running=False, msg=f"{status.capitalize()}: {SCAN_STATE['done']} spectra, "
                                      f"{n_ids_total} identifications, ref found in {n_found}. "
                                      f"Saved {os.path.basename(db_path)}")
    except Exception as exc:
        traceback.print_exc()
        msg = f"Scan failed: {exc}"
        if info is not None:
            # the run exists on disk: record the failure there too
            info.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                        n_spectra_done=SCAN_STATE["done"], n_ref_found=n_found,
                        n_identifications=n_ids_total, duration_s=round(time.time() - t0, 1))
            msg += f" (partial run kept as {os.path.basename(db_path)})"
            _record_failed_run(con, stem, info, stored)
        _set_state(running=False, error=str(exc), msg=msg)
    finally:
        if con is not None:
            try:
                con.close()
            except Exception:
                pass


def nearby_counted(path):
    """True once every hit of the scan database has its n_nearby count
    (new scans: from the start; older scans: after backfill_nearby)."""
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            row = con.execute("SELECT value FROM run_info WHERE key = 'nearby_counted'").fetchone()
        finally:
            con.close()
        return bool(row and json.loads(row[0]))
    except Exception:
        return False


def backfill_nearby(path, progress=None):
    """One-off for scans made before n_nearby existed: load each spectrum with
    hits once, simulate it as the scan did (same settings, the reference the
    scan used) and store each hit's count of other simulated lines within
    ± match window of the peak.  progress(done, total) is called as it goes.
    Spectra that cannot be loaded keep NULL (they sort last).
    Returns the number of such spectra."""
    con = sqlite3.connect(path, timeout=30)
    try:
        cols = [r[1] for r in con.execute("PRAGMA table_info(identifications)")]
        if "n_nearby" not in cols:
            con.execute("ALTER TABLE identifications ADD COLUMN n_nearby INTEGER")
            con.commit()
        s = parse_settings(read_kv(con, "settings"))
        ion_db, ref_mq = build_ion_database(*s["ion_key"])
        tol = float(s["match_tol_hz"])
        specs = con.execute("SELECT spectrum_id, name, ref_found FROM spectra "
                            "WHERE ref_found IS NOT NULL AND n_ids > 0 ORDER BY seq").fetchall()
        failed = 0
        for k, (sid, name, ref) in enumerate(specs):
            if progress:
                progress(k, len(specs))
            try:
                ff, pp = load_spectrum(s["spectra_dir"], name)
                sims, _, _ = physics.simulate_spectrum(ion_db, ref_mq, float(ref), s, ff, pp)
            except Exception:
                traceback.print_exc()
                failed += 1
                continue
            rows = con.execute("SELECT id, peak_freq, sim_freq FROM identifications "
                               "WHERE spectrum_id = ?", (sid,)).fetchall()
            if not rows:
                continue
            ids, pf, sf = zip(*rows)
            n = nearby_counts(pf, sf, sims["f_sim"], tol)
            con.executemany("UPDATE identifications SET n_nearby = ? WHERE id = ?",
                            [(int(c), int(i)) for c, i in zip(n, ids)])
            if k % 20 == 0:
                con.commit()
        _kv_insert(con, "run_info", {"nearby_counted": True, "nearby_failed": failed})
        con.commit()
        if progress:
            progress(len(specs), len(specs))
        return failed
    finally:
        con.close()


def _record_failed_run(con, stem, info, stored):
    """Write a failed run's run_info to its database, .json file and index.csv.
    Each step on its own, so one failing (e.g. the database itself) does not
    stop the others."""
    steps = []
    if con is not None:
        def _db():
            # drop what was not committed yet (it may hold a half-written
            # spectrum), then count what the database really holds
            con.rollback()
            q = lambda sql: con.execute(sql).fetchone()[0]
            info.update(n_spectra_done=q("SELECT COUNT(*) FROM spectra"),
                        n_ref_found=q("SELECT COUNT(*) FROM spectra WHERE ref_status='found'"),
                        n_identifications=q("SELECT COUNT(*) FROM identifications"))
            _kv_insert(con, "run_info", info)
            con.commit()
        steps.append(("database", _db))

    def _json():
        with open(os.path.join(SCANS_DIR, stem + ".json"), "w") as fh:
            json.dump({"run_info": info, "settings": stored}, fh, indent=2)
    steps.append(("json", _json))
    steps.append(("index.csv", lambda: _append_index(stem, info, stored)))
    for what, step in steps:
        try:
            step()
        except Exception as exc:
            print(f"Theia: could not record the failed scan in its {what}: {exc}", file=sys.stderr)


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


def read_run_info(con, path):
    """run_info of a scan database.  A run still marked "running" that is not
    the scan in progress was cut off (Theia closed or killed mid-scan): it is
    reported as "interrupted" (the database itself is not changed)."""
    info = read_kv(con, "run_info")
    if info.get("status") == "running":
        with SCAN_LOCK:
            live = (SCAN_STATE["running"] and SCAN_STATE["db_path"] is not None
                    and os.path.abspath(SCAN_STATE["db_path"]) == os.path.abspath(path))
        if not live:
            info["status"] = "interrupted"
    return info


def list_databases():
    """[(path, run_info, settings)] newest first."""
    out = []
    for path in sorted(glob.glob(os.path.join(SCANS_DIR, "*.sqlite")), reverse=True):
        try:
            con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            info, st = read_run_info(con, path), read_kv(con, "settings")
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
