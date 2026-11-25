#!/usr/bin/env python3
"""
Plot a spectrogram from a single ESR injection .npz file.

Opens an interactive browser page with a 2-D heatmap and 3-D surface
side by side.  Scale, time-averaging, and axis ranges can all be
adjusted in the browser without re-running the script.

Usage
-----
  python plot_spectrogram.py                                   # reopen the last session
  python plot_spectrogram.py --new                             # choose the file in the GUI
  python plot_spectrogram.py <file> [<freq_min> <freq_max>] [--linear] [--time-avg N]

Session memory
--------------
  The last loaded file, its frequency crop and the view settings (frequency /
  time sliders, title, dB, time-avg, 1-D overlay, 3-D camera angle) are saved
  to data/<experiment>/settings/spectrogram_prefs.json as you change them. Running with no arguments
  reopens exactly that view; with nothing saved (or the file gone) it opens
  the file chooser. --linear / --time-avg on a bare run override the saved
  values.

Arguments (all optional — the file and frequency range can be chosen in the GUI)
---------
  file          Path to a .npz file (freq, time, power arrays)
  freq_min      Lower frequency bound (same units as the freq array); default: full range
  freq_max      Upper frequency bound
  --linear      Start in linear-power mode instead of dB (can be toggled in the browser)
  --time-avg N  Initial time-frame averaging (can be changed in the browser)

Example
-------
  python plot_spectrogram.py 1081.5.npz 245000 246000
  python plot_spectrogram.py 1081.5.npz 245000 246000 --linear
  python plot_spectrogram.py 1081.5.npz 245000 246000 --time-avg 5
"""

import sys
import os
import argparse

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Plot ESR injection spectrogram")
    parser.add_argument("file",     nargs="?", default=None,
                        help="Path to .npz file (optional; choose in the GUI otherwise)")
    parser.add_argument("freq_min", nargs="?", type=float, default=None,
                        help="Lower frequency bound (default: full range)")
    parser.add_argument("freq_max", nargs="?", type=float, default=None,
                        help="Upper frequency bound (default: full range)")
    # default=None (not False) so a bare run can tell "not given" from "given"
    parser.add_argument("--linear", action="store_true", default=None,
                        help="Start in linear-power mode instead of dB (toggleable in browser)")
    parser.add_argument("--time-avg", type=int, metavar="N", default=None,
                        help="Average every N consecutive time frames (reduces time axis by factor N)")
    parser.add_argument("--new", action="store_true",
                        help="Ignore the saved session and open the file chooser")
    return parser.parse_args()


def load(path):
    """Load and return freq_1d (F,), time_1d (T,), power_2d (T, F)."""
    with np.load(path) as data:    # closed even if a key is missing
        freq  = data['freq'].ravel()   # (1, F) → (F,)
        time  = data['time'].ravel()   # (T, 1) → (T,)
        power = data['power']          # (T, F)

    # Normalise time to start at 0
    time = time - time[0]

    return freq, time, power


def slice_freq(freq, power, freq_min, freq_max):
    """Return (freq_slice, power_slice) restricted to [freq_min, freq_max]."""
    mask = (freq >= freq_min) & (freq <= freq_max)
    if not mask.any():
        raise ValueError(
            f"No frequency bins found in [{freq_min}, {freq_max}]. "
            f"File covers [{freq.min():.1f}, {freq.max():.1f}]."
        )
    return freq[mask], power[:, mask]


# ---------------------------------------------------------------------------
# File loader (shared by the landing page and the spectrogram page)
# ---------------------------------------------------------------------------

_HERE       = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.dirname(_HERE))
from core import baseline, browser, icon, jsonfile, logcapture, logpanel, offline, paths, tabwatch, trackpad

# output also to data/logs/ and the page's Log panel (core/logcapture.py)
if __name__ == "__main__":
    logcapture.start("spectrogram")
PREFS_PATH  = paths.settings_file("spectrogram_prefs.json")
DECAY_PATH  = os.path.join(_HERE, "decay_fit.py")


import threading
_PREFS_LOCK = threading.RLock()   # serialise read-modify-write of the prefs file


def read_prefs():
    """The prefs file as a dict; {} if missing.  A damaged file is set aside
    (its backup restored if there is one) with a warning: core/jsonfile.py."""
    return jsonfile.read(PREFS_PATH, {})


def write_prefs(prefs):
    # atomic, previous version kept as .bak (core/jsonfile.py); raises on failure
    jsonfile.write(PREFS_PATH, prefs, indent=None)


def _same_crop(a, b):
    """True if two requested freq bounds (float or None) are the same."""
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) < 1e-9


def saved_view(file_path, fmin_req, fmax_req):
    """Saved GUI view for this exact file + crop, else {}."""
    sess = read_prefs().get("session") or {}
    if (isinstance(sess, dict) and file_path
            and sess.get("file") == os.path.abspath(file_path)
            and _same_crop(sess.get("fMin"), fmin_req)
            and _same_crop(sess.get("fMax"), fmax_req)
            and isinstance(sess.get("view"), dict)):
        return sess["view"]
    return {}


# ---------------------------------------------------------------------------
# Simulated ion lines (physics in core/physics.py, shared with theia/theia.py)
# ---------------------------------------------------------------------------
# The reference is the strongest bin of the summed power (all time frames, the
# whole uncropped file). It is taken to be `ref_name` at
# `ref_harmonic`; every other ion in the selection is placed relative to it.

# Panel defaults. The physics keys are placeholders: their values come from the
# shared physics settings (data/<exp>/settings/physics.json); the "sim" entry
# in spectrogram_prefs.json keeps only this tool's overrides of them.
SIM_DEFAULTS = {
    "show":               True,
    "ref_name":           "",
    "ref_harmonic":       0,
    "z_lims":             "",
    "n_lims":             "",
    "el1":                "",
    "el2":                "",
    "mq_width":           0.0,
    "gamma_t":            0.0,
    "harmonics":          "",
    "correction_enabled": False,
    "corr_a":             0.0,
    "corr_b":             0.0,
    "corr_c":             0.0,
    "corr_profile":       "",
    "ref_search_hz":      0.0,
    "min_half_life":      0.0,
    "min_half_life_unit": "ms",
    "exclude_unknown_hl": False,
    "show_excluded":      True,
}

from core import corrections, physics
from core.physics import HL_UNITS, HL_CUT_UNITS, hl_allowed, build_ion_database, simulate_frequencies


def _ints(v, n, what):
    import re
    try:
        vals = [int(float(x)) for x in re.split(r"[,\s]+", str(v).strip()) if x]
    except ValueError:
        raise ValueError(f"{what}: expected integers, got {v!r}")
    if n is not None and len(vals) != n:
        raise ValueError(f"{what}: expected {n} integers, got {v!r}")
    return vals


def _num(v, what, default=None):
    if v is None or str(v).strip() == "":
        if default is None:
            raise ValueError(f"{what} is empty")
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{what}: not a number ({v!r})")


def parse_sim_settings(raw):
    """Raw panel values -> typed settings. Raises ValueError with a readable message."""
    s = {}
    s["ref_name"] = str(raw.get("ref_name") or "").strip()
    if not s["ref_name"]:
        raise ValueError("reference ion name is empty")
    s["ref_harmonic"] = int(_num(raw.get("ref_harmonic"), "reference harmonic"))
    if s["ref_harmonic"] <= 0:
        raise ValueError("reference harmonic must be > 0")
    s["z_lims"] = _ints(raw.get("z_lims"), 2, "Z limits")
    s["n_lims"] = _ints(raw.get("n_lims"), 2, "N limits")
    s["el1"] = _ints(raw.get("el1"), 2, "el1")
    s["el2"] = _ints(raw.get("el2"), 2, "el2")
    s["mq_width"] = _num(raw.get("mq_width"), "m/q window")
    s["gamma_t"] = _num(raw.get("gamma_t"), "γt")
    s["harmonics"] = _ints(raw.get("harmonics"), None, "harmonics")
    if not s["harmonics"]:
        raise ValueError("harmonics list is empty")
    s["correction_enabled"] = bool(raw.get("correction_enabled"))
    s["corr_a"] = _num(raw.get("corr_a"), "a", 0.0)
    s["corr_b"] = _num(raw.get("corr_b"), "b", 0.0)
    s["corr_c"] = _num(raw.get("corr_c"), "c", 0.0)
    # correction profile name -> general and per-harmonic a, b, c (core/corrections.py)
    s["corr_profile"] = str(raw.get("corr_profile") or "").strip()
    physics.resolve_correction(s)
    s["ref_search_hz"] = _num(raw.get("ref_search_hz"), "reference search window", 0.0)
    s["min_half_life"] = _num(raw.get("min_half_life"), "min half-life", 0.0)
    s["min_half_life_unit"] = raw.get("min_half_life_unit") or "ms"
    if s["min_half_life_unit"] not in HL_UNITS:
        raise ValueError(f"half-life unit '{s['min_half_life_unit']}' not recognised")
    s["exclude_unknown_hl"] = bool(raw.get("exclude_unknown_hl"))
    return s


def sim_prefs():
    """Panel values: SIM_DEFAULTS, then the shared physics, then the saved overrides."""
    saved = read_prefs().get("sim")
    saved = {k: v for k, v in saved.items() if k in SIM_DEFAULTS} if isinstance(saved, dict) else {}
    return physics.with_shared(SIM_DEFAULTS, saved)


def compute_sim_lines(freq, sum_all, raw):
    """Reference from the strongest bin of `sum_all` (power summed over all
    frames, aligned with `freq` - the whole uncropped file), then all
    simulated lines inside that frequency range. Returns a JSON-able dict."""
    s = parse_sim_settings(raw)
    i_ref = int(np.nanargmax(sum_all))
    ref_freq = float(freq[i_ref])
    ion_db, ref_mq = build_ion_database(tuple(s["z_lims"]), tuple(s["n_lims"]),
                                        tuple(s["el1"]), tuple(s["el2"]),
                                        s["ref_name"], float(s["mq_width"]))
    allowed = hl_allowed(ion_db, s)
    # every harmonic from the reference ion's own peak in it (in the summed power),
    # with that harmonic's own correction
    order = np.argsort(freq)
    sims, ref_info, warns = physics.simulate_spectrum(
        ion_db, ref_mq, ref_freq, s, np.asarray(freq, float)[order], np.asarray(sum_all, float)[order])
    lo, hi = float(np.min(freq)), float(np.max(freq))
    sims = sims[(sims["f_sim"] >= lo) & (sims["f_sim"] <= hi)].reset_index(drop=True)
    rows = sims["ion_row"].to_numpy()
    names = ion_db["ion"].to_numpy()[rows]
    hover = ion_db["hover"].to_numpy()[rows]
    lines = []
    def _hl_txt(t, u):
        t = "" if t is None or (isinstance(t, float) and np.isnan(t)) else str(t).strip()
        u = "" if u is None or (isinstance(u, float) and np.isnan(u)) else str(u).strip()
        return (t + (" " + u if u else "")).strip()
    hl_txt = [_hl_txt(t, u) for t, u in zip(ion_db["T"].to_numpy()[rows], ion_db["T_unit"].to_numpy()[rows])]
    for name, h, f, hv, ok, ht in zip(names, sims["harmonic"], sims["f_sim"], hover, allowed[rows], hl_txt):
        lines.append({
            "t_half": ht,
            "f": float(f),
            "ion": str(name),
            "h": int(h),
            "excluded": not bool(ok),
            "is_ref": str(name) == s["ref_name"] and int(h) == s["ref_harmonic"],
            "label": f"{hv}<br>Harmonic: {int(h)}<br>f_sim = {float(f):,.1f} Hz",
        })
    return {"ref_freq": ref_freq, "ref_name": s["ref_name"], "ref_harmonic": s["ref_harmonic"],
            "n_ions": int(len(ion_db)), "lines": lines, "warnings": warns,
            "ref_by_h": {str(h): d["f"] for h, d in ref_info.items()},
            "ref_rows": physics.reference_rows(ref_info, s["ref_harmonic"]),
            "ref_search_hz": float(s.get("ref_search_hz") or 0)}


import re as _re
_ION_RE  = _re.compile(r"(?<![\w>])([mnpqrx]?)(\d{1,3})([A-Z][a-z]?)(\d{1,3})\+")
_HARM_RE = _re.compile(r"(?<!\S)h(\d+)(?!\S)")
# "m207Tl80+ h213" -> "m207Tl80+, h213" (adds the comma if it is missing)
_ION_HARM_RE = _re.compile(r"((?<![\w>])[mnpqrx]?\d{1,3}[A-Z][a-z]?\d{1,3}\+)(\s+)(h\d+)(?!\S)")


def format_title_html(text):
    """Plain title -> plotly HTML: m207Tl80+ -> <sup>207m</sup>Tl<sup>80+</sup>,
    h213 -> <i>h</i> = 213 (same rules as formatTitleHtml() in the page)."""
    t = str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = _ION_HARM_RE.sub(r"\1,\2\3", t)
    t = _ION_RE.sub(lambda m: f"<sup>{m.group(2)}{m.group(1)}</sup>{m.group(3)}<sup>{m.group(4)}+</sup>", t)
    return _HARM_RE.sub(lambda m: f"<i>h</i> = {m.group(1)}", t)


def make_title(path, fmin, fmax, linear=False, ion=None):
    return (f"{os.path.basename(path)}   "
            f"({'linear' if linear else 'dB'})"
            + (f"   ·   {ion}" if ion else ""))


# Plain strings (not f-strings) so CSS/JS braces need no escaping.
# Loading screen of the spectrogram page: covers the page from its first bytes
# until the plots are drawn and the data the page fetches itself (1-D
# spectrum, whole-file spectrum, simulated lines) has arrived, so the page
# appears only when ready.  Same look as the home page's starting page
# (home.py _STARTING_HTML); ?asph_t0=<ms> (set by that page) continues its timer.
BOOT_OVERLAY = """
<div id="asphBoot" style="position:fixed;inset:0;z-index:2147483647;background:#fff;color:#222;
  display:flex;align-items:center;justify-content:center;transition:opacity .15s;
  font-family:-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif">
 <style>
  #asphBoot .spin{width:34px;height:34px;margin:0 auto 18px;border:4px solid #fff4ec;
    border-top-color:#f57d15;border-radius:50%;animation:asphSpin .8s linear infinite}
  @keyframes asphSpin{to{transform:rotate(360deg)}}
 </style>
 <div style="text-align:center">
  <div class="spin"></div>
  <div id="asphBootH" style="font-size:20px;font-weight:600;margin:0 0 14px">Loading Spectrogram&hellip;</div>
  <div id="asphBootT" style="color:#777;font-size:13px">0.0 s</div>
 </div>
</div>
<script>
(function() {
  var m = /[?&]asph_t0=(\\d+)/.exec(location.search);
  var t0 = m ? +m[1] : (performance.timeOrigin || Date.now());
  if (m) document.getElementById('asphBootH').innerHTML = 'Starting Spectrogram&hellip;';
  if (m && history.replaceState) {   // a reload starts its own timer
    history.replaceState(null, '', location.pathname + location.search.replace(/([?&])asph_t0=\\d+&?/, '$1').replace(/[?&]$/, '') + location.hash);
  }
  var tick = setInterval(function() {
    var el = document.getElementById('asphBootT');
    if (el) el.textContent = ((Date.now() - t0) / 1000).toFixed(1) + ' s';
  }, 100);
  // count the page's own requests (not the periodic polls) to know when its data is in
  var busy = 0, lastDone = Date.now(), loaded = false, done = false;
  var poll = /\\/_alive|\\/_aslog|\\/_touchpad/;
  var orig = window.fetch;
  window.fetch = function(u) {
    var mine = !poll.test(String(u && u.url || u));
    if (mine) busy++;
    var fin = function() { if (mine) { busy--; lastDone = Date.now(); } };
    var pr = orig.apply(this, arguments);
    pr.then(fin, fin);
    return pr;
  };
  function plotsDrawn() {
    var ids = ['heatPlot', 'specPlot', 'fullPlot'], ok = true;
    ids.forEach(function(id) {
      var d = document.getElementById(id);
      if (d && d.offsetParent !== null && d.offsetHeight > 0 && !d._fullLayout) ok = false;
    });
    return ok;
  }
  function hide() {
    if (done) return; done = true; clearInterval(tick);
    var o = document.getElementById('asphBoot');
    if (!o) return;
    requestAnimationFrame(function() { requestAnimationFrame(function() {
      o.style.opacity = '0';
      setTimeout(function() { o.remove(); }, 160);
    }); });
  }
  window.addEventListener('load', function() { loaded = true; });
  var started = Date.now();
  (function check() {
    if (done) return;
    if (loaded && busy === 0 && Date.now() - lastDone > 250 && plotsDrawn()) return hide();
    if (Date.now() - started > 30000) return hide();      // never hide the page for good
    setTimeout(check, 50);
  })();
})();
</script>
"""


LOADER_CSS = """
  .loader-row { grid-column: 1 / -1; display: flex; align-items: center; gap: 8px;
                flex-wrap: wrap; padding-bottom: 8px; border-bottom: 1px solid #374151; }
  .loader-row .lbl { font-size: 12px; font-weight: 600; text-transform: uppercase;
                     letter-spacing: .06em; color: #f97316; }
  .loader-row input[type=text] { flex: 1; min-width: 260px; background: #374151;
                     border: 1px solid #4b5563; border-radius: 4px; color: #e5e7eb;
                     padding: 4px 8px; font-size: 13px; }
  .loader-row input[type=number] { width: 96px; background: #374151; border: 1px solid #4b5563;
                     border-radius: 4px; color: #e5e7eb; padding: 3px 6px; font-size: 13px; }
  .loader-row input:focus { outline: none; border-color: #f97316; }
  .loader-row button { background: #f97316; border: none; border-radius: 4px; color: #111827;
                     font-size: 12px; font-weight: 700; padding: 4px 12px; cursor: pointer; }
  .loader-row button.sec { background: #374151; color: #e5e7eb; border: 1px solid #4b5563; }
  .loader-row button:hover { filter: brightness(1.1); }
  #ldStatus { font-size: 11px; }
  #ldPanel { grid-column: 1 / -1; background: #111827; border: 1px solid #374151;
             border-radius: 6px; max-height: 320px; overflow-y: auto; margin-bottom: 6px; }
  #ldPanel .hdr { position: sticky; top: 0; background: #1f2937; padding: 6px 10px;
                  display: flex; gap: 8px; align-items: center; border-bottom: 1px solid #374151; }
  #ldPanel .hdr input { flex: 1; background: #374151; border: 1px solid #4b5563; border-radius: 4px;
                        color: #e5e7eb; padding: 3px 6px; font-size: 12px; }
  #ldPanel .item { display: flex; gap: 12px; padding: 4px 12px; cursor: pointer;
                   font-variant-numeric: tabular-nums; }
  #ldPanel .item:hover { background: #1f2937; }
  #ldPanel .item.sel { background: #7c2d12; }
  #ldPanel .item .nm { flex: 1; }
  #ldPanel .item.dir .nm { color: #fbbf24; }
  #ldPanel .item .meta { color: #9ca3af; font-size: 11px; width: 150px; text-align: right; }
  #ldPanel .empty { padding: 8px 12px; color: #9ca3af; }
  #ldPanel .conn { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; padding: 6px 10px;
                   border-bottom: 1px solid #374151; color: #9ca3af; font-size: 12px; }
  #ldPanel .conn .chip { background: #1f2937; color: #e5e7eb; border: 1px solid #f97316; border-radius: 12px;
                         padding: 2px 9px; font-size: 12px; cursor: pointer; font-weight: 500; }
  #ldPanel .conn .chip.here { background: #f97316; color: #111827; }
  #ldPanel .conn .chip:disabled { opacity: .45; cursor: default; }
"""

LOADER_HTML = """
  <div class="loader-row">
    <span class="lbl">File</span>
    <input type="text" id="ldPath" placeholder="/path/to/file.npz" value="__PATH__"
           onkeydown="if(event.key==='Enter') ldLoad()">
    <button class="sec" onclick="ldToggle()" title="Browse for .npz files">Browse ▾</button>
    <span class="lbl" title="Leave blank to load the full frequency range">Freq (Hz)</span>
    <input type="number" id="ldFmin" step="any" placeholder="min (full)" value="__FMIN__">
    <input type="number" id="ldFmax" step="any" placeholder="max (full)" value="__FMAX__">
    <button onclick="ldLoad()" title="Load this file (or re-crop the current one)">Load</button>
    <span id="ldStatus"></span>
  </div>
  <div id="ldPanel" style="display:none;"></div>
"""

LOADER_JS = """
var LD_INIT = __INIT__;
var ldCurDir = null;

function ldStatus(msg, kind) {
  var s = document.getElementById('ldStatus');
  s.textContent = msg;
  s.style.color = kind === 'err' ? '#f87171' : kind === 'ok' ? '#4ade80' : '#fbbf24';
}

function ldToggle(open) {
  var p = document.getElementById('ldPanel');
  var show = (open === undefined) ? (p.style.display === 'none') : open;
  p.style.display = show ? 'block' : 'none';
  if (show) ldList(ldCurDir || LD_INIT.dir);
}

function _fmtSize(b) {
  if (b >= 1e9) return (b / 1e9).toFixed(2) + ' GB';
  if (b >= 1e6) return (b / 1e6).toFixed(1) + ' MB';
  if (b >= 1e3) return (b / 1e3).toFixed(0) + ' kB';
  return b + ' B';
}

async function ldList(dir) {
  try {
    var r = await fetch('/list-dir?path=' + encodeURIComponent(dir || ''));
    var d = await r.json();
    if (!r.ok) { ldStatus('✗ ' + (d.error || 'cannot list folder'), 'err'); return; }
    ldCurDir = d.path;
    var p = document.getElementById('ldPanel');
    p.innerHTML = '';
    var hdr = document.createElement('div'); hdr.className = 'hdr';
    var up = document.createElement('button'); up.textContent = '↑ Up'; up.className = 'sec';
    up.style.cssText = 'background:#374151;color:#e5e7eb;border:1px solid #4b5563;border-radius:4px;padding:2px 8px;cursor:pointer;';
    up.onclick = function() { ldList(d.parent); };
    var inp = document.createElement('input'); inp.value = d.path; inp.title = 'Type a folder and press Enter';
    inp.onkeydown = function(e) { if (e.key === 'Enter') ldList(inp.value); };
    hdr.appendChild(up); hdr.appendChild(inp); p.appendChild(hdr);
    // the experiment's connected data folders (home page → Folders), one click away
    if (d.connected && d.connected.length) {
      var cn = document.createElement('div'); cn.className = 'conn';
      var lb = document.createElement('span'); lb.textContent = 'Connected Folders:'; cn.appendChild(lb);
      d.connected.forEach(function(f) {
        var b = document.createElement('button');
        b.textContent = f.name === f.kind ? f.name : (f.name + ' · ' + f.kind);
        b.title = f.path + (f.ok ? '' : '  (missing)');
        b.className = 'chip' + (f.path.replace(/\\/$/, '') === d.path.replace(/\\/$/, '') ? ' here' : '');
        if (!f.ok) b.disabled = true;
        b.onclick = function() { ldList(f.path); };
        cn.appendChild(b);
      });
      p.appendChild(cn);
    }
    if (d.note) {
      var nt = document.createElement('div'); nt.className = 'item';
      nt.style.cssText = 'color:#fbbf24;cursor:default;white-space:normal;';
      nt.textContent = d.note;
      p.appendChild(nt);
    }
    var cur = document.getElementById('ldPath').value;
    d.dirs.forEach(function(name) {
      var it = document.createElement('div'); it.className = 'item dir';
      var nm = document.createElement('span'); nm.className = 'nm'; nm.textContent = '▸ ' + name + '/';
      it.appendChild(nm);
      it.onclick = function() { ldList(d.path.replace(/\\/$/, '') + '/' + name); };
      p.appendChild(it);
    });
    d.files.forEach(function(f) {
      var full = d.path.replace(/\\/$/, '') + '/' + f.name;
      var it = document.createElement('div'); it.className = 'item' + (full === cur ? ' sel' : '');
      var nm = document.createElement('span'); nm.className = 'nm'; nm.textContent = f.name;
      var mt = document.createElement('span'); mt.className = 'meta';
      mt.textContent = _fmtSize(f.size) + '   ' + f.mtime;
      it.appendChild(nm); it.appendChild(mt);
      it.title = 'Click to select, double-click to load';
      it.onclick = function() {
        document.getElementById('ldPath').value = full;
        Array.prototype.forEach.call(p.querySelectorAll('.item.sel'), function(x) { x.classList.remove('sel'); });
        it.classList.add('sel');
      };
      it.ondblclick = function() { document.getElementById('ldPath').value = full; ldLoad(); };
      p.appendChild(it);
    });
    if (!d.dirs.length && !d.files.length) {
      var e = document.createElement('div'); e.className = 'empty'; e.textContent = 'No folders or .npz files here';
      p.appendChild(e);
    }
  } catch (e) { ldStatus('✗ ' + e.message, 'err'); }
}

async function ldLoad() {
  var path = document.getElementById('ldPath').value.trim();
  if (!path) { ldStatus('✗ choose a file first', 'err'); return; }
  var fMinRaw = document.getElementById('ldFmin').value.trim();
  var fMaxRaw = document.getElementById('ldFmax').value.trim();
  if (fMinRaw !== '' && fMaxRaw !== '' && +fMinRaw >= +fMaxRaw) {
    ldStatus('✗ freq min must be < max', 'err'); return;
  }
  var db  = document.getElementById('dbCheck');
  var nav = document.getElementById('nAvg');
  var payload = {
    file:   path,
    fMin:   fMinRaw === '' ? null : +fMinRaw,
    fMax:   fMaxRaw === '' ? null : +fMaxRaw,
    linear: db ? !db.checked : false,
    nAvg:   nav ? Math.max(1, Math.round(+nav.value || 1)) : 1
  };
  ldStatus('Loading…');
  try {
    var r = await fetch('/load', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    var d = await r.json();
    if (!r.ok) { ldStatus('✗ ' + (d.error || 'load failed'), 'err'); return; }
    ldStatus('✓ Loaded', 'ok');
    // replace (not a new history entry), so the tab stays closable by script
    window.location.replace('/spectrogram.html?t=' + Date.now());
  } catch (e) { ldStatus('✗ ' + e.message, 'err'); }
}
"""


COLORSCALE_NAMES = ["Inferno", "Magma", "Plasma", "Viridis", "Cividis", "Turbo", "Hot", "Jet",
                    "Electric", "Blackbody", "Greys", "Blues", "YlGnBu", "Ice", "Haline"]


def colorscales_dict():
    """{name: [[pos, colour], ...]} for the colour-scheme dropdown."""
    import plotly.colors as pc
    out = {}
    for n in COLORSCALE_NAMES:
        try:
            out[n] = [[float(p), c] for p, c in pc.get_colorscale(n)]
        except Exception:      # very old plotly: name unknown -> leave it out
            pass
    if "Inferno" not in out:   # always keep the default available
        out["Inferno"] = [[0.0, "rgb(0,0,4)"], [0.1, "rgb(27,11,65)"], [0.2, "rgb(74,12,107)"],
                          [0.3, "rgb(119,28,109)"], [0.4, "rgb(163,47,92)"], [0.5, "rgb(200,70,67)"],
                          [0.6, "rgb(229,103,37)"], [0.7, "rgb(246,142,9)"], [0.8, "rgb(252,185,22)"],
                          [0.9, "rgb(252,226,83)"], [1.0, "rgb(252,255,164)"]]
    return out


SIM_CSS = """
  .sim-row .opt-grp input[type=text] { width: 80px; background: #374151;
                     border: 1px solid #4b5563; border-radius: 4px; color: #e5e7eb;
                     padding: 3px 6px; font-size: 13px; }
  .sim-row .opt-grp input[type=text]:focus { outline: none; border-color: #f97316; }
  #colorscaleSel, .sim-row .opt-grp select { background: #374151; border: 1px solid #4b5563; border-radius: 4px;
                     color: #e5e7eb; padding: 3px 4px; font-size: 13px; }
  table.ref-tbl { border-collapse: collapse; font-size: 11px; margin-top: 2px; }
  table.ref-tbl th { background: #f97316; color: #fff; font-weight: 600; padding: 2px 8px; }
  table.ref-tbl td { padding: 2px 8px; text-align: right; font-family: monospace;
                     border-bottom: 1px solid #374151; }
  table.ref-tbl td.st { text-align: left; font-family: inherit; }
  table.ref-tbl td.found { color: #4ade80; font-weight: 600; }
  table.ref-tbl td.not_found { color: #f87171; font-weight: 600; }
  .sim-modal { display: none; position: fixed; inset: 0; z-index: 1000;
               background: rgba(0,0,0,.6); align-items: center; justify-content: center; }
  .sim-modal.open { display: flex; }
  .sim-dialog { background: #1f2937; border: 1px solid #374151; border-radius: 8px;
                box-shadow: 0 10px 40px rgba(0,0,0,.5); width: min(760px, calc(100vw - 32px));
                max-height: calc(100vh - 32px); overflow-y: auto; }
  .sim-head { display: flex; align-items: center; justify-content: space-between;
              padding: 10px 16px; border-bottom: 1px solid #374151; }
  .sim-head #simTitle, .sim-head #decayTitle, .sim-head #generalTitle, .sim-head #fileTitle,
  .sim-head #searchTitle { font-size: 13px; font-weight: 600; text-transform: uppercase;
                        letter-spacing: .06em; color: #f97316; }
  .sim-x { background: none; border: none; color: #9ca3af; font-size: 20px; line-height: 1;
           cursor: pointer; padding: 0 4px; }
  .sim-x:hover { color: #e5e7eb; }
  .sim-body { display: flex; flex-direction: column; gap: 12px; padding: 14px 16px 16px; }
  .sim-body .opt-grp { flex-wrap: wrap; }
  .file-dialog { width: min(1000px, calc(100vw - 32px)); }
  .srch-hint { color: #9ca3af; font-size: 12px; }
  .opt-grp button.quit-btn { background: #b91c1c; color: #fef2f2; }
  .opt-grp button.quit-btn:hover { background: #dc2626; }
  .opt-grp button.quit-btn.armed { background: #fef2f2; color: #b91c1c; }
  .reg-banner { display: none; position: fixed; top: 12px; left: 50%; transform: translateX(-50%);
                z-index: 900; background: #f97316; color: #111827; font-weight: 700; font-size: 13px;
                padding: 8px 16px; border-radius: 6px; box-shadow: 0 4px 16px rgba(0,0,0,.4); }
  #heatDiv.drawing { outline: 2px dashed #f97316; outline-offset: 2px; }
  #heatDiv.drawing .nsewdrag { cursor: crosshair !important; }
  .opt-grp button:disabled { opacity: .35; cursor: not-allowed; }
  .srch-list { background: #111827; border: 1px solid #374151; border-radius: 6px;
               max-height: 360px; overflow-y: auto; }
  .srch-list:empty { display: none; }
  .srch-list .item { display: grid; grid-template-columns: 1.2fr .6fr 1.6fr 1fr; gap: 12px;
                     padding: 5px 12px; cursor: pointer; font-variant-numeric: tabular-nums; }
  .srch-list .item:hover, .srch-list .item:focus { background: #7c2d12; outline: none; }
  .srch-list .hdr { display: grid; grid-template-columns: 1.2fr .6fr 1.6fr 1fr; gap: 12px;
                    padding: 5px 12px; position: sticky; top: 0; background: #1f2937;
                    color: #9ca3af; font-size: 11px; text-transform: uppercase; letter-spacing: .06em; }
  .srch-list .ex { color: #9ca3af; }
  .opt-grp span.dsug-status,
  .opt-grp span#srchStatus { font-size: 11px; text-transform: none; letter-spacing: normal;
                             font-weight: 400; }
  .file-dialog .loader-row { border-bottom: none; padding-bottom: 0; }
  .full-toggle { background: #1f2937; border: 1px solid #374151; border-radius: 6px;
                 color: #e5e7eb; font-size: 12px; font-weight: 600; text-transform: uppercase;
                 letter-spacing: .06em; padding: 6px 12px; cursor: pointer; width: 100%;
                 text-align: left; margin-bottom: 4px; }
  .full-toggle:hover { border-color: #f97316; }
  .full-toggle #fullArrow { color: #f97316; display: inline-block; width: 12px; }
  .opt-grp span#simStatus { font-size: 11px; margin-left: 4px; text-transform: none;
                            letter-spacing: normal; font-weight: 400; }
  .opt-grp span#blStatus { font-size: 11px; margin-left: 4px; text-transform: none;
                           letter-spacing: normal; font-weight: 400; }
  .bl-grid { display: grid; grid-template-columns: max-content max-content; justify-content: start;
             gap: 6px 14px; align-items: center; margin: 4px 0 6px 2px; }
  .bl-grid label { font-size: 12px; color: #d1d5db; }
  .bl-grid input[type=text] { width: 90px; background: #374151; border: 1px solid #4b5563;
                              border-radius: 4px; color: #f3f4f6; padding: 2px 6px; font-size: 12px; }
  .bl-grid input[type=text]:focus { outline: none; border-color: #f97316; }
"""


def sim_panel_html():
    """HTML for the 'Simulated ions' settings row, pre-filled from prefs."""
    p = sim_prefs()
    esc = lambda v: str(v).replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")
    chk = lambda k: "checked" if p.get(k) else ""

    def txt(key, width, tip, placeholder=""):
        return (f'<input type="text" id="sim_{key}" style="width:{width}px;" '
                f'value="{esc(p.get(key, ""))}" placeholder="{placeholder}" title="{esc(tip)}" '
                f'onkeydown="if(event.key===\'Enter\') runSim()">')

    units = "".join(f'<option value="{u}"{" selected" if u == p.get("min_half_life_unit") else ""}>{u}</option>'
                    for u in HL_CUT_UNITS)
    corr_name = corrections.resolve(p.get("corr_profile"), p.get("corr_a"), p.get("corr_b"),
                                    p.get("corr_c"))["name"]
    return f"""
  <div id="simModal" class="sim-modal" onclick="if(event.target===this) simClose()">
   <div class="sim-dialog sim-row" role="dialog" aria-modal="true" aria-labelledby="simTitle">
    <div class="sim-head">
      <span id="simTitle">PID Settings</span>
      <button class="sim-x" onclick="simClose()" title="Close (Esc)">×</button>
    </div>
    <div class="sim-body">
    <div class="opt-grp">
      <input type="checkbox" id="sim_show" {chk("show")} onchange="runSim()">
      <label for="sim_show" title="Draw simulated ion frequencies on the 2-D plot">show</label>
      <input type="checkbox" id="sim_show_excluded" {chk("show_excluded")} onchange="runSim()">
      <label for="sim_show_excluded" title="Also draw ions excluded by the half-life cut (dotted)">excluded</label>
    </div>
    <div class="opt-grp" title="The strongest bin of the power summed over ALL time frames (whole uncropped file) is taken to be this ion at this harmonic">
      <span>Ref</span>
      {txt("ref_name", 90, "Reference ion, e.g. 208Pb80+")}
      <label>h</label>
      {txt("ref_harmonic", 44, "Harmonic of the reference peak")}
      <span style="text-transform:none;">Other h ±</span>
      {txt("ref_search_hz", 44, "Each harmonic's lines are placed from the reference ion's own peak in that harmonic: nearest peak to f_ref/h_ref·h within ± this (Hz). 0 = off")}
    </div>
    <div class="opt-grp">
      <span>Z</span>{txt("z_lims", 64, "Z min, max")}
      <span>N</span>{txt("n_lims", 64, "N min, max")}
    </div>
    <div class="opt-grp" title="Diagonal cuts (as in Theia): two (A, Z) points defining the selection band">
      <span style="text-transform:none;">el1</span>{txt("el1", 64, "el1")}
      <span style="text-transform:none;">el2</span>{txt("el2", 64, "el2")}
    </div>
    <div class="opt-grp">
      <span style="text-transform:none;">m/q ±</span>{txt("mq_width", 48, "m/q window around the reference ion")}
      <span style="text-transform:none;">γt</span>{txt("gamma_t", 54, "Transition gamma γt")}
    </div>
    <div class="opt-grp">
      <span>Harmonics</span>{txt("harmonics", 96, "Harmonics to simulate, comma-separated")}
    </div>
    <div class="opt-grp" title="Second-order correction applied to the revolution frequency: f + a·f² + b·f + c">
      <input type="checkbox" id="sim_correction_enabled" {chk("correction_enabled")}>
      <label for="sim_correction_enabled">correction</label>
      <select id="sim_corr_profile" onchange="corrPick()" onfocus="corrRefresh()" style="max-width:200px;"
              title="Correction profile (saved fits of the 2nd-order correction, possibly one per harmonic); picking one fills a, b, c with its general values"><option value="{esc(corr_name)}" selected>{esc(corr_name)}</option></select>
      <span>a</span>{txt("corr_a", 150, "a")}
      <span>b</span>{txt("corr_b", 150, "b")}
      <span>c</span>{txt("corr_c", 150, "c")}
    </div>
    <div class="opt-grp" id="sim_corr_info" style="display:block;font-size:11px;opacity:.85;max-width:760px;"></div>
    <div class="opt-grp" title="Ions with a half-life below this are drawn dotted (as 'excluded'). 0 = off">
      <span>T½ ≥</span>{txt("min_half_life", 44, "Minimum half-life (0 = off)")}
      <select id="sim_min_half_life_unit">{units}</select>
      <input type="checkbox" id="sim_exclude_unknown_hl" {chk("exclude_unknown_hl")}>
      <label for="sim_exclude_unknown_hl" title="Also exclude ions with an unknown half-life">+ unknown</label>
    </div>
    <div class="opt-grp">
      <button onclick="runSim()">Simulate</button>
      <span id="simStatus"></span>
    </div>
    <div class="opt-grp" id="simRefTable" style="display:block;"></div>
    </div>
   </div>
  </div>
"""


def baseline_prefs():
    """Saved baseline settings: {"on": bool, **BrPLS parameters}."""
    raw = read_prefs().get("baseline")
    raw = raw if isinstance(raw, dict) else {}
    try:
        params = baseline.parse_params(raw)
    except ValueError:
        params = dict(baseline.DEFAULTS)
    return {"on": bool(raw.get("on")), **params}


def baseline_panel_html(info):
    """Popup with the BrPLS baseline-removal settings. info: the loaded
    page's baseline state {"on", "params", "error"}."""
    import json
    p = baseline_prefs()
    esc = lambda v: str(v).replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")
    rows = "".join(
        f'<label for="bl_{k}" title="{esc(tip)}">{esc(label)}</label>'
        f'<input type="text" id="bl_{k}" value="{p[k]:g}" placeholder="{baseline.DEFAULTS[k]:g}" '
        f'title="{esc(tip)}" onkeydown="if(event.key===\'Enter\') baselineApply()">'
        for k, (label, tip) in baseline.FIELDS.items())
    err = info.get("error")
    status = (f'<span id="blStatus" style="color:#f87171;">✗ {esc(err)}</span>' if err else
              f'<span id="blStatus" style="color:#4ade80;">✓ on: {esc(baseline.describe(info["params"]))}</span>'
              if info.get("on") else '<span id="blStatus"></span>')
    return f"""
  <div id="blModal" class="sim-modal" onclick="if(event.target===this) modalClose('blModal')">
   <div class="sim-dialog" role="dialog" aria-modal="true" aria-labelledby="blTitle">
    <div class="sim-head">
      <span id="blTitle">Baseline Removal</span>
      <button class="sim-x" onclick="modalClose('blModal')" title="Close (Esc)">×</button>
    </div>
    <div class="sim-body">
    <div class="opt-grp">
      <input type="checkbox" id="bl_on" {"checked" if p["on"] else ""}>
      <label for="bl_on" title="Fit and subtract a baseline in every time frame (linear power), over the loaded frequency window">remove baseline (BrPLS) from every frame</label>
    </div>
    <div class="srch-hint">pybaselines BrPLS, fitted to each frame's linear power over the loaded
      frequency window, before time averaging. Decay fits started from this page use the same
      corrected frames. Empty = default.</div>
    <div class="bl-grid">{rows}</div>
    <div class="opt-grp">
      <button onclick="baselineApply()">Apply</button>
      <button onclick="baselineDefaults()" title="Fill in pybaselines' defaults">Defaults</button>
      {status}
    </div>
    </div>
   </div>
  </div>
  <script>
  var BL_DEFAULTS = {json.dumps(baseline.DEFAULTS)};
  function baselineDefaults() {{
    Object.keys(BL_DEFAULTS).forEach(function(k) {{ document.getElementById('bl_' + k).value = BL_DEFAULTS[k]; }});
  }}
  async function baselineApply() {{
    var st = document.getElementById('blStatus');
    var body = {{on: document.getElementById('bl_on').checked}};
    Object.keys(BL_DEFAULTS).forEach(function(k) {{ body[k] = document.getElementById('bl_' + k).value.trim(); }});
    var nav = document.getElementById('nAvg'), db = document.getElementById('dbCheck');
    body.nAvg = nav ? Math.max(1, Math.round(+nav.value || 1)) : 1;
    body.linear = db ? !db.checked : false;
    st.style.color = '#d1d5db';
    st.textContent = body.on ? 'Removing the baseline from every frame…' : 'Reloading without baseline removal…';
    try {{
      var r = await fetch('/baseline', {{method: 'POST', headers: {{'Content-Type': 'application/json'}},
                                         body: JSON.stringify(body)}});
      var d = await r.json();
      if (!r.ok) {{ st.style.color = '#f87171'; st.textContent = '✗ ' + (d.error || 'failed'); return; }}
      window.location.replace('/spectrogram.html?t=' + Date.now());
    }} catch (e) {{ st.style.color = '#f87171'; st.textContent = '✗ ' + e.message; }}
  }}
  </script>
"""


SIM_JS = """
// ── Simulated ion lines on the 2-D plot (physics from core/physics.py, as in Theia) ────
var SIM = null;              // last /simulate result
var SIM_TRACES = [2, 3, 4, 5, 6, 7];  // heatPlot: outlines (2-4) then white lines (5-7)
                                      // for normal, excluded, reference
var SIM_KEYS = ['ref_name', 'ref_harmonic', 'z_lims', 'n_lims', 'el1', 'el2', 'mq_width',
                'gamma_t', 'harmonics', 'corr_a', 'corr_b', 'corr_c', 'min_half_life',
                'min_half_life_unit', 'corr_profile', 'ref_search_hz'];
var SIM_CHECKS = ['show', 'show_excluded', 'correction_enabled', 'exclude_unknown_hl'];

// ── Correction profiles (core/corrections.py) ───────────────────────────────
// The profile name is the setting; picking a profile fills a, b, c with its
// general values and locks them ('custom' unlocks them).
var CORR_PROFILES = [];
function corrLock(locked) {
  ['corr_a', 'corr_b', 'corr_c'].forEach(function(k) {
    var el = document.getElementById('sim_' + k);
    el.disabled = locked; el.style.opacity = locked ? 0.6 : 1;
  });
}
async function corrRefresh() {
  var q = new URLSearchParams();
  ['corr_a', 'corr_b', 'corr_c', 'ref_name', 'ref_harmonic', 'gamma_t', 'harmonics',
   'corr_profile'].forEach(function(k) {
    q.set(k, document.getElementById('sim_' + k).value);
  });
  try {
    var r = await fetch('/correction-profiles?' + q.toString());
    var d = await r.json();
    if (!r.ok) throw new Error(d.error || 'error');
    CORR_PROFILES = d.profiles;
    var sel = document.getElementById('sim_corr_profile');
    sel.innerHTML = '';
    d.profiles.concat([{name: d.custom, label: d.custom_label, describe: ''}]).forEach(function(p) {
      var o = document.createElement('option');
      o.value = p.name; o.textContent = p.label || p.name; o.title = p.describe || '';
      sel.appendChild(o);
    });
    sel.value = d.match || d.custom;
    var pm = d.profiles.find(function(x) { return x.name === sel.value; });
    if (pm) {                    // the boxes show the profile's general a, b, c
      document.getElementById('sim_corr_a').value = pm.a_text;
      document.getElementById('sim_corr_b').value = pm.b_text;
      document.getElementById('sim_corr_c').value = pm.c_text;
    }
    corrLock(sel.value !== d.custom);
    var info = document.getElementById('sim_corr_info');
    info.innerHTML = '';
    d.info.forEach(function(line) {
      var div = document.createElement('div');
      div.textContent = line;
      info.appendChild(div);
    });
    (d.warnings || []).forEach(function(line) {
      var div = document.createElement('div');
      div.textContent = '⚠ ' + line;
      div.style.color = '#fbbf24';
      info.appendChild(div);
    });
  } catch (e) {
    document.getElementById('sim_corr_info').textContent = '✗ correction profiles: ' + e.message;
  }
}
function corrPick() {
  var name = document.getElementById('sim_corr_profile').value;
  var p = CORR_PROFILES.find(function(x) { return x.name === name; });
  if (p) {
    document.getElementById('sim_corr_a').value = p.a_text;
    document.getElementById('sim_corr_b').value = p.b_text;
    document.getElementById('sim_corr_c').value = p.c_text;
  }
  corrRefresh();
}
document.addEventListener('DOMContentLoaded', function() {
  ['corr_a', 'corr_b', 'corr_c', 'ref_name', 'ref_harmonic', 'gamma_t', 'harmonics'].forEach(function(k) {
    var el = document.getElementById('sim_' + k);
    if (el) el.addEventListener('change', corrRefresh);
  });
  corrRefresh();
});

function _simStatus(msg, kind) {
  var s = document.getElementById('simStatus');
  s.textContent = msg;
  s.style.color = kind === 'err' ? '#f87171' : kind === 'ok' ? '#4ade80' : '#fbbf24';
}

function _simRaw() {
  var raw = {};
  SIM_KEYS.forEach(function(k) { raw[k] = document.getElementById('sim_' + k).value; });
  SIM_CHECKS.forEach(function(k) { raw[k] = document.getElementById('sim_' + k).checked; });
  return raw;
}

var _simSeq = 0;
async function runSim() {
  var raw = _simRaw();
  var seq = ++_simSeq;
  if (!raw.show) {
    // just save the toggle and clear the lines
    fetch('/simulate', {method: 'POST', headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(Object.assign({save_only: true}, raw))});
    _renderRefTable(null); _renderSim(); _renderFull(); _simStatus('');
    return;
  }
  _simStatus('Simulating…');
  try {
    var r = await fetch('/simulate', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(raw)
    });
    var d = await r.json();
    if (seq !== _simSeq) return;
    if (!r.ok) { SIM = null; _renderRefTable(null); _renderSim(); _renderFull(); _simStatus('✗ ' + (d.error || 'error'), 'err'); return; }
    SIM = d;
    _renderRefTable(d);
    _renderSim();
    _renderFull();
    _simStatus('✓ ref ' + d.ref_name + ' h' + d.ref_harmonic + ' = ' +
               d.ref_freq.toLocaleString('en-GB', {maximumFractionDigits: 1}) + ' Hz · ' +
               d.lines.length + ' lines across the whole file' +
               (d.warnings && d.warnings.length ? ' · ⚠ ' + d.warnings.join(' · ⚠ ') : ''),
               d.warnings && d.warnings.length ? 'warn' : 'ok');
  } catch (e) {
    if (seq === _simSeq) _simStatus('✗ ' + e.message, 'err');
  }
}

// Where the reference ion's peak is in each harmonic of the summed power
function _renderRefTable(d) {
  var el = document.getElementById('simRefTable');
  var rows = (d && d.ref_rows) || [];
  if (!rows.length) { el.innerHTML = ''; return; }
  var fmt = function(v, sign) {
    if (v === null || v === undefined) return '—';
    var s = v.toLocaleString('en-GB', {minimumFractionDigits: 1, maximumFractionDigits: 1});
    return sign && v >= 0 ? '+' + s : s;
  };
  var esc = function(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;'); };
  var note = d.ref_search_hz > 0 ? 'searched within ±' + d.ref_search_hz + ' Hz of f_ref / h_ref × h'
                                 : 'search off (ref ± = 0): predicted positions used';
  var h = '<div style="font-weight:600;margin-top:4px;">' + esc(d.ref_name) +
          ' peak in each harmonic <span style="font-weight:400;opacity:.7;">(' + note +
          ')</span></div><table class="ref-tbl"><tr><th>h</th><th>Status</th>' +
          '<th>Expected (Hz)</th><th>Found at (Hz)</th><th>Offset (Hz)</th></tr>';
  rows.forEach(function(r) {
    var shown = r.status === 'reference' || r.status === 'found';
    h += '<tr><td>' + r.h + '</td><td class="st ' + r.status + '">' + esc(r.label) + '</td><td>' +
         fmt(r.predicted) + '</td><td>' + (shown ? fmt(r.f) : '—') + '</td><td>' +
         fmt(r.offset, true) + '</td></tr>';
  });
  el.innerHTML = h + '</table>';
}

// Redraw the lines that fall inside the current frequency-slider range
function _renderSim() {
  if (typeof _heatReady === 'undefined' || !_heatReady) return;
  var groups = [{x: [], y: [], h: []}, {x: [], y: [], h: []}, {x: [], y: [], h: []}];
  var show = document.getElementById('sim_show').checked;
  if (show && SIM && T_RAW.length) {
    var showEx = document.getElementById('sim_show_excluded').checked;
    var fMin = +document.getElementById('fMin').value;
    var fMax = +document.getElementById('fMax').value;
    // Plotly only shows a tooltip near a data point, so each line gets a
    // point every few pixels across the visible time window (plus the ends
    // of the full time range so the line still spans the data when zoomed out)
    var t0 = T_RAW[0], t1 = T_RAW[T_RAW.length - 1];
    var v0 = Math.max(t0, +document.getElementById('tMin').value);
    var v1 = Math.min(t1, +document.getElementById('tMax').value);
    if (!(v1 > v0)) { v0 = t0; v1 = t1; }
    var gd = document.getElementById('heatPlot');
    var wPx = (gd && gd._fullLayout && gd._fullLayout.xaxis) ? gd._fullLayout.xaxis._length : 800;
    var vis = SIM.lines.filter(function(l) {
      return l.f >= fMin && l.f <= fMax && !(l.excluded && !l.is_ref && !showEx);
    });
    var N = Math.max(20, Math.ceil(wPx / 6));                     // ~every 6 px
    N = Math.min(N, Math.max(20, Math.floor(60000 / Math.max(vis.length, 1))));
    var xs = [t0];
    for (var i = 0; i < N; i++) xs.push(v0 + (v1 - v0) * i / (N - 1));
    xs.push(t1);
    vis.forEach(function(l) {
      var g = groups[l.is_ref ? 2 : (l.excluded ? 1 : 0)];
      for (var i = 0; i < xs.length; i++) { g.x.push(xs[i]); g.y.push(l.f); g.h.push(l.label); }
      g.x.push(null); g.y.push(null); g.h.push(null);
    });
  }
  // same geometry for the dark outlines and the white lines on top
  var both = groups.concat(groups);
  Plotly.restyle('heatPlot', {
    x: both.map(function(g) { return g.x; }),
    y: both.map(function(g) { return g.y; }),
    hovertext: both.map(function(g) { return g.h; })
  }, SIM_TRACES);
}

// ── Whole-spectrum 1-D plot (all frames, whole loaded range) with sim lines ──
var FULL = null;   // {y, n} summed power, aligned with F_1D

async function _fetchFull() {
  try {
    var r = await fetch('/spectrum-file');       // whole uncropped file, all frames
    var d = await r.json();
    if (!r.ok) throw new Error(d.error || 'error');
    FULL = d;
    _renderFull();
  } catch (e) { console.log('[full1d] ' + e.message); }
}

// ── Search ions: list an ion's simulated lines, load ±width around one ─────
var SEARCH_HALF_WIDTH_HZ = 10000;   // default when the width box is empty/invalid
function _srchHalfWidthHz() {
  var el = document.getElementById('srchWidth');
  var v = el ? parseFloat(el.value) : NaN;
  return (isFinite(v) && v > 0) ? v * 1000 : SEARCH_HALF_WIDTH_HZ;   // box is in kHz
}
function srchWidthChanged() {
  var el = document.getElementById('srchWidth');
  var v = parseFloat(el.value);
  el.style.borderColor = (el.value.trim() === '' || (isFinite(v) && v > 0)) ? '' : '#f87171';
  savePref('srchWidth', (isFinite(v) && v > 0) ? String(v) : '');
  // refresh the per-line tooltips with the new range
  if (document.getElementById('srchIon').value.trim()) searchIonsSoon();
}
var _srchTimer = null, _srchSeq = 0;
function _srchStatus(msg, kind) {
  var s = document.getElementById('srchStatus');
  s.textContent = msg;
  s.style.color = kind === 'err' ? '#f87171' : kind === 'ok' ? '#4ade80' : '#fbbf24';
}
function searchIonsSoon() { clearTimeout(_srchTimer); _srchTimer = setTimeout(searchIons, 350); }
async function searchIons() {
  clearTimeout(_srchTimer);
  var q = document.getElementById('srchIon').value.trim();
  var list = document.getElementById('srchList');
  if (!q) { list.innerHTML = ''; _srchStatus(''); return; }
  var seq = ++_srchSeq;
  _srchStatus('Searching…');
  try {
    var r = await fetch('/search-ion', {method: 'POST', headers: {'Content-Type': 'application/json'},
                                        body: JSON.stringify({q: q})});
    var d = await r.json();
    if (seq !== _srchSeq) return;
    list.innerHTML = '';
    if (!r.ok) { _srchStatus('✗ ' + (d.error || 'error'), 'err'); return; }
    if (!d.hits.length) { _srchStatus('No simulated lines for "' + q + '" in this file', 'err'); return; }
    _srchStatus(d.n + ' line' + (d.n === 1 ? '' : 's') + (d.n > d.hits.length ? ' (first ' + d.hits.length + ' shown)' : ''), 'ok');
    var hdr = document.createElement('div'); hdr.className = 'hdr';
    ['Ion', 'Harmonic', 'f_sim (Hz)', 'T½'].forEach(function(t) {
      var c = document.createElement('span'); c.textContent = t; hdr.appendChild(c);
    });
    list.appendChild(hdr);
    d.hits.forEach(function(l) {
      var it = document.createElement('div');
      it.className = 'item' + (l.excluded ? ' ex' : '');
      it.tabIndex = 0;
      var hw = _srchHalfWidthHz();
      it.title = 'Load ' + Math.round(l.f - hw).toLocaleString('en-GB') + ' – ' +
                 Math.round(l.f + hw).toLocaleString('en-GB') + ' Hz' +
                 (l.excluded ? '  (below the T½ cut)' : '');
      [l.ion + (l.is_ref ? '  (ref)' : ''), 'h' + l.h,
       l.f.toLocaleString('en-GB', {maximumFractionDigits: 1}), l.t_half || '?'].forEach(function(t) {
        var c = document.createElement('span'); c.textContent = t; it.appendChild(c);
      });
      it.onclick = function() { loadAroundIon(d.file, l); };
      it.onkeydown = function(e) { if (e.key === 'Enter') loadAroundIon(d.file, l); };
      list.appendChild(it);
    });
  } catch (e) { if (seq === _srchSeq) _srchStatus('✗ ' + e.message, 'err'); }
}
// Load the same file cropped to f ± width (Width box, kHz) through the same /load request as
// File Settings → Load (keeps the current dB / time-avg); errors shown here
async function loadAroundIon(file, l, statusFn) {
  var _st = statusFn || _srchStatus;
  var hw = _srchHalfWidthHz();
  var fMin = l.f - hw, fMax = l.f + hw;
  document.getElementById('ldPath').value = file;
  document.getElementById('ldFmin').value = fMin.toFixed(1);
  document.getElementById('ldFmax').value = fMax.toFixed(1);
  var db  = document.getElementById('dbCheck');
  var nav = document.getElementById('nAvg');
  _st('Loading ' + l.ion + ', h' + l.h + '…');
  try {
    var r = await fetch('/load', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({file: file, fMin: fMin, fMax: fMax,
                            linear: db ? !db.checked : false,
                            nAvg: nav ? Math.max(1, Math.round(+nav.value || 1)) : 1,
                            ion: l.ion + ', h' + l.h})
    });
    var d = await r.json();
    if (!r.ok) { _st('✗ ' + (d.error || 'load failed'), 'err'); return; }
    // replace (not a new history entry), so the tab stays closable by script
    window.location.replace('/spectrogram.html?t=' + Date.now());
  } catch (e) { _st('✗ ' + e.message, 'err'); }
}

// ── Step buttons: toolbar h±1, q±1, A±1 for the ion loaded by Search ions ───
// CUR_ION_LABEL ("m186Hf72+, h213") is set by the page.  h±1: same ion, next
// harmonic.  q±1: same nuclide (isomer kept), same harmonic.  A±1: neighbouring
// isotope (ground state), same charge and harmonic.  The targets come from the
// same simulated lines as Search ions, and a button is disabled when its line
// isn't among them (outside the file, outside the Z/N range, etc.).
var _stepLines = {};   // {'h-1': {file, line}, 'q1': …, 'A-1': …}
function _harmStatus(msg, kind) {
  var s = document.getElementById('harmStatus');
  if (!s) return;
  s.textContent = msg;
  s.style.color = kind === 'err' ? '#f87171' : kind === 'ok' ? '#4ade80' : '#fbbf24';
}
// "m186Hf72+" -> {iso: 'm', A: 186, sym: 'Hf', q: 72}, or null
function _parseIon(name) {
  var m = /^([a-z]?)(\\d+)([A-Z][a-z]?)(\\d+)\\+$/.exec(String(name).trim());
  return m ? {iso: m[1], A: +m[2], sym: m[3], q: +m[4]} : null;
}
async function _harmInit() {
  var grp = document.getElementById('harmGrp');
  if (!grp || typeof CUR_ION_LABEL !== 'string' || !CUR_ION_LABEL) return;
  var m = /^(.*),\\s*h(\\d+)$/.exec(CUR_ION_LABEL.trim());
  if (!m) return;
  var ion = m[1].trim(), h = parseInt(m[2], 10), p = _parseIon(ion);
  grp.style.display = 'inline-flex';
  document.getElementById('stepBox').style.display = 'inline-flex';
  document.getElementById('harmCur').textContent = 'h' + h;
  // what each button loads: [key, button id, target ion, harmonic, tooltip name]
  var steps = [['h-1', 'harmDn', ion, h - 1], ['h1', 'harmUp', ion, h + 1]];
  if (p) {
    document.getElementById('qGrp').style.display = 'inline-flex';
    document.getElementById('aGrp').style.display = 'inline-flex';
    document.getElementById('qCur').textContent = p.q + '+';
    document.getElementById('aCur').textContent = (p.iso ? p.iso : '') + p.A + p.sym;
    steps.push(['q-1', 'qDn', p.iso + p.A + p.sym + (p.q - 1) + '+', h],
               ['q1',  'qUp', p.iso + p.A + p.sym + (p.q + 1) + '+', h],
               ['A-1', 'aDn', (p.A - 1) + p.sym + p.q + '+', h],
               ['A1',  'aUp', (p.A + 1) + p.sym + p.q + '+', h]);
  }
  steps.forEach(function(s) { document.getElementById(s[1]).disabled = true; });
  try {
    // one /search-ion request per distinct target ion (exact name matched below)
    var names = steps.map(function(s) { return s[2]; })
                     .filter(function(n, i, a) { return a.indexOf(n) === i && !/(^|[^\\d])0\\+$/.test(n); });
    var res = await Promise.all(names.map(function(n) {
      return fetch('/search-ion', {method: 'POST', headers: {'Content-Type': 'application/json'},
                                   body: JSON.stringify({q: n})})
        .then(function(r) { return r.json().then(function(d) { return {ok: r.ok, d: d}; }); });
    }));
    var byName = {}, err = null;
    names.forEach(function(n, i) {
      if (!res[i].ok) { err = res[i].d.error || 'error'; return; }
      byName[n] = res[i].d;
    });
    if (err) { _harmStatus('✗ ' + err, 'err'); return; }
    var hw = _srchHalfWidthHz();
    steps.forEach(function(s) {
      var b = document.getElementById(s[1]), d = byName[s[2]];
      var l = d && d.hits.filter(function(x) { return x.ion === s[2] && x.h === s[3]; })[0];
      if (!l) { b.title = s[2] + ', h' + s[3] + ' is not among the simulated lines in this file'; return; }
      _stepLines[s[0]] = {file: d.file, line: l};
      b.disabled = false;
      b.title = 'Load ' + s[2] + ', h' + l.h + ': ' +
                Math.round(l.f - hw).toLocaleString('en-GB') + ' – ' +
                Math.round(l.f + hw).toLocaleString('en-GB') + ' Hz' +
                (l.excluded ? '  (below the T½ cut)' : '');
    });
  } catch (e) { _harmStatus('✗ ' + e.message, 'err'); }
}
// kind 'h' (harmonic), 'q' (charge state) or 'A' (isotope); step ±1
function ionStep(kind, step) {
  var t = _stepLines[kind + step];
  if (t) loadAroundIon(t.file, t.line, _harmStatus);
}
function harmStep(step) { ionStep('h', step); }
window.addEventListener('load', _harmInit);

// ── Decay-fit region: slider window or a box dragged on the 2-D heatmap ─────
// Drawing uses the heatmap's normal drag-to-zoom: while drawing, the zoom
// ranges are captured as the box and the view is put back to the sliders.
var _regDrawing = false;
var _regTarget  = 's';     // box being drawn: 's' standard, 'm' mother, 'b' daughter
var _regLoaded  = false;   // saved region read back from the server yet?

// ── Decay Fit popout tabs: "Standard Fit" (exponential) and "Bateman Fit".
//    Each tab has its own region, title, γ and Run. Standard: sliders or one
//    drawn box. Bateman: a drawn MOTHER box and a drawn DAUGHTER box; both are
//    fitted over the time range where the two boxes overlap.
//    Box / element keys: 's' (no suffix), 'b' (suffix "B": Bateman tab and its
//    daughter box), 'm' (suffix "M": mother box). The heatmap shows the open
//    tab's box(es).
var DECAY_TAB = 's';
var _SFX = {s: '', b: 'B', m: 'M'};
function _did(id, key) { return id + _SFX[key || DECAY_TAB]; }
function _d(id, key)   { return document.getElementById(_did(id, key)); }
function decayTab(tab, save) {
  DECAY_TAB = tab === 'b' ? 'b' : 's';
  ['s', 'b'].forEach(function(t) {
    var on = t === DECAY_TAB, T = t.toUpperCase();
    var btn = document.getElementById('dtab' + T);
    btn.classList.toggle('active', on);
    btn.setAttribute('aria-selected', on ? 'true' : 'false');
    document.getElementById('dpane' + T).style.display = on ? '' : 'none';
  });
  if (save !== false) savePref('decayTab', DECAY_TAB === 'b' ? 'bateman' : 'standard');
  _regSync();
}

var _BOX_IDS = ['regT0', 'regT1', 'regF0', 'regF1'];
function _regBoxValues(key) {
  var v = _BOX_IDS.map(function(id) {
    var x = _d(id, key).value.trim();
    return x === '' ? NaN : +x;
  });
  return v.every(isFinite) ? v : null;
}
// standard tab: sliders or drawn box; the Bateman boxes are always drawn boxes
function _regIsBox(key) { return (key || DECAY_TAB) === 's' ? _d('regBox', 's').checked : true; }

function _regSetFields(v, key) {
  _BOX_IDS.forEach(function(id, i) {
    _d(id, key).value = v ? (i < 2 ? (+v[i]).toFixed(3) : (+v[i]).toFixed(1)) : '';
  });
}

function _regSync() {
  var box = _regIsBox('s');
  _BOX_IDS.forEach(function(id) { _d(id, 's').disabled = !box; });
  _d('regDrawBtn', 's').disabled = !box;
  _regDrawShape();
}

// boxes outlined on the plots for the open tab: [{v, color, fill, label}]
function _regShownBoxes() {
  var out = [];
  if (DECAY_TAB === 's') {
    var v = _regBoxValues('s');
    if (_regIsBox('s') && v) out.push({v: v, color: '#f97316', fill: 'rgba(249,115,22,0.08)', label: ''});
  } else {
    var m = _regBoxValues('m'), d = _regBoxValues('b');
    if (m) out.push({v: m, color: '#38bdf8', fill: 'rgba(56,189,248,0.08)', label: 'Mother'});
    if (d) out.push({v: d, color: '#f97316', fill: 'rgba(249,115,22,0.08)', label: 'Daughter'});
  }
  return out;
}

// A box [t0, t1, f0, f1] clipped to the visible window (time/frequency sliders).
// Returns null when there is nothing to draw, else the clipped corners and,
// per edge, whether it is a real box edge (solid) or where the box runs past
// the edge of the plot (cut → dashed line along the plot border).
function _regClipped(v) {
  if (!v) return null;
  var T0 = +document.getElementById('tMin').value, T1 = +document.getElementById('tMax').value;
  var F0 = +document.getElementById('fMin').value, F1 = +document.getElementById('fMax').value;
  var t0 = Math.min(v[0], v[1]), t1 = Math.max(v[0], v[1]);
  var f0 = Math.min(v[2], v[3]), f1 = Math.max(v[2], v[3]);
  var c = {t0: Math.max(t0, T0), t1: Math.min(t1, T1), f0: Math.max(f0, F0), f1: Math.min(f1, F1)};
  if (!(c.t1 > c.t0) || !(c.f1 > c.f0)) return null;       // box entirely outside the view
  // An edge is "cut" only if the box runs past the plotted range by more than
  // half a time frame / frequency bin (the sliders' steps).  A box drawn to the
  // border of the plot can land a hair outside it (zoom pixels, the rounding of
  // the region fields): that is not a cut and stays solid.
  var step = function(id) {
    var s = parseFloat(document.getElementById(id).step);
    return isFinite(s) && s > 0 ? s : 0;
  };
  var dt = step('tMin') / 2, df = step('fMin') / 2;
  // edges: [[ta, fa], [tb, fb], cut?]
  c.edges = [
    [[c.t0, c.f0], [c.t0, c.f1], t0 < T0 - dt],    // left   (start time)
    [[c.t1, c.f0], [c.t1, c.f1], t1 > T1 + dt],    // right  (end time)
    [[c.t0, c.f0], [c.t1, c.f0], f0 < F0 - df],    // bottom (low freq)
    [[c.t0, c.f1], [c.t1, c.f1], f1 > F1 + df]     // top    (high freq)
  ];
  return c;
}

// outline of the fit region(s) on the heatmap (orange; the Bateman mother box
// blue, both labelled); parts cut off by the view are drawn dashed along the
// plot border
function _regDrawShape() {
  if (typeof _heatReady !== 'undefined' && _heatReady) {
    var shapes = [];
    _regShownBoxes().forEach(function(b) {
      var c = _regClipped(b.v);
      if (!c) return;
      var rect = {type: 'rect', xref: 'x', yref: 'y', x0: c.t0, x1: c.t1, y0: c.f0, y1: c.f1,
                  line: {width: 0}, fillcolor: b.fill};
      if (b.label) rect.label = {text: b.label, textposition: 'top left',
                                 font: {color: b.color, size: 12}};
      shapes.push(rect);
      c.edges.forEach(function(e) {
        shapes.push({type: 'line', xref: 'x', yref: 'y',
                     x0: e[0][0], y0: e[0][1], x1: e[1][0], y1: e[1][1],
                     line: {color: b.color, width: e[2] ? 2.5 : 2, dash: e[2] ? 'dash' : 'solid'}});
      });
    });
    Plotly.relayout('heatPlot', {shapes: shapes});
  }
  _regDraw3d();
}

// the same region(s) as orange wireframe boxes in the 3-D plot: rectangles at
// the bottom and top of the z-axis range joined by vertical edges. Real edges
// go in trace 1 (solid), edges where the view cuts a box in trace 2 (dashed).
function _regDraw3d() {
  var gd = document.getElementById('specPlot');
  if (!gd || !gd.data || gd.data.length < 3) return;
  var sol = {x: [], y: [], z: []}, dsh = {x: [], y: [], z: []};
  function seg(o, a, b) {
    o.x.push(a[0], b[0], null); o.y.push(a[1], b[1], null); o.z.push(a[2], b[2], null);
  }
  var zr = (gd.layout && gd.layout.scene && gd.layout.scene.zaxis && gd.layout.scene.zaxis.range) ||
           (gd._fullLayout && gd._fullLayout.scene && gd._fullLayout.scene.zaxis.range);
  _regShownBoxes().forEach(function(b) {
    var c = _regClipped(b.v);
    if (!c || !zr) return;
    // keep every edge a little inside the plot's floor, ceiling and walls: a
    // line lying exactly on one of those planes fights with the grey
    // background drawn there and breaks up into dots
    var IN = 0.005;                              // fraction of each axis span
    function inside(v, r) {
      if (!r) return v;
      var lo = Math.min(r[0], r[1]), hi = Math.max(r[0], r[1]), d = (hi - lo) * IN;
      return Math.min(Math.max(v, lo + d), hi - d);
    }
    var fl = gd._fullLayout && gd._fullLayout.scene;
    var xr = fl && fl.xaxis && fl.xaxis.range, yr = fl && fl.yaxis && fl.yaxis.range;
    var P = function(t, f, z) { return [inside(t, xr), inside(f, yr), z]; };
    var dz = (Math.max(zr[0], zr[1]) - Math.min(zr[0], zr[1])) * IN;
    var z0 = Math.min(zr[0], zr[1]) + dz, z1 = Math.max(zr[0], zr[1]) - dz;
    c.edges.forEach(function(e) {               // bottom and top outlines
      var o = e[2] ? dsh : sol;
      seg(o, P(e[0][0], e[0][1], z0), P(e[1][0], e[1][1], z0));
      seg(o, P(e[0][0], e[0][1], z1), P(e[1][0], e[1][1], z1));
    });
    // vertical edges at the corners: dashed if the corner lies on a cut edge
    var L = c.edges[0][2], R = c.edges[1][2], B = c.edges[2][2], T = c.edges[3][2];
    [[c.t0, c.f0, L || B], [c.t1, c.f0, R || B], [c.t1, c.f1, R || T], [c.t0, c.f1, L || T]]
      .forEach(function(k) { seg(k[2] ? dsh : sol, P(k[0], k[1], z0), P(k[0], k[1], z1)); });
  });
  Plotly.restyle('specPlot', {x: [sol.x, dsh.x], y: [sol.y, dsh.y], z: [sol.z, dsh.z]}, [1, 2]);
}

function setRegionMode() {
  savePref('decayRegion', _regIsBox('s') ? 'box' : 'sliders');
  _regSync();
}
function regFieldsChanged() { _regDrawShape(); saveSession(); }
function clearRegionBox(key) { _regSetFields(null, key || DECAY_TAB); _regDrawShape(); saveSession(); }

function _initRegion(v, vB, vM) {
  _regSetFields(v, 's');
  _regSetFields(vB, 'b');
  _regSetFields(vM, 'm');
  _regSync();
  var gd = document.getElementById('heatPlot');
  if (gd && gd.on) gd.on('plotly_relayout', _regOnRelayout);
  // the page is built once per server start; re-read the saved state so a
  // browser refresh keeps the mode and the boxes
  fetch('/get-region').then(function(r) { return r.json(); }).then(function(d) {
    _d(d.mode === 'box' ? 'regBox' : 'regSliders', 's').checked = true;
    if (d.box)  _regSetFields(d.box, 's');
    if (d.boxB) _regSetFields(d.boxB, 'b');
    if (d.boxM) _regSetFields(d.boxM, 'm');
    _d('decayTitleBox', 's').value = d.title || '';
    _d('decayTitleBox', 'b').value = d.titleB || '';
    _regSync();
    _regLoaded = true;
  }).catch(function() { _regLoaded = true; });
}

var _REG_WHAT = {s: 'the decay-fit region', m: 'the MOTHER region', b: 'the DAUGHTER region'};
// key: which box to draw ('s', 'm' or 'b'); default: the open tab's box
function startRegionDraw(key) {
  _regTarget = key || DECAY_TAB;
  if (_regTarget === 's') { _d('regBox', 's').checked = true; setRegionMode(); }
  modalClose('decayModal');
  _regDrawing = true;
  Plotly.relayout('heatPlot', {dragmode: 'zoom'});
  var ban = document.getElementById('regBanner');
  ban.textContent = 'Drag a box on the 2-D plot to set ' + _REG_WHAT[_regTarget] + ' · Esc to cancel';
  ban.style.display = 'block';
  document.getElementById('heatDiv').classList.add('drawing');
}
function _regStopDrawing() {
  _regDrawing = false;
  document.getElementById('regBanner').style.display = 'none';
  document.getElementById('heatDiv').classList.remove('drawing');
}
function cancelRegionDraw() {
  if (!_regDrawing) return;
  _regStopDrawing();
  modalOpen('decayModal', _did('regDrawBtn', _regTarget));
}

function _regOnRelayout(ev) {
  if (!_regDrawing || !ev) return;
  var hasX = ev['xaxis.range[0]'] !== undefined, hasY = ev['yaxis.range[0]'] !== undefined;
  if (!hasX && !hasY) return;              // not a zoom on the heatmap axes
  var gd = document.getElementById('heatPlot');
  var xr = hasX ? [ev['xaxis.range[0]'], ev['xaxis.range[1]']] : gd._fullLayout.xaxis.range.slice();
  var yr = hasY ? [ev['yaxis.range[0]'], ev['yaxis.range[1]']] : gd._fullLayout.yaxis.range.slice();
  _regStopDrawing();
  _regSetFields([Math.min(xr[0], xr[1]), Math.max(xr[0], xr[1]),
                 Math.min(yr[0], yr[1]), Math.max(yr[0], yr[1])], _regTarget);
  update();                                // put the view back to the slider window
  _regDrawShape();
  saveSession();
  modalOpen('decayModal', _did('regT0', _regTarget));
}

// a drawn box as {tMin, tMax, fMin, fMax} (throws if missing or empty)
function _boxWindow(key, what) {
  var v = _regBoxValues(key);
  if (!v) throw new Error('draw ' + what + ' first');
  var w = {tMin: Math.min(v[0], v[1]), tMax: Math.max(v[0], v[1]),
           fMin: Math.min(v[2], v[3]), fMax: Math.max(v[2], v[3])};
  if (!(w.tMax > w.tMin) || !(w.fMax > w.fMin)) throw new Error(what + ' has zero size');
  return w;
}

// window used by the fit of the open tab.
//   Standard: sliders, or the drawn box.
//   Bateman: daughter box frequencies (fMin/fMax), mother box frequencies
//   (pMin/pMax), time = where the two boxes overlap.
function decayWindow() {
  if (DECAY_TAB === 'b') {
    var m = _boxWindow('m', 'the mother box'), d = _boxWindow('b', 'the daughter box');
    var t0 = Math.max(m.tMin, d.tMin), t1 = Math.min(m.tMax, d.tMax);
    if (!(t1 > t0)) throw new Error('the mother and daughter boxes do not overlap in time');
    return {tMin: t0, tMax: t1, fMin: d.fMin, fMax: d.fMax, pMin: m.fMin, pMax: m.fMax};
  }
  if (!_regIsBox('s')) {
    return {fMin: +document.getElementById('fMin').value, fMax: +document.getElementById('fMax').value,
            tMin: +document.getElementById('tMin').value, tMax: +document.getElementById('tMax').value};
  }
  return _boxWindow('s', 'a region (or choose "sliders")');
}

// ── Decay-fit title: pick from simulated ions inside the fit region ─────────
//    (Bateman tab: ions in the mother box, then in the daughter box)
function _sugStatus(msg, kind) {
  var s = _d('decaySugStatus');
  s.textContent = msg;
  s.style.color = kind === 'err' ? '#f87171' : kind === 'ok' ? '#4ade80' : '#fbbf24';
}
async function suggestDecayTitle() {
  var list = _d('decaySugList');
  var tbox = _d('decayTitleBox');           // this tab's title box
  list.innerHTML = '';
  var w;
  try { w = decayWindow(); } catch (e) { _sugStatus('✗ ' + e.message, 'err'); return; }
  var ranges = DECAY_TAB === 'b'
    ? [['Mother Box', w.pMin, w.pMax], ['Daughter Box', w.fMin, w.fMax]]
    : [['', w.fMin, w.fMax]];
  _sugStatus('Looking…');
  try {
    var found = 0;
    for (var k = 0; k < ranges.length; k++) {
      var r = await fetch('/region-ions', {method: 'POST', headers: {'Content-Type': 'application/json'},
                                           body: JSON.stringify({fMin: ranges[k][1], fMax: ranges[k][2]})});
      var d = await r.json();
      if (!r.ok) { _sugStatus('✗ ' + (d.error || 'error'), 'err'); return; }
      if (!d.hits.length) continue;
      found += d.n;
      var hdr = document.createElement('div'); hdr.className = 'hdr';
      [ranges[k][0] ? ranges[k][0] + ': Ion' : 'Ion', 'Harmonic', 'f_sim (Hz)', 'From Centre'].forEach(function(t) {
        var c = document.createElement('span'); c.textContent = t; hdr.appendChild(c);
      });
      list.appendChild(hdr);
      d.hits.forEach(function(l) {
        var it = document.createElement('div');
        it.className = 'item' + (l.excluded ? ' ex' : ''); it.tabIndex = 0;
        var txt = l.ion + ', h' + l.h;
        it.title = 'Use "' + txt + '" as the title' + (l.excluded ? '  (below the T½ cut)' : '');
        [l.ion + (l.is_ref ? '  (ref)' : ''), 'h' + l.h,
         l.f.toLocaleString('en-GB', {maximumFractionDigits: 1}),
         (l.df >= 0 ? '+' : '−') + Math.abs(l.df).toLocaleString('en-GB', {maximumFractionDigits: 0}) + ' Hz'
        ].forEach(function(t) { var c = document.createElement('span'); c.textContent = t; it.appendChild(c); });
        var pick = function() {
          tbox.value = txt;
          list.innerHTML = ''; _sugStatus('');
          saveSession();
          tbox.focus();
        };
        it.onclick = pick;
        it.onkeydown = function(e) { if (e.key === 'Enter') pick(); };
        list.appendChild(it);
      });
    }
    if (!found) { _sugStatus('No simulated ions in ' + (ranges.length > 1 ? 'these boxes' : 'this region'), 'err'); return; }
    _sugStatus(found + ' in ' + (ranges.length > 1 ? 'the boxes' : 'region') + ', nearest centre first, click one', 'ok');
  } catch (e) { _sugStatus('✗ ' + e.message, 'err'); }
}

// ── Quit: stop the Python server and close the tab ──────────────────────────
// First click arms the button (avoids quitting by accident); second click quits.
var _quitTimer = null;
async function quitApp() {
  var b = document.getElementById('quitBtn');
  if (!b.classList.contains('armed')) {
    b.classList.add('armed'); b.textContent = 'Click again to quit';
    _quitTimer = setTimeout(function() { b.classList.remove('armed'); b.textContent = 'Quit'; }, 3000);
    return;
  }
  clearTimeout(_quitTimer);
  try {   // save the current view straight away (saveSession() is delayed)
    await fetch('/save-session', {method: 'POST', headers: {'Content-Type': 'application/json'},
                                  body: JSON.stringify(_sessionState())});
  } catch (e) {}
  try { await fetch('/quit', {method: 'POST'}); } catch (e) {}
  window.close();          // works only for tabs a script opened; otherwise...
  setTimeout(function() {  // ...leave a clear message in the tab instead
    document.body.innerHTML =
      '<div style="font-family:system-ui,sans-serif;color:#e5e7eb;padding:40px;font-size:16px;">' +
      'plot_spectrogram.py has stopped. You can close this tab.</div>';
  }, 300);
}

// ── Popout dialogs (simulated ions, decay fit) ──────────────────────────────
var _modalLastFocus = null;
function modalOpen(id, focusId) {
  document.querySelectorAll('.sim-modal.open').forEach(function(m) { m.classList.remove('open'); });
  _modalLastFocus = document.activeElement;
  document.getElementById(id).classList.add('open');
  var f = focusId && document.getElementById(focusId);
  if (f) f.focus();
}
function modalClose(id) {
  document.getElementById(id).classList.remove('open');
  if (_modalLastFocus) _modalLastFocus.focus();
}
function simOpen()  { modalOpen('simModal', 'sim_ref_name'); corrRefresh(); }
function simClose() { modalClose('simModal'); }
document.addEventListener('keydown', function(e) {
  if (e.key !== 'Escape') return;
  if (_regDrawing) { cancelRegionDraw(); return; }
  var m = document.querySelector('.sim-modal.open');
  if (m) modalClose(m.id);
});

// ── Collapsible whole-file 1-D plot (open/closed state saved in prefs) ──────
function _fullOpen() { return document.getElementById('fullPlot').style.display !== 'none'; }
function _setFull(open) {
  document.getElementById('fullPlot').style.display = open ? '' : 'none';
  document.getElementById('fullArrow').textContent = open ? '▾' : '▸';
  document.getElementById('fullToggle').setAttribute('aria-expanded', open ? 'true' : 'false');
}
function toggleFull() {
  var open = !_fullOpen();
  _setFull(open);
  savePref('fullOpen', open ? '1' : '0');
  if (open) {
    _renderFull();
    document.getElementById('fullDiv').scrollIntoView({behavior: 'smooth', block: 'end'});
  }
}

// ── 2-D / 3-D plots fill the window height above the "Whole-file spectrum"
//    bar (the bar sits at the bottom of the window; the expanded 1-D plot
//    is below it and the page scrolls) ─────────────────────────────────────
var _fitTimer = null;
// height that makes the 2-D/3-D plots fill the window above the bottom bar
function plotHeight() {
  var plots = document.getElementById('plots');
  var bar   = document.getElementById('fullToggle');
  if (!plots || !bar) return 680;
  var top   = plots.getBoundingClientRect().top + window.scrollY;
  var boxV  = 2 * 6 + 2;                                  // panel padding + border
  var below = 8 + bar.offsetHeight + 4 + 8;               // gap, bar, gap, body padding
  return Math.max(420, Math.floor(window.innerHeight - top - boxV - below));
}
function fitPlots() {
  if (typeof Plotly === 'undefined') return;
  var h = plotHeight();
  var setH = function(id, hh) {
    var gd = document.getElementById(id);
    if (!gd || !gd.layout) return null;
    gd.style.height = hh + 'px';        // the exported 3-D div has a fixed 680px style
    var wrap = gd.parentElement;        // ...and so does plotly's wrapper around it
    if (wrap && wrap.id !== 'heatDiv' && wrap.id !== 'surfDiv') wrap.style.height = hh + 'px';
    return gd.layout.height !== hh ? Plotly.relayout(id, {height: hh}) : null;
  };
  var jobs = ['heatPlot', 'specPlot'].map(function(id) { return setH(id, h); }).filter(Boolean);
  Promise.all(jobs).then(function() {
    // trim any remaining overflow so the bar sits exactly at the bottom
    var over = document.documentElement.scrollHeight - window.innerHeight
               - (_fullOpen() ? document.getElementById('fullPlot').offsetHeight : 0);
    if (over > 0 && h - over >= 420) {
      setH('heatPlot', h - over);
      setH('specPlot', h - over);
    }
    _renderSim();     // hover-point spacing depends on the plot width
  });
}
window.addEventListener('resize', function() {
  clearTimeout(_fitTimer); _fitTimer = setTimeout(fitPlots, 150);
});

// Built the same way as Theia's spectrum figure so it stays responsive:
// the spectrum is one WebGL trace with hover switched off, and each line group
// is ONE WebGL trace (20 points per line, so tooltips work along it).
function _renderFull() {
  if (!FULL || !_fullOpen()) return;   // drawn when expanded (needs real width)
  var useDb = document.getElementById('dbCheck').checked;
  if (useDb && !FULL.ydb) {
    FULL.ydb = FULL.y.map(function(v) { return v > 0 ? 10 * Math.log10(v) : NaN; });
  }
  var y = useDb ? FULL.ydb : FULL.y;
  var key = useDb ? 'db' : 'lin';
  if (!FULL.range || FULL.range.key !== key) {
    var lo = Infinity, hi = -Infinity;
    for (var i = 0; i < y.length; i++) {
      var v = y[i];
      if (!isFinite(v)) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (!isFinite(lo)) { lo = 0; hi = 1; }
    var pad = (hi - lo) * 0.05 || 1;
    FULL.range = {key: key, lo: lo - pad, hi: hi + pad};
  }
  var y0 = FULL.range.lo, y1 = FULL.range.hi;
  var pLabel = useDb ? 'Summed Power (dB)' : 'Summed Power (linear)';

  // vertical simulated lines over the whole file (sliders ignored)
  // a point every few pixels up each line so its tooltip works anywhere on it
  var fp = document.getElementById('fullPlot');
  var hPx = (fp && fp._fullLayout && fp._fullLayout.yaxis) ? fp._fullLayout.yaxis._length : 300;
  var N = Math.max(20, Math.ceil(hPx / 6)), ys = [];
  for (var k = 0; k < N; k++) ys.push(y0 + (y1 - y0) * k / (N - 1));
  var groups = [{x: [], y: [], h: []}, {x: [], y: [], h: []}, {x: [], y: [], h: []}];
  if (document.getElementById('sim_show').checked && SIM) {
    var showEx = document.getElementById('sim_show_excluded').checked;
    SIM.lines.forEach(function(l) {
      if (l.excluded && !l.is_ref && !showEx) return;
      var g = groups[l.is_ref ? 2 : (l.excluded ? 1 : 0)];
      for (var k = 0; k < N; k++) { g.x.push(l.f); g.y.push(ys[k]); g.h.push(l.label); }
      g.x.push(null); g.y.push(null); g.h.push(null);
    });
  }
  var styles = [
    {name: 'Simulated',          line: {color: 'rgba(229,231,235,0.7)', width: 1, dash: 'dash'}},
    {name: 'Simulated (T½ cut)', line: {color: 'rgba(156,163,175,0.5)', width: 1, dash: 'dot'}},
    {name: 'Reference',          line: {color: 'rgba(255,255,255,0.95)', width: 1.5}}
  ];
  var traces = [{
    type: 'scattergl', mode: 'lines', x: FULL.f, y: y, name: pLabel, showlegend: false,
    hoverinfo: 'skip', line: {shape: 'hvh', color: 'rgba(56,189,248,0.95)', width: 1}
  }];
  groups.forEach(function(g, i) {
    traces.push({type: 'scattergl', mode: 'lines', x: g.x, y: g.y, hovertext: g.h,
                 hoverinfo: 'text', connectgaps: false, showlegend: false,
                 name: styles[i].name, line: styles[i].line});
  });
  Plotly.react('fullPlot', traces, {
    template: {},
    title: {text: 'Whole File: All ' + FULL.n + ' Frames, ' +
                  Math.round(FULL.f[0]).toLocaleString('en-GB') + ' – ' +
                  Math.round(FULL.f[FULL.f.length - 1]).toLocaleString('en-GB') + ' Hz',
            x: 0.5, xanchor: 'center', font: {size: 18, color: '#e5e7eb'},
            // below the Plotly toolbar (top-right) so it is never covered
            yref: 'container', y: 1, yanchor: 'top', pad: {t: 34}},
    xaxis: {title: {text: 'Frequency (Hz)'}, tickformat: ',d',
            color: '#9ca3af', gridcolor: '#374151', zerolinecolor: '#374151'},
    yaxis: {title: {text: pLabel}, range: [y0, y1],
            color: '#9ca3af', gridcolor: '#374151', zerolinecolor: '#374151'},
    paper_bgcolor: '#111827', plot_bgcolor: '#1f2937', font: {color: '#e5e7eb'},
    hovermode: 'x', dragmode: 'pan', uirevision: 'full',   // as Theia
    margin: {l: 90, r: 20, t: 70, b: 50}, height: 380
  }, {responsive: true, scrollZoom: true});
}
"""


# ---------------------------------------------------------------------------
# "Save to Log Book" popout (results log book: core/results_logbook.py)
# ---------------------------------------------------------------------------
LOGBOOK_CSS = """
  .sim-head #lbTitle { font-size: 13px; font-weight: 600; text-transform: uppercase;
                       letter-spacing: .06em; color: #f97316; }
  .lb-dialog { width: min(860px, calc(100vw - 32px)); }
  .lb-dialog label.f { display: block; font-size: 11px; color: #9ca3af; text-transform: uppercase;
                       letter-spacing: .06em; margin-bottom: 3px; }
  .lb-dialog input[type=text], .lb-dialog textarea { background: #374151; border: 1px solid #4b5563;
                       border-radius: 4px; color: #e5e7eb; padding: 4px 7px; font-size: 13px;
                       width: 100%; box-sizing: border-box; font-family: inherit; }
  .lb-dialog input[type=text]:focus, .lb-dialog textarea:focus { outline: none; border-color: #f97316; }
  .lb-dialog textarea { min-height: 70px; }
  .lb-top { display: grid; grid-template-columns: 1fr 2fr; gap: 12px; }
  .lb-vals-box { max-height: 340px; overflow-y: auto; border: 1px solid #374151; border-radius: 4px; }
  table.lb-vals { border-collapse: collapse; width: 100%; font-size: 12px; }
  table.lb-vals thead th { position: sticky; top: 0; z-index: 1; }
  table.lb-vals th { background: #374151; color: #d1d5db; font-weight: 600; text-align: left;
                     padding: 3px 6px; }
  table.lb-vals td { padding: 2px 4px; border-bottom: 1px solid #374151; }
  table.lb-vals td.lbl { color: #e5e7eb; white-space: nowrap; }
  table.lb-vals input[type=text] { font-family: monospace; font-size: 12px; padding: 2px 5px; }
  .lb-plots { display: flex; flex-wrap: wrap; gap: 12px; }
  .lb-plots label { display: flex; flex-direction: column; gap: 4px; font-size: 12px; color: #d1d5db;
                    cursor: pointer; }
  .lb-plots img { height: 110px; border: 1px solid #4b5563; border-radius: 4px; background: #111827; }
  .lb-dialog input[type=checkbox] { accent-color: #f97316; }
  .lb-foot { display: flex; gap: 10px; align-items: center; }
  .lb-foot button.primary { background: #f97316; color: #111827; font-weight: 700; }
  #lbStatus { font-size: 12px; }
"""

LOGBOOK_HTML = """
  <div id="lbModal" class="sim-modal" onclick="if(event.target===this) modalClose('lbModal')">
   <div class="sim-dialog lb-dialog" role="dialog" aria-modal="true" aria-labelledby="lbTitle">
    <div class="sim-head">
      <span id="lbTitle">Save to Log Book</span>
      <button class="sim-x" onclick="modalClose('lbModal')" title="Close (Esc)">×</button>
    </div>
    <div class="sim-body">
      <div class="lb-top">
        <div><label class="f" for="lbIon">Ion (tab)</label>
          <input type="text" id="lbIon" list="lbIons" autocomplete="off" placeholder="e.g. 198Ir, m186Hf72+">
          <datalist id="lbIons"></datalist></div>
        <div><label class="f" for="lbEntryTitle">Title</label>
          <input type="text" id="lbEntryTitle" autocomplete="off"></div>
      </div>
      <div><label class="f">Values to Send (edit if needed)</label>
        <div class="lb-vals-box"><table class="lb-vals"><thead><tr><th></th><th>Quantity</th><th>Value</th><th>± Error</th>
          <th>Unit</th></tr></thead><tbody id="lbVals"></tbody></table></div>
        <button onclick="lbAddRow()" style="margin-top:6px;">+ Add Value</button></div>
      <div><label class="f">Plots (as shown now)</label><div id="lbPlots" class="lb-plots"></div></div>
      <div><label class="f" for="lbNotes">Notes</label><textarea id="lbNotes"></textarea></div>
      <div class="lb-foot">
        <button class="primary" id="lbSaveBtn" onclick="lbSave()">Save</button>
        <button onclick="modalClose('lbModal')">Cancel</button>
        <span id="lbStatus"></span>
      </div>
    </div>
   </div>
  </div>
"""

LOGBOOK_JS = """
// ── Save to log book (results log book, one tab per ion) ────────────────────
var LB_PLOTS = [];       // [{png: dataURL, caption}] snapshots taken when the popout opened
var LB_GROUP = '';       // group of the rows added next ('settings': folded in the log book)
function _lbStatus(msg, kind) {
  var s = document.getElementById('lbStatus');
  s.textContent = msg;
  s.style.color = kind === 'err' ? '#f87171' : kind === 'ok' ? '#4ade80' : '#fbbf24';
}
function _lbCell(tr, value, cls) {
  var td = document.createElement('td');
  var inp = document.createElement('input'); inp.type = 'text'; inp.value = value == null ? '' : value;
  if (cls) inp.className = cls;
  td.appendChild(inp); tr.appendChild(td); return inp;
}
function lbAddRow(item) {
  item = item || {label: '', value: '', checked: true, editable: true};
  var tr = document.createElement('tr');
  tr.dataset.group = item.group || LB_GROUP;
  var td0 = document.createElement('td');
  var c = document.createElement('input'); c.type = 'checkbox'; c.checked = item.checked !== false;
  td0.appendChild(c); tr.appendChild(td0);
  if (item.editable) { _lbCell(tr, item.label, 'lbl'); }
  else { var tl = document.createElement('td'); tl.className = 'lbl'; tl.textContent = item.label;
         tl.dataset.label = item.label; tr.appendChild(tl); }
  _lbCell(tr, item.value); _lbCell(tr, item.error); _lbCell(tr, item.unit);
  document.getElementById('lbVals').appendChild(tr);
}
function _lbRows() {
  var out = [];
  document.querySelectorAll('#lbVals tr').forEach(function(tr) {
    var tds = tr.children;
    if (!tds[0].firstChild.checked) return;
    var lab = tds[1].dataset.label !== undefined ? tds[1].dataset.label : tds[1].firstChild.value;
    out.push({label: lab, value: tds[2].firstChild.value, error: tds[3].firstChild.value,
              unit: tds[4].firstChild.value, group: tr.dataset.group || ''});
  });
  return out;
}
async function _lbSnap(id, caption) {
  var gd = document.getElementById(id);
  if (!gd || !gd.data || gd.offsetParent === null || !gd.offsetWidth) return null;
  try {
    var png = await Plotly.toImage(gd, {format: 'png', width: gd.offsetWidth, height: gd.offsetHeight,
                                        scale: 2});
    return {png: png, caption: caption};
  } catch (e) { return null; }
}
async function lbOpen() {
  modalOpen('lbModal', 'lbIon');
  _lbStatus('');
  document.getElementById('lbSaveBtn').disabled = false;
  // ion + harmonic from the "Search Ions" label ("m186Hf72+, h213"), else the decay-fit title
  var lab = (typeof CUR_ION_LABEL === 'string' && CUR_ION_LABEL) ? CUR_ION_LABEL
            : _d('decayTitleBox').value;
  var parts = String(lab || '').split(',');
  var ion = parts[0].trim().split(' ')[0];
  var hm = null;
  parts.concat(String(lab || '').split(' ')).forEach(function(p) {
    p = p.trim(); if (!hm && p.length > 1 && p[0] === 'h' && !isNaN(+p.slice(1))) hm = +p.slice(1);
  });
  document.getElementById('lbIon').value = _parseIon(ion) ? ion : '';
  document.getElementById('lbEntryTitle').value = document.getElementById('plotTitle').value;
  document.getElementById('lbNotes').value = '';
  var tb = document.getElementById('lbVals'); tb.innerHTML = '';
  var fMin = +document.getElementById('fMin').value, fMax = +document.getElementById('fMax').value;
  var tMin = +document.getElementById('tMin').value, tMax = +document.getElementById('tMax').value;
  LB_GROUP = '';
  if (hm) lbAddRow({label: 'Harmonic h', value: hm});
  LB_GROUP = 'settings';   // everything below is a setting
  lbAddRow({label: 'Frequency Window', value: fMin.toFixed(1) + ' – ' + fMax.toFixed(1), unit: 'Hz'});
  lbAddRow({label: 'Time Window', value: tMin.toFixed(3) + ' – ' + tMax.toFixed(3), unit: 's'});
  // every setting of the page, all ticked; options not used say "off"
  lbAddRow({label: 'Time Average', value: APPLIED_N > 1 ? APPLIED_N : 'off', unit: APPLIED_N > 1 ? 'frames' : ''});
  var ionLab = (typeof CUR_ION_LABEL === 'string' && CUR_ION_LABEL) ? CUR_ION_LABEL : 'none';
  lbAddRow({label: 'Ion Loaded (Search Ions)', value: ionLab});
  try {   // file, frequency crop, baseline, PID settings: from the server
    var rs = await fetch('/logbook-settings'); var ds = await rs.json();
    (ds.rows || []).forEach(function(r) { lbAddRow(r); });
  } catch (e) {}
  var chk = function(id) { var el = document.getElementById(id); return el && el.checked ? 'on' : 'off'; };
  var val = function(id) { var el = document.getElementById(id); return el ? String(el.value).trim() : ''; };
  // decay-fit settings of the open tab of the Decay Fit popout
  var bat = DECAY_TAB === 'b';
  var dval = function(id) { return val(_did(id)); };
  var boxTxt = function(rb) {
    return rb ? ('t ' + (+rb[0]).toFixed(3) + ' – ' + (+rb[1]).toFixed(3) + ' s, f ' +
                 (+rb[2]).toFixed(1) + ' – ' + (+rb[3]).toFixed(1) + ' Hz') : 'none';
  };
  if (bat) {
    lbAddRow({label: 'Decay-Fit Region', value: 'mother + daughter boxes (time overlap)'});
    lbAddRow({label: 'Mother Box', value: boxTxt(_regBoxValues('m'))});
    lbAddRow({label: 'Daughter Box', value: boxTxt(_regBoxValues('b'))});
  } else {
    lbAddRow({label: 'Decay-Fit Region', value: _regIsBox('s') ? 'drawn box' : 'sliders'});
    lbAddRow({label: 'Drawn Box', value: boxTxt(_regBoxValues('s'))});
  }
  lbAddRow({label: 'Decay-Fit Model', value: bat ? 'Bateman' : 'exponential decay'});
  lbAddRow({label: 'Offset', value: bat ? 'with offset (+ C)'
            : (chk('noOffset') === 'on' ? 'no offset' : 'with offset (+ C)')});
  lbAddRow({label: 'Fixed T½ Isomer (c.o.m.)', value: (bat && val('hlIso')) || 'off',
            unit: (bat && val('hlIso')) ? 's' : ''});
  lbAddRow({label: 'Fixed T½ Ground State (c.o.m.)', value: (bat && val('hlGs')) || 'off',
            unit: (bat && val('hlGs')) ? 's' : ''});
  lbAddRow({label: 'Lorentz Factor γ', value: dval('gammaVal') || 'off'});
  lbAddRow({label: 'Decay-Fit Title', value: dval('decayTitleBox') || 'automatic'});
  lbAddRow({label: 'Log Panel (data − C)', value: bat ? 'off' : chk('logPlotChk')});
  lbAddRow({label: 'Residual Panels', value: bat ? 'off' : chk('residChk')});
  lbAddRow({label: 'Power Scale', value: chk('dbCheck') === 'on' ? 'dB' : 'linear'});
  lbAddRow({label: 'Colour Scheme', value: (typeof CUR_CS === 'string') ? CUR_CS : val('colorscaleSel')});
  lbAddRow({label: '3-D Style', value: chk('surface3d') === 'on' ? 'surface' : 'bars'});
  lbAddRow({label: '1-D Overlay', value: chk('show1d')});
  lbAddRow({label: 'Plot Title', value: val('plotTitle') || 'none'});
  LB_GROUP = '';           // rows added with "+ Add Value" are results
  // existing tabs for the ion box
  try {
    var r = await fetch('/logbook-ions'); var d = await r.json();
    var dl = document.getElementById('lbIons'); dl.innerHTML = '';
    (d.ions || []).forEach(function(n) { var o = document.createElement('option'); o.value = n; dl.appendChild(o); });
  } catch (e) {}
  // plots as they are on screen now
  var box = document.getElementById('lbPlots'); box.innerHTML = 'Taking snapshots…';
  LB_PLOTS = (await Promise.all([_lbSnap('heatPlot', '2-D spectrogram'), _lbSnap('specPlot', '3-D view'),
                                 _lbSnap('fullPlot', 'Whole-file spectrum')])).filter(Boolean);
  box.innerHTML = '';
  if (!LB_PLOTS.length) box.textContent = 'No plot to send.';
  LB_PLOTS.forEach(function(p, i) {
    var l = document.createElement('label');
    var c = document.createElement('input'); c.type = 'checkbox'; c.dataset.i = i;
    c.checked = p.caption !== 'Whole-file spectrum';   // 2-D and 3-D ticked
    var img = document.createElement('img'); img.src = p.png;
    var cap = document.createElement('span'); cap.textContent = p.caption;
    var row = document.createElement('span'); row.appendChild(c); row.appendChild(cap);
    l.appendChild(img); l.appendChild(row); box.appendChild(l);
  });
}
async function lbSave() {
  var ion = document.getElementById('lbIon').value.trim();
  if (!ion) { _lbStatus('✗ Choose or type an ion', 'err'); return; }
  var plots = [];
  document.querySelectorAll('#lbPlots input[type=checkbox]').forEach(function(c) {
    if (c.checked) plots.push(LB_PLOTS[+c.dataset.i]);
  });
  var body = {ion: ion, title: document.getElementById('lbEntryTitle').value,
              notes: document.getElementById('lbNotes').value, values: _lbRows(), plots: plots,
              ionLabel: (typeof CUR_ION_LABEL === 'string') ? CUR_ION_LABEL : ''};
  _lbStatus('Saving…');
  try {
    var r = await fetch('/logbook-save', {method: 'POST', headers: {'Content-Type': 'application/json'},
                                          body: JSON.stringify(body)});
    var d = await r.json();
    if (!r.ok) { _lbStatus('✗ ' + (d.error || 'error'), 'err'); return; }
    _lbStatus('✓ Saved to the log book: ' + d.ion, 'ok');
    document.getElementById('lbSaveBtn').disabled = true;
    setTimeout(function() { modalClose('lbModal'); }, 1200);
  } catch (e) { _lbStatus('✗ ' + e.message, 'err'); }
}
"""


def logbook_settings_rows(state):
    """Settings the server knows for the popout: file, frequency crop, baseline,
    PID settings (all ticked; options not used say "off")."""
    from core import results_logbook
    rows = [{"label": "File", "value": state.get("npz_path") or "none"}]   # tagged "settings" below
    la = state.get("load_args") or {}
    lo, hi = la.get("fmin"), la.get("fmax")
    rows.append({"label": "Frequency Crop (loaded)",
                 "value": "full file" if lo is None and hi is None
                 else f"{'start' if lo is None else f'{float(lo):.1f}'} – {'end' if hi is None else f'{float(hi):.1f}'}",
                 "unit": "" if lo is None and hi is None else "Hz"})
    bl = state.get("baseline")
    if bl and bl.get("on"):
        par = ", ".join(f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}"
                        for k, v in (bl.get("params") or {}).items())
        win = bl.get("window") or []
        rows.append({"label": "Baseline Removal", "value": baseline.METHOD.upper()
                     + (f", window {float(win[0]):.1f} – {float(win[1]):.1f} Hz"
                        if len(win) == 2 and None not in win else "")
                     + (f" ({par})" if par else "")})
    else:
        rows.append({"label": "Baseline Removal", "value": "off"})
    sim = sim_prefs()
    rows += results_logbook.settings_rows(sim, list(SIM_DEFAULTS))
    return [dict(r, group="settings") for r in rows]


def logbook_save(body, npz_path):
    """POST /logbook-save: add the entry from the popout; returns the ion key."""
    from core import results_logbook
    plots = [p for p in (body.get("plots") or []) if isinstance(p, dict) and p.get("png")]
    context = {"tool": "Spectrogram", "file": npz_path or ""}
    if body.get("ionLabel"):
        context["ion loaded (Search ions)"] = body["ionLabel"]
    _, key = results_logbook.add_entry(body.get("ion", ""), "Spectrogram", title=body.get("title", ""),
                                       notes=body.get("notes", ""), values=body.get("values") or [],
                                       plots=plots, context=context)
    print(f"Results log book: saved to {key} ({results_logbook.logbook_path()})")
    return key


def loader_parts(file_path=None, fmin=None, fmax=None):
    """Return (html, js) for the loader row, pre-filled."""
    import json
    prefs = read_prefs()
    if file_path:
        start_dir = os.path.dirname(os.path.abspath(file_path))
    else:
        start_dir = prefs.get("lastDir") or paths.spectrograms()
    esc = lambda v: str(v).replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")
    html = (LOADER_HTML
            .replace("__PATH__", esc(os.path.abspath(file_path)) if file_path else "")
            .replace("__FMIN__", "" if fmin is None else esc(repr(float(fmin))))
            .replace("__FMAX__", "" if fmax is None else esc(repr(float(fmax)))))
    js = LOADER_JS.replace("__INIT__", json.dumps({"dir": start_dir}).replace("</", "<\\/"))
    # every page has the loader: close the tab when this program stops
    js += tabwatch.watch_js("plot_spectrogram.py", dark=True)
    js += trackpad.js()        # touchpad zoom in the plots (core/trackpad.py)
    return html, js


def build_landing_page():
    html, js = loader_parts()
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
{icon.LINK_TAG}
<title>Asphodel — Open Spectrogram</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; padding: 8px; background: #111827; color: #e5e7eb;
         font-family: system-ui, sans-serif; font-size: 13px; }}
  #controls {{ display: grid; grid-template-columns: 1fr; gap: 12px;
               background: #1f2937; border-radius: 8px; padding: 12px 16px; }}
  .hint {{ color: #9ca3af; }}
{LOADER_CSS}
</style>
</head>
<body>
<div id="controls">
{html}
  <div class="hint">Choose an .npz spectrogram (browse, or type a path), optionally set a
  frequency range, and click <b>Load</b>. Leave the range blank to load the full file.</div>
</div>
<script>
{js}
ldToggle(true);
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Interactive browser plot (2-D heatmap + 3-D surface, side by side)
# ---------------------------------------------------------------------------

def _box_row(key, what, title):
    """One drawn-box row of the Decay Fit popout: Draw button, t and f fields, Clear."""
    S = {"s": "", "b": "B", "m": "M"}[key]
    arg = "" if key == "s" else f"'{key}'"
    return f"""    <div class="opt-grp" title="{title}">
      <span>{what}</span>
      <button id="regDrawBtn{S}" onclick="startRegionDraw({arg})"
              title="Close this dialog and drag the {what.lower()} on the 2-D plot">Draw</button>
      <span>t (s)</span>
      <input type="number" id="regT0{S}" step="any" style="width:72px;" oninput="regFieldsChanged()">
      <label>–</label>
      <input type="number" id="regT1{S}" step="any" style="width:72px;" oninput="regFieldsChanged()">
      <span>f (Hz)</span>
      <input type="number" id="regF0{S}" step="any" style="width:104px;" oninput="regFieldsChanged()">
      <label>–</label>
      <input type="number" id="regF1{S}" step="any" style="width:104px;" oninput="regFieldsChanged()">
      <button onclick="clearRegionBox({arg})" title="Remove this box">Clear</button>
    </div>"""


def decay_pane_html(tab, prefs, reg_box_mode):
    """Controls of one Decay Fit tab: "s" = Standard Fit (exponential decay),
    "b" = Bateman Fit. The tabs are independent: each has its own region,
    title, γ and Run. Ids in the Bateman tab end in "B" (mother box: "M").

    Standard: region = sliders or one drawn box; "no offset"; log / residual panels.
    Bateman: a drawn mother box and a drawn daughter box (joint parent +
    daughter fit, both over the time range where the boxes overlap); fixed t½."""
    b = tab == "b"
    S = "B" if b else ""                                   # id suffix
    pk = (lambda k: k + "B") if b else (lambda k: k)       # pref key
    # the Bateman tab's γ starts from the old shared value
    pref = (lambda k: prefs.get(pk(k), prefs.get(k, ""))) if b else (lambda k: prefs.get(k, ""))
    chk = lambda k: "checked" if pref(k) else ""
    hide = ' style="display:none;"' if tab != "s" else ""
    if b:
        region = (_box_row("m", "Mother Box",
                           "Box around the MOTHER (parent) peak: fitted as A·e^(-t/τ_A) + C_A") + "\n" +
                  _box_row("b", "Daughter Box",
                           "Box around the DAUGHTER peak: Bateman growth with no offset, τ_A shared with the mother") + """
    <div class="opt-grp"><span class="dbox-note">Both are fitted over the time range where the two boxes overlap.</span></div>""")
        model = """
    <div class="opt-grp" id="grpFixedHl" title="Known c.o.m.-frame half-lives (s), converted to lab frame with γ. Fixes the corresponding τ in the Bateman fit. Requires γ.">
      <span>Fixed t½ (s)</span>
      <input type="number" id="hlIso" min="0" step="any" placeholder="isomer"
             style="width:72px;" value="{hi}" oninput="savePref('hlIso', this.value)">
      <input type="number" id="hlGs" min="0" step="any" placeholder="g.s."
             style="width:72px;" value="{hg}" oninput="savePref('hlGs', this.value)">
    </div>""".format(hi=prefs.get("hlIso", ""), hg=prefs.get("hlGs", ""))
        plots = ""          # log / residual panels: single-window fits only
    else:
        box, sl = ("checked", "") if reg_box_mode else ("", "checked")
        region = f"""    <div class="opt-grp" title="Which time/frequency window the fit uses">
      <span>Region</span>
      <input type="radio" name="decayRegion" id="regSliders" value="sliders" {sl}
             onchange="setRegionMode()">
      <label for="regSliders">sliders</label>
      <input type="radio" name="decayRegion" id="regBox" value="box" {box}
             onchange="setRegionMode()">
      <label for="regBox">drawn box</label>
    </div>
""" + _box_row("s", "Box", "Fit region from the drawn box (editable)")
        model = """
    <div class="opt-grp">
      <span>Offset</span>
      <input type="checkbox" id="noOffset">
      <label for="noOffset" title="Fit A·exp(-t/τ) with no constant background">no offset</label>
    </div>"""
        plots = f"""
    <div class="opt-grp" title="Extra panels in the fit plot">
      <span>Plots</span>
      <input type="checkbox" id="logPlotChk" {chk('logPlot')}
             onchange="savePref('logPlot', this.checked ? '1' : '')">
      <label for="logPlotChk" title="Semi-log panel to the right: data − C and fit − C on a log axis (C = fitted background), so an exponential decay is a straight line">log (data − C)</label>
      <input type="checkbox" id="residChk" {chk('residuals')}
             onchange="savePref('residuals', this.checked ? '1' : '')">
      <label for="residChk" title="Residual panel under each plot: (data − fit)/σ">residuals</label>
    </div>"""
    return f"""    <div class="sim-body dpane" id="dpane{tab.upper()}" role="tabpanel" aria-labelledby="dtab{tab.upper()}"{hide}>
{region}
    <div class="opt-grp" style="width:100%;" title="Title of the decay-fit plot. Empty = automatic (file, range, model)">
      <span>Title</span>
      <input type="text" id="decayTitleBox{S}" placeholder="automatic (file, range, model)"
             style="flex:1; width:auto; min-width:260px;" oninput="saveSession()">
      <button onclick="suggestDecayTitle()"
              title="List simulated ions inside the fit region{' (mother and daughter boxes)' if b else ''} to pick a title from">Suggest ▾</button>
      <span id="decaySugStatus{S}" class="dsug-status"></span>
    </div>
    <div id="decaySugList{S}" class="srch-list"></div>{model}
    <div class="opt-grp">
      <span style="text-transform:none;">γ</span>
      <input type="number" id="gammaVal{S}" min="0" step="0.01"
             placeholder="γ" title="Lorentz factor γ (optional{'; required for fixed half-lives' if b else ''})"
             style="width:64px;" value="{pref('gamma')}"
             oninput="savePref('{pk('gamma')}', this.value)">
    </div>{plots}
    <div class="opt-grp">
      <button onclick="runDecayFit()">Run {'Bateman' if b else 'Standard'} Fit</button>
      <span id="decayStatus{S}" class="decay-status"></span>
    </div>
    </div>"""


def build_page(freq, time, power, title, log_scale=True, n_avg=1, file_path=None,
               fmin_req=None, fmax_req=None, ion=None, bl_info=None):
    """Build the spectrogram page. Returns (html, state) for the server."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        sys.exit(
            "plotly is required for --plot3d.  Install it with:\n"
            "  pip install plotly"
        )
    import tempfile, webbrowser, json, subprocess as _subprocess

    _npz_path   = os.path.abspath(file_path) if file_path else ""
    # Load persisted preferences (gamma value, etc.)
    _prefs = read_prefs()
    loader_html, loader_js = loader_parts(file_path, fmin_req, fmax_req)
    _saved_gamma = str(_prefs.get("gamma", ""))
    _saved_hl_iso = str(_prefs.get("hlIso", ""))
    _saved_hl_gs  = str(_prefs.get("hlGs", ""))
    _saved_pmin   = str(_prefs.get("parentMin", ""))
    _saved_pmax   = str(_prefs.get("parentMax", ""))
    _saved_srch_w = str(_prefs.get("srchWidth", "10"))
    # Decay Fit popout: tab open at start (old "bateman" checkbox pref as fallback)
    _decay_tab = {"bateman": "b", "standard": "s"}.get(_prefs.get("decayTab"),
                                                         "b" if _prefs.get("bateman") else "s")
    _chk_bars3d     = "" if _prefs.get("bars3d") == "0" else "checked"   # bars by default
    _chk_surface3d  = "" if _chk_bars3d else "checked"
    _cs_all  = colorscales_dict()
    _cs_name = _prefs.get("colorscale") if _prefs.get("colorscale") in _cs_all else "Inferno"
    _cs_options = "".join(f'<option value="{n}"{" selected" if n == _cs_name else ""}>{n}</option>'
                          for n in COLORSCALE_NAMES if n in _cs_all)
    _cs_json = json.dumps(_cs_all)

    # Saved view for this file + crop (sliders, title, camera, 1-D overlay)
    _view = saved_view(file_path, fmin_req, fmax_req)
    _default_title = title
    _vt = _view.get("title")
    if isinstance(_vt, str) and _vt.strip():
        title = _vt
    _chk_show1d = "" if _view.get("show1d") is False else "checked"
    _reg_box_mode     = _prefs.get("decayRegion") == "box"
    _chk_reg_box      = "checked" if _reg_box_mode else ""
    _chk_reg_sliders  = "" if _reg_box_mode else "checked"
    _rb = _view.get("regBox")
    _reg_init_js = json.dumps([float(v) for v in _rb]
                              if isinstance(_rb, list) and len(_rb) == 4 else None)
    _rbb = _view.get("regBoxB")
    _reg_init_js_b = json.dumps([float(v) for v in _rbb]
                                if isinstance(_rbb, list) and len(_rbb) == 4 else None)
    _decay_pane_s = decay_pane_html("s", _prefs, _reg_box_mode)
    _rbm = _view.get("regBoxM")
    _reg_init_js_m = json.dumps([float(v) for v in _rbm]
                                if isinstance(_rbm, list) and len(_rbm) == 4 else None)
    _decay_pane_b = decay_pane_html("b", _prefs, True)
    if _decay_tab == "b":   # open on the Bateman tab: swap which pane starts hidden
        _decay_pane_s = _decay_pane_s.replace('role="tabpanel" aria-labelledby="dtabS">',
                                              'role="tabpanel" aria-labelledby="dtabS" style="display:none;">', 1)
        _decay_pane_b = _decay_pane_b.replace(' style="display:none;">', '>', 1)
    _sim_panel = sim_panel_html()
    _bl = bl_info or {}
    _bl_panel = baseline_panel_html(_bl)
    _bl_btn = "Baseline ✓…" if _bl.get("on") else "Baseline…"
    _full_open_js = "false" if _prefs.get("fullOpen") == "0" else "true"

    z_label = "Power (dB)" if log_scale else "Power (linear)"

    # Full-resolution arrays for the 1-D summed spectrum (served on demand).
    # Captured here because `time` is re-bound to the time module further down.
    _t_full = np.asarray(time, dtype=float)
    _p_full = power
    f_1d_json = json.dumps(np.asarray(freq, dtype=float).tolist())

    # Downsample if very large to keep the browser responsive
    # Target ≤ 500 points per axis
    MAX_PTS = 500
    t_step = max(1, len(time) // MAX_PTS)
    f_step = max(1, len(freq) // MAX_PTS)
    t_ds   = time[::t_step]
    f_ds   = freq[::f_step]
    # Raw linear power, downsampled, frequency-axis flipped (high→low for display)
    z_ds   = power[::t_step, ::f_step][:, ::-1]   # shape (T_ds, F_ds)
    # dB of baseline-removed data: values ≤ 0 are drawn at this floor (1st
    # percentile of the positive values) instead of at 10·log10(5e-324)
    _pos = z_ds[z_ds > 0]
    db_floor = float(np.percentile(_pos, 1)) if (_bl.get("on") and _pos.size) else 5e-324
    f_plot = f_ds[::-1]

    if t_step > 1 or f_step > 1:
        print(
            f"Downsampled for 3-D plot: {len(t_ds)} time pts × {len(f_ds)} freq pts "
            f"(original: {len(time)} × {len(freq)})"
        )

    # Apply initial time averaging in Python for first render
    def _py_rebin(t, z, n):
        T, F = z.shape
        T_trim = (T // n) * n
        z_r = z[:T_trim].reshape(T // n, n, F).mean(axis=1)
        t_r = t[:T_trim].reshape(T // n, n).mean(axis=1)
        return t_r, z_r

    if n_avg > 1:
        t_init, z_init = _py_rebin(t_ds, z_ds, n_avg)
    else:
        t_init, z_init = t_ds.copy(), z_ds.copy()

    fig = go.Figure(data=[go.Surface(
        x=[0, 1], y=[0, 1], z=[[0, 0], [0, 0]],   # placeholder: real data built in the page
        colorscale=_cs_all[_cs_name],
        colorbar=dict(
            title=dict(text=z_label, side="right"),
            thickness=18,
        ),
        hovertemplate=(
            "Time: %{x:.2f} s<br>"
            "Freq: %{y:,.0f} Hz<br>"
            + z_label + ": %{z:.2f}<extra></extra>"
        ),
    )])

    # Axis bounds and bin sizes for slider step sizes
    f_min, f_max = float(freq.min()), float(freq.max())
    df = float(freq[1] - freq[0]) if len(freq) > 1 else max(f_max - f_min, 1.0)

    # Initial time slider bounds (after initial averaging)
    t_init_min = float(t_init[0])
    t_init_max = float(t_init[-1])
    dt_init = float(t_init[1] - t_init[0]) if len(t_init) > 1 else max(t_init_max - t_init_min, 1e-6)

    # Restore saved slider positions, clamped to what this data allows
    def _restore_range(lo_key, hi_key, lo, hi):
        try:
            a = min(max(float(_view[lo_key]), lo), hi)
            b = min(max(float(_view[hi_key]), lo), hi)
            if a < b:
                return a, b
        except (KeyError, TypeError, ValueError):
            pass
        return lo, hi

    f_sel_min, f_sel_max = _restore_range("fMin", "fMax", f_min, f_max)
    t_sel_min, t_sel_max = _restore_range("tMin", "tMax", t_init_min, t_init_max)

    _camera = _view.get("camera") if isinstance(_view.get("camera"), dict) else None
    if _camera is None:
        _camera = dict(eye=dict(x=1.6, y=-1.6, z=0.8))

    _scene = dict(
        xaxis=dict(title="Time (s)"),
        yaxis=dict(title="Frequency (Hz)", tickformat=",d",
                   range=[f_sel_max, f_sel_min]),
        zaxis=dict(title=z_label),
        camera=_camera,
    )
    if (t_sel_min, t_sel_max) != (t_init_min, t_init_max):
        _scene["xaxis"]["range"] = [t_sel_min, t_sel_max]

    fig.update_layout(
        # title below the Plotly toolbar (top-right) so it is never covered
        title=dict(text=format_title_html(title), x=0.5, xanchor="center", font_size=18,
                   yref="container", y=1, yanchor="top", pad=dict(t=34)),
        scene=_scene,
        margin=dict(l=0, r=0, t=80, b=0),
        height=680,
    )

    # Embed raw (downsampled) data as JSON for JS re-rendering
    z_raw_json = json.dumps(z_ds.tolist())
    t_raw_json = json.dumps(t_ds.tolist())
    f_raw_json = json.dumps(f_ds.tolist())   # freq array, low-to-high (for 2-D heatmap)
    db_checked = "checked" if log_scale else ""
    n_avg_init = n_avg

    # The 3-D plot is created in the page by _initSurface() from Z_RAW, so only
    # its style and layout are embedded (not a second copy of the data), and
    # plotly.js is served by this program from the plotly package (no internet
    # needed: core/offline.py)
    from plotly.utils import PlotlyJSONEncoder
    _lay = fig.to_plotly_json()["layout"]      # incl. plotly's default template (3-D panes)
    surf_layout_json = json.dumps(_lay, cls=PlotlyJSONEncoder).replace("</", "<\\/")
    surf_style_json = json.dumps({
        "type": "surface",
        "colorbar": {"title": {"text": z_label, "side": "right"}, "thickness": 18},
    }).replace("</", "<\\/")
    plot_div = (f'<script src="{offline.PLOTLY_JS_URL}" charset="utf-8"></script>\n'
                f'<div id="specPlot" style="width:100%;"></div>')
    _default_title_js = json.dumps(_default_title).replace("</", "<\\/")
    title_html = title.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
{icon.LINK_TAG}
<title>{title_html}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; padding: 8px; background: #111827; color: #e5e7eb;
         font-family: system-ui, sans-serif; font-size: 13px; }}
  #controls {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px;
               background: #1f2937; border-radius: 8px; padding: 12px 16px;
               margin-bottom: 6px; }}
  .title-row {{ flex: 1; min-width: 240px; display: flex; align-items: center;
                gap: 10px; margin: 0; }}
  .title-row label {{ color: #9ca3af; white-space: nowrap; font-size: 12px;
                      font-weight: 600; text-transform: uppercase;
                      letter-spacing: .06em; color: #f97316; }}
  .title-row input[type=text] {{ flex: 1; background: #374151;
                                  border: 1px solid #4b5563; border-radius: 4px;
                                  color: #e5e7eb; padding: 4px 8px; font-size: 13px; }}
  .title-row input[type=text]:focus {{ outline: none; border-color: #f97316; }}
  .grp h4 {{ margin: 0 0 8px; font-size: 12px; font-weight: 600;
              text-transform: uppercase; letter-spacing: .06em; color: #f97316; }}
  .row {{ display: flex; align-items: center; gap: 8px; margin: 4px 0; }}
  .row label {{ width: 28px; color: #9ca3af; }}
  .row input[type=range] {{ flex: 1; accent-color: #f97316; cursor: pointer; }}
  .row button.step {{ width: 22px; height: 20px; padding: 0; background: #374151;
                      border: 1px solid #4b5563; border-radius: 4px; color: #e5e7eb;
                      font-size: 13px; line-height: 1; cursor: pointer; }}
  .row button.step:hover {{ border-color: #f97316; color: #f97316; }}
  .row span {{ width: 110px; text-align: right; color: #d1d5db; font-size: 11px;
               font-variant-numeric: tabular-nums; }}
  .opts-row {{ grid-column: 1 / -1; display: flex; align-items: center;
               gap: 24px; padding-top: 8px; border-top: 1px solid #374151;
               flex-wrap: wrap; }}
  .opt-grp {{ display: flex; align-items: center; gap: 6px; }}
  .opt-grp span {{ font-size: 12px; font-weight: 600; text-transform: uppercase;
                   letter-spacing: .06em; color: #f97316; margin-right: 2px; }}
  .opt-grp label {{ color: #9ca3af; }}
  .opt-grp input[type=checkbox], .opt-grp input[type=radio] {{ accent-color: #f97316; width: 14px; height: 14px;
                                    cursor: pointer; }}
  .opt-grp input[type=number] {{ width: 60px; background: #374151;
                                  border: 1px solid #4b5563; border-radius: 4px;
                                  color: #e5e7eb; padding: 3px 6px; font-size: 13px; }}
  .opt-grp input[type=number]:focus {{ outline: none; border-color: #f97316; }}
  .opt-grp button {{ background: #f97316; border: none; border-radius: 4px;
                     color: #111827; font-size: 12px; font-weight: 700;
                     padding: 4px 12px; cursor: pointer; }}
  .opt-grp button:hover {{ background: #fb923c; }}
  #stepBox {{ align-items: center; gap: 10px; padding: 3px 8px;
              border: 1px solid #4b5563; border-radius: 6px; }}
  .opt-grp input:disabled {{ opacity: .35; cursor: not-allowed; }}
  .opt-grp input:disabled + label {{ opacity: .35; }}
  .opt-grp button:disabled {{ opacity: .35; cursor: not-allowed; }}
  #discardMsg {{ font-size: 11px; color: #fbbf24; margin-left: 4px; }}
  .opt-grp span.decay-status {{ font-size: 11px; margin-left: 4px; text-transform: none;
                                letter-spacing: normal; font-weight: 400; }}
  .dtab-bar {{ display: flex; gap: 2px; padding: 10px 16px 0; border-bottom: 1px solid #374151; }}
  .dtab {{ background: none; border: 1px solid transparent; border-bottom: none;
           border-radius: 6px 6px 0 0; color: #9ca3af; font-size: 12px; font-weight: 600;
           text-transform: uppercase; letter-spacing: .06em; padding: 6px 14px;
           margin-bottom: -1px; cursor: pointer; }}
  .dtab:hover {{ color: #e5e7eb; }}
  #decayModal .sim-dialog {{ width: min(880px, calc(100vw - 32px)); }}
  .opt-grp span.dbox-note {{ font-size: 11px; color: #9ca3af; text-transform: none;
                             letter-spacing: normal; font-weight: 400; }}
  .dtab.active {{ color: #f97316; background: #1f2937; border-color: #374151;
                  box-shadow: inset 0 2px 0 #f97316; }}
  #plots {{ display: flex; gap: 10px; align-items: stretch; }}
  #heatDiv, #surfDiv {{ flex: 1; min-width: 0; border: 1px solid #374151; border-radius: 8px;
                        padding: 6px; background: #111827; overflow: hidden; }}
  #heatDiv.drawing {{ outline: 2px dashed #f97316; outline-offset: 2px; }}
{LOADER_CSS}
{SIM_CSS}
{LOGBOOK_CSS}
</style>
</head>
<body>
{BOOT_OVERLAY}
<div id="controls">
  <div id="fileModal" class="sim-modal" onclick="if(event.target===this) modalClose('fileModal')">
   <div class="sim-dialog file-dialog" role="dialog" aria-modal="true" aria-labelledby="fileTitle">
    <div class="sim-head">
      <span id="fileTitle">File Settings</span>
      <button class="sim-x" onclick="modalClose('fileModal')" title="Close (Esc)">×</button>
    </div>
    <div class="sim-body">
{loader_html}
    </div>
   </div>
  </div>
  <div class="grp">
    <h4>Frequency (Hz)</h4>
    <div class="row">
      <label>Min</label>
      <input type="range" id="fMin" min="{f_min}" max="{f_max}"
             step="{df}" value="{f_sel_min}" oninput="update()">
      <span id="fMinVal">{f_sel_min:,.0f}</span>
    </div>
    <div class="row">
      <label>Max</label>
      <input type="range" id="fMax" min="{f_min}" max="{f_max}"
             step="{df}" value="{f_sel_max}" oninput="update()">
      <span id="fMaxVal">{f_sel_max:,.0f}</span>
    </div>
  </div>
  <div class="grp">
    <h4>Time (s)</h4>
    <div class="row">
      <label>Min</label>
      <input type="range" id="tMin" min="{t_init_min}" max="{t_init_max}"
             step="{dt_init}" value="{t_sel_min}" oninput="update()">
      <button class="step" onclick="stepTime('tMin', -1)" title="Back one time frame">−</button>
      <button class="step" onclick="stepTime('tMin', 1)" title="Forward one time frame">+</button>
      <span id="tMinVal">{t_sel_min:.3f}</span>
    </div>
    <div class="row">
      <label>Max</label>
      <input type="range" id="tMax" min="{t_init_min}" max="{t_init_max}"
             step="{dt_init}" value="{t_sel_max}" oninput="update()">
      <button class="step" onclick="stepTime('tMax', -1)" title="Back one time frame">−</button>
      <button class="step" onclick="stepTime('tMax', 1)" title="Forward one time frame">+</button>
      <span id="tMaxVal">{t_sel_max:.3f}</span>
    </div>
  </div>
  <div class="opts-row">
    <div class="opt-grp">
      <span>Time Avg</span>
      <input type="number" id="nAvg" min="1" step="1" value="{n_avg_init}">
      <button onclick="updateSurface()">Apply</button>
      <span id="discardMsg"></span>
    </div>
    <div class="opt-grp">
      <button onclick="modalOpen('generalModal', 'colorscaleSel')" title="Colour scheme, dB scale, 3-D style, 1-D overlay">General Settings…</button>
      <button onclick="modalOpen('fileModal', 'ldPath')" title="Load a file / change the frequency crop">File Settings…</button>
      <button onclick="simOpen()" title="Particle-identification settings: simulated ion lines">PID Settings…</button>
      <button id="blBtn" onclick="modalOpen('blModal', 'bl_lam')" title="Remove a BrPLS baseline from every time frame">{_bl_btn}</button>
      <button onclick="modalOpen('searchModal', 'srchIon')" title="Find an ion's simulated lines and load a window around one">Search Ions…</button>
      <button onclick="modalOpen('decayModal', DECAY_TAB === 'b' ? 'dtabB' : 'dtabS')" title="Decay-fit settings and Run (Standard and Bateman tabs)">Decay Fit…</button>
      <button onclick="lbOpen()" title="Send this view (plots, window, ion) to the results log book">Save to Log Book…</button>
      <span id="stepBox" style="display:none;">
      <span id="harmGrp" style="display:none; align-items:center; gap:6px;">
        <button id="harmDn" onclick="harmStep(-1)" disabled>h−1</button>
        <span id="harmCur" title="Harmonic of the ion loaded by Search ions"></span>
        <button id="harmUp" onclick="harmStep(1)" disabled>h+1</button>
      </span>
      <span id="qGrp" style="display:none; align-items:center; gap:6px;">
        <button id="qDn" onclick="ionStep('q', -1)" disabled>q−1</button>
        <span id="qCur" title="Charge state of the ion loaded by Search ions"></span>
        <button id="qUp" onclick="ionStep('q', 1)" disabled>q+1</button>
      </span>
      <span id="aGrp" style="display:none; align-items:center; gap:6px;">
        <button id="aDn" onclick="ionStep('A', -1)" disabled>A−1</button>
        <span id="aCur" title="Isotope of the ion loaded by Search ions (A±1 goes to the ground state)"></span>
        <button id="aUp" onclick="ionStep('A', 1)" disabled>A+1</button>
      </span>
      </span>
      <span id="harmStatus" style="text-transform:none; font-weight:400;"></span>
    </div>
    <div class="title-row">
      <label for="plotTitle">Title</label>
      <input type="text" id="plotTitle" value="{title_html}" oninput="updateTitle()">
    </div>
    <div class="opt-grp">
      <button id="quitBtn" class="quit-btn" onclick="quitApp()"
              title="Stop plot_spectrogram.py and close this tab">Quit</button>
    </div>
  </div>
{LOGBOOK_HTML}
  <div id="decayModal" class="sim-modal" onclick="if(event.target===this) modalClose('decayModal')">
   <div class="sim-dialog sim-row" role="dialog" aria-modal="true" aria-labelledby="decayTitle">
    <div class="sim-head">
      <span id="decayTitle">Decay Fit</span>
      <button class="sim-x" onclick="modalClose('decayModal')" title="Close (Esc)">×</button>
    </div>
    <div class="dtab-bar" role="tablist" aria-label="Fit model">
      <button id="dtabS" class="dtab" role="tab" aria-controls="dpaneS" onclick="decayTab('s')"
              title="Single exponential A·e^(-t/τ) (+ C)">Standard Fit</button>
      <button id="dtabB" class="dtab" role="tab" aria-controls="dpaneB" onclick="decayTab('b')"
              title="Bateman two-member chain A→B→C (optionally parent + daughter together)">Bateman Fit</button>
    </div>
{_decay_pane_s}
{_decay_pane_b}
   </div>
  </div>
  <div id="generalModal" class="sim-modal" onclick="if(event.target===this) modalClose('generalModal')">
   <div class="sim-dialog" role="dialog" aria-modal="true" aria-labelledby="generalTitle">
    <div class="sim-head">
      <span id="generalTitle">General Settings</span>
      <button class="sim-x" onclick="modalClose('generalModal')" title="Close (Esc)">×</button>
    </div>
    <div class="sim-body">
    <div class="opt-grp">
      <span>Colour Scheme</span>
      <select id="colorscaleSel" onchange="setColorscale(this.value)"
              title="Colour scale for the 2-D heatmap and 3-D plot">{_cs_options}</select>
    </div>
    <div class="opt-grp">
      <span>Scale</span>
      <input type="checkbox" id="dbCheck" {db_checked} onchange="updateSurface()">
      <label for="dbCheck">dB</label>
    </div>
    <div class="opt-grp">
      <span>3D</span>
      <input type="radio" name="style3d" id="bars3d" value="bars" {_chk_bars3d}
             onchange="_set3dStyle()">
      <label for="bars3d" title="Draw each point of the 3-D view as a flat-topped square bar, like the heatmap cells">bars</label>
      <input type="radio" name="style3d" id="surface3d" value="surface" {_chk_surface3d}
             onchange="_set3dStyle()">
      <label for="surface3d" title="Smooth surface joining neighbouring points (the original 3-D style)">surface</label>
    </div>
    <div class="opt-grp">
      <span>1D Overlay</span>
      <input type="checkbox" id="show1d" {_chk_show1d} onchange="_renderSpec1d(); saveSession()">
      <label for="show1d">show</label>
    </div>
    </div>
   </div>
  </div>
  <div id="searchModal" class="sim-modal" onclick="if(event.target===this) modalClose('searchModal')">
   <div class="sim-dialog sim-row" role="dialog" aria-modal="true" aria-labelledby="searchTitle">
    <div class="sim-head">
      <span id="searchTitle">Search Ions</span>
      <button class="sim-x" onclick="modalClose('searchModal')" title="Close (Esc)">×</button>
    </div>
    <div class="sim-body">
    <div class="opt-grp">
      <span>Ion</span>
      <input type="text" id="srchIon" placeholder="e.g. 186Hf or m186Hf72+" style="width:220px;"
             oninput="searchIonsSoon()" onkeydown="if(event.key==='Enter') searchIons()">
      <button onclick="searchIons()">Search</button>
      <span title="Half-width of the frequency window loaded around the chosen line">±</span>
      <input type="number" id="srchWidth" min="0" step="any" style="width:72px;" value="{_saved_srch_w}"
             placeholder="10" title="Half-width of the frequency window loaded around the chosen line (kHz)"
             oninput="srchWidthChanged()" onkeydown="if(event.key==='Enter') searchIons()">
      <span>kHz</span>
      <span id="srchStatus"></span>
    </div>
    <div class="srch-hint">Simulated lines across the whole file (current PID settings). Click one to load
      ± the width above around it (default ±10 kHz).</div>
    <div id="srchList" class="srch-list"></div>
    </div>
   </div>
  </div>
{_sim_panel}
{_bl_panel}
</div>
<div id="regBanner" class="reg-banner">Drag a box on the 2-D plot to set the decay-fit region · Esc to cancel</div>
<div id="plots">
  <div id="heatDiv"><div id="heatPlot"></div></div>
  <div id="surfDiv">{plot_div}</div>
</div>
<div id="fullDiv" style="margin-top:8px;">
  <button id="fullToggle" class="full-toggle" onclick="toggleFull()" aria-expanded="true">
    <span id="fullArrow">▾</span> Whole-file spectrum
  </button>
  <div id="fullPlot"></div>
</div>
<script>
{loader_js}
{SIM_JS}
{LOGBOOK_JS}
const Z_RAW = {z_raw_json};
const SURF_LAYOUT = {surf_layout_json};
const SURF_STYLE  = {surf_style_json};
const T_RAW = {t_raw_json};
const F_PLOT = {f_raw_json};  // freq array, low-to-high (2-D heatmap y-axis)
const F_1D   = {f_1d_json};   // full-resolution freq array for the 1-D spectrum
const DEFAULT_TITLE = {_default_title_js};
var CUR_ION_LABEL = {json.dumps(ion or "")};   // Search ions label, e.g. "m186Hf72+, h213"
var APPLIED_N = {n_avg_init};
const DB_FLOOR = {db_floor!r};   // smallest power drawn in dB (raised when the baseline is removed)
var LAST_CAMERA = {json.dumps(_camera)};

var _heatReady = true;

// Colour scales as explicit [pos, colour] arrays (from plotly.py), so every
// scheme works whatever plotly.js version the CDN serves
const COLORSCALES = {_cs_json};
var CUR_CS = {json.dumps(_cs_name)};

function setColorscale(name) {{
  if (!COLORSCALES[name]) return;
  CUR_CS = name;
  Plotly.restyle('heatPlot', {{ colorscale: [COLORSCALES[name]] }}, [0]);
  Plotly.restyle('specPlot', {{ colorscale: [COLORSCALES[name]] }}, [0]);
  savePref('colorscale', name);
}}

function toDb(Z) {{
  var eps = DB_FLOOR;
  return Z.map(function(row) {{ return row.map(function(v) {{ return 10 * Math.log10(Math.max(v, eps)); }}); }});
}}

function rebin(Z, t, n) {{
  if (n <= 1) return {{Z: Z, t: t}};
  var T = Z.length, F = Z[0].length;
  var Ttrim = Math.floor(T / n) * n;
  var Tout = Ttrim / n;
  var Zout = [], tout = [];
  for (var i = 0; i < Tout; i++) {{
    var row = new Array(F).fill(0.0);
    var tsum = 0.0;
    for (var j = 0; j < n; j++) {{
      var src = Z[i * n + j];
      for (var f = 0; f < F; f++) row[f] += src[f];
      tsum += t[i * n + j];
    }}
    Zout.push(row.map(function(v) {{ return v / n; }}));
    tout.push(tsum / n);
  }}
  return {{Z: Zout, t: tout}};
}}

function transpose(Z) {{
  if (!Z.length) return [];
  return Z[0].map(function(_, f) {{ return Z.map(function(row) {{ return row[f]; }}); }});
}}

function resetTimeSliders(t) {{
  var tMin = t[0], tMax = t[t.length - 1];
  var dt = t.length > 1 ? t[1] - t[0] : Math.max(tMax - tMin, 1e-6);
  var sMin = document.getElementById('tMin');
  var sMax = document.getElementById('tMax');
  sMin.min = tMin; sMin.max = tMax; sMin.step = dt; sMin.value = tMin;
  sMax.min = tMin; sMax.max = tMax; sMax.step = dt; sMax.value = tMax;
  document.getElementById('tMinVal').textContent = fmt(tMin, 3);
  document.getElementById('tMaxVal').textContent = fmt(tMax, 3);
}}

// ── 2-D heatmap ─────────────────────────────────────────────────────────────
// Z_RAW has freq reversed (high→low) for the 3-D surface display.
// For the heatmap we un-flip freq, then transpose to (F, T) as Plotly expects.

function _heat2dZ(rebinnedZ, useDb) {{
  var Zunflip = rebinnedZ.map(function(row) {{ return row.slice().reverse(); }});
  return transpose(useDb ? toDb(Zunflip) : Zunflip);
}}

function _initHeatmap(h, rebinned, mm) {{
  var useDb = document.getElementById('dbCheck').checked;
  var zLabel = useDb ? 'Power (dB)' : 'Power (linear)';
  Plotly.newPlot('heatPlot', [{{
    type: 'heatmap',
    x: rebinned.t,
    y: F_PLOT,
    z: _heat2dZ(rebinned.Z, useDb),
    colorscale: COLORSCALES[CUR_CS],
    zmin: mm.min, zmax: mm.max, zauto: false,
    colorbar: {{ title: {{ text: zLabel, side: 'right' }} }},
    hovertemplate: 'Time: %{{x:.2f}} s<br>Freq: %{{y:,.0f}} Hz<br>' + zLabel + ': %{{z:.2f}}<extra></extra>'
  }}, {{
    // 1-D summed spectrum overlay: runs along the frequency (y) axis,
    // with summed power on its own x-axis along the top
    type: 'scattergl', mode: 'lines',
    x: [], y: [],
    xaxis: 'x2', yaxis: 'y',
    line: {{ color: 'rgba(56,189,248,0.9)', width: 1.2 }},
    showlegend: false,
    hovertemplate: 'Freq: %{{y:,.0f}} Hz<br>Summed: %{{x:.4g}}<extra></extra>'
  }}, {{
    // Simulated ion lines (filled by _renderSim). Each group is drawn twice:
    // a wide dark outline underneath (traces 2-4) and a thin white line on
    // top (traces 5-7), with identical px dash patterns so they line up.
    type: 'scatter', mode: 'lines', x: [], y: [], hoverinfo: 'skip',
    connectgaps: false, showlegend: false, name: 'Simulated outline',
    line: {{ color: 'rgba(0,0,0,0.75)', width: 3.5, dash: '6px,4px' }}
  }}, {{
    type: 'scatter', mode: 'lines', x: [], y: [], hoverinfo: 'skip',
    connectgaps: false, showlegend: false, name: 'Simulated (T½ cut) outline',
    line: {{ color: 'rgba(0,0,0,0.6)', width: 3, dash: '2px,3px' }}
  }}, {{
    type: 'scatter', mode: 'lines', x: [], y: [], hoverinfo: 'skip',
    connectgaps: false, showlegend: false, name: 'Reference outline',
    line: {{ color: 'rgba(0,0,0,0.8)', width: 4 }}
  }}, {{
    type: 'scatter', mode: 'lines', x: [], y: [], hovertext: [], hoverinfo: 'text',
    connectgaps: false, showlegend: false, name: 'Simulated',
    line: {{ color: 'rgba(255,255,255,0.95)', width: 1.2, dash: '6px,4px' }}
  }}, {{
    type: 'scatter', mode: 'lines', x: [], y: [], hovertext: [], hoverinfo: 'text',
    connectgaps: false, showlegend: false, name: 'Simulated (T½ cut)',
    line: {{ color: 'rgba(220,220,220,0.85)', width: 1, dash: '2px,3px' }}
  }}, {{
    type: 'scatter', mode: 'lines', x: [], y: [], hovertext: [], hoverinfo: 'text',
    connectgaps: false, showlegend: false, name: 'Reference',
    line: {{ color: 'rgba(255,255,255,1)', width: 1.8 }}
  }}], {{
    template: {{}},
    title: {{ text: formatTitleHtml(document.getElementById('plotTitle').value),
              // below the Plotly toolbar (top-right) so it is never covered
              x: 0.5, xanchor: 'center', yref: 'container', y: 1, yanchor: 'top', pad: {{ t: 34 }},
              font: {{ size: 18, color: '#e5e7eb' }} }},
    xaxis2: {{ overlaying: 'x', side: 'top', title: {{ text: '', standoff: 4 }},
               showgrid: false, zeroline: false,
               color: 'rgb(56,189,248)', tickfont: {{ size: 10 }} }},
    xaxis: {{ title: {{ text: 'Time (s)' }},
              range: [+document.getElementById('tMin').value, +document.getElementById('tMax').value],
              color: '#9ca3af', gridcolor: '#374151', zerolinecolor: '#374151' }},
    yaxis: {{ title: {{ text: 'Frequency (Hz)', standoff: 18 }}, automargin: true,
              tickformat: ',d', autorange: false,
              range: [+document.getElementById('fMax').value, +document.getElementById('fMin').value],
              color: '#9ca3af', gridcolor: '#374151', zerolinecolor: '#374151' }},
    paper_bgcolor: '#111827', plot_bgcolor: '#1f2937',
    font: {{ color: '#e5e7eb' }},
    // one tooltip: the nearest item only (older plotly.js defaults to 'x',
    // which shows every line at that time); sim lines win within 10 px,
    // otherwise the heatmap cell under the mouse is shown
    hovermode: 'closest', hoverdistance: 10,
    margin: {{ l: 110, r: 20, t: 120, b: 60 }},
    height: h,     // fills the window (plotHeight); refit on resize by fitPlots()
  }}, {{ responsive: true }});
}}

function _updateHeatmap() {{
  if (!_heatReady) return;
  var n = Math.max(1, Math.round(+document.getElementById('nAvg').value || 1));
  var useDb = document.getElementById('dbCheck').checked;
  var rebinned = rebin(Z_RAW, T_RAW, n);
  var zLabel = useDb ? 'Power (dB)' : 'Power (linear)';
  Plotly.restyle('heatPlot', {{
    x: [rebinned.t],
    z: [_heat2dZ(rebinned.Z, useDb)],
    'colorbar.title.text': [zLabel],
    hovertemplate: ['Time: %{{x:.2f}} s<br>Freq: %{{y:,.0f}} Hz<br>' + zLabel + ': %{{z:.2f}}<extra></extra>']
  }}, [0]);
  Plotly.relayout('heatPlot', {{
    'xaxis.range': [+document.getElementById('tMin').value, +document.getElementById('tMax').value],
    'yaxis.range': [+document.getElementById('fMax').value, +document.getElementById('fMin').value]
  }});
}}

// ── 1-D summed spectrum ─────────────────────────────────────────────────────
// Sum of all full-resolution time frames within the time-slider window,
// computed server-side so no frames or freq bins are lost to downsampling.
var _spec1dY = null, _spec1dN = 0, _spec1dT = null;
var _spec1dTimer = null, _spec1dSeq = 0;

function _fetchSpec1d() {{
  clearTimeout(_spec1dTimer);
  _spec1dTimer = setTimeout(async function() {{
    var seq  = ++_spec1dSeq;
    var n    = Math.max(1, Math.round(+document.getElementById('nAvg').value || 1));
    var dtDs = T_RAW.length > 1 ? T_RAW[1] - T_RAW[0] : 0;
    var half = n * dtDs / 2;   // half-width of one displayed (averaged) time bin
    var tMin = +document.getElementById('tMin').value;
    var tMax = +document.getElementById('tMax').value;
    try {{
      var r = await fetch('/spectrum1d?tmin=' + (tMin - half) + '&tmax=' + (tMax + half));
      var d = await r.json();
      if (seq !== _spec1dSeq) return;          // a newer request superseded this one
      if (!r.ok) throw new Error(d.error || 'error');
      _spec1dY = d.y; _spec1dN = d.n; _spec1dT = [d.t0, d.t1];
      _renderSpec1d();
    }} catch (e) {{
      console.log('[spec1d] ' + e.message);
    }}
  }}, 120);
}}

function _spec1dYRange(y, fMin, fMax) {{
  var lo = Infinity, hi = -Infinity;
  for (var i = 0; i < F_1D.length; i++) {{
    if (F_1D[i] < fMin || F_1D[i] > fMax) continue;
    var v = y[i];
    if (!isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }}
  if (!isFinite(lo)) return null;
  var pad = (hi - lo) * 0.05 || Math.abs(hi) * 0.05 || 1;
  return [lo - pad, hi + pad];
}}

function _renderSpec1d() {{
  if (!_heatReady || !_spec1dY) return;
  var show = document.getElementById('show1d').checked;
  if (!show) {{
    Plotly.restyle('heatPlot', {{ visible: [false] }}, [1]);
    Plotly.relayout('heatPlot', {{ 'xaxis2.visible': false }});
    return;
  }}
  var useDb = document.getElementById('dbCheck').checked;
  var y = useDb ? _spec1dY.map(function(v) {{ return v > 0 ? 10 * Math.log10(v) : NaN; }})
                : _spec1dY;
  var fMin = +document.getElementById('fMin').value;
  var fMax = +document.getElementById('fMax').value;
  var pLabel = useDb ? 'Summed Power (dB)' : 'Summed Power (linear)';
  var info = _spec1dN + ' frame' + (_spec1dN === 1 ? '' : 's') +
             (_spec1dN > 0 ? ', t = ' + _spec1dT[0].toFixed(2) + '–' + _spec1dT[1].toFixed(2) + ' s' : '');
  var pr = _spec1dYRange(y, fMin, fMax);
  Plotly.restyle('heatPlot', {{
    x: [y], y: [F_1D], visible: [true],
    hovertemplate: ['Freq: %{{y:,.0f}} Hz<br>' + pLabel + ': %{{x:.4g}}<extra></extra>']
  }}, [1]);
  var lay = {{ 'xaxis2.visible': true, 'xaxis2.title.text': pLabel + '  (' + info + ')' }};
  if (pr) {{ lay['xaxis2.range'] = pr; lay['xaxis2.autorange'] = false; }}
  else    {{ lay['xaxis2.autorange'] = true; }}
  Plotly.relayout('heatPlot', lay);
}}

// ── Decay-fit controls: enable only what applies to the chosen mode ──────────
// ── Persisted preferences ────────────────────────────────────────────────────
function savePref(key, value) {{
  fetch('/save-pref', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{key: key, value: value}})
  }});
}}

// ── Session memory: the bare `python plot_spectrogram.py` reopens this view ──
function _sessionState() {{
  var gd  = document.getElementById('specPlot');
  var cam = (gd && gd.layout && gd.layout.scene && gd.layout.scene.camera) || LAST_CAMERA;
  var title = document.getElementById('plotTitle').value;
  return {{
    linear: !document.getElementById('dbCheck').checked,
    nAvg:   APPLIED_N,
    view: {{
      fMin:   +document.getElementById('fMin').value,
      fMax:   +document.getElementById('fMax').value,
      tMin:   +document.getElementById('tMin').value,
      tMax:   +document.getElementById('tMax').value,
      title:  title !== DEFAULT_TITLE ? title : '',
      show1d: document.getElementById('show1d').checked,
      // undefined (= keep the saved box) until the saved region has been read back
      regBox:  _regLoaded ? _regBoxValues('s') : undefined,
      regBoxB: _regLoaded ? _regBoxValues('b') : undefined,     // Bateman daughter box
      regBoxM: _regLoaded ? _regBoxValues('m') : undefined,     // Bateman mother box
      decayTitle:  _regLoaded ? document.getElementById('decayTitleBox').value : undefined,
      decayTitleB: _regLoaded ? document.getElementById('decayTitleBoxB').value : undefined,
      camera: cam
    }}
  }};
}}

var _sessTimer = null;
function saveSession() {{
  clearTimeout(_sessTimer);
  _sessTimer = setTimeout(function() {{
    fetch('/save-session', {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify(_sessionState())
    }}).catch(function() {{}});
  }}, 400);
}}

// Save after 3-D camera drags/zooms. Wait until the scene is fully built,
// because listeners attached earlier are dropped by plotly.
function _hookCamera() {{
  var gd = document.getElementById('specPlot');
  var ready = gd && gd.on && gd._fullLayout && gd._fullLayout.scene &&
              gd._fullLayout.scene._scene;
  if (!ready) {{ setTimeout(_hookCamera, 200); return; }}
  gd.on('plotly_relayout', function(ev) {{
    if (ev && ev['scene.camera']) LAST_CAMERA = ev['scene.camera'];
    saveSession();
  }});
  gd.addEventListener('mouseup', saveSession);
  gd.addEventListener('wheel', saveSession, {{passive: true}});
}}

// Flush the latest state when the tab is closed
window.addEventListener('beforeunload', function() {{
  if (navigator.sendBeacon) {{
    navigator.sendBeacon('/save-session', new Blob([JSON.stringify(_sessionState())],
                                                   {{type: 'application/json'}}));
  }}
}});

// ── Main update functions ────────────────────────────────────────────────────

async function runDecayFit() {{
  var bateman = DECAY_TAB === 'b';          // the tab whose Run was pressed
  var status  = _d('decayStatus');
  status.textContent = 'Launching…';
  status.style.color = '#fbbf24';
  try {{
    var gammaRaw = _d('gammaVal').value.trim();
    var hlIsoRaw = document.getElementById('hlIso').value.trim();
    var hlGsRaw  = document.getElementById('hlGs').value.trim();
    var win = decayWindow();   // sliders / drawn box / mother + daughter boxes (throws if invalid)
    if (bateman && (hlIsoRaw !== '' || hlGsRaw !== '') && gammaRaw === '') {{
      throw new Error('fixed half-lives need γ (they are c.o.m. values)');
    }}
    // Only pass the GUI title if the user edited it (the decay plot builds its own default)
    // decay-plot title from the dialog's own box (empty = decay_fit.py's automatic title)
    var titleFull = _d('decayTitleBox').value.trim() || null;
    // Bateman: always the joint fit, mother box frequencies = parent window
    var parentWindow = bateman ? [win.pMin, win.pMax] : null;
    var payload = {{
      fMin:  win.fMin,
      fMax:  win.fMax,
      tMin:  win.tMin,
      tMax:  win.tMax,
      nAvg:  Math.max(1, Math.round(+document.getElementById('nAvg').value || 1)),
      gamma:    gammaRaw !== '' ? +gammaRaw : null,
      noOffset: !bateman && document.getElementById('noOffset').checked,
      logPlot:   !bateman && document.getElementById('logPlotChk').checked,
      residuals: !bateman && document.getElementById('residChk').checked,
      bateman:  bateman,
      hlIso:    (bateman && hlIsoRaw !== '') ? +hlIsoRaw : null,
      hlGs:     (bateman && hlGsRaw  !== '') ? +hlGsRaw  : null,
      titleFull: titleFull,
      parentWindow: parentWindow
    }};
    var r = await fetch('/run-decay-fit', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify(payload)
    }});
    var data = await r.json();
    if (r.ok) {{
      status.textContent = 'Fitting…';
      status.title = '';
      _watchDecayFit(data.id, status);
    }} else {{
      status.textContent = '✗ ' + (data.error || 'Error');
      status.style.color = '#f87171';
    }}
  }} catch(e) {{
    status.textContent = '✗ ' + e.message;
    status.style.color = '#f87171';
  }}
}}

// Follow a started fit until its window is drawn, or report why it died
var _decayWatch = 0;
function _watchDecayFit(id, status) {{
  var token = ++_decayWatch, t0 = Date.now();
  function done(text, color, tip) {{
    status.textContent = text; status.style.color = color; status.title = tip || '';
  }}
  async function poll() {{
    if (token !== _decayWatch) return;            // a newer fit took over the status
    try {{
      var r = await fetch('/decay-fit-status?id=' + encodeURIComponent(id));
      var d = await r.json();
      if (!r.ok) return done('✗ ' + (d.error || 'Error'), '#f87171');
      if (d.state === 'shown') return done('✓ Fit window open', '#4ade80');
      if (d.state === 'failed') {{
        var msg = d.message || '';
        if (msg.length > 140) msg = msg.slice(0, 137) + '…';   // full log in the tooltip
        return done('✗ Decay fit ' + (d.reason || 'failed') + (msg ? ': ' + msg : ''),
                    '#f87171', d.log || '');
      }}
      if (Date.now() - t0 > 600000) return done('Still fitting after 10 min…', '#fbbf24');
      setTimeout(poll, 500);
    }} catch (e) {{ done('✗ ' + e.message, '#f87171'); }}
  }}
  poll();
}}

// 3-D style radio buttons: bars / surface (saved as the "bars3d" pref)
function _set3dStyle() {{
  var bars = document.getElementById('bars3d').checked;
  savePref('bars3d', bars ? '1' : '0');
  _renderSurface();
}}

// Ion names / harmonic labels in titles: m207Tl80+ -> <sup>207m</sup>Tl<sup>80+</sup>,
// h213 -> <i>h</i> = 213 (same rules as format_title_html() in Python)
function formatTitleHtml(text) {{
  var t = String(text).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  t = t.replace(/((?:^|[^\\w>])[mnpqrx]?\\d{{1,3}}[A-Z][a-z]?\\d{{1,3}}\\+)(\\s+)(h\\d+)(?=\\s|$)/g, '$1,$2$3');
  t = t.replace(/(^|[^\\w>])([mnpqrx]?)(\\d{{1,3}})([A-Z][a-z]?)(\\d{{1,3}})\\+/g,
                function(_, pre, iso, a, el, q) {{ return pre + '<sup>' + a + iso + '</sup>' + el + '<sup>' + q + '+</sup>'; }});
  return t.replace(/(^|\\s)h(\\d+)(?=\\s|$)/g, function(_, pre, n) {{ return pre + '<i>h</i> = ' + n; }});
}}

function updateTitle() {{
  var t = formatTitleHtml(document.getElementById('plotTitle').value);
  Plotly.relayout('specPlot', {{'title.text': t}});
  if (_heatReady) Plotly.relayout('heatPlot', {{'title.text': t}});
  saveSession();
}}

// ── 3-D surface: smooth or square bars ──────────────────────────────────────
// Bars are drawn as a "stepped" surface: every grid point becomes a flat
// square top spanning its cell, with near-vertical walls to its neighbours.
// Each axis is expanded to two vertices per cell, inset by a tiny epsilon so
// adjacent cells' vertices stay distinct.
function _cellEdges(c) {{
  var n = c.length;
  if (n === 1) return [c[0] - 0.5, c[0] + 0.5];
  var e = [c[0] - (c[1] - c[0]) / 2];
  for (var i = 0; i < n - 1; i++) e.push((c[i] + c[i + 1]) / 2);
  e.push(c[n - 1] + (c[n - 1] - c[n - 2]) / 2);
  return e;
}}

function _stepAxis(c) {{
  var e = _cellEdges(c), out = [];
  for (var i = 0; i < c.length; i++) {{
    var eps = (e[i + 1] - e[i]) * 1e-4;   // works for increasing or decreasing axes
    out.push(e[i] + eps, e[i + 1] - eps);
  }}
  return out;
}}

// Trace data for the 3-D plot (smooth surface or stepped "bars"), as plain
// values; _renderSurface wraps them for Plotly.update, _initSurface uses them
// directly in Plotly.newPlot.
function _surfaceData(rebinned) {{
  var useDb  = document.getElementById('dbCheck').checked;
  var bars   = document.getElementById('bars3d').checked;
  var zLabel = useDb ? 'Power (dB)' : 'Power (linear)';
  var Zt     = transpose(useDb ? toDb(rebinned.Z) : rebinned.Z);   // (F_desc, T)
  var fDesc  = F_PLOT.slice().reverse();                            // matches Z_RAW's flipped freq
  var upd;
  if (bars) {{
    var t = rebinned.t, nT = t.length, nF = fDesc.length;
    var Zs = [], cd = [];
    for (var j = 0; j < nF; j++) {{
      var zr = new Array(2 * nT), cr = new Array(2 * nT), src = Zt[j];
      for (var i = 0; i < nT; i++) {{
        zr[2 * i] = zr[2 * i + 1] = src[i];
        cr[2 * i] = cr[2 * i + 1] = [t[i], fDesc[j]];
      }}
      Zs.push(zr, zr.slice()); cd.push(cr, cr);
    }}
    upd = {{
      x: _stepAxis(t), y: _stepAxis(fDesc), z: Zs, customdata: cd,
      hovertemplate: 'Time: %{{customdata[0]:.2f}} s<br>Freq: %{{customdata[1]:,.0f}} Hz<br>' +
                     zLabel + ': %{{z:.2f}}<extra></extra>'
    }};
  }} else {{
    upd = {{
      x: rebinned.t, y: fDesc, z: Zt, customdata: null,
      hovertemplate: 'Time: %{{x:.2f}} s<br>Freq: %{{y:,.0f}} Hz<br>' + zLabel + ': %{{z:.2f}}<extra></extra>'
    }};
  }}
  upd.zLabel = zLabel;
  return upd;
}}

function _renderSurface(rebinned) {{
  if (!rebinned) {{
    var n = Math.max(1, Math.round(+document.getElementById('nAvg').value || 1));
    rebinned = rebin(Z_RAW, T_RAW, n);
  }}
  var d = _surfaceData(rebinned);
  // one redraw: data + z-axis title together
  Plotly.update('specPlot', {{
    x: [d.x], y: [d.y], z: [d.z], customdata: [d.customdata], hovertemplate: [d.hovertemplate],
    'colorbar.title.text': [d.zLabel]
  }}, {{'scene.zaxis.title.text': d.zLabel}}, [0]);
}}

// Create the 3-D plot once, already sized, styled and colour-ranged
function _initSurface(h, rebinned, mm) {{
  var d = _surfaceData(rebinned);
  var trace = Object.assign({{}}, SURF_STYLE, {{
    x: d.x, y: d.y, z: d.z, customdata: d.customdata, hovertemplate: d.hovertemplate,
    cmin: mm.min, cmax: mm.max, cauto: false
  }});
  trace.colorbar = Object.assign({{}}, SURF_STYLE.colorbar, {{title: {{text: d.zLabel, side: 'right'}}}});
  trace.colorscale = COLORSCALES[CUR_CS];
  var layout = JSON.parse(JSON.stringify(SURF_LAYOUT));
  layout.height = h;
  layout.scene = layout.scene || {{}};
  layout.scene.zaxis = Object.assign({{}}, layout.scene.zaxis, {{range: [mm.min, mm.max], title: {{text: d.zLabel}}}});
  var gd = document.getElementById('specPlot');
  gd.style.height = h + 'px';
  // trace 1: decay-fit region box, solid edges (filled by _regDraw3d)
  var regTrace = {{type: 'scatter3d', mode: 'lines', x: [], y: [], z: [],
                  connectgaps: false, showlegend: false, hoverinfo: 'skip',
                  line: {{color: '#f97316', width: 5}}}};
  // trace 2: parts of the box cut off by the view, dashed along the plot border
  var regCut = Object.assign({{}}, regTrace, {{line: {{color: '#f97316', width: 5, dash: 'dash'}}}});
  return Plotly.newPlot('specPlot', [trace, regTrace, regCut], layout, {{responsive: true}});
}}

function updateSurface() {{
  var n = Math.max(1, Math.round(+document.getElementById('nAvg').value || 1));
  document.getElementById('nAvg').value = n;
  APPLIED_N = n;
  var useDb = document.getElementById('dbCheck').checked;
  var rebinned = rebin(Z_RAW, T_RAW, n);
  _renderSurface(rebinned);
  resetTimeSliders(rebinned.t);
  var T = Z_RAW.length;
  var discarded = T % n;
  document.getElementById('discardMsg').textContent =
    discarded > 0 ? '⚠︎ ' + discarded + ' of ' + T + ' frames discarded' : '';
  if (_heatReady) _updateHeatmap();
  // Rescale colorscale to the highest power in the (now full-range) window
  _rescaleToWindow(rebinned.Z, rebinned.t);
  // Time sliders were reset → re-sum the 1-D spectrum
  _fetchSpec1d();
  _renderFull();
  saveSession();
}}

// −/+ buttons: move a time slider by one displayed (time-averaged) frame,
// keeping min < max. The step is done here by bin index rather than with the
// browser's stepUp()/stepDown(): those compare positions exactly, so a value
// a rounding error off the slider grid (common with float times) can make a
// click snap to the same bin and appear to do nothing (seen in Firefox).
function stepTime(id, dir) {{
  var s  = document.getElementById(id);
  var mn = +s.min, mx = +s.max, st = +s.step;
  if (!(st > 0)) return;
  var nMax = Math.round((mx - mn) / st);
  var iOld = Math.min(nMax, Math.max(0, Math.round((+s.value - mn) / st)));
  var iNew = Math.min(nMax, Math.max(0, iOld + dir));
  if (iNew === iOld) return;                      // already at the end
  var v = Math.min(mx, mn + iNew * st);
  var other = +document.getElementById(id === 'tMin' ? 'tMax' : 'tMin').value;
  if (id === 'tMin' ? v >= other - st / 2 : v <= other + st / 2) return;   // would cross
  s.value = v;
  update();
}}

function fmt(v, dec) {{
  return dec === 0 ? Math.round(v).toLocaleString('en-GB') : v.toFixed(3);
}}

// Return {{min, max}} of displayed values within the current time/freq window.
// Z is Z_RAW shape (T_ds, F_ds) with freq flipped (index 0 = highest freq).
// F_PLOT is low-to-high, so freq for Z[i][j] = F_PLOT[F_ds - 1 - j].
function windowMinMax(Z, t, tMin, tMax, fMin, fMax, useDb) {{
  var eps = DB_FLOOR;
  var F = Z.length > 0 ? Z[0].length : 0;
  var minVal = Infinity, maxVal = -Infinity;
  for (var i = 0; i < Z.length; i++) {{
    if (t[i] < tMin || t[i] > tMax) continue;
    for (var j = 0; j < F; j++) {{
      var freqJ = F_PLOT[F - 1 - j];
      if (freqJ < fMin || freqJ > fMax) continue;
      var v = Z[i][j];
      if (useDb) v = 10 * Math.log10(Math.max(v, eps));
      if (v < minVal) minVal = v;
      if (v > maxVal) maxVal = v;
    }}
  }}
  if (!isFinite(minVal)) {{ minVal = 0; maxVal = 0; }}
  return {{ min: minVal, max: maxVal }};
}}

// Read slider values, compute visible-window min/max, and push to both plots.
function _rescaleToWindow(Z, t) {{
  var fMin = +document.getElementById('fMin').value;
  var fMax = +document.getElementById('fMax').value;
  var tMin = +document.getElementById('tMin').value;
  var tMax = +document.getElementById('tMax').value;
  var useDb = document.getElementById('dbCheck').checked;
  var mm = windowMinMax(Z, t, tMin, tMax, fMin, fMax, useDb);
  // Colour range (both plots) and 3-D z-axis range: one redraw per plot
  Plotly.update('specPlot', {{ cmin: [mm.min], cmax: [mm.max], cauto: [false] }},
                {{ 'scene.zaxis.range': [mm.min, mm.max] }}, [0]);
  if (_heatReady) Plotly.restyle('heatPlot', {{ zmin: [mm.min], zmax: [mm.max], zauto: [false] }}, [0]);
  _regDrawShape();   // region box: re-clip to the new window, 3-D box to the new z-range
}}

function update() {{
  var fMin = +document.getElementById('fMin').value;
  var fMax = +document.getElementById('fMax').value;
  var tMin = +document.getElementById('tMin').value;
  var tMax = +document.getElementById('tMax').value;
  if (fMin >= fMax || tMin >= tMax) return;
  document.getElementById('fMinVal').textContent = fmt(fMin, 0);
  document.getElementById('fMaxVal').textContent = fmt(fMax, 0);
  document.getElementById('tMinVal').textContent = fmt(tMin, 3);
  document.getElementById('tMaxVal').textContent = fmt(tMax, 3);
  // Freq axis is reversed (high→low), so swap range order for 3-D plotly
  Plotly.relayout('specPlot', {{
    'scene.yaxis.range': [fMax, fMin],
    'scene.xaxis.range': [tMin, tMax]
  }});
  if (_heatReady) {{
    Plotly.relayout('heatPlot', {{
      'xaxis.range': [tMin, tMax],
      'yaxis.range': [fMax, fMin]
    }});
  }}
  // Rescale colorscale to the highest power in the new visible window
  var n = Math.max(1, Math.round(+document.getElementById('nAvg').value || 1));
  var rebinned = rebin(Z_RAW, T_RAW, n);
  _rescaleToWindow(rebinned.Z, rebinned.t);
  // 1-D spectrum: re-sum if the time window moved, otherwise just re-range
  if (_lastT1d[0] !== tMin || _lastT1d[1] !== tMax) {{
    _lastT1d = [tMin, tMax];
    _fetchSpec1d();
  }} else {{
    _renderSpec1d();
  }}
  _renderSim();
  saveSession();
}}
var _lastT1d = [null, null];

// Initialise the 2-D heatmap and 1-D spectrum immediately on load
// Draw each plot once: data, colour range, z-range and height all known up front
(function() {{
  var n = Math.max(1, Math.round(+document.getElementById('nAvg').value || 1));
  var rebinned = rebin(Z_RAW, T_RAW, n);
  var mm = windowMinMax(rebinned.Z, rebinned.t,
                        +document.getElementById('tMin').value, +document.getElementById('tMax').value,
                        +document.getElementById('fMin').value, +document.getElementById('fMax').value,
                        document.getElementById('dbCheck').checked);
  var h = plotHeight();
  _initHeatmap(h, rebinned, mm);
  _initSurface(h, rebinned, mm);
}})();
_fetchSpec1d();
_setFull({_full_open_js});
// the page is built once per load; re-read the saved state so a browser refresh keeps it
fetch('/get-pref?key=fullOpen').then(function(r) {{ return r.json(); }}).then(function(d) {{
  var open = d.value !== '0';
  if (open !== _fullOpen()) {{ _setFull(open); if (open) _renderFull(); }}
}}).catch(function() {{}});
_fetchFull();
fitPlots();   // only redraws if the window changed size meanwhile
_hookCamera();
decayTab('{_decay_tab}', false);
_initRegion({_reg_init_js}, {_reg_init_js_b}, {_reg_init_js_m});
if (document.getElementById('sim_show').checked) runSim();
</script>
</body>
</html>"""

    state = dict(npz_path=_npz_path, t_full=_t_full, p_full=_p_full,
                 f_full=np.asarray(freq, dtype=float),
                 sum_all=np.asarray(_p_full, dtype=float).sum(axis=0))
    return html, state


# ---------------------------------------------------------------------------
# Local server: serves the current page, loads files, runs decay fits
# ---------------------------------------------------------------------------

def load_dataset(path, fmin=None, fmax=None, linear=False, n_avg=1, ion=None, bl=None):
    """Load + crop a file and build its page. Returns (html, state).

    bl: baseline settings {"on", **BrPLS parameters} to use; None = the saved
    ones (and then a failing baseline fit loads the raw data with a warning
    instead of failing the load).
    ion: label added to the title (set by Search ions, e.g. "m186Hf72+ h213").
    If not given, it is kept from the saved session when the same file and
    crop are reloaded (bare re-run, refresh), and dropped otherwise.
    """
    path = os.path.abspath(os.path.expanduser(paths.remap_legacy(path)))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"file not found — {path}")
    if not ion:
        prev = read_prefs().get("session") or {}
        if (isinstance(prev, dict) and prev.get("file") == path
                and _same_crop(prev.get("fMin"), fmin) and _same_crop(prev.get("fMax"), fmax)):
            ion = prev.get("ion") or None
    freq, time, power = load(path)
    lo = float(freq.min()) if fmin is None else float(fmin)
    hi = float(freq.max()) if fmax is None else float(fmax)
    freq_s, power_s = slice_freq(freq, power, lo, hi)
    print(f"Loaded {os.path.basename(path)}: {power_s.shape[1]} freq bins × "
          f"{power_s.shape[0]} time bins  [{lo:,.0f} – {hi:,.0f} Hz]")
    # optional BrPLS baseline removal from every frame, over this window
    explicit = bl is not None
    bl = bl if explicit else baseline_prefs()
    bl_info = {"on": False}
    if bl.get("on"):
        params = baseline.parse_params(bl)
        try:
            power_s = baseline.remove_cached(path, freq_s, power_s, (lo, hi), params)
            bl_info = {"on": True, "params": params, "window": [lo, hi]}
        except Exception as exc:
            if explicit:
                raise
            print(f"Baseline removal failed, showing the raw data: {exc}")
            bl_info = {"on": False, "error": f"baseline removal failed: {exc}"}
    html, state = build_page(freq_s, time, power_s, make_title(path, lo, hi, linear, ion),
                             log_scale=not linear, n_avg=max(1, int(n_avg)),
                             file_path=path, fmin_req=fmin, fmax_req=fmax, ion=ion,
                             bl_info=bl_info)
    state["baseline"] = bl_info if bl_info.get("on") else None
    state["load_args"] = dict(path=path, fmin=fmin, fmax=fmax, ion=ion)
    # Whole (uncropped) file, summed over all frames: used for the simulated-
    # line reference and the whole-file 1-D plot
    state["file_freq"] = np.asarray(freq, dtype=float)
    state["file_sum"]  = np.asarray(power, dtype=float).sum(axis=0)
    state["file_n"]    = int(power.shape[0])
    _PREFS_LOCK.acquire()
    prefs = read_prefs()
    prefs["lastDir"] = os.path.dirname(path)
    # Remember this file/crop for a bare re-run; keep the saved view only if
    # it belongs to this same file and crop
    prefs["session"] = dict(
        file=path,
        fMin=None if fmin is None else float(fmin),
        fMax=None if fmax is None else float(fmax),
        linear=bool(linear),
        nAvg=max(1, int(n_avg)),
        ion=ion or None,
        view=saved_view(path, fmin, fmax),
    )
    try:
        write_prefs(prefs)
    except Exception as exc:
        jsonfile.warn(f"could not save the spectrogram prefs to {PREFS_PATH}: "
                      f"{type(exc).__name__}: {exc}")
    finally:
        _PREFS_LOCK.release()
    return html, state


def serve(html, state):
    import http.server, socket, threading, time, json, webbrowser, datetime
    import subprocess as _subprocess
    from urllib.parse import urlparse, parse_qs

    STATE = dict(state or {})
    STOP  = threading.Event()          # set by the page's Quit button
    # ...or once the last browser tab of this program has been closed
    TABS  = tabwatch.TabTracker(on_last_closed=STOP.set, name="plot_spectrogram.py")
    STATE["html"] = html

    # decay fits started from the page: id -> {proc, ready file, stderr file}
    import tempfile, itertools, shutil
    FIT_DIR  = tempfile.mkdtemp(prefix="asphodel_decayfit_")
    FITS     = {}
    _fit_ids = itertools.count(1)

    def _fit_error(fit):
        """Short reason why a decay fit ended without its window."""
        try:
            with open(fit["err"], errors="replace") as fh:
                lines = [l.strip() for l in fh if l.strip()]
        except OSError:
            lines = []
        rc = fit["proc"].returncode
        if rc < 0:
            import signal as _signal
            try:
                why = f"killed by {_signal.Signals(-rc).name}"
            except ValueError:
                why = f"killed by signal {-rc}"
        else:
            why = f"exit code {rc}" if rc else "ended without opening a window"
        # prefer the last error-looking line (traceback end, Qt message, sys.exit text)
        msg = next((l for l in reversed(lines)
                    if any(k in l for k in ("Error", "error", "failed", "Could not", "Aborted"))),
                   lines[-1] if lines else "")
        return {"reason": why, "message": msg, "log": "\n".join(lines[-15:])}

    # a free port, or the one the home page chose (it opens the tab itself
    # and needs to know where the spectrogram will be)
    try:
        port = int(os.environ.pop("ASPHODEL_PORT", "") or 0)
    except ValueError:
        port = 0
    if not port:
        with socket.socket() as _s:
            _s.bind(("127.0.0.1", 0))
            port = _s.getsockname()[1]

    class _Handler(http.server.BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json"):
            if isinstance(body, (dict, list)):
                body = json.dumps(body)
            if isinstance(body, str) and ctype.startswith("text/html"):
                body = logpanel.inject_html(body, "Spectrogram")   # Log button + panel
            data = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _body(self):
            length = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(length) or b"{}")

        def do_GET(self):
            url = urlparse(self.path)
            q   = parse_qs(url.query)
            if url.path in ("/", "/spectrogram.html"):
                self._send(200, STATE["html"], "text/html; charset=utf-8")

            elif url.path == offline.PLOTLY_JS_URL:
                # plotly.js from the installed plotly package (core/offline.py)
                data = offline.plotly_js_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/javascript")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "max-age=86400")
                self.end_headers()
                self.wfile.write(data)

            elif url.path == "/_aslog":
                # this program's output for the page's Log panel (core/logpanel.py)
                try:
                    since = int(float(q.get("since", ["0"])[0]))
                except (ValueError, OverflowError):
                    since = 0
                self._send(200, logpanel.http_lines(since))

            elif url.path == icon.ICON_URL:
                # browser-tab icon: core/icons/spectrogram.png, else asphodel.png
                self._send(200, icon.png_bytes("spectrogram"), "image/png")

            elif url.path == tabwatch.ALIVE_PATH:
                # regular check from the page (its tab closes once this stops
                # answering; the program stops once no tab checks in)
                TABS.seen(q.get("id", [""])[0])
                self._send(200, {"ok": True})

            elif url.path == "/logbook-settings":
                # file, crop, baseline and PID settings for the "Save to Log Book" popout
                try:
                    self._send(200, {"rows": logbook_settings_rows(STATE)})
                except Exception as exc:
                    self._send(200, {"rows": [], "error": str(exc)})

            elif url.path == "/logbook-ions":
                # existing tabs of the results log book, for the popout's ion box
                try:
                    from core import results_logbook
                    self._send(200, {"ions": results_logbook.ion_names()})
                except Exception as exc:
                    self._send(200, {"ions": [], "error": str(exc)})

            elif url.path == "/get-region":
                # saved decay-fit region mode + drawn box for the loaded file/crop
                sess = read_prefs().get("session") or {}
                view = saved_view(STATE.get("npz_path"), sess.get("fMin"), sess.get("fMax")) \
                    if STATE.get("npz_path") else {}
                # (Standard Fit tab; Bateman Fit tab: daughter box "B", mother box "M")
                prefs = read_prefs()
                rb, rbb = view.get("regBox"), view.get("regBoxB")
                self._send(200, {"mode": prefs.get("decayRegion") or "sliders",
                                 "box": rb if isinstance(rb, list) and len(rb) == 4 else None,
                                 "title": view.get("decayTitle") or "",
                                 "boxB": rbb if isinstance(rbb, list) and len(rbb) == 4 else None,
                                 "boxM": (view.get("regBoxM") if isinstance(view.get("regBoxM"), list)
                                          and len(view.get("regBoxM")) == 4 else None),
                                 "titleB": view.get("decayTitleB") or ""})

            elif url.path == "/decay-fit-status":
                # state of a fit started by /run-decay-fit: running (still fitting),
                # shown (its window was drawn), crashed / no-window (+ reason)
                fit = FITS.get(q.get("id", [""])[0])
                if fit is None:
                    return self._send(404, {"error": "unknown fit"})
                if os.path.exists(fit["ready"]):
                    return self._send(200, {"state": "shown"})
                if fit["proc"].poll() is None:
                    return self._send(200, {"state": "running"})
                self._send(200, {"state": "failed", **_fit_error(fit)})

            elif url.path == "/correction-profiles":
                # profiles for the PID popup's dropdown, the one matching the
                # a, b, c given, and its fit summary + warnings
                def arg(k):
                    return q.get(k, [""])[0]
                profs = corrections.load_profiles()
                r = corrections.resolve(arg("corr_profile"), arg("corr_a"), arg("corr_b"),
                                        arg("corr_c"), profiles=profs)
                p, name = r["profile"], r["name"]
                try:
                    hs = physics.canonical("harmonics", arg("harmonics"))
                except (TypeError, ValueError):
                    hs = None
                info = (["Values typed in (no profile), used for every harmonic."] if p is None else
                        [corrections.describe(p)] + corrections.coefficient_lines(p))
                warns = ([] if p is None else
                         corrections.warnings(p, arg("ref_name"), arg("ref_harmonic"),
                                              arg("gamma_t"), harmonics=hs))
                self._send(200, {
                    "profiles": [{"name": x["name"], "describe": corrections.describe(x),
                                  "a_text": physics._fmt_num(x["a"]),
                                  "b_text": physics._fmt_num(x["b"]),
                                  "c_text": physics._fmt_num(x["c"])} for x in profs],
                    "match": name, "custom": corrections.CUSTOM,
                    "custom_label": corrections.CUSTOM_LABEL, "info": info, "warnings": warns})

            elif url.path == trackpad.URL:
                self._send(200, trackpad.response())

            elif url.path == "/get-pref":
                key = q.get("key", [""])[0]
                if key not in ("fullOpen", "decayRegion"):
                    return self._send(400, {"error": f"unknown pref key: {key!r}"})
                self._send(200, {"value": read_prefs().get(key)})

            elif url.path == "/spectrum-file":
                # whole uncropped file, summed over all time frames
                if STATE.get("file_sum") is None:
                    return self._send(400, {"error": "no file loaded"})
                self._send(200, {"f": STATE["file_freq"].tolist(),
                                 "y": STATE["file_sum"].tolist(),
                                 "n": STATE["file_n"]})

            elif url.path == "/spectrum1d":
                if STATE.get("t_full") is None:
                    return self._send(400, {"error": "no file loaded"})
                try:
                    t_full, p_full = STATE["t_full"], STATE["p_full"]
                    tmin = float(q.get("tmin", ["-inf"])[0])
                    tmax = float(q.get("tmax", ["inf"])[0])
                    i0   = int(np.searchsorted(t_full, tmin, side="left"))
                    i1   = int(np.searchsorted(t_full, tmax, side="left"))
                    n    = max(i1 - i0, 0)
                    y    = p_full[i0:i1].sum(axis=0) if n > 0 else np.zeros(p_full.shape[1])
                    self._send(200, {
                        "y":  np.asarray(y, dtype=float).tolist(),
                        "n":  n,
                        "t0": float(t_full[i0])     if n > 0 else 0.0,
                        "t1": float(t_full[i1 - 1]) if n > 0 else 0.0,
                    })
                except Exception as exc:
                    self._send(500, {"error": str(exc)})

            elif url.path == "/list-dir":
                try:
                    d = q.get("path", [""])[0].strip() or read_prefs().get("lastDir") or paths.spectrograms()
                    if not d:            # no spectrogram folder connected (and none used yet)
                        try:
                            conn = [{"name": f["name"], "path": f["path"],
                                     "kind": paths.DATA_KINDS.get(f["kind"], f["kind"]),
                                     "ok": os.path.isdir(f["path"])} for f in paths.connected("data")]
                        except Exception:
                            conn = []
                        self._send(200, {"path": "", "parent": "", "dirs": [], "files": [],
                                         "connected": conn,
                                         "note": paths.no_data_folder("spectrograms")})
                        return
                    d = os.path.abspath(os.path.expanduser(d))
                    if os.path.isfile(d):
                        d = os.path.dirname(d)
                    if not os.path.isdir(d):
                        raise FileNotFoundError(f"not a folder: {d}")
                    dirs, files = [], []
                    for name in sorted(os.listdir(d), key=str.lower):
                        if name.startswith("."):
                            continue
                        full = os.path.join(d, name)
                        if os.path.isdir(full):
                            dirs.append(name)
                        elif name.lower().endswith(".npz"):
                            st = os.stat(full)
                            files.append({
                                "name":  name,
                                "size":  st.st_size,
                                "mtime": datetime.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M"),
                            })
                    try:     # the experiment's connected data folders (home page → Folders)
                        conn = [{"name": f["name"], "path": f["path"],
                                 "kind": paths.DATA_KINDS.get(f["kind"], f["kind"]),
                                 "ok": os.path.isdir(f["path"])} for f in paths.connected("data")]
                    except Exception:
                        conn = []
                    self._send(200, {"path": d, "parent": os.path.dirname(d),
                                     "dirs": dirs, "files": files, "connected": conn})
                except Exception as exc:
                    self._send(400, {"error": str(exc)})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if urlparse(self.path).path == tabwatch.BYE_PATH:
                # the page's tab is being closed or reloaded
                TABS.bye(parse_qs(urlparse(self.path).query).get("id", [""])[0])
                self._send(200, {"ok": True})
            elif self.path == "/_aslog/open":
                self._send(200, logpanel.http_open(self._body()))
            elif self.path == "/logbook-save":
                try:
                    key = logbook_save(self._body(), STATE.get("npz_path"))
                    self._send(200, {"ion": key})
                except Exception as exc:
                    self._send(400, {"error": str(exc)})
            elif self.path == "/load":
                try:
                    b = self._body()
                    html_new, st = load_dataset(b.get("file", ""), b.get("fMin"), b.get("fMax"),
                                                bool(b.get("linear")), int(b.get("nAvg") or 1),
                                                ion=(str(b["ion"]).strip() or None) if b.get("ion") else None)
                    STATE.clear(); STATE.update(st); STATE["html"] = html_new
                    self._send(200, {"status": "loaded"})
                except Exception as exc:
                    self._send(400, {"error": str(exc)})

            elif self.path == "/baseline":
                # save the baseline settings and reload the current file with them
                try:
                    b = self._body()
                    la = STATE.get("load_args")
                    if not la:
                        raise ValueError("no file loaded")
                    bl = {"on": bool(b.get("on")), **baseline.parse_params(b)}
                    html_new, st = load_dataset(la["path"], la["fmin"], la["fmax"],
                                                bool(b.get("linear")), int(b.get("nAvg") or 1),
                                                ion=la["ion"], bl=bl)
                    with _PREFS_LOCK:
                        prefs = read_prefs()
                        prefs["baseline"] = bl
                        write_prefs(prefs)
                    STATE.clear(); STATE.update(st); STATE["html"] = html_new
                    self._send(200, {"status": "loaded"})
                except Exception as exc:
                    self._send(400, {"error": str(exc)})

            elif self.path == "/run-decay-fit":
                try:
                    body = self._body()
                    if not STATE.get("npz_path"):
                        raise ValueError("no file loaded")
                    if not os.path.isfile(DECAY_PATH):
                        raise FileNotFoundError(f"decay_fit.py not found at {DECAY_PATH}")
                    cmd = [
                        sys.executable, DECAY_PATH,
                        STATE["npz_path"],
                        str(body["fMin"]), str(body["fMax"]),
                        "--tmin",     str(body["tMin"]),
                        "--tmax",     str(body["tMax"]),
                        "--time-avg", str(int(body.get("nAvg", 1))),
                    ]
                    if STATE.get("baseline"):
                        # same corrected frames as the page: BrPLS over the loaded window
                        blw = STATE["baseline"]["window"]
                        cmd += ["--baseline", json.dumps(STATE["baseline"]["params"]),
                                "--baseline-window", repr(float(blw[0])), repr(float(blw[1]))]
                    if body.get("gamma") is not None:
                        cmd += ["--gamma", str(body["gamma"])]
                    if body.get("noOffset"):
                        cmd += ["--no-offset"]
                    if body.get("errCoeffs") is not None:
                        ea, eb = body["errCoeffs"]
                        cmd += ["--err-coeffs", str(float(ea)), str(float(eb))]
                    if body.get("logPlot") and not body.get("parentWindow"):
                        cmd += ["--log-plot"]
                    if body.get("residuals") and not body.get("parentWindow"):
                        cmd += ["--residuals"]
                    if body.get("unweighted"):
                        cmd += ["--unweighted"]
                    if body.get("bateman"):
                        cmd += ["--bateman"]
                        if body.get("hlIso") is not None:
                            cmd += ["--halflife-isomer", str(float(body["hlIso"]))]
                        if body.get("hlGs") is not None:
                            cmd += ["--halflife-groundstate", str(float(body["hlGs"]))]
                        if body.get("parentWindow") is not None:
                            pw0, pw1 = body["parentWindow"]
                            cmd += ["--parent-window", str(float(pw0)), str(float(pw1))]
                    skip = body.get("skip") or [0, 0]
                    if int(skip[0]) > 0 or int(skip[1]) > 0:
                        cmd += ["--skip", str(int(skip[0])), str(int(skip[1]))]
                    if body.get("titleFull"):
                        cmd += ["--title-full", str(body["titleFull"])]
                    elif body.get("title"):
                        cmd += ["--title", str(body["title"])]
                    fid   = str(next(_fit_ids))
                    ready = os.path.join(FIT_DIR, f"fit{fid}.ready")
                    err   = os.path.join(FIT_DIR, f"fit{fid}.err")
                    env   = dict(os.environ, ASPHODEL_READY_FILE=ready)
                    # stderr to a file (read back if the fit dies); stdout stays
                    # on the terminal as before
                    with open(err, "wb") as errfh:
                        # stdout: this program's original one (the fit keeps its own
                        # log, core/logcapture.py)
                        proc = _subprocess.Popen(cmd, env=env, stderr=errfh,
                                                 stdout=logcapture.child_stdio().get("stdout"))
                    FITS[fid] = {"proc": proc, "ready": ready, "err": err}
                    self._send(200, {"status": "launched", "id": fid})
                except Exception as exc:
                    self._send(500, {"error": str(exc)})

            elif self.path == "/region-ions":
                # simulated lines (current PID settings) inside [fMin, fMax],
                # T1/2-cut ions included (greyed in the list), nearest to the
                # window centre first
                try:
                    b = self._body()
                    f0, f1 = sorted((float(b["fMin"]), float(b["fMax"])))
                    if STATE.get("file_sum") is None:
                        raise ValueError("no file loaded")
                    res = compute_sim_lines(STATE["file_freq"], STATE["file_sum"], sim_prefs())
                    c = 0.5 * (f0 + f1)
                    hits = [l for l in res["lines"] if f0 <= l["f"] <= f1]
                    hits.sort(key=lambda l: abs(l["f"] - c))
                    for l in hits:
                        l.pop("label", None)
                        l["df"] = l["f"] - c
                    self._send(200, {"hits": hits[:200], "n": len(hits)})
                except ImportError as exc:
                    self._send(400, {"error": f"{exc} (needs pandas, and core/nubase_reader.py + "
                                              f"core/nubase_4.mas20)"})
                except Exception as exc:
                    self._send(400, {"error": str(exc)})

            elif self.path == "/search-ion":
                # simulated lines (current PID settings, whole file) whose ion
                # name contains the query, case-insensitive
                try:
                    q = str(self._body().get("q", "")).strip().lower()
                    if not q:
                        raise ValueError("enter an ion, e.g. 186Hf")
                    if STATE.get("file_sum") is None:
                        raise ValueError("no file loaded")
                    res = compute_sim_lines(STATE["file_freq"], STATE["file_sum"], sim_prefs())
                    hits = sorted((l for l in res["lines"] if q in l["ion"].lower()),
                                  key=lambda l: l["f"])
                    for l in hits:
                        l.pop("label", None)
                    self._send(200, {"file": STATE.get("npz_path", ""), "hits": hits[:300],
                                     "n": len(hits), "ref_freq": res["ref_freq"]})
                except ImportError as exc:
                    self._send(400, {"error": f"{exc} (needs pandas, and core/nubase_reader.py + "
                                              f"core/nubase_4.mas20)"})
                except Exception as exc:
                    self._send(400, {"error": str(exc)})

            elif self.path == "/simulate":
                try:
                    body = self._body()
                    with _PREFS_LOCK:
                        prefs = read_prefs()
                        # keep only the values that differ from the shared physics
                        prefs["sim"] = physics.strip_shared(
                            {k: body[k] for k in SIM_DEFAULTS if k in body})
                        write_prefs(prefs)
                    if body.get("save_only"):
                        return self._send(200, {"status": "saved"})
                    if STATE.get("file_sum") is None:
                        raise ValueError("no file loaded")
                    # reference + lines over the whole uncropped file
                    self._send(200, compute_sim_lines(STATE["file_freq"], STATE["file_sum"], body))
                except ImportError as exc:
                    self._send(400, {"error": f"{exc} (needs pandas, and core/nubase_reader.py + "
                                              f"core/nubase_4.mas20)"})
                except Exception as exc:
                    self._send(400, {"error": str(exc)})

            elif self.path == "/quit":
                # Quit button: reply first, then let the main loop shut down
                self._send(200, {"status": "stopping"})
                STOP.set()

            elif self.path == "/save-session":
                # GUI view state for the current file (see saveSession() in the page)
                try:
                    body = self._body()
                    with _PREFS_LOCK:
                        prefs = read_prefs()
                        sess = prefs.get("session")
                        if (isinstance(sess, dict) and STATE.get("npz_path")
                                and sess.get("file") == STATE["npz_path"]):
                            if isinstance(body.get("view"), dict):
                                new_view = dict(body["view"])
                                old_view = sess.get("view") if isinstance(sess.get("view"), dict) else {}
                                for k in ("regBox", "decayTitle", "regBoxB", "decayTitleB", "regBoxM"):   # page not ready yet
                                    if k not in new_view and k in old_view:
                                        new_view[k] = old_view[k]
                                sess["view"] = new_view
                            if "linear" in body:
                                sess["linear"] = bool(body["linear"])
                            if body.get("nAvg") is not None:
                                sess["nAvg"] = max(1, int(body["nAvg"]))
                            write_prefs(prefs)
                    self._send(200, {"status": "saved"})
                except Exception as exc:
                    self._send(500, {"error": str(exc)})

            elif self.path == "/save-pref":
                try:
                    body = self._body()
                    pref_key = body.get("key")
                    pref_val = body.get("value", "")
                    if pref_key not in ("gamma", "hlIso", "hlGs", "bateman", "decayTab",
                                        "parentMin", "parentMax", "bars3d", "fullOpen", "colorscale", "decayRegion",
                                        "logPlot", "residuals", "srchWidth",
                                        "gammaB"):
                        raise ValueError(f"unknown pref key: {pref_key!r}")
                    with _PREFS_LOCK:
                        prefs = read_prefs()
                        if pref_val == "" or pref_val is None:
                            prefs.pop(pref_key, None)
                        else:
                            prefs[pref_key] = pref_val
                        write_prefs(prefs)
                    self._send(200, {"status": "saved"})
                except Exception as exc:
                    self._send(500, {"error": str(exc)})
            else:
                self._send(404, {"error": "not found"})

        def log_message(self, *args):
            pass  # silence request logs

    # Threaded so a spare browser connection can never block the page load
    _srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    threading.Thread(target=_srv.serve_forever, daemon=True).start()

    url = f"http://127.0.0.1:{port}/spectrogram.html"
    print(f"Serving spectrogram GUI at {url}  (Ctrl-C to quit)")
    if not browser.tab_opened_by_launcher():   # (the home page opened it already)
        browser.open_url(url)    # suite's Firefox window, else default browser

    try:
        while not STOP.is_set():
            time.sleep(0.2)
        time.sleep(0.3)                # let the reply / last saves finish
        print("Stopped from the browser (Quit).")
    except KeyboardInterrupt:
        pass
    _srv.shutdown()
    shutil.rmtree(FIT_DIR, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.time_avg is not None and args.time_avg < 1:
        sys.exit("Error: --time-avg N requires N ≥ 1.")
    if (args.freq_min is None) != (args.freq_max is None):
        sys.exit("Error: give both freq_min and freq_max, or neither (full range).")

    if args.file is not None:
        try:
            html, state = load_dataset(args.file, args.freq_min, args.freq_max,
                                       linear=bool(args.linear),
                                       n_avg=args.time_avg or 1)
        except Exception as exc:
            sys.exit(f"Error: {exc}")
        return serve(html, state)

    # Bare run → reopen the last session if there is one
    sess = {} if args.new else (read_prefs().get("session") or {})
    if isinstance(sess, dict) and sess.get("file") and os.path.isfile(sess["file"]):
        linear = bool(sess.get("linear")) if args.linear is None else True
        n_avg  = int(sess.get("nAvg") or 1) if args.time_avg is None else args.time_avg
        print(f"Reopening last session: {sess['file']}  (use --new for the file chooser)")
        try:
            html, state = load_dataset(sess["file"], sess.get("fMin"), sess.get("fMax"),
                                       linear=linear, n_avg=n_avg)
            return serve(html, state)
        except Exception as exc:
            print(f"Could not reopen last session ({exc}); opening the file chooser.")
    elif isinstance(sess, dict) and sess.get("file"):
        print(f"Last file no longer exists ({sess['file']}); opening the file chooser.")

    # Nothing to reopen → open the GUI on the file chooser
    serve(build_landing_page(), {})


if __name__ == "__main__":
    main()
