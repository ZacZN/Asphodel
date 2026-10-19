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
  to .asphodel_prefs.json as you change them. Running with no arguments
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
    data = np.load(path)
    freq  = data['freq'].ravel()   # (1, F) → (F,)
    time  = data['time'].ravel()   # (T, 1) → (T,)
    power = data['power']          # (T, F)
    data.close()

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
PREFS_PATH  = os.path.join(_HERE, ".asphodel_prefs.json")
DECAY_PATH  = os.path.join(_HERE, "decay_fit.py")


import threading
_PREFS_LOCK = threading.RLock()   # serialise read-modify-write of the prefs file


def read_prefs():
    import json
    try:
        with open(PREFS_PATH, "r", encoding="utf-8") as fh:
            return json.loads(fh.read())
    except Exception:
        return {}


def write_prefs(prefs):
    import json
    tmp = PREFS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(prefs))
    os.replace(tmp, PREFS_PATH)   # atomic: never leaves a half-written prefs file


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


def make_title(path, fmin, fmax, linear=False):
    return (f"{os.path.basename(path)}   "
            f"[{fmin:,.0f} – {fmax:,.0f} Hz]   "
            f"({'linear' if linear else 'dB'})")


# Plain strings (not f-strings) so CSS/JS braces need no escaping.
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
    window.location.href = '/spectrogram.html?t=' + Date.now();
  } catch (e) { ldStatus('✗ ' + e.message, 'err'); }
}
"""


def loader_parts(file_path=None, fmin=None, fmax=None):
    """Return (html, js) for the loader row, pre-filled."""
    import json
    prefs = read_prefs()
    if file_path:
        start_dir = os.path.dirname(os.path.abspath(file_path))
    else:
        start_dir = prefs.get("lastDir") or os.getcwd()
    esc = lambda v: str(v).replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")
    html = (LOADER_HTML
            .replace("__PATH__", esc(os.path.abspath(file_path)) if file_path else "")
            .replace("__FMIN__", "" if fmin is None else esc(repr(float(fmin))))
            .replace("__FMAX__", "" if fmax is None else esc(repr(float(fmax)))))
    js = LOADER_JS.replace("__INIT__", json.dumps({"dir": start_dir}).replace("</", "<\\/"))
    return html, js


def build_landing_page():
    html, js = loader_parts()
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Asphodel — open spectrogram</title>
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

def build_page(freq, time, power, title, log_scale=True, n_avg=1, file_path=None,
               fmin_req=None, fmax_req=None):
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
    _saved_err_a = str(_prefs.get("errA", ""))
    _saved_err_b = str(_prefs.get("errB", ""))
    _saved_hl_iso = str(_prefs.get("hlIso", ""))
    _saved_hl_gs  = str(_prefs.get("hlGs", ""))
    _saved_skip0  = str(_prefs.get("skipStart", ""))
    _saved_skip1  = str(_prefs.get("skipEnd", ""))
    _saved_pmin   = str(_prefs.get("parentMin", ""))
    _saved_pmax   = str(_prefs.get("parentMax", ""))
    _chk_bateman    = "checked" if _prefs.get("bateman")    else ""
    _chk_unweighted = "checked" if _prefs.get("unweighted") else ""
    _chk_bars3d     = "" if _prefs.get("bars3d") == "0" else "checked"   # bars by default

    # Saved view for this file + crop (sliders, title, camera, 1-D overlay)
    _view = saved_view(file_path, fmin_req, fmax_req)
    _default_title = title
    _vt = _view.get("title")
    if isinstance(_vt, str) and _vt.strip():
        title = _vt
    _chk_show1d = "" if _view.get("show1d") is False else "checked"

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

    eps = np.finfo(float).tiny
    z_plot_init = (10.0 * np.log10(np.maximum(z_init, eps)) if log_scale else z_init).T

    fig = go.Figure(data=[go.Surface(
        x=t_init,
        y=f_plot,
        z=z_plot_init,
        colorscale="Inferno",
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
        title=dict(text=title, x=0.5, xanchor="center", font_size=14),
        scene=_scene,
        margin=dict(l=0, r=0, t=60, b=0),
        height=680,
    )

    # Embed raw (downsampled) data as JSON for JS re-rendering
    z_raw_json = json.dumps(z_ds.tolist())
    t_raw_json = json.dumps(t_ds.tolist())
    f_raw_json = json.dumps(f_ds.tolist())   # freq array, low-to-high (for 2-D heatmap)
    db_checked = "checked" if log_scale else ""
    n_avg_init = n_avg

    plot_div = fig.to_html(full_html=False, include_plotlyjs="cdn", div_id="specPlot",
                           config={"responsive": True})
    _default_title_js = json.dumps(_default_title).replace("</", "<\\/")
    title_html = title.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title_html}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; padding: 8px; background: #111827; color: #e5e7eb;
         font-family: system-ui, sans-serif; font-size: 13px; }}
  #controls {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px;
               background: #1f2937; border-radius: 8px; padding: 12px 16px;
               margin-bottom: 6px; }}
  .title-row {{ grid-column: 1 / -1; display: flex; align-items: center;
                gap: 10px; margin-bottom: 4px; }}
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
  .row span {{ width: 110px; text-align: right; color: #d1d5db; font-size: 11px;
               font-variant-numeric: tabular-nums; }}
  .opts-row {{ grid-column: 1 / -1; display: flex; align-items: center;
               gap: 24px; padding-top: 8px; border-top: 1px solid #374151;
               flex-wrap: wrap; }}
  .opt-grp {{ display: flex; align-items: center; gap: 6px; }}
  .opt-grp span {{ font-size: 12px; font-weight: 600; text-transform: uppercase;
                   letter-spacing: .06em; color: #f97316; margin-right: 2px; }}
  .opt-grp label {{ color: #9ca3af; }}
  .opt-grp input[type=checkbox] {{ accent-color: #f97316; width: 14px; height: 14px;
                                    cursor: pointer; }}
  .opt-grp input[type=number] {{ width: 60px; background: #374151;
                                  border: 1px solid #4b5563; border-radius: 4px;
                                  color: #e5e7eb; padding: 3px 6px; font-size: 13px; }}
  .opt-grp input[type=number]:focus {{ outline: none; border-color: #f97316; }}
  .opt-grp button {{ background: #f97316; border: none; border-radius: 4px;
                     color: #111827; font-size: 12px; font-weight: 700;
                     padding: 4px 12px; cursor: pointer; }}
  .opt-grp button:hover {{ background: #fb923c; }}
  .opt-grp input:disabled {{ opacity: .35; cursor: not-allowed; }}
  .opt-grp input:disabled + label {{ opacity: .35; }}
  .opt-grp button:disabled {{ opacity: .35; cursor: not-allowed; }}
  #discardMsg {{ font-size: 11px; color: #fbbf24; margin-left: 4px; }}
  #decayStatus {{ font-size: 11px; margin-left: 4px; }}
  #plots {{ display: flex; gap: 8px; align-items: flex-start; }}
  #heatDiv {{ flex: 1; min-width: 0; }}
  #surfDiv {{ flex: 1; min-width: 0; }}
{LOADER_CSS}
</style>
</head>
<body>
<div id="controls">
{loader_html}
  <div class="title-row">
    <label>Title</label>
    <input type="text" id="plotTitle" value="{title_html}" oninput="updateTitle()">
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
      <span id="tMinVal">{t_sel_min:.3f}</span>
    </div>
    <div class="row">
      <label>Max</label>
      <input type="range" id="tMax" min="{t_init_min}" max="{t_init_max}"
             step="{dt_init}" value="{t_sel_max}" oninput="update()">
      <span id="tMaxVal">{t_sel_max:.3f}</span>
    </div>
  </div>
  <div class="opts-row">
    <div class="opt-grp">
      <span>Scale</span>
      <input type="checkbox" id="dbCheck" {db_checked} onchange="updateSurface()">
      <label for="dbCheck">dB</label>
    </div>
    <div class="opt-grp">
      <span>3D</span>
      <input type="checkbox" id="bars3d" {_chk_bars3d}
             onchange="savePref('bars3d', this.checked ? '1' : '0'); _renderSurface()">
      <label for="bars3d" title="Draw each point of the 3-D view as a flat-topped square bar (like the heatmap cells) instead of a smooth surface">bars</label>
    </div>
    <div class="opt-grp">
      <span>1D overlay</span>
      <input type="checkbox" id="show1d" {_chk_show1d} onchange="_renderSpec1d(); saveSession()">
      <label for="show1d">show</label>
    </div>
    <div class="opt-grp">
      <span>Time avg</span>
      <input type="number" id="nAvg" min="1" step="1" value="{n_avg_init}">
      <button onclick="updateSurface()">Apply</button>
      <span id="discardMsg"></span>
    </div>
  </div>
  <div class="opts-row decay-row">
    <div class="opt-grp"><span style="color:#e5e7eb;">Decay fit</span></div>
    <div class="opt-grp">
      <span>Model</span>
      <input type="checkbox" id="batemanChk" {_chk_bateman}
             onchange="savePref('bateman', this.checked ? '1' : ''); _syncDecayControls()">
      <label for="batemanChk" title="Bateman two-member chain A→B→C instead of a single exponential">Bateman</label>
      <input type="checkbox" id="noOffset">
      <label for="noOffset" title="Simple decay only: fit A·exp(-t/τ) with no constant background">no offset</label>
    </div>
    <div class="opt-grp" title="Bateman only. Frequency window of the PARENT peak (Hz). When set, parent and daughter are fitted together with a shared τ_A: parent = A·e^(-t/τ_A) + C_A (with offset), daughter = the frequency-slider window, Bateman growth with no offset. Leave blank for a single-window Bateman fit.">
      <span>Parent window</span>
      <input type="number" id="parentMin" step="any" placeholder="f min"
             style="width:84px;" value="{_saved_pmin}" oninput="savePref('parentMin', this.value)">
      <input type="number" id="parentMax" step="any" placeholder="f max"
             style="width:84px;" value="{_saved_pmax}" oninput="savePref('parentMax', this.value)">
      <button id="parentFromSliders" onclick="_parentFromSliders()"
              title="Copy the current frequency-slider range into the parent window">← sliders</button>
    </div>
    <div class="opt-grp" title="Known c.o.m.-frame half-lives (s), converted to lab frame with γ. Fixes the corresponding τ in the Bateman fit. Requires γ.">
      <span>Fixed t½ (s)</span>
      <input type="number" id="hlIso" min="0" step="any" placeholder="isomer"
             style="width:72px;" value="{_saved_hl_iso}" oninput="savePref('hlIso', this.value)">
      <input type="number" id="hlGs" min="0" step="any" placeholder="g.s."
             style="width:72px;" value="{_saved_hl_gs}" oninput="savePref('hlGs', this.value)">
    </div>
    <div class="opt-grp">
      <span>γ</span>
      <input type="number" id="gammaVal" min="0" step="0.01"
             placeholder="γ" title="Lorentz factor γ (optional; required for fixed half-lives)"
             style="width:64px;" value="{_saved_gamma}"
             oninput="savePref('gamma', this.value)">
    </div>
    <div class="opt-grp">
      <span title="Error bar on each point: σ = a·I + b. Leave both blank to estimate a, b automatically from the fit residuals.">σ =</span>
      <input type="number" id="errA" step="any"
             placeholder="auto" title="Error-bar slope a in σ = a·I + b (single-frame). Blank = estimate from residuals"
             style="width:64px;" value="{_saved_err_a}"
             oninput="savePref('errA', this.value)">
      <label>·I +</label>
      <input type="number" id="errB" step="any"
             placeholder="auto" title="Error-bar intercept b in σ = a·I + b (single-frame). Blank = estimate from residuals"
             style="width:64px;" value="{_saved_err_b}"
             oninput="savePref('errB', this.value)">
      <input type="checkbox" id="unweightedChk" {_chk_unweighted}
             onchange="savePref('unweighted', this.checked ? '1' : ''); _syncDecayControls()">
      <label for="unweightedChk" title="Single unweighted fit, no error bars">unweighted</label>
    </div>
    <div class="opt-grp" title="Skip the first X and last Y time bins (applied before the time-slider window)">
      <span>Skip bins</span>
      <input type="number" id="skipStart" min="0" step="1" placeholder="first"
             style="width:56px;" value="{_saved_skip0}" oninput="savePref('skipStart', this.value)">
      <input type="number" id="skipEnd" min="0" step="1" placeholder="last"
             style="width:56px;" value="{_saved_skip1}" oninput="savePref('skipEnd', this.value)">
    </div>
    <div class="opt-grp">
      <button onclick="runDecayFit()">Run</button>
      <span id="decayStatus"></span>
    </div>
  </div>
</div>
<div id="plots">
  <div id="heatDiv"><div id="heatPlot"></div></div>
  <div id="surfDiv">{plot_div}</div>
</div>
<script>
{loader_js}
const Z_RAW = {z_raw_json};
const T_RAW = {t_raw_json};
const F_PLOT = {f_raw_json};  // freq array, low-to-high (2-D heatmap y-axis)
const F_1D   = {f_1d_json};   // full-resolution freq array for the 1-D spectrum
const DEFAULT_TITLE = {_default_title_js};
var APPLIED_N = {n_avg_init};
var LAST_CAMERA = {json.dumps(_camera)};

var _heatReady = true;

// Inferno colorscale defined explicitly so it works regardless of plotly.js version
const INFERNO = [
  [0.0,  'rgb(0,0,4)'],    [0.1,  'rgb(27,11,65)'],
  [0.2,  'rgb(74,12,107)'], [0.3,  'rgb(119,28,109)'],
  [0.4,  'rgb(163,47,92)'], [0.5,  'rgb(200,70,67)'],
  [0.6,  'rgb(229,103,37)'],[0.7,  'rgb(246,142,9)'],
  [0.8,  'rgb(252,185,22)'],[0.9,  'rgb(252,226,83)'],
  [1.0,  'rgb(252,255,164)']
];

function toDb(Z) {{
  var eps = 5e-324;
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

function _initHeatmap() {{
  var n = Math.max(1, Math.round(+document.getElementById('nAvg').value || 1));
  var useDb = document.getElementById('dbCheck').checked;
  var rebinned = rebin(Z_RAW, T_RAW, n);
  var zLabel = useDb ? 'Power (dB)' : 'Power (linear)';
  Plotly.newPlot('heatPlot', [{{
    type: 'heatmap',
    x: rebinned.t,
    y: F_PLOT,
    z: _heat2dZ(rebinned.Z, useDb),
    colorscale: INFERNO,
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
  }}], {{
    template: {{}},
    title: {{ text: document.getElementById('plotTitle').value,
              x: 0.5, xanchor: 'center', y: 0.985, yanchor: 'top',
              font: {{ size: 14, color: '#e5e7eb' }} }},
    xaxis2: {{ overlaying: 'x', side: 'top', title: {{ text: '', standoff: 4 }},
               showgrid: false, zeroline: false,
               color: 'rgb(56,189,248)', tickfont: {{ size: 10 }} }},
    xaxis: {{ title: 'Time (s)',
              range: [+document.getElementById('tMin').value, +document.getElementById('tMax').value],
              color: '#9ca3af', gridcolor: '#374151', zerolinecolor: '#374151' }},
    yaxis: {{ title: 'Frequency (Hz)', tickformat: ',d', autorange: 'reversed',
              range: [+document.getElementById('fMax').value, +document.getElementById('fMin').value],
              color: '#9ca3af', gridcolor: '#374151', zerolinecolor: '#374151' }},
    paper_bgcolor: '#111827', plot_bgcolor: '#1f2937',
    font: {{ color: '#e5e7eb' }},
    margin: {{ l: 90, r: 20, t: 110, b: 60 }},
    height: 680,
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
  var pLabel = useDb ? 'Summed power (dB)' : 'Summed power (linear)';
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
function _syncDecayControls() {{
  var bateman    = document.getElementById('batemanChk').checked;
  var unweighted = document.getElementById('unweightedChk').checked;
  document.getElementById('noOffset').disabled = bateman;   // Bateman always fits C
  document.getElementById('hlIso').disabled    = !bateman;
  document.getElementById('hlGs').disabled     = !bateman;
  document.getElementById('parentMin').disabled = !bateman;
  document.getElementById('parentMax').disabled = !bateman;
  document.getElementById('parentFromSliders').disabled = !bateman;
  document.getElementById('errA').disabled     = unweighted;
  document.getElementById('errB').disabled     = unweighted;
}}

function _parentFromSliders() {{
  var fMin = document.getElementById('fMin').value;
  var fMax = document.getElementById('fMax').value;
  document.getElementById('parentMin').value = fMin;
  document.getElementById('parentMax').value = fMax;
  savePref('parentMin', fMin);
  savePref('parentMax', fMax);
}}

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
  var status = document.getElementById('decayStatus');
  status.textContent = 'Launching…';
  status.style.color = '#fbbf24';
  try {{
    var gammaRaw = document.getElementById('gammaVal').value.trim();
    var bateman  = document.getElementById('batemanChk').checked;
    var unweighted = document.getElementById('unweightedChk').checked;
    var hlIsoRaw = document.getElementById('hlIso').value.trim();
    var hlGsRaw  = document.getElementById('hlGs').value.trim();
    var skip0    = Math.max(0, Math.round(+document.getElementById('skipStart').value || 0));
    var skip1    = Math.max(0, Math.round(+document.getElementById('skipEnd').value || 0));
    if (bateman && (hlIsoRaw !== '' || hlGsRaw !== '') && gammaRaw === '') {{
      throw new Error('fixed half-lives need γ (they are c.o.m. values)');
    }}
    // Only pass the GUI title if the user edited it (the decay plot builds its own default)
    var guiTitle = document.getElementById('plotTitle').value.trim();
    var titleArg = (guiTitle !== '' && guiTitle !== DEFAULT_TITLE) ? guiTitle : null;
    var pMinRaw = document.getElementById('parentMin').value.trim();
    var pMaxRaw = document.getElementById('parentMax').value.trim();
    var parentWindow = null;
    if (bateman && (pMinRaw !== '' || pMaxRaw !== '')) {{
      if (pMinRaw === '' || pMaxRaw === '' || +pMinRaw >= +pMaxRaw) {{
        throw new Error('parent window needs f min < f max');
      }}
      parentWindow = [+pMinRaw, +pMaxRaw];
    }}
    var errARaw  = document.getElementById('errA').value.trim();
    var errBRaw  = document.getElementById('errB').value.trim();
    // Blank box counts as 0; both blank → unweighted fit
    var errCoeffs = (errARaw === '' && errBRaw === '') ? null
                  : [errARaw === '' ? 0 : +errARaw, errBRaw === '' ? 0 : +errBRaw];
    var payload = {{
      fMin:  +document.getElementById('fMin').value,
      fMax:  +document.getElementById('fMax').value,
      tMin:  +document.getElementById('tMin').value,
      tMax:  +document.getElementById('tMax').value,
      nAvg:  Math.max(1, Math.round(+document.getElementById('nAvg').value || 1)),
      gamma:    gammaRaw !== '' ? +gammaRaw : null,
      noOffset: !bateman && document.getElementById('noOffset').checked,
      errCoeffs: unweighted ? null : errCoeffs,
      unweighted: unweighted,
      bateman:  bateman,
      hlIso:    (bateman && hlIsoRaw !== '') ? +hlIsoRaw : null,
      hlGs:     (bateman && hlGsRaw  !== '') ? +hlGsRaw  : null,
      skip:     [skip0, skip1],
      title:    titleArg,
      parentWindow: parentWindow
    }};
    var r = await fetch('/run-decay-fit', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify(payload)
    }});
    var data = await r.json();
    if (r.ok) {{
      status.textContent = '✓ Launched';
      status.style.color = '#4ade80';
    }} else {{
      status.textContent = '✗ ' + (data.error || 'Error');
      status.style.color = '#f87171';
    }}
  }} catch(e) {{
    status.textContent = '✗ ' + e.message;
    status.style.color = '#f87171';
  }}
}}

function updateTitle() {{
  var t = document.getElementById('plotTitle').value;
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

function _renderSurface(rebinned) {{
  if (!rebinned) {{
    var n = Math.max(1, Math.round(+document.getElementById('nAvg').value || 1));
    rebinned = rebin(Z_RAW, T_RAW, n);
  }}
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
      x: [_stepAxis(t)], y: [_stepAxis(fDesc)], z: [Zs], customdata: [cd],
      hovertemplate: ['Time: %{{customdata[0]:.2f}} s<br>Freq: %{{customdata[1]:,.0f}} Hz<br>' +
                      zLabel + ': %{{z:.2f}}<extra></extra>']
    }};
  }} else {{
    upd = {{
      x: [rebinned.t], y: [fDesc], z: [Zt], customdata: [null],
      hovertemplate: ['Time: %{{x:.2f}} s<br>Freq: %{{y:,.0f}} Hz<br>' + zLabel + ': %{{z:.2f}}<extra></extra>']
    }};
  }}
  upd['colorbar.title.text'] = [zLabel];
  Plotly.restyle('specPlot', upd, [0]);
  Plotly.relayout('specPlot', {{'scene.zaxis.title.text': zLabel}});
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
  saveSession();
}}

function fmt(v, dec) {{
  return dec === 0 ? Math.round(v).toLocaleString('en-GB') : v.toFixed(3);
}}

// Return {{min, max}} of displayed values within the current time/freq window.
// Z is Z_RAW shape (T_ds, F_ds) with freq flipped (index 0 = highest freq).
// F_PLOT is low-to-high, so freq for Z[i][j] = F_PLOT[F_ds - 1 - j].
function windowMinMax(Z, t, tMin, tMax, fMin, fMax, useDb) {{
  var eps = 5e-324;
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
  // Colorscale bounds (both plots)
  Plotly.restyle('specPlot', {{ cmin: [mm.min], cmax: [mm.max], cauto: [false] }}, [0]);
  if (_heatReady) Plotly.restyle('heatPlot', {{ zmin: [mm.min], zmax: [mm.max], zauto: [false] }}, [0]);
  // 3-D z-axis height range
  Plotly.relayout('specPlot', {{ 'scene.zaxis.range': [mm.min, mm.max] }});
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
  saveSession();
}}
var _lastT1d = [null, null];

// Initialise the 2-D heatmap and 1-D spectrum immediately on load
_syncDecayControls();
_initHeatmap();
_fetchSpec1d();
// Align colorscale max across both plots using full data range
(function() {{
  var n = Math.max(1, Math.round(+document.getElementById('nAvg').value || 1));
  var rebinned = rebin(Z_RAW, T_RAW, n);
  if (document.getElementById('bars3d').checked) _renderSurface(rebinned);
  _rescaleToWindow(rebinned.Z, rebinned.t);
}})();
_hookCamera();
</script>
</body>
</html>"""

    state = dict(npz_path=_npz_path, t_full=_t_full, p_full=_p_full)
    return html, state


# ---------------------------------------------------------------------------
# Local server: serves the current page, loads files, runs decay fits
# ---------------------------------------------------------------------------

def load_dataset(path, fmin=None, fmax=None, linear=False, n_avg=1):
    """Load + crop a file and build its page. Returns (html, state)."""
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"file not found — {path}")
    freq, time, power = load(path)
    lo = float(freq.min()) if fmin is None else float(fmin)
    hi = float(freq.max()) if fmax is None else float(fmax)
    freq_s, power_s = slice_freq(freq, power, lo, hi)
    print(f"Loaded {os.path.basename(path)}: {power_s.shape[1]} freq bins × "
          f"{power_s.shape[0]} time bins  [{lo:,.0f} – {hi:,.0f} Hz]")
    html, state = build_page(freq_s, time, power_s, make_title(path, lo, hi, linear),
                             log_scale=not linear, n_avg=max(1, int(n_avg)),
                             file_path=path, fmin_req=fmin, fmax_req=fmax)
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
        view=saved_view(path, fmin, fmax),
    )
    try:
        write_prefs(prefs)
    except Exception:
        pass
    finally:
        _PREFS_LOCK.release()
    return html, state


def serve(html, state):
    import http.server, socket, threading, time, json, webbrowser, datetime
    import subprocess as _subprocess
    from urllib.parse import urlparse, parse_qs

    STATE = dict(state or {})
    STATE["html"] = html

    with socket.socket() as _s:
        _s.bind(("127.0.0.1", 0))
        port = _s.getsockname()[1]

    class _Handler(http.server.BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json"):
            if isinstance(body, (dict, list)):
                body = json.dumps(body)
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
                    d = q.get("path", [""])[0].strip() or read_prefs().get("lastDir") or os.getcwd()
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
                    self._send(200, {"path": d, "parent": os.path.dirname(d),
                                     "dirs": dirs, "files": files})
                except Exception as exc:
                    self._send(400, {"error": str(exc)})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path == "/load":
                try:
                    b = self._body()
                    html_new, st = load_dataset(b.get("file", ""), b.get("fMin"), b.get("fMax"),
                                                bool(b.get("linear")), int(b.get("nAvg") or 1))
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
                    if body.get("gamma") is not None:
                        cmd += ["--gamma", str(body["gamma"])]
                    if body.get("noOffset"):
                        cmd += ["--no-offset"]
                    if body.get("errCoeffs") is not None:
                        ea, eb = body["errCoeffs"]
                        cmd += ["--err-coeffs", str(float(ea)), str(float(eb))]
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
                    if body.get("title"):
                        cmd += ["--title", str(body["title"])]
                    _subprocess.Popen(cmd)
                    self._send(200, {"status": "launched"})
                except Exception as exc:
                    self._send(500, {"error": str(exc)})

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
                                sess["view"] = body["view"]
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
                    if pref_key not in ("gamma", "errA", "errB", "hlIso", "hlGs",
                                        "skipStart", "skipEnd", "bateman", "unweighted",
                                        "parentMin", "parentMax", "bars3d"):
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
    webbrowser.open(url)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        _srv.shutdown()


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
