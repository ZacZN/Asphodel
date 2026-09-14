#!/usr/bin/env python3
"""
Plot a spectrogram from a single ESR injection .npz file.

Usage
-----
  python plot_spectrogram.py <file> <freq_min> <freq_max> [--linear] [--plot3d] [--time-avg N]

Arguments
---------
  file          Path to a .npz file (freq, time, power arrays)
  freq_min      Lower frequency bound (same units as the freq array)
  freq_max      Upper frequency bound
  --linear      Plot raw power instead of 10·log10(power)
                (3-D: sets initial GUI state; can be toggled in the browser)
  --plot3d      Interactive 3-D surface plot (uses plotly; opens in browser)
  --time-avg N  Average every N consecutive time frames (e.g. 5 → T/5 frames)
                (3-D: sets initial GUI state; can be changed in the browser)

Example
-------
  python plot_spectrogram.py 1081.5.npz 245000 246000
  python plot_spectrogram.py 1081.5+1082.0.npz 245000 246000 --plot3d
  python plot_spectrogram.py 1081.5.npz 245000 246000 --linear --plot3d
  python plot_spectrogram.py 1081.5.npz 245000 246000 --time-avg 5
"""

import sys
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def parse_args():
    parser = argparse.ArgumentParser(description="Plot ESR injection spectrogram")
    parser.add_argument("file",     help="Path to .npz file")
    parser.add_argument("freq_min", type=float, help="Lower frequency bound")
    parser.add_argument("freq_max", type=float, help="Upper frequency bound")
    parser.add_argument("--linear", action="store_true",
                        help="Plot linear power instead of dB (log scale)")
    parser.add_argument("--plot3d", action="store_true",
                        help="Interactive 3-D surface plot via plotly (opens in browser)")
    parser.add_argument("--time-avg", type=int, metavar="N", default=None,
                        help="Average every N consecutive time frames (reduces time axis by factor N)")
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


def to_db(power):
    """Convert power to dB, guarding against zeros/negatives."""
    eps = np.finfo(float).tiny
    return 10.0 * np.log10(np.maximum(power, eps))


# ---------------------------------------------------------------------------
# 2-D matplotlib plot (original)
# ---------------------------------------------------------------------------

def plot_2d(freq, time, power, title, log_scale=True):
    fig, ax = plt.subplots(figsize=(12, 5))

    z = to_db(power) if log_scale else power

    # pcolormesh expects edges; build them from centres
    def edges(arr):
        d = np.diff(arr)
        return np.concatenate([[arr[0] - d[0] / 2],
                                arr[:-1] + d / 2,
                                [arr[-1] + d[-1] / 2]])

    t_edges = edges(time)
    f_edges = edges(freq)

    mesh = ax.pcolormesh(
        t_edges, f_edges, z.T,          # z.T → (F, T) for (y, x) orientation
        cmap="viridis",
        shading="flat",
        rasterized=True,
    )

    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("Power (dB)" if log_scale else "Power (linear)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Frequency (Hz)", fontsize=11)
    ax.set_title(title, fontsize=12, pad=8)

    # Frequency axis: show full integers, no scientific notation
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.tick_params(labelsize=9)

    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Time rebinning
# ---------------------------------------------------------------------------

def rebin_time(time, power, n):
    """Average every N consecutive time frames.

    Trailing frames that don't fill a complete group are discarded.
    Averaging is done in linear power space.

    Returns (time_rebinned, power_rebinned) with shape (T//N, F).
    """
    T, F = power.shape
    T_trim = (T // n) * n
    power_r = power[:T_trim].reshape(T // n, n, F).mean(axis=1)
    time_r  = time[:T_trim].reshape(T // n, n).mean(axis=1)
    return time_r, power_r


# ---------------------------------------------------------------------------
# 3-D interactive plotly plot
# ---------------------------------------------------------------------------

def plot_3d(freq, time, power, title, log_scale=True, n_avg=1):
    try:
        import plotly.graph_objects as go
    except ImportError:
        sys.exit(
            "plotly is required for --plot3d.  Install it with:\n"
            "  pip install plotly"
        )
    import tempfile, webbrowser, json

    z_label = "Power (dB)" if log_scale else "Power (linear)"

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

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font_size=14),
        scene=dict(
            xaxis=dict(title="Time (s)"),
            yaxis=dict(title="Frequency (Hz)", tickformat=",d",
                       range=[float(freq.max()), float(freq.min())]),
            zaxis=dict(title=z_label),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.8)),
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        height=680,
    )

    # Axis bounds and bin sizes for slider step sizes
    f_min, f_max = float(freq.min()), float(freq.max())
    df = float(freq[1] - freq[0]) if len(freq) > 1 else max(f_max - f_min, 1.0)

    # Initial time slider bounds (after initial averaging)
    t_init_min = float(t_init[0])
    t_init_max = float(t_init[-1])
    dt_init = float(t_init[1] - t_init[0]) if len(t_init) > 1 else max(t_init_max - t_init_min, 1e-6)

    # Embed raw (downsampled) data as JSON for JS re-rendering
    z_raw_json = json.dumps(z_ds.tolist())
    t_raw_json = json.dumps(t_ds.tolist())
    f_raw_json = json.dumps(f_ds.tolist())   # freq array, low-to-high (for 2-D heatmap)
    db_checked = "checked" if log_scale else ""

    plot_div = fig.to_html(full_html=False, include_plotlyjs="cdn", div_id="specPlot",
                           config={"responsive": True})
    title_html = title.replace("&", "&amp;").replace('"', "&quot;")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
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
  #discardMsg {{ font-size: 11px; color: #fbbf24; margin-left: 4px; }}
  #plots {{ display: flex; gap: 8px; align-items: flex-start; }}
  #heatDiv {{ display: none; flex: 1; min-width: 0; }}
  #surfDiv {{ flex: 1; min-width: 0; }}
</style>
</head>
<body>
<div id="controls">
  <div class="title-row">
    <label>Title</label>
    <input type="text" id="plotTitle" value="{title_html}" oninput="updateTitle()">
  </div>
  <div class="grp">
    <h4>Frequency (Hz)</h4>
    <div class="row">
      <label>Min</label>
      <input type="range" id="fMin" min="{f_min}" max="{f_max}"
             step="{df}" value="{f_min}" oninput="update()">
      <span id="fMinVal">{f_min:,.0f}</span>
    </div>
    <div class="row">
      <label>Max</label>
      <input type="range" id="fMax" min="{f_min}" max="{f_max}"
             step="{df}" value="{f_max}" oninput="update()">
      <span id="fMaxVal">{f_max:,.0f}</span>
    </div>
  </div>
  <div class="grp">
    <h4>Time (s)</h4>
    <div class="row">
      <label>Min</label>
      <input type="range" id="tMin" min="{t_init_min}" max="{t_init_max}"
             step="{dt_init}" value="{t_init_min}" oninput="update()">
      <span id="tMinVal">{t_init_min:.3f}</span>
    </div>
    <div class="row">
      <label>Max</label>
      <input type="range" id="tMax" min="{t_init_min}" max="{t_init_max}"
             step="{dt_init}" value="{t_init_max}" oninput="update()">
      <span id="tMaxVal">{t_init_max:.3f}</span>
    </div>
  </div>
  <div class="opts-row">
    <div class="opt-grp">
      <span>Scale</span>
      <input type="checkbox" id="dbCheck" {db_checked} onchange="updateSurface()">
      <label for="dbCheck">dB</label>
    </div>
    <div class="opt-grp">
      <span>Time avg</span>
      <input type="number" id="nAvg" min="1" step="1" value="{n_avg}">
      <button onclick="updateSurface()">Apply</button>
      <span id="discardMsg"></span>
    </div>
    <div class="opt-grp">
      <span>View</span>
      <input type="checkbox" id="show2D" onchange="toggle2D()">
      <label for="show2D">Show 2D</label>
    </div>
  </div>
</div>
<div id="plots">
  <div id="heatDiv"><div id="heatPlot"></div></div>
  <div id="surfDiv">{plot_div}</div>
</div>
<script>
const Z_RAW = {z_raw_json};
const T_RAW = {t_raw_json};
const F_PLOT = {f_raw_json};  // freq array, low-to-high (2-D heatmap y-axis)

var _heatReady = false;

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

function toggle2D() {{
  var show = document.getElementById('show2D').checked;
  document.getElementById('heatDiv').style.display = show ? 'block' : 'none';
  if (show) {{
    if (!_heatReady) {{ _initHeatmap(); _heatReady = true; }}
    else {{ _updateHeatmap(); }}
  }}
  // Resize the 3-D plot to fit its new (narrower or wider) container
  setTimeout(function() {{ Plotly.Plots.resize(document.getElementById('specPlot')); }}, 0);
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
  }}], {{
    template: {{}},
    title: {{ text: document.getElementById('plotTitle').value,
              x: 0.5, xanchor: 'center', font: {{ size: 14, color: '#e5e7eb' }} }},
    xaxis: {{ title: 'Time (s)',
              range: [+document.getElementById('tMin').value, +document.getElementById('tMax').value],
              color: '#9ca3af', gridcolor: '#374151', zerolinecolor: '#374151' }},
    yaxis: {{ title: 'Frequency (Hz)', tickformat: ',d', autorange: 'reversed',
              range: [+document.getElementById('fMax').value, +document.getElementById('fMin').value],
              color: '#9ca3af', gridcolor: '#374151', zerolinecolor: '#374151' }},
    paper_bgcolor: '#111827', plot_bgcolor: '#1f2937',
    font: {{ color: '#e5e7eb' }},
    margin: {{ l: 90, r: 20, t: 60, b: 60 }},
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

// ── Main update functions ────────────────────────────────────────────────────

function updateTitle() {{
  var t = document.getElementById('plotTitle').value;
  Plotly.relayout('specPlot', {{'title.text': t}});
  if (_heatReady) Plotly.relayout('heatPlot', {{'title.text': t}});
}}

function updateSurface() {{
  var n = Math.max(1, Math.round(+document.getElementById('nAvg').value || 1));
  document.getElementById('nAvg').value = n;
  var useDb = document.getElementById('dbCheck').checked;
  var rebinned = rebin(Z_RAW, T_RAW, n);
  var Zdisp = useDb ? toDb(rebinned.Z) : rebinned.Z;
  var Zt = transpose(Zdisp);
  var zLabel = useDb ? 'Power (dB)' : 'Power (linear)';
  var hoverTmpl = 'Time: %{{x:.2f}} s<br>Freq: %{{y:,.0f}} Hz<br>' + zLabel + ': %{{z:.2f}}<extra></extra>';
  Plotly.restyle('specPlot', {{
    x: [rebinned.t],
    z: [Zt],
    'colorbar.title.text': [zLabel],
    hovertemplate: [hoverTmpl]
  }}, [0]);
  Plotly.relayout('specPlot', {{'scene.zaxis.title.text': zLabel}});
  resetTimeSliders(rebinned.t);
  var T = Z_RAW.length;
  var discarded = T % n;
  document.getElementById('discardMsg').textContent =
    discarded > 0 ? '⚠︎ ' + discarded + ' of ' + T + ' frames discarded' : '';
  if (_heatReady) _updateHeatmap();
}}

function fmt(v, dec) {{
  return dec === 0 ? Math.round(v).toLocaleString('en-GB') : v.toFixed(3);
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
}}
</script>
</body>
</html>"""

    import http.server, socket, threading, time

    # Write HTML to a temp directory
    tmp_dir = tempfile.mkdtemp()
    html_path = os.path.join(tmp_dir, "spectrogram.html")
    with open(html_path, "w", encoding="utf-8") as _f:
        _f.write(html)

    # Find a free port
    with socket.socket() as _s:
        _s.bind(("127.0.0.1", 0))
        port = _s.getsockname()[1]

    # Serve over localhost — bypasses snap AppArmor restrictions on file:// URLs
    class _Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=tmp_dir, **kwargs)
        def log_message(self, *args):
            pass  # silence request logs

    _srv = http.server.HTTPServer(("127.0.0.1", port), _Handler)
    threading.Thread(target=_srv.serve_forever, daemon=True).start()

    url = f"http://127.0.0.1:{port}/spectrogram.html"
    print(f"Serving 3-D plot at {url}  (Ctrl-C to quit)")
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

    if not os.path.isfile(args.file):
        sys.exit(f"Error: file not found — {args.file}")

    freq, time, power = load(args.file)
    freq_s, power_s = slice_freq(freq, power, args.freq_min, args.freq_max)

    scale_label = "linear" if args.linear else "dB"
    title = (
        f"{os.path.basename(args.file)}   "
        f"[{args.freq_min:,.0f} – {args.freq_max:,.0f} Hz]   "
        f"({scale_label})"
    )

    n_avg = args.time_avg  # None or an integer

    if args.plot3d:
        # For 3-D mode all options (scale, time-avg) are available as GUI controls;
        # CLI flags set their initial values only.
        init_n = n_avg if (n_avg is not None and n_avg >= 1) else 1
        if n_avg is not None and n_avg < 1:
            sys.exit("Error: --time-avg N requires N ≥ 1.")
        print(f"Plotting {power_s.shape[1]} freq bins × {power_s.shape[0]} time bins")
        plot_3d(freq_s, time, power_s, title,
                log_scale=not args.linear, n_avg=init_n)
    else:
        if n_avg is not None:
            if n_avg < 2:
                sys.exit("Error: --time-avg N requires N ≥ 2.")
            T = power_s.shape[0]
            time, power_s = rebin_time(time, power_s, n_avg)
            print(
                f"Time rebinned: {T} frames → {len(time)} frames "
                f"(averaged every {n_avg}, {T % n_avg} trailing frame(s) discarded)"
            )
            title = title.rstrip(")") + f", time-avg {n_avg})"

        print(f"Plotting {power_s.shape[1]} freq bins × {power_s.shape[0]} time bins")
        plot_2d(freq_s, time, power_s, title, log_scale=not args.linear)


if __name__ == "__main__":
    main()
