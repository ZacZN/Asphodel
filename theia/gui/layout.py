"""
Page layout: the tab bar with every tab, the coordinate readout; sets app.layout.

Part of Theia (theia/theia.py); split out of theia.py in Nov 2026.
"""

import os

from dash import dcc, html, Input, Output, State, no_update

from core import corrections, logpanel, paths, summing
from core.scan import load_config

from .common import ACCENT, BOX_STYLE, CFG
from .app import app
from .info import info_modal
from .corr_profiles import corr_profile_state
from .folder_browser import folder_browser
from .viewer import load_viewer_config, v_zoom_width_saved, VIEWER_KEYS, viewer_tab
from .scan_tab import scan_main, sidebar
from .results import results_tab
from .further_files import pid_modal
from .further_fit import SRCH_DEFAULT_KHZ, srch_modal
from .isocurve import iso_modal
from .further_layout import further_tab
from .logbook import lb_modal


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
    # Log: this program's output (core/logpanel.py), left of Quit
    logpanel.dash_button(className="btn btn-outline-secondary btn-sm",
                         style={"position": "absolute", "top": "7px", "right": "188px",
                                "zIndex": "20"}),
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
        dcc.Tab(label="Further Output", value="further", style=TAB_STYLE, selected_style=TAB_SEL,
                children=[further_tab]),
    # Tab bar stops short of the right edge to leave room for the Quit and
    # Settings buttons
    ], style={"height": "42px", "width": "calc(100% - 268px)"}),
    # Polls the background scan; only enabled while a scan is running
    dcc.Interval(id="poll", interval=700, disabled=True),
    dcc.Store(id="scan-finished"),
    # Hidden hooks used by the clientside callbacks that attach the readout
    *[html.Div(id=f"_readout-hook-{gid}", style={"display": "none"}) for gid in READOUT_GRAPHS],
    html.Div(id="_snippet-resize-hook", style={"display": "none"}),
    html.Div(id="_hits-fill-hook", style={"display": "none"}),
    html.Div(id="_hits-page-hook", style={"display": "none"}),
    html.Div(id="_snippet-width-saved", style={"display": "none"}),
    html.Div(id="_snippet-excluded-hook", style={"display": "none"}),
    # Further output tab: saved state handed to the restore logic on page load
    folder_browser,
    info_modal,
    pid_modal,
    iso_modal,
    srch_modal,
    lb_modal,
    dcc.Store(id="further-restore", data=None),
    # the saved state waits here until the Further Output tab is first opened
    # (restore_further_when_opened below), so page load stays fast
    dcc.Store(id="further-restore-pending", data=None),
    dcc.Store(id="sum-view", data=None),
    html.Div(id="_sum-view-hook", style={"display": "none"}),
    html.Div(id="_further-saved", style={"display": "none"}),
    # Settings-file problems (damaged file set aside, failed save): core/jsonfile.py
    html.Div(id="settings-warn", children=[],
             style={"position": "fixed", "top": "48px", "right": "14px", "zIndex": 3000,
                    "maxWidth": "560px"}),
    dcc.Interval(id="settings-warn-poll", interval=3000),
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
    update_fit from the further-restore store, which is filled the first time
    the Further Output tab is opened (restore_further_when_opened)."""
    # Viewer tab: last saved viewer settings
    vcfg = load_viewer_config()
    for k in VIEWER_KEYS:
        comp = _find_component(_BASE_LAYOUT, f"v-{k}")
        if comp is not None:
            comp.value = vcfg[k] if k == "correction_enabled" else str(vcfg[k])
    comp = _find_component(_BASE_LAYOUT, "v-correction-inputs")
    if comp is not None:
        comp.style = {"display": "block" if vcfg["correction_enabled"] else "none"}
    comp = _find_component(_BASE_LAYOUT, "v-zoom-width")
    if comp is not None:
        comp.value = v_zoom_width_saved()

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
        ("sum-window", "value", st.get("window") or summing.DEFAULT_WINDOW_HZ),
        ("fit-shape", "value", fit.get("shape") or "gauss"),
        ("fit-bg", "value", fit.get("bg") or "const"),
        ("further-restore-pending", "data", (dict(st, files=files)
                                             if st.get("summed") and restorable else None)),
        ("pid-settings", "data", st.get("pid")),
        ("further-source", "value", st.get("source") or "sum"),
        ("fsrc-dir", "value", st.get("file_dir") or os.path.join(paths.summed_1d(), "")),
        ("fsrc-name", "value", st.get("file_name")),
        ("srch-width", "value", load_config().get("search_width_khz") or SRCH_DEFAULT_KHZ),
    ]:
        comp = _find_component(_BASE_LAYOUT, cid)
        if comp is not None:
            setattr(comp, prop, value)

    # correction profiles: options (profiles saved since the last load), the
    # stored profile (or the one matching the a, b, c shown) and the locked boxes
    stored_profile = {"s-": load_config().get("corr_profile"), "v-": vcfg.get("corr_profile")}
    for prefix in ("s-", "v-"):
        def val(k):
            comp = _find_component(_BASE_LAYOUT, f"{prefix}{k}")
            return None if comp is None else getattr(comp, "value", None)
        value, opts, info, locked, prof = corr_profile_state(
            stored_profile[prefix], val("corr_a"), val("corr_b"), val("corr_c"), val("ref_name"),
            val("ref_harmonic"), val("gamma_t"), val("ref_frequency"), val("harmonics"))
        for cid, prop, v in ((f"{prefix}corr_profile", "options", opts),
                             (f"{prefix}corr_profile", "value", value),
                             (f"{prefix}corr_info", "children", info)):
            comp = _find_component(_BASE_LAYOUT, cid)
            if comp is not None:
                setattr(comp, prop, v)
        for k in ("corr_a", "corr_b", "corr_c"):
            comp = _find_component(_BASE_LAYOUT, f"{prefix}{k}")
            if comp is not None:
                comp.disabled = locked
                if prof is not None:          # show the profile's general a, b, c
                    v = prof[k[-1]]
                    comp.value = str(v) if isinstance(comp.value, str) else float(v)
    comp = _find_component(_BASE_LAYOUT, "pid-corr_profile")
    if comp is not None:
        comp.options = corrections.options()
    return _BASE_LAYOUT


@app.callback(
    Output("further-restore", "data"),
    Output("further-restore-pending", "data"),
    Input("tabs", "value"),
    State("further-restore-pending", "data"),
)
def restore_further_when_opened(tab, pending):
    """Rebuild the Further Output plot saved last time only when that tab is
    first opened (it can take several seconds), not on every page load."""
    if tab == "further" and pending:
        return pending, None
    return no_update, no_update


app.layout = logpanel.dash_layout(serve_layout, "Theia")
logpanel.register_dash(app)

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

        // Keep the hover labels on screen.  Plotly keeps them inside the plot,
        // but the plot can reach below the visible part of the page (the tab
        // scrolls), and stacked labels of nearby lines (hovermode "x") are
        // pushed downwards.  After each hover, shift the label layer up (or
        // down) so all labels sit inside the visible area: the window, cut by
        // every scrolling box around the plot, and above the coordinate readout.
        var fitHover = function() {
            var hl = gd._fullLayout && gd._fullLayout._hoverlayer && gd._fullLayout._hoverlayer.node();
            if (!hl) return;
            watchLayer(hl);
            // each label is a <g class="hovertext" transform="translate(x,y)"> holding its box
            // and its text; it is moved through that same attribute, so box and text stay
            // together (a CSS transform on the whole layer moved the boxes without their
            // text in Firefox)
            var labels = hl.querySelectorAll('g.hovertext'), items = [], fresh = false;
            for (var i = 0; i < labels.length; i++) {
                var g = labels[i];
                var m = /translate\\(\\s*([-\\d.e]+)[ ,]+([-\\d.e]+)\\s*\\)/.exec(g.getAttribute('transform') || '');
                if (!m) continue;
                var x = parseFloat(m[1]), y = parseFloat(m[2]);
                var sh = g._theiaShift;
                if (sh && Math.abs(y - sh.to) < 0.01 && Math.abs(x - sh.x) < 0.01) y = sh.from;  // our own move
                else fresh = true;                                                        // drawn by Plotly
                items.push({g: g, x: x, y: y});
            }
            if (!items.length || !fresh) return;          // nothing new since the last fit
            // measure at Plotly's positions
            items.forEach(function(it) { it.g.setAttribute('transform', 'translate(' + it.x + ',' + it.y + ')'); });
            var top = Infinity, bot = -Infinity, left = Infinity, right = -Infinity;
            items.forEach(function(it) {
                var r = it.g.getBoundingClientRect();
                if (!r.height) return;
                top = Math.min(top, r.top); bot = Math.max(bot, r.bottom);
                left = Math.min(left, r.left); right = Math.max(right, r.right);
            });
            var dy = 0;
            if (isFinite(top)) {
                var vTop = 0, vBot = window.innerHeight;
                for (var el = gd.parentElement; el && el !== document.body; el = el.parentElement) {
                    if (/(auto|scroll|hidden)/.test(getComputedStyle(el).overflowY)) {
                        var bx = el.getBoundingClientRect();
                        vTop = Math.max(vTop, bx.top); vBot = Math.min(vBot, bx.bottom);
                    }
                }
                if (readout.style.display !== 'none') {
                    var ro = readout.getBoundingClientRect();
                    if (ro.height && right > ro.left && left < ro.right) vBot = Math.min(vBot, ro.top);
                }
                var pad = 4;
                if (bot > vBot - pad) dy = (vBot - pad) - bot;
                if (top + dy < vTop + pad) dy = (vTop + pad) - top;     // too tall: keep the top visible
                dy = Math.round(dy);
            }
            items.forEach(function(it) {
                var to = it.y + dy;
                it.g.setAttribute('transform', 'translate(' + it.x + ',' + to + ')');
                it.g._theiaShift = {x: it.x, from: it.y, to: to};
            });
        };
        // Plotly redraws the labels on every mouse move (it only reports a hover when
        // the hovered points change), so refit whenever the label layer changes; the
        // observer runs before the browser paints, so a moved label never flashes
        var watchLayer = function(hl) {
            if (hl._theiaObs || !window.MutationObserver) return;
            hl._theiaObs = new MutationObserver(function() { fitHover(); });
            hl._theiaObs.observe(hl, {childList: true, subtree: true, attributes: true,
                                      attributeFilter: ['transform']});
        };
        gd.on('plotly_hover', fitHover);
        gd.on('plotly_afterplot', function() {
            var hl = gd._fullLayout && gd._fullLayout._hoverlayer && gd._fullLayout._hoverlayer.node();
            if (hl) watchLayer(hl);
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
