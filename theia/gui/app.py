"""
The Dash app object, the page CSS and the browser-tab hooks.

Part of Theia (theia/theia.py); split out of theia.py in Nov 2026.
"""

import dash

from core import bootscreen, icon, offline, tabwatch, trackpad


# Bootstrap CSS served from core/static/ (not cdn.jsdelivr.net), so Theia
# works without internet access: core/offline.py
app = dash.Dash(__name__, external_stylesheets=[offline.BOOTSTRAP_CSS_URL],
                suppress_callback_exceptions=True, title="Theia")
offline.serve_static(app.server)
app.title = "Theia"

# Results tables: move the page buttons from below the table to the top right,
# level with the Export button (both 30 px tall), so nothing is left below it.
_TABLE_CSS = """
#hits-table { position: relative; }
#hits-table .previous-next-container {
    position: absolute; top: 0; right: 0; margin: 0; padding: 0;
    height: 30px; display: flex; align-items: center;
}
#hits-table .previous-next-container button {
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

_INPUT_CSS = """
/* dcc.Input is a box (.dash-input-container, which gets our className/style)
   holding the real <input>. Make the <input> fill the box; otherwise some
   browsers keep it at its default ~20 characters and long text such as a
   folder path is cut off in the middle of the box. */
.dash-input-container { display: flex; align-items: center; }
.dash-input-container > input.dash-input-element { flex: 1 1 auto; width: 100%; min-width: 0; }
/* fold-away sections (html.Details): orange disclosure marker */
.theia-fold > summary::marker { color: #f57d15; }
.theia-fold > summary:hover { color: #f57d15; }
"""

_HARM_CSS = """
/* Viewer harmonic toggles: checklist labels drawn as on/off buttons */
.harm-toggle label { display: block; width: 90px; border: 1px solid #6c757d; border-radius: 4px;
    padding: 2px 8px; margin: 0 0 5px 0; font-size: 13px; text-align: center;
    cursor: pointer; user-select: none; color: #6c757d; background: #ffffff; }
.harm-toggle label:hover { border-color: #f57d15; }
.harm-toggle label:has(input:checked) { background: #f57d15; border-color: #f57d15;
    color: #ffffff; font-weight: 600; }
body.dark #v-legend-frame { border-color: #4a4a4a !important; }
body.dark .harm-toggle label { background: #1c1c1c; color: #d6d6d6; border-color: #9a9a9a; }
body.dark .harm-toggle label:has(input:checked) { background: #f57d15; border-color: #f57d15;
    color: #ffffff; }
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
body.dark .send-pid-card { background: #1c1c1c !important; border-color: #6a6a6a !important; }
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
/* hover labels and their leader lines (theia/gui/layout.py): colours here are before the
   inversion, so black shows as white.  White borders on the labels; the leaders' casing and
   dot rings white (they would invert to black and vanish on the black background) */
body.dark .js-plotly-plot .hoverlayer g.hovertext > path { stroke: #000000 !important; stroke-width: 1.5px !important; }
body.dark .js-plotly-plot .theia-leader-halo { stroke: #000000 !important; stroke-opacity: 1 !important; stroke-width: 5px !important; }
body.dark .js-plotly-plot .theia-leader-line { stroke-width: 2px !important; }
body.dark .js-plotly-plot .theia-leader-dot { stroke: #000000 !important; stroke-width: 2px !important; r: 4px; }
"""
app.index_string = app.index_string.replace(
    "{%css%}", "{%css%}\n<style>" + _TABLE_CSS + _FB_CSS + _CALIB_CSS + _FIT_CSS + _INPUT_CSS + _HARM_CSS + _DARK_CSS + "</style>")
# close this tab when Theia stops (Quit / Stop on the home page, Quit here, crash)
app.index_string = app.index_string.replace(
    "{%scripts%}", "{%scripts%}\n<script>" + tabwatch.watch_js("Theia") + "</script>")
trackpad.install_dash(app)      # touchpad zoom in the plots (core/trackpad.py)
# loading screen until the page (and each tab when first opened) is ready (core/bootscreen.py)
bootscreen.install_dash(app, "Theia", tabs="tabs")
icon.use_in_dash(app, "theia")   # browser-tab icon: core/icons/theia.png, else asphodel.png
