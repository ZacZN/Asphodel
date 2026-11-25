"""
Shared settings, styles and layout helpers of Theia's GUI.

Part of Theia (theia/theia.py); split out of theia.py in Nov 2026.
"""

from dash import dash_table, dcc, html

from core import paths
from core.physics import HL_CUT_UNITS, HL_UNITS
from core.scan import load_config


CONFIRMED_DIR = paths.outputs("confirmed")
VIEWER_CONFIG_PATH = paths.settings_file("viewer_config.json")
ACCENT      = "#f57d15"
# h±1 / q±1 / A±1 step buttons: each pair is "−1  value  +1"
STEP_PAIR_STYLE = {"display": "inline-flex", "alignItems": "center", "gap": "6px"}
STEP_CUR_STYLE = {"fontSize": "13px", "fontWeight": "600"}
ACCENT_PALE = "#fff4ec"
# outline of the boxes (plots, panels, pop-ups); dark mode greys it via .theia-box
BOX_BORDER = "1px solid black"


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

REF_TABLE_COLS = [("h", "h"), ("status", "Status"), ("predicted", "Expected (Hz)"),
                  ("freq", "Found at (Hz)"), ("offset", "Offset (Hz)")]


def _ref_table_records(rows, ref_f_label=None):
    """physics.reference_rows -> table records.  ref_f_label: status shown
    for the reference harmonic (e.g. the scan's 'found' / 'predicted')."""
    out = []
    for r in rows:
        st = r["status"]
        label = (ref_f_label or "reference") if st == "reference" else r["label"]
        shown = st in ("reference", "found")
        out.append({"h": r["h"], "status": label, "predicted": f"{r['predicted']:.1f}",
                    "freq": f"{r['f']:.1f}" if shown else "—",
                    "offset": "—" if r["offset"] is None else f"{r['offset']:+.1f}"})
    return out


def ref_peak_table(id_, records, columns=REF_TABLE_COLS, mark_h=None):
    """Small table of where the reference ion's peak is in each harmonic
    (mark_h: harmonic whose row is outlined)."""
    mark = ([{"if": {"filter_query": f"{{h}} = {int(mark_h)}"}, "border": f"2px solid {ACCENT}"}]
            if mark_h is not None else [])
    return dash_table.DataTable(
        id=id_, columns=[{"name": n, "id": c} for c, n in columns], data=records, fill_width=False,
        style_table={"overflowX": "auto"}, style_header=TABLE_HEADER, style_cell=TABLE_CELL,
        style_data_conditional=TABLE_ODD + [
            {"if": {"filter_query": '{status} = "not found"', "column_id": "status"},
             "color": "#c0392b", "fontWeight": "600"},
            {"if": {"filter_query": '{status} = "found"', "column_id": "status"},
             "color": "#2e7d32", "fontWeight": "600"},
            {"if": {"column_id": "status"}, "textAlign": "left"}] + mark)

# key -> (component id, property)
SETTING_FIELDS = {}


def inp(label, key, number=False, help=None, step="any", placeholder=None):
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
    if placeholder:
        kw["placeholder"] = placeholder
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
        html.Label("Exclude Ions with Half-Life Below  [0 = off]", style=LABEL_STYLE),
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


# column headings of the tables made by _table (the ids stay the database column
# names, and an exported CSV keeps them as its header row)
COLUMN_TITLES = {"spectrum": "Spectrum", "ion": "Ion", "harmonic": "Harmonic",
                 "peak_freq": "Peak Freq (Hz)", "sim_freq": "Sim Freq (Hz)", "delta_hz": "Δ (Hz)", "n_nearby": "Nearby Lines",
                 "power": "Power", "snr": "SNR", "kind": "Kind", "closest_for_ion": "Closest for Ion",
                 "closest_for_peak": "Closest for Peak", "key": "Key", "value": "Value"}


def _table(id_, columns, page_size=20, height=None, overflow_x="auto", **kw):
    kw.setdefault("export_headers", "ids")
    return dash_table.DataTable(
        id=id_, columns=[{"name": COLUMN_TITLES.get(c, c), "id": c} for c in columns], data=[],
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
BOX_STYLE = {"border": BOX_BORDER, "borderRadius": "6px", "padding": "10px",
             "background": "#ffffff"}
# box round the residuals plot on the Further output page
RESID_BOX_SHOWN = {**BOX_STYLE, "padding": "8px 10px", "marginTop": "10px", "display": "block"}
RESID_BOX_HIDDEN = {**RESID_BOX_SHOWN, "display": "none"}
TABLE_PAGE_SIZE = 15    # rows per page of the hits table
TABLE_ROW_PX = 30       # DataTable row height
FILTER_HEAD_PX = 120    # height of the filter block above the hits table
# heading of the fold-away per-harmonic tables in the reference boxes
REF_FOLD_SUMMARY = {"fontWeight": "600", "fontSize": "13px", "cursor": "pointer"}
# sub-tab bars (Scan Results, Further output)
SUBTAB_STYLE = {"padding": "6px 14px", "fontWeight": "600", "fontSize": "13px"}
SUBTAB_SEL = {**SUBTAB_STYLE, "borderTop": f"3px solid {ACCENT}"}
# the two Scan Results pages: two columns side by side
# Both pages sit in the same cell of RESULTS_PAGES, which fills the height left
# under the database row / sub-tab bar (the tab is a flex column), down to a
# minimum below which the tab scrolls.  The page not shown is made invisible
# rather than display: none, so it keeps its size: tables in a display: none
# page crash Dash's DataTable (endless resize loop) when their data changes,
# and plots would come back at the wrong size.
RESULTS_PAGES = {"display": "grid", "gridTemplateColumns": "minmax(0, 1fr)",
                 "gridTemplateRows": "minmax(0, 1fr)", "flex": "1 1 0", "minHeight": "560px"}
RESULTS_PAGE_GRID = {"display": "grid", "gridTemplateColumns": "minmax(0, 1fr) minmax(0, 1fr)",
                     "gridTemplateRows": "minmax(0, 1fr)", "columnGap": "16px",
                     "alignItems": "stretch", "gridArea": "1 / 1", "minHeight": "0",
                     "position": "relative", "zIndex": "1"}
# The hidden page goes underneath: Plotly's drag layers set their own
# pointer-events, so without this an invisible plot would catch the mouse
# over the page shown (crosshair, no clicks).
RESULTS_PAGE_HIDDEN = {**RESULTS_PAGE_GRID, "visibility": "hidden", "pointerEvents": "none",
                       "zIndex": "-1"}
# a box that fills its grid cell, its contents stacked; the plot in it grows
# (on a screen too small for everything at its minimum size the box scrolls)
FILL_BOX = {**BOX_STYLE, "minWidth": "0", "minHeight": "0", "display": "flex",
            "flexDirection": "column", "overflowY": "auto"}
# Run info / settings tables: long values (e.g. corr_by_h) wrap onto several
# lines inside the table's width instead of widening it
KV_WRAP = dict(
    style_data={"whiteSpace": "pre-line", "height": "auto"},   # pre-line: keep \n line breaks
    style_cell_conditional=[
        {"if": {"column_id": "key"}, "whiteSpace": "nowrap", "width": "1%"},
        {"if": {"column_id": "value"}, "textAlign": "left", "overflowWrap": "anywhere",
         "wordBreak": "break-word"},
    ],
)


ISO_MODAL_SHOWN = {"display": "flex", "position": "fixed", "inset": "0", "zIndex": "1000",
                   "background": "rgba(0,0,0,0.35)", "alignItems": "center",
                   "justifyContent": "center"}
ISO_MODAL_HIDDEN = {**ISO_MODAL_SHOWN, "display": "none"}
