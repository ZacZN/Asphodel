"""
Viewer tab: one spectrum with the simulated ion lines (was viewer.py).

Part of Theia (theia/theia.py); split out of theia.py in Nov 2026.
"""

import os
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State

from core import corrections, jsonfile, paths, physics
from core.nubase_reader import df_nubase, diag_restrict, index_by_name, rect_restrict, unpack_npz
from core.physics import apply_correction, build_hover_label, delta_f, ion_base_name
from core.scan import _load_json

from .common import (ACCENT, BOX_BORDER, BOX_STYLE, HELP_STYLE, INPUT_STYLE, LABEL_STYLE,
                     ref_peak_table, _ref_table_records, section, VIEWER_CONFIG_PATH)
from .app import app
from .logbook import logbook_button
from .figures import empty_fig
from .corr_profiles import corr_profile_ui, register_corr_profile
from .folder_browser import list_npz, with_browse


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
                                correction_enabled=False, corr_a=0.0, corr_b=0.0, corr_c=0.0,
                                ref_rev_by_h=None, corr_by_h=None):
    """ref_rev_by_h: {h: reference revolution frequency for that harmonic}
    (from its own reference peak); corr_by_h: {h: (a, b, c)} (else corr_a/b/c)."""
    result = df.sort_values(mq_col).copy()
    for h in harmonics:
        ref_mq = result.loc[ref_ion_name, mq_col]
        f_ref_h = (ref_rev_by_h or {}).get(int(h), ref_freq)
        abc = physics._corr_for(h, corr_by_h, (corr_a, corr_b, corr_c))

        def _direct(row_mq, _h=h, _ref_mq=ref_mq, _corr=correction_enabled, _f=f_ref_h, _abc=abc):
            if row_mq == _ref_mq:
                f_true = _f
            else:
                f_true = _f + delta_f(_f, _ref_mq, row_mq - _ref_mq, gamma_t)
            # correction applied to the revolution frequency before harmonic scaling
            if _corr:
                f_true = apply_correction(f_true, *_abc)
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
    every charge state of that isotope ('204Hg' matches '204Hg79+').
    Case-insensitive, so '205tl' and '205TL81+' work too."""
    name = ion_name.lower()
    base = ion_base_name(ion_name).lower()
    for h in highlight_list:
        h = h.lower()
        if ("+" in h and name == h) or ("+" not in h and base == h):
            return True
    return False


# Viewer: non-highlighted simulated lines (same grey as "Simulated (no peak)")
V_SIM_STYLE = dict(color="#bbbbbb", width=1, dash="dash")
# Viewer: one colour per highlighted ion and harmonic (Plotly Dark24 without
# its near-black and grey entries, so they stand out from the spectrum and
# the grey lines)
V_HIGHLIGHT_COLORS = ["#2E91E5", "#E15F99", "#1CA71C", "#FB0D0D", "#DA16FF", "#B68100",
                      "#750D86", "#EB663B", "#511CFB", "#00A08B", "#FB00D1", "#FC0080",
                      "#B2828D", "#6C7C32", "#862A16", "#A777F1", "#620042", "#1616A7",
                      "#DA60CA", "#6C4516", "#0D2A63", "#AF0038"]


def v_build_figure(file_path, spectrum_name, ref_name, ref_frequency, ref_harmonic,
                   z_lims, n_lims, el1, el2, mq_width, gamma_t, harmonics,
                   correction_enabled=False, corr_a=0.0, corr_b=0.0, corr_c=0.0,
                   highlight_ions=None, shown_harmonics=None, lines_out=None,
                   corr_by_h=None, ref_search_hz=0.0, warnings_out=None, refs_out=None):
    """Run the full computation and return (figure, error text).

    shown_harmonics: harmonics to draw (None = all); set by the toggle box.
    lines_out: if a list, filled with the highlighted lines inside the
    spectrum, every harmonic (also hidden ones), sorted by frequency:
    dicts {ion, h, f, shown} (for the Zoom / ◀ / ▶ buttons)."""
    highlight_ions = highlight_ions or []
    try:
        full_database = pd.concat([v_get_chargestate(e, z_lims, n_lims, el1, el2)
                                   for e in (0, 1, 2, 3)])
        sorted_db = full_database.sort_values("m/q", ascending=False)

        mq_ref = sorted_db.loc[ref_name, "m/q"]
        db = sorted_db[(sorted_db["m/q"] >= mq_ref - mq_width) &
                       (sorted_db["m/q"] <= mq_ref + mq_width)]

        data = unpack_npz(file_path.rstrip("/") + "/" + spectrum_name)
        ff, pp = data["arr_0"], data["arr_1"]

        # each harmonic from the reference ion's own peak in it (core/physics.py)
        ref_freq = ref_frequency / ref_harmonic
        fs = np.asarray(ff, dtype=float)
        ps = np.asarray(pp, dtype=float)
        refs, rinfo = physics.harmonic_references(fs, ps, ref_frequency, ref_harmonic, harmonics,
                                                  ref_search_hz)
        if warnings_out is not None:
            warnings_out += physics.reference_warnings(rinfo, ref_name, ref_search_hz)
        if refs_out is not None:
            refs_out += physics.reference_rows(rinfo, ref_harmonic)
        db = v_compute_frequencies_multi(db, "m/q", ref_name, ref_freq, gamma_t, harmonics,
                                         correction_enabled=correction_enabled,
                                         corr_a=corr_a, corr_b=corr_b, corr_c=corr_c,
                                         ref_rev_by_h={h: f / h for h, f in refs.items()},
                                         corr_by_h=corr_by_h)

        n_points = 20
        fig = go.Figure()
        # "hvh" centres each step on its frequency value
        fig.add_trace(go.Scattergl(x=ff, y=pp, mode="lines", name="Spectrum", hoverinfo="skip",
                                   line=dict(shape="hvh", color="black", width=1)))

        ymin, ymax = np.nanmin(pp), np.nanmax(pp)
        pad = (ymax - ymin) * 0.05
        y_range = np.linspace(ymin - pad, ymax + pad, n_points)
        freq_min, freq_max = np.nanmin(ff), np.nanmax(ff)

        HIGHLIGHT_WIDTH = 3
        multi_harmonic = len(harmonics) > 1
        shown = [h for h in harmonics if shown_harmonics is None or h in shown_harmonics]

        # highlighted lines of every harmonic, for the zoom buttons
        if lines_out is not None and highlight_ions:
            hi_names = [n for n in dict.fromkeys(db.index)
                        if v_ion_matches_highlight(str(n), highlight_ions)]
            for h in harmonics:
                col = f"frequency_direct_{h}"
                for ion_name in hi_names:
                    f = db.loc[ion_name, col]
                    if isinstance(f, pd.Series):     # (duplicate ion name)
                        f = f.iloc[0]
                    if pd.isna(f) or f < freq_min or f > freq_max:
                        continue
                    lines_out.append({"ion": str(ion_name), "h": int(h), "f": float(f),
                                      "shown": h in shown})
            lines_out.sort(key=lambda l: l["f"])

        highlighted = {}            # (ion, h) -> [xs, ys, hover, f]
        first_sim = True

        for h in shown:
            col = f"frequency_direct_{h}"
            xs_norm, ys_norm, hover_norm = [], [], []
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
                    d = highlighted.setdefault((str(ion_name), h), [[], [], [], f])
                    d[0] += seg_x; d[1] += seg_y; d[2] += seg_h
                else:
                    xs_norm += seg_x; ys_norm += seg_y; hover_norm += seg_h
            # all other simulated lines: grey, as "Simulated (no peak)" in the
            # other tabs; one legend entry for all harmonics (legend group)
            if xs_norm:
                fig.add_trace(go.Scattergl(x=xs_norm, y=ys_norm, mode="lines",
                                           line=V_SIM_STYLE, name="Simulated",
                                           legendgroup="sim", showlegend=first_sim,
                                           hoverinfo="text", hovertext=hover_norm,
                                           connectgaps=False))
                first_sim = False

        # highlighted ions: a colour and a legend entry for each ion and harmonic
        highlighted_annotations = []
        for i, ((ion_name, h), (xs, ys, hov, f)) in enumerate(sorted(highlighted.items())):
            color = V_HIGHLIGHT_COLORS[i % len(V_HIGHLIGHT_COLORS)]
            fig.add_trace(go.Scattergl(x=xs, y=ys, mode="lines",
                                       line=dict(dash="solid", color=color, width=HIGHLIGHT_WIDTH),
                                       name=f"{ion_name} (h={h})", hoverinfo="text", hovertext=hov,
                                       connectgaps=False))
            highlighted_annotations.append((ion_name, f, h, color))

        # labelled arrows for highlighted ions; three offsets so labels at
        # similar frequencies don't sit exactly on top of each other
        y_tip = ymax + pad
        AY_OFFSETS = [-38, -58, -78]
        for idx, (ion_name, freq, h, color) in enumerate(sorted(highlighted_annotations,
                                                                key=lambda a: a[1])):
            text = ion_name if not multi_harmonic else \
                f"{ion_name}<br><span style='font-size:10px'>h={h}</span>"
            fig.add_annotation(x=freq, y=y_tip, text=text, showarrow=True, arrowhead=2,
                               arrowsize=1, arrowwidth=1.5, arrowcolor=color, ax=0,
                               ay=AY_OFFSETS[idx % len(AY_OFFSETS)],
                               font=dict(size=11, color=color),
                               bgcolor="rgba(255,255,255,0.88)", bordercolor=color,
                               borderwidth=1, borderpad=3, align="center")

        # (the title is the heading of the plot's box, see viewer_tab)
        fig.update_layout(xaxis_title="Frequency (Hz)", yaxis_title="Schottky Power (a.u.)",
                          hovermode="x", dragmode="pan", template="plotly_white",
                          # top margin: room for the plot controls (mode bar)
                          # above the legend box, so they don't overlap
                          font=dict(size=18), margin=dict(l=60, r=20, t=50, b=50),
                          # keep the zoom when toggling harmonics or changing
                          # settings; a new spectrum starts unzoomed
                          uirevision=f"{file_path}|{spectrum_name}")
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
    corr_profile       = "",
    ref_search_hz      = 0.0,
    highlight_ions     = "",
)
VIEWER_KEYS = list(VIEWER_DEFAULTS)


def load_viewer_config():
    """Viewer settings: VIEWER_DEFAULTS, shared physics, then the viewer's overrides."""
    return physics.with_shared(VIEWER_DEFAULTS, _load_json(VIEWER_CONFIG_PATH))


# Viewer settings saved in viewer_config.json that are not plot inputs (so
# changing them doesn't redraw the plot); kept when the others are saved
V_EXTRA_KEYS = ("zoom_width_khz",)
V_ZOOM_DEFAULT_KHZ = 10.0


def v_zoom_width_saved():
    v = _load_json(VIEWER_CONFIG_PATH).get("zoom_width_khz")
    try:
        v = float(v)
    except (TypeError, ValueError):
        return V_ZOOM_DEFAULT_KHZ
    return v if np.isfinite(v) and v > 0 else V_ZOOM_DEFAULT_KHZ


def save_viewer_config(cfg):
    """Save the viewer settings in cfg; the V_EXTRA_KEYS not in cfg keep
    their saved values."""
    try:
        with jsonfile.lock(VIEWER_CONFIG_PATH):
            old = _load_json(VIEWER_CONFIG_PATH)
            extra = {k: cfg.get(k, old.get(k)) for k in V_EXTRA_KEYS
                     if cfg.get(k, old.get(k)) is not None}
            main = {k: v for k, v in cfg.items() if k not in V_EXTRA_KEYS}
            jsonfile.write(VIEWER_CONFIG_PATH, {**physics.strip_shared(main), **extra})
    except Exception as exc:
        jsonfile.warn(f"could not save the viewer settings to {VIEWER_CONFIG_PATH}: "
                      f"{type(exc).__name__}: {exc}")


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
    html.H5("Viewer Settings", style={"fontWeight": "700", "marginBottom": "4px"}),
    *section("Spectrum"),
    with_browse(v_input("File Path", "file_path"), "btn-browse-v-file_path"),
    html.Div([
        html.Label("Spectrum", style=LABEL_STYLE),
        dcc.Dropdown(id="v-spectrum_name", value=str(VCFG["spectrum_name"]), clearable=False,
                     options=[{"label": n, "value": n} for n in list_npz(VCFG["file_path"])],
                     placeholder="No .npz files in this folder",
                     style={"fontSize": "13px"}),
    ], style=INPUT_STYLE),
    *section("Reference Ion"),
    v_input("Ion Name", "ref_name"),
    v_input("Frequency (Hz)", "ref_frequency", True, 1),
    v_input("Harmonic", "ref_harmonic", True, 1),
    v_input("Ref. Peak Search in the Other Harmonics ± (Hz)", "ref_search_hz", True, "any",
            help="Each harmonic's lines are placed from the reference ion's own peak in that "
                 "harmonic (nearest to f_ref/h_ref·h). 0 = off"),
    *section("Ion Selection"),
    v_input("Z Limits  [min, max]", "z_lims"),
    v_input("N Limits  [min, max]", "n_lims"),
    v_input("el1  [A, Z]", "el1"),
    v_input("el2  [A, Z]", "el2"),
    v_input("m/q window  (± value)", "mq_width", True, 0.001),
    *section("Ring / Harmonics"),
    v_input("γ_t", "gamma_t", True, 0.001),
    v_input("Harmonics (comma-separated)", "harmonics"),
    *section("Frequency Correction"),
    dcc.Checklist(id="v-correction_enabled",
                  options=[{"label": " Second order correction", "value": "on"}],
                  value=VCFG["correction_enabled"], style={"fontSize": "13px", "marginBottom": "10px"}),
    html.Div([
        corr_profile_ui("v-", VCFG["corr_profile"], VCFG["corr_a"], VCFG["corr_b"], VCFG["corr_c"],
                        ref_name=VCFG["ref_name"], ref_h=VCFG["ref_harmonic"],
                        gamma_t=VCFG["gamma_t"], ref_f=VCFG["ref_frequency"],
                        harmonics=VCFG["harmonics"]),
        v_input("a  (f² coefficient)", "corr_a", True, "any"),
        v_input("b  (f coefficient)", "corr_b", True, "any"),
        v_input("c  (constant)", "corr_c", True, "any"),
    ], id="v-correction-inputs",
       style={"display": "block" if VCFG["correction_enabled"] else "none"}),
    *section("Highlight Ions"),
    v_input("Ion Names (comma-separated)", "highlight_ions", placeholder="e.g. 204Hg, 208Tl79+",
            help="Omit charge state to highlight all charge states (upper or lower case)"),
    # Zoom to each highlighted line in turn (left to right, every harmonic)
    html.Div([
        html.Div([
            html.Label("Zoom Window  ±", style={**LABEL_STYLE, "margin": "0 6px 0 0"}),
            dcc.Input(id="v-zoom-width", type="number", min=0, step="any", debounce=True,
                      value=v_zoom_width_saved(), placeholder="10",
                      className="form-control form-control-sm",
                      style={"width": "80px", "fontSize": "13px", "marginRight": "4px"}),
            html.Span("kHz", style={"fontSize": "13px"}),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "6px"}),
        html.Div([
            html.Button("Zoom", id="v-zoom-btn", className="btn btn-sm", disabled=True,
                        title="Zoom to the left-most highlighted line",
                        style={"backgroundColor": ACCENT, "color": "white"}),
            html.Button("◀", id="v-zoom-prev", className="btn btn-outline-secondary btn-sm",
                        disabled=True, title="Previous highlighted line (lower frequency)"),
            html.Button("▶", id="v-zoom-next", className="btn btn-outline-secondary btn-sm",
                        disabled=True, title="Next highlighted line (higher frequency)"),
            html.Button("Full View", id="v-zoom-full", className="btn btn-outline-secondary btn-sm",
                        disabled=True, title="Show the whole spectrum again"),
        ], style={"display": "flex", "gap": "6px", "flexWrap": "wrap"}),
        html.Div(id="v-zoom-status", style={"fontSize": "12px", "marginTop": "6px",
                                            "minHeight": "16px"}),
        dcc.Store(id="v-zoom-lines", data={"src": "", "lines": []}),
        dcc.Store(id="v-zoom-idx", data=None),
        dcc.Store(id="v-zoom", data=None),
        html.Div(id="_v-zoom-hook", style={"display": "none"}),
    ], style=INPUT_STYLE),
    html.Hr(style={"margin": "10px 0"}),
    html.Div(id="v-error-box", style={"color": "#c0392b", "fontSize": "12px",
                                      "whiteSpace": "pre-wrap"}),
    html.Div(style={"height": "30px"}),
], className="theia-sidebar",
   style={"width": "300px", "minWidth": "300px", "padding": "14px 18px", "background": "#f9f9f9",
          "overflowY": "auto", "borderRight": "1px solid #dddddd", "height": "calc(100vh - 42px)"})

def _v_harm_options(text):
    """Toggle-box options for the harmonics text ('212, 213, 214')."""
    try:
        hs = [int(p.strip()) for p in str(text).split(",") if p.strip()]
    except ValueError:
        hs = []
    return [{"label": f"h = {h}", "value": h} for h in hs], hs


_V_HARM_OPTS, _V_HARM_ON = _v_harm_options(VCFG["harmonics"])

viewer_tab = html.Div([
    viewer_sidebar,
    html.Div([
        # plot in a box with its title as the heading, as in the Scan Results tab
        html.Div([
            html.Div([
                html.H6("Calculated Frequencies Overlayed with Data",
                        style={"fontWeight": "600", "margin": "0"}),
                logbook_button("viewer", marginLeft="auto"),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "6px"}),
            html.Div([
                dcc.Graph(id="viewer-graph", style={"height": "100%"}, responsive=True,
                          config={"scrollZoom": True, "displayModeBar": True,
                                  "displaylogo": False}),
                # show/hide all lines (grey and highlighted) of each harmonic;
                # placed in the plot just under the legend by the clientside
                # callback v-harm-place (hidden until placed)
                html.Div([
                    html.Div("Harmonics", style={"fontSize": "15px", "marginBottom": "4px",
                                                 "color": "#2a3f5f"}),
                    dcc.Checklist(id="v-harm-toggle", options=_V_HARM_OPTS, value=_V_HARM_ON,
                                  className="harm-toggle", inputStyle={"display": "none"}),
                ], id="v-harm-overlay", title="Click to show or hide all simulated lines of a harmonic",
                   style={"position": "absolute", "top": "0", "left": "0", "zIndex": "10",
                          "visibility": "hidden"}),
                # one box round the legend and the toggles, same outline as the
                # plot's box (border only; clicks pass through to the legend)
                html.Div(id="v-legend-frame",
                         style={"position": "absolute", "top": "0", "left": "0", "zIndex": "9",
                                "border": BOX_BORDER, "borderRadius": "6px",
                                "pointerEvents": "none", "visibility": "hidden"}),
            ], style={"position": "relative", "flex": "1 1 0", "minHeight": "0"}),
            html.Div(id="v-ref-table", style={"flex": "0 0 auto", "maxHeight": "30vh",
                                              "overflowY": "auto"}),
            html.Div(id="_v-harm-hook", style={"display": "none"}),
        ], className="theia-box",
           style={**BOX_STYLE, "height": "100%", "display": "flex", "flexDirection": "column",
                    "boxSizing": "border-box"}),
    ], style={"flex": "1", "minWidth": "0", "height": "calc(100vh - 42px)",
              "padding": "10px 14px"}),
    html.Div(id="_viewer-saved", style={"display": "none"}),
], style={"display": "flex"})


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


register_corr_profile("s-", as_text=False, has_ref_freq=True)
register_corr_profile("v-", as_text=False, has_ref_freq=True)
register_corr_profile("pid-", as_text=True, has_ref_freq=False)


@app.callback(Output("v-correction-inputs", "style"), Input("v-correction_enabled", "value"))
def v_toggle_correction_inputs(enabled):
    return {"display": "block" if enabled else "none"}


@app.callback(Output("_viewer-saved", "children"), *_V_INPUTS, prevent_initial_call=True)
def v_persist_config(*values):
    save_viewer_config(dict(zip(VIEWER_KEYS, values)))
    return dash.no_update


def _v_ints(s):
    return [int(p.strip()) for p in str(s).split(",") if p.strip()]


@app.callback(Output("v-harm-toggle", "options"), Output("v-harm-toggle", "value"),
              Input("v-harmonics", "value"),
              State("v-harm-toggle", "options"), State("v-harm-toggle", "value"),
              prevent_initial_call=True)
def v_harmonic_toggles(text, old_opts, old_on):
    """New harmonics list: keep hidden the ones that were hidden, show new ones."""
    opts, hs = _v_harm_options(text)
    if not hs:
        return dash.no_update, dash.no_update
    before = {o["value"] for o in (old_opts or [])}
    on = set(old_on or [])
    return opts, [h for h in hs if h in on or h not in before]


# Put the harmonic toggles inside the plot, just under the legend; moved
# again after every redraw (legend length changes) and on window resize.
app.clientside_callback(
    """
    function(fig) {
        var tries = 0;
        function setup() {
            var gd = document.querySelector('#viewer-graph .js-plotly-plot');
            var ov = document.getElementById('v-harm-overlay');
            if (!gd || !ov || typeof gd.on !== 'function') {
                if (tries++ < 50) { setTimeout(setup, 100); }
                return;
            }
            var fr = document.getElementById('v-legend-frame');
            function place() {
                var lg = gd.querySelector('.legend');
                if (!lg) {
                    ov.style.visibility = 'hidden';
                    if (fr) { fr.style.visibility = 'hidden'; }
                    return;
                }
                var r = lg.getBoundingClientRect();
                var w = ov.parentElement.getBoundingClientRect();
                ov.style.left = (r.left - w.left + 10) + 'px';
                ov.style.top = (r.bottom - w.top + 6) + 'px';
                ov.style.visibility = 'visible';
                if (fr) {                        // box round legend + toggles
                    var o = ov.getBoundingClientRect();
                    var pad = 6;
                    fr.style.left = (r.left - w.left - pad) + 'px';
                    fr.style.top = (r.top - w.top - pad) + 'px';
                    fr.style.width = (Math.max(r.right, o.right) - r.left + 2 * pad) + 'px';
                    fr.style.height = (o.bottom - r.top + pad) + 'px';
                    fr.style.visibility = 'visible';
                }
            }
            if (!gd._harmHook) {
                gd._harmHook = true;
                gd.on('plotly_afterplot', place);
                window.addEventListener('resize', function() { setTimeout(place, 100); });
            }
            setTimeout(place, 0);
        }
        setup();
        return '';
    }
    """,
    Output("_v-harm-hook", "children"),
    Input("viewer-graph", "figure"),
)


@app.callback(Output("viewer-graph", "figure"), Output("v-error-box", "children"),
              Output("v-zoom-lines", "data"), Output("v-ref-table", "children"), *_V_INPUTS,
              Input("v-harm-toggle", "value"))
def v_update_figure(*values):
    shown = values[-1]
    v = dict(zip(VIEWER_KEYS, values[:-1]))
    src = f"{v['file_path']}|{v['spectrum_name']}"
    if not v["spectrum_name"]:
        return (empty_fig("No spectrum selected (no .npz files in this folder?)"), "",
                {"src": src, "lines": []}, None)
    lines, warns, refs = [], [], []
    try:
        corr = corrections.resolve(v.get("corr_profile"), v["corr_a"], v["corr_b"], v["corr_c"])
        fig, err = v_build_figure(
            file_path=v["file_path"] or "", spectrum_name=v["spectrum_name"] or "",
            ref_name=v["ref_name"] or "", ref_frequency=float(v["ref_frequency"] or 0),
            ref_harmonic=int(float(v["ref_harmonic"] or 1)),
            z_lims=_v_ints(v["z_lims"]), n_lims=_v_ints(v["n_lims"]),
            el1=_v_ints(v["el1"]), el2=_v_ints(v["el2"]),
            mq_width=float(v["mq_width"] or 0.01), gamma_t=float(v["gamma_t"] or 1.399),
            harmonics=_v_ints(v["harmonics"]),
            correction_enabled=bool(v["correction_enabled"]),
            corr_a=float(corr["general"][0]), corr_b=float(corr["general"][1]),
            corr_c=float(corr["general"][2]), corr_by_h=corr["by_h"],
            ref_search_hz=float(v.get("ref_search_hz") or 0),
            highlight_ions=v_parse_highlight_ions(v["highlight_ions"] or ""),
            shown_harmonics=None if shown is None else [int(h) for h in shown],
            lines_out=lines, warnings_out=warns, refs_out=refs)
        if warns and not err:
            err = "\n".join("⚠ " + w for w in warns)
        return fig, err, {"src": src, "lines": lines}, v_ref_table(refs, v["ref_name"],
                                                                    v.get("ref_search_hz"))
    except Exception as exc:
        return empty_fig(), f"Parse error: {exc}", {"src": src, "lines": []}, None


def v_ref_table(rows, ref_name, window):
    """Viewer: reference peak in each harmonic, under the plot."""
    if not rows:
        return None
    w = float(window or 0)
    note = (f"searched within ±{w:g} Hz of f_ref / h_ref × h" if w > 0 else
            "search off (Reference search ± = 0): predicted positions used")
    return html.Div([
        html.Div([html.Span(f"{(ref_name or 'Reference').strip()} peak in each harmonic",
                            style={"fontWeight": "600", "marginRight": "10px"}),
                  html.Span(note, style={"color": "#888", "fontSize": "11px"})],
                 style={"fontSize": "13px", "margin": "6px 0 4px 0"}),
        ref_peak_table("v-ref-dt", _ref_table_records(rows))])


def _v_zoom_hw(width_khz):
    """Half-width (Hz) of the zoom window from the ± kHz box (default 10 kHz)."""
    try:
        v = float(width_khz)
    except (TypeError, ValueError):
        return V_ZOOM_DEFAULT_KHZ * 1000.0
    return v * 1000.0 if np.isfinite(v) and v > 0 else V_ZOOM_DEFAULT_KHZ * 1000.0


def _v_save_extra(key, value):
    """Save one of V_EXTRA_KEYS in viewer_config.json, other entries untouched."""
    try:
        jsonfile.update(VIEWER_CONFIG_PATH, {key: value})
    except Exception as exc:
        jsonfile.warn(f"could not save the viewer settings to {VIEWER_CONFIG_PATH}: "
                      f"{type(exc).__name__}: {exc}")


@app.callback(
    Output("v-zoom", "data"),
    Output("v-zoom-idx", "data"),
    Output("v-zoom-width", "style"),
    Input("v-zoom-btn", "n_clicks"),
    Input("v-zoom-prev", "n_clicks"),
    Input("v-zoom-next", "n_clicks"),
    Input("v-zoom-full", "n_clicks"),
    Input("v-zoom-lines", "data"),
    Input("v-zoom-width", "value"),
    State("v-zoom-idx", "data"),
    prevent_initial_call=True,
)
def v_zoom_nav(_z, _p, _n, _f, store, width, cur):
    """Zoom: left-most highlighted line; ◀ / ▶: previous / next line;
    Full view: unzoom.  cur (v-zoom-idx) is the line zoomed to: {i, src,
    ion, h, f}, or None when not zoomed.  New lines (plot redrawn): stay on
    the same ion and harmonic if it is still highlighted, and zoom again only
    if the spectrum or the line's frequency changed (so a manual pan isn't
    undone by e.g. a harmonic toggle)."""
    trig = dash.callback_context.triggered[0]["prop_id"] if dash.callback_context.triggered else ""
    store = store or {}
    lines = store.get("lines") or []
    src = store.get("src")
    nu = dash.no_update

    w_style = {"width": "80px", "fontSize": "13px", "marginRight": "4px"}
    try:
        wv = float(width) if width not in (None, "") else None
    except (TypeError, ValueError):
        wv = -1
    if wv is not None and not (np.isfinite(wv) and wv > 0):
        w_style["borderColor"] = "#c0392b"
    hw = _v_zoom_hw(width)

    def zoom_to(i, reapply=False):
        l = lines[i]
        return ({"x": [l["f"] - hw, l["f"] + hw], "reapply": reapply, "t": time.time()},
                {"i": i, "src": src, "ion": l["ion"], "h": l["h"], "f": l["f"]}, w_style)

    if trig.startswith("v-zoom-full"):
        return {"full": True, "t": time.time()}, None, w_style
    if trig.startswith("v-zoom-btn"):
        return zoom_to(0) if lines else (nu, None, w_style)
    if trig.startswith("v-zoom-prev") or trig.startswith("v-zoom-next"):
        if not lines:
            return nu, None, w_style
        step = -1 if trig.startswith("v-zoom-prev") else 1
        i = 0 if cur is None else min(max(cur["i"] + step, 0), len(lines) - 1)
        return zoom_to(i)
    if trig.startswith("v-zoom-width"):
        if wv is not None and wv > 0:
            _v_save_extra("zoom_width_khz", wv)
        if cur is None or cur["i"] >= len(lines):
            return nu, nu, w_style
        return zoom_to(cur["i"])
    # v-zoom-lines: the plot was redrawn
    if cur is None:
        return nu, nu, w_style
    for i, l in enumerate(lines):
        if (l["ion"], l["h"]) == (cur["ion"], cur["h"]):
            if src != cur["src"] or l["f"] != cur["f"]:
                return zoom_to(i, reapply=True)
            return nu, {**cur, "i": i}, w_style
    return nu, None, w_style


@app.callback(
    Output("v-zoom-btn", "disabled"),
    Output("v-zoom-prev", "disabled"),
    Output("v-zoom-next", "disabled"),
    Output("v-zoom-full", "disabled"),
    Output("v-zoom-status", "children"),
    Input("v-zoom-lines", "data"),
    Input("v-zoom-idx", "data"),
    Input("v-highlight_ions", "value"),
)
def v_zoom_buttons(store, cur, hl):
    lines = (store or {}).get("lines") or []
    n = len(lines)
    if not n:
        msg = ("No highlighted lines in this spectrum" if (hl or "").strip()
               else "Highlight ions above to zoom to their lines")
        return True, True, True, cur is None, html.Span(msg, style={"color": "#888"})
    if cur is None or cur["i"] >= n:
        return (False, True, False, True,
                html.Span(f"{n} highlighted line{'' if n == 1 else 's'} in this spectrum",
                          style={"color": "#888"}))
    i = cur["i"]
    l = lines[i]
    txt = f"{i + 1} / {n}:  {l['ion']}  h{l['h']}  ({l['f']:,.1f} Hz)"
    kids = [html.Span(txt, style={"fontWeight": "600"})]
    if not l.get("shown", True):
        kids.append(html.Span(f"  h{l['h']} hidden", style={"color": "#888"},
                              title="This harmonic is switched off in the Harmonics box, "
                                    "so its lines are not drawn"))
    return False, i == 0, i == n - 1, False, html.Span(kids)


# Zoom the Viewer plot to the chosen window, y fitted to the data inside it;
# {"full": true} shows the whole spectrum.  "reapply": the plot is being
# redrawn (new spectrum), so the zoom is applied again once it has been drawn.
app.clientside_callback(
    """
    function(z) {
        if (!z) return window.dash_clientside.no_update;
        function apply() {
            var gd = document.querySelector('#viewer-graph .js-plotly-plot');
            if (!gd || !gd._fullData || !gd._fullData.length) return;
            if (z.full) {
                Plotly.relayout(gd, {'xaxis.autorange': true, 'yaxis.autorange': true});
                return;
            }
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
        }
        apply();
        if (z.reapply) { setTimeout(apply, 250); }
        return window.dash_clientside.no_update;
    }
    """,
    Output("_v-zoom-hook", "children"),
    Input("v-zoom", "data"),
    prevent_initial_call=True,
)
