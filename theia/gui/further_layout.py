"""
Further output tab: page layout (Aligning and fitting, Isochronicity curve).

Part of Theia (theia/theia.py); split out of theia.py in Nov 2026.
"""

import os

from dash import dash_table, dcc, html

from core import paths, summing

from .common import (ACCENT, ACCENT_PALE, BOX_STYLE, LABEL_STYLE, RESID_BOX_HIDDEN,
                     STEP_CUR_STYLE, STEP_PAIR_STYLE, SUBTAB_SEL, SUBTAB_STYLE, TABLE_CELL,
                     TABLE_HEADER, TABLE_ODD)
from .figures import empty_fig
from .info import info_button
from .further_files import SUM_CONTROLS_SHOWN
from .further_fit import (CALIB_BORDER, CALIB_TD, CALIB_TH, fit_columns, FIT_PANEL_HIDDEN, FIX_ON,
                          FIX_PARAMS, LIMIT_KEYS, SHAPE_LABELS, SUGG_BOX_STYLE)
from .isocurve import isochronicity_tab


align_fit_tab = html.Div([
    html.H5("Aligning and Fitting", style={"fontWeight": "700"}),
    # Data source: the sum of the spectra in the confirmation files, or one
    # spectrum file (e.g. a Tartarus sum, or the file open in the Viewer)
    html.Div([
        html.Span("Data Source:", style={**LABEL_STYLE, "margin": "0 8px 0 0"}),
        dcc.RadioItems(id="further-source", value="sum", inline=True,
                       options=[{"label": "Confirmation files (summed)", "value": "sum"},
                                {"label": "Single spectrum file", "value": "file"}],
                       inputStyle={"marginRight": "4px"},
                       labelStyle={"marginRight": "14px", "fontSize": "13px"}),
    ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}),
    html.Div([
        # Numbered list of loaded confirmation files (any number); "+ Add File"
        # opens a picker, choosing a file appends it to the list.
        html.Div(id="further-list"),
        html.Div([
            html.Button("+ Add File", id="btn-further-add", className="btn btn-sm",
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
        html.Button("Use Viewer Spectrum", id="btn-fsrc-viewer",
                    className="btn btn-outline-secondary btn-sm",
                    title="Load the spectrum that is open in the Viewer tab"),
    ], id="further-file-src", style={"display": "none"}),
    dcc.Store(id="fsrc-load-req", data=None),
    dcc.Store(id="further-files", data=[]),
    html.Div([
        # Sum the spectra referenced by the loaded files, optionally aligned.
        html.Div([
            html.Div([
                html.Span("Sum Spectra", style={"fontWeight": "700", "marginRight": "4px"}),
                info_button("sum", {"marginRight": "14px"}),
                html.Span("Align On:", style={**LABEL_STYLE, "marginRight": "8px"}),
                dcc.RadioItems(id="sum-align", value="none", inline=True,
                               options=[{"label": "Not aligned", "value": "none"},
                                        {"label": "Strongest peak", "value": "strongest"},
                                        {"label": "Identified peak", "value": "identified"}],
                               inputStyle={"marginRight": "4px"},
                               labelStyle={"marginRight": "14px", "fontSize": "13px"}),
                html.Span([html.Span("±", style={"marginRight": "3px"}),
                           dcc.Input(id="sum-window", type="number", min=1,
                                     value=summing.DEFAULT_WINDOW_HZ, debounce=True,
                                     style={"width": "70px", "fontSize": "13px"}),
                           html.Span(" Hz", style={"marginLeft": "3px"})],
                          id="sum-window-box",
                          title="Strongest peak: each spectrum's peak is looked for within ± "
                                "this of the previous spectrum's peak",
                          style={"display": "none", "fontSize": "13px", "marginRight": "14px"}),
                dcc.Dropdown(id="sum-ion", options=[], placeholder="Ion / harmonic…",
                             clearable=False, style={"fontSize": "13px", "width": "240px",
                                                     "marginRight": "14px"}),
                dcc.Checklist(id="sum-unsure", value=["on"],
                              options=[{"label": " include unsure", "value": "on"}],
                              style={"fontSize": "13px", "marginRight": "14px"}),
                html.Button("Sum Spectra", id="btn-sum", className="btn btn-sm",
                            style={"backgroundColor": ACCENT, "color": "white",
                                   "marginRight": "14px"}),
            ], id="sum-only-controls", style=SUM_CONTROLS_SHOWN),
            html.Button("Load Spectrum", id="btn-fsrc-load", className="btn btn-sm",
                        title="Show the chosen spectrum file for fitting",
                        style={"backgroundColor": ACCENT, "color": "white", "marginRight": "14px",
                               "display": "none"}),
            html.Button("PID Settings", id="btn-pid", className="btn btn-outline-secondary btn-sm",
                        title="Settings used for the simulated ion lines",
                        style={"marginRight": "8px"}),
            html.Button("Search Ions…", id="btn-srch", className="btn btn-outline-secondary btn-sm",
                        title="Find an ion's simulated lines and zoom to a window around one",
                        style={"marginRight": "8px"}),
            # h−1 / h+1, q−1 / q+1 (charge state) and A−1 / A+1 (isotope) for
            # the ion chosen in Search ions
            # each pair has the current value between its buttons (as in the
            # spectrogram page), all three pairs inside one border
            html.Span([
                html.Span([
                    html.Button("h−1", id="btn-harm-dn", className="btn btn-outline-secondary btn-sm", disabled=True),
                    html.Span(id="harm-cur", style=STEP_CUR_STYLE),
                    html.Button("h+1", id="btn-harm-up", className="btn btn-outline-secondary btn-sm", disabled=True),
                ], style=STEP_PAIR_STYLE),
                html.Span([
                    html.Button("q−1", id="btn-q-dn", className="btn btn-outline-secondary btn-sm", disabled=True),
                    html.Span(id="q-cur", style=STEP_CUR_STYLE,
                              title="Charge state of the ion chosen in Search ions"),
                    html.Button("q+1", id="btn-q-up", className="btn btn-outline-secondary btn-sm", disabled=True),
                ], style=STEP_PAIR_STYLE),
                html.Span([
                    html.Button("A−1", id="btn-a-dn", className="btn btn-outline-secondary btn-sm", disabled=True),
                    html.Span(id="a-cur", style=STEP_CUR_STYLE,
                              title="Isotope of the ion chosen in Search ions "
                                    "(A±1 goes to the ground state)"),
                    html.Button("A+1", id="btn-a-up", className="btn btn-outline-secondary btn-sm", disabled=True),
                ], style=STEP_PAIR_STYLE),
            ], id="harm-grp", style={"display": "none"}),
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
            html.Button("Energy Calculation", id="btn-energy-mode",
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
                    ("Name", "Mass Excess (keV)", "Frequency (Hz)", "FWHM (Hz)")])),
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
                html.Button("Reset Calibration", id="btn-energy-reset",
                            className="btn btn-outline-secondary btn-sm", style={"marginRight": "8px"}),
                html.Button("Clear Pairs", id="btn-energy-clear-pairs",
                            className="btn btn-outline-secondary btn-sm"),
            ], style={"margin": "6px 0"}),
            dash_table.DataTable(
                id="energy-table", data=[],
                columns=[{"name": "Pair", "id": "pair"}, {"name": "f_A (Hz)", "id": "f_A"},
                         {"name": "f_B (Hz)", "id": "f_B"}, {"name": "f_B − f_A (Hz)", "id": "df"},
                         {"name": "E(B) − E(A) (keV)", "id": "de"},
                         {"name": "Frequencies From", "id": "src"}],
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
                html.Button("Peak Fitting", id="btn-fit-mode",
                            className="btn btn-outline-secondary btn-sm",
                            title="Turn on, then click peaks in the plot to add them to the fit",
                            style={"marginRight": "12px", "fontWeight": "600"}),
                info_button("fit"),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}),
            html.Div([
                html.Span("Shape for New Peaks:", style={**LABEL_STYLE, "margin": "0 6px 0 0",
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
                html.Span("Fit Region (Hz):", style={**LABEL_STYLE, "margin": "0 6px 0 0",
                                                     "minWidth": "135px"}),
                dcc.Input(id="fit-region-lo", type="number", step="any", debounce=True,
                          className="form-control form-control-sm",
                          style={"width": "170px", "marginRight": "4px"}),
                html.Span("–", style={"margin": "0 4px"}),
                dcc.Input(id="fit-region-hi", type="number", step="any", debounce=True,
                          className="form-control form-control-sm",
                          style={"width": "170px", "marginRight": "8px"}),
                html.Button("Region = View", id="btn-fit-region-view",
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
                html.Button("Clear Peaks", id="btn-fit-clear",
                            className="btn btn-outline-secondary btn-sm",
                            style={"marginRight": "8px"}),
                html.Button("Send to Isocurve", id="btn-iso-send",
                            className="btn btn-outline-secondary btn-sm",
                            title="Send the fitted peaks (each needs an ion name and harmonic) "
                                  "to the Isochronicity Curve tab"),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "6px"}),
            html.Div(id="fit-status", style={"fontSize": "12px"}),
            html.Div(id="iso-send-status", style={"fontSize": "12px", "marginTop": "2px"}),
        ]),
            html.Div([
                html.Div(["Fit Parameters", info_button("params", {"marginLeft": "6px"}),
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

        # each plot in its own box, as in the Scan Results tab
        html.Div(
            dcc.Graph(id="sum-graph", figure=empty_fig("Load files and click “Sum Spectra”"),
                      # doubleClick off: a touchpad double-tap (to drag/pan)
                      # otherwise triggers Plotly's reset/autoscale, which
                      # redraws all ~131k points and stalls the page.
                      config={"scrollZoom": True, "displaylogo": False, "doubleClick": False},
                      style={"height": "560px"}),
            className="theia-box", style={**BOX_STYLE, "padding": "8px 10px"}),
        # residuals box: shown by show_residuals() once there is a fit
        html.Div(
            dcc.Graph(id="fit-resid", figure=empty_fig(),
                      config={"scrollZoom": True, "displaylogo": False, "doubleClick": False},
                      style={"height": "200px"}),
            id="fit-resid-box", className="theia-box", style=RESID_BOX_HIDDEN),
    ], id="further-content", className="theia-box", style={**BOX_STYLE, "padding": "8px 10px"}),
], id="further-page-align", style={"padding": "14px 18px", "overflowY": "auto",
                                   "height": "calc(100vh - 82px)"})


# The sub-tab bar only switches which page is visible: both pages stay
# mounted, so plots keep their zoom and browser-side listeners.
FURTHER_PAGES = {"align": "further-page-align", "iso": "further-page-iso"}
further_tab = html.Div([
    dcc.Tabs(id="further-subtabs", value="align", children=[
        dcc.Tab(label="Aligning and Fitting", value="align", style=SUBTAB_STYLE,
                selected_style=SUBTAB_SEL),
        dcc.Tab(label="Isochronicity Curve", value="iso", style=SUBTAB_STYLE,
                selected_style=SUBTAB_SEL),
    ], style={"height": "40px"}),
    align_fit_tab,
    isochronicity_tab,
])
