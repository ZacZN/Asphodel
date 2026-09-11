from my_utils.nubase import *
from my_utils.data_processing import *
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import os

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc


# ---------------------------------------------------------------------------
# Physics / data functions  (unchanged from original)
# ---------------------------------------------------------------------------

def delta_f(ref_frequency, ref_mq, delta_mq, gamma_t):
    return (-1 / (gamma_t**2)) * (delta_mq / ref_mq) * ref_frequency


def get_chargestate(electrons, z_lims, n_lims, el1, el2):
    dataframe = df_nubase(electrons=electrons)
    dataframe = rect_restrict(df=dataframe, z_lims=z_lims, n_lims=n_lims)
    dataframe = diag_restrict(df=dataframe, el1=el1, el2=el2)
    dataframe = index_by_name(df=dataframe, electrons=electrons)
    return dataframe


def compute_frequencies_multi(df, mq_col, ref_ion_name, ref_freq, gamma_t, harmonics,
                              correction_enabled=False, corr_a=0.0, corr_b=0.0, corr_c=0.0):
    result = df.sort_values(mq_col).copy()
    for h in harmonics:
        ref_mq = result.loc[ref_ion_name, mq_col]

        def _direct(row_mq, _h=h, _ref_mq=ref_mq):
            if row_mq == _ref_mq:
                f_true = ref_freq
            else:
                delta_mq = row_mq - _ref_mq
                f_true = ref_freq + delta_f(ref_freq, _ref_mq, delta_mq, gamma_t)
            # Apply correction to the true (revolution) frequency before harmonic scaling
            if correction_enabled:
                f_true = apply_correction(f_true, corr_a, corr_b, corr_c)
            return f_true * _h

        result[f"frequency_direct_{h}"] = result[mq_col].apply(_direct)
    return result


def apply_correction(freq, a, b, c):
    """Return freq + a*freq**2 + b*freq + c (second-order polynomial correction)."""
    return freq + a * freq**2 + b * freq + c


def build_figure(file_path, spectrum_name, ref_name, ref_frequency, ref_harmonic,
                 z_lims, n_lims, el1, el2, mq_width, gamma_t, harmonics,
                 correction_enabled=False, corr_a=0.0, corr_b=0.0, corr_c=0.0):
    """Run the full computation and return a Plotly figure, or an error figure."""
    try:
        # Ion database
        fully_stripped = get_chargestate(0, z_lims, n_lims, el1, el2)
        h_like         = get_chargestate(1, z_lims, n_lims, el1, el2)
        he_like        = get_chargestate(2, z_lims, n_lims, el1, el2)
        li_like        = get_chargestate(3, z_lims, n_lims, el1, el2)
        full_database  = pd.concat([fully_stripped, h_like, he_like, li_like])
        sorted_db      = full_database.sort_values("m/q", ascending=False)

        # m/q window
        mq_ref   = sorted_db.loc[ref_name, "m/q"]
        mq_lower = mq_ref - mq_width
        mq_upper = mq_ref + mq_width
        db = sorted_db[(sorted_db["m/q"] >= mq_lower) & (sorted_db["m/q"] <= mq_upper)]

        # Frequencies
        ref_freq = ref_frequency / ref_harmonic
        db = compute_frequencies_multi(
            db, "m/q", ref_name, ref_freq, gamma_t, harmonics,
            correction_enabled=correction_enabled,
            corr_a=corr_a, corr_b=corr_b, corr_c=corr_c,
        )

        # Spectrum
        fullname = file_path.rstrip("/") + "/" + spectrum_name
        data = unpack_npz(fullname)
        ff, pp = data["arr_0"], data["arr_1"]

        # Build figure
        n_points = 20
        fig = go.Figure()
        fig.add_trace(go.Scattergl(
            x=ff, y=pp,
            mode="lines",
            line=dict(shape="hv", color="black", width=1),
            name="Spectrum",
            hoverinfo="skip"
        ))

        ymin, ymax = np.nanmin(pp), np.nanmax(pp)
        pad = (ymax - ymin) * 0.05
        y_range = np.linspace(ymin - pad, ymax + pad, n_points)

        colors = px.colors.qualitative.Plotly
        freq_cols = [f"frequency_direct_{h}" for h in harmonics]
        for i, col in enumerate(freq_cols):
            color = colors[i % len(colors)]
            xs, ys, hover = [], [], []
            labels = db.index
            for ion_name, f in zip(labels, db[col]):
                if pd.isna(f):
                    continue
                xs    += [f] * n_points + [None]
                ys    += list(y_range)  + [None]
                hover += [str(ion_name)] * n_points + [None]
            fig.add_trace(go.Scattergl(
                x=xs, y=ys,
                mode="lines",
                line=dict(dash="dash", color=color, width=1),
                name=col,
                hoverinfo="text",
                hovertext=hover,
                connectgaps=False
            ))

        fig.update_layout(
            title="Calculated Frequencies overlayed with data",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Schottky Power (a.u.)",
            hovermode="x",
            dragmode="pan",
            template="plotly_white",
            font=dict(size=18),
            margin=dict(l=60, r=20, t=50, b=50),
        )
        return fig, ""

    except Exception as exc:
        fig = go.Figure()
        fig.update_layout(
            title=f"Error: {exc}",
            template="plotly_white",
            annotations=[dict(text=str(exc), x=0.5, y=0.5, showarrow=False,
                              font=dict(size=16, color="red"), xref="paper", yref="paper")]
        )
        return fig, str(exc)


# ---------------------------------------------------------------------------
# Default values and persistent config
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    file_path          = "/home/zn00092/Desktop/1dSpectra/SplitTDMS_Output_2^17_nobaseline/",
    spectrum_name      = "1129.1.noBKG.npz",
    ref_name           = "204Hg79+",
    ref_frequency      = 411910517,
    ref_harmonic       = 213,
    z_lims             = "60, 83",
    n_lims             = "100, 127",
    el1                = "156, 66",
    el2                = "200, 82",
    mq_width           = 0.01,
    gamma_t            = 1.399,
    harmonics          = "212, 213, 214",
    correction_enabled = [],
    corr_a             = 0,
    corr_b             = 0,
    corr_c             = 0,
)

# JSON file written next to viewer.py so it survives restarts
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viewer_config.json")


def load_config() -> dict:
    """Return saved settings merged on top of DEFAULTS; fall back silently on any error."""
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as fh:
                return {**DEFAULTS, **json.load(fh)}
        except Exception:
            pass
    return dict(DEFAULTS)


def save_config(cfg: dict):
    """Persist settings to JSON next to viewer.py (best-effort; failures are ignored)."""
    try:
        with open(CONFIG_PATH, "w") as fh:
            json.dump(cfg, fh, indent=2)
    except Exception:
        pass


# Loaded once at startup; every sidebar widget uses this for its initial value
CFG = load_config()


# ---------------------------------------------------------------------------
# Dash layout
# ---------------------------------------------------------------------------
LABEL_STYLE  = {"fontWeight": "600", "marginBottom": "2px", "fontSize": "13px"}
INPUT_STYLE  = {"width": "100%", "marginBottom": "12px"}

def make_input(label, id_, value, input_type="text", step=None):
    kwargs = dict(id=id_, value=str(value), debounce=True,
                  style={"width": "100%", "fontSize": "13px"},
                  className="form-control form-control-sm")
    if input_type == "number":
        kwargs["type"] = "number"
        if step is not None:
            kwargs["step"] = step
    return html.Div([
        html.Label(label, style=LABEL_STYLE),
        dcc.Input(**kwargs),
    ], style=INPUT_STYLE)


sidebar = html.Div([
    html.H5("Parameters", style={"marginBottom": "16px", "fontWeight": "700"}),

    html.Hr(style={"marginBottom": "12px"}),
    html.P("Spectrum", style={"fontWeight": "600", "color": "#555", "marginBottom": "6px", "fontSize": "12px"}),
    make_input("File path",      "file-path",     CFG["file_path"]),
    make_input("Spectrum name",  "spectrum-name", CFG["spectrum_name"]),

    html.Hr(style={"marginBottom": "12px"}),
    html.P("Reference ion", style={"fontWeight": "600", "color": "#555", "marginBottom": "6px", "fontSize": "12px"}),
    make_input("Ion name",        "ref-name",      CFG["ref_name"]),
    make_input("Frequency (Hz)",  "ref-frequency", CFG["ref_frequency"], "number", 1),
    make_input("Harmonic",        "ref-harmonic",  CFG["ref_harmonic"],  "number", 1),

    html.Hr(style={"marginBottom": "12px"}),
    html.P("Ion selection", style={"fontWeight": "600", "color": "#555", "marginBottom": "6px", "fontSize": "12px"}),
    make_input("Z limits  [min, max]",  "z-lims",  CFG["z_lims"]),
    make_input("N limits  [min, max]",  "n-lims",  CFG["n_lims"]),
    make_input("el1  [N, Z]",           "el1",     CFG["el1"]),
    make_input("el2  [N, Z]",           "el2",     CFG["el2"]),
    make_input("m/q window  (± value)", "mq-width", CFG["mq_width"], "number", 0.001),

    html.Hr(style={"marginBottom": "12px"}),
    html.P("Ring / harmonics", style={"fontWeight": "600", "color": "#555", "marginBottom": "6px", "fontSize": "12px"}),
    make_input("γ_t",       "gamma-t",   CFG["gamma_t"],  "number", 0.001),
    make_input("Harmonics (comma-separated)", "harmonics", CFG["harmonics"]),

    html.Hr(style={"marginBottom": "12px"}),
    html.P("Frequency correction", style={"fontWeight": "600", "color": "#555", "marginBottom": "6px", "fontSize": "12px"}),
    html.Div([
        dcc.Checklist(
            id="correction-enabled",
            options=[{"label": " Second order correction", "value": "on"}],
            value=CFG["correction_enabled"],
            style={"fontSize": "13px", "marginBottom": "10px"},
        ),
    ]),
    html.Div(id="correction-inputs", children=[
        make_input("a  (f² coefficient)", "corr-a", CFG["corr_a"], "number", "any"),
        make_input("b  (f coefficient)",  "corr-b", CFG["corr_b"], "number", "any"),
        make_input("c  (constant)",       "corr-c", CFG["corr_c"], "number", "any"),
    ], style={"display": "block" if CFG["correction_enabled"] else "none"}),

    html.Hr(style={"marginBottom": "12px"}),
    html.Div(id="error-box", style={"color": "red", "fontSize": "12px", "whiteSpace": "pre-wrap"}),

], style={
    "width": "280px",
    "minWidth": "280px",
    "padding": "18px",
    "background": "#f8f9fa",
    "overflowY": "auto",
    "borderRight": "1px solid #dee2e6",
    "height": "100vh",
})

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Theia Spectrum Viewer"

app.layout = html.Div([
    sidebar,
    html.Div([
        dcc.Graph(
            id="spectrum-graph",
            style={"height": "100vh"},
            config={"scrollZoom": True, "displayModeBar": True},
        )
    ], style={"flex": "1", "minWidth": "0"}),
    # Hidden hook used by the clientside callback to attach the readout
    html.Div(id="_js-hook", style={"display": "none"}),
    # Dummy store — target for the config-persist callback
    dcc.Store(id="_config-store"),
], style={"display": "flex", "flexDirection": "row", "height": "100vh", "overflow": "hidden"})


# ---------------------------------------------------------------------------
# Clientside callback — injects the coordinate readout after each render
# ---------------------------------------------------------------------------
app.clientside_callback(
    """
    function(figure) {
        setTimeout(function() {
            // In Dash, getElementById gives the wrapper div; the actual Plotly
            // element (which has _fullLayout / p2d) is the .js-plotly-plot inside it.
            var gd = document.querySelector('#spectrum-graph .js-plotly-plot');
            if (!gd) return;

            // Create readout once, reuse on subsequent renders
            var readout = document.getElementById('_coord-readout');
            if (!readout) {
                readout = document.createElement('div');
                readout.id = '_coord-readout';
                readout.style.cssText = [
                    'position:fixed',
                    'bottom:12px',
                    'right:16px',
                    'background:rgba(30,30,30,0.80)',
                    'color:#e8e8e8',
                    'padding:5px 12px',
                    'border-radius:5px',
                    'font-family:monospace',
                    'font-size:16px',
                    'pointer-events:none',
                    'z-index:9999',
                    'letter-spacing:0.03em',
                    'white-space:pre'
                ].join(';');
                readout.innerText = 'x: ---\\ny: ---';
                document.body.appendChild(readout);

                gd.addEventListener('mousemove', function(evt) {
                    var layout = gd._fullLayout;
                    if (!layout) return;
                    var rect = gd.getBoundingClientRect();
                    var mx = evt.clientX - rect.left - layout.margin.l;
                    var my = evt.clientY - rect.top  - layout.margin.t;
                    var x = layout.xaxis.p2d(mx);
                    var y = layout.yaxis.p2d(my);
                    if (isFinite(x) && isFinite(y)) {
                        readout.innerText = 'x: ' + x.toFixed(0) + ' Hz\\ny: ' + y.toExponential(3);
                    }
                });

                gd.addEventListener('mouseleave', function() {
                    readout.innerText = 'x: ---\\ny: ---';
                });
            }
        }, 150);
        return window.dash_clientside.no_update;
    }
    """,
    Output("_js-hook", "children"),
    Input("spectrum-graph", "figure"),
    prevent_initial_call=False,
)


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------
def parse_int_pair(s):
    parts = [x.strip() for x in s.split(",")]
    return [int(p) for p in parts]

def parse_int_list(s):
    parts = [x.strip() for x in s.split(",")]
    return [int(p) for p in parts]


@app.callback(
    Output("correction-inputs", "style"),
    Input("correction-enabled", "value"),
)
def toggle_correction_inputs(enabled):
    if enabled:
        return {"display": "block"}
    return {"display": "none"}


@app.callback(
    Output("_config-store", "data"),
    Input("file-path",          "value"),
    Input("spectrum-name",      "value"),
    Input("ref-name",           "value"),
    Input("ref-frequency",      "value"),
    Input("ref-harmonic",       "value"),
    Input("z-lims",             "value"),
    Input("n-lims",             "value"),
    Input("el1",                "value"),
    Input("el2",                "value"),
    Input("mq-width",           "value"),
    Input("gamma-t",            "value"),
    Input("harmonics",          "value"),
    Input("correction-enabled", "value"),
    Input("corr-a",             "value"),
    Input("corr-b",             "value"),
    Input("corr-c",             "value"),
    prevent_initial_call=True,
)
def persist_config(file_path, spectrum_name, ref_name, ref_frequency, ref_harmonic,
                   z_lims, n_lims, el1, el2, mq_width, gamma_t, harmonics,
                   correction_enabled, corr_a, corr_b, corr_c):
    save_config(dict(
        file_path          = file_path,
        spectrum_name      = spectrum_name,
        ref_name           = ref_name,
        ref_frequency      = ref_frequency,
        ref_harmonic       = ref_harmonic,
        z_lims             = z_lims,
        n_lims             = n_lims,
        el1                = el1,
        el2                = el2,
        mq_width           = mq_width,
        gamma_t            = gamma_t,
        harmonics          = harmonics,
        correction_enabled = correction_enabled,
        corr_a             = corr_a,
        corr_b             = corr_b,
        corr_c             = corr_c,
    ))
    return dash.no_update


@app.callback(
    Output("spectrum-graph", "figure"),
    Output("error-box", "children"),
    Input("file-path",          "value"),
    Input("spectrum-name",      "value"),
    Input("ref-name",           "value"),
    Input("ref-frequency",      "value"),
    Input("ref-harmonic",       "value"),
    Input("z-lims",             "value"),
    Input("n-lims",             "value"),
    Input("el1",                "value"),
    Input("el2",                "value"),
    Input("mq-width",           "value"),
    Input("gamma-t",            "value"),
    Input("harmonics",          "value"),
    Input("correction-enabled", "value"),
    Input("corr-a",             "value"),
    Input("corr-b",             "value"),
    Input("corr-c",             "value"),
)
def update_figure(file_path, spectrum_name, ref_name, ref_frequency, ref_harmonic,
                  z_lims_str, n_lims_str, el1_str, el2_str,
                  mq_width, gamma_t, harmonics_str,
                  correction_enabled, corr_a, corr_b, corr_c):
    try:
        z_lims   = parse_int_pair(z_lims_str)
        n_lims   = parse_int_pair(n_lims_str)
        el1      = parse_int_pair(el1_str)
        el2      = parse_int_pair(el2_str)
        harmonics = parse_int_list(harmonics_str)
        fig, err = build_figure(
            file_path           = file_path or "",
            spectrum_name       = spectrum_name or "",
            ref_name            = ref_name or "",
            ref_frequency       = float(ref_frequency or 0),
            ref_harmonic        = int(ref_harmonic or 1),
            z_lims              = z_lims,
            n_lims              = n_lims,
            el1                 = el1,
            el2                 = el2,
            mq_width            = float(mq_width or 0.01),
            gamma_t             = float(gamma_t or 1.399),
            harmonics           = harmonics,
            correction_enabled  = bool(correction_enabled),
            corr_a              = float(corr_a or 0),
            corr_b              = float(corr_b or 0),
            corr_c              = float(corr_c or 0),
        )
        return fig, err
    except Exception as exc:
        empty = go.Figure()
        empty.update_layout(template="plotly_white")
        return empty, f"Parse error: {exc}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import threading
    import webbrowser
    url = "http://127.0.0.1:8050/"
    print(f"Starting Theia viewer at {url}")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(debug=False, host="127.0.0.1", port=8050)
