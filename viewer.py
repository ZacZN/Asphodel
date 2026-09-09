from my_utils.nubase import *
from my_utils.data_processing import *
import matplotlib.pyplot as plt
import argparse
import tomli as tomllib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import tempfile
import webbrowser


class ConfigObject:
    """Converts a dictionary to object"""
    def __init__(self, data: dict):
        for key, value in data.items():
            # Convert nested dictionaries as well
            if isinstance(value, dict):
                setattr(self, key, ConfigObject())
            else:
                setattr(self, key, value)


def load_config(filepath: str) -> ConfigObject:
    with open(filepath, "rb") as f:
        data=tomllib.load(f)

    return ConfigObject(data)


# Take NUBASE data and make a pandas dataframe for one chargestate
# Restrict by N and Z bounds given in config
# Restrict by diagonal line based on elements given in config
# Then return the dataframe
def get_chargestate(electrons, config):
    dataframe=df_nubase(electrons=electrons)
    dataframe=rect_restrict(df=dataframe, z_lims=config.z_lims, n_lims=config.n_lims)
    dataframe=diag_restrict(df=dataframe, el1=config.el1, el2=config.el2)
    dataframe=index_by_name(df=dataframe, electrons=electrons)

    return dataframe


# Define the storage ring equation
# Will need to make gamma_t a variable later
def delta_f(ref_frequency, ref_mq, delta_mq, gamma_t):
    return (-1 / (gamma_t**2)) * (delta_mq / ref_mq) * ref_frequency


# Compute frequencies of other ions based on the reference
# Return a database with the frequencies appended as a new column
def compute_frequencies(df, mq_col, ref_ion_name, ref_freq, delta_f_func, gamma_t, harmonic):
    df = df.sort_values(mq_col).copy()

    if ref_ion_name not in df.index:
        raise KeyError(f"Ion '{ref_ion_name}' not found in list of ions")

    ref_mq = df.loc[ref_ion_name, mq_col]

    def _direct(row_mq):
        if row_mq == ref_mq:
            return ref_freq * harmonic
        delta_mq = row_mq - ref_mq

        return (ref_freq + delta_f_func(ref_freq, ref_mq, delta_mq, gamma_t)) * harmonic

    df["frequency_direct"] = df[mq_col].apply(_direct)

    return df


def compute_frequencies_multi(df, mq_col, ref_ion_name, ref_freq, delta_f_func, gamma_t, harmonics):
    result = df.sort_values(mq_col).copy()

    for h in harmonics:
        temp = compute_frequencies(
            df, mq_col, ref_ion_name, ref_freq, delta_f_func, gamma_t, harmonic=h
        )
        result[f"frequency_direct_{h}"] = temp["frequency_direct"]

    return result


def plot_spectrum_with_ions(frequency, signal, df, freq_cols=("frequency_direct",),
                             label_col=None, title="Calculated Frequencies overlayed with data",
                             n_points=20, colors=None):

    fig = go.Figure()

    # Plot the histogram
    fig.add_trace(go.Scattergl(
        x=frequency,
        y=signal,
        mode="lines",
        line=dict(shape="hv", color="black", width=1),
        name="Spectrum",
        hoverinfo="skip"
    ))

    ymin, ymax = np.nanmin(signal), np.nanmax(signal)
    pad = (ymax - ymin) * 0.05
    ymin, ymax = ymin - pad, ymax + pad
    y_range = np.linspace(ymin, ymax, n_points)

    def add_ion_lines(freq_series, color, name):
        xs, ys, hover = [], [], []
        labels = df[label_col] if label_col else df.index
        for ion_name, f in zip(labels, freq_series):
            if pd.isna(f):
                continue
            xs += [f] * n_points + [None]
            ys += list(y_range) + [None]
            hover += [str(ion_name)] * n_points + [None]

        fig.add_trace(go.Scattergl(
            x=xs, y=ys,
            mode="lines",
            line=dict(dash="dash", color=color, width=1),
            name=name,
            hoverinfo="text",
            hovertext=hover,
            connectgaps=False
        ))

    # Default color cycle if none provided
    if colors is None:
        colors = px.colors.qualitative.Plotly  # or any list of hex colors

    for i, col in enumerate(freq_cols):
        color = colors[i % len(colors)]
        add_ion_lines(df[col], color, col)

    fig.update_layout(
        title=title,
        xaxis_title="Frequency (Hz)",
        yaxis_title="Schottky Power (a.u.)",
        hovermode="x",
        dragmode="pan",
        template="plotly_white"
    )

    return fig


def show_with_coord_readout(fig, config=None):
    """Render the figure in a browser with a live mouse-coordinate readout overlay."""
    js = """
(function() {
    var gd = document.querySelector('.js-plotly-plot');
    if (!gd) return;

    var readout = document.createElement('div');
    readout.style.cssText = [
        'position:fixed',
        'bottom:12px',
        'right:16px',
        'background:rgba(30,30,30,0.80)',
        'color:#e8e8e8',
        'padding:5px 12px',
        'border-radius:5px',
        'font-family:monospace',
        'font-size:13px',
        'pointer-events:none',
        'z-index:9999',
        'letter-spacing:0.03em'
    ].join(';');
    readout.innerText = 'x: ---   y: ---';
    document.body.appendChild(readout);

    gd.addEventListener('mousemove', function(evt) {
        var layout = gd._fullLayout;
        if (!layout) return;
        var rect = gd.getBoundingClientRect();
        var mx = evt.clientX - rect.left - layout.margin.l;
        var my = evt.clientY - rect.top - layout.margin.t;
        var x = layout.xaxis.p2d(mx);
        var y = layout.yaxis.p2d(my);
        if (isFinite(x) && isFinite(y)) {
            readout.innerText = 'x: ' + x.toFixed(0) + ' Hz   y: ' + y.toExponential(3);
        }
    });

    gd.addEventListener('mouseleave', function() {
        readout.innerText = 'x: ---   y: ---';
    });
})();
"""
    config = config or {}
    html = fig.to_html(full_html=True, include_plotlyjs="cdn", config=config, post_script=js)
    # Write beside the script, not to /tmp — Firefox snap can't access /tmp
    import os
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_viewer_output.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    webbrowser.open("file://" + out_path)


def main():
    config = load_config("viewer_config.toml")
    # Create dataframes for each chargestate
    fully_stripped = get_chargestate(0, config)
    h_like = get_chargestate(1, config)
    he_like = get_chargestate(2, config)
    li_like = get_chargestate(3, config)
    # Join dataframes into one
    full_database = pd.concat([fully_stripped, h_like, he_like, li_like])
    # Sort by m/q
    sorted_database = full_database.sort_values(by="m/q", ascending=False)
    # Set m/q bounds and restrict to that range
    mq_ref = sorted_database.loc[config.ref_name, "m/q"]
    mq_upper = mq_ref + config.mq_width
    mq_lower = mq_ref - config.mq_width
    mq_restricted_database = sorted_database[(sorted_database["m/q"] >= mq_lower) & (sorted_database["m/q"] <= mq_upper)]
    # Calculate frequencies
    ref_freq = config.ref_frequency / config.ref_harmonic
    # frequencies_database = compute_frequencies(
    #     mq_restricted_database,
    #     mq_col="m/q",
    #     ref_ion_name=config.ref_name,
    #     ref_freq=ref_freq,
    #     delta_f_func=delta_f,
    #     gamma_t=config.gamma_t,
    #     harmonic=213
    # )

    frequencies_database = compute_frequencies_multi(
        mq_restricted_database,
        mq_col="m/q",
        ref_ion_name=config.ref_name,
        ref_freq=ref_freq,
        delta_f_func=delta_f,
        gamma_t=config.gamma_t,
        harmonics=config.harmonics
    )

    print(frequencies_database)

    # Load the spectrum
    fullname = config.file_path + config.spectrum_name
    data = unpack_npz(fullname)
    ff = data["arr_0"]
    pp = data["arr_1"]

    freq_cols = [f"frequency_direct_{h}" for h in config.harmonics]
    fig = plot_spectrum_with_ions(ff, pp, frequencies_database, freq_cols=freq_cols)
    show_with_coord_readout(fig, config={"scrollZoom": True})


if __name__ == "__main__":
    main()
