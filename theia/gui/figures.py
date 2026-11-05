"""
Spectrum figures: simulated ion lines and detected peaks.

Part of Theia (theia/theia.py); split out of theia.py in Nov 2026.
"""

import numpy as np
import plotly.graph_objects as go

from .common import ACCENT, ACCENT_PALE


# ===========================================================================
# Figures
# ===========================================================================

def _vlines(xs, y0, y1, hover, n_points=20):
    """Vertical lines as one trace.  Each line has n_points points (as in
    viewer.py) so its tooltip appears anywhere along it, not just at the ends."""
    ys = list(np.linspace(y0, y1, n_points))
    X, Y, H = [], [], []
    for x, h in zip(xs, hover):
        X += [x] * n_points + [None]
        Y += ys + [None]
        H += [h] * n_points + [None]
    return X, Y, H


def empty_fig(msg=""):
    fig = go.Figure()
    fig.update_layout(template="plotly_white", margin=dict(l=60, r=20, t=40, b=40))
    if msg:
        fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                           font=dict(size=15, color="#888"))
    return fig


EXCLUDED_TRACE = "Excluded (half-life cut)"


def add_sim_lines(fig, sims, ion_db, matches, y0, y1, highlight_ion=None,
                  excluded=None, show_excluded=True):
    """Draw simulated ion frequencies as vertical dashed lines with tooltips.

    Grey: no peak matched; dark grey: identified (a peak within the match
    window); orange: `highlight_ion`; light dotted: excluded from
    identification by the half-life cut (`excluded`, bool per ion_db row;
    hidden if not show_excluded).  `matches` needs columns ion, harmonic.
    """
    if sims is None or not len(sims):
        return
    ions = ion_db["ion"].to_numpy()
    names = ions[sims["ion_row"].to_numpy()]
    hover = ion_db["hover"].to_numpy()[sims["ion_row"].to_numpy()]
    labels = [f"{lab}<br>Harmonic: {h}<br>f_sim = {f:.1f} Hz" for lab, h, f in
              zip(hover, sims["harmonic"], sims["f_sim"])]
    matched = set() if matches is None else set(zip(matches["ion"], matches["harmonic"]))
    is_m = np.array([(n, h) in matched for n, h in zip(names, sims["harmonic"])])
    is_hi = np.array([highlight_ion is not None and n == highlight_ion for n in names])
    is_ex = (np.zeros(len(names), dtype=bool) if excluded is None
             else np.asarray(excluded, dtype=bool)[sims["ion_row"].to_numpy()])
    is_ex &= ~is_hi
    groups = [
        (~is_m & ~is_hi & ~is_ex, "Simulated (no peak)", dict(color="#bbbbbb", width=1, dash="dash")),
        (is_m & ~is_hi & ~is_ex, "Simulated (identified)", dict(color="#555555", width=1.5, dash="dash")),
        (is_hi, highlight_ion or "", dict(color=ACCENT, width=3, dash="solid")),
    ]
    # the excluded trace is always added (hidden if not show_excluded) so the
    # snippet's toggle can show/hide it in the browser without a redraw
    groups.append((is_ex, EXCLUDED_TRACE, dict(color="#c8c8c8", width=1, dash="dot")))
    for mask, name, style in groups:
        if mask.any():
            X, Y, H = _vlines(sims["f_sim"].to_numpy()[mask], y0, y1,
                              np.array(labels, dtype=object)[mask])
            fig.add_trace(go.Scattergl(x=X, y=Y, mode="lines", name=name, line=style,
                                       hoverinfo="text", hovertext=H, connectgaps=False,
                                       visible=True if (name != EXCLUDED_TRACE or show_excluded)
                                       else False))


def spectrum_figure(ff, pp, peaks, base, sig, s, r, title, xrange=None, highlight_ion=None):
    fig = go.Figure()
    # "hvh" centres each step on its frequency value (vertical edges halfway
    # between bins); "hv" would start each step at its value.
    fig.add_trace(go.Scattergl(x=ff, y=pp, mode="lines", name="Spectrum", hoverinfo="skip",
                               line=dict(shape="hvh", color="black", width=1)))
    if base is not None and sig is not None:
        fig.add_trace(go.Scattergl(x=ff, y=base + s["k_height"] * sig, mode="lines",
                                   name=f"baseline + {s['k_height']:g}σ", hoverinfo="skip",
                                   line=dict(color="#999", width=1, dash="dot")))
    ymin, ymax = float(np.nanmin(pp)), float(np.nanmax(pp))
    pad = 0.05 * (ymax - ymin or 1)

    # simulated ions
    add_sim_lines(fig, r.get("sims"), r.get("ion_db"), r.get("matches"),
                  ymin - pad, ymax + pad, highlight_ion,
                  excluded=r.get("excluded"), show_excluded=r.get("show_excluded", True))

    # reference window + found reference
    pred = r.get("predicted")
    if pred is not None:
        w = s["ref_window_hz"]
        fig.add_vrect(x0=pred - w, x1=pred + w, fillcolor=ACCENT_PALE, opacity=0.7,
                      line_width=0, layer="below")
        if s.get("ref_min_height", 0) > 0:
            fig.add_shape(type="line", x0=pred - w, x1=pred + w, y0=s["ref_min_height"],
                          y1=s["ref_min_height"], line=dict(color=ACCENT, width=1, dash="dot"))
    if r.get("ref_used") is not None:
        fig.add_vline(x=r["ref_used"], line=dict(color=ACCENT, width=2),
                      annotation_text=f"ref {s['ref_name']} ({r['status']})",
                      annotation_font_color=ACCENT)

    # peaks
    if peaks is not None and len(peaks):
        for kind, sym, col in [("max", "triangle-up", ACCENT), ("shoulder", "diamond", "#444")]:
            p = peaks[peaks["kind"] == kind]
            if len(p):
                hov = [f"{kind}<br>f = {f:.2f} Hz<br>P = {pw:.4g}<br>SNR = {sn:.1f}"
                       for f, pw, sn in zip(p["freq"], p["power"], p["snr"])]
                fig.add_trace(go.Scatter(x=p["freq"], y=p["power"], mode="markers",
                                         name=f"{'Maxima' if kind == 'max' else 'Shoulders'} ({len(p)})",
                                         marker=dict(symbol=sym, color=col, size=9,
                                                     line=dict(color="black", width=0.6)),
                                         hoverinfo="text", hovertext=hov))
    fig.update_layout(title=title, xaxis_title="Frequency (Hz)", yaxis_title="Schottky Power (a.u.)",
                      hovermode="x", dragmode="pan", template="plotly_white",
                      font=dict(size=14), margin=dict(l=60, r=20, t=50, b=40),
                      legend=dict(orientation="h", y=-0.15, x=0), uirevision=title)
    if xrange:
        fig.update_xaxes(range=xrange, autorange=False)
        m = (ff >= xrange[0]) & (ff <= xrange[1])
        if m.any():
            lo, hi = float(pp[m].min()), float(pp[m].max())
            fig.update_yaxes(range=[lo - 0.08 * (hi - lo or 1), hi + 0.12 * (hi - lo or 1)])
    return fig
