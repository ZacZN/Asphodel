"""
The ⓘ buttons and their help texts.

Part of Theia (theia/theia.py); split out of theia.py in Nov 2026.
"""

from dash import html

from .common import BOX_BORDER


# ---- ⓘ info buttons: the help text of a section, shown in a popup ----------
INFO_TEXTS = {
    "sum": ("Sum spectra", [
        "Each spectrum is added once, however many identifications or files it appears in.",
        "Aligned spectra are shifted by whole bins (never interpolated) so the chosen peak sits "
        "where it is in the first spectrum; what is left over (at most half a bin) is shown.",
        "Strongest peak: the first spectrum's strongest bin, then in each next spectrum the "
        "strongest bin within ± the window (Hz) of the previous spectrum's peak, so another ion "
        "becoming the strongest elsewhere does not move the alignment.",
        "The plot opens zoomed to ±5 kHz around the ion/harmonic chosen in the dropdown.",
        "With an identified peak, spectra without that ion/harmonic are left out.",
        "Bins not covered by every shifted spectrum are dropped.",
        "Simulated ion lines: the strongest peak of the sum is taken as the reference ion (at its "
        "reference harmonic), with the scan settings stored in the first loaded file.  Ions in the "
        "loaded files are dark dashed, others light dashed, half-life-excluded dotted.",
        "Data source 'Single spectrum file': any .npz spectrum (e.g. a Tartarus sum in "
        "summed_1d, or the file open in the Viewer via 'Use Viewer spectrum') is shown as it is, "
        "for fitting and the energy calculation.  Its simulated lines use the Viewer settings "
        "(no half-life cut), with the strongest peak as the reference.",
        "Search ions…: type part of an ion name to list its simulated lines on the plotted "
        "spectrum; click one to zoom to f_sim ± the width (kHz, default ±10).  h−1 / h+1 then "
        "zoom to the same ion at the neighbouring harmonic.",
    ]),
    "fit": ("Peak fitting", [
        "Switch it on and click a peak to add it (the start values snap to the nearest maximum).",
        "Edit any value in the table to adjust the curve by hand, or press Fit.  Delete a row to "
        "remove a peak.",
        "The fit region is set by the two orange lines (set from the view when the first peak is "
        "added).  Drag a line, type the limits, or press 'Region = view'.",
        "Drag a peak's × marker to move its centre and amplitude together (cursor ✥); drag below "
        "the marker to move the centre only (cursor ↔); drag a ◆ handle at half height to change "
        "the FWHM.",
        "Each peak has its own line shape (shape column); 'Shape for new peaks' sets the shape of "
        "peaks added from then on.",
        "χ²/ndf uses a noise σ per bin estimated from the summed spectrum; uncertainties are "
        "scaled by √(χ²/ndf).",
        "'Send to isocurve' sends every fitted peak to the Isochronicity Curve tab.  Each peak "
        "needs an ion name with its charge (e.g. 207Tl81+, m207Tl81+) and a harmonic: type them "
        "in the ion / h columns, or you are asked for the missing ones (with nearby simulated "
        "ions as suggestions).  Press Fit first.",
    ]),
    "energy": ("Energy calculation", [
        "Step 1 – calibration: click two peaks, give their NUBASE names (or click a suggested "
        "nearby simulated ion) and press Calibrate.",
        "A click on a peak of the main fit (within half its FWHM) uses its fitted centre; anywhere "
        "else the peak is fitted on its own in a small window (shape for new peaks + constant "
        "background).",
        "Calibration: k = (ME₂ − ME₁) / (f₂ − f₁) from NUBASE (for two states of the same nuclide "
        "the excitation energies are used).",
        "Step 2: click peaks in pairs A, B:  ΔE = E(B) − E(A) = k·(f_B − f_A).",
        "A linear keV-per-Hz calibration is only accurate for pairs close in frequency to the "
        "calibration peaks (ideally the same A/q).",
    ]),
    "iso": ("Isochronicity curve", [
        "Points come from 'Send to isocurve' in the peak-fitting box of Aligning and fitting: "
        "one row per ion and harmonic, with the fitted centre, FWHM, amplitude, area (± 1σ) and "
        "the revolution frequency f_rev = centre / h.",
        "They are saved in outputs/isocurve/isocurve_points.csv of the active experiment, so "
        "they are kept between sessions.",
        "Sending the same ion and harmonic again replaces its point (you are asked first).",
        "Delete a row (×) to remove the point from the file.",
        "'fixed' lists parameters held fixed in the fit (their uncertainty is 0).",
    ]),
    "isofit": ("Isochronicity curve fit", [
        "σ_T(T) = √( ((1/γ² − 1/γ_t²) · σp/p · T)² + σ_sys² ),  with 1/γ² = 1 − (C / (c·T))², "
        "C the ring circumference.  Fitted parameters: γ_t, σp/p, σ_sys, within the min – max "
        "bounds.",
        "Weighted least squares with err σ(T) as absolute errors (as absolute_sigma=True); "
        "χ²/ndf = Σ((σ(T) − fit) / err σ(T))² / (N − 3).  Several γ_t start values are tried "
        "and the best fit kept.",
        "The fit is redone whenever the points, the 'in fit' ticks or the settings change.",
        "Leave a point out: click its ☑ in the 'in fit' column, or click the point on the plot "
        "(click again to put it back).  Points with err σ(T) = 0 cannot be fitted ('–').",
        "Systematic floor: auto (χ²/ndf = 1): a constant f is added in quadrature to every "
        "err σ(T), √(err² + f²), with f chosen so that the fit has χ²/ndf = 1 (f = 0 if it is "
        "already ≤ 1).  f covers what the peak-fit errors miss (alignment, background, line "
        "shape, close neighbours) and is shown with the result; treat it as an upper limit.",
        "Error bars show the combined error used in the fit: statistical ⊕ bin width ⊕ floor "
        "(whichever are ticked); the table lists the statistical and total errors.",
        "Bin-width error: bin = recording frequency / number of frequency bins (type e.g. "
        "10e6 and 2^17 → 76.294 Hz).  The true position is spread evenly across a bin, so the "
        "1σ error is bin/√12 (22.02 Hz for 76.294 Hz bins; ±½ bin is the maximum, not 1σ).  It "
        "is taken as an extra FWHM error and converted like err σ(T): (bin/√12 / centre) · T · "
        "1000 / 2√(2 ln 2), added in quadrature.  With "
        "the auto floor also ticked, the floor is fitted on top of it (what is left over).",
        "Settings are saved per experiment (settings/isocurve.json).",
        "'Save figure' draws the plot and residuals with matplotlib and saves PNG + PDF in "
        "outputs/figures/.",
    ]),
    "params": ("Fit parameters", [
        "centre, amplitude (height above the background), FWHM and – for pseudo-Voigt – η "
        "(Lorentzian fraction) can be edited; the curve is redrawn straight away.",
        "± columns are the 1σ uncertainties from the last Fit (cleared when you change anything "
        "by hand).",
        "Click a 'fix' cell (☐ / ☑) to hold that parameter at its value during the Fit; fixed "
        "parameters show 'fixed' instead of an uncertainty.  b0 / b1 have their own fix boxes.",
        "Tick 'limit columns' to set a min / max for each peak's centre and FWHM (blank = no "
        "extra limit).  Centre limits are drawn as a grey band.",
        "Background = b0 + b1·(f − f_mid), where f_mid is the centre of the fit region.",
    ]),
}


def info_button(key, style=None):
    return html.Button("ⓘ", id={"type": "info-btn", "key": key}, n_clicks=0,
                       className="btn btn-outline-secondary btn-sm", title="Information",
                       style={"padding": "0 7px", "borderRadius": "50%", "fontWeight": "700",
                              "lineHeight": "1.4", **(style or {})})


info_modal = html.Div([
    html.Div([
        html.Div([
            html.Span(id="info-title", style={"fontWeight": "700", "fontSize": "15px"}),
            html.Button("✕", id="info-close", className="btn btn-outline-secondary btn-sm"),
        ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center",
                  "marginBottom": "8px"}),
        html.Div(id="info-body", style={"fontSize": "13px"}),
    ], className="theia-box",
       style={"border": BOX_BORDER, "borderRadius": "6px", "background": "white",
              "width": "620px", "maxWidth": "92vw", "padding": "14px 16px",
              "boxShadow": "0 8px 30px rgba(0,0,0,0.3)"}),
], id="info-modal", n_clicks=0,
   style={"display": "none", "position": "fixed", "inset": "0", "zIndex": "1000",
          "background": "rgba(0,0,0,0.35)", "alignItems": "center", "justifyContent": "center"})
