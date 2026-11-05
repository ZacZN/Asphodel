"""
2nd-order correction profile dropdowns (Scan settings, Viewer, PID settings).

Part of Theia (theia/theia.py); split out of theia.py in Nov 2026.
"""

import dash
from dash import dcc, html, Input, Output

from core import corrections, physics

from .common import HELP_STYLE, INPUT_STYLE, LABEL_STYLE
from .app import app


# ---------------------------------------------------------------------------
# 2nd-order correction profiles (core/corrections.py): a dropdown next to the
# a, b, c boxes of the Scan settings ("s-"), the Viewer ("v-") and the PID
# settings popup ("pid-").  Picking a profile fills a, b, c; typing a, b, c
# selects the matching profile (or "custom").
# ---------------------------------------------------------------------------

def corr_profile_state(name, a, b, c, ref_name=None, ref_h=None, gamma_t=None, ref_f=None,
                       harmonics=None):
    """(dropdown value, options, info children, boxes locked, profile) for the
    stored profile name and these a, b, c ('' = the profile matching a, b, c)."""
    profs = corrections.load_profiles()
    r = corrections.resolve(name, a, b, c, profiles=profs)
    p = r["profile"]
    if p is None:
        info = ("The stored profile no longer exists. " if r["missing"] else "") + \
            "Values typed in (no profile), used for every harmonic."
    else:
        try:
            hs = physics.canonical("harmonics", harmonics) if harmonics not in (None, "") else None
        except (TypeError, ValueError):
            hs = None
        warn = corrections.warnings(p, ref_name, ref_h, gamma_t, ref_f, p["a"], p["b"], p["c"],
                                    harmonics=hs)
        info = ([html.Div(corrections.describe(p))]
                + [html.Div(t) for t in corrections.coefficient_lines(p)]
                + [html.Div("⚠ " + w, style={"color": "#b45309"}) for w in warn])
    return r["name"], corrections.options(profs), info, p is not None, p


def corr_profile_ui(prefix, name=None, a=None, b=None, c=None, **settings):
    """Profile dropdown + info line for the a, b, c boxes with id prefix."""
    value, opts, info, _, _ = corr_profile_state(name, a, b, c, **settings)
    return html.Div([
        html.Label("Correction profile", style=LABEL_STYLE),
        dcc.Dropdown(id=f"{prefix}corr_profile", options=opts, value=value, clearable=False,
                     style={"fontSize": "13px"}),
        html.Small(info, id=f"{prefix}corr_info", style=HELP_STYLE),
    ], style=INPUT_STYLE)


def register_corr_profile(prefix, as_text, has_ref_freq):
    """Callback linking the profile dropdown and the a, b, c boxes of prefix.
    The profile *name* is the setting; picking a profile fills a, b, c with
    its general values and locks them; 'custom' unlocks them.
    as_text: the boxes are text inputs (else number inputs)."""
    ref_inputs = [Input(f"{prefix}ref_name", "value"), Input(f"{prefix}ref_harmonic", "value"),
                  Input(f"{prefix}gamma_t", "value"), Input(f"{prefix}harmonics", "value")]
    if has_ref_freq:
        ref_inputs.append(Input(f"{prefix}ref_frequency", "value"))

    @app.callback(
        Output(f"{prefix}corr_a", "value", allow_duplicate=True),
        Output(f"{prefix}corr_b", "value", allow_duplicate=True),
        Output(f"{prefix}corr_c", "value", allow_duplicate=True),
        Output(f"{prefix}corr_a", "disabled"),
        Output(f"{prefix}corr_b", "disabled"),
        Output(f"{prefix}corr_c", "disabled"),
        Output(f"{prefix}corr_profile", "options"),
        Output(f"{prefix}corr_info", "children"),
        Input(f"{prefix}corr_profile", "value"),
        Input(f"{prefix}corr_a", "value"),
        Input(f"{prefix}corr_b", "value"),
        Input(f"{prefix}corr_c", "value"),
        *ref_inputs,
        prevent_initial_call=True,
    )
    def _corr_profile(name, a, b, c, ref_name, ref_h, gamma_t, harmonics, ref_f=None):
        nu = dash.no_update
        if not name:                      # not filled in yet (PID popup closed)
            return nu, nu, nu, nu, nu, nu, nu, nu
        value, opts, info, locked, p = corr_profile_state(name, a, b, c, ref_name, ref_h,
                                                          gamma_t, ref_f, harmonics)
        out = [nu, nu, nu]
        if p is not None and not all(corrections._same(x, y) for x, y in
                                     zip((a, b, c), (p["a"], p["b"], p["c"]))):
            out = ([physics._fmt_num(v) for v in (p["a"], p["b"], p["c"])] if as_text
                   else [float(v) for v in (p["a"], p["b"], p["c"])])
        return (*out, locked, locked, locked, opts, info)
