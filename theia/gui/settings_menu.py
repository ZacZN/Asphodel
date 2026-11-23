"""
Settings pop-up menu, dark mode, Quit and settings-file warnings (all pages).

Part of Theia (theia/theia.py); split out of theia.py in Nov 2026.
"""

import os
import threading

import dash
import dash_bootstrap_components as dbc
from dash import html, Input, Output, State
from dash.exceptions import PreventUpdate

from core import jsonfile, tabwatch
from core.scan import load_config, save_config

from .app import app


# ---------------------------------------------------------------------------
# Quit
# ---------------------------------------------------------------------------

# The page checks in regularly: its tab closes when Theia stops, and Theia
# stops (a running scan too) a few seconds after its last tab is closed.
_TABS = tabwatch.TabTracker(on_last_closed=lambda: os._exit(0), name="Theia")
tabwatch.install_flask(app.server, _TABS)


@app.server.route("/_theia_quit", methods=["POST"])
def _theia_quit():
    """Stop the program shortly after answering (so the browser gets the
    reply).  The Further output state is already saved after every change."""
    print("Quit pressed in the browser: stopping Theia.")
    threading.Timer(0.5, lambda: os._exit(0)).start()
    return "stopping"


app.clientside_callback(
    """
    function(n) {
        if (!n) return window.dash_clientside.no_update;
        if (!window.confirm('Quit Theia?\\n\\nThis stops Theia ' +
                            '(a scan that is running is stopped too).')) {
            return window.dash_clientside.no_update;
        }
        fetch('_theia_quit', {method: 'POST'}).catch(function() {}).then(function() {
            window.close();
            // browsers only let a script close tabs it opened itself; if this
            // tab is still open, replace the page with a message
            setTimeout(function() {
                document.body.innerHTML =
                    '<div style="font-family:sans-serif;padding:60px;text-align:center;">' +
                    '<h3 style="color:#f57d15">Theia Has Stopped</h3>' +
                    '<p>You can close this tab.</p></div>';
            }, 300);
        });
        return window.dash_clientside.no_update;
    }
    """,
    Output("_quit-hook", "children"),
    Input("btn-quit", "n_clicks"),
    prevent_initial_call=True,
)


# ===========================================================================
# Callbacks — settings popup menu and dark mode (all pages)
# ===========================================================================

app.clientside_callback(
    """
    function(n) {
        var menu = document.getElementById('settings-menu');
        var btn = document.getElementById('btn-settings');
        if (!menu || !btn) return window.dash_clientside.no_update;
        if (!window._theiaMenuListener) {
            // close when clicking anywhere outside the menu and the button
            window._theiaMenuListener = true;
            document.addEventListener('mousedown', function(evt) {
                var m = document.getElementById('settings-menu');
                var b = document.getElementById('btn-settings');
                if (m && m.style.display !== 'none' && !m.contains(evt.target)
                        && !(b && b.contains(evt.target))) {
                    m.style.display = 'none';
                }
            });
            document.addEventListener('keydown', function(evt) {
                var m = document.getElementById('settings-menu');
                if (evt.key === 'Escape' && m) m.style.display = 'none';
            });
        }
        if (n) menu.style.display = (menu.style.display === 'none') ? 'block' : 'none';
        return window.dash_clientside.no_update;
    }
    """,
    Output("_settings-menu-hook", "children"),
    Input("btn-settings", "n_clicks"),
)

app.clientside_callback(
    """
    function(value) {
        document.body.classList.toggle('dark', !!(value && value.length));
        return window.dash_clientside.no_update;
    }
    """,
    Output("_dark-mode-hook", "children"),
    Input("dark-mode", "value"),
)


@app.callback(
    Output("dark-mode", "value"),
    Input("tabs", "value"),
)
def restore_dark_mode(_tab):
    """Show the last saved dark-mode choice when the page (re)loads."""
    return load_config().get("dark_mode", [])


@app.callback(
    Output("_dark-mode-hook", "title"),
    Input("dark-mode", "value"),
    prevent_initial_call=True,
)
def remember_dark_mode(value):
    """Save the choice so it is restored next time Theia runs."""
    save_config({"dark_mode": value or []})
    return dash.no_update


# ===========================================================================
# Settings-file warnings (core/jsonfile.py): damaged files set aside, failed saves
# ===========================================================================

@app.callback(
    Output("settings-warn", "children"),
    Input("settings-warn-poll", "n_intervals"),
    State("settings-warn", "children"),
)
def show_settings_warnings(_n, shown):
    new = jsonfile.take_warnings()
    if not new:
        raise PreventUpdate
    # keep the alerts still open (a dismissed one comes back with is_open False)
    shown = [c for c in (shown or [])
             if isinstance(c, dict) and (c.get("props") or {}).get("is_open", True) is not False]
    for text in new:
        shown.append(dbc.Alert([html.B("Settings file: "), text], color="warning",
                               dismissable=True, style={"fontSize": "13px", "marginBottom": "6px"}))
    return shown[-6:]
