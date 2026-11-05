"""
Web resources served locally, so the tools work without internet access
(e.g. on a beamtime computer):

  plotly.js      the copy inside the installed plotly Python package
                 (same version plotly.py itself uses);
  Bootstrap CSS  core/static/bootstrap.min.css (Bootstrap 5.3.6, MIT licence,
                 core/static/bootstrap-LICENSE.txt), the version Theia's
                 dash-bootstrap-components asked for from cdn.jsdelivr.net.

Dash apps:   offline.serve_static(app.server) and
             dash.Dash(..., external_stylesheets=[offline.BOOTSTRAP_CSS_URL])
Own server:  answer PLOTLY_JS_URL with plotly_js_bytes() (spectrogram).
"""

import os
from functools import lru_cache

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
STATIC_PREFIX = "/_asphodel_static/"
BOOTSTRAP_CSS_URL = STATIC_PREFIX + "bootstrap.min.css"
PLOTLY_JS_URL = STATIC_PREFIX + "plotly.min.js"


def plotly_js_path():
    import plotly
    return os.path.join(os.path.dirname(plotly.__file__), "package_data", "plotly.min.js")


@lru_cache(maxsize=1)
def plotly_js_bytes():
    """plotly.min.js from the installed plotly package."""
    with open(plotly_js_path(), "rb") as fh:
        return fh.read()


def serve_static(server):
    """Answer STATIC_PREFIX<file> on a Flask server (a Dash app's app.server):
    the files in core/static/ and plotly.min.js."""
    from flask import Response, abort, send_from_directory

    @server.route(STATIC_PREFIX + "<path:name>")
    def _asphodel_static(name):
        if name == "plotly.min.js":
            return Response(plotly_js_bytes(), mimetype="text/javascript",
                            headers={"Cache-Control": "max-age=86400"})
        if name.endswith(".map"):
            abort(404)                      # source maps are not shipped
        return send_from_directory(STATIC_DIR, name, max_age=86400)
