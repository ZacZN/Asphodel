"""Browser-tab icons (favicons) for the Asphodel tools.

Icons live in core/icons/<name>.png. A tool asks for its own name
("theia", "spectrogram", ...) and falls back to asphodel.png until it has
an image of its own, so adding e.g. core/icons/theia.png is all it takes
to give Theia a different icon.

Every tool serves its icon at ICON_URL:
  * Dash apps:   icon.use_in_dash(app, "theia")
  * other pages: put icon.LINK_TAG in <head> and answer ICON_URL with
                 icon.png_bytes("spectrogram")
"""
import os

ICON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons")
DEFAULT = "asphodel"
ICON_URL = "/_asphodel_icon.png"
LINK_TAG = f'<link rel="icon" type="image/png" href="{ICON_URL}">'


def icon_path(name=DEFAULT):
    """core/icons/<name>.png, or the suite icon if that tool has none yet."""
    p = os.path.join(ICON_DIR, f"{name}.png")
    return p if os.path.isfile(p) else os.path.join(ICON_DIR, f"{DEFAULT}.png")


def png_bytes(name=DEFAULT):
    try:
        with open(icon_path(name), "rb") as fh:
            return fh.read()
    except OSError:
        return b""


def use_in_dash(app, name=DEFAULT):
    """Serve the icon from the Dash app and point the page's tab icon at it."""
    import flask

    @app.server.route(ICON_URL)
    def _asphodel_icon():
        data = png_bytes(name)
        if not data:
            flask.abort(404)
        resp = flask.Response(data, mimetype="image/png")
        resp.headers["Cache-Control"] = "no-cache"
        return resp

    # {%favicon%} is Dash's placeholder for assets/favicon.ico
    if "{%favicon%}" in app.index_string:
        app.index_string = app.index_string.replace("{%favicon%}", LINK_TAG)
    else:
        app.index_string = app.index_string.replace("</head>", LINK_TAG + "</head>", 1)
