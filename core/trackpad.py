"""
Touchpad zoom in the suite's Plotly graphs.

Plotly zooms on every wheel event it gets. A mouse wheel sends a few large
steps, which works well; a touchpad sends a stream of small ones, and three
things went wrong with that:

* too sensitive: every touchpad event counted as much as Plotly's per-step
  zoom, so a short finger movement zoomed a long way;
* slow / lagging: while a plot was redrawing (the Segrè chart takes a few
  hundred ms) the touchpad events queued up, and were all zoomed afterwards;
* coasting: Firefox keeps sending scroll events after the fingers leave the
  touchpad (kinetic scrolling), so the zoom carried on after letting go.

The page script `js()` sits in front of Plotly's wheel handling (a capture
listener on the window). For a 2-D Plotly plot with scrollZoom on:

* mouse wheel (line / page steps, or large whole-pixel steps): passed to
  Plotly as before, the same zoom per click;
* touchpad: the events of one animation frame are added up, scaled by
  BASE x the speed setting, and handed to Plotly as one wheel step at the
  latest mouse position. Events that were already older than STALE_MS when
  the page got to them (they waited for a redraw) are dropped, and so is the
  rest of a gesture once it looks like coasting (the steps shrinking at a
  steady rate frame after frame). Sideways swipes do nothing; ctrl + wheel
  (pinch) is left to the browser.

Other wheel events (page scrolling, 3-D plots, plots without scrollZoom) are
not touched. Kinetic scrolling itself is switched off in the suite's own
Firefox profile (core/browser.py, apz.gtk.kinetic_scroll.enabled).

The speed is a suite setting (data/asphodel_settings.json, key SETTING_KEY,
1 = default), set on the home page. Pages fetch it from URL when they load
and whenever their tab gets the focus again, so a change applies when you
switch back to a tool.

Use in a tool:

    trackpad.install_dash(app)                         # Dash apps
    ... page js += trackpad.js(); GET trackpad.URL -> trackpad.response()  # http.server
"""

import json

from . import jsonfile, paths

URL = "/_touchpad"
SETTING_KEY = "touchpad_zoom_speed"
DEFAULT_SPEED = 1.0
SPEEDS = (0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0)   # choices on the home page
MIN_SPEED, MAX_SPEED = 0.1, 10.0

BASE = 0.35          # Plotly wheel units per touchpad pixel at speed 1 (Plotly itself: 1)
STALE_MS = 80        # touchpad events older than this when handled are dropped
GAP_MS = 250         # a pause this long ends a touchpad gesture
COAST_STEPS = 4      # this many steadily shrinking steps in a row = coasting


def speed():
    """The touchpad zoom speed (1 = default)."""
    try:
        v = float(jsonfile.read(paths.SUITE_SETTINGS, {}).get(SETTING_KEY, DEFAULT_SPEED))
    except (TypeError, ValueError, AttributeError):
        return DEFAULT_SPEED
    if not (v == v) or v <= 0:          # NaN / nonsense
        return DEFAULT_SPEED
    return min(max(v, MIN_SPEED), MAX_SPEED)


def set_speed(v):
    v = min(max(float(v), MIN_SPEED), MAX_SPEED)
    jsonfile.update(paths.SUITE_SETTINGS, {SETTING_KEY: v})
    return v


def response():
    """Body of the GET URL reply (JSON)."""
    return {"speed": speed()}


def install_flask(server):
    """The speed route on a Flask server (also Dash's app.server)."""
    from flask import jsonify

    def _speed():
        r = jsonify(response())
        r.headers["Cache-Control"] = "no-store"
        return r

    server.add_url_rule(URL, "touchpad_speed", _speed, methods=["GET"])


def install_dash(app):
    """Route + page script for a Dash app."""
    install_flask(app.server)
    app.index_string = app.index_string.replace(
        "{%scripts%}", "{%scripts%}\n<script>" + js() + "</script>")


def js(url=URL):
    """Page script (no <script> tags)."""
    return """
(function() {
  // Touchpad zoom for the Plotly plots of this page: core/trackpad.py
  if (window.__asphodelTouchpad) return;
  window.__asphodelTouchpad = true;
  var URL = %(url)s, BASE = %(base)s, STALE_MS = %(stale)s, GAP_MS = %(gap)s,
      COAST_STEPS = %(coast)s, SPEED = %(speed)s;
  function load() {
    fetch(URL + '?t=' + Date.now(), {cache: 'no-store'})
      .then(function(r) { return r.json(); })
      .then(function(d) { var v = Number(d && d.speed); if (isFinite(v) && v > 0) SPEED = v; })
      .catch(function() {});
  }
  load();
  window.addEventListener('focus', load);
  document.addEventListener('visibilitychange', function() { if (!document.hidden) load(); });

  var ours = new WeakSet();        // the wheel events this script hands to Plotly
  var g = {last: -1e9, pad: false, prev: 0, ratio: 0, steady: 0, coasting: false};
  var pend = null, raf = 0;

  // 2-D Plotly plot with scroll zoom under the event, else null
  function plotOf(t) {
    if (!t || !t.closest) return null;
    var gd = t.closest('.js-plotly-plot');
    if (!gd || !t.closest('.draglayer')) return null;
    var sz = gd._context && gd._context.scrollZoom;
    return sz ? gd : null;
  }
  function send(target, x, y, sx, sy, dy) {
    var ev = new WheelEvent('wheel', {deltaY: dy, deltaX: 0, deltaMode: 0,
      clientX: x, clientY: y, screenX: sx, screenY: sy, bubbles: true, cancelable: true});
    ours.add(ev);
    target.dispatchEvent(ev);
  }
  function flush() {
    raf = 0;
    var p = pend; pend = null;
    if (!p || !p.dy) return;
    // the plot may have rebuilt its drag layer since: target the element now under the mouse
    var t = document.elementFromPoint(p.x, p.y);
    if (!plotOf(t)) t = p.target.isConnected ? p.target : null;
    if (!t) return;
    send(t, p.x, p.y, p.sx, p.sy, p.dy * BASE * SPEED);
  }

  window.addEventListener('wheel', function(e) {
    if (ours.has(e) || e.ctrlKey) return;            // our own step / pinch
    if (!plotOf(e.target)) return;                   // not a plot: page scrolling etc.
    var mode = e.deltaMode, dy = e.deltaY, dx = e.deltaX;   // deltaMode read first (Firefox)
    var now = performance.now();
    var newGesture = now - g.last > GAP_MS;
    if (newGesture) {
      // a wheel click: line / page steps, or a large whole-pixel step with no sideways part
      // (fractional line steps come from a touchpad)
      g.pad = mode === 1 ? !Number.isInteger(dy)
            : !(mode === 2 || (dx === 0 && Math.abs(dy) >= 50 && Number.isInteger(dy)));
      g.prev = 0; g.ratio = 0; g.steady = 0; g.coasting = false;
    }
    g.last = now;
    e.preventDefault();
    e.stopImmediatePropagation();
    if (!g.pad) {
      // mouse wheel: as before (Plotly limits a step to 20 units anyway)
      var px = mode === 1 ? dy * 16 : mode === 2 ? dy * 400 : dy;
      send(e.target, e.clientX, e.clientY, e.screenX, e.screenY, px);
      return;
    }
    // touchpad
    if (now - e.timeStamp > STALE_MS) return;        // waited for a redraw: drop
    if (Math.abs(dx) > Math.abs(dy)) return;         // sideways swipe
    var a = Math.abs(dy);
    if (g.coasting) {
      if (a > g.prev * 1.05) g.coasting = false;     // fingers back on the touchpad
      else { g.prev = a; return; }
    }
    if (g.prev > 0 && a > 0 && a <= g.prev) {
      var r = a / g.prev;
      g.steady = (g.ratio > 0 && r > 0.5 && r < 0.995 && Math.abs(r - g.ratio) < 0.04) ? g.steady + 1 : 0;
      g.ratio = r;
      if (g.steady >= COAST_STEPS) { g.coasting = true; g.prev = a; pend = null; return; }
    } else { g.steady = 0; g.ratio = 0; }
    g.prev = a;
    if (!pend) pend = {dy: 0};
    pend.dy += dy; pend.target = e.target;
    pend.x = e.clientX; pend.y = e.clientY; pend.sx = e.screenX; pend.sy = e.screenY;
    if (!raf) raf = requestAnimationFrame(flush);
  }, {capture: true, passive: false});
})();
""" % {"url": json.dumps(url), "base": BASE, "stale": STALE_MS, "gap": GAP_MS,
       "coast": COAST_STEPS, "speed": json.dumps(speed())}
