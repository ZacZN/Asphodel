"""
Touchpad (and mouse-wheel) zoom in the suite's Plotly graphs.

Plotly zooms on every wheel event it gets. A mouse wheel sends a few large
steps; a touchpad sends a stream of small ones, and with that:

* every Plotly zoom step redraws the layout shapes at the new scale. The
  Segrè chart's squares are shapes (~120k characters of path), so a step
  took ~80 ms (15 ms without the shapes) and the touchpad stream lagged;
* 50 ms after the last wheel event Plotly does a full redraw (the Segrè
  chart: a few hundred ms), so every short pause in a touchpad gesture
  froze the page while the fingers kept moving;
* Firefox kept sending scroll events after the fingers left the touchpad
  (kinetic scrolling), so the zoom carried on after letting go.

The page script `js()` sits in front of Plotly's wheel handling (a capture
listener on the window). For a 2-D Plotly plot with scrollZoom on:

* every zoom step is handed to Plotly with the shapes left out of its
  per-step redraw (the axes' _shapeIndices emptied for the duration of the
  call); the shape layers are moved with an SVG transform instead (exact for
  linear axes: a zoom is a scale + shift in pixels), clipped to the plot
  area. Plotly's full redraw at the end draws them properly again and the
  transform is removed. Plots with log / category axes or several subplots
  are zoomed by Plotly as before.
* mouse wheel (line / page steps, or large whole-pixel steps): one Plotly
  step per click, the same zoom per click as before;
* touchpad: the events of one animation frame are added up, scaled by
  BASE x the speed setting, and handed to Plotly as up to MAX_STEPS steps
  (Plotly limits one step to 20 units = 1.105x) at the latest mouse
  position. Plotly's full redraw, which it schedules 50 ms (REDRAW_MS)
  after each step, is held back while the fingers are on the touchpad
  (events less than HOLD_MS apart; the setTimeout call it makes during our
  step is caught) and run once when the gesture ends, or at once when a
  mouse button is pressed. The rest of a
  gesture is dropped once it looks like coasting (steps shrinking at a
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
SPEEDS = (0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0)   # choices on the home page
MIN_SPEED, MAX_SPEED = 0.1, 10.0

BASE = 0.5           # Plotly wheel units per touchpad pixel at speed 1 (Plotly itself: 1)
MAX_STEPS = 4        # Plotly steps per frame at most (one step: up to 20 units)
GAP_MS = 250         # a pause this long ends a touchpad gesture (mouse / touchpad decided anew)
HOLD_MS = 150        # fingers taken as still on the touchpad this long after an event
REDRAW_MS = 50       # Plotly's REDRAWDELAY: its full redraw this long after a wheel step
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
  // Touchpad / wheel zoom for the Plotly plots of this page: core/trackpad.py
  if (window.__asphodelTouchpad) return;
  window.__asphodelTouchpad = true;
  var URL = %(url)s, BASE = %(base)s, MAX_STEPS = %(steps)s, GAP_MS = %(gap)s,
      HOLD_MS = %(hold)s, REDRAW_MS = %(redraw)s, COAST_STEPS = %(coast)s, SPEED = %(speed)s;
  function load() {
    fetch(URL + '?t=' + Date.now(), {cache: 'no-store'})
      .then(function(r) { return r.json(); })
      .then(function(d) { var v = Number(d && d.speed); if (isFinite(v) && v > 0) SPEED = v; })
      .catch(function() {});
  }
  load();
  window.addEventListener('focus', load);
  document.addEventListener('visibilitychange', function() { if (!document.hidden) load(); });

  var NS = 'http://www.w3.org/2000/svg';
  var ours = new WeakSet();        // the wheel events this script hands to Plotly
  var g = {last: -1e9, pad: false, prev: 0, ratio: 0, steady: 0, coasting: false};
  var pend = null, raf = 0;
  var realTimeout = window.setTimeout.bind(window);
  var held = null, endTimer = 0;   // Plotly's full redraw, held back during a touchpad gesture

  // 2-D Plotly plot with scroll zoom under the element, else null
  function plotOf(t) {
    if (!t || !t.closest) return null;
    var gd = t.closest('.js-plotly-plot');
    if (!gd || !t.closest('.draglayer')) return null;
    var sz = gd._context && gd._context.scrollZoom;
    return sz ? gd : null;
  }

  // ---- shapes moved by a transform during the zoom instead of redrawn per step
  // one subplot, linear axes, with shapes: {xa, ya, layers} else null
  function shapeInfo(gd) {
    var fl = gd._fullLayout;
    if (!fl || !fl._plots || Object.keys(fl._plots).length !== 1) return null;
    var xa = fl.xaxis, ya = fl.yaxis;
    if (!xa || !ya || xa.type !== 'linear' || ya.type !== 'linear') return null;
    if (!(xa._shapeIndices || []).length && !(ya._shapeIndices || []).length) return null;
    var layers = [];
    var add = function(sel) { var n = sel && sel.node && sel.node(); if (n && layers.indexOf(n) < 0) layers.push(n); };
    add(fl._shapeUpperLayer); add(fl._shapeLowerLayer);
    for (var k in fl._plots) add(fl._plots[k].shapelayer);
    return layers.length ? {xa: xa, ya: ya, layers: layers} : null;
  }
  function hook(gd) {
    if (!gd.on || gd.__tpEv === gd._ev) return;
    gd.__tpEv = gd._ev;
    gd.on('plotly_afterplot', function() { unmove(gd); });
    gd.on('plotly_relayout', function(ev) {
      for (var k in (ev || {})) if (/^[xy]axis\\d*\\.(range|autorange)/.test(k)) { unmove(gd); return; }
    });
  }
  function unmove(gd) {
    var st = gd.__tpMove;
    if (!st) return;
    gd.__tpMove = null;
    st.layers.forEach(function(n) { n.removeAttribute('transform'); });
    st.parents.forEach(function(n) { n.removeAttribute('clip-path'); });
    if (st.clip && st.clip.parentNode) st.clip.parentNode.removeChild(st.clip);
  }
  function lin(ax) { return [ax.r2l(ax.range[0]), ax.r2l(ax.range[1])]; }
  function place(gd) {
    var st = gd.__tpMove;
    if (!st) return;
    var xa = st.xa, ya = st.ya, nx = lin(xa), ny = lin(ya), r = st.r;
    if (nx[1] === nx[0] || ny[1] === ny[0]) return;
    var sx = (r[1] - r[0]) / (nx[1] - nx[0]), sy = (r[3] - r[2]) / (ny[1] - ny[0]);
    var tx = xa._offset * (1 - sx) + (r[0] - nx[0]) / (nx[1] - nx[0]) * xa._length;
    var ty = (ya._offset + ya._length) * (1 - sy) - (r[2] - ny[0]) / (ny[1] - ny[0]) * ya._length;
    var tf = 'matrix(' + sx + ',0,0,' + sy + ',' + tx + ',' + ty + ')';
    st.layers.forEach(function(n) { n.setAttribute('transform', tf); });
  }
  function startMove(gd, info) {
    var xa = info.xa, ya = info.ya, x = lin(xa), y = lin(ya);
    var st = {xa: xa, ya: ya, layers: info.layers, r: [x[0], x[1], y[0], y[1]], parents: [], clip: null};
    // clip the moved layers to the plot area (their own clip paths move with them)
    var svg = gd.querySelector('svg.main-svg');
    var defs = svg && (svg.querySelector('defs') || svg.insertBefore(document.createElementNS(NS, 'defs'), svg.firstChild));
    if (defs) {
      var id = 'tpclip-' + Math.random().toString(36).slice(2, 9);
      var cp = document.createElementNS(NS, 'clipPath'); cp.setAttribute('id', id);
      var rc = document.createElementNS(NS, 'rect');
      rc.setAttribute('x', xa._offset); rc.setAttribute('y', ya._offset);
      rc.setAttribute('width', xa._length); rc.setAttribute('height', ya._length);
      cp.appendChild(rc); defs.appendChild(cp); st.clip = cp;
      info.layers.forEach(function(n) {
        var p = n.parentNode;
        if (p && st.parents.indexOf(p) < 0) { p.setAttribute('clip-path', 'url(#' + id + ')'); st.parents.push(p); }
      });
    }
    gd.__tpMove = st;
  }

  // Plotly's redraw timer set during fn() is kept in `held` instead of started
  function holding(fn) {
    var st = window.setTimeout;
    window.setTimeout = function(cb, ms) {
      if (ms === REDRAW_MS && typeof cb === 'function' && arguments.length <= 2) { held = cb; return 0; }
      return st.apply(window, arguments);
    };
    try { fn(); } finally { window.setTimeout = st; }
  }
  function release() {
    clearTimeout(endTimer); endTimer = 0;
    var cb = held; held = null;
    if (cb) cb();
  }
  window.addEventListener('mousedown', release, true);

  // one Plotly zoom step (dy in Plotly units), shapes moved instead of redrawn;
  // hold: keep Plotly's full redraw back (touchpad)
  function step(target, x, y, sx, sy, dy, hold) {
    var gd = plotOf(target);
    var info = gd && shapeInfo(gd);
    var ev = new WheelEvent('wheel', {deltaY: dy, deltaX: 0, deltaMode: 0,
      clientX: x, clientY: y, screenX: sx, screenY: sy, bubbles: true, cancelable: true});
    ours.add(ev);
    var send = function() { target.dispatchEvent(ev); };
    var run = hold ? function() { holding(send); } : send;
    if (!info) { run(); return; }
    hook(gd);
    if (gd.__tpMove && gd.__tpMove.xa !== info.xa) unmove(gd);   // replotted meanwhile
    if (!gd.__tpMove) startMove(gd, info);
    var xs = info.xa._shapeIndices, ys = info.ya._shapeIndices;
    info.xa._shapeIndices = []; info.ya._shapeIndices = [];
    try { run(); }
    finally { info.xa._shapeIndices = xs; info.ya._shapeIndices = ys; }
    place(gd);
  }
  function targetAt(p) {
    var t = document.elementFromPoint(p.x, p.y);
    if (plotOf(t)) return t;
    return p.target && p.target.isConnected && plotOf(p.target) ? p.target : null;
  }
  function flush() {
    raf = 0;
    var p = pend; pend = null;
    if (!p || !p.dy) return;
    var t = targetAt(p);
    if (!t) return;
    var d = p.dy * BASE * SPEED;
    for (var i = 0; i < MAX_STEPS && Math.abs(d) > 1e-6; i++) {
      var s = Math.max(-20, Math.min(20, d));
      step(t, p.x, p.y, p.sx, p.sy, s, true);
      d -= s;
    }
  }
  window.addEventListener('wheel', function(e) {
    if (ours.has(e) || e.ctrlKey) return;            // our own step / pinch
    if (!plotOf(e.target)) return;                   // not a plot: page scrolling etc.
    var mode = e.deltaMode, dy = e.deltaY, dx = e.deltaX;   // deltaMode read first (Firefox)
    var now = performance.now();
    if (now - g.last > GAP_MS) {
      // a wheel click: line / page steps, or a large whole-pixel step with no sideways
      // part (fractional line steps come from a touchpad)
      g.pad = mode === 1 ? !Number.isInteger(dy)
            : !(mode === 2 || (dx === 0 && Math.abs(dy) >= 50 && Number.isInteger(dy)));
      g.prev = 0; g.ratio = 0; g.steady = 0; g.coasting = false;
    }
    g.last = now;
    e.preventDefault();
    e.stopImmediatePropagation();
    if (!g.pad) {
      release();
      // mouse wheel: one step per click as before (Plotly limits a step to 20 units)
      var px = mode === 1 ? dy * 16 : mode === 2 ? dy * 400 : dy;
      step(e.target, e.clientX, e.clientY, e.screenX, e.screenY, px);
      return;
    }
    // touchpad
    // Plotly's full redraw once the fingers have been still for HOLD_MS
    clearTimeout(endTimer);
    endTimer = realTimeout(release, HOLD_MS);
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
""" % {"url": json.dumps(url), "base": BASE, "steps": MAX_STEPS, "gap": GAP_MS,
       "hold": HOLD_MS, "redraw": REDRAW_MS, "coast": COAST_STEPS, "speed": json.dumps(speed())}
