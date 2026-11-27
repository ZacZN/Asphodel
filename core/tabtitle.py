"""
Browser-tab title while a Dash page is busy.

Dash's default swaps the tab title for "Updating..." while a callback runs,
so the tool's name disappears.  install_dash(app) turns that off and instead
keeps the name with a spinning braille character after it
("Theia ⠋", "Theia ⠙", ...), so the name itself never moves, while callbacks are running.

    from core import tabtitle
    tabtitle.install_dash(app)

Called from core/icon.py use_in_dash(), so every Dash tool that sets its tab
icon gets it.

How "busy" is seen: window.fetch is wrapped to count Dash callback requests
(_dash-update-component).  Callbacks fired only by a dcc.Interval (log panel,
file watches) are ignored, as in core/bootscreen.py, so the tab doesn't
flicker every few seconds.  The spinner appears only after DELAY_MS, so
quick callbacks don't flash it.
"""

DELAY_MS = 150      # callbacks shorter than this show no spinner
FRAME_MS = 100      # spinner speed

FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

_JS = r"""
<script>
(function() {
  if (window.__asphTabTitle) return;
  window.__asphTabTitle = true;
  var FRAMES = "__FRAMES__", DELAY = __DELAY__, STEP = __STEP__;
  var SUFFIX = new RegExp(' [' + FRAMES + ']$');
  var busy = 0, timer = null, spin = null, i = 0, base = null;

  function periodic(init) {
    if (!init || typeof init.body !== 'string') return false;
    try {
      var c = JSON.parse(init.body).changedPropIds || [];
      return c.length > 0 && c.every(function(p) { return /\.n_intervals$/.test(p); });
    } catch (e) { return false; }
  }
  function start() {
    timer = null;
    if (busy <= 0 || spin) return;
    base = document.title.replace(SUFFIX, '');
    spin = setInterval(function() {
      i = (i + 1) % FRAMES.length;
      document.title = base + ' ' + FRAMES[i];
    }, STEP);
    document.title = base + ' ' + FRAMES[i];
  }
  function stop() {
    if (timer) { clearTimeout(timer); timer = null; }
    if (spin) { clearInterval(spin); spin = null; document.title = base; }
  }

  var orig = window.fetch;
  window.fetch = function(input, init) {
    var u = String(input && input.url || input);
    var mine = u.indexOf('_dash-update-component') >= 0 && !periodic(init);
    if (mine) {
      busy++;
      if (!timer && !spin) timer = setTimeout(start, DELAY);
    }
    var pr = orig.apply(this, arguments);
    if (mine) {
      var fin = function() { busy--; if (busy <= 0) { busy = 0; stop(); } };
      pr.then(fin, fin);
    }
    return pr;
  };
})();
</script>
"""


def js():
    """The <script> that runs the title spinner."""
    return (_JS.replace("__FRAMES__", FRAMES).replace("__DELAY__", str(int(DELAY_MS)))
            .replace("__STEP__", str(int(FRAME_MS))))


def install_dash(app):
    """Keep the tool's name in the tab title (no "Updating..."), with a spinner while busy."""
    app.config.update_title = None
    # in <head>, so fetch is wrapped before the Dash renderer starts
    if "</head>" in app.index_string and "__asphTabTitle" not in app.index_string:
        app.index_string = app.index_string.replace("</head>", js() + "</head>", 1)
