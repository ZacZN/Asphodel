"""
Loading screen for the suite's Dash pages (Theia; others can use it too).

A full-page "Starting <tool>…" screen, the same as the home page's starting
page (home.py _STARTING_HTML) and the spectrogram's (BOOT_OVERLAY in
spectrogram/plot_spectrogram.py), that covers the page from its first bytes
until it is ready: Dash has built the page, every callback the page load
started has answered (chains included) and every visible plot is drawn.
?asph_t0=<ms> in the URL (set by the home page's starting page) continues its
timer, so the two screens look like one.

With tabs=<id of a dcc.Tabs>, the screen also covers the first opening of each
tab ("Loading <tab label>…"), whose callbacks can take a few seconds (e.g.
Theia's Scan Results and Further Output, which load only when first opened).

    from core import bootscreen
    bootscreen.install_dash(app, "Theia", tabs="tabs")

How "ready" is seen: window.fetch is wrapped to count the page's own requests
(Dash's _dash-layout, _dash-dependencies and callback requests, the page's own
routes); periodic ones are ignored (callbacks fired only by a dcc.Interval,
and /_alive, /_aslog, /_touchpad).  Ready = no request in flight for QUIET_MS
and every visible dcc.Graph drawn.  The screen never stays longer than
MAX_S seconds.
"""

QUIET_MS = 300
MAX_S = 60

_HTML = r"""
<div id="asphBoot" style="position:fixed;inset:0;z-index:2147483647;background:#fff;color:#222;
  display:flex;align-items:center;justify-content:center;transition:opacity .15s;
  font-family:-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif">
 <style>
  #asphBoot .spin{width:34px;height:34px;margin:0 auto 18px;border:4px solid #fff4ec;
    border-top-color:#f57d15;border-radius:50%;animation:asphSpin .8s linear infinite}
  @keyframes asphSpin{to{transform:rotate(360deg)}}
 </style>
 <div style="text-align:center">
  <div class="spin"></div>
  <div id="asphBootH" style="font-size:20px;font-weight:600;margin:0 0 14px">Loading __NAME__&hellip;</div>
  <div id="asphBootT" style="color:#777;font-size:13px">0.0 s</div>
 </div>
</div>
<script>
(function() {
  var NAME = "__NAME__", TABS = "__TABS__", QUIET = __QUIET__, MAX = __MAX__ * 1000;
  var o = document.getElementById('asphBoot');
  var m = /[?&]asph_t0=(\d+)/.exec(location.search);
  var t0 = m ? +m[1] : (performance.timeOrigin || Date.now());
  if (m) document.getElementById('asphBootH').innerHTML = 'Starting ' + NAME + '&hellip;';
  if (m && history.replaceState) {   // a reload starts its own timer
    history.replaceState(null, '', location.pathname + location.search.replace(/([?&])asph_t0=\d+&?/, '$1').replace(/[?&]$/, '') + location.hash);
  }

  // ---- the page's own requests ------------------------------------------
  var busy = 0, lastAct = Date.now();
  var POLL = /\/_alive|\/_aslog|\/_touchpad/;
  function periodic(u, init) {
    if (POLL.test(u)) return true;
    if (u.indexOf('_dash-update-component') < 0 || !init || typeof init.body !== 'string') return false;
    try {
      var c = JSON.parse(init.body).changedPropIds || [];
      return c.length > 0 && c.every(function(p) { return /\.n_intervals$/.test(p); });
    } catch (e) { return false; }
  }
  var orig = window.fetch;
  window.fetch = function(input, init) {
    var u = String(input && input.url || input);
    var mine = !periodic(u, init);
    if (mine) { busy++; lastAct = Date.now(); }
    var fin = function() { if (mine) { busy--; lastAct = Date.now(); } };
    var pr = orig.apply(this, arguments);
    pr.then(fin, fin);
    return pr;
  };

  // ---- show / hide --------------------------------------------------------
  var shownAt = 0, showing = true, tick = null, loaded = false;
  function startTimer(from) {
    clearInterval(tick);
    tick = setInterval(function() {
      document.getElementById('asphBootT').textContent = ((Date.now() - from) / 1000).toFixed(1) + ' s';
    }, 100);
  }
  function graphsDrawn() {
    var ok = true;
    document.querySelectorAll('.dash-graph').forEach(function(g) {
      if (g.offsetParent === null || g.offsetHeight === 0) return;     // hidden
      var p = g.querySelector('.js-plotly-plot');
      if (!p || !p._fullLayout) ok = false;
    });
    return ok;
  }
  function ready() {
    return loaded && busy === 0 && Date.now() - lastAct > QUIET
      && Date.now() - shownAt > QUIET && graphsDrawn()
      && document.querySelector('#react-entry-point') && !document.querySelector('._dash-loading');
  }
  function hide() {
    showing = false; clearInterval(tick);
    requestAnimationFrame(function() { requestAnimationFrame(function() {
      if (showing) return;
      o.style.opacity = '0';
      setTimeout(function() { if (!showing) o.style.display = 'none'; }, 160);
    }); });
  }
  function show(title) {
    document.getElementById('asphBootH').innerHTML = title;
    document.getElementById('asphBootT').textContent = '0.0 s';
    o.style.display = 'flex'; o.style.opacity = '1';
    showing = true; shownAt = Date.now(); lastAct = Date.now();
    startTimer(shownAt);
    check();
  }
  function check() {
    if (!showing) return;
    if (ready() || Date.now() - shownAt > MAX) return hide();
    setTimeout(check, 50);
  }
  window.addEventListener('load', function() { loaded = true; });
  shownAt = Date.now();
  startTimer(t0);
  check();

  // ---- first opening of each tab -----------------------------------------
  if (TABS) {
    var seen = {};
    document.addEventListener('click', function(ev) {
      var tab = ev.target.closest && ev.target.closest('#' + TABS + ' .tab');
      if (!tab || showing) return;
      var label = (tab.textContent || '').trim();
      if (!label || seen[label] || tab.classList.contains('tab--selected')) return;
      seen[label] = true;
      show('Loading ' + label.replace(/&/g, '&amp;').replace(/</g, '&lt;') + '&hellip;');
    }, true);
    // the tab shown at page load counts as opened
    (function markFirst() {
      var sel = document.querySelector('#' + TABS + ' .tab--selected');
      if (sel) seen[(sel.textContent || '').trim()] = true;
      else setTimeout(markFirst, 200);
    })();
  }
})();
</script>
"""


def html(name, tabs=""):
    """The loading screen's HTML (overlay + script) for a page called `name`."""
    esc = name.replace("&", "&amp;").replace("<", "&lt;").replace('"', "&quot;")
    return (_HTML.replace("__NAME__", esc).replace("__TABS__", tabs or "")
            .replace("__QUIET__", str(int(QUIET_MS))).replace("__MAX__", str(int(MAX_S))))


def install_dash(app, name, tabs=""):
    """Put the loading screen at the top of a Dash app's page body."""
    if "{%app_entry%}" not in app.index_string:
        return
    app.index_string = app.index_string.replace(
        "{%app_entry%}", html(name, tabs) + "\n{%app_entry%}", 1)
