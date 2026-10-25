"""
Browser tabs and their programs stop together.

Each page of a browser-based tool (home page, Theia, Spectrogram) asks its own
server every couple of seconds whether it is still running (GET ALIVE_PATH,
with an id for the tab).  That works both ways:

* The program stops (Quit / Stop on the home page, its own Quit button, a
  crash): two checks in a row cannot reach the server, so the page closes its
  tab.  Browsers only let a page close a tab that a script opened or one with
  a single history entry (true for a tab the tool opened itself); otherwise
  the page is replaced by a short "has stopped" message.

* The tab is closed: the page says goodbye (POST ALIVE_PATH/bye, sent with
  navigator.sendBeacon when the page goes away).  When no tab of the program
  is left for GRACE_S seconds, the program stops.  The wait means a page
  reload (goodbye, then a new tab id a moment later) does not stop it.  A tab
  that vanishes without a goodbye (browser killed) is dropped after STALE_S
  seconds without a check (background tabs may be throttled to about one
  check a minute, so this is long).  A program whose page never opened is
  never stopped this way.

Use in a tool:

    tracker = tabwatch.TabTracker(on_last_closed=lambda: os._exit(0))
    tabwatch.install_flask(app.server, tracker)        # Dash / Flask apps
    ... "<script>" + tabwatch.watch_js("Theia") + "</script>" in the page

or, for an http.server handler, call tracker.seen(id) on GET ALIVE_PATH and
tracker.bye(id) on POST ALIVE_PATH + "/bye" (the id is the ?id= query value).
"""

import json
import threading
import time

ALIVE_PATH = "/_alive"
BYE_PATH = ALIVE_PATH + "/bye"
GRACE_S = 5.0           # no tab left for this long -> the program stops
STALE_S = 180.0         # a tab not heard from for this long is taken as gone


class TabTracker:
    """Keeps the ids of the open tabs; calls on_last_closed() once, when the
    last tab has been gone for `grace` seconds.  Uses time.monotonic(), which
    does not run while the computer sleeps, so waking up does not count as
    the tabs having gone quiet."""

    def __init__(self, on_last_closed, grace=GRACE_S, stale=STALE_S, name=""):
        self._cb, self._grace, self._stale, self._name = on_last_closed, grace, stale, name
        self._tabs = {}                  # id -> last check (monotonic)
        self._seen_any = False
        self._empty_since = None
        self._fired = False
        self._lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True, name="tabwatch").start()

    def seen(self, tab_id):
        with self._lock:
            self._tabs[str(tab_id or "?")] = time.monotonic()
            self._seen_any = True
            self._empty_since = None

    def bye(self, tab_id):
        with self._lock:
            self._tabs.pop(str(tab_id or "?"), None)

    def open_tabs(self):
        with self._lock:
            return len(self._tabs)

    def _loop(self):
        while not self._fired:
            time.sleep(0.5)
            now = time.monotonic()
            with self._lock:
                for k, t in list(self._tabs.items()):
                    if now - t > self._stale:
                        self._tabs.pop(k, None)
                if not self._seen_any or self._tabs:
                    self._empty_since = None
                    continue
                if self._empty_since is None:
                    self._empty_since = now
                    continue
                if now - self._empty_since < self._grace:
                    continue
                self._fired = True
            if self._name:
                print(f"{self._name}: its browser tab was closed, stopping.", flush=True)
            try:
                self._cb()
            except Exception as exc:          # never let the watcher die silently
                print(f"tabwatch: stopping failed: {exc}", flush=True)


def install_flask(server, tracker):
    """Routes on a Flask server (also Dash's app.server) for the page's checks."""
    from flask import request

    def _alive():
        tracker.seen(request.args.get("id"))
        return "ok"

    def _bye():
        tracker.bye(request.args.get("id"))
        return "ok"

    server.add_url_rule(ALIVE_PATH, "tabwatch_alive", _alive, methods=["GET"])
    server.add_url_rule(BYE_PATH, "tabwatch_bye", _bye, methods=["POST"])


def watch_js(app_name, alive_path=ALIVE_PATH, interval_ms=2000, misses=2, dark=False):
    """JavaScript for the page: regular checks with this tab's id, goodbye
    when the page goes away, and closing the tab once the server has stopped."""
    colour, accent = ("#e5e7eb", "#f57d15") if dark else ("#222222", "#f57d15")
    msg = ('<div style="font-family:system-ui,sans-serif;padding:60px;text-align:center;'
           f'color:{colour}">'
           f'<h3 style="color:{accent}">{app_name} has stopped</h3>'
           '<p>You can close this tab.</p></div>')
    return f"""
(function() {{
  // {app_name} and this tab stop together: the tab closes when {app_name}
  // stops (Quit / Stop on the home page, its own Quit button, a crash), and
  // {app_name} stops a few seconds after its last tab is closed.  Only an
  // unreachable server counts as stopped; any reply, even slow, is fine.
  if (window.__asphodelTabWatch) return;
  window.__asphodelTabWatch = true;
  var tabId = Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
  var url = {json.dumps(alive_path)};
  var missed = 0, done = false, busy = false;
  function stopped() {{
    done = true;
    try {{ window.close(); }} catch (e) {{}}
    setTimeout(function() {{ document.body.innerHTML = {json.dumps(msg)}; }}, 300);
  }}
  function check() {{
    if (done || busy) return;
    busy = true;
    fetch(url + '?id=' + tabId + '&t=' + Date.now(), {{cache: 'no-store'}})
      .then(function() {{ missed = 0; }}, function() {{ missed += 1; }})
      .then(function() {{ busy = false; if (missed >= {int(misses)}) stopped(); }});
  }}
  check();
  setInterval(check, {int(interval_ms)});
  // goodbye when the tab is closed or reloaded (a reload checks in again at once)
  window.addEventListener('pagehide', function() {{
    if (done) return;
    try {{ navigator.sendBeacon(url + '/bye?id=' + tabId); }} catch (e) {{}}
  }});
  // back from the browser's page cache: check in again straight away
  window.addEventListener('pageshow', function(ev) {{ if (ev.persisted) check(); }});
}})();
"""
