"""
Browser tabs of the suite's tools close themselves when their program stops.

The page asks its own server every couple of seconds whether it is still
running (GET ALIVE_PATH, answered by each tool's server).  When two checks in
a row cannot reach it - the program was stopped from the home page (Quit or
Stop), stopped with its own Quit button, or crashed - the page closes its tab.

Browsers only let a page close a tab that a script opened, or one that has a
single entry in its history (true for a tab the tool opened itself with
webbrowser.open).  If the browser refuses, the page is replaced with a short
"has stopped" message instead.

Use:  put  watch_js("Theia")  into a <script> of the page, and answer
GET ALIVE_PATH with any 200 reply on the tool's server.
"""

import json

ALIVE_PATH = "/_alive"


def watch_js(app_name, alive_path=ALIVE_PATH, interval_ms=2000, misses=2, dark=False):
    """JavaScript that closes the tab once the tool's server has stopped."""
    colour, accent = ("#e5e7eb", "#f57d15") if dark else ("#222222", "#f57d15")
    msg = ('<div style="font-family:system-ui,sans-serif;padding:60px;text-align:center;'
           f'color:{colour}">'
           f'<h3 style="color:{accent}">{app_name} has stopped</h3>'
           '<p>You can close this tab.</p></div>')
    return f"""
(function() {{
  // Close this tab when {app_name} stops (Quit / Stop on the home page, its
  // own Quit button, or a crash).  Only an unreachable server counts; any
  // reply, even slow, means it is still running.
  if (window.__asphodelTabWatch) return;
  window.__asphodelTabWatch = true;
  var missed = 0, done = false, busy = false;
  function stopped() {{
    done = true;
    try {{ window.close(); }} catch (e) {{}}
    setTimeout(function() {{ document.body.innerHTML = {json.dumps(msg)}; }}, 300);
  }}
  setInterval(function() {{
    if (done || busy) return;
    busy = true;
    fetch({json.dumps(alive_path)} + '?t=' + Date.now(), {{cache: 'no-store'}})
      .then(function() {{ missed = 0; }}, function() {{ missed += 1; }})
      .then(function() {{ busy = false; if (missed >= {int(misses)}) stopped(); }});
  }}, {int(interval_ms)});
}})();
"""
