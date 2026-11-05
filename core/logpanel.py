"""
"Log" panels that show a tool's own output (core/logcapture.py) in its own
window, so nothing that would go to the terminal is lost when the suite is
started from the Asphodel launcher.

  Dash apps (home page, Theia, Segrè chart, GUI correction):
      app.layout = logpanel.dash_layout(serve_layout, "Theia")   # adds the panel
      logpanel.register_dash(app)                                # its callbacks
      ... logpanel.dash_button(...) ...                          # in the title bar
    The "Log" button (with a ⚠ count of warnings and errors) opens a panel
    with the latest output, refreshed while it is open.  An app without a
    title-bar button gets a floating one at the bottom left (floating=True).  The
    home page (files=True) can also show the saved logs of every tool,
    including tools without a window of their own (Sum spectra).

  Spectrogram page (own HTTP server):  html_snippet() before </body> and the
    GET /_aslog, POST /_aslog/open endpoints answered with http_lines() /
    open_target().

  matplotlib windows (decay fit, correction plot):  add_mpl_button(fig, rect)
    adds an "Output" button that opens a window with the latest output.

  Tk (LevelSchemer):  open_tk_window(root) opens a window with the output.
"""

import html as _html
import os
import subprocess

from . import logcapture

PANEL_LINES = 400            # lines shown in a panel
ACCENT = "#f57d15"
PALE = "#fff4ec"
ERR_COLOUR = "#b3261e"
WARN_COLOUR = "#9a4d00"


def open_target(what, path=None):
    """Open this run's log file ("file"), another log file (path) or the log
    folder ("folder") with the desktop's default application.  Returns a
    short message."""
    target = logcapture.LOG_DIR if what == "folder" else (path or logcapture.log_path())
    if not target or not os.path.exists(target):
        return "No log file yet."
    try:
        subprocess.Popen(["xdg-open", target], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
    except OSError as exc:
        return f"Could not open it ({exc}); it is {target}"
    return f"Opened {target}"


def _badge_text(c):
    n = c["warning"] + c["error"]
    return f"⚠ {n}" if n else ""


# ---------------------------------------------------------------------------
# Dash
# ---------------------------------------------------------------------------

_BTN_STYLE = {"position": "fixed", "left": "10px", "bottom": "10px", "zIndex": 3100,
              "padding": "3px 10px", "fontSize": "12px", "border": "1px solid #bbb",
              "borderRadius": "4px", "background": "#fff", "color": "#333", "cursor": "pointer",
              "boxShadow": "0 1px 3px rgba(0,0,0,.15)"}
_PANEL_STYLE = {"position": "fixed", "left": "10px", "bottom": "10px", "zIndex": 3100,
                "width": "min(960px, calc(100vw - 20px))", "height": "48vh",
                "background": "#fff", "border": "1px solid #bbb", "borderRadius": "4px",
                "boxShadow": "0 4px 16px rgba(0,0,0,.2)", "flexDirection": "column",
                "fontSize": "12px", "color": "#222"}
_SMALL_BTN = {"fontSize": "12px", "padding": "1px 8px", "marginLeft": "6px",
              "border": "1px solid #bbb", "borderRadius": "3px", "background": "#f9f9f9",
              "cursor": "pointer"}


def dash_button(className=None, style=None):
    """The Log button, for an app's title bar."""
    from dash import html
    return html.Button(["Log ", html.Span(id="aslog-badge",
                                          style={"color": ACCENT, "fontWeight": "700"})],
                       id="aslog-btn", n_clicks=0, className=className, style=style,
                       title="Output of this program (what would appear in a terminal)")


def dash_component(title, files=False, floating=False):
    """The Log panel (hidden until the Log button is clicked); floating=True
    adds a Log button at the bottom left.  files=True: a dropdown to show the
    saved log of any tool (home page)."""
    from dash import dcc, html
    head = [html.B(f"Output · {title}"),
            html.Span(id="aslog-path", style={"color": "#777", "marginLeft": "10px",
                                              "overflow": "hidden", "textOverflow": "ellipsis",
                                              "whiteSpace": "nowrap", "flex": "1"}),
            html.Button("Open log file", id="aslog-open-file", n_clicks=0, style=_SMALL_BTN),
            html.Button("Open log folder", id="aslog-open-folder", n_clicks=0, style=_SMALL_BTN),
            html.Button("Close", id="aslog-close", n_clicks=0, style=_SMALL_BTN)]
    rows = [html.Div(head, style={"display": "flex", "alignItems": "center",
                                  "padding": "6px 10px", "borderBottom": f"2px solid {ACCENT}",
                                  "background": "#f9f9f9"})]
    if files:
        rows.append(html.Div([
            html.Span("Show: ", style={"marginRight": "6px"}),
            dcc.Dropdown(id="aslog-file", clearable=False, value="",
                         style={"flex": "1", "fontSize": "12px"})],
            style={"display": "flex", "alignItems": "center", "padding": "4px 10px",
                   "borderBottom": "1px solid #ddd"}))
    rows.append(html.Div(id="aslog-msg", style={"padding": "0 10px", "color": "#777"}))
    # column-reverse keeps the view at the newest (bottom) line
    rows.append(html.Div(html.Div(id="aslog-body"),
                         style={"flex": "1", "overflowY": "auto", "display": "flex",
                                "flexDirection": "column-reverse", "padding": "4px 10px",
                                "fontFamily": "monospace", "whiteSpace": "pre-wrap",
                                "wordBreak": "break-word"}))
    panel = {**_PANEL_STYLE, "display": "none"}
    if floating:
        panel["bottom"] = "44px"
    return html.Div(id="aslog-root", children=(
        [dash_button(style=_BTN_STYLE)] if floating else []) + [
        html.Div(rows, id="aslog-panel", style=panel),
        dcc.Interval(id="aslog-poll", interval=1500),
    ])


def dash_layout(serve_layout, title, files=False, floating=False):
    """A layout function: serve_layout() with the Log button and panel added
    to its root element (app.layout = logpanel.dash_layout(serve_layout, "Theia"))."""
    def layout():
        root = serve_layout() if callable(serve_layout) else serve_layout
        kids = root.children
        kids = list(kids) if isinstance(kids, (list, tuple)) else ([kids] if kids is not None else [])
        # a layout function may hand back the same object every time (Theia):
        # add the panel only once
        if not any(getattr(k, "id", None) == "aslog-root" for k in kids):
            root.children = kids + [dash_component(title, files, floating)]
        return root
    return layout


def _dash_rows(entries):
    from dash import html
    out = []
    for _i, t, stream, level, text in entries:
        st = {}
        if level == "error":
            st = {"color": ERR_COLOUR, "background": "#fdf0ef"}
        elif level == "warning":
            st = {"color": WARN_COLOUR, "background": PALE}
        out.append(html.Div([html.Span(f"{t} " if t else "", style={"color": "#999"}), text],
                            style=st))
    return out or [html.Div("(no output yet)", style={"color": "#999"})]


def register_dash(app, files=False):
    """Callbacks of dash_component (call once per app)."""
    import dash
    from dash import Input, Output, State, ctx

    @app.callback(Output("aslog-panel", "style"),
                  Input("aslog-btn", "n_clicks"), Input("aslog-close", "n_clicks"),
                  State("aslog-panel", "style"), prevent_initial_call=True)
    def _aslog_toggle(_a, _b, style):
        is_open = (style or {}).get("display") == "flex"
        want = False if ctx.triggered_id == "aslog-close" else not is_open
        return {**(style or _PANEL_STYLE), "display": "flex" if want else "none"}

    inputs = [Input("aslog-poll", "n_intervals")]
    if files:
        inputs.append(Input("aslog-file", "value"))

    outputs = [Output("aslog-badge", "children"), Output("aslog-body", "children"),
               Output("aslog-path", "children")]
    if files:
        outputs.append(Output("aslog-file", "options"))

    states = [State("aslog-file", "options")] if files else []

    # the panel's style is an input too, so opening it shows the output at once
    @app.callback(*outputs, *inputs, Input("aslog-panel", "style"), *states)
    def _aslog_refresh(_n, *rest):
        chosen = rest[0] if files else ""
        style = rest[1] if files else rest[0]
        old_opts = rest[2] if files else None
        badge = _badge_text(logcapture.counts())
        if (style or {}).get("display") != "flex":
            return (badge, dash.no_update, dash.no_update) + ((dash.no_update,) if files else ())
        own = logcapture.log_path()
        if chosen and chosen != own:
            entries = logcapture.read_file_lines(chosen, PANEL_LINES)
            path = chosen
        else:
            entries = logcapture.lines(limit=PANEL_LINES)[1]
            path = own or "(not being saved to a file)"
        out = (badge, _dash_rows(entries), path)
        if files:
            opts = [{"label": "This run of the home page", "value": ""}]
            import datetime
            import re
            found = []
            for p, t, _m in logcapture.log_files():
                if p == own:
                    continue
                # start time from the file name (stable, so the list does not
                # change while a tool is writing)
                mt = re.search(r"_(\d{8}_\d{6})_\d+\.log$", p)
                try:
                    start = datetime.datetime.strptime(mt.group(1), "%Y%m%d_%H%M%S")
                except (AttributeError, ValueError):
                    start = datetime.datetime.fromtimestamp(0)
                found.append((start, t, p))
            for start, t, p in sorted(found, reverse=True):
                opts.append({"label": f"{t} · started {start:%d %b %H:%M:%S}", "value": p})
            # unchanged list: leave the dropdown alone (it is reset by every update)
            out += (dash.no_update if opts == old_opts else opts,)
        return out

    file_state = [State("aslog-file", "value")] if files else []

    @app.callback(Output("aslog-msg", "children"),
                  Input("aslog-open-file", "n_clicks"), Input("aslog-open-folder", "n_clicks"),
                  *file_state, prevent_initial_call=True)
    def _aslog_open(_a, _b, *chosen):
        if ctx.triggered_id == "aslog-open-folder":
            return open_target("folder")
        return open_target("file", (chosen[0] if chosen else None) or None)


# ---------------------------------------------------------------------------
# Plain HTML page (spectrogram)
# ---------------------------------------------------------------------------

def http_lines(since=0):
    """JSON-able answer for GET /_aslog?since=N."""
    nxt, entries = logcapture.lines(since=since, limit=PANEL_LINES)
    return {"next": nxt, "counts": logcapture.counts(), "path": logcapture.log_path() or "",
            "lines": [[t, lvl, text] for _i, t, _s, lvl, text in entries]}


def html_snippet(title):
    """Log button + panel + script for a plain HTML page (put before </body>)."""
    t = _html.escape(title)
    return f"""
<style>
#aslog-btn{{position:fixed;left:10px;bottom:10px;z-index:3100;padding:3px 10px;font-size:12px;
  border:1px solid #bbb;border-radius:4px;background:#fff;color:#333;cursor:pointer;
  box-shadow:0 1px 3px rgba(0,0,0,.15);font-family:sans-serif}}
#aslog-btn b{{color:{ACCENT}}}
#aslog-btn.inbar{{position:static;box-shadow:none;margin-right:8px;font-weight:600;padding:4px 12px}}
#aslog-panel{{position:fixed;left:10px;bottom:44px;z-index:3100;width:min(960px,calc(100vw - 20px));
  height:48vh;background:#fff;border:1px solid #bbb;border-radius:4px;box-shadow:0 4px 16px rgba(0,0,0,.2);
  display:none;flex-direction:column;font-size:12px;color:#222;font-family:sans-serif}}
#aslog-head{{display:flex;align-items:center;padding:6px 10px;border-bottom:2px solid {ACCENT};background:#f9f9f9}}
#aslog-head button{{font-size:12px;padding:1px 8px;margin-left:6px;border:1px solid #bbb;border-radius:3px;
  background:#f9f9f9;cursor:pointer;color:#222}}
#aslog-path{{color:#777;margin-left:10px;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
#aslog-msg{{padding:0 10px;color:#777}}
#aslog-scroll{{flex:1;overflow-y:auto;display:flex;flex-direction:column-reverse;padding:4px 10px;
  font-family:monospace;white-space:pre-wrap;word-break:break-word}}
#aslog-body .e{{color:{ERR_COLOUR};background:#fdf0ef}} #aslog-body .w{{color:{WARN_COLOUR};background:{PALE}}}
#aslog-body .t{{color:#999}}
</style>
<button id="aslog-btn" title="Output of this program (what would appear in a terminal)">Log <b id="aslog-badge"></b></button>
<div id="aslog-panel"><div id="aslog-head"><b>Output · {t}</b><span id="aslog-path"></span>
<button id="aslog-of">Open log file</button><button id="aslog-od">Open log folder</button>
<button id="aslog-x">Close</button></div><div id="aslog-msg"></div>
<div id="aslog-scroll"><div id="aslog-body"></div></div></div>
<script>
(function(){{
  var next = 0, open = false, body = document.getElementById('aslog-body');
  function esc(s){{return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}}
  async function poll(){{
    try {{
      var r = await fetch('/_aslog?since=' + (open ? next : 1e12));
      var d = await r.json();
      var n = d.counts.warning + d.counts.error;
      document.getElementById('aslog-badge').textContent = n ? '\\u26a0 ' + n : '';
      document.getElementById('aslog-path').textContent = d.path || '(not being saved to a file)';
      if (open) {{
        var h = '';
        d.lines.forEach(function(l){{
          var c = l[1] === 'error' ? 'e' : (l[1] === 'warning' ? 'w' : '');
          h += '<div class="' + c + '"><span class="t">' + esc(l[0]) + ' </span>' + esc(l[2]) + '</div>';
        }});
        if (h) body.insertAdjacentHTML('beforeend', h);
        while (body.childNodes.length > {PANEL_LINES}) body.removeChild(body.firstChild);
        next = d.next;
      }}
    }} catch (e) {{}}
  }}
  function show(v){{
    open = v; document.getElementById('aslog-panel').style.display = v ? 'flex' : 'none';
    if (v) {{ next = 0; body.innerHTML = ''; poll(); }}
  }}
  async function openIt(what){{
    try {{
      var r = await fetch('/_aslog/open', {{method:'POST', headers:{{'Content-Type':'application/json'}},
                                           body: JSON.stringify({{what: what}})}});
      document.getElementById('aslog-msg').textContent = (await r.json()).message || '';
    }} catch (e) {{}}
  }}
  // in the page's toolbar, left of its Quit button, when there is one
  var quit = document.getElementById('quitBtn'), btn = document.getElementById('aslog-btn');
  if (quit && quit.parentNode) {{
    btn.classList.add('inbar'); quit.parentNode.insertBefore(btn, quit);
    document.getElementById('aslog-panel').style.bottom = '10px';
  }}
  btn.onclick = function(){{ show(!open); }};
  document.getElementById('aslog-x').onclick = function(){{ show(false); }};
  document.getElementById('aslog-of').onclick = function(){{ openIt('file'); }};
  document.getElementById('aslog-od').onclick = function(){{ openIt('folder'); }};
  poll(); setInterval(poll, 1500);
}})();
</script>
"""


def inject_html(page, title):
    """`page` with the Log snippet put before its last </body> (unchanged if it has none)."""
    i = page.rfind("</body>")
    if i < 0 or 'id="aslog-btn"' in page:
        return page
    return page[:i] + html_snippet(title) + page[i:]


# ---------------------------------------------------------------------------
# matplotlib
# ---------------------------------------------------------------------------

def show_mpl_output(title=None):
    """A matplotlib window with the latest output of this program."""
    import matplotlib.pyplot as plt
    entries = logcapture.lines(limit=70)[1]
    fig = plt.figure(figsize=(10, 7))
    try:
        fig.canvas.manager.set_window_title(f"Output · {title or logcapture.tool() or ''}")
    except AttributeError:
        pass
    fig.text(0.01, 0.985, f"Log file: {logcapture.log_path() or '(not being saved)'}",
             fontsize=8, color="#777", va="top", family="monospace")
    y, dy = 0.955, 0.0135
    for _i, t, _s, level, text in entries:
        colour = {"error": ERR_COLOUR, "warning": WARN_COLOUR}.get(level, "#222")
        shown = text if len(text) <= 160 else text[:157] + "…"
        fig.text(0.01, y, f"{t}  {shown}", fontsize=7, family="monospace", color=colour,
                 va="top")
        y -= dy
    if not entries:
        fig.text(0.01, 0.95, "(no output yet)", fontsize=8, color="#999", va="top")
    fig.show()
    return fig


def add_mpl_buttons(rect, title=None):
    """add_mpl_button on every open figure that has none yet; rect is a list
    or a function of the figure returning one."""
    import matplotlib.pyplot as plt
    for n in plt.get_fignums():
        f = plt.figure(n)
        if not hasattr(f, "_asphodel_output_button"):
            add_mpl_button(f, rect(f) if callable(rect) else rect, title)


def add_mpl_button(fig, rect, title=None):
    """An "Output" button on a matplotlib figure, at rect = [left, bottom,
    width, height] in figure coordinates."""
    from matplotlib.widgets import Button
    ax = fig.add_axes(rect)
    try:
        ax.set_in_layout(False)                # not moved by constrained layout
    except AttributeError:
        pass
    btn = Button(ax, "Output", color=PALE, hovercolor=ACCENT)
    btn.label.set_fontsize(12)
    btn.on_clicked(lambda _e: show_mpl_output(title))
    fig._asphodel_output_button = btn          # keep a reference (or the button stops working)
    return btn


# ---------------------------------------------------------------------------
# Tk
# ---------------------------------------------------------------------------

def open_tk_window(root, title=None):
    """A Tk window with this program's output, updated every second."""
    import tkinter as tk
    from tkinter.scrolledtext import ScrolledText
    win = tk.Toplevel(root)
    win.title(f"Output · {title or logcapture.tool() or ''}")
    win.geometry("900x420")
    bar = tk.Frame(win)
    bar.pack(fill="x")
    tk.Label(bar, text=f"Log file: {logcapture.log_path() or '(not being saved)'}",
             fg="#777", anchor="w").pack(side="left", fill="x", expand=True, padx=6)
    tk.Button(bar, text="Open log folder", command=lambda: open_target("folder")).pack(side="right")
    tk.Button(bar, text="Open log file", command=lambda: open_target("file")).pack(side="right")
    txt = ScrolledText(win, font=("monospace", 9), wrap="word")
    txt.pack(fill="both", expand=True)
    txt.tag_configure("error", foreground=ERR_COLOUR, background="#fdf0ef")
    txt.tag_configure("warning", foreground=WARN_COLOUR, background=PALE)
    txt.tag_configure("time", foreground="#999")
    state = {"next": 0}

    def refresh():
        try:
            if not win.winfo_exists():
                return
        except tk.TclError:
            return
        nxt, entries = logcapture.lines(since=state["next"], limit=PANEL_LINES)
        state["next"] = nxt
        if entries:
            at_end = txt.yview()[1] > 0.999
            txt.configure(state="normal")
            for _i, t, _s, level, text in entries:
                txt.insert("end", f"{t} ", ("time",))
                txt.insert("end", text + "\n", (level,) if level else ())
            txt.configure(state="disabled")
            if at_end:
                txt.see("end")
        win.after(1000, refresh)

    txt.configure(state="disabled")
    refresh()
    return win


def http_open(body):
    """Answer for POST /_aslog/open ({"what": "file" | "folder"})."""
    what = (body or {}).get("what")
    return {"message": open_target("folder" if what == "folder" else "file")}


__all__ = ["dash_component", "dash_layout", "register_dash", "html_snippet", "inject_html", "http_lines",
           "http_open", "add_mpl_button", "add_mpl_buttons", "show_mpl_output", "open_tk_window", "open_target"]
