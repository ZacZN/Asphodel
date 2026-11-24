"""
startup_timing.py  –  how long each Asphodel tool takes to start, step by step

Run it from the repo folder with the suite's Python, with the suite closed:

    .venv/bin/python tests/startup_timing.py              # all web tools, 2 runs each
    .venv/bin/python tests/startup_timing.py --runs 3 --tools theia,segre
    .venv/bin/python tests/startup_timing.py --no-browser # Python side only

For every tool (home page, Theia, Segrè chart, Results Log Book, GUI Second
Order Correction, Spectrogram) and every run it measures:

  1. Python start      the interpreter starting (before any suite code)
  2. Until server      imports, building the page, loading data (the
                       spectrogram reopens your last session), until the
                       tool's web server starts
  3. Server answering  until the port accepts connections
  4. Browser           the page opened in the suite's Firefox window, through
                       a small recording proxy: page and script files, the
                       Dash callbacks fired when the page loads (the slowest
                       are listed), until nothing has been requested for
                       1.5 s ("settled"; periodic polls are ignored)

Once per tool it also lists the slowest imports (python -X importtime), and
once at the start how long the suite's Firefox takes to start (or to open a
tab, if it is already running) and whether the desktop launcher wraps Python
in `conda run`.

What it changes: nothing in your settings or data.  The tools run with their
normal settings, but their output goes to this script instead of the log
files (so your data/logs/ is not pruned).  Each tool is stopped after its
run; its tab then closes itself.  If the suite's Firefox was not running, it
is started and left open.  The report is printed and saved to
<workspace>/logs/startup_timing_<date>_<time>.txt.

Tools whose port is already in use (already running) are skipped: close the
suite first.  Not measured: LevelSchemer (Tk window), Decay Fit
(matplotlib, started from the spectrogram) and Sum Spectra (command line).
"""

import argparse
import datetime
import http.client
import http.server
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THIS = os.path.abspath(__file__)

# name: (label, script, port name in core/ports.py or None, page path)
TOOLS = {
    "home":           ("Home Page", "home.py", "home", "/"),
    "theia":          ("Theia", "theia/theia.py", "theia", "/"),
    "segre":          ("Segrè Chart", "reference/segre_chart.py", "segre", "/"),
    "logbook":        ("Results Log Book", "reference/results_logbook.py", "logbook", "/"),
    "gui_correction": ("GUI Second Order Correction", "tartarus/gui_2nd_order_correction.py",
                       "gui_correction", "/"),
    "spectrogram":    ("Spectrogram", "spectrogram/plot_spectrogram.py", None, "/spectrogram.html"),
}

ENV_EVENTS = "ASPHODEL_TIMING_EVENTS"     # child writes its events (JSON lines) here
ENV_STOP = "ASPHODEL_TIMING_STOP"         # child exits when its server would start
QUIET_S = 1.5                             # no requests for this long = page settled
SETTLE_TIMEOUT_S = 90.0
READY_TIMEOUT_S = 180.0
IGNORED_PATHS = ("/_alive", "/_aslog", "/_touchpad", "/favicon.ico")


# ===========================================================================
# Child mode: run one tool with timing hooks
# ===========================================================================

def _child(script, args):
    t_python = time.time()
    events = os.environ.get(ENV_EVENTS)
    stop = os.environ.get(ENV_STOP) == "1"

    def event(name, **kw):
        if events:
            with open(events, "a") as fh:
                fh.write(json.dumps(dict(event=name, t=time.time(), **kw)) + "\n")

    event("python", t_python=t_python)
    print("ASPHODEL-TIMING-START", file=sys.stderr, flush=True)   # import list starts here
    sys.path.insert(0, REPO)
    from core import logcapture
    logcapture.start = lambda *a, **k: None        # don't write to (and prune) data/logs/

    def server_starting(port):
        event("server", port=port)
        if stop:
            sys.stdout.flush()
            os._exit(0)

    # Dash (werkzeug) and the spectrogram both serve with an http.server.HTTPServer
    orig_serve = http.server.HTTPServer.serve_forever

    def serve_forever(self, *a, **k):
        server_starting(self.server_address[1])
        return orig_serve(self, *a, **k)
    http.server.HTTPServer.serve_forever = serve_forever

    import runpy
    path = os.path.join(REPO, script)
    sys.argv = [path, *args]
    sys.path.insert(0, os.path.dirname(path))
    runpy.run_path(path, run_name="__main__")


# ===========================================================================
# Recording proxy (sees every request the page makes)
# ===========================================================================

class Recorder:
    def __init__(self):
        self.lock = threading.Lock()
        self.reqs = []          # dicts: start, end, method, path, status, bytes, label, ignored
        self.active = 0

    def snapshot(self):
        with self.lock:
            return list(self.reqs), self.active


def _callback_label(body):
    """(label, ignored) for a Dash callback request body."""
    try:
        d = json.loads(body)
    except ValueError:
        return "callback", False
    out = d.get("output", "")
    changed = d.get("changedPropIds") or []
    periodic = bool(changed) and all(c.endswith(".n_intervals") for c in changed)
    trig = ", ".join(changed) if changed else "on load"
    trig = trig if len(trig) < 70 else trig[:67] + "..."
    out = out if len(out) < 90 else out[:87] + "..."
    return f"{out}   (trigger: {trig})", periodic


def make_proxy(target_port, rec):
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _forward(self):
            t0 = time.time()
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n) if n else b""
            path = self.path
            label, ignored = path.split("?")[0], path.startswith(IGNORED_PATHS)
            if self.command == "POST" and "_dash-update-component" in path:
                label, ignored = _callback_label(body)
            with rec.lock:
                rec.active += 0 if ignored else 1
            headers = {k: v for k, v in self.headers.items()
                       if k.lower() not in ("host", "connection", "accept-encoding",
                                            "keep-alive", "content-length")}
            status, data, rh = 502, b"", []
            try:
                c = http.client.HTTPConnection("127.0.0.1", target_port, timeout=120)
                c.request(self.command, path, body=body or None, headers=headers)
                r = c.getresponse()
                status, data, rh = r.status, r.read(), r.getheaders()
                c.close()
            except OSError as exc:
                data = str(exc).encode()
            t1 = time.time()
            with rec.lock:
                rec.active -= 0 if ignored else 1
                rec.reqs.append(dict(start=t0, end=t1, method=self.command, path=path,
                                     status=status, bytes=len(data), label=label,
                                     ignored=ignored))
            try:
                self.send_response(status)
                for k, v in rh:
                    if k.lower() not in ("transfer-encoding", "connection", "content-length",
                                         "keep-alive"):
                        self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(data)
            except OSError:
                pass

        do_GET = do_POST = do_HEAD = do_PUT = do_DELETE = _forward

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def wait_settled(rec, t_open):
    """Seconds from t_open until the page settled, or None on timeout."""
    deadline = t_open + SETTLE_TIMEOUT_S
    while time.time() < deadline:
        time.sleep(0.1)
        reqs, active = rec.snapshot()
        rel = [r for r in reqs if not r["ignored"]]
        if not rel or active:
            continue
        last = max(r["end"] for r in rel)
        if time.time() - last >= QUIET_S:
            return last - t_open
    return None


# ===========================================================================
# Helpers
# ===========================================================================

def port_open(port):
    with socket.socket() as s:
        s.settimeout(0.2)
        return s.connect_ex(("127.0.0.1", port)) == 0


def read_events(path):
    out = []
    try:
        with open(path) as fh:
            for line in fh:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    except OSError:
        pass
    return out


def stop_proc(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait(timeout=5)


def tail(path, n=15):
    try:
        with open(path, errors="replace") as fh:
            return "".join(fh.readlines()[-n:])
    except OSError:
        return ""


def mb(n):
    return f"{n / 1e6:.1f} MB" if n >= 1e5 else f"{n / 1e3:.0f} kB"


class Report:
    def __init__(self):
        self.lines = []

    def __call__(self, text=""):
        print(text, flush=True)
        self.lines.append(text)


# ===========================================================================
# Measurements
# ===========================================================================

def launcher_info(say):
    say("Launcher")
    say("--------")
    say(f"Python used here: {sys.executable}")
    t = time.time()
    subprocess.run([sys.executable, "-c", "pass"])
    say(f"  bare Python start:                     {time.time() - t:5.2f} s")
    apps = os.path.expanduser("~/.local/share/applications")
    exec_line = None
    try:
        for f in sorted(os.listdir(apps)):
            if f.startswith("asphodel") and f.endswith(".desktop"):
                with open(os.path.join(apps, f)) as fh:
                    text = fh.read()
                if REPO in text:
                    m = re.search(r"^Exec=(.*)$", text, re.M)
                    if m:
                        exec_line = m.group(1)
                        say(f"Desktop launcher {f}: {exec_line}")
                        break
    except OSError:
        pass
    if exec_line is None:
        say("  (no desktop launcher for this folder found)")
    else:
        try:
            parts = shlex.split(exec_line)
        except ValueError:
            parts = []
        if "conda" in os.path.basename(parts[0] if parts else "") and parts[1:2] == ["run"]:
            cmd = [p for p in parts if not p.endswith("home.py")] + ["-c", "pass"]
            t = time.time()
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            say(f"  Python start through `conda run`:      {time.time() - t:5.2f} s"
                "  (added before the home page starts)")
        else:
            say("  launcher starts Python directly (no conda run)")
    say()


def firefox_start(say):
    """Time the suite's Firefox opening a tiny local page."""
    from core import browser
    say("Suite Firefox")
    say("-------------")
    exe = browser.firefox()
    if not exe:
        say("  Firefox not found: the tools open in the default browser; not measured.")
        say()
        return
    was_running = browser._running(browser.profile_dir(exe))
    hits = {}

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            hits.setdefault(self.path.split("?")[0], time.time())
            page = (b"<!doctype html><title>Asphodel timing</title><p>Timing test, "
                    b"this tab can be closed.<script>fetch('/loaded').then(function()"
                    b"{window.close()})</script>")
            body = page if self.path.startswith("/start") else b"ok"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    t0 = time.time()
    browser.open_url(f"http://127.0.0.1:{srv.server_address[1]}/start", new_window=not was_running)
    while time.time() - t0 < 60 and "/loaded" not in hits:
        time.sleep(0.02)
    srv.shutdown()
    what = "open a tab in the running suite Firefox" if was_running else \
        "start the suite Firefox (it was not running)"
    if "/loaded" in hits:
        say(f"  {what}: page requested after {hits['/start'] - t0:5.2f} s, "
            f"loaded after {hits['/loaded'] - t0:5.2f} s")
    else:
        say(f"  {what}: no page load within 60 s")
    if not was_running:
        say("  (a cold start happens whenever the home page opens with the suite Firefox closed)")
    say()


def import_breakdown(name, say, top=10):
    label, script, _, _ = TOOLS[name]
    with tempfile.TemporaryDirectory() as tmp:
        err = os.path.join(tmp, "importtime.txt")
        env = dict(os.environ, ASPHODEL_NO_BROWSER="1", **{ENV_STOP: "1"})
        with open(err, "w") as fh:
            subprocess.run([sys.executable, "-X", "importtime", THIS, "--child", script],
                           cwd=REPO, env=env, stdout=subprocess.DEVNULL, stderr=fh,
                           timeout=READY_TIMEOUT_S)
        rows, started = [], False
        with open(err, errors="replace") as fh:
            for line in fh:
                started = started or line.startswith("ASPHODEL-TIMING-START")
                if not started:
                    continue
                m = re.match(r"import time:\s+(\d+)\s+\|\s+(\d+)\s+\|( *)(\S+)", line)
                if m and len(m.group(3)) <= 1:          # top-level imports only
                    rows.append((int(m.group(2)) / 1e6, m.group(4)))
    rows.sort(reverse=True)
    if rows:
        say("  Slowest imports (cumulative, first import of each):")
        for s, mod in [r for r in rows if r[0] >= 0.01][:top]:
            say(f"    {s:5.2f} s  {mod}")


def run_tool(name, run_no, use_browser, say):
    label, script, port_name, page = TOOLS[name]
    from core import ports
    fixed_port = ports.port(port_name) if port_name and name != "home" else None
    if fixed_port and port_open(fixed_port):
        say(f"  run {run_no}: port {fixed_port} is in use ({label} already running?) – skipped")
        return False
    tmp = tempfile.mkdtemp(prefix="asph_timing_")
    ev_file, out_file = os.path.join(tmp, "events.jsonl"), os.path.join(tmp, "output.txt")
    env = dict(os.environ, ASPHODEL_NO_BROWSER="1", **{ENV_EVENTS: ev_file})
    env.pop(ENV_STOP, None)
    t0 = time.time()
    with open(out_file, "w") as out:
        proc = subprocess.Popen([sys.executable, THIS, "--child", script], cwd=REPO, env=env,
                                stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        port, t_server, t_python = None, None, None
        while time.time() - t0 < READY_TIMEOUT_S and proc.poll() is None:
            for e in read_events(ev_file):
                if e["event"] == "python":
                    t_python = e["t"]
                elif e["event"] == "server":
                    port, t_server = e["port"], e["t"]
            if port and port_open(port):
                break
            time.sleep(0.02)
        t_ready = time.time()
        if not (port and port_open(port)):
            say(f"  run {run_no}: did not start (exit code {proc.poll()}). Last output:")
            say("    " + tail(out_file).replace("\n", "\n    "))
            return False
        say(f"  run {run_no}:")
        if t_python:
            say(f"    Python start                          {t_python - t0:5.2f} s")
        if t_server:
            say(f"    imports + setup until server starts   {t_server - (t_python or t0):5.2f} s")
        say(f"    server answering (since launch)       {t_ready - t0:5.2f} s")
        if not use_browser:
            return True

        rec = Recorder()
        proxy = make_proxy(port, rec)
        from core import browser
        t_open = time.time()
        browser.open_url(f"http://127.0.0.1:{proxy.server_address[1]}{page}")
        settled = wait_settled(rec, t_open)
        proxy.shutdown()
        reqs = [r for r in rec.snapshot()[0] if not r["ignored"]]
        if not reqs:
            say("    browser: no requests arrived within 90 s (did the tab open?)")
            return True
        cbs = [r for r in reqs if r["method"] == "POST" and "_dash-update-component" in r["path"]]
        files = [r for r in reqs if r not in cbs]
        first = min(r["start"] for r in reqs)
        say(f"    browser asks for the page after       {first - t_open:5.2f} s")
        if files:
            say(f"    page + script files loaded after      {max(r['end'] for r in files) - t_open:5.2f} s"
                f"   ({len(files)} requests, {mb(sum(r['bytes'] for r in files))})")
        if cbs:
            busy = sum(r["end"] - r["start"] for r in cbs)
            say(f"    callbacks on load: {len(cbs)}, first at {min(r['start'] for r in cbs) - t_open:.2f} s, "
                f"last done {max(r['end'] for r in cbs) - t_open:.2f} s  (server time summed {busy:.2f} s)")
            for r in sorted(cbs, key=lambda r: r["end"] - r["start"], reverse=True)[:5]:
                say(f"      {1000 * (r['end'] - r['start']):6.0f} ms  {mb(r['bytes']):>8}  {r['label']}")
        slow_other = [r for r in files if r["end"] - r["start"] > 0.3]
        for r in sorted(slow_other, key=lambda r: r["end"] - r["start"], reverse=True)[:5]:
            say(f"      {1000 * (r['end'] - r['start']):6.0f} ms  {mb(r['bytes']):>8}  "
                f"{r['method']} {r['path'][:80]}")
        if settled is None:
            say(f"    page did not settle within {SETTLE_TIMEOUT_S:.0f} s")
        else:
            say(f"    page settled after                    {settled:5.2f} s")
            say(f"    TOTAL launch → settled page           {t_open - t0 + settled:5.2f} s"
                + ("   (+ the tool's own fixed 1 s wait before it opens the browser)"
                   if name != "spectrogram" else ""))
        return True
    finally:
        stop_proc(proc)
        if name == "spectrogram" and run_no == 1:
            txt = tail(out_file, 40)
            if "Reopening last session" in txt:
                say("    (spectrogram reopened your last session file)")
            elif "file chooser" in txt or "Serving" in txt:
                say("    (spectrogram opened on the file chooser: no last session)")
        time.sleep(1.0)               # let the port close and the tab notice


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--child", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    ap.add_argument("--runs", type=int, default=2, help="runs per tool (default 2)")
    ap.add_argument("--tools", default=",".join(TOOLS),
                    help="comma-separated, from: " + ", ".join(TOOLS))
    ap.add_argument("--no-browser", action="store_true", help="skip everything in Firefox")
    ap.add_argument("--no-imports", action="store_true", help="skip the import lists")
    a = ap.parse_args()
    if a.child:
        return _child(a.child[0], a.child[1:])

    sys.path.insert(0, REPO)
    from core import paths
    names = [t.strip() for t in a.tools.split(",") if t.strip()]
    bad = [t for t in names if t not in TOOLS]
    if bad:
        sys.exit(f"Unknown tool(s): {', '.join(bad)}. Choose from: {', '.join(TOOLS)}")

    say = Report()
    say(f"Asphodel startup timing · {datetime.datetime.now():%Y-%m-%d %H:%M:%S} · {REPO}")
    try:
        say(f"Experiment: {paths.experiment()}   CPU cores: {os.cpu_count()}   "
            f"Load average: {', '.join(f'{x:.1f}' for x in os.getloadavg())}")
    except Exception:
        pass
    say()
    launcher_info(say)
    if not a.no_browser:
        firefox_start(say)
    for name in names:
        say(f"{TOOLS[name][0]}  ({TOOLS[name][1]})")
        say("-" * (len(TOOLS[name][0]) + len(TOOLS[name][1]) + 4))
        for i in range(1, a.runs + 1):
            if not run_tool(name, i, not a.no_browser, say):
                break
        if not a.no_imports:
            try:
                import_breakdown(name, say)
            except Exception as exc:
                say(f"  (import list failed: {exc})")
        say()

    try:
        logs = os.path.join(paths.DATA_ROOT, "logs")
        os.makedirs(logs, exist_ok=True)
        out = os.path.join(logs, f"startup_timing_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt")
        with open(out, "w") as fh:
            fh.write("\n".join(say.lines) + "\n")
        print(f"Report saved to {out}")
    except OSError as exc:
        print(f"Could not save the report: {exc}")


if __name__ == "__main__":
    main()
