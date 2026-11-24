#!/usr/bin/env python3
"""
Smoke test: does a fresh Asphodel installation start?

    .venv/bin/python tests/smoke_test.py            # in a throw-away workspace
    .venv/bin/python tests/smoke_test.py --keep     # keep that workspace to look at

Run it with the Python Asphodel uses (.venv/bin/python after ./install.sh).
It never touches your own workspace: it sets ASPHODEL_WORKSPACE to a new
temporary folder, takes the first-run defaults without asking
(ASPHODEL_NONINTERACTIVE) and opens no browser (ASPHODEL_NO_BROWSER).

Checks:
  1. every .py file compiles
  2. the required packages import
  3. home.py starts, runs the first-run setup (workspace + Experiment1) and
     serves its page
  4. Theia, the Segrè chart, the GUI correction tool and the results log book
     start and serve their pages (on ports 8952-8958, so Asphodel can be open
     meanwhile; a tool is skipped if its port is in use anyway)
  5. the command-line tools answer --help (decay fit, sum spectra)

Exit code 0 if everything passed.
"""
import argparse
import compileall
import importlib
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGES = ["dash", "dash_bootstrap_components", "plotly", "flask", "numpy", "scipy", "pandas",
            "matplotlib", "pybaselines", "openpyxl", "odf"]
# the test runs on its own ports (core/ports.py + this offset), so it also
# works while Asphodel is open
TEST_PORT_OFFSET = 900
sys.path.insert(0, REPO)
from core.ports import BASE  # noqa: E402
# (name, script, port or None = read the address from the output)
SERVERS = [("Home Page", "home.py", None),
           ("Theia", "theia/theia.py", BASE["theia"] + TEST_PORT_OFFSET),
           ("Segrè Chart", "reference/segre_chart.py", BASE["segre"] + TEST_PORT_OFFSET),
           ("GUI Correction", "tartarus/gui_2nd_order_correction.py",
            BASE["gui_correction"] + TEST_PORT_OFFSET),
           ("Results Log Book", "reference/results_logbook.py", BASE["logbook"] + TEST_PORT_OFFSET)]
CLI_HELP = ["spectrogram/decay_fit.py", "tartarus/sum_spectra.py"]
START_TIMEOUT = 90          # s per server

RESULTS = []


def report(ok, name, detail=""):
    RESULTS.append(ok)
    mark = "PASS" if ok else ("SKIP" if ok is None else "FAIL")
    print(f"[{mark}] {name}" + (f": {detail}" if detail else ""), flush=True)


def port_open(port):
    with socket.socket() as s:
        s.settimeout(0.2)
        return s.connect_ex(("127.0.0.1", port)) == 0


def get(url):
    with urllib.request.urlopen(url, timeout=20) as r:
        return r.status, r.read()


def start_server(name, script, port, env, logdir):
    if port and port_open(port):
        report(None, name, f"port {port} already in use (is Asphodel running?)")
        return
    log = os.path.join(logdir, os.path.basename(script) + ".out")
    with open(log, "w") as out:
        proc = subprocess.Popen([sys.executable, os.path.join(REPO, script)], cwd=REPO, env=env,
                                stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        url, t0 = None, time.time()
        while time.time() - t0 < START_TIMEOUT:
            if proc.poll() is not None:
                break
            if port is None:
                m = re.search(r"http://127\.0\.0\.1:(\d+)/", open(log).read())
                if m and port_open(int(m.group(1))):
                    url = m.group(0)
                    break
            elif port_open(port):
                url = f"http://127.0.0.1:{port}/"
                break
            time.sleep(0.5)
        if not url:
            tail = open(log).read()[-2000:]
            report(False, name, f"did not start (exit code {proc.poll()}); last output:\n{tail}")
            return
        status, _ = get(url)
        status2, body = get(url + "_dash-layout")
        ok = status == 200 and status2 == 200 and len(body) > 100
        report(ok, name, f"{url} -> {status}, layout {status2} ({len(body)} bytes)")
    except Exception as e:
        report(False, name, f"{type(e).__name__}: {e}\n{open(log).read()[-2000:]}")
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(10)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep", action="store_true", help="keep the temporary workspace")
    args = ap.parse_args()

    print(f"Asphodel smoke test\n  repo:   {REPO}\n  python: {sys.executable} ({sys.version.split()[0]})")

    # 1. compile
    ok = compileall.compile_dir(REPO, quiet=1, rx=re.compile(r"/(\.venv|\.git|data|Claude outputs)/"),
                                force=True, legacy=False)
    report(bool(ok), "all .py files compile")

    # 2. packages
    bad = []
    for m in PACKAGES:
        try:
            importlib.import_module(m)
        except Exception as e:
            bad.append(f"{m} ({e})")
    report(not bad, "required packages import", ", ".join(bad))

    # 3.-5. in a throw-away workspace
    tmp = tempfile.mkdtemp(prefix="asphodel_smoke_")
    ws = os.path.join(tmp, "workspace")
    env = dict(os.environ, ASPHODEL_WORKSPACE=ws, ASPHODEL_NONINTERACTIVE="1",
               ASPHODEL_PORT_OFFSET=str(TEST_PORT_OFFSET),
               ASPHODEL_NO_BROWSER="1", MPLBACKEND=os.environ.get("MPLBACKEND", "Agg"))
    env.pop("ASPHODEL_EXPERIMENT", None)
    print(f"  workspace: {ws}")
    try:
        for name, script, port in SERVERS:
            start_server(name, script, port, env, tmp)
            if script == "home.py":
                exp = os.path.join(ws, "Experiment1")
                report(os.path.isdir(os.path.join(exp, "settings")),
                       "first-run setup created the workspace and Experiment1", exp)
        for script in CLI_HELP:
            r = subprocess.run([sys.executable, os.path.join(REPO, script), "--help"], cwd=REPO,
                               env=env, capture_output=True, text=True, timeout=120)
            report(r.returncode == 0, f"{script} --help",
                   "" if r.returncode == 0 else (r.stdout + r.stderr)[-1500:])
    finally:
        if args.keep:
            print(f"  kept: {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)

    failed = sum(1 for r in RESULTS if r is False)
    skipped = sum(1 for r in RESULTS if r is None)
    print(f"\n{len(RESULTS) - failed - skipped} passed, {failed} failed, {skipped} skipped")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
