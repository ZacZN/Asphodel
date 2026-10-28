#!/usr/bin/env python3
"""
home.py  –  Asphodel home page

The suite's launcher and overview, as a Dash app (http://127.0.0.1:8050/):

  * experiment profiles: switch the active experiment, create a new one
    (each experiment has its own data folder, settings, outputs and notes)
  * tools, grouped by section (Theia, Spectrogram, Tartarus, Elysium): start,
    stop and status.  Every tool is its own program; the home page starts it
    as a separate process with the experiment fixed (ASPHODEL_EXPERIMENT), so
    a running tool stays on its experiment if the home page switches.
  * data files of the experiment, with "open with" buttons
  * outputs: scan runs, confirmed identifications, decay fits, figures
  * the shared physics settings (physics.json) and the overrides of each tool
  * notes per experiment
  * "Sync with APPC256": sends new confirmed lists to APPC256 and pulls the
    summed spectrograms you pick from it (core/sync.py)

Usage
-----
    python home.py            (opens the browser)

The Quit button (top right) closes all running tools and the home page, after
asking. Ctrl-C in the terminal stops only the home page; the tools keep running.
"""

import datetime
import glob
import itertools
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import webbrowser

sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
from core import browser, icon, paths, physics, sync, tabwatch, version

import dash
from dash import dcc, html, Input, Output, State, ALL, ctx

REPO = paths.REPO_ROOT
ACCENT, PALE, SIDE, LINE = "#f57d15", "#fff4ec", "#f9f9f9", "#dddddd"
HOME_PORT = 8050
THEIA_PORT = 8052

# ===========================================================================
# Tools
# ===========================================================================
# key: (section, name, script relative to the repo, description, single instance?)
TOOLS = {
    "theia":           ("Theia", "Theia", "theia/theia.py",
                        "Viewer (one 1-D spectrum with simulated ion lines), batch scanner, "
                        "scan results and further output.", True),
    "spectrogram":     ("Spectrogram", "Spectrogram", "spectrogram/plot_spectrogram.py",
                        "2-D / 3-D time × frequency view, simulated lines, region selection "
                        "for decay fits.", False),
    # no card: started from the spectrogram, or re-run from the Decay fits tab
    "decay_fit":       (None, "Decay fit", "spectrogram/decay_fit.py",
                        "Exponential / Bateman / joint fits and half-lives.", False),
    "sum_spectra":     ("Tartarus", "Sum spectra", "tartarus/sum_spectra.py",
                        "Sum all 1-D spectra of a folder (raw and drift-aligned).", False),
    "correction_plot": ("Tartarus", "2nd-order correction", "tartarus/new_2nd_order_correction_plot.py",
                        "f_diff vs f_sim from a RionID table and the 2nd-order polynomial fit.", False),
    "intensity_fit":   ("Tartarus", "Intensity variation fit", "tartarus/intensity_variation_fit.py",
                        "Linear fit of σ vs intensity from a spreadsheet.", False),
    "levelschemer":    ("Elysium", "LevelSchemer", "elysium/levelschemer.py",
                        "Draw nuclear level schemes (Tk window).", False),
}
SECTIONS = ["Theia", "Spectrogram", "Tartarus", "Elysium"]
# sections shown side by side on one row (each takes as many columns as it has cards)
SECTION_ROWS = [["Theia", "Spectrogram", "Elysium"], ["Tartarus", "Reference"]]

# NUBASE2020 paper and its page index (core/build_nubase_index.py)
NUBASE_PDF = os.path.join(REPO, "core", "nubase2020.pdf")
NUBASE_PAGES = os.path.join(REPO, "core", "nubase2020_pages.json")
NUBASE_URL = "/nubase2020.pdf"

_PROCS = {}                 # key -> list of (Popen, experiment)
_PROCS_LOCK = threading.Lock()


def _port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        return s.connect_ex(("127.0.0.1", port)) == 0


def running(key):
    """[(pid, experiment)] of the live processes of a tool started from here."""
    with _PROCS_LOCK:
        live = [(p, e) for p, e in _PROCS.get(key, []) if p.poll() is None]
        _PROCS[key] = live
    return [(p.pid, e) for p, e in live]


def start_tool(key, args=()):
    """Start a tool as its own process, fixed to the active experiment."""
    section, name, script, _, single = TOOLS[key]
    path = os.path.join(REPO, script)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{script} not found")
    if key == "theia" and _port_open(THEIA_PORT):
        browser.open_url(f"http://127.0.0.1:{THEIA_PORT}/")
        return f"Theia is already running – opened it in the browser."
    if single and running(key):
        return f"{name} is already running."
    exp = paths.experiment()
    env = dict(os.environ, **{paths.ENV_EXPERIMENT: exp})
    # own process group, so Stop/Quit also end what the tool starts itself
    # (e.g. decay fits opened from the spectrogram), and Ctrl-C in the home
    # page's terminal does not reach the tools
    proc = subprocess.Popen([sys.executable, path, *[str(a) for a in args]], cwd=REPO, env=env,
                            start_new_session=True)
    with _PROCS_LOCK:
        _PROCS.setdefault(key, []).append((proc, exp))
    shown = " ".join(shlex.quote(str(a)) for a in args)
    return f"Started {name} ({exp})" + (f": {shown}" if shown else "") + "."


def _terminate(proc):
    """SIGTERM to the tool's whole process group (the tool and anything it started)."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            proc.terminate()
        except ProcessLookupError:
            pass


def running_summary():
    """Running tools as text lines, for the Quit confirmation."""
    lines = []
    for key, t in TOOLS.items():
        n = len(running(key))
        if key == "theia":
            if _port_open(THEIA_PORT):
                lines.append("Theia" + ("" if n else " (started elsewhere)"))
        elif n:
            lines.append(f"{n} × {t[1]}" if n > 1 else t[1])
    if sync.JOB.state.get("running"):
        lines.append("Pull from APPC256 (cancelled; the partial file is removed)")
    return lines


def stop_everything():
    """Stop every tool started from here, and Theia even if started elsewhere."""
    if sync.JOB.cancel():
        for _ in range(50):                    # let it remove the partial file
            if not sync.JOB.state.get("running"):
                break
            time.sleep(0.1)
    with _PROCS_LOCK:
        procs = [p for lst in _PROCS.values() for p, _ in lst if p.poll() is None]
    for p in procs:
        _terminate(p)
    if _port_open(THEIA_PORT) and not running("theia"):
        try:
            import urllib.request
            urllib.request.urlopen(urllib.request.Request(
                f"http://127.0.0.1:{THEIA_PORT}/_theia_quit", method="POST"), timeout=2)
        except Exception:
            pass
    return len(procs)


def stop_tool(key):
    name = TOOLS[key][1]
    with _PROCS_LOCK:
        procs = [p for p, _ in _PROCS.get(key, []) if p.poll() is None]
    if not procs:
        if key == "theia" and _port_open(THEIA_PORT):
            return "Theia was not started from here – use its Quit button."
        return f"{name} is not running."
    for p in procs:
        _terminate(p)
    return f"Stopped {name} ({len(procs)} process{'es' if len(procs) > 1 else ''})."


def theia_status():
    if _port_open(THEIA_PORT):
        own = running("theia")
        return True, (f"running ({own[0][1]}) · 127.0.0.1:{THEIA_PORT}" if own
                      else f"running · 127.0.0.1:{THEIA_PORT} (started elsewhere)")
    return False, "stopped"


def tool_status(key):
    if key == "theia":
        return theia_status()
    live = running(key)
    if not live:
        return False, "not running"
    exps = sorted({e for _, e in live})
    return True, f"{len(live)} running ({', '.join(exps)})"


# ===========================================================================
# Settings files of the tools (for "open with" and the overrides list)
# ===========================================================================

def _read_json(path):
    try:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _update_json(path, changes):
    d = _read_json(path)
    d.update(changes)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=2)
    os.replace(tmp, path)


def tool_overrides():
    """{tool label: {key: value}} physics values a tool keeps for itself."""
    out = {}
    for label, fname, sub in (("Theia scanner", "scanner_config.json", None),
                              ("Theia viewer", "viewer_config.json", None),
                              ("Spectrogram", "spectrogram_prefs.json", "sim")):
        d = _read_json(paths.settings_file(fname))
        if sub:
            d = d.get(sub) or {}
        shared = physics.load_shared()
        ov = {k: v for k, v in d.items()
              if k in physics.PHYSICS_KEYS and not physics._same(k, v, shared[k])}
        if ov:
            out[label] = ov
    return out


# ===========================================================================
# Data and outputs of the experiment
# ===========================================================================

def _size(n):
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if n < 1000 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000


def _mtime(path):
    try:
        return datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime("%d %b %Y %H:%M")
    except OSError:
        return ""


def folder_summary(d, pattern="*.npz", subfolders=False):
    files = glob.glob(os.path.join(d, pattern))
    if subfolders:
        files += glob.glob(os.path.join(d, "*", pattern))
    total = 0
    for f in files:
        try:
            total += os.path.getsize(f)
        except OSError:
            pass
    return len(files), total


def data_rows():
    """Rows of the data-file table: the 1-D spectra folder as one row, every
    spectrogram and summed 1-D file on its own."""
    rows = []
    d1 = paths.spectra_1d()
    n, total = folder_summary(d1)
    rows.append(dict(name=f"spectra_1d/  ({n} files)", kind="1-D folder", size=total,
                     mtime=_mtime(d1), path=d1))
    for kind, d in (("spectrogram", paths.spectrograms()), ("summed 1-D", paths.summed_1d())):
        files = sorted(glob.glob(os.path.join(d, "*.npz")))
        if kind == "spectrogram":          # also one subfolder down (as synced from APPC256)
            files += sorted(glob.glob(os.path.join(d, "*", "*.npz")))
        for f in files:
            rows.append(dict(name=os.path.relpath(f, d), kind=kind, size=os.path.getsize(f),
                             mtime=_mtime(f), path=f))
    return rows


def scan_runs():
    path = os.path.join(paths.outputs("scans"), "index.csv")
    try:
        import csv
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
    except OSError:
        return []
    return rows[::-1]                                  # newest first


def confirmed_files():
    out = []
    for f in sorted(glob.glob(os.path.join(paths.outputs("confirmed"), "*.csv")),
                    key=os.path.getmtime, reverse=True):
        head = {}
        try:
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    if not line.startswith("# "):
                        break
                    k, _, v = line[2:].partition(": ")
                    head[k.strip()] = v.strip()
        except OSError:
            continue
        out.append(dict(path=f, name=os.path.basename(f), saved=head.get("saved", ""),
                        accepted=head.get("accepted", ""), unsure=head.get("unsure", ""),
                        filter=head.get("filter", ""),
                        scan=os.path.basename(head.get("database", ""))))
    return out


def fit_records():
    out = []
    for f in sorted(glob.glob(os.path.join(paths.outputs("fits"), "*.json")), reverse=True):
        d = _read_json(f)
        if d.get("format") != "asphodel-decay-fit":
            continue
        d["_path"] = f
        out.append(d)
    return out


def figure_files():
    d = paths.outputs("figures")
    return [f for f in sorted(glob.glob(os.path.join(d, "*")), key=os.path.getmtime, reverse=True)
            if os.path.isfile(f)]


def fmt_time(s, err=None):
    """Seconds -> value ± error in a readable unit."""
    if s is None:
        return "–"
    for unit, f in (("y", 3.15576e7), ("d", 86400), ("h", 3600), ("min", 60), ("s", 1),
                    ("ms", 1e-3), ("µs", 1e-6)):
        if abs(s) >= f or unit == "µs":
            v = s / f
            e = (err / f) if err is not None else None
            return f"{v:.4g} ± {e:.2g} {unit}" if e is not None else f"{v:.4g} {unit}"


def fit_rerun_args(rec):
    """decay_fit.py command-line arguments that redo a saved fit."""
    st = rec.get("settings") or {}
    w = st.get("freq_window_hz") or [None, None]
    args = [rec["file"], repr(w[0]), repr(w[1])]
    if st.get("parent_window_hz"):
        args += ["--parent-window", *map(repr, st["parent_window_hz"])]
    elif st.get("bateman"):
        args.append("--bateman")
    if st.get("with_offset") is False and not st.get("bateman"):
        args.append("--no-offset")
    for key, opt in (("tmin_s", "--tmin"), ("tmax_s", "--tmax"), ("gamma", "--gamma"),
                     ("halflife_isomer_com_s", "--halflife-isomer"),
                     ("halflife_groundstate_com_s", "--halflife-groundstate"),
                     ("time_avg", "--time-avg")):
        if st.get(key) is not None:
            args += [opt, repr(st[key])]
    if st.get("skip_bins") and any(st["skip_bins"]):
        args += ["--skip", *map(str, st["skip_bins"])]
    if st.get("err_coeffs"):
        args += ["--err-coeffs", *map(repr, st["err_coeffs"])]
    if st.get("unweighted"):
        args.append("--unweighted")
    bl = st.get("baseline")
    if bl and bl.get("params") and bl.get("window_hz"):
        args += ["--baseline", json.dumps(bl["params"]),
                 "--baseline-window", *map(repr, bl["window_hz"])]
    if st.get("title_full"):
        args += ["--title-full", st["title_full"]]
    elif st.get("title"):
        args += ["--title", st["title"]]
    elif rec.get("label"):                     # records saved before title/title_full were stored
        args += ["--title", rec["label"]]
    return args


def open_in_system(path):
    """Open a file with the desktop's default program (figures, PDFs)."""
    subprocess.Popen(["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ===========================================================================
# Layout helpers
# ===========================================================================

_ACT_IDS = itertools.count()


def act(action, payload="", label="", cls="btn small", title=None):
    """Button whose click is handled by the one 'actions' callback."""
    return html.Button(label, id={"type": "act", "a": action, "p": str(payload),
                                  "u": next(_ACT_IDS)},
                       className=cls, n_clicks=0, title=title)


def table(header, rows):
    return html.Table([html.Thead(html.Tr([html.Th(h) for h in header])),
                       html.Tbody([html.Tr([html.Td(c) for c in r]) for r in rows])])


def section(title, *children, extra=None):
    return html.Section([html.H2([title, extra] if extra else title), *children])


def fold_section(title, *children):
    """A section whose heading opens/closes it; closed when the page loads."""
    return html.Section(html.Details([html.Summary(title), *children],
                                     open=False, className="fold main-fold"))


def _status_children(on, text):
    return [html.Span(className="dot on" if on else "dot"), html.Span(text)]


def tool_card(key):
    section_, name, script, desc, _ = TOOLS[key]
    on, text = tool_status(key)
    buttons = {
        "theia": [act("start", "theia", "Open", "btn primary small"),
                  act("stop", "theia", "Stop")],
        "spectrogram": [act("start", "spectrogram", "Open", "btn primary small",
                            title="Reopens the last session, or the file chooser"),
                        act("stop", "spectrogram", "Stop all")],
        "levelschemer": [act("start", "levelschemer", "Open", "btn primary small",
                             title="Reopens the last scheme"),
                         act("stop", "levelschemer", "Stop all")],
    }.get(key, [])
    return html.Div([
        html.Div(name, className="name"),
        html.Div(script, className="from"),
        html.P(desc),
        html.Div(_status_children(on, text), id={"type": "status", "k": key}, className="status"),
        html.Div(buttons, className="btns"),
    ], className="tool")


def _opts(files, base=None):
    return [{"label": os.path.relpath(f, base) if base else os.path.basename(f), "value": f}
            for f in files]


def tool_forms():
    """Small forms for the command-line tools."""
    exp_dir = paths.experiment_dir()
    xlsx = sorted(glob.glob(os.path.join(paths.rionid(), "*.xls*")))
    sheets = sorted(f for ext in ("*.ods", "*.xlsx", "*.xls", "*.csv")
                    for f in glob.glob(os.path.join(exp_dir, ext)))
    schemes = sorted(glob.glob(os.path.join(paths.levelschemes(), "*.json")))
    ods_default = os.path.join(exp_dir, "Schottky_Uncertainties.ods")
    lbl = lambda t: html.Label(t, className="flabel")
    return {
        "sum_spectra": html.Div([
            lbl("Input folder"), dcc.Input(id="ss-in", type="text", value=paths.spectra_1d(), className="wide"),
            lbl("Output folder"), dcc.Input(id="ss-out", type="text", value=paths.summed_1d(), className="wide"),
            html.Button("Run", id="ss-run", className="btn primary small", n_clicks=0),
        ], className="form"),
        "correction_plot": html.Div([
            lbl("RionID table"),
            dcc.Dropdown(id="cp-file", options=_opts(xlsx), clearable=False,
                         value=xlsx[0] if xlsx else None),
            lbl("Sheet"), dcc.Input(id="cp-sheet", type="text", value="New2ndOrderCorrection", className="wide"),
            html.Button("Run", id="cp-run", className="btn primary small", n_clicks=0),
        ], className="form"),
        "intensity_fit": html.Div([
            lbl("Spreadsheet"),
            dcc.Dropdown(id="if-file", options=_opts(sheets), clearable=False,
                         value=ods_default if ods_default in sheets else (sheets[0] if sheets else None)),
            lbl("Exclude ions (space separated, optional)"),
            dcc.Input(id="if-exclude", type="text", className="wide", placeholder="e.g. 206Pb79+ 208Pb80+"),
            html.Button("Run", id="if-run", className="btn primary small", n_clicks=0),
        ], className="form"),
        "levelschemer": html.Div([
            lbl("Open a scheme"),
            dcc.Dropdown(id="ls-file", options=_opts(schemes), clearable=False,
                         value=schemes[0] if schemes else None),
            html.Button("Open scheme", id="ls-run", className="btn small", n_clicks=0),
        ], className="form"),
    }


def nubase_card():
    """'View NUBASE': open the NUBASE2020 paper in a tab, or at a nuclide's page."""
    pages = _read_json(NUBASE_PAGES) or {}
    have_pdf = os.path.isfile(NUBASE_PDF)
    if not have_pdf:
        body = [html.Div("core/nubase2020.pdf not found", className="note")]
    else:
        body = [
            html.Div(html.A("Open NUBASE", href=NUBASE_URL, target="_blank",
                            className="btn primary small",
                            title="NUBASE2020 (Kondev et al., Chin. Phys. C 45, 030001) in a new tab"),
                     className="btns"),
            html.Div([
                html.Label("Go to a nuclide (e.g. 198Ir, Ir198, Ir-198)", className="flabel"),
                dcc.Input(id="nub-q", type="text", list="nub-names", debounce=False,
                          className="wide", placeholder="198Ir", autoComplete="off", n_submit=0),
                html.Datalist(id="nub-names", children=[html.Option(value=k) for k in pages]),
                html.Div([html.A("Open at page", id="nub-go", href=NUBASE_URL, target="asphodel-nubase",
                                 className="btn small"),
                          html.Span(id="nub-msg", className="note")], className="below"),
            ], className="form"),
            dcc.Store(id="nub-index", data=pages),
        ]
    return html.Div([
        html.Div("View NUBASE", className="name"),
        html.Div("core/nubase2020.pdf", className="from"),
        html.P("The NUBASE2020 evaluation: table of nuclides, isomers, half-lives, "
               "spins and decay modes."),
        *body,
    ], className="tool")


def tools_section():
    forms = tool_forms()
    rows = []
    for row in SECTION_ROWS:
        blocks, ncols = [], 0
        for sec in row:
            cards = []
            for key, t in TOOLS.items():
                if t[0] != sec:
                    continue
                card = tool_card(key)
                if key in forms:
                    card.children.append(forms[key])
                cards.append(card)
            if sec == "Reference":
                cards.append(nubase_card())
            n = max(len(cards), 1)
            ncols += n
            blocks.append(html.Div([html.H3(sec, className="sec"),
                                    html.Div(cards, className="tools",
                                             style={"gridTemplateColumns": f"repeat({n}, 1fr)"})],
                                   className="tsec", style={"gridColumn": f"span {n}"}))
        rows.append(html.Div(blocks, className="trow",
                             style={"gridTemplateColumns": f"repeat({ncols}, minmax(0, 1fr))"}))
    return section("Tools", *rows)


def data_section():
    rows = []
    for r in data_rows():
        if r["kind"] == "1-D folder":
            btns = [act("viewer_dir", r["path"], "Viewer"), act("scan_dir", r["path"], "Scan in Theia")]
        elif r["kind"] == "spectrogram":
            btns = [act("spectrogram_file", r["path"], "Spectrogram")]
        else:
            btns = [act("viewer_file", r["path"], "Viewer")]
        rows.append([html.Span(r["name"], className="mono"), r["kind"], _size(r["size"]),
                     r["mtime"], html.Span(btns)])
    return fold_section("Data files",
                   html.Div(paths.experiment_dir(), className="note mono"),
                   table(["File", "Type", "Size", "Modified", "Open with"], rows))


def outputs_section():
    # scan runs
    runs = scan_runs()
    keys = [k for k in ("file", "created", "label", "status", "n_files", "n_spectra_done",
                        "n_identifications", "settings_hash") if runs and k in runs[0]]
    scans = table(keys, [[r.get(k, "") for k in keys] for r in runs]) if runs else \
        html.P("No scan runs yet.", className="note")
    # confirmed
    conf = confirmed_files()
    confirmed = table(["File", "Saved", "Accepted", "Unsure", "Filter", "From scan"],
                      [[html.Span(c["name"], className="mono"), c["saved"], c["accepted"], c["unsure"],
                        c["filter"], c["scan"]] for c in conf]) if conf else \
        html.P("No confirmed identifications yet.", className="note")
    # fits
    recs = fit_records()
    fit_rows = []
    for r in recs:
        res = r.get("result") or {}
        hls = res.get("half_lives") or {}
        hl = hls.get("decay") or hls.get("A") or {}
        st = r.get("settings") or {}
        w = st.get("freq_window_hz") or [None, None]
        chi = res.get("chi2_ndf")
        fit_rows.append([
            (r.get("saved") or "").replace("T", " "), r.get("label") or "–", r.get("mode", ""),
            fmt_time(hl.get("lab_s"), hl.get("lab_err_s")),
            fmt_time(hl.get("com_s"), hl.get("com_err_s")) if "com_s" in hl else "–",
            html.Span(os.path.basename(r.get("file", "")), className="mono"),
            f"{w[0]:,.0f} – {w[1]:,.0f} Hz" if None not in w else "",
            f"{chi:.3g}" if chi is not None else "–",
            html.Span([act("rerun_fit", r["_path"], "Re-run"),
                       act("fit_spectrogram", r["_path"], "Spectrogram")])])
    fits = table(["Saved", "Label", "Model", "T½ (lab)", "T½ (c.o.m.)", "File", "Window", "χ²/ndf", ""],
                 fit_rows) if fit_rows else \
        html.P("No saved fits yet – use “Save fit parameters” in a decay-fit window.", className="note")
    # figures
    figs = figure_files()
    figures = table(["Figure", "Modified", ""],
                    [[html.Span(os.path.basename(f), className="mono"), _mtime(f),
                      act("open_file", f, "Open")] for f in figs]) if figs else \
        html.P("No figures yet.", className="note")
    return fold_section("Outputs", dcc.Tabs([
        dcc.Tab(label=f"Scan runs ({len(runs)})", children=[scans,
                html.Div(act("start", "theia", "Open Theia (Scan Results)"), className="below")]),
        dcc.Tab(label=f"Confirmed IDs ({len(conf)})", children=[confirmed]),
        dcc.Tab(label=f"Decay fits ({len(recs)})", children=[fits]),
        dcc.Tab(label=f"Figures ({len(figs)})", children=[figures]),
    ], className="otabs"))


# --------------------------------------------------------------------------- sidebar

PHYS_LABELS = {
    "ref_name": "Reference ion", "ref_frequency": "f_ref (Hz)", "ref_harmonic": "Ref. harmonic",
    "gamma_t": "γ_t", "harmonics": "Harmonics", "z_lims": "Z min, max", "n_lims": "N min, max",
    "el1": "el1 (A, Z)", "el2": "el2 (A, Z)", "mq_width": "m/q window (±)",
    "correction_enabled": "2nd-order correction", "corr_a": "a", "corr_b": "b", "corr_c": "c",
    "min_half_life": "Min T½", "min_half_life_unit": "T½ unit",
    "exclude_unknown_hl": "Exclude unknown T½",
}


def _phys_text(k, v):
    if isinstance(v, list):
        return ", ".join(str(x) for x in v)
    if isinstance(v, float):
        return physics._fmt_num(v)
    return str(v)


def physics_panel():
    sh = physics.load_shared()
    fields = []
    for k in physics.PHYSICS_KEYS:
        if k in ("correction_enabled", "exclude_unknown_hl"):
            inp = dcc.Checklist(id={"type": "phys", "k": k}, options=[{"label": "", "value": "on"}],
                                value=["on"] if sh[k] else [], className="chk")
        elif k == "min_half_life_unit":
            inp = dcc.Dropdown(id={"type": "phys", "k": k}, options=physics.HL_CUT_UNITS,
                               value=sh[k], clearable=False, className="unit")
        else:
            inp = dcc.Input(id={"type": "phys", "k": k}, type="text", value=_phys_text(k, sh[k]),
                            debounce=True)
        fields += [html.Label(PHYS_LABELS.get(k, k)), inp]
    ov = tool_overrides()
    ov_lines = ([html.Div([html.B(t + ": "), ", ".join(f"{k} = {_phys_text(k, v)}" for k, v in d.items())])
                 for t, d in ov.items()] or [html.Div("none")])
    return html.Div([
        html.Div(fields, className="kv"),
        html.Div([html.Button("Save physics", id="phys-save", className="btn primary small", n_clicks=0),
                  html.Span(id="phys-msg", className="note")], className="below"),
        html.Div([html.Div("Tool overrides (tool settings that differ from these):", className="note"),
                  html.Div(ov_lines, className="note")], className="below"),
        html.Div("Tools read these when they start (Theia: on page reload).", className="note"),
    ])


def changelog_panel(n=20):
    """The last n commits (git log), newest first, with their version."""
    entries = version.changelog(n)
    if not entries:
        rows = [html.Div("git log not available", className="note")]
    else:
        rows = [html.Div([
            html.Span(f"v{e['version']}" if e["version"] else "", className="cl-ver"),
            html.Span(e["date"], className="cl-date"),
            html.Div(e["subject"], className="cl-msg"),
        ], className="cl-row") for e in entries]
    return html.Div([html.H3("Changelog"), html.Div(rows, className="cl-list")],
                    className="changelog")


def sidebar():
    srcs = []
    for title, d in (("1-D spectra", paths.spectra_1d()), ("Spectrograms", paths.spectrograms()),
                     ("Summed 1-D", paths.summed_1d())):
        n, total = folder_summary(d, subfolders=(title == "Spectrograms"))
        srcs.append(html.Div([html.B(title), html.Code(os.path.relpath(d, paths.experiment_dir())),
                              html.Div(f"{n} .npz · {_size(total)}")], className="src"))
    nub = os.path.join(paths.REPO_ROOT, "core", "nubase_4.mas20")
    return html.Aside([
        html.Div([
            html.H3("Data sources"), *srcs,
            html.Details([html.Summary("Shared physics"), physics_panel()],
                         open=False, className="fold"),
            html.H3("Reference data"),
            html.Div([html.B("NUBASE"), html.Code("core/nubase_4.mas20"),
                      html.Div(_mtime(nub))], className="src"),
        ], className="side-top"),
        # suite icon and version, in the free space at the bottom of the sidebar
        # (version: core/version.py, goes up by one with every commit)
        html.Div([
            html.Img(src=icon.ICON_URL, alt="Asphodel"),
            html.Div(f"Asphodel v{version.version()}", className="brand-name"),
            changelog_panel(),
        ], className="brand"),
    ])


def notes_section():
    try:
        with open(paths.notes_file(), encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        text = ""
    return section("Notes", dcc.Textarea(id="notes", value=text, className="notes",
                                         placeholder="Notes for this experiment…"),
                   html.Div([html.Button("Save notes", id="notes-save", className="btn small", n_clicks=0),
                             html.Span(id="notes-msg", className="note")], className="below"))


def header():
    exps = paths.list_experiments()
    cur = paths.experiment()
    return html.Header([
        html.Div(["Asphodel", html.Span(" ·")], className="logo"),
        dcc.Dropdown(id="exp-select", options=exps, value=cur, clearable=False, className="exp"),
        html.Button("+ New experiment…", id="exp-new-open", className="btn small", n_clicks=0),
        html.Span(paths.experiment_dir(), className="path"),
        html.Span(className="spacer"),
        html.Button("⇅ Sync with APPC256", id="sync-open", className="btn small", n_clicks=0,
                    title="Send new confirmed lists to APPC256, pull new summed spectrograms"),
        html.Button("↻ Refresh", id="refresh", className="btn small", n_clicks=0,
                    title="Re-read files and outputs"),
        html.Button("⏻ Quit", id="quit", className="btn small quit", n_clicks=0,
                    title="Close all tools and the home page (asks first)"),
        dcc.Store(id="quit-info"),
        html.Div(id="quit-done", style={"display": "none"}),
    ])


def new_experiment_modal():
    exps = paths.list_experiments()
    return html.Div(id="exp-modal", className="modal", style={"display": "none"}, children=html.Div([
        html.H2("New experiment"),
        html.Label("Name", className="flabel"),
        dcc.Input(id="exp-name", type="text", placeholder="e.g. E0204", className="wide"),
        html.Label("Data folder (empty = data/<name> in the Asphodel folder)", className="flabel"),
        dcc.Input(id="exp-dir", type="text", className="wide", placeholder=paths.DATA_ROOT + "/<name>"),
        html.Label("Copy the shared physics settings from", className="flabel"),
        dcc.Dropdown(id="exp-copy", options=[{"label": "(built-in defaults)", "value": ""}] + exps,
                     value=paths.experiment(), clearable=False),
        html.Div([html.Button("Create", id="exp-create", className="btn primary small", n_clicks=0),
                  html.Button("Cancel", id="exp-cancel", className="btn small", n_clicks=0),
                  html.Span(id="exp-msg", className="note")], className="below"),
    ], className="modal-box"))


def sync_modal():
    try:
        cfg = sync.settings()
        where = f"{cfg['host']}:{cfg['remote_root']}"
    except sync.SyncError as exc:
        where = f"⚠ {exc}"
    hidden = {"display": "none"}
    return html.Div(id="sync-modal", className="modal", style=hidden, children=html.Div([
        html.H2("Sync with APPC256"),
        html.Div([html.Span(where, className="mono"),
                  " (change in settings/sync.json of the experiment)"], className="note"),
        html.Div("Sends confirmed lists that APPC256 doesn't have yet; lists the summed "
                 "spectrograms this computer doesn't have, to pull the ones you tick. "
                 "Nothing is overwritten or deleted on either side.", className="note below"),
        html.Label("lxlogin password (not stored; leave empty if a connection to appc256 is "
                   "already open, e.g. an ssh session from the last 4 h)", className="flabel"),
        dcc.Input(id="sync-pw", type="password", className="wide", autoComplete="off"),
        html.Div([html.Button("Check and send confirmed lists", id="sync-check",
                              className="btn primary small", n_clicks=0),
                  html.Button("Close", id="sync-close", className="btn small", n_clicks=0)],
                 className="below"),
        dcc.Loading(html.Div(id="sync-result", className="below"), type="dot", color=ACCENT),
        html.Div(id="sync-pull-box", style=hidden, className="below", children=[
            html.Label("Summed spectrograms on APPC256 that are not here (subfolders are kept)",
                       className="flabel"),
            dcc.Checklist(id="sync-pull-list", options=[], value=[], className="pull-list"),
            html.Div(id="sync-space", className="note below"),
            html.Div([html.Button("Pull selected", id="sync-pull", className="btn primary small", n_clicks=0),
                      html.Button("Cancel pull", id="sync-cancel", className="btn small", n_clicks=0),
                      html.Span(id="sync-msg", className="note")], className="below"),
        ]),
        html.Div(id="sync-progress", className="below"),
        dcc.Store(id="sync-data"),
    ], className="modal-box wide-box"))


def serve_layout():
    return html.Div([
        dcc.Location(id="loc", refresh=True),
        dcc.Interval(id="tick", interval=2000),
        header(),
        html.Div(id="msg", className="banner", children="Ready."),
        html.Div([
            sidebar(),
            html.Main([tools_section(), data_section(), outputs_section(), notes_section()]),
        ], className="wrap"),
        new_experiment_modal(),
        sync_modal(),
    ], className="page")


# ===========================================================================
# App
# ===========================================================================

CSS = f"""
:root{{--accent:{ACCENT};--pale:{PALE};--side:{SIDE};--line:{LINE};--muted:#777;--ok:#2e7d32}}
*{{box-sizing:border-box}}
body{{margin:0;font-family:-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;color:#222;background:#fff;font-size:14px}}
header{{height:52px;display:flex;align-items:center;gap:14px;padding:0 18px;border-bottom:3px solid var(--accent)}}
header .logo{{font-weight:800;font-size:18px;letter-spacing:.5px}} header .logo span{{color:var(--accent)}}
header .exp{{width:180px;font-size:13px}} header .path{{color:var(--muted);font-family:monospace;font-size:12px}}
header .spacer{{flex:1}}
.banner{{background:var(--pale);border-bottom:1px solid var(--line);padding:6px 18px;font-size:12px;color:#8a4a10;min-height:28px}}
/* page = header + banner + (sidebar | main) filling the window; the main
   section scrolls on its own, the sidebar stays put */
body{{overflow:hidden}}
.page{{height:100vh;display:flex;flex-direction:column}}
.page>header,.page>.banner{{flex:none}}
.wrap{{display:flex;flex:1 1 auto;min-height:0}}
aside{{width:330px;min-width:330px;background:var(--side);border-right:1px solid var(--line);padding:14px 16px;
  display:flex;flex-direction:column;overflow:hidden}}
/* sidebar split in two equal halves: top = data sources, physics, reference
   data (scrolls if too long); bottom = icon, version and changelog (fixed) */
.side-top{{flex:1 1 0;min-height:0;overflow-y:auto}}
.brand{{text-align:center;flex:1 1 0;min-height:0;display:flex;flex-direction:column;
  border-top:1px solid var(--line);margin-top:10px;padding-top:14px}}
.brand img{{width:120px;height:120px;display:block;margin:0 auto 8px;flex:none}}
.brand-name{{font-weight:700;font-size:14px;color:#555;letter-spacing:.3px}}
/* changelog fills the rest of the sidebar; its bottom is the window's bottom */
.changelog{{text-align:left;margin-top:12px;flex:1 1 auto;min-height:0;display:flex;flex-direction:column}}
.cl-list{{flex:1 1 auto;min-height:0;overflow-y:auto;background:#fff;border:1px solid var(--line);border-radius:5px;padding:2px 8px}}
.cl-row{{padding:5px 0;border-bottom:1px solid var(--line);font-size:11px}}
.cl-row:last-child{{border-bottom:none}}
.cl-ver{{font-weight:700;color:var(--accent);margin-right:6px;font-family:monospace}}
.cl-date{{color:var(--muted)}}
.cl-msg{{color:#333;margin-top:1px;word-break:break-word}}
aside h3{{font-size:12px;text-transform:uppercase;letter-spacing:.8px;color:var(--muted);margin:18px 0 8px}}
aside h3:first-child{{margin-top:0}}
details.fold{{margin:18px 0 8px}}
details.fold>summary{{font-size:12px;text-transform:uppercase;letter-spacing:.8px;color:var(--muted);font-weight:700;cursor:pointer;list-style:none;margin-bottom:8px}}
details.fold>summary::-webkit-details-marker{{display:none}}
details.fold>summary::before{{content:"\\25B8";display:inline-block;width:12px;color:var(--accent);transition:transform .15s}}
details.fold[open]>summary::before{{transform:rotate(90deg)}}
details.fold>summary:hover{{color:var(--accent)}}
details.main-fold{{margin:0}}
details.main-fold>summary{{font-size:15px;text-transform:none;letter-spacing:0;color:#222;margin:0 0 10px;padding-bottom:5px;border-bottom:1px solid var(--line)}}
.src{{background:#fff;border:1px solid var(--line);border-radius:5px;padding:7px 9px;margin-bottom:6px;font-size:12px}}
.src b{{display:block;font-size:13px}} .src code{{color:var(--muted);font-size:11px;word-break:break-all;display:block}}
.kv{{display:grid;grid-template-columns:auto 1fr;gap:4px 8px;font-size:12px;align-items:center}}
.kv label{{color:var(--muted)}} .kv input{{width:100%;font-family:monospace;font-size:12px;padding:2px 5px;border:1px solid var(--line);border-radius:3px}}
.kv .unit{{font-size:12px}}
.btn{{display:inline-block;border:1px solid var(--line);background:#fff;border-radius:4px;padding:5px 10px;font-size:12px;cursor:pointer;color:#222;margin:0 6px 4px 0}}
.btn:hover{{border-color:var(--accent)}}
.btn.primary{{background:var(--accent);border-color:var(--accent);color:#fff;font-weight:600}}
.btn.small{{padding:3px 8px;font-size:11px}}
main{{flex:1;padding:16px 20px;min-width:0;overflow-y:auto}}
h2{{font-size:15px;margin:0 0 10px;padding-bottom:5px;border-bottom:1px solid var(--line)}}
h3.sec{{font-size:12px;text-transform:uppercase;letter-spacing:.8px;color:var(--accent);margin:4px 0 6px}}
.trow{{display:grid;gap:12px;margin-bottom:10px}}
.tsec{{display:flex;flex-direction:column;min-width:0}}
.tsec .tools{{flex:1}}
section{{margin-bottom:24px}}
.tools{{display:grid;gap:12px}}
.tool{{border:1px solid var(--line);border-radius:6px;padding:12px;background:#fff;border-top:3px solid var(--accent)}}
.tool .name{{font-weight:700;font-size:15px}} .tool .from{{font-size:11px;color:var(--muted);font-family:monospace}}
.tool p{{font-size:12px;color:#555;margin:6px 0 8px}}
.status{{font-size:11px;margin-bottom:8px}}
.dot{{display:inline-block;width:8px;height:8px;border-radius:50%;background:#bbb;margin-right:5px}} .dot.on{{background:var(--ok)}}
.form{{border-top:1px dashed var(--line);margin-top:8px;padding-top:8px;font-size:12px}}
.flabel{{display:block;color:var(--muted);font-size:11px;margin:6px 0 2px}}
.form input,.wide{{width:100%;font-size:12px;padding:3px 6px;border:1px solid var(--line);border-radius:3px}}
.form .btn{{margin-top:8px}}
.row2{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
th{{background:var(--accent);color:#fff;text-align:left;padding:6px 8px;font-weight:600}}
td{{padding:5px 8px;border-bottom:1px solid var(--line);white-space:nowrap}}
tbody tr:nth-child(even) td{{background:var(--pale)}}
.mono{{font-family:monospace}}
.note{{font-size:11px;color:var(--muted)}}
.below{{margin-top:8px}}
.otabs{{margin-bottom:8px}}
.notes{{width:100%;min-height:140px;font-family:monospace;font-size:12px;border:1px solid var(--line);border-radius:4px;padding:8px}}
.modal{{position:fixed;inset:0;background:rgba(0,0,0,.35);z-index:1000;align-items:center;justify-content:center}}
.modal-box{{background:#fff;border-radius:8px;padding:18px 22px;width:520px;border-top:4px solid var(--accent)}}
.chk input{{width:auto}}
@media (max-width:1100px){{.trow,.tools{{grid-template-columns:1fr !important}} .tsec{{grid-column:auto !important}}}}
.modal-box.wide-box{{width:640px;max-height:90vh;overflow:auto}}
.pull-list label{{display:flex;align-items:center;gap:6px;font-family:monospace;font-size:12px;padding:2px 0}}
.pbar{{height:8px;background:var(--line);border-radius:4px;overflow:hidden;margin:4px 0}}
.pbar div{{height:100%;background:var(--accent)}}
a.btn{{text-decoration:none}}
.btn.quit{{color:#c62828;border-color:#c62828}} .btn.quit:hover{{background:#c62828;color:#fff}}
"""

app = dash.Dash(__name__, title="Asphodel", suppress_callback_exceptions=True)
app.index_string = app.index_string.replace("</head>", f"<style>{CSS}</style></head>")
# tab and home page stop together: closing the home tab/window works like Quit
app.index_string = app.index_string.replace(
    "{%scripts%}", "{%scripts%}\n<script>" + tabwatch.watch_js("Asphodel") + "</script>")
icon.use_in_dash(app, "asphodel")          # browser-tab icon (core/icons/)
app.layout = serve_layout


@app.callback(
    Output("msg", "children"),
    Input({"type": "act", "a": ALL, "p": ALL, "u": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def actions(clicks):
    """Every button made with act(): start/stop tools and the 'open with' buttons."""
    tid = ctx.triggered_id
    if not tid or not any(clicks) or not (ctx.triggered and ctx.triggered[0]["value"]):
        return dash.no_update
    a, p = tid["a"], tid["p"]
    try:
        if a == "start":
            return start_tool(p)
        if a == "stop":
            return stop_tool(p)
        if a == "spectrogram_file":
            return start_tool("spectrogram", [p])
        if a in ("viewer_file", "viewer_dir"):
            d, name = (os.path.dirname(p), os.path.basename(p)) if a == "viewer_file" else (p, None)
            changes = {"file_path": os.path.join(d, "")}
            if name:
                changes["spectrum_name"] = name
            _update_json(paths.settings_file("viewer_config.json"), changes)
            return start_tool("theia") + " Viewer set to " + (name or os.path.basename(d.rstrip("/"))) + \
                (" – reload the Theia page if it was already open." if _port_open(THEIA_PORT) else "")
        if a == "scan_dir":
            _update_json(paths.settings_file("scanner_config.json"), {"spectra_dir": os.path.join(p, "")})
            return start_tool("theia") + " Scan folder set to " + os.path.basename(p.rstrip("/")) + \
                " (Scan tab; reload the page if Theia was already open)."
        if a == "rerun_fit":
            return start_tool("decay_fit", fit_rerun_args(_read_json(p)))
        if a == "fit_spectrogram":
            rec = _read_json(p)
            w = (rec.get("settings") or {}).get("freq_window_hz") or []
            pad = (w[1] - w[0]) * 2 if len(w) == 2 else 0
            args = [rec["file"]] + ([repr(w[0] - pad), repr(w[1] + pad)] if pad else [])
            return start_tool("spectrogram", args)
        if a == "open_file":
            open_in_system(p)
            return f"Opened {os.path.basename(p)}."
    except Exception as exc:
        return f"⚠ {exc}"
    return dash.no_update


@app.callback(
    Output("msg", "children", allow_duplicate=True),
    Input("ss-run", "n_clicks"), Input("cp-run", "n_clicks"),
    Input("if-run", "n_clicks"), Input("ls-run", "n_clicks"),
    State("ss-in", "value"), State("ss-out", "value"),
    State("cp-file", "value"), State("cp-sheet", "value"),
    State("if-file", "value"), State("if-exclude", "value"),
    State("ls-file", "value"),
    prevent_initial_call=True,
)
def run_forms(_s, _c, _i, _l, ss_in, ss_out,
              cp_file, cp_sheet, if_file, if_ex, ls_file):
    """The 'Run' buttons of the tool forms."""
    t = ctx.triggered_id
    try:
        if t == "ss-run":
            if not ss_in or not ss_out:
                return "⚠ Sum spectra: give the input and output folders."
            return start_tool("sum_spectra", [os.path.expanduser(ss_in), os.path.expanduser(ss_out)])
        if t == "cp-run":
            if not cp_file:
                return "⚠ No RionID table in rionid/."
            return start_tool("correction_plot", [cp_file, cp_sheet or "New2ndOrderCorrection"])
        if t == "if-run":
            if not if_file:
                return "⚠ No spreadsheet in the experiment folder."
            ex = shlex.split(if_ex or "")
            return start_tool("intensity_fit", [if_file] + (["--exclude", *ex] if ex else []))
        if t == "ls-run":
            if not ls_file:
                return "⚠ No scheme files in levelschemes/."
            return start_tool("levelschemer", [ls_file])
    except Exception as exc:
        return f"⚠ {exc}"
    return dash.no_update


@app.callback(
    Output({"type": "status", "k": ALL}, "children"),
    Input("tick", "n_intervals"),
    State({"type": "status", "k": ALL}, "id"),
)
def refresh_status(_n, ids):
    return [_status_children(*tool_status(i["k"])) for i in ids]


@app.callback(
    Output("phys-msg", "children"),
    Input("phys-save", "n_clicks"),
    State({"type": "phys", "k": ALL}, "value"),
    State({"type": "phys", "k": ALL}, "id"),
    prevent_initial_call=True,
)
def save_physics(_n, values, ids):
    vals = {i["k"]: v for i, v in zip(ids, values)}
    try:
        checked = {k: physics.canonical(k, v) for k, v in vals.items()}
        if not checked["harmonics"]:
            raise ValueError("harmonics list is empty")
        for k in ("z_lims", "n_lims", "el1", "el2"):
            if len(checked[k]) != 2:
                raise ValueError(f"{PHYS_LABELS[k]} needs two numbers")
        physics.save_shared(checked)
    except (TypeError, ValueError) as exc:
        return f"⚠ Not saved: {exc}"
    return f"Saved {datetime.datetime.now():%H:%M:%S}."


@app.callback(
    Output("notes-msg", "children"),
    Input("notes-save", "n_clicks"),
    State("notes", "value"),
    prevent_initial_call=True,
)
def save_notes(_n, text):
    path = paths.notes_file()
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text or "")
    os.replace(tmp, path)
    return f"Saved {datetime.datetime.now():%H:%M:%S}."


@app.callback(
    Output("loc", "href"),
    Input("exp-select", "value"),
    Input("refresh", "n_clicks"),
    prevent_initial_call=True,
)
def switch_experiment(exp, _r):
    """Change the active experiment (or just refresh): reload the page."""
    if ctx.triggered_id == "exp-select" and exp and exp != paths.experiment():
        paths.set_experiment(exp)
    return "/"


@app.callback(
    Output("exp-modal", "style"),
    Output("exp-msg", "children"),
    Output("loc", "href", allow_duplicate=True),
    Input("exp-new-open", "n_clicks"), Input("exp-cancel", "n_clicks"), Input("exp-create", "n_clicks"),
    State("exp-name", "value"), State("exp-dir", "value"), State("exp-copy", "value"),
    prevent_initial_call=True,
)
def new_experiment(_o, _c, _k, name, data_dir, copy_from):
    shown, hidden = {"display": "flex"}, {"display": "none"}
    t = ctx.triggered_id
    if t == "exp-new-open":
        return shown, "", dash.no_update
    if t == "exp-cancel":
        return hidden, "", dash.no_update
    try:
        paths.create_experiment(name, (data_dir or "").strip() or None, copy_from or None)
        paths.set_experiment(name.strip())
    except (ValueError, OSError) as exc:
        return shown, f"⚠ {exc}", dash.no_update
    return hidden, "", "/"


# ---------------------------------------------------------------------------- sync with APPC256

@app.callback(
    Output("sync-modal", "style"),
    Input("sync-open", "n_clicks"), Input("sync-close", "n_clicks"),
    prevent_initial_call=True,
)
def sync_modal_show(_o, _c):
    return {"display": "flex"} if ctx.triggered_id == "sync-open" else {"display": "none"}


@app.callback(
    Output("sync-result", "children"),
    Output("sync-pull-list", "options"),
    Output("sync-pull-list", "value"),
    Output("sync-pull-box", "style"),
    Output("sync-data", "data"),
    Input("sync-check", "n_clicks"),
    State("sync-pw", "value"),
    prevent_initial_call=True,
)
def sync_check(_n, pw):
    """Compare both sides, send the new confirmed lists, offer the new spectrograms."""
    hidden = {"display": "none"}
    try:
        res = sync.check(pw)
        lines = []
        if res["remote_confirmed_missing"]:
            lines.append(f"⚠ {sync.remote_dir(res['cfg'], 'remote_confirmed')} does not exist on "
                         f"APPC256 – nothing sent.")
        elif res["to_push"]:
            sent = sync.push_confirmed(res["to_push"], pw, res["exp"])
            lines.append(html.Div([html.B(f"Sent {len(sent)} confirmed list(s):"),
                                   html.Ul([html.Li(n, className="mono") for n in sent])]))
        else:
            lines.append("Confirmed lists: APPC256 already has all of them.")
        if res["remote_only_confirmed"]:
            lines.append(f"({len(res['remote_only_confirmed'])} confirmed list(s) exist only on "
                         f"APPC256; they are not pulled.)")
        if res["remote_spectrograms_missing"]:
            lines.append(f"⚠ {sync.remote_dir(res['cfg'], 'remote_spectrograms')} does not exist on APPC256.")
        elif not res["to_pull"]:
            lines.append("Spectrograms: this computer already has all of them.")
    except sync.SyncError as exc:
        return f"⚠ {exc}", [], [], hidden, None
    except OSError as exc:
        return f"⚠ {exc}", [], [], hidden, None
    opts = [{"label": f" {c['path']}  ({_size(c['size'])}" + (" + .txt" if len(c["files"]) > 1 else "") + ")",
             "value": c["path"]} for c in res["to_pull"]]
    data = {"exp": res["exp"], "sizes": {c["path"]: c["size"] for c in res["to_pull"]},
            "files": {c["path"]: c["files"] for c in res["to_pull"]}}
    return ([html.Div(l) for l in lines], opts, [],
            {"display": "block"} if opts else hidden, data)


@app.callback(
    Output("sync-space", "children"),
    Input("sync-pull-list", "value"),
    State("sync-data", "data"),
)
def sync_space(sel, data):
    if not data:
        return ""
    import shutil
    need = sum(data["sizes"].get(n, 0) for n in sel or [])
    free = shutil.disk_usage(paths.spectrograms(data["exp"])).free
    txt = f"Selected {len(sel or [])} file(s), {_size(need)} · free here: {_size(free)}"
    if need + sync.FREE_MARGIN > free:
        txt += f" – ⚠ too much (keeps {_size(sync.FREE_MARGIN)} spare)"
    return txt


@app.callback(
    Output("sync-msg", "children"),
    Input("sync-pull", "n_clicks"), Input("sync-cancel", "n_clicks"),
    State("sync-pull-list", "value"), State("sync-pw", "value"), State("sync-data", "data"),
    prevent_initial_call=True,
)
def sync_pull(_p, _c, sel, pw, data):
    if ctx.triggered_id == "sync-cancel":
        return "Cancelling…" if sync.JOB.cancel() else "No pull running."
    if not data:
        return "⚠ Press “Check” first."
    items = [f for n in sorted(sel or []) if n in data["files"] for f in data["files"][n]]
    try:
        sync.JOB.start(items, pw, data["exp"])
    except (sync.SyncError, OSError) as exc:
        return f"⚠ {exc}"
    return f"Pulling {len(items)} file(s)…" if items else "⚠ Nothing selected."


@app.callback(
    Output("sync-progress", "children"),
    Input("tick", "n_intervals"),
    State("sync-modal", "style"),
)
def sync_progress(_n, style):
    if not style or style.get("display") == "none":
        return dash.no_update
    st = sync.JOB.status()
    if "items" not in st:
        return ""
    n = len(st["items"])
    out = []
    if st["running"]:
        size, got = st["current_size"], st["current_got"]
        pct = 100 * got / size if size else 0
        out += [html.Div(f"Pulling {st['index'] + 1}/{n}: {st['current']} – "
                         f"{_size(got)} of {_size(size)}", className="note"),
                html.Div(html.Div(style={"width": f"{pct:.1f}%"}), className="pbar")]
    else:
        if st.get("cancelled"):
            out.append(html.Div("Pull cancelled.", className="note"))
        if st.get("error"):
            out.append(html.Div(f"⚠ {st['error']}", className="note"))
        if not st.get("cancelled") and not st.get("error"):
            out.append(html.Div("Pull finished.", className="note"))
    if st["done"]:
        out.append(html.Div(f"Pulled into {os.path.relpath(st['dest'], paths.experiment_dir(st['exp']))}/: "
                            + ", ".join(st["done"])
                            + ("" if st["running"] else " – press ↻ Refresh to see them in Data files."),
                            className="note"))
    if st["skipped"]:
        out.append(html.Div("Already here, skipped: " + ", ".join(st["skipped"]), className="note"))
    return out


@app.callback(
    Output("quit-info", "data"),
    Input("quit", "n_clicks"),
    prevent_initial_call=True,
)
def quit_info(n):
    """Quit pressed: hand the list of running tools to the confirmation dialog."""
    return {"n": n, "tools": running_summary()}


def _home_tab_closed():
    """The home page's last tab/window was closed: same as Quit."""
    n = stop_everything()
    print(f"Closed {n} tool process(es), stopping the home page.", flush=True)
    os._exit(0)


_TABS = tabwatch.TabTracker(on_last_closed=_home_tab_closed, name="Asphodel home page")
tabwatch.install_flask(app.server, _TABS)


@app.server.route(NUBASE_URL)
def _nubase_pdf():
    """The NUBASE2020 paper, shown by the browser's PDF viewer (#page=N works)."""
    from flask import send_file
    return send_file(NUBASE_PDF, mimetype="application/pdf")


# View NUBASE: the link follows the search box ("198Ir" -> page 140); Enter
# opens it in one reused tab ("asphodel-nubase"); ?p= makes that tab reload
# when only the page changes.
app.clientside_callback(
    r"""
    function(q, nsub, idx) {
        var base = '%s';
        // "198Ir", "ir198", "Ir-198", "198Ir77+", "m186Hf", "186Hfm" -> "198Ir"
        var s = (q || '').replace(/[\s\-]+/g, '').replace(/\d+\+$/, '');
        if (!s) return [base, ''];
        var key = null, m = /^[a-z]?(\d{1,3})([a-zA-Z]{1,3})$/.exec(s)
                         || /^([a-zA-Z]{1,2})(\d{1,3})[a-z]?$/.exec(s);
        if (m) {
            var A = /^\d/.test(m[1]) ? m[1] : m[2], el = /^\d/.test(m[1]) ? m[2] : m[1];
            for (var n = Math.min(2, el.length); n >= 1 && !key; n--) {
                var k = A + el.charAt(0).toUpperCase() + el.slice(1, n).toLowerCase();
                if (k in idx) key = k;
            }
            if (!key && el.toLowerCase() === 'n' && (A + 'n') in idx) key = A + 'n';   // free neutron
        }
        if (!key) return [base, '✗ ' + q.trim() + ' is not in NUBASE2020'];
        var url = base + '?p=' + idx[key] + '#page=' + idx[key];
        var trig = window.dash_clientside.callback_context.triggered.map(function(t) { return t.prop_id; });
        if (trig.indexOf('nub-q.n_submit') >= 0) window.open(url, 'asphodel-nubase');
        return [url, key + ' → page ' + idx[key] + (trig.indexOf('nub-q.n_submit') >= 0 ? '' : ' (Enter to open)')];
    }
    """ % NUBASE_URL,
    Output("nub-go", "href"),
    Output("nub-msg", "children"),
    Input("nub-q", "value"),
    Input("nub-q", "n_submit"),
    State("nub-index", "data"),
    prevent_initial_call=True,
)


@app.server.route("/_asphodel_quit", methods=["POST"])
def _quit_route():
    """Close all tools, then stop the home page."""
    n = stop_everything()
    print(f"Quit pressed in the browser: closed {n} tool process(es), stopping the home page.")
    threading.Timer(1.0, lambda: os._exit(0)).start()
    return "stopping"


app.clientside_callback(
    r"""
    function(info) {
        if (!info) return window.dash_clientside.no_update;
        var msg = 'Quit Asphodel?\n\n';
        if (info.tools && info.tools.length) {
            msg += 'This also closes all running tools:\n  • ' + info.tools.join('\n  • ') +
                   '\n\nSave your work in them first (unsaved changes are lost).';
        } else {
            msg += 'No tools are running.';
        }
        if (!window.confirm(msg)) return window.dash_clientside.no_update;
        fetch('_asphodel_quit', {method: 'POST'}).catch(function() {}).then(function() {
            window.close();
            // a script can only close tabs it opened itself; otherwise show a message
            setTimeout(function() {
                document.body.innerHTML =
                    '<div style="font-family:sans-serif;padding:60px;text-align:center;">' +
                    '<h3 style="color:#f57d15">Asphodel has stopped</h3>' +
                    '<p>All tools are closed. You can close this tab; start again with ' +
                    '<code>python home.py</code>.</p></div>';
            }, 300);
        });
        return 'quit';
    }
    """,
    Output("quit-done", "children"),
    Input("quit-info", "data"),
    prevent_initial_call=True,
)


# Browsers that take --new-window, by the name of their .desktop file / command
_NEW_WINDOW_BROWSERS = [
    ("firefox", ["firefox", "firefox-esr"]),
    ("google-chrome", ["google-chrome", "google-chrome-stable"]),
    ("chromium", ["chromium", "chromium-browser"]),
    ("brave", ["brave-browser", "brave"]),
    ("microsoft-edge", ["microsoft-edge", "microsoft-edge-stable"]),
    ("vivaldi", ["vivaldi", "vivaldi-stable"]),
]


def _default_browser_commands():
    """Commands of the default browser first (xdg-settings), then the others."""
    try:
        desktop = subprocess.run(["xdg-settings", "get", "default-web-browser"],
                                 capture_output=True, text=True, timeout=3).stdout.strip().lower()
    except (OSError, subprocess.SubprocessError):
        desktop = ""
    first = [cmds for name, cmds in _NEW_WINDOW_BROWSERS if name in desktop]
    return first + [cmds for name, cmds in _NEW_WINDOW_BROWSERS if name not in desktop]


def open_in_new_window(url):
    """Open the home page in a new browser window (the tools' tabs then open
    in that window too).  With Firefox this is the suite's own Firefox
    window (core/browser.py: own profile, asphodel icon in the dock);
    otherwise the default browser's --new-window option, falling back to
    Python's webbrowser if no known browser is found."""
    import shutil
    if browser.firefox() and browser.open_url(url, new_window=True):
        return
    for cmds in _default_browser_commands():
        for cmd in cmds:
            exe = shutil.which(cmd)
            if exe:
                try:
                    subprocess.Popen([exe, "--new-window", url], stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL, start_new_session=True)
                    return
                except OSError:
                    continue
    webbrowser.open_new(url)


def _free_port(preferred):
    if not _port_open(preferred):
        return preferred
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


if __name__ == "__main__":
    import logging
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    port = _free_port(HOME_PORT)
    url = f"http://127.0.0.1:{port}/"
    print(f"Asphodel home page at {url}  (Quit button: closes everything; Ctrl-C: home page only)")
    threading.Timer(1.0, lambda: open_in_new_window(url)).start()
    app.run(debug=False, host="127.0.0.1", port=port)
