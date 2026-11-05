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
  * folders: the data and output folders the tools use, connected per
    experiment ("Connect data folder…" / "Connect output folder…", each with a
    kind; core/paths.py keeps them in <experiment>/settings/folders.json)
  * the shared physics settings (physics.json) and the overrides of each tool
  * lab book per experiment: time-stamped notes under a heading per day
    (<experiment>/notes.md, append-only; core/labbook.py)
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
from core import (browser, corrections, folderpick, icon, jsonfile, labbook, logcapture, logpanel,
                  paths, physics, sync, tabwatch, version)

# output also to data/logs/ and the page's Log panel (core/logcapture.py)
if __name__ == "__main__":
    logcapture.start("home")

import dash
from dash import dcc, html, Input, Output, State, ALL, ctx

REPO = paths.REPO_ROOT
ACCENT, PALE, SIDE, LINE = "#f57d15", "#fff4ec", "#f9f9f9", "#dddddd"
HOME_PORT = 8050
THEIA_PORT = 8052
SEGRE_PORT = 8054             # reference/segre_chart.py
GUICORR_PORT = 8056           # tartarus/gui_2nd_order_correction.py

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
    # (tartarus/new_2nd_order_correction_plot.py has no card any more: the GUI
    #  correction tool replaces it and opens its figure with its "Plot" button)
    "gui_correction":  ("Tartarus", "GUI Second Order Correction", "tartarus/gui_2nd_order_correction.py",
                        "Simulated lines on a spectrum (strongest peak = reference): drag them into "
                        "place, assign peaks, fit and save a correction per harmonic.", True),
    "levelschemer":    ("Elysium", "LevelSchemer", "elysium/levelschemer.py",
                        "Draw nuclear level schemes (Tk window).", False),
    "segre":           ("Reference", "Segrè chart", "reference/segre_chart.py",
                        "Chart of nuclides from NUBASE2020: colour by half-life, decay mode or "
                        "identified ions; ground states or isomers; hover for the data.", True),
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
    if key == "segre" and _port_open(SEGRE_PORT):
        browser.open_url(f"http://127.0.0.1:{SEGRE_PORT}/")
        return "The Segrè chart is already running – opened it in the browser."
    if key == "gui_correction" and _port_open(GUICORR_PORT):
        browser.open_url(f"http://127.0.0.1:{GUICORR_PORT}/")
        return "GUI Second Order Correction is already running – opened it in the browser."
    if single and running(key):
        return f"{name} is already running."
    exp = paths.experiment()
    env = dict(os.environ, **{paths.ENV_EXPERIMENT: exp})
    # own process group, so Stop/Quit also end what the tool starts itself
    # (e.g. decay fits opened from the spectrogram), and Ctrl-C in the home
    # page's terminal does not reach the tools
    # the tool keeps its own log (core/logcapture.py): give it the home page's
    # original stdout/stderr, not the home page's capture
    proc = subprocess.Popen([sys.executable, path, *[str(a) for a in args]], cwd=REPO, env=env,
                            start_new_session=True, **logcapture.child_stdio())
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
    """A JSON file as a dict; {} if missing.  A damaged file is set aside
    (its backup restored if there is one) with a warning: core/jsonfile.py."""
    return jsonfile.read(path, {})


def _update_json(path, changes):
    jsonfile.update(path, changes)


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
        "gui_correction": [act("start", "gui_correction", "Open", "btn primary small",
                               title="Opens in a new tab, at the last spectrum"),
                           act("stop", "gui_correction", "Stop")],
        "segre": [act("start", "segre", "Open", "btn primary small",
                      title="Opens the chart in a new tab"),
                  act("stop", "segre", "Stop")],
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


def _folder_opts(group, kind=None):
    """Dropdown options: the connected folders of a group (of one kind first)."""
    fs = paths.connected(group)
    if kind:
        fs = [f for f in fs if f["kind"] == kind] + [f for f in fs if f["kind"] != kind]
    return [{"label": f"{f['name']}  ·  {f['path']}", "value": f["path"]} for f in fs]


def _files_in(group, kinds, patterns):
    """Files matching the patterns in the connected folders of the given kinds
    (labelled 'folder name / file'), the default folder's first."""
    out, seen = [], set()
    for f in paths.connected(group):
        if kinds and f["kind"] not in kinds:
            continue
        for pat in patterns:
            for fp in sorted(glob.glob(os.path.join(f["path"], pat))):
                if fp not in seen:
                    seen.add(fp)
                    out.append({"label": f"{f['name']} / {os.path.basename(fp)}", "value": fp})
    return out


def tool_forms():
    """Small forms for the command-line tools.  Folders and files come from the
    connected folders (Folders section below)."""
    schemes = _files_in("data", ["levelschemes"], ["*.json"])
    lbl = lambda t: html.Label(t, className="flabel")
    return {
        "sum_spectra": html.Div([
            lbl("Input folder"),
            dcc.Dropdown(id="ss-in", options=_folder_opts("data", "spectra_1d"), clearable=False,
                         value=paths.spectra_1d()),
            lbl("Output folder"),
            dcc.Dropdown(id="ss-out", options=_folder_opts("data", "summed_1d"), clearable=False,
                         value=paths.summed_1d()),
            lbl("Peak tracking window ± (Hz)"),
            html.Div(dcc.Input(id="ss-window", type="number", min=1, value=500, debounce=True),
                     title="The drift-corrected sum follows the strongest peak: each spectrum's "
                           "peak is looked for within ± this of the previous one's"),
            html.Button("Run", id="ss-run", className="btn primary small", n_clicks=0),
        ], className="form"),
        "levelschemer": html.Div([
            lbl("Open a scheme"),
            dcc.Dropdown(id="ls-file", options=schemes, clearable=False,
                         value=schemes[0]["value"] if schemes else None),
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
        # go-to-nuclide form first, the Open NUBASE button last (bottom of the card)
        body = [
            dcc.Store(id="nub-index", data=pages),
            html.Div([
                html.Label("Go to a nuclide (e.g. 198Ir, Ir198, Ir-198)", className="flabel"),
                dcc.Input(id="nub-q", type="text", list="nub-names", debounce=False,
                          className="wide", placeholder="198Ir", autoComplete="off", n_submit=0),
                html.Datalist(id="nub-names", children=[html.Option(value=k) for k in pages]),
                html.Div([html.A("Open at page", id="nub-go", href=NUBASE_URL, target="asphodel-nubase",
                                 className="btn small"),
                          html.Span(id="nub-msg", className="note")], className="below"),
            ], className="form"),
            html.Div(html.A("Open NUBASE", href=NUBASE_URL, target="_blank",
                            className="btn primary small",
                            title="NUBASE2020 (Kondev et al., Chin. Phys. C 45, 030001) in a new tab"),
                     className="btns"),
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
                    # the card's own buttons (Open / Stop) stay last, at the bottom of the
                    # card, with the form above them; cards without such buttons end
                    # with the form and its Run button
                    if card.children[-1].children:
                        card.children.insert(len(card.children) - 1, forms[key])
                    else:
                        card.children[-1] = forms[key]
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


# --------------------------------------------------------------------------- sidebar

PHYS_LABELS = {
    "ref_name": "Reference ion", "ref_frequency": "f_ref (Hz)", "ref_harmonic": "Ref. harmonic",
    "gamma_t": "γ_t", "harmonics": "Harmonics", "z_lims": "Z min, max", "n_lims": "N min, max",
    "el1": "el1 (A, Z)", "el2": "el2 (A, Z)", "mq_width": "m/q window (±)",
    "correction_enabled": "2nd-order correction", "corr_a": "a", "corr_b": "b", "corr_c": "c",
    "min_half_life": "Min T½", "min_half_life_unit": "T½ unit",
    "exclude_unknown_hl": "Exclude unknown T½",
    "corr_profile": "Correction profile", "ref_search_hz": "Ref. peak search, other h ± (Hz)",
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
        if k == "corr_profile":
            continue                       # drawn above a, b, c
        if k == "corr_a":
            # correction profiles (core/corrections.py): the profile *name* is stored
            # (physics.json "corr_profile"); a profile fills a, b, c (its general values)
            # and they can only be typed in for "custom"
            profs = corrections.load_profiles()
            r = corrections.resolve(sh.get("corr_profile"), sh["corr_a"], sh["corr_b"], sh["corr_c"],
                                    profiles=profs)
            fields += [html.Label("Correction profile"),
                       dcc.Dropdown(id={"type": "phys", "k": "corr_profile"},
                                    options=corrections.options(profs), value=r["name"],
                                    clearable=False, className="unit")]
            if r["name"] != corrections.CUSTOM:
                inp.disabled = True
        if k in ("corr_b", "corr_c") and corrections.resolve(
                sh.get("corr_profile"), sh["corr_a"], sh["corr_b"], sh["corr_c"])["name"] != corrections.CUSTOM:
            inp.disabled = True
        fields += [html.Label(PHYS_LABELS.get(k, k)), inp]
        if k == "corr_c":
            fields += [html.Div(id="phys-corr-info", className="note",
                                style={"gridColumn": "1 / -1"})]
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
    nub = os.path.join(paths.REPO_ROOT, "core", "nubase_4.mas20")
    return html.Aside([
        html.Div([
            html.Details([html.Summary("Shared physics"), physics_panel()],
                         open=False, className="fold first"),
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


# ===========================================================================
# Folders (connected data / output folders, core/paths.py)
# ===========================================================================

def _folder_rows(group):
    kinds = paths.FOLDER_GROUPS[group]
    fs = paths.connected(group)
    first_of_kind = {}
    for f in fs:
        first_of_kind.setdefault(f["kind"], f["path"])
    rows = []
    for f in fs:
        ok = os.path.isdir(f["path"])
        if ok:
            n, total = folder_summary(f["path"], "*")
            contents = f"{n} item{'s' if n != 1 else ''}"
        else:
            contents = html.Span("✗ missing", className="bad")
        is_default = first_of_kind.get(f["kind"]) == f["path"]
        fid = {"g": group, "p": f["path"]}
        rows.append(html.Tr([
            html.Td(html.B(f["name"])),
            html.Td(dcc.Dropdown(id={"type": "fold-kind", **fid}, value=f["kind"], clearable=False,
                                 options=[{"label": v, "value": k} for k, v in kinds.items()],
                                 className="kind-dd")),
            html.Td(html.Span(f["path"], className="mono"), className="fpath"),
            html.Td(contents),
            html.Td([
                html.Span("in use", className="inuse",
                          title=f"Tools use this folder for '{kinds[f['kind']]}'")
                if is_default and f["kind"] != "other" else
                (html.Button("Use", id={"type": "fold-act", "a": "default", **fid}, className="btn small",
                             n_clicks=0, title=f"Make the tools use this folder for '{kinds[f['kind']]}'")
                 if f["kind"] != "other" else ""),
                html.Button("Open", id={"type": "fold-act", "a": "open", **fid}, className="btn small",
                            n_clicks=0, title="Open in the file manager"),
                dcc.ConfirmDialogProvider(
                    html.Button("Disconnect", className="btn small"),
                    id={"type": "fold-act", "a": "disconnect", **fid},
                    message=f"Disconnect {f['path']}?\n\nThe folder and its files are not touched; "
                            "it is only removed from the list."),
            ], className="factions"),
        ]))
    if not rows:
        return html.P("No folders connected.", className="note")
    return html.Table([html.Thead(html.Tr([html.Th(h) for h in
                                           ("Name", "Kind", "Folder", "Contents", "")])),
                       html.Tbody(rows)], className="ftable")


def _folder_group(group, title, button):
    kinds = paths.FOLDER_GROUPS[group]
    return html.Div([
        html.Div([
            html.H3(title),
            html.Button(button, id={"type": "fold-pick", "g": group}, className="btn primary small",
                        n_clicks=0, title="Choose a folder with the desktop's folder dialog"),
        ], className="fhead"),
        html.Div([
            dcc.Input(id={"type": "fold-path", "g": group}, type="text", className="wide",
                      placeholder="… or paste a folder path", debounce=False, n_submit=0),
            dcc.Dropdown(id={"type": "fold-newkind", "g": group}, value="auto", clearable=False,
                         options=[{"label": "kind: guess", "value": "auto"}] +
                                 [{"label": v, "value": k} for k, v in kinds.items()],
                         className="kind-dd"),
            html.Button("Add", id={"type": "fold-add", "g": group}, className="btn small", n_clicks=0),
        ], className="fadd"),
        html.Div(_folder_rows(group), id={"type": "fold-list", "g": group}),
    ], className="fgroup")


def folders_section():
    return section(
        "Folders",
        html.P(["Where the tools read their data and save their output, for this experiment. "
                "Each folder has a kind; a tool uses the folder marked ", html.I("in use"),
                " for the kind it needs, and its folder and file choosers list all connected folders. "
                "Without a folder of a kind, the experiment's standard sub-folder is used."],
               className="note"),
        html.Div([_folder_group("data", "Data folders", "Connect data folder…"),
                  _folder_group("output", "Output folders", "Connect output folder…")],
                 className="fgrid"),
        dcc.Store(id="fold-ver", data=0),
    )


def _labbook_log():
    """The lab book, newest day first (and newest entry first within a day),
    with the notes from before the lab book at the end."""
    earlier, days = labbook.parse(paths.notes_file())
    today = datetime.date.today()
    out = []
    for day in reversed(days):
        d = day["date"]
        label = f"{d:%A} {d.day} {d:%B %Y}"
        if d == today:
            label = "Today · " + label
        elif d == today - datetime.timedelta(days=1):
            label = "Yesterday · " + label
        n = len(day["entries"])
        out.append(html.Div([
            html.Div([html.Span(label), html.Span(f"{n} entr{'y' if n == 1 else 'ies'}", className="lb-n")],
                     className="lb-day"),
            *[html.Div([html.Div(e["time"], className="lb-time"),
                        html.Div(e["text"], className="lb-text")], className="lb-entry")
              for e in reversed(day["entries"])],
        ], className="lb-dayblock"))
    if earlier:
        out.append(html.Div([html.Div("Earlier notes (before the lab book)", className="lb-day"),
                             html.Div(earlier, className="lb-text lb-earlier")], className="lb-dayblock"))
    if not out:
        out = [html.Div("No entries yet.", className="note")]
    return out


def notes_section():
    return section(
        "Lab book",
        dcc.Textarea(id="notes", value="", className="notes",
                     placeholder="New entry for today (time-stamped when added)…"),
        html.Div([html.Button("Add entry", id="notes-add", className="btn primary small", n_clicks=0),
                  html.Span(id="notes-msg", className="note"),
                  html.Span("Entries can't be changed once added; add a new one to correct.",
                            className="note lb-hint")], className="below"),
        html.Div(_labbook_log(), id="notes-log", className="lb-log below"),
    )


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
        logpanel.dash_button(className="btn small"),
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
        html.Div(id="settings-banner", className="banner settings-warn",
                 style={"display": "none"}),
        html.Div([
            sidebar(),
            html.Main([tools_section(), folders_section(), notes_section()]),
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
.settings-warn{{border-left:4px solid var(--accent);color:#222}}
.settings-warn ul{{margin:2px 0 2px 18px;padding:0}}
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
.tool{{border:1px solid var(--line);border-radius:6px;padding:12px;background:#fff;border-top:3px solid var(--accent);
  display:flex;flex-direction:column}}
/* the card's last block (its buttons, or the form ending in its Run button) sits at the
   bottom of the card, so the buttons of the cards in a row line up at the bottom left */
.tool>:last-child{{margin-top:auto}}
/* Folders section */
.fgrid{{display:grid;grid-template-columns:1fr;gap:16px}}
.fgroup{{border:1px solid var(--line);border-radius:6px;padding:12px;border-top:3px solid var(--accent);min-width:0}}
.fhead{{display:flex;align-items:center;justify-content:space-between;gap:10px}}
.fhead h3{{margin:0;font-size:15px}}
.fadd{{display:flex;gap:6px;align-items:center;margin:10px 0 8px}}
.fadd .wide{{flex:1}} .fadd .btn{{margin:0}}
.kind-dd{{width:170px;font-size:12px}}
.ftable{{width:100%;border-collapse:collapse;font-size:12px}}
.ftable th{{text-align:left;padding:4px 6px}}
.ftable td{{border-bottom:1px solid var(--line);padding:4px;vertical-align:middle}}
.ftable .fpath{{word-break:break-all;width:55%}}
.ftable .factions{{white-space:nowrap;min-width:230px}} .ftable .factions .btn{{margin:0 4px 0 0}}
.inuse{{display:inline-block;font-size:11px;color:var(--ok);font-weight:700;margin-right:6px}}
.bad{{color:#c62828}}
details.fold.first{{margin-top:0}}

.tool .form+.btns{{padding-top:10px}}
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
.notes{{width:100%;min-height:70px;font-family:monospace;font-size:12px;border:1px solid var(--line);border-radius:4px;padding:8px;resize:vertical}}
/* Lab book */
.lb-hint{{margin-left:10px}}
.lb-log{{max-height:480px;overflow-y:auto;border:1px solid var(--line);border-radius:4px}}
.lb-day{{position:sticky;top:0;z-index:1;display:flex;justify-content:space-between;align-items:baseline;
  background:var(--pale);border-bottom:2px solid var(--accent);padding:5px 10px;font-weight:700;font-size:13px}}
.lb-n{{font-weight:400;font-size:11px;color:var(--muted)}}
.lb-entry{{display:flex;gap:12px;padding:6px 10px;border-bottom:1px solid var(--line)}}
.lb-time{{flex:none;width:42px;font-family:monospace;font-size:12px;color:var(--accent);font-weight:700;padding-top:1px}}
.lb-text{{flex:1;min-width:0;white-space:pre-wrap;word-break:break-word;font-family:monospace;font-size:12px}}
.lb-earlier{{padding:6px 10px;color:#444}}
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
app.layout = logpanel.dash_layout(serve_layout, "Home page", files=True)
logpanel.register_dash(app, files=True)


@app.callback(
    Output("msg", "children"),
    Input({"type": "act", "a": ALL, "p": ALL, "u": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def actions(clicks):
    """Every button made with act(): start/stop tools, open a folder."""
    tid = ctx.triggered_id
    if not tid or not any(clicks) or not (ctx.triggered and ctx.triggered[0]["value"]):
        return dash.no_update
    a, p = tid["a"], tid["p"]
    try:
        if a == "start":
            return start_tool(p)
        if a == "stop":
            return stop_tool(p)
        if a == "open_file":
            open_in_system(p)
            return f"Opened {os.path.basename(p)}."
    except Exception as exc:
        return f"⚠ {exc}"
    return dash.no_update


@app.callback(
    Output("msg", "children", allow_duplicate=True), Output("fold-ver", "data"),
    Output({"type": "fold-path", "g": ALL}, "value"),
    Input({"type": "fold-pick", "g": ALL}, "n_clicks"),
    Input({"type": "fold-add", "g": ALL}, "n_clicks"),
    Input({"type": "fold-path", "g": ALL}, "n_submit"),
    State({"type": "fold-path", "g": ALL}, "value"),
    State({"type": "fold-newkind", "g": ALL}, "value"),
    State("fold-ver", "data"),
    prevent_initial_call=True,
)
def connect_folders(_pick, _add, _sub, typed, newkinds, ver):
    """'Connect … folder' (desktop folder dialog) and 'Add' (a pasted path)."""
    t = ctx.triggered_id
    nu = dash.no_update
    groups = [x["id"]["g"] for x in ctx.inputs_list[0]]
    keep = [nu] * len(groups)
    if not t or not (ctx.triggered and ctx.triggered[0]["value"]):
        return nu, nu, keep
    g = t["g"]
    i = groups.index(g)
    kind = None if newkinds[i] == "auto" else newkinds[i]
    word = "data" if g == "data" else "output"
    try:
        if t["type"] == "fold-pick":
            start = next((f["path"] for f in paths.connected(g) if os.path.isdir(f["path"])),
                         paths.experiment_dir())
            path = folderpick.pick_folder(f"Connect {word} folder ({paths.experiment()})", start)
            if not path:
                return "Nothing connected (dialog cancelled).", nu, keep
        else:
            path = (typed[i] or "").strip()
            if not path:
                return "⚠ Paste a folder path first.", nu, keep
        f = paths.connect_folder(g, path, kind)
    except Exception as exc:
        return f"⚠ {exc}", nu, keep
    cleared = list(keep)
    cleared[i] = ""
    return (f"Connected {word} folder {f['path']} as '{paths.FOLDER_GROUPS[g][f['kind']]}'.",
            (ver or 0) + 1, cleared)


@app.callback(
    Output("msg", "children", allow_duplicate=True), Output("fold-ver", "data", allow_duplicate=True),
    Input({"type": "fold-act", "a": ALL, "g": ALL, "p": ALL}, "n_clicks"),
    Input({"type": "fold-act", "a": ALL, "g": ALL, "p": ALL}, "submit_n_clicks"),
    Input({"type": "fold-kind", "g": ALL, "p": ALL}, "value"),
    State("fold-ver", "data"),
    prevent_initial_call=True,
)
def folder_actions(_c, _s, _kinds, ver):
    """Use / Open / Disconnect a connected folder, or change its kind."""
    t = ctx.triggered_id
    nu = dash.no_update
    if not t or not ctx.triggered:
        return nu, nu
    g, p = t["g"], t["p"]
    val = ctx.triggered[0]["value"]
    try:
        if t["type"] == "fold-kind":
            cur = next((f for f in paths.connected(g) if f["path"] == p), None)
            if cur is None or cur["kind"] == val:       # (re)drawn, not changed
                return nu, nu
            paths.update_folder(g, p, kind=val)
            return f"{p}: kind set to '{paths.FOLDER_GROUPS[g][val]}'.", (ver or 0) + 1
        if not val:
            return nu, nu
        if t["a"] == "open":
            open_in_system(p)
            return f"Opened {p}.", nu
        if t["a"] == "default":
            paths.make_default(g, p)
            return f"Tools now use {p}.", (ver or 0) + 1
        if t["a"] == "disconnect":
            paths.disconnect_folder(g, p)
            return f"Disconnected {p} (its files are untouched).", (ver or 0) + 1
    except Exception as exc:
        return f"⚠ {exc}", nu
    return nu, nu


@app.callback(
    Output({"type": "fold-list", "g": ALL}, "children"),
    Input("fold-ver", "data"),
    State({"type": "fold-list", "g": ALL}, "id"),
    prevent_initial_call=True,
)
def redraw_folders(_ver, ids):
    return [_folder_rows(i["g"]) for i in ids]


@app.callback(
    Output("msg", "children", allow_duplicate=True),
    Input("ss-run", "n_clicks"),
    Input("ls-run", "n_clicks"),
    State("ss-in", "value"), State("ss-out", "value"), State("ss-window", "value"),
    State("ls-file", "value"),
    prevent_initial_call=True,
)
def run_forms(_s, _l, ss_in, ss_out, ss_window, ls_file):
    """The 'Run' buttons of the tool forms."""
    t = ctx.triggered_id
    try:
        if t == "ss-run":
            if not ss_in or not ss_out:
                return "⚠ Sum spectra: give the input and output folders."
            try:
                window = float(ss_window)
            except (TypeError, ValueError):
                window = 0.0
            if not window > 0:
                return "⚠ Sum spectra: the peak tracking window must be a number above 0 Hz."
            return start_tool("sum_spectra", [os.path.expanduser(ss_in), os.path.expanduser(ss_out),
                                              "--window", f"{window:g}"])
        if t == "ls-run":
            if not ls_file:
                return "⚠ No scheme files in the connected level-scheme folders (Folders below)."
            return start_tool("levelschemer", [ls_file])
    except Exception as exc:
        return f"⚠ {exc}"
    return dash.no_update


# Settings-file problems (core/jsonfile.py): damaged files set aside by any
# tool (found as *.corrupt-* files, shown until they are deleted) and failed
# saves of the home page itself (last few)
_SAVE_WARNINGS = []


def _damaged_settings_files():
    folders = {paths.DATA_ROOT, os.path.dirname(paths.settings_file("physics.json")),
               paths.outputs("identified")}           # Segrè chart's identified list
    out = []
    for f in sorted(folders):
        out += jsonfile.damaged_files(f)
    return out


@app.callback(
    Output("settings-banner", "children"),
    Output("settings-banner", "style"),
    Input("tick", "n_intervals"),
)
def settings_banner(_n):
    _SAVE_WARNINGS.extend(jsonfile.take_warnings())
    del _SAVE_WARNINGS[:-5]
    try:
        damaged = _damaged_settings_files()
    except Exception:
        damaged = []
    if not damaged and not _SAVE_WARNINGS:
        return [], {"display": "none"}
    kids = []
    if damaged:
        kids.append(html.Div([
            html.B("Damaged settings files were set aside: "),
            "the tools used the backup (.bak) or their defaults instead. Check them, then "
            "delete them to clear this message."]))
        kids.append(html.Ul([html.Li(html.Code(p)) for p in damaged[:10]]
                            + ([html.Li(f"… and {len(damaged) - 10} more")]
                               if len(damaged) > 10 else [])))
    for w in _SAVE_WARNINGS:
        kids.append(html.Div(["Settings file: ", w]))
    return kids, {}


@app.callback(
    Output({"type": "status", "k": ALL}, "children"),
    Input("tick", "n_intervals"),
    State({"type": "status", "k": ALL}, "id"),
)
def refresh_status(_n, ids):
    return [_status_children(*tool_status(i["k"])) for i in ids]


@app.callback(
    Output({"type": "phys", "k": "corr_a"}, "value"),
    Output({"type": "phys", "k": "corr_b"}, "value"),
    Output({"type": "phys", "k": "corr_c"}, "value"),
    Output({"type": "phys", "k": "corr_a"}, "disabled"),
    Output({"type": "phys", "k": "corr_b"}, "disabled"),
    Output({"type": "phys", "k": "corr_c"}, "disabled"),
    Output({"type": "phys", "k": "corr_profile"}, "options"),
    Output("phys-corr-info", "children"),
    Input({"type": "phys", "k": "corr_profile"}, "value"),
    Input({"type": "phys", "k": "corr_a"}, "value"),
    Input({"type": "phys", "k": "corr_b"}, "value"),
    Input({"type": "phys", "k": "corr_c"}, "value"),
    Input({"type": "phys", "k": "ref_name"}, "value"),
    Input({"type": "phys", "k": "ref_harmonic"}, "value"),
    Input({"type": "phys", "k": "ref_frequency"}, "value"),
    Input({"type": "phys", "k": "gamma_t"}, "value"),
    Input({"type": "phys", "k": "harmonics"}, "value"),
    Input("tick", "n_intervals"),
)
def corr_profile(name, a, b, c, ref_name, ref_h, ref_f, gamma_t, harmonics, _tick):
    """Correction profile dropdown: a profile fills a, b, c with its general values
    (boxes locked); 'custom' unlocks them.  Shows the profile's fit, its
    per-harmonic coefficients and warnings if it was fitted with another setup."""
    nu = dash.no_update
    profs = corrections.load_profiles()
    opts = corrections.options(profs)
    p = next((q for q in profs if q["name"] == name), None) if name != corrections.CUSTOM else None
    out_abc = [nu, nu, nu]
    if p is not None:
        if not all(corrections._same(x, y) for x, y in zip((a, b, c), (p["a"], p["b"], p["c"]))):
            out_abc = [physics._fmt_num(p["a"]), physics._fmt_num(p["b"]), physics._fmt_num(p["c"])]
        a, b, c = p["a"], p["b"], p["c"]
        try:
            hs = physics.canonical("harmonics", harmonics)
        except (TypeError, ValueError):
            hs = None
        warn = corrections.warnings(p, ref_name, ref_h, gamma_t, ref_f, a, b, c, harmonics=hs)
        info = ([html.Div(corrections.describe(p))]
                + [html.Div(t) for t in corrections.coefficient_lines(p)]
                + [html.Div("⚠ " + w, style={"color": "#b45309"}) for w in warn])
    else:
        info = ("Profile no longer exists – " if name and name != corrections.CUSTOM else "") + \
            "Values typed in (no profile), used for every harmonic."
    lock = p is not None
    return (*out_abc, lock, lock, lock, opts, info)


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
    Output("notes-msg", "children"), Output("notes-log", "children"), Output("notes", "value"),
    Input("notes-add", "n_clicks"),
    State("notes", "value"),
    prevent_initial_call=True,
)
def add_note(_n, text):
    """Append the text as a time-stamped entry under today's heading."""
    try:
        when = labbook.add_entry(paths.notes_file(), text, experiment=paths.experiment())
    except ValueError as exc:
        return f"⚠ Not added: {exc}.", dash.no_update, dash.no_update
    except OSError as exc:
        return f"⚠ Not added: {exc}", dash.no_update, dash.no_update
    return f"Added {when:%H:%M}.", _labbook_log(), ""


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
