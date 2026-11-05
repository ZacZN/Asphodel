"""
Output capture for the Asphodel tools: everything a tool writes to the
terminal (print, warnings, tracebacks, messages from libraries and from
programs it starts) also goes to a log file and to a buffer its own window
shows ("Log" / "Output" panel, core/logpanel.py).

    logcapture.start("theia")          # first thing a tool does when run

Started from a terminal, the output still appears there as well; started from
the Asphodel launcher (no terminal) the log file and the panel are where it
can be read.

Files: data/logs/<tool>_<YYYYmmdd_HHMMSS>_<pid>.log, one per run; the last
KEEP runs of each tool are kept (older ones are deleted when a new run starts).

How: file descriptors 1 and 2 (stdout, stderr) are pointed at pipes; a
background thread copies everything from the pipes to the original
stdout/stderr (terminal, or wherever the launcher sent them), the log file and
an in-memory buffer of the latest lines.  Working on the descriptors, not only
on sys.stdout, also catches output from C libraries, from logging handlers
made before start() and from child processes.  A tool that starts another
Asphodel tool passes child_stdio() to subprocess.Popen, so that tool's output
goes to its own log rather than into this one as well.
"""

import atexit
import collections
import datetime
import glob
import os
import re
import sys
import threading

from . import paths

LOG_DIR = os.path.join(paths.DATA_ROOT, "logs")
KEEP = 10                         # runs kept per tool
BUFFER_LINES = 5000               # lines kept in memory for the panel

_STATE = {"tool": None, "path": None, "file": None, "saved": {}, "threads": []}
_LOCK = threading.Lock()
_LINES = collections.deque(maxlen=BUFFER_LINES)   # (index, "HH:MM:SS", stream, level, text)
_COUNTER = {"next": 0, "warning": 0, "error": 0}

_ERROR_RE = re.compile(r"Traceback \(most recent call last\)|\b\w*(Error|Exception)\b[:(]|"
                       r"\bfailed\b|\bFAILED\b|\bCRITICAL\b|\bFatal\b|\bAborted\b")
_WARN_RE = re.compile(r"\bwarn(ing|ings|ed)?\b|\bWARNING\b|Asphodel settings:|\bdeprecat", re.I)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start(tool):
    """Start capturing this process's output for `tool` (once per process).
    Returns the log file path, or None if capture could not be set up (the
    output then goes only where it went before)."""
    with _LOCK:
        if _STATE["tool"] is not None:
            return _STATE["path"]
        _STATE["tool"] = tool
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        _prune(tool)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(LOG_DIR, f"{tool}_{stamp}_{os.getpid()}.log")
        fh = open(path, "a", encoding="utf-8", errors="replace", buffering=1)
        fh.write(f"# Asphodel {tool} · started {datetime.datetime.now():%Y-%m-%d %H:%M:%S} · "
                 f"pid {os.getpid()} · {' '.join(sys.argv)}\n")
    except OSError as exc:
        print(f"Asphodel: could not open a log file in {LOG_DIR}: {exc}", file=sys.stderr)
        return None
    _STATE.update(path=path, file=fh)
    for stream, fd in (("out", 1), ("err", 2)):
        try:
            _capture_fd(stream, fd)
        except OSError as exc:
            _add_line(stream if stream == "err" else "err",
                      f"Asphodel: could not capture {stream}: {exc}")
    # print promptly (not only when a buffer fills)
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass
    atexit.register(_stop)
    return path


def log_path():
    """This run's log file (None before start())."""
    return _STATE["path"]


def tool():
    return _STATE["tool"]


def lines(since=0, limit=None):
    """(next index, [(index, time, stream, level, text)]) of the buffered
    lines with index >= since (the last `limit` of them if given)."""
    with _LOCK:
        out = [ln for ln in _LINES if ln[0] >= since]
        nxt = _COUNTER["next"]
    if limit is not None:
        out = out[-limit:]
    return nxt, out


def counts():
    """{"warning": n, "error": n} lines seen since start()."""
    with _LOCK:
        return {"warning": _COUNTER["warning"], "error": _COUNTER["error"]}


def child_stdio():
    """Popen keyword arguments that give a child process this process's
    original stdout/stderr (not the capture pipes), e.g.
    subprocess.Popen(cmd, **logcapture.child_stdio())."""
    saved = _STATE["saved"]
    out = {}
    if "out" in saved:
        out["stdout"] = saved["out"]
    if "err" in saved:
        out["stderr"] = saved["err"]
    return out


def log_files():
    """[(path, tool, mtime)] of every log file, newest first."""
    out = []
    for p in glob.glob(os.path.join(LOG_DIR, "*.log")):
        name = os.path.basename(p)
        m = re.match(r"(.+?)_\d{8}_\d{6}_\d+\.log$", name)
        try:
            out.append((p, m.group(1) if m else name, os.path.getmtime(p)))
        except OSError:
            continue
    return sorted(out, key=lambda t: t[2], reverse=True)


def classify(stream, text, in_traceback=False):
    """("error" / "warning" / "", still in a traceback?) for one line."""
    if text.startswith("Traceback (most recent call last)"):
        return "error", True
    if in_traceback:
        if text.startswith((" ", "\t")) or not text.strip():
            return "error", True
        return "error", False                  # the final "SomeError: message" line
    if _ERROR_RE.search(text):
        return "error", False
    if _WARN_RE.search(text):
        return "warning", False
    return "", False


def read_file_lines(path, limit=400):
    """The last `limit` lines of a log file, classified like the buffer:
    [(index, time, stream, level, text)] (time and stream unknown: '')."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            raw = fh.read().splitlines()
    except OSError as exc:
        return [(0, "", "err", "error", f"could not read {path}: {exc}")]
    out, tb = [], False
    for i, ln in enumerate(raw):
        m = re.match(r"\[(\d\d:\d\d:\d\d)\]( !)? (.*)$", ln)
        t, stream, text = (m.group(1), "err" if m.group(2) else "out", m.group(3)) if m \
            else ("", "out", ln)
        level, tb = classify(stream, text, tb)
        out.append((i, t, stream, level, text))
    return out[-limit:]


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _prune(tool):
    """Delete this tool's oldest logs so that, with the new one, KEEP remain."""
    mine = [p for p, t, _ in log_files() if t == tool]
    for p in mine[KEEP - 1:]:
        try:
            os.remove(p)
        except OSError:
            pass


_TB = {"out": False, "err": False}


def _add_line(stream, text):
    now = datetime.datetime.now().strftime("%H:%M:%S")
    with _LOCK:
        level, _TB[stream] = classify(stream, text, _TB[stream])
        i = _COUNTER["next"]
        _COUNTER["next"] = i + 1
        if level:
            _COUNTER[level] += 1
        _LINES.append((i, now, stream, level, text))
        fh = _STATE["file"]
    if fh is not None:
        try:
            fh.write(f"[{now}]{' !' if stream == 'err' else ''} {text}\n")
        except (OSError, ValueError):
            pass


def _capture_fd(stream, fd):
    """Point fd at a pipe; a thread copies the pipe to the original fd, the
    log file and the buffer."""
    try:
        saved = os.dup(fd)
    except OSError:
        saved = None                            # fd was not open: nothing to echo to
    r, w = os.pipe()
    sys_stream = sys.stdout if fd == 1 else sys.stderr
    try:
        if sys_stream is not None:
            sys_stream.flush()
    except (OSError, ValueError):
        pass
    os.dup2(w, fd)
    os.close(w)
    if saved is not None:
        _STATE["saved"][stream] = saved

    def pump():
        partial = b""
        while True:
            try:
                chunk = os.read(r, 65536)
            except OSError:
                break
            if not chunk:
                break
            if saved is not None:
                try:
                    os.write(saved, chunk)
                except OSError:
                    pass
            partial += chunk
            *done, partial = partial.split(b"\n")
            for ln in done:
                _add_line(stream, ln.decode("utf-8", "replace").rstrip("\r"))
        if partial:
            _add_line(stream, partial.decode("utf-8", "replace"))
        os.close(r)

    t = threading.Thread(target=pump, name=f"logcapture-{stream}", daemon=True)
    t.start()
    _STATE["threads"].append((fd, saved, t))


def _stop():
    """At exit: flush, give fds 1/2 back (closing the pipes) and let the
    threads write the last lines."""
    for s in (sys.stdout, sys.stderr):
        try:
            if s is not None:
                s.flush()
        except (OSError, ValueError):
            pass
    for fd, saved, _ in _STATE["threads"]:
        try:
            if saved is not None:
                os.dup2(saved, fd)
            else:
                os.close(fd)
        except OSError:
            pass
    for _, _, t in _STATE["threads"]:
        t.join(timeout=1.0)                     # a child still holding the pipe: give up
    fh = _STATE["file"]
    if fh is not None:
        try:
            fh.write(f"# ended {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")
            fh.close()
        except (OSError, ValueError):
            pass
        _STATE["file"] = None
