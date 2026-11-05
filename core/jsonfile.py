"""
Safe reading and writing of the suite's JSON settings files.

Every tool keeps its settings (and some saved state) in small JSON files,
mostly in data/<experiment>/settings/.  This module makes sure a damaged
file is never silently replaced and a failed save is never silently lost:

  read(path, default)
      Missing file  -> `default` (a fresh copy), as before.
      Damaged file  -> (not valid JSON, or not the expected type) the file is
                       moved aside to  <name>.corrupt-<YYYYmmdd_HHMMSS>  so
                       nothing overwrites it, and a warning is recorded.  If a
                       backup  <name>.bak  (the previous good version) is
                       readable it is restored and used; otherwise `default`.
      Unreadable    -> (permissions, I/O error) `default` + a warning; the
                       file is left where it is.

  write(path, data)
      Keeps the current file (if it is valid JSON) as  <name>.bak,  writes to a
      uniquely named temporary file in the same folder and renames it into
      place, so a crash never leaves half a file and two writers never share
      a temporary file.  Raises OSError / TypeError on failure.

  lock(path)
      Context manager serialising a read-modify-write of `path`: a lock per
      file shared by the threads of this process, plus an flock on a hidden
      lock file (.<name>.lock) shared with the other Asphodel processes (the
      home page and the tools run as separate programs).

  update(path, changes, default=None)
      read + dict.update + write under the lock.

  warn(text) / take_warnings()
      Problems are printed to the terminal and queued; a GUI shows them with
      take_warnings() (Theia: a box at the top right).  damaged_files(folder)
      lists the  *.corrupt-*  files left in a folder (home page banner).
"""

import contextlib
import datetime
import glob
import json
import os
import shutil
import sys
import tempfile
import threading

try:
    import fcntl
except ImportError:                     # not on Linux / macOS: in-process lock only
    fcntl = None

BAK_SUFFIX = ".bak"
CORRUPT_TAG = ".corrupt-"

_GUARD = threading.Lock()
_LOCKS = {}                             # absolute path -> threading.RLock
_HELD = threading.local()               # per thread: {path: depth} of flocks held
_WARNINGS = []                          # queued for take_warnings()


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------

def warn(text):
    """Report a settings-file problem: printed now, shown later by a GUI."""
    print(f"Asphodel settings: {text}", file=sys.stderr, flush=True)
    with _GUARD:
        _WARNINGS.append(text)


def take_warnings():
    """Warnings recorded since the last call (oldest first)."""
    with _GUARD:
        out = list(_WARNINGS)
        _WARNINGS.clear()
    return out


def damaged_files(folder):
    """Paths of the damaged files set aside in `folder` (newest first)."""
    if not folder or not os.path.isdir(folder):
        return []
    hits = glob.glob(os.path.join(folder, f"*{CORRUPT_TAG}*"))
    return sorted(hits, key=os.path.getmtime, reverse=True)


# ---------------------------------------------------------------------------
# Locking
# ---------------------------------------------------------------------------

def _thread_lock(path):
    with _GUARD:
        lk = _LOCKS.get(path)
        if lk is None:
            lk = _LOCKS[path] = threading.RLock()
        return lk


@contextlib.contextmanager
def lock(path):
    """Hold the lock of `path` (re-entrant within a thread)."""
    path = os.path.abspath(path)
    with _thread_lock(path):
        held = getattr(_HELD, "d", None)
        if held is None:
            held = _HELD.d = {}
        if held.get(path):                      # this thread already holds the flock
            held[path] += 1
            try:
                yield
            finally:
                held[path] -= 1
            return
        fh = None
        if fcntl is not None:
            folder, name = os.path.split(path)
            try:
                os.makedirs(folder, exist_ok=True)
                fh = open(os.path.join(folder, f".{name}.lock"), "a")
                fcntl.flock(fh, fcntl.LOCK_EX)
            except OSError:
                if fh is not None:
                    fh.close()
                fh = None                       # e.g. read-only folder: thread lock only
        held[path] = 1
        try:
            yield
        finally:
            held.pop(path, None)
            if fh is not None:
                try:
                    fcntl.flock(fh, fcntl.LOCK_UN)
                finally:
                    fh.close()


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def _copy_default(default):
    return json.loads(json.dumps(default)) if default is not None else None


def _parse(path, kind):
    """Value of the JSON file at path; ValueError if it is not valid JSON or
    not of type `kind` (None = any type)."""
    with open(path, encoding="utf-8") as fh:
        value = json.load(fh)
    if kind is not None and not isinstance(value, kind):
        raise ValueError(f"expected a JSON {kind.__name__}, found {type(value).__name__}")
    return value


def read(path, default=None, kind=dict):
    """Contents of a JSON settings file (see the module docstring)."""
    try:
        return _parse(path, kind)
    except FileNotFoundError:
        return _copy_default(default)
    except OSError as exc:
        warn(f"could not read {path} ({exc}); using defaults for now")
        return _copy_default(default)
    except ValueError as exc:
        reason = str(exc)
    # damaged: set it aside (under the lock, re-checking: another writer may
    # have replaced it meanwhile)
    with lock(path):
        try:
            return _parse(path, kind)
        except FileNotFoundError:
            return _copy_default(default)
        except (OSError, ValueError):
            pass
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        aside = f"{path}{CORRUPT_TAG}{stamp}"
        n = 1
        while os.path.exists(aside):
            n += 1
            aside = f"{path}{CORRUPT_TAG}{stamp}_{n}"
        try:
            os.replace(path, aside)
        except OSError as exc:
            warn(f"{path} is damaged ({reason}) and could not be set aside ({exc}); "
                 f"using defaults for now")
            return _copy_default(default)
        name = os.path.basename(path)
        bak = path + BAK_SUFFIX
        try:
            value = _parse(bak, kind)
        except (OSError, ValueError):
            warn(f"{name} was damaged ({reason}); it was moved to {os.path.basename(aside)} "
                 f"and defaults are used (no backup to restore)")
            return _copy_default(default)
        try:
            shutil.copy2(bak, path)
            how = "restored"
        except OSError:
            how = "used (could not copy it back into place)"
        warn(f"{name} was damaged ({reason}); it was moved to {os.path.basename(aside)} and "
             f"the previous version ({name}{BAK_SUFFIX}) was {how}")
        return value


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def write(path, data, text=None, ensure_ascii=True, indent=2):
    """Write `data` as JSON (or `text`, already-formatted JSON) to path,
    keeping the previous good version as <path>.bak.  Raises on failure."""
    if text is None:
        text = json.dumps(data, indent=indent, ensure_ascii=ensure_ascii)
    if not text.endswith("\n"):
        text += "\n"
    folder = os.path.dirname(os.path.abspath(path))
    with lock(path):
        os.makedirs(folder, exist_ok=True)
        if os.path.exists(path):
            try:
                _parse(path, None)               # back up only a good version
                shutil.copy2(path, path + BAK_SUFFIX)
            except (OSError, ValueError):
                pass
        fd, tmp = tempfile.mkstemp(dir=folder, prefix=f".{os.path.basename(path)}.",
                                   suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
                fh.flush()
                os.fsync(fh.fileno())
            if os.path.exists(path):
                shutil.copymode(path, tmp)
            else:
                os.chmod(tmp, 0o644)             # mkstemp makes it private (0600)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


def update(path, changes, default=None):
    """Merge `changes` into the JSON object at path (read + update + write,
    under the lock).  Returns the merged dict."""
    with lock(path):
        d = read(path, default if default is not None else {})
        d.update(changes)
        write(path, d)
        return d


def try_write(path, data, what=None, **kw):
    """write(), reporting a failure with warn() instead of raising.
    Returns True if the file was written."""
    try:
        write(path, data, **kw)
        return True
    except Exception as exc:
        warn(f"could not save {what or os.path.basename(path)} to {path}: "
             f"{type(exc).__name__}: {exc}")
        return False
