"""
Synchronise files between this computer and APPC256 (scp via lxlogin).

    push   outputs/confirmed/*.csv                    ->  <remote_root>/confirmed/
    pull   <remote_root>/summed_spectrograms/*.npz    ->  spectrograms/   (picked by hand)

"New" means the file name does not exist on the other side yet. Nothing is
ever overwritten or deleted on either side. A pull downloads into
<name>.part and renames it when complete, so an interrupted pull never
leaves a truncated spectrogram behind.

Connection
----------
Uses the "appc256" entry of ~/.ssh/config (ProxyJump lxlogin, ControlMaster).
The lxlogin password is passed to ssh through a one-off SSH_ASKPASS helper in
a private temporary folder (mode 0700, removed after every command). The helper
answers at most one password prompt per command, so a wrong password is not
retried. If a connection to appc256 is already open (ControlPersist keeps it
for 4 h, also when it was opened in a terminal), no password is needed.

Settings (optional, per experiment): <experiment>/settings/sync.json, e.g.

    {"host": "appc256",
     "remote_root": "/home/gpuuser/personall_directories/zac/python-scripts/Tartarus",
     "remote_confirmed": "confirmed",
     "remote_spectrograms": "summed_spectrograms"}

Missing keys fall back to DEFAULTS below.
"""

import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
import threading
import time

from core import paths

DEFAULTS = {
    "host": "appc256",
    "remote_root": "/home/gpuuser/personall_directories/zac/python-scripts/Tartarus",
    "remote_confirmed": "confirmed",
    "remote_spectrograms": "summed_spectrograms",
}
SSH_OPTS = ["-o", "ConnectTimeout=30", "-o", "ServerAliveInterval=30", "-o", "ForwardX11=no"]
LIST_TIMEOUT = 120          # s, for listing and for pushing the (small) confirmed lists
FREE_MARGIN = 2e9           # bytes kept free on this disk when pulling
PART = ".part"

# names / paths that go through the remote shell unquoted (scp's remote side)
_SAFE_NAME = re.compile(r"^[A-Za-z0-9._+=,@%-]+$")
_SAFE_PATH = re.compile(r"^[A-Za-z0-9._+=,@%/~-]+$")

ASKPASS = r"""#!/bin/sh
# Asphodel sync: answers one password prompt per ssh/scp command (see core/sync.py)
d="$ASPHODEL_ASKPASS_DIR"
case "$1" in *assword*) ;; *) exit 1 ;; esac
[ -s "$d/pw" ] || exit 1
mkdir "$d/used" 2>/dev/null || exit 1
cat "$d/pw"
echo
"""


class SyncError(Exception):
    pass


def size_text(n):
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if n < 1000 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000


def settings(exp=None):
    """Sync settings of the experiment (DEFAULTS + settings/sync.json)."""
    cfg = dict(DEFAULTS)
    try:
        with open(paths.settings_file("sync.json", exp), encoding="utf-8") as fh:
            d = json.load(fh)
        cfg.update({k: v.strip() for k, v in d.items()
                    if k in DEFAULTS and isinstance(v, str) and v.strip()})
    except (OSError, ValueError, AttributeError):
        pass
    for k in ("remote_root", "remote_confirmed", "remote_spectrograms"):
        if not _SAFE_PATH.match(cfg[k]):
            raise SyncError(f"sync.json: {k} = {cfg[k]!r} contains characters scp can't take unquoted.")
    return cfg


def remote_dir(cfg, key):
    return cfg["remote_root"].rstrip("/") + "/" + cfg[key].strip("/")


# --------------------------------------------------------------------------- running ssh / scp

class _Askpass:
    """Environment for one ssh/scp command that answers one password prompt."""

    def __init__(self, password):
        self.password = password or ""
        self.asked = False

    def __enter__(self):
        self.dir = tempfile.mkdtemp(prefix="asphodel_ssh_")          # mode 0700
        fd = os.open(os.path.join(self.dir, "pw"), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write(self.password)
        script = os.path.join(self.dir, "askpass.sh")
        with open(script, "w") as fh:
            fh.write(ASKPASS)
        os.chmod(script, 0o700)
        env = dict(os.environ, SSH_ASKPASS=script, SSH_ASKPASS_REQUIRE="force",
                   ASPHODEL_ASKPASS_DIR=self.dir)
        env.setdefault("DISPLAY", ":0")     # older OpenSSH only uses askpass with a DISPLAY
        return env

    def __exit__(self, *exc):
        self.asked = os.path.isdir(os.path.join(self.dir, "used"))
        shutil.rmtree(self.dir, ignore_errors=True)


def _explain(err, password, asked):
    lines = [l.strip() for l in err.splitlines() if l.strip()]
    tail = " | ".join(lines[-3:]) or "no error message"
    if "Permission denied" in err or "Authentication failed" in err or "Too many authentication" in err:
        if not password:
            return ("Login needed: enter the lxlogin password (there is no open connection to "
                    "appc256 to reuse).")
        if asked:
            return f"Login failed – wrong password? ({tail})"
    return tail


def _kill(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass


def _run(cmd, password, timeout=None, on_start=None):
    """Run ssh/scp without a terminal; return stdout, raise SyncError on failure.

    Output goes to temporary files rather than pipes: a ControlPersist master
    started by this command lives on in the background and would keep a pipe open.
    """
    ap = _Askpass(password)
    with ap as env, tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                    env=env, start_new_session=True)
        except FileNotFoundError:
            raise SyncError(f"{cmd[0]} not found on this computer.")
        if on_start:
            on_start(proc)
        try:
            rc = proc.wait(timeout)
        except subprocess.TimeoutExpired:
            _kill(proc)
            proc.wait()
            raise SyncError(f"{cmd[0]} gave no answer within {timeout} s (network or lxlogin down?).")
        out.seek(0)
        err.seek(0)
        stdout = out.read().decode(errors="replace")
        stderr = err.read().decode(errors="replace")
    if rc < 0:
        raise SyncError("cancelled")
    if rc != 0:
        raise SyncError(_explain(stderr, password, ap.asked))
    return stdout


# --------------------------------------------------------------------------- what is where

def remote_listing(password, cfg):
    """{"confirmed": {name: size} or None, "spectrograms": {...} or None} on the remote
    machine (None: folder missing). Only plain files at the top of each folder."""
    keys = {"confirmed": "remote_confirmed", "spectrograms": "remote_spectrograms"}
    parts = []
    for k, ck in keys.items():
        d = shlex.quote(remote_dir(cfg, ck))
        parts.append(f"echo '@@{k}'; if [ -d {d} ]; then find {d} -mindepth 1 -maxdepth 1 -type f "
                     f"-printf '%f\\t%s\\n'; else echo '@@missing'; fi")
    text = _run(["ssh", *SSH_OPTS, cfg["host"], "; ".join(parts)], password, LIST_TIMEOUT)
    res, cur = {}, None
    for line in text.splitlines():
        if line.startswith("@@") and line[2:] in keys:
            cur = line[2:]
            res[cur] = {}
        elif line == "@@missing" and cur:
            res[cur] = None
        elif cur and res[cur] is not None and "\t" in line:
            name, _, size = line.rpartition("\t")
            res[cur][name] = int(size) if size.isdigit() else 0
    if set(res) != set(keys):
        raise SyncError("Unexpected answer from appc256: " + (text.strip()[-200:] or "nothing"))
    return res


def local_confirmed(exp=None):
    d = paths.outputs("confirmed", exp)
    return sorted(n for n in os.listdir(d) if n.endswith(".csv") and os.path.isfile(os.path.join(d, n)))


def local_spectrograms(exp=None):
    d = paths.spectrograms(exp)
    os.makedirs(d, exist_ok=True)
    return {n for n in os.listdir(d) if os.path.isfile(os.path.join(d, n)) and not n.endswith(PART)}


def check(password, exp=None):
    """Compare both sides. Returns a dict for the home page."""
    exp = exp or paths.experiment()
    cfg = settings(exp)
    rem = remote_listing(password, cfg)
    loc_conf = local_confirmed(exp)
    loc_spec = local_spectrograms(exp)
    rc, rs = rem["confirmed"], rem["spectrograms"]
    return {
        "exp": exp,
        "cfg": cfg,
        "remote_confirmed_missing": rc is None,
        "remote_spectrograms_missing": rs is None,
        "to_push": [n for n in loc_conf if rc is not None and n not in rc],
        "remote_only_confirmed": sorted(n for n in (rc or {}) if n not in loc_conf),
        "to_pull": sorted([n, s] for n, s in (rs or {}).items()
                          if n.endswith(".npz") and n not in loc_spec),
        "free": shutil.disk_usage(paths.spectrograms(exp)).free,
    }


# --------------------------------------------------------------------------- push

def push_confirmed(names, password, exp=None):
    """Copy the named confirmed lists to the remote confirmed folder (one scp)."""
    if not names:
        return []
    cfg = settings(exp)
    d = paths.outputs("confirmed", exp)
    files = [os.path.join(d, n) for n in names]
    _run(["scp", "-p", "-q", *SSH_OPTS, *files, f"{cfg['host']}:{remote_dir(cfg, 'remote_confirmed')}/"],
         password, LIST_TIMEOUT)
    return list(names)


# --------------------------------------------------------------------------- pull (background)

class PullJob:
    """Pulls spectrograms one by one in a background thread; one job at a time."""

    def __init__(self):
        self._lock = threading.Lock()
        self._proc = None
        self._cancel = False
        self.state = {"running": False}

    def start(self, items, password, exp):
        """items: [[name, size], ...] as returned by check()."""
        with self._lock:
            if self.state.get("running"):
                raise SyncError("A pull is already running.")
            if not items:
                raise SyncError("Nothing selected.")
            for name, _ in items:
                if not _SAFE_NAME.match(name):
                    raise SyncError(f"Can't pull {name!r}: unusual characters in the name.")
            dest = paths.spectrograms(exp)
            need = sum(s for _, s in items)
            free = shutil.disk_usage(dest).free
            if need + FREE_MARGIN > free:
                raise SyncError(f"Not enough space: the selection needs {size_text(need)}, "
                                f"{size_text(free)} is free (keeping {size_text(FREE_MARGIN)} spare).")
            self._cancel = False
            self.state = {"running": True, "exp": exp, "dest": dest, "items": items, "index": 0,
                          "done": [], "skipped": [], "error": None, "cancelled": False,
                          "started": time.time(), "finished": None}
        threading.Thread(target=self._work, args=(items, password, exp, dest), daemon=True).start()

    def _work(self, items, password, exp, dest):
        try:
            cfg = settings(exp)
            src = remote_dir(cfg, "remote_spectrograms")
            for i, (name, size) in enumerate(items):
                with self._lock:
                    self.state["index"] = i
                    if self._cancel:
                        break
                final = os.path.join(dest, name)
                part = final + PART
                if os.path.exists(final):
                    self.state["skipped"].append(name)
                    continue
                try:
                    _run(["scp", "-p", "-q", *SSH_OPTS, f"{cfg['host']}:{src}/{name}", part],
                         password, None, on_start=self._set_proc)
                    got = os.path.getsize(part)
                    if size and got != size:
                        raise SyncError(f"{name}: got {got} bytes, expected {size}")
                    if os.path.exists(final):          # appeared meanwhile: keep the existing one
                        os.remove(part)
                        self.state["skipped"].append(name)
                    else:
                        os.replace(part, final)
                        self.state["done"].append(name)
                except Exception:
                    try:
                        os.remove(part)
                    except OSError:
                        pass
                    raise
        except Exception as exc:
            with self._lock:
                if self._cancel:
                    self.state["cancelled"] = True
                else:
                    self.state["error"] = str(exc)
        finally:
            with self._lock:
                self.state["cancelled"] = self.state["cancelled"] or self._cancel
                self.state["running"] = False
                self.state["finished"] = time.time()
                self._proc = None

    def _set_proc(self, proc):
        with self._lock:
            self._proc = proc
            if self._cancel:
                _kill(proc)

    def cancel(self):
        with self._lock:
            if not self.state.get("running"):
                return False
            self._cancel = True
            if self._proc is not None:
                _kill(self._proc)
        return True

    def status(self):
        """Copy of the state plus the bytes received of the current file."""
        with self._lock:
            st = json.loads(json.dumps(self.state))
        if st.get("running"):
            name, size = st["items"][st["index"]]
            try:
                got = os.path.getsize(os.path.join(st["dest"], name + PART))
            except OSError:
                got = 0
            st.update(current=name, current_size=size, current_got=got)
        return st


JOB = PullJob()
