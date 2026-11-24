"""
Where Asphodel keeps its own files: the "workspace".

The workspace holds what the suite writes itself: asphodel_settings.json
(active experiment, experiment profiles), each experiment's folder with its
settings/ (physics.json, folders.json, tool settings), default outputs/ and
notes.md, and logs/.  Input data can be anywhere: it is connected per
experiment on the home page.

Which folder is the workspace, first match wins:

  1. the environment variable ASPHODEL_WORKSPACE (one run, e.g. tests)
  2. <repo>/.asphodel_local.json  {"workspace": "/path/to/folder", ...}
     (written by the first-run setup, core/firstrun.py; git ignores it)
  3. <repo>/data, if that folder exists (the layout before the suite was
     installable: existing setups keep working without any change)

The choice is kept inside the code folder (the git clone), not in the home
folder, so every clone is its own installation: a second clone (e.g. to test
a fresh install) has its own first-run setup and workspace and never changes
the settings of another clone.

If none of these gives a folder, workspace() returns None and home.py runs the
first-run setup, which suggests DEFAULT_WORKSPACE.

This module imports nothing from core, so it can be used before core.paths
(which reads the workspace when it is imported).
"""
import json
import os

ENV_WORKSPACE = "ASPHODEL_WORKSPACE"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEGACY_WORKSPACE = os.path.join(REPO_ROOT, "data")
DEFAULT_WORKSPACE = os.path.join(os.path.expanduser("~"), "Asphodel-data")


def config_file():
    """This clone's own settings (workspace, port offset); git ignores it."""
    return os.path.join(REPO_ROOT, ".asphodel_local.json")


_OLD_CONFIG_FILE = os.path.join(REPO_ROOT, ".asphodel_workspace.json")   # older name of the file


def _read_config():
    for f in (config_file(), _OLD_CONFIG_FILE):
        try:
            with open(f) as fh:
                d = json.load(fh)
            return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            continue
    return {}


def _write_config(d):
    tmp = config_file() + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(d, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, config_file())


def _norm(p):
    return os.path.abspath(os.path.expanduser(p))


def workspace():
    """The workspace folder (absolute path), or None if none is set up yet."""
    env = (os.environ.get(ENV_WORKSPACE) or "").strip()
    if env:
        return _norm(env)
    cfg = (_read_config().get("workspace") or "").strip()
    if cfg:
        return _norm(cfg)
    if os.path.isdir(LEGACY_WORKSPACE):
        return LEGACY_WORKSPACE
    return None


def source():
    """Where workspace() comes from: 'env', 'config', 'legacy' or None."""
    if (os.environ.get(ENV_WORKSPACE) or "").strip():
        return "env"
    if (_read_config().get("workspace") or "").strip():
        return "config"
    if os.path.isdir(LEGACY_WORKSPACE):
        return "legacy"
    return None


def set_workspace(path):
    """Remember `path` as this clone's workspace (in <repo>/.asphodel_local.json)
    and create the folder. Returns the absolute path."""
    p = _norm(path)
    os.makedirs(p, exist_ok=True)
    d = _read_config()
    d["workspace"] = p
    _write_config(d)
    return p


# --------------------------------------------------------------------------- ports
#
# Every tool that is a web page listens on a fixed port (core/ports.py). Two
# copies of Asphodel (e.g. your working copy and a test clone) can run at the
# same time when they use different ports: each clone adds its port offset.
# The working copy has offset 0; install.sh gives a second copy the next free
# multiple of 100 (8150, 8152, ... instead of 8050, 8052, ...).

ENV_PORT_OFFSET = "ASPHODEL_PORT_OFFSET"


def port_offset():
    """This clone's port offset: ASPHODEL_PORT_OFFSET, else .asphodel_local.json, else 0."""
    for v in (os.environ.get(ENV_PORT_OFFSET), _read_config().get("port_offset")):
        try:
            if v is not None and str(v).strip() != "":
                return int(v)
        except (TypeError, ValueError):
            pass
    return 0


def offset_of_clone(repo):
    """The port offset saved in another clone (0 if it has none)."""
    for name in (".asphodel_local.json", ".asphodel_workspace.json"):
        try:
            with open(os.path.join(repo, name)) as fh:
                return int((json.load(fh) or {}).get("port_offset") or 0)
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    return 0


def assign_port_offset(other_clones):
    """Give this clone a port offset that none of `other_clones` uses (keeps
    the current one if it is already free). Returns the offset."""
    used = {offset_of_clone(r) for r in other_clones
            if os.path.realpath(r) != os.path.realpath(REPO_ROOT)}
    cur = int(_read_config().get("port_offset") or 0)
    if cur not in used:
        off = cur
    else:
        off = 100
        while off in used:
            off += 100
    d = _read_config()
    d["port_offset"] = off
    _write_config(d)
    return off


if __name__ == "__main__":
    # python -m core.workspace assign-port-offset <other clone folder> ...   (install.sh)
    import sys
    if sys.argv[1:2] == ["assign-port-offset"]:
        print(assign_port_offset(sys.argv[2:]))
    else:
        print(f"workspace: {workspace()} ({source()}); port offset: {port_offset()}")
