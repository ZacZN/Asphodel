"""
Where Asphodel keeps its own files: the "workspace".

The workspace holds what the suite writes itself: asphodel_settings.json
(active experiment, experiment profiles), each experiment's folder with its
settings/ (physics.json, folders.json, tool settings), default outputs/ and
notes.md, and logs/.  Input data can be anywhere: it is connected per
experiment on the home page.

Which folder is the workspace, first match wins:

  1. the environment variable ASPHODEL_WORKSPACE (one run, e.g. tests)
  2. <repo>/.asphodel_workspace.json  {"workspace": "/path/to/folder"}
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
    """The file that remembers this clone's workspace."""
    return os.path.join(REPO_ROOT, ".asphodel_workspace.json")


def _read_config():
    try:
        with open(config_file()) as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


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
    """Remember `path` as this clone's workspace (in <repo>/.asphodel_workspace.json)
    and create the folder. Returns the absolute path."""
    p = _norm(path)
    os.makedirs(p, exist_ok=True)
    d = _read_config()
    d["workspace"] = p
    tmp = config_file() + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(d, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, config_file())
    return p
