"""
Where the Asphodel suite keeps its data, outputs and settings.

Layout (all under <repo>/data/, which git ignores):

    data/
      asphodel_settings.json      {"experiment": "E0203"}  - the active experiment
      <experiment>/
        spectra_1d/               1-D spectra (.npz)
        spectrograms/             summed 2-D spectrograms (.npz)
        summed_1d/                summed 1-D spectra (.npz)
        levelschemes/             LevelSchemer files (.json)
        rionid/                   RionID tables (.xlsx)
        settings/                 per-tool GUI settings (.json)
        outputs/scans|confirmed|fits|figures/

Every function takes an optional `exp`; without it the active experiment is used.
Folders that tools write into (outputs/<kind>, settings/) are created on demand.

Use from a script in a sub-folder (e.g. theia/scanner.py):

    import os, sys
    sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core import paths
"""

import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(REPO_ROOT, "data")
SUITE_SETTINGS = os.path.join(DATA_ROOT, "asphodel_settings.json")

OUTPUT_KINDS = ("scans", "confirmed", "fits", "figures")


# --------------------------------------------------------------------------- experiment

def _read_suite_settings():
    try:
        with open(SUITE_SETTINGS, encoding="utf-8") as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def list_experiments():
    """Experiment folders in data/ (sorted)."""
    try:
        return sorted(n for n in os.listdir(DATA_ROOT)
                      if not n.startswith(".") and os.path.isdir(os.path.join(DATA_ROOT, n)))
    except OSError:
        return []


def experiment():
    """Name of the active experiment.

    Read from data/asphodel_settings.json. If that file has no experiment and
    data/ holds exactly one experiment folder, that one is used.
    """
    name = (_read_suite_settings().get("experiment") or "").strip()
    if name:
        return name
    exps = list_experiments()
    if len(exps) == 1:
        return exps[0]
    raise RuntimeError(
        f"No active experiment set. Put {{\"experiment\": \"<name>\"}} in {SUITE_SETTINGS} "
        f"(experiments found in data/: {', '.join(exps) or 'none'}).")


def set_experiment(name):
    """Make `name` the active experiment (kept in data/asphodel_settings.json)."""
    d = _read_suite_settings()
    d["experiment"] = name
    os.makedirs(DATA_ROOT, exist_ok=True)
    with open(SUITE_SETTINGS, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=2)


# --------------------------------------------------------------------------- folders

def experiment_dir(exp=None):
    return os.path.join(DATA_ROOT, exp or experiment())


def spectra_1d(exp=None):
    return os.path.join(experiment_dir(exp), "spectra_1d")


def spectrograms(exp=None):
    return os.path.join(experiment_dir(exp), "spectrograms")


def summed_1d(exp=None):
    return os.path.join(experiment_dir(exp), "summed_1d")


def levelschemes(exp=None):
    return os.path.join(experiment_dir(exp), "levelschemes")


def rionid(exp=None):
    return os.path.join(experiment_dir(exp), "rionid")


def outputs(kind, exp=None):
    """outputs/<kind> folder (scans, confirmed, fits or figures), created if missing."""
    if kind not in OUTPUT_KINDS:
        raise ValueError(f"Unknown output kind {kind!r}; expected one of {OUTPUT_KINDS}")
    d = os.path.join(experiment_dir(exp), "outputs", kind)
    os.makedirs(d, exist_ok=True)
    return d


def settings_file(name, exp=None):
    """Path of a per-tool settings file in <experiment>/settings/ (folder created if missing)."""
    d = os.path.join(experiment_dir(exp), "settings")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


# --------------------------------------------------------------------------- old locations

def _legacy_map():
    """(old folder, new folder) pairs for paths stored before the suite existed.

    All of the old data belongs to E0203, so the targets are fixed to E0203
    rather than to the active experiment.
    """
    home = os.path.expanduser("~")
    desk = os.path.join(home, "Desktop")
    exp = "E0203"
    pairs = []
    for old in ("Asphodel/summed", "Asphodel_old/summed"):
        pairs.append((old, spectrograms(exp)))
    for old in ("Tartarus", "Tartarus_old"):
        pairs.append((old, summed_1d(exp)))
    for old in ("Theia", "Theia_old"):
        pairs.append((old + "/scans", os.path.join(experiment_dir(exp), "outputs", "scans")))
        pairs.append((old + "/confirmed", os.path.join(experiment_dir(exp), "outputs", "confirmed")))
    for old in ("Elysium", "Elysium_old"):
        pairs.append((old + "/levelschemes", levelschemes(exp)))
    pairs.append(("1dSpectra/SplitTDMS_Output_2^17_nobaseline", spectra_1d(exp)))
    return [(os.path.join(desk, o), n) for o, n in pairs]


def remap_legacy(path):
    """Translate a file or folder path from the old Desktop layout to data/.

    Returns the new path if `path` lies in one of the old folders and the new
    path exists; otherwise returns `path` unchanged. A trailing '/' is kept.
    Records on disk (scan databases, confirmed CSVs) are never rewritten.
    """
    if not path:
        return path
    trailing = path.endswith(os.sep)
    p = os.path.normpath(os.path.expanduser(path))
    for old, new in _legacy_map():
        if p == old or p.startswith(old + os.sep):
            cand = new + p[len(old):]
            if os.path.exists(cand):
                return cand + (os.sep if trailing and not cand.endswith(os.sep) else "")
            break
    return path
