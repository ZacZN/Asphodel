"""
Where the Asphodel suite keeps its data, outputs and settings.

Layout (all under the workspace folder, see core/workspace.py: chosen at
first start, e.g. ~/Asphodel-data; <repo>/data/ on setups from before that):

    <workspace>/
      asphodel_settings.json      active experiment + experiment profiles (see below)
      <experiment>/               (default location; a profile can point anywhere)
        settings/                 per-tool GUI settings (.json)
        outputs/scans|confirmed|fits|figures|isocurve|identified|logbook/
        notes.md                  notes from the home page

Input data is not in the workspace: its folders are connected per experiment
on the home page and can be anywhere. (Experiments from before that, e.g.
E0203, still have spectra_1d/, spectrograms/, summed_1d/, levelschemes/ and
rionid/ sub-folders; those are used as long as they hold files.)

Every function takes an optional `exp`; without it the active experiment is used.
Folders that tools write into (outputs/<kind>, settings/) are created on demand.

Use from a script in a sub-folder (e.g. theia/theia.py):

    import os, sys
    sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core import paths
"""

import os
import shutil

from . import workspace as _workspace

REPO_ROOT = _workspace.REPO_ROOT
# the workspace (DEFAULT_WORKSPACE if none is set up yet: home.py runs the
# first-run setup before importing this module, so that only happens when a
# tool is started on its own before Asphodel was ever set up)
DATA_ROOT = _workspace.workspace() or _workspace.DEFAULT_WORKSPACE
SUITE_SETTINGS = os.path.join(DATA_ROOT, "asphodel_settings.json")

OUTPUT_KINDS = ("scans", "confirmed", "fits", "figures", "isocurve", "identified", "logbook")


# --------------------------------------------------------------------------- experiments (profiles)
#
# Each experiment is a profile with its own data folder; everything of that
# experiment (spectra, settings incl. physics.json, outputs, notes) lives in
# that folder. data/asphodel_settings.json holds the active experiment and the
# profiles:
#
#   {"experiment": "E0203",
#    "profiles": {"E0203": {"data_dir": "/home/.../Asphodel/data/E0203"}}}
#
# A folder in data/ without a profile entry is a profile too (data_dir =
# data/<name>). The environment variable ASPHODEL_EXPERIMENT overrides the
# active experiment for one process: the home page starts every tool with it,
# so a running tool stays on its experiment when the home page switches.

ENV_EXPERIMENT = "ASPHODEL_EXPERIMENT"

# sub-folders made for every new experiment: its settings and the default
# output folders. No input-data folders: the user connects those (anywhere)
# on the home page, so a new experiment starts with none.
EXPERIMENT_SUBDIRS = ("settings", "outputs/scans", "outputs/confirmed", "outputs/fits",
                      "outputs/figures")
# the standard data sub-folders of the layout before connected folders; still
# used (listed as connected) where they exist and hold files, e.g. E0203
LEGACY_DATA_SUBDIRS = ("spectra_1d", "spectrograms", "summed_1d", "levelschemes", "rionid")


def _read_suite_settings():
    from . import jsonfile                 # jsonfile imports nothing from here
    return jsonfile.read(SUITE_SETTINGS, {})


def _write_suite_settings(d):
    from . import jsonfile
    jsonfile.write(SUITE_SETTINGS, d)


def profiles():
    """{name: {"data_dir": path}} for every experiment: the registered profiles
    plus any folder in data/ that is not registered."""
    out = {}
    try:
        for n in os.listdir(DATA_ROOT):
            if not n.startswith(".") and os.path.isdir(os.path.join(DATA_ROOT, n)):
                out[n] = {"data_dir": os.path.join(DATA_ROOT, n)}
    except OSError:
        pass
    reg = _read_suite_settings().get("profiles") or {}
    if isinstance(reg, dict):
        for n, p in reg.items():
            if isinstance(p, dict):
                out[n] = dict(out.get(n, {}), **p)
    for p in out.values():
        p["data_dir"] = os.path.abspath(os.path.expanduser(p.get("data_dir") or ""))
    return dict(sorted(out.items()))


def list_experiments():
    """Names of all experiments (sorted)."""
    return list(profiles())


def experiment():
    """Name of the active experiment.

    ASPHODEL_EXPERIMENT if set, else data/asphodel_settings.json. If neither
    names one and there is exactly one experiment, that one is used.
    """
    name = (os.environ.get(ENV_EXPERIMENT) or "").strip()
    if name:
        return name
    name = (_read_suite_settings().get("experiment") or "").strip()
    if name:
        return name
    exps = list_experiments()
    if len(exps) == 1:
        return exps[0]
    raise RuntimeError(
        f"No active experiment set. Put {{\"experiment\": \"<name>\"}} in {SUITE_SETTINGS} "
        f"(experiments found: {', '.join(exps) or 'none'}).")


def set_experiment(name):
    """Make `name` the active experiment (kept in data/asphodel_settings.json)."""
    if name not in profiles():
        raise ValueError(f"Unknown experiment {name!r}")
    from . import jsonfile
    with jsonfile.lock(SUITE_SETTINGS):
        d = _read_suite_settings()
        d["experiment"] = name
        _write_suite_settings(d)


def create_experiment(name, data_dir=None, copy_physics_from=None):
    """New experiment profile: register it, create its data folder with the
    standard sub-folders and, optionally, copy physics.json from another
    experiment. Existing files in data_dir are left alone. Returns data_dir."""
    name = (name or "").strip()
    if not name or os.sep in name or name.startswith("."):
        raise ValueError("Give the experiment a name without '/' (e.g. E0203).")
    if name in profiles():
        raise ValueError(f"An experiment called {name!r} already exists.")
    data_dir = os.path.abspath(os.path.expanduser(data_dir or os.path.join(DATA_ROOT, name)))
    for sub in EXPERIMENT_SUBDIRS:
        os.makedirs(os.path.join(data_dir, sub), exist_ok=True)
    if copy_physics_from:
        src = os.path.join(experiment_dir(copy_physics_from), "settings", "physics.json")
        dst = os.path.join(data_dir, "settings", "physics.json")
        if os.path.isfile(src) and not os.path.exists(dst):
            shutil.copyfile(src, dst)
    from . import jsonfile
    with jsonfile.lock(SUITE_SETTINGS):
        d = _read_suite_settings()
        reg = d.get("profiles") if isinstance(d.get("profiles"), dict) else {}
        reg[name] = {"data_dir": data_dir}
        d["profiles"] = reg
        _write_suite_settings(d)
    return data_dir


# --------------------------------------------------------------------------- folders

def experiment_dir(exp=None):
    """Data folder of the experiment (its profile's data_dir; data/<name> if
    it has no profile entry)."""
    name = exp or experiment()
    p = profiles().get(name)
    return p["data_dir"] if p else os.path.join(DATA_ROOT, name)


def notes_file(exp=None):
    """Free-text notes of the experiment (shown on the home page)."""
    return os.path.join(experiment_dir(exp), "notes.md")


# --------------------------------------------------------------------------- connected folders
#
# The folders the tools read from and write to are "connected" on the home
# page, per experiment, in <experiment>/settings/folders.json:
#
#   {"data":   [{"path": "...", "kind": "spectra_1d", "name": "1-D Spectra"}, ...],
#    "output": [{"path": "...", "kind": "scans", "name": "Scan Runs"}, ...]}
#
# Each folder has a kind; a tool that needs e.g. the 1-D spectra uses the first
# connected data folder of kind "spectra_1d" (spectra_1d() below), and its
# folder choosers list all connected folders.  Without a connected output
# folder of a kind the experiment's outputs/<kind> is used; without a connected
# data folder of a kind there is none ("") and the tool asks for one.
# Without folders.json the list is _default_folders() below.

DATA_KINDS = {"spectra_1d": "1-D Spectra", "spectrograms": "Spectrograms", "summed_1d": "Summed 1-D",
              "levelschemes": "Level Schemes", "rionid": "RionID Tables", "other": "Other Data"}
OUTPUT_KIND_NAMES = {"scans": "Scan Runs", "confirmed": "Confirmed IDs", "fits": "Decay Fits",
                     "figures": "Figures", "isocurve": "Isochronicity Curve",
                     "identified": "Identified Ions", "logbook": "Results Log Book",
                     "other": "Other Output"}
FOLDER_GROUPS = {"data": DATA_KINDS, "output": OUTPUT_KIND_NAMES}
FOLDERS_FILE = "folders.json"


def default_folder(group, kind, exp=None):
    """The standard sub-folder of the experiment for a kind (the pre-existing layout)."""
    base = experiment_dir(exp)
    if group == "data":
        return os.path.join(base, kind) if kind != "other" else base
    return os.path.join(base, "outputs", kind) if kind != "other" else os.path.join(base, "outputs")


def _has_files(d):
    try:
        return any(not n.startswith(".") for n in os.listdir(d))
    except OSError:
        return False


def _default_folders(exp=None):
    """Connected folders of an experiment without folders.json: the default
    output folders, and as data folders only the standard data sub-folders
    that exist and hold files (an experiment from before connected folders,
    e.g. E0203). A new experiment has no data folders until some are connected."""
    return {"data": [{"path": default_folder("data", k, exp), "kind": k, "name": n}
                     for k, n in DATA_KINDS.items()
                     if k in LEGACY_DATA_SUBDIRS and _has_files(default_folder("data", k, exp))],
            "output": [{"path": default_folder("output", k, exp), "kind": k, "name": n}
                       for k, n in OUTPUT_KIND_NAMES.items() if k != "other"]}


def _folders_path(exp=None):
    return os.path.join(experiment_dir(exp), "settings", FOLDERS_FILE)


def load_folders(exp=None):
    """{"data": [...], "output": [...]}: the connected folders of the experiment
    (the standard sub-folders until anything is connected or removed)."""
    from . import jsonfile
    d = jsonfile.read(_folders_path(exp), None)
    if d is None:                          # missing (or damaged: set aside + warning)
        return _default_folders(exp)
    out = {}
    for g, kinds in FOLDER_GROUPS.items():
        lst = []
        for f in (d.get(g) or []) if isinstance(d, dict) else []:
            if isinstance(f, dict) and f.get("path"):
                kind = f.get("kind") if f.get("kind") in kinds else "other"
                name = (f.get("name") or "").strip() or os.path.basename(f["path"].rstrip("/"))
                if name.lower() == kinds[kind].lower():      # a standard name saved in older
                    name = kinds[kind]                       # capitalisation ("1-D spectra")
                lst.append({"path": os.path.abspath(os.path.expanduser(f["path"])), "kind": kind,
                            "name": name})
        out[g] = lst
    return out


def save_folders(d, exp=None):
    from . import jsonfile
    jsonfile.write(_folders_path(exp), d)


def guess_kind(group, path):
    """A kind for a newly connected folder, from its name and contents."""
    name = os.path.basename(os.path.abspath(path).rstrip("/")).lower()
    for k in FOLDER_GROUPS[group]:
        if k != "other" and k in name.replace("-", "_").replace(" ", "_"):
            return k
    if group == "data":
        try:
            files = os.listdir(path)
        except OSError:
            files = []
        if any(f.lower().endswith((".xlsx", ".xls")) for f in files):
            return "rionid"
        if any(f.lower().endswith(".npz") for f in files):
            return "spectra_1d"
    return "other"


def connect_folder(group, path, kind=None, name=None, exp=None):
    """Add a folder to the experiment's connected folders (or update its kind /
    name if it is already there). Returns the folder entry."""
    if group not in FOLDER_GROUPS:
        raise ValueError(f"Unknown folder group {group!r}")
    p = os.path.abspath(os.path.expanduser((path or "").strip()))
    if not (path or "").strip() or not os.path.isdir(p):
        raise ValueError(f"Not a folder: {path}")
    kind = kind if kind in FOLDER_GROUPS[group] else guess_kind(group, p)
    with _folders_lock(exp):
        d = load_folders(exp)
        entry = next((f for f in d[group] if f["path"] == p), None)
        if entry is None:
            entry = {"path": p, "kind": kind,
                     "name": (name or "").strip() or os.path.basename(p.rstrip("/"))}
            d[group].append(entry)
        else:
            entry["kind"] = kind
            if (name or "").strip():
                entry["name"] = name.strip()
        save_folders(d, exp)
    return entry


def _folders_lock(exp=None):
    """Lock for a read-modify-write of folders.json (core/jsonfile.py)."""
    from . import jsonfile
    return jsonfile.lock(_folders_path(exp))


def disconnect_folder(group, path, exp=None):
    with _folders_lock(exp):
        d = load_folders(exp)
        d[group] = [f for f in d[group] if f["path"] != path]
        save_folders(d, exp)


def update_folder(group, path, exp=None, **changes):
    """Change a connected folder's kind or name."""
    with _folders_lock(exp):
        d = load_folders(exp)
        for f in d[group]:
            if f["path"] == path:
                if changes.get("kind") in FOLDER_GROUPS[group]:
                    f["kind"] = changes["kind"]
                if (changes.get("name") or "").strip():
                    f["name"] = changes["name"].strip()
        save_folders(d, exp)


def make_default(group, path, exp=None):
    """Move a folder in front of the others of its kind: the tools use it."""
    with _folders_lock(exp):
        d = load_folders(exp)
        f = next((x for x in d[group] if x["path"] == path), None)
        if f is not None:
            d[group].remove(f)
            d[group].insert(0, f)
            save_folders(d, exp)


def connected(group, kind=None, exp=None):
    """Connected folders of a group ("data" or "output"), optionally of one kind,
    defaults first: [{"path", "kind", "name"}]."""
    lst = load_folders(exp)[group]
    return [f for f in lst if kind is None or f["kind"] == kind]


def folder_for(group, kind, exp=None):
    """The folder a tool uses for a kind: the first connected folder of that kind
    that exists. If there is none: for output, the experiment's standard
    outputs/<kind>; for data, "" (no folder: the user has to connect one on the
    home page first; no_data_folder() gives the message to show)."""
    for f in connected(group, kind, exp):
        if os.path.isdir(f["path"]):
            return f["path"]
    return default_folder(group, kind, exp) if group == "output" else ""


def no_data_folder(kind):
    """Message for a tool that needs a data folder of `kind` when none is connected."""
    return (f"No {DATA_KINDS.get(kind, kind)} folder is connected for this experiment: "
            f"connect one on the home page (Folders → Connect Data Folder…), or type a path.")


# The data folders below are "" when no folder of the kind is connected.

def spectra_1d(exp=None):
    return folder_for("data", "spectra_1d", exp)


def spectrograms(exp=None):
    return folder_for("data", "spectrograms", exp)


def summed_1d(exp=None):
    return folder_for("data", "summed_1d", exp)


def levelschemes(exp=None):
    return folder_for("data", "levelschemes", exp)


def rionid(exp=None):
    return folder_for("data", "rionid", exp)


def outputs(kind, exp=None):
    """Output folder of a kind (scans, confirmed, fits, figures, isocurve,
    identified, logbook): the connected one, else outputs/<kind>; created if missing."""
    if kind not in OUTPUT_KINDS:
        raise ValueError(f"Unknown output kind {kind!r}; expected one of {OUTPUT_KINDS}")
    d = folder_for("output", kind, exp)
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
        pairs.append((old, default_folder("data", "spectrograms", exp)))
    for old in ("Tartarus", "Tartarus_old"):
        pairs.append((old, default_folder("data", "summed_1d", exp)))
    for old in ("Theia", "Theia_old"):
        pairs.append((old + "/scans", os.path.join(experiment_dir(exp), "outputs", "scans")))
        pairs.append((old + "/confirmed", os.path.join(experiment_dir(exp), "outputs", "confirmed")))
    for old in ("Elysium", "Elysium_old"):
        pairs.append((old + "/levelschemes", default_folder("data", "levelschemes", exp)))
    pairs.append(("1dSpectra/SplitTDMS_Output_2^17_nobaseline", default_folder("data", "spectra_1d", exp)))
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
