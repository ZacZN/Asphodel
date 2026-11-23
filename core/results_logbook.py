"""
results_logbook.py  –  the experiment's results log book (one page per ion)

Results about each ion (half-lives, peak frequencies, plots, notes) are sent
here from the tools ("Save to log book" in the decay-fit window, the
spectrogram page and Theia's Viewer / Scan Results) and shown by
reference/results_logbook.py, one vertical tab per ion.

A tab ("ion") is a nuclide in one energy state: 198Ir and 198mIr are separate
tabs; the charge state and harmonic are stored on each entry (as values).

Storage (per experiment, output kind "logbook", see core/paths.py):

    outputs/logbook/results_logbook.json
    outputs/logbook/plots/<entry id>_<n>.png

results_logbook.json (format asphodel-results-logbook v1):

    {"format": "asphodel-results-logbook", "version": 1,
     "ions": {"198mIr": {"A": 198, "sym": "Ir", "Z": 77, "N": 121, "state": "m",
                         "created": "2026-11-23T10:00:00"}},
     "entries": [{"id": "a1b2c3d4e5", "ion": "198mIr",
                  "created": "...", "updated": "...",
                  "source": "Decay fit",          # tool that sent it
                  "title": "...", "notes": "...",
                  "values": [{"label": "T½ (c.o.m.)", "value": 1.4, "error": 0.1,
                              "unit": "s"}, ...],
                  "plots": [{"file": "plots/a1b2c3d4e5_0.png", "caption": "..."}],
                  "context": {"file": "...", ...}}]}   # where it came from (read-only)

Every change is a read-modify-write under jsonfile.lock (shared between the
processes), so the tools and the log book page can write at the same time.
"""
import base64
import datetime
import os
import re
import uuid

from core import jsonfile, paths

FILE = "results_logbook.json"
FORMAT = "asphodel-results-logbook"
PLOTS = "plots"

SYMBOLS = ("n H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
           "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr "
           "Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra "
           "Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts "
           "Og").split()
Z_OF = {s.lower(): z for z, s in enumerate(SYMBOLS)}
STATES = "mnpqrx"                    # isomer letters as in NUBASE

# accepted ion names: 198Ir, 198mIr, m198Ir, Ir198, Ir-198, Ir198m, 198Ir77+, m186Hf72+,
# 186m Hf, 186Hf(m)
_RE_A_FIRST = re.compile(r"^([mnpqrx]?)\s*(\d+)\s*([mnpqrx]?)\s*\(?([mnpqrx]?)\)?\s*-?\s*([A-Za-z]{1,2})"
                         r"\s*\(?([mnpqrx]?)\)?\s*(?:(\d+)\s*\+)?$")
_RE_SYM_FIRST = re.compile(r"^([mnpqrx]?)\s*([A-Za-z]{1,2})\s*-?\s*(\d+)\s*([mnpqrx]?)\s*(?:(\d+)\s*\+)?$")


class IonError(ValueError):
    pass


def logbook_path(exp=None):
    return os.path.join(paths.outputs("logbook", exp), FILE)


def plots_dir(exp=None):
    d = os.path.join(paths.outputs("logbook", exp), PLOTS)
    os.makedirs(d, exist_ok=True)
    return d


def plot_path(rel, exp=None):
    """Absolute path of a plot stored as 'plots/<name>.png' (None if it escapes the folder)."""
    base = paths.outputs("logbook", exp)
    p = os.path.normpath(os.path.join(base, rel))
    return p if p.startswith(os.path.join(base, PLOTS) + os.sep) else None


def mtime(exp=None):
    try:
        return os.path.getmtime(logbook_path(exp))
    except OSError:
        return 0.0


# --------------------------------------------------------------------------- ion names

def parse_ion(text):
    """'198Ir', '198mIr', 'm198Ir', 'Ir-198', '198Ir77+', 'm186Hf72+' ->
    {"key": "198mIr", "A", "sym", "Z", "N", "state": "g"/"m"/..., "charge": int or None}.
    Raises IonError if it can't be read."""
    t = str(text or "").strip()
    if not t:
        raise IonError("no ion given")
    m = _RE_A_FIRST.match(t)
    if m:
        pre, a, s1, s2, sym, s3, q = m.groups()
        state = pre or s1 or s2 or s3
    else:
        m = _RE_SYM_FIRST.match(t)
        if not m:
            raise IonError(f"can't read the ion name '{t}' (e.g. 198Ir, 198mIr, Ir-198, 198Ir77+)")
        pre, sym, a, s1, q = m.groups()
        state = pre or s1
    z = Z_OF.get(sym.lower())
    if z is None:
        # '198mIr' read as sym='mIr' can't happen (1-2 letters), but 'Mir' etc. can
        raise IonError(f"unknown element '{sym}' in '{t}'")
    sym = SYMBOLS[z]
    a = int(a)
    if a < z:
        raise IonError(f"A = {a} is smaller than Z = {z} in '{t}'")
    state = state or "g"
    charge = int(q) if q else None
    if charge is not None and charge > z:
        raise IonError(f"charge {charge}+ is more than Z = {z} in '{t}'")
    return {"key": ion_key(a, sym, state), "A": a, "sym": sym, "Z": z, "N": a - z,
            "state": state, "charge": charge}


def ion_key(a, sym, state="g"):
    return f"{a}{'' if state in ('g', '', None) else state}{sym}"


def ion_sort_key(ion):
    st = ion.get("state", "g")
    return (ion.get("Z", 0), ion.get("A", 0), -1 if st == "g" else STATES.find(st))


# --------------------------------------------------------------------------- file

def _empty():
    return {"format": FORMAT, "version": 1, "ions": {}, "entries": []}


def load(exp=None):
    d = jsonfile.read(logbook_path(exp), None)
    if not isinstance(d, dict):
        return _empty()
    d.setdefault("format", FORMAT)
    d.setdefault("version", 1)
    if not isinstance(d.get("ions"), dict):
        d["ions"] = {}
    if not isinstance(d.get("entries"), list):
        d["entries"] = []
    d["entries"] = [e for e in d["entries"] if isinstance(e, dict) and e.get("id")]
    return d


def ions(exp=None, d=None):
    """[(key, ion dict, number of entries)] sorted by Z, A, state."""
    d = d or load(exp)
    count = {}
    for e in d["entries"]:
        count[e.get("ion")] = count.get(e.get("ion"), 0) + 1
    out = [(k, v, count.get(k, 0)) for k, v in d["ions"].items()]
    return sorted(out, key=lambda x: ion_sort_key(x[1]))


def ion_names(exp=None):
    return [k for k, _, _ in ions(exp)]


def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def _ensure_ion(d, info):
    k = info["key"]
    if k not in d["ions"]:
        d["ions"][k] = {"A": info["A"], "sym": info["sym"], "Z": info["Z"], "N": info["N"],
                        "state": info["state"], "created": _now()}
    return k


def _num(v):
    """Number if it reads as one, else the stripped text (None for empty)."""
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v if v == v else None          # NaN -> None
    s = str(v).strip()
    if not s:
        return None
    try:
        f = float(s)
        return int(f) if re.fullmatch(r"[+-]?\d+", s) else f
    except ValueError:
        return s


def clean_values(values):
    out = []
    for v in values or []:
        if not isinstance(v, dict):
            continue
        label = str(v.get("label") or "").strip()
        value = _num(v.get("value"))
        if not label and value is None:
            continue
        item = {"label": label or "value", "value": value}
        err = _num(v.get("error"))
        if err is not None:
            item["error"] = err
        unit = str(v.get("unit") or "").strip()
        if unit:
            item["unit"] = unit
        out.append(item)
    return out


def _png_bytes(png):
    """bytes, a 'data:image/png;base64,...' URL or a path to a .png file -> bytes."""
    if isinstance(png, (bytes, bytearray)):
        return bytes(png)
    s = str(png or "")
    if s.startswith("data:"):
        head, _, b64 = s.partition(",")
        if "image/png" not in head:
            raise ValueError("only PNG images can be stored")
        return base64.b64decode(b64)
    if s and os.path.isfile(s):
        with open(s, "rb") as fh:
            return fh.read()
    raise ValueError("no image data")


def _store_plots(eid, plots, exp=None, start=0):
    out = []
    for i, p in enumerate(plots or [], start=start):
        if not isinstance(p, dict) or not p.get("png"):
            continue
        data = _png_bytes(p["png"])
        if not data.startswith(b"\x89PNG"):
            raise ValueError("the plot is not a PNG image")
        rel = f"{PLOTS}/{eid}_{i}.png"
        with open(os.path.join(plots_dir(exp), f"{eid}_{i}.png"), "wb") as fh:
            fh.write(data)
        out.append({"file": rel, "caption": str(p.get("caption") or "").strip()})
    return out


def add_entry(ion, source, title="", notes="", values=(), plots=(), context=None, exp=None):
    """Add an entry to an ion's tab (made if new).  `ion` is a name parse_ion
    reads; a charge state in it ('198Ir77+') is added as the value 'Charge state q'
    unless the values already have one.  `plots` = [{"png": bytes | data URL |
    path, "caption": str}].  Returns (entry id, ion key)."""
    info = parse_ion(ion)
    vals = clean_values(values)
    if info["charge"] is not None and not any(v["label"].lower().startswith("charge") for v in vals):
        vals.insert(0, {"label": "Charge state q", "value": info["charge"]})
    eid = uuid.uuid4().hex[:10]
    stored = _store_plots(eid, plots, exp)          # before the lock: files only
    path = logbook_path(exp)
    with jsonfile.lock(path):
        d = load(exp)
        key = _ensure_ion(d, info)
        now = _now()
        d["entries"].append({"id": eid, "ion": key, "created": now, "updated": now,
                             "source": str(source or ""), "title": str(title or "").strip(),
                             "notes": str(notes or "").rstrip(), "values": vals, "plots": stored,
                             "context": context if isinstance(context, dict) else {}})
        jsonfile.write(path, d, indent=1, ensure_ascii=False)
    return eid, key


def get_entry(eid, exp=None):
    return next((e for e in load(exp)["entries"] if e["id"] == eid), None)


def update_entry(eid, title=None, notes=None, values=None, ion=None, keep_plots=None,
                 captions=None, exp=None):
    """Edit an entry.  `ion` moves it to another tab (made if new);
    `keep_plots` = list of plot files to keep (others are deleted);
    `captions` = {file: caption}.  Returns the ion key it is under now."""
    path = logbook_path(exp)
    removed = []
    with jsonfile.lock(path):
        d = load(exp)
        e = next((x for x in d["entries"] if x["id"] == eid), None)
        if e is None:
            raise KeyError("the entry is no longer in the log book")
        if title is not None:
            e["title"] = str(title).strip()
        if notes is not None:
            e["notes"] = str(notes).rstrip()
        if values is not None:
            e["values"] = clean_values(values)
        if ion is not None and str(ion).strip():
            e["ion"] = _ensure_ion(d, parse_ion(ion))
        if keep_plots is not None:
            keep = set(keep_plots)
            removed = [p["file"] for p in e.get("plots", []) if p.get("file") not in keep]
            e["plots"] = [p for p in e.get("plots", []) if p.get("file") in keep]
        if captions:
            for p in e.get("plots", []):
                if p.get("file") in captions:
                    p["caption"] = str(captions[p["file"]] or "").strip()
        e["updated"] = _now()
        jsonfile.write(path, d, indent=1, ensure_ascii=False)
    _remove_files(removed, exp)
    return e["ion"]


def delete_entry(eid, exp=None):
    """Delete an entry and its plot files.  The ion's tab stays (delete_ion
    removes an empty one)."""
    path = logbook_path(exp)
    with jsonfile.lock(path):
        d = load(exp)
        e = next((x for x in d["entries"] if x["id"] == eid), None)
        if e is None:
            return False
        d["entries"] = [x for x in d["entries"] if x["id"] != eid]
        jsonfile.write(path, d, indent=1, ensure_ascii=False)
    _remove_files([p.get("file") for p in e.get("plots", [])], exp)
    return True


def delete_ion(key, exp=None):
    """Remove an ion's tab; only allowed when it has no entries."""
    path = logbook_path(exp)
    with jsonfile.lock(path):
        d = load(exp)
        if any(e.get("ion") == key for e in d["entries"]):
            raise ValueError(f"{key} still has entries – delete them first")
        if d["ions"].pop(key, None) is None:
            return False
        jsonfile.write(path, d, indent=1, ensure_ascii=False)
    return True


def _remove_files(rels, exp=None):
    for rel in rels:
        p = plot_path(rel, exp) if rel else None
        if p and os.path.isfile(p):
            try:
                os.remove(p)
            except OSError as exc:
                print(f"Results log book: could not remove {p}: {exc}")


# --------------------------------------------------------------------------- display helpers

def fmt_number(v, err=None):
    """value ± error, the error with 2 significant figures setting the decimals."""
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return str(v) + (f" ± {err}" if err not in (None, "") else "")
    if isinstance(err, (int, float)) and err > 0 and v == v:
        import math
        dec = max(0, 1 - int(math.floor(math.log10(err))))
        if abs(v) >= 1e12 or dec > 10:
            return f"{v:.6g} ± {err:.2g}"
        return f"{v:.{dec}f} ± {err:.{dec}f}"
    if isinstance(v, int):
        return str(v)
    return f"{v:.6g}"


def fmt_value(item):
    s = fmt_number(item.get("value"), item.get("error"))
    return s + (f" {item['unit']}" if item.get("unit") else "")


def round_pair(v, err=None):
    """Round the error to 2 significant figures and the value to one more
    decimal than the error (6 significant figures without an error)."""
    import math
    if not isinstance(v, (int, float)) or v != v:
        return v, err
    if isinstance(err, (int, float)) and err > 0:
        dec = 1 - int(math.floor(math.log10(err)))
        return round(v, dec + 1), round(err, dec)
    return (float(f"{v:.6g}") if isinstance(v, float) else v), err


def fmt_time_value(seconds, err=None):
    """A time in s as a value dict in a readable unit (ms, s, min, h, d), rounded."""
    units = [("d", 86400.0), ("h", 3600.0), ("min", 60.0), ("s", 1.0), ("ms", 1e-3), ("µs", 1e-6)]
    for u, f in units:
        if abs(seconds) >= f or u == "µs":
            v, e = round_pair(seconds / f, None if err is None else err / f)
            return {"value": v, "error": e, "unit": u}
