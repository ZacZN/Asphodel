"""
Your own identifications for the Segrè chart, shared by the tools.

The file is data/<exp>/outputs/identified/segre_identified.json (format
asphodel-segre-identified v1, see reference/segre_chart.py for every field).
The Segrè chart reads and edits it; other tools (Theia's "Send to PID plot")
add entries through add_identification().  The open chart notices the file
changed and redraws.
"""
import datetime
import os
import re
import uuid

from core import jsonfile, paths

IDS_FILE = "segre_identified.json"
FORMAT = "asphodel-segre-identified"
ION_RE = re.compile(r"([a-z]?)(\d+)([A-Z][a-z]?)(\d+)\+$")     # 'm186Hf72+' -> m, 186, Hf, 72


def ids_path(exp=None):
    return os.path.join(paths.outputs("identified", exp), IDS_FILE)


def ion_name(e):
    iso = "" if e.get("isomer", "g") == "g" else e["isomer"]
    return f"{iso}{e['A']}{e['sym']}{e['charge']}+"


def add_identification(Z, N, ion, harmonic, status="confident", notes="", exp=None):
    """Add one of your own entries (ion like '197Os76+' or 'm186Hf72+', with Z,
    N of the nuclide).  An entry with the same nuclide, energy state, charge
    and harmonic already there is left as it is.
    Returns (added, message); raises if the file can't be written."""
    m = ION_RE.match(str(ion).strip())
    if not m:
        raise ValueError(f"can't read the ion name '{ion}'")
    iso, a, sym, q = m.group(1) or "g", int(m.group(2)), m.group(3), int(m.group(4))
    Z, N, h = int(Z), int(N), int(harmonic)
    entry = {"Z": Z, "N": N, "A": a, "sym": sym, "isomer": iso, "charge": q, "harmonic": h,
             "status": "unsure" if status == "unsure" else "confident",
             "notes": str(notes or "").strip(),
             "updated": datetime.datetime.now().isoformat(timespec="seconds")}
    path = ids_path(exp)
    with jsonfile.lock(path):                 # also held across processes (the chart may be saving)
        d = jsonfile.read(path, {})
        if not isinstance(d, dict):
            d = {}
        d.setdefault("format", FORMAT)
        d.setdefault("version", 1)
        entries = d.setdefault("entries", [])
        same = next((x for x in entries if isinstance(x, dict) and x.get("Z") == Z and x.get("N") == N
                     and x.get("isomer", "g") == iso and x.get("charge") == q
                     and x.get("harmonic") == h), None)
        if same is not None:
            return False, (f"{ion_name(same)} h{h} is already in the Segrè chart "
                           f"({same.get('status', 'confident')}) – left as it is.")
        entry["id"] = uuid.uuid4().hex[:10]
        entries.append(entry)
        jsonfile.write(path, d, indent=1, ensure_ascii=False)
    return True, f"✓ Added {ion_name(entry)} h{h} to the Segrè chart ({entry['status']})."
