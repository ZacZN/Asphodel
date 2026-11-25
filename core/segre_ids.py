"""
Your own identifications for the Segrè chart, shared by the tools.

The file is data/<exp>/outputs/identified/segre_identified.json (format
asphodel-segre-identified v1, see reference/segre_chart.py for every field).
The Segrè chart reads and edits it; other tools (Theia's "Send to PID Plot")
add entries through add_identification().  The open chart notices the file
changed and redraws.

An entry sent from Theia also has "peaks": one dict per spectrum / peak it
was sent from, with the same columns as a Theia confirmed list (spectrum,
seq, spectrum_id, peak_id, peak_freq, sim_freq, delta_hz, power, snr, kind,
...) plus "database", "run_info", "scan_settings" and "sent".  Sending the
same ion again from another peak adds that peak to the entry; sending it as
confident upgrades an unsure entry.
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


def _peak_key(p):
    return (os.path.basename(str(p.get("database", ""))), p.get("spectrum_id"), p.get("peak_id"))


def peak_label(p):
    """'spectrum, peak 123456.7 Hz' for messages."""
    try:
        return f"{p.get('spectrum', '?')}, peak {float(p['peak_freq']):.1f} Hz"
    except (KeyError, TypeError, ValueError):
        return str(p.get("spectrum", "?"))


def add_identification(Z, N, ion, harmonic, status="confident", notes="", exp=None, peak=None):
    """Add one of your own entries (ion like '197Os76+' or 'm186Hf72+', with Z,
    N of the nuclide), optionally with the peak it was found in (a dict, see
    the module docstring).  If an entry with the same nuclide, energy state,
    charge and harmonic is already there, the peak is added to it (its status
    and notes are left as they are, except that status "confident" upgrades an
    unsure entry; "unsure" never downgrades a confident one); the same peak
    twice is skipped.
    Returns (changed, message); raises if the file can't be written."""
    m = ION_RE.match(str(ion).strip())
    if not m:
        raise ValueError(f"can't read the ion name '{ion}'")
    iso, a, sym, q = m.group(1) or "g", int(m.group(2)), m.group(3), int(m.group(4))
    Z, N, h = int(Z), int(N), int(harmonic)
    entry = {"Z": Z, "N": N, "A": a, "sym": sym, "isomer": iso, "charge": q, "harmonic": h,
             "status": "unsure" if status == "unsure" else "confident",
             "notes": str(notes or "").strip(),
             "updated": datetime.datetime.now().isoformat(timespec="seconds")}
    if peak:
        peak = dict(peak, sent=entry["updated"])
        entry["peaks"] = [peak]
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
            # confident upgrades an unsure entry; unsure never downgrades a confident one
            upgrade = entry["status"] == "confident" and same.get("status", "confident") == "unsure"
            up_msg = " Now confident." if upgrade else ""
            if upgrade:
                same["status"] = "confident"
            peaks = same.setdefault("peaks", []) if peak else None
            if peak and not isinstance(peaks, list):
                peaks = same["peaks"] = []
            dup = bool(peak) and any(isinstance(x, dict) and _peak_key(x) == _peak_key(peak)
                                     for x in peaks)
            if not peak or dup:
                where = f" from {peak_label(peak)}" if peak else ""
                if not upgrade:
                    return False, (f"{ion_name(same)} h{h}{where} is already in the Segrè chart "
                                   f"({same.get('status', 'confident')}) – left as it is.")
                same["updated"] = entry["updated"]
                jsonfile.write(path, d, indent=1, ensure_ascii=False)
                return True, f"✓ {ion_name(same)} h{h}{where} is already in the Segrè chart.{up_msg}"
            peaks.append(peak)
            same["updated"] = entry["updated"]
            jsonfile.write(path, d, indent=1, ensure_ascii=False)
            return True, (f"✓ Added {peak_label(peak)} to {ion_name(same)} h{h} in the Segrè chart "
                          f"({len(peaks)} peak{'s' if len(peaks) != 1 else ''}, "
                          f"{same.get('status', 'confident')}).{up_msg}")
        entry["id"] = uuid.uuid4().hex[:10]
        entries.append(entry)
        jsonfile.write(path, d, indent=1, ensure_ascii=False)
    return True, (f"✓ Added {ion_name(entry)} h{h} to the Segrè chart ({entry['status']})"
                  + (f" with {peak_label(peak)}." if peak else "."))
