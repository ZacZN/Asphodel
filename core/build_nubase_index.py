"""build_nubase_index.py — page index for the NUBASE2020 paper (core).

Writes core/nubase2020_pages.json: {"198Ir": 140, ...}, the first page of
Table I in core/nubase2020.pdf that lists each nuclide (its ground-state
line; an isomer line if the ground state isn't there). The home page's
"View NUBASE" panel uses it to open the PDF at a nuclide. Only needed again
if the PDF changes; needs PyMuPDF (pip install pymupdf).

    python core/build_nubase_index.py
"""
import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PDF = os.path.join(HERE, "nubase2020.pdf")
MAS = os.path.join(HERE, "nubase_4.mas20")
OUT = os.path.join(HERE, "nubase2020_pages.json")
TABLE_HEADER = "Table I. The NUBASE2020 table"


def nuclide_names(mas=MAS):
    """Ground-state names in nubase_4.mas20, e.g. '198Ir', '1n'."""
    names = set()
    with open(mas, encoding="utf-8") as fh:
        for line in fh.readlines()[25:]:
            m = re.match(r"(\d+)([A-Z][a-z]?|n)$", line[11:16].strip())
            if m and line[7:8] == "0":
                names.add(m.group(1) + m.group(2))
    return names


def build(pdf=PDF, mas=MAS):
    import pymupdf
    names = nuclide_names(mas)
    gs, iso = collections.defaultdict(set), collections.defaultdict(set)
    for p, page in enumerate(pymupdf.open(pdf), start=1):
        text = page.get_text()
        if TABLE_HEADER not in text:
            continue
        lines = [l.strip() for l in text.splitlines()]
        for j, line in enumerate(lines):
            # a remark ("T :", "D :", ...) can wrap onto a line starting "131Pr ..."
            if j and lines[j - 1].endswith(":"):
                continue
            m = (re.match(r"^(\d{1,3})([A-Za-z]{1,3})\b", line)
                 or re.match(r"^(\d{1,3})([A-Z][a-z]?) [i-z]\b", line))
            if not m:
                continue
            a, s = m.groups()
            if a + s in names:
                gs[a + s].add(p)                      # ground-state line
            elif len(s) >= 2 and a + s[:-1] in names and s[-1] in "ijmnpqrx":
                iso[a + s[:-1]].add(p)                # isomer line, e.g. '198Irm'
    # pages never go backwards with A: each A starts no earlier than the last
    a_of = lambda k: int(re.match(r"\d+", k).group())
    pages = {k: gs[k] or iso[k] for k in set(gs) | set(iso)}
    idx, start = {}, 0
    for a in sorted({a_of(k) for k in pages}):
        ks = [k for k in pages if a_of(k) == a]
        start = min(p for k in ks for p in pages[k] if p >= start)
        for k in ks:
            ok = [p for p in pages[k] if p >= start]
            if ok:
                idx[k] = min(ok)
    missing = sorted(names - set(idx), key=lambda k: (a_of(k), k))
    return dict(sorted(idx.items(), key=lambda kv: (a_of(kv[0]), kv[1], kv[0]))), missing


if __name__ == "__main__":
    idx, missing = build()
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(idx, fh, separators=(",", ":"))
    print(f"{len(idx)} nuclides -> {OUT}")
    if missing:
        print("not found in the PDF:", " ".join(missing))
        sys.exit(1)
