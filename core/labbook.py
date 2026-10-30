"""
labbook.py  –  the experiment's lab book (shown on the home page)

The notes of an experiment are a diary of time-stamped entries, kept in
<experiment>/notes.md as plain Markdown that reads well on its own:

    # Lab book – E0203

    ## Earlier notes              (text from before the lab book, if any)

    ...

    ## 2026-10-30 Friday

    ### 14:32

    Entry text ...

    ### 15:10

    ...

The file is append-only: a new entry is added at the end (under a new day
heading if it is the first entry of the day); nothing already written is
changed.  The home page shows it newest first.

Inside an entry, a line that would look like a day or time heading is
written with a leading backslash, so it can't be mistaken for one.
"""
import datetime
import os
import re

DAY_RE = re.compile(r"^## (\d{4}-\d{2}-\d{2})\b.*$")
TIME_RE = re.compile(r"^### (\d{1,2}:\d{2})\s*$")
TITLE_RE = re.compile(r"^# .*$")
EARLIER = "## Earlier notes"


def _read(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return ""


def _escape(text):
    out = []
    for line in text.splitlines():
        if DAY_RE.match(line) or TIME_RE.match(line) or line.startswith("\\#"):
            line = "\\" + line
        out.append(line)
    return "\n".join(out)


def _unescape(line):
    return line[1:] if line.startswith("\\#") else line


def parse(path):
    """-> (earlier, days): earlier = text from before the lab book ("" if
    none); days = [{"date": date, "entries": [{"time": "14:32", "text": ...}]}]
    in file order (oldest first)."""
    earlier, days = [], []
    entry = None
    for line in _read(path).splitlines():
        m = DAY_RE.match(line)
        if m:
            try:
                d = datetime.date.fromisoformat(m.group(1))
            except ValueError:
                d = None
            if d is not None:
                days.append({"date": d, "entries": []})
                entry = None
                continue
        if days:
            m = TIME_RE.match(line)
            if m:
                entry = {"time": m.group(1), "lines": []}
                days[-1]["entries"].append(entry)
                continue
            if entry is not None:
                entry["lines"].append(_unescape(line))
            # text between a day heading and its first time is ignored (blank)
        else:
            earlier.append(line)
    # earlier notes: drop the "# Lab book" title and the "## Earlier notes" heading
    earlier = [l for l in earlier if not (l.strip() == EARLIER or TITLE_RE.match(l) and l.startswith("# Lab book"))]
    for day in days:
        for e in day["entries"]:
            e["text"] = "\n".join(e.pop("lines")).strip("\n")
    return "\n".join(earlier).strip("\n"), days


def add_entry(path, text, experiment="", now=None):
    """Append one time-stamped entry. Returns the datetime used."""
    text = (text or "").strip("\n")
    if not text.strip():
        raise ValueError("the entry is empty")
    now = now or datetime.datetime.now()
    old = _read(path)
    _, days = parse(path)
    parts = []
    if not days:
        # first entry: start the lab book; keep any older text as "Earlier notes"
        if old.strip():
            body = old.strip("\n")
            if not old.lstrip().startswith("# Lab book"):
                body = f"# Lab book – {experiment}\n\n{EARLIER}\n\n{body}".replace(" – \n", "\n")
                # rewrite once, only to add the two headings in front of the old text
                tmp = path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as fh:
                    fh.write(body + "\n")
                os.replace(tmp, path)
                old = body + "\n"
        else:
            parts.append(f"# Lab book – {experiment}\n" if experiment else "# Lab book\n")
    if not days or days[-1]["date"] != now.date():
        parts.append(f"## {now:%Y-%m-%d %A}\n")
    parts.append(f"### {now:%H:%M}\n")
    parts.append(_escape(text) + "\n")
    sep = "" if (not old or old.endswith("\n\n")) else ("\n" if old.endswith("\n") else "\n\n")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(sep + "\n".join(parts))
    return now
