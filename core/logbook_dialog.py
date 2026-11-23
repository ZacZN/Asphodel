#!/usr/bin/env python3
"""
logbook_dialog.py  –  "Save to Log Book" popout for the desktop tools (Tk)

The decay-fit window (matplotlib) opens this as its own process, so Tk and the
plot window's GUI toolkit never share a process:

    python core/logbook_dialog.py <spec.json>

spec.json:

    {"source": "Decay Fit", "title": "...", "notes": "",
     "slots": [{"label": "Ion", "value": "m186Hf72+"}],          # 1 or 2 ion fields
     "items": [{"label": "T½ (c.o.m.)", "value": 1.65, "error": 0.09, "unit": "s",
                "slot": 0,               # sent with that ion only; null = with every ion
                "checked": true}],
     "plots": [{"path": "/tmp/fit.png", "caption": "Decay fit", "checked": true}],
     "context": {...}}                                             # stored as it is

The window shows every value (editable, with a tick box each), the plot
(thumbnail, tick box), a title and notes; Save adds one entry per filled-in
ion field to the results log book (core/results_logbook.py) of the
experiment in ASPHODEL_EXPERIMENT.  The spec file and the plot files are
removed when the window closes.
"""
import json
import os
import sys

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import results_logbook as rlb

ACCENT, PALE, SIDE, LINE = "#f57d15", "#fff4ec", "#f9f9f9", "#dddddd"


def _str(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.10g}"
    return str(v)


def run(spec, spec_path=None):
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title(f"Save to Log Book · {spec.get('source', '')}")
    root.configure(bg="white")
    root.minsize(560, 300)
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure("TCombobox", fieldbackground="white")

    head = tk.Frame(root, bg=ACCENT, height=4)
    head.pack(fill="x")
    body = tk.Frame(root, bg="white", padx=14, pady=10)
    body.pack(fill="both", expand=True)

    def label(parent, text, **kw):
        return tk.Label(parent, text=text.upper(), fg="#777", bg="white",
                        font=("TkDefaultFont", 8, "bold"), anchor="w", **kw)

    # --- ions
    names = []
    try:
        names = rlb.ion_names()
    except Exception as exc:
        print(f"Results log book: could not read it: {exc}")
    slots = spec.get("slots") or [{"label": "Ion", "value": ""}]
    ion_vars = []
    fr = tk.Frame(body, bg="white")
    fr.pack(fill="x")
    for i, s in enumerate(slots):
        col = tk.Frame(fr, bg="white")
        col.pack(side="left", fill="x", expand=True, padx=(0, 10))
        label(col, s.get("label", "Ion") + " (tab)").pack(fill="x")
        v = tk.StringVar(value=s.get("value") or "")
        ttk.Combobox(col, textvariable=v, values=names, width=18).pack(fill="x")
        ion_vars.append(v)
    tk.Label(body, text="e.g. 198Ir, 198mIr, m186Hf72+ – a new name makes a new tab"
                        + ("; leave one empty to send to the other ion only" if len(slots) > 1 else ""),
             fg="#777", bg="white", anchor="w", font=("TkDefaultFont", 8)).pack(fill="x")

    label(body, "Title").pack(fill="x", pady=(8, 0))
    title_var = tk.StringVar(value=spec.get("title") or "")
    tk.Entry(body, textvariable=title_var, relief="solid", bd=1,
             highlightthickness=0).pack(fill="x")

    # --- values
    label(body, "Values to Send (edit if needed)").pack(fill="x", pady=(10, 2))
    vals = tk.Frame(body, bg="white")
    vals.pack(fill="x")
    for c, t in enumerate(["", "Quantity", "Value", "± Error", "Unit", ""]):
        tk.Label(vals, text=t, bg=PALE, fg="#555", anchor="w",
                 font=("TkDefaultFont", 9, "bold")).grid(row=0, column=c, sticky="ew", padx=1)
    vals.columnconfigure(1, weight=2)
    vals.columnconfigure(2, weight=2)
    vals.columnconfigure(3, weight=1)
    rows = []

    def add_row(item, editable_label=False):
        r = len(rows) + 1
        chk = tk.BooleanVar(value=item.get("checked", True))
        tk.Checkbutton(vals, variable=chk, bg="white", activebackground="white",
                       selectcolor="white", highlightthickness=0).grid(row=r, column=0)
        lab = tk.StringVar(value=item.get("label", ""))
        if editable_label:
            tk.Entry(vals, textvariable=lab, relief="solid", bd=1, width=18).grid(
                row=r, column=1, sticky="ew", padx=1, pady=1)
        else:
            tk.Label(vals, textvariable=lab, bg="white", anchor="w").grid(row=r, column=1, sticky="ew")
        vs = []
        for c, k, w in ((2, "value", 16), (3, "error", 10), (4, "unit", 6)):
            sv = tk.StringVar(value=_str(item.get(k)))
            tk.Entry(vals, textvariable=sv, relief="solid", bd=1, width=w,
                     font=("monospace", 9)).grid(row=r, column=c, sticky="ew", padx=1, pady=1)
            vs.append(sv)
        slot = item.get("slot")
        if slot is not None and len(slots) > 1:
            tk.Label(vals, text=f"→ {slots[slot].get('label', '')}", fg="#8a4a10", bg="white",
                     font=("TkDefaultFont", 8)).grid(row=r, column=5, sticky="w", padx=4)
        rows.append({"chk": chk, "label": lab, "value": vs[0], "error": vs[1], "unit": vs[2],
                     "slot": slot})

    for item in spec.get("items") or []:
        add_row(item)
    tk.Button(body, text="+ Add Value", relief="solid", bd=1, bg="white", activebackground=PALE,
              command=lambda: add_row({"label": "", "checked": True}, editable_label=True)
              ).pack(anchor="w", pady=(4, 0))

    # --- plots
    plots = []
    images = []                        # keep references (else Tk drops the pictures)
    if spec.get("plots"):
        label(body, "Plots").pack(fill="x", pady=(10, 2))
        for p in spec["plots"]:
            if not p.get("path") or not os.path.isfile(p["path"]):
                continue
            row = tk.Frame(body, bg="white")
            row.pack(fill="x")
            chk = tk.BooleanVar(value=p.get("checked", True))
            tk.Checkbutton(row, variable=chk, bg="white", activebackground="white",
                           selectcolor="white", highlightthickness=0).pack(side="left")
            try:
                img = tk.PhotoImage(file=p["path"])
                k = max(1, -(-img.width() // 260))
                img = img.subsample(k, k)
                images.append(img)
                tk.Label(row, image=img, bg="white", relief="solid", bd=1).pack(side="left", padx=4)
            except tk.TclError:
                pass
            cap = tk.StringVar(value=p.get("caption") or "")
            col = tk.Frame(row, bg="white")
            col.pack(side="left", fill="x", expand=True, padx=6)
            label(col, "Caption").pack(fill="x")
            tk.Entry(col, textvariable=cap, relief="solid", bd=1).pack(fill="x")
            plots.append({"chk": chk, "path": p["path"], "caption": cap})

    # --- notes
    label(body, "Notes").pack(fill="x", pady=(10, 0))
    notes = tk.Text(body, height=4, relief="solid", bd=1, wrap="word", font=("TkDefaultFont", 10))
    notes.pack(fill="both", expand=True)
    if spec.get("notes"):
        notes.insert("1.0", spec["notes"])

    # --- buttons
    foot = tk.Frame(body, bg="white")
    foot.pack(fill="x", pady=(10, 0))
    msg = tk.Label(foot, text="", fg="#8a4a10", bg="white", anchor="w", justify="left",
                   wraplength=420)

    def save():
        chosen = [(i, v.get().strip()) for i, v in enumerate(ion_vars) if v.get().strip()]
        if not chosen:
            msg.config(text="✗ Choose or type an ion.", fg="#c62828")
            return
        for _, name in chosen:
            try:
                rlb.parse_ion(name)
            except ValueError as exc:
                msg.config(text=f"✗ {exc}", fg="#c62828")
                return
        sel_plots = [{"png": p["path"], "caption": p["caption"].get()} for p in plots if p["chk"].get()]
        done = []
        try:
            for i, name in chosen:
                values = [{"label": r["label"].get(), "value": r["value"].get(),
                           "error": r["error"].get(), "unit": r["unit"].get()}
                          for r in rows if r["chk"].get() and r["slot"] in (None, i)]
                _, key = rlb.add_entry(name, spec.get("source", ""), title=title_var.get(),
                                       notes=notes.get("1.0", "end"), values=values,
                                       plots=sel_plots, context=spec.get("context") or {})
                done.append(key)
        except Exception as exc:
            msg.config(text=f"✗ Could not save: {exc}" + (f" (saved to {', '.join(done)})" if done else ""),
                       fg="#c62828")
            return
        text = f"✓ Saved to the log book: {', '.join(done)}"
        print(f"Results log book: {text[2:]} ({rlb.logbook_path()})")
        msg.config(text=text, fg="#2e7d32")
        save_btn.config(state="disabled")
        root.after(1200, root.destroy)

    save_btn = tk.Button(foot, text="Save", command=save, bg=ACCENT, fg="white", relief="flat",
                         activebackground="#e06d0a", activeforeground="white", padx=14)
    save_btn.pack(side="left")
    tk.Button(foot, text="Cancel", command=root.destroy, relief="solid", bd=1, bg="white",
              activebackground=PALE, padx=10).pack(side="left", padx=8)
    msg.pack(side="left", fill="x", expand=True)
    root.bind("<Escape>", lambda _e: root.destroy())
    root.mainloop()


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: logbook_dialog.py <spec.json>")
    path = sys.argv[1]
    with open(path, encoding="utf-8") as fh:
        spec = json.load(fh)
    try:
        run(spec, path)
    finally:
        # the spec and the plot files were written for this dialog only
        for p in [path] + [q.get("path") for q in spec.get("plots") or [] if spec.get("temp_plots", True)]:
            try:
                if p:
                    os.remove(p)
            except OSError:
                pass


if __name__ == "__main__":
    main()
