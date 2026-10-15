#!/usr/bin/env python3
"""
LevelSchemer - a small GUI for drawing publication-quality nuclear level
schemes with matplotlib.

Requirements: Python 3.8+, matplotlib (>= 3.3), Tkinter (ships with Python;
on Linux you may need `sudo apt install python3-tk`).

Run:   python levelschemer.py [scheme.json]

Mouse controls in the preview
  Left-click            select a level / transition
  Drag a level line     shift where it is drawn (its energy is unchanged)
  Drag an arrow         move it sideways
  Drag any label        reposition it
  Right-drag level->level   create a transition between them
  Double-click empty    add a new level at that energy / column
Keys: Delete = delete selection, Ctrl+Z / Ctrl+Y = undo / redo, Ctrl+S = save
"""

import copy
import datetime
import inspect
import json
import os
import pprint
import re
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, ArrowStyle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_agg import FigureCanvasAgg

APP_NAME = "LevelSchemer"
HIGHLIGHT = "#f57d15"

DEFAULT_STYLE = {
    # figure
    "fig_width": 3.4, "fig_height": 4.0, "dpi": 600, "tight_bbox": True,
    "margin_left": 0.22, "margin_right": 0.18, "margin_bottom": 0.06, "margin_top": 0.06,
    "y_margin": 0.05,
    # fonts
    "font_family": "serif", "mathtext_fontset": "stix", "usetex": False,
    "font_size": 9.0, "gamma_font_size": 8.5,
    # levels
    "column_width": 1.0, "column_gap": 0.8, "level_lw": 1.0, "level_color": "black",
    "energy_fmt": "{:.0f}", "energy_italic": True,
    "spin_side": "left", "energy_side": "right", "label_pad": 4.0,
    "auto_spin_format": True,
    # transitions
    "arrow_max_width": 5.0, "arrow_min_width": 0.8, "head_length": 6.0, "head_extra": 4.0,
    "gamma_label_side": "left", "gamma_label_rotation": "0", "gamma_label_pad": 3.0,
    # annotations
    "title": "", "title_size": 12.0, "column_titles": "",
    "offsets": {},
}


# =============================================================================
# Rendering core. These functions are copied verbatim into exported scripts,
# so they only depend on re, FancyArrowPatch and ArrowStyle.
# =============================================================================

def style_rc(style):
    """matplotlib rcParams for a given style dict."""
    return {
        "font.family": style["font_family"],
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": style["mathtext_fontset"],
        "font.size": float(style["font_size"]),
        "text.usetex": bool(style["usetex"]),
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    }


def fmt_energy(value, style):
    try:
        return style["energy_fmt"].format(value)
    except Exception:
        return "{:g}".format(value)


def format_spin(s, auto=True):
    """'(23/2-)' -> '(23/2$^{-}$)'. Strings already containing $ are left alone."""
    if not s:
        return ""
    if not auto or "$" in s:
        return s
    return re.sub(r"(\d+(?:/\d+)?)\s*([+\-])",
                  lambda m: m.group(1) + "$^{" + m.group(2) + "}$", s)


def format_lifetime(s, style):
    """'5.89(18) us' -> '5.89(18) μs'."""
    if not s:
        return ""
    mu = r"$\mu$" if style.get("usetex") else "\u03bc"
    return re.sub(r"(\d|\))\s*us\b", lambda m: m.group(1) + " " + mu + "s", s.strip())


def column_x0(col, style):
    return int(col) * (float(style["column_width"]) + float(style["column_gap"]))


def level_y(level):
    return float(level["energy"]) + float(level.get("y_shift") or 0.0)


def level_xspan(level, style):
    x0 = column_x0(level.get("column", 0), style)
    w = float(style["column_width"])
    return x0 + w * float(level.get("x0", 0.0)), x0 + w * float(level.get("x1", 1.0))


def render(fig, scheme, style, hits=None, selected=None, limits=None, highlight="#f57d15"):
    """Draw a level scheme into `fig`. Returns the axes.

    scheme: {"levels": [...], "transitions": [...]}
    hits/selected/limits/highlight are only used by the interactive editor.
    """
    fig.clf()
    fig.patch.set_facecolor("white")
    ml, mr = float(style["margin_left"]), float(style["margin_right"])
    mb, mt = float(style["margin_bottom"]), float(style["margin_top"])
    ax = fig.add_axes([ml, mb, max(0.05, 1 - ml - mr), max(0.05, 1 - mb - mt)])
    ax.set_axis_off()

    levels = {l["id"]: l for l in scheme.get("levels", [])}
    if not levels:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        if hits is not None:
            ax.text(0.5, 0.5, "Double-click here to add a level", ha="center",
                    va="center", color="0.6", transform=ax.transAxes)
        return ax

    fs = float(style["font_size"])
    gfs = float(style["gamma_font_size"])
    cw = float(style["column_width"])
    pad = float(style["label_pad"])

    def reg(*item):
        if hits is not None:
            hits.append(item)

    def offs(obj, part):
        o = (obj.get("offsets") or {}).get(part) or (0, 0)
        return float(o[0]), float(o[1])

    def is_sel(kind, id_):
        return selected is not None and tuple(selected) == (kind, id_)

    # --- limits -------------------------------------------------------------
    ys = [level_y(l) for l in levels.values()]
    ymin, ymax = min(ys), max(ys)
    ncol = max(int(l.get("column", 0)) for l in levels.values()) + 1
    xmin, xmax = 0.0, column_x0(ncol - 1, style) + cw
    if limits:
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])
    else:
        span = max(ymax - ymin, 100.0)
        m = float(style["y_margin"]) * span
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin - m, ymax + m)

    # --- levels -------------------------------------------------------------
    for l in scheme["levels"]:
        x0, x1 = level_xspan(l, style)
        y = level_y(l)
        sel = is_sel("level", l["id"])
        col = highlight if sel else (l.get("color") or style["level_color"])
        lw = float(l.get("lw") or style["level_lw"])
        ls = (0, (4, 2)) if l.get("tentative") else "-"
        ax.plot([x0, x1], [y, y], color=col, lw=lw, ls=ls, solid_capstyle="butt",
                clip_on=False, zorder=2)
        reg("level", l["id"], "line", ("seg", (x0, y, x1, y), lw / 2 + 3))

        spin = format_spin(l.get("spin", ""), style["auto_spin_format"])
        if spin and style["spin_side"] != "none":
            left = style["spin_side"] == "left"
            dx, dy = offs(l, "spin")
            t = ax.annotate(spin, (x0 if left else x1, y),
                            xytext=((-pad if left else pad) + dx, dy),
                            textcoords="offset points", ha="right" if left else "left",
                            va="center", fontsize=fs, color=col, annotation_clip=False)
            reg("level", l["id"], "spin", ("text", t))

        elab = l.get("energy_label") or fmt_energy(float(l["energy"]), style)
        if elab and style["energy_side"] != "none":
            left = style["energy_side"] == "left"
            dx, dy = offs(l, "energy")
            t = ax.annotate(elab, (x0 if left else x1, y),
                            xytext=((-pad if left else pad) + dx, dy),
                            textcoords="offset points", ha="right" if left else "left",
                            va="center", fontsize=fs, color=col,
                            fontstyle="italic" if style["energy_italic"] else "normal",
                            annotation_clip=False)
            reg("level", l["id"], "energy", ("text", t))

        life = format_lifetime(l.get("lifetime", ""), style)
        if life:
            dx, dy = offs(l, "life")
            t = ax.annotate(life, (x1, y), xytext=(-1 + dx, lw / 2 + 1.5 + dy),
                            textcoords="offset points", ha="right", va="bottom",
                            fontsize=fs, color=col, annotation_clip=False)
            reg("level", l["id"], "life", ("text", t))

    # --- transitions --------------------------------------------------------
    trans = scheme.get("transitions", [])
    imax = max([float(t.get("intensity") or 0) for t in trans] + [1e-12])
    wmin, wmax = float(style["arrow_min_width"]), float(style["arrow_max_width"])
    hl = float(style["head_length"])
    gpad = float(style["gamma_label_pad"])
    for t in trans:
        a, b = levels.get(t["from"]), levels.get(t["to"])
        if a is None or b is None:
            continue
        ca, cb = int(a.get("column", 0)), int(b.get("column", 0))
        xa = column_x0(ca, style) + cw * float(t.get("x", 0.5))
        if ca == cb:
            xb = xa
        else:
            xt = t.get("x_to")
            xb = column_x0(cb, style) + cw * float(t.get("x", 0.5) if xt is None else xt)
        ya, yb = level_y(a), level_y(b)
        inten = float(t.get("intensity") or 0)
        tw = wmin + (wmax - wmin) * max(inten, 0) / imax
        hw = tw + float(style["head_extra"])
        sel = is_sel("trans", t["id"])
        col = highlight if sel else (t.get("color") or "black")
        if t.get("tentative"):
            # dashed shaft (stops at the head) + solid head with an invisible tail
            ax.add_patch(FancyArrowPatch((xa, ya), (xb, yb), arrowstyle="-", mutation_scale=1,
                                         shrinkA=0, shrinkB=hl * 0.9, lw=max(0.6 * tw, 0.6),
                                         ls=(0, (3, 2)), color=col, clip_on=False, zorder=3))
            ast = ArrowStyle("Simple", head_length=hl, head_width=hw, tail_width=0.0)
            patch = FancyArrowPatch((xa, ya), (xb, yb), arrowstyle=ast, mutation_scale=1,
                                    shrinkA=0, shrinkB=0, lw=0, fc=col, ec="none",
                                    clip_on=False, zorder=3)
        else:
            ast = ArrowStyle("Simple", head_length=hl, head_width=hw, tail_width=tw)
            patch = FancyArrowPatch((xa, ya), (xb, yb), arrowstyle=ast, mutation_scale=1,
                                    shrinkA=0, shrinkB=0, lw=0, fc=col, ec="none",
                                    clip_on=False, zorder=3)
        ax.add_patch(patch)
        reg("trans", t["id"], "arrow", ("seg", (xa, ya, xb, yb), hw / 2 + 2))

        label = t.get("label") or fmt_energy(abs(float(a["energy"]) - float(b["energy"])), style)
        side = t.get("label_side") or "default"
        if side == "default":
            side = style["gamma_label_side"]
        if side != "none" and label:
            left = side == "left"
            dx, dy = offs(t, "label")
            txt = ax.annotate(label, ((xa + xb) / 2, (ya + yb) / 2),
                              xytext=((-1 if left else 1) * (tw / 2 + gpad) + dx, dy),
                              textcoords="offset points", ha="right" if left else "left",
                              va="center", rotation=float(style["gamma_label_rotation"] or 0),
                              fontsize=gfs, color=col, annotation_clip=False)
            reg("trans", t["id"], "label", ("text", txt))

    # --- titles -------------------------------------------------------------
    ylo = ax.get_ylim()[0]
    xlo, xhi = ax.get_xlim()
    raw = (style.get("column_titles") or "").strip()
    titles = [s.strip() for s in raw.split("|")] if raw else []
    for c, s in enumerate(titles):
        if s:
            dx, dy = offs(style, "col%d" % c)
            tt = ax.annotate(s, (column_x0(c, style) + cw / 2, ylo), xytext=(dx, dy),
                             textcoords="offset points", ha="center", va="top",
                             fontsize=fs, annotation_clip=False)
            reg("style", 0, "col%d" % c, ("text", tt))
    if style.get("title"):
        dx, dy = offs(style, "title")
        tt = ax.annotate(style["title"], ((xmin + xmax) / 2, ylo),
                         xytext=(dx, dy - (fs * 1.8 if titles else 0)),
                         textcoords="offset points", ha="center", va="top",
                         fontsize=float(style["title_size"]), annotation_clip=False)
        reg("style", 0, "title", ("text", tt))
    return ax


EXPORTED_FUNCTIONS = [style_rc, fmt_energy, format_spin, format_lifetime,
                      column_x0, level_y, level_xspan, render]


# =============================================================================
# Model helpers
# =============================================================================

def new_level(id_, energy, column=0):
    return {"id": id_, "energy": float(energy), "spin": "", "energy_label": "",
            "lifetime": "", "column": int(column), "x0": 0.0, "x1": 1.0, "lw": None,
            "color": "", "tentative": False, "y_shift": 0.0, "offsets": {}}


def new_transition(id_, from_id, to_id, x=0.5):
    return {"id": id_, "from": from_id, "to": to_id, "label": "", "intensity": 10.0,
            "x": x, "x_to": None, "label_side": "default", "color": "",
            "tentative": False, "offsets": {}}


def example_scheme():
    """The reference scheme: 13/2+ isomer with a 23/2- isomer above it."""
    L = [new_level(1, 1556), new_level(2, 2278), new_level(3, 2506), new_level(4, 3316)]
    L[0].update(spin="13/2+", lifetime="1.09(4) ms", lw=3.0)
    L[1].update(spin="(15/2+)", tentative=True, x0=0.12, x1=0.42)
    L[2].update(spin="(17/2+)")
    L[3].update(spin="(23/2-)", lifetime="5.89(18) us", lw=3.0)
    T = [new_transition(1, 4, 3, 0.55), new_transition(2, 3, 1, 0.70),
         new_transition(3, 3, 2, 0.27), new_transition(4, 2, 1, 0.27)]
    T[0]["intensity"] = 100
    T[1]["intensity"] = 100
    T[2].update(intensity=8, label="227")
    T[3]["intensity"] = 8
    return {"levels": L, "transitions": T}


def merged_style(style):
    s = copy.deepcopy(DEFAULT_STYLE)
    s.update(style or {})
    if not isinstance(s.get("offsets"), dict):
        s["offsets"] = {}
    return s


def normalise_scheme(scheme):
    out = {"levels": [], "transitions": []}
    for l in scheme.get("levels", []):
        d = new_level(l["id"], l["energy"])
        d.update(l)
        out["levels"].append(d)
    for t in scheme.get("transitions", []):
        d = new_transition(t["id"], t["from"], t["to"])
        d.update(t)
        out["transitions"].append(d)
    return out


def save_figure(scheme, style, path):
    with matplotlib.rc_context(style_rc(style)):
        fig = Figure(figsize=(float(style["fig_width"]), float(style["fig_height"])))
        FigureCanvasAgg(fig)
        render(fig, scheme, style)
        fig.savefig(path, dpi=int(style["dpi"]),
                    bbox_inches="tight" if style["tight_bbox"] else None,
                    pad_inches=0.03, facecolor="white")


SCRIPT_HEADER = '''#!/usr/bin/env python3
"""Level scheme exported from LevelSchemer on {date}.

Edit SCHEME / STYLE below and re-run to regenerate {out}.
Offsets are label nudges in points; x / x0 / x1 are fractions of a column width.
"""
import re

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, ArrowStyle

SCHEME = {scheme}

STYLE = {style}

'''

SCRIPT_FOOTER = '''

if __name__ == "__main__":
    plt.rcParams.update(style_rc(STYLE))
    fig = plt.figure(figsize=(STYLE["fig_width"], STYLE["fig_height"]))
    render(fig, SCHEME, STYLE)
    fig.savefig("{out}", dpi=STYLE["dpi"],
                bbox_inches="tight" if STYLE["tight_bbox"] else None, pad_inches=0.03)
    plt.show()
'''


def build_script(scheme, style, out_name="level_scheme.pdf"):
    src = "\n\n".join(inspect.getsource(f) for f in EXPORTED_FUNCTIONS)
    head = SCRIPT_HEADER.format(
        date=datetime.date.today().isoformat(), out=out_name,
        scheme=pprint.pformat(scheme, sort_dicts=False, width=100),
        style=pprint.pformat(style, sort_dicts=False, width=100))
    return head + src + SCRIPT_FOOTER.format(out=out_name)


# =============================================================================
# GUI widgets
# =============================================================================

class ScrollFrame(ttk.Frame):
    """A vertically scrollable frame; put widgets in .inner."""

    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        bg = ttk.Style().lookup("TFrame", "background") or "#d9d9d9"
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0, background=bg)
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(
            scrollregion=self.canvas.bbox("all")))
        win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(win, width=e.width))
        self.canvas.configure(yscrollcommand=vsb.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self.bind("<Enter>", lambda e: self._wheel(True))
        self.bind("<Leave>", lambda e: self._wheel(False))

    def _wheel(self, on):
        if on:
            self.bind_all("<MouseWheel>", lambda e: self.canvas.yview_scroll(
                int(-e.delta / (120 if abs(e.delta) >= 120 else 1)), "units"))
            self.bind_all("<Button-4>", lambda e: self.canvas.yview_scroll(-2, "units"))
            self.bind_all("<Button-5>", lambda e: self.canvas.yview_scroll(2, "units"))
        else:
            for s in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                self.unbind_all(s)


class PropertyForm(ttk.Frame):
    """Label/editor grid bound to a dict. Field spec: (key, label, kind[, options]).

    kinds: float, optfloat, int, str, bool, choice, level, color, header
    """

    def __init__(self, master, fields, on_change):
        super().__init__(master)
        self.fields, self.vars, self.widgets = {}, {}, {}
        self.on_change = on_change
        self.target = None
        self._loading = False
        self.level_map, self.level_rev = {}, {}
        row = 0
        for spec in fields:
            key, label, kind = spec[0], spec[1], spec[2]
            opts = spec[3] if len(spec) > 3 else None
            if kind == "header":
                ttk.Label(self, text=label, style="Header.TLabel").grid(
                    row=row, column=0, columnspan=2, sticky="w", padx=4, pady=(8, 2))
                row += 1
                continue
            self.fields[key] = (kind, opts)
            ttk.Label(self, text=label).grid(row=row, column=0, sticky="w", padx=(6, 4), pady=1)
            if kind == "bool":
                var = tk.BooleanVar()
                w = ttk.Checkbutton(self, variable=var, command=lambda k=key: self._commit(k))
            elif kind in ("choice", "level"):
                var = tk.StringVar()
                w = ttk.Combobox(self, textvariable=var, values=opts or [],
                                 state="readonly", width=22)
                w.bind("<<ComboboxSelected>>", lambda e, k=key: self._commit(k))
            elif kind == "color":
                var = tk.StringVar()
                w = ttk.Frame(self)
                ent = ttk.Entry(w, textvariable=var, width=16)
                ent.pack(side="left", fill="x", expand=True)
                ttk.Button(w, text="Pick", width=5,
                           command=lambda k=key: self._pick_color(k)).pack(side="left", padx=(2, 0))
                for ev in ("<Return>", "<KP_Enter>", "<FocusOut>"):
                    ent.bind(ev, lambda e, k=key: self._commit(k))
            else:
                var = tk.StringVar()
                w = ttk.Entry(self, textvariable=var, width=24)
                for ev in ("<Return>", "<KP_Enter>", "<FocusOut>"):
                    w.bind(ev, lambda e, k=key: self._commit(k))
            w.grid(row=row, column=1, sticky="ew", padx=(0, 6), pady=1)
            self.vars[key], self.widgets[key] = var, w
            row += 1
        self.columnconfigure(1, weight=1)

    def set_level_choices(self, pairs):
        self.level_map = {d: i for i, d in pairs}
        self.level_rev = {i: d for i, d in pairs}
        for k, (kind, _) in self.fields.items():
            if kind == "level":
                self.widgets[k].configure(values=[d for _, d in pairs])

    def load(self, target):
        self.target = target
        self._loading = True
        try:
            for k, (kind, _) in self.fields.items():
                v = None if target is None else target.get(k)
                var = self.vars[k]
                if kind == "bool":
                    var.set(bool(v))
                elif kind == "level":
                    var.set(self.level_rev.get(v, ""))
                elif v is None:
                    var.set("")
                elif isinstance(v, float):
                    var.set("{:g}".format(v))
                else:
                    var.set(str(v))
        finally:
            self._loading = False

    def _parse(self, kind, raw):
        raw = raw.strip() if isinstance(raw, str) else raw
        if kind == "float":
            return float(raw)
        if kind == "optfloat":
            return None if raw == "" else float(raw)
        if kind == "int":
            return int(float(raw))
        if kind == "bool":
            return bool(raw)
        if kind == "level":
            return self.level_map[raw]
        if kind == "color":
            if raw and not mcolors.is_color_like(raw):
                raise ValueError(raw)
            return raw
        return raw

    def _commit(self, key):
        if self._loading or self.target is None:
            return
        kind, _ = self.fields[key]
        try:
            val = self._parse(kind, self.vars[key].get())
        except (ValueError, KeyError):
            self.bell()
            self.load(self.target)
            return
        if self.target.get(key) == val:
            return
        self.on_change(self.target, key, val)

    def _pick_color(self, key):
        if self.target is None:
            return
        cur = self.vars[key].get() or "black"
        try:
            init = mcolors.to_hex(cur)
        except ValueError:
            init = "#000000"
        res = colorchooser.askcolor(color=init, parent=self)
        if res and res[1]:
            self.vars[key].set(res[1])
            self._commit(key)


LEVEL_FIELDS = [
    ("energy", "Energy (keV)", "float"),
    ("spin", "Spin/parity, e.g. (23/2-)", "str"),
    ("energy_label", "Energy label (blank = auto)", "str"),
    ("lifetime", "Lifetime label, e.g. 5.9(2) us", "str"),
    ("column", "Column (0, 1, 2 ...)", "int"),
    ("x0", "Line start (0-1 of column)", "float"),
    ("x1", "Line end (0-1 of column)", "float"),
    ("lw", "Line width pt (blank = default)", "optfloat"),
    ("color", "Colour (blank = default)", "color"),
    ("tentative", "Tentative (dashed)", "bool"),
    ("y_shift", "Drawn offset (keV)", "float"),
]

TRANS_FIELDS = [
    ("from", "From level", "level"),
    ("to", "To level", "level"),
    ("label", "Label (blank = auto Eγ)", "str"),
    ("intensity", "Intensity (sets width)", "float"),
    ("x", "x position (0-1 of column)", "float"),
    ("x_to", "x at destination column", "optfloat"),
    ("label_side", "Label side", "choice", ["default", "left", "right", "none"]),
    ("color", "Colour (blank = black)", "color"),
    ("tentative", "Tentative (dashed)", "bool"),
]

STYLE_FIELDS = [
    (None, "Figure", "header"),
    ("fig_width", "Width (inches)", "float"),
    ("fig_height", "Height (inches)", "float"),
    ("margin_left", "Left margin (fraction)", "float"),
    ("margin_right", "Right margin (fraction)", "float"),
    ("margin_bottom", "Bottom margin (fraction)", "float"),
    ("margin_top", "Top margin (fraction)", "float"),
    ("y_margin", "Vertical padding (fraction)", "float"),
    ("dpi", "PNG export dpi", "int"),
    ("tight_bbox", "Crop to content on export", "bool"),
    (None, "Fonts", "header"),
    ("font_family", "Font family", "choice", ["serif", "sans-serif"]),
    ("mathtext_fontset", "Math font set", "choice", ["stix", "cm", "stixsans", "dejavusans", "dejavuserif"]),
    ("font_size", "Label size (pt)", "float"),
    ("gamma_font_size", "γ-label size (pt)", "float"),
    ("usetex", "Use LaTeX (needs TeX install)", "bool"),
    (None, "Levels", "header"),
    ("column_width", "Column width", "float"),
    ("column_gap", "Gap between columns", "float"),
    ("level_lw", "Default line width (pt)", "float"),
    ("level_color", "Default colour", "color"),
    ("energy_fmt", "Energy format", "str"),
    ("energy_italic", "Italic energies", "bool"),
    ("spin_side", "Spin labels", "choice", ["left", "right", "none"]),
    ("energy_side", "Energy labels", "choice", ["right", "left", "none"]),
    ("label_pad", "Label gap (pt)", "float"),
    ("auto_spin_format", "Auto-superscript parity", "bool"),
    (None, "Transitions", "header"),
    ("arrow_max_width", "Max arrow width (pt)", "float"),
    ("arrow_min_width", "Min arrow width (pt)", "float"),
    ("head_length", "Head length (pt)", "float"),
    ("head_extra", "Head extra width (pt)", "float"),
    ("gamma_label_side", "Label side", "choice", ["left", "right", "none"]),
    ("gamma_label_rotation", "Label rotation", "choice", ["0", "90"]),
    ("gamma_label_pad", "Label gap (pt)", "float"),
    (None, "Annotations", "header"),
    ("title", "Title, e.g. $^{94}$Pd", "str"),
    ("title_size", "Title size (pt)", "float"),
    ("column_titles", "Column titles (a | b | c)", "str"),
]

HELP_TEXT = """\
Building a scheme
  • Levels tab: add levels and type the energy, spin (e.g. (23/2-)) and lifetime
    (e.g. 5.89(18) us). Parities are superscripted and 'us' becomes μs automatically.
    LaTeX mathtext works in any label, e.g. $^{94}$Pd.
  • Transitions tab: pick From/To levels. Eγ is computed from the level energies
    unless you type a label. Arrow width scales with intensity.
  • Style tab: figure size, fonts, arrow widths, label placement.

Mouse (preview)
  • Click: select. Drag a level line up/down to separate crowded levels
    (only where it's drawn changes, not the energy).
  • Drag an arrow sideways to move it; drag any label to nudge it.
  • Right-drag from one level to another to create a transition.
  • Double-click empty space to add a level there.
  • 'Reset labels' puts the selected item's labels back.

Output
  • File ▸ Export figure: PDF / SVG / EPS / PNG at the size set in the Style tab.
  • File ▸ Export Python script: a standalone matplotlib script that reproduces
    the figure, for tweaking by hand or including in your analysis repo.
  • Save / Open store the scheme as JSON.

Keys: Delete, Ctrl+Z / Ctrl+Y, Ctrl+S, Ctrl+O, Ctrl+N
"""


# =============================================================================
# Application
# =============================================================================

class App:
    def __init__(self, root, path=None):
        self.root = root
        self.scheme = example_scheme()
        self.style = merged_style({})
        self.selected = None
        self.undo_stack, self.redo_stack = [], []
        self.path, self.dirty = None, False
        self.hits, self.ax = [], None
        self.drag = self.link = None
        self._syncing = False

        st = ttk.Style()
        st.configure("Header.TLabel", font=("TkDefaultFont", 10, "bold"))
        self._build_menu()
        self._build_layout()
        self._bind_keys()
        root.protocol("WM_DELETE_WINDOW", self.quit)
        if path:
            self.open_file(path)
        self.refresh_all()
        self.update_title()

    # ------------------------------------------------------------------ layout
    def _build_menu(self):
        m = tk.Menu(self.root)
        f = tk.Menu(m, tearoff=0)
        f.add_command(label="New", accelerator="Ctrl+N", command=self.new_file)
        f.add_command(label="Open…", accelerator="Ctrl+O", command=self.open_dialog)
        f.add_command(label="Save", accelerator="Ctrl+S", command=self.save)
        f.add_command(label="Save as…", command=self.save_as)
        f.add_separator()
        f.add_command(label="Export figure (PDF/SVG/EPS/PNG)…", accelerator="Ctrl+E",
                      command=self.export_figure)
        f.add_command(label="Export Python script…", command=self.export_script)
        f.add_separator()
        f.add_command(label="Load example scheme", command=self.load_example)
        f.add_separator()
        f.add_command(label="Quit", command=self.quit)
        m.add_cascade(label="File", menu=f)
        e = tk.Menu(m, tearoff=0)
        e.add_command(label="Undo", accelerator="Ctrl+Z", command=self.undo)
        e.add_command(label="Redo", accelerator="Ctrl+Y", command=self.redo)
        e.add_separator()
        e.add_command(label="Add level", command=self.add_level)
        e.add_command(label="Add transition", command=self.add_transition)
        e.add_command(label="Delete selected", accelerator="Del", command=self.delete_selected)
        e.add_command(label="Reset labels of selected", command=self.reset_labels)
        e.add_command(label="Reset all label positions", command=self.reset_all_labels)
        m.add_cascade(label="Edit", menu=e)
        h = tk.Menu(m, tearoff=0)
        h.add_command(label="Quick guide", command=self.show_help)
        m.add_cascade(label="Help", menu=h)
        self.root.config(menu=m)

    def _build_layout(self):
        pw = ttk.PanedWindow(self.root, orient="horizontal")
        pw.pack(fill="both", expand=True)

        left = ttk.Frame(pw, width=380)
        self.nb = ttk.Notebook(left)
        self.nb.pack(fill="both", expand=True, padx=4, pady=4)

        # Levels tab
        lt = ttk.Frame(self.nb)
        self.nb.add(lt, text="Levels")
        self.lv_tree = self._make_tree(lt, [("E", "Energy", 70), ("spin", "Spin", 80),
                                            ("life", "Lifetime", 90), ("col", "Col", 35)])
        self.lv_tree.bind("<<TreeviewSelect>>", lambda e: self._tree_select("level", self.lv_tree))
        bf = ttk.Frame(lt)
        bf.pack(fill="x", padx=4)
        for txt, cmd in (("Add", self.add_level), ("Duplicate", self.duplicate_selected),
                         ("Delete", self.delete_selected), ("Reset labels", self.reset_labels)):
            ttk.Button(bf, text=txt, command=cmd).pack(side="left", padx=1, pady=3)
        self.lv_form = PropertyForm(lt, LEVEL_FIELDS, self.on_item_change)
        self.lv_form.pack(fill="x", pady=4)

        # Transitions tab
        tt = ttk.Frame(self.nb)
        self.nb.add(tt, text="Transitions")
        self.tr_tree = self._make_tree(tt, [("E", "Eγ", 60), ("from", "From", 80),
                                            ("to", "To", 80), ("I", "I", 50)])
        self.tr_tree.bind("<<TreeviewSelect>>", lambda e: self._tree_select("trans", self.tr_tree))
        bf = ttk.Frame(tt)
        bf.pack(fill="x", padx=4)
        for txt, cmd in (("Add", self.add_transition), ("Duplicate", self.duplicate_selected),
                         ("Delete", self.delete_selected), ("Reset labels", self.reset_labels)):
            ttk.Button(bf, text=txt, command=cmd).pack(side="left", padx=1, pady=3)
        self.tr_form = PropertyForm(tt, TRANS_FIELDS, self.on_item_change)
        self.tr_form.pack(fill="x", pady=4)

        # Style tab
        sf = ScrollFrame(self.nb)
        self.nb.add(sf, text="Style")
        self.st_form = PropertyForm(sf.inner, STYLE_FIELDS, self.on_style_change)
        self.st_form.pack(fill="x", pady=4)
        pw.add(left, weight=0)

        # Preview
        right = ttk.Frame(pw)
        self.preview_box = tk.Frame(right, background="#8a8a8a")
        self.preview_box.pack(fill="both", expand=True)
        self.fig = Figure(figsize=(self.style["fig_width"], self.style["fig_height"]))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.preview_box)
        self.cwidget = self.canvas.get_tk_widget()
        self.cwidget.configure(highlightthickness=0, borderwidth=0)
        self.cwidget.bind("<Configure>", self._on_canvas_configure, add="+")
        self.preview_box.bind("<Configure>", lambda e: self._layout_canvas())
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        pw.add(right, weight=1)

        self.status_var = tk.StringVar(value="Ready. Help ▸ Quick guide explains the mouse controls.")
        ttk.Label(self.root, textvariable=self.status_var, anchor="w", relief="sunken",
                  padding=(6, 2)).pack(fill="x", side="bottom")

    def _make_tree(self, master, cols):
        fr = ttk.Frame(master)
        fr.pack(fill="both", expand=True, padx=4, pady=(4, 0))
        tree = ttk.Treeview(fr, columns=[c[0] for c in cols], show="headings",
                            height=8, selectmode="browse")
        for key, head, w in cols:
            tree.heading(key, text=head)
            tree.column(key, width=w, anchor="w")
        sb = ttk.Scrollbar(fr, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        return tree

    def _bind_keys(self):
        r = self.root
        r.bind("<Control-z>", lambda e: self.undo())
        r.bind("<Control-y>", lambda e: self.redo())
        r.bind("<Control-Z>", lambda e: self.redo())
        r.bind("<Control-s>", lambda e: self.save())
        r.bind("<Control-o>", lambda e: self.open_dialog())
        r.bind("<Control-n>", lambda e: self.new_file())
        r.bind("<Control-e>", lambda e: self.export_figure())
        r.bind("<Delete>", self._on_delete_key)

    def _on_delete_key(self, event):
        if isinstance(self.root.focus_get(), (ttk.Entry, tk.Entry, ttk.Combobox)):
            return
        self.delete_selected()

    def _layout_canvas(self):
        """Keep the preview at the export aspect ratio (true WYSIWYG)."""
        bw, bh = self.preview_box.winfo_width(), self.preview_box.winfo_height()
        if bw < 20 or bh < 20:
            return
        fw, fh = float(self.style["fig_width"]), float(self.style["fig_height"])
        scale = min((bw - 20) / fw, (bh - 20) / fh)
        w, h = max(int(fw * scale), 10), max(int(fh * scale), 10)
        self.cwidget.place(x=(bw - w) // 2, y=(bh - h) // 2, width=w, height=h)

    def _on_canvas_configure(self, event):
        fw, fh = float(self.style["fig_width"]), float(self.style["fig_height"])
        if event.width > 10:
            self.fig.set_dpi(event.width / fw)
            self.fig.set_size_inches(fw, fh, forward=False)
            self.redraw()

    # ------------------------------------------------------------ model utils
    def level(self, id_):
        return next((l for l in self.scheme["levels"] if l["id"] == id_), None)

    def trans(self, id_):
        return next((t for t in self.scheme["transitions"] if t["id"] == id_), None)

    def selected_obj(self):
        if not self.selected:
            return None
        kind, id_ = self.selected
        return self.level(id_) if kind == "level" else self.trans(id_)

    @staticmethod
    def next_id(items):
        return max([i["id"] for i in items] + [0]) + 1

    def level_name(self, l):
        return "{:g}  {}".format(float(l["energy"]), l.get("spin", "")).strip() + "  #%d" % l["id"]

    # ------------------------------------------------------------- undo/redo
    def snapshot(self):
        return json.dumps({"scheme": self.scheme, "style": self.style})

    def push_undo(self):
        self.undo_stack.append(self.snapshot())
        del self.undo_stack[:-200]
        self.redo_stack.clear()
        self.dirty = True
        self.update_title()

    def _restore(self, snap):
        d = json.loads(snap)
        self.scheme, self.style = d["scheme"], d["style"]
        if not self.selected_obj():
            self.selected = None
        self._layout_canvas()
        self.refresh_all()

    def undo(self):
        if self.undo_stack:
            self.redo_stack.append(self.snapshot())
            self._restore(self.undo_stack.pop())
            self.dirty = True
            self.update_title()

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(self.snapshot())
            self._restore(self.redo_stack.pop())
            self.dirty = True
            self.update_title()

    # --------------------------------------------------------------- refresh
    def refresh_all(self):
        self.refresh_trees()
        self.load_forms()
        self.redraw()

    def refresh_trees(self):
        self._syncing = True
        try:
            lv = self.lv_tree
            lv.delete(*lv.get_children())
            for l in sorted(self.scheme["levels"],
                            key=lambda l: (l.get("column", 0), -float(l["energy"]))):
                lv.insert("", "end", iid=str(l["id"]), values=(
                    "{:g}".format(float(l["energy"])), l.get("spin", ""),
                    l.get("lifetime", ""), l.get("column", 0)))
            tr = self.tr_tree
            tr.delete(*tr.get_children())
            for t in self.scheme["transitions"]:
                a, b = self.level(t["from"]), self.level(t["to"])
                if a and b:
                    e = t.get("label") or fmt_energy(abs(float(a["energy"]) - float(b["energy"])), self.style)
                    fa, fb = "{:g}".format(float(a["energy"])), "{:g}".format(float(b["energy"]))
                else:
                    e, fa, fb = "?", "?", "?"
                tr.insert("", "end", iid=str(t["id"]), values=(e, fa, fb, "{:g}".format(float(t.get("intensity") or 0))))
            for tree, kind in ((lv, "level"), (tr, "trans")):
                if self.selected and self.selected[0] == kind and tree.exists(str(self.selected[1])):
                    tree.selection_set(str(self.selected[1]))
                    tree.see(str(self.selected[1]))
                else:
                    tree.selection_set(())
        finally:
            self.root.after_idle(self._end_sync)

    def _end_sync(self):
        self._syncing = False

    def load_forms(self):
        pairs = [(l["id"], self.level_name(l))
                 for l in sorted(self.scheme["levels"], key=lambda l: -float(l["energy"]))]
        self.tr_form.set_level_choices(pairs)
        kind = self.selected[0] if self.selected else None
        self.lv_form.load(self.selected_obj() if kind == "level" else None)
        self.tr_form.load(self.selected_obj() if kind == "trans" else None)
        self.st_form.load(self.style)

    def redraw(self, limits=None):
        matplotlib.rcParams.update(style_rc(self.style))
        self.hits = []
        try:
            self.ax = render(self.fig, self.scheme, self.style, hits=self.hits,
                             selected=self.selected, limits=limits, highlight=HIGHLIGHT)
            self.canvas.draw()
        except Exception as exc:  # e.g. bad mathtext or missing LaTeX
            self.status("Drawing error: %s" % str(exc).splitlines()[0][:200])

    def status(self, msg):
        self.status_var.set(msg)

    def update_title(self):
        name = os.path.basename(self.path) if self.path else "untitled"
        self.root.title("%s - %s%s" % (APP_NAME, name, " *" if self.dirty else ""))

    # ------------------------------------------------------------- selection
    def select(self, sel, from_canvas=False):
        self.selected = tuple(sel) if sel else None
        if from_canvas and self.selected:
            self.nb.select(0 if self.selected[0] == "level" else 1)
        self.refresh_trees()
        self.load_forms()
        self.redraw()

    def _tree_select(self, kind, tree):
        if self._syncing:
            return
        s = tree.selection()
        if not s:
            return
        sel = (kind, int(s[0]))
        if sel != self.selected:
            self.select(sel)

    # --------------------------------------------------------------- editing
    def on_item_change(self, target, key, val):
        self.push_undo()
        target[key] = val
        self.refresh_trees()
        self.load_forms()
        self.redraw()

    def on_style_change(self, target, key, val):
        self.push_undo()
        target[key] = val
        if key in ("fig_width", "fig_height"):
            self._layout_canvas()
        if key in ("energy_fmt",):
            self.refresh_trees()
        self.redraw()

    def add_level(self, energy=None, column=None):
        lv = self.scheme["levels"]
        cur = self.selected_obj() if self.selected and self.selected[0] == "level" else None
        if column is None:
            column = cur.get("column", 0) if cur else 0
        if energy is None:
            same = [float(l["energy"]) for l in lv if l.get("column", 0) == column]
            energy = (max(same) + 500.0) if same else 0.0
        self.push_undo()
        l = new_level(self.next_id(lv), energy, column)
        lv.append(l)
        self.select(("level", l["id"]), from_canvas=True)
        self.status("Added level at %g keV. Set its spin and lifetime in the form." % energy)

    def _auto_x(self, a, b):
        col = a.get("column", 0)
        lo, hi = sorted((level_y(a), level_y(b)))
        used = []
        for t in self.scheme["transitions"]:
            ta, tb = self.level(t["from"]), self.level(t["to"])
            if not ta or not tb or ta.get("column", 0) != col:
                continue
            tlo, thi = sorted((level_y(ta), level_y(tb)))
            if tlo < hi and thi > lo:
                used.append(float(t.get("x", 0.5)))
        for x in (0.5, 0.35, 0.65, 0.2, 0.8, 0.1, 0.9):
            if all(abs(x - u) > 0.08 for u in used):
                return x
        return 0.5

    def add_transition(self, from_id=None, to_id=None, x=None):
        lv = self.scheme["levels"]
        if len(lv) < 2:
            messagebox.showinfo(APP_NAME, "Add at least two levels first.")
            return
        if from_id is None:
            cur = self.selected_obj() if self.selected and self.selected[0] == "level" else None
            a = cur or max(lv, key=lambda l: float(l["energy"]))
            lower = [l for l in lv if float(l["energy"]) < float(a["energy"])]
            same = [l for l in lower if l.get("column", 0) == a.get("column", 0)]
            if not lower:
                a = max(lv, key=lambda l: float(l["energy"]))
                lower = [l for l in lv if l is not a]
                same = [l for l in lower if l.get("column", 0) == a.get("column", 0)]
            b = max(same or lower, key=lambda l: float(l["energy"]))
            from_id, to_id = a["id"], b["id"]
        a, b = self.level(from_id), self.level(to_id)
        self.push_undo()
        t = new_transition(self.next_id(self.scheme["transitions"]), from_id, to_id,
                           self._auto_x(a, b) if x is None else x)
        self.scheme["transitions"].append(t)
        self.select(("trans", t["id"]), from_canvas=True)
        self.status("Added %s transition. Edit it in the Transitions tab."
                    % fmt_energy(abs(float(a["energy"]) - float(b["energy"])), self.style))

    def duplicate_selected(self):
        obj = self.selected_obj()
        if not obj:
            return
        self.push_undo()
        new = copy.deepcopy(obj)
        if self.selected[0] == "level":
            new["id"] = self.next_id(self.scheme["levels"])
            new["energy"] = float(new["energy"]) + 100.0
            self.scheme["levels"].append(new)
        else:
            new["id"] = self.next_id(self.scheme["transitions"])
            new["x"] = min(float(new.get("x", 0.5)) + 0.12, 1.0)
            self.scheme["transitions"].append(new)
        self.select((self.selected[0], new["id"]))

    def delete_selected(self):
        obj = self.selected_obj()
        if not obj:
            return
        self.push_undo()
        if self.selected[0] == "level":
            self.scheme["levels"].remove(obj)
            before = len(self.scheme["transitions"])
            self.scheme["transitions"] = [t for t in self.scheme["transitions"]
                                          if obj["id"] not in (t["from"], t["to"])]
            n = before - len(self.scheme["transitions"])
            self.status("Deleted level%s." % (" and %d transition(s)" % n if n else ""))
        else:
            self.scheme["transitions"].remove(obj)
            self.status("Deleted transition.")
        self.select(None)

    def reset_labels(self):
        obj = self.selected_obj()
        if not obj:
            return
        self.push_undo()
        obj["offsets"] = {}
        if self.selected[0] == "level":
            obj["y_shift"] = 0.0
        self.refresh_all()

    def reset_all_labels(self):
        self.push_undo()
        for o in self.scheme["levels"] + self.scheme["transitions"]:
            o["offsets"] = {}
        self.style["offsets"] = {}
        self.refresh_all()

    # ----------------------------------------------------------- mouse input
    def hit_test(self, px, py):
        if self.ax is None:
            return None
        try:
            renderer = self.canvas.get_renderer()
        except Exception:
            return None
        k = self.fig.dpi / 72.0
        texts = [h for h in self.hits if h[3][0] == "text"]
        arrows = [h for h in self.hits if h[2] == "arrow"]
        lines = [h for h in self.hits if h[2] == "line"]
        for h in reversed(texts):
            bb = h[3][1].get_window_extent(renderer)
            if bb.x0 - 2 <= px <= bb.x1 + 2 and bb.y0 - 2 <= py <= bb.y1 + 2:
                return h
        best, bestd = None, 1e9
        for group in (arrows, lines):
            for h in group:
                (x0, y0, x1, y1), tol = h[3][1], h[3][2]
                (ax0, ay0), (ax1, ay1) = self.ax.transData.transform([(x0, y0), (x1, y1)])
                dx, dy = ax1 - ax0, ay1 - ay0
                L2 = dx * dx + dy * dy
                u = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax0) * dx + (py - ay0) * dy) / L2))
                d = ((ax0 + u * dx - px) ** 2 + (ay0 + u * dy - py) ** 2) ** 0.5
                if d <= max(tol * k, 4) and d < bestd:
                    best, bestd = h, d
            if best:
                return best
        return None

    def _container(self, kind, id_):
        if kind == "level":
            return self.level(id_)
        if kind == "trans":
            return self.trans(id_)
        return self.style

    def on_press(self, ev):
        if self.ax is None:
            return
        hit = self.hit_test(ev.x, ev.y)
        if ev.button == 1:
            if ev.dblclick:
                if hit is None:
                    x, y = self.ax.transData.inverted().transform((ev.x, ev.y))
                    step = float(self.style["column_width"]) + float(self.style["column_gap"])
                    self.add_level(energy=float(round(y)), column=max(0, int(x // step)))
                return
            if hit is None:
                if self.selected:
                    self.select(None)
                return
            kind, id_, part, _ = hit
            if kind in ("level", "trans") and self.selected != (kind, id_):
                self.select((kind, id_), from_canvas=True)
            obj = self._container(kind, id_)
            if part == "line":
                orig = float(obj.get("y_shift") or 0.0)
            elif part == "arrow":
                orig = float(obj.get("x", 0.5))
            else:
                orig = list((obj.get("offsets") or {}).get(part) or [0.0, 0.0])
            bbox = self.ax.bbox
            xl, yl = self.ax.get_xlim(), self.ax.get_ylim()
            self.drag = {"kind": kind, "id": id_, "part": part, "x0": ev.x, "y0": ev.y,
                         "orig": orig, "pushed": False, "limits": (xl, yl),
                         "sy": (yl[1] - yl[0]) / bbox.height,
                         "sx": (xl[1] - xl[0]) / bbox.width / float(self.style["column_width"])}
        elif ev.button == 3 and hit and hit[0] == "level":
            x, y = self.ax.transData.inverted().transform((ev.x, ev.y))
            a = self.level(hit[1])
            self.link = {"from": hit[1], "x": x, "y": level_y(a)}
            self.link["line"], = self.ax.plot([x, x], [level_y(a), y], color=HIGHLIGHT,
                                             lw=1.5, ls="--", clip_on=False)
            self.canvas.draw_idle()
            self.status("Release on the destination level to create a transition.")

    def on_motion(self, ev):
        if self.link:
            x, y = self.ax.transData.inverted().transform((ev.x, ev.y))
            self.link["line"].set_data([self.link["x"], self.link["x"]], [self.link["y"], y])
            self.canvas.draw_idle()
            return
        d = self.drag
        if not d:
            if self.ax is not None:
                h = self.hit_test(ev.x, ev.y)
                self.cwidget.configure(cursor="fleur" if h else "")
            return
        dxp, dyp = ev.x - d["x0"], ev.y - d["y0"]
        if not d["pushed"]:
            if abs(dxp) + abs(dyp) < 3:
                return
            self.push_undo()
            d["pushed"] = True
        obj = self._container(d["kind"], d["id"])
        if obj is None:
            return
        part = d["part"]
        if part == "line":
            obj["y_shift"] = round(d["orig"] + dyp * d["sy"], 1)
        elif part == "arrow":
            obj["x"] = round(min(1.3, max(-0.3, d["orig"] + dxp * d["sx"])), 3)
        else:
            k = 72.0 / self.fig.dpi
            obj.setdefault("offsets", {})[part] = [round(d["orig"][0] + dxp * k, 1),
                                                   round(d["orig"][1] + dyp * k, 1)]
        self.redraw(limits=d["limits"])

    def on_release(self, ev):
        if self.link:
            link, self.link = self.link, None
            hit = self.hit_test(ev.x, ev.y)
            if hit and hit[0] == "level" and hit[1] != link["from"]:
                a = self.level(link["from"])
                x0 = column_x0(a.get("column", 0), self.style)
                frac = (link["x"] - x0) / float(self.style["column_width"])
                self.add_transition(link["from"], hit[1], round(min(1, max(0, frac)), 3))
            else:
                self.status("No transition created (release on a different level).")
                self.redraw()
            return
        if self.drag:
            moved = self.drag["pushed"]
            self.drag = None
            if moved:
                self.load_forms()
                self.redraw()

    # ------------------------------------------------------------------ files
    def _confirm_discard(self):
        if not self.dirty:
            return True
        r = messagebox.askyesnocancel(APP_NAME, "Save changes to the current scheme?")
        if r is None:
            return False
        if r:
            return self.save()
        return True

    def new_file(self):
        if not self._confirm_discard():
            return
        self.scheme = {"levels": [], "transitions": []}
        self.style = merged_style({})
        self.path, self.dirty, self.selected = None, False, None
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._layout_canvas()
        self.refresh_all()
        self.update_title()

    def load_example(self):
        self.push_undo()
        self.scheme = example_scheme()
        self.selected = None
        self.refresh_all()

    def open_dialog(self):
        if not self._confirm_discard():
            return
        p = filedialog.askopenfilename(filetypes=[("Level scheme", "*.json"), ("All files", "*")])
        if p:
            self.open_file(p)

    def open_file(self, path):
        try:
            with open(path, encoding="utf-8") as fh:
                d = json.load(fh)
            self.scheme = normalise_scheme(d)
            self.style = merged_style(d.get("style"))
        except Exception as exc:
            messagebox.showerror(APP_NAME, "Could not open %s:\n%s" % (path, exc))
            return
        self.path, self.dirty, self.selected = path, False, None
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._layout_canvas()
        self.refresh_all()
        self.update_title()
        self.status("Opened %s" % path)

    def save(self):
        if not self.path:
            return self.save_as()
        try:
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump({"format": "levelschemer", "version": 1,
                           "levels": self.scheme["levels"],
                           "transitions": self.scheme["transitions"],
                           "style": self.style}, fh, indent=1, ensure_ascii=False)
        except Exception as exc:
            messagebox.showerror(APP_NAME, "Could not save:\n%s" % exc)
            return False
        self.dirty = False
        self.update_title()
        self.status("Saved %s" % self.path)
        return True

    def save_as(self):
        p = filedialog.asksaveasfilename(defaultextension=".json",
                                         filetypes=[("Level scheme", "*.json")])
        if not p:
            return False
        self.path = p
        return self.save()

    def _stem(self):
        return os.path.splitext(os.path.basename(self.path))[0] if self.path else "level_scheme"

    def export_figure(self):
        p = filedialog.asksaveasfilename(
            defaultextension=".pdf", initialfile=self._stem() + ".pdf",
            filetypes=[("PDF", "*.pdf"), ("SVG", "*.svg"), ("EPS", "*.eps"), ("PNG", "*.png")])
        if not p:
            return
        try:
            save_figure(self.scheme, self.style, p)
        except Exception as exc:
            messagebox.showerror(APP_NAME, "Export failed:\n%s" % exc)
            return
        finally:
            matplotlib.rcParams.update(style_rc(self.style))
        self.status("Exported %s" % p)

    def export_script(self):
        p = filedialog.asksaveasfilename(defaultextension=".py",
                                         initialfile=self._stem() + "_plot.py",
                                         filetypes=[("Python", "*.py")])
        if not p:
            return
        try:
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(build_script(self.scheme, self.style, self._stem() + ".pdf"))
        except Exception as exc:
            messagebox.showerror(APP_NAME, "Could not write script:\n%s" % exc)
            return
        self.status("Wrote %s" % p)

    def show_help(self):
        w = tk.Toplevel(self.root)
        w.title("Quick guide")
        txt = tk.Text(w, width=86, height=30, wrap="word", padx=10, pady=10)
        txt.insert("1.0", HELP_TEXT)
        txt.configure(state="disabled")
        txt.pack(fill="both", expand=True)

    def quit(self):
        if self._confirm_discard():
            self.root.destroy()


def main():
    root = tk.Tk()
    root.geometry("1150x760")
    App(root, sys.argv[1] if len(sys.argv) > 1 else None)
    root.mainloop()


if __name__ == "__main__":
    main()
