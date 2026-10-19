#!/usr/bin/env python3
"""
LevelSchemer - a small GUI for drawing publication-quality nuclear level
schemes with matplotlib.

Requirements: Python 3.8+, matplotlib (>= 3.3), Tkinter (ships with Python;
on Linux you may need `sudo apt install python3-tk`).

Run:   python levelschemer.py [scheme.json]

A figure holds one or more schemes (e.g. neighbouring nuclei) drawn side by
side on a common energy scale. Each scheme can be shifted in energy and has
its own title.

Mouse controls in the preview
  Left-click            select a level / transition / scheme title
  Drag a level line     shift where it is drawn (its energy is unchanged)
  Drag an arrow         move it sideways
  Drag any label        reposition it
  Right-drag level->level   create a transition between them
  Double-click empty    add a new level at that energy / scheme / column
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
from matplotlib.transforms import Affine2D, ScaledTranslation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_agg import FigureCanvasAgg

APP_NAME = "LevelSchemer"
HIGHLIGHT = "#f57d15"
FILE_VERSION = 3

DEFAULT_STYLE = {
    # figure
    "fig_width": 3.4, "fig_height": 4.0, "dpi": 600, "tight_bbox": True,
    "margin_left": 0.22, "margin_right": 0.18, "margin_bottom": 0.06, "margin_top": 0.06,
    "y_margin": 0.05,
    # fonts
    "font_family": "serif", "mathtext_fontset": "stix", "usetex": False,
    "font_size": 9.0, "gamma_font_size": 8.5,
    # schemes
    "scheme_gap": 1.0, "scheme_title_pos": "below", "scheme_title_size": 10.0,
    # levels
    "column_width": 1.0, "column_gap": 0.8, "level_lw": 1.0, "level_color": "black",
    "energy_italic": True,
    "spin_side": "left", "energy_side": "right", "label_pad": 4.0,
    "auto_spin_format": True,
    "fan_close_levels": True, "label_min_sep": 11.0, "level_min_gap": 4.0,
    "connector_stub": 4.0, "connector_diag": 8.0,
    # transitions
    "arrow_max_width": 5.0, "arrow_min_width": 0.8, "head_length": 6.0, "head_extra": 4.0,
    "intensity_norm": "per scheme",
    "gamma_label_side": "left", "gamma_label_rotation": "0", "gamma_label_pad": 3.0,
    "show_bwu": True,
    # annotations
    "title": "", "title_size": 12.0,
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


def energy_text(level):
    """A level's energy exactly as typed in the GUI (e.g. '1556', '2506.0', '1.642')."""
    t = str(level.get("energy_text") or "").strip()
    if t:
        return t
    v = float(level["energy"])
    return str(int(v)) if v.is_integer() else repr(v)


def gamma_energy_text(a, b):
    """Eγ from two levels, with as many decimals as the more precise of the two."""
    d = max(len(m.group(1)) if m else 0
            for m in (re.search(r"\.(\d+)", energy_text(x)) for x in (a, b)))
    return "{:.{}f}".format(abs(float(a["energy"]) - float(b["energy"])), d)


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


def spread_positions(ys, sep, extra=None):
    """Spread 1-D label positions so neighbours are at least `sep` apart.

    extra[i] is additional room needed above label i (e.g. a lifetime printed
    over the energy). Labels that would collide are merged into groups; each group
    is laid out at that spacing and centred on its members' wanted positions.
    Returns the new positions in the input order.
    """
    n = len(ys)
    extra = list(extra) if extra else [0.0] * n
    order = sorted(range(n), key=lambda i: ys[i])

    def place(g):
        offs, o = [], 0.0
        for k, j in enumerate(g):
            if k:
                o += sep + extra[g[k - 1]]
            offs.append(o)
        c = sum(ys[j] - offs[k] for k, j in enumerate(g)) / len(g)
        return [c + o for o in offs]

    groups = []
    for i in order:
        groups.append([i])
        while len(groups) > 1:
            g1, g2 = groups[-2], groups[-1]
            if place(g2)[0] - (place(g1)[-1] + extra[g1[-1]]) < sep:
                groups[-2:] = [g1 + g2]
            else:
                break
    out = list(ys)
    for g in groups:
        for j, p in zip(g, place(g)):
            out[j] = p
    return out


def fan_close_levels(items, min_sep, min_gap, life_h=0.0):
    """Handle levels in one column that are too close for their labels.

    items: [(id, y_pts, has_lifetime), ...] with y in printed points.
    Each level closer than `min_sep` to the one below is nudged up (if needed) so
    the drawn lines are at least `min_gap` apart; levels above are not moved.
    Labels of those levels (and any neighbours they push) are then fanned out to
    `min_sep` spacing, with `life_h` extra room above a level that has a lifetime.
    Returns (raise_pts {id: dy}, label_offset_pts {id: dy from the drawn line}).
    """
    raised, drawn, close = {}, [], set()
    prev = None                              # (id, true y, drawn y)
    for id_, y, life in sorted(items, key=lambda it: it[1]):
        q = y
        if prev is not None and y - prev[1] < min_sep:
            close.update((prev[0], id_))
            q = max(y, prev[2] + min_gap)
        if q != y:
            raised[id_] = q - y
        drawn.append((id_, q, life))
        prev = (id_, y, q)
    pos = spread_positions([q for _, q, _ in drawn], min_sep,
                           [life_h if (life and id_ in close) else 0.0 for id_, _, life in drawn])
    labels = {}
    for (id_, q, _), p in zip(drawn, pos):
        if id_ in close or abs(p - q) > 0.25:
            labels[id_] = p - q
    return raised, labels


def format_bwu(multipolarity, value):
    """('E2', '4.5(3)') -> 'B(E2) = 4.5(3) W.u.'; '2.1(+5-4)' becomes 2.1 with
    superscript +5 / subscript -4. Returns '' when no value is given."""
    v = (value or "").strip()
    if not v:
        return ""
    m = re.match(r"^(.*?)\(\s*\+\s*(\d+)\s*[-−]\s*(\d+)\s*\)\s*$", v)
    if m:
        v = m.group(1).strip() + r"$^{+%s}_{-%s}$" % (m.group(2), m.group(3))
    rel = "="
    m = re.match(r"^(<=|>=|[<>≤≥~≈])\s*(.*)$", v)
    if m:                                   # limits / approximate: B(E2) > 12 W.u.
        rel = {"<=": "≤", ">=": "≥", "~": "≈"}.get(m.group(1), m.group(1))
        v = m.group(2)
    mp = (multipolarity or "").strip()
    return ("B(%s) %s " % (mp, rel) if mp else ("" if rel == "=" else rel + " ")) + v + " W.u."


def element_z(symbol):
    """Proton number for an element symbol ('pd' -> 46), or None if unknown."""
    names = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni "
             "Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe "
             "Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg "
             "Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg "
             "Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og").split()
    s = (symbol or "").strip().capitalize()
    return names.index(s) + 1 if s in names else None


def scheme_title(s):
    """Build the mathtext title of a scheme from its nuclide fields.

    Fields: mass (A), isomer ('m', 'm1'...), symbol, show_z, show_n, charge ('46+'),
    note (plain text after the name). A non-empty 'title' overrides everything.
    """
    if (s.get("title") or "").strip():
        return s["title"]
    sym = (s.get("symbol") or "").strip()
    sym = sym[:1].upper() + sym[1:].lower()
    a = str(s.get("mass") if s.get("mass") is not None else "").strip()
    iso = (s.get("isomer") or "").strip()
    z = element_z(sym)
    pre_sup = a + (r"\mathrm{%s}" % iso if iso else "")
    pre_sub = str(z) if (s.get("show_z") and z) else ""
    k = len(a + iso) - len(pre_sub)
    if pre_sub and k > 0:            # right-align Z under a longer mass/isomer label
        pre_sub = r"\phantom{\mathrm{%s}}%s" % ((a + iso)[:k], pre_sub)
    n = ""
    if s.get("show_n") and z and a.isdigit() and int(a) >= z:
        n = str(int(a) - z)
    charge = re.sub(r"([+\-])", r"{\1}", (s.get("charge") or "").replace(" ", ""))
    out = ""
    if pre_sup or pre_sub:
        out += "${}" + ("^{%s}" % pre_sup if pre_sup else "") + ("_{%s}" % pre_sub if pre_sub else "") + "$"
    out += sym
    if charge or n:
        out += "${}" + ("^{%s}" % charge if charge else "") + ("_{%s}" % n if n else "") + "$"
    note = (s.get("note") or "").strip()
    if note:
        out += (" " if out else "") + note
    return out


def column_x0(col, style):
    """Left edge of a column, relative to the left edge of its scheme."""
    return int(col) * (float(style["column_width"]) + float(style["column_gap"]))


def scheme_ncol(scheme):
    return max([int(l.get("column", 0)) for l in scheme.get("levels", [])] + [0]) + 1


def scheme_layout(doc, style):
    """[(scheme, x_left, width), ...] for the schemes placed left to right."""
    cw, cg = float(style["column_width"]), float(style["column_gap"])
    out, x = [], 0.0
    for s in doc.get("schemes", []):
        n = scheme_ncol(s)
        w = n * cw + (n - 1) * cg
        out.append((s, x, w))
        x += w + float(style["scheme_gap"])
    return out


def level_y(level, shift=0.0):
    """Drawn height: energy + the scheme's shift + the level's drawing nudge."""
    return float(level["energy"]) + float(shift or 0.0) + float(level.get("y_shift") or 0.0)


def level_xspan(level, style, base=0.0):
    x0 = base + column_x0(level.get("column", 0), style)
    w = float(style["column_width"])
    return x0 + w * float(level.get("x0", 0.0)), x0 + w * float(level.get("x1", 1.0))


def render(fig, doc, style, hits=None, selected=None, limits=None, highlight="#f57d15"):
    """Draw one or more level schemes, side by side, into `fig`. Returns the axes.

    doc: {"schemes": [{"title", "shift", "column_titles", "levels", "transitions"}, ...]}
    hits/selected/limits/highlight are only used by the interactive editor.
    """
    fig.clf()
    fig.patch.set_facecolor("white")
    ml, mr = float(style["margin_left"]), float(style["margin_right"])
    mb, mt = float(style["margin_bottom"]), float(style["margin_top"])
    ax = fig.add_axes([ml, mb, max(0.05, 1 - ml - mr), max(0.05, 1 - mb - mt)])
    ax.set_axis_off()

    layout = scheme_layout(doc, style)
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

    # --- limits (common energy scale for all schemes) ------------------------
    ys = [level_y(l, s.get("shift")) for s, _, _ in layout for l in s.get("levels", [])]
    xmax = (layout[-1][1] + layout[-1][2]) if layout else 1.0
    if limits:
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])
    elif ys:
        ymin, ymax = min(ys), max(ys)
        m = float(style["y_margin"]) * max(ymax - ymin, 100.0)
        ax.set_xlim(0.0, xmax)
        ax.set_ylim(ymin - m, ymax + m)
    else:
        ax.set_xlim(0.0, xmax)
        ax.set_ylim(0.0, 1000.0)
    if not ys and hits is not None:
        ax.text(0.5, 0.5, "Double-click here to add a level", ha="center",
                va="center", color="0.6", transform=ax.transAxes)

    wmin, wmax = float(style["arrow_min_width"]), float(style["arrow_max_width"])
    hl = float(style["head_length"])
    gpad = float(style["gamma_label_pad"])
    all_i = [float(t.get("intensity") or 0) for s, _, _ in layout for t in s.get("transitions", [])]
    global_imax = max(all_i + [1e-12])

    # points per energy unit on the printed figure (for fanning out close levels)
    fig_h = fig.get_size_inches()[1] * 72.0 * max(0.05, 1 - mb - mt)
    ylo0, yhi0 = ax.get_ylim()
    pts_per_kev = fig_h / max(yhi0 - ylo0, 1e-9)
    stub, diag = float(style["connector_stub"]), float(style["connector_diag"])
    lowest_label = None                  # lowest fanned label (points), to keep titles clear

    for s, base, _w in layout:
        shift = float(s.get("shift") or 0.0)
        levels = {l["id"]: l for l in s.get("levels", [])}

        # --- nudge close levels apart and fan out their labels (per column) ---
        raised, fanned = {}, {}
        if style.get("fan_close_levels", True):
            cols = {}
            for l in s.get("levels", []):
                cols.setdefault(int(l.get("column", 0)), []).append(
                    (l["id"], level_y(l, shift) * pts_per_kev, bool(l.get("lifetime"))))
            for items in cols.values():
                r, f = fan_close_levels(items, float(style["label_min_sep"]),
                                        float(style["level_min_gap"]), 1.05 * fs)
                raised.update(r)
                fanned.update(f)
        ydraw = {i: level_y(levels[i], shift) + dy / pts_per_kev for i, dy in raised.items()}
        for i, dy in fanned.items():
            yl = ydraw.get(i, level_y(levels[i], shift)) * pts_per_kev + dy
            lowest_label = yl if lowest_label is None else min(lowest_label, yl)

        def side_label(text, l, part, left, anchor, col, **kw):
            """Spin/energy label beside the line, or fanned out with a dashed connector:
            horizontal stub at the level, diagonal, horizontal stub into the label."""
            sgn = -1 if left else 1
            dx, dy = offs(l, part)
            if l["id"] in fanned:
                dyl = fanned[l["id"]] + dy
                xs = [0, sgn * stub, sgn * (stub + diag) + dx, sgn * (2 * stub + diag) + dx]
                ys = [0, 0, dyl, dyl]
                tr = (Affine2D().scale(1 / 72.0) + fig.dpi_scale_trans
                      + ScaledTranslation(anchor[0], anchor[1], ax.transData))
                ax.plot(xs, ys, transform=tr, color=col, lw=0.6, ls=(0, (2, 1.5)),
                        clip_on=False, zorder=2)
                xt = sgn * (2 * stub + diag + pad) + dx
            else:
                dyl, xt = dy, sgn * pad + dx
            t = ax.annotate(text, anchor, xytext=(xt, dyl), textcoords="offset points",
                            ha="right" if left else "left", va="center", fontsize=fs,
                            color=col, annotation_clip=False, **kw)
            reg("level", l["id"], part, ("text", t))
            return xt, dyl

        # --- levels ---------------------------------------------------------
        for l in s.get("levels", []):
            x0, x1 = level_xspan(l, style, base)
            y = ydraw.get(l["id"], level_y(l, shift))
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
                side_label(spin, l, "spin", left, (x0 if left else x1, y), col)

            elab = l.get("energy_label") or energy_text(l)
            epos = None
            if elab and style["energy_side"] != "none":
                left = style["energy_side"] == "left"
                eanchor = (x0 if left else x1, y)
                epos = (left, eanchor) + side_label(
                    elab, l, "energy", left, eanchor, col,
                    fontstyle="italic" if style["energy_italic"] else "normal")

            life = format_lifetime(l.get("lifetime", ""), style)
            if life:
                dx, dy = offs(l, "life")
                if l["id"] in fanned and epos:   # keep it just above the fanned energy label
                    left, eanchor, xt, dyl = epos
                    t = ax.annotate(life, eanchor, xytext=(xt + dx, dyl + 0.6 * fs + dy),
                                    textcoords="offset points", ha="right" if left else "left",
                                    va="bottom", fontsize=fs, color=col, annotation_clip=False)
                else:
                    t = ax.annotate(life, (x1, y), xytext=(-1 + dx, lw / 2 + 1.5 + dy),
                                    textcoords="offset points", ha="right", va="bottom",
                                    fontsize=fs, color=col, annotation_clip=False)
                reg("level", l["id"], "life", ("text", t))

        # --- transitions ----------------------------------------------------
        trans = s.get("transitions", [])
        if style.get("intensity_norm") == "global":
            imax = global_imax
        else:
            imax = max([float(t.get("intensity") or 0) for t in trans] + [1e-12])
        for t in trans:
            a, b = levels.get(t["from"]), levels.get(t["to"])
            if a is None or b is None:
                continue
            ca, cb = int(a.get("column", 0)), int(b.get("column", 0))
            xa = base + column_x0(ca, style) + cw * float(t.get("x", 0.5))
            if ca == cb:
                xb = xa
            else:
                xt = t.get("x_to")
                xb = base + column_x0(cb, style) + cw * float(t.get("x", 0.5) if xt is None else xt)
            ya = ydraw.get(a["id"], level_y(a, shift))
            yb = ydraw.get(b["id"], level_y(b, shift))
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
            else:
                ast = ArrowStyle("Simple", head_length=hl, head_width=hw, tail_width=tw)
            ax.add_patch(FancyArrowPatch((xa, ya), (xb, yb), arrowstyle=ast, mutation_scale=1,
                                         shrinkA=0, shrinkB=0, lw=0, fc=col, ec="none",
                                         clip_on=False, zorder=3))
            reg("trans", t["id"], "arrow", ("seg", (xa, ya, xb, yb), hw / 2 + 2))

            label = t.get("label") or gamma_energy_text(a, b)
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

            bl = format_bwu(t.get("multipolarity"), t.get("bwu")) if style.get("show_bwu", True) else ""
            if bl:
                # opposite side of the arrow from the Eγ label
                left = side == "right"
                dx, dy = offs(t, "bwu")
                txt = ax.annotate(bl, ((xa + xb) / 2, (ya + yb) / 2),
                                  xytext=((-1 if left else 1) * (tw / 2 + gpad) + dx, dy),
                                  textcoords="offset points", ha="right" if left else "left",
                                  va="center", rotation=float(style["gamma_label_rotation"] or 0),
                                  fontsize=gfs, color=col, annotation_clip=False)
                reg("trans", t["id"], "bwu", ("text", txt))

    # --- titles: column titles, then scheme titles, then figure title --------
    ylo, yhi = ax.get_ylim()
    sts = float(style["scheme_title_size"])
    below = 0.0
    if lowest_label is not None:         # fanned labels below the bottom of the axes
        below = max(0.0, ylo * pts_per_kev - (lowest_label - 0.7 * fs) + 2.0)
    any_cols = False
    for s, base, _w in layout:
        raw = (s.get("column_titles") or "").strip()
        sel = is_sel("scheme", s["id"])
        for c, txt in enumerate([p.strip() for p in raw.split("|")] if raw else []):
            if txt:
                any_cols = True
                dx, dy = offs(s, "col%d" % c)
                tt = ax.annotate(txt, (base + column_x0(c, style) + cw / 2, ylo),
                                 xytext=(dx, dy), textcoords="offset points", ha="center",
                                 va="top", fontsize=fs, color=highlight if sel else "black",
                                 annotation_clip=False)
                reg("scheme", s["id"], "col%d" % c, ("text", tt))
    if any_cols:
        below += fs * 1.6
    pos = style.get("scheme_title_pos", "below")
    any_titles = False
    for s, base, w in layout:
        title = scheme_title(s)
        if not title or pos == "none":
            continue
        any_titles = True
        sel = is_sel("scheme", s["id"])
        dx, dy = offs(s, "title")
        if pos == "above":
            tt = ax.annotate(title, (base + w / 2, yhi), xytext=(dx, fs * 1.4 + dy),
                             textcoords="offset points", ha="center", va="bottom",
                             fontsize=sts, color=highlight if sel else "black",
                             annotation_clip=False)
        else:
            tt = ax.annotate(title, (base + w / 2, ylo), xytext=(dx, dy - below),
                             textcoords="offset points", ha="center", va="top",
                             fontsize=sts, color=highlight if sel else "black",
                             annotation_clip=False)
        reg("scheme", s["id"], "title", ("text", tt))
    if any_titles and pos == "below":
        below += sts * 1.7
    if style.get("title"):
        dx, dy = offs(style, "title")
        tt = ax.annotate(style["title"], (xmax / 2, ylo), xytext=(dx, dy - below),
                         textcoords="offset points", ha="center", va="top",
                         fontsize=float(style["title_size"]), annotation_clip=False)
        reg("style", 0, "title", ("text", tt))
    return ax


EXPORTED_FUNCTIONS = [style_rc, energy_text, gamma_energy_text, format_spin, format_lifetime,
                      spread_positions, fan_close_levels, format_bwu, element_z, scheme_title, column_x0,
                      scheme_ncol, scheme_layout, level_y, level_xspan, render]


# =============================================================================
# Model helpers
# =============================================================================

def new_level(id_, energy, column=0):
    return {"id": id_, "energy": float(energy), "spin": "", "energy_label": "",
            "lifetime": "", "column": int(column), "x0": 0.0, "x1": 1.0, "lw": None,
            "color": "", "tentative": False, "y_shift": 0.0, "energy_text": "",
            "offsets": {}}


def new_transition(id_, from_id, to_id, x=0.5):
    return {"id": id_, "from": from_id, "to": to_id, "label": "", "intensity": 10.0,
            "x": x, "x_to": None, "label_side": "default", "color": "",
            "tentative": False, "multipolarity": "", "bwu": "", "offsets": {}}


def new_scheme(id_, title=""):
    return {"id": id_, "mass": None, "isomer": "", "symbol": "", "show_z": False,
            "show_n": False, "charge": "", "note": "", "title": title,
            "shift": 0.0, "column_titles": "", "offsets": {}, "levels": [], "transitions": []}


_OLD_TITLE_RE = re.compile(
    r"^\s*\$\s*(?:\{\})?\^\{?\s*(\d+)\s*(?:\\mathrm\{(\w+)\}|([a-z]\d?))?\s*\}?\s*\$"   # $^{94m}$
    r"\s*([A-Za-z]{1,2})"                                                        # Ru
    r"(?:\s*\$\s*(?:\{\})?\^\{?\s*(\d+\s*[+-])\s*\}?\s*\$)?"                      # $^{46+}$
    r"\s*(.*?)\s*$")                                                             # note


def split_title(title):
    """Convert an old mathtext title like '$^{96}$Pd (Exp.)' into nuclide fields.

    Returns a dict of fields, or {"title": title} when it isn't recognisable.
    """
    m = _OLD_TITLE_RE.match(title or "")
    if not m or element_z(m.group(4)) is None or "$" in m.group(6):
        return {"title": title or ""}
    return {"mass": int(m.group(1)), "isomer": m.group(2) or m.group(3) or "",
            "symbol": m.group(4).capitalize(), "charge": (m.group(5) or "").replace(" ", ""),
            "note": m.group(6), "title": ""}


def scheme_label(s):
    """Plain-text name for lists and the scheme selector, e.g. '94mRu (Exp.)'."""
    if (s.get("title") or "").strip():
        return s["title"]
    parts = "%s%s%s" % (s.get("mass") if s.get("mass") is not None else "",
                        s.get("isomer") or "", (s.get("symbol") or "").capitalize())
    if s.get("charge"):
        parts += " " + s["charge"]
    if s.get("note"):
        parts += " " + s["note"]
    return parts.strip()


def empty_doc():
    return {"schemes": [new_scheme(1)]}


def example_doc():
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
    s = new_scheme(1)
    s.update(levels=L, transitions=T)
    return {"schemes": [s]}


def merged_style(style):
    s = copy.deepcopy(DEFAULT_STYLE)
    s.update(style or {})
    s.pop("column_titles", None)          # moved into each scheme in file v2
    s.pop("energy_fmt", None)             # energies are now shown as typed
    s.pop("energy_decimals", None)
    for k in ("stack_close_levels", "stack_width", "stack_gap", "spread_labels"):
        s.pop(k, None)                    # options from earlier close-level layouts
    if not isinstance(s.get("offsets"), dict):
        s["offsets"] = {}
    return s


def _norm_scheme(raw, sid):
    s = new_scheme(sid)
    s.update({k: v for k, v in raw.items() if k not in ("levels", "transitions")})
    s["id"] = sid
    if "symbol" not in raw:                      # file v1/v2: title was typed as mathtext
        s.update(split_title(raw.get("title", "")))
    for l in raw.get("levels", []):
        d = new_level(l["id"], l["energy"])
        d.update(l)
        s["levels"].append(d)
    for t in raw.get("transitions", []):
        d = new_transition(t["id"], t["from"], t["to"])
        d.update(t)
        s["transitions"].append(d)
    return s


def normalise_doc(d):
    """Accept v2 files ({"schemes": [...]}) and v1 files (one scheme at top level)."""
    if "schemes" in d:
        schemes = [_norm_scheme(s, i + 1) for i, s in enumerate(d["schemes"])]
    else:
        old = (d.get("style") or {})
        s = _norm_scheme({"levels": d.get("levels", []), "transitions": d.get("transitions", []),
                          "column_titles": old.get("column_titles", "")}, 1)
        offs = old.get("offsets") or {}
        s["offsets"] = {k: v for k, v in offs.items() if k.startswith("col")}
        schemes = [s]
    if not schemes:
        schemes = [new_scheme(1)]
    # make level / transition ids unique across the whole document
    nl = nt = 0
    for s in schemes:
        remap = {}
        for l in s["levels"]:
            nl += 1
            remap[l["id"]] = nl
            l["id"] = nl
        for t in s["transitions"]:
            nt += 1
            t["id"] = nt
            t["from"] = remap.get(t["from"], -1)
            t["to"] = remap.get(t["to"], -1)
    return {"schemes": schemes}


def save_figure(doc, style, path):
    with matplotlib.rc_context(style_rc(style)):
        fig = Figure(figsize=(float(style["fig_width"]), float(style["fig_height"])))
        FigureCanvasAgg(fig)
        render(fig, doc, style)
        fig.savefig(path, dpi=int(style["dpi"]),
                    bbox_inches="tight" if style["tight_bbox"] else None,
                    pad_inches=0.03, facecolor="white")


SCRIPT_HEADER = '''#!/usr/bin/env python3
"""Level scheme(s) exported from LevelSchemer on {date}.

Edit DOC / STYLE below and re-run to regenerate {out}.
Each entry in DOC["schemes"] is drawn side by side on a common energy scale
(shifted by its "shift" in keV). Offsets are label nudges in points;
x / x0 / x1 are fractions of a column width.
"""
import re

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, ArrowStyle
from matplotlib.transforms import Affine2D, ScaledTranslation

DOC = {doc}

STYLE = {style}

'''

SCRIPT_FOOTER = '''

if __name__ == "__main__":
    plt.rcParams.update(style_rc(STYLE))
    fig = plt.figure(figsize=(STYLE["fig_width"], STYLE["fig_height"]))
    render(fig, DOC, STYLE)
    fig.savefig("{out}", dpi=STYLE["dpi"],
                bbox_inches="tight" if STYLE["tight_bbox"] else None, pad_inches=0.03)
    plt.show()
'''


def build_script(doc, style, out_name="level_scheme.pdf"):
    src = "\n\n".join(inspect.getsource(f) for f in EXPORTED_FUNCTIONS)
    head = SCRIPT_HEADER.format(
        date=datetime.date.today().isoformat(), out=out_name,
        doc=pprint.pformat(doc, sort_dicts=False, width=100),
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

    kinds: float, optfloat, int, optint, str, bool, choice, level, color, header,
           computed (read-only text; options = function(target) -> str)
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
            if kind == "computed":
                var = tk.StringVar()
                w = ttk.Label(self, textvariable=var, foreground="#666666")
            elif kind == "bool":
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
            for k, (kind, opts) in self.fields.items():
                v = None if target is None else target.get(k)
                var = self.vars[k]
                if kind == "computed":
                    var.set(opts(target))
                elif kind == "energy":
                    var.set(energy_text(target) if target is not None else "")
                elif kind == "bool":
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
        if kind == "energy":                 # value + the text exactly as typed
            return (float(raw), raw)
        if kind == "optfloat":
            return None if raw == "" else float(raw)
        if kind == "int":
            return int(float(raw))
        if kind == "optint":
            return None if raw == "" else int(raw)
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
        if kind == "computed":
            return
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


def _zn_info(s):
    if s is None:
        return ""
    if (s.get("title") or "").strip():
        return "Override title is set, so these fields are ignored"
    z = element_z(s.get("symbol"))
    if not (s.get("symbol") or "").strip():
        return "Enter an element symbol"
    if z is None:
        return "Unknown element symbol"
    a = s.get("mass")
    return "Z = %d" % z + (", N = %d" % (a - z) if isinstance(a, int) and a >= z else "")


SCHEME_FIELDS = [
    (None, "Nuclide title", "header"),
    ("mass", "Mass number A", "optint"),
    ("isomer", "Isomer label (m, m1, m2 …)", "str"),
    ("symbol", "Element symbol, e.g. Pd", "str"),
    ("zn", "", "computed", _zn_info),
    ("show_z", "Show Z (subscript before)", "bool"),
    ("show_n", "Show N (subscript after)", "bool"),
    ("charge", "Charge state, e.g. 46+", "str"),
    ("note", "Note after name, e.g. (Exp.)", "str"),
    ("title", "Override title (optional)", "str"),
    (None, "Placement", "header"),
    ("shift", "Energy shift (keV)", "float"),
    ("column_titles", "Column titles (a | b | c)", "str"),
]

LEVEL_FIELDS = [
    ("energy", "Energy (keV)", "energy"),
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
    ("multipolarity", "Multipolarity, e.g. E2", "str"),
    ("bwu", "B(σλ) in W.u., e.g. 4.5(3)", "str"),
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
    (None, "Schemes", "header"),
    ("scheme_gap", "Gap between schemes", "float"),
    ("scheme_title_pos", "Scheme titles", "choice", ["below", "above", "none"]),
    ("scheme_title_size", "Scheme title size (pt)", "float"),
    (None, "Levels", "header"),
    ("column_width", "Column width", "float"),
    ("column_gap", "Gap between columns", "float"),
    ("level_lw", "Default line width (pt)", "float"),
    ("level_color", "Default colour", "color"),
    ("energy_italic", "Italic energies", "bool"),
    ("spin_side", "Spin labels", "choice", ["left", "right", "none"]),
    ("energy_side", "Energy labels", "choice", ["right", "left", "none"]),
    ("label_pad", "Label gap (pt)", "float"),
    ("auto_spin_format", "Auto-superscript parity", "bool"),
    ("fan_close_levels", "Fan out labels of close levels", "bool"),
    ("label_min_sep", "Closeness threshold / label spacing (pt)", "float"),
    ("level_min_gap", "Min. gap between close lines (pt)", "float"),
    ("connector_stub", "Connector horizontal stubs (pt)", "float"),
    ("connector_diag", "Connector diagonal reach (pt)", "float"),
    (None, "Transitions", "header"),
    ("arrow_max_width", "Max arrow width (pt)", "float"),
    ("arrow_min_width", "Min arrow width (pt)", "float"),
    ("head_length", "Head length (pt)", "float"),
    ("head_extra", "Head extra width (pt)", "float"),
    ("intensity_norm", "Width scaling", "choice", ["per scheme", "global"]),
    ("gamma_label_side", "Label side", "choice", ["left", "right", "none"]),
    ("gamma_label_rotation", "Label rotation", "choice", ["0", "90"]),
    ("gamma_label_pad", "Label gap (pt)", "float"),
    ("show_bwu", "Show B(σλ) labels", "bool"),
    (None, "Annotations", "header"),
    ("title", "Figure title (under everything)", "str"),
    ("title_size", "Title size (pt)", "float"),
]

HELP_TEXT = """\
Schemes
  • A figure holds one or more schemes (e.g. neighbouring nuclei) side by side on a
    common energy scale. Pick the active scheme in the selector at the top left;
    the Levels and Transitions tabs always show the active scheme.
  • Schemes tab: add, duplicate, delete and reorder (◀ ▶) schemes, and set each
    scheme's title, energy shift and column titles.
  • Titles are built from fields: mass number A, isomer label (m, m1 …), element
    symbol, optional Z / N subscripts (worked out from the symbol and A), charge
    state (e.g. 46+) and a note (e.g. (Exp.)). A=94, isomer=m, symbol=Ru gives ⁹⁴ᵐRu.
    For anything else (e.g. 'Shell model') type an Override title, which replaces
    the built one; it accepts plain text or mathtext.
  • Energy shift (keV) moves the whole scheme up or down on the common scale, e.g.
    to align a chosen level across schemes. Energy labels still show the
    energies you typed.
  • Clicking anything in the preview makes its scheme the active one.

Building a scheme
  • Levels tab: add levels and type the energy, spin (e.g. (23/2-)) and lifetime
    (e.g. 5.89(18) us). Parities are superscripted and 'us' becomes μs automatically.
    LaTeX mathtext works in any label.
  • Transitions tab: pick From/To levels (within the active scheme). Eγ is computed
    from the level energies unless you type a label. Arrow width scales with
    intensity (per scheme, or across all schemes: Style ▸ Width scaling).
  • Transition strengths: enter a Multipolarity (E2, M1 …) and B(σλ) in W.u.
    (e.g. 4.5(3), >12 or 2.1(+5-4) for asymmetric errors). It is drawn as
    'B(E2) = 4.5(3) W.u.' on the opposite side of the arrow from Eγ, can be
    dragged, and can be hidden for the whole figure with Style ▸ Show B(σλ) labels.
  • Energy labels show each level energy exactly as you typed it (2186, 2186.3,
    1.642 …). Automatic γ energies use as many decimals as the more precise of
    the two levels.
  • Style tab: figure size, fonts, gaps, arrow widths, label placement.

Mouse (preview)
  • Click: select. Drag a level line up/down to separate crowded levels
    (only where it's drawn changes, not the energy).
  • Levels too close for their labels to fit are handled automatically: the upper
    level is nudged up just enough to leave a visible gap (levels above it are not
    moved), and the labels of the close levels are fanned out (lower ones down,
    upper ones up, groups evenly spaced). Each fanned label is joined to its level by
    a dashed connector: a horizontal stub at the level, a diagonal, and a horizontal
    stub into the label; the lifetime stays just above the energy. Energy labels keep
    the true energies. Tune it in Style ▸ Levels or switch it off; labels can still
    be dragged afterwards.
  • Drag an arrow sideways to move it; drag any label (including scheme titles)
    to nudge it.
  • Right-drag from one level to another in the same scheme to create a transition.
  • Double-click empty space to add a level to the scheme under the mouse.
  • 'Reset labels' puts the selected item's labels back.

Output
  • File ▸ Export figure: PDF / SVG / EPS / PNG at the size set in the Style tab.
  • File ▸ Export Python script: a standalone matplotlib script that reproduces
    the figure.
  • Save / Open store all schemes in one JSON file (older single-scheme files open
    as one scheme).

Keys: Delete, Ctrl+Z / Ctrl+Y, Ctrl+S, Ctrl+O, Ctrl+N
"""


# =============================================================================
# Application
# =============================================================================

class App:
    TAB_SCHEMES, TAB_LEVELS, TAB_TRANS, TAB_STYLE = range(4)

    def __init__(self, root, path=None):
        self.root = root
        self.doc = empty_doc()
        self.active = self.doc["schemes"][0]["id"]
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
        e.add_command(label="Add scheme", command=self.add_scheme)
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

    def _buttons(self, master, items):
        bf = ttk.Frame(master)
        bf.pack(fill="x", padx=4)
        for txt, cmd in items:
            ttk.Button(bf, text=txt, command=cmd,
                       width=3 if len(txt) == 1 else None).pack(side="left", padx=1, pady=3)

    def _build_layout(self):
        pw = ttk.PanedWindow(self.root, orient="horizontal")
        pw.pack(fill="both", expand=True)

        left = ttk.Frame(pw, width=380)
        top = ttk.Frame(left)
        top.pack(fill="x", padx=6, pady=(6, 0))
        ttk.Label(top, text="Active scheme:").pack(side="left")
        self.scheme_var = tk.StringVar()
        self.scheme_combo = ttk.Combobox(top, textvariable=self.scheme_var, state="readonly")
        self.scheme_combo.pack(side="left", fill="x", expand=True, padx=(4, 2))
        self.scheme_combo.bind("<<ComboboxSelected>>", self._on_scheme_combo)
        ttk.Button(top, text="+", width=3, command=self.add_scheme).pack(side="left")

        self.nb = ttk.Notebook(left)
        self.nb.pack(fill="both", expand=True, padx=4, pady=4)

        # Schemes tab
        sc = ttk.Frame(self.nb)
        self.nb.add(sc, text="Schemes")
        self.sc_tree = self._make_tree(sc, [("pos", "#", 30), ("title", "Title", 150),
                                            ("shift", "Shift", 60), ("n", "Levels", 55)], height=4)
        self.sc_tree.bind("<<TreeviewSelect>>", self._on_scheme_tree)
        self._buttons(sc, (("Add", self.add_scheme), ("Duplicate", self.duplicate_scheme),
                           ("Delete", self.delete_scheme), ("◀", lambda: self.move_scheme(-1)),
                           ("▶", lambda: self.move_scheme(1))))
        self.sc_form = PropertyForm(sc, SCHEME_FIELDS, self.on_item_change)
        self.sc_form.pack(fill="x", pady=4)
        ttk.Label(sc, foreground="#666666", wraplength=340, justify="left", text=(
            "Fill in A and the element symbol and the title is typeset for you; "
            "Z and N are worked out automatically. The shift moves the whole scheme "
            "on the shared energy scale (labels keep the energies you typed)."
        )).pack(fill="x", padx=8, pady=4)

        # Levels tab
        lt = ttk.Frame(self.nb)
        self.nb.add(lt, text="Levels")
        self.lv_tree = self._make_tree(lt, [("E", "Energy", 70), ("spin", "Spin", 80),
                                            ("life", "Lifetime", 90), ("col", "Col", 35)], height=6)
        self.lv_tree.bind("<<TreeviewSelect>>", lambda e: self._tree_select("level", self.lv_tree))
        self._buttons(lt, (("Add", self.add_level), ("Duplicate", self.duplicate_selected),
                           ("Delete", self.delete_selected), ("Reset labels", self.reset_labels)))
        self.lv_form = PropertyForm(lt, LEVEL_FIELDS, self.on_item_change)
        self.lv_form.pack(fill="x", pady=4)

        # Transitions tab
        tt = ttk.Frame(self.nb)
        self.nb.add(tt, text="Transitions")
        self.tr_tree = self._make_tree(tt, [("E", "Eγ", 60), ("from", "From", 80),
                                            ("to", "To", 80), ("I", "I", 50)])
        self.tr_tree.bind("<<TreeviewSelect>>", lambda e: self._tree_select("trans", self.tr_tree))
        self._buttons(tt, (("Add", self.add_transition), ("Duplicate", self.duplicate_selected),
                           ("Delete", self.delete_selected), ("Reset labels", self.reset_labels)))
        self.tr_form = PropertyForm(tt, TRANS_FIELDS, self.on_item_change)
        self.tr_form.pack(fill="x", pady=4)

        # Style tab
        sf = ScrollFrame(self.nb)
        self.nb.add(sf, text="Style")
        self.st_form = PropertyForm(sf.inner, STYLE_FIELDS, self.on_style_change)
        self.st_form.pack(fill="x", pady=4)
        self.nb.select(self.TAB_LEVELS)
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

    def _make_tree(self, master, cols, height=8):
        fr = ttk.Frame(master)
        fr.pack(fill="both", expand=True, padx=4, pady=(4, 0))
        tree = ttk.Treeview(fr, columns=[c[0] for c in cols], show="headings",
                            height=height, selectmode="browse")
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
    def schemes(self):
        return self.doc["schemes"]

    def scheme(self, sid):
        return next((s for s in self.schemes() if s["id"] == sid), None)

    def cur(self):
        s = self.scheme(self.active)
        if s is None:
            s = self.schemes()[0]
            self.active = s["id"]
        return s

    def all_levels(self):
        return [l for s in self.schemes() for l in s["levels"]]

    def all_trans(self):
        return [t for s in self.schemes() for t in s["transitions"]]

    def level(self, id_):
        return next((l for l in self.all_levels() if l["id"] == id_), None)

    def trans(self, id_):
        return next((t for t in self.all_trans() if t["id"] == id_), None)

    def owner(self, kind, id_):
        """The scheme containing a level / transition (or the scheme itself)."""
        if kind == "scheme":
            return self.scheme(id_)
        key = "levels" if kind == "level" else "transitions"
        return next((s for s in self.schemes() if any(o["id"] == id_ for o in s[key])), None)

    def selected_obj(self):
        if not self.selected:
            return None
        kind, id_ = self.selected
        if kind == "level":
            return self.level(id_)
        if kind == "trans":
            return self.trans(id_)
        if kind == "scheme":
            return self.scheme(id_)
        return None

    @staticmethod
    def next_id(items):
        return max([i["id"] for i in items] + [0]) + 1

    def level_name(self, l):
        return ("%s  %s" % (energy_text(l), l.get("spin", ""))).strip() + "  #%d" % l["id"]

    def scheme_name(self, s):
        i = self.schemes().index(s) + 1
        return "%d: %s" % (i, scheme_label(s) or "(untitled)")

    def layout(self):
        return scheme_layout(self.doc, self.style)

    # ------------------------------------------------------------- undo/redo
    def snapshot(self):
        return json.dumps({"doc": self.doc, "style": self.style, "active": self.active})

    def push_undo(self):
        self.undo_stack.append(self.snapshot())
        del self.undo_stack[:-200]
        self.redo_stack.clear()
        self.dirty = True
        self.update_title()

    def _restore(self, snap):
        d = json.loads(snap)
        self.doc, self.style, self.active = d["doc"], d["style"], d["active"]
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
            cur = self.cur()
            names = [self.scheme_name(s) for s in self.schemes()]
            self.scheme_combo.configure(values=names)
            self.scheme_var.set(self.scheme_name(cur))

            sc = self.sc_tree
            sc.delete(*sc.get_children())
            for i, s in enumerate(self.schemes()):
                sc.insert("", "end", iid=str(s["id"]), values=(
                    i + 1, scheme_label(s) or "(untitled)",
                    "{:g}".format(float(s.get("shift") or 0)), len(s["levels"])))
            sc.selection_set(str(cur["id"]))

            lv = self.lv_tree
            lv.delete(*lv.get_children())
            for l in sorted(cur["levels"],
                            key=lambda l: (l.get("column", 0), -float(l["energy"]))):
                lv.insert("", "end", iid=str(l["id"]), values=(
                    energy_text(l), l.get("spin", ""),
                    l.get("lifetime", ""), l.get("column", 0)))
            tr = self.tr_tree
            tr.delete(*tr.get_children())
            for t in cur["transitions"]:
                a, b = self.level(t["from"]), self.level(t["to"])
                if a and b:
                    e = t.get("label") or gamma_energy_text(a, b)
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
        cur = self.cur()
        pairs = [(l["id"], self.level_name(l))
                 for l in sorted(cur["levels"], key=lambda l: -float(l["energy"]))]
        self.tr_form.set_level_choices(pairs)
        kind = self.selected[0] if self.selected else None
        self.sc_form.load(cur)
        self.lv_form.load(self.selected_obj() if kind == "level" else None)
        self.tr_form.load(self.selected_obj() if kind == "trans" else None)
        self.st_form.load(self.style)

    def redraw(self, limits=None):
        matplotlib.rcParams.update(style_rc(self.style))
        self.hits = []
        try:
            self.ax = render(self.fig, self.doc, self.style, hits=self.hits,
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
        if self.selected:
            s = self.owner(*self.selected)
            if s is not None:
                self.active = s["id"]
        if from_canvas and self.selected:
            self.nb.select({"scheme": self.TAB_SCHEMES, "level": self.TAB_LEVELS,
                            "trans": self.TAB_TRANS}.get(self.selected[0], self.TAB_LEVELS))
        self.refresh_trees()
        self.load_forms()
        self.redraw()

    def set_active(self, sid):
        if sid == self.active:
            return
        self.active = sid
        if self.selected and (self.owner(*self.selected) or {}).get("id") != sid:
            self.selected = None
        self.refresh_all()

    def _on_scheme_combo(self, event):
        i = self.scheme_combo.current()
        if 0 <= i < len(self.schemes()):
            self.set_active(self.schemes()[i]["id"])

    def _on_scheme_tree(self, event):
        if self._syncing:
            return
        s = self.sc_tree.selection()
        if s:
            self.set_active(int(s[0]))

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
        if key == "energy":                  # keep the energy exactly as typed
            val, raw = val
            if target.get("energy") == val and energy_text(target) == raw:
                return
            self.push_undo()
            target["energy"], target["energy_text"] = val, raw
            self.refresh_trees()
            self.load_forms()
            self.redraw()
            return
        if key == "symbol":
            val = val.strip().capitalize()
            if target.get(key) == val:
                self.load_forms()
                return
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
        self.redraw()

    # schemes
    def add_scheme(self):
        self.push_undo()
        s = new_scheme(self.next_id(self.schemes()))
        self.schemes().append(s)
        self.active = s["id"]
        self.selected = None
        self.nb.select(self.TAB_SCHEMES)
        self.refresh_all()
        self.status("Added scheme %d. Give it a title, then add levels to it." % len(self.schemes()))

    def duplicate_scheme(self):
        src = self.cur()
        self.push_undo()
        s = copy.deepcopy(src)
        s["id"] = self.next_id(self.schemes())
        nl, nt = self.next_id(self.all_levels()), self.next_id(self.all_trans())
        remap = {}
        for i, l in enumerate(s["levels"]):
            remap[l["id"]] = nl + i
            l["id"] = nl + i
        for i, t in enumerate(s["transitions"]):
            t["id"] = nt + i
            t["from"], t["to"] = remap.get(t["from"], -1), remap.get(t["to"], -1)
        self.schemes().insert(self.schemes().index(src) + 1, s)
        self.active = s["id"]
        self.selected = None
        self.refresh_all()

    def delete_scheme(self, sid=None):
        s = self.scheme(sid) if sid is not None else self.cur()
        if len(self.schemes()) == 1:
            messagebox.showinfo(APP_NAME, "A figure needs at least one scheme.")
            return
        if s["levels"] and not messagebox.askyesno(
                APP_NAME, "Delete scheme '%s' and its %d level(s)?"
                % (scheme_label(s) or "(untitled)", len(s["levels"]))):
            return
        self.push_undo()
        i = self.schemes().index(s)
        self.schemes().remove(s)
        self.active = self.schemes()[max(0, i - 1)]["id"]
        self.selected = None
        self.refresh_all()
        self.status("Deleted scheme.")

    def move_scheme(self, step):
        s = self.cur()
        L = self.schemes()
        i = L.index(s)
        j = i + step
        if not 0 <= j < len(L):
            return
        self.push_undo()
        L[i], L[j] = L[j], L[i]
        self.refresh_all()

    # levels / transitions
    def add_level(self, energy=None, column=None, scheme_id=None):
        if scheme_id is not None:
            self.active = scheme_id
        sch = self.cur()
        lv = sch["levels"]
        sel = self.selected_obj() if self.selected and self.selected[0] == "level" else None
        if sel is not None and sel not in lv:
            sel = None
        if column is None:
            column = sel.get("column", 0) if sel else 0
        if energy is None:
            same = [float(l["energy"]) for l in lv if l.get("column", 0) == column]
            energy = (max(same) + 500.0) if same else 0.0
        self.push_undo()
        l = new_level(self.next_id(self.all_levels()), energy, column)
        lv.append(l)
        self.select(("level", l["id"]), from_canvas=True)
        self.status("Added level at %g keV to %s. Set its spin and lifetime in the form."
                    % (energy, self.scheme_name(sch)))

    def _auto_x(self, sch, a, b):
        col = a.get("column", 0)
        lo, hi = sorted((level_y(a), level_y(b)))
        used = []
        for t in sch["transitions"]:
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
        if from_id is not None:
            self.active = self.owner("level", from_id)["id"]
        sch = self.cur()
        lv = sch["levels"]
        if len(lv) < 2:
            messagebox.showinfo(APP_NAME, "Add at least two levels to this scheme first.")
            return
        if from_id is None:
            cur = self.selected_obj() if self.selected and self.selected[0] == "level" else None
            if cur is not None and cur not in lv:
                cur = None
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
        t = new_transition(self.next_id(self.all_trans()), from_id, to_id,
                           self._auto_x(sch, a, b) if x is None else x)
        sch["transitions"].append(t)
        self.select(("trans", t["id"]), from_canvas=True)
        self.status("Added %s transition. Edit it in the Transitions tab."
                    % gamma_energy_text(a, b))

    def duplicate_selected(self):
        obj = self.selected_obj()
        if not obj:
            return
        kind = self.selected[0]
        if kind == "scheme":
            return self.duplicate_scheme()
        sch = self.owner(kind, obj["id"])
        self.push_undo()
        new = copy.deepcopy(obj)
        if kind == "level":
            new["id"] = self.next_id(self.all_levels())
            new["energy"] = float(new["energy"]) + 100.0
            new["energy_text"] = ""
            sch["levels"].append(new)
        else:
            new["id"] = self.next_id(self.all_trans())
            new["x"] = min(float(new.get("x", 0.5)) + 0.12, 1.0)
            sch["transitions"].append(new)
        self.select((kind, new["id"]))

    def delete_selected(self):
        obj = self.selected_obj()
        if not obj:
            return
        kind = self.selected[0]
        if kind == "scheme":
            return self.delete_scheme(obj["id"])
        sch = self.owner(kind, obj["id"])
        self.push_undo()
        if kind == "level":
            sch["levels"].remove(obj)
            before = len(sch["transitions"])
            sch["transitions"] = [t for t in sch["transitions"]
                                  if obj["id"] not in (t["from"], t["to"])]
            n = before - len(sch["transitions"])
            self.status("Deleted level%s." % (" and %d transition(s)" % n if n else ""))
        else:
            sch["transitions"].remove(obj)
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
        for o in self.all_levels() + self.all_trans() + self.schemes():
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
        if kind == "scheme":
            return self.scheme(id_)
        return self.style

    def scheme_at(self, x):
        """(scheme, x_left) whose block (incl. half the gaps around it) contains x."""
        lay = self.layout()
        half = float(self.style["scheme_gap"]) / 2
        for s, base, w in lay:
            if x < base + w + half:
                return s, base
        return lay[-1][0], lay[-1][1]

    def on_press(self, ev):
        if self.ax is None:
            return
        hit = self.hit_test(ev.x, ev.y)
        if ev.button == 1:
            if ev.dblclick:
                if hit is None:
                    x, y = self.ax.transData.inverted().transform((ev.x, ev.y))
                    s, base = self.scheme_at(x)
                    step = float(self.style["column_width"]) + float(self.style["column_gap"])
                    col = int(min(max(0, (x - base) // step), scheme_ncol(s) - 1))
                    e = float(round(y - float(s.get("shift") or 0)))
                    self.add_level(energy=e, column=col, scheme_id=s["id"])
                return
            if hit is None:
                if self.selected:
                    self.select(None)
                return
            kind, id_, part, _ = hit
            if kind in ("level", "trans", "scheme") and self.selected != (kind, id_):
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
            ya = level_y(a, self.owner("level", hit[1]).get("shift"))
            self.link = {"from": hit[1], "x": x, "y": ya}
            self.link["line"], = self.ax.plot([x, x], [ya, y], color=HIGHLIGHT,
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
                sa, sb = self.owner("level", link["from"]), self.owner("level", hit[1])
                if sa is not sb:
                    self.status("Transitions must connect levels within the same scheme.")
                    self.redraw()
                    return
                a = self.level(link["from"])
                base = next(b for s, b, _ in self.layout() if s is sa)
                x0 = base + column_x0(a.get("column", 0), self.style)
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

    def _set_doc(self, doc, style):
        self.doc, self.style = doc, style
        self.active = doc["schemes"][0]["id"]
        self.selected = None

    def new_file(self):
        if not self._confirm_discard():
            return
        self._set_doc(empty_doc(), merged_style({}))
        self.path, self.dirty = None, False
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._layout_canvas()
        self.refresh_all()
        self.update_title()

    def load_example(self):
        self.push_undo()
        self.doc = example_doc()
        self.active, self.selected = 1, None
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
            doc = normalise_doc(d)
            style = merged_style(d.get("style"))
        except Exception as exc:
            messagebox.showerror(APP_NAME, "Could not open %s:\n%s" % (path, exc))
            return
        self._set_doc(doc, style)
        self.path, self.dirty = path, False
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._layout_canvas()
        self.refresh_all()
        self.update_title()
        self.status("Opened %s (%d scheme%s)" % (path, len(doc["schemes"]),
                                                 "" if len(doc["schemes"]) == 1 else "s"))

    def save(self):
        if not self.path:
            return self.save_as()
        try:
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump({"format": "levelschemer", "version": FILE_VERSION,
                           "schemes": self.doc["schemes"], "style": self.style},
                          fh, indent=1, ensure_ascii=False)
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
            save_figure(self.doc, self.style, p)
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
                fh.write(build_script(self.doc, self.style, self._stem() + ".pdf"))
        except Exception as exc:
            messagebox.showerror(APP_NAME, "Could not write script:\n%s" % exc)
            return
        self.status("Wrote %s" % p)

    def show_help(self):
        w = tk.Toplevel(self.root)
        w.title("Quick guide")
        txt = tk.Text(w, width=86, height=34, wrap="word", padx=10, pady=10)
        txt.insert("1.0", HELP_TEXT)
        txt.configure(state="disabled")
        txt.pack(fill="both", expand=True)

    def quit(self):
        if self._confirm_discard():
            self.root.destroy()


def main():
    root = tk.Tk()
    root.geometry("1150x780")
    App(root, sys.argv[1] if len(sys.argv) > 1 else None)
    root.mainloop()


if __name__ == "__main__":
    main()
