"""
Fit the 2nd-order frequency correction from the "New2ndOrderCorrection" sheet.

The sheet gives the identified ions with their harmonic and measured frequency
(Ion, Harmonic, f_exp).  The simulated frequencies are calculated here, with the
suite's own masses (core.nubase_reader: NUBASE mass excess, electron masses and
NIST electron binding energies) and no correction applied, so the fit always
matches the masses the tools use.  Any f_sim / f_sim_true / f_diff columns in
the sheet are ignored (f_sim is only compared in the printout).

Simulation (revolution frequencies, as core.physics.simulate_frequencies):
    f_ref     = f_exp(ref) / h(ref)                  reference ion from the sheet
    f_sim     = f_ref + delta_f(f_ref, mq_ref, mq - mq_ref, gamma_t)
    f_exp_rev = f_exp / h,   f_diff = f_exp_rev - f_sim
gamma_t from the shared physics settings (data/<experiment>/settings/physics.json).

Fit model (shifted form, passes exactly through (x0, 0), x = f_sim):
    f_diff = s1*(x - x0) + s2*(x - x0)^2
Equivalent expanded form (the a, b, c of core.physics.apply_correction):
    f_diff = a*x^2 + b*x + c
    a = s2,  b = s1 - 2*s2*x0,  c = s2*x0^2 - s1*x0

Usage:
    python new_2nd_order_correction_plot.py [spreadsheet] [sheet name] [--weighted]
    python new_2nd_order_correction_plot.py --points points.json   (from the GUI tool)

--weighted: weight the points by 1/sigma^2, sigma = uncertainty of the simulated
frequency from the m/q uncertainties (NUBASE (+) NIST) of the ion and the
reference.  The measured frequencies have no uncertainty in the sheet.

Saving: the plot window has a name box and a "Save as profile" button.  Each
saved fit becomes a correction profile (core/corrections.py, stored in
data/<experiment>/settings/correction_profiles.json) that the Viewer, Scan,
PID settings, spectrogram and home page offer in a dropdown.  With "also make
it the default" ticked, a, b, c are also written to the shared physics
settings (physics.json).

The default spreadsheet is data/<experiment>/rionid/186hf_rionid-v8_PID.xlsx.
"""
import argparse
import datetime
import json
import os
import re
import sys
from functools import lru_cache

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from matplotlib.widgets import Button, CheckButtons, TextBox

from core import corrections, logcapture, logpanel, paths, physics

# output also to data/logs/ and this tool's Output window (core/logcapture.py)
if __name__ == "__main__":
    logcapture.start("correction_plot")
from core.nubase_reader import MASS_MODEL, df_nubase

plt.rcParams["font.size"] = 2 * plt.rcParams["font.size"]  # all fonts doubled

DEFAULT_FILE = "186hf_rionid-v8_PID.xlsx"
DEFAULT_SHEET = "New2ndOrderCorrection"
COLUMNS = ["Ion", "Harmonic", "f_exp"]
ION_RE = re.compile(r"\s*([a-z]?)(\d+)([A-Z][a-z]?)(\d+)\+\s*")


# ---------------------------------------------------------------------------
# Spreadsheet
# ---------------------------------------------------------------------------

def read_sheet(filename, sheet):
    """Return (table, ref_ion, ref_harmonic_cell).

    table: Ion, Harmonic, f_exp (+ f_sim_sheet if the sheet has an f_sim column),
    from the header row containing 'Ion' and 'f_exp'.  ref_ion: the value next to
    the 'Reference' cell above the table; ref_harmonic_cell: next to 'Harmonic'
    (None if absent)."""
    raw = pd.read_excel(filename, sheet_name=sheet, header=None)

    def value_after(label, stop):
        for r in range(stop):
            cells = [c for c in raw.iloc[r].tolist() if not (isinstance(c, float) and np.isnan(c))]
            if len(cells) >= 2 and str(cells[0]).strip() == label:
                return cells[1]
        return None

    for r in range(len(raw)):
        row = raw.iloc[r].astype(str).str.strip().tolist()
        if "Ion" in row and "f_exp" in row:
            df = raw.iloc[r + 1:].copy()
            df.columns = [str(c).strip() for c in raw.iloc[r]]
            cols = COLUMNS + (["f_sim"] if "f_sim" in df.columns else [])
            df = df[cols].dropna(subset=["Ion"])
            df["Ion"] = df["Ion"].astype(str).str.strip()
            df["Harmonic"] = pd.to_numeric(df["Harmonic"]).astype(int)
            df["f_exp"] = pd.to_numeric(df["f_exp"])
            if "f_sim" in df.columns:
                # formulas (e.g. '=E6') are not evaluated by pandas -> NaN
                df["f_sim_sheet"] = pd.to_numeric(df.pop("f_sim"), errors="coerce")
            ref_ion = value_after("Reference", r)
            if ref_ion is None:
                raise ValueError(f"No 'Reference' cell above the table in sheet '{sheet}'")
            ref_h = value_after("Harmonic", r)
            return df.reset_index(drop=True), str(ref_ion).strip(), ref_h
    raise ValueError(f"No header row with 'Ion' and 'f_exp' found in sheet '{sheet}'")


# ---------------------------------------------------------------------------
# Masses and simulated frequencies
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _charge_table(electrons):
    return df_nubase(electrons=electrons)


def ion_mq(name):
    """m/q (u/e), its uncertainty and the number of bound electrons for an ion
    name like '208Pb80+' or 'm186Hf72+' (suite masses incl. electron binding)."""
    m = ION_RE.fullmatch(name)
    if not m:
        raise ValueError(f"'{name}' is not an ion name like 208Pb80+ or m186Hf72+")
    iso, A, sym, q = m.group(1), m.group(2), m.group(3), int(m.group(4))
    base = _charge_table(0)
    rows = base[base["Element"].astype(str).str.strip() == f"{A}{sym}"]
    rows = rows[rows["Isomer_Flag"] == iso] if iso else rows[rows["Isomer_Flag"].isna()]
    if not len(rows):
        raise ValueError(f"'{iso}{A}{sym}' is not in NUBASE")
    Z = int(rows["ZZZi"].iloc[0])
    if not 1 <= q <= Z:
        raise ValueError(f"charge {q}+ is not possible for {sym} (Z = {Z})")
    d = _charge_table(Z - q)
    r = d.loc[rows.index[0]]
    mq, dmq = float(r["m/q"]), float(r["m/q_unc"])
    if not np.isfinite(mq):
        raise ValueError(f"no mass for {name} (NUBASE mass excess or NIST energies missing)")
    return mq, dmq, Z - q


def simulate(df, ref_ion, gamma_t):
    """Add electrons, mq, f_exp_true, f_sim_true, f_sim_unc, f_sim, f_diff
    (all *_true / f_diff / f_sim_unc are revolution frequencies in Hz)."""
    ref = df[df["Ion"] == ref_ion]
    if ref.empty:
        raise ValueError(f"Reference ion {ref_ion} is not in the table")
    ref_f, ref_h = float(ref["f_exp"].iloc[0]), int(ref["Harmonic"].iloc[0])
    mq_ref, dmq_ref, _ = ion_mq(ref_ion)
    f_ref = ref_f / ref_h

    out = df.copy()
    vals = [ion_mq(n) for n in out["Ion"]]
    out["electrons"] = [v[2] for v in vals]
    out["mq"] = [v[0] for v in vals]
    out["mq_unc"] = [v[1] for v in vals]
    out["f_exp_true"] = out["f_exp"] / out["Harmonic"]
    out["f_sim_true"] = f_ref + physics.delta_f(f_ref, mq_ref, out["mq"] - mq_ref, gamma_t)
    # d f / d mq = -f_ref / (gamma_t^2 mq_ref); ion and reference uncertainties
    # in quadrature (the reference itself: 0)
    is_ref = out["Ion"] == ref_ion
    dmq = np.where(is_ref, 0.0, np.hypot(out["mq_unc"], dmq_ref))
    out["f_sim_unc"] = f_ref / (gamma_t ** 2 * mq_ref) * dmq
    out["f_sim"] = out["f_sim_true"] * out["Harmonic"]
    out["f_diff"] = out["f_exp_true"] - out["f_sim_true"]
    return out, ref_f, ref_h, f_ref


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

# anchored_quadratic_fit is shared with the GUI correction tool (core/corrections.py)
anchored_quadratic_fit = corrections.anchored_quadratic_fit


def pretty_ion(name):
    """'208Pb80+' -> mathtext ^{208}Pb^{80+}"""
    m = re.fullmatch(r"([a-z]?)(\d+)([A-Za-z]+)(\d+[+-])", name)
    if not m:
        return name
    iso, A, el, q = m.groups()
    return rf"$^{{{A}{iso}}}\mathrm{{{el}}}^{{{q}}}$"


def default_profile_name(weighted, n_points):
    mode = "weighted" if weighted else "unweighted"
    return f"{datetime.datetime.now():%Y-%m-%d %H:%M} {mode}, e-binding, {n_points} ions"


def save_profile(name, make_default, coeffs, uncs, fit_info, shifted=None):
    """Save a correction profile; with make_default also write a, b, c to
    physics.json.  shifted: corrections.shifted_block (s1, s2, x0 and their
    covariance).  Returns a status message (raises ValueError on a bad name)."""
    (c2, c1, c0), (dc2, dc1, dc0) = coeffs, uncs
    prof = {"name": name.strip(), "a": float(c2), "b": float(c1), "c": float(c0),
            "a_unc": float(dc2), "b_unc": float(dc1), "c_unc": float(dc0),
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "fit": fit_info, "notes": ""}
    if shifted:
        prof["shifted"] = shifted
    prof = corrections.save_profile(prof)
    msg = f"Saved profile '{prof['name']}'"
    print(f"\n{msg} in {corrections.profiles_path()}")
    if make_default:
        physics.save_shared({"corr_profile": prof["name"],
                             "corr_a": float(c2), "corr_b": float(c1), "corr_c": float(c0)})
        msg += " and made it the default (physics.json)"
        print(f"a, b, c written to {physics.shared_path()}")
    return msg + ". Restart open tools / reload Theia to see it."


def draw_figure(names, x, y, fit_mask, x0, ref_ion, ref_f, s1, s2, abc, dabc, rms, chi2ndf,
                weighted=False, sig=None, mode="unweighted", controls=True, title=None):
    """The correction figure (f_diff vs f_sim with labels, fit, coefficient box,
    residuals).  Returns (fig, gridspec); with controls the gridspec has two
    extra rows for the profile controls."""
    names = pd.Series(list(names)).reset_index(drop=True)
    x, y = np.asarray(x, float), np.asarray(y, float)
    fit_mask = np.asarray(fit_mask, bool)
    sig = np.full(len(x), np.nan) if sig is None else np.asarray(sig, float)
    (c2, c1, c0), (dc2, dc1, dc0) = abc, dabc
    resid = y - (s1 * (x - x0) + s2 * (x - x0) ** 2)
    # ---- Plot ----
    # constrained layout re-fits margins to the window size, so labels never get cut off
    fig = plt.figure(figsize=(22, 16), layout="constrained")
    fig.get_layout_engine().set(hspace=0.0, h_pad=0.05)
    gs = (fig.add_gridspec(4, 1, height_ratios=[3, 1, 0.16, 0.1]) if controls
          else fig.add_gridspec(2, 1, height_ratios=[3, 1]))
    ax = fig.add_subplot(gs[0])
    axr = fig.add_subplot(gs[1], sharex=ax)
    ax.tick_params(labelbottom=False)
    yerr = sig[fit_mask] if weighted else None
    ax.errorbar(x[fit_mask], y[fit_mask], yerr=yerr, fmt="o", color="C0", zorder=3,
                capsize=4 if weighted else 0, label="Data")
    ax.scatter([x0], [0], color="C3", marker="*", s=360, zorder=4,
               label=f"Anchor ({pretty_ion(ref_ion)}, $f_{{exp}}$ = {ref_f!r} Hz)")

    # Group nearby points into clusters and stack their labels to avoid overlap
    xr, yr = np.ptp(x) or 1.0, np.ptp(y) or 1.0
    order = np.argsort(x)
    clusters, cur = [], [order[0]]
    for i in order[1:]:
        j = cur[-1]
        if abs(x[i] - x[j]) < 0.02 * xr and abs(y[i] - y[j]) < 0.05 * yr:
            cur.append(i)
        else:
            clusters.append(cur)
            cur = [i]
    clusters.append(cur)
    # Split each cluster: upper half of labels go above-left of the fit line,
    # the rest below-right. Every label gets a leader line to its point.
    line_kw = dict(arrowstyle="-", color="black", lw=1.2, shrinkA=2, shrinkB=3)
    for cl in clusters:
        cl = sorted(cl, key=lambda k: y[k])  # ascending f_diff
        n_above = (len(cl) + 1) // 2 if len(cl) > 1 else 0
        above, below = cl[len(cl) - n_above:], cl[:len(cl) - n_above][::-1]
        # Above-line labels: rightmost point gets the rightmost (and highest) label,
        # so leader lines fan out without crossing
        above = sorted(above, key=lambda k: x[k], reverse=True)
        for i, k in enumerate(above):
            ax.annotate(pretty_ion(names[k]), (x[k], y[k]), textcoords="offset points",
                        xytext=(-20 - 90 * i, 45 + 20 * (len(above) - 1 - i)), fontsize=18,
                        ha="center", va="center", arrowprops=dict(**line_kw, relpos=(0.5, 0)))
        for i, k in enumerate(below):   # highest first, stacked downwards
            ax.annotate(pretty_ion(names[k]), (x[k], y[k]), textcoords="offset points",
                        xytext=(40 + 20 * i, -30 - 34 * i), fontsize=18, ha="left", va="center",
                        arrowprops=dict(**line_kw, relpos=(0, 0.5)))

    pad = 0.05 * (x.max() - x.min())
    xx = np.linspace(x.min() - pad, x.max() + pad, 500)
    ax.plot(xx, s1 * (xx - x0) + s2 * (xx - x0) ** 2, color="C1",
            label=f"Anchored 2nd-order fit ({mode})")
    ax.axhline(0, color="grey", lw=0.8, ls="--")

    txt = (
        r"Expanded: $f_{diff} = a x^2 + b x + c$" "\n"
        rf"  $a$ = {c2:.6e} ± {dc2:.2e} Hz$^{{-1}}$" "\n"
        rf"  $b$ = {c1:.6e} ± {dc1:.2e}" "\n"
        rf"  $c$ = {c0:.6e} ± {dc0:.2e} Hz" "\n\n"
        rf"RMS residual = {rms:.3f} Hz"
        + (f"\nχ²/ndf = {chi2ndf:.3g}" if weighted else "")
    )
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left", fontsize=18,
            family="monospace", bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.9))

    ax.set_ylabel(r"$f_{diff} = f_{exp,true} - f_{sim,true}$ (Hz)")
    ax.set_title(title or "Second-order Correction (masses incl. electron binding)")
    ax.ticklabel_format(axis="x", useOffset=False, style="plain")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    # ---- Residuals panel ----
    axr.errorbar(x[fit_mask], resid[fit_mask], yerr=yerr, fmt="o", color="C0", zorder=3,
                 capsize=4 if weighted else 0)
    axr.scatter([x0], [resid[~fit_mask][0]], color="C3", marker="*", s=360, zorder=4)
    axr.axhline(0, color="C1", lw=1.5)
    axr.set_xlabel(r"$f_{sim,true}$ (Hz)")
    axr.set_ylabel("Residual (Hz)")
    axr.ticklabel_format(axis="x", useOffset=False, style="plain")
    axr.grid(alpha=0.3)
    fig.align_ylabels([ax, axr])

    return fig, gs


def plot_points(path):
    """--points FILE: the same figure for points given in a JSON file (from the
    GUI correction tool: one harmonic's assigned peaks), without the profile
    controls.  JSON: {"points": [{"ion", "x", "y"}] (x = f_sim, y = f_diff,
    revolution Hz), "x0", "ref_ion", "ref_f", "h", "file"}."""
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    pts = d["points"]
    names = [d["ref_ion"]] + [p["ion"] for p in pts]
    x = np.array([d["x0"]] + [p["x"] for p in pts], float)
    y = np.array([0.0] + [p["y"] for p in pts], float)
    fit_mask = np.array([False] + [True] * len(pts))
    res = corrections.fit_correction(x[fit_mask], y[fit_mask], float(d["x0"]))
    print(f"{len(pts)} points + anchor {d['ref_ion']} (h{d['h']}), "
          f"a = {res['abc'][0]:.16e}, b = {res['abc'][1]:.16e}, c = {res['abc'][2]:.16e}, "
          f"RMS = {res['rms']:.3f} Hz (rev)")
    title = (f"Second-order Correction h{d['h']} – {os.path.basename(d.get('file') or '')}"
             "\n(masses incl. electron binding, GUI assignments)")
    draw_figure(names, x, y, fit_mask, float(d["x0"]), d["ref_ion"], d["ref_f"], res["s1"],
                res["s2"], res["abc"], res["abc_unc"], res["rms"], res["chi2_ndf"],
                controls=False, title=title)
    try:
        os.remove(path)                    # a temporary file written by the GUI
    except OSError:
        pass
    logpanel.add_mpl_buttons([0.925, 0.955, 0.07, 0.04], "Correction plot")
    plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("spreadsheet", nargs="?", default=os.path.join(paths.rionid(), DEFAULT_FILE))
    ap.add_argument("sheet", nargs="?", default=DEFAULT_SHEET)
    ap.add_argument("--weighted", action="store_true",
                    help="weight points by 1/sigma^2 of the simulated frequency")
    ap.add_argument("--points", metavar="JSON",
                    help="plot points from a JSON file (GUI correction tool) instead of a sheet")
    args = ap.parse_args()
    if args.points:
        return plot_points(args.points)

    df, ref_ion, ref_h_cell = read_sheet(args.spreadsheet, args.sheet)
    gamma_t = float(physics.load_shared()["gamma_t"])
    df, ref_f, ref_h, f_ref = simulate(df, ref_ion, gamma_t)
    if ref_h_cell is not None and int(ref_h_cell) != ref_h:
        print(f"Warning: sheet 'Harmonic' cell is {ref_h_cell}, but {ref_ion} is listed at "
              f"h{ref_h}; using h{ref_h}")

    print(f"Spreadsheet: {args.spreadsheet}  [{args.sheet}]")
    print(f"Reference {ref_ion}: f_exp = {ref_f!r} Hz at h{ref_h} -> f_rev = {f_ref:.6f} Hz; "
          f"gamma_t = {gamma_t} (physics.json)")
    print("Masses: NUBASE + electron masses + NIST electron binding; no correction applied.\n")
    show = df[["Ion", "electrons", "Harmonic", "f_exp", "f_sim", "f_sim_true", "f_sim_unc", "f_diff"]].copy()
    if "f_sim_sheet" in df:
        show["sheet f_sim - f_sim"] = df["f_sim_sheet"] - df["f_sim"]
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}", "display.width", 200):
        print(show.to_string(index=False))

    x = df["f_sim_true"].to_numpy(float)
    y = df["f_diff"].to_numpy(float)
    sig = df["f_sim_unc"].to_numpy(float)
    x0 = float(x[df["Ion"] == ref_ion][0])
    fit_mask = (df["Ion"] != ref_ion).to_numpy()   # anchor point is enforced, not fitted

    if args.weighted:
        bad = fit_mask & ~(np.isfinite(sig) & (sig > 0))
        if bad.any():
            raise ValueError("no simulated-frequency uncertainty for: "
                             + ", ".join(df["Ion"][bad]))
    (s1, s2), cov, _, chi2ndf = anchored_quadratic_fit(
        x[fit_mask], y[fit_mask], x0, sig[fit_mask] if args.weighted else None)
    ds1, ds2 = np.sqrt(np.diag(cov))

    # Expanded coefficients and their propagated uncertainties
    c2, c1, c0 = s2, s1 - 2 * s2 * x0, s2 * x0 ** 2 - s1 * x0
    J = np.array([[0.0, 1.0],            # dc2/d(s1,s2)
                  [1.0, -2 * x0],        # dc1/d(s1,s2)
                  [-x0, x0 ** 2]])       # dc0/d(s1,s2)
    cov_c = J @ cov @ J.T
    dc2, dc1, dc0 = np.sqrt(np.diag(cov_c))

    resid = y - (s1 * (x - x0) + s2 * (x - x0) ** 2)
    rms = np.sqrt(np.mean(resid[fit_mask] ** 2))
    mode = "weighted by 1/sigma_sim^2" if args.weighted else "unweighted"

    print(f"\nFit ({mode}), {fit_mask.sum()} points + anchor {ref_ion}:")
    print(f"Shifted (s1 (x-x0) + s2 (x-x0)^2): s1 = {s1:.6e} ± {ds1:.2e},  "
          f"s2 = {s2:.6e} ± {ds2:.2e},  x0 = {x0:.6f} Hz")
    # Full 17 significant figures: the three terms are ~1e6 Hz and cancel near x0,
    # so truncated coefficients give errors of order 1 Hz
    print("Expanded (a x^2 + b x + c), full precision:")
    print(f"  a = {c2:.16e} ± {dc2:.2e}")
    print(f"  b = {c1:.16e} ± {dc1:.2e}")
    print(f"  c = {c0:.16e} ± {dc0:.2e}")
    print(f"RMS residual = {rms:.3f} Hz (revolution frequency), "
          f"≈ {rms * ref_h:.1f} Hz at h{ref_h}")
    if args.weighted:
        print(f"chi2/ndf = {chi2ndf:.3g} (simulation uncertainties only)")

    fig, gs = draw_figure(df["Ion"], x, y, fit_mask, x0, ref_ion, ref_f, s1, s2, (c2, c1, c0),
                          (dc2, dc1, dc0), rms, chi2ndf, weighted=args.weighted, sig=sig,
                          mode=mode)

    # ---- Save as correction profile ----
    fit_info = {
        "ref_ion": ref_ion, "ref_f_exp": ref_f, "ref_harmonic": ref_h, "f_ref_rev": f_ref,
        "gamma_t": gamma_t, "mass_model": MASS_MODEL, "weighted": bool(args.weighted),
        "n_points": int(fit_mask.sum()), "rms_rev_hz": float(rms),
        "chi2_ndf": float(chi2ndf) if args.weighted else None,
        "ions": [f"{ion} h{h}" for ion, h in zip(df["Ion"][fit_mask], df["Harmonic"][fit_mask])],
        "source_file": os.path.abspath(args.spreadsheet), "sheet": args.sheet,
    }
    # controls on one row, the save status on its own full-width row below
    ctrl = gs[2].subgridspec(1, 4, width_ratios=[6, 3.2, 1.6, 5], wspace=0.04)
    ax_name, ax_chk, ax_btn = (fig.add_subplot(ctrl[0, i]) for i in range(3))
    ax_msg = fig.add_subplot(gs[3])
    name_box = TextBox(ax_name, "Profile name ",
                       initial=default_profile_name(args.weighted, int(fit_mask.sum())))
    name_box.label.set_fontsize(16)
    name_box.text_disp.set_fontsize(15)
    chk = CheckButtons(ax_chk, ["also make it the default"], [False])
    for t in chk.labels:
        t.set_fontsize(15)
    btn = Button(ax_btn, "Save as profile")
    btn.label.set_fontsize(16)
    ax_msg.set_axis_off()
    msg = ax_msg.text(0.0, 0.5, "", va="center", ha="left", fontsize=14,
                      transform=ax_msg.transAxes)

    def on_save(_event):
        try:
            text = save_profile(name_box.text, chk.get_status()[0], (c2, c1, c0),
                                (dc2, dc1, dc0), fit_info,
                                corrections.shifted_block(s1, s2, x0, cov))
            msg.set_text("✓ " + text)
            msg.set_color("darkgreen")
        except (ValueError, OSError) as exc:
            msg.set_text(f"✗ Not saved: {exc}")
            msg.set_color("firebrick")
        fig.canvas.draw_idle()

    btn.on_clicked(on_save)
    logpanel.add_mpl_buttons([0.925, 0.955, 0.07, 0.04], "Correction plot")
    plt.show()


if __name__ == "__main__":
    main()
