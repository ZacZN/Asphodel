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

--weighted: weight the points by 1/sigma^2, sigma = uncertainty of the simulated
frequency from the m/q uncertainties (NUBASE (+) NIST) of the ion and the
reference.  The measured frequencies have no uncertainty in the sheet.

After the plot window is closed the script offers to write a, b, c to the
shared physics settings.

The default spreadsheet is data/<experiment>/rionid/186hf_rionid-v8_PID.xlsx.
"""
import argparse
import os
import re
import sys
from functools import lru_cache

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import paths, physics
from core.nubase_reader import df_nubase

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

def anchored_quadratic_fit(x, y, x0, sigma=None):
    """Least squares for y = s1*dx + s2*dx^2 (no intercept), dx = x - x0.
    With sigma: weighted by 1/sigma^2.  The covariance is scaled by the
    residual variance (chi^2/ndf) in both cases."""
    dx = x - x0
    X = np.column_stack([dx, dx ** 2])
    w = np.ones_like(y) if sigma is None else 1.0 / np.asarray(sigma, float)
    Xw, yw = X * w[:, None], y * w
    coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid = y - X @ coef
    dof = len(y) - X.shape[1]
    s2 = np.sum((resid * w) ** 2) / dof if dof > 0 else np.nan
    cov = s2 * np.linalg.inv(Xw.T @ Xw)
    return coef, cov, resid, s2


def pretty_ion(name):
    """'208Pb80+' -> mathtext ^{208}Pb^{80+}"""
    m = re.fullmatch(r"([a-z]?)(\d+)([A-Za-z]+)(\d+[+-])", name)
    if not m:
        return name
    iso, A, el, q = m.groups()
    return rf"$^{{{A}{iso}}}\mathrm{{{el}}}^{{{q}}}$"


def offer_save(c2, c1, c0):
    """Ask in the terminal whether to write a, b, c to the shared physics settings."""
    path = physics.shared_path()
    old = physics.load_shared()
    print(f"\nShared physics settings: {path}")
    print(f"  current: a = {old['corr_a']!r}, b = {old['corr_b']!r}, c = {old['corr_c']!r}")
    print(f"  new:     a = {float(c2)!r}, b = {float(c1)!r}, c = {float(c0)!r}")
    try:
        answer = input("Write the new a, b, c to physics.json? [y/N] ").strip().lower()
    except EOFError:
        answer = ""
    if answer in ("y", "yes"):
        physics.save_shared({"corr_a": float(c2), "corr_b": float(c1), "corr_c": float(c0)})
        print("Saved. Restart open tools (Theia, spectrograms) to use the new correction.")
    else:
        print("Not saved.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("spreadsheet", nargs="?", default=os.path.join(paths.rionid(), DEFAULT_FILE))
    ap.add_argument("sheet", nargs="?", default=DEFAULT_SHEET)
    ap.add_argument("--weighted", action="store_true",
                    help="weight points by 1/sigma^2 of the simulated frequency")
    args = ap.parse_args()

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

    # ---- Plot ----
    # constrained layout re-fits margins to the window size, so labels never get cut off
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(22, 15), sharex=True, layout="constrained",
                                  gridspec_kw=dict(height_ratios=[3, 1]))
    fig.get_layout_engine().set(hspace=0.0, h_pad=0.05)
    yerr = sig[fit_mask] if args.weighted else None
    ax.errorbar(x[fit_mask], y[fit_mask], yerr=yerr, fmt="o", color="C0", zorder=3,
                capsize=4 if args.weighted else 0, label="Data")
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
            ax.annotate(pretty_ion(df["Ion"][k]), (x[k], y[k]), textcoords="offset points",
                        xytext=(-20 - 90 * i, 45 + 20 * (len(above) - 1 - i)), fontsize=18,
                        ha="center", va="center", arrowprops=dict(**line_kw, relpos=(0.5, 0)))
        for i, k in enumerate(below):   # highest first, stacked downwards
            ax.annotate(pretty_ion(df["Ion"][k]), (x[k], y[k]), textcoords="offset points",
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
        + (f"\nχ²/ndf = {chi2ndf:.3g}" if args.weighted else "")
    )
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left", fontsize=18,
            family="monospace", bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.9))

    ax.set_ylabel(r"$f_{diff} = f_{exp,true} - f_{sim,true}$ (Hz)")
    ax.set_title("Second-order Correction (masses incl. electron binding)")
    ax.ticklabel_format(axis="x", useOffset=False, style="plain")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    # ---- Residuals panel ----
    axr.errorbar(x[fit_mask], resid[fit_mask], yerr=yerr, fmt="o", color="C0", zorder=3,
                 capsize=4 if args.weighted else 0)
    axr.scatter([x0], [resid[~fit_mask][0]], color="C3", marker="*", s=360, zorder=4)
    axr.axhline(0, color="C1", lw=1.5)
    axr.set_xlabel(r"$f_{sim,true}$ (Hz)")
    axr.set_ylabel("Residual (Hz)")
    axr.ticklabel_format(axis="x", useOffset=False, style="plain")
    axr.grid(alpha=0.3)
    fig.align_ylabels([ax, axr])
    plt.show()

    offer_save(c2, c1, c0)


if __name__ == "__main__":
    main()
