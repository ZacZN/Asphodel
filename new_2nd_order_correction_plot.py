"""
Plot f_diff vs f_sim_true from the "New2ndOrderCorrection" sheet and fit a
2nd-order polynomial anchored at the reference ion (f_diff = 0 at 208Pb80+).

Fit model (shifted form, passes exactly through (x0, 0)):
    f_diff = a*(x - x0) + b*(x - x0)^2
Equivalent expanded form:
    f_diff = c0 + c1*x + c2*x^2
    c2 = b,  c1 = a - 2*b*x0,  c0 = b*x0^2 - a*x0

Usage:
    python new_2nd_order_correction_plot.py [spreadsheet] [sheet name]
"""
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 2 * plt.rcParams["font.size"]  # all fonts doubled

FILENAME = sys.argv[1] if len(sys.argv) > 1 else "/home/zn00092/Desktop/rionid-v8-186Hf/186hf_rionid-v8_PID.xlsx"
SHEET = sys.argv[2] if len(sys.argv) > 2 else "New2ndOrderCorrection"
ANCHOR_ION = "208Pb80+"
COLUMNS = ["Ion", "Harmonic", "f_exp", "f_sim", "f_exp_true", "f_sim_true", "f_diff"]


def read_table(filename, sheet):
    """Find the header row containing 'Ion' and return the table below it."""
    raw = pd.read_excel(filename, sheet_name=sheet, header=None)
    for r in range(len(raw)):
        row = raw.iloc[r].astype(str).str.strip().tolist()
        if "Ion" in row and "f_diff" in row:
            df = raw.iloc[r + 1:].copy()
            df.columns = [str(c).strip() for c in raw.iloc[r]]
            df = df[COLUMNS].dropna(subset=["Ion"])
            for c in COLUMNS[1:]:
                df[c] = pd.to_numeric(df[c])
            df["Ion"] = df["Ion"].astype(str).str.strip()
            return df.reset_index(drop=True)
    raise ValueError(f"No header row with 'Ion' and 'f_diff' found in sheet '{sheet}'")


def pretty_ion(name):
    """'208Pb80+' -> mathtext ^{208}Pb^{80+}"""
    m = re.fullmatch(r"(\d+)([A-Za-z]+)(\d+[+-])", name)
    if not m:
        return name
    A, el, q = m.groups()
    return rf"$^{{{A}}}\mathrm{{{el}}}^{{{q}}}$"


def anchored_quadratic_fit(x, y, x0):
    """Least squares for y = a*dx + b*dx^2 (no intercept), dx = x - x0."""
    dx = x - x0
    X = np.column_stack([dx, dx**2])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = len(y) - X.shape[1]
    s2 = resid @ resid / dof if dof > 0 else np.nan
    cov = s2 * np.linalg.inv(X.T @ X)
    return coef, cov, resid


def main():
    df = read_table(FILENAME, SHEET)
    print(df.to_string())

    anchor = df[df["Ion"] == ANCHOR_ION]
    if anchor.empty:
        raise ValueError(f"Anchor ion {ANCHOR_ION} not found in table")
    x0 = float(anchor["f_sim_true"].iloc[0])
    y0 = float(anchor["f_diff"].iloc[0])
    if y0 != 0:
        print(f"Warning: f_diff of {ANCHOR_ION} is {y0}, not 0; anchoring at (x0, 0) anyway")

    x = df["f_sim_true"].to_numpy(float)
    y = df["f_diff"].to_numpy(float)
    fit_mask = df["Ion"] != ANCHOR_ION  # anchor point is enforced, not fitted

    (a, b), cov, _ = anchored_quadratic_fit(x[fit_mask], y[fit_mask], x0)
    da, db = np.sqrt(np.diag(cov))

    # Expanded coefficients and their propagated uncertainties
    c2, c1, c0 = b, a - 2 * b * x0, b * x0**2 - a * x0
    J = np.array([[0.0, 1.0],            # dc2/d(a,b)
                  [1.0, -2 * x0],        # dc1/d(a,b)
                  [-x0, x0**2]])         # dc0/d(a,b)
    cov_c = J @ cov @ J.T
    dc2, dc1, dc0 = np.sqrt(np.diag(cov_c))

    resid = y - (a * (x - x0) + b * (x - x0) ** 2)
    rms = np.sqrt(np.mean(resid[fit_mask] ** 2))

    print(f"\nShifted (s1 (x-x0) + s2 (x-x0)^2): s1 = {a:.6e} ± {da:.2e},  s2 = {b:.6e} ± {db:.2e},  x0 = {x0:.6f} Hz")
    print(f"Expanded (a x^2 + b x + c): a = {c2:.6e} ± {dc2:.2e},  b = {c1:.6e} ± {dc1:.2e},  c = {c0:.6e} ± {dc0:.2e}")
    print(f"RMS residual = {rms:.3f} Hz")

    # ---- Plot ----
    # constrained layout re-fits margins to the window size, so labels never get cut off
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(22, 15), sharex=True, layout="constrained",
                                  gridspec_kw=dict(height_ratios=[3, 1]))
    fig.get_layout_engine().set(hspace=0.0, h_pad=0.05)
    ax.scatter(x[fit_mask], y[fit_mask], color="C0", zorder=3, label="Data")
    ax.scatter([x0], [0], color="C3", marker="*", s=360, zorder=4,
               label=f"Anchor ({pretty_ion(ANCHOR_ION)})")

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
                        xytext=(-20 - 90 * i, 45 + 20 * (len(above) - 1 - i)), fontsize=18, ha="center", va="center",
                        arrowprops=dict(**line_kw, relpos=(0.5, 0)))
        for i, k in enumerate(below):   # highest first, stacked downwards
            ax.annotate(pretty_ion(df["Ion"][k]), (x[k], y[k]), textcoords="offset points",
                        xytext=(40 + 20 * i, -30 - 34 * i), fontsize=18, ha="left", va="center",
                        arrowprops=dict(**line_kw, relpos=(0, 0.5)))

    pad = 0.05 * (x.max() - x.min())
    xx = np.linspace(x.min() - pad, x.max() + pad, 500)
    ax.plot(xx, a * (xx - x0) + b * (xx - x0) ** 2, color="C1", label="Anchored 2nd-order fit")
    ax.axhline(0, color="grey", lw=0.8, ls="--")

    txt = (
        r"Expanded: $f_{diff} = a x^2 + b x + c$" "\n"
        rf"  $a$ = {c2:.6e} ± {dc2:.2e} Hz$^{{-1}}$" "\n"
        rf"  $b$ = {c1:.6e} ± {dc1:.2e}" "\n"
        rf"  $c$ = {c0:.6e} ± {dc0:.2e} Hz" "\n\n"
        rf"RMS residual = {rms:.3f} Hz"
    )
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left", fontsize=18,
            family="monospace", bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.9))

    ax.set_ylabel(r"$f_{diff} = f_{exp,true} - f_{sim,true}$ (Hz)")
    ax.set_title("Second-order Correction")
    ax.ticklabel_format(axis="x", useOffset=False, style="plain")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    # ---- Residuals panel ----
    axr.scatter(x[fit_mask], resid[fit_mask], color="C0", zorder=3)
    axr.scatter([x0], [resid[~fit_mask][0]], color="C3", marker="*", s=360, zorder=4)
    axr.axhline(0, color="C1", lw=1.5)
    axr.set_xlabel(r"$f_{sim,true}$ (Hz)")
    axr.set_ylabel("Residual (Hz)")
    axr.ticklabel_format(axis="x", useOffset=False, style="plain")
    axr.grid(alpha=0.3)
    fig.align_ylabels([ax, axr])
    plt.show()


if __name__ == "__main__":
    main()
