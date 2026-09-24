"""
Linear fit of sigma vs intensity, with residuals.

Spreadsheet columns (by position): ion name, intensity, sigma.
Works with .ods, .xlsx/.xls or .csv, with or without a header row.
Reading .ods needs odfpy:  pip install odfpy

Usage:
    python intensity_variation_fit.py data.ods
    python intensity_variation_fit.py data.ods --exclude 206Pb79+ 208Pb80+

--exclude takes one or more ion names; each matches any ion whose name
starts with it (so "206Pb" matches "206Pb79+"). Excluded points are left
out of the fit and R^2 but still plotted as open grey circles.
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_data(path):
    reader = pd.read_csv if path.lower().endswith(".csv") else pd.read_excel
    raw = reader(path, header=None)
    # Remove completely empty rows/columns (e.g. data not starting in cell A1)
    df = raw.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    if df.shape[1] < 3:
        raise ValueError(f"Expected 3 columns, found {df.shape[1]}. Raw sheet:\n{raw.head(10)}")
    df = df.iloc[:, :3]
    df.columns = ["ion", "intensity", "sigma"]

    def to_num(col):
        # Also accept numbers stored as text, including comma decimals ("1,23")
        s = col.astype(str).str.strip().str.replace(",", ".", regex=False)
        return pd.to_numeric(s, errors="coerce")

    df["intensity"] = to_num(df["intensity"])
    df["sigma"] = to_num(df["sigma"])
    good = df.dropna(subset=["intensity", "sigma"]).reset_index(drop=True)

    print(f"Read {len(good)} data rows from {path}")
    if len(good) < 2:
        raise ValueError(
            "Fewer than 2 numeric rows found in columns 2 and 3.\n"
            f"First rows of the sheet as read:\n{raw.head(10)}"
        )
    return good


def linear_fit(x, y):
    """Least-squares fit y = m*x + c. Returns params, their errors, and covariance."""
    (m, c), cov = np.polyfit(x, y, 1, cov=True)
    m_err, c_err = np.sqrt(np.diag(cov))
    return m, c, m_err, c_err


def main(path, exclude=()):
    df = read_data(path)
    names = np.array([str(n).strip() for n in df["ion"]])
    x, y = df["intensity"].values, df["sigma"].values

    excl = np.zeros(len(df), dtype=bool)
    for e in exclude:
        hit = np.array([n.startswith(e) for n in names])
        if not hit.any():
            print(f"Warning: --exclude '{e}' matched no ions")
        excl |= hit
    if excl.any():
        print("Excluded from fit:", ", ".join(names[excl]))
    inc = ~excl
    if inc.sum() < 3:
        raise ValueError("Need at least 3 points left in the fit.")

    m, c, m_err, c_err = linear_fit(x[inc], y[inc])
    resid = y - (m * x + c)
    r2 = 1 - np.sum(resid[inc]**2) / np.sum((y[inc] - y[inc].mean())**2)

    print(f"Points in fit = {inc.sum()}")
    print(f"slope     = {m:.6g} ± {m_err:.2g}")
    print(f"intercept = {c:.6g} ± {c_err:.2g}")
    print(f"R^2       = {r2:.4f}")

    fig, (ax, axr) = plt.subplots(
        2, 1, sharex=True, figsize=(8, 7),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )
    excl_style = dict(ls="none", marker="o", mfc="none", mec="grey")

    # Data + fit
    xx = np.linspace(x.min(), x.max(), 200)
    ax.plot(x[inc], y[inc], "o", label="data")
    if excl.any():
        ax.plot(x[excl], y[excl], label="excluded", **excl_style)
    ax.plot(xx, m * xx + c, "-",
            label=f"fit: σ = ({m:.3g})·I + ({c:.3g})\n$R^2$ = {r2:.3f}")
    for xi, yi, n in zip(x, y, names):
        ax.annotate(n, (xi, yi), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_ylabel("Sigma")
    ax.legend()
    ax.grid(alpha=0.3)

    # Residuals (excluded points shown relative to the same fit)
    axr.axhline(0, color="k", lw=0.8)
    axr.plot(x[inc], resid[inc], "o")
    if excl.any():
        axr.plot(x[excl], resid[excl], **excl_style)
    for xi, ri, n in zip(x, resid, names):
        axr.annotate(n, (xi, ri), textcoords="offset points", xytext=(4, 4), fontsize=7)
    axr.margins(y=0.3)  # room for labels
    axr.set_xlabel("Intensity")
    axr.set_ylabel("Residual")
    axr.grid(alpha=0.3)

    fig.savefig("linear_fit.png", dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Linear fit of sigma vs intensity.")
    ap.add_argument("file", nargs="?", default="data.ods", help="spreadsheet (.ods/.xlsx/.csv)")
    ap.add_argument("--exclude", nargs="+", default=[], metavar="ION",
                    help="ion name(s) to leave out of the fit, e.g. 206Pb79+")
    args = ap.parse_args()
    main(args.file, args.exclude)
