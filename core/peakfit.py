"""
Peak-fitting engine shared by Theia (theia/gui/further_fit.py) and the
spectrogram (energy calculation): line shapes, least-squares fit over a
region, start values for a clicked peak, noise estimate.

No GUI code here.  Moved out of theia/gui/further_fit.py in Nov 2026.

Parameters per peak: centre (Hz), amplitude (peak height above background),
FWHM (Hz) and, for pseudo-Voigt, eta (Lorentzian fraction, 0..1).  All
shapes are written in terms of FWHM so switching shape keeps the width.
Fitting is done in offset frequency u = f - xref (xref = middle of the fit
region) to keep the numbers well conditioned.
"""

import numpy as np


_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def _peak_shape(u, c, a, w, eta, shape):
    w = max(abs(w), 1e-9)
    g = np.exp(-0.5 * ((u - c) / (w * _FWHM_TO_SIGMA)) ** 2)
    if shape == "gauss":
        return a * g
    lor = 1.0 / (1.0 + ((u - c) / (w / 2.0)) ** 2)
    if shape == "lorentz":
        return a * lor
    eta = min(max(eta, 0.0), 1.0)
    return a * (eta * lor + (1.0 - eta) * g)


def _peak_area(a, w, eta, shape):
    ga = a * w * _FWHM_TO_SIGMA * np.sqrt(2 * np.pi)
    la = a * np.pi * w / 2.0
    if shape == "gauss":
        return ga
    if shape == "lorentz":
        return la
    return eta * la + (1 - eta) * ga


def _n_bg(bg):
    return {"none": 0, "const": 1, "linear": 2}.get(bg, 0)


def _shapes(state):
    """Line shape of each peak (each peak has its own)."""
    default = state.get("shape") or "gauss"
    return [pk.get("shape") or default for pk in state["peaks"]]


def _pack(state):
    """state -> (p0, names) for the parameters; centres as offsets from xref.
    eta is a parameter only for pseudo-Voigt peaks."""
    xref = state["xref"]
    p, names = [], []
    for i, (pk, shape) in enumerate(zip(state["peaks"], _shapes(state))):
        p += [pk["centre"] - xref, pk["amp"], pk["fwhm"]]
        names += [(i, "centre"), (i, "amp"), (i, "fwhm")]
        if shape == "pvoigt":
            p.append(pk.get("eta", 0.5))
            names.append((i, "eta"))
    for j in range(_n_bg(state["bg"])):
        p.append(state["bgp"][j])
        names.append((None, f"b{j}"))
    return np.array(p, dtype=float), names


def _model_from_vector(u, p, shapes, bg):
    """Model for the parameter vector laid out as in _pack."""
    y = np.zeros_like(u, dtype=float)
    q = 0
    for shape in shapes:
        c, a, w = p[q:q + 3]
        q += 3
        eta = 0.0
        if shape == "pvoigt":
            eta = p[q]
            q += 1
        y += _peak_shape(u, c, a, w, eta, shape)
    nb = _n_bg(bg)
    if nb >= 1:
        y += p[q]
    if nb == 2:
        y += p[q + 1] * u
    return y


def _region_mask(ff, region):
    lo, hi = region
    return (ff >= lo) & (ff <= hi)


def run_fit(ff, pp, sigma, state):
    """Least-squares fit over the region, starting from the current values.
    Returns the updated state (values, uncertainties, chi2/ndf)."""
    from scipy.optimize import curve_fit
    m = _region_mask(ff, state["region"]) & np.isfinite(pp)
    u, y = ff[m] - state["xref"], pp[m]
    p0, names = _pack(state)
    lo_u, hi_u = float(u.min()), float(u.max())
    bin_hz = float(np.median(np.diff(ff[m])))
    xref = state["xref"]
    lb, ub = [], []
    cb = state.get("centre_bounds")             # optional per-peak centre limits (Hz)
    fb = state.get("fwhm_bounds")

    def _lim(pk, key):
        v = (pk.get("lim") or {}).get(key)
        return None if v is None or not np.isfinite(v) else float(v)
    for (i, what) in names:
        pk = state["peaks"][i] if i is not None else {}
        if what == "centre":
            lo, hi = lo_u, hi_u
            if cb and cb[i]:
                lo, hi = max(lo, cb[i][0] - xref), min(hi, cb[i][1] - xref)
            if _lim(pk, "centre_min") is not None:
                lo = max(lo, _lim(pk, "centre_min") - xref)
            if _lim(pk, "centre_max") is not None:
                hi = min(hi, _lim(pk, "centre_max") - xref)
            if not lo < hi:
                raise ValueError(f"peak {i + 1}: the centre limits do not overlap the fit region")
            lb.append(lo); ub.append(hi)
        elif what == "amp":
            lb.append(0.0); ub.append(np.inf)
        elif what == "fwhm":
            lo, hi = 0.05 * bin_hz, max(hi_u - lo_u, bin_hz)
            if fb and fb[i]:
                lo, hi = fb[i]
            if _lim(pk, "fwhm_min") is not None:
                lo = max(_lim(pk, "fwhm_min"), 1e-9)
            if _lim(pk, "fwhm_max") is not None:
                hi = _lim(pk, "fwhm_max")
            if not lo < hi:
                raise ValueError(f"peak {i + 1}: FWHM min must be below FWHM max")
            lb.append(lo); ub.append(hi)
        elif what == "eta":
            lb.append(0.0); ub.append(1.0)
        else:
            lb.append(-np.inf); ub.append(np.inf)
    lb, ub = np.array(lb), np.array(ub)

    # fixed parameters are left out of the fit (held at their current value)
    bg_fix = set(state.get("bg_fix") or [])
    free = np.array([(what not in bg_fix) if i is None
                     else (what not in (state["peaks"][i].get("fix") or []))
                     for (i, what) in names], dtype=bool)
    n_free = int(free.sum())
    if n_free == 0:
        raise ValueError("every parameter is fixed: free at least one")
    if len(u) <= n_free:
        raise ValueError(f"Only {len(u)} bins in the fit region for {n_free} free parameters")
    for q in np.flatnonzero(free):                # start strictly inside the bounds
        if not (lb[q] < p0[q] < ub[q]):
            span = (ub[q] - lb[q]) if np.isfinite(ub[q] - lb[q]) else max(abs(p0[q]), 1.0)
            p0[q] = min(max(p0[q], lb[q] + 1e-6 * span), ub[q] - 1e-6 * span)
    shapes, bg = _shapes(state), state["bg"]
    n_peaks = len(shapes)
    p_all = p0.copy()

    def full(pf):
        p = p_all.copy()
        p[free] = pf
        return p

    def f_free(uu, *pf):
        return _model_from_vector(uu, full(np.array(pf)), shapes, bg)

    # absolute_sigma=False: the covariance is scaled by chi2/ndf, i.e. the
    # uncertainties are scaled by sqrt(chi2/ndf).  The noise sigma is only a
    # rough estimate (and real peaks are not perfect line shapes), so this
    # lets the scatter of the data about the fit set the uncertainties.
    popt_f, pcov_f = curve_fit(f_free, u, y, p0=p0[free], sigma=np.full(len(u), sigma),
                               absolute_sigma=False, bounds=(lb[free], ub[free]), maxfev=20000)
    popt = full(popt_f)
    pcov = np.zeros((len(p0), len(p0)))
    pcov[np.ix_(free, free)] = pcov_f
    ok_cov = np.all(np.isfinite(pcov_f))
    perr = np.sqrt(np.clip(np.diag(pcov), 0, None)) if ok_cov else np.full(len(popt), np.nan)
    perr[~free] = 0.0
    resid = y - _model_from_vector(u, popt, shapes, bg)
    ndf = max(len(u) - n_free, 1)

    new = dict(state)
    new["peaks"] = [dict(pk) for pk in state["peaks"]]
    new["bgp"] = list(state.get("bgp") or [0.0, 0.0])
    unc = {"peaks": [{} for _ in state["peaks"]], "bgp": [None, None]}
    for (i, what), v, e in zip(names, popt, perr):
        if i is None:
            j = int(what[1])
            new["bgp"][j], unc["bgp"][j] = float(v), float(e)
        else:
            new["peaks"][i][what] = float(v + state["xref"]) if what == "centre" else float(v)
            unc["peaks"][i][what] = float(e)
    # area uncertainty: linear error propagation with a numerical gradient
    for i in range(n_peaks):
        idx = [k for k, (j, _) in enumerate(names) if j == i]
        keys = [names[k][1] for k in idx]

        def area_of(vec, shape=shapes[i]):
            d = dict(zip(keys, vec))
            return _peak_area(d["amp"], d["fwhm"], d.get("eta", 0.5), shape)
        v0 = popt[idx].astype(float)
        grad = np.zeros(len(idx))
        for q in range(len(idx)):
            h = 1e-6 * max(abs(v0[q]), 1.0)
            vp = v0.copy()
            vp[q] += h
            grad[q] = (area_of(vp) - area_of(v0)) / h
        sub = pcov[np.ix_(idx, idx)]
        unc["peaks"][i]["area"] = (float(np.sqrt(max(grad @ sub @ grad, 0.0)))
                                   if ok_cov else float("nan"))
    new.update(unc=unc, chi2=float(np.sum((resid / sigma) ** 2) / ndf), ndf=int(ndf),
               sigma=float(sigma), fitted=True)
    return new


def _snap_peak(ff, pp, x, bgp, bg):
    """Start values for a peak clicked at x: centre and amplitude from the
    highest bin within +-3 bins, FWHM from where the data fall to half height."""
    k0 = int(np.clip(np.searchsorted(ff, x), 0, len(ff) - 1))
    lo, hi = max(k0 - 3, 0), min(k0 + 4, len(ff))
    k = lo + int(np.nanargmax(pp[lo:hi]))
    base = bgp[0] if _n_bg(bg) >= 1 else 0.0
    amp = max(float(pp[k] - base), 0.0)
    half = base + amp / 2.0
    left = k
    while left > 0 and pp[left - 1] > half and k - left < 200:
        left -= 1
    right = k
    while right < len(pp) - 1 and pp[right + 1] > half and right - k < 200:
        right += 1
    bin_hz = float(np.median(np.diff(ff[max(k - 5, 0):k + 6])))
    fwhm = max((right - left + 1) * bin_hz, 1.5 * bin_hz)
    return {"centre": float(ff[k]), "amp": amp, "fwhm": fwhm, "eta": 0.5}


def noise_sigma(pp):
    """Noise sigma per bin from the bin-to-bin differences (robust to peaks)."""
    pp = np.asarray(pp, dtype=float)
    d = np.diff(pp[np.isfinite(pp)])
    sigma = float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)) if len(d) else 1.0
    return sigma or 1.0
