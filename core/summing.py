"""
Summing 1-D spectra in whole bins (no interpolation), shared by Theia's
Further output tab and Tartarus sum_spectra.py.

    positions, lost = track_strongest(spectra, window_hz=500)
    freq, total, info = sum_whole_bins(spectra, positions)

spectra: list of (freq, power) arrays, in time order.

Alignment ("strongest peak", tracked):  the first spectrum's position is its
strongest bin; every next spectrum's is its strongest bin within ±window of
the previous spectrum's position, so a different ion becoming the strongest
somewhere else in a spectrum cannot pull the alignment away.  If that bin is
on the edge of the window and the power still rises outside it (the peak has
moved further than the window), or the window holds
no data, the spectrum is left out of the sum (reported in `lost`) and the
next spectrum is searched around the last position found.

Shifting:  each spectrum is moved by a whole number of bins so that its
position lands on the target, the first spectrum's position (round((target -
position) / bin width)); the power values are never interpolated, so the
noise of each bin stays that of the measured bins.  What is left of the
offset (at most half a bin) is reported per spectrum.

Edges:  only the bins every (shifted) spectrum covers are kept, so every bin
of the sum holds the same number of spectra.

The spectra must have the same bin width (their frequency grids may be offset
from each other by any amount: the offset is included in the shift).
"""

import numpy as np

DEFAULT_WINDOW_HZ = 500.0


def bin_width(ff):
    ff = np.asarray(ff, dtype=float)
    if len(ff) < 2:
        raise ValueError("a spectrum needs at least two bins")
    return float((ff[-1] - ff[0]) / (len(ff) - 1))


def track_strongest(spectra, window_hz=DEFAULT_WINDOW_HZ):
    """Position (Hz) of the tracked strongest peak in every spectrum, or None
    where it was not found within ±window of the previous position (see the
    module docstring).  Returns (positions, lost): lost = [(index, reason)]."""
    window_hz = float(window_hz)
    if not window_hz > 0:
        raise ValueError("the peak-tracking window must be > 0 Hz")
    positions, lost, prev = [], [], None
    for i, (ff, pp) in enumerate(spectra):
        ff = np.asarray(ff, dtype=float)
        pp = np.asarray(pp, dtype=float)
        if prev is None:
            sel = np.isfinite(pp)
        else:
            sel = np.isfinite(pp) & (np.abs(ff - prev) <= window_hz)
        if not sel.any():
            positions.append(None)
            lost.append((i, f"no data within ±{window_hz:g} Hz of the previous peak"))
            continue
        idx = np.flatnonzero(sel)
        j = idx[int(np.argmax(pp[idx]))]
        rising_out = ((j == idx[0] and j > 0 and pp[j - 1] > pp[j])
                      or (j == idx[-1] and j < len(pp) - 1 and pp[j + 1] > pp[j]))
        if prev is not None and rising_out:
            # strongest on the window edge and the power still rises outside
            # it: the peak has moved further than the window
            positions.append(None)
            lost.append((i, f"peak not inside ±{window_hz:g} Hz of the previous one "
                            f"(power still rising beyond the window edge)"))
            continue
        prev = float(ff[j])
        positions.append(prev)
    return positions, lost


def sum_whole_bins(spectra, positions=None, target=None):
    """Sum of the spectra, each shifted by whole bins so that positions[i]
    lands on target (default: the first position), cropped to the bins every
    spectrum covers.  positions=None: no alignment (grid offsets only).
    Entries of positions that are None are left out.

    Returns (freq, total, info); info: n (spectra summed), used (indices),
    shift_bins, shift_hz, residual_hz (target - position - shift, |.| <= half
    a bin), bin_hz, target, cropped_low / cropped_high (bins dropped at each
    end of the first spectrum's range)."""
    if not spectra:
        raise ValueError("no spectra to sum")
    if positions is not None and len(positions) != len(spectra):
        raise ValueError("one position per spectrum is needed")
    used = [i for i in range(len(spectra)) if positions is None or positions[i] is not None]
    if not used:
        raise ValueError("no spectrum has the alignment peak")
    ff0 = np.asarray(spectra[used[0]][0], dtype=float)
    bw = bin_width(ff0)
    if positions is not None and target is None:
        target = float(positions[used[0]])

    offsets, shifts, residuals = [], [], []
    for i in used:
        ff = np.asarray(spectra[i][0], dtype=float)
        if not np.isclose(bin_width(ff), bw, rtol=1e-6, atol=0.0):
            raise ValueError(f"spectrum {i + 1} has a bin width of {bin_width(ff):.6g} Hz, the first "
                             f"has {bw:.6g} Hz: spectra with different bin widths cannot be summed "
                             f"in whole bins")
        want = (target - float(positions[i])) if positions is not None else 0.0
        grid_off = (ff[0] - ff0[0]) / bw            # where this spectrum's bin 0 sits, in bins
        k = int(round(want / bw + grid_off))        # index of its bin 0 on the first one's grid
        shift_bins = k - grid_off                   # what the shift moved it, in bins
        offsets.append(k)
        shifts.append(int(round(k - round(grid_off))))
        residuals.append(want - shift_bins * bw)

    lo = max(offsets)
    hi = min(k + len(spectra[i][1]) for k, i in zip(offsets, used))
    if hi <= lo:
        raise ValueError("after shifting, no frequency bin is covered by every spectrum")
    total = np.zeros(hi - lo, dtype=float)
    for k, i in zip(offsets, used):
        pp = np.asarray(spectra[i][1], dtype=float)
        total += pp[lo - k:hi - k]
    # frequencies: the first spectrum's own values where it has them
    idx = np.arange(lo, hi)
    freq = ff0[0] + idx * bw
    inside = (idx >= 0) & (idx < len(ff0))
    freq[inside] = ff0[idx[inside]]
    info = dict(n=len(used), used=used, shift_bins=np.array(shifts), bin_hz=bw,
                shift_hz=np.array(shifts, dtype=float) * bw, residual_hz=np.array(residuals),
                target=target, cropped_low=max(lo, 0), cropped_high=max(len(ff0) - hi, 0))
    return freq, total, info


class LazySpectra:
    """A list of spectra read from disk only when used (so summing a large
    folder never holds every spectrum in memory): LazySpectra(paths, load)
    with load(path) -> (freq, power)."""

    def __init__(self, items, load):
        self.items = list(items)
        self.load = load

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.load(self.items[i])

    def __iter__(self):
        for it in self.items:
            yield self.load(it)
