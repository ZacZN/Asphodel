# Theia scanner — design notes

Recreated from the Theia project doc `scanner_design.md` (via the Asphodel handover note, 22 Oct 2026). The design decisions are unchanged; the file locations were updated for the Asphodel suite (see the last section).

## Overview

`theia/scanner.py` is a Dash GUI on port 8052. Physics copied from viewer.py: vectorised `compute_frequencies_multi`, with the same 2nd-order correction applied to the revolution frequency before harmonic scaling.

## Decisions (confirmed by Zac, 2026-09-30)

- **Reference tracking:** spectra sorted by natural filename order. Search ±window around the ref found in the previous spectrum (the first spectrum uses the nominal value). Pick the strongest (or nearest) detected maximum, with optional fallback to the predicted position.
- **Peak detection** runs on raw data, with no smoothing:
  - Noise is estimated blockwise with median/MAD.
  - A maximum is kept if prominence/σ_saddle ≥ k_prom and height/σ_bg ≥ k_height, where σ_saddle = sqrt(σ_bg² + (r·saddle level)²). This handles signal-dependent noise on big peaks.
  - Shoulders are found from a finite-step second difference at scale s, excluding anything within ±s of an accepted maximum.
- **Matching:** every peak within ±x Hz of a simulated (ion, harmonic) is recorded, with closest_for_ion / closest_for_peak flags.
- **Output:** `scan_<timestamp>_<settings-hash>.sqlite` + `.json` sidecar + `index.csv`. Tables: run_info, settings, spectra, peaks (all, with identified flag), identifications, view hits.
- **Synthetic-data test** (gamma noise, 2^17 bins): k_height=5 gives ~0–5 false maxima per spectrum; k_height=3 gives ~180.
- **Defaults:** k_prom 5, k_height 5, r 0.05, shoulder scale 10 Hz, k_shoulder 5.

## Changes in the Asphodel suite (23 Oct 2026)

- The ion simulation (`build_ion_database`, `simulate_frequencies`, the correction and the half-life cut) now lives in `core/physics.py`, shared with `spectrogram/plot_spectrogram.py`. NUBASE is read by `core/nubase_reader.py`.
- Scans are written to `data/<experiment>/outputs/scans/` (was `scans/` next to the script); confirmed identifications to `data/<experiment>/outputs/confirmed/`.
- Settings: `data/<experiment>/settings/scanner_config.json` and `viewer_config.json`. The physics values (reference ion, γ_t, harmonics, Z/N limits, correction, half-life cut) default to the shared `data/<experiment>/settings/physics.json`; the scanner's files keep only its overrides.
