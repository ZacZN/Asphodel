# Asphodel suite — notes

Structure, decisions and history of the Asphodel suite. Started as the handover note from the Theia project (22 Oct 2026) and kept up to date since. The same text is the reference doc of the "Asphodel" Claude project.

## Goal

Combine two existing codebases into one analysis suite called **Asphodel**, with a home screen / dashboard to manage data files and outputs and to launch the tools.

- **Theia** (`scanner.py`, Dash app, port 8052). Tabs: Viewer (one 1-D spectrum with simulated ion frequencies overlaid), Scan (batch peak detection + identification over a folder), Scan Results, Further output (confirmed IDs, aligning/fitting, isochronicity curve).
- **Old Asphodel scripts**:
  - `plot_spectrogram.py`: its own `http.server` + Plotly/JS page on a random port, showing a 2-D/3-D time × frequency view with simulated ion lines. It launches `decay_fit.py` via subprocess.
  - `decay_fit.py`: exponential / Bateman fits and half-lives.
  - `intensity_variation_fit.py`.
  - `asphodel_scripts_guide.pdf`.

## Decisions so far (confirmed by Zac)

- Start from a **new folder** named `Asphodel`. The existing folders get renamed (e.g. `Asphodel_old`, `Theia_old`) and are kept until the new suite works end to end.
- The old Claude project called "Asphodel" gets renamed (e.g. "Asphodel (old scripts)"). This new project is "Asphodel".
- **Keep the git history** of both Theia and the old Asphodel repo. Do this by merging the repos, not by copying files.
- **Separate data folder** collecting all data in one place, outside the code repo.
- **Data folder (22 Oct):** `~/Desktop/Asphodel/data/` (inside the repo folder, git-ignored). One experiment so far, `E0203/` with `spectra_1d/`, `spectrograms/`, `summed_1d/`, `outputs/{scans,confirmed,fits,figures}`. Outputs live per experiment. Filled with hard links (`cp -al`) because the disk is 91% full (data ~27 GB, 22 GB free); anything new is written only into the new structure.
- **Tartarus joins the suite (23 Oct)** as its own section `tartarus/` (sum_spectra.py, new_2nd_order_correction_plot.py), merged with its git history like the other two. Its .npz sums live in `data/E0203/summed_1d/`. Data folder filled and verified (file counts match sources).
- **Elysium joins the suite (23 Oct)** as `elysium/` (levelschemer.py), merged with its history from `main`. Level scheme files live per experiment in `data/<experiment>/levelschemes/` (copied, not hard-linked, since schemes are edited in place). Old folders renamed `Tartarus_old`, `Elysium_old`.
- **Merge from the `WIP` branches** of both old repos (22 Oct). `main` in both is weeks behind; `GUI-Decayfit` (Asphodel) is already in `WIP`. Old folders are renamed to `Theia_old` / `Asphodel_old`, and the new empty `~/Desktop/Asphodel` exists.

## Proposed approach (agreed in outline, details still open)

1. **Home screen as a launcher**: a new Dash app (`home.py`). It indexes the data folders, `scans/index.csv`, `confirmed/*.csv` and fit outputs. It starts the tools as separate processes with arguments filled in (e.g. `plot_spectrogram.py <file> fmin fmax`). The tools stay separate apps; there's no rewrite of `plot_spectrogram.py` into Dash.
2. **Shared core**: one copy of `nubase_reader.py`, `nubase_4.mas20`, the ion-frequency simulation and the physics settings. Right now these are duplicated: the simulation in `plot_spectrogram.py` was ported from `scanner.py`, and reference ion, γ_t, harmonics and correction coefficients are stored separately in `viewer_config.json`, `scanner_config.json` and `.asphodel_prefs.json`.
3. **One git repo** containing both histories.

A mockup of the home screen was made: `suite_home_mockup.html`. It has:

- a sidebar with data sources, shared physics settings and the NUBASE file
- tool cards (Viewer, Scanner, Spectrogram, Decay fit) with running/stopped status
- a "resume last session" bar
- a data-file table with "open with" buttons
- output tabs: scan runs, confirmed IDs, decay fits, figures
- an activity log and a notes panel.

The styling matches Theia: accent #f57d15, pale #fff4ec, sidebar #f9f9f9, borders #dddddd.

## Proposed folder layout (to confirm)

```
~/Desktop/Asphodel/              code, git repo
  home.py                        new launcher / dashboard
  core/                          nubase_reader.py, nubase_4.mas20, physics/simulation, shared settings
  theia/                         scanner.py (+ old/)
  spectrogram/                   plot_spectrogram.py, decay_fit.py, intensity_variation_fit.py
  docs/                          asphodel_scripts_guide.pdf, scanner_design.md

~/Desktop/<DATA_FOLDER>/         data, not in git (name to be decided)
  <experiment>/                  e.g. E0203
    spectra_1d/                  from 1dSpectra/SplitTDMS_Output_2^17_nobaseline/
    spectrograms/                from Asphodel_old/summed/
    summed_1d/                   from Tartarus/
    outputs/scans/  outputs/confirmed/  outputs/fits/  outputs/figures/
```

## Migration steps

**Status (22 Oct 2026): done.** `~/Desktop/Asphodel` is a git repo on `main` with both `WIP` histories merged (theia/ and spectrogram/) plus a top-level `.gitignore`. Newer git needed `--no-rebase` on the two pulls. Old folders untouched.

Run these by hand. Check the branch names first with `git branch` in each repo; `main` is assumed below.

```bash
# 0. Commit any outstanding work in both old repos
cd ~/Desktop/Theia    && git status
cd ~/Desktop/Asphodel && git status

# 1. Rename the old folders
cd ~/Desktop
mv Asphodel Asphodel_old
mv Theia Theia_old

# 2. In temporary clones, move each repo's contents into a subfolder (history is kept)
git clone Theia_old /tmp/theia_tmp
cd /tmp/theia_tmp && mkdir theia && git mv $(git ls-tree --name-only HEAD) theia/ \
  && git commit -m "Move Theia into theia/ for merge into Asphodel suite"

git clone ~/Desktop/Asphodel_old /tmp/asph_tmp
cd /tmp/asph_tmp && mkdir spectrogram && git mv $(git ls-tree --name-only HEAD) spectrogram/ \
  && git commit -m "Move Asphodel scripts into spectrogram/ for merge into suite"

# 3. Create the new repo and merge both histories
mkdir ~/Desktop/Asphodel && cd ~/Desktop/Asphodel && git init
git commit --allow-empty -m "Start Asphodel suite"
git pull /tmp/theia_tmp WIP --allow-unrelated-histories --no-edit
git pull /tmp/asph_tmp  WIP --allow-unrelated-histories --no-edit
git log --oneline --graph | head -30        # both histories should be there

# 4. Later restructuring (core/, docs/) as normal git mv commits; history follows renames
#    (use `git log --follow <file>`).
```

Notes:

- `git mv $(git ls-tree ...)` also moves `.gitignore` into the subfolder. Add a top-level `.gitignore` afterwards (`__pycache__/`, `*.sqlite`, `*.npz`, `_viewer_output.html`, etc.).
- Check that no large data (`.npz`, `summed/`) is tracked in either old repo before merging.
- The data is copied into the data folder separately (not via git). Copy and verify before removing anything from the old locations.

## Paths (done 23 Oct)

- `core/paths.py`: `DATA_ROOT = <repo>/data`; active experiment from `data/asphodel_settings.json` (`{"experiment": "E0203"}`); helpers `spectra_1d()`, `spectrograms()`, `summed_1d()`, `levelschemes()`, `rionid()`, `outputs(kind)`, `settings_file(name)`, `experiment_dir()`, `set_experiment()`.
- Scripts import it with `sys.path.insert(1, <repo root>)` then `from core import paths`.
- Per-experiment tool settings in `data/<exp>/settings/`: `scanner_config.json`, `viewer_config.json`, `spectrogram_prefs.json` (was `.asphodel_prefs.json`), `levelschemer_settings.json` (was `~/.config/levelschemer/settings.json`). Old configs migrated with paths rewritten.
- `remap_legacy(path)`: translates old Desktop paths (1dSpectra/..., Asphodel(_old)/summed, Tartarus(_old), Theia(_old)/scans|confirmed, Elysium(_old)/levelschemes) to data/E0203 when loading; old scan records are never rewritten (hard-linked).
- `scans/index.csv` had its hard link broken, since the scanner appends to it.
- Default inputs: `data/E0203/Schottky_Uncertainties.ods` (intensity_variation_fit), `data/E0203/rionid/186hf_rionid-v8_PID.xlsx` (correction plot). Figures go to `outputs/figures/`.

## Shared core (done 23 Oct)

- `core/nubase_reader.py` + `core/nubase_4.mas20`: single copy (the theia/ and spectrogram/ copies were removed).
- `core/physics.py`: simulation (`build_ion_database`, `simulate_frequencies`, `delta_f`, `apply_correction`, half-life cut, hover labels), used by scanner.py and plot_spectrogram.py. The Viewer tab still has its own `v_compute_frequencies_multi` (different output format) but uses the shared `delta_f`/`apply_correction`.
- Shared physics defaults per experiment: `data/<exp>/settings/physics.json` (edit by hand until the home screen exists). Tools load `physics.with_shared(TOOL_DEFAULTS, saved)` and save `physics.strip_shared(cfg)`, so tool configs hold only overrides. Decided by Zac: shared default harmonics 212, 213, 214; the spectrogram keeps 211–215 as its override.

## docs/ (done 23 Oct)

`docs/asphodel_scripts_guide.pdf` (moved from spectrogram/, outdated: to be rewritten once the structure settles), `docs/scanner_design.md` (recreated from the handover copy, with the new locations), `docs/suite_notes.md` (copy of this note), `docs/suite_home_mockup.html`.

## Home page (first version 23 Oct)

- `home.py` (repo root, Dash, port 8050 or a free one). Decided by Zac: all tools stay separate programs; the home page starts/stops them as processes with `ASPHODEL_EXPERIMENT` set, so a running tool keeps its experiment. Live messaging between tools (a hub in home.py) comes later, once the interactions are listed.
- Sections: Theia (theia/theia.py, renamed from scanner.py), Spectrogram (plot_spectrogram.py, decay_fit.py), Tartarus (sum_spectra.py, new_2nd_order_correction_plot.py, intensity_variation_fit.py moved here), Elysium (levelschemer.py).
- Contents: experiment switcher + new experiment, tool cards with status and small forms, data-file table with "open with", outputs tabs (scan runs, confirmed IDs, decay fits with Re-run/Spectrogram, figures), shared physics editor with the tool overrides, NUBASE info, notes (`<data_dir>/notes.md`). Left out for now: resume-last-session bar, activity log, diffing scan runs.
- Experiment profiles (`core/paths.py`): `data/asphodel_settings.json` holds `experiment` and `profiles` {name: {data_dir}}; a profile's data folder can be anywhere; a folder in data/ without an entry is also a profile.
- Decay fit has no card (Zac): it is started from the spectrogram or re-run from the Decay fits tab. A ⏻ Quit button in the header stops the home page; tools started from it keep running.
- "Open with Viewer / Scan in Theia" works through the settings files (viewer_config.json file_path/spectrum_name, scanner_config.json spectra_dir), then reload the Theia page.

## Things that will break and need fixing

- **Absolute paths** in `.asphodel_prefs.json` (`lastDir`, `session.file` → `.../Asphodel/summed/...`) and `viewer_config.json` (`file_path` → `Tartarus/`). `scanner_config.json` `spectra_dir` → `1dSpectra/...`.
- **Paths relative to script location**: `scanner.py` writes `scans/` and `confirmed/` next to itself (`HERE`). `plot_spectrogram.py` finds `decay_fit.py` via `_HERE` and imports `nubase_reader` from its own folder. Both need pointing at `core/` and at the data/outputs folder.
- ~~`decay_fit.py` only plots/prints.~~ Done 23 Oct: a "Save fit parameters" button in the plot window (top left) writes one JSON per fit to `data/<exp>/outputs/fits/fit_<stamp>_<file>[_<label>].json` (format `asphodel-decay-fit` v1: file, label, mode decay/bateman/joint, settings, parameters with value/error/status, half-lives lab + c.o.m., χ²/ndf, error model; no data points). Decided by Zac: saving only on request, no figure saved.

## Current physics settings (E0203, 208Pb80+ setting)

Reference 208Pb80+, harmonic 214, f_ref 412391232 Hz, γ_t 1.399. Harmonics 211–215 (spectrogram) / 212–214 (viewer). Z 60–83, N 100–127. el1 156,66; el2 200,82. mq_width 0.2. Correction on: a 2.8498393862845746e-07, b -1.0910358215490124, c 1044186.4522976408. Min half-life 1 ms.

## scanner.py design notes (copied from Theia project doc `scanner_design.md`)

Dash GUI on port 8052. Physics copied from viewer.py: vectorised `compute_frequencies_multi`, with the same 2nd-order correction applied to the revolution frequency before harmonic scaling.

Decisions (confirmed by Zac, 2026-09-30):

- **Reference tracking:** spectra sorted by natural filename order. Search ±window around the ref found in the previous spectrum (the first spectrum uses the nominal value). Pick the strongest (or nearest) detected maximum, with optional fallback to the predicted position.
- **Peak detection** runs on raw data, with no smoothing:
  - Noise is estimated blockwise with median/MAD.
  - A maximum is kept if prominence/σ_saddle ≥ k_prom and height/σ_bg ≥ k_height, where σ_saddle = sqrt(σ_bg² + (r·saddle level)²). This handles signal-dependent noise on big peaks.
  - Shoulders are found from a finite-step second difference at scale s, excluding anything within ±s of an accepted maximum.
- **Matching:** every peak within ±x Hz of a simulated (ion, harmonic) is recorded, with closest_for_ion / closest_for_peak flags.
- **Output:** `scans/scan_<timestamp>_<settings-hash>.sqlite` + `.json` sidecar + `index.csv`. Tables: run_info, settings, spectra, peaks (all, with identified flag), identifications, view hits.
- **Synthetic-data test** (gamma noise, 2^17 bins): k_height=5 gives ~0–5 false maxima per spectrum; k_height=3 gives ~180.
- **Defaults:** k_prom 5, k_height 5, r 0.05, shoulder scale 10 Hz, k_shoulder 5.

## How Zac wants Claude to work

- Always ask for clarification rather than making assumptions.
- After any code change, give a copy-pasteable git commit as commands: `git add <files>` then `git commit -m "..."`.
- PDFs: greyscale text with #f57d15 as the only accent colour (see the pdf-style skill).

## Open questions (not yet decided)

1. Name of the data folder and how experiments are named/organised inside it.
2. Tools as separate browser tabs launched from the home screen, or eventually one window?
3. Do outputs (scans, confirmed, fits) live in the data folder per experiment, or in the repo?
4. Which mockup features to keep: notes panel, activity log, diffing two scan runs, "resume last session".
5. Is `Tartarus/` only data, or also code that should join the suite? Same question for other Desktop folders (e.g. `DetectPeaks`, `NoiseAnalysis`, `Elysium`).
