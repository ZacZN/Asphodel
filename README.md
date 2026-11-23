# Asphodel

Analysis suite for Schottky data from the ESR storage ring at GSI: isotope identification from revolution frequencies, spectrograms and decay fits, spectrum summing and corrections, level schemes and a Segrè chart. It brings together the earlier Theia, Asphodel, Tartarus and Elysium tools.

Every tool is its own program. The home page starts and stops them, keeps the experiment settings and the connected data folders, and shows the outputs.

## Requirements

- Linux (macOS and Windows are not supported yet)
- Python 3.9 or newer, with `venv` and Tk
- Recommended: Firefox (Asphodel runs in its own Firefox window) and zenity (folder dialogs)

On Ubuntu/Debian:

```bash
sudo apt install git python3 python3-venv python3-tk zenity
```

## Install

```bash
git clone git@github.com:ZacZN/Asphodel.git      # or https://github.com/ZacZN/Asphodel.git
cd Asphodel
./install.sh
```

`install.sh` does three things. It creates a Python environment in `Asphodel/.venv/` and installs the packages from `requirements.txt`. It adds the `asphodel` command in `~/.local/bin`. It adds an "Asphodel" entry to the app menu. Nothing else on the system changes.

Start Asphodel from the app menu, or by running `asphodel` in a terminal.

### First start

The first start asks for two things:

- **Workspace folder** (default `~/Asphodel-data`). This is where Asphodel keeps its own files: settings, outputs (scans, confirmed IDs, fits, figures), logs and notes, with one sub-folder per experiment.
- **Name of the first experiment.**

Your input data does not have to be in the workspace. Connect data folders from anywhere on the home page ("Connect data folder…").

The workspace choice is saved in `~/.config/asphodel/config.json`. To use a different folder for one run, set `ASPHODEL_WORKSPACE=/path/to/folder`.

## Update

```bash
cd Asphodel
git pull
./install.sh        # installs any new packages
```

## Uninstall

```bash
./install.sh --uninstall
```

This removes `.venv/`, the `asphodel` command and the app-menu entry. Your workspace (settings, outputs) and the `Asphodel` folder are left alone. Delete them by hand if you no longer need them.

## Testing a fresh install (for developers)

```bash
tests/fresh_install_test.sh                                   # last commit of this repo
tests/fresh_install_test.sh git@github.com:ZacZN/Asphodel.git # what is on GitHub
tests/fresh_install_test.sh --try                             # then start it as the new user
```

The script clones the repo (committed files only) into a folder under `/tmp`. It then runs `install.sh` and `tests/smoke_test.py` as a new user would: in an empty fake home folder, with a clean environment and the system Python. Your own home folder, launcher, conda environment and workspace are not used or changed.

`tests/smoke_test.py` can also be run on its own with an installed Python (`.venv/bin/python tests/smoke_test.py`). It checks that everything compiles and imports, and that the home page, Theia, the Segrè chart, the GUI correction tool and the results log book start and serve their pages. It runs in a temporary workspace and opens no browser.

GitHub Actions runs the same fresh-install test on Ubuntu 22.04 and 24.04 for every push (`.github/workflows/install-test.yml`).
