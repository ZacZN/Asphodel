# Asphodel

Analysis suite for Schottky data from the ESR storage ring at GSI: isotope identification from revolution frequencies, spectrograms and decay fits, spectrum summing and corrections, level schemes and a Segrè chart.

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

The workspace choice is saved inside the Asphodel folder, in `.asphodel_local.json` (git ignores it). To use a different folder for one run, set `ASPHODEL_WORKSPACE=/path/to/folder`.

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

This removes `.venv/` and this copy's command and app-menu entry. Your workspace (settings, outputs) and the `Asphodel` folder are left alone. Delete them by hand if you no longer need them.

## Testing a fresh install (for developers)

Every clone is its own installation: its Python environment (`.venv/`) and its workspace choice live inside the clone. So you can test a fresh install in your own account, next to your working copy, without changing it:

```bash
cd /tmp                       # anywhere outside your working copy
git clone https://github.com/ZacZN/Asphodel.git Asphodel-test
cd Asphodel-test
./install.sh
```

`install.sh` notices that your working copy already has the "Asphodel" launcher and the `asphodel` command. It asks what to do, and by default adds the test copy separately, as "Asphodel (Asphodel-test)" and `asphodel-asphodel-test`. It also gives the test copy its own ports (8150–8158 instead of 8050–8058), so both copies can be open at the same time. Start "Asphodel (Asphodel-test)" and you get the first-run setup exactly as a new user does, while your working copy stays open. The test copy's home page shows "Copy: Asphodel-test" under the version, and its browser tab is titled "Asphodel (Asphodel-test)". Both copies open their pages in the same Asphodel Firefox window.

To remove the test copy afterwards:

```bash
cd /tmp/Asphodel-test
./install.sh --uninstall      # removes its .venv/, launcher and command
cd .. && rm -rf Asphodel-test
rm -rf ~/Asphodel-data        # its workspace, if you used the default
```

`tests/smoke_test.py` is a quick automatic check of an installed copy (`.venv/bin/python tests/smoke_test.py`). It checks that everything compiles and imports, and that the home page, Theia, the Segrè chart, the GUI correction tool and the results log book start and serve their pages. It runs in a temporary workspace on its own ports (8950–8958) and opens no browser, so it also works while Asphodel is open.
