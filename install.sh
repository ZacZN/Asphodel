#!/usr/bin/env bash
# Install Asphodel for the current user (Linux).
#
#   ./install.sh                 install / update (run again after `git pull`)
#   ./install.sh --no-launcher   skip the app-menu launcher and the `asphodel` command
#   ./install.sh --uninstall     remove .venv/, the launcher and the command
#                                (your workspace and its data are kept)
#   PYTHON=python3.12 ./install.sh   use a particular Python
#
# What it does:
#   1. checks Python (3.9 or newer) with venv and Tk
#   2. creates .venv/ in this folder and installs requirements.txt into it
#   3. adds the command ~/.local/bin/asphodel and the "Asphodel" app-menu launcher
# Nothing outside this folder, ~/.local/bin/asphodel and
# ~/.local/share/applications/asphodel.desktop is changed. The workspace
# (settings, outputs) is chosen the first time Asphodel starts.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$REPO/.venv"
BIN_DIR="${XDG_BIN_HOME:-$HOME/.local/bin}"
WRAPPER="$BIN_DIR/asphodel"
DESKTOP_FILE="${XDG_DATA_HOME:-$HOME/.local/share}/applications/asphodel.desktop"
PYTHON="${PYTHON:-python3}"
LAUNCHER=1
MIN_PY="3.9"

say()  { printf '\033[1;33m==>\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31mError:\033[0m %s\n' "$*" >&2; exit 1; }

for arg in "$@"; do
    case "$arg" in
        --no-launcher) LAUNCHER=0 ;;
        --uninstall)
            say "Removing Asphodel's environment and launchers"
            if [ -f "$WRAPPER" ] && grep -qF "$REPO" "$WRAPPER"; then rm -f "$WRAPPER"; echo "  removed $WRAPPER"; fi
            if [ -f "$DESKTOP_FILE" ] && grep -qF "$REPO" "$DESKTOP_FILE"; then rm -f "$DESKTOP_FILE"; echo "  removed $DESKTOP_FILE"; fi
            if [ -d "$VENV" ]; then rm -rf "$VENV"; echo "  removed $VENV"; fi
            echo "Your workspace (see ~/.config/asphodel/config.json) and this folder were left alone."
            exit 0 ;;
        -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
        *) fail "unknown option $arg (see ./install.sh --help)" ;;
    esac
done

[ "$(uname -s)" = "Linux" ] || fail "only Linux is supported for now."
cd "$REPO"

# ---------------------------------------------------------------- 1. Python
say "Checking Python"
command -v "$PYTHON" >/dev/null || fail "$PYTHON not found. Install Python $MIN_PY or newer (e.g. sudo apt install python3)."
"$PYTHON" -c "import sys; sys.exit(sys.version_info < (${MIN_PY/./, }))" \
    || fail "Python $MIN_PY or newer is needed; $PYTHON is $("$PYTHON" -V 2>&1)."
echo "  $("$PYTHON" -V 2>&1) at $(command -v "$PYTHON")"

missing=()
"$PYTHON" -c "import venv, ensurepip" 2>/dev/null || missing+=("python3-venv")
"$PYTHON" -c "import tkinter" 2>/dev/null || missing+=("python3-tk")
if [ ${#missing[@]} -gt 0 ]; then
    fail "missing system packages: ${missing[*]}
       On Ubuntu/Debian:  sudo apt install ${missing[*]}
       (on Fedora: python3-tkinter; conda Pythons already include both)"
fi
command -v zenity >/dev/null || command -v kdialog >/dev/null \
    || echo "  note: zenity not found; folder dialogs fall back to a plain Tk dialog (sudo apt install zenity)"
command -v firefox >/dev/null || command -v firefox-esr >/dev/null \
    || echo "  note: Firefox not found; the pages open in your default browser instead of their own window"

# ---------------------------------------------------------------- 2. environment
if [ -x "$VENV/bin/python" ] && ! "$VENV/bin/python" -c "pass" 2>/dev/null; then
    say "Existing .venv is broken; recreating it"; rm -rf "$VENV"
fi
if [ ! -x "$VENV/bin/python" ]; then
    say "Creating the Python environment in .venv/"
    "$PYTHON" -m venv "$VENV"
fi
say "Installing Python packages (this can take a few minutes the first time)"
"$VENV/bin/python" -m pip install --disable-pip-version-check -q --upgrade pip
"$VENV/bin/python" -m pip install --disable-pip-version-check -q -r requirements.txt

say "Checking the installation"
"$VENV/bin/python" - <<'PY'
import importlib, sys
mods = ["dash", "dash_bootstrap_components", "plotly", "flask", "numpy", "scipy", "pandas",
        "matplotlib", "pybaselines", "openpyxl", "odf", "tkinter"]
bad = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        bad.append(f"{m}: {e}")
sys.path.insert(0, ".")
try:
    from core import version
    print("  Asphodel", version.version())
except Exception as e:
    bad.append(f"core: {e}")
if bad:
    sys.exit("  failed imports:\n    " + "\n    ".join(bad))
print("  all packages import")
PY

# ---------------------------------------------------------------- 3. launchers
if [ "$LAUNCHER" = 1 ]; then
    say "Adding the asphodel command and the app-menu launcher"
    mkdir -p "$BIN_DIR"
    cat > "$WRAPPER" <<EOF
#!/bin/sh
# Asphodel launcher, written by $REPO/install.sh
exec "$VENV/bin/python" "$REPO/home.py" "\$@"
EOF
    chmod +x "$WRAPPER"
    echo "  $WRAPPER"
    "$VENV/bin/python" -m core.browser install | sed 's/^/  /'
    case ":$PATH:" in
        *":$BIN_DIR:"*) ;;
        *) echo "  note: $BIN_DIR is not on your PATH; add it (e.g. in ~/.bashrc) to type 'asphodel' in a terminal" ;;
    esac
fi

say "Done."
if [ "$LAUNCHER" = 1 ]; then
    echo "Start Asphodel from the app menu, or run:  asphodel"
else
    echo "Start Asphodel with:  $VENV/bin/python $REPO/home.py"
fi
echo "The first start asks where to keep the workspace (default ~/Asphodel-data)."
