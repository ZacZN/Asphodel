#!/usr/bin/env bash
# Install Asphodel for the current user (Linux).
#
#   ./install.sh                     install / update (run again after `git pull`)
#   ./install.sh --no-launcher       skip the app-menu launcher and the `asphodel` command
#   ./install.sh --separate-launcher if another copy of Asphodel already has the
#   ./install.sh --replace-launcher    launcher: add this one next to it / take it over
#                                      (without either option you are asked)
#   ./install.sh --uninstall         remove .venv/ and this copy's launcher and command
#                                    (the workspace and its data are kept)
#   PYTHON=python3.12 ./install.sh   use a particular Python
#
# What it does:
#   1. checks Python (3.9 or newer) with venv and Tk
#   2. creates .venv/ in this folder and installs requirements.txt into it
#   3. if other copies of Asphodel are installed, gives this one its own ports
#   4. adds the command ~/.local/bin/asphodel and the "Asphodel" app-menu launcher
#
# Every clone is its own installation: its environment (.venv/) and its own
# settings (.asphodel_local.json: workspace, chosen at the first start, and
# port offset) live in the clone. A second copy (e.g. a clone to test a fresh
# install) gets other ports, so it can be open next to your working copy, and
# its own "Asphodel (<folder>)" launcher and asphodel-<folder> command, unless
# you choose to replace the existing ones.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$REPO/.venv"
BIN_DIR="${XDG_BIN_HOME:-$HOME/.local/bin}"
APPS_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
PYTHON="${PYTHON:-python3}"
LAUNCHER=1
LAUNCHER_CHOICE=""          # "", separate, replace
NAME="$(basename "$REPO")"
SLUG="$(printf '%s' "$NAME" | tr 'A-Z' 'a-z' | tr -c 'a-z0-9._-' '_')"
MIN_PY="3.9"

say()  { printf '\033[1;33m==>\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31mError:\033[0m %s\n' "$*" >&2; exit 1; }

for arg in "$@"; do
    case "$arg" in
        --no-launcher) LAUNCHER=0 ;;
        --separate-launcher) LAUNCHER_CHOICE=separate ;;
        --replace-launcher) LAUNCHER_CHOICE=replace ;;
        --uninstall)
            say "Removing this copy's environment, launcher and command"
            for f in "$APPS_DIR"/asphodel*.desktop "$BIN_DIR"/asphodel*; do
                if [ -f "$f" ] && grep -qF "$REPO/home.py" "$f"; then rm -f "$f"; echo "  removed $f"; fi
            done
            if [ -d "$VENV" ]; then rm -rf "$VENV"; echo "  removed $VENV"; fi
            ws="$(sed -n 's/.*"workspace": *"\(.*\)".*/\1/p' "$REPO/.asphodel_local.json" "$REPO/.asphodel_workspace.json" 2>/dev/null | head -1 || true)"
            echo "Left alone: this folder${ws:+ and the workspace $ws}."
            echo "Delete them by hand if you no longer need them."
            exit 0 ;;
        -h|--help) sed -n '2,27p' "$0"; exit 0 ;;
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

# ---------------------------------------------------------------- 3. ports
# Other copies of Asphodel installed for this user (found through their
# launchers). If there are any, this copy gets its own ports so both can be
# open at the same time (core/workspace.py, core/ports.py).
others=()
for f in "$APPS_DIR"/asphodel*.desktop; do
    [ -f "$f" ] || continue
    p="$(sed -n 's/^Path=//p' "$f" | head -1)"
    if [ -n "$p" ] && [ "$p" != "$REPO" ] && [ -f "$p/home.py" ]; then others+=("$p"); fi
done
if [ ${#others[@]} -gt 0 ]; then
    off="$("$VENV/bin/python" -m core.workspace assign-port-offset "${others[@]}")"
    if [ "$off" != 0 ]; then
        say "Other copies of Asphodel found: this copy uses ports $((8050 + off))–$((8058 + off)), so it can run next to them"
    fi
fi

# ---------------------------------------------------------------- 4. launchers
# does a launcher / command file start another copy of Asphodel?
other_copy() { [ -f "$1" ] && ! grep -qF "$REPO/home.py" "$1"; }

if [ "$LAUNCHER" = 1 ]; then
    DESKTOP_FILE="$APPS_DIR/asphodel.desktop"
    WRAPPER="$BIN_DIR/asphodel"
    MENU_NAME="Asphodel"
    if other_copy "$DESKTOP_FILE" || other_copy "$WRAPPER"; then
        other="$(sed -n 's/^Path=//p' "$DESKTOP_FILE" 2>/dev/null || true)"
        echo
        echo "Another copy of Asphodel already has the \"Asphodel\" launcher and the asphodel command${other:+ ($other)}."
        if [ -z "$LAUNCHER_CHOICE" ]; then
            if [ -t 0 ]; then
                echo "  [s] add this copy separately, as \"Asphodel ($NAME)\" and asphodel-$SLUG  (default)"
                echo "  [r] replace them: they will start this copy instead"
                echo "  [n] no launcher for this copy"
                read -r -p "Choice [s/r/n]: " ans
                case "${ans:-s}" in
                    r|R) LAUNCHER_CHOICE=replace ;;
                    n|N) LAUNCHER=0 ;;
                    *)   LAUNCHER_CHOICE=separate ;;
                esac
            else
                LAUNCHER_CHOICE=separate
            fi
        fi
        if [ "$LAUNCHER_CHOICE" = separate ]; then
            DESKTOP_FILE="$APPS_DIR/asphodel-$SLUG.desktop"
            WRAPPER="$BIN_DIR/asphodel-$SLUG"
            MENU_NAME="Asphodel ($NAME)"
        fi
    fi
fi

if [ "$LAUNCHER" = 1 ]; then
    say "Adding the command and the app-menu launcher \"$MENU_NAME\""
    mkdir -p "$BIN_DIR"
    cat > "$WRAPPER" <<EOF
#!/bin/sh
# Asphodel launcher, written by $REPO/install.sh
exec "$VENV/bin/python" "$REPO/home.py" "\$@"
EOF
    chmod +x "$WRAPPER"
    echo "  $WRAPPER"
    "$VENV/bin/python" -m core.browser install "$DESKTOP_FILE" "$MENU_NAME" | sed 's/^/  /'
    case ":$PATH:" in
        *":$BIN_DIR:"*) ;;
        *) echo "  note: $BIN_DIR is not on your PATH; add it (e.g. in ~/.bashrc) to start Asphodel from a terminal" ;;
    esac
fi

say "Done."
if [ "$LAUNCHER" = 1 ]; then
    echo "Start Asphodel from the app menu (\"$MENU_NAME\"), or run:  $(basename "$WRAPPER")"
else
    echo "Start Asphodel with:  $VENV/bin/python $REPO/home.py"
fi
if grep -qs '"workspace"' "$REPO/.asphodel_local.json" "$REPO/.asphodel_workspace.json" || [ -d "$REPO/data" ]; then
    echo "This copy's workspace is already set up."
else
    echo "The first start asks where to keep the workspace (default ~/Asphodel-data)."
fi
