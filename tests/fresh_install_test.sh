#!/usr/bin/env bash
# Install Asphodel the way a new user would, without touching your own setup.
#
#   tests/fresh_install_test.sh                      # clone of this repo's last commit
#   tests/fresh_install_test.sh git@github.com:ZacZN/Asphodel.git    # what is on GitHub
#   tests/fresh_install_test.sh --branch WIP         # another branch
#   tests/fresh_install_test.sh --try                # afterwards, start Asphodel as that user
#   tests/fresh_install_test.sh --keep               # keep the test folder to look at
#   tests/fresh_install_test.sh --python /usr/bin/python3.10
#
# How it pretends to be a new user:
#   * git clone, so only committed files are there (uncommitted or git-ignored
#     files such as data/ are missing, exactly as for someone else)
#   * a fake, empty home folder (HOME=<test>/home): no ~/.config/asphodel,
#     no launcher, no pip cache, no Firefox profile
#   * a clean environment (env -i) with only the system PATH, so your conda
#     environment and shell settings are not used. The system python3 is used
#     unless --python is given.
# Then it runs ./install.sh and tests/smoke_test.py. With --try it then starts
# Asphodel as that user, so you see the first-run window and can click around
# (close it with the Quit button).
#
# Your own home folder, launcher and workspace are never touched; everything
# lives in a folder under /tmp, removed at the end unless --keep or --try.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="$HERE"
BRANCH=""
KEEP=0
TRY=0
PY="/usr/bin/python3"

while [ $# -gt 0 ]; do
    case "$1" in
        --branch) BRANCH="$2"; shift ;;
        --keep) KEEP=1 ;;
        --try) TRY=1; KEEP=1 ;;
        --python) PY="$2"; shift ;;
        -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
        -*) echo "unknown option $1" >&2; exit 2 ;;
        *) SOURCE="$1" ;;
    esac
    shift
done

say() { printf '\n\033[1;33m==== %s\033[0m\n' "$*"; }

if [ "$SOURCE" = "$HERE" ] && [ -n "$(git -C "$HERE" status --porcelain --untracked-files=no)" ]; then
    echo "Note: this repo has uncommitted changes; the test uses the last commit only."
fi

TEST="$(mktemp -d /tmp/asphodel_fresh_XXXX)"
FAKE_HOME="$TEST/home"
mkdir -p "$FAKE_HOME"
cleanup() { if [ "$KEEP" = 0 ]; then rm -rf "$TEST"; fi; }
trap cleanup EXIT

say "Cloning $SOURCE${BRANCH:+ (branch $BRANCH)}"
# (cloned with your real HOME, so your SSH keys work for a GitHub URL)
git clone -q ${BRANCH:+--branch "$BRANCH"} "$SOURCE" "$TEST/Asphodel"
git -C "$TEST/Asphodel" log -1 --format='  commit %h  %s  (%cd)' --date=short

# a clean environment, like a new user's login
CLEAN_ENV=(env -i HOME="$FAKE_HOME" USER="${USER:-user}" LANG="${LANG:-C.UTF-8}"
           PATH="$FAKE_HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"
           TERM="${TERM:-xterm}" PYTHON="$PY")
# (display, plus network proxy settings so pip works behind a proxy)
for v in DISPLAY WAYLAND_DISPLAY XDG_RUNTIME_DIR DBUS_SESSION_BUS_ADDRESS XAUTHORITY \
         HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy \
         PIP_INDEX_URL PIP_CONFIG_FILE PIP_CERT REQUESTS_CA_BUNDLE SSL_CERT_FILE; do
    if [ -n "${!v:-}" ]; then CLEAN_ENV+=("$v=${!v}"); fi
done
# keep X11 access working after HOME changes
if [ -z "${XAUTHORITY:-}" ] && [ -f "$HOME/.Xauthority" ]; then CLEAN_ENV+=("XAUTHORITY=$HOME/.Xauthority"); fi

say "Running ./install.sh as a new user (HOME=$FAKE_HOME)"
"${CLEAN_ENV[@]}" bash -c "cd '$TEST/Asphodel' && ./install.sh"

say "Smoke test"
status=0
"${CLEAN_ENV[@]}" "$TEST/Asphodel/.venv/bin/python" "$TEST/Asphodel/tests/smoke_test.py" || status=$?

if [ "$TRY" = 1 ]; then
    say "Starting Asphodel as the new user (Quit button on the home page to finish)"
    "${CLEAN_ENV[@]}" "$FAKE_HOME/.local/bin/asphodel" || true
fi

say "Result"
if [ "$status" = 0 ]; then echo "Fresh install OK."; else echo "Fresh install had FAILURES (see above)."; fi
if [ "$KEEP" = 1 ]; then
    echo "Test folder kept: $TEST   (remove with: rm -rf $TEST)"
fi
exit "$status"
