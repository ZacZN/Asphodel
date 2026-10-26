"""Open the Asphodel pages in their own Firefox window.

The suite runs in a separate Firefox instance with its own profile and the
window name/class "asphodel". The desktop (dock, Alt-Tab, window list) then
treats it as a separate app and shows the asphodel icon instead of Firefox's,
provided the launcher from `install_desktop_file()` is installed:

    python -m core.browser install      # once, from the repo folder

Your normal Firefox (and its profile) is not touched. Without Firefox the
pages open in the default browser as before.
"""
import os
import shutil
import subprocess
import sys
import webbrowser

APP_ID = "asphodel"                       # window name/class (= Wayland app_id)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ICON_PNG = os.path.join(REPO, "core", "icons", "asphodel.png")
DESKTOP_FILE = os.path.expanduser(f"~/.local/share/applications/{APP_ID}.desktop")

# Firefox preferences for the suite's profile: no first-run / welcome pages,
# no default-browser question, no warning when the pages close their tabs.
_USER_JS = """\
user_pref("browser.aboutwelcome.enabled", false);
user_pref("trailhead.firstrun.didSeeAboutWelcome", true);
user_pref("browser.startup.homepage_override.mstone", "ignore");
user_pref("startup.homepage_welcome_url", "");
user_pref("startup.homepage_welcome_url.additional", "");
user_pref("browser.startup.page", 0);
user_pref("browser.shell.checkDefaultBrowser", false);
user_pref("datareporting.policy.dataSubmissionPolicyBypassNotification", true);
user_pref("toolkit.telemetry.reportingpolicy.firstRun", false);
user_pref("browser.tabs.warnOnClose", false);
user_pref("browser.warnOnQuit", false);
user_pref("browser.sessionstore.resume_from_crash", false);
"""


def firefox():
    """Path of the firefox command, or None."""
    for cmd in ("firefox", "firefox-esr"):
        exe = shutil.which(cmd)
        if exe:
            return exe
    return None


def _is_snap(exe):
    return os.path.realpath(exe).startswith("/snap/") or os.path.isdir("/snap/firefox")


def profile_dir(exe):
    """The suite's own Firefox profile. Snap Firefox (Ubuntu's default) may
    only use folders under ~/snap/firefox/, so it goes there in that case."""
    if _is_snap(exe):
        return os.path.expanduser("~/snap/firefox/common/asphodel-profile")
    return os.path.expanduser("~/.local/share/asphodel/firefox-profile")


def _prepare_profile(path):
    os.makedirs(path, exist_ok=True)
    uj = os.path.join(path, "user.js")
    if not os.path.isfile(uj):
        with open(uj, "w") as fh:
            fh.write(_USER_JS)


def _running(path):
    """True while a Firefox instance is using the profile (its lock exists)."""
    return os.path.lexists(os.path.join(path, "lock"))


def open_url(url, new_window=False):
    """Open url in the suite's Firefox window (a new tab there, or a new
    window). Returns True if Firefox was used, False if it fell back to the
    default browser."""
    exe = firefox()
    if not exe:
        (webbrowser.open_new if new_window else webbrowser.open)(url)
        return False
    prof = profile_dir(exe)
    try:
        _prepare_profile(prof)
    except OSError:
        webbrowser.open(url)
        return False
    cmd = [exe, "--name", APP_ID, "--class", APP_ID, "--profile", prof]
    if not _running(prof):
        # start the suite's instance even if your normal Firefox is open;
        # once it runs, later calls hand their URL to it instead
        cmd.append("--new-instance")
    cmd += ["--new-window" if new_window else "--new-tab", url]
    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
        return True
    except OSError:
        webbrowser.open(url)
        return False


def install_desktop_file(python=None):
    """Write ~/.local/share/applications/asphodel.desktop: an 'Asphodel'
    launcher (starts home.py) whose StartupWMClass matches the window, so the
    dock shows the asphodel icon for the suite's Firefox window."""
    python = python or sys.executable
    text = (
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=Asphodel\n"
        "Comment=ESR analysis suite\n"
        f"Exec={python} {os.path.join(REPO, 'home.py')}\n"
        f"Path={REPO}\n"
        f"Icon={ICON_PNG}\n"
        "Terminal=false\n"
        f"StartupWMClass={APP_ID}\n"
        "Categories=Science;Education;\n"
    )
    os.makedirs(os.path.dirname(DESKTOP_FILE), exist_ok=True)
    with open(DESKTOP_FILE, "w") as fh:
        fh.write(text)
    try:
        subprocess.run(["update-desktop-database", os.path.dirname(DESKTOP_FILE)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
    except (OSError, subprocess.SubprocessError):
        pass
    return DESKTOP_FILE


if __name__ == "__main__":
    if sys.argv[1:] == ["install"]:
        print("Wrote", install_desktop_file())
        print("Log out and back in (or restart GNOME Shell) if the icon does not show at once.")
    else:
        print(__doc__)
