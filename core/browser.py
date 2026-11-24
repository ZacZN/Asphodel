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
user_pref("apz.gtk.kinetic_scroll.enabled", false);
"""
# (the last line: no coasting after the fingers leave the touchpad, so a
# touchpad zoom in the plots stops when you let go; core/trackpad.py)


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
    try:
        with open(uj) as fh:
            have = fh.read()
    except OSError:
        have = ""
    # add the lines that are missing (a profile made by an older version);
    # Firefox reads user.js when the suite's Firefox starts
    missing = [ln for ln in _USER_JS.splitlines() if ln and ln not in have.splitlines()]
    if missing:
        with open(uj, "a") as fh:
            if have and not have.endswith("\n"):
                fh.write("\n")
            fh.write("\n".join(missing) + "\n")


def _running(path):
    """True while a Firefox instance is using the profile (its lock exists)."""
    return os.path.lexists(os.path.join(path, "lock"))


def no_browser():
    """True when ASPHODEL_NO_BROWSER is set (tests, servers): the tools then
    only print their address and open no browser."""
    return os.environ.get("ASPHODEL_NO_BROWSER", "").strip() not in ("", "0")


def open_url(url, new_window=False):
    """Open url in the suite's Firefox window (a new tab there, or a new
    window). Returns True if Firefox was used, False if it fell back to the
    default browser (or if no browser is to be opened, see no_browser())."""
    if no_browser():
        print(f"(ASPHODEL_NO_BROWSER set: not opening {url})")
        return False
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


def when_ready(port, action, timeout=60.0):
    """Run action() in a background thread as soon as 127.0.0.1:port accepts
    connections (checked every 50 ms), instead of after a fixed wait.  Call it
    just before app.run(): the page then opens the moment the server is up.
    Gives up silently after `timeout` seconds (the server did not start)."""
    import socket
    import threading
    import time

    def wait():
        end = time.time() + timeout
        while time.time() < end:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.2)
                if s.connect_ex(("127.0.0.1", port)) == 0:
                    action()
                    return
            time.sleep(0.05)

    threading.Thread(target=wait, daemon=True).start()


# Set by the home page when it has already opened this tool's tab (showing
# "Starting <tool>…", which switches to the tool once its server answers):
# the tool then must not open a second tab.  Read once and removed, so
# programs this tool starts in turn open their own tabs as usual.
ENV_TAB_OPENED = "ASPHODEL_TAB_OPENED"
_TAB_OPENED = os.environ.pop(ENV_TAB_OPENED, "").strip() == "1"


def tab_opened_by_launcher():
    """True if the home page already opened this tool's browser tab."""
    return _TAB_OPENED


def open_when_ready(url, port, new_window=False):
    """open_url(url) as soon as the tool's server on `port` answers (nothing
    if the home page already opened the tab, see tab_opened_by_launcher)."""
    if not _TAB_OPENED:
        when_ready(port, lambda: open_url(url, new_window=new_window))


def _conda_exe(prefix):
    """The conda (or mamba) command that manages the env at prefix, or None."""
    for exe in (os.environ.get("CONDA_EXE"),
                os.path.join(os.path.dirname(os.path.dirname(prefix)), "bin", "conda"),
                shutil.which("conda")):
        if exe and os.path.isfile(exe) and os.access(exe, os.X_OK):
            return exe
    return None


def launch_command(python=None):
    """Command line that starts home.py. In a conda env the env is activated
    first (`conda run -p <env>`): a desktop launcher does not activate it, and
    without that Qt cannot find its platform plugins, so matplotlib windows
    such as the decay fit abort before they open."""
    python = python or sys.executable
    home = os.path.join(REPO, "home.py")
    prefix = os.path.dirname(os.path.dirname(os.path.abspath(python)))
    conda = _conda_exe(prefix) if os.path.isdir(os.path.join(prefix, "conda-meta")) else None
    if conda:
        return [conda, "run", "-p", prefix, "--no-capture-output", "python", home]
    return [python, home]


def _desktop_exec(args):
    """Quote arguments for a .desktop Exec line."""
    def q(a):
        if not any(c in a for c in ' \t\n"\'\\><~|&;$*?#()`'):
            return a
        return '"' + a.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`").replace("$", "\\$") + '"'
    return " ".join(q(a) for a in args)


def install_desktop_file(python=None, path=None, name="Asphodel"):
    """Write ~/.local/share/applications/asphodel.desktop (or `path`): an
    app-menu launcher called `name` (starts this clone's home.py, with its
    conda env activated) whose StartupWMClass matches the window, so the dock
    shows the asphodel icon for the suite's Firefox window."""
    path = path or DESKTOP_FILE
    text = (
        "[Desktop Entry]\n"
        "Type=Application\n"
        f"Name={name}\n"
        "Comment=ESR analysis suite\n"
        f"Exec={_desktop_exec(launch_command(python))}\n"
        f"Path={REPO}\n"
        f"Icon={ICON_PNG}\n"
        "Terminal=false\n"
        f"StartupWMClass={APP_ID}\n"
        "Categories=Science;Education;\n"
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)
    try:
        subprocess.run(["update-desktop-database", os.path.dirname(path)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
    except (OSError, subprocess.SubprocessError):
        pass
    return path


if __name__ == "__main__":
    if sys.argv[1:2] == ["install"]:
        # python -m core.browser install [<desktop file> [<menu name>]]
        args = sys.argv[2:]
        print("Wrote", install_desktop_file(path=args[0] if args else None,
                                            name=args[1] if len(args) > 1 else "Asphodel"))
        print("Launcher runs:", _desktop_exec(launch_command()))
        print("Log out and back in (or restart GNOME Shell) if the icon does not show at once.")
    else:
        print(__doc__)
