"""Native folder chooser for the browser-based tools.

The tools are web pages served from this computer, so a page cannot open the
desktop's folder dialog itself; the server opens it instead and returns the
chosen path.  Tries zenity (GNOME), kdialog (KDE), then a small Tk dialog in
its own process (so it never blocks a Dash/Flask thread's Tk state).

    path = pick_folder("Connect a data folder", initial="/home/me/data")
    # -> "/home/me/data/E0203/spectra_1d" or None if cancelled / no dialog available
"""
import os
import shutil
import subprocess
import sys

_TK_SNIPPET = r"""
import sys, tkinter as tk
from tkinter import filedialog
root = tk.Tk(); root.withdraw()
try:
    root.attributes('-topmost', True)
except Exception:
    pass
p = filedialog.askdirectory(title=sys.argv[1], initialdir=sys.argv[2] or None, mustexist=True)
print(p or "", end="")
"""


def available():
    """Name of the dialog that would be used, or None."""
    for cmd in ("zenity", "kdialog"):
        if shutil.which(cmd):
            return cmd
    try:
        import tkinter  # noqa: F401  (only checks that Tk is installed)
        return "tk"
    except ImportError:
        return None


def pick_folder(title="Choose a folder", initial=None, timeout=600):
    """Open the desktop's folder dialog; the chosen folder, or None."""
    initial = initial if initial and os.path.isdir(initial) else os.path.expanduser("~")
    tool = available()
    if tool == "zenity":
        cmd = ["zenity", "--file-selection", "--directory", f"--title={title}",
               f"--filename={os.path.join(initial, '')}"]
    elif tool == "kdialog":
        cmd = ["kdialog", "--title", title, "--getexistingdirectory", initial]
    elif tool == "tk":
        cmd = [sys.executable, "-c", _TK_SNIPPET, title, initial]
    else:
        raise RuntimeError("No folder dialog available (install zenity, or paste the path instead).")
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    path = (out.stdout or "").strip().splitlines()[-1:] if out.returncode == 0 else []
    path = path[0].strip() if path else ""
    return path if path and os.path.isdir(path) else None
