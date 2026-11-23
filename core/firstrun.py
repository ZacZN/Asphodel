"""
First-run setup: choose the workspace folder and create the first experiment.

home.py calls ensure_workspace() before anything else.  It does nothing when a
workspace is set up and has at least one experiment.  Otherwise it asks, with
defaults filled in:

  * the workspace folder (default ~/Asphodel-data; core/workspace.py), where
    Asphodel keeps settings, outputs, logs and notes.  Input data does not
    have to be there: data folders are connected per experiment later.
  * the name of the first experiment (default "Experiment1")

How it asks: a small Tk window on a desktop; a prompt in the terminal when
there is no desktop but a terminal; with neither (or ASPHODEL_NONINTERACTIVE=1,
used by the tests) the defaults are taken without asking.

Order matters: this module must run before core.paths is imported, because
paths reads the workspace once, at import.
"""
import os
import sys

from . import workspace

DEFAULT_EXPERIMENT = "Experiment1"
ENV_NONINTERACTIVE = "ASPHODEL_NONINTERACTIVE"

INTRO = ("Asphodel needs a folder for its own files: settings, outputs (scans, "
         "confirmed IDs, fits, figures), logs and notes, one sub-folder per experiment.\n\n"
         "Your input data (spectra, spectrograms, ...) can stay where it is: you connect "
         "those folders on the home page later.")


# --------------------------------------------------------------------------- checks

def _has_experiments(ws):
    """True if the workspace has at least one experiment (a sub-folder or a
    registered profile)."""
    if not ws or not os.path.isdir(ws):
        return False
    try:
        if any(not n.startswith(".") and n != "logs" and os.path.isdir(os.path.join(ws, n))
               for n in os.listdir(ws)):
            return True
    except OSError:
        return False
    import json
    try:
        with open(os.path.join(ws, "asphodel_settings.json")) as fh:
            return bool((json.load(fh) or {}).get("profiles"))
    except (OSError, ValueError, AttributeError):
        return False


def valid_experiment_name(name):
    """None if `name` can be an experiment name, else the reason why not."""
    name = (name or "").strip()
    if not name:
        return "Give the experiment a name."
    if "/" in name or os.sep in name or name.startswith(".") or name == "logs":
        return "The name can't contain '/', start with '.' or be 'logs'."
    return None


# --------------------------------------------------------------------------- asking

def _ask_tk(ws_default, exp_default, ask_ws):
    """Tk dialog; (workspace, experiment) or None if cancelled. Raises if Tk
    can't open a window."""
    import tkinter as tk
    from tkinter import filedialog, ttk

    root = tk.Tk()
    root.title("Welcome to Asphodel")
    try:
        root.iconphoto(True, tk.PhotoImage(file=os.path.join(workspace.REPO_ROOT, "core", "icons",
                                                             "asphodel.png")))
    except Exception:
        pass
    result = {}
    frm = ttk.Frame(root, padding=16)
    frm.grid(sticky="nsew")
    ttk.Label(frm, text="Set Up Asphodel", font=("TkDefaultFont", 14, "bold")).grid(
        row=0, column=0, columnspan=3, sticky="w")
    ttk.Label(frm, text=INTRO if ask_ws else
              f"The workspace {ws_default} has no experiment yet.", wraplength=460,
              justify="left").grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 12))

    ws_var = tk.StringVar(value=ws_default)
    exp_var = tk.StringVar(value=exp_default)
    row = 2
    if ask_ws:
        ttk.Label(frm, text="Workspace Folder:").grid(row=row, column=0, sticky="w")
        ttk.Entry(frm, textvariable=ws_var, width=44).grid(row=row, column=1, sticky="ew", padx=6)

        def browse():
            start = ws_var.get()
            while start and not os.path.isdir(start):
                start = os.path.dirname(start)
            p = filedialog.askdirectory(parent=root, title="Workspace Folder",
                                        initialdir=start or os.path.expanduser("~"))
            if p:
                ws_var.set(p)
        ttk.Button(frm, text="Browse…", command=browse).grid(row=row, column=2)
        row += 1
    ttk.Label(frm, text="First Experiment:").grid(row=row, column=0, sticky="w", pady=(8, 0))
    ttk.Entry(frm, textvariable=exp_var, width=24).grid(row=row, column=1, sticky="w", padx=6,
                                                        pady=(8, 0))
    row += 1
    msg = ttk.Label(frm, text="", foreground="#b00020")
    msg.grid(row=row, column=0, columnspan=3, sticky="w", pady=(8, 0))
    row += 1

    def ok(*_):
        err = valid_experiment_name(exp_var.get())
        if ask_ws and not ws_var.get().strip():
            err = "Choose a workspace folder."
        if err:
            msg.config(text=err)
            return
        result["v"] = (ws_var.get().strip(), exp_var.get().strip())
        root.destroy()

    btns = ttk.Frame(frm)
    btns.grid(row=row, column=0, columnspan=3, sticky="e", pady=(12, 0))
    ttk.Button(btns, text="Cancel", command=root.destroy).pack(side="right")
    ttk.Button(btns, text="Create", command=ok).pack(side="right", padx=6)
    root.bind("<Return>", ok)
    root.bind("<Escape>", lambda *_: root.destroy())
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass
    root.mainloop()
    return result.get("v")


def _ask_terminal(ws_default, exp_default, ask_ws):
    print("\n=== Asphodel first-run setup ===\n")
    ws = ws_default
    if ask_ws:
        print(INTRO + "\n")
        ws = input(f"Workspace folder [{ws_default}]: ").strip() or ws_default
    while True:
        exp = input(f"First experiment [{exp_default}]: ").strip() or exp_default
        err = valid_experiment_name(exp)
        if not err:
            return ws, exp
        print(err)


def _interactive_mode():
    if os.environ.get(ENV_NONINTERACTIVE, "").strip() not in ("", "0"):
        return None
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return "tk"
    if sys.stdin and sys.stdin.isatty():
        return "terminal"
    return None


# --------------------------------------------------------------------------- main

def ensure_workspace():
    """Make sure a workspace with at least one experiment exists, asking the
    user if needed. Returns the workspace path; exits if the user cancels."""
    ws = workspace.workspace()
    if ws and _has_experiments(ws):
        return ws

    ask_ws = ws is None
    ws_default = ws or workspace.DEFAULT_WORKSPACE
    mode = _interactive_mode()
    answer = None
    if mode == "tk":
        try:
            answer = _ask_tk(ws_default, DEFAULT_EXPERIMENT, ask_ws)
        except Exception as e:                        # no Tk, or no usable display
            print(f"Asphodel setup: no setup window ({e}).", file=sys.stderr)
            mode = "terminal" if sys.stdin and sys.stdin.isatty() else None
            if mode is None:
                answer = (ws_default, DEFAULT_EXPERIMENT)
        else:
            if answer is None:
                sys.exit("Asphodel setup cancelled: nothing was created.")
    if mode == "terminal" and answer is None:
        try:
            answer = _ask_terminal(ws_default, DEFAULT_EXPERIMENT, ask_ws)
        except (EOFError, KeyboardInterrupt):
            sys.exit("\nAsphodel setup cancelled: nothing was created.")
    if answer is None:
        answer = (ws_default, DEFAULT_EXPERIMENT)

    ws_new, exp = answer
    if ask_ws or os.path.abspath(os.path.expanduser(ws_new)) != ws:
        if workspace.source() == "env":
            ws = os.path.abspath(os.path.expanduser(ws_new))
            os.makedirs(ws, exist_ok=True)
            os.environ[workspace.ENV_WORKSPACE] = ws
        else:
            ws = workspace.set_workspace(ws_new)
    if "core.paths" in sys.modules:
        raise RuntimeError("core.firstrun must run before core.paths is imported")
    from . import paths                               # reads the workspace now
    if not paths.list_experiments():
        paths.create_experiment(exp)
        paths.set_experiment(exp)
    print(f"Asphodel workspace: {ws}  (experiment: {paths.experiment()})")
    return ws


if __name__ == "__main__":
    ensure_workspace()
