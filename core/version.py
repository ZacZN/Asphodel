"""Asphodel version number: MAJOR.MINOR.PATCH

PATCH goes up by one with every git commit, counted from BASE_COMMIT (the
last commit before this numbering started), so nothing has to be edited
by hand to bump it:

    0.1.0  the commit that added this file (and the work before committing it)
    0.1.1  the next commit, and so on

To start a new minor/major series (e.g. 0.2.0), set MAJOR_MINOR to "0.2"
and BASE_COMMIT to the hash of the current last commit (`git rev-parse HEAD`),
then commit that change: it becomes 0.2.0.
"""
import os
import subprocess

MAJOR_MINOR = "0.1"
BASE_COMMIT = "e447f061e75f2a4b9a5d28829de52e2f12d18b1c"

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def commits_since_base():
    """Number of commits after BASE_COMMIT, or None if git can't tell."""
    try:
        out = subprocess.run(["git", "-C", REPO, "rev-list", "--count", f"{BASE_COMMIT}..HEAD"],
                             capture_output=True, text=True, timeout=5)
        return int(out.stdout.strip()) if out.returncode == 0 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def version():
    """'0.1.N' (N = commits after the one that added this file), or '0.1'
    when git is not available."""
    n = commits_since_base()
    if n is None:
        return MAJOR_MINOR
    return f"{MAJOR_MINOR}.{max(n - 1, 0)}"



def _git(*args):
    out = subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True, timeout=5)
    if out.returncode != 0:
        raise OSError(out.stderr.strip())
    return out.stdout


def changelog(n=20):
    """The last n commits, newest first: [{"version": "0.1.3" or None,
    "date": "26 Oct 2026", "subject": "..."}]. Commits from before the
    numbering started have no version. [] if git is not available."""
    try:
        log = _git("log", f"-n{n}", "--date=format:%d %b %Y", "--format=%H%x1f%ad%x1f%s")
        after_base = set(_git("rev-list", f"{BASE_COMMIT}..HEAD").split())
        out = []
        for line in log.splitlines():
            h, date, subject = (line.split("\x1f") + ["", ""])[:3]
            ver = None
            if h in after_base:
                k = int(_git("rev-list", "--count", f"{BASE_COMMIT}..{h}").strip())
                ver = f"{MAJOR_MINOR}.{max(k - 1, 0)}"
            out.append({"version": ver, "date": date, "subject": subject})
        return out
    except (OSError, ValueError, subprocess.SubprocessError):
        return []


__version__ = version()
