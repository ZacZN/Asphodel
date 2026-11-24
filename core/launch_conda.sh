#!/bin/bash
# Start a program with a conda environment activated, without `conda run`
# (which takes about 2 s before Python even starts).  Used by the desktop
# launcher when Asphodel runs in a conda env (core/browser.py launch_command):
#
#     bash core/launch_conda.sh <env prefix> <script.py> [args...]
#
# Does what `conda activate` does for a program: the env's bin/ first on PATH,
# the CONDA_* variables, and the env's activation scripts (etc/conda/activate.d,
# e.g. compilers, ROOT), so libraries and Qt find their files as in a terminal.
prefix="$1"; shift
export CONDA_PREFIX="$prefix"
export CONDA_DEFAULT_ENV="$(basename "$prefix")"
export CONDA_SHLVL=1
export PATH="$prefix/bin:$PATH"
for f in "$prefix"/etc/conda/activate.d/*.sh; do
    [ -r "$f" ] && . "$f" >/dev/null 2>&1
done
exec "$prefix/bin/python" "$@"
