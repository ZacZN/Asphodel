#!/bin/bash
# Start a program with a conda environment activated, without `conda run`
# (which takes about 2 s before Python even starts).  Used by the desktop
# launcher when Asphodel runs in a conda env (core/browser.py launch_command):
#
#     bash core/launch_conda.sh <env prefix> <script.py> [args...]
#
# Sets what Asphodel needs from `conda activate`: the env's bin/ first on PATH
# and the CONDA_* variables.  The env's own activation scripts
# (etc/conda/activate.d: compilers, ROOT, ...) are skipped: Asphodel does not
# use them and they took about 0.4 s.  Set ASPHODEL_CONDA_ACTIVATE_D=1 to run
# them anyway (e.g. if a tool ever needs something they set up).
prefix="$1"; shift
export CONDA_PREFIX="$prefix"
export CONDA_DEFAULT_ENV="$(basename "$prefix")"
export CONDA_SHLVL=1
export PATH="$prefix/bin:$PATH"
if [ "${ASPHODEL_CONDA_ACTIVATE_D:-0}" = "1" ]; then
    for f in "$prefix"/etc/conda/activate.d/*.sh; do
        [ -r "$f" ] && . "$f" >/dev/null 2>&1
    done
fi
exec "$prefix/bin/python" "$@"
