"""
theia.py  –  Theia: spectrum viewer and batch scanner

Tabs: Viewer (one spectrum with the simulated ion frequencies overlaid; was
viewer.py), Scan, Scan Results and Further output.

The scanner loops over every 1-D Schottky spectrum (.npz, same format as viewer.py) in a
directory, detects peaks in the raw data, tracks the (drifting) reference peak
from spectrum to spectrum, simulates the NUBASE ion frequencies from that
reference and identifies every detected peak lying within ±x Hz of a simulated
ion.  All results go into a new SQLite database per run, which can be loaded
and searched in the "Scan Results" tab.

Usage
-----
    python theia/theia.py      (or start it from the Asphodel home page)
then open  http://127.0.0.1:8052/

The scan engine (peak detection, identification, settings, databases, the
scan worker) is in core/scan.py; this file is the GUI.

Output  (in the active experiment's data folder, see core/paths.py)
------
    data/<experiment>/outputs/scans/
      index.csv                               one row per run (human readable)
      scan_<YYYYmmdd_HHMMSS>_<hash>.sqlite    one database per run
      scan_<YYYYmmdd_HHMMSS>_<hash>.json      settings used for that run
    data/<experiment>/outputs/confirmed/      confirmed identifications (*.csv)
    data/<experiment>/settings/               scanner_config.json, viewer_config.json

<hash> is a hash of the scan settings, so runs with identical settings share
the same hash.

Peak detection, identification and the scan databases: see core/scan.py.

Code layout
-----------
    theia/theia.py          this file: starts Theia
    theia/gui/              the GUI, one module per tab or part:
      common.py             shared settings, styles, layout helpers
      app.py                the Dash app object, page CSS
      figures.py            spectrum figures
      info.py               ⓘ help buttons and texts
      corr_profiles.py      correction-profile dropdowns
      folder_browser.py     Browse… pop-up
      viewer.py             Viewer tab
      scan_tab.py           Scan tab
      logbook.py            "Save to Log Book" popout (Viewer, Scan Results)
      results.py            Scan Results tab (incl. confirmation mode, reference tracking)
      further_files.py      Further output: confirmation files, summing, PID settings
      further_fit.py        Further output: peak fitting, Search ions
      isocurve.py           Further output: isochronicity curve
      energy.py             Further output: energy calculation
      further_layout.py     Further output: page layout
      settings_menu.py      Settings menu, dark mode, Quit, settings-file warnings
      layout.py             the page (tab bar, all tabs), app.layout
    core/scan.py            the scan engine (no GUI)
Each module imports what it uses from the ones above it in this list; a few
functions import from a later module inside the function (marked).
"""

import logging
import os
import sys
import threading

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import browser, logcapture

# output also to data/logs/ and the page's Log panel (core/logcapture.py)
if __name__ == "__main__":
    logcapture.start("theia")

# importing the modules builds the page and registers every callback
import gui.common, gui.app, gui.figures, gui.info, gui.corr_profiles  # noqa: E401, F401
import gui.folder_browser, gui.logbook, gui.viewer, gui.scan_tab, gui.results  # noqa: E401, F401
import gui.further_files, gui.further_fit, gui.isocurve, gui.energy  # noqa: E401, F401
import gui.further_layout, gui.settings_menu, gui.layout  # noqa: E401, F401
from gui.app import app


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    # Hide the per-request access log lines; warnings and errors are still shown
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    url = "http://127.0.0.1:8052/"
    print(f"Starting Theia at {url}")
    # in the suite's Firefox window (core/browser.py), else the default browser
    threading.Timer(1.0, lambda: browser.open_url(url)).start()
    app.run(debug=False, host="127.0.0.1", port=8052)
