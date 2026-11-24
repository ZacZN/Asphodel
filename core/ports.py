"""
Ports of the Asphodel tools that are web pages (all on 127.0.0.1).

    from core import ports
    ports.port("theia")         # 8052, or 8152 etc. in a second copy

Each clone adds its port offset (core/workspace.py: 0 for your working copy,
100, 200, ... for further copies installed next to it), so two copies can be
open at the same time. The spectrogram picks a free port itself.
"""
from . import workspace

BASE = {
    "home": 8050,             # home.py (a free port if 8050+offset is taken)
    "theia": 8052,            # theia/theia.py
    "segre": 8054,            # reference/segre_chart.py
    "gui_correction": 8056,   # tartarus/gui_2nd_order_correction.py
    "logbook": 8058,          # reference/results_logbook.py
}


def offset():
    return workspace.port_offset()


def port(name):
    """The port of tool `name` in this clone."""
    return BASE[name] + offset()
