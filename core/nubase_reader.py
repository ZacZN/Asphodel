"""nubase_reader.py — self-contained NuBase2020 reader for the Asphodel suite (core).

Provides all ion-database and spectrum-loading helpers that viewer.py needs,
with no dependency on external personal modules.  The NuBase data file
(nubase_4.mas20) is expected to live in the same directory as this script.
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path to the bundled NuBase2020 data file
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_NUBASE_FILE = os.path.join(_HERE, "nubase_4.mas20")


# ---------------------------------------------------------------------------
# NuBase parsing
# ---------------------------------------------------------------------------

def get_base_dataframe():
    """Read and format the NuBase2020 data in preparation for calculating m/q.

    Returns
    -------
    base_dataframe : pandas.DataFrame
        NuBase2020 as a pandas DataFrame.
    """
    # Column names matching the fixed-width format of nubase_4.mas20
    col_names = [
        "AAA", "ZZZi", "Element", "Isomer_Flag", "m.e.", "m.e._unc",
        "E_x", "E_x_unc", "Orig", "Isom_Unc", "Isom_Inv", "T",
        "T_unit", "T_unc", "Jpi", "Ensdf_Year", "Discovery", "BR"
    ]

    # 0-indexed column slices
    col_specs = [
        (0, 3), (4, 8), (11, 16), (16, 17), (18, 31), (31, 42),
        (42, 54), (54, 65), (65, 67), (67, 68), (68, 69), (69, 78),
        (78, 80), (81, 88), (88, 102), (102, 104), (114, 118), (119, 209)
    ]

    df = pd.read_fwf(
        _NUBASE_FILE,
        colspecs=col_specs,
        names=col_names,
        skiprows=25,
    )

    # Keep only the columns we need downstream
    compact_df = df[[
        "AAA", "ZZZi", "Element", "Isomer_Flag",
        "m.e.", "m.e._unc",
        "E_x", "E_x_unc",
        "T", "T_unit", "T_unc",
        "BR",
    ]].copy()

    # Record whether m.e. was theoretical (#), then strip # and convert to float
    col_str = compact_df["m.e."].astype(str)
    compact_df["m.e. sys?"] = col_str.str.contains("#")
    compact_df["m.e."] = col_str.str.replace("#", "", regex=False)

    col_str = compact_df["m.e._unc"].astype(str)
    compact_df["m.e._unc"] = col_str.str.replace("#", "", regex=False)

    compact_df["m.e."]     = pd.to_numeric(compact_df["m.e."],     errors="coerce")
    compact_df["m.e._unc"] = pd.to_numeric(compact_df["m.e._unc"], errors="coerce")

    # Strip the isomer character from ZZZi to get plain Z, then convert
    compact_df["ZZZi"] = df["ZZZi"].astype(str).str.strip().str[:-1]
    compact_df["ZZZi"] = pd.to_numeric(compact_df["ZZZi"], errors="coerce")
    compact_df["AAA"]  = pd.to_numeric(compact_df["AAA"],  errors="coerce")

    # Neutron number column
    compact_df.insert(2, "NNN", compact_df["AAA"] - compact_df["ZZZi"])

    # Nuclear mass in unified atomic mass units
    ELECTRON_MASS_U = 0.00054858
    compact_df["nuclear_mass_u"] = (
        compact_df["AAA"]
        + compact_df["m.e."] / 931494.1
        - compact_df["ZZZi"] * ELECTRON_MASS_U
    )
    compact_df["nuclear_mass_unc_u"] = compact_df["m.e._unc"] / 931494.1

    return compact_df


def df_nubase(electrons):
    """Return a NuBase2020 DataFrame with m/q calculated for a given charge state.

    Parameters
    ----------
    electrons : int
        Number of bound electrons (0 = fully stripped).

    Returns
    -------
    pandas.DataFrame
    """
    base_df = get_base_dataframe()
    ELECTRON_MASS_U = 0.00054858
    q = base_df["ZZZi"] - electrons
    base_df["m/q"]     = (base_df["nuclear_mass_u"]     + ELECTRON_MASS_U * electrons) / q
    base_df["m/q_unc"] =  base_df["nuclear_mass_unc_u"] / q
    return base_df


def rect_restrict(df, z_lims, n_lims):
    """Select a rectangular region of the nuclear chart.

    Parameters
    ----------
    df : pandas.DataFrame
    z_lims : (int, int)  — (Z_min, Z_max)
    n_lims : (int, int)  — (N_min, N_max)

    Returns
    -------
    pandas.DataFrame
    """
    mask = (
        (df["ZZZi"] >= z_lims[0]) & (df["ZZZi"] <= z_lims[1]) &
        (df["NNN"]  >= n_lims[0]) & (df["NNN"]  <= n_lims[1])
    )
    return df[mask]


def diag_restrict(df, el1, el2):
    """Remove nuclides above the diagonal line connecting two reference nuclei.

    Parameters
    ----------
    df : pandas.DataFrame
    el1 : (int, int)  — (A, Z) of the first reference nucleus
    el2 : (int, int)  — (A, Z) of the second reference nucleus

    Returns
    -------
    pandas.DataFrame
    """
    A1, Z1 = el1;  N1 = A1 - Z1
    A2, Z2 = el2;  N2 = A2 - Z2
    gradient  = (Z2 - Z1) / (N2 - N1)
    intercept = Z2 - gradient * N2
    return df[df["ZZZi"] < gradient * df["NNN"] + intercept]


def index_by_name(df, electrons):
    """Set the DataFrame index to ion names (e.g. ``204Au79+``, ``m186Hf72+``).

    Parameters
    ----------
    df : pandas.DataFrame
    electrons : int

    Returns
    -------
    pandas.DataFrame
    """
    df = df.copy()
    df.ZZZi = df.ZZZi.astype(int)
    df.NNN  = df.NNN.astype(int)

    has_isomer = df["Isomer_Flag"].notna()
    names = np.where(
        has_isomer,
        df["Isomer_Flag"] + df["Element"] + (df["ZZZi"] - electrons).astype(str) + "+",
        df["Element"]                      + (df["ZZZi"] - electrons).astype(str) + "+",
    )
    df.index = names
    return df


# ---------------------------------------------------------------------------
# Spectrum loading
# ---------------------------------------------------------------------------

def unpack_npz(filepath, namespace=None, verbose=False):
    """Load an .npz file and return a dict of its arrays.

    Parameters
    ----------
    filepath : str
        Path to the .npz file.
    namespace : dict, optional
        If provided, arrays are also injected into this namespace.
    verbose : bool
        Print array names, shapes, and dtypes when True.

    Returns
    -------
    dict
        Mapping of array name → numpy array.
    """
    data = np.load(filepath)
    arrays = {key: data[key] for key in data.files}

    if namespace is not None:
        namespace.update(arrays)

    if verbose:
        for key, arr in arrays.items():
            print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")

    return arrays
