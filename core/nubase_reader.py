"""nubase_reader.py — self-contained NuBase2020 reader for the Asphodel suite (core).

Provides all ion-database and spectrum-loading helpers that viewer.py needs,
with no dependency on external personal modules.  The NuBase data file
(nubase_4.mas20) and the NIST ionization energies
(nist_ionization_energies.csv) are expected to live in the same directory as
this script.

Ion masses include the electron binding energies:

    m_ion(Z, q) = M_atom - q m_e + S(Z, q) / c^2

M_atom from the NUBASE mass excess, S(Z, q) = the sum of the first q
successive ionization energies of element Z (NIST ASD), i.e. the binding
energy of the q electrons removed from the neutral atom.  q = Z gives the
bare nucleus (S = total electron binding energy of the atom).
"""

import csv
import os
import re
from functools import lru_cache

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths to the bundled data files
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_NUBASE_FILE = os.path.join(_HERE, "nubase_4.mas20")
_IE_FILE = os.path.join(_HERE, "nist_ionization_energies.csv")

# ---------------------------------------------------------------------------
# Constants (AME2020: Wang et al., Chin. Phys. C 45, 030003 (2021))
# ---------------------------------------------------------------------------
AMU_KEV = 931494.10242               # 1 u in keV
ELECTRON_MASS_U = 5.48579909065e-4   # electron mass in u

# Short name of the mass model above, stored with fits that depend on it
# (e.g. 2nd-order correction profiles); change it if the mass calculation changes
MASS_MODEL = "NUBASE2020 + NIST electron binding"


# ---------------------------------------------------------------------------
# Electron binding energies (NIST ASD ionization energies)
# ---------------------------------------------------------------------------

def _nist_cell(text):
    """NIST CSV cell '="13.598"' -> '13.598' (Excel protection stripped)."""
    text = (text or "").strip()
    if text.startswith("="):
        text = text[1:]
    return text.strip('"').strip()


@lru_cache(maxsize=1)
def ionization_energies():
    """NIST ASD ionization energies as a DataFrame with one row per (Z, q):
    Z, q (charge of the ion the electron is removed from), IE_eV, IE_unc_eV
    (0 where NIST gives none), kind ('measured', 'theoretical' for values in
    ( ), 'semi-empirical' for values in [ ]).  Rows without a value are left out."""
    rows = []
    with open(_IE_FILE, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = [h.strip() for h in next(reader)]
        col = {name: i for i, name in enumerate(header)}
        i_z, i_q = col["At. num"], col["Ion Charge"]
        i_pre, i_ie = col["Prefix"], col["Ionization Energy (b) (eV)"]
        i_unc = col["Uncertainty (c) (eV)"]
        for r in reader:
            if len(r) <= i_unc:
                continue                       # notes at the end of the file
            cells = [_nist_cell(c) for c in r]
            try:
                z, q = int(cells[i_z]), int(cells[i_q].replace("+", ""))
            except ValueError:
                continue
            ie = re.sub(r"[^0-9.eE+-]", "", cells[i_ie])
            if not ie:
                continue                       # unknown value
            try:
                unc = float(re.sub(r"[^0-9.eE+-]", "", cells[i_unc]) or 0.0)
            except ValueError:
                unc = 0.0
            kind = {"(": "theoretical", "[": "semi-empirical"}.get(cells[i_pre], "measured")
            rows.append((z, q, float(ie), unc, kind))
    return pd.DataFrame(rows, columns=["Z", "q", "IE_eV", "IE_unc_eV", "kind"])


@lru_cache(maxsize=1)
def _removed_binding_table():
    """Arrays S[Z, q] and S_unc[Z, q] in eV for 0 <= q <= Z: binding energy of
    the q electrons removed from neutral Z (sum of the first q ionization
    energies; uncertainties added in quadrature).  NaN where an ionization
    energy is missing."""
    ie = ionization_energies()
    z_max = int(ie["Z"].max())
    e = np.full((z_max + 1, z_max + 1), np.nan)
    u = np.full((z_max + 1, z_max + 1), np.nan)
    for z, q, val, unc in ie[["Z", "q", "IE_eV", "IE_unc_eV"]].itertuples(index=False):
        if 0 <= q < z <= z_max:
            e[z, q], u[z, q] = val, unc
    S = np.full((z_max + 1, z_max + 2), np.nan)
    S_unc = np.full_like(S, np.nan)
    S[:, 0] = S_unc[:, 0] = 0.0
    # cumulative sums along q; a missing energy makes every later sum NaN
    S[:, 1:] = np.cumsum(e, axis=1)
    S_unc[:, 1:] = np.sqrt(np.cumsum(u ** 2, axis=1))
    qs = np.arange(z_max + 2)
    above = qs[None, :] > np.arange(z_max + 1)[:, None]      # q > Z: impossible
    S[above] = S_unc[above] = np.nan
    return S, S_unc


def removed_binding_ev(Z, q):
    """Binding energy (eV) of the q electrons removed from the neutral atom Z,
    and its uncertainty.  Works on scalars and arrays; NaN where unknown or
    impossible (q < 0, q > Z)."""
    S, S_unc = _removed_binding_table()
    Z = np.asarray(Z, dtype=float)
    q = np.asarray(q, dtype=float)
    ok = (np.isfinite(Z) & np.isfinite(q) & (Z >= 0) & (Z < S.shape[0])
          & (q >= 0) & (q < S.shape[1]))
    zi = np.where(ok, Z, 0).astype(int)
    qi = np.where(ok, q, 0).astype(int)
    val = np.where(ok, S[zi, qi], np.nan)
    unc = np.where(ok, S_unc[zi, qi], np.nan)
    if val.ndim == 0:
        return float(val), float(unc)
    return val, unc


def ion_electron_energy_kev(Z, q):
    """Mass of ion (Z, q) minus the neutral-atom mass, in keV:
    -q m_e c^2 + S(Z, q).  Returns (value, uncertainty)."""
    s, s_unc = removed_binding_ev(Z, q)
    return (-np.asarray(q, dtype=float) * ELECTRON_MASS_U * AMU_KEV + np.asarray(s) / 1000.0,
            np.asarray(s_unc) / 1000.0)


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

    # Atomic mass in u (from the mass excess) and its uncertainty
    compact_df["atomic_mass_u"] = compact_df["AAA"] + compact_df["m.e."] / AMU_KEV
    compact_df["atomic_mass_unc_u"] = compact_df["m.e._unc"] / AMU_KEV

    # Nuclear mass in u: all Z electrons removed, their total binding energy
    # added back (bare ion, q = Z)
    s, s_unc = removed_binding_ev(compact_df["ZZZi"], compact_df["ZZZi"])
    compact_df["nuclear_mass_u"] = (
        compact_df["atomic_mass_u"]
        - compact_df["ZZZi"] * ELECTRON_MASS_U
        + s / 1000.0 / AMU_KEV
    )
    compact_df["nuclear_mass_unc_u"] = np.hypot(compact_df["atomic_mass_unc_u"],
                                                s_unc / 1000.0 / AMU_KEV)

    return compact_df


def df_nubase(electrons):
    """Return a NuBase2020 DataFrame with m/q calculated for a given charge state.

    m_ion = M_atom - q m_e + S(Z, q)/c^2, with S the binding energy of the q
    removed electrons (NIST ionization energies).  m/q_unc combines the NUBASE
    mass-excess and NIST uncertainties in quadrature.  m/q is NaN where the
    ionization energies are unknown or the charge state is impossible.

    Parameters
    ----------
    electrons : int
        Number of bound electrons (0 = fully stripped).

    Returns
    -------
    pandas.DataFrame
    """
    base_df = get_base_dataframe()
    q = base_df["ZZZi"] - electrons
    s, s_unc = removed_binding_ev(base_df["ZZZi"], q)
    base_df["e_binding_removed_keV"] = s / 1000.0
    base_df["e_binding_removed_unc_keV"] = s_unc / 1000.0
    ion_mass_u = base_df["atomic_mass_u"] - q * ELECTRON_MASS_U + s / 1000.0 / AMU_KEV
    ion_mass_unc_u = np.hypot(base_df["atomic_mass_unc_u"], s_unc / 1000.0 / AMU_KEV)
    q_pos = q.where(q > 0)                      # q <= 0: no ion -> NaN
    base_df["ion_mass_u"] = ion_mass_u
    base_df["m/q"]     = ion_mass_u / q_pos
    base_df["m/q_unc"] = ion_mass_unc_u / q_pos
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
