"""
Pillar 1 — Solid-Acid Electrolyte: Tunnelling-Enhanced Proton Transport

Implements:
  - Eq 2: WKB tunnelling correction kappa(T) via numerical integration
  - Double-well potential V(x) = V_b * [1 - (x/a)^2]^2
  - Conductivity sigma(T) = sigma_0 * kappa(T) * exp(-E_a / k_B T) / T
  - ASR(T) = t_electrolyte / sigma(T)
  - Cell voltage: V_cell = E_nernst - eta_act - j * ASR(T) - eta_conc
  - Strain doping: reducing barrier width a by a fraction to simulate GB compression
"""

import numpy as np
from scipy import integrate

# Physical constants
K_B = 8.617333e-5       # Boltzmann constant in eV/K
HBAR = 6.582119e-16     # reduced Planck constant in eV*s
M_P = 1.6726219e-27     # proton mass in kg
M_P_EV = 938.272e6      # proton mass in eV/c^2
EV_TO_J = 1.602176634e-19  # eV to Joules


def double_well_potential(x, V_b, a):
    """
    Symmetric quartic double-well: V(x) = V_b * [1 - (x/a)^2]^2
    Barrier height V_b at x=0, minima at x = +/- a.

    Parameters:
        x: position in meters
        V_b: barrier height in eV
        a: half-width (distance from center to minimum) in meters
    Returns:
        V(x) in eV
    """
    return V_b * (1.0 - (x / a) ** 2) ** 2


def _wkb_action_at_energy(E, V_b, a):
    """
    Compute the WKB tunnelling action integral at a given energy E:
        S(E) = (2/hbar) * integral_{x1}^{x2} sqrt(2 * m_p * (V(x) - E)) dx

    where x1, x2 are the classical turning points where V(x) = E.

    For the quartic double-well V(x) = V_b * [1-(x/a)^2]^2,
    setting V(x) = E gives: [1-(x/a)^2]^2 = E/V_b
    => (x/a)^2 = 1 - sqrt(E/V_b)  and  (x/a)^2 = 1 + sqrt(E/V_b)

    The inner turning points (the barrier region) are:
      x1 = -a * sqrt(1 - sqrt(E/V_b))
      x2 = +a * sqrt(1 - sqrt(E/V_b))

    We need E < V_b for tunnelling through the barrier.
    """
    if E >= V_b or E < 0:
        return 0.0

    ratio = np.sqrt(E / V_b)
    arg = 1.0 - ratio
    if arg <= 0:
        return 0.0

    x_turn = a * np.sqrt(arg)
    # Turning points are at -x_turn and +x_turn

    def integrand(x):
        V = double_well_potential(x, V_b, a)
        diff = V - E
        if diff <= 0:
            return 0.0
        # sqrt(2 * m_p * diff_in_joules) but we keep everything in eV and meters
        # 2*m_p*(V-E) has units of kg*eV, need to convert eV to J for proper units
        # sqrt(2 * m_p_kg * (V-E)_eV * eV_to_J) gives units of kg*m/s
        return np.sqrt(2.0 * M_P * diff * EV_TO_J)

    # Integrate from -x_turn to +x_turn
    # By symmetry, = 2 * integral from 0 to x_turn
    result, _ = integrate.quad(integrand, 0, x_turn, limit=100)
    action = 2.0 * result * 2.0 / (HBAR * EV_TO_J)  # hbar in J*s
    return action


def tunnelling_kappa(T, V_b, a, n_energy=50):
    """
    Compute the WKB tunnelling transmission enhancement kappa(T) (Eq 2).

    kappa(T) = (1/k_B*T) * integral_0^{V_b} exp(-action(E)) * exp(-E/k_B*T) dE

    Parameters:
        T: temperature in Kelvin
        V_b: barrier height in eV
        a: half-width of double-well in Angstroms (converted to meters internally)
        n_energy: number of energy points for integration
    Returns:
        kappa: dimensionless tunnelling enhancement factor
    """
    a_m = a * 1e-10  # Angstroms to meters
    k_BT = K_B * T   # in eV

    if k_BT < 1e-10:
        return 1.0

    def integrand(E):
        action = _wkb_action_at_energy(E, V_b, a_m)
        transmission = np.exp(-action)
        boltzmann = np.exp(-E / k_BT)
        return transmission * boltzmann

    result, _ = integrate.quad(integrand, 0, V_b * 0.999, limit=100)
    kappa = result / k_BT
    return max(kappa, 1.0)  # kappa >= 1 (at minimum, classical rate)


def conductivity(T, V_b, a, sigma_0=1.0e4, E_a=0.35):
    """
    Proton conductivity with tunnelling correction.
    sigma(T) = sigma_0 * kappa(T) * exp(-E_a / k_B*T) / T

    Parameters:
        T: temperature in K
        V_b: barrier height in eV
        a: barrier half-width in Angstroms
        sigma_0: pre-exponential factor in S*cm^-1*K
        E_a: activation energy in eV
    Returns:
        sigma in S/cm
    """
    kappa = tunnelling_kappa(T, V_b, a)
    k_BT = K_B * T
    return sigma_0 * kappa * np.exp(-E_a / k_BT) / T


def asr(T, V_b, a, t_elec=50e-4, sigma_0=1.0e4, E_a=0.35):
    """
    Area-specific resistance ASR(T) = t_electrolyte / sigma(T).

    Parameters:
        T: temperature in K
        t_elec: electrolyte thickness in cm (default 50 micron = 50e-4 cm)
    Returns:
        ASR in Ohm*cm^2
    """
    sig = conductivity(T, V_b, a, sigma_0, E_a)
    if sig < 1e-15:
        return 1e6  # Effectively infinite resistance
    return t_elec / sig


def cell_voltage(j, T, V_b, a, t_elec=50e-4, sigma_0=1.0e4, E_a=0.35):
    """
    Cell voltage under load.
    V_cell = E_nernst - eta_act - j * ASR(T) - eta_conc

    Parameters:
        j: current density in A/cm^2
        T: temperature in K
    Returns:
        V_cell in Volts
    """
    # Nernst OCV for H2/O2 at 1 atm (simplified)
    E_nernst = 1.229 - 0.00085 * (T - 298.15)

    # Activation overpotential (Tafel equation, simplified)
    j0 = 1e-3  # Exchange current density A/cm^2
    alpha = 0.5
    if j > 0:
        eta_act = (K_B * T / alpha) * np.log(j / j0 + 1.0) * (EV_TO_J / EV_TO_J)
        # Convert from eV to V: K_B is in eV/K, so K_B*T/alpha is in eV = volts for single charge
        eta_act = (K_B * T / alpha) * np.log(j / j0 + 1.0)
    else:
        eta_act = 0.0

    # Ohmic overpotential
    asr_val = asr(T, V_b, a, t_elec, sigma_0, E_a)
    eta_ohm = j * asr_val

    # Concentration overpotential (simplified)
    j_L = 2.0  # Limiting current density A/cm^2
    if j < j_L:
        eta_conc = -K_B * T * np.log(1.0 - j / j_L)
    else:
        eta_conc = 1.0  # Saturated

    V_cell = E_nernst - eta_act - eta_ohm - eta_conc
    return max(V_cell, 0.0)


def run_simulation(params: dict) -> dict:
    """
    Run the full Pillar 1 simulation.

    Parameters (from frontend):
        V_b: barrier height in eV (default 0.3)
        a: barrier half-width in Angstroms (default 0.2)
        strain_pct: grain-boundary strain percentage (default 0, range 0-10)
        T_min: min temperature in C (default 150)
        T_max: max temperature in C (default 230)
        j_max: max current density A/cm^2 (default 1.5)

    Returns dict with all plot data as lists (JSON-serializable).
    """
    V_b = float(params.get("V_b", 0.3))
    a = float(params.get("a", 0.2))
    strain_pct = float(params.get("strain_pct", 0.0))
    T_min_C = float(params.get("T_min", 150))
    T_max_C = float(params.get("T_max", 230))
    j_max = float(params.get("j_max", 1.5))

    # Temperature array in Kelvin
    T_C = np.linspace(T_min_C, T_max_C, 40)
    T_K = T_C + 273.15

    # Current density array
    j_arr = np.linspace(0.001, j_max, 50)

    # --- Chart 1: kappa(T) with and without strain ---
    a_strained = a * (1.0 - strain_pct / 100.0)

    kappa_no_strain = []
    kappa_strained = []
    for T in T_K:
        kappa_no_strain.append(tunnelling_kappa(T, V_b, a))
        if strain_pct > 0:
            kappa_strained.append(tunnelling_kappa(T, V_b, a_strained))

    # --- Chart 2: sigma(T) with and without strain ---
    sigma_no_strain = []
    sigma_strained = []
    for T in T_K:
        sigma_no_strain.append(conductivity(T, V_b, a))
        if strain_pct > 0:
            sigma_strained.append(conductivity(T, V_b, a_strained))

    # --- Chart 3: Polarization curve V_cell vs j at mid-temperature ---
    T_mid = T_K[len(T_K) // 2]
    vcell_no_strain = []
    vcell_strained = []
    for j in j_arr:
        vcell_no_strain.append(cell_voltage(j, T_mid, V_b, a))
        if strain_pct > 0:
            vcell_strained.append(cell_voltage(j, T_mid, V_b, a_strained))

    # --- Chart 4: ASR vs T ---
    asr_no_strain = []
    asr_strained = []
    for T in T_K:
        asr_no_strain.append(asr(T, V_b, a))
        if strain_pct > 0:
            asr_strained.append(asr(T, V_b, a_strained))

    # Key metric: ASR at operating temperature (mid-range)
    key_asr = asr(T_mid, V_b, a_strained if strain_pct > 0 else a)

    return {
        "temperatures_C": T_C.tolist(),
        "kappa_no_strain": kappa_no_strain,
        "kappa_strained": kappa_strained if strain_pct > 0 else None,
        "sigma_no_strain": sigma_no_strain,
        "sigma_strained": sigma_strained if strain_pct > 0 else None,
        "j_arr": j_arr.tolist(),
        "vcell_no_strain": vcell_no_strain,
        "vcell_strained": vcell_strained if strain_pct > 0 else None,
        "asr_no_strain": asr_no_strain,
        "asr_strained": asr_strained if strain_pct > 0 else None,
        "polarization_temp_C": round(T_mid - 273.15, 1),
        "key_asr": round(key_asr, 4),
        "strain_pct": strain_pct,
    }
