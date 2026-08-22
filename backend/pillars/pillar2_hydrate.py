"""
Pillar 2 — Clathrate-Hydrate Hydrogen Storage Cartridge

Implements:
  - Eq 5: Langmuir cage occupancy theta_{i,g} for sII hydrate
  - Eq 6: Water chemical potential change Delta_mu_w
  - Eq 7: Gravimetric H2 storage capacity wt%
  - Kihara cell-potential Langmuir constants C_{i,g}(T)
  - Defect-mediated lattice pre-straining: dose -> cage radius shift -> multi-occupancy
"""

import numpy as np
from scipy import integrate

# Physical constants
K_B_JK = 1.380649e-23   # Boltzmann constant J/K
R_GAS = 8.314462         # Gas constant J/(mol*K)
N_A = 6.02214076e23      # Avogadro's number

# sII clathrate structure constants
NU_SMALL = 2.0 / 17.0   # Small cages (5^12) per water molecule
NU_LARGE = 1.0 / 17.0   # Large cages (5^12 6^4) per water molecule
N_WATER = 136            # Water molecules per unit cell in sII
N_SMALL = 16             # Small cages per unit cell
N_LARGE = 8              # Large cages per unit cell

# Cage radii in Angstroms (sII)
R_SMALL_BASE = 3.91      # Small cage mean radius
R_LARGE_BASE = 4.73      # Large cage mean radius

# Molecular masses in g/mol
M_H2 = 2.016
M_THF = 72.11
M_DIOX = 74.08
M_WATER = 18.015

# Kihara potential parameters for H2 in water cages
# epsilon/k_B in K, sigma in Angstroms, core_radius in Angstroms
KIHARA_H2 = {"epsilon_over_kB": 36.0, "sigma": 2.97, "core_a": 0.3275}
KIHARA_THF = {"epsilon_over_kB": 225.0, "sigma": 5.1, "core_a": 0.0}


def kihara_cell_potential(r, R_cage, eps_over_kB, sigma, core_a):
    """
    Kihara cell potential for a guest molecule at distance r from cage center.
    w(r) = 2 * z * epsilon * [sigma^12/(R^11 * r) * (delta^10 + core_a/R * delta^11)
                              - sigma^6/(R^5 * r) * (delta^4 + core_a/R * delta^5)]
    Simplified spherical-cell approximation.
    """
    R = R_cage * 1e-10  # to meters
    r_m = r * 1e-10
    sig = sigma * 1e-10
    a_c = core_a * 1e-10
    eps = eps_over_kB * K_B_JK

    if r_m >= R - a_c or r_m < 1e-15:
        return 1e10  # Very large energy = excluded

    z = 20  # Coordination number for sII small cage (approximate)
    # Use the simplified McKoy-Sinanoglu form
    rho = r_m / R
    sig_R = sig / R

    def delta_n(n):
        val = ((1.0 - rho - sig_R) ** (-n) - (1.0 + rho - sig_R) ** (-n)) / n
        return val

    try:
        w = 2.0 * z * eps * (
            sig_R ** 12 * (delta_n(10) + a_c / R * delta_n(11))
            - sig_R ** 6 * (delta_n(4) + a_c / R * delta_n(5))
        )
    except (ZeroDivisionError, OverflowError):
        w = 1e10

    return w


def langmuir_constant(T, R_cage, guest_params, z_coord=20):
    """
    Compute Langmuir constant C(T) for a guest in a cage of radius R_cage.

    C(T) = (4*pi / k_B*T) * integral_0^{R_cage} exp(-w(r)/k_B*T) * r^2 dr

    Parameters:
        T: temperature in K
        R_cage: cage radius in Angstroms
        guest_params: dict with epsilon_over_kB, sigma, core_a
    Returns:
        C in 1/Pa (Langmuir constant)
    """
    kBT = K_B_JK * T

    def integrand(r_ang):
        w = kihara_cell_potential(
            r_ang, R_cage,
            guest_params["epsilon_over_kB"],
            guest_params["sigma"],
            guest_params["core_a"]
        )
        # w is in Joules — clamp exponent to prevent overflow/underflow
        exponent = -w / kBT
        if exponent > 500 or exponent < -500:
            return 0.0 if exponent < -500 else 0.0
        exponent = max(min(exponent, 500), -500)
        return np.exp(exponent) * (r_ang * 1e-10) ** 2

    r_max = R_cage * 0.90  # Stay well away from the wall singularity
    result, _ = integrate.quad(integrand, 0.05, r_max, limit=80)
    C = 4.0 * np.pi / kBT * result * 1e-10  # Convert dr from Angstroms
    # Guard against NaN/Inf
    if not np.isfinite(C):
        C = 0.0
    return C


def cage_occupancy(C_values, fugacities):
    """
    Eq 5: Langmuir adsorption occupancy.
    theta_{i,g} = C_{i,g} * f_g / (1 + sum_g' C_{i,g'} * f_{g'})

    Parameters:
        C_values: dict of {guest_name: C_value} for this cage type
        fugacities: dict of {guest_name: fugacity_in_Pa}
    Returns:
        dict of {guest_name: occupancy}
    """
    denom = 1.0
    for g, C in C_values.items():
        val = C * fugacities.get(g, 0.0)
        if not np.isfinite(val):
            val = 1e15  # Very large but finite
        denom += val

    if not np.isfinite(denom) or denom < 1e-15:
        denom = 1e15  # Prevent division by zero/NaN

    thetas = {}
    for g, C in C_values.items():
        val = C * fugacities.get(g, 0.0)
        if not np.isfinite(val):
            val = 1e15
        theta = val / denom
        thetas[g] = min(max(float(theta), 0.0), 1.0)  # Clamp to [0, 1]

    return thetas


def water_chemical_potential(thetas_small, thetas_large):
    """
    Eq 6: Delta_mu_w / RT = -sum_i nu_i * ln(1 - sum_g theta_{i,g})

    Returns Delta_mu_w / RT (dimensionless).
    """
    sum_small = sum(thetas_small.values())
    sum_large = sum(thetas_large.values())

    # Clamp to avoid log(0)
    sum_small = min(sum_small, 0.9999)
    sum_large = min(sum_large, 0.9999)

    return -(NU_SMALL * np.log(1.0 - sum_small) + NU_LARGE * np.log(1.0 - sum_large))


def gravimetric_wt_pct(thetas_small, thetas_large, promoter_mass=M_THF,
                       promoter_in_large=True):
    """
    Eq 7: wt% H2 from cage occupancies.

    n_H2 = N_SMALL * theta_H2_small + N_LARGE * theta_H2_large * (1 if promoter not in large else 0)
    n_promoter = N_LARGE * theta_promoter_large (if promoter_in_large)
    n_water = N_WATER
    """
    n_H2_small = N_SMALL * thetas_small.get("H2", 0.0)
    n_H2_large = N_LARGE * thetas_large.get("H2", 0.0)
    n_H2 = n_H2_small + n_H2_large

    n_promoter = 0.0
    if promoter_in_large:
        n_promoter = N_LARGE * thetas_large.get("promoter", 0.0)

    mass_H2 = n_H2 * M_H2
    mass_promoter = n_promoter * promoter_mass
    mass_water = N_WATER * M_WATER

    total = mass_H2 + mass_promoter + mass_water
    if total < 1e-10:
        return 0.0
    return (mass_H2 / total) * 100.0


def h2_fugacity(P_MPa, T):
    """
    Approximate H2 fugacity from pressure using a simple virial correction.
    f = P * exp(B*P/RT) where B is the second virial coefficient.
    """
    P_Pa = P_MPa * 1e6
    # Second virial coefficient for H2 (approximate, in m^3/mol)
    B = 1.5e-5  # Simplified
    f = P_Pa * np.exp(B * P_Pa / (R_GAS * T))
    return f


def run_simulation(params: dict) -> dict:
    """
    Run the full Pillar 2 simulation.

    Parameters:
        P_min: min pressure in MPa (default 1)
        P_max: max pressure in MPa (default 15)
        T: temperature in K (default 279)
        promoter_ratio: promoter sub-stoichiometric ratio 0-1 (default 0.5)
        defect_dose: lattice defect dose 0-1 (default 0)
        promoter_type: 'THF' or 'DIOX' (default 'THF')
    """
    P_min = float(params.get("P_min", 1.0))
    P_max = float(params.get("P_max", 15.0))
    T = float(params.get("T", 279.0))
    promoter_ratio = float(params.get("promoter_ratio", 0.5))
    defect_dose = float(params.get("defect_dose", 0.0))
    promoter_type = params.get("promoter_type", "THF")

    promoter_mass = M_THF if promoter_type == "THF" else M_DIOX

    # Apply defect dose: elongates cage radii, enabling multi-occupancy
    # Dose 0->1 maps to 0->3% lattice elongation (from the cited study)
    lattice_strain = defect_dose * 0.03
    R_small = R_SMALL_BASE * (1.0 + lattice_strain)
    R_large = R_LARGE_BASE * (1.0 + lattice_strain)

    # Compute Langmuir constants at this temperature
    C_H2_small = langmuir_constant(T, R_small, KIHARA_H2)
    C_H2_large = langmuir_constant(T, R_large, KIHARA_H2)

    # For promoter: use a simplified large Langmuir constant
    # (promoter fills large cages preferentially)
    C_prom_large = langmuir_constant(T, R_large, KIHARA_THF) * promoter_ratio

    # --- Chart 1: Occupancy vs Pressure ---
    P_arr = np.linspace(P_min, P_max, 50)
    theta_H2_small_arr = []
    theta_H2_large_arr = []
    theta_prom_large_arr = []
    wt_pct_arr = []
    delta_mu_arr = []

    for P in P_arr:
        f_H2 = h2_fugacity(P, T)
        # Promoter fugacity is set by its liquid-phase activity (simplified as constant)
        f_prom = promoter_ratio * 1e5  # Simplified

        C_small = {"H2": C_H2_small}
        C_large = {"H2": C_H2_large, "promoter": C_prom_large}
        fug = {"H2": f_H2, "promoter": f_prom}

        thetas_s = cage_occupancy(C_small, fug)
        thetas_l = cage_occupancy(C_large, fug)

        theta_H2_small_arr.append(thetas_s.get("H2", 0.0))
        theta_H2_large_arr.append(thetas_l.get("H2", 0.0))
        theta_prom_large_arr.append(thetas_l.get("promoter", 0.0))

        wt = gravimetric_wt_pct(thetas_s, thetas_l, promoter_mass)
        wt_pct_arr.append(wt)

        dmu = water_chemical_potential(thetas_s, thetas_l)
        delta_mu_arr.append(dmu)

    # --- Chart 2: wt% vs defect dose at fixed pressure ---
    P_fixed = (P_min + P_max) / 2.0
    dose_arr = np.linspace(0, 1.0, 30)
    wt_vs_dose = []
    for dose in dose_arr:
        strain = dose * 0.03
        R_s = R_SMALL_BASE * (1.0 + strain)
        R_l = R_LARGE_BASE * (1.0 + strain)
        C_s = langmuir_constant(T, R_s, KIHARA_H2)
        C_l = langmuir_constant(T, R_l, KIHARA_H2)
        C_pl = langmuir_constant(T, R_l, KIHARA_THF) * promoter_ratio

        f_H2 = h2_fugacity(P_fixed, T)
        f_prom = promoter_ratio * 1e5

        ts = cage_occupancy({"H2": C_s}, {"H2": f_H2})
        tl = cage_occupancy({"H2": C_l, "promoter": C_pl}, {"H2": f_H2, "promoter": f_prom})
        wt = gravimetric_wt_pct(ts, tl, promoter_mass)
        wt_vs_dose.append(wt)

    # --- Chart 3: P-T stability (wt% heatmap) ---
    T_arr = np.linspace(270, 285, 10)
    P_stab = np.linspace(P_min, P_max, 10)
    wt_heatmap = []
    for t in T_arr:
        row = []
        for p in P_stab:
            C_s = langmuir_constant(t, R_small, KIHARA_H2)
            C_l = langmuir_constant(t, R_large, KIHARA_H2)
            C_pl = langmuir_constant(t, R_large, KIHARA_THF) * promoter_ratio
            f_H2 = h2_fugacity(p, t)
            f_prom = promoter_ratio * 1e5
            ts = cage_occupancy({"H2": C_s}, {"H2": f_H2})
            tl = cage_occupancy({"H2": C_l, "promoter": C_pl}, {"H2": f_H2, "promoter": f_prom})
            wt = gravimetric_wt_pct(ts, tl, promoter_mass)
            row.append(round(wt, 4))
        wt_heatmap.append(row)

    # Key metric
    key_wt = wt_pct_arr[len(wt_pct_arr) // 2] if wt_pct_arr else 0.0

    return {
        "pressures_MPa": P_arr.tolist(),
        "theta_H2_small": theta_H2_small_arr,
        "theta_H2_large": theta_H2_large_arr,
        "theta_promoter_large": theta_prom_large_arr,
        "wt_pct_vs_P": wt_pct_arr,
        "delta_mu": delta_mu_arr,
        "dose_arr": dose_arr.tolist(),
        "wt_vs_dose": wt_vs_dose,
        "T_stability": T_arr.tolist(),
        "P_stability": P_stab.tolist(),
        "wt_heatmap": wt_heatmap,
        "key_wt_pct": round(key_wt, 3),
        "temperature_K": T,
        "defect_dose": defect_dose,
    }
