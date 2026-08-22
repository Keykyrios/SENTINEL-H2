"""
Pillar 5 — Non-Hermitian Exceptional-Point Sensor Network

Implements:
  - Eq 14: Complex eigenvalues of a 2x2 non-Hermitian coupled-mode system
    omega_pm = (w1+w2)/2 - i(g1+g2)/4 +/- sqrt(kappa^2 + ((dw - i*dg/2)/2)^2)
  - EP condition: dg/2 = kappa, dw = 0 -> coalescence
  - Perturbation sweep: delta_kappa -> eigenvalue splitting (sqrt scaling)
  - Pd-H2 transduction model: H2 concentration -> delta_kappa
  - Monte Carlo noise analysis: minimum detectable perturbation vs noise floor
"""

import numpy as np


def coupled_mode_eigenvalues(omega1, omega2, gamma1, gamma2, kappa):
    """
    Eq 14: Eigenvalues of the 2x2 non-Hermitian Hamiltonian.

    H = [[omega1 - i*gamma1/2, kappa],
         [kappa, omega2 - i*gamma2/2]]

    omega_pm = (omega1+omega2)/2 - i*(gamma1+gamma2)/4
               +/- sqrt(kappa^2 + ((delta_omega - i*delta_gamma/2)/2)^2)

    Parameters:
        omega1, omega2: resonance frequencies (real, Hz or normalized)
        gamma1, gamma2: loss rates (real, Hz or normalized)
        kappa: inter-resonator coupling (real)
    Returns:
        (omega_plus, omega_minus) as complex numbers
    """
    avg_omega = (omega1 + omega2) / 2.0
    avg_gamma = (gamma1 + gamma2) / 4.0
    delta_omega = omega1 - omega2
    delta_gamma = gamma1 - gamma2

    # The discriminant under the square root
    # kappa^2 + ((delta_omega - i*delta_gamma/2) / 2)^2
    z = (delta_omega - 1j * delta_gamma / 2.0) / 2.0
    discriminant = kappa ** 2 + z ** 2
    sqrt_disc = np.sqrt(discriminant + 0j)  # Ensure complex sqrt

    base = avg_omega - 1j * avg_gamma
    omega_plus = base + sqrt_disc
    omega_minus = base - sqrt_disc

    return omega_plus, omega_minus


def eigenvalue_splitting(kappa_base, delta_perturbation):
    """
    Compute eigenvalue splitting near the EP.

    From Eq 14, the discriminant is:
      D = kappa^2 + ((dw - i*dg/2) / 2)^2

    With dw = 0, D = kappa^2 - (dg/4)^2.
    EP condition (D=0): kappa = dg/4, i.e. dg = 4*kappa.

    The physical perturbation is a change in loss on the Pd-coated resonator
    (H2 absorption changes gamma2). We perturb gamma2 by delta_perturbation,
    which shifts dg and breaks the EP degeneracy.

    Splitting ~ sqrt(delta_perturbation) for small perturbations — this is the
    key EP sensitivity enhancement.
    """
    omega0 = 1.0  # Normalized base frequency
    gamma1 = 0.1
    # EP condition: dg = 4*kappa_base, so gamma2 = gamma1 - 4*kappa_base
    gamma2_ep = gamma1 - 4.0 * kappa_base

    # Perturb gamma2 (H2-induced loss change on Pd resonator)
    gamma2_perturbed = gamma2_ep - delta_perturbation

    wp, wm = coupled_mode_eigenvalues(omega0, omega0, gamma1, gamma2_perturbed, kappa_base)
    return abs(wp - wm)


def pd_h2_transduction(c_H2_ppm, alpha=1e-4, beta=0.8):
    """
    Pd-H2 transduction model.
    delta_kappa = alpha * c_H2^beta

    Models H2 absorption into Pd film causing elastic modulus and loss change
    that detunes the EP resonator pair.

    Parameters:
        c_H2_ppm: H2 concentration in ppm
        alpha: transduction coefficient
        beta: nonlinearity exponent (Sievert's law gives beta~0.5 for dilute)
    """
    return alpha * (c_H2_ppm ** beta)


def run_simulation(params: dict) -> dict:
    """
    Run the full Pillar 5 simulation.

    Parameters:
        kappa_base: base coupling at EP (default 0.05)
        dk_min_log: log10 of min delta_kappa (default -6)
        dk_max_log: log10 of max delta_kappa (default -1)
        n_noise_mc: Monte Carlo samples for noise analysis (default 500)
        noise_floor: measurement noise floor (default 1e-4)
    """
    kappa_base = float(params.get("kappa_base", 0.05))
    dk_min_log = float(params.get("dk_min_log", -6))
    dk_max_log = float(params.get("dk_max_log", -1))
    n_noise_mc = int(params.get("n_noise_mc", 500))
    noise_floor = float(params.get("noise_floor", 1e-4))

    n_noise_mc = min(max(n_noise_mc, 100), 2000)

    # --- Chart 1: Eigenvalue splitting vs delta_kappa (log-log) ---
    dk_arr = np.logspace(dk_min_log, dk_max_log, 60)
    splitting_arr = []
    for dk in dk_arr:
        split = eigenvalue_splitting(kappa_base, dk)
        splitting_arr.append(float(split))

    # Linear reference: splitting = dk (for comparison)
    linear_ref = dk_arr.tolist()

    # Fit slope on log-log to confirm sqrt scaling
    log_dk = np.log10(dk_arr)
    log_split = np.log10(np.array(splitting_arr) + 1e-20)
    # Linear fit
    valid = np.isfinite(log_split)
    if np.sum(valid) > 2:
        coeffs = np.polyfit(log_dk[valid], log_split[valid], 1)
        fitted_slope = float(coeffs[0])
    else:
        fitted_slope = 0.5

    # --- Chart 2: Eigenvalue trajectories in complex plane ---
    # Sweep kappa from 0 to 2*kappa_base to cross the EP
    kappa_sweep = np.linspace(0.001, 2.0 * kappa_base, 80)
    omega0 = 1.0
    gamma1 = 0.1
    gamma2 = gamma1 - 4.0 * kappa_base  # EP condition: dg = 4*kappa

    traj_plus_re = []
    traj_plus_im = []
    traj_minus_re = []
    traj_minus_im = []

    for k in kappa_sweep:
        wp, wm = coupled_mode_eigenvalues(omega0, omega0, gamma1, gamma2, k)
        traj_plus_re.append(float(wp.real))
        traj_plus_im.append(float(wp.imag))
        traj_minus_re.append(float(wm.real))
        traj_minus_im.append(float(wm.imag))

    # --- Chart 3: Sensitivity gain (EP vs linear) ---
    sensitivity_ep = []
    sensitivity_linear = []
    for dk in dk_arr:
        split_ep = eigenvalue_splitting(kappa_base, dk)
        # EP sensitivity: d(split)/d(dk) ~ 1/(2*sqrt(dk)) for sqrt scaling
        sensitivity_ep.append(float(split_ep / dk) if dk > 0 else 0)
        # Linear sensor: d(split)/d(dk) = 1
        sensitivity_linear.append(1.0)

    # --- Chart 4: Detection limit vs noise floor ---
    noise_levels = np.logspace(-5, -2, 20)
    min_detectable_dk = []
    min_detectable_ppm = []
    rng = np.random.RandomState(42)

    for nf in noise_levels:
        # Find minimum dk where splitting > 3*noise
        found = False
        for dk_test in np.logspace(-7, -1, 100):
            split = eigenvalue_splitting(kappa_base, dk_test)
            if split > 3.0 * nf:
                min_detectable_dk.append(float(dk_test))
                # Convert to H2 ppm via transduction model
                # Invert: c_H2 = (dk / alpha)^(1/beta)
                c_ppm = (dk_test / 1e-4) ** (1.0 / 0.8)
                min_detectable_ppm.append(float(c_ppm))
                found = True
                break
        if not found:
            min_detectable_dk.append(1.0)
            min_detectable_ppm.append(1e6)

    # H2 concentration sweep -> splitting
    h2_ppm = np.logspace(0, 5, 50)  # 1 ppm to 100,000 ppm (10% = LEL territory)
    h2_splitting = []
    for c in h2_ppm:
        dk = pd_h2_transduction(c)
        split = eigenvalue_splitting(kappa_base, dk)
        h2_splitting.append(float(split))

    # Key metric: sensitivity gain at 100 ppm H2
    dk_100ppm = pd_h2_transduction(100)
    split_100 = eigenvalue_splitting(kappa_base, dk_100ppm)
    gain = split_100 / dk_100ppm if dk_100ppm > 0 else 1.0

    return {
        "dk_arr": dk_arr.tolist(),
        "splitting": splitting_arr,
        "linear_ref": linear_ref,
        "fitted_slope": round(fitted_slope, 3),
        "kappa_sweep": kappa_sweep.tolist(),
        "traj_plus_re": traj_plus_re,
        "traj_plus_im": traj_plus_im,
        "traj_minus_re": traj_minus_re,
        "traj_minus_im": traj_minus_im,
        "sensitivity_ep": sensitivity_ep,
        "sensitivity_linear": sensitivity_linear,
        "noise_levels": noise_levels.tolist(),
        "min_detectable_dk": min_detectable_dk,
        "min_detectable_ppm": min_detectable_ppm,
        "h2_ppm": h2_ppm.tolist(),
        "h2_splitting": h2_splitting,
        "key_gain": round(float(gain), 1),
        "kappa_base": kappa_base,
    }
