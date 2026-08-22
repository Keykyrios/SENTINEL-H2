"""
Pillar 4 — Acoustic-Emission Inversion for Stack State-of-Health

Implements:
  - Eq 12: Forward degradation model V(t) = V0 - k1*dECSA - k2*dR_ohm - k3*d_eta_mt
  - Synthetic AE feature generator (hit rate, energy, b-value)
  - Eq 13: Bayesian inversion p(theta_deg | y_AE) via Metropolis-Hastings MCMC
  - Posterior recovery of degradation parameters from noisy AE observations
"""

import numpy as np


# --- Forward Degradation Model (Eq 12) ---

def ecsa_loss(t_hours, k_ecsa=0.018):
    """
    ECSA loss follows diffusion-limited Pt dissolution: dECSA ~ sqrt(t).
    Calibrated to ~55% loss at 900h (from cited literature).
    dECSA(t) = k_ecsa * sqrt(t)
    """
    return k_ecsa * np.sqrt(t_hours)


def ohmic_increase(t_hours, k_ohm=5e-5):
    """
    Ohmic resistance increases roughly linearly with time (membrane thinning).
    dR_ohm(t) = k_ohm * t
    """
    return k_ohm * t_hours


def mass_transport_loss(t_hours, k_mt=2e-6):
    """
    Mass transport degradation (GDL/PTFE aging).
    d_eta_mt(t) = k_mt * t^1.2 (slightly superlinear)
    """
    return k_mt * t_hours ** 1.2


def voltage_decay(t_hours, V0=0.72, k1=0.15, k2=0.8, k3=0.6,
                  k_ecsa=0.018, k_ohm=5e-5, k_mt=2e-6):
    """
    Eq 12: V(t) = V0 - k1*dECSA(t) - k2*dR_ohm(t) - k3*d_eta_mt(t)
    """
    dECSA = ecsa_loss(t_hours, k_ecsa)
    dR = ohmic_increase(t_hours, k_ohm)
    dMT = mass_transport_loss(t_hours, k_mt)
    return V0 - k1 * dECSA - k2 * dR - k3 * dMT


# --- Synthetic AE Feature Generator ---

def generate_ae_features(theta_deg, noise_std=0.05, rng=None):
    """
    Generate synthetic AE feature vector from degradation state.

    theta_deg = [dECSA, dR_ohm, d_eta_mt]

    AE features:
    - hit_rate: correlated with dECSA (microcracking) and dR_ohm
    - energy: correlated with d_eta_mt (flooding events) and dECSA
    - b_value: Gutenberg-Richter b-value, decreases as larger cracks form

    Returns [hit_rate, energy, b_value]
    """
    if rng is None:
        rng = np.random.RandomState()

    dECSA, dR, dMT = theta_deg

    # Hit rate increases with catalyst degradation
    hit_rate = 20.0 + 150.0 * dECSA + 80.0 * dR + rng.normal(0, noise_std * 30)
    hit_rate = max(hit_rate, 0.0)

    # AE energy increases with all degradation channels
    energy = 0.5 + 3.0 * dECSA + 2.0 * dMT + 1.0 * dR + rng.normal(0, noise_std * 0.5)
    energy = max(energy, 0.0)

    # b-value decreases as larger defects form (more big events)
    b_value = 1.5 - 0.8 * dECSA - 0.3 * dMT + rng.normal(0, noise_std * 0.15)
    b_value = max(b_value, 0.3)

    return np.array([hit_rate, energy, b_value])


# --- Bayesian Inversion (Eq 13) via Metropolis-Hastings MCMC ---

def ae_log_likelihood(theta_deg, y_ae_obs, noise_std=0.05):
    """
    Log-likelihood: p(y_AE | theta_deg)
    Assumes Gaussian noise model.
    """
    y_pred = generate_ae_features(theta_deg, noise_std=0.0)
    residual = y_ae_obs - y_pred
    # Noise variance per feature
    sigma = np.array([noise_std * 30, noise_std * 0.5, noise_std * 0.15])
    sigma = np.maximum(sigma, 1e-6)
    return -0.5 * np.sum((residual / sigma) ** 2)


def voltage_log_prior(theta_deg, t_hours, V_observed, V0=0.72):
    """
    Physics-informed prior from the voltage decay model.
    Penalizes degradation states inconsistent with observed voltage.
    """
    dECSA, dR, dMT = theta_deg
    # All components must be non-negative
    if dECSA < 0 or dR < 0 or dMT < 0:
        return -1e10

    V_pred = V0 - 0.15 * dECSA - 0.8 * dR - 0.6 * dMT
    # Gaussian prior centered on observed voltage
    sigma_v = 0.01
    log_p = -0.5 * ((V_pred - V_observed) / sigma_v) ** 2
    return log_p


def run_mcmc(y_ae_obs, t_hours, V_observed, n_samples=2000, noise_std=0.05):
    """
    Metropolis-Hastings MCMC for Bayesian inversion.

    Samples from: p(theta_deg | y_AE) propto p(y_AE | theta_deg) * p(theta_deg)
    """
    rng = np.random.RandomState(42)

    # Initial guess
    theta = np.array([0.3, 0.02, 0.01])
    proposal_std = np.array([0.02, 0.002, 0.001])

    samples = []
    log_p_current = ae_log_likelihood(theta, y_ae_obs, noise_std) + \
                    voltage_log_prior(theta, t_hours, V_observed)

    n_accept = 0
    for i in range(n_samples):
        # Propose
        theta_prop = theta + rng.normal(0, proposal_std)
        theta_prop = np.maximum(theta_prop, 0.0)  # Non-negative constraint

        log_p_prop = ae_log_likelihood(theta_prop, y_ae_obs, noise_std) + \
                     voltage_log_prior(theta_prop, t_hours, V_observed)

        # Accept/reject
        log_alpha = log_p_prop - log_p_current
        if np.log(rng.random()) < log_alpha:
            theta = theta_prop
            log_p_current = log_p_prop
            n_accept += 1

        samples.append(theta.copy())

    return np.array(samples), n_accept / n_samples


def run_simulation(params: dict) -> dict:
    """
    Run the full Pillar 4 simulation.

    Parameters:
        t_max_hours: max simulation time in hours (default 900)
        noise_std: AE noise level (default 0.05)
        n_mcmc: MCMC samples (default 2000)
        V0: initial cell voltage (default 0.72)
    """
    t_max = float(params.get("t_max_hours", 900))
    noise_std = float(params.get("noise_std", 0.05))
    n_mcmc = int(params.get("n_mcmc", 2000))
    V0 = float(params.get("V0", 0.72))

    n_mcmc = min(max(n_mcmc, 500), 5000)

    rng = np.random.RandomState(42)

    # --- Chart 1: Voltage decay over time ---
    t_arr = np.linspace(0, t_max, 60)
    v_arr = [voltage_decay(t, V0) for t in t_arr]

    # --- Ground truth degradation trajectory ---
    dECSA_true = [ecsa_loss(t) for t in t_arr]
    dR_true = [ohmic_increase(t) for t in t_arr]
    dMT_true = [mass_transport_loss(t) for t in t_arr]

    # --- Run Bayesian inversion at several time points ---
    inversion_times = np.linspace(t_max * 0.1, t_max, 8)
    posterior_means = {"dECSA": [], "dR_ohm": [], "d_eta_mt": []}
    posterior_stds = {"dECSA": [], "dR_ohm": [], "d_eta_mt": []}
    inversion_t_list = []

    for t_inv in inversion_times:
        # True degradation state at this time
        true_theta = np.array([
            ecsa_loss(t_inv),
            ohmic_increase(t_inv),
            mass_transport_loss(t_inv)
        ])

        # Generate noisy AE observation
        y_ae = generate_ae_features(true_theta, noise_std, rng)

        # Observed voltage (with some noise)
        V_obs = voltage_decay(t_inv, V0) + rng.normal(0, 0.005)

        # Run MCMC
        samples, accept_rate = run_mcmc(y_ae, t_inv, V_obs, n_mcmc, noise_std)

        # Burn-in: discard first 30%
        burn = int(n_mcmc * 0.3)
        post_samples = samples[burn:]

        posterior_means["dECSA"].append(float(np.mean(post_samples[:, 0])))
        posterior_means["dR_ohm"].append(float(np.mean(post_samples[:, 1])))
        posterior_means["d_eta_mt"].append(float(np.mean(post_samples[:, 2])))

        posterior_stds["dECSA"].append(float(np.std(post_samples[:, 0])))
        posterior_stds["dR_ohm"].append(float(np.std(post_samples[:, 1])))
        posterior_stds["d_eta_mt"].append(float(np.std(post_samples[:, 2])))

        inversion_t_list.append(float(t_inv))

    # --- Last inversion: full posterior histogram data ---
    last_true = np.array([ecsa_loss(t_max), ohmic_increase(t_max), mass_transport_loss(t_max)])
    y_ae_last = generate_ae_features(last_true, noise_std, rng)
    V_obs_last = voltage_decay(t_max, V0) + rng.normal(0, 0.005)
    samples_last, accept_last = run_mcmc(y_ae_last, t_max, V_obs_last, n_mcmc, noise_std)
    burn = int(n_mcmc * 0.3)
    post_last = samples_last[burn:]

    # Histogram bins
    n_bins = 30
    hist_ecsa, edges_ecsa = np.histogram(post_last[:, 0], bins=n_bins)
    hist_rohm, edges_rohm = np.histogram(post_last[:, 1], bins=n_bins)
    hist_mt, edges_mt = np.histogram(post_last[:, 2], bins=n_bins)

    # Key metric: SoH estimate (1 - dECSA_mean / dECSA_max)
    ecsa_mean = float(np.mean(post_last[:, 0]))
    soh = max(0.0, 1.0 - ecsa_mean / 0.55) * 100  # Relative to 55% max ECSA loss

    return {
        "time_hours": t_arr.tolist(),
        "voltage_decay": v_arr,
        "dECSA_true": dECSA_true,
        "dR_true": dR_true,
        "dMT_true": dMT_true,
        "inversion_times": inversion_t_list,
        "posterior_means": posterior_means,
        "posterior_stds": posterior_stds,
        "hist_ecsa": {"counts": hist_ecsa.tolist(), "edges": edges_ecsa.tolist()},
        "hist_rohm": {"counts": hist_rohm.tolist(), "edges": edges_rohm.tolist()},
        "hist_mt": {"counts": hist_mt.tolist(), "edges": edges_mt.tolist()},
        "true_at_end": {
            "dECSA": float(last_true[0]),
            "dR_ohm": float(last_true[1]),
            "d_eta_mt": float(last_true[2]),
        },
        "key_soh_pct": round(soh, 1),
        "accept_rate": round(accept_last, 3),
    }
