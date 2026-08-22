<p align="center">
  <strong><code>S E N T I N E L – H₂</code></strong>
</p>

<h3 align="center">
  Multi-Physics Co-Design Architecture for Solid-Acid Fuel Cell Vehicles<br/>
  with Lattice-Engineered Hydride-Hydrate Storage &amp; Exceptional-Point Safety Sensing
</h3>

<p align="center">
  <a href="https://drive.google.com/file/d/1smEpJe-KCvLJNbkREYmiGVrvDrOEtt_x/view?usp=sharing"><strong>📄 Read the Whitepaper</strong></a>
  &nbsp;·&nbsp;
  <a href="#quick-start">Quick Start</a>
  &nbsp;·&nbsp;
  <a href="#the-six-pillars">The Six Pillars</a>
  &nbsp;·&nbsp;
  <a href="#unified-system-functional">Unified Ψ Functional</a>
</p>

---

## What is SENTINEL-H₂?

**S**olid-acid **E**lectrolyte with **N**on-Hermitian sensing, **T**unnelling-enhanced transport, **I**nversion-based diagnostics, **N**euromorphic fusion **E**ngine, and **L**attice-strained **H**ydrate storage for **H₂** mobility.

A fuel-cell-electric passenger vehicle architecture engineered for Indian operating conditions, submitted to the **Suzuki Igniters Innovation Challenge 2026** (Case Study E, Option 1). The architecture integrates six subsystems — each grounded in peer-reviewed physics — into a unified simulation console that evaluates the coupled system performance functional over drive-cycle and climate envelopes.

This repository contains the **interactive simulation dashboard**: a FastAPI backend implementing all six physics modules with real numerical solvers, served to a vanilla HTML/CSS/JS frontend with Plotly visualizations.

---

## Quick Start

```bash
# Install dependencies
cd backend
pip install fastapi uvicorn numpy scipy pydantic

# Run the server
python main.py
# → http://127.0.0.1:8000/
```

No Redis. No Celery. No Docker. Background compute is handled by `ThreadPoolExecutor`.

---

## Architecture

```
SENTINEL-H₂/
├── backend/
│   ├── main.py                          # FastAPI app + static serving
│   ├── job_manager.py                   # ThreadPoolExecutor job queue
│   ├── config.py                        # Configuration constants
│   ├── orchestrator.py                  # Unified Ψ functional (Eq 20)
│   ├── requirements.txt
│   └── pillars/
│       ├── pillar1_electrolyte.py       # WKB tunnelling, ASR model
│       ├── pillar2_hydrate.py           # Kihara cell potential, vdW-P
│       ├── pillar3_gdl.py              # D2Q9 LBM, Shan-Chen two-phase
│       ├── pillar4_ae.py               # Bayesian MCMC AE inversion
│       ├── pillar5_ep.py               # Non-Hermitian EP eigenvalues
│       └── pillar6_hdc.py              # 10K-dim HDC classifier
└── frontend/
    ├── index.html                       # SPA shell
    ├── css/style.css                    # Design system
    └── js/app.js                        # Tab routing, Plotly rendering
```

### Data flow

```
┌──────────────┐      POST /api/pillar/{id}/run       ┌─────────────────────┐
│              │ ──────────────────────────────────── ▶ │   ThreadPoolExecutor │
│   Browser    │                                       │   (JobManager)       │
│   (SPA)      │ ◀ ──── GET /api/jobs/{id}/status ──── │                     │
│              │         { status, result, elapsed }    │   pillar1..6.py     │
└──────────────┘                                       │   orchestrator.py   │
                                                       └─────────────────────┘
```

---

## The Six Pillars

### Pillar 1 — Superprotonic Solid-Acid Electrolyte with Tunnelling-Enhanced Transport

**Module:** [`pillar1_electrolyte.py`](backend/pillars/pillar1_electrolyte.py)

CsH₂PO₄ undergoes a superprotonic phase transition at ~230°C. Below this transition, proton transport proceeds via a Grotthuss mechanism where the O–H···O double-well barrier is narrow enough (~0.1–0.3 Å) for quantum tunnelling to matter. The net hopping rate uses a Bell-type tunnelling correction:

$$k(T) = \kappa(T) \cdot \nu_0 \exp\!\left(-\frac{E_a}{k_B T}\right)$$

where the tunnelling transmission enhancement κ(T) is computed from a one-dimensional WKB integral over the double-well potential:

$$\boxed{\kappa(T) = \frac{1}{k_B T}\int_0^{V_b} \exp\!\left(-\frac{2}{\hbar}\int_{x_1(E)}^{x_2(E)}\sqrt{2m_p\bigl(V(x)-E\bigr)}\,dx\right)\exp\!\left(-\frac{E}{k_B T}\right)dE}$$

- **m_p**: proton mass
- **V(x)**: double-well potential along the O–H···O coordinate
- **V_b**: barrier height
- **x₁(E), x₂(E)**: classical turning points at energy E

The cell voltage model combines Nernst OCV with activation, ohmic, and concentration overpotentials:

$$V_{\text{cell}} = E_{\text{Nernst}} - \eta_{\text{act}} - \eta_{\text{ohm}} - \eta_{\text{conc}}$$

$$\eta_{\text{ohm}} = j \cdot \text{ASR}(T) = j \cdot \frac{t_{\text{electrolyte}}}{\sigma(T)}$$

where σ(T) = σ₀ κ(T) exp(−E_a/k_BT)/T folds in the tunnelling correction.

**Simulation outputs:** Tunnelling factor vs temperature, conductivity Arrhenius plot, polarization curves, ASR vs strain.

---

### Pillar 2 — Defect-Engineered Clathrate-Hydrate Hydrogen Storage

**Module:** [`pillar2_hydrate.py`](backend/pillars/pillar2_hydrate.py)

Structure-II clathrate hydrates store H₂ at dramatically reduced pressure (5 MPa vs 300 MPa for pure H₂ hydrate) when a promoter guest (THF) co-occupies the large 5¹²6⁴ cages. The thermodynamics follow the van der Waals–Platteeuw statistical model.

**Cage occupancy** (Langmuir adsorption):

$$\theta_{i,g} = \frac{C_{i,g}\, f_g}{1 + \sum_{g'} C_{i,g'}\, f_{g'}}$$

where f_g is guest fugacity and C_{i,g}(T) is the Langmuir constant computed from a **Kihara cell-potential integral** over the cage geometry:

$$C_{i,g}(T) = \frac{4\pi}{k_B T}\int_0^{R_{\text{cage}}} \exp\!\left(-\frac{w(r)}{k_B T}\right) r^2\, dr$$

**Water chemical potential change** on hydrate formation:

$$\frac{\Delta\mu_w^{\,\beta-H}}{RT} = -\sum_i \nu_i \ln\!\left(1 - \sum_g \theta_{i,g}\right)$$

with ν_small = 2/17, ν_large = 1/17 for sII.

**Gravimetric storage capacity:**

$$\text{wt\% H}_2 = \frac{n_{H_2} M_{H_2}}{n_{H_2} M_{H_2} + n_{\text{promoter}} M_{\text{promoter}} + n_{\text{water}} M_{\text{water}}} \times 100$$

The **novel engineering proposal**: controlled ion-beam defect pre-treatment elongates cage radii by up to 3%, enabling multi-occupancy of small cages and shifting practical wt% from ~1% to the 3.5–4.5% range at 5–12 MPa.

**Simulation outputs:** Occupancy vs pressure isotherms, wt% vs defect dose, P-T stability heatmap.

---

### Pillar 3 — Lattice-Boltzmann Two-Phase GDL Simulation

**Module:** [`pillar3_gdl.py`](backend/pillars/pillar3_gdl.py)

The gas diffusion layer governs reactant delivery and liquid-water rejection. India's climate extremes (5–50°C, 10–95% RH) demand a graded porosity design optimized across the full envelope.

**Throat conductance** (Hagen–Poiseuille):

$$g_{ij} = \frac{\pi r_{ij}^4}{8\mu L_{ij}}$$

**Capillary entry pressure** (Young–Laplace):

$$P_c = \frac{2\gamma\cos\theta}{r_{ij}}$$

**D2Q9 BGK streaming-collision** (Lattice Boltzmann):

$$f_i(\mathbf{x}+\mathbf{e}_i\Delta t,\, t+\Delta t) - f_i(\mathbf{x},t) = -\frac{1}{\tau}\bigl[f_i(\mathbf{x},t) - f_i^{\text{eq}}(\mathbf{x},t)\bigr]$$

with relaxation time τ related to kinematic viscosity by ν = c_s²(τ − 1/2)Δt.

**Shan-Chen pseudopotential** (two-phase interaction):

$$\mathbf{F}(\mathbf{x}) = -G\,\psi(\mathbf{x})\sum_i w_i\,\psi(\mathbf{x}+\mathbf{e}_i)\,\mathbf{e}_i$$

where ψ(ρ) = 1 − exp(−ρ). The porosity is graded from ~55% at the channel interface to ~75% at the catalyst layer.

**Simulation outputs:** Density field heatmap, saturation vs time, porosity profile, T×RH flooding probability heatmap.

---

### Pillar 4 — Acoustic-Emission Bayesian Inversion for Stack State-of-Health

**Module:** [`pillar4_ae.py`](backend/pillars/pillar4_ae.py)

Non-invasive health monitoring via acoustic emission, formulated as a Bayesian inverse problem.

**Forward degradation model:**

$$V(t) = V_0 - k_1\,\Delta\text{ECSA}(t) - k_2\,\Delta R_\Omega(t) - k_3\,\Delta\eta_{\text{mt}}(t)$$

where:
- ΔECSA(t) = k_ecsa · √t (Pt dissolution, diffusion-limited)
- ΔR_Ω(t) = k_ohm · t (membrane thinning, linear)
- Δη_mt(t) = k_mt · t^1.2 (GDL/PTFE aging, superlinear)

**Bayesian inversion** via Metropolis-Hastings MCMC:

$$p\bigl(\boldsymbol{\theta}_{\text{deg}} \mid \mathbf{y}_{AE}\bigr) \;\propto\; p\bigl(\mathbf{y}_{AE} \mid \boldsymbol{\theta}_{\text{deg}}\bigr)\, p\bigl(\boldsymbol{\theta}_{\text{deg}}\bigr)$$

The likelihood p(y_AE | θ_deg) is a forward AE-signature model mapping degradation state → acoustic features (hit rate, energy, b-value). The prior p(θ_deg) regularizes via the physics-informed voltage model.

The MCMC sampler recovers the posterior distribution of degradation parameters from noisy AE observations, yielding a continuously updated SoH estimate.

**Simulation outputs:** Voltage decay trajectory, posterior mean vs ground truth, posterior histograms, MCMC trace.

---

### Pillar 5 — Non-Hermitian Exceptional-Point Sensor Network

**Module:** [`pillar5_ep.py`](backend/pillars/pillar5_ep.py)

Two coupled resonators with unequal loss form a non-Hermitian system whose eigenvalues are:

$$\boxed{\omega_{\pm} = \frac{\omega_1+\omega_2}{2} - i\frac{\gamma_1+\gamma_2}{4} \pm \sqrt{\kappa^2 + \left(\frac{\Delta\omega - i\Delta\gamma/2}{2}\right)^2}}$$

At the **exceptional point** (EP), the discriminant vanishes:

$$\kappa^2 + \left(\frac{-i\Delta\gamma/2}{2}\right)^2 = 0 \quad\Rightarrow\quad \Delta\gamma = 4\kappa, \quad \Delta\omega = 0$$

Near the EP, a perturbation δκ (from H₂ absorption into a Pd film changing resonator loss) produces eigenvalue splitting that scales as:

$$\Delta\omega_{\text{split}} \propto \sqrt{\delta\kappa}$$

The differential sensitivity **diverges** as δκ → 0:

$$\frac{d(\Delta\omega_{\text{split}})}{d(\delta\kappa)} \;\to\; \infty$$

This is the key advantage over conventional linear-response sensors: the sensor is most sensitive exactly in the weak-perturbation, near-threshold regime that matters for early leak detection (before reaching the 4% LEL).

**Pd-H₂ transduction model:** H₂ concentration → δκ via Sievert's law:

$$\delta\kappa = \alpha \cdot c_{H_2}^\beta$$

**Verified:** Log-log slope of splitting vs perturbation = **0.505** (confirms √δκ scaling).

**Simulation outputs:** Eigenvalue splitting (log-log), complex-plane trajectories, sensitivity gain vs perturbation, detection limit vs noise floor, H₂ concentration response.

---

### Pillar 6 — Neuromorphic Hyperdimensional Computing Fusion Engine

**Module:** [`pillar6_hdc.py`](backend/pillars/pillar6_hdc.py)

All sensor streams are fused into a single hazard classification using 10,000-dimensional hypervectors. HDC represents information as pseudo-random bipolar vectors combined via hardware-cheap operations.

**Encoding:** Each sensor channel k is encoded into a bipolar hypervector H_k ∈ {−1, +1}^D via thermometer encoding, then bound with a fixed role vector:

$$\mathbf{B}_k = \mathbf{H}_k \odot \mathbf{R}_k$$

**Bundling** (majority-sum across all K channels):

$$\mathbf{S}(t) = \text{sign}\!\left(\sum_{k=1}^{K} \mathbf{B}_k(t)\right)$$

**Classification** by cosine similarity against learned prototypes:

$$\hat{c} = \arg\max_{c}\; \frac{\mathbf{S}(t)\cdot \mathbf{P}_c}{\lVert\mathbf{S}(t)\rVert\,\lVert\mathbf{P}_c\rVert}$$

**Online prototype adaptation** (single-pass, no backpropagation):

$$\mathbf{P}_c \leftarrow \mathbf{P}_c + \eta\bigl(\mathbf{S}(t) - \mathbf{P}_c\bigr)$$

Five hazard classes:

| Class | Meaning |
|-------|---------|
| `NORMAL` | All systems nominal |
| `LEAK_MINOR` | Sub-LEL H₂ detected |
| `LEAK_MAJOR` | Near-LEL H₂ concentration |
| `STACK_DEGRADING` | AE-inferred SoH decline |
| `FIRE_PRECURSOR` | Joint leak + degradation + thermal anomaly |

**Simulation outputs:** Confusion matrix, per-class accuracy, cosine similarity distributions, prototype adaptation convergence.

---

## Unified System Functional

The six pillars are coupled — not merely juxtaposed — through a single scalar **system safety-performance functional** Ψ, evaluated over the drive-cycle time horizon [0, T] and climate envelope Ω:

$$\boxed{\Psi = \int_0^{\mathcal{T}}\!\!\int_\Omega \underbrace{\bigl[\kappa(T)\,\sigma_0\,e^{-E_a/k_BT}\bigr]}_{\text{§3: tunnelling transport}} \cdot \underbrace{\bigl[1-\sum_g\theta_{i,g}(P,T)\bigr]^{-1}}_{\text{§4: hydrate storage margin}} \cdot \underbrace{\exp\!\bigl[-\tfrac{1}{2}(\Delta\omega_{\text{split}})^{-2}\bigr]}_{\text{§7: EP leak sensitivity}} \cdot \underbrace{\mathcal{L}(\mathbf{y}_{AE}\mid\boldsymbol{\theta}_{\text{deg}})}_{\text{§6: AE likelihood}} \;d\Omega\,dt \;-\; \lambda\!\sum_{c\in\mathcal{P}}\lVert\mathbf{S}(t)-\mathbf{P}_c\rVert_H}$$

**Module:** [`orchestrator.py`](backend/orchestrator.py) — evaluates Ψ numerically over simulated drive-cycles.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/pillar/{id}/run` | Submit simulation job (id: 1–6) |
| `POST` | `/api/system/run` | Submit unified Ψ evaluation |
| `GET` | `/api/jobs/{job_id}/status` | Poll job status and retrieve results |
| `GET` | `/` | Serve the simulation console UI |

**Request body:**
```json
{
  "params": {
    "T": 200,
    "strain": 0.05
  }
}
```

**Response (status endpoint):**
```json
{
  "status": "completed",
  "elapsed": 3.2,
  "result": { ... },
  "pillar": "Solid-Acid Electrolyte"
}
```

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.13, FastAPI, Uvicorn |
| Compute | NumPy, SciPy (quad integration, MCMC) |
| Concurrency | `concurrent.futures.ThreadPoolExecutor` |
| Frontend | Vanilla HTML/CSS/JS, Plotly.js |
| Fonts | Newsreader (serif), Public Sans (UI), Fragment Mono (data) |

---

## Design Philosophy

The frontend follows a strict **Anti-Slop Contract** — an instrument-panel aesthetic inspired by flight-deck and laboratory-rack interfaces:

- **No** purple/blue gradients, no glassmorphism, no generic rounded shadow-cards
- **Asymmetric rail layout**: left navigation rail + split control/output panels
- **Semantic accent**: copper-amber `#FF7A1A` used only for LIVE/active states
- **Typography hierarchy**: Newsreader for headings, Public Sans for UI, Fragment Mono for numerical readouts
- **Real data-driven motion**: charts animate only when data arrives, never decoratively

---

## References

Key papers underpinning the physics:

1. Lee & Tuckerman (2008). *Structure and Proton Transport in Superprotonic CsH₂PO₄.* J. Phys. Chem. C, 112, 9917–9930.
2. Singh et al. (2022). *Nanocomposite CsH₂PO₄/NaH₂PO₄/TiO₂ proton conductor.* Results in Chemistry, 4, 100262.
3. Florusse et al. (2004). *Stable Low-Pressure H₂ Clusters in Binary Clathrate Hydrate.* Science, 306(5695), 469–471.
4. Lee et al. (2005). *Tuning clathrate hydrates for hydrogen storage.* Nature, 434, 743–746.
5. Hondo et al. (2025). *H₂ storage via clathrate hydrates under mild conditions.* Chem. Eng. J., 524, 169469.
6. Chen et al. (2017). *Exceptional points enhance sensing in an optical microcavity.* Nature, 548, 192–196.
7. Hodaei et al. (2017). *Enhanced sensitivity at higher-order exceptional points.* Nature, 548, 187–191.
8. Wiersig (2016). *Sensors operating at exceptional points.* Phys. Rev. A, 93, 033809.
9. HDC Berkeley (2022). *Low-Power Hyperdimensional Computing Processors for Sensor Fusion.* EECS-2022-118.
10. U.S. DOE (2025). *Onboard Type IV Compressed H₂ Storage — Cost & Performance.* Record #24006.

---

## Authors

**Mitrajit Ghorui** · **Dhruv Vaghela** · **Ishan Nepal**

Team SENTINEL-H₂ — Suzuki Igniters Innovation Challenge 2026

---

## License

See [LICENSE](LICENSE) for details.
