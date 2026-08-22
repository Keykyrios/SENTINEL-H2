/* ============================================================
   SENTINEL-H₂ Simulation Console — Application Logic
   Tab routing, API calls, job polling, Plotly chart rendering.
   ============================================================ */

(function () {
  'use strict';

  // --- Plotly theme matching design tokens ---
  const PLOT_BG = '#16170F';
  const PLOT_PAPER = '#16170F';
  const PLOT_GRID = 'rgba(232,228,214,0.06)';
  const PLOT_LINE = 'rgba(232,228,214,0.15)';
  const PLOT_TEXT = '#A9A697';
  const PLOT_INK = '#E8E4D6';
  const ACCENT = '#FF7A1A';
  const ACCENT_DIM = '#7A4A20';
  const OK_CLR = '#6FA96B';
  const WARN_CLR = '#D9A441';
  const DANGER_CLR = '#C4523A';

  const LAYOUT_BASE = {
    paper_bgcolor: PLOT_PAPER,
    plot_bgcolor: PLOT_BG,
    font: { family: "'Fragment Mono', monospace", size: 11, color: PLOT_TEXT },
    margin: { l: 50, r: 20, t: 32, b: 40 },
    xaxis: { gridcolor: PLOT_GRID, linecolor: PLOT_LINE, zerolinecolor: PLOT_LINE, tickfont: { size: 10 } },
    yaxis: { gridcolor: PLOT_GRID, linecolor: PLOT_LINE, zerolinecolor: PLOT_LINE, tickfont: { size: 10 } },
  };

  const PLOT_CFG = { responsive: true, displayModeBar: false };

  function layout(overrides) {
    const base = JSON.parse(JSON.stringify(LAYOUT_BASE));
    return deepMerge(base, overrides || {});
  }

  function deepMerge(target, source) {
    for (const key of Object.keys(source)) {
      if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
        if (!target[key]) target[key] = {};
        deepMerge(target[key], source[key]);
      } else {
        target[key] = source[key];
      }
    }
    return target;
  }

  // --- Logging ---
  const logBody = document.getElementById('log-body');
  function log(msg, cls) {
    const now = new Date();
    const ts = now.toLocaleTimeString('en-GB', { hour12: false });
    const line = document.createElement('div');
    line.className = 'log-line';
    line.innerHTML = `<span class="log-time">${ts}</span><span class="log-msg ${cls || ''}">${msg}</span>`;
    logBody.prepend(line);
  }

  // --- Tab Routing ---
  const railItems = document.querySelectorAll('.rail-item');
  const views = document.querySelectorAll('.pillar-view');

  function switchTab(pillar) {
    railItems.forEach(r => r.classList.toggle('active', r.dataset.pillar === String(pillar)));
    views.forEach(v => {
      const viewId = v.id.replace('view-', '');
      v.classList.toggle('active', viewId === String(pillar));
    });
  }

  railItems.forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.pillar));
  });

  // --- API ---
  const API = '';

  async function postRun(endpoint, params) {
    const res = await fetch(API + endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ params }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  async function getStatus(jobId) {
    const res = await fetch(API + `/api/jobs/${jobId}/status`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  // --- Job Polling ---
  const statusDot = document.getElementById('global-status-dot');
  const stripStatus = document.getElementById('strip-status');

  function setGlobalStatus(state) {
    statusDot.className = 'status-dot ' + state;
    stripStatus.textContent = state === 'running' ? 'RUNNING' : state === 'done' ? 'COMPLETE' : state === 'error' ? 'FAULT' : 'IDLE';
  }

  async function runAndPoll(endpoint, params, onComplete, btnEl) {
    btnEl.classList.add('running');
    btnEl.textContent = 'RUNNING…';
    setGlobalStatus('running');

    try {
      const { job_id } = await postRun(endpoint, params);
      log(`Job submitted: ${job_id.slice(0, 8)}…`);

      const poll = async () => {
        const status = await getStatus(job_id);
        if (status.status === 'completed') {
          setGlobalStatus('done');
          btnEl.classList.remove('running');
          btnEl.textContent = 'RUN SIMULATION';
          log(`Job complete (${status.elapsed}s)`, 'ok');
          onComplete(status.result);
        } else if (status.status === 'failed') {
          setGlobalStatus('error');
          btnEl.classList.remove('running');
          btnEl.textContent = 'RUN SIMULATION';
          log(`Job failed: ${(status.error || '').slice(0, 120)}`, 'error');
        } else {
          setTimeout(poll, 600);
        }
      };
      setTimeout(poll, 400);
    } catch (err) {
      setGlobalStatus('error');
      btnEl.classList.remove('running');
      btnEl.textContent = 'RUN SIMULATION';
      log(`Error: ${err.message}`, 'error');
    }
  }

  // --- Slider Binding Helper ---
  function bindSlider(id, valId, fmt) {
    const slider = document.getElementById(id);
    const valEl = document.getElementById(valId);
    if (!slider || !valEl) return;
    const update = () => { valEl.textContent = fmt(slider.value); };
    slider.addEventListener('input', update);
    update();
  }

  // ============================================================
  // PILLAR 1 — Electrolyte
  // ============================================================
  bindSlider('p1-vb', 'p1-vb-val', v => parseFloat(v).toFixed(2) + ' eV');
  bindSlider('p1-a', 'p1-a-val', v => parseFloat(v).toFixed(3) + ' Å');
  bindSlider('p1-strain', 'p1-strain-val', v => parseFloat(v).toFixed(1) + ' %');

  document.getElementById('run-p1').addEventListener('click', function () {
    const params = {
      V_b: parseFloat(document.getElementById('p1-vb').value),
      a: parseFloat(document.getElementById('p1-a').value),
      strain_pct: parseFloat(document.getElementById('p1-strain').value),
      T_min: parseFloat(document.getElementById('p1-tmin').value),
      T_max: parseFloat(document.getElementById('p1-tmax').value),
    };
    runAndPoll('/api/pillar/1/run', params, renderP1, this);
  });

  function renderP1(d) {
    // Chart 1: kappa(T)
    const traces1 = [{ x: d.temperatures_C, y: d.kappa_no_strain, mode: 'lines', line: { color: PLOT_INK, width: 1.5 }, name: 'Unstrained' }];
    if (d.kappa_strained) traces1.push({ x: d.temperatures_C, y: d.kappa_strained, mode: 'lines', line: { color: ACCENT, width: 1.5 }, name: `Strained (${d.strain_pct}%)` });
    Plotly.newPlot('chart-p1-kappa', traces1, layout({ title: { text: 'κ(T) Tunnelling Enhancement', font: { size: 12 } }, xaxis: { title: 'T (°C)' }, yaxis: { title: 'κ', type: 'log' } }), PLOT_CFG);

    // Chart 2: sigma(T)
    const traces2 = [{ x: d.temperatures_C, y: d.sigma_no_strain, mode: 'lines', line: { color: PLOT_INK, width: 1.5 }, name: 'Unstrained' }];
    if (d.sigma_strained) traces2.push({ x: d.temperatures_C, y: d.sigma_strained, mode: 'lines', line: { color: ACCENT, width: 1.5 }, name: 'Strained' });
    Plotly.newPlot('chart-p1-sigma', traces2, layout({ title: { text: 'σ(T) Conductivity', font: { size: 12 } }, xaxis: { title: 'T (°C)' }, yaxis: { title: 'S/cm', type: 'log' } }), PLOT_CFG);

    // Chart 3: Polarization curve
    const traces3 = [{ x: d.j_arr, y: d.vcell_no_strain, mode: 'lines', line: { color: PLOT_INK, width: 1.5 }, name: 'Unstrained' }];
    if (d.vcell_strained) traces3.push({ x: d.j_arr, y: d.vcell_strained, mode: 'lines', line: { color: ACCENT, width: 1.5 }, name: 'Strained' });
    Plotly.newPlot('chart-p1-vcell', traces3, layout({ title: { text: `V-j Curve at ${d.polarization_temp_C}°C`, font: { size: 12 } }, xaxis: { title: 'j (A/cm²)' }, yaxis: { title: 'V (V)' } }), PLOT_CFG);

    // Chart 4: ASR vs T
    const traces4 = [{ x: d.temperatures_C, y: d.asr_no_strain, mode: 'lines', line: { color: PLOT_INK, width: 1.5 }, name: 'Unstrained' }];
    if (d.asr_strained) traces4.push({ x: d.temperatures_C, y: d.asr_strained, mode: 'lines', line: { color: ACCENT, width: 1.5 }, name: 'Strained' });
    // Reference line at 0.3 ohm*cm^2
    traces4.push({ x: [d.temperatures_C[0], d.temperatures_C[d.temperatures_C.length - 1]], y: [0.3, 0.3], mode: 'lines', line: { color: WARN_CLR, width: 1, dash: 'dot' }, name: 'Target' });
    Plotly.newPlot('chart-p1-asr', traces4, layout({ title: { text: 'ASR(T)', font: { size: 12 } }, xaxis: { title: 'T (°C)' }, yaxis: { title: 'Ω·cm²', type: 'log' } }), PLOT_CFG);

    document.getElementById('metric-p1').innerHTML = `${d.key_asr} <span class="metric-unit">Ω·cm²</span>`;
  }

  // ============================================================
  // PILLAR 2 — Hydrate Storage
  // ============================================================
  bindSlider('p2-temp', 'p2-temp-val', v => parseFloat(v).toFixed(1) + ' K');
  bindSlider('p2-pratio', 'p2-pratio-val', v => parseFloat(v).toFixed(2));
  bindSlider('p2-dose', 'p2-dose-val', v => parseFloat(v).toFixed(2));

  document.getElementById('run-p2').addEventListener('click', function () {
    const params = {
      T: parseFloat(document.getElementById('p2-temp').value),
      P_min: parseFloat(document.getElementById('p2-pmin').value),
      P_max: parseFloat(document.getElementById('p2-pmax').value),
      promoter_ratio: parseFloat(document.getElementById('p2-pratio').value),
      defect_dose: parseFloat(document.getElementById('p2-dose').value),
    };
    runAndPoll('/api/pillar/2/run', params, renderP2, this);
  });

  function renderP2(d) {
    // Chart 1: Cage occupancy vs P
    Plotly.newPlot('chart-p2-occupancy', [
      { x: d.pressures_MPa, y: d.theta_H2_small, mode: 'lines', line: { color: PLOT_INK, width: 1.5 }, name: 'θ H₂ small' },
      { x: d.pressures_MPa, y: d.theta_H2_large, mode: 'lines', line: { color: ACCENT, width: 1.5 }, name: 'θ H₂ large' },
      { x: d.pressures_MPa, y: d.theta_promoter_large, mode: 'lines', line: { color: WARN_CLR, width: 1.5 }, name: 'θ promoter' },
    ], layout({ title: { text: 'Cage Occupancy vs P', font: { size: 12 } }, xaxis: { title: 'P (MPa)' }, yaxis: { title: 'θ' } }), PLOT_CFG);

    // Chart 2: wt% vs P
    Plotly.newPlot('chart-p2-wt', [
      { x: d.pressures_MPa, y: d.wt_pct_vs_P, mode: 'lines', line: { color: ACCENT, width: 1.5 }, name: 'wt% H₂' },
    ], layout({ title: { text: 'Gravimetric Capacity vs P', font: { size: 12 } }, xaxis: { title: 'P (MPa)' }, yaxis: { title: 'wt% H₂' } }), PLOT_CFG);

    // Chart 3: wt% vs defect dose
    Plotly.newPlot('chart-p2-dose', [
      { x: d.dose_arr, y: d.wt_vs_dose, mode: 'lines', line: { color: ACCENT, width: 1.5 } },
    ], layout({ title: { text: 'Capacity vs Defect Dose', font: { size: 12 } }, xaxis: { title: 'Dose (normalized)' }, yaxis: { title: 'wt% H₂' } }), PLOT_CFG);

    // Chart 4: P-T stability heatmap
    Plotly.newPlot('chart-p2-stability', [{
      z: d.wt_heatmap, x: d.P_stability, y: d.T_stability, type: 'heatmap',
      colorscale: [[0, PLOT_BG], [1, ACCENT]], showscale: true,
      colorbar: { tickfont: { size: 10 }, title: { text: 'wt%', font: { size: 10 } } },
    }], layout({ title: { text: 'P-T Stability (wt%)', font: { size: 12 } }, xaxis: { title: 'P (MPa)' }, yaxis: { title: 'T (K)' } }), PLOT_CFG);

    document.getElementById('metric-p2').innerHTML = `${d.key_wt_pct} <span class="metric-unit">wt% H₂</span>`;
  }

  // ============================================================
  // PILLAR 3 — GDL / LBM
  // ============================================================
  bindSlider('p3-temp', 'p3-temp-val', v => v + ' °C');
  bindSlider('p3-rh', 'p3-rh-val', v => v + ' %');
  bindSlider('p3-steps', 'p3-steps-val', v => v);

  document.getElementById('run-p3').addEventListener('click', function () {
    const params = {
      Nx: parseInt(document.getElementById('p3-nx').value),
      Ny: parseInt(document.getElementById('p3-ny').value),
      porosity_bottom: parseFloat(document.getElementById('p3-por-bot').value),
      porosity_top: parseFloat(document.getElementById('p3-por-top').value),
      T_ambient: parseFloat(document.getElementById('p3-temp').value),
      RH: parseFloat(document.getElementById('p3-rh').value),
      n_steps: parseInt(document.getElementById('p3-steps').value),
    };
    runAndPoll('/api/pillar/3/run', params, renderP3, this);
  });

  function renderP3(d) {
    // Chart 1: Saturation heatmap (rho field)
    Plotly.newPlot('chart-p3-satmap', [{
      z: d.rho_field, type: 'heatmap',
      colorscale: [[0, '#1D1E14'], [0.3, ACCENT_DIM], [0.7, ACCENT], [1, PLOT_INK]],
      showscale: true, colorbar: { tickfont: { size: 10 }, title: { text: 'ρ', font: { size: 10 } } },
    }], layout({ title: { text: 'Water Saturation Map', font: { size: 12 } }, xaxis: { title: 'x' }, yaxis: { title: 'y' } }), PLOT_CFG);

    // Chart 2: Saturation vs time
    const satT = d.saturation_history.map(s => s.step);
    const satV = d.saturation_history.map(s => s.saturation);
    Plotly.newPlot('chart-p3-sattime', [
      { x: satT, y: satV, mode: 'lines', line: { color: ACCENT, width: 1.5 } },
    ], layout({ title: { text: 'Saturation vs Step', font: { size: 12 } }, xaxis: { title: 'LBM Step' }, yaxis: { title: 'Saturation' } }), PLOT_CFG);

    // Chart 3: Porosity profile
    const yIdx = d.porosity_profile.map((_, i) => i);
    Plotly.newPlot('chart-p3-porosity', [
      { x: d.porosity_profile, y: yIdx, mode: 'lines', line: { color: PLOT_INK, width: 1.5 }, orientation: 'h' },
    ], layout({ title: { text: 'Porosity Profile', font: { size: 12 } }, xaxis: { title: 'Porosity' }, yaxis: { title: 'Through-thickness' } }), PLOT_CFG);

    // Chart 4: Climate flood probability heatmap
    Plotly.newPlot('chart-p3-climate', [{
      z: d.flood_probability, x: d.RH_sweep, y: d.T_sweep, type: 'heatmap',
      colorscale: [[0, PLOT_BG], [1, DANGER_CLR]], showscale: true,
      colorbar: { tickfont: { size: 10 }, title: { text: 'Flood P', font: { size: 10 } } },
    }], layout({ title: { text: 'Flood Probability (T × RH)', font: { size: 12 } }, xaxis: { title: 'RH (%)' }, yaxis: { title: 'T (°C)' } }), PLOT_CFG);

    document.getElementById('metric-p3').innerHTML = `${d.final_saturation} <span class="metric-unit">fraction</span>`;
  }

  // ============================================================
  // PILLAR 4 — AE Inversion
  // ============================================================
  bindSlider('p4-tmax', 'p4-tmax-val', v => v + ' h');
  bindSlider('p4-noise', 'p4-noise-val', v => parseFloat(v).toFixed(2));
  bindSlider('p4-mcmc', 'p4-mcmc-val', v => v);
  bindSlider('p4-v0', 'p4-v0-val', v => parseFloat(v).toFixed(2) + ' V');

  document.getElementById('run-p4').addEventListener('click', function () {
    const params = {
      t_max_hours: parseFloat(document.getElementById('p4-tmax').value),
      noise_std: parseFloat(document.getElementById('p4-noise').value),
      n_mcmc: parseInt(document.getElementById('p4-mcmc').value),
      V0: parseFloat(document.getElementById('p4-v0').value),
    };
    runAndPoll('/api/pillar/4/run', params, renderP4, this);
  });

  function renderP4(d) {
    // Chart 1: Voltage decay
    Plotly.newPlot('chart-p4-voltage', [
      { x: d.time_hours, y: d.voltage_decay, mode: 'lines', line: { color: PLOT_INK, width: 1.5 }, name: 'V(t)' },
    ], layout({ title: { text: 'Voltage Decay', font: { size: 12 } }, xaxis: { title: 'Hours' }, yaxis: { title: 'V (V)' } }), PLOT_CFG);

    // Chart 2: Posterior recovery vs ground truth
    Plotly.newPlot('chart-p4-recovery', [
      { x: d.time_hours, y: d.dECSA_true, mode: 'lines', line: { color: PLOT_INK, width: 1, dash: 'dot' }, name: 'True ΔECSA' },
      { x: d.inversion_times, y: d.posterior_means.dECSA, mode: 'markers', marker: { color: ACCENT, size: 6 }, error_y: { type: 'data', array: d.posterior_stds.dECSA, visible: true, color: ACCENT_DIM }, name: 'Posterior ΔECSA' },
    ], layout({ title: { text: 'ΔECSA Recovery', font: { size: 12 } }, xaxis: { title: 'Hours' }, yaxis: { title: 'ΔECSA' } }), PLOT_CFG);

    // Chart 3: Degradation channels
    Plotly.newPlot('chart-p4-posterior', [
      { x: d.time_hours, y: d.dECSA_true, mode: 'lines', line: { color: ACCENT, width: 1.5 }, name: 'ΔECSA' },
      { x: d.time_hours, y: d.dR_true, mode: 'lines', line: { color: WARN_CLR, width: 1.5 }, name: 'ΔR_Ω' },
      { x: d.time_hours, y: d.dMT_true, mode: 'lines', line: { color: OK_CLR, width: 1.5 }, name: 'Δη_mt' },
    ], layout({ title: { text: 'Degradation Channels', font: { size: 12 } }, xaxis: { title: 'Hours' }, yaxis: { title: 'Magnitude' } }), PLOT_CFG);

    // Chart 4: Posterior histogram at end
    const hE = d.hist_ecsa;
    const binCenters = hE.edges.slice(0, -1).map((e, i) => (e + hE.edges[i + 1]) / 2);
    Plotly.newPlot('chart-p4-hist', [
      { x: binCenters, y: hE.counts, type: 'bar', marker: { color: ACCENT_DIM }, name: 'ΔECSA posterior' },
      { x: [d.true_at_end.dECSA, d.true_at_end.dECSA], y: [0, Math.max(...hE.counts)], mode: 'lines', line: { color: ACCENT, width: 2 }, name: 'True value' },
    ], layout({ title: { text: 'ΔECSA Posterior at End', font: { size: 12 } }, xaxis: { title: 'ΔECSA' }, yaxis: { title: 'Count' }, bargap: 0.05 }), PLOT_CFG);

    document.getElementById('metric-p4').innerHTML = `${d.key_soh_pct} <span class="metric-unit">%</span>`;
  }

  // ============================================================
  // PILLAR 5 — EP Sensor
  // ============================================================
  bindSlider('p5-kappa', 'p5-kappa-val', v => parseFloat(v).toFixed(3));
  bindSlider('p5-noise', 'p5-noise-val', v => parseFloat(v).toExponential(1));

  document.getElementById('run-p5').addEventListener('click', function () {
    const params = {
      kappa_base: parseFloat(document.getElementById('p5-kappa').value),
      dk_min_log: parseFloat(document.getElementById('p5-dkmin').value),
      dk_max_log: parseFloat(document.getElementById('p5-dkmax').value),
      noise_floor: parseFloat(document.getElementById('p5-noise').value),
    };
    runAndPoll('/api/pillar/5/run', params, renderP5, this);
  });

  function renderP5(d) {
    // Chart 1: Splitting vs delta_kappa (log-log)
    Plotly.newPlot('chart-p5-splitting', [
      { x: d.dk_arr, y: d.splitting, mode: 'lines', line: { color: ACCENT, width: 1.5 }, name: `EP (slope ≈ ${d.fitted_slope})` },
      { x: d.dk_arr, y: d.linear_ref, mode: 'lines', line: { color: PLOT_TEXT, width: 1, dash: 'dot' }, name: 'Linear ref' },
    ], layout({ title: { text: 'Eigenvalue Splitting vs δκ', font: { size: 12 } }, xaxis: { title: 'δκ', type: 'log' }, yaxis: { title: '|Δω|', type: 'log' } }), PLOT_CFG);

    // Chart 2: Complex plane trajectories
    Plotly.newPlot('chart-p5-complex', [
      { x: d.traj_plus_re, y: d.traj_plus_im, mode: 'lines', line: { color: ACCENT, width: 1.5 }, name: 'ω₊' },
      { x: d.traj_minus_re, y: d.traj_minus_im, mode: 'lines', line: { color: PLOT_INK, width: 1.5 }, name: 'ω₋' },
    ], layout({ title: { text: 'Eigenvalue Trajectories', font: { size: 12 } }, xaxis: { title: 'Re(ω)' }, yaxis: { title: 'Im(ω)' } }), PLOT_CFG);

    // Chart 3: Sensitivity gain
    Plotly.newPlot('chart-p5-gain', [
      { x: d.dk_arr, y: d.sensitivity_ep, mode: 'lines', line: { color: ACCENT, width: 1.5 }, name: 'EP sensor' },
      { x: d.dk_arr, y: d.sensitivity_linear, mode: 'lines', line: { color: PLOT_TEXT, width: 1, dash: 'dot' }, name: 'Linear' },
    ], layout({ title: { text: 'Sensitivity Gain', font: { size: 12 } }, xaxis: { title: 'δκ', type: 'log' }, yaxis: { title: 'Gain (split/δκ)', type: 'log' } }), PLOT_CFG);

    // Chart 4: Detection limit vs noise
    Plotly.newPlot('chart-p5-detect', [
      { x: d.noise_levels, y: d.min_detectable_ppm, mode: 'lines+markers', line: { color: ACCENT, width: 1.5 }, marker: { size: 4 }, name: 'Min H₂ (ppm)' },
    ], layout({ title: { text: 'Detection Limit vs Noise', font: { size: 12 } }, xaxis: { title: 'Noise Floor', type: 'log' }, yaxis: { title: 'Min H₂ (ppm)', type: 'log' } }), PLOT_CFG);

    document.getElementById('metric-p5').innerHTML = `${d.key_gain} <span class="metric-unit">×</span>`;
  }

  // ============================================================
  // PILLAR 6 — HDC Fusion
  // ============================================================
  bindSlider('p6-ntrain', 'p6-ntrain-val', v => v);
  bindSlider('p6-ntest', 'p6-ntest-val', v => v);
  bindSlider('p6-eta', 'p6-eta-val', v => parseFloat(v).toFixed(2));
  bindSlider('p6-noise', 'p6-noise-val', v => parseFloat(v).toFixed(2));
  bindSlider('p6-retrain', 'p6-retrain-val', v => v);

  document.getElementById('run-p6').addEventListener('click', function () {
    const params = {
      n_train: parseInt(document.getElementById('p6-ntrain').value),
      n_test: parseInt(document.getElementById('p6-ntest').value),
      eta: parseFloat(document.getElementById('p6-eta').value),
      noise_level: parseFloat(document.getElementById('p6-noise').value),
      n_retrain: parseInt(document.getElementById('p6-retrain').value),
    };
    runAndPoll('/api/pillar/6/run', params, renderP6, this);
  });

  function renderP6(d) {
    const cls = d.class_names;

    // Chart 1: Confusion matrix
    Plotly.newPlot('chart-p6-confusion', [{
      z: d.confusion_matrix, x: cls, y: cls, type: 'heatmap',
      colorscale: [[0, PLOT_BG], [0.5, ACCENT_DIM], [1, ACCENT]],
      showscale: false,
      text: d.confusion_matrix.map(row => row.map(String)),
      texttemplate: '%{text}', textfont: { size: 11, color: PLOT_INK },
    }], layout({ title: { text: 'Confusion Matrix', font: { size: 12 } }, xaxis: { title: 'Predicted', tickangle: -45 }, yaxis: { title: 'True', autorange: 'reversed' } }), PLOT_CFG);

    // Chart 2: Accuracy over retraining passes
    const passes = d.accuracy_over_iterations.map(a => a.pass);
    const accs = d.accuracy_over_iterations.map(a => a.accuracy);
    Plotly.newPlot('chart-p6-accuracy', [
      { x: passes, y: accs, mode: 'lines+markers', line: { color: ACCENT, width: 1.5 }, marker: { size: 5 } },
    ], layout({ title: { text: 'Accuracy vs Retrain Pass', font: { size: 12 } }, xaxis: { title: 'Pass', dtick: 1 }, yaxis: { title: 'Accuracy (%)', range: [0, 100] } }), PLOT_CFG);

    // Chart 3: Per-class accuracy
    Plotly.newPlot('chart-p6-perclass', [
      { x: cls, y: d.per_class_accuracy, type: 'bar', marker: { color: ACCENT_DIM } },
    ], layout({ title: { text: 'Per-Class Accuracy', font: { size: 12 } }, xaxis: { tickangle: -30 }, yaxis: { title: '%', range: [0, 100] } }), PLOT_CFG);

    // Chart 4: Accuracy vs noise
    const noiseLevels = d.acc_vs_noise.map(a => a.noise);
    const noiseAccs = d.acc_vs_noise.map(a => a.accuracy);
    Plotly.newPlot('chart-p6-noise', [
      { x: noiseLevels, y: noiseAccs, mode: 'lines+markers', line: { color: ACCENT, width: 1.5 }, marker: { size: 4 } },
    ], layout({ title: { text: 'Accuracy vs Sensor Noise', font: { size: 12 } }, xaxis: { title: 'Noise Level' }, yaxis: { title: 'Accuracy (%)', range: [0, 100] } }), PLOT_CFG);

    document.getElementById('metric-p6').innerHTML = `${d.key_accuracy} <span class="metric-unit">%</span>`;
  }

  // ============================================================
  // SYSTEM — Unified Ψ
  // ============================================================
  bindSlider('sys-nclim', 'sys-nclim-val', v => v);

  document.getElementById('run-sys').addEventListener('click', function () {
    const params = {
      T_min_C: parseFloat(document.getElementById('sys-tmin').value),
      T_max_C: parseFloat(document.getElementById('sys-tmax').value),
      RH_min: parseFloat(document.getElementById('sys-rhmin').value),
      RH_max: parseFloat(document.getElementById('sys-rhmax').value),
      n_climate: parseInt(document.getElementById('sys-nclim').value),
    };
    this.textContent = 'RUN SYSTEM SWEEP';
    runAndPoll('/api/system/run', params, renderSys, this);
  });

  function renderSys(d) {
    // Chart 1: Psi vs time + drive cycle speed
    Plotly.newPlot('chart-sys-psi', [
      { x: d.drive_time_s, y: d.psi_vs_time, mode: 'lines', line: { color: ACCENT, width: 1.5 }, name: 'Ψ(t)', yaxis: 'y' },
      { x: d.drive_time_s, y: d.drive_speed_kmh, mode: 'lines', line: { color: PLOT_TEXT, width: 1, dash: 'dot' }, name: 'Speed', yaxis: 'y2' },
    ], layout({
      title: { text: 'Ψ over Drive Cycle', font: { size: 12 } },
      xaxis: { title: 'Time (s)' },
      yaxis: { title: 'Ψ', side: 'left' },
      yaxis2: { title: 'km/h', side: 'right', overlaying: 'y', showgrid: false, tickfont: { size: 10, color: PLOT_TEXT } },
      legend: { x: 0.01, y: 0.99, bgcolor: 'rgba(0,0,0,0)' },
    }), PLOT_CFG);

    // Chart 2: Psi climate heatmap
    Plotly.newPlot('chart-sys-heatmap', [{
      z: d.psi_heatmap, x: d.RH_arr_pct, y: d.T_arr_C, type: 'heatmap',
      colorscale: [[0, PLOT_BG], [1, ACCENT]], showscale: true,
      colorbar: { tickfont: { size: 10 }, title: { text: 'Ψ avg', font: { size: 10 } } },
    }], layout({ title: { text: 'Ψ Climate Envelope', font: { size: 12 } }, xaxis: { title: 'RH (%)' }, yaxis: { title: 'T (°C)' } }), PLOT_CFG);

    // Chart 3: KPI radar
    const kpis = d.kpis;
    const kpiNames = ['Range (km)', 'ASR (Ω·cm²)', 'Leak Latency (ms)', 'FP Rate (%)', 'SoH Error (%)'];
    const kpiVals = [kpis.range_km / 500, 1 - kpis.asr_ohm_cm2, 1 - kpis.leak_latency_ms / 1000, 1 - kpis.fp_rate_pct / 100, 1 - kpis.soh_error_pct / 100];
    Plotly.newPlot('chart-sys-kpi', [{
      type: 'scatterpolar', r: kpiVals.concat([kpiVals[0]]), theta: kpiNames.concat([kpiNames[0]]),
      fill: 'toself', fillcolor: 'rgba(255,122,26,0.15)', line: { color: ACCENT, width: 1.5 },
    }], layout({
      title: { text: 'System KPIs', font: { size: 12 } },
      polar: { bgcolor: PLOT_BG, radialaxis: { visible: true, range: [0, 1], gridcolor: PLOT_GRID, linecolor: PLOT_LINE, tickfont: { size: 9 } }, angularaxis: { gridcolor: PLOT_GRID, linecolor: PLOT_LINE, tickfont: { size: 9, color: PLOT_TEXT } } },
    }), PLOT_CFG);

    // Chart 4: Pillar factor contributions
    const factorNames = ['Transport (P1)', 'Storage (P2)', 'EP (P5)', 'AE (P4)'];
    const factorVals = [d.avg_factors.transport, d.avg_factors.storage, d.avg_factors.ep, d.avg_factors.ae];
    Plotly.newPlot('chart-sys-factors', [
      { x: factorNames, y: factorVals, type: 'bar', marker: { color: [ACCENT, WARN_CLR, OK_CLR, PLOT_INK] } },
    ], layout({ title: { text: 'Pillar Contributions (avg)', font: { size: 12 } }, yaxis: { title: 'Factor value' } }), PLOT_CFG);

    document.getElementById('metric-sys').textContent = d.psi_integrated;
    document.getElementById('strip-psi').textContent = d.psi_integrated;
  }

  // --- Init ---
  log('SENTINEL-H₂ console ready.');
  switchTab('1');

})();
