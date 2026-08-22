"""
Pillar 6 — Neuromorphic Hyperdimensional Computing Fusion Engine

Implements:
  - Eq 15: Binding B_k = H_k (XOR-equivalent) R_k on bipolar {-1,+1}^D vectors
  - Eq 16: Bundling S(t) = sign(sum B_k(t))
  - Eq 18: Classification c_hat = argmax_c cos(S(t), P_c)
  - Eq 19: Online prototype update P_c <- P_c + eta*(S(t) - P_c)
  - Synthetic multi-channel sensor streams from Pillars 1-5
  - 5 hazard classes: normal, leak-minor, leak-major, stack-degrading, fire-precursor
"""

import numpy as np

D = 10000  # Hypervector dimensionality

CLASSES = ["normal", "leak_minor", "leak_major", "stack_degrading", "fire_precursor"]
N_CLASSES = len(CLASSES)

# Sensor channels: EP splitting, AE b-value, cartridge P/T margin, electrolyte ASR
N_CHANNELS = 4
CHANNEL_NAMES = ["ep_splitting", "ae_bvalue", "cartridge_margin", "asr"]


def generate_role_vectors(n_channels, D, seed=42):
    """
    Generate fixed, near-orthogonal role vectors R_k for each channel.
    Bipolar {-1, +1}^D, generated randomly (with high D, near-orthogonal by construction).
    """
    rng = np.random.RandomState(seed)
    roles = np.where(rng.random((n_channels, D)) > 0.5, 1, -1).astype(np.int8)
    return roles


def thermometer_encode(value, min_val, max_val, D, seed_offset=0):
    """
    Encode a scalar value into a bipolar hypervector using thermometer encoding.
    Maps value in [min_val, max_val] to a position in the D-dimensional space.
    """
    rng = np.random.RandomState(int(abs(value * 1000)) % (2**31) + seed_offset)
    # Fraction of dimensions to flip
    frac = (value - min_val) / max(max_val - min_val, 1e-10)
    frac = np.clip(frac, 0, 1)
    n_flip = int(frac * D)

    hv = np.ones(D, dtype=np.int8)
    indices = rng.choice(D, size=n_flip, replace=False)
    hv[indices] = -1
    return hv


def bind(hv1, hv2):
    """Eq 15: Element-wise binding (XOR equivalent for bipolar = element-wise multiply)."""
    return (hv1 * hv2).astype(np.int8)


def bundle(bound_vectors):
    """Eq 16: Majority-sum bundling S(t) = sign(sum B_k(t))."""
    total = np.sum(bound_vectors, axis=0)
    return np.where(total >= 0, 1, -1).astype(np.int8)


def cosine_similarity(a, b):
    """Cosine similarity between two bipolar vectors."""
    dot = np.dot(a.astype(np.float64), b.astype(np.float64))
    norm_a = np.sqrt(np.dot(a.astype(np.float64), a.astype(np.float64)))
    norm_b = np.sqrt(np.dot(b.astype(np.float64), b.astype(np.float64)))
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return dot / (norm_a * norm_b)


def classify(state_hv, prototypes):
    """Eq 18: Classification by cosine similarity to prototypes."""
    sims = []
    for p in prototypes:
        sims.append(cosine_similarity(state_hv, p))
    return int(np.argmax(sims)), sims


def update_prototype(prototype, state_hv, eta=0.1):
    """Eq 19: Online prototype update P_c <- P_c + eta*(S(t) - P_c)."""
    updated = prototype.astype(np.float64) + eta * (state_hv.astype(np.float64) - prototype.astype(np.float64))
    return np.where(updated >= 0, 1, -1).astype(np.int8)


def generate_synthetic_stream(n_samples, noise_level=0.1, seed=42):
    """
    Generate synthetic multi-channel sensor data labeled by hazard class.

    Normal: all channels near nominal
    Leak minor: EP splitting elevated
    Leak major: EP splitting high + cartridge margin drops
    Stack degrading: AE b-value drops + ASR rises
    Fire precursor: EP + AE + cartridge all anomalous
    """
    rng = np.random.RandomState(seed)

    # Nominal ranges for each channel [min, max]
    # ep_splitting: 0-1 (normalized), ae_bvalue: 0.5-1.5, cartridge_margin: 0-1, asr: 0-1
    samples = []
    labels = []

    samples_per_class = n_samples // N_CLASSES

    for cls_idx in range(N_CLASSES):
        for _ in range(samples_per_class):
            if cls_idx == 0:  # normal
                ep = rng.uniform(0.0, 0.15)
                ae = rng.uniform(1.2, 1.5)
                cm = rng.uniform(0.7, 1.0)
                asr_val = rng.uniform(0.1, 0.3)
            elif cls_idx == 1:  # leak_minor
                ep = rng.uniform(0.3, 0.6)
                ae = rng.uniform(1.0, 1.4)
                cm = rng.uniform(0.5, 0.9)
                asr_val = rng.uniform(0.1, 0.35)
            elif cls_idx == 2:  # leak_major
                ep = rng.uniform(0.6, 1.0)
                ae = rng.uniform(0.8, 1.3)
                cm = rng.uniform(0.1, 0.5)
                asr_val = rng.uniform(0.2, 0.4)
            elif cls_idx == 3:  # stack_degrading
                ep = rng.uniform(0.0, 0.2)
                ae = rng.uniform(0.4, 0.8)
                cm = rng.uniform(0.5, 0.9)
                asr_val = rng.uniform(0.5, 0.9)
            else:  # fire_precursor
                ep = rng.uniform(0.7, 1.0)
                ae = rng.uniform(0.3, 0.7)
                cm = rng.uniform(0.0, 0.3)
                asr_val = rng.uniform(0.6, 1.0)

            # Add noise
            ep += rng.normal(0, noise_level * 0.1)
            ae += rng.normal(0, noise_level * 0.15)
            cm += rng.normal(0, noise_level * 0.1)
            asr_val += rng.normal(0, noise_level * 0.1)

            samples.append([ep, ae, cm, asr_val])
            labels.append(cls_idx)

    # Shuffle
    order = rng.permutation(len(samples))
    samples = [samples[i] for i in order]
    labels = [labels[i] for i in order]

    return samples, labels


def run_simulation(params: dict) -> dict:
    """
    Run the full Pillar 6 HDC simulation.

    Parameters:
        D: hypervector dimensionality (default 10000)
        n_train: training samples (default 500)
        n_test: test samples (default 200)
        eta: learning rate (default 0.1)
        noise_level: sensor noise level (default 0.1)
        n_retrain: number of retraining passes (default 3)
    """
    dim = int(params.get("D", D))
    n_train = int(params.get("n_train", 500))
    n_test = int(params.get("n_test", 200))
    eta = float(params.get("eta", 0.1))
    noise_level = float(params.get("noise_level", 0.1))
    n_retrain = int(params.get("n_retrain", 3))

    dim = min(max(dim, 1000), 10000)
    n_train = min(max(n_train, 100), 2000)
    n_test = min(max(n_test, 50), 500)

    # Generate role vectors
    roles = generate_role_vectors(N_CHANNELS, dim)

    # Generate training and test data
    train_samples, train_labels = generate_synthetic_stream(n_train, noise_level, seed=42)
    test_samples, test_labels = generate_synthetic_stream(n_test, noise_level, seed=99)

    # --- Training: build prototypes ---
    prototypes = [np.zeros(dim, dtype=np.float64) for _ in range(N_CLASSES)]
    class_counts = [0] * N_CLASSES

    accuracy_over_iterations = []

    for retrain_pass in range(n_retrain):
        for i, (sample, label) in enumerate(zip(train_samples, train_labels)):
            # Encode each channel
            hvs = []
            for ch in range(N_CHANNELS):
                hv = thermometer_encode(sample[ch], 0.0, 1.5, dim, seed_offset=ch * 1000)
                bound_hv = bind(hv, roles[ch])
                hvs.append(bound_hv)

            # Bundle
            state_hv = bundle(np.array(hvs))

            # Update prototype (Eq 19)
            prototypes[label] = prototypes[label] + state_hv.astype(np.float64)
            class_counts[label] += 1

        # Binarize prototypes
        proto_binary = []
        for p in prototypes:
            proto_binary.append(np.where(p >= 0, 1, -1).astype(np.int8))

        # Evaluate accuracy after this pass
        correct = 0
        for sample, label in zip(test_samples, test_labels):
            hvs = []
            for ch in range(N_CHANNELS):
                hv = thermometer_encode(sample[ch], 0.0, 1.5, dim, seed_offset=ch * 1000)
                bound_hv = bind(hv, roles[ch])
                hvs.append(bound_hv)
            state_hv = bundle(np.array(hvs))
            pred, _ = classify(state_hv, proto_binary)
            if pred == label:
                correct += 1

        acc = correct / len(test_samples) * 100.0
        accuracy_over_iterations.append({"pass": retrain_pass + 1, "accuracy": round(acc, 1)})

    # --- Final evaluation: confusion matrix ---
    confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    all_sims = []

    for sample, label in zip(test_samples, test_labels):
        hvs = []
        for ch in range(N_CHANNELS):
            hv = thermometer_encode(sample[ch], 0.0, 1.5, dim, seed_offset=ch * 1000)
            bound_hv = bind(hv, roles[ch])
            hvs.append(bound_hv)
        state_hv = bundle(np.array(hvs))
        pred, sims = classify(state_hv, proto_binary)
        confusion[label][pred] += 1
        all_sims.append(sims)

    # Per-class accuracy
    per_class_acc = []
    for c in range(N_CLASSES):
        total = np.sum(confusion[c])
        correct = confusion[c][c]
        per_class_acc.append(round(correct / max(total, 1) * 100, 1))

    # False positive rate for each class
    fp_rates = []
    for c in range(N_CLASSES):
        fp = np.sum(confusion[:, c]) - confusion[c][c]
        tn = np.sum(confusion) - np.sum(confusion[c, :]) - np.sum(confusion[:, c]) + confusion[c][c]
        fp_rate = fp / max(fp + tn, 1) * 100
        fp_rates.append(round(fp_rate, 1))

    # --- Noise sweep: accuracy vs noise level ---
    noise_sweep = np.linspace(0.01, 0.5, 10)
    acc_vs_noise = []
    for nl in noise_sweep:
        noisy_test, noisy_labels = generate_synthetic_stream(100, nl, seed=77)
        correct = 0
        for sample, label in zip(noisy_test, noisy_labels):
            hvs = []
            for ch in range(N_CHANNELS):
                hv = thermometer_encode(sample[ch], 0.0, 1.5, dim, seed_offset=ch * 1000)
                bound_hv = bind(hv, roles[ch])
                hvs.append(bound_hv)
            state_hv = bundle(np.array(hvs))
            pred, _ = classify(state_hv, proto_binary)
            if pred == label:
                correct += 1
        acc_vs_noise.append({"noise": round(float(nl), 3), "accuracy": round(correct / 100 * 100, 1)})

    # Overall accuracy (key metric)
    overall_acc = float(np.trace(confusion)) / float(np.sum(confusion)) * 100

    return {
        "confusion_matrix": confusion.tolist(),
        "class_names": CLASSES,
        "per_class_accuracy": per_class_acc,
        "fp_rates": fp_rates,
        "accuracy_over_iterations": accuracy_over_iterations,
        "acc_vs_noise": acc_vs_noise,
        "key_accuracy": round(overall_acc, 1),
        "dimensionality": dim,
    }
