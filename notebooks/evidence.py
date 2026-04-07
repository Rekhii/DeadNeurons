"""
Generate evidence charts for DeadNeurons.
1. Neuron health over training (activation heatmap)
2. Kill + recovery visualization
3. Multi-run consistency across seeds
4. Per-session accuracy distribution
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, '.')
from src.model.classifier import SelfImprovingClassifier
from src.features.extractor import SpikeFeatureExtractor

os.makedirs('figures', exist_ok=True)


# CHART 1: Neuron Health Over Training with Kill + Recovery
print("Generating Chart 1: Neuron health over training...")

rng = np.random.default_rng(42)
X = rng.normal(0, 1, (200, 20))
y = (X[:, 0] + X[:, 1] + X[:, 2] > 0).astype(int)
X_train, X_test = X[:160], X[160:]
y_train, y_test = y[:160], y[160:]

clf = SelfImprovingClassifier(n_features=20, n_hidden=32, learning_rate=0.05)

# Train 50 epochs normally
all_means = []
all_epochs = []
accuracies = []

for epoch in range(50):
    output, cache = clf.forward(X_train)
    grads = clf.backward(y_train, cache)
    clf.W1 -= clf.lr * grads['dW1']
    clf.b1 -= clf.lr * grads['db1']
    clf.W2 -= clf.lr * grads['dW2']
    clf.b2 -= clf.lr * grads['db2']
    _, fc = clf.forward(X_train)
    clf.observe(fc)
    clf.diagnose()
    clf.correct(clf.diagnose())
    all_means.append(clf.activation_means[-1].copy())
    accuracies.append(clf.compute_accuracy(X_test, y_test))

# Kill 8 neurons at epoch 50
kill_ids = [0, 3, 7, 11, 15, 19, 23, 27]
clf.kill_neurons(kill_ids)
accuracies.append(clf.compute_accuracy(X_test, y_test))

# Train 50 more epochs with self-improvement
for epoch in range(50):
    output, cache = clf.forward(X_train)
    grads = clf.backward(y_train, cache)
    clf.W1 -= clf.lr * grads['dW1']
    clf.b1 -= clf.lr * grads['db1']
    clf.W2 -= clf.lr * grads['dW2']
    clf.b2 -= clf.lr * grads['db2']
    _, fc = clf.forward(X_train)
    clf.observe(fc)
    diag = clf.diagnose()
    clf.correct(diag)
    all_means.append(clf.activation_means[-1].copy())
    accuracies.append(clf.compute_accuracy(X_test, y_test))

means_array = np.array(all_means)  # (100, 32)

fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]})

# Heatmap of neuron activations
ax1 = axes[0]
im = ax1.imshow(means_array.T, aspect='auto', cmap='viridis',
                extent=[0, 100, 31.5, -0.5], vmin=0, vmax=np.percentile(means_array, 95))
ax1.axvline(x=50, color='red', linewidth=2, linestyle='--', label='Neurons killed')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Hidden Neuron Index', fontsize=12)
ax1.set_title('Hidden Neuron Activation Over Training\n(8 neurons killed at epoch 50, self-improvement recovers them)',
              fontsize=13, fontweight='bold')
ax1.legend(loc='upper right', fontsize=10)

# Mark killed neurons
for kid in kill_ids:
    ax1.annotate('', xy=(50, kid), fontsize=6, color='red')

cbar = plt.colorbar(im, ax=ax1, label='Mean Activation')

# Accuracy curve
ax2 = axes[1]
ax2.plot(range(len(accuracies)), accuracies, color='#4ade80', linewidth=2)
ax2.axvline(x=50, color='red', linewidth=2, linestyle='--')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Test Accuracy', fontsize=12)
ax2.set_ylim(0.4, 0.8)
ax2.set_title('Accuracy: Drop at Kill, Recovery via Self-Improvement', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/neuron_health.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: figures/neuron_health.png")



# CHART 2: Multi-Run Consistency (5 seeds)
print("Generating Chart 2: Multi-run consistency...")

extractor = SpikeFeatureExtractor('data/synthetic')
extractor.load_data()
datasets = extractor.extract_all()

sessions_to_test = [0, 5, 10, 14, 21]
seeds = [42, 123, 456, 789, 1024]
results = {}

for sid in sessions_to_test:
    X, y_labels, meta = datasets[sid]
    session_accs = []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(y_labels))
        split = int(0.8 * len(y_labels))
        Xtr, Xte = X[idx[:split]], X[idx[split:]]
        ytr, yte = y_labels[idx[:split]], y_labels[idx[split:]]

        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0) + 1e-8
        Xtr_n = (Xtr - mu) / sd
        Xte_n = (Xte - mu) / sd

        nc = min(50, Xtr_n.shape[0] - 1)
        cov = Xtr_n.T @ Xtr_n / (len(Xtr_n) - 1)
        evals, evecs = np.linalg.eigh(cov)
        isort = np.argsort(evals)[::-1]
        evecs = evecs[:, isort[:nc]]
        Xtr_n = Xtr_n @ evecs
        Xte_n = Xte_n @ evecs

        c = SelfImprovingClassifier(n_features=nc, n_hidden=32, learning_rate=0.01, seed=seed)
        c.reg = 0.1
        c.fit(Xtr_n, ytr, Xte_n, yte, epochs=150, verbose=False)
        acc = c.compute_accuracy(Xte_n, yte)
        session_accs.append(acc)

    results[sid] = {
        'mouse': meta['mouse'],
        'accs': session_accs,
        'mean': np.mean(session_accs),
        'std': np.std(session_accs)
    }
    print(f"  Session {sid} ({meta['mouse']}): {np.mean(session_accs):.3f} +/- {np.std(session_accs):.3f}")

fig, ax = plt.subplots(figsize=(10, 5))
x_pos = range(len(sessions_to_test))
means = [results[s]['mean'] for s in sessions_to_test]
stds = [results[s]['std'] for s in sessions_to_test]
labels = [f"S{s}\n{results[s]['mouse']}" for s in sessions_to_test]

bars = ax.bar(x_pos, means, yerr=stds, capsize=8, color='#4ade80',
              edgecolor='#166534', linewidth=1.5, alpha=0.85, error_kw={'linewidth': 2})
ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, label='Chance level')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel('Test Accuracy', fontsize=12)
ax.set_title('Multi-Run Consistency: 5 Random Seeds per Session\n(Error bars = 1 standard deviation)',
             fontsize=13, fontweight='bold')
ax.set_ylim(0.4, 1.05)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(i, m + s + 0.02, f'{m:.1%}\n+/-{s:.1%}', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/multi_run_consistency.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: figures/multi_run_consistency.png")



# CHART 3: All 26 Sessions Accuracy Distribution
print("Generating Chart 3: All sessions accuracy...")

all_accs = []
all_mice = []
for sid in range(26):
    X, yl, meta = datasets[sid]
    r = np.random.default_rng(42)
    idx = r.permutation(len(yl))
    split = int(0.8 * len(yl))
    Xtr, Xte = X[idx[:split]], X[idx[split:]]
    ytr, yte = yl[idx[:split]], yl[idx[split:]]
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0) + 1e-8
    Xtr_n = (Xtr - mu) / sd
    Xte_n = (Xte - mu) / sd
    nc = min(50, Xtr_n.shape[0] - 1)
    cov = Xtr_n.T @ Xtr_n / (len(Xtr_n) - 1)
    evals, evecs = np.linalg.eigh(cov)
    isort = np.argsort(evals)[::-1]
    evecs = evecs[:, isort[:nc]]
    Xtr_n = Xtr_n @ evecs
    Xte_n = Xte_n @ evecs
    c = SelfImprovingClassifier(n_features=nc, n_hidden=32, learning_rate=0.01, seed=42)
    c.reg = 0.1
    c.fit(Xtr_n, ytr, Xte_n, yte, epochs=150, verbose=False)
    acc = c.compute_accuracy(Xte_n, yte)
    all_accs.append(acc)
    all_mice.append(meta['mouse'])

# Color by mouse
mouse_colors = {
    'Cori': '#ef4444', 'Forssmann': '#f97316', 'Hench': '#eab308',
    'Lederberg': '#22c55e', 'Moniz': '#3b82f6', 'Muller': '#8b5cf6',
    'Radnitz': '#ec4899'
}
colors = [mouse_colors[m] for m in all_mice]

fig, ax = plt.subplots(figsize=(14, 5))
bars = ax.bar(range(26), all_accs, color=colors, edgecolor='#333', linewidth=0.8)
ax.axhline(y=0.5, color='white', linestyle='--', linewidth=1, alpha=0.5, label='Chance')
ax.axhline(y=np.mean(all_accs), color='#4ade80', linestyle='-', linewidth=2, alpha=0.7,
           label=f'Mean: {np.mean(all_accs):.1%}')
ax.set_xlabel('Session', fontsize=12)
ax.set_ylabel('Test Accuracy', fontsize=12)
ax.set_title('Decoding Accuracy Across All 26 Steinmetz Sessions\n(Color = mouse)',
             fontsize=13, fontweight='bold')
ax.set_ylim(0.4, 1.05)
ax.set_xticks(range(26))
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2, axis='y')

# Legend for mice
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=m) for m, c in mouse_colors.items()]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9, ncol=2)

plt.tight_layout()
plt.savefig('figures/all_sessions.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: figures/all_sessions.png")

print("\nAll charts generated in figures/")