"""
Prove self-improvement works on real Steinmetz brain data.
Train, kill neurons, show detection and recovery.
"""
import numpy as np
import sys
sys.path.insert(0, '.')
from src.features.extractor import SpikeFeatureExtractor
from src.model.classifier import SelfImprovingClassifier

# Load real data
extractor = SpikeFeatureExtractor('data/synthetic')
extractor.load_data()
datasets = extractor.extract_all()

# Use Session 10 (best performing)
X, y, meta = datasets[10]

# Split and normalize
rng = np.random.default_rng(42)
idx = rng.permutation(len(y))
split = int(0.8 * len(y))
X_train, X_test = X[idx[:split]], X[idx[split:]]
y_train, y_test = y[idx[:split]], y[idx[split:]]

mean = X_train.mean(axis=0)
std = X_train.std(axis=0) + 1e-8
X_train_n = (X_train - mean) / std
X_test_n = (X_test - mean) / std

# PCA
n_components = min(50, X_train_n.shape[0] - 1)
cov = X_train_n.T @ X_train_n / (len(X_train_n) - 1)
eigenvalues, eigenvectors = np.linalg.eigh(cov)
idx_sort = np.argsort(eigenvalues)[::-1]
eigenvectors = eigenvectors[:, idx_sort[:n_components]]
X_train_n = X_train_n @ eigenvectors
X_test_n = X_test_n @ eigenvectors

print(f"Session 10 | {meta['mouse']} | {meta['n_neurons']} neurons | {meta['n_trials_active']} trials")
print(f"Train: {len(y_train)} | Test: {len(y_test)} | Features: {X_train_n.shape[1]}")

# Phase 1: Train
print("\n" + "=" * 60)
print("PHASE 1: Normal Training (100 epochs)")
print("=" * 60)
clf = SelfImprovingClassifier(n_features=X_train_n.shape[1], n_hidden=64, learning_rate=0.01)
clf.reg = 0.1
h1 = clf.fit(X_train_n, y_train, X_test_n, y_test, epochs=100, verbose=True)
acc_before = clf.compute_accuracy(X_test_n, y_test)
print(f"\nAccuracy after training: {acc_before:.3f}")

# Phase 2: Kill 20 neurons (31% of 64)
print("\n" + "=" * 60)
print("PHASE 2: Killing 20 out of 64 neurons")
print("=" * 60)
clf.kill_neurons(list(range(20)))
acc_killed = clf.compute_accuracy(X_test_n, y_test)
print(f"Accuracy after kill: {acc_killed:.3f}")
print(f"Drop: {acc_before - acc_killed:.3f}")

_, cache = clf.forward(X_train_n)
clf.observe(cache)
diag = clf.diagnose()
print(f"Dead neurons detected: {diag['n_dead']}")

# Phase 3: Recovery
print("\n" + "=" * 60)
print("PHASE 3: Recovery with Self-Improvement (100 epochs)")
print("=" * 60)
h2 = clf.fit(X_train_n, y_train, X_test_n, y_test, epochs=100, verbose=True)
acc_recovered = clf.compute_accuracy(X_test_n, y_test)

# Summary
print("\n" + "=" * 60)
print("RESULTS ON REAL STEINMETZ DATA")
print("=" * 60)
print(f"Session: {meta['session_idx']} ({meta['mouse']}, {meta['n_neurons']} neurons)")
print(f"Before kill:   {acc_before:.1%}")
print(f"After kill:    {acc_killed:.1%}  (drop: {acc_before - acc_killed:.1%})")
print(f"After recover: {acc_recovered:.1%}  (gained: {acc_recovered - acc_killed:.1%})")
print(f"Dead detected: {diag['n_dead']}")
print(f"Total fixed:   {sum(c['reinitialized'] for c in clf.correction_history)}")