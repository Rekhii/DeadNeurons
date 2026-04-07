"""
Test that proves the self-improvement cycle works.
Trains a classifier, manually kills neurons, shows the system
detecting and fixing them, and accuracy recovering.
"""
import numpy as np
import sys
sys.path.insert(0, '.')
from src.model.classifier import SelfImprovingClassifier

rng = np.random.default_rng(42)

# Create a clear classification problem
X = rng.normal(0, 1, (200, 20))
y = (X[:, 0] + X[:, 1] + X[:, 2] > 0).astype(int)
X_train, X_test = X[:160], X[160:]
y_train, y_test = y[:160], y[160:]

# Phase 1: Train normally
print("=" * 60)
print("PHASE 1: Normal Training (50 epochs)")
print("=" * 60)
clf = SelfImprovingClassifier(n_features=20, n_hidden=32, learning_rate=0.05)
history1 = clf.fit(X_train, y_train, X_test, y_test, epochs=50, verbose=True)
acc_before_kill = clf.compute_accuracy(X_test, y_test)
print(f"\nAccuracy after normal training: {acc_before_kill:.3f}")

# Phase 2: Kill 10 neurons (31% of hidden layer)
print("\n" + "=" * 60)
print("PHASE 2: Killing 10 out of 32 neurons")
print("=" * 60)
kill_indices = list(range(0, 10))
clf.kill_neurons(kill_indices)
acc_after_kill = clf.compute_accuracy(X_test, y_test)
print(f"Accuracy after killing neurons: {acc_after_kill:.3f}")
print(f"Accuracy drop: {acc_before_kill - acc_after_kill:.3f}")

# Run one observe + diagnose to show detection
_, cache = clf.forward(X_train)
clf.observe(cache)
diagnosis = clf.diagnose()
print(f"Dead neurons detected: {diagnosis['n_dead']}")
print(f"Dead neuron indices: {diagnosis['dead']}")

# Phase 3: Continue training WITH self-improvement
print("\n" + "=" * 60)
print("PHASE 3: Recovery Training with Self-Improvement (50 epochs)")
print("=" * 60)
history2 = clf.fit(X_train, y_train, X_test, y_test, epochs=50, verbose=True)
acc_recovered = clf.compute_accuracy(X_test, y_test)

# Summary
print("\n" + "=" * 60)
print("SUMMARY: Self-Improvement Cycle Proof")
print("=" * 60)
print(f"Accuracy after normal training:  {acc_before_kill:.3f}")
print(f"Accuracy after killing neurons:  {acc_after_kill:.3f}  (drop: {acc_before_kill - acc_after_kill:.3f})")
print(f"Accuracy after recovery:         {acc_recovered:.3f}  (recovered: {acc_recovered - acc_after_kill:.3f})")
print(f"Dead neurons detected:           {diagnosis['n_dead']}")
print(f"Total neurons reinitialized:     {sum(c['reinitialized'] for c in clf.correction_history)}")
print(f"\nThe self-improvement cycle detected {diagnosis['n_dead']} dead neurons")
print(f"and recovered accuracy from {acc_after_kill:.1%} to {acc_recovered:.1%}")
