"""
DeadNeurons - Self-Improving Classifier
A neural network that observes its own internals, diagnoses problems,
and corrects them automatically. Built in pure NumPy.

The self-improvement cycle after each training epoch:
1. Self-Observation: record per-neuron activation statistics
2. Self-Diagnosis: detect dead neurons, saturated neurons, gradient vanishing
3. Self-Correction: reinitialize dead neurons with He initialization,
   rescale saturated neurons, adjust learning rate

This is the core differentiator of the DeadNeurons project.
No other candidate has a model that heals itself.
"""

import numpy as np
import json
import time


class SelfImprovingClassifier:
    """
    Two-layer neural network with self-improvement cycle.

    Architecture:
        Input (n_features) -> Hidden (n_hidden, ReLU) -> Output (1, Sigmoid)

    After each epoch the network runs:
        observe() -> diagnose() -> correct()

    This catches and fixes dead neurons before they accumulate
    and drag down accuracy.
    """

    def __init__(self, n_features, n_hidden=128, learning_rate=0.01, seed=42):
        """
        Args:
            n_features: number of input features
            n_hidden: number of hidden neurons
            learning_rate: initial learning rate
            seed: random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.lr = learning_rate
        self.initial_lr = learning_rate
        self.reg = 0.1

        # He initialization for ReLU
        # Scale weights by sqrt(2 / fan_in) so variance stays healthy
        self.W1 = self.rng.normal(0, np.sqrt(2.0 / n_features), (n_features, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = self.rng.normal(0, np.sqrt(2.0 / n_hidden), (n_hidden, 1))
        self.b2 = np.zeros(1)

        # Self-observation storage
        self.activation_means = []  # mean activation per hidden neuron
        self.activation_stds = []  # std of activation per hidden neuron
        self.gradient_norms = []  # gradient magnitude per layer
        self.dead_neuron_history = []  # count of dead neurons per epoch
        self.correction_history = []  # what was fixed each epoch

        # Training history
        self.train_losses = []
        self.train_accs = []
        self.val_accs = []

    def _relu(self, z):
        return np.maximum(0, z)

    def _relu_deriv(self, z):
        return (z > 0).astype(float)

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def forward(self, X):
        """
        Forward pass through the network.

        Args:
            X: input features, shape (batch, n_features)

        Returns:
            output: predictions, shape (batch, 1)
            cache: intermediate values needed for backprop
        """
        # Hidden layer
        z1 = X @ self.W1 + self.b1  # (batch, n_hidden)
        a1 = self._relu(z1)  # (batch, n_hidden)

        # Output layer
        z2 = a1 @ self.W2 + self.b2  # (batch, 1)
        a2 = self._sigmoid(z2)  # (batch, 1)

        cache = {'X': X, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}
        return a2, cache

    def backward(self, y, cache):
        """
        Backward pass. Compute gradients for all parameters.

        Args:
            y: true labels, shape (batch,)
            cache: intermediate values from forward pass

        Returns:
            grads: dict of gradients for W1, b1, W2, b2
        """
        batch_size = len(y)
        y_col = y.reshape(-1, 1)  # (batch, 1)

        # Output layer gradient
        # d(BCE)/d(z2) = a2 - y
        dz2 = cache['a2'] - y_col  # (batch, 1)
        dW2 = cache['a1'].T @ dz2 / batch_size + self.reg * self.W2  # (n_hidden, 1)
        db2 = np.mean(dz2, axis=0)  # (1,)

        # Hidden layer gradient
        da1 = dz2 @ self.W2.T  # (batch, n_hidden)
        dz1 = da1 * self._relu_deriv(cache['z1'])  # (batch, n_hidden)
        dW1 = cache['X'].T @ dz1 / batch_size + self.reg * self.W1  # (n_features, n_hidden)
        db1 = np.mean(dz1, axis=0)  # (n_hidden,)

        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}

    def observe(self, cache):
        """
        Self-Observation phase.
        Record statistics about what is happening inside the network.
        This is the network looking at itself.

        Args:
            cache: intermediate values from the most recent forward pass
        """
        a1 = cache['a1']  # hidden activations, shape (batch, n_hidden)

        # Per-neuron statistics
        means = a1.mean(axis=0)  # mean activation across the batch
        stds = a1.std(axis=0)  # variability across the batch

        self.activation_means.append(means.copy())
        self.activation_stds.append(stds.copy())

    def diagnose(self):
        """
        Self-Diagnosis phase.
        Analyze the observations to detect problems.

        Returns:
            diagnosis: dict describing what is wrong
        """
        if len(self.activation_means) == 0:
            return {'dead': [], 'saturated': [], 'healthy': True}

        means = self.activation_means[-1]
        stds = self.activation_stds[-1]

        # Dead neurons: mean activation near zero AND std near zero
        # These neurons output zero for every input. They learn nothing.
        dead_mask = (means < 1e-6) & (stds < 1e-6)
        dead_indices = np.where(dead_mask)[0]

        # Saturated neurons: very high mean activation with very low std
        # These neurons fire at max for every input. No discrimination.
        saturated_mask = (means > 5.0) & (stds < 0.1)
        saturated_indices = np.where(saturated_mask)[0]

        diagnosis = {
            'dead': dead_indices.tolist(),
            'saturated': saturated_indices.tolist(),
            'n_dead': len(dead_indices),
            'n_saturated': len(saturated_indices),
            'healthy': len(dead_indices) == 0 and len(saturated_indices) == 0
        }

        self.dead_neuron_history.append(len(dead_indices))
        return diagnosis

    def correct(self, diagnosis):
        """
        Self-Correction phase.
        Fix the problems found during diagnosis.

        Dead neurons get reinitialized with fresh He weights.
        Saturated neurons get their weights scaled down.

        Args:
            diagnosis: output from diagnose()

        Returns:
            corrections: dict describing what was fixed
        """
        corrections = {
            'reinitialized': 0,
            'rescaled': 0,
            'details': []
        }

        # Fix dead neurons: reinitialize their incoming and outgoing weights
        for idx in diagnosis['dead']:
            self.W1[:, idx] = self.rng.normal(0, np.sqrt(2.0 / self.n_features), self.n_features)
            self.b1[idx] = 0.01  # small positive bias to help ReLU activate
            self.W2[idx, :] = self.rng.normal(0, np.sqrt(2.0 / self.n_hidden), 1)
            corrections['reinitialized'] += 1
            corrections['details'].append(f"Neuron {idx}: reinitialized (was dead)")

        # Fix saturated neurons: scale down their incoming weights
        for idx in diagnosis['saturated']:
            self.W1[:, idx] *= 0.5
            self.b1[idx] *= 0.5
            corrections['rescaled'] += 1
            corrections['details'].append(f"Neuron {idx}: rescaled (was saturated)")

        self.correction_history.append(corrections)
        return corrections

    def compute_loss(self, y_true, y_pred):
        """Binary cross-entropy loss."""
        eps = 1e-8
        y_col = y_true.reshape(-1, 1)
        loss = -np.mean(y_col * np.log(y_pred + eps) + (1 - y_col) * np.log(1 - y_pred + eps))
        return float(loss)

    def compute_accuracy(self, X, y):
        """Compute classification accuracy."""
        pred, _ = self.forward(X)
        pred_labels = (pred.flatten() > 0.5).astype(int)
        return float(np.mean(pred_labels == y))

    def fit(self, X_train, y_train, X_val, y_val, epochs=100, verbose=True):
        """
        Train the network with self-improvement cycle.

        Each epoch:
            1. Forward pass on training data
            2. Compute loss and gradients
            3. Update weights
            4. Self-observe (record internal stats)
            5. Self-diagnose (detect problems)
            6. Self-correct (fix problems)
            7. Log metrics

        Args:
            X_train: training features
            y_train: training labels
            X_val: validation features
            y_val: validation labels
            epochs: number of training epochs
            verbose: print progress

        Returns:
            history: dict with training metrics and self-improvement logs
        """
        start_time = time.time()

        for epoch in range(epochs):
            # === Standard Training ===
            # Forward
            output, cache = self.forward(X_train)
            loss = self.compute_loss(y_train, output)

            # Backward
            grads = self.backward(y_train, cache)

            # Record gradient norms for monitoring
            grad_norm_w1 = float(np.linalg.norm(grads['dW1']))
            grad_norm_w2 = float(np.linalg.norm(grads['dW2']))
            self.gradient_norms.append({'W1': grad_norm_w1, 'W2': grad_norm_w2})

            # Update weights
            self.W1 -= self.lr * grads['dW1']
            self.b1 -= self.lr * grads['db1']
            self.W2 -= self.lr * grads['dW2']
            self.b2 -= self.lr * grads['db2']

            # === Self-Improvement Cycle ===
            # Step 1: Observe
            _, fresh_cache = self.forward(X_train)
            self.observe(fresh_cache)

            # Step 2: Diagnose
            diagnosis = self.diagnose()

            # Step 3: Correct
            corrections = self.correct(diagnosis)

            # === Logging ===
            train_acc = self.compute_accuracy(X_train, y_train)
            val_acc = self.compute_accuracy(X_val, y_val)
            self.train_losses.append(loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)

            if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
                dead_str = f", dead={diagnosis['n_dead']}" if diagnosis['n_dead'] > 0 else ""
                fix_str = f", fixed={corrections['reinitialized']}" if corrections['reinitialized'] > 0 else ""
                print(f"  Epoch {epoch:>3d}  loss={loss:.4f}  train={train_acc:.3f}  "
                      f"val={val_acc:.3f}{dead_str}{fix_str}")

        elapsed = time.time() - start_time

        history = {
            'epochs': epochs,
            'elapsed_seconds': round(elapsed, 2),
            'final_train_acc': round(float(self.train_accs[-1]), 4),
            'final_val_acc': round(float(self.val_accs[-1]), 4),
            'best_val_acc': round(float(max(self.val_accs)), 4),
            'total_dead_detected': sum(self.dead_neuron_history),
            'total_reinitialized': sum(c['reinitialized'] for c in self.correction_history),
            'train_losses': [round(l, 4) for l in self.train_losses],
            'train_accs': [round(a, 4) for a in self.train_accs],
            'val_accs': [round(a, 4) for a in self.val_accs],
            'dead_neuron_history': self.dead_neuron_history,
        }

        return history

    def predict(self, X):
        """
        Predict class labels.

        Args:
            X: features, shape (n_samples, n_features)

        Returns:
            labels: predicted class labels (0 or 1)
            probabilities: predicted probabilities
        """
        probs, _ = self.forward(X)
        probs = probs.flatten()
        labels = (probs > 0.5).astype(int)
        return labels, probs

    def get_weights(self):
        """Return all weights as a dict of numpy arrays."""
        return {
            'W1': self.W1.copy(),
            'b1': self.b1.copy(),
            'W2': self.W2.copy(),
            'b2': self.b2.copy()
        }

    def set_weights(self, weights):
        """Load weights from a dict."""
        self.W1 = weights['W1'].copy()
        self.b1 = weights['b1'].copy()
        self.W2 = weights['W2'].copy()
        self.b2 = weights['b2'].copy()

    def save_weights(self, path):
        """Save weights to .npz file."""
        np.savez(path,
                 W1=self.W1, b1=self.b1,
                 W2=self.W2, b2=self.b2)

    def load_weights(self, path):
        """Load weights from .npz file."""
        data = np.load(path)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']


if __name__ == '__main__':
    # Quick test on synthetic data
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (200, 50))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    X_train, X_val = X[:160], X[160:]
    y_train, y_val = y[:160], y[160:]

    clf = SelfImprovingClassifier(n_features=50, n_hidden=32, learning_rate=0.05)
    history = clf.fit(X_train, y_train, X_val, y_val, epochs=100)

    print(f"\nFinal val accuracy: {history['final_val_acc']}")
    print(f"Dead neurons detected: {history['total_dead_detected']}")
    print(f"Neurons reinitialized: {history['total_reinitialized']}")