"""
DeadNeurons - Feature Extractor
Extracts mean firing rates per neuron per time window from Steinmetz spike data.
Each session produces a feature matrix X and label vector y.
"""

import numpy as np
import json
import os


class SpikeFeatureExtractor:
    """
    Extracts features from Steinmetz Neuropixels spike data.

    The Steinmetz dataset contains recordings from mice performing a visual
    decision task. Each session has a different number of neurons recorded
    across multiple brain regions. The mouse sees two screens with different
    contrast levels and turns a wheel left or right to indicate the brighter side.

    Features: mean firing rate per neuron in 4 time windows per trial.
    Labels: 0 = turned left, 1 = turned right.
    Trials where the mouse didn't respond are excluded.
    """

    def __init__(self, data_dir):
        """
        Args:
            data_dir: path to directory containing steinmetz_part0.npz and steinmetz_part1.npz
        """
        self.data_dir = data_dir
        self.sessions = None
        self.n_sessions = 0

        # Time windows in bins (10ms per bin, 250 bins = 2.5 seconds)
        # Pre-stimulus:  0-62    (before visual input)
        # Stimulus:      62-125  (visual processing)
        # Decision:      125-187 (motor planning)
        # Post-decision: 187-250 (feedback period)
        self.windows = [
            (0, 62),
            (62, 125),
            (125, 187),
            (187, 250)
        ]
        self.window_names = ['pre_stim', 'stimulus', 'decision', 'post_decision']

    def load_data(self):
        """Load all sessions from both .npz files."""
        parts = []
        for fname in ['steinmetz_part0.npz', 'steinmetz_part1.npz']:
            path = os.path.join(self.data_dir, fname)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing data file: {path}")
            dat = np.load(path, allow_pickle=True)['dat']
            parts.append(dat)

        self.sessions = np.concatenate(parts)
        self.n_sessions = len(self.sessions)
        print(f"Loaded {self.n_sessions} sessions")
        return self

    def extract_session(self, session_idx):
        """
        Extract features and labels from a single session.

        Args:
            session_idx: integer index of the session (0 to n_sessions-1)

        Returns:
            X: feature matrix, shape (n_active_trials, n_neurons * 4)
            y: label vector, shape (n_active_trials,), values 0 or 1
            meta: dict with session metadata
        """
        if self.sessions is None:
            raise RuntimeError("Call load_data() first")

        s = self.sessions[session_idx]
        spks = s['spks']  # (neurons, trials, time_bins)
        resp = s['response']  # (trials,)

        # Keep only trials where the mouse responded
        active_mask = resp != 0
        n_active = int(np.sum(active_mask))

        if n_active == 0:
            return None, None, None

        spks_active = spks[:, active_mask, :]

        # Convert labels: -1 (left) -> 0, +1 (right) -> 1
        labels = ((resp[active_mask] + 1) // 2).astype(int)

        nn = spks_active.shape[0]  # neurons in this session
        nt = spks_active.shape[1]  # active trials
        n_bins = spks_active.shape[2]

        # Adjust windows if session has different bin count
        if n_bins != 250:
            quarter = n_bins // 4
            local_windows = [
                (0, quarter),
                (quarter, 2 * quarter),
                (2 * quarter, 3 * quarter),
                (3 * quarter, n_bins)
            ]
        else:
            local_windows = self.windows

        # Build feature matrix: mean firing rate per neuron per window
        X = np.zeros((nt, nn * len(local_windows)))
        for j, (start, end) in enumerate(local_windows):
            rates = spks_active[:, :, start:end].mean(axis=2)  # (nn, nt)
            X[:, j * nn:(j + 1) * nn] = rates.T  # (nt, nn)

        meta = {
            'session_idx': session_idx,
            'mouse': str(s['mouse_name']),
            'date': str(s['date_exp']),
            'n_neurons': nn,
            'n_trials_total': len(resp),
            'n_trials_active': n_active,
            'n_features': X.shape[1],
            'brain_areas': list(np.unique(s['brain_area'])),
            'label_counts': {
                'left': int(np.sum(labels == 0)),
                'right': int(np.sum(labels == 1))
            }
        }

        return X, labels, meta

    def extract_all(self):
        """
        Extract features from all sessions.

        Returns:
            datasets: list of (X, y, meta) tuples, one per session
        """
        datasets = []
        for i in range(self.n_sessions):
            X, y, meta = self.extract_session(i)
            if X is not None:
                datasets.append((X, y, meta))

        total_trials = sum(d[0].shape[0] for d in datasets)
        print(f"Extracted {len(datasets)} sessions, {total_trials} total trials")
        return datasets

    def train_test_split(self, X, y, test_ratio=0.2, seed=42):
        """
        Shuffle and split a single session's data.

        Args:
            X: feature matrix
            y: label vector
            test_ratio: fraction held out for testing
            seed: random seed for reproducibility

        Returns:
            X_train, X_test, y_train, y_test
        """
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(y))
        split = int((1 - test_ratio) * len(y))

        return X[idx[:split]], X[idx[split:]], y[idx[:split]], y[idx[split:]]

    def normalize(self, X_train, X_test):
        """
        Zero mean, unit variance normalization.
        Fit on train, apply to both train and test.

        Args:
            X_train: training features
            X_test: test features

        Returns:
            X_train_norm, X_test_norm, mean, std
        """
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 1e-8

        return (X_train - mean) / std, (X_test - mean) / std, mean, std


if __name__ == '__main__':
    # Quick test
    extractor = SpikeFeatureExtractor('data/synthetic')
    extractor.load_data()

    datasets = extractor.extract_all()

    # Print summary
    print(f"\n{'Session':<8} {'Mouse':<12} {'Trials':<8} {'Neurons':<8} {'Features'}")
    print("-" * 55)
    for X, y, meta in datasets:
        print(f"{meta['session_idx']:<8} {meta['mouse']:<12} {meta['n_trials_active']:<8} "
              f"{meta['n_neurons']:<8} {meta['n_features']}")