import numpy as np
import json
from datetime import datetime


class DriftDetector:

    def __init__(self, n_bins=10, psi_threshold=0.2):
        """
        Args:
            n_bins: number of bins for PSI histogram
            psi_threshold: PSI above this means significant drift
        """
        self.n_bins = n_bins
        self.psi_threshold = psi_threshold
        self.reference = None
        self.bin_edges = []

    def fit(self, X_reference):
        """
        Store the training data distribution as the reference.
        Called once after training with the training features.

        Args:
            X_reference: training feature matrix, shape (n_samples, n_features)
        """
        self.reference = X_reference.copy()
        self.bin_edges = []
        for col in range(X_reference.shape[1]):
            edges = np.histogram_bin_edges(X_reference[:, col], bins=self.n_bins)
            self.bin_edges.append(edges)

    def compute_psi(self, expected, actual):
        """
        Compute PSI between two distributions.

        PSI = sum( (actual_pct - expected_pct) * ln(actual_pct / expected_pct) )

        PSI < 0.1:  no significant drift
        PSI 0.1-0.2: moderate drift
        PSI > 0.2:  significant drift

        Args:
            expected: array of bin counts from training data
            actual: array of bin counts from new data

        Returns:
            psi: float
        """
        eps = 1e-6
        expected_pct = expected / expected.sum() + eps
        actual_pct = actual / actual.sum() + eps

        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(psi)

    def detect(self, X_new):
        """
        Check if new data has drifted from the reference.

        Args:
            X_new: new feature matrix, shape (n_samples, n_features)

        Returns:
            report: dict with drift scores and alert status
        """
        if self.reference is None:
            raise RuntimeError('Call fit() first with training data')

        psi_scores = []
        drifted_features = []

        for col in range(X_new.shape[1]):
            ref_hist, _ = np.histogram(self.reference[:, col], bins=self.bin_edges[col])
            new_hist, _ = np.histogram(X_new[:, col], bins=self.bin_edges[col])

            psi = self.compute_psi(ref_hist, new_hist)
            psi_scores.append(psi)

            if psi > self.psi_threshold:
                drifted_features.append(col)

        mean_psi = float(np.mean(psi_scores))
        max_psi = float(np.max(psi_scores))
        drifted = mean_psi > self.psi_threshold

        report = {
            'timestamp': datetime.now().isoformat(),
            'mean_psi': round(mean_psi, 4),
            'max_psi': round(max_psi, 4),
            'threshold': self.psi_threshold,
            'drifted': drifted,
            'n_drifted_features': len(drifted_features),
            'n_total_features': X_new.shape[1],
        }

        status = 'DRIFT DETECTED' if drifted else 'NO DRIFT'
        print(f"  [Drift] {status} | mean_psi={mean_psi:.4f} | "
              f"drifted_features={len(drifted_features)}/{X_new.shape[1]}")

        return report


if __name__ == '__main__':
    detector = DriftDetector()

    # Create reference data
    rng = np.random.default_rng(42)
    X_ref = rng.normal(0, 1, (200, 10))
    detector.fit(X_ref)

    # Test 1: Same distribution (no drift)
    X_same = rng.normal(0, 1, (200, 10))
    print("Same distribution:")
    detector.detect(X_same)

    # Test 2: Shifted distribution (drift)
    X_shifted = rng.normal(2, 1, (200, 10))
    print("Shifted distribution:")
    detector.detect(X_shifted)