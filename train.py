"""
DeadNeurons - Training Script
Trains the self-improving classifier on Steinmetz spike data.
Every run is automatically logged to SQLite via the experiment tracker.

Usage:
    python train.py                    # train all sessions
    python train.py --session 0        # train session 0 only
    python train.py --session 0 --epochs 200
    python train.py --no-track         # skip tracking
"""

import numpy as np
import json
import argparse
import os
import sys
import time
from src.registry.registry import ModelRegistry
from src.features.extractor import SpikeFeatureExtractor
from src.model.classifier import SelfImprovingClassifier
from src.tracking.tracker import ExperimentTracker


def train_session(X, y, meta, n_hidden=128, epochs=150, lr=0.01, reg=0.1,
                  pca_components=50, seed=42):
    """
    Train the self-improving classifier on one session.

    Args:
        X: feature matrix from extractor
        y: labels
        meta: session metadata
        n_hidden: hidden layer size
        epochs: training epochs
        lr: learning rate
        reg: L2 regularization strength
        pca_components: number of PCA components
        seed: random seed

    Returns:
        history: training history dict
        clf: trained classifier
        data_splits: dict with train/test data and normalization params
    """
    # Split
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    split = int(0.8 * len(y))
    X_train, X_test = X[idx[:split]], X[idx[split:]]
    y_train, y_test = y[idx[:split]], y[idx[split:]]

    # Normalize
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train_n = (X_train - mean) / std
    X_test_n = (X_test - mean) / std

    # PCA: reduce to at most n_components dimensions
    n_components = min(pca_components, X_train_n.shape[0] - 1, X_train_n.shape[1])
    cov = X_train_n.T @ X_train_n / (len(X_train_n) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx_sort = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx_sort[:n_components]]
    X_train_n = X_train_n @ eigenvectors
    X_test_n = X_test_n @ eigenvectors

    # Train
    clf = SelfImprovingClassifier(
        n_features=X_train_n.shape[1],
        n_hidden=n_hidden,
        learning_rate=lr,
        seed=seed
    )
    clf.reg = reg

    print(f"\nSession {meta['session_idx']} | {meta['mouse']} | "
          f"{meta['n_neurons']} neurons | {meta['n_trials_active']} trials")
    print(f"  Train: {len(y_train)} | Test: {len(y_test)} | "
          f"Features: {X.shape[1]} -> {n_components} (PCA)")

    history = clf.fit(X_train_n, y_train, X_test_n, y_test, epochs=epochs)

    # Final test accuracy
    test_acc = clf.compute_accuracy(X_test_n, y_test)
    history['test_acc'] = round(test_acc, 4)
    history['session_meta'] = meta

    # Chance level
    chance = max(np.mean(y_test), 1 - np.mean(y_test))
    history['chance_level'] = round(float(chance), 4)
    history['above_chance'] = bool(test_acc > chance)

    print(f"  Result: test={test_acc:.3f} vs chance={chance:.3f} "
          f"| dead_fixed={history['total_reinitialized']}")

    data_splits = {
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'mean': mean, 'std': std,
        'eigenvectors': eigenvectors
    }

    return history, clf, data_splits


def main():
    parser = argparse.ArgumentParser(description='DeadNeurons Training')
    parser.add_argument('--session', type=int, default=None,
                        help='Train specific session (default: all)')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Training epochs (default: 150)')
    parser.add_argument('--hidden', type=int, default=32,
                        help='Hidden layer size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate (default: 0.01)')
    parser.add_argument('--reg', type=float, default=0.1,
                        help='L2 regularization (default: 0.1)')
    parser.add_argument('--pca', type=int, default=50,
                        help='PCA components (default: 50)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--no-track', action='store_true',
                        help='Skip experiment tracking')
    args = parser.parse_args()

    # Hyperparams dict for tracking
    hyperparams = {
        'hidden': args.hidden,
        'lr': args.lr,
        'reg': args.reg,
        'epochs': args.epochs,
        'pca_components': args.pca,
        'seed': args.seed,
        'session': args.session if args.session is not None else 'all'
    }

    # Initialize tracker
    tracker = None
    run_id = None
    if not args.no_track:
        tracker = ExperimentTracker('experiments.db')
        run_id = tracker.start_run(hyperparams)

    # Load data
    extractor = SpikeFeatureExtractor('data/synthetic')
    extractor.load_data()
    datasets = extractor.extract_all()

    # Create output directory for weights
    os.makedirs('weights', exist_ok=True)

    run_start = time.time()

    try:
        if args.session is not None:
            # Train single session
            X, y, meta = datasets[args.session]
            history, clf, splits = train_session(
                X, y, meta,
                n_hidden=args.hidden,
                epochs=args.epochs,
                lr=args.lr,
                reg=args.reg,
                pca_components=args.pca,
                seed=args.seed
            )

            # Save weights
            weight_path = f"weights/session_{args.session}.npz"
            clf.save_weights(weight_path)
            print(f"\n  Weights saved: {weight_path}")

            # Log to tracker
            if tracker:
                tracker.log_session_result(run_id, args.session, meta, history)

            # Summary for single session
            summary = {
                'n_sessions': 1,
                'mean_accuracy': history['test_acc'],
                'best_accuracy': history['test_acc'],
                'worst_accuracy': history['test_acc'],
                'above_chance': 1 if history['above_chance'] else 0,
                'total_dead_detected': history['total_dead_detected'],
                'total_reinitialized': history['total_reinitialized']
            }

        else:
            # Train all sessions
            all_results = []

            for X, y, meta in datasets:
                history, clf, splits = train_session(
                    X, y, meta,
                    n_hidden=args.hidden,
                    epochs=args.epochs,
                    lr=args.lr,
                    reg=args.reg,
                    pca_components=args.pca,
                    seed=args.seed
                )

                # Save weights per session
                sid = meta['session_idx']
                clf.save_weights(f"weights/session_{sid}.npz")

                # Log to tracker
                if tracker:
                    tracker.log_session_result(run_id, sid, meta, history)

                all_results.append(history)

            # Summary
            accs = [r['test_acc'] for r in all_results]
            above = sum(1 for r in all_results if r['above_chance'])
            total_dead = sum(r['total_dead_detected'] for r in all_results)
            total_fixed = sum(r['total_reinitialized'] for r in all_results)

            print(f"\n{'='*55}")
            print(f"SUMMARY")
            print(f"{'='*55}")
            print(f"Sessions trained:     {len(all_results)}")
            print(f"Mean test accuracy:   {np.mean(accs):.3f}")
            print(f"Best session:         {np.max(accs):.3f}")
            print(f"Worst session:        {np.min(accs):.3f}")
            print(f"Above chance:         {above}/{len(all_results)}")
            print(f"Dead neurons found:   {total_dead}")
            print(f"Neurons reinitialized:{total_fixed}")

            summary = {
                'n_sessions': len(all_results),
                'mean_accuracy': round(float(np.mean(accs)), 4),
                'best_accuracy': round(float(np.max(accs)), 4),
                'worst_accuracy': round(float(np.min(accs)), 4),
                'above_chance': above,
                'total_dead_detected': total_dead,
                'total_reinitialized': total_fixed
            }

            # Save summary to file
            with open('weights/training_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\nSummary saved: weights/training_summary.json")


        # Register model in registry and try to promote
        registry = ModelRegistry('rekhi/deadneurons-registry')

        # Collect weights from the last trained classifier
        # For all-sessions run, we register the overall summary
        # For single session, we register that session's weights
        version = registry.register_model(
            weights=clf.get_weights(),
            config=hyperparams,
            metrics=summary
        )
        registry.promote(version)
        # End tracking
        run_duration = time.time() - run_start
        if tracker:
            tracker.end_run(run_id, summary, run_duration)

    except Exception as e:
        if tracker and run_id:
            tracker.fail_run(run_id, str(e))
        raise


if __name__ == '__main__':
    main()