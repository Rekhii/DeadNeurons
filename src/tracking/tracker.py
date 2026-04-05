"""
DeadNeurons - Experiment Tracker
Logs every training run to SQLite automatically.
No MLflow dependency. Built from scratch.

Usage:
    # In code
    tracker = ExperimentTracker('experiments.db')
    run_id = tracker.start_run(hyperparams)
    tracker.log_session_result(run_id, session_idx, metrics)
    tracker.end_run(run_id, summary)

    # CLI
    python -m src.tracking.tracker list
    python -m src.tracking.tracker show <run_id>
    python -m src.tracking.tracker compare <run_id_1> <run_id_2>
    python -m src.tracking.tracker delete <run_id>
"""

import sqlite3
import json
import time
import subprocess
import os
import uuid
from datetime import datetime


class ExperimentTracker:
    """
    Tracks training experiments in SQLite.

    Each run gets a unique ID, timestamp, git commit hash,
    hyperparameters, and per-session results. Everything is
    queryable and comparable.
    """

    def __init__(self, db_path='experiments.db'):
        """
        Args:
            db_path: path to SQLite database file
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute('''
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                git_commit TEXT,
                status TEXT DEFAULT 'running',
                hyperparams TEXT,
                summary TEXT,
                duration_seconds REAL
            )
        ''')

        c.execute('''
            CREATE TABLE IF NOT EXISTS session_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                session_idx INTEGER NOT NULL,
                mouse TEXT,
                n_neurons INTEGER,
                n_trials INTEGER,
                n_features INTEGER,
                train_acc REAL,
                val_acc REAL,
                test_acc REAL,
                chance_level REAL,
                above_chance INTEGER,
                dead_detected INTEGER,
                dead_fixed INTEGER,
                best_val_acc REAL,
                epochs INTEGER,
                duration_seconds REAL,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
        ''')

        conn.commit()
        conn.close()

    def _get_git_commit(self):
        """Get current git commit hash. Returns 'unknown' if not in a git repo."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--short', 'HEAD'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return 'unknown'

    def start_run(self, hyperparams):
        """
        Start a new experiment run.

        Args:
            hyperparams: dict of hyperparameters
                Example: {'hidden': 32, 'lr': 0.01, 'epochs': 150,
                         'reg': 0.1, 'pca_components': 50, 'seed': 42}

        Returns:
            run_id: unique identifier for this run
        """
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
        timestamp = datetime.now().isoformat()
        git_commit = self._get_git_commit()

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            'INSERT INTO runs (run_id, timestamp, git_commit, status, hyperparams) VALUES (?, ?, ?, ?, ?)',
            (run_id, timestamp, git_commit, 'running', json.dumps(hyperparams))
        )
        conn.commit()
        conn.close()

        print(f"  [Tracker] Run started: {run_id}")
        return run_id

    def log_session_result(self, run_id, session_idx, meta, history):
        """
        Log results for one session within a run.

        Args:
            run_id: the run this session belongs to
            session_idx: which session (0-25)
            meta: session metadata from extractor
            history: training history from classifier
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            INSERT INTO session_results
            (run_id, session_idx, mouse, n_neurons, n_trials, n_features,
             train_acc, val_acc, test_acc, chance_level, above_chance,
             dead_detected, dead_fixed, best_val_acc, epochs, duration_seconds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            run_id,
            session_idx,
            meta.get('mouse', ''),
            meta.get('n_neurons', 0),
            meta.get('n_trials_active', 0),
            meta.get('n_features', 0),
            history.get('final_train_acc', 0),
            history.get('final_val_acc', 0),
            history.get('test_acc', 0),
            history.get('chance_level', 0),
            1 if history.get('above_chance', False) else 0,
            history.get('total_dead_detected', 0),
            history.get('total_reinitialized', 0),
            history.get('best_val_acc', 0),
            history.get('epochs', 0),
            history.get('elapsed_seconds', 0)
        ))
        conn.commit()
        conn.close()

    def end_run(self, run_id, summary, duration):
        """
        Mark a run as completed and store the summary.

        Args:
            run_id: the run to close
            summary: dict with overall results
            duration: total run time in seconds
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            'UPDATE runs SET status=?, summary=?, duration_seconds=? WHERE run_id=?',
            ('completed', json.dumps(summary), duration, run_id)
        )
        conn.commit()
        conn.close()
        print(f"  [Tracker] Run completed: {run_id}")

    def fail_run(self, run_id, error_msg):
        """Mark a run as failed."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            'UPDATE runs SET status=?, summary=? WHERE run_id=?',
            ('failed', json.dumps({'error': error_msg}), run_id)
        )
        conn.commit()
        conn.close()

    def list_runs(self, limit=20):
        """List recent runs."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            SELECT run_id, timestamp, status, git_commit, hyperparams, summary, duration_seconds
            FROM runs ORDER BY timestamp DESC LIMIT ?
        ''', (limit,))
        rows = c.fetchall()
        conn.close()
        return rows

    def get_run(self, run_id):
        """Get full details of a specific run."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute('SELECT * FROM runs WHERE run_id=?', (run_id,))
        run = c.fetchone()

        c.execute('''
            SELECT session_idx, mouse, n_neurons, n_trials, n_features,
                   train_acc, val_acc, test_acc, chance_level, above_chance,
                   dead_detected, dead_fixed, best_val_acc, epochs, duration_seconds
            FROM session_results WHERE run_id=? ORDER BY session_idx
        ''', (run_id,))
        sessions = c.fetchall()

        conn.close()
        return run, sessions

    def delete_run(self, run_id):
        """Delete a run and its session results."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('DELETE FROM session_results WHERE run_id=?', (run_id,))
        c.execute('DELETE FROM runs WHERE run_id=?', (run_id,))
        conn.commit()
        conn.close()
        print(f"  Deleted run: {run_id}")


def _print_run_list(rows):
    """Pretty print a list of runs."""
    if not rows:
        print("No runs found.")
        return

    print(f"\n{'Run ID':<28} {'Status':<10} {'Commit':<8} {'Accuracy':<10} {'Duration'}")
    print("-" * 75)

    for row in rows:
        run_id, timestamp, status, git_commit, hyperparams, summary, duration = row
        acc = ''
        if summary:
            s = json.loads(summary)
            acc = f"{s.get('mean_accuracy', '')}"
        dur = f"{duration:.1f}s" if duration else ''
        print(f"{run_id:<28} {status:<10} {git_commit:<8} {acc:<10} {dur}")


def _print_run_detail(run, sessions):
    """Pretty print full run details."""
    if not run:
        print("Run not found.")
        return

    run_id, timestamp, git_commit, status, hyperparams, summary, duration = run

    print(f"\nRun: {run_id}")
    print(f"Time: {timestamp}")
    print(f"Status: {status}")
    print(f"Commit: {git_commit}")
    print(f"Duration: {duration:.1f}s" if duration else "Duration: -")

    if hyperparams:
        hp = json.loads(hyperparams)
        print(f"Hyperparams: {hp}")

    if summary:
        s = json.loads(summary)
        print(f"\nSummary:")
        for k, v in s.items():
            if k != 'per_session':
                print(f"  {k}: {v}")

    if sessions:
        print(f"\n{'Sess':<5} {'Mouse':<12} {'Neurons':<8} {'Trials':<7} {'Test':<7} {'Chance':<7} {'Dead':<5} {'Fixed'}")
        print("-" * 65)
        for s in sessions:
            idx, mouse, neurons, trials, features, train, val, test, chance, above, dead, fixed, best_val, epochs, dur = s
            print(f"{idx:<5} {mouse:<12} {neurons:<8} {trials:<7} {test:<7.3f} {chance:<7.3f} {dead:<5} {fixed}")


def _compare_runs(tracker, id1, id2):
    """Compare two runs side by side."""
    run1, sess1 = tracker.get_run(id1)
    run2, sess2 = tracker.get_run(id2)

    if not run1 or not run2:
        print("One or both runs not found.")
        return

    s1 = json.loads(run1[5]) if run1[5] else {}
    s2 = json.loads(run2[5]) if run2[5] else {}
    hp1 = json.loads(run1[4]) if run1[4] else {}
    hp2 = json.loads(run2[4]) if run2[4] else {}

    print(f"\n{'Metric':<25} {'Run 1':<20} {'Run 2':<20}")
    print("-" * 65)
    print(f"{'Run ID':<25} {id1:<20} {id2:<20}")
    print(f"{'Commit':<25} {run1[2]:<20} {run2[2]:<20}")

    # Compare hyperparams
    all_keys = set(list(hp1.keys()) + list(hp2.keys()))
    for k in sorted(all_keys):
        v1 = str(hp1.get(k, '-'))
        v2 = str(hp2.get(k, '-'))
        marker = ' *' if v1 != v2 else ''
        print(f"{k:<25} {v1:<20} {v2:<20}{marker}")

    # Compare results
    print()
    acc1 = str(s1.get('mean_accuracy', '-'))
    acc2 = str(s2.get('mean_accuracy', '-'))
    print(f"{'Mean accuracy':<25} {acc1:<20} {acc2:<20}")

    best1 = str(s1.get('best_accuracy', '-'))
    best2 = str(s2.get('best_accuracy', '-'))
    print(f"{'Best accuracy':<25} {best1:<20} {best2:<20}")

    dead1 = str(s1.get('total_dead_detected', '-'))
    dead2 = str(s2.get('total_dead_detected', '-'))
    print(f"{'Dead neurons':<25} {dead1:<20} {dead2:<20}")

    print("\n* = different between runs")


if __name__ == '__main__':
    import sys

    tracker = ExperimentTracker('experiments.db')

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m src.tracking.tracker list")
        print("  python -m src.tracking.tracker show <run_id>")
        print("  python -m src.tracking.tracker compare <run_id1> <run_id2>")
        print("  python -m src.tracking.tracker delete <run_id>")
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == 'list':
        rows = tracker.list_runs()
        _print_run_list(rows)

    elif cmd == 'show' and len(sys.argv) >= 3:
        run, sessions = tracker.get_run(sys.argv[2])
        _print_run_detail(run, sessions)

    elif cmd == 'compare' and len(sys.argv) >= 4:
        _compare_runs(tracker, sys.argv[2], sys.argv[3])

    elif cmd == 'delete' and len(sys.argv) >= 3:
        tracker.delete_run(sys.argv[2])

    else:
        print(f"Unknown command: {cmd}")