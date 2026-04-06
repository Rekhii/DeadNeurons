import numpy as np
import json
from datetime import datetime


class DataValidator:

    def __init__(self):
        # Store validation results
        self.results = []

    def validate_session(self, spks, response, session_idx=0):
        """
        Run all checks on one session's data.
        Returns (passed, report) where passed is True/False
        and report is a dict with details.
        """
        checks = []

        # Check 1: Shape
        # spks should be 3D: (neurons, trials, time_bins)
        if len(spks.shape) != 3:
            checks.append({
                'check': 'shape',
                'passed': False,
                'reason': f'Expected 3D array, got {len(spks.shape)}D'
            })
        else:
            checks.append({
                'check': 'shape',
                'passed': True,
                'detail': f'{spks.shape[0]} neurons, {spks.shape[1]} trials, {spks.shape[2]} bins'
            })

        # Check 2: No NaN or Inf values
        has_nan = np.any(np.isnan(spks))
        has_inf = np.any(np.isinf(spks))
        checks.append({
            'check': 'values',
            'passed': not has_nan and not has_inf,
            'reason': f'NaN={has_nan}, Inf={has_inf}' if has_nan or has_inf else 'clean'
        })

        # Check 3: Non-negative spike counts
        has_negative = np.any(spks < 0)
        checks.append({
            'check': 'non_negative',
            'passed': not has_negative,
            'reason': 'Negative spike counts found' if has_negative else 'all non-negative'
        })

        # Check 4: Minimum trial count
        # Need at least 20 active trials to train meaningfully
        active_trials = int(np.sum(response != 0))
        checks.append({
            'check': 'trial_count',
            'passed': active_trials >= 20,
            'reason': f'{active_trials} active trials' + (' (too few)' if active_trials < 20 else '')
        })

        # Check 5: Label balance
        # Neither class should be less than 10% of active trials
        if active_trials > 0:
            left = np.sum(response == -1)
            right = np.sum(response == 1)
            minority_ratio = min(left, right) / active_trials
            checks.append({
                'check': 'label_balance',
                'passed': minority_ratio >= 0.1,
                'reason': f'left={int(left)}, right={int(right)}, minority={minority_ratio:.2f}'
            })
        else:
            checks.append({
                'check': 'label_balance',
                'passed': False,
                'reason': 'No active trials'
            })

        # Build report
        passed = all(c['passed'] for c in checks)
        report = {
            'session_idx': session_idx,
            'timestamp': datetime.now().isoformat(),
            'passed': passed,
            'checks': checks
        }
        self.results.append(report)

        return passed, report

    def print_report(self, report):
        status = 'PASSED' if report['passed'] else 'FAILED'
        print(f"  Session {report['session_idx']}: {status}")
        for c in report['checks']:
            mark = 'OK' if c['passed'] else 'FAIL'
            reason = c.get('reason', c.get('detail', ''))
            print(f"    [{mark}] {c['check']}: {reason}")

if __name__ == '__main__':
    # Quick test with fake data
    import numpy as np

    validator = DataValidator()

    # Good data
    spks = np.random.poisson(5, (100, 50, 250)).astype(float)
    response = np.random.choice([-1, 0, 1], size=50)
    passed, report = validator.validate_session(spks, response, session_idx=0)
    validator.print_report(report)

    # Bad data: has NaN
    spks_bad = spks.copy()
    spks_bad[0, 0, 0] = np.nan
    passed, report = validator.validate_session(spks_bad, response, session_idx=1)
    validator.print_report(report)