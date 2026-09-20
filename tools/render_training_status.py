#!/usr/bin/env python3
"""Render the dated training inventory without exposing operational logs."""
import json
from pathlib import Path


def render(status):
    lines = ['# Fine-tuning progress', '', 'Snapshot: ' + status['as_of_utc'] + '.', '',
             'The completion campaign contains 20 jobs: 18 completed with finite logged losses, '
             'one completed with a numerical issue, and one still training. '
             'BDD RepPoints v3 is a separate replacement candidate.', '',
             '[Full model zoo](../README.md) · [Machine-readable status](../training/status.json)', '',
             '| Domain | Recipe | Steps / schedule | Status |',
             '| --- | --- | ---: | --- |']
    names = {'completed_log_no_nonfinite_loss': 'Completed',
             'completed_with_nonfinite_loss': 'Numerical issue', 'training': 'Training'}
    for row in status['models']:
        lines.append('| {} | [{}](../{}) | {:,} / {:,} | {} |'.format(
            row['domain'], row['model'], row['config'], row['completed_steps'],
            row['total_steps'], names[row['status']]))
    lines += ['', 'Completed schedules are checked against final checkpoint-save and validation '
              'records. Loss status comes from periodic training logs. '
              'The v3 repair additionally checks every training batch.', '',
              'Accuracy results are in the domain model tables. '
              'See [numerical stability](numerical-stability.md) for the RepPoints repair '
              'and [dataset details](datasets.md) for the project splits.', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    (root / 'docs/training-status.md').write_text(render(json.loads((root / 'training/status.json').read_text())))
