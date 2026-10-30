#!/usr/bin/env python3
# Compare saved Safety policies with distance A* on seen and unseen vessels.
import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from src.policy_store import load_deployed_policy
from src.simulation.live_controller import MAX_LIVE_STEPS
from src.simulation.microdose_env import ENVIRONMENT_PROFILES, SEEN_STUDY_LAYOUTS, UNSEEN_LAYOUTS, INPUTS, environment_fingerprint
from src.simulation.rewards import SAFETY_REWARD
from src.simulation.rl_environment import BenchmarkWorld, evaluate
from src.simulation.safety_metrics import SAFETY_CALIBRATION

COMPARISONS = {
    'astar_vs_rl': ('astar_distance', 'rl_safety'),
    'rl_safety': ('astar_distance', 'rl_safety', 'sarsa_safety'),
}
LABELS = {'astar_distance': 'A*', 'rl_safety': 'Q-learning', 'sarsa_safety': 'SARSA'}
CONTROLLERS = {'rl_safety': 'Q-learning · Safety', 'sarsa_safety': 'SARSA · Safety'}
COLORS = {'astar_distance': '#737c8c', 'rl_safety': '#7250a2', 'sarsa_safety': '#218b65'}
METRICS = ('success', 'off_target', 'timeout', 'steps', 'safety_score', 'safety_return',
           'observed_risk_exposure', 'collision_attempts', 'drift_events', 'runtime_ms', 'policy_coverage')



def write_csv(path, rows):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)



def summarize(rows):
    groups = defaultdict(list)

    for row in rows:
        # Keep the pooled results and the individual tumour results.

        for target in ('all', row['target']):
            key = (target, row['mode'], row['split'], row['algorithm'])
            groups[key].append(row)

    summary = []

    for (target, mode, split, algorithm), group in groups.items():
        record = dict(target=target, mode=mode, split=split, algorithm=algorithm,
                      episodes=len(group), vessels=len({row['layout'] for row in group}))

        for metric in METRICS:
            record[metric] = mean(row[metric] for row in group)
        successful = [row['steps'] for row in group if row['success']]
        record['successful_steps'] = mean(successful) if successful else None
        summary.append(record)
    return summary



def plot_comparison(summary, algorithms, output):
    modes = {'deterministic': 'No stochastic drift', 'low': 'Low stochasticity',
             'stochastic': 'High stochasticity', 'very_high': 'Very high stochasticity'}
    panels = (
        ('success', 'Deliveries completed', 'Completed deliveries (%)'),
        ('successful_steps', 'Travel to successful delivery', 'Mean moves (successful trials)'),
        ('safety_score', 'Observed tissue protection', 'Safety score (0–100)'),
        ('safety_return', 'Total task reward', 'Safety return (reward points)'),
    )

    for mode in dict.fromkeys(r['mode'] for r in summary):
        fig, axes = plt.subplots(2, 4, figsize=(14, 8))
        trial_counts = []

        for row_index, split in enumerate(('seen', 'unseen')):
            group = [r for r in summary if r['target'] == 'all' and r['mode'] == mode and r['split'] == split]
            trial_counts.append(group[0]['episodes'])
            layout_ids = SEEN_STUDY_LAYOUTS if split == 'seen' else UNSEEN_LAYOUTS
            layout_label = f'{split.capitalize()} layouts {layout_ids[0]}–{layout_ids[-1]}'
            results = {row['algorithm']: row for row in group}

            for ax, (metric, title, units) in zip(axes[row_index], panels):
                values = [results[algorithm][metric] for algorithm in algorithms]
                ax.bar([LABELS[a] for a in algorithms], [v if v is not None else 0 for v in values],
                       color=[COLORS[a] for a in algorithms], width=.6)

                for index, value in enumerate(values):
                    if value is None:
                        ax.text(index, 0, 'No deliveries', ha='center', fontsize=8)
                ax.set_title(f'{layout_label}\n{title}', fontsize=10)
                ax.set_ylabel(units, fontsize=9)
                ax.tick_params(axis='x', labelsize=9)
                ax.grid(axis='y', alpha=.2)
                ax.set_axisbelow(True)
                if metric == 'success':
                    ax.set_ylim(0, 1.05)
                    ax.yaxis.set_major_formatter(PercentFormatter(1))
                if metric == 'safety_score':
                    ax.set_ylim(0, 100)
        labels = [LABELS[algorithm] for algorithm in algorithms]
        comparison = ', '.join(labels[:-1]) + ' and ' + labels[-1]
        fig.suptitle(f'{comparison}: tumour delivery in {modes[mode].lower()}\n'
                     f'{len(SEEN_STUDY_LAYOUTS)} seen and {len(UNSEEN_LAYOUTS)} unseen vessels; targets A, B and C; 50-unit dose; 160-move limit', fontsize=13)
        fig.text(.5, .045, f'{trial_counts[0]} seen and {trial_counts[1]} unseen trials per controller. '
                 'RL uses saved Safety policies; unseen trials transfer the layout-0 policy.',
                 ha='center', fontsize=9)
        fig.text(.5, .018, 'Safety score measures observed tissue exposure and contacts. '
                 'Safety return also includes delivery, travel, flow and failure penalties.',
                 ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .075, 1, .92), h_pad=2)
        fig.savefig(output / f'comparison_{mode}.png', dpi=160)
        plt.close(fig)



def run_comparison(name, args):
    algorithms = COMPARISONS[name]
    output = args.output_root / name
    output.mkdir(parents=True, exist_ok=True)
    rows, policy_sources = [], {}
    # Seen layouts use their own saved table. unseen layouts use layout 0.

    for target in args.targets:
        for mode in args.modes:
            for layout in (*SEEN_STUDY_LAYOUTS, *UNSEEN_LAYOUTS):
                split = 'unseen' if layout in UNSEEN_LAYOUTS else 'seen'
                world = BenchmarkWorld(layout, mode, target, MAX_LIVE_STEPS, 50)
                policies = {}

                for algorithm in algorithms:
                    if algorithm in CONTROLLERS:
                        policy, info = load_deployed_policy(CONTROLLERS[algorithm], target, mode, layout)
                        policies[algorithm] = policy
                        policy_sources[info['checkpoint']] = info

                for run in range(args.runs):
                    # Reuse the seed for each controller, rather than giving one an easier draw.
                    seed = 3000000 + layout * 1000 + run

                    for algorithm in algorithms:
                        record = evaluate(world, algorithm, seed, policies.get(algorithm))
                        record.update(target=target, split=split,
                                      source_layout=0 if split == 'unseen' else layout)
                        rows.append(record)
                print(f'{name}: {target} / {mode} / {split} layout {layout}', flush=True)
    summary = summarize(rows)
    write_csv(output / 'episodes.csv', rows)
    write_csv(output / 'summary.csv', summary)
    config = dict(comparison=name, algorithms=algorithms, targets=args.targets, modes=args.modes,
                  runs_per_target_layout=args.runs, seen_layouts=SEEN_STUDY_LAYOUTS, unseen_layouts=UNSEEN_LAYOUTS,
                  horizon=MAX_LIVE_STEPS, dose=50, seed_base=3000000, paired_seeds=True,
                  unseen_policy_source_layout=0, training_on_unseen=False, policy_sources=policy_sources,
                  safety_reward=asdict(SAFETY_REWARD),
                  environment_fingerprint=environment_fingerprint(),
                  healthy_cell_release_probability=INPUTS['reward']['healthy_cell_release_probability'],
                  safety_calibration=asdict(SAFETY_CALIBRATION),
                  selection='Fixed controllers; no sensor classification or route-risk selection in this experiment',
                  evaluator='Shared batch evaluator; A* chooses from the current microscope estimate. Live fixed-reference recovery and fleet conflicts are not evaluated here.')
    (output / 'config.json').write_text(json.dumps(config, indent=2)+'\n')
    plot_comparison(summary, algorithms, output)
    print(f'Saved {len(rows)} episodes: {output}', flush=True)



def validate_shared_trials(output_root):
    shared = COMPARISONS['astar_vs_rl']
    comparisons = []

    for name in COMPARISONS:
        records = {}

        with (output_root / name / 'episodes.csv').open() as file:
            for row in csv.DictReader(file):
                if row['algorithm'] not in shared:
                    continue
                key = tuple(row[field] for field in ('algorithm', 'target', 'mode', 'layout', 'seed'))
                # Timing changes between runs. the trajectory measurements should not.
                row.pop('runtime_ms')
                records[key] = row
        comparisons.append(records)
    if comparisons[0] != comparisons[1]:
        raise ValueError('Shared controller trials disagree across comparisons.')
    print(f'Validated {len(comparisons[0])} shared A*/Q-learning trials.', flush=True)



def main():
    parser = argparse.ArgumentParser(description='Compare saved Safety policies with distance A* on seen and unseen vessels.')
    parser.add_argument('--comparison', choices=('all', *COMPARISONS), default='all')
    parser.add_argument('--runs', type=int, default=50)
    parser.add_argument('--targets', nargs='+', choices=list('ABC'), default=list('ABC'))
    parser.add_argument('--modes', nargs='+', choices=tuple(ENVIRONMENT_PROFILES), default=list(ENVIRONMENT_PROFILES))
    parser.add_argument('--output-root', type=Path, default=ROOT / 'outputs' / 'experiments')
    args = parser.parse_args()
    if args.runs < 1:
        parser.error('runs must be positive')
    args.targets = list(dict.fromkeys(args.targets))
    args.modes = list(dict.fromkeys(args.modes))
    comparisons = COMPARISONS if args.comparison == 'all' else [args.comparison]

    for name in comparisons:
        run_comparison(name, args)
    if args.comparison == 'all':
        validate_shared_trials(args.output_root)


if __name__ == '__main__':
    main()
