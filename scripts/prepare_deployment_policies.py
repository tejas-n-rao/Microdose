#!/usr/bin/env python3
# Prepare target-specific Safety policies offline. reuse current SQL models.
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.algorithms.deployment import POLICY_DIR, policy_key
from src.algorithms.q_learning import QLearningPolicy
from src.algorithms.sarsa import SARSAPolicy
from src.policy_store import load_deployed_policy
from src.storage import DB_PATH, import_checkpoints, save_model
from src.simulation.rl_environment import BenchmarkWorld
from src.simulation.live_controller import POLICIES, MAX_LIVE_STEPS
from src.simulation.microdose_env import environment_fingerprint
from src.simulation.rewards import SAFETY_REWARD

CONTROLLER_KEYS = {'q_learning_safety': 'Q-learning · Safety', 'sarsa_safety': 'SARSA · Safety'}



def train_checkpoint(task):
    target, mode, layout, controller, episodes = task
    learner, objective = POLICIES[controller]
    cls = QLearningPolicy if learner == 'q_learning' else SARSAPolicy
    world = BenchmarkWorld(layout, mode, target, MAX_LIVE_STEPS, 50)
    policy = cls(objective, 10000 + layout).train(world, episodes=episodes)
    key = policy_key(target, controller, mode, layout)
    buffer = BytesIO()
    policy.save(buffer)
    info = dict(checkpoint=f'{key}.npz', target=target, mode=mode, layout=layout,
        learner=learner, objective=objective, dose=50, max_steps=MAX_LIVE_STEPS,
        training_episodes=len(policy.training_curve), training_seed=10000+layout,
        reward_weights=asdict(SAFETY_REWARD), source='offline training',
        environment_fingerprint=environment_fingerprint())
    return key, info, buffer.getvalue()



def main():
    parser = argparse.ArgumentParser(description='Prepare target-specific Safety policies offline; reuse current SQL models.')
    parser.add_argument('--episodes', type=int, default=12000)
    parser.add_argument('--targets', nargs='+', choices=list('ABC'), default=list('ABC'))
    parser.add_argument('--layouts', nargs='+', type=int, choices=range(5), default=list(range(5)))
    parser.add_argument('--controllers', nargs='+', choices=CONTROLLER_KEYS, default=list(CONTROLLER_KEYS))
    parser.add_argument('--modes', nargs='+', choices=['deterministic','low','stochastic','very_high'],
                        default=['deterministic','low','stochastic','very_high'])
    parser.add_argument('--workers', type=int, default=4, help='Independent training processes; SQL writes stay in the parent.')
    parser.add_argument('--force', action='store_true', help='Retrain instead of reusing SQL checkpoints.')
    args = parser.parse_args()
    if args.episodes < 1 or args.workers < 1:
        parser.error('episodes and workers must be positive')
    import_checkpoints(POLICY_DIR)
    tasks = []

    for target in dict.fromkeys(args.targets):
        for mode in dict.fromkeys(args.modes):
            for layout in dict.fromkeys(args.layouts):
                for controller in dict.fromkeys(CONTROLLER_KEYS[key] for key in args.controllers):
                    if not args.force:
                        try:
                            _, info = load_deployed_policy(controller, target, mode, layout)
                            if info['training_episodes'] == args.episodes:
                                continue
                        except (FileNotFoundError, ValueError):
                            pass
                    tasks.append((target, mode, layout, controller, args.episodes))

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, (key, info, data) in enumerate(pool.map(train_checkpoint, tasks), start=1):
            save_model(key, info, data)
            print(f'Ready {index}/{len(tasks)}: {key}', flush=True)
    print(f'Deployment models: {DB_PATH}', flush=True)


if __name__ == '__main__':
    main()
