# SQL-backed deployment adapter. learner implementations remain unchanged.
from dataclasses import asdict
from io import BytesIO
from src.algorithms.deployment import POLICY_DIR, policy_key
from src.algorithms.q_learning import QLearningPolicy
from src.algorithms.sarsa import SARSAPolicy
from src.simulation.live_controller import POLICIES, MAX_LIVE_STEPS
from src.simulation.microdose_env import UNSEEN_LAYOUTS, environment_fingerprint
from src.simulation.rewards import SAFETY_REWARD
from src.storage import DB_PATH, import_checkpoints, read_model


def load_deployed_policy(controller, target, mode, layout, *, database_path=DB_PATH):
    source_layout = 0 if layout in UNSEEN_LAYOUTS else layout
    key = policy_key(target, controller, mode, source_layout)
    try:
        info, data = read_model(key, database_path)
    except FileNotFoundError:
        import_checkpoints(POLICY_DIR, database_path)
        info, data = read_model(key, database_path)
    if (info['target'], info['mode'], info['layout']) != (target.upper(), mode, source_layout):
        raise ValueError('Saved policy does not match the selected target and environment')
    learner, objective = POLICIES[controller]
    if info['reward_weights'] != asdict(SAFETY_REWARD):
        raise ValueError('Saved policy reward configuration is outdated; refresh models offline')
    if info.get('environment_fingerprint') != environment_fingerprint():
        raise ValueError('Saved policy dynamics are outdated; retrain policies offline')
    if info['max_steps'] != MAX_LIVE_STEPS:
        raise ValueError('Saved policy horizon does not match the live mission')
    cls = QLearningPolicy if learner == 'q_learning' else SARSAPolicy
    policy = cls.load(BytesIO(data))
    if policy.objective != objective:
        raise ValueError('Saved policy objective does not match the selected controller')
    info.update(evaluation_layout=layout, split='unseen' if layout in UNSEEN_LAYOUTS else 'seen',
                source_layout=source_layout, storage='sqlite')
    return policy, info
