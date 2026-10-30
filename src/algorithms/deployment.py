# Checkpoint names and the legacy export directory used by SQL migration.


# This was generated with an AI: I was able to write the scripts for Qlearning/Sarsa, 
# whose policies were saved. but I was unable to make deploy it live. 

from pathlib import Path
from src.simulation.live_controller import POLICIES

POLICY_DIR = Path(__file__).resolve().parents[2] / 'outputs' / 'deployed_policies'



def policy_key(target, controller, mode, layout):
    learner, objective = POLICIES[controller]
    return f'{target.upper()}_{learner}_{objective}_{mode}_layout{layout}'
