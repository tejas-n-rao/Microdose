# Safety reward components. Hazard is charged once per move, never at release.
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RewardWeights:
    step: float = 0.75
    hazard: float = 15.0
    flow: float = 2.0
    collision: float = 10.0
    off_target: float = 50.0
    timeout: float = 120.0
    therapy: float = 1.0

    def components(self, *, hazard=0.0, flow=0.0, collision=False,
                   movement=False, terminal=None, therapeutic_value=0.0):
        return {
            'time': -self.step if movement else 0.0,
            'hazard': -self.hazard * hazard if movement else 0.0,
            'flow': -self.flow * flow if movement else 0.0,
            'collision': -self.collision if collision else 0.0,
            'therapy': self.therapy * therapeutic_value if terminal == 'delivery' else 0.0,
            'off_target': -self.off_target if terminal == 'off_target' else 0.0,
            'timeout': -self.timeout if terminal == 'timeout' else 0.0,
        }


SAFETY_REWARD = RewardWeights()


def therapy_components(dose, steps, config):
    efficacy = max(config['efficacy_floor'], math.exp(-config['efficacy_decay_per_step'] * steps))
    benefit = config['benefit_max'] * dose**2 / (config['benefit_ec50']**2 + dose**2)
    toxicity = config['toxicity_max'] * dose**3 / (config['toxicity_tc50']**3 + dose**3)
    return {'benefit': benefit * efficacy, 'toxicity': -toxicity}
