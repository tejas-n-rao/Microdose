"""Synthetic environment classifier; supplied detection rates, no route inputs.

Rates are P(alert | actual mode), not posterior probabilities. The simulator
supplies actual mode; only the returned prediction is used for admission.
"""
import random
from dataclasses import dataclass, asdict
from uuid import uuid4


@dataclass(frozen=True)
class EnvironmentObservation:
    observation_id: str
    predicted_mode: str

    def as_dict(self):
        return asdict(self)


def observe_environment(actual_mode, rng=None):
    from src.verification.smt_verifier import RULES
    rates = RULES['environment_sensor']['very_high_alert_rates']
    if actual_mode not in rates:
        raise ValueError('Unknown environment mode')
    alert = (rng or random).random() < rates[actual_mode]
    predicted = 'very_high' if alert else ('stochastic' if actual_mode == 'very_high' else actual_mode)
    return EnvironmentObservation(uuid4().hex, predicted).as_dict()


def is_very_high_alert(observation):
    return bool(observation and observation.get('predicted_mode') == 'very_high')
