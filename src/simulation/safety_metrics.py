# Reward-independent safety proxy from an estimated risk map and contact logs.
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SafetyCalibration:
    risk_budget: float = 5.0
    contact_budget: float = 5.0

    def score(self, observed_risk_exposure, collision_attempts):
        if self.risk_budget <= 0 or self.contact_budget <= 0:
            raise ValueError('Safety reference budgets must be positive')
        if observed_risk_exposure < 0 or collision_attempts < 0:
            raise ValueError('Safety measurements must be nonnegative')
        return 100 * math.exp(-observed_risk_exposure/self.risk_budget
                              - collision_attempts/self.contact_budget)


SAFETY_CALIBRATION = SafetyCalibration()
