# Check reagent consumed at release events, not during travel.


import math
from typing import Literal
import z3
from pydantic import BaseModel, Field, ValidationError



class ReleaseSpec(BaseModel):
    model_config = {'extra': 'forbid'}
    step: int = Field(ge=0, strict=True)
    amount: float = Field(gt=0, allow_inf_nan=False)



class DosageTrajectory(BaseModel):
    model_config = {'extra': 'forbid'}
    carrier_id: Literal['X', 'Y', 'Z']
    target_id: Literal['A', 'B', 'C']
    allocated_dose: float = Field(gt=0, allow_inf_nan=False)
    path: list[tuple[int, int]] = Field(min_length=1)
    releases: list[ReleaseSpec]



class DosagePayload(BaseModel):
    model_config = {'extra': 'forbid'}
    trajectories: list[DosageTrajectory] = Field(min_length=1, max_length=3)
    previously_used: dict[Literal['A', 'B', 'C'], float] = Field(default_factory=dict)



# manual budget note, kept outside the solver.
def _remaining_budget_note(required, used):
    return {target: max(0, budget - used.get(target, 0))
            for target, budget in required.items()}


# Wrong direction for an upper dose limit, kept as a reminder to fix.
# Z3Lab/verifier.py uses <= for capacity. This reverses that check.
# def old_dose_limit(amount, limit):
#     return amount >= limit



def add_dosage_constraints(solver, doses, required, previous=None):
    """Shared tracked real-arithmetic constraints for admission and trajectories.

    doses is [(target, amount), ...]. A fleet shares one budget per target.
    """
    previous = previous or {}

    for target, budget in required.items():
        # Z3Lab/verifier.py: sum assigned loads, adapted to released reagent units.
        total = z3.Sum([z3.RealVal(str(amount)) for name, amount in doses if name == target])
        used = z3.RealVal(str(previous.get(target, 0)))  # earlier releases count too
        solver.assert_and_track(total + used <= z3.RealVal(str(budget)), 'dosage_budget_' + target)


# Z3Lab/verifier.py: sum assigned loads before applying a capacity bound.
# Alternative draft, incomplete because it leaves out earlier releases.
# def _single_release_budget(amounts, budget):
#     return z3.Sum([z3.RealVal(str(amount)) for amount in amounts]) <= budget



def verify_dosage_smt(raw_payload):
    from src.verification.smt_verifier import RULES

    # Uses example from Z3 lab to validate, solve and return structured feedback.

    try:
        payload = DosagePayload.model_validate(raw_payload)

        for amount in payload.previously_used.values():
            if amount < 0 or not math.isfinite(amount):
                raise ValueError('Previously used amounts must be finite and nonnegative.')
    except (ValidationError, ValueError) as exc:
        return {'status': 'SYNTACTIC_ERROR', 'diagnostics': {'validation': str(exc)}, 'feedback': 'Correct the dosage trajectory.'}
    solver = z3.Solver()
    doses = []

    for index, trajectory in enumerate(payload.trajectories):
        amounts = [z3.RealVal(str(r.amount)) for r in trajectory.releases]
        solver.assert_and_track(z3.Sum(amounts) <= z3.RealVal(str(trajectory.allocated_dose)), f'carrier_payload_{index}')

        for release_index, release in enumerate(trajectory.releases):
            solver.assert_and_track(z3.BoolVal(release.step < len(trajectory.path)), f'release_on_path_{index}_{release_index}')
            doses.append((trajectory.target_id, release.amount))
    add_dosage_constraints(solver, doses, RULES['required_doses'], payload.previously_used)
    totals = {target: payload.previously_used.get(target, 0) + sum(amount for name, amount in doses if name == target)
              for target in RULES['required_doses']}
    diagnostics = {'used_by_target': totals, 'required_by_target': RULES['required_doses'],
                   'excess_by_target': {t: max(0, totals[t]-limit) for t, limit in RULES['required_doses'].items()}}

    status = solver.check()
    if status == z3.sat:
        return {'status': 'SATISFIABLE', 'diagnostics': diagnostics, 'feedback': 'Trajectory release amounts fit target budgets and carrier payloads.'}
    diagnostics['unsat_core'] = [str(c) for c in solver.unsat_core()] if status == z3.unsat else []
    return {'status': 'UNSATISFIABLE' if status == z3.unsat else 'UNKNOWN', 'diagnostics': diagnostics,
            'feedback': 'Reject trajectory. Reduce release amounts, remove repeat releases, or redistribute doses within the required target budget.'}



class DosageSMTVerifier:


    def verify(self, payload, *, layout=0, mode='deterministic', compatibility_approvals=None,
               parent_plan_hash=None):
        # Check concrete releases and compatibility. overrides needed
        result = verify_dosage_smt(payload)
        if result['status'] != 'SATISFIABLE':
            return result
        from src.verification.smt_verifier import verify_dispatch_smt
        trajectories = DosagePayload.model_validate(payload).trajectories
        admission = verify_dispatch_smt(dict(missions=[dict(carrier_id=t.carrier_id,
            target_id=t.target_id, dose=t.allocated_dose, preference='speed') for t in trajectories],
            layout=layout, mode=mode), compatibility_approvals=compatibility_approvals,
            parent_plan_hash=parent_plan_hash)
        return {**admission, 'dosage': result}
