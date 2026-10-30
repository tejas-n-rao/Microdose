# Mission arguments, tool messages and deployment results.

# Used React and Z3 lab files as references for the tools made. 

from dataclasses import dataclass
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator
from src.robot_identity import ROBOT_CARRIERS



class MissionParseError(RuntimeError):
    pass



class MissionClarificationRequired(MissionParseError):


    def __init__(self, questions, transcript, partial=None, alternatives=None, compatibility_options=None,
                 smt_rejection=None, sensor_prompt=None):
        super().__init__('Mission needs clarification: ' + ' '.join(questions))
        self.questions, self.transcript = questions, transcript
        self.partial, self.alternatives = partial or {}, alternatives or []
        self.compatibility_options = compatibility_options or []
        self.smt_rejection = smt_rejection or {}
        self.sensor_prompt = sensor_prompt or {}



# ReAct Lab/sriroute_tools.py: implemented one validated argument model per tool
class ToolArgs(BaseModel):
    model_config = {'extra': 'forbid'}



class CarrierMissionArgs(ToolArgs):
    target_id: Literal['A', 'B', 'C'] | None = None
    dose: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    carrier_id: Literal['X', 'Y', 'Z'] | None = None
    preference: Literal['safety', 'speed'] | None = None
    evidence: dict[str, str] = Field(default_factory=dict)


    @field_validator('target_id', 'carrier_id', mode='before')
    @classmethod
    def normalize_ids(cls, value):
        return value.strip().upper() if isinstance(value, str) else value


    @field_validator('carrier_id', mode='before')
    @classmethod
    def normalize_robot(cls, value):
        if isinstance(value, str):
            normalized = value.strip().upper()
            return ROBOT_CARRIERS.get(normalized, normalized)
        return value

# ReAct Lab/sriroute_tools.py: the simpler one-request argument shape.
# Alternative draft only, this would omit fleet and evidence fields.
# class SingleDeliveryArgs(ToolArgs):
#     target_id: str
#     carrier_id: str
#     dose: float



class SetMissionArgs(CarrierMissionArgs):
    layout: int | None = Field(default=None, ge=0, le=7, strict=True)
    mode: Literal['deterministic', 'low', 'stochastic', 'very_high'] | None = None
    urgent_target: Literal['A', 'B', 'C'] | None = None
    execution: Literal['simultaneous', 'sequential'] | None = None
    missions: list[CarrierMissionArgs] | None = Field(default=None, min_length=1, max_length=3)



class EmptyArgs(ToolArgs):
    pass



class AskUserArgs(ToolArgs):
    questions: list[str] = Field(min_length=1)



class FinishArgs(ToolArgs):
    final_answer: str



# thought, action and action_input message fields.
class AgentStep(ToolArgs):
    thought: str
    action: str
    action_input: dict[str, Any]

# i follwo sriroute_tools.py as dispatch selects the matching argument model.
TOOL_ARG_MODELS = {'set_mission': SetMissionArgs, 'verify_mission': EmptyArgs,
                   'verify_dispatch_smt': EmptyArgs, 'propose_alternatives': EmptyArgs, 'request_compatibility_override': EmptyArgs,
                   'ask_user': AskUserArgs, 'finish': FinishArgs}



@dataclass(frozen=True)
class MissionSpec:
    target_id: str
    dose: float
    carrier_id: str
    layout: int
    mode: str
    preference: str
    controller: str
    verification: dict[str, Any]
    transcript: list[dict[str, str]]
    sequence_index: int = 0
    execution: str = 'simultaneous'



# Small scratch helper for a console summary, not used by the UI.
def _requested_targets_note(missions):
    return tuple(dict.fromkeys(mission.target_id for mission in missions))



@dataclass(frozen=True)
class FleetMissionSpec:
    missions: list[MissionSpec]
    layout: int
    mode: str
    transcript: list[dict[str, str]]
    execution: str = 'simultaneous'
