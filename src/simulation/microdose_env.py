from __future__ import annotations

import math
import random
from dataclasses import dataclass
from functools import lru_cache
import json
from hashlib import sha256
from pathlib import Path
from typing import Callable

from src.algorithms import GridWorld, Position

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUTS_PATH = PROJECT_ROOT / "config" / "microdose_inputs.json"



def load_inputs(path: Path = INPUTS_PATH) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


INPUTS = load_inputs()



def environment_fingerprint():
    # A checkpoint trained under old physics must not be silently reused.
    return sha256(json.dumps(INPUTS, sort_keys=True).encode()).hexdigest()


GRID = int(INPUTS["grid"]["width"])
START: Position = tuple(INPUTS["grid"]["start"])  # type: ignore[assignment]
LAYOUT_SEEDS = tuple(range(5))
SEEN_STUDY_LAYOUTS = (0, 1, 2, 3, 4)
UNSEEN_LAYOUTS = (5, 6, 7)
MIDDLE_CORRIDOR_HEALTHY_LAYOUTS = (0, 2)
ROBOT_CARRIERS = ('X', 'Y', 'Z')



def vessel_label(seed):
    value = layout_seed(seed)
    if value in UNSEEN_LAYOUTS:
        return f"UNSEEN vessel {UNSEEN_LAYOUTS.index(value)+1} · layout {value}"
    return f"Seen vessel {value+1} · layout {value}"


TUMOURS = {
    tumour_id: {
        **data,
        "true": tuple(data["true"]),
        "estimated": tuple(data["estimated"]),
    }
    for tumour_id, data in INPUTS["tumours"].items()
}



@dataclass(frozen=True)
class VesselLayout:
    traversable: frozenset[Position]
    blocked: frozenset[Position]
    boundary: frozenset[Position]
    deadspace: frozenset[Position]
    center_chaos: tuple[tuple[Position, float], ...]
    paths: tuple[tuple[Position, ...], ...]



def layout_seed(seed: int | None) -> int:
    return 0 if seed is None else int(seed) if int(seed) in UNSEEN_LAYOUTS else int(seed) % len(LAYOUT_SEEDS)



def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))



def _seeded_layout_value(seed: int | None, salt: int) -> random.Random:
    return random.Random(layout_seed(seed) * 10_003 + salt)



def _tube_cells(
    x_start: int,
    x_end: int,
    center_y_fn: Callable[[int], float],
    half_width: float,
    center_chaos: dict[Position, float],
    traversable: set[Position],
    chaos_weight: float,
) -> None:
    for x in range(max(0, x_start), min(GRID, x_end + 1)):
        center_y = center_y_fn(x)
        y_min = max(0, int(math.floor(center_y - half_width - 1)))
        y_max = min(GRID - 1, int(math.ceil(center_y + half_width + 1)))

        for y in range(y_min, y_max + 1):
            distance = abs(y - center_y)
            if distance <= half_width:
                position = (x, y)
                traversable.add(position)
                normalized_center = max(0.0, 1.0 - distance / max(half_width, 1.0))
                center_chaos[position] = max(center_chaos.get(position, 0.0), chaos_weight * normalized_center)



def _centerline_path(x_start: int, x_end: int, center_y_fn: Callable[[int], float]) -> tuple[Position, ...]:
    # Rasterize a smooth centreline as a cardinal, contiguous path.
    path: list[Position] = []
    previous: Position | None = None

    for x in range(max(0, x_start), min(GRID, x_end + 1)):
        y = int(round(clamp(center_y_fn(x), 0, GRID - 1)))
        if previous is None:
            path.append((x, y))
        else:
            px, py = previous
            if x != px:
                path.append((x, py))

            while path[-1][1] != y:
                cy = path[-1][1]
                path.append((x, cy + (1 if y > cy else -1)))
        previous = (x, y)
    return tuple(dict.fromkeys(path))



@lru_cache(maxsize=None)
def vessel_layout(seed: int | None = None) -> VesselLayout:
    rng = _seeded_layout_value(seed, salt=907)
    center_y = 24 + rng.randint(-2, 2)
    left_join = 8 + rng.randint(-1, 2)
    right_join = 41 + rng.randint(-2, 2)
    top_amp = 13 + rng.randint(-2, 2)
    bottom_amp = 12 + rng.randint(-2, 3)
    island_cx = 25 + rng.randint(-2, 2)
    island_cy = center_y + rng.randint(-1, 1)
    island_rx = 7.2 + rng.uniform(-1.0, 1.2)
    island_ry = 4.3 + rng.uniform(-0.8, 0.8)
    phase = rng.uniform(0, math.tau)
    wiggle = rng.uniform(0.5, 1.4)
    branch_width = 3.2 + rng.uniform(-0.3, 0.5)
    main_width = 4.2 + rng.uniform(-0.2, 0.4)
    # Evaluation-only geometry variants. IDs 5–7 are never mapped onto training IDs.
    if layout_seed(seed) == 5:
        top_amp, bottom_amp, branch_width, main_width = 17, 8, 2.4, 3.8
        island_rx, island_ry = 8.0, 5.0
    elif layout_seed(seed) == 6:
        top_amp, bottom_amp, branch_width, main_width = 10, 17, 2.6, 4.0
        island_cx, island_rx, island_ry = 28, 8.8, 4.7
    elif layout_seed(seed) == 7:
        top_amp, bottom_amp, branch_width, main_width = 15, 14, 2.0, 3.0
        island_rx, island_ry, wiggle = 9.0, 3.8, 1.8
    chaos_weight = float(INPUTS["field_model"].get("center_chaos", {}).get("weight", 0.16))

    traversable: set[Position] = set()
    center_chaos: dict[Position, float] = {}


    def main_y(x: int) -> float:
        return center_y + wiggle * math.sin(x / 6.5 + phase)


    def arch_y(x: int, amplitude: float, sign: int) -> float:
        span = max(1, right_join - left_join)
        t = clamp((x - left_join) / span, 0.0, 1.0)
        return center_y + sign * amplitude * math.sin(math.pi * t) + wiggle * math.sin(x / 5.0 + phase / 2)

    _tube_cells(0, GRID - 1, main_y, main_width, center_chaos, traversable, chaos_weight=chaos_weight)
    _tube_cells(left_join, right_join, lambda x: arch_y(x, top_amp, -1), branch_width, center_chaos, traversable, chaos_weight=chaos_weight * 0.7)
    _tube_cells(left_join, right_join, lambda x: arch_y(x, bottom_amp, 1), branch_width, center_chaos, traversable, chaos_weight=chaos_weight * 0.7)
    paths = (
        _centerline_path(0, GRID - 1, main_y),
        _centerline_path(left_join, right_join, lambda x: arch_y(x, top_amp, -1)),
        _centerline_path(left_join, right_join, lambda x: arch_y(x, bottom_amp, 1)),
    )
    defined_paths = set().union(*paths)
    traversable |= defined_paths

    for position in defined_paths:
        center_chaos.setdefault(position, chaos_weight * 0.45)

    # Keep the reference central void while preserving a narrow, explicit
    # middle corridor. Together with the upper and lower arches this gives
    # every layout three actual routes through the vessel.

    for x, y in list(traversable):
        ellipse = ((x - island_cx) / island_rx) ** 2 + ((y - island_cy) / island_ry) ** 2
        if ellipse <= 1.0 and (x, y) not in defined_paths:
            traversable.remove((x, y))
            center_chaos.pop((x, y), None)

    explicit_blocked = {tuple(point) for point in INPUTS["grid"].get("blocked", [])}
    traversable -= explicit_blocked  # type: ignore[arg-type]
    all_cells = {(x, y) for y in range(GRID) for x in range(GRID)}
    blocked = all_cells - traversable
    boundary = {
        position
        for position in blocked
        if any(neighbor in traversable for neighbor in _cardinal_candidates(position))
    }
    deadspace = blocked - boundary

    return VesselLayout(
        traversable=frozenset(traversable),
        blocked=frozenset(blocked),
        boundary=frozenset(boundary),
        deadspace=frozenset(deadspace),
        center_chaos=tuple(sorted(center_chaos.items())),
        paths=paths,
    )



def _cardinal_candidates(position: Position) -> list[Position]:
    x, y = position
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]



def center_chaos_lookup(seed: int | None = None) -> dict[Position, float]:
    return dict(vessel_layout(seed).center_chaos)



@lru_cache(maxsize=None)
def wall_distance_lookup(seed: int | None = None) -> dict[Position, int]:
    layout = vessel_layout(seed)
    distances: dict[Position, int] = {}

    for position in layout.traversable:
        distances[position] = min(manhattan(position, boundary) for boundary in layout.boundary)
    return distances



@lru_cache(maxsize=None)
def local_opening_depth_lookup(seed: int | None = None) -> dict[Position, int]:
    wall_distance = wall_distance_lookup(seed)
    depths: dict[Position, int] = {}

    for x, y in wall_distance:
        local_depth = 0

        for nx in range(max(0, x - 4), min(GRID, x + 5)):
            for ny in range(max(0, y - 4), min(GRID, y + 5)):
                if manhattan((x, y), (nx, ny)) <= 4:
                    local_depth = max(local_depth, wall_distance.get((nx, ny), 0))
        depths[(x, y)] = local_depth
    return depths



def generate_vein_obstacles(seed: int | None = None) -> set[Position]:
    return set(vessel_layout(seed).blocked)



def configured_blocked(seed: int | None = None) -> set[Position]:
    explicit = {tuple(point) for point in INPUTS["grid"].get("blocked", [])}
    if INPUTS["grid"].get("obstacle_style") == "veins":
        explicit |= generate_vein_obstacles(seed)
    return explicit  # type: ignore[return-value]


BLOCKED: set[Position] = configured_blocked(0)



@dataclass(frozen=True)
class EnvironmentProfile:
    label: str
    perception_noise: float
    drift_noise_bonus: float
    hazard_decay: float
    hazard_diffusion: float
    flow_decay: float
    hazard_source_scale: float
    flow_source_scale: float
    uncertainty_scale: float
    horizontal_slip: float
    left_bias: float
    vertical_slip: float
    down_bias: float


ENVIRONMENT_PROFILES: dict[str, EnvironmentProfile] = {
    mode: EnvironmentProfile(**profile)
    for mode, profile in INPUTS["environment_profiles"].items()
}



def gaussian(x: int, y: int, cx: float, cy: float, sigma: float) -> float:
    dx = x - cx
    dy = y - cy
    return math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))



def manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])



def seeded_rng(seed: int | None, salt: int) -> random.Random:
    return random.Random(layout_seed(seed) * 10_003 + salt)



def seeded_position(
    base: Position,
    jitter: int,
    seed: int | None,
    salt: int,
    blocked: set[Position] | None = None,
    avoid: set[Position] | None = None,
) -> Position:
    blocked = blocked if blocked is not None else configured_blocked(seed)
    avoid = avoid or set()
    rng = seeded_rng(seed, salt)

    for _attempt in range(200):
        candidate = (
            int(clamp(base[0] + rng.randint(-jitter, jitter), 0, GRID - 1)),
            int(clamp(base[1] + rng.randint(-jitter, jitter), 0, GRID - 1)),
        )
        if candidate not in blocked and candidate not in avoid:
            return candidate
    return nearest_traversable(base, blocked, avoid)



def nearest_traversable(
    base: Position,
    blocked: set[Position] | frozenset[Position],
    avoid: set[Position] | None = None,
) -> Position:
    avoid = avoid or set()
    bx, by = base
    candidates = [
        (x, y)
        for y in range(GRID)
        for x in range(GRID)
        if (x, y) not in blocked and (x, y) not in avoid
    ]
    if not candidates:
        return base
    return min(candidates, key=lambda point: (manhattan(point, (bx, by)), point[1], point[0]))



def seeded_start(seed: int | None = None) -> Position:
    jitter = int(INPUTS["grid"].get("seed_jitter", 0))
    blocked = configured_blocked(seed)
    return seeded_position(START, jitter, seed, salt=101, blocked=blocked)



@lru_cache(maxsize=None)
def seeded_robot_starts(seed: int | None = None) -> dict[str, Position]:
    # Return three distinct, visible launch cells near the vessel entrance.
    anchor = seeded_start(seed)
    traversable = vessel_layout(seed).traversable
    entrance_x = min(x for x, _y in traversable)
    candidates = [p for p in traversable if p[0] <= entrance_x + 7]
    desired = ((anchor[0], anchor[1] - 3), anchor, (anchor[0], anchor[1] + 3))
    starts: dict[str, Position] = {}
    occupied: set[Position] = set()

    for carrier, target in zip(ROBOT_CARRIERS, desired):
        available = [p for p in candidates if p not in occupied]
        separated = [p for p in available if all(manhattan(p, q) >= 2 for q in occupied)]
        pool = separated or available
        chosen = min(pool, key=lambda p: (manhattan(p, target), p[0], p[1]))
        starts[carrier] = chosen
        occupied.add(chosen)
    return starts



def seeded_robot_start(seed: int | None = None, carrier_id: str | None = None) -> Position:
    return seeded_robot_starts(seed).get(str(carrier_id).upper(), seeded_start(seed))



def seeded_tumours(seed: int | None = None) -> dict[str, dict]:
    tumours: dict[str, dict] = {}
    blocked = configured_blocked(seed)
    occupied = set(seeded_robot_starts(seed).values())

    for index, (tumour_id, tumour) in enumerate(TUMOURS.items()):
        base_true = tumour["true"]
        jitter = int(tumour.get("seed_jitter", 0))
        true = seeded_position(base_true, jitter, seed, salt=300 + index, blocked=blocked, avoid=occupied)
        occupied.add(true)
        estimated_offset = (
            tumour["estimated"][0] - base_true[0],
            tumour["estimated"][1] - base_true[1],
        )
        estimated = (
            int(clamp(true[0] + estimated_offset[0], 0, GRID - 1)),
            int(clamp(true[1] + estimated_offset[1], 0, GRID - 1)),
        )
        tumours[tumour_id] = {
            **tumour,
            "true": true,
            "estimated": estimated,
        }
    return tumours



def seeded_healthy_tissue(seed: int | None = None) -> list[dict]:
    blobs: list[dict] = []
    blocked = configured_blocked(seed)
    wall_distance = wall_distance_lookup(seed)
    local_depth = local_opening_depth_lookup(seed)
    candidates = [
        position
        for position, distance in wall_distance.items()
        if distance == 1 and local_depth.get(position, 0) >= 2 and position not in blocked
    ]
    if not candidates:
        candidates = [
            position
            for position, distance in wall_distance.items()
            if 1 <= distance <= 2 and position not in blocked
        ]
    if not candidates:
        candidates = [position for position in wall_distance if position not in blocked]
    occupied = set(seeded_robot_starts(seed).values())
    occupied.update(tumour["true"] for tumour in seeded_tumours(seed).values())

    for index, blob in enumerate(INPUTS["field_model"].get("healthy_tissue", [])):
        rng = seeded_rng(seed, salt=700 + index)
        ordered = candidates[:]
        rng.shuffle(ordered)
        if layout_seed(seed) in MIDDLE_CORRIDOR_HEALTHY_LAYOUTS and index < 2:
            # Put two visible tissue patches directly on the middle route. A*
            # remains distance-only and crosses it when shortest. the safety
            # learner is rewarded for taking an arch when that reduces risk.
            route = vessel_layout(seed).paths[0]
            middle = [p for p in route if GRID * 0.38 <= p[0] <= GRID * 0.62 and p not in occupied]
            target_x = GRID // 2 + (-5 if index == 0 else 5)
            center = min(middle, key=lambda p: (abs(p[0] - target_x), abs(p[1] - GRID // 2)))
        else:
            center = next(
                (
                    candidate
                    for candidate in ordered
                    if candidate not in occupied
                    and all(manhattan(candidate, existing) >= int(blob.get("min_spacing", 6)) for existing in occupied)
                ),
                nearest_traversable((int(round(blob["cx"])), int(round(blob["cy"]))), blocked, occupied),
            )
        occupied.add(center)
        blobs.append(
            {
                **blob,
                "cx": float(center[0]),
                "cy": float(center[1]),
                "sigma": max(0.95, float(blob.get("sigma", 2.0)) * 0.55),
            }
        )
    return blobs



def vessel_safety_bonus(position: Position, seed: int | None = None) -> float:
    safety_config = INPUTS["field_model"].get("wall_safety", {})
    wall_distance = wall_distance_lookup(seed).get(position, 0)
    local_depth = local_opening_depth_lookup(seed).get(position, 0)
    thick_depth = int(safety_config.get("thick_opening_min_depth", 3))
    near_wall_bonus = float(safety_config.get("near_wall_bonus", 0.0))
    center_bonus = float(safety_config.get("narrow_center_bonus", 0.0))

    if 1 <= wall_distance <= 2 and local_depth >= thick_depth:
        return near_wall_bonus * (1.0 - abs(wall_distance - 1.5) / 2.0)

    if local_depth < thick_depth and wall_distance == local_depth:
        return center_bonus

    return 0.0



def healthy_tissue_risk_from_blobs(position: Position, healthy_tissue: list[dict]) -> float:
    x, y = position
    return sum(
        blob["weight"] * gaussian(x, y, blob["cx"], blob["cy"], blob["sigma"])
        for blob in healthy_tissue
    )



def create_fields(seed: int | None = None) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    field_model = INPUTS["field_model"]
    shear_band = field_model["shear_band"]
    flow_config = field_model["flow"]
    blocked = configured_blocked(seed)
    chaos_by_cell = center_chaos_lookup(seed)
    healthy_tissue = seeded_healthy_tissue(seed)
    hazard: list[list[float]] = []
    flow: list[list[float]] = []
    variance: list[list[float]] = []

    for y in range(GRID):
        hazard_row: list[float] = []
        flow_row: list[float] = []
        variance_row: list[float] = []

        for x in range(GRID):
            if (x, y) in blocked:
                hazard_row.append(0.0)
                flow_row.append(0.0)
                variance_row.append(0.0)
                continue
            healthy_tissue_risk = healthy_tissue_risk_from_blobs((x, y), healthy_tissue)
            shear_cost = (
                shear_band["cost"]
                if y == shear_band["y"] and shear_band["x_min_exclusive"] < x < shear_band["x_max_exclusive"]
                else 0
            )
            center_chaos = chaos_by_cell.get((x, y), 0.0)
            safety_bonus = vessel_safety_bonus((x, y), seed)
            edge_flow = abs(math.sin((x + 2) * flow_config["x_frequency"]) * math.cos((y + 1) * flow_config["y_frequency"]))
            right_extra = flow_config["right_side_extra_cost"] if x > flow_config["right_side_extra_cost_x_gt"] else 0
            total_risk = healthy_tissue_risk + shear_cost + center_chaos - safety_bonus
            hazard_row.append(clamp(total_risk, 0, 1))
            flow_row.append(clamp(edge_flow * flow_config["scale"] + right_extra + center_chaos * 0.7, 0, 1))
            variance_row.append(clamp(0.08 + edge_flow * 0.18 + total_risk * 0.16, 0, 0.48))
        hazard.append(hazard_row)
        flow.append(flow_row)
        variance.append(variance_row)

    return hazard, flow, variance



def matrix_to_cost_dict(matrix: list[list[float]]) -> dict[Position, float]:
    return {
        (x, y): matrix[y][x]
        for y in range(len(matrix))
        for x in range(len(matrix[y]))
    }



def make_grid(hazard: list[list[float]], flow: list[list[float]], blocked: set[Position] | None = None) -> GridWorld:
    blocked = blocked if blocked is not None else BLOCKED
    return GridWorld.from_obstacles(
        width=GRID,
        height=GRID,
        obstacles=blocked,
        hazard_mean=matrix_to_cost_dict(hazard),
        flow_cost=matrix_to_cost_dict(flow),
    )



def is_traversable(position: Position, blocked: set[Position] = BLOCKED) -> bool:
    x, y = position
    return 0 <= x < GRID and 0 <= y < GRID and position not in blocked



def neighbors(position: Position, blocked: set[Position] = BLOCKED) -> list[Position]:
    candidates = _cardinal_candidates(position)
    return [candidate for candidate in candidates if is_traversable(candidate, blocked)]



def unit_transition_cost() -> Callable[[Position, Position, Position], float]:
    # Distance A* is deliberately risk-blind: its operational objective is the
    # declared speed reward, whose only non-zero movement term is step cost.
    return lambda _state, _action, _next_state: 1.0



def apply_slip(
    intended: Position,
    profile: EnvironmentProfile,
    rng: random.Random,
    blocked: set[Position] = BLOCKED,
) -> tuple[Position, bool]:
    if profile.horizontal_slip == 0 and profile.vertical_slip == 0:
        return intended, False

    horizontal_slip = rng.random() < profile.horizontal_slip
    vertical_slip = rng.random() < profile.vertical_slip
    dx = -1 if horizontal_slip and rng.random() < profile.left_bias else 1 if horizontal_slip else 0
    dy = 1 if vertical_slip and rng.random() < profile.down_bias else -1 if vertical_slip else 0

    x, y = intended
    candidates = [
        (x + dx, y + dy),
        (x + dx, y),
        (x, y + dy),
        intended,
    ]
    moved = next((candidate for candidate in candidates if is_traversable(candidate, blocked)), intended)
    return moved, moved != intended



def microscope_estimate(
    actual: Position,
    profile: EnvironmentProfile,
    rng: random.Random,
    drifted: bool,
    blocked: set[Position] = BLOCKED,
) -> Position:
    sensor_config = INPUTS.get("sensors", {})
    base_noise = float(sensor_config.get("microscope_position_noise", profile.perception_noise))
    drift_bonus = float(sensor_config.get("microscope_drift_noise_bonus", profile.drift_noise_bonus))
    noise_chance = base_noise + (drift_bonus if drifted else 0)
    if noise_chance == 0:
        return actual

    dx = rng.choice([-1, 0, 1]) if rng.random() < noise_chance else 0
    dy = rng.choice([-1, 0, 1]) if rng.random() < noise_chance else 0
    estimated = (
        int(clamp(actual[0] + dx, 0, GRID - 1)),
        int(clamp(actual[1] + dy, 0, GRID - 1)),
    )
    if is_traversable(estimated, blocked):
        return estimated
    return nearest_traversable(estimated, blocked)



def near_tumour_release_reward(actual: Position, target: dict) -> float:
    reward_config = INPUTS["reward"]
    radius = int(reward_config.get("near_tumour_release_radius", 1))
    if manhattan(actual, target["true"]) <= radius:
        return float(reward_config.get("near_tumour_release_bonus", 0))
    return 0.0



def tumour_reached(actual: Position, target: dict) -> bool:
    return manhattan(actual, target["true"]) <= int(target.get("radius", 0))



def release_sensor_thinks_reached(
    actual: Position,
    estimated: Position,
    target: dict,
    rng: random.Random,
    accuracy: float | None = None,
) -> bool:
    sensor_config = INPUTS.get("sensors", {})
    sensor_accuracy = accuracy if accuracy is not None else float(sensor_config.get("tumour_reach_accuracy", 0.9))
    goal = target["true"]
    radius = int(target.get("radius", 0))
    actual_reached = tumour_reached(actual, target)
    estimated_reached = manhattan(estimated, goal) <= radius

    if not actual_reached and not estimated_reached:
        return False

    if rng.random() < sensor_accuracy:
        return actual_reached
    return not actual_reached
