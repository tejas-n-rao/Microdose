from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Callable, Iterable

Position = tuple[int, int]



# Derives heavily from Lab1/environment.py: It is the same snippet, just adapted to my framework
@dataclass
class GridWorld:
    # Minimal grid abstraction shared by planners and controllers.

    width: int
    height: int
    obstacles: set[Position] = field(default_factory=set)
    hazard_mean: dict[Position, float] = field(default_factory=dict)
    flow_cost: dict[Position, float] = field(default_factory=dict)


    @classmethod
    def from_obstacles(
        cls,
        width: int,
        height: int,
        obstacles: Iterable[Position],
        hazard_mean: dict[Position, float] | None = None,
        flow_cost: dict[Position, float] | None = None,
    ) -> "GridWorld":
        return cls(
            width=width,
            height=height,
            obstacles=set(obstacles),
            hazard_mean=hazard_mean or {},
            flow_cost=flow_cost or {},
        )


    def is_valid_coord(self, x: int, y: int) -> bool:
        # same coordinate bounds check as environment.py
        return 0 <= x < self.width and 0 <= y < self.height


    def is_valid(self, position: Position) -> bool:
        x, y = position
        return self.is_valid_coord(x, y) and position not in self.obstacles


    def get_neighbors(self, position: Position) -> list[Position]:
        x, y = position

        # four cardinal neighbours like in reference script, with tuple positions here.
        candidates = [(x, y + 1), (x + 1, y), (x, y - 1), (x - 1, y)]
        return [candidate for candidate in candidates if self.is_valid(candidate)]  # only moves that fit


    def manhattan_distance(self, a: Position, b: Position) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])


    def euclidean_distance(self, a: Position, b: Position) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


    def chebyshev_distance(self, a: Position, b: Position) -> float:
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


    def transition_cost(
        self,
        state: Position,
        action: Position,
        next_state: Position,
        cost_fn: Callable[[Position, Position, Position], float] | None = None,
    ) -> float:
        if cost_fn is not None:
            return cost_fn(state, action, next_state)

        return 1.0
