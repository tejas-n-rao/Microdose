from __future__ import annotations
import heapq
import time
from dataclasses import dataclass
from typing import Callable, Literal

from .grid import GridWorld, Position

HeuristicName = Literal["manhattan", "euclidean", "chebyshev", "dijkstra"] # Ended up using manhattan because of gridwerld setting
CostFunction = Callable[[Position, Position, Position], float]



@dataclass
class SearchNode:
    # Emulates Lab1/models.py

    x: int
    y: int
    g: float = float("inf")
    h: float = float("inf")
    f: float = float("inf")
    parent: "SearchNode | None" = None


    @property
    def position(self) -> Position:
        return (self.x, self.y)


    def __lt__(self, other: "SearchNode") -> bool:
        return self.f < other.f


    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SearchNode):
            return False
        return self.position == other.position


    def __hash__(self) -> int:
        return hash(self.position)



@dataclass(frozen=True)
class AStarResult:
    path: list[Position]
    nodes_expanded: int
    elapsed_seconds: float
    total_cost: float
    heuristic: HeuristicName


# Lab1/planner.py: the goal branch originally reconstructed the path inline.
# Alternative draft, not called by search.
# def _inline_parent_walk(node):
#     path = []
#     while node is not None:
#         path.append(node.position)
#         node = node.parent
#     return path[::-1]



class AStarPlanner:
    # A* search over a MicroDose grid.


    def __init__(self, grid: GridWorld):
        self.grid = grid


    def search(
        self,
        start: Position,
        goal: Position,
        heuristic: HeuristicName = "manhattan",
        cost_fn: CostFunction | None = None,
    ) -> AStarResult:
        if not self.grid.is_valid(start):
            raise ValueError(f"Start {start} is outside the grid or blocked.")
        if not self.grid.is_valid(goal):
            raise ValueError(f"Goal {goal} is outside the grid or blocked.")

        started_at = time.time()
        nodes_expanded = 0

        start_node = SearchNode(start[0], start[1], g=0.0)
        start_node.h = self._heuristic(start, goal, heuristic)
        start_node.f = start_node.g + start_node.h

        # follows snippet in planner.py
        # heap and position lookup follow a_star().
        open_heap: list[SearchNode] = []
        heapq.heappush(open_heap, start_node)
        open_by_position: dict[Position, SearchNode] = {start: start_node}
        closed_positions: set[Position] = set()

        while open_heap:
            current = heapq.heappop(open_heap)
            if open_by_position.get(current.position) is not current:
                continue

            if current.position == goal:
                path = self._reconstruct_path(current)
                return AStarResult(
                    path=path,
                    nodes_expanded=nodes_expanded,
                    elapsed_seconds=time.time() - started_at,
                    total_cost=current.g,
                    heuristic=heuristic,
                )

            del open_by_position[current.position]
            closed_positions.add(current.position)
            nodes_expanded += 1

            for next_position in self.grid.get_neighbors(current.position):
                if next_position in closed_positions:
                    continue

                action = (
                    next_position[0] - current.x,
                    next_position[1] - current.y,
                )
                tentative_g = current.g + self.grid.transition_cost(
                    current.position,
                    action,
                    next_position,
                    cost_fn,
                )

                existing = open_by_position.get(next_position)
                if existing is not None and tentative_g >= existing.g:
                    continue

                # Fresh heap entries avoid the score-mutation issue in the lab.
                neighbor = SearchNode(next_position[0], next_position[1])
                neighbor.parent = current
                neighbor.g = tentative_g
                neighbor.h = self._heuristic(next_position, goal, heuristic)
                neighbor.f = neighbor.g + neighbor.h
                open_by_position[next_position] = neighbor
                heapq.heappush(open_heap, neighbor)

        return AStarResult(
            path=[],
            nodes_expanded=nodes_expanded,
            elapsed_seconds=time.time() - started_at,
            total_cost=float("inf"),
            heuristic=heuristic,
        )


    def _heuristic(self, position: Position, goal: Position, heuristic: HeuristicName) -> float:
        if heuristic == "manhattan":
            return self.grid.manhattan_distance(position, goal)
        if heuristic == "euclidean":
            return self.grid.euclidean_distance(position, goal)
        if heuristic == "chebyshev":
            return self.grid.chebyshev_distance(position, goal)
        raise ValueError(f"Unknown heuristic: {heuristic}")


    def _reconstruct_path(self, node: SearchNode) -> list[Position]:
        path: list[Position] = []
        current: SearchNode | None = node

        while current is not None:
            path.append(current.position)
            current = current.parent
        path.reverse()
        return path
