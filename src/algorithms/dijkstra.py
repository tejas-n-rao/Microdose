from __future__ import annotations

import heapq
import time

from .a_star import AStarResult, CostFunction, SearchNode
from .grid import GridWorld, Position



class DijkstraPlanner:
    # Dijkstra search over the same GridWorld interface used by A*.


    def __init__(self, grid: GridWorld):
        self.grid = grid


    def search(
        self,
        start: Position,
        goal: Position,
        cost_fn: CostFunction | None = None,
    ) -> AStarResult:
        if not self.grid.is_valid(start):
            raise ValueError(f"Start {start} is outside the grid or blocked.")
        if not self.grid.is_valid(goal):
            raise ValueError(f"Goal {goal} is outside the grid or blocked.")

        started_at = time.time()
        nodes_expanded = 0

        # Lab1/planner.py: same frontier setup, extended here with h = 0.
        start_node = SearchNode(start[0], start[1], g=0.0, h=0.0, f=0.0)
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
                    heuristic="dijkstra",
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

                # Use a fresh entry so changing g cannot scramble the heap.
                neighbor = SearchNode(next_position[0], next_position[1])
                neighbor.parent = current
                neighbor.g = tentative_g
                neighbor.h = 0.0
                neighbor.f = tentative_g
                open_by_position[next_position] = neighbor
                heapq.heappush(open_heap, neighbor)

        return AStarResult(
            path=[],
            nodes_expanded=nodes_expanded,
            elapsed_seconds=time.time() - started_at,
            total_cost=float("inf"),
            heuristic="dijkstra",
        )


    def _reconstruct_path(self, node: SearchNode) -> list[Position]:
        # Lab1/planner.py: walk parents back to the start and reverse the list.
        path: list[Position] = []
        current: SearchNode | None = node

        while current is not None:
            path.append(current.position)
            current = current.parent
        path.reverse()
        return path
