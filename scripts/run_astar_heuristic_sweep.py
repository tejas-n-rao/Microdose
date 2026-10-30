# Compare A* heuristics and Dijkstra on seen vessel maps.
import argparse
from collections import deque
import csv
import json
import os
from pathlib import Path
from statistics import mean, median
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.algorithms.a_star import AStarPlanner
from src.algorithms.dijkstra import DijkstraPlanner
from src.robot_identity import robot_label
from src.simulation import microdose_env as env

HEURISTICS = ('manhattan', 'euclidean', 'chebyshev', 'dijkstra')
COLORS = ('#7250a2', '#218b65', '#737c8c', '#bc8140')
LABELS = ('A*\nManhattan', 'A*\nEuclidean', 'A*\nChebyshev', 'Dijkstra\nh = 0')



def shortest_distance(grid, start, goal):
    # BFS is the independent check for these unit-cost paths.
    queue, visited = deque([(start, 0)]), {start}

    while queue:
        cell, distance = queue.popleft()
        if cell == goal:
            return distance

        for neighbor in grid.get_neighbors(cell):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))
    raise ValueError(f'No route from {start} to {goal}')



def check_path(grid, result, start, goal, optimum):
    path = result.path
    if not path or path[0] != start or path[-1] != goal:
        raise ValueError('The path does not reach the requested endpoints')
    if any(b not in grid.get_neighbors(a) for a, b in zip(path, path[1:])):
        raise ValueError('The path crosses a blocked cell or skips a move')
    if len(path) - 1 != optimum or result.total_cost != optimum:
        raise ValueError('Planner did not find the BFS-shortest route')



def write_csv(path, rows):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)



def main():
    parser = argparse.ArgumentParser(description='Compare A* heuristics and Dijkstra on seen vessel maps.')
    parser.add_argument('--repeats', type=int, default=10, help='Timing repeats per route and heuristic.')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/experiments/astar_heuristics')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    rows = []
    layouts = env.SEEN_STUDY_LAYOUTS

    for layout in layouts:
        hazard, flow, _ = env.create_fields(layout)
        grid = env.make_grid(hazard, flow, env.configured_blocked(layout))
        planner = AStarPlanner(grid)
        baseline = DijkstraPlanner(grid)

        for robot, start in env.seeded_robot_starts(layout).items():
            for target, tumour in env.seeded_tumours(layout).items():
                goal = tumour['true']
                optimum = shortest_distance(grid, start, goal)
                timings = {h: [] for h in HEURISTICS}
                results = {}

                for repeat in range(args.repeats):
                    # Rotate the order so one heuristic is not always timed first.
                    offset = repeat % len(HEURISTICS)

                    for heuristic in HEURISTICS[offset:] + HEURISTICS[:offset]:
                        began = perf_counter()
                        # Dijkstra is the same shortest-path task without heuristic guidance.
                        if heuristic == 'dijkstra':
                            result = baseline.search(start, goal, env.unit_transition_cost())
                        else:
                            result = planner.search(start, goal, heuristic, env.unit_transition_cost())
                        timings[heuristic].append(1000 * (perf_counter() - began))
                        check_path(grid, result, start, goal, optimum)
                        results[heuristic] = result

                for heuristic, result in results.items():
                    rows.append(dict(layout=layout, split='seen', robot=robot_label(robot), target=target,
                        heuristic=heuristic, path_moves=len(result.path)-1, optimal_moves=optimum,
                        nodes_expanded=result.nodes_expanded, median_runtime_ms=median(timings[heuristic])))
    summary = []

    for heuristic in HEURISTICS:
        group = [r for r in rows if r['heuristic'] == heuristic]
        summary.append(dict(split='seen', heuristic=heuristic, routes=len(group),
            mean_path_moves=mean(r['path_moves'] for r in group),
            mean_nodes_expanded=mean(r['nodes_expanded'] for r in group),
            median_runtime_ms=median(r['median_runtime_ms'] for r in group)))
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'routes.csv', rows)
    write_csv(output / 'summary.csv', summary)
    config = dict(heuristics=HEURISTICS, baseline='Dijkstra (zero heuristic)', layouts=layouts,
        robots=['R1', 'R2', 'R3'], targets=list('ABC'), repeats=args.repeats, cost='one per cardinal move',
        validation='Every timed path checked against BFS',
        scope='Static planning only; no drift, release, safety scoring, training or controller selection')
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))

    for ax, metric, title in zip(axes,
        ('mean_path_moves', 'mean_nodes_expanded', 'median_runtime_ms'),
        ('Shortest path length (moves)', 'Cells examined during search', 'Planning time (milliseconds)')):
        ax.bar(LABELS, [r[metric] for r in summary], color=COLORS)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis='x', labelsize=9)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle(f'A* heuristics and Dijkstra on seen layouts {layouts[0]}–{layouts[-1]}\n'
                 f'3 robot starts × 3 tumour goals on each of {len(layouts)} layouts', fontsize=13)
    fig.text(.5, .025, f'All paths match breadth-first search. {args.repeats} timing repeats per route.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, .88))
    fig.savefig(output / 'heuristic_sweep.png', dpi=160)
    plt.close(fig)

    for row in summary:
        print(f"{row['split']:6} {row['heuristic']:10} | {row['mean_path_moves']:.1f} moves | "
              f"{row['mean_nodes_expanded']:.1f} nodes | {row['median_runtime_ms']:.3f} ms")
    print(f'Saved {len(rows)} route comparisons: {output}')


if __name__ == '__main__':
    main()
