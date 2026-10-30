# Shared structured/fleet microscope and world rendering.
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
from src.algorithms.grid import Position
from src.robot_identity import robot_label
from src.simulation.microdose_env import (GRID, vessel_label, seeded_healthy_tissue,
    healthy_tissue_risk_from_blobs, seeded_robot_starts)

ROBOT_COLORS = {'X': '#2563eb', 'Y': '#d97706', 'Z': '#7c3aed'}


def initial_robot_states(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    # Small display states for the three defined launch positions.
    return {carrier: {'carrier_id': carrier, 'actual': position, 'estimated': position}
            for carrier, position in seeded_robot_starts(state['seed']).items()}


def world_model_frame(state: dict[str, Any]) -> np.ndarray:
    # Canonical muted-purple world background for every deployment view.
    frame = np.ones((GRID, GRID, 3), dtype=float)
    for y in range(GRID):
        for x in range(GRID):
            position = (x, y)
            if position in state['blocked']:
                continue
            risk = float(state['hazard'][y][x])
            chaos = float(state['center_chaos'].get(position, 0.0))
            if chaos > 0.04:
                frame[y, x] = np.array([0.43, 0.39, 0.45]) * (0.88 + min(chaos, 0.18))
            else:
                low_risk = np.array([0.50, 0.46, 0.55])
                high_risk = np.array([0.70, 0.32, 0.22])
                risk_scale = min(max(risk - 0.08, 0.0) * 1.9, 1.0)
                frame[y, x] = low_risk * (1 - risk_scale) + high_risk * risk_scale
    for x, y in state['boundary']:
        frame[y, x] = np.array([0.02, 0.02, 0.02])
    return np.clip(frame, 0, 1)


def healthy_marker_cells(seed, blocked):
    # Select the five strongest visible cells per synthetic tissue blob.
    cells = set()
    for blob in seeded_healthy_tissue(seed):
        cx, cy = int(round(blob['cx'])), int(round(blob['cy']))
        candidates = [(healthy_tissue_risk_from_blobs((x, y), [blob]), (x, y))
                      for y in range(max(0, cy-2), min(GRID, cy+3))
                      for x in range(max(0, cx-2), min(GRID, cx+3))
                      if (x, y) not in blocked]
        cells.update(position for _, position in sorted(candidates, reverse=True)[:5])
    return sorted(cells)


def plot_world_preview(state: dict[str, Any]):
    # Neutral vessel preview with all launch positions and no selected mission.
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.imshow(world_model_frame(state), origin='upper')
    for index, (x, y) in enumerate(healthy_marker_cells(state['seed'], state['blocked'])):
        ax.add_patch(plt.Rectangle(
            (x - .5, y - .5), 1, 1, facecolor='#9b6b38', edgecolor='#f7e8c9',
            linewidth=.7, alpha=.82, label='healthy tissue risk' if index == 0 else None,
            zorder=3))
    for name, tumour in state['tumours'].items():
        x, y = tumour['true']
        ax.scatter(x, y, marker='s', s=160, c='#c79bd8', edgecolors='white', zorder=10)
        ax.annotate(name, (x, y), ha='center', va='center', color='white', weight='bold', zorder=11)
    for carrier, position in seeded_robot_starts(state['seed']).items():
        ax.scatter(*position, color=ROBOT_COLORS[carrier], edgecolors='white', s=120, zorder=13)
        ax.annotate(robot_label(carrier), position, xytext=(5, -8), textcoords='offset points',
                    color='white', weight='bold', zorder=14)
    ax.legend(loc='upper left', fontsize=7)
    ax.set_title('World Model · ' + vessel_label(state['seed']))
    ax.set_xlim(-.5, GRID - .5)
    ax.set_ylim(GRID - .5, -.5)
    tick_step = 4 if GRID > 28 else 2 if GRID > 20 else 1
    ax.set_xticks(range(0, GRID, tick_step))
    ax.set_yticks(range(0, GRID, tick_step))
    fig.tight_layout()
    return fig

def plot_microscope_view(state: dict[str, Any], fleet_states=None, *, show_robots=True,
                         highlight_target=True):
    hazard = np.array(state["hazard"], dtype=float)
    flow = np.array(state["flow"], dtype=float)
    blocked = state["blocked"]
    boundary = state["boundary"]
    deadspace = state["deadspace"]
    center_chaos = state["center_chaos"]
    noise_rng = np.random.default_rng(int(state["seed"]) * 1009)
    frame = np.ones((GRID, GRID), dtype=float)
    traversable_mask = np.ones((GRID, GRID), dtype=bool)
    for x, y in blocked:
        traversable_mask[y, x] = False

    vessel_texture = 0.54 - hazard * 0.20 + flow * 0.05
    vessel_texture += noise_rng.normal(0, 0.09, size=frame.shape)
    frame[traversable_mask] = vessel_texture[traversable_mask]

    for (x, y), chaos in center_chaos.items():
        frame[y, x] = min(frame[y, x], 0.30 - chaos * 0.35)

    yy, xx = np.mgrid[0:GRID, 0:GRID]
    for x, y in healthy_marker_cells(state['seed'], blocked):
        frame[y, x] = 0.22

    for tumour_name, tumour in state["tumours"].items():
        tx, ty = tumour["true"]
        radius = float(tumour.get("radius", 1))
        conformal_radius = float(tumour.get("conformal_radius", radius + 1))
        distance = np.sqrt((xx - tx) ** 2 + (yy - ty) ** 2)
        halo = distance <= conformal_radius
        rim = (distance > radius + 0.35) & halo
        core = distance <= radius + 0.35
        selected = highlight_target and tumour_name == state["target_id"]
        frame[halo] = np.maximum(frame[halo], 0.72 if selected else 0.62)
        frame[rim] = np.minimum(frame[rim], 0.16)
        frame[core] = np.maximum(frame[core], 1.0 if selected else 0.90)

    for x, y in deadspace:
        frame[y, x] = 1.0

    for x, y in boundary:
        frame[y, x] = 0.02


    def paint_pixel(x: int, y: int, value: float) -> None:
        if 0 <= x < GRID and 0 <= y < GRID:
            frame[y, x] = value

    robots = (list(fleet_states.values()) if fleet_states else [state]) if show_robots else []
    for robot in robots:
        actual_x, actual_y = robot['actual']
        for dx, dy, value in [(0, 0, 0.), (1, 0, 1.), (-1, 0, 1.), (0, 1, 1.), (0, -1, 1.)]:
            paint_pixel(actual_x+dx, actual_y+dy, value)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.imshow(np.clip(frame, 0, 1), cmap='gray', origin='upper', vmin=0, vmax=1)
    for robot in robots:
        x, y = robot['estimated']
        label = robot_label(robot.get('carrier_id'))
        ax.scatter([x], [y], s=220, facecolors='none', edgecolors='1.0', linewidth=2.8,
                   label=label + ' microscope estimate')
        ax.scatter([x], [y], s=90, marker='x', color='0.0', linewidth=2.)
        ax.annotate(label, (x, y), xytext=(5, -8), textcoords='offset points', color='white',
                    bbox=dict(facecolor='black', alpha=.7, pad=1))

    ax.set_title("Microscope · " + vessel_label(state["seed"]))
    ax.set_xlim(-0.5, GRID - 0.5)
    ax.set_ylim(GRID - 0.5, -0.5)
    tick_step = 4 if GRID > 28 else 2 if GRID > 20 else 1
    ax.set_xticks(range(0, GRID, tick_step))
    ax.set_yticks(range(0, GRID, tick_step))
    ax.grid(color="0.85", alpha=0.18)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", fontsize=8, facecolor="0.15", edgecolor="0.8", labelcolor="white")
    fig.tight_layout()
    return fig
