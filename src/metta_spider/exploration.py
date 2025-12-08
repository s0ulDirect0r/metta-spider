"""
Exploration logic for MettaSpider.

Implements frontier-based exploration to fully map the arena
before starting resource gathering.

The key insight: we don't need to VISIT every cell, just OBSERVE it.
With an 11x11 observation window, walking along the "frontier" (edge
of known territory) efficiently sweeps new areas into view.
"""

from __future__ import annotations

from metta_spider.types import CellType, SpiderState
from metta_spider.pathfinding import (
    is_within_bounds,
    is_traversable,
    get_neighbors,
    shortest_path,
    manhattan_distance,
)


def find_frontier_cells(state: SpiderState) -> list[tuple[int, int]]:
    """
    Find all "frontier" cells - FREE cells adjacent to UNKNOWN cells.

    These are the cells we should move toward to expand our map knowledge.
    Standing on a frontier cell lets us observe the UNKNOWN cells next to it.

    Returns:
        List of (row, col) positions that are on the frontier.
    """
    frontier = []

    # Check every cell we know is FREE
    for row in range(state.map_height):
        for col in range(state.map_width):
            if state.occupancy[row][col] != CellType.FREE.value:
                continue

            # Is this cell adjacent to any UNKNOWN cell?
            for nr, nc in get_neighbors(row, col):
                if is_within_bounds(state, nr, nc):
                    if state.occupancy[nr][nc] == CellType.UNKNOWN.value:
                        frontier.append((row, col))
                        break  # Only add once even if multiple unknown neighbors

    return frontier


def find_nearest_frontier(state: SpiderState) -> tuple[int, int] | None:
    """
    Find the nearest frontier cell to the agent.

    Returns:
        Position of nearest frontier cell, or None if no frontier exists
        (meaning exploration is complete).
    """
    frontier = find_frontier_cells(state)

    if not frontier:
        return None

    current = (state.row, state.col)

    # Find nearest by Manhattan distance
    # (In practice, we'll pathfind to it, but this is a quick heuristic
    # for choosing which frontier cell to target)
    return min(frontier, key=lambda pos: manhattan_distance(current, pos))


def is_exploration_complete(state: SpiderState) -> bool:
    """
    Check if exploration is complete.

    Exploration is complete when there are no more frontier cells -
    meaning every UNKNOWN cell is unreachable from any FREE cell.

    This happens when:
    1. We've observed the entire reachable map
    2. All remaining UNKNOWN cells are behind walls
    """
    frontier = find_frontier_cells(state)
    return len(frontier) == 0


def get_exploration_target(state: SpiderState) -> tuple[int, int] | None:
    """
    Get the next exploration target.

    This uses a simple strategy: go to the nearest frontier cell.
    More sophisticated strategies could consider:
    - Clustering frontiers and going to the biggest cluster
    - Preferring frontiers in unexplored directions
    - Spiral patterns

    But simple is good for now.
    """
    return find_nearest_frontier(state)


def path_to_exploration_target(state: SpiderState) -> list[tuple[int, int]]:
    """
    Find path to the best exploration target.

    Returns:
        Path to frontier cell, or empty list if exploration is complete.
    """
    frontier = find_frontier_cells(state)

    if not frontier:
        return []

    current = (state.row, state.col)

    # If we're AT a frontier cell, we need to move toward UNKNOWN territory
    # Pick a direction that leads to unexplored cells
    if current in frontier:
        # Find an adjacent UNKNOWN cell and move toward it
        for nr, nc in get_neighbors(current[0], current[1]):
            if is_within_bounds(state, nr, nc):
                if state.occupancy[nr][nc] == CellType.UNKNOWN.value:
                    # Can't walk into UNKNOWN, but find adjacent FREE cell toward it
                    # Actually, just pick another frontier cell that isn't us
                    other_frontiers = [f for f in frontier if f != current]
                    if other_frontiers:
                        target = min(other_frontiers, key=lambda p: manhattan_distance(current, p))
                        return shortest_path(state, current, [target], allow_goal_blocked=False)
                    # No other frontiers - we need to step toward unknown
                    # Return a "fake" path to move in a direction
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        adj_r, adj_c = current[0] + dr, current[1] + dc
                        if is_traversable(state, adj_r, adj_c):
                            return [(adj_r, adj_c)]
        return []

    # Find path to nearest frontier
    target = min(frontier, key=lambda p: manhattan_distance(current, p))
    return shortest_path(state, current, [target], allow_goal_blocked=False)


def count_unknown_cells(state: SpiderState) -> int:
    """
    Count how many cells are still UNKNOWN.

    Useful for tracking exploration progress.
    """
    count = 0
    for row in range(state.map_height):
        for col in range(state.map_width):
            if state.occupancy[row][col] == CellType.UNKNOWN.value:
                count += 1
    return count


def count_explored_cells(state: SpiderState) -> int:
    """
    Count how many cells have been observed (FREE or OBSTACLE).

    Useful for tracking exploration progress.
    """
    count = 0
    for row in range(state.map_height):
        for col in range(state.map_width):
            if state.occupancy[row][col] != CellType.UNKNOWN.value:
                count += 1
    return count


def get_exploration_progress(state: SpiderState) -> tuple[int, int]:
    """
    Get exploration progress as (explored, frontier_remaining).

    Returns:
        (number of explored cells, number of frontier cells remaining)
    """
    explored = count_explored_cells(state)
    frontier = len(find_frontier_cells(state))
    return (explored, frontier)
