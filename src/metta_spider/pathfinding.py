"""
Pathfinding utilities for MettaSpider.

Uses BFS (Breadth-First Search) for shortest path navigation.
BFS is optimal for unweighted grids - it guarantees the shortest path.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from metta_spider.types import SpiderState

from metta_spider.types import CellType


def is_within_bounds(state: SpiderState, row: int, col: int) -> bool:
    """Check if position is within map bounds."""
    return 0 <= row < state.map_height and 0 <= col < state.map_width


def is_traversable(state: SpiderState, row: int, col: int) -> bool:
    """
    Check if a cell can be walked through.

    A cell is traversable if:
    1. It's within map bounds
    2. It's marked as FREE (not UNKNOWN or OBSTACLE)
    3. No other agent is standing there
    """
    if not is_within_bounds(state, row, col):
        return False

    # Don't walk through other agents
    if (row, col) in state.agent_positions:
        return False

    # Only traverse cells we KNOW are free
    return state.occupancy[row][col] == CellType.FREE.value


def get_neighbors(row: int, col: int) -> list[tuple[int, int]]:
    """
    Get the 4 cardinal neighbors of a cell.

    Returns positions for: north, south, east, west
    """
    return [
        (row - 1, col),  # north
        (row + 1, col),  # south
        (row, col + 1),  # east
        (row, col - 1),  # west
    ]


def compute_adjacent_cells(
    state: SpiderState,
    target: tuple[int, int]
) -> list[tuple[int, int]]:
    """
    Get traversable cells adjacent to target.

    Used when we want to reach a cell adjacent to something
    (like an extractor or station) rather than the cell itself.
    """
    goals = []
    for nr, nc in get_neighbors(target[0], target[1]):
        if is_traversable(state, nr, nc):
            goals.append((nr, nc))
    return goals


def shortest_path(
    state: SpiderState,
    start: tuple[int, int],
    goals: list[tuple[int, int]],
    allow_goal_blocked: bool = False,
) -> list[tuple[int, int]]:
    """
    Find shortest path from start to any goal using BFS.

    Args:
        state: Current agent state (for map knowledge)
        start: Starting position (row, col)
        goals: List of acceptable goal positions
        allow_goal_blocked: If True, allow pathfinding to goals even if they're
                           marked as obstacles (useful for reaching extractors/stations)

    Returns:
        List of positions to visit (excluding start, including goal).
        Empty list if no path exists.

    How BFS works:
    1. Start at 'start', add to queue
    2. Pop from queue, check if it's a goal
    3. If not, add all unvisited traversable neighbors to queue
    4. Record where we came from for each cell
    5. When we hit a goal, trace back to build the path
    """
    if not goals:
        return []

    goal_set = set(goals)

    # Check if we're already at a goal
    if start in goal_set:
        return []

    # BFS queue and visited tracking
    queue: deque[tuple[int, int]] = deque([start])
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

    def can_walk(row: int, col: int) -> bool:
        """Check if we can walk to this cell."""
        # Goals might be "blocked" (extractors/stations are obstacles)
        # but we still want to pathfind TO them
        if (row, col) in goal_set and allow_goal_blocked:
            return is_within_bounds(state, row, col)
        return is_traversable(state, row, col)

    while queue:
        current = queue.popleft()

        # Found a goal!
        if current in goal_set:
            return _reconstruct_path(came_from, current)

        # Explore neighbors
        for nr, nc in get_neighbors(current[0], current[1]):
            if (nr, nc) not in came_from and can_walk(nr, nc):
                came_from[(nr, nc)] = current
                queue.append((nr, nc))

    # No path found
    return []


def _reconstruct_path(
    came_from: dict[tuple[int, int], tuple[int, int] | None],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    """
    Trace back from goal to start to build the path.

    Returns path from start to goal (excluding start, including goal).
    """
    path = []
    current = goal

    while came_from[current] is not None:
        path.append(current)
        current = came_from[current]  # type: ignore

    path.reverse()
    return path


def find_path_to_target(
    state: SpiderState,
    target: tuple[int, int],
    reach_adjacent: bool = False,
) -> list[tuple[int, int]]:
    """
    High-level pathfinding: find path from current position to target.

    Args:
        state: Current agent state
        target: Where we want to go
        reach_adjacent: If True, path to an adjacent cell (for using objects)
                       If False, path to the target itself

    Returns:
        List of positions to walk through, or empty if no path.
    """
    start = (state.row, state.col)

    if reach_adjacent:
        # Find path to any cell adjacent to target
        goals = compute_adjacent_cells(state, target)
        if not goals:
            # No traversable adjacent cells known yet - try anyway
            # (might discover them as we get closer)
            goals = [
                (nr, nc) for nr, nc in get_neighbors(target[0], target[1])
                if is_within_bounds(state, nr, nc)
            ]
        return shortest_path(state, start, goals, allow_goal_blocked=False)
    else:
        # Path directly to target
        return shortest_path(state, start, [target], allow_goal_blocked=True)


def manhattan_distance(pos1: tuple[int, int], pos2: tuple[int, int]) -> int:
    """
    Calculate Manhattan distance between two positions.

    Manhattan distance = |row1 - row2| + |col1 - col2|
    This is the minimum steps to get from pos1 to pos2 in a grid
    (assuming no obstacles).
    """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def is_adjacent(pos1: tuple[int, int], pos2: tuple[int, int]) -> bool:
    """Check if two positions are cardinally adjacent (not diagonal)."""
    return manhattan_distance(pos1, pos2) == 1


def direction_to_target(
    from_pos: tuple[int, int],
    to_pos: tuple[int, int]
) -> str | None:
    """
    Get the cardinal direction from one position to an adjacent position.

    Returns "north", "south", "east", "west", or None if not adjacent.
    """
    dr = to_pos[0] - from_pos[0]
    dc = to_pos[1] - from_pos[1]

    if dr == -1 and dc == 0:
        return "north"
    elif dr == 1 and dc == 0:
        return "south"
    elif dr == 0 and dc == 1:
        return "east"
    elif dr == 0 and dc == -1:
        return "west"
    return None
