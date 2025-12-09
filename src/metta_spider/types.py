"""
Data types for the MettaSpider policy.

SpiderState holds all persistent state between steps.
Phase enum defines the agent's high-level behavioral modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from mettagrid.simulator import Action


class CellType(Enum):
    """Occupancy map cell states."""
    UNKNOWN = 0  # Haven't observed this cell yet
    FREE = 1     # Passable (can walk through)
    OBSTACLE = 2 # Impassable (walls, stations, extractors)


class Phase(Enum):
    """
    High-level behavioral phases for the agent.

    The agent progresses through these phases:
    1. EXPLORE - Build complete map knowledge before doing anything else
    2. GATHER - Collect resources from extractors
    3. DEPOSIT - Deposit resources to shared chest (gatherers only)
    4. WITHDRAW - Withdraw resources from chest (assembler only)
    5. ASSEMBLE - Combine resources into hearts at assembler
    6. DELIVER - Deposit hearts into chest

    RECHARGE can interrupt any phase when energy is low.
    """
    EXPLORE = "explore"    # Initial phase: map the entire arena
    GATHER = "gather"      # Collect resources from extractors
    DEPOSIT = "deposit"    # Deposit resources to chest (gatherers)
    WITHDRAW = "withdraw"  # Withdraw resources from chest (assembler)
    ASSEMBLE = "assemble"  # Make hearts at assembler
    DELIVER = "deliver"    # Deposit hearts to chest
    RECHARGE = "recharge"  # Recharge energy at charger


# ============================================================================
# Role System
# ============================================================================

class AgentRole(Enum):
    """
    Fixed role assignments for multi-agent coordination.

    Each agent specializes in one resource:
    - All agents gather their assigned resource and deposit to chest
    - All agents can assemble hearts when they have all resources
    - The "assembler" role handles silicon + coordination
    """
    CARBON = "carbon"        # Agent 0: gathers carbon
    OXYGEN = "oxygen"        # Agent 1: gathers oxygen
    GERMANIUM = "germanium"  # Agent 2: gathers germanium
    SILICON = "silicon"      # Agent 3: gathers silicon + assembles hearts


# Role assignment by agent_id
ROLE_BY_AGENT_ID: dict[int, AgentRole] = {
    0: AgentRole.CARBON,
    1: AgentRole.OXYGEN,
    2: AgentRole.GERMANIUM,
    3: AgentRole.SILICON,
}


@dataclass
class SharedTeamState:
    """
    Shared state between all agents on the team.

    This is the "team brain" - any agent can read/write to this to share
    discoveries and coordinate actions. Held by the MettaSpiderPolicy and
    passed to each agent's policy.

    Thread safety: In the game sim, agents step sequentially (not in parallel),
    so we don't need locks. Each agent sees the most recent state.
    """

    # ========================================
    # Shared Map Knowledge
    # ========================================
    # Best known extractor for each resource type
    # Dict[resource_type -> position or None]
    extractors: dict[str, tuple[int, int] | None] = field(
        default_factory=lambda: {
            "carbon": None,
            "oxygen": None,
            "germanium": None,
            "silicon": None,
        }
    )

    # Key station positions (first agent to find them shares with team)
    assembler: Optional[tuple[int, int]] = None
    chest: Optional[tuple[int, int]] = None
    charger: Optional[tuple[int, int]] = None

    # ========================================
    # Exploration Coordination
    # ========================================
    # Frontier cells that have been tried and found unreachable
    # This prevents multiple agents from getting stuck on the same bad frontier
    tried_frontiers: set[tuple[int, int]] = field(default_factory=set)

    # Number of agents that have completed exploration
    exploration_done_count: int = 0

    # ========================================
    # Resource Tracking (for assembler)
    # ========================================
    # What the team has deposited to the chest (so assembler knows when to withdraw)
    # Reset after assembler withdraws
    deposited_resources: dict[str, int] = field(
        default_factory=lambda: {
            "carbon": 0,
            "oxygen": 0,
            "germanium": 0,
            "silicon": 0,
        }
    )


@dataclass
class ExtractorInfo:
    """Tracks a discovered extractor."""
    position: tuple[int, int]
    resource_type: str  # "carbon", "oxygen", "germanium", "silicon"
    cooldown_remaining: int = 0
    clipped: bool = False
    remaining_uses: int = 999


@dataclass
class SpiderState:
    """
    Complete state for one spider agent.

    This is the agent's "memory" - everything it knows and tracks
    between steps. The simulator only gives us observations, so we
    maintain our own understanding of the world here.
    """

    # ========================================
    # Identity
    # ========================================
    agent_id: int
    step_count: int = 0

    # ========================================
    # Position (tracked via movement deltas)
    # ========================================
    # We start at the center of a large virtual map and track position
    # relative to that starting point using our movement actions.
    row: int = 0
    col: int = 0

    # ========================================
    # Map Knowledge
    # ========================================
    # Virtual map size - large enough for any arena
    map_height: int = 200
    map_width: int = 200

    # Occupancy grid: CellType values (UNKNOWN, FREE, OBSTACLE)
    # Initialized to all UNKNOWN, updated as we observe
    occupancy: list[list[int]] = field(default_factory=list)

    # Cells we've directly observed (within our 11x11 view at some point)
    seen: set[tuple[int, int]] = field(default_factory=set)

    # ========================================
    # Discovered Objects
    # ========================================
    # Extractors by resource type
    extractors: dict[str, list[ExtractorInfo]] = field(
        default_factory=lambda: {"carbon": [], "oxygen": [], "germanium": [], "silicon": []}
    )

    # Key stations (only need one of each)
    assembler: Optional[tuple[int, int]] = None
    chest: Optional[tuple[int, int]] = None
    charger: Optional[tuple[int, int]] = None

    # Heart recipe (discovered from assembler observation)
    heart_recipe: Optional[dict[str, int]] = None

    # ========================================
    # Inventory (read from observation tokens)
    # ========================================
    carbon: int = 0
    oxygen: int = 0
    germanium: int = 0
    silicon: int = 0
    hearts: int = 0
    energy: int = 100

    # Crafting items (for unclipping, if needed)
    decoder: int = 0
    modulator: int = 0
    resonator: int = 0
    scrambler: int = 0

    # ========================================
    # Phase and Goals
    # ========================================
    phase: Phase = Phase.EXPLORE
    phase_before_recharge: Optional[Phase] = None  # To restore after recharging

    target: Optional[tuple[int, int]] = None
    target_resource: Optional[str] = None  # During GATHER, which resource we're collecting

    # ========================================
    # Exploration State
    # ========================================
    exploration_complete: bool = False

    # ========================================
    # Vibe (affects station interactions)
    # ========================================
    current_vibe: str = "default"

    # ========================================
    # Movement and Stuck Detection
    # ========================================
    last_action: Optional[Action] = None
    position_history: list[tuple[int, int]] = field(default_factory=list)
    stuck_detected: bool = False

    # Flag to prevent position update when using objects
    # (moving into an extractor/station doesn't change position)
    using_object_this_step: bool = False

    # ========================================
    # Path Caching
    # ========================================
    cached_path: Optional[list[tuple[int, int]]] = None
    cached_path_target: Optional[tuple[int, int]] = None

    # ========================================
    # Other Agents (for collision avoidance)
    # ========================================
    agent_positions: set[tuple[int, int]] = field(default_factory=set)

    # ========================================
    # Extractor Usage State
    # ========================================
    waiting_at_extractor: Optional[tuple[int, int]] = None
    wait_steps: int = 0
    pending_resource: Optional[str] = None
    pending_amount: int = 0


def create_initial_state(agent_id: int, map_size: int = 200) -> SpiderState:
    """
    Create a fresh SpiderState for a new agent.

    The agent starts at the center of a virtual map. All cells are UNKNOWN
    until observed. Position (100, 100) is arbitrary - we just need a
    reference point to track relative movement.
    """
    center = map_size // 2

    # Initialize occupancy grid to all UNKNOWN
    occupancy = [[CellType.UNKNOWN.value] * map_size for _ in range(map_size)]

    return SpiderState(
        agent_id=agent_id,
        map_height=map_size,
        map_width=map_size,
        row=center,
        col=center,
        occupancy=occupancy,
    )
