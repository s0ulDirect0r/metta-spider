"""
MettaSpider - Explore-First Policy for Alignment League.

Strategy: Fully explore the map before gathering resources.
Hypothesis: Complete map knowledge leads to better decisions than
the baseline's interleaved explore-while-working approach.

Phases:
1. EXPLORE - Map the entire arena first
2. GATHER - Collect resources from known extractors
3. ASSEMBLE - Make hearts at assembler
4. DELIVER - Deposit hearts into chest
(RECHARGE interrupts any phase when energy is low)
"""

from __future__ import annotations

import atexit
import json
import logging
import random
from typing import Any, Optional

# Set up file-only logging - don't propagate to root logger (which prints to terminal)
LOG_FILE = "/tmp/metta_spider.log"
TRACE_FILE = "/tmp/metta_spider.jsonl"

logger = logging.getLogger("metta_spider")
logger.setLevel(logging.DEBUG)
logger.propagate = False  # Don't send to root logger / terminal
logger.handlers = []
_fh = logging.FileHandler(LOG_FILE, mode='w')
_fh.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(_fh)

# Structured trace file (JSON lines)
_trace_file = open(TRACE_FILE, 'w')
atexit.register(_trace_file.close)

def trace(event: str, agent_id: int, step: int, **data: Any) -> None:
    """Write a structured trace event as JSON line."""
    record = {"event": event, "agent": agent_id, "step": step, **data}
    _trace_file.write(json.dumps(record) + "\n")
    _trace_file.flush()
    # Also write human-readable version to text log
    detail = ", ".join(f"{k}={v}" for k, v in data.items()) if data else ""
    logger.debug(f"[{event}] agent={agent_id} step={step} {detail}")

from mettagrid.config.vibes import VIBE_BY_NAME
from mettagrid.policy.policy import MultiAgentPolicy, StatefulAgentPolicy, StatefulPolicyImpl
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action
from mettagrid.simulator.interface import AgentObservation

from metta_spider.types import (
    AgentRole,
    CellType,
    ExtractorInfo,
    Phase,
    ROLE_BY_AGENT_ID,
    SharedTeamState,
    SpiderState,
    create_initial_state,
)
from metta_spider.pathfinding import (
    is_adjacent,
    is_traversable,
    is_within_bounds,
    find_path_to_target,
    direction_to_target,
    manhattan_distance,
    get_neighbors,
)
from metta_spider.exploration import (
    is_exploration_complete,
    path_to_exploration_target,
)


# ============================================================================
# Constants
# ============================================================================

# Energy thresholds for recharge decisions
ENERGY_LOW = 30   # Enter RECHARGE when below this
ENERGY_HIGH = 80  # Exit RECHARGE when above this

# Stuck detection
POSITION_HISTORY_SIZE = 30

# Resource to vibe mapping
# resource_a = extract/withdraw, resource_b = deposit
RESOURCE_TO_VIBE = {
    "carbon": "carbon_a",
    "oxygen": "oxygen_a",
    "germanium": "germanium_a",
    "silicon": "silicon_a",
}

RESOURCE_TO_DEPOSIT_VIBE = {
    "carbon": "carbon_b",
    "oxygen": "oxygen_b",
    "germanium": "germanium_b",
    "silicon": "silicon_b",
}


# ============================================================================
# Main Policy Implementation
# ============================================================================

class SpiderPolicyImpl(StatefulPolicyImpl[SpiderState]):
    """
    Core policy logic for a single spider agent.

    This class handles:
    - Observation parsing
    - State updates
    - Phase transitions
    - Action selection

    Each agent has a role that determines its behavior:
    - CARBON/OXYGEN/GERMANIUM: Gather that resource and deposit to chest
    - SILICON: Gather silicon, deposit, then withdraw all and assemble hearts
    """

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        agent_id: int,
        team_state: SharedTeamState,
        role: AgentRole,
    ):
        self._agent_id = agent_id
        self._policy_env_info = policy_env_info
        self._team_state = team_state
        self._role = role

        # Observation grid dimensions (typically 11x11, so half-radius is 5)
        self._obs_hr = policy_env_info.obs_height // 2
        self._obs_wr = policy_env_info.obs_width // 2

        # Action helpers
        self._actions = policy_env_info.actions
        self._move_deltas = {
            "north": (-1, 0),
            "south": (1, 0),
            "east": (0, 1),
            "west": (0, -1),
        }

        # Tag ID to name lookup (for parsing observations)
        self._tag_names = policy_env_info.tag_id_to_name

        trace("role_assigned", agent_id, 0, role=role.value)

    # ========================================================================
    # State Initialization
    # ========================================================================

    def initial_agent_state(self) -> SpiderState:
        """Create initial state for this agent."""
        state = create_initial_state(self._agent_id)

        # Try to get heart recipe from environment config
        for protocol in self._policy_env_info.assembler_protocols:
            if protocol.output_resources.get("heart", 0) > 0:
                state.heart_recipe = dict(protocol.input_resources)
                state.heart_recipe.pop("energy", None)  # Don't track energy as gatherable
                break

        trace("init", self._agent_id, 0, recipe=state.heart_recipe)

        return state

    # ========================================================================
    # Main Step Function
    # ========================================================================

    def step_with_state(
        self,
        obs: AgentObservation,
        state: SpiderState
    ) -> tuple[Action, SpiderState]:
        """
        Main decision function - called every tick.

        This is the OODA loop:
        1. Observe - parse tokens into structured data
        2. Orient - update map, position, inventory
        3. Decide - choose phase and action
        4. Act - return the action
        """
        state.step_count += 1

        # ========================================
        # OBSERVE: Parse observation
        # ========================================
        # Save energy BEFORE reading new inventory (for move success detection)
        prev_energy = state.energy
        self._read_inventory(state, obs)
        self._update_position(state, prev_energy)
        self._update_map_from_observation(state, obs)

        # ========================================
        # ORIENT: Update phase
        # ========================================
        self._update_phase(state)

        # ========================================
        # Handle vibe changes
        # ========================================
        desired_vibe = self._get_desired_vibe(state)
        if state.current_vibe != desired_vibe:
            state.current_vibe = desired_vibe
            action = self._change_vibe(desired_vibe)
            state.last_action = action
            return action, state

        # ========================================
        # Check for stuck state
        # ========================================
        if state.stuck_detected:
            action = self._escape_stuck(state)
            if action:
                state.last_action = action
                return action, state

        # ========================================
        # DECIDE & ACT: Execute phase
        # ========================================
        action = self._execute_phase(state)
        state.last_action = action


        return action, state

    # ========================================================================
    # Observation Parsing
    # ========================================================================

    def _read_inventory(self, state: SpiderState, obs: AgentObservation) -> None:
        """Read inventory from observation tokens at center cell.

        Important: Tokens with value 0 are not sent, so we collect into a dict
        and use .get(key, 0) to default missing values to 0.
        """
        center = (self._obs_hr, self._obs_wr)

        # Collect inventory tokens into dict
        inv = {}
        for tok in obs.tokens:
            if tok.location != center:
                continue
            name = tok.feature.name
            if name.startswith("inv:"):
                resource = name[4:]  # Remove "inv:" prefix
                inv[resource] = tok.value

        # Update state with defaults for missing tokens
        state.carbon = inv.get("carbon", 0)
        state.oxygen = inv.get("oxygen", 0)
        state.germanium = inv.get("germanium", 0)
        state.silicon = inv.get("silicon", 0)
        state.hearts = inv.get("heart", 0)
        state.energy = inv.get("energy", 0)
        state.decoder = inv.get("decoder", 0)
        state.modulator = inv.get("modulator", 0)
        state.resonator = inv.get("resonator", 0)
        state.scrambler = inv.get("scrambler", 0)

    def _update_position(self, state: SpiderState, prev_energy: int) -> None:
        """Update position based on last action and energy change.

        We detect move success/failure by checking energy change:
        - Successful move: -2 energy cost, +1 regen = net -1
        - Failed move: no cost, +1 regen = net +1

        Edge cases:
        - At energy cap (255): no regen, so success=-2, fail=0
        - First step: no prev_energy, assume success
        """
        # Only update if we moved (not if we used an object)
        if state.last_action and not state.using_object_this_step:
            action_name = state.last_action.name
            if action_name.startswith("move_"):
                direction = action_name[5:]  # Remove "move_" prefix
                if direction in self._move_deltas:
                    dr, dc = self._move_deltas[direction]

                    # Detect if move succeeded via energy change
                    energy_delta = state.energy - prev_energy

                    # Move succeeded if energy decreased (or stayed same at cap with no regen)
                    # Move failed if energy increased (got regen but no cost)
                    move_succeeded = energy_delta <= 0

                    # Edge case: at energy cap, failed move shows delta=0
                    # But successful move at cap shows delta=-2
                    # So delta <= 0 still works, but delta == 0 at cap is ambiguous
                    # If at cap and delta == 0, check if target is known obstacle
                    if energy_delta == 0 and prev_energy >= 254:
                        new_row = state.row + dr
                        new_col = state.col + dc
                        if is_within_bounds(state, new_row, new_col):
                            if state.occupancy[new_row][new_col] == CellType.OBSTACLE.value:
                                move_succeeded = False

                    if move_succeeded:
                        state.row += dr
                        state.col += dc

        state.using_object_this_step = False

        # Update position history for stuck detection
        current_pos = (state.row, state.col)
        state.position_history.append(current_pos)
        if len(state.position_history) > POSITION_HISTORY_SIZE:
            state.position_history.pop(0)

        # Detect stuck (oscillating between positions)
        self._detect_stuck(state)

    def _detect_stuck(self, state: SpiderState) -> None:
        """Detect if agent is stuck by checking recent position diversity.

        Instead of detecting specific loop patterns (2-pos, 3-pos, etc.),
        we count unique positions in recent history. If the agent visited
        12 positions but only 4 or fewer were unique, it's oscillating.
        """
        history = state.position_history
        state.stuck_detected = False

        if len(history) < 12:
            return

        recent = history[-12:]  # Last 12 positions
        unique_count = len(set(recent))

        # If we've only visited 4 or fewer unique positions in 12 steps, we're stuck
        if unique_count <= 4:
            state.stuck_detected = True
            trace("stuck", self._agent_id, state.step_count, unique_positions=unique_count)

    def _update_map_from_observation(
        self,
        state: SpiderState,
        obs: AgentObservation
    ) -> None:
        """Update occupancy map and discover objects from observation."""
        # Clear agent positions (will be rebuilt)
        state.agent_positions.clear()

        # Collect features by position
        position_data: dict[tuple[int, int], dict] = {}

        for tok in obs.tokens:
            obs_r, obs_c = tok.location

            # Skip center (that's inventory)
            if obs_r == self._obs_hr and obs_c == self._obs_wr:
                continue

            # Convert to world coordinates
            world_r = obs_r - self._obs_hr + state.row
            world_c = obs_c - self._obs_wr + state.col

            if not is_within_bounds(state, world_r, world_c):
                continue

            pos = (world_r, world_c)

            # Mark as seen
            state.seen.add(pos)

            # Collect features for this position
            if pos not in position_data:
                position_data[pos] = {"tags": [], "cooldown": 0, "clipped": 0, "remaining": 999}

            name = tok.feature.name
            value = tok.value

            if name == "tag":
                position_data[pos]["tags"].append(value)
            elif name == "cooldown_remaining":
                position_data[pos]["cooldown"] = value
            elif name == "clipped":
                position_data[pos]["clipped"] = value
            elif name == "remaining_uses":
                position_data[pos]["remaining"] = value
            elif name == "agent:group":
                # Another agent at this position
                state.agent_positions.add(pos)
            elif name.startswith("protocol_output:"):
                resource = name[len("protocol_output:"):]
                if resource == "heart" and value > 0:
                    # This is an assembler that makes hearts - get the recipe
                    position_data[pos]["makes_hearts"] = True

        # First pass: mark all observed cells as FREE
        for obs_r in range(2 * self._obs_hr + 1):
            for obs_c in range(2 * self._obs_wr + 1):
                world_r = obs_r - self._obs_hr + state.row
                world_c = obs_c - self._obs_wr + state.col
                if is_within_bounds(state, world_r, world_c):
                    if state.occupancy[world_r][world_c] == CellType.UNKNOWN.value:
                        state.occupancy[world_r][world_c] = CellType.FREE.value

        # Second pass: process objects at each position
        for pos, data in position_data.items():
            if not data["tags"]:
                continue

            # Get primary tag name
            tag_id = data["tags"][0]
            obj_name = self._tag_names.get(tag_id, f"unknown_{tag_id}").lower()

            r, c = pos

            # Handle different object types
            if "wall" in obj_name or "#" in obj_name:
                state.occupancy[r][c] = CellType.OBSTACLE.value

            elif "extractor" in obj_name:
                state.occupancy[r][c] = CellType.OBSTACLE.value
                resource_type = obj_name.replace("_extractor", "").replace("clipped_", "")
                self._discover_extractor(state, pos, resource_type, data)

            elif "assembler" in obj_name:
                state.occupancy[r][c] = CellType.OBSTACLE.value
                if state.assembler is None:
                    state.assembler = pos
                    trace("discovered", self._agent_id, state.step_count, type="assembler", pos=pos)
                # Share with team
                if self._team_state.assembler is None:
                    self._team_state.assembler = pos
                    trace("shared", self._agent_id, state.step_count, type="assembler", pos=pos)

            elif "chest" in obj_name:
                state.occupancy[r][c] = CellType.OBSTACLE.value
                if state.chest is None:
                    state.chest = pos
                    trace("discovered", self._agent_id, state.step_count, type="chest", pos=pos)
                # Share with team
                if self._team_state.chest is None:
                    self._team_state.chest = pos
                    trace("shared", self._agent_id, state.step_count, type="chest", pos=pos)

            elif "charger" in obj_name or "solar" in obj_name:
                state.occupancy[r][c] = CellType.OBSTACLE.value
                if state.charger is None:
                    state.charger = pos
                    trace("discovered", self._agent_id, state.step_count, type="charger", pos=pos)
                # Share with team
                if self._team_state.charger is None:
                    self._team_state.charger = pos
                    trace("shared", self._agent_id, state.step_count, type="charger", pos=pos)

    def _discover_extractor(
        self,
        state: SpiderState,
        pos: tuple[int, int],
        resource_type: str,
        data: dict
    ) -> None:
        """Add or update an extractor in state, and share with team."""
        if resource_type not in state.extractors:
            return

        # Check if we already know about this extractor
        for ext in state.extractors[resource_type]:
            if ext.position == pos:
                # Update existing
                ext.cooldown_remaining = data.get("cooldown", 0)
                ext.clipped = data.get("clipped", 0) > 0
                ext.remaining_uses = data.get("remaining", 999)
                return

        # Add new extractor to local state
        state.extractors[resource_type].append(ExtractorInfo(
            position=pos,
            resource_type=resource_type,
            cooldown_remaining=data.get("cooldown", 0),
            clipped=data.get("clipped", 0) > 0,
            remaining_uses=data.get("remaining", 999),
        ))
        trace("discovered", self._agent_id, state.step_count, type=f"{resource_type}_extractor", pos=pos)

        # Share with team (first extractor of this type wins)
        if self._team_state.extractors[resource_type] is None:
            self._team_state.extractors[resource_type] = pos
            trace("shared", self._agent_id, state.step_count,
                  type=f"{resource_type}_extractor", pos=pos)

    # ========================================================================
    # Phase Management
    # ========================================================================

    def _update_phase(self, state: SpiderState) -> None:
        """Update phase based on current state and role.

        Role-based behavior:
        - Gatherers: EXPLORE → GATHER → DEPOSIT → GATHER (loop)
        - Assembler: EXPLORE → WITHDRAW → ASSEMBLE → DELIVER → WITHDRAW (loop)
        """
        # Priority 1: Recharge if energy low
        if state.energy < ENERGY_LOW:
            if state.phase != Phase.RECHARGE:
                trace("phase", self._agent_id, state.step_count,
                      old=state.phase.value, new="recharge", reason="low_energy", energy=state.energy)
                state.phase_before_recharge = state.phase
                state.phase = Phase.RECHARGE
                state.target = None
                state.cached_path = None
            return

        # If recharging, wait until energy is high enough
        if state.phase == Phase.RECHARGE:
            if state.energy >= ENERGY_HIGH:
                # Restore previous phase
                prev = state.phase_before_recharge or Phase.EXPLORE
                trace("phase", self._agent_id, state.step_count,
                      old="recharge", new=prev.value, reason="recharged", energy=state.energy)
                state.phase = prev
                state.phase_before_recharge = None
                state.target = None
                state.cached_path = None
            return

        # Priority 2: If exploring, check if complete
        if state.phase == Phase.EXPLORE:
            if is_exploration_complete(state):
                # Guard: only count exploration once per agent
                if not state.exploration_complete:
                    ext_counts = {r: len(e) for r, e in state.extractors.items() if e}
                    # Track exploration completion for team coordination
                    self._team_state.exploration_done_count += 1
                    state.exploration_complete = True
                    trace("phase", self._agent_id, state.step_count,
                          old="explore", new="gather", reason="exploration_complete",
                          role=self._role.value, extractors=ext_counts)
                else:
                    # Agent already explored, just transition without re-counting
                    trace("phase", self._agent_id, state.step_count,
                          old="explore", new="gather", reason="re_explored")

                # All roles go to GATHER after exploration
                # SILICON role will later do WITHDRAW/ASSEMBLE via _update_phase_silicon_assembler
                state.phase = Phase.GATHER
                state.target = None
                state.cached_path = None
            return

        # Role-specific phase transitions
        # SILICON role also handles assembly (dual role: gatherer + assembler)
        if self._role == AgentRole.SILICON:
            self._update_phase_silicon_assembler(state)
        else:
            self._update_phase_gatherer(state)

    def _update_phase_gatherer(self, state: SpiderState) -> None:
        """Phase transitions for gatherer agents (CARBON, OXYGEN, GERMANIUM roles).

        Loop: GATHER → DEPOSIT → GATHER
        """
        my_resource = self._role.value  # "carbon", "oxygen", or "germanium"

        # If we have our resource, go deposit it
        current_amount = getattr(state, my_resource, 0)
        if current_amount > 0 and state.phase == Phase.GATHER:
            trace("phase", self._agent_id, state.step_count,
                  old="gather", new="deposit", reason="have_resource",
                  resource=my_resource, amount=current_amount)
            state.phase = Phase.DEPOSIT
            state.target = None
            state.cached_path = None
            return

        # If in DEPOSIT and we've deposited (no more resource), go back to GATHER
        if state.phase == Phase.DEPOSIT and current_amount == 0:
            trace("phase", self._agent_id, state.step_count,
                  old="deposit", new="gather", reason="deposited")
            state.phase = Phase.GATHER
            state.target = None
            state.cached_path = None
            return

        # Default: stay in current phase or go to GATHER
        if state.phase not in (Phase.GATHER, Phase.DEPOSIT, Phase.EXPLORE, Phase.RECHARGE):
            state.phase = Phase.GATHER
            state.target = None
            state.cached_path = None

    def _update_phase_silicon_assembler(self, state: SpiderState) -> None:
        """Phase transitions for SILICON agent (dual role: gatherer + assembler).

        Loop: GATHER → DEPOSIT → WITHDRAW → ASSEMBLE → DELIVER → (repeat)

        This agent gathers silicon like other gatherers, but also withdraws
        the other resources and assembles hearts.
        """
        # If we have hearts, deliver them
        if state.hearts > 0:
            if state.phase != Phase.DELIVER:
                trace("phase", self._agent_id, state.step_count,
                      old=state.phase.value, new="deliver", reason="have_hearts",
                      hearts=state.hearts)
                state.phase = Phase.DELIVER
                state.target = None
                state.cached_path = None
            return

        # If we can assemble, do it
        if self._can_assemble(state):
            if state.phase != Phase.ASSEMBLE:
                trace("phase", self._agent_id, state.step_count,
                      old=state.phase.value, new="assemble", reason="have_resources")
                state.phase = Phase.ASSEMBLE
                state.target = None
                state.cached_path = None
            return

        # If we just delivered, go back to gathering silicon
        if state.phase == Phase.DELIVER:
            trace("phase", self._agent_id, state.step_count,
                  old="deliver", new="gather", reason="delivered")
            state.phase = Phase.GATHER
            state.target = None
            state.cached_path = None
            return

        # If we have silicon, deposit it then go to withdraw
        if state.silicon > 0 and state.phase == Phase.GATHER:
            trace("phase", self._agent_id, state.step_count,
                  old="gather", new="deposit", reason="have_silicon",
                  silicon=state.silicon)
            state.phase = Phase.DEPOSIT
            state.target = None
            state.cached_path = None
            return

        # If we just deposited, go to withdraw to get other resources
        if state.phase == Phase.DEPOSIT and state.silicon == 0:
            trace("phase", self._agent_id, state.step_count,
                  old="deposit", new="withdraw", reason="deposited_silicon")
            state.phase = Phase.WITHDRAW
            state.target = None
            state.cached_path = None
            return

        # If we're withdrawing but don't have enough resources, stay withdrawing
        # Check if team has deposited enough for us to assemble
        if state.phase == Phase.WITHDRAW:
            # Stay in withdraw - the _do_withdraw will handle the logic
            # But if we've been withdrawing too long without success, go gather more
            return

        # Default: go to GATHER (to get more silicon)
        if state.phase not in (Phase.GATHER, Phase.DEPOSIT, Phase.WITHDRAW, Phase.ASSEMBLE, Phase.DELIVER, Phase.EXPLORE, Phase.RECHARGE):
            state.phase = Phase.GATHER
            state.target = None
            state.cached_path = None

    def _can_assemble(self, state: SpiderState) -> bool:
        """Check if we have enough resources to make a heart."""
        if state.heart_recipe is None:
            return False

        return (
            state.carbon >= state.heart_recipe.get("carbon", 0) and
            state.oxygen >= state.heart_recipe.get("oxygen", 0) and
            state.germanium >= state.heart_recipe.get("germanium", 0) and
            state.silicon >= state.heart_recipe.get("silicon", 0)
        )

    def _get_desired_vibe(self, state: SpiderState) -> str:
        """Get the vibe we should be in for current phase and role.

        Vibe reference:
        - resource_a: Extract from extractor (GATHER phase), also withdraw from chest
        - resource_b: Deposit resource to chest (DEPOSIT phase)
        - heart_a: Assemble hearts (ASSEMBLE phase)
        - heart_b: Deposit hearts to chest (DELIVER phase)

        For WITHDRAW, we need resource_a vibes to withdraw from chest.
        """
        if state.phase == Phase.ASSEMBLE:
            return "heart_a"
        elif state.phase == Phase.DELIVER:
            return "heart_b"  # Deposit hearts to chest
        elif state.phase == Phase.DEPOSIT:
            # All gatherers (including SILICON) deposit their resource
            resource = self._role.value  # "carbon", "oxygen", "germanium", "silicon"
            return f"{resource}_b"  # Deposit vibe
        elif state.phase == Phase.WITHDRAW:
            # SILICON agent withdraws resources - need the "extract" vibe for each resource
            # We'll cycle through resources we need
            return self._get_withdraw_vibe(state)
        elif state.phase == Phase.GATHER:
            # All gatherers gather their assigned resource
            if self._role in (AgentRole.CARBON, AgentRole.OXYGEN, AgentRole.GERMANIUM, AgentRole.SILICON):
                return f"{self._role.value}_a"  # Extract vibe
            elif state.target_resource:
                return RESOURCE_TO_VIBE.get(state.target_resource, "default")
        return "default"

    def _get_withdraw_vibe(self, state: SpiderState) -> str:
        """Get the correct vibe for withdrawing resources from chest.

        The assembler needs to withdraw each resource type, so we check
        which resource we're missing most and set that vibe.
        """
        if state.heart_recipe is None:
            return "carbon_a"  # Default

        # Find which resource we need most
        deficits = {
            "carbon": max(0, state.heart_recipe.get("carbon", 0) - state.carbon),
            "oxygen": max(0, state.heart_recipe.get("oxygen", 0) - state.oxygen),
            "germanium": max(0, state.heart_recipe.get("germanium", 0) - state.germanium),
            "silicon": max(0, state.heart_recipe.get("silicon", 0) - state.silicon),
        }

        # Get resource with highest deficit
        max_resource = max(deficits.items(), key=lambda x: x[1])
        if max_resource[1] > 0:
            return f"{max_resource[0]}_a"  # Extract/withdraw vibe

        return "default"

    # ========================================================================
    # Phase Execution
    # ========================================================================

    def _execute_phase(self, state: SpiderState) -> Action:
        """Execute action for current phase."""
        if state.phase == Phase.EXPLORE:
            return self._do_explore(state)
        elif state.phase == Phase.GATHER:
            return self._do_gather(state)
        elif state.phase == Phase.DEPOSIT:
            return self._do_deposit(state)
        elif state.phase == Phase.WITHDRAW:
            return self._do_withdraw(state)
        elif state.phase == Phase.ASSEMBLE:
            return self._do_assemble(state)
        elif state.phase == Phase.DELIVER:
            return self._do_deliver(state)
        elif state.phase == Phase.RECHARGE:
            return self._do_recharge(state)
        return self._noop()

    def _do_explore(self, state: SpiderState) -> Action:
        """Execute exploration phase."""
        # Get path to nearest frontier
        path = path_to_exploration_target(state)

        if not path:
            # No frontier found - exploration complete!
            state.exploration_complete = True
            state.phase = Phase.GATHER
            return self._noop()

        # Take next step
        return self._follow_path(state, path)

    def _do_gather(self, state: SpiderState) -> Action:
        """Execute gathering phase.

        Role-aware: gatherers only gather their assigned resource.
        """
        # If we're already waiting at an extractor, handle that first
        # Don't recalculate target - finish what we started
        if state.waiting_at_extractor is not None:
            return self._handle_extractor_wait(state)

        # Role-aware: determine what resource to gather
        if self._role in (AgentRole.CARBON, AgentRole.OXYGEN, AgentRole.GERMANIUM, AgentRole.SILICON):
            # All gatherers (including SILICON) gather their assigned resource
            my_resource = self._role.value
            extractor = self._find_extractor_for_resource(state, my_resource)
            resource = my_resource
        else:
            # Fallback: use deficit-based logic
            deficits = self._calculate_deficits(state)
            extractor, resource = self._find_needed_extractor(state, deficits)

        if extractor is None:
            # Don't know where to find what we need - explore more!
            # Reset exploration state and go back to exploring
            state.exploration_complete = False
            state.phase = Phase.EXPLORE
            trace("gather_fail", self._agent_id, state.step_count,
                  reason="no_extractor", resource=resource, role=self._role.value)
            return self._do_explore(state)

        state.target_resource = resource
        trace("gather_target", self._agent_id, state.step_count,
              resource=resource, target=extractor.position, role=self._role.value)

        # Navigate to extractor
        current = (state.row, state.col)

        if is_adjacent(current, extractor.position):
            # We're adjacent - use it!
            if extractor.cooldown_remaining > 0 or extractor.clipped:
                # Not ready, wait
                return self._noop()

            # Record pre-use amount and initiate use
            state.pending_resource = resource
            state.pending_amount = getattr(state, resource, 0)
            state.waiting_at_extractor = extractor.position
            state.wait_steps = 0

            return self._use_object(state, extractor.position)

        # Move toward extractor
        path = find_path_to_target(state, extractor.position, reach_adjacent=True)
        if path:
            return self._follow_path(state, path)

        return self._noop()

    def _find_extractor_for_resource(
        self,
        state: SpiderState,
        resource: str
    ) -> ExtractorInfo | None:
        """Find an extractor for a specific resource type."""
        extractors = state.extractors.get(resource, [])

        # Filter to available extractors
        available = [e for e in extractors if not e.clipped and e.remaining_uses > 0]

        if available:
            current = (state.row, state.col)
            return min(available, key=lambda e: manhattan_distance(current, e.position))

        # Check shared team state for fallback
        shared_pos = self._team_state.extractors.get(resource)
        if shared_pos:
            # Create a temporary ExtractorInfo from shared position
            # Note: Uses default values (cooldown=0, clipped=False) - agent will
            # discover real state when adjacent. May waste steps if unusable.
            trace("extractor_fallback", self._agent_id, state.step_count,
                  resource=resource, pos=shared_pos, reason="using_team_shared")
            return ExtractorInfo(position=shared_pos, resource_type=resource)

        return None

    def _handle_extractor_wait(self, state: SpiderState) -> Action:
        """Handle waiting for resource from extractor."""
        resource = state.pending_resource
        if resource is None:
            # Shouldn't happen, but clear state and continue
            state.waiting_at_extractor = None
            return self._noop()

        # Check if we received the resource
        current_amount = getattr(state, resource, 0)
        if current_amount > state.pending_amount:
            # Success! Clear waiting state
            gained = current_amount - state.pending_amount
            trace("gathered", self._agent_id, state.step_count,
                  resource=resource, gained=gained, total=current_amount)
            state.waiting_at_extractor = None
            state.pending_resource = None
            state.pending_amount = 0
            state.wait_steps = 0
            return self._noop()  # Next step will find new target

        # Still waiting - check timeout
        state.wait_steps += 1
        if state.wait_steps > 20:  # Generous timeout
            # Timeout - clear and try again
            state.waiting_at_extractor = None
            state.pending_resource = None
            state.pending_amount = 0
            state.wait_steps = 0

        return self._noop()

    def _handle_deposit_verification(self, state: SpiderState) -> Action:
        """Verify that a pending deposit succeeded by checking inventory decreased."""
        resource = state.pending_deposit_resource
        if resource is None:
            return self._noop()

        current_amount = getattr(state, resource, 0)

        # Deposit succeeded if we now have less of the resource
        if current_amount < state.pending_deposit_amount:
            deposited = state.pending_deposit_amount - current_amount
            trace("deposit_verified", self._agent_id, state.step_count,
                  resource=resource, deposited=deposited)

            # NOW update team's deposited_resources (verified)
            self._team_state.deposited_resources[resource] += deposited

            # Clear pending state
            state.pending_deposit_resource = None
            state.pending_deposit_amount = 0
        else:
            # Deposit may have failed or is still processing - clear and retry
            trace("deposit_unverified", self._agent_id, state.step_count,
                  resource=resource, expected_decrease=state.pending_deposit_amount,
                  current=current_amount)
            state.pending_deposit_resource = None
            state.pending_deposit_amount = 0

        return self._noop()

    def _do_deposit(self, state: SpiderState) -> Action:
        """Execute deposit phase (gatherers depositing resources to chest).

        Gatherers deposit their assigned resource to the shared chest,
        making it available for the assembler to withdraw.
        """
        # Check if we have a pending deposit to verify
        if state.pending_deposit_resource is not None:
            return self._handle_deposit_verification(state)

        # Use local chest if known, otherwise try team's shared chest
        chest = state.chest or self._team_state.chest
        if chest is None:
            trace("deposit_fail", self._agent_id, state.step_count, reason="no_chest")
            return self._noop()

        # Update local state from team state if needed
        if state.chest is None and self._team_state.chest is not None:
            state.chest = self._team_state.chest

        current = (state.row, state.col)
        my_resource = self._role.value  # "carbon", "oxygen", "germanium", "silicon"
        current_amount = getattr(state, my_resource, 0)

        if is_adjacent(current, chest):
            # We're at the chest - initiate deposit
            trace("deposit", self._agent_id, state.step_count,
                  pos=chest, resource=my_resource, amount=current_amount)

            # Store pending deposit - will verify on next step
            state.pending_deposit_resource = my_resource
            state.pending_deposit_amount = current_amount

            return self._use_object(state, chest)

        # Move toward chest
        path = find_path_to_target(state, chest, reach_adjacent=True)
        if path:
            return self._follow_path(state, path)

        trace("deposit_fail", self._agent_id, state.step_count, reason="no_path", chest=chest)
        return self._noop()

    def _handle_withdraw_verification(self, state: SpiderState) -> Action:
        """Verify that a pending withdraw succeeded by checking inventory increased."""
        if not state.pending_withdraw:
            return self._noop()

        # Check which resources increased
        resources_gained: dict[str, int] = {}
        for resource in ["carbon", "oxygen", "germanium", "silicon"]:
            before = state.inventory_before_withdraw.get(resource, 0)
            current = getattr(state, resource, 0)
            if current > before:
                resources_gained[resource] = current - before

        if resources_gained:
            trace("withdraw_verified", self._agent_id, state.step_count,
                  gained=resources_gained)

            # Reset deposited_resources for resources we successfully withdrew
            for resource, amount in resources_gained.items():
                # Decrease deposited count (don't go negative)
                self._team_state.deposited_resources[resource] = max(
                    0, self._team_state.deposited_resources[resource] - amount
                )
        else:
            trace("withdraw_unverified", self._agent_id, state.step_count,
                  inventory_before=state.inventory_before_withdraw)

        # Clear pending state
        state.pending_withdraw = False
        state.inventory_before_withdraw = {}
        return self._noop()

    def _do_withdraw(self, state: SpiderState) -> Action:
        """Execute withdraw phase (assembler withdrawing resources from chest).

        The assembler checks if the team has deposited enough resources,
        then withdraws them from the chest.
        """
        # Check if we have a pending withdraw to verify
        if state.pending_withdraw:
            return self._handle_withdraw_verification(state)

        # Use local chest if known, otherwise try team's shared chest
        chest = state.chest or self._team_state.chest
        if chest is None:
            trace("withdraw_fail", self._agent_id, state.step_count, reason="no_chest")
            return self._noop()

        # Update local state from team state if needed
        if state.chest is None and self._team_state.chest is not None:
            state.chest = self._team_state.chest

        current = (state.row, state.col)

        if is_adjacent(current, chest):
            # We're at the chest - initiate withdraw
            # Store inventory before withdraw to verify success
            state.pending_withdraw = True
            state.inventory_before_withdraw = {
                "carbon": state.carbon,
                "oxygen": state.oxygen,
                "germanium": state.germanium,
                "silicon": state.silicon,
            }
            trace("withdraw", self._agent_id, state.step_count,
                  pos=chest, deposited=self._team_state.deposited_resources)
            return self._use_object(state, chest)

        # Move toward chest
        path = find_path_to_target(state, chest, reach_adjacent=True)
        if path:
            return self._follow_path(state, path)

        trace("withdraw_fail", self._agent_id, state.step_count, reason="no_path", chest=chest)
        return self._noop()

    def _do_assemble(self, state: SpiderState) -> Action:
        """Execute assembly phase."""
        # Use local assembler if known, otherwise try team's shared assembler
        assembler = state.assembler or self._team_state.assembler
        if assembler is None:
            trace("assemble_fail", self._agent_id, state.step_count, reason="no_assembler")
            return self._noop()

        # Update local state from team state if needed
        if state.assembler is None and self._team_state.assembler is not None:
            state.assembler = self._team_state.assembler

        current = (state.row, state.col)

        if is_adjacent(current, assembler):
            # Use assembler
            trace("assemble", self._agent_id, state.step_count, pos=assembler)
            return self._use_object(state, assembler)

        # Move toward assembler
        path = find_path_to_target(state, assembler, reach_adjacent=True)
        if path:
            return self._follow_path(state, path)

        trace("assemble_fail", self._agent_id, state.step_count, reason="no_path", assembler=assembler)
        return self._noop()

    def _do_deliver(self, state: SpiderState) -> Action:
        """Execute delivery phase (depositing hearts to chest)."""
        # Use local chest if known, otherwise try team's shared chest
        chest = state.chest or self._team_state.chest
        if chest is None:
            trace("deliver_fail", self._agent_id, state.step_count, reason="no_chest")
            return self._noop()

        # Update local state from team state if needed
        if state.chest is None and self._team_state.chest is not None:
            state.chest = self._team_state.chest

        current = (state.row, state.col)

        if is_adjacent(current, chest):
            # Use chest
            trace("deliver", self._agent_id, state.step_count, pos=chest, hearts=state.hearts)
            return self._use_object(state, chest)

        # Move toward chest
        path = find_path_to_target(state, chest, reach_adjacent=True)
        if path:
            return self._follow_path(state, path)

        trace("deliver_fail", self._agent_id, state.step_count, reason="no_path", chest=chest)
        return self._noop()

    def _do_recharge(self, state: SpiderState) -> Action:
        """Execute recharge phase."""
        # Use local charger if known, otherwise try team's shared charger
        charger = state.charger or self._team_state.charger
        if charger is None:
            return self._noop()

        # Update local state from team state if needed
        if state.charger is None and self._team_state.charger is not None:
            state.charger = self._team_state.charger

        current = (state.row, state.col)

        if is_adjacent(current, charger):
            # Use charger
            return self._use_object(state, charger)

        # Move toward charger
        path = find_path_to_target(state, charger, reach_adjacent=True)
        if path:
            return self._follow_path(state, path)

        return self._noop()

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _calculate_deficits(self, state: SpiderState) -> dict[str, int]:
        """Calculate how many more of each resource we need."""
        if state.heart_recipe is None:
            # Default recipe if not discovered
            return {"carbon": 1, "oxygen": 1, "germanium": 1, "silicon": 1}

        return {
            "carbon": max(0, state.heart_recipe.get("carbon", 0) - state.carbon),
            "oxygen": max(0, state.heart_recipe.get("oxygen", 0) - state.oxygen),
            "germanium": max(0, state.heart_recipe.get("germanium", 0) - state.germanium),
            "silicon": max(0, state.heart_recipe.get("silicon", 0) - state.silicon),
        }

    def _find_needed_extractor(
        self,
        state: SpiderState,
        deficits: dict[str, int]
    ) -> tuple[ExtractorInfo | None, str]:
        """Find nearest extractor for a resource we need."""
        current = (state.row, state.col)

        # Sort by deficit (highest first)
        sorted_resources = sorted(deficits.items(), key=lambda x: -x[1])

        for resource, deficit in sorted_resources:
            if deficit <= 0:
                continue

            extractors = state.extractors.get(resource, [])

            # Filter to available extractors
            available = [e for e in extractors if not e.clipped and e.remaining_uses > 0]

            if available:
                # Return nearest
                nearest = min(available, key=lambda e: manhattan_distance(current, e.position))
                return nearest, resource

        return None, ""

    def _follow_path(self, state: SpiderState, path: list[tuple[int, int]]) -> Action:
        """Take the next step along a path."""
        if not path:
            return self._noop()

        next_pos = path[0]
        current = (state.row, state.col)

        # Check for agent collision
        if next_pos in state.agent_positions:
            # Someone's in the way - try random direction
            return self._random_move(state)

        direction = direction_to_target(current, next_pos)
        if direction:
            return self._actions.move.Move(direction)

        return self._noop()

    def _use_object(self, state: SpiderState, target: tuple[int, int]) -> Action:
        """Move into an object to use it."""
        state.using_object_this_step = True

        current = (state.row, state.col)
        direction = direction_to_target(current, target)

        if direction:
            return self._actions.move.Move(direction)

        return self._noop()

    def _escape_stuck(self, state: SpiderState) -> Action | None:
        """Try to escape when stuck.

        Don't change phases - just do a random move to break out of the loop.
        The frontier may still be reachable via a different route.
        """
        trace("escape_stuck", self._agent_id, state.step_count, phase=state.phase.value)

        # Clear stuck state and history
        state.stuck_detected = False
        state.position_history.clear()  # Clear history so we don't immediately re-detect
        state.cached_path = None

        # Take a random step to break out
        return self._random_move(state)

    def _random_move(self, state: SpiderState) -> Action:
        """Move in a random valid direction."""
        directions = ["north", "south", "east", "west"]
        random.shuffle(directions)

        for direction in directions:
            dr, dc = self._move_deltas[direction]
            nr, nc = state.row + dr, state.col + dc

            if is_traversable(state, nr, nc):
                if (nr, nc) not in state.agent_positions:
                    return self._actions.move.Move(direction)

        return self._noop()

    def _change_vibe(self, vibe_name: str) -> Action:
        """Change to a different vibe."""
        change_vibe_cfg = getattr(self._actions, "change_vibe", None)
        if change_vibe_cfg is None:
            return self._noop()
        if not getattr(change_vibe_cfg, "enabled", True):
            return self._noop()

        vibe = VIBE_BY_NAME.get(vibe_name)
        if vibe is None:
            return self._noop()

        return self._actions.change_vibe.ChangeVibe(vibe)

    def _noop(self) -> Action:
        """Return a no-op action."""
        return self._actions.noop.Noop()


# ============================================================================
# Policy Wrapper Classes
# ============================================================================

class MettaSpiderPolicy(MultiAgentPolicy):
    """
    Multi-agent policy wrapper for MettaSpider.

    This is the entry point that the game calls. It creates per-agent
    policies that share a common team state for coordination.

    Coordination Strategy:
    - Agents 0-2 are gatherers (carbon, oxygen, germanium)
    - Agent 3 is the assembler
    - Gatherers deposit resources to chest
    - Assembler withdraws from chest and makes hearts
    """

    short_names = ["metta_spider", "spider"]

    def __init__(self, policy_env_info: PolicyEnvInterface, **kwargs):
        super().__init__(policy_env_info, **kwargs)
        self._agent_policies: dict[int, StatefulAgentPolicy[SpiderState]] = {}

        # Shared state for team coordination
        self._team_state = SharedTeamState()

    def agent_policy(self, agent_id: int) -> StatefulAgentPolicy[SpiderState]:
        if agent_id not in self._agent_policies:
            # Get role for this agent (defaults to SILICON for unknown IDs - they assemble)
            role = ROLE_BY_AGENT_ID.get(agent_id, AgentRole.SILICON)

            impl = SpiderPolicyImpl(
                self._policy_env_info,
                agent_id,
                team_state=self._team_state,
                role=role,
            )
            self._agent_policies[agent_id] = StatefulAgentPolicy(
                impl,
                self._policy_env_info,
                agent_id=agent_id,
            )
        return self._agent_policies[agent_id]
