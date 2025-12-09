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
    CellType,
    ExtractorInfo,
    Phase,
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

    Each agent is self-sufficient: gathers all resources, assembles hearts,
    and delivers them. Works with any number of agents (1, 4, or N).
    """

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        agent_id: int,
        team_state: SharedTeamState,
    ):
        self._agent_id = agent_id
        self._policy_env_info = policy_env_info
        self._team_state = team_state

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

        trace("init_agent", agent_id, 0)

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

        # If team already knows this extractor is depleted, mark it immediately
        team_knows_depleted = pos in self._team_state.depleted_extractors

        # Check if we already know about this extractor
        for ext in state.extractors[resource_type]:
            if ext.position == pos:
                # Update existing - but preserve depleted marking
                ext.cooldown_remaining = data.get("cooldown", 0)
                ext.clipped = data.get("clipped", 0) > 0
                # Sync with team's depleted knowledge
                if team_knows_depleted:
                    ext.remaining_uses = 0
                elif ext.remaining_uses != 0:
                    ext.remaining_uses = data.get("remaining", 999)
                return

        # Add new extractor to local state
        # If team knows it's depleted, mark it as such from the start
        remaining = 0 if team_knows_depleted else data.get("remaining", 999)
        state.extractors[resource_type].append(ExtractorInfo(
            position=pos,
            resource_type=resource_type,
            cooldown_remaining=data.get("cooldown", 0),
            clipped=data.get("clipped", 0) > 0,
            remaining_uses=remaining,
        ))
        trace("discovered", self._agent_id, state.step_count,
              type=f"{resource_type}_extractor", pos=pos, team_depleted=team_knows_depleted)

        # Share with team (first extractor of this type wins)
        if self._team_state.extractors[resource_type] is None:
            self._team_state.extractors[resource_type] = pos
            trace("shared", self._agent_id, state.step_count,
                  type=f"{resource_type}_extractor", pos=pos)

    # ========================================================================
    # Phase Management
    # ========================================================================

    def _update_phase(self, state: SpiderState) -> None:
        """Update phase based on current state.

        Simple priority-based logic - each agent is self-sufficient:
        1. Recharge if energy low
        2. Deliver if have hearts
        3. Assemble if have all resources
        4. Continue exploring if not complete
        5. Default: GATHER
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

        # Priority 2: Deliver if we have hearts
        if state.hearts > 0:
            if state.phase != Phase.DELIVER:
                trace("phase", self._agent_id, state.step_count,
                      old=state.phase.value, new="deliver", reason="have_hearts", hearts=state.hearts)
                state.phase = Phase.DELIVER
                state.target = None
                state.cached_path = None
            return

        # Priority 3: Assemble if we have all resources
        if self._can_assemble(state):
            if state.phase != Phase.ASSEMBLE:
                trace("phase", self._agent_id, state.step_count,
                      old=state.phase.value, new="assemble", reason="have_resources")
                state.phase = Phase.ASSEMBLE
                state.target = None
                state.cached_path = None
            return

        # Priority 4: If exploring, check if complete
        if state.phase == Phase.EXPLORE:
            if is_exploration_complete(state):
                # Guard: only count exploration once per agent
                if not state.exploration_complete:
                    ext_counts = {r: len(e) for r, e in state.extractors.items() if e}
                    self._team_state.exploration_done_count += 1
                    state.exploration_complete = True
                    trace("phase", self._agent_id, state.step_count,
                          old="explore", new="gather", reason="exploration_complete",
                          extractors=ext_counts)
                else:
                    trace("phase", self._agent_id, state.step_count,
                          old="explore", new="gather", reason="re_explored")

                state.phase = Phase.GATHER
                state.target = None
                state.cached_path = None
            return

        # Default: if not in a valid phase, go to GATHER
        if state.phase not in (Phase.GATHER, Phase.EXPLORE, Phase.RECHARGE):
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
        """Get the vibe we should be in for current phase.

        Vibe reference:
        - resource_a: Extract from extractor (GATHER phase)
        - heart_a: Assemble hearts (ASSEMBLE phase)
        - heart_b: Deposit hearts to chest (DELIVER phase)
        """
        if state.phase == Phase.ASSEMBLE:
            return "heart_a"
        elif state.phase == Phase.DELIVER:
            return "heart_b"  # Deposit hearts to chest
        elif state.phase == Phase.GATHER:
            # Use the vibe for whatever resource we're currently targeting
            if state.target_resource:
                return RESOURCE_TO_VIBE.get(state.target_resource, "default")
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

        Deficit-based: gather whichever resource we need most.
        """
        # If we're already waiting at an extractor, handle that first
        wait_action = self._handle_extractor_wait(state)
        if wait_action is not None:
            return wait_action
        # If None, continue below to find next target

        # Find which resource we need most and its nearest extractor
        deficits = self._calculate_deficits(state)
        extractor, resource = self._find_needed_extractor(state, deficits)

        if extractor is None:
            # Don't know where to find what we need - explore more!
            state.exploration_complete = False
            state.phase = Phase.EXPLORE
            trace("gather_fail", self._agent_id, state.step_count,
                  reason="no_extractor", deficits=deficits)
            return self._do_explore(state)

        state.target_resource = resource
        trace("gather_target", self._agent_id, state.step_count,
              resource=resource, target=extractor.position)

        # Navigate to extractor
        current = (state.row, state.col)

        if is_adjacent(current, extractor.position):
            # We're adjacent - check if usable

            # Skip if depleted or clipped - find another extractor
            # Note: depleted extractors often show cooldown=255 instead of remaining_uses=0
            is_depleted = (
                extractor.remaining_uses == 0 or
                extractor.clipped or
                extractor.cooldown_remaining >= 250  # Effectively infinite = depleted
            )
            if is_depleted:
                trace("extractor_unusable", self._agent_id, state.step_count,
                      resource=resource, pos=extractor.position,
                      remaining=extractor.remaining_uses, clipped=extractor.clipped,
                      cooldown=extractor.cooldown_remaining)
                # Mark it locally AND share with team so others avoid it
                extractor.remaining_uses = 0
                self._team_state.depleted_extractors.add(extractor.position)
                trace("shared_depleted", self._agent_id, state.step_count,
                      pos=extractor.position, resource=resource)
                # Clear target and let next iteration find a different extractor
                state.target_resource = None
                return self._noop()

            # Wait if on cooldown (normal cooldown, not depleted)
            if extractor.cooldown_remaining > 0:
                state.wait_steps += 1
                # Set waiting state so timeout can trigger
                if state.waiting_at_extractor is None:
                    state.waiting_at_extractor = extractor.position
                    state.pending_resource = resource
                    state.pending_amount = getattr(state, resource, 0)
                    trace("extractor_cooldown", self._agent_id, state.step_count,
                          resource=resource, pos=extractor.position, cooldown=extractor.cooldown_remaining)
                return self._noop()

            # Ready to use - record pre-use amount and initiate
            state.pending_resource = resource
            state.pending_amount = getattr(state, resource, 0)
            state.waiting_at_extractor = extractor.position
            state.wait_steps = 0

            trace("extractor_use", self._agent_id, state.step_count,
                  resource=resource, pos=extractor.position)
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

    def _handle_extractor_wait(self, state: SpiderState) -> Action | None:
        """Handle waiting for resource from extractor.

        Returns:
            Action if still waiting (noop)
            None if done waiting (success or timeout) - lets _do_gather continue
        """
        resource = state.pending_resource
        if resource is None or state.waiting_at_extractor is None:
            return None  # Not waiting, continue with gather logic

        # Check if we received the resource
        current_amount = getattr(state, resource, 0)
        if current_amount > state.pending_amount:
            # Success! Clear waiting state
            gained = current_amount - state.pending_amount
            trace("gathered", self._agent_id, state.step_count,
                  resource=resource, gained=gained, total=current_amount)
            self._clear_waiting_state(state)
            return None  # Continue to find next target

        # Look up extractor to get cooldown-based timeout
        extractor = self._find_extractor_at_position(state, state.waiting_at_extractor)
        max_wait = (extractor.cooldown_remaining + 5) if extractor else 15

        state.wait_steps += 1
        if state.wait_steps > max_wait:
            # Timeout - mark extractor as depleted and try another
            self._mark_extractor_depleted(state, resource, state.waiting_at_extractor)
            trace("extractor_timeout", self._agent_id, state.step_count,
                  resource=resource, pos=state.waiting_at_extractor, waited=state.wait_steps)
            self._clear_waiting_state(state)
            return None  # Let _do_gather try again immediately

        return self._noop()

    def _clear_waiting_state(self, state: SpiderState) -> None:
        """Clear extractor waiting state."""
        state.waiting_at_extractor = None
        state.pending_resource = None
        state.pending_amount = 0
        state.wait_steps = 0

    def _find_extractor_at_position(
        self,
        state: SpiderState,
        pos: tuple[int, int]
    ) -> ExtractorInfo | None:
        """Find extractor info at a given position."""
        for extractors in state.extractors.values():
            for ext in extractors:
                if ext.position == pos:
                    return ext
        return None

    def _mark_extractor_depleted(
        self,
        state: SpiderState,
        resource: str,
        pos: tuple[int, int]
    ) -> None:
        """Mark an extractor as depleted locally AND share with team."""
        extractors = state.extractors.get(resource, [])
        for ext in extractors:
            if ext.position == pos:
                ext.remaining_uses = 0
                # Share with team so all agents avoid this extractor
                self._team_state.depleted_extractors.add(pos)
                trace("shared_depleted", self._agent_id, state.step_count,
                      pos=pos, resource=resource)
                return

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
        """Find extractor using agent_id to distribute across team.

        Instead of all agents picking the same "nearest" extractor, we:
        1. Filter out team-known depleted extractors
        2. Find candidates within tolerance distance of nearest
        3. Use agent_id to pick deterministically from candidates
        4. Fall back to team-shared positions if local list is empty

        This distributes agents across available extractors.
        """
        current = (state.row, state.col)

        # Sort by deficit (highest first)
        sorted_resources = sorted(deficits.items(), key=lambda x: -x[1])

        for resource, deficit in sorted_resources:
            if deficit <= 0:
                continue

            extractors = state.extractors.get(resource, [])

            # Filter: available AND not team-depleted
            available = [
                e for e in extractors
                if not e.clipped
                and e.remaining_uses > 0
                and e.position not in self._team_state.depleted_extractors
            ]

            # If no local extractors, check team-shared position
            if not available:
                shared_pos = self._team_state.extractors.get(resource)
                if shared_pos and shared_pos not in self._team_state.depleted_extractors:
                    trace("extractor_fallback", self._agent_id, state.step_count,
                          resource=resource, pos=shared_pos, reason="using_team_shared")
                    return ExtractorInfo(position=shared_pos, resource_type=resource), resource
                continue

            if len(available) == 1:
                return available[0], resource

            # Distribute agents across near-equal options
            sorted_by_dist = sorted(
                available, key=lambda e: manhattan_distance(current, e.position)
            )
            nearest_dist = manhattan_distance(current, sorted_by_dist[0].position)
            tolerance = 5  # Extractors within 5 steps considered equally close
            candidates = [
                e for e in sorted_by_dist
                if manhattan_distance(current, e.position) <= nearest_dist + tolerance
            ]

            # Use agent_id to pick deterministically
            idx = state.agent_id % len(candidates)
            return candidates[idx], resource

        return None, ""

    def _follow_path(self, state: SpiderState, path: list[tuple[int, int]]) -> Action:
        """Take the next step along a path, with smart collision handling.

        Instead of immediately falling back to random moves when blocked,
        we wait briefly (2 steps) to let the other agent move, then try
        to find an alternate path around them.
        """
        if not path:
            return self._noop()

        next_pos = path[0]
        current = (state.row, state.col)

        # Check for agent collision
        if next_pos in state.agent_positions:
            state.collision_wait_steps += 1

            # Wait briefly to let other agent move
            if state.collision_wait_steps < 3:
                return self._noop()

            # Waited too long - try to find alternate path
            state.collision_wait_steps = 0

            if len(path) > 1:
                dest = path[-1]
                blocked_r, blocked_c = next_pos
                # Temporarily mark blocked cell as obstacle
                old_value = state.occupancy[blocked_r][blocked_c]
                state.occupancy[blocked_r][blocked_c] = CellType.OBSTACLE.value
                # Use reach_adjacent=False since dest is already the cell we want
                # (original path already computed adjacency if needed)
                alt_path = find_path_to_target(state, dest, reach_adjacent=False)
                state.occupancy[blocked_r][blocked_c] = old_value

                if alt_path:
                    # Found alternate route - follow it (but don't recurse forever)
                    alt_next = alt_path[0]
                    if alt_next not in state.agent_positions:
                        direction = direction_to_target(current, alt_next)
                        if direction:
                            return self._actions.move.Move(direction)

            # No alternate path found - fall back to random
            return self._random_move(state)

        # Clear wait counter when path is clear
        state.collision_wait_steps = 0

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

        # Clear any pending extractor wait (might be stuck at a depleted extractor)
        if state.waiting_at_extractor is not None:
            # Mark extractor as depleted since we're stuck trying to use it
            if state.pending_resource:
                self._mark_extractor_depleted(state, state.pending_resource, state.waiting_at_extractor)
            self._clear_waiting_state(state)

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

    Each agent is self-sufficient (gathers all resources, assembles, delivers).
    Agents share map discoveries but work independently.
    Works with any number of agents (1, 4, or N).
    """

    short_names = ["metta_spider", "spider"]

    def __init__(self, policy_env_info: PolicyEnvInterface, **kwargs):
        super().__init__(policy_env_info, **kwargs)
        self._agent_policies: dict[int, StatefulAgentPolicy[SpiderState]] = {}

        # Shared state for team coordination (map sharing only)
        self._team_state = SharedTeamState()

    def agent_policy(self, agent_id: int) -> StatefulAgentPolicy[SpiderState]:
        if agent_id not in self._agent_policies:
            impl = SpiderPolicyImpl(
                self._policy_env_info,
                agent_id,
                team_state=self._team_state,
            )
            self._agent_policies[agent_id] = StatefulAgentPolicy(
                impl,
                self._policy_env_info,
                agent_id=agent_id,
            )
        return self._agent_policies[agent_id]
