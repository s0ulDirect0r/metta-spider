# Policy API Reference

> How to implement policies for the Alignment League.
> Load this file when building or modifying agent policies.

## Policy Class Hierarchy

```
MultiAgentPolicy (factory)
    └── agent_policy(agent_id) → AgentPolicy

AgentPolicy (per-agent interface)
    └── step(obs) → Action

StatefulPolicyImpl[StateType] (stateful logic)
    └── step_with_state(obs, state) → (Action, StateType)

StatefulAgentPolicy (wrapper)
    └── Wraps StatefulPolicyImpl, manages state between steps
```

---

## Implementing a Policy

### Basic Structure

```python
from mettagrid.policy.policy import MultiAgentPolicy, StatefulAgentPolicy, StatefulPolicyImpl
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action
from mettagrid.simulator.interface import AgentObservation

class MyAgentState:
    """Per-agent persistent state."""
    def __init__(self):
        self.step_count = 0
        self.position = (0, 0)
        # ... other state

class MyPolicyImpl(StatefulPolicyImpl[MyAgentState]):
    """Per-agent decision logic."""

    def __init__(self, policy_env_info: PolicyEnvInterface, agent_id: int):
        self._agent_id = agent_id
        self._actions = policy_env_info.actions
        self._obs_hr = policy_env_info.obs_height // 2  # 5 for 11x11
        self._obs_wr = policy_env_info.obs_width // 2

    def initial_agent_state(self) -> MyAgentState:
        """Called once at episode start."""
        return MyAgentState()

    def step_with_state(
        self, obs: AgentObservation, state: MyAgentState
    ) -> tuple[Action, MyAgentState]:
        """Called every step. Returns action and updated state."""
        state.step_count += 1

        # Parse observation
        # Update state
        # Decide action

        action = self._actions.move.Move("north")
        return action, state


class MyPolicy(MultiAgentPolicy):
    """Factory that creates per-agent policies."""

    short_names = ["my_policy"]  # For CLI: -p class=my_policy

    def __init__(self, policy_env_info: PolicyEnvInterface):
        super().__init__(policy_env_info)
        self._agent_policies = {}

    def agent_policy(self, agent_id: int) -> StatefulAgentPolicy:
        if agent_id not in self._agent_policies:
            impl = MyPolicyImpl(self._policy_env_info, agent_id)
            self._agent_policies[agent_id] = StatefulAgentPolicy(
                impl, self._policy_env_info, agent_id=agent_id
            )
        return self._agent_policies[agent_id]
```

---

## PolicyEnvInterface

Environment information provided to policies:

```python
policy_env_info.num_agents         # Number of agents
policy_env_info.obs_height         # 11
policy_env_info.obs_width          # 11
policy_env_info.actions            # Action factory object
policy_env_info.action_names       # List of action strings
policy_env_info.tag_id_to_name     # Dict[int, str] for object identification
policy_env_info.assembler_protocols # Recipe information
policy_env_info.observation_space  # gym.spaces.Box
policy_env_info.action_space       # gym.spaces.Discrete
```

---

## Actions

### Creating Actions

```python
# From policy_env_info.actions
actions = policy_env_info.actions

# Movement
actions.move.Move("north")   # Returns Action(name="move_north")
actions.move.Move("south")
actions.move.Move("east")
actions.move.Move("west")

# Vibe change
actions.change_vibe.ChangeVibe(vibe)  # vibe is a Vibe object

# No-op
actions.noop.Noop()
```

### Vibe Actions

```python
from mettagrid.config.vibes import VIBE_BY_NAME

# Get vibe object
vibe = VIBE_BY_NAME["carbon_a"]
action = actions.change_vibe.ChangeVibe(vibe)

# Common vibes
VIBE_BY_NAME["resource_a"]   # Extract/withdraw
VIBE_BY_NAME["resource_b"]   # Deposit
VIBE_BY_NAME["heart_a"]      # Assemble
VIBE_BY_NAME["default"]      # No special interaction
VIBE_BY_NAME["carbon_a"]     # Carbon-specific
VIBE_BY_NAME["oxygen_a"]     # Oxygen-specific
VIBE_BY_NAME["germanium_a"]  # Germanium-specific
VIBE_BY_NAME["silicon_a"]    # Silicon-specific
```

---

## Parsing Observations

### Token Structure

```python
for token in obs.tokens:
    feature_name = token.feature.name   # e.g., "inv:carbon", "tag"
    location = token.location           # (row, col) in 11x11 grid
    value = token.value                 # 0-255
```

### Reading Inventory (at center cell)

```python
def read_inventory(obs: AgentObservation, obs_hr: int, obs_wr: int) -> dict:
    center = (obs_hr, obs_wr)  # (5, 5) for 11x11
    inventory = {}

    for token in obs.tokens:
        if token.location == center and token.feature.name.startswith("inv:"):
            resource = token.feature.name[4:]  # Remove "inv:" prefix
            inventory[resource] = token.value

    return inventory
```

### Reading Nearby Objects

```python
def parse_objects(
    obs: AgentObservation,
    tag_id_to_name: dict[int, str],
    agent_row: int,
    agent_col: int,
    obs_hr: int,
    obs_wr: int,
) -> dict:
    objects = {}  # (world_row, world_col) -> object_info

    for token in obs.tokens:
        obs_r, obs_c = token.location

        # Convert to world coordinates
        world_r = obs_r - obs_hr + agent_row
        world_c = obs_c - obs_wr + agent_col

        if token.feature.name == "tag":
            obj_name = tag_id_to_name.get(token.value, "unknown")
            objects[(world_r, world_c)] = {"name": obj_name}

        elif token.feature.name == "cooldown_remaining":
            if (world_r, world_c) in objects:
                objects[(world_r, world_c)]["cooldown"] = token.value

        elif token.feature.name == "clipped":
            if (world_r, world_c) in objects:
                objects[(world_r, world_c)]["clipped"] = token.value > 0

        elif token.feature.name == "remaining_uses":
            if (world_r, world_c) in objects:
                objects[(world_r, world_c)]["remaining_uses"] = token.value

    return objects
```

### Detecting Other Agents

```python
def find_other_agents(obs: AgentObservation, obs_hr: int, obs_wr: int) -> set:
    agent_positions = set()

    for token in obs.tokens:
        if token.feature.name == "agent:group":
            obs_r, obs_c = token.location
            # Skip center (self)
            if (obs_r, obs_c) != (obs_hr, obs_wr):
                agent_positions.add((obs_r, obs_c))

    return agent_positions
```

---

## State Management Patterns

### Position Tracking

```python
class AgentState:
    row: int = 100  # Start at center of virtual map
    col: int = 100
    last_action: Action = None

def update_position(state: AgentState, move_deltas: dict):
    """Call at start of step to update position from last action."""
    if state.last_action is None:
        return

    action_name = state.last_action.name
    if action_name.startswith("move_"):
        direction = action_name[5:]  # Remove "move_" prefix
        if direction in move_deltas:
            dr, dc = move_deltas[direction]
            state.row += dr
            state.col += dc
```

### Occupancy Grid

```python
from enum import Enum

class CellType(Enum):
    UNKNOWN = 0
    FREE = 1
    OBSTACLE = 2

class AgentState:
    occupancy: list[list[int]]  # 2D grid of CellType values
    map_height: int = 200
    map_width: int = 200

def update_occupancy(state, parsed_objects, obs_hr, obs_wr):
    """Update occupancy from observation."""
    # Mark all observed cells as FREE
    for obs_r in range(2 * obs_hr + 1):
        for obs_c in range(2 * obs_wr + 1):
            world_r = obs_r - obs_hr + state.row
            world_c = obs_c - obs_wr + state.col
            if 0 <= world_r < state.map_height and 0 <= world_c < state.map_width:
                state.occupancy[world_r][world_c] = CellType.FREE.value

    # Mark obstacles
    for pos, obj in parsed_objects.items():
        r, c = pos
        if is_obstacle(obj["name"]):
            state.occupancy[r][c] = CellType.OBSTACLE.value
```

---

## Common Patterns

### Pathfinding

```python
from collections import deque

def shortest_path(
    state,
    start: tuple[int, int],
    goals: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    """BFS pathfinding. Returns path or None if unreachable."""
    if start in goals:
        return []

    queue = deque([(start, [])])
    visited = {start}

    while queue:
        (r, c), path = queue.popleft()

        for dr, dc in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
            nr, nc = r + dr, c + dc
            next_pos = (nr, nc)

            if next_pos in visited:
                continue
            if not is_traversable(state, nr, nc):
                continue

            new_path = path + [next_pos]

            if next_pos in goals:
                return new_path

            visited.add(next_pos)
            queue.append((next_pos, new_path))

    return None  # No path found


def is_traversable(state, r: int, c: int) -> bool:
    if r < 0 or r >= state.map_height or c < 0 or c >= state.map_width:
        return False
    return state.occupancy[r][c] == CellType.FREE.value
```

### Stuck Detection

```python
class AgentState:
    position_history: list[tuple[int, int]] = []
    stuck_detected: bool = False

def check_stuck(state, max_history: int = 30):
    """Detect oscillation loops."""
    state.position_history.append((state.row, state.col))
    if len(state.position_history) > max_history:
        state.position_history.pop(0)

    # Check for 2-position loop (A->B->A->B->A->B)
    if len(state.position_history) >= 6:
        h = state.position_history
        if (h[-1] == h[-3] == h[-5] and
            h[-2] == h[-4] == h[-6] and
            h[-1] != h[-2]):
            state.stuck_detected = True
```

### Phase State Machine

```python
from enum import Enum

class Phase(Enum):
    EXPLORE = "explore"
    GATHER = "gather"
    ASSEMBLE = "assemble"
    DELIVER = "deliver"
    RECHARGE = "recharge"

def update_phase(state):
    """Priority-based phase selection."""
    # Priority 1: Recharge if low energy
    if state.energy < 35:
        state.phase = Phase.RECHARGE
        return

    # Priority 2: Deliver if have hearts
    if state.hearts > 0:
        state.phase = Phase.DELIVER
        return

    # Priority 3: Assemble if have all resources
    if has_all_resources(state):
        state.phase = Phase.ASSEMBLE
        return

    # Priority 4: Gather (default)
    state.phase = Phase.GATHER
```

---

## Registration and CLI

### Registering a Policy

Add `short_names` class attribute:

```python
class MyPolicy(MultiAgentPolicy):
    short_names = ["my_policy", "mp"]  # Both work in CLI
```

### Using Your Policy

```bash
# By full path
cogames play -m training_facility -p class=metta_spider.agent.MettaSpiderPolicy

# By shortname (if registered in cogames)
cogames play -m training_facility -p class=scripted_baseline
```

---

## Testing Policies

```bash
# Quick visual test
cogames play -m training_facility -p class=your.policy.Class --render unicode --steps 100

# Headless eval
cogames eval -m training_facility -p class=your.policy.Class -e 10 --steps 1000

# Check trace logs
cat /tmp/metta_spider.jsonl | jq -r '.event' | sort | uniq -c | sort -rn
```
