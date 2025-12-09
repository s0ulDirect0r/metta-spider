# Mettagrid Engine Reference

> Core game engine for the Alignment League multi-agent cooperative game.
> Load this file when working on agent logic, observation parsing, or game mechanics.

## Overview

Mettagrid is a C++ game engine with Python bindings. It's a grid-based simulation where agents navigate, gather resources, and cooperate to assemble and deliver hearts.

**Key characteristics:**
- Grid-based discrete world
- Token-based observations (egocentric 11x11 view)
- Action-based control (move, vibe, noop)
- Partial observability (agents only see nearby cells)

---

## Observation System

### AgentObservation Structure

```python
@dataclass
class AgentObservation:
    agent_id: int
    tokens: Sequence[ObservationToken]

@dataclass
class ObservationToken:
    feature: ObservationFeatureSpec  # name + normalization
    location: tuple[int, int]        # (row, col) in 11x11 grid
    value: int                       # 0-255
    raw_token: tuple[int, int, int]  # raw C++ data
```

### Grid Layout

- **Size**: 11x11 centered on agent
- **Agent position**: Always at center (5, 5)
- **Coordinate system**: (0,0) = top-left, (10,10) = bottom-right

### Coordinate Conversion

```python
# Observation coords → World coords
world_row = obs_row - 5 + agent_row
world_col = obs_col - 5 + agent_col

# Movement deltas
MOVE_DELTAS = {
    "north": (-1, 0),
    "south": (+1, 0),
    "east":  (0, +1),
    "west":  (0, -1),
}
```

### Feature Types

#### Inventory Features (at center cell only)
| Feature | Description | Range |
|---------|-------------|-------|
| `inv:carbon` | Carbon held | 0-255 |
| `inv:oxygen` | Oxygen held | 0-255 |
| `inv:germanium` | Germanium held | 0-255 |
| `inv:silicon` | Silicon held | 0-255 |
| `inv:heart` | Hearts held | 0-255 |
| `inv:energy` | Current energy | 0-255 |
| `inv:decoder` | Decoder items | 0-255 |
| `inv:modulator` | Modulator items | 0-255 |
| `inv:resonator` | Resonator items | 0-255 |
| `inv:scrambler` | Scrambler items | 0-255 |

#### Spatial Features (at object positions)
| Feature | Description | Range |
|---------|-------------|-------|
| `tag` | Object type ID | 0-N (use tag_id_to_name) |
| `cooldown_remaining` | Steps until extractor ready | 0-255 |
| `clipped` | Extractor is broken | 0 or 1 |
| `remaining_uses` | Extractor uses left | 0-999 (capped at 255 in obs) |

#### Agent Features
| Feature | Description |
|---------|-------------|
| `agent:group` | Another agent at this position (team ID) |
| `agent:frozen` | Agent is frozen |

#### Protocol Features (at stations)
| Feature | Description |
|---------|-------------|
| `protocol_input:resource` | Input requirement |
| `protocol_output:resource` | Output produced |

### Important: Zero-Value Tokens

**Tokens with value 0 are NOT sent.** Always use defaults:
```python
value = token_dict.get("inv:carbon", 0)
```

---

## Action System

### Action Types

```python
# Movement (costs 2 energy if successful)
self._actions.move.Move("north")  # or "south", "east", "west"

# Vibe change (sets interaction mode)
self._actions.change_vibe.ChangeVibe(vibe)

# No-op (wait, get +1 energy regen)
self._actions.noop.Noop()
```

### Move Success Detection

```python
prev_energy = state.energy
# ... execute move ...
energy_delta = new_energy - prev_energy

if energy_delta <= 0:  # Cost paid or at cap
    move_succeeded = True
else:  # Got regen but no cost = blocked
    move_succeeded = False
```

### Vibes (Interaction Modes)

Must set correct vibe before interacting with stations:

| Vibe | Purpose |
|------|---------|
| `resource_a` | Extract from extractor OR withdraw from chest |
| `resource_b` | Deposit resource to chest |
| `heart_a` | Assemble hearts at assembler |
| `default` / `heart_b` | Deposit hearts to chest |

Resource-specific vibes: `carbon_a`, `oxygen_a`, `germanium_a`, `silicon_a`

---

## Game Mechanics

### Resources

Four types gathered from extractors:
- **Carbon** - from carbon extractors
- **Oxygen** - from oxygen extractors
- **Germanium** - from germanium extractors
- **Silicon** - from silicon extractors

### Energy System

| Mechanic | Value |
|----------|-------|
| Starting energy | 100 |
| Passive regen | +1/step (unless at cap) |
| Move cost | 2 energy |
| Energy cap | 255 |
| Exhaustion | Can't move if energy < 2 |

### Heart Assembly

1. Gather all 4 resources (amounts from recipe)
2. Navigate to assembler
3. Set vibe to `heart_a`
4. Move adjacent to assembler (interaction is automatic)
5. Navigate to chest
6. Set vibe to `default` or `heart_b`
7. Move adjacent to chest to deposit

**Recipe**: Discovered from assembler's `protocol_output:heart` tokens

### Extractors

| State | Meaning |
|-------|---------|
| `cooldown_remaining > 0` | Can't use yet |
| `cooldown_remaining == 0` | Ready to use |
| `remaining_uses == 0` | Depleted |
| `clipped == 1` | Broken (needs unclip item) |

**Detection of depletion**: After using, if no resource gained and cooldown very high (≥250), extractor is depleted.

### Stations

All stations are obstacles (can't walk through):
- **Assembler** - converts resources to hearts
- **Chest** - storage and goal delivery point
- **Charger** - recharges energy

---

## Object Detection

### Tag Parsing

```python
tag_id = token.value  # from "tag" feature
object_name = self._tag_names[tag_id].lower()

# Identify object type
if "wall" in object_name or object_name == "#":
    cell_type = OBSTACLE
elif "extractor" in object_name:
    resource = object_name.replace("_extractor", "").replace("clipped_", "")
elif object_name == "assembler":
    ...
elif object_name == "chest":
    ...
elif "charger" in object_name or "solar" in object_name:
    ...
elif object_name == "agent":
    # Another agent
    ...
```

---

## Occupancy Grid

```python
class CellType(Enum):
    UNKNOWN = 0   # Never observed
    FREE = 1      # Passable
    OBSTACLE = 2  # Wall, station, extractor
```

### Update Pattern

1. First pass: Mark ALL observed cells as FREE
2. Second pass: Override with OBSTACLE for walls/stations/extractors
3. Track other agents separately (they move)

---

## Multi-Agent Coordination

### Agent Detection

```python
# Check for agent:group feature at position
if token.feature.name == "agent:group":
    other_agent_at_position = True
```

### Collision Avoidance

```python
# Before moving to cell (nr, nc)
if (nr, nc) in state.agent_occupancy:
    # Cell blocked by another agent
    try_alternative_direction()
```

---

## PolicyEnvInterface

Provided by engine, contains environment configuration:

```python
policy_env_info.obs_height         # 11
policy_env_info.obs_width          # 11
policy_env_info.actions            # Action factory
policy_env_info.tag_id_to_name     # Tag ID → object name
policy_env_info.assembler_protocols # Recipe information
policy_env_info.action_names       # List of action strings
```

---

## Critical Imports

```python
from mettagrid.simulator import Action, AgentObservation
from mettagrid.policy.policy import MultiAgentPolicy, StatefulPolicyImpl, StatefulAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.config.mettagrid_config import CardinalDirection, CardinalDirections
```
