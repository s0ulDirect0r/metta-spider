# CoGames Framework Reference

> CLI and framework for running/evaluating policies in the Alignment League.
> Load this file when working on policy integration, evaluation, or CLI usage.

## Overview

CoGames is the Python framework that wraps mettagrid, providing:
- CLI for play/evaluate/train
- Policy loading and registration
- Multi-episode evaluation
- Replay recording and playback

---

## CLI Commands

### Play (Single Episode)

```bash
# Run with GUI
cogames play -m training_facility -p class=metta_spider.agent.MettaSpiderPolicy --render gui

# Run with unicode terminal rendering
cogames play -m training_facility -p class=scripted_baseline --render unicode --steps 500

# Run headless
cogames play -m training_facility -p class=scripted_baseline --render none
```

### Evaluate (Multi-Episode)

```bash
# Basic evaluation
cogames eval -m training_facility -p class=metta_spider.agent.MettaSpiderPolicy -e 10

# With specific agent count
cogames eval -m training_facility -c 4 -p class=metta_spider.agent.MettaSpiderPolicy -e 10 --steps 1000

# Save replays
cogames eval -m training_facility -p class=scripted_baseline -e 5 --save-replay ./replays
```

### List Available

```bash
# List missions
cogames missions

# List registered policies (shows shortnames)
cogames policies
```

---

## Game Loop

### Flow

```
cogames play/evaluate
    ↓
PolicyEnvInterface.from_mg_cfg(env_cfg)
    ↓
initialize_or_load_policy(env_interface, policy_spec)
    ↓
[policy.agent_policy(i) for i in range(num_agents)]
    ↓
Rollout(config, agent_policies)
    ↓
rollout.run_until_done()
    └── while not sim.is_done():
        ├── for each agent:
        │   ├── obs = agent.observation
        │   ├── action = policy.step(obs)
        │   └── agent.set_action(action)
        ├── sim.step()  # C++ physics
        └── event_handlers.on_step()  # render, log
```

### Key Classes

**Rollout** (`mettagrid.simulator.rollout`):
```python
class Rollout:
    def __init__(
        self,
        config: MettaGridConfig,
        policies: list[AgentPolicy],
        max_action_time_ms: int = 10000,
        render_mode: Optional[RenderMode] = None,
        seed: int = 0,
        event_handlers: Optional[list[SimulatorEventHandler]] = None,
    )

    def step(self) -> None
    def run_until_done(self) -> None
    def is_done(self) -> bool
```

**Simulation** (`mettagrid.simulator.simulator`):
```python
class Simulation:
    def agents(self) -> list[SimulationAgent]
    def step(self) -> None
    def is_done(self) -> bool
    def current_step: int
    def episode_rewards: list[float]
```

**SimulationAgent**:
```python
class SimulationAgent:
    def set_action(self, action: Action | str) -> None
    @property
    def observation(self) -> AgentObservation
    @property
    def inventory(self) -> Dict[str, int]
    @property
    def step_reward(self) -> float
```

---

## Policy Specification

### PolicySpec

```python
@dataclass
class PolicySpec:
    class_path: str           # Full path or shortname
    data_path: Optional[str]  # Checkpoint file
    init_kwargs: dict[str, Any]
```

### CLI Policy Syntax

```bash
# Full class path
-p class=cogames.policy.scripted_agent.baseline_agent.BaselinePolicy

# Shortname (if registered)
-p class=scripted_baseline

# With checkpoint
-p class=lstm data=/path/to/model.pt

# External policy (your code)
-p class=metta_spider.agent.MettaSpiderPolicy
```

---

## Evaluation Output

### Key Metrics

| Metric | Description |
|--------|-------------|
| `chest.heart.amount` | Hearts deposited (THE GOAL) |
| `Per-Agent Reward` | Composite reward score |
| `Action Timeouts` | Policies exceeding time budget |

### Reading Results

```
Average Game Stats:
  chest.heart.amount: 1.2  # Target: maximize this

Per-Agent Reward:
  Policy A: 45.3
  Policy B: 42.1
```

---

## Event Handlers

Hook into game lifecycle:

```python
class SimulatorEventHandler:
    def set_simulation(self, simulation: Simulation) -> None
    def on_episode_start(self) -> None
    def on_step(self) -> None
    def on_episode_end(self) -> None
    def on_close(self) -> None
```

Used for:
- Rendering (GUI, terminal)
- Replay logging
- Statistics tracking

---

## Mission Configuration

**MettaGridConfig** defines:
- `game.num_agents` - Number of agents
- `game.max_steps` - Episode length (0 = no limit)
- `game.obs` - Observation dimensions
- `game.actions` - Available actions
- `game.objects` - Extractors, stations, walls
- `game.map_builder` - Map generation

### Loading Missions

```python
from cogames.game import load_mission_config
config = load_mission_config(Path("mission.yaml"))
```

---

## Baseline Agent Reference

**Location**: `cogames.policy.scripted_agent.baseline_agent`

### Architecture

```python
class BaselinePolicy(MultiAgentPolicy):
    short_names = ["scripted_baseline"]

    def agent_policy(self, agent_id: int) -> StatefulAgentPolicy:
        return StatefulAgentPolicy(
            BaselineAgentPolicyImpl(self._policy_env_info, agent_id, hyperparams),
            self._policy_env_info,
            agent_id=agent_id,
        )
```

### State Machine

```
EXPLORE ──────────────────────────────────────────┐
    │ (map complete or found what we need)        │
    ▼                                             │
GATHER ◄──────────────────────────────────────────┤
    │ (have all resources)                        │
    ▼                                             │
ASSEMBLE                                          │
    │ (have hearts)                               │
    ▼                                             │
DELIVER ──────────────────────────────────────────┘
    │ (delivered)
    └──► back to GATHER

RECHARGE interrupts any phase when energy < 35
```

### Key Design Patterns

1. **Interleaved exploration** - Explores as needed while gathering
2. **Priority ordering** - RECHARGE > DELIVER > ASSEMBLE > GATHER
3. **Stuck detection** - Position history tracks oscillation loops
4. **Path caching** - Reuses paths until target changes or blocked
5. **Vibe management** - Changes vibe before phase actions

---

## Debugging

### Trace Logs

```bash
# Spider agent logs to /tmp/metta_spider.jsonl
cat /tmp/metta_spider.jsonl | jq .

# Event counts
cat /tmp/metta_spider.jsonl | jq -r '.event' | sort | uniq -c | sort -rn

# Find stuck patterns
grep -E "stuck|escape" /tmp/metta_spider.jsonl
```

### Healthy vs Broken Runs

| Metric | Healthy | Broken |
|--------|---------|--------|
| stuck events | < 10 | 50+ |
| noops | < 1000 | 7000+ |
| gathered events | matches expected | same target repeating |

---

## Critical File Locations

```
metta/packages/cogames/src/cogames/
├── main.py              # CLI entry point
├── play.py              # Single episode runner
├── evaluate.py          # Multi-episode evaluation
├── game.py              # Mission loading
└── policy/
    └── scripted_agent/
        ├── baseline_agent.py  # Reference implementation
        ├── types.py           # State dataclasses
        ├── pathfinding.py     # A* algorithm
        └── utils.py           # Helpers

metta/packages/mettagrid/python/src/mettagrid/
├── simulator/
│   ├── simulator.py     # Core simulation
│   ├── rollout.py       # Episode runner
│   └── interface.py     # Action, AgentObservation
├── policy/
│   ├── policy.py        # Policy base classes
│   └── policy_env_interface.py
└── config/
    ├── mettagrid_config.py  # Game configuration
    └── id_map.py            # Feature ID mapping
```
