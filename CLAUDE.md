# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **Alignment League** submission for the Cogs vs Clips game, built on the MettaGrid/CoGames framework. The goal is to develop AI agents ("Cogs") that cooperate to produce HEARTs by gathering resources, operating machinery, and assembling components.

The project name is `metta-spider` and exports a `MettaSpiderPolicy` class.

## Requirements

- **Python 3.12** (required - mettagrid C++ bindings don't work with 3.13+)

```bash
# Install Python 3.12 if needed
uv python install 3.12.11

# Create venv with correct Python
uv venv .venv --python 3.12
source .venv/bin/activate
```

## Development Commands

Policy syntax uses comma-separated `key=value` pairs: `class=<name>,data=<weights>,proportion=<float>`

```bash
# Install dependencies
uv pip install -e .

# List available missions and policies
cogames missions
cogames policies

# Play easy_mode with baseline scripted policy (good for testing)
cogames play -m easy_mode -p class=baseline

# Play with text renderer (avoids GUI hangs)
cogames play -m easy_mode -p class=baseline --render text --steps 100

# Play with your policy
cogames play -m easy_mode -p class=metta_spider

# Evaluate your policy
cogames eval -m easy_mode -p class=metta_spider

# Evaluate on multiple missions
cogames eval -set integrated_evals -p class=metta_spider

# Train an LSTM policy on easy_mode
cogames train -m easy_mode -p class=lstm

# Submit to Alignment League (requires login)
cogames login
cogames submit -p class=metta_spider -n "My Submission Name"

# Dry run submission validation
cogames submit -p class=metta_spider -n "Test" --dry-run
```

### easy_mode variants

The `easy_mode` mission uses simplified rules for training:
- `lonely_heart` - Heart crafting needs only 1 of each resource
- `heart_chorus` - Reward shaping for hearts and diverse inventories
- `pack_rat` - All capacity limits raised to 255

## Architecture

### Policy Structure

```
src/metta_spider/
├── __init__.py          # Exports MettaSpiderPolicy
└── agent.py             # Policy implementation
```

The policy must implement the `MultiAgentPolicy` interface from `mettagrid.policy.policy`:

- **`MettaSpiderPolicy(MultiAgentPolicy)`**: Factory that creates per-agent policies
  - `short_names`: List of aliases for CLI registration (e.g., `["metta_spider", "spider"]`)
  - `agent_policy(agent_id) -> AgentPolicy`: Returns policy for specific agent

- **`SpiderAgentPolicy(AgentPolicy)`**: Per-agent decision-making
  - `step(obs: AgentObservation) -> Action`: Core method - takes observation, returns action

### Observation Format

Observations are token-based: `AgentObservation.tokens` contains a list of tokens where each token has:
- `location`: Packed coordinate (upper 4 bits = row, lower 4 bits = col), `0xFF` = empty
- `feature_id`: What feature this token represents
- `value`: The feature value

Agent is centered at location `0x55` (row 5, col 5) in an 11x11 observation window.

Access environment info via `self._policy_env_info`:
- `action_names`: List of available action names
- `obs_features`: List of observation features
- `tags`: Object type tags
- `assembler_protocols`: Recipes for assemblers

### Available Actions

Return actions by name: `Action(name="move_north")`, `Action(name="noop")`, etc.

- **Movement**: `move_north`, `move_south`, `move_east`, `move_west`
- **Vibes**: `change_vibe_heart_a`, `change_vibe_carbon_a`, etc.
- **Noop**: `noop`

Moving into a station/object triggers interaction. Vibes affect interaction protocols.

### Game Mechanics (Cogs vs Clips)

**Goal**: Produce HEARTs and deposit them in chests.

**Resources**: carbon, oxygen, germanium, silicon (gather from extractors)

**Stations**:
- **Extractors**: Harvest resources (carbon, oxygen, germanium, silicon extractors)
- **Assemblers**: Convert resources to HEARTs (requires specific vibe + adjacent Cogs)
- **Chests**: Store/retrieve resources and HEARTs (vibe controls deposit/withdraw)
- **Solar Arrays**: Recharge energy

**Energy**: Most actions cost energy. Passive +1/turn, solar arrays give +50.

## Reference Implementations

Look at `../cogames/src/cogames/policy/scripted_agent/` for examples:
- `baseline_agent.py`: Full-featured scripted agent with pathfinding
- `starter_agent.py`: Simpler agent for reference
- `types.py`: Data structures for agent state
- `utils.py`: Helper functions (manhattan_distance, position_to_direction, etc.)

## Key Dependencies

- `mettagrid`: Core simulation and policy interfaces
- `cogames`: Game environment, CLI, and built-in policies
