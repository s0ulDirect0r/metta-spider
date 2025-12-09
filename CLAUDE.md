# CLAUDE.md – metta-spider

> Issue tracking: `bd` commands (not markdown TODOs). See AGENTS.md.

## Role

**ML/RL specialist** building winning policies for Alignment League. Primary goal: produce more HEARTs than competitors.

Think like an RL researcher:
- Debug by instrumenting/visualizing, not guessing
- Ask "what helps the policy learn?" not just "what's correct"
- When stuck, simplify the problem first

---

## Quick Reference

```bash
# Run policy (MUST use full class path)
cogames play -m easy_mode -p class=metta_spider.agent.MettaSpiderPolicy --render unicode --steps 100

# Evaluate
cogames eval -m easy_mode -p class=metta_spider.agent.MettaSpiderPolicy
cogames eval -set integrated_evals -p class=metta_spider.agent.MettaSpiderPolicy

# Baseline comparison
cogames play -m easy_mode -p class=baseline --render text --steps 100

# View traces (structured JSON)
cat /tmp/metta_spider.jsonl | jq .

# View text log
cat /tmp/metta_spider.log
```

**Requirements**: Python 3.12 (mettagrid C++ bindings fail on 3.13+)

---

## Architecture

```
src/metta_spider/
├── __init__.py      # Exports MettaSpiderPolicy
├── agent.py         # Main policy logic (SpiderPolicyImpl)
├── types.py         # SpiderState, SharedTeamState, Phase, AgentRole
├── pathfinding.py   # A* pathfinding, direction helpers
└── exploration.py   # Frontier-based exploration
```

**Key classes:**
- `MettaSpiderPolicy(MultiAgentPolicy)` → factory, creates per-agent policies
- `SpiderPolicyImpl(StatefulPolicyImpl)` → core decision logic
- `SpiderState` → per-agent persistent state (map, inventory, phase)
- `SharedTeamState` → team coordination (shared discoveries, deposit tracking)

---

## Current Implementation

### Role Assignment

| Agent ID | Role | Behavior |
|----------|------|----------|
| 0 | CARBON | Gather carbon → deposit to chest |
| 1 | OXYGEN | Gather oxygen → deposit to chest |
| 2 | GERMANIUM | Gather germanium → deposit to chest |
| 3 | SILICON | Gather silicon → deposit → withdraw all → assemble → deliver |

### Phase State Machine

```
EXPLORE ─────────────────────────────────────────────────┐
    │ (map complete)                                     │
    ▼                                                    │
GATHER ◄─────────────────────────────────────────────┐   │
    │ (have resource)                                │   │
    ▼                                                │   │
DEPOSIT ─────────────────────────────────────────────┘   │
    │ (deposited, silicon role only)                     │
    ▼                                                    │
WITHDRAW ────────────────────────────────────────────────┤
    │ (have all resources)                               │
    ▼                                                    │
ASSEMBLE                                                 │
    │ (have hearts)                                      │
    ▼                                                    │
DELIVER ─────────────────────────────────────────────────┘
    │ (delivered)
    └──► back to GATHER

RECHARGE interrupts any phase when energy < 30
```

### Coordination via SharedTeamState

- **Shared discoveries**: First agent to find chest/assembler/charger/extractors shares with team
- **Deposit tracking**: `deposited_resources` dict so assembler knows when resources available
- **Exploration count**: Track how many agents finished exploring

### Trace Logging

Structured JSON traces to `/tmp/metta_spider.jsonl`:
```json
{"event": "phase", "agent": 0, "step": 42, "old": "explore", "new": "gather"}
{"event": "gathered", "agent": 0, "step": 55, "resource": "carbon", "gained": 1}
```

Events: `role_assigned`, `init`, `discovered`, `shared`, `phase`, `gathered`, `deposit`, `withdraw`, `assemble`, `deliver`, `stuck`, `escape_stuck`

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Short class path `class=MettaSpiderPolicy` | Policy not found | Use `class=metta_spider.agent.MettaSpiderPolicy` |
| Wrong location unpacking | Agent acts on wrong positions | Row = `loc >> 4`, Col = `loc & 0xF` (but we use tuple now) |
| Ignoring energy | Agent stops mid-episode | Check against `ENERGY_LOW=30` threshold |
| Wrong vibe for interaction | Interaction does nothing | `resource_a` = extract/withdraw, `resource_b` = deposit |
| Not finding stations | Agents wander | Check trace for `discovered` events, verify exploration completes |
| Stuck in oscillation | Agent bounces between positions | `stuck_detected` flag triggers `_escape_stuck()` |

---

## Success Criteria

| Mission | Target | Notes |
|---------|--------|-------|
| easy_mode (500 steps) | ≥3 hearts | With `lonely_heart` rules (1 resource each) |
| integrated_evals | Top 50% | Diverse map layouts |

**Health checks:**
- No "stuck" periods >12 steps with ≤4 unique positions
- Energy never hits 0 (planning failure)
- All 4 extractors + chest found (check trace for `discovered` events)
- Phase transitions happening (check trace for `phase` events)

---

## Game Mechanics

**Goal**: Gather resources → assemble HEARTs → deposit in chest

**Resources**: carbon, oxygen, germanium, silicon (from extractors)

**Vibes** (must set correct vibe before interaction):
- `resource_a`: Extract from extractor OR withdraw from chest
- `resource_b`: Deposit resource to chest
- `heart_a`: Assemble hearts at assembler
- `heart_b`: Deposit hearts to chest

**Energy**: Most actions cost energy. Passive +1/turn. Solar arrays give +50.

---

## Reference Code

Baseline scripted agent: `../cogames/src/cogames/policy/scripted_agent/`
- `baseline_agent.py` - full pathfinding implementation
- `utils.py` - `manhattan_distance`, `position_to_direction`, etc.

---

## Extended Context

For MARL architecture decisions, training pipelines, or reward shaping, see `.claude/marl-knowledge.md`. Covers:
- Credit assignment strategies
- Coordination mechanisms (explicit vs emergent)
- Architecture options (set encoders, CNNs, attention)
- Training phases (BC → RL fine-tuning → self-play)
- Debugging patterns for learned policies
