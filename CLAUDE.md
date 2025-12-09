# CLAUDE.md – metta-spider

> Issue tracking: `bd` commands (not markdown TODOs). See AGENTS.md.

## Role

**Game AI engineer** building a scripted agent for Alignment League.
This is a **Finite State Machine**, not machine learning (yet).

Think like a game AI programmer:
- State machines need robust transitions AND fallback behaviors
- If the agent is stuck, it's missing a transition condition or fallback
- Debug by tracing state + observations, not guessing
- Every action should have a "what if this fails?" plan

---

## Mental Model

This is a **cooperative multi-agent game** where agents must:
1. Explore to find resources and stations
2. Gather 4 resource types from extractors
3. Assemble resources into HEARTs
4. Deposit HEARTs in chest

Each agent runs a **finite state machine** with phases:
EXPLORE → GATHER → ASSEMBLE → DELIVER (+ RECHARGE interrupt)

**The implementation challenge**: Agents need to handle failure gracefully.
What if an extractor is depleted? What if pathfinding fails?
What if another agent is blocking? The baseline agent handles
these; ours often doesn't.

---

## Strategic Questions (The Real Challenges)

These are the *design* questions that matter more than fixing bugs:

1. **Agent specialization**: Should each agent gather all 4 resources, or specialize? Assembly lines vs generalists?

2. **Coordination**: How do N agents divide work without collision? Who goes where?

3. **Partial observability**: Agents only see 11x11. How to reason about unseen map state?

4. **Efficiency vs robustness**: Aggressive strategies score higher until they break. How defensive?

5. **Meta-game**: What beats the baseline? What beats *that*?

We can't explore these until the FSM stops breaking on basic edge cases.

---

## Known Issues

| Issue | Symptom | Root Cause |
|-------|---------|------------|
| Stuck at depleted extractor | 76+ stuck events per episode | No fallback when extractor gives nothing |
| Oscillation loops | Agent bounces between 2 positions | escape_stuck doesn't clear waiting state |
| Excessive noops | 7000+ noops per 1000 steps | Waiting at broken extractors |

---

## Debugging Workflow

```bash
# Run with trace
cogames play -m training_facility -p class=metta_spider.agent.MettaSpiderPolicy --render none --steps 500

# Check event counts (healthy: <10 stuck, broken: 50+)
cat /tmp/metta_spider.jsonl | jq -r '.event' | sort | uniq -c | sort -rn

# Find stuck patterns
grep -E "stuck|escape" /tmp/metta_spider.jsonl

# Trace specific failure window
cat /tmp/metta_spider.jsonl | jq -c 'select(.step > 100 and .step < 150)'

# See what resource agent is stuck trying to gather
cat /tmp/metta_spider.jsonl | jq -c 'select(.event == "gather_target")' | tail -20
```

**Healthy run**: <10 stuck events, gathered events match expected resources
**Broken run**: 50+ stuck events, same gather_target repeating forever

---

## Reading Eval Output

The goal is to maximize hearts. Key metric: `chest.heart.amount` in "Average Game Stats".

- `chest.heart.amount` = actual hearts deposited (the goal)
- `Per-Agent Reward` = composite score (resources, movement, etc.) - useful but not the goal

Current performance on `training_facility` (max 2 hearts possible):
- Baseline 1 agent: **1.0 hearts** (consistent)
- Spider 1 agent: **0.2 hearts** (inconsistent - sometimes 2, usually 0)

The spider can hit 2 hearts (optimal) but has poor consistency.

---

## Quick Reference

```bash
# List available policies (shows shortnames)
cogames policies

# Run spider policy
cogames play -m training_facility -p class=metta_spider.agent.MettaSpiderPolicy --render unicode --steps 100

# Run baseline for comparison (use shortname)
cogames play -m training_facility -p class=scripted_baseline --render text --steps 100

# Evaluate on specific mission with N agents
cogames eval -m training_facility -c 1 -p class=metta_spider.agent.MettaSpiderPolicy -e 10 --steps 1000
cogames eval -m training_facility -c 4 -p class=metta_spider.agent.MettaSpiderPolicy -e 10 --steps 1000

# List available missions
cogames missions

# View traces (structured JSON)
cat /tmp/metta_spider.jsonl | jq .

# View text log
cat /tmp/metta_spider.log
```

**Requirements**: Python 3.12 (mettagrid C++ bindings fail on 3.13+)

---

## Architecture

```text
src/metta_spider/
├── __init__.py      # Exports MettaSpiderPolicy
├── agent.py         # Main policy logic (SpiderPolicyImpl)
├── types.py         # SpiderState, SharedTeamState, Phase
├── pathfinding.py   # A* pathfinding, direction helpers
└── exploration.py   # Frontier-based exploration
```

**Key classes:**
- `MettaSpiderPolicy(MultiAgentPolicy)` → factory, creates per-agent policies
- `SpiderPolicyImpl(StatefulPolicyImpl)` → core decision logic
- `SpiderState` → per-agent persistent state (map, inventory, phase)
- `SharedTeamState` → team coordination (shared map discoveries)

---

## Phase State Machine

```text
EXPLORE ─────────────────────────────────────────────────┐
    │ (map complete)                                     │
    ▼                                                    │
GATHER ◄─────────────────────────────────────────────────┤
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

---

## FSM Design Principles

**Every state needs:**
1. **Entry condition** - when do we enter this state?
2. **Exit condition** - when do we leave?
3. **Failure fallback** - what if we can't make progress?
4. **Timeout** - max time before forcing a transition

**Current gaps:**
- GATHER has no fallback when extractor is depleted
- No timeout on waiting for resources
- escape_stuck doesn't reset gathering state

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

## Deep Reference Documentation

For detailed API/engine reference, load these files as needed:

| File | When to Load |
|------|--------------|
| `.claude/spider-context.md` | Spider-specific design, failure modes, debugging |
| `.claude/mettagrid-reference.md` | Observation parsing, actions, game mechanics |
| `.claude/cogames-reference.md` | CLI usage, evaluation, game loop |
| `.claude/policy-api-reference.md` | Building/modifying policy classes |
| `.claude/marl-knowledge.md` | MARL architecture (future ML phase) |

**Baseline agent source** (reference implementation):
- `scripted_baseline` = `cogames.policy.scripted_agent.baseline_agent.BaselinePolicy`
- Located at: `../metta/packages/cogames/src/cogames/policy/scripted_agent/`

---

## Future: ML Phase

When the FSM is robust, we can use it for:
- Behavioral cloning (imitation learning from the scripted agent)
- RL fine-tuning
- Self-play and population-based training

But that's Phase 2. Phase 1 is making the FSM actually work.

For MARL architecture notes when we get there, see `.claude/marl-knowledge.md`.
