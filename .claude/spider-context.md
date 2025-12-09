# Spider Agent Context

> Design decisions, differences from baseline, and known failure modes.
> Load this when working on spider-specific issues.

## Design Philosophy

**Core hypothesis**: Complete map knowledge before gathering leads to better decisions than the baseline's interleaved explore-while-working approach.

**Execution**: Each agent is self-sufficient (gathers all 4 resources, assembles, delivers) but shares map discoveries with teammates.

---

## Key Differences from Baseline

| Aspect | Baseline | Spider |
|--------|----------|--------|
| **Exploration** | Interleaved - explores as needed while gathering | Full exploration first, then gather |
| **Map storage** | Per-agent only | Per-agent + SharedTeamState |
| **Depleted extractors** | Detected locally, not shared | Shared via `depleted_extractors` set |
| **Stuck detection** | Loop pattern matching (A→B→A→B) | Unique position count in window |
| **Extractor selection** | Nearest available | Distributed by agent_id to reduce collisions |
| **Collision handling** | Random escape | Wait 3 steps, then find alternate path |

---

## Architecture

```
MettaSpiderPolicy (MultiAgentPolicy)
    └── creates SpiderPolicyImpl per agent
        └── all share same SharedTeamState

SharedTeamState:
    - extractors: Dict[resource → position]  # First discovered
    - assembler, chest, charger positions
    - depleted_extractors: Set[position]     # Shared blacklist
    - tried_frontiers: Set[position]         # Failed exploration targets

SpiderState (per-agent):
    - position (row, col)
    - occupancy grid (200x200)
    - seen cells
    - local extractor list with cooldown/clipped/remaining
    - inventory
    - phase, target, path cache
```

---

## Phase Flow

```
Agent starts at EXPLORE
    ↓
EXPLORE until frontier is empty
    ↓
Switch to GATHER
    ↓
GATHER → ASSEMBLE → DELIVER → back to GATHER
    (RECHARGE can interrupt any phase)
```

### EXPLORE
- Uses frontier-based exploration
- Frontier = FREE cells adjacent to UNKNOWN cells
- When frontier empty → exploration complete → GATHER

### GATHER
- Calculate deficits (what we need for heart recipe)
- Find extractor for highest-deficit resource
- Navigate to extractor, use it, wait for resource
- If extractor depleted → mark in team state, find another

### ASSEMBLE
- Navigate to assembler
- Set vibe to `heart_a`
- Move adjacent to trigger assembly

### DELIVER
- Navigate to chest
- Set vibe to `heart_b`
- Move adjacent to deposit hearts

### RECHARGE
- Triggered when energy < 30
- Navigate to charger
- Wait until energy > 80
- Resume previous phase

---

## Known Failure Modes

### 1. Stuck at Depleted Extractor
**Symptom**: 76+ stuck events, agent oscillates near extractor
**Root cause**: Extractor shows cooldown but never gives resource
**Detection**: `cooldown_remaining >= 250` = effectively depleted
**Current fix**: Mark depleted in team state, share with all agents

### 2. Exploration Never Completes
**Symptom**: Agent stays in EXPLORE forever
**Root cause**: Frontier cells exist but are unreachable (behind walls)
**Detection**: Path to frontier fails repeatedly
**Current fix**: Mark as tried_frontier, skip it

### 3. Multiple Agents at Same Extractor
**Symptom**: Agents collide, waste time waiting
**Root cause**: All agents pick "nearest" = same one
**Current fix**: Use `agent_id % len(candidates)` to distribute

### 4. Oscillation Between Positions
**Symptom**: Agent bounces A→B→A→B forever
**Root cause**: Two equally-attractive targets, path switches each step
**Detection**: Unique positions in last 12 steps ≤ 4
**Current fix**: Clear history on escape, random move to break loop

### 5. Path Cache Invalidation
**Symptom**: Agent walks into wall or known obstacle
**Root cause**: Cached path becomes invalid after map update
**Detection**: Next step in path is now obstacle
**Current fix**: Invalidate cache when target changes or path blocked

---

## Trace Events Reference

The spider logs structured events to `/tmp/metta_spider.jsonl`:

| Event | Meaning |
|-------|---------|
| `init` | Agent initialized with recipe |
| `phase` | Phase transition (old, new, reason) |
| `discovered` | Found station/extractor |
| `shared` | Shared discovery with team |
| `gather_target` | Selected resource and extractor to gather |
| `extractor_use` | Started using extractor |
| `extractor_cooldown` | Waiting for cooldown |
| `extractor_unusable` | Found depleted/clipped extractor |
| `gathered` | Successfully received resource |
| `extractor_timeout` | Gave up waiting, marking depleted |
| `shared_depleted` | Marked extractor depleted in team state |
| `stuck` | Detected stuck state |
| `escape_stuck` | Attempting escape |
| `assemble` | Using assembler |
| `deliver` | Using chest |

### Healthy Run Signature
```
< 10 stuck events
gathered events for each resource type
phase transitions: explore → gather → assemble → deliver
```

### Broken Run Signature
```
50+ stuck events
Same gather_target repeating forever
extractor_timeout events
No gathered events
```

---

## Debugging Workflow

```bash
# Run with trace
cogames play -m training_facility -p class=metta_spider.agent.MettaSpiderPolicy --render none --steps 500

# Event summary
cat /tmp/metta_spider.jsonl | jq -r '.event' | sort | uniq -c | sort -rn

# Find stuck patterns
grep -E "stuck|escape" /tmp/metta_spider.jsonl | head -20

# Trace gather failures
cat /tmp/metta_spider.jsonl | jq -c 'select(.event == "gather_target" or .event == "extractor_timeout")' | tail -30

# Check phase transitions
cat /tmp/metta_spider.jsonl | jq -c 'select(.event == "phase")'

# Compare to baseline
cogames eval -m training_facility -p class=scripted_baseline -e 10 --steps 1000
cogames eval -m training_facility -p class=metta_spider.agent.MettaSpiderPolicy -e 10 --steps 1000
```

---

## Current Performance

On `training_facility` (max 2 hearts):

| Policy | Agents | Hearts | Notes |
|--------|--------|--------|-------|
| Baseline | 1 | ~1.0 | Consistent |
| Spider | 1 | ~0.2 | High variance, sometimes 2, usually 0 |

**Problem**: Spider can achieve optimal (2 hearts) but fails most runs due to getting stuck.

---

## Open Questions

1. **Is full exploration worth it?** Takes time, baseline interleaves and wins.
2. **Multi-agent coordination**: Current sharing helps but agents still collide.
3. **Extractor depletion**: Detection is fragile (cooldown=250 heuristic).
4. **Energy management**: RECHARGE threshold may be too conservative.
