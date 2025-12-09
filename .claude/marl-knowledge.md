# MARL Knowledge Base

Load this when working on: policy architecture, training pipelines, reward shaping, or debugging learned behavior.

---

## MARL Challenge Hierarchy

| Level | Approach | When to Use |
|-------|----------|-------------|
| 1 | Independent learning | Simple tasks, minimal coordination needed |
| 2 | CTDE (QMIX, MAPPO) | Need coordination but decentralized execution |
| 3 | Communication learning | Env supports message channels |
| 4 | Partner modeling | Heterogeneous teammates or adversaries |

**Current approach**: Level 2 with explicit role assignment (simpler than learned coordination).

---

## Credit Assignment

| Strategy | Pros | Cons |
|----------|------|------|
| Team reward only | Simple | Hard to attribute; slow learning |
| Role-based shaping | Clear attribution | Manual reward design |
| Counterfactual (COMA) | Principled | Computationally expensive |
| Difference rewards | Theoretically grounded | Requires counterfactual simulation |

**For this project**: Start with role-based shaping:
- +0.1 picking up resource
- +0.2 depositing to chest
- +0.5 assembling heart
- +1.0 delivering heart

---

## Coordination Mechanisms

**Explicit (current)**:
- Fixed role assignment via `ROLE_BY_AGENT_ID`
- Phase-based state machine
- Shared team state (chest location, extractor positions)

**Emergent (future)**:
- Agents learn specialization through training
- Risk: suboptimal equilibria
- Mitigation: population-based training, diversity bonuses

---

## Partial Observability (11x11 window)

| Strategy | Implementation |
|----------|----------------|
| Memory (LSTM/GRU) | Add recurrent layer to policy network |
| Belief state | Maintain map of seen/unseen (`state.occupancy`, `state.seen`) |
| Exploration bonus | Reward visiting unseen cells |

**Current**: Explicit map building (handcrafted belief state).

---

## Architecture Options

**Token observations → policy:**

1. **Set encoder** (DeepSets, Transformer)
   - Handles variable token count naturally
   - May be overkill for this domain

2. **Grid reconstruction + CNN**
   - Preserve spatial structure
   - Sparse observations waste compute

3. **Attention over tokens**
   - Query "nearest extractor" directly
   - Modern MARL standard

**Multi-agent:**

| Approach | Pro | Con |
|----------|-----|-----|
| Parameter sharing + agent ID | More data per param | Harder to specialize |
| Separate networks | Natural specialization | 4x parameters |

---

## Training Pipeline

```text
Phase 1: Scripted baseline ← WE ARE HERE
    └─ Understand environment, collect traces

Phase 2: Behavioral cloning
    └─ Imitate scripted policy, avoid cold-start

Phase 3: RL fine-tuning (MAPPO)
    └─ KL penalty to stay near scripted prior
    └─ Curriculum: lonely_heart → easy_mode → harder

Phase 4: Self-play / population training
    └─ Prevent overfitting to specific strategies
```

---

## Debugging Learned Policies

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| All agents do same thing | No agent ID feature | Add agent ID input |
| One agent dominates | Credit imbalance | Per-role reward shaping |
| Constant collisions | No avoidance learned | Collision penalty |
| Training unstable | Non-stationarity | Freeze others, CTDE |
| Learns then forgets | Catastrophic forgetting | Replay buffer, EWC |
| Works easy, fails hard | Overfitting | Curriculum, randomization |

---

## Environment-Specific Notes

1. **Vibe = discrete mode switching** (like options framework)
2. **Energy is a hard constraint** – budget it, never hit 0
3. **Chest = implicit communication** – coordinate via deposits/withdrawals
4. **Assembly adjacency** – spatial coordination constraint
5. **Map variability** – need diverse training maps for generalization

---

## Key Papers (for deeper dives)

- **QMIX**: Value decomposition for cooperative MARL
- **MAPPO**: Multi-agent PPO, surprisingly strong baseline
- **COMA**: Counterfactual credit assignment
- **MAVEN**: Committed exploration for coordination
