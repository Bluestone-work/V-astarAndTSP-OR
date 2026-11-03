# Replay and learner alignment

Current implementation: `jas0644-aligned-v2`; see
[the full paper audit](manuscript_code_alignment.md).

The proposed replay remains exactly the submitted Eqs.18-19 protocol:

```
p = .25*log(1+max(reward,0)) + .75*log(1+abs(td_error)) + 1e-5
P(i) = p_i / sum(p)
y = reward + gamma*(1-done)*max(Q_target(next_state))
loss = mean((y-Q_online(state,action))**2)
```

One FIFO, replacement sampling, formula-based insertion, pre-gradient sampled
priority refresh, no auxiliary high-reward buffer, no priority exponent and no
IS weights in the proposed arm. Nonpositive rewards have no reward bonus;
collision priority is not guaranteed to exceed other events.

The separate standard PER arm uses `(|delta|+1e-5)^.6` and normalized
`(N P)^(-beta)` loss weights. Beta increases from .4 to 1 over 200000 optimizer
steps. The uniform arm remains unweighted. Online sampling-event counters use
actual environment termination/waypoint events, not reward thresholds.

GridWorld now returns the complete composite reward including SMR. The agent
stores that reward without charging SMR twice. A legacy external environment
without `reward_includes_smoothness` can still use the documented agent fallback.
The policy uses plain epsilon-greedy action selection without Q-value smoothing.
Targets initialize as exact copies, soft-update with tau=.01 after every
optimizer step, use a one-step ordinary-DQN maximum, and clip gradient norm to 1.
Epsilon decays once per environment interaction with a .01 floor.

Historical checkpoints/results do not validate this state/reward/controller
revision. The old public source is preserved as `legacy/agents_original.py`.
New checkpoints include code protocol, environment settings and source hashes;
loading incompatible historical checkpoints requires an explicit transfer flag.
