# Local View Navigation Based on Global-Local Collaborative Framework in High-Density Scenes

TSP-OR means **Temporal-State Processing and Optimized Replay DQN**.
The framework combines Voronoi-A* guidance, causal four-frame local observations,
weighted obstacle-motion predictions and dual-criterion replay.

The current implementation protocol is `jas0644-aligned-v2`. Read the
[paper/code alignment audit](docs/manuscript_code_alignment.md) for the exact
formula corrections, remaining manuscript edits, historical-result boundaries,
and reviewer requests that still require new experiments. The author's response
letter has not been supplied, so its commitments have not yet been verified.

## Implementation

- `global_planner.py`: physical cell dimensions, finite-segment clearance,
  explicit robot footprint/margin, Voronoi graph pruning, safe grid fallback,
  rounded-route validation and sparse key points.
- `environment.py`: one observation per control timestep, six-component rewards,
  explicit termination reasons, local sensing and real event replanning.
- `obstacle_prediction.py`: shared causal prediction, CV and explicitly defined
  historical occupancy aggregation baselines.
- `agents.py`, `manuscript_replay.py`: Eq.18-19 replay, uniform replay and standard
  proportional PER with IS loss weights; one-step DQN targets.
- `models.py`: BiLSTM/UniLSTM × Dueling/Plain factorial networks.
- `navigation_protocol.py`: version and the full 9×9 transition penalty matrix.
- `motion_control.py`: optional bounded unicycle feedback tracker.
- `configs/revision_experiment.json`: current experiment settings.
- `create_voronoi.ipynb`: visualization using the same production planner.

## Training and verification

Use an environment with NumPy, SciPy, PyTorch, pytest, TensorBoard, tqdm and the
plotting dependencies. On this workstation the Python interpreter is
`/home/wj/miniconda3/envs/navq/bin/python`.
Interactive windows additionally need pygame; headless GIF rendering does not.

```bash
python -m pytest tests -q
python train.py --map-file grid_map2.txt --seed 0

# Short verification only; not a paper result
python train.py --seed 91 --episodes 1 --max-steps 3 --device cpu \
  --save-path experiments/results/aligned_v2/smoke.pth

# Independent seeds and matched replay baselines
python -m experiments.replay_comparison --seeds 0 1 2 3 4
python -m experiments.architecture_ablation --seeds 0 1 2 3 4
python -m experiments.proposed_prediction
```

Training defaults to 1000 episodes, a 2000-step cap, actual event replanning,
per-step exploration decay and a final checkpoint. Each run saves the seed,
actual environment interactions, full source snapshot/hashes, model/environment
settings, sampled-event counters and per-episode reward/loss/TD diagnostics.
It does not claim that every run uses exactly two million interactions or that
within-run reward variation measures independent-training reproducibility.
Resume/reuse requires matching source hashes and experiment budgets; repeated
seeds and relabelling short training runs as full runs are rejected.

The training CLI also accepts `--prediction-mode`, `--avoidance-reward-form`,
`--reward-scales` and `--smoothing-factor` for declared module/reward variants.
All formal variants must be independently trained on a frozen source version.

## Evaluation

```bash
python -m experiments.proposed_evaluation \
  --map-file grid_map2.txt --training-map grid_map2.txt --seed 0 \
  --checkpoint experiments/results/aligned_v2/grid_map2_manuscript_seed0.pth \
  --scenarios 300 --levels 1 2 3 4 \
  --output-dir experiments/results/aligned_v2/evaluation_seed0
```

The default controller is `full`. Actual calls, costs, fallbacks, collisions,
timeouts and recovery events are recorded per scenario. Geometry is configurable
with `--robot-radius`, `--cell-size` and `--extra-margin` in physical map units.
Task throughput is successful completions divided by total episode wall time;
policy latency, control-cycle latency and whole-task time are separate metrics.
Grid action counts are not physical navigation seconds.

`experiments.global_local_ablation` and `experiments.closed_loop_noise` use the
same required arguments and real evaluator. They add `--controllers` and
`--perturbations`, respectively. Fixed-checkpoint controller interventions must
be described separately from independently retrained architecture/replay variants.
`experiments.prediction_metrics` forwards to the shared three-step diagnostic.

For an additional continuous-motion simulation:

```bash
python -m experiments.kinematic_closed_loop \
  --checkpoint experiments/results/aligned_v2/grid_map2_manuscript_seed0.pth \
  --map-file grid_map2.txt --robot-radius .2
```

This simulator applies bounded linear/angular speed and acceleration, feedback,
controlled stopping and swept-clearance checks. It is a simulation adapter with
sequential dynamic grid obstacles, not physical-robot validation. Localization
noise, sensing noise and actuation delay are optional test parameters.

## Evidence and historical versions

New checkpoints record `jas0644-aligned-v2` and their training source hashes.
Old checkpoints or changed training sources are rejected by default. The
explicit historical-load flag is only for separately labelled transfer tests.
No old checkpoint or result is silently converted into a new-version result.

`experiments/results/proposed_suite/`, its queue scripts and its reports refer
to the prior frozen run. The original public agent is preserved under `legacy/`;
the complete pre-alignment source/notebooks are archived under
`experiments/results/alignment_20261021_before/`.

Existing `data/` tables and GIFs are historical. Current code changes and smoke
runs do not establish the PDF's numerical performance or a universal collision
avoidance guarantee. M-DQN, IDDQN and RecurrentPPO historical sources/checkpoints
are absent; their multi-seed results cannot be regenerated from text tables.
