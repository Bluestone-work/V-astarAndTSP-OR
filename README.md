Notice that we are writing the description and tutorials to our source code and other related materials, in order to foster future research in this robot collision avoidance line. They will be found in our repository in Github https://github.com/Bluestone-work/V_AstarAndTSP-OR, before publication.

# V-A* and TSP-OR: Multi-Waypoint Path Planning with Dynamic Obstacles

This repository contains the implementation of **TSP-OR (Traveling Salesman Problem with Obstacle avoidance using Reinforcement learning)**, a deep reinforcement learning approach for multi-waypoint path planning in dynamic environments with Voronoi diagram-based path skeleton generation.

## Visual Demonstrations

### Indoor Scenario - TSP-OR Navigation
<p align="center">
  <img src="gifs/Indoor_TSP-OR.gif" alt="TSP-OR in Indoor Environment" width="600"/>
  <br>
  <em>TSP-OR successfully navigating through a cluttered indoor environment with dynamic obstacles</em>
</p>

### Maze A Scenario - TSP-OR Navigation
<p align="center">
  <img src="gifs/MazeA_TSP-OR.gif" alt="TSP-OR in Maze A" width="600"/>
  <br>
  <em>TSP-OR demonstrating robust multi-waypoint navigation in complex maze layout</em>
</p>

## 📊 Abstract
Autonomous Mobile Robots (AMRs) deployed in intelligent warehouses and similar scenarios must navigate in large-scale, obstacle-dense environments where only the permanent structural layout is known a priori. In such settings, the environment is cluttered with unknown temporary static obstacles and unpredictable dynamic entities, creating a high-density, dynamically evolving scenario. However, existing hybrid planners that rely on pre-calculated geometric paths are prone to frequent global re-planning when unexpected obstacles invalidate the assumed path structure. Meanwhile, standard end-to-end Deep Reinforcement Learning (DRL) methods have limitations of slow convergence and poor long-horizon guidance in large maps. To overcome these limitations, we propose a novel global-local collaborative navigation model for providing long-horizon topological guidance and avoiding fast-changing obstacles. First, the global planner constructs a Voronoi diagram from the a priori known structural map. Second, a Voronoi–A* search is performed to extract sparse key path points that serve as flexible sub-goals. Third, for local control, we introduce the Temporal-State Processed and Optimized Replay DQN (TSP-OR DQN), which integrates temporal feature modeling, motion prediction, and a prioritized replay mechanism to handle partial observability. Finally, extensive evaluation across four environments and four obstacle density levels demonstrates that our framework consistently outperforms state-of-the-art baselines. The proposed method maintains a success rate of 60\%--90\% in highly cluttered indoor and maze layouts and exceeds 90\% in open scenarios. Furthermore, the learned policy exhibits strong generalization to unseen maps and novel obstacle configurations.

## Project Structure

```
V-A*_and_TSP-OR/
├── agents.py                    # TSP-OR DQN agent implementation
├── environment.py               # Grid world environment with dynamic 
├── train.py                     # Training script
├── evaluate.py                 # Comprehensive testing script
├── create_voronoi.ipynb        # Voronoi diagram generation and path skeleton 
├── convert_img_to_grid.ipynb  # Image to grid map conversion
├── generate_test_environments.py # Generate fixed test scenarios
├── gym_gridworld_env.py        # OpenAI Gym compatible environment wrapper
├── data/                        # Test results for different algorithms and scenarios
│   ├── *_Indoor_Level*.txt     # Indoor scenario results
│   ├── *_MazeA_Level*.txt      # Maze A scenario results
│   ├── *_MazeB_Level*.txt      # Maze B scenario results
│   └── *_Outdoor_Level*.txt    # Outdoor scenario results
├── models/                      # Trained model checkpoints
├── maps/                        # Prior map
└── gifs/                        # Visualization GIFs
```

## Usage

### 1. Generate Voronoi Path Skeleton

Use the Jupyter notebook to create Voronoi diagrams and extract path skeletons:

```bash
jupyter notebook create_voronoi.ipynb
```

This will:
- Generate Voronoi diagrams from obstacle maps
- Extract and simplify path skeletons

### 2. Train the Agent

Train the TSP-OR agent with curriculum learning:

```bash
# Basic training
python train.py

# With custom parameters
python train.py --episodes 1000 --map grid_map2.txt
```

### 3. Generate Test Environments

Create fixed test scenarios for reproducible evaluation:

```bash
# Generate test environments with different obstacle densities
python generate_test_environments.py --map grid_map.txt --output_dir test_environments_g

# Generate for specific obstacle counts
python generate_test_environments.py --map grid_map2.txt --obs_counts 10,20,30,40,50 --num_envs 100
```

### 4. Test the Agent

Run comprehensive evaluations using the interactive testing script:

```bash
python evaluate.py
```
View Training Progress (if using TensorBoard)

tensorboard --logdir runs/

The script provides three testing modes:

**Mode 1: Random Environment Testing**
- Tests agent performance across multiple randomly generated scenarios
- Supports 4 difficulty levels with varying obstacle densities
- Generates detailed performance metrics and statistical analysis

**Mode 2: Fixed Environment Testing**
- Evaluates on pre-generated reproducible test scenarios
- Ensures consistent benchmarking across different runs
- Automatically saves GIF visualizations every 100 episodes

**Mode 3: Environment Visualization**
- Visualize fixed environment configurations
- Randomly samples and displays environment layouts
- Useful for understanding test scenario complexity
