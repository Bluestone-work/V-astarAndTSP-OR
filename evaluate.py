import numpy as np
import os
import torch
import time
import matplotlib.pyplot as plt
from environment import GridWorld
from agents import RainbowDQN
from tqdm import tqdm
try:
    from statistical_tests import (
        calculate_confidence_interval,
        independent_t_test,
        calculate_effect_size
    )
    _STATS_AVAILABLE = True
except ImportError:
    print("[Warning] statistical_tests.py not found, statistical comparison features will be unavailable")
    _STATS_AVAILABLE = False

try:
    import pynvml
    _GPU_AVAILABLE = True
    pynvml.nvmlInit()
    _GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)  # Default GPU 0
except Exception as e:
    print("[Monitor Info] No available pynvml or GPU detected, GPU monitoring will be disabled.")
    _GPU_AVAILABLE = False


def get_device(prefer_gpu=1):
    """
    Smart device selection
    
    Args:
        prefer_gpu: Preferred GPU number (0 or 1), default is 1 (second card)
    
    Returns:
        torch.device: Selected device
    """
    if not torch.cuda.is_available():
        print(f"[Device] CUDA not available, using CPU")
        return torch.device("cpu")
    
    gpu_count = torch.cuda.device_count()
    print(f"[Device] Detected {gpu_count} GPU(s)")
    
    # If the specified GPU exists, use it
    if prefer_gpu < gpu_count:
        device = torch.device(f"cuda:{prefer_gpu}")
        
        # Check GPU memory usage
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(prefer_gpu)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used_gb = mem_info.used / 1024**3
            total_gb = mem_info.total / 1024**3
            free_gb = mem_info.free / 1024**3
            
            print(f"[Device] GPU {prefer_gpu} memory: {used_gb:.2f}GB / {total_gb:.2f}GB (remaining {free_gb:.2f}GB)")
            
            if free_gb < 2.0:
                print(f"[Warning] GPU {prefer_gpu} memory less than 2GB, may run out of memory")
        except:
            pass
        
        print(f"[Device] Using GPU {prefer_gpu} (cuda:{prefer_gpu})")
        return device
    else:
        # If specified GPU doesn't exist, use the first one
        device = torch.device("cuda:0")
        print(f"[Device] GPU {prefer_gpu} not found, using GPU 0 (cuda:0)")
        return device
    _GPU_HANDLE = None

def calculate_path_metrics(path):
    if len(path) < 2:
        return 0.0, 1.0
    
    path_length = 0
    for i in range(1, len(path)):
        dx = path[i][0] - path[i-1][0]
        dy = path[i][1] - path[i-1][1]
        path_length += np.sqrt(dx**2 + dy**2)
    
    if len(path) < 3:
        return path_length, 1.0
    
    angles = []
    for i in range(1, len(path)-1):
        v1 = np.array(path[i]) - np.array(path[i-1])
        v2 = np.array(path[i+1]) - np.array(path[i])
        
        if np.all(v1 == 0) or np.all(v2 == 0):
            continue
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        angles.append(angle)
    
    smoothness = 1.0 - (np.mean(angles) / np.pi if angles else 0)
    return path_length, smoothness

def run_rl_algorithm_without_viz(env, agent, max_steps=1000, resample_obstacles=False):
    """Run RL algorithm (including our method, Paper 1, Paper 2) without visualization
    
    Args:
        env: Environment instance
        agent: Agent instance (RainbowDQN, PaperDQNAgent, Paper2DQNAgent)
        max_steps: Maximum steps
        resample_obstacles: Whether to resample obstacles (True=random env, False=fixed env)
    
    Returns:
        (success, path, steps, metrics): Success status, path, steps, detailed metrics dict
        metrics contains:
            - local_decision_times: Decision time list per step (ms)
            - replan_count: Number of replans
            - dynamic_obstacle_encounters: Dynamic obstacle encounters
            - reaction_steps: Reaction steps for each encounter
    """
    # Ensure model is in evaluation mode
    original_epsilon = agent.epsilon
    agent.epsilon = 0
    if hasattr(agent, 'q_network'):
        agent.q_network.eval()
    
    state = env.reset(resample_obstacles=resample_obstacles)
    done = False
    path = [tuple(env.agent_pos)]
    steps = 0
    success = False
    
    # New metrics collection
    metrics = {
        'local_decision_times': [],  # Decision time per step (ms)
        'replan_count': 0,  # Number of replans
        'dynamic_obstacle_encounters': 0,  # Dynamic obstacle encounter count
        'reaction_steps': [],  # Steps needed for each response
    }
    
    last_action = None  # Used to detect replanning
    in_reaction = False  # Whether reacting to dynamic obstacles
    reaction_start_step = 0  # Reaction start step
    
    while not done and steps < max_steps:
        try:
            # Measure local decision time
            decision_start = time.time()
            
            # Select action - compatible with different method names
            if hasattr(agent, 'select_action'):
                action = agent.select_action(state)
            elif hasattr(agent, 'act'):
                action = agent.act(state)
            else:
                raise AttributeError(f"Agent {type(agent).__name__} does not have 'select_action' or 'act' method")
            
            decision_time = (time.time() - decision_start) * 1000  # Convert to milliseconds
            metrics['local_decision_times'].append(decision_time)
            
            # Detect replanning (action change)
            if last_action is not None:
                # Define action vectors (9 actions: 8 directions + stop)
                action_vectors = [
                    (-1, 0), (1, 0), (0, -1), (0, 1),  # up down left right
                    (-1, -1), (-1, 1), (1, -1), (1, 1),  # diagonal
                    (0, 0)  # stop
                ]
                
                if last_action < len(action_vectors) and action < len(action_vectors):
                    last_vec = np.array(action_vectors[last_action])
                    curr_vec = np.array(action_vectors[action])
                    
                    # If action direction changes more than 90 degrees, consider as replanning
                    if np.linalg.norm(last_vec) > 0 and np.linalg.norm(curr_vec) > 0:
                        dot_product = np.dot(last_vec, curr_vec)
                        norm_product = np.linalg.norm(last_vec) * np.linalg.norm(curr_vec)
                        if dot_product / norm_product < 0:  # angle > 90 degrees
                            metrics['replan_count'] += 1
            
            last_action = action
            
            # Detect if close to dynamic obstacles (distance <= 2)
            near_dynamic_obstacle = False
            if hasattr(env, 'dynamic_obstacles'):
                for obs in env.dynamic_obstacles:
                    obs_pos = np.array(obs['position'])
                    agent_pos = np.array(env.agent_pos)
                    distance = np.linalg.norm(agent_pos - obs_pos)
                    if distance <= 2.0:
                        near_dynamic_obstacle = True
                        if not in_reaction:
                            # Start new reaction
                            in_reaction = True
                            reaction_start_step = steps
                            metrics['dynamic_obstacle_encounters'] += 1
                        break
            
            # If was reacting before, but now far from dynamic obstacles
            if in_reaction and not near_dynamic_obstacle:
                reaction_duration = steps - reaction_start_step
                metrics['reaction_steps'].append(reaction_duration)
                in_reaction = False
                
            next_state, reward, done = env.step(action)
            
            state = next_state
            path.append(tuple(env.agent_pos))
            steps += 1
            
            # Check if goal reached (using distance, tolerance 0.5)
            if np.linalg.norm(np.array(env.agent_pos) - np.array(env.goal)) < 0.5:
                success = True
                done = True
                
        except Exception as e:
            print(f"Error executing RL algorithm action: {e}")
            import traceback
            traceback.print_exc()
            agent.epsilon = original_epsilon  # Restore original epsilon
            return False, path, steps, metrics
    
    # If still reacting at end, record final reaction duration
    if in_reaction:
        reaction_duration = steps - reaction_start_step
        metrics['reaction_steps'].append(reaction_duration)
    
    agent.epsilon = original_epsilon  # Restore original epsilon
    return success, path, steps, metrics

def run_rl_algorithm_with_gif(env, agent, max_steps=1000, gif_path="test.gif", title_prefix="Test"):
    """Run RL algorithm and record GIF animation
    
    Args:
        env: Environment
        agent: Agent
        max_steps: Maximum steps
        gif_path: GIF save path
        title_prefix: Title prefix
    
    Returns:
        success: Success status
        path: Path
        steps: Steps
    """
    import imageio
    from visualization import _render_env_frame
    
    # Save original epsilon
    original_epsilon = agent.epsilon if hasattr(agent, 'epsilon') else 0.0
    if hasattr(agent, 'epsilon'):
        agent.epsilon = 0.0  # No exploration during testing
    
    state = env.reset(resample_obstacles=False)
    done = False
    path = [tuple(env.agent_pos)]
    steps = 0
    success = False
    total_reward = 0
    frames = []
    
    # Initialize dynamic obstacle path recording
    obstacle_paths = {i: [tuple(obs['position'])] for i, obs in enumerate(env.dynamic_obstacles)}
    
    # Record initial frame
    frames.append(_render_env_frame(
        env, path,
        title=f"{title_prefix} | Step 0 | Reward: 0.00",
        obstacle_paths=list(obstacle_paths.values())
    ))
    
    while not done and steps < max_steps:
        try:
            # Select action
            if hasattr(agent, 'select_action'):
                action = agent.select_action(state)
            elif hasattr(agent, 'act'):
                action = agent.act(state)
            else:
                raise AttributeError(f"Agent {type(agent).__name__} does not have 'select_action' or 'act' method")
            
            next_state, reward, done = env.step(action)
            
            # Update dynamic obstacle paths
            for i, obs in enumerate(env.dynamic_obstacles):
                obstacle_paths[i].append(tuple(obs['position']))
            
            state = next_state
            total_reward += reward
            path.append(tuple(env.agent_pos))
            steps += 1
            
            # Record frame
            frames.append(_render_env_frame(
                env, path,
                title=f"{title_prefix} | Step {steps} | Reward: {total_reward:.2f}",
                obstacle_paths=list(obstacle_paths.values())
            ))
            
            # Check if goal reached
            if np.linalg.norm(np.array(env.agent_pos) - np.array(env.goal)) < 0.5:
                success = True
                done = True
                # Add success frame
                frames.append(_render_env_frame(
                    env, path,
                    title=f"{title_prefix} | Success! Total Reward: {total_reward:.2f}",
                    obstacle_paths=list(obstacle_paths.values())
                ))
                
        except Exception as e:
            print(f"Error executing RL algorithm action: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # Restore original epsilon
    if hasattr(agent, 'epsilon'):
        agent.epsilon = original_epsilon
    
    # Save GIF
    if frames:
        try:
            import imageio
            imageio.mimsave(gif_path, frames, duration=0.1)
        except Exception as e:
            print(f"Failed to save GIF: {e}")
    
    return success, path, steps

def visualize_sample_fixed_envs(config_file, num_samples=10, save_dir="visualizations/fixed_envs"):
    """Randomly sample from fixed environment configurations for visualization
    
    Args:
        config_file: Path to environment configuration file
        num_samples: Number of samples to draw for visualization
        save_dir: Directory to save visualization images
    """
    print(f"\n{'='*60}")
    print(f"Visualizing fixed environment configurations")
    print(f"{'='*60}")
    
    # Load configurations
    try:
        configs = GridWorld.load_configs_from_file(config_file)
        print(f"Loaded {len(configs)} fixed environment configurations")
    except Exception as e:
        print(f"Failed to load configuration file: {e}")
        return
    
    num_samples = min(num_samples, len(configs))
    sample_indices = np.random.choice(len(configs), num_samples, replace=False)
    print(f"Randomly selected {num_samples} environments for visualization")
    
    os.makedirs(save_dir, exist_ok=True)

    for idx, config_idx in enumerate(sample_indices):
        config = configs[config_idx]
        
        print(f"\n--- Environment #{config_idx+1} ---")
  
        obstacle_mode = config.get('obstacle_mode', 'point')
        block_size_range = config.get('block_size_range', (2, 5))
        
        env = GridWorld(
            grid_map=config.get('base_grid_map'),
            start=config['start'],
            goal=config['goal'],
            goal_list=config.get('goal_list', []),
            obs=0,
            dobs=0,
            obstacle_mode=obstacle_mode,
            block_size_range=block_size_range
        )
        env.load_obstacle_config(config)
        env.reset(resample_obstacles=False)
        
        start = config['start']
        goal = config['goal']
        static_obs = np.sum(config['base_grid_map'] == 1)
        dynamic_obs = len(config['dynamic_obstacles'])
        
        print(f"  Start: {start}")
        print(f"  Goal: {goal}")
        print(f"  Obstacle Mode: {obstacle_mode}")
        if obstacle_mode == 'block':
            print(f"  Block Size Range: {block_size_range}")
        print(f"  Static Obstacles: {static_obs}")
        print(f"  Dynamic Obstacles: {dynamic_obs}")
        
        fig, ax = plt.subplots(figsize=(10, 10))

        grid_map = config['base_grid_map']
        ax.imshow(grid_map, cmap='binary', origin='upper')

        for obs_data in config['dynamic_obstacles']:
            obs_pos = obs_data['position']
            circle = plt.Circle((obs_pos[1], obs_pos[0]), 0.3, color='red', alpha=0.6, zorder=5)
            ax.add_patch(circle)
            velocity = obs_data.get('velocity', [0, 0])
            if velocity[0] != 0 or velocity[1] != 0:
                ax.arrow(obs_pos[1], obs_pos[0], velocity[1]*2, velocity[0]*2, 
                        head_width=0.3, head_length=0.3, fc='darkred', ec='darkred', alpha=0.7, zorder=6)
        ax.plot(start[1], start[0], 'go', markersize=15, label='Start', zorder=10)
        ax.plot(goal[1], goal[0], 'bs', markersize=15, label='Goal', zorder=10)
        ax.set_title(f'Fixed Environment #{config_idx+1}\n'
                    f'Static Obs: {static_obs}, Dynamic Obs: {dynamic_obs}', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Y', fontsize=12)
        ax.set_ylabel('X', fontsize=12)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, grid_map.shape[1]-0.5)
        ax.set_ylim(grid_map.shape[0]-0.5, -0.5)
        save_path = os.path.join(save_dir, f'env_{config_idx+1:04d}.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved to: {save_path}")
    
    print(f"\n{'='*60}")
    print(f"Visualization completed! Generated {num_samples} images")
    print(f"Save directory: {save_dir}")
    print(f"{'='*60}")


def batch_test_rl_on_fixed_envs(grid_map, nodes, config_file, model_path, 
                                 agent_class, agent_name, agent_params=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load fixed environment configurations
    print(f"\n{'='*60}")
    print(f"Testing from fixed configurations {agent_name}")
    print(f"{'='*60}")
    
    try:
        configs = GridWorld.load_configs_from_file(config_file)
        print(f"✓ Loaded {len(configs)} fixed environment configurations")
    except Exception as e:
        print(f"✗ Failed to load configuration file: {e}")
        return None
    
    # 创建临时Environment用于初始化agent
    temp_env = GridWorld(
        grid_map=grid_map,
        start=nodes[0],
        goal=nodes[-1],
        goal_list=nodes,
        obs=0,  # 会从配置加载
        dobs=0
    )
    
    # 设置默认参数
    default_params = {
        'env': temp_env,
        'gamma': 0.99,
        'epsilon': 0.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995,
        'learning_rate': 5e-3,
        'batch_size': 64,
        'memory_size': 200000,
        'device': device
    }
    
    if agent_params:
        default_params.update(agent_params)
    
    # 创建agent并加载Model
    agent = agent_class(**default_params)
    
    if not os.path.exists(model_path):
        print(f"✗ Model file does not exist: {model_path}")
        return None
    
    agent.load_model(model_path)
    print(f"✓ Model loaded successfully: {model_path}")
    
    # Test results (extended metrics)
    results = {
        'success': [],
        'time': [],
        'path_length': [],
        'smoothness': [],
        'steps': [],
        # New performance metrics
        'local_decision_times': [],  # All decision times for each test
        'avg_local_decision_time': [],  # Average decision time for each test
        'replan_count': [],  # Number of replans
        'replan_frequency': [],  # Replan frequency (times/step)
        'dynamic_obstacle_encounters': [],  # Dynamic obstacle encounter count
        'avg_reaction_steps': [],  # Average reaction steps
        'control_latency': [],  # End-to-end control latency (ms)
    }
    
    print(f"\nStarting test {len(configs)} environments...")
    
    # Create GIF save directory
    gif_dir = f"./gif/fixed_env_{agent_name.replace(' ', '_')}"
    os.makedirs(gif_dir, exist_ok=True)
    
    progress_bar = tqdm(enumerate(configs), total=len(configs), desc=f"{agent_name}Test progress", unit="env")
    
    for i, config in progress_bar:
        # Create environment and load configuration
        env = GridWorld(
            grid_map=grid_map,
            start=config['start'],
            goal=config['goal'],
            goal_list=config.get('goal_list', []),
            obs=0,
            dobs=0
        )
        
        env.load_obstacle_config(config)
        # Important: must call reset to apply loaded config, but don't resample obstacles
        env.reset(resample_obstacles=False)
        
        # Determine if frame recording needed (every 100 times)
        record_frames = ((i + 1) % 100 == 0)
        
        # Run test
        start_time = time.time()
        if record_frames:
            # Test with GIF recording (not collecting detailed metrics)
            success, path, steps = run_rl_algorithm_with_gif(
                env, agent, max_steps=1000, 
                gif_path=os.path.join(gif_dir, f"env_{i+1:04d}.gif"),
                title_prefix=f"Fixed Env #{i+1}"
            )
            tqdm.write(f"  Saved GIF: {gif_dir}/env_{i+1:04d}.gif")
            # Fill default metrics for GIF test
            metrics = {
                'local_decision_times': [],
                'replan_count': 0,
                'dynamic_obstacle_encounters': 0,
                'reaction_steps': []
            }
        else:
            # Normal test (collecting detailed metrics)
            success, path, steps, metrics = run_rl_algorithm_without_viz(env, agent, max_steps=1000)
        
        run_time = time.time() - start_time
        
        results['success'].append(success)
        results['time'].append(run_time)
        results['steps'].append(steps)
        
        # Collect new metrics
        if metrics['local_decision_times']:
            results['local_decision_times'].append(metrics['local_decision_times'])
            results['avg_local_decision_time'].append(np.mean(metrics['local_decision_times']))
            # Control latency = decision time + env step time
            avg_decision_time = np.mean(metrics['local_decision_times'])
            step_time = (run_time * 1000) / steps if steps > 0 else 0  # Total time / steps
            results['control_latency'].append(step_time)  # End-to-end latency
        else:
            results['local_decision_times'].append([])
            results['avg_local_decision_time'].append(0)
            results['control_latency'].append(0)
        
        results['replan_count'].append(metrics['replan_count'])
        results['replan_frequency'].append(metrics['replan_count'] / steps if steps > 0 else 0)
        results['dynamic_obstacle_encounters'].append(metrics['dynamic_obstacle_encounters'])
        
        if metrics['reaction_steps']:
            results['avg_reaction_steps'].append(np.mean(metrics['reaction_steps']))
        else:
            results['avg_reaction_steps'].append(0)
        
        if success and len(path) > 1:
            path_length, smoothness = calculate_path_metrics(path)
            results['path_length'].append(path_length)
            results['smoothness'].append(smoothness)
        else:
            results['path_length'].append(0)
            results['smoothness'].append(0)
        
        # Update progress bar showing current success rate
        current_success_rate = sum(results['success']) / len(results['success']) * 100
        progress_bar.set_postfix({'Success rate': f'{current_success_rate:.1f}%'})
    
    # Calculate statistics
    success_count = sum(results['success'])
    success_rate = success_count / len(configs) * 100
    
    # Calculate Wilson 95% confidence interval (for success rate)
    from scipy import stats
    n = len(configs)
    p_hat = success_count / n
    z = 1.96  # 95% confidence level
    wilson_center = (p_hat + z**2/(2*n)) / (1 + z**2/n)
    wilson_width = z * np.sqrt((p_hat*(1-p_hat) + z**2/(4*n)) / n) / (1 + z**2/n)
    wilson_ci_lower = wilson_center - wilson_width
    wilson_ci_upper = wilson_center + wilson_width
    
    print(f"\n{'='*60}")
    print(f"{agent_name}Test Results")
    print(f"{'='*60}")
    print(f"Success rate: {success_rate:.2f}% ({success_count}/{len(configs)})")
    print(f"  Wilson 95% CI: [{wilson_ci_lower*100:.2f}%, {wilson_ci_upper*100:.2f}%]")
    
    if success_count > 0:
        successful_indices = [i for i, s in enumerate(results['success']) if s]
        
        # Extract success cases data for confidence interval calculation
        success_times = np.array([results['time'][i] for i in successful_indices])
        success_steps = np.array([results['steps'][i] for i in successful_indices])
        # Only count valid path metrics in success cases (exclude 0 padding after failure)
        success_path_lengths = np.array([results['path_length'][i] for i in successful_indices if results['path_length'][i] > 0])
        success_smoothness = np.array([results['smoothness'][i] for i in successful_indices if results['smoothness'][i] > 0])
        success_local_dt = np.array([results['avg_local_decision_time'][i] for i in successful_indices])
        success_replan_freq = np.array([results['replan_frequency'][i] for i in successful_indices])
        success_control_latency = np.array([results['control_latency'][i] for i in successful_indices])
        
        # Calculate mean and 95% confidence interval
        if _STATS_AVAILABLE:
            from statistical_tests import calculate_confidence_interval
            
            time_mean, time_ci_low, time_ci_high = calculate_confidence_interval(success_times)
            steps_mean, steps_ci_low, steps_ci_high = calculate_confidence_interval(success_steps)
            # Path length and smoothness: only calculate CI for valid values
            if len(success_path_lengths) > 0:
                path_mean, path_ci_low, path_ci_high = calculate_confidence_interval(success_path_lengths)
            else:
                path_mean = path_ci_low = path_ci_high = 0.0
            if len(success_smoothness) > 0:
                smooth_mean, smooth_ci_low, smooth_ci_high = calculate_confidence_interval(success_smoothness)
            else:
                smooth_mean = smooth_ci_low = smooth_ci_high = 0.0
            local_dt_mean, local_dt_ci_low, local_dt_ci_high = calculate_confidence_interval(success_local_dt)
            replan_mean, replan_ci_low, replan_ci_high = calculate_confidence_interval(success_replan_freq)
            latency_mean, latency_ci_low, latency_ci_high = calculate_confidence_interval(success_control_latency)
        else:
            # If statistical_tests unavailable, use simple mean ± std
            time_mean = np.mean(success_times)
            steps_mean = np.mean(success_steps)
            path_mean = np.mean(success_path_lengths) if len(success_path_lengths) > 0 else 0.0
            smooth_mean = np.mean(success_smoothness) if len(success_smoothness) > 0 else 0.0
            local_dt_mean = np.mean(success_local_dt)
            replan_mean = np.mean(success_replan_freq)
            latency_mean = np.mean(success_control_latency)
        
        avg_reaction = np.mean([results['avg_reaction_steps'][i] for i in successful_indices if results['avg_reaction_steps'][i] > 0])
        
        print(f"\nBasic performance metrics (mean [95% CI]):")
        if _STATS_AVAILABLE:
            print(f"  Average time: {time_mean:.3f} [{time_ci_low:.3f}, {time_ci_high:.3f}] seconds")
            print(f"  Average steps: {steps_mean:.1f} [{steps_ci_low:.1f}, {steps_ci_high:.1f}]")
            print(f"  Average path length: {path_mean:.2f} [{path_ci_low:.2f}, {path_ci_high:.2f}]")
            print(f"  Average smoothness: {smooth_mean:.3f} [{smooth_ci_low:.3f}, {smooth_ci_high:.3f}]")
        else:
            std_time = np.std(success_times)
            std_steps = np.std(success_steps)
            print(f"  Average time: {time_mean:.3f} ± {std_time:.3f} seconds")
            print(f"  Average steps: {steps_mean:.1f} ± {std_steps:.1f}")
            print(f"  Average path length: {path_mean:.2f}")
            print(f"  Average smoothness: {smooth_mean:.3f}")
        
        print(f"\nReal-time performance metrics (mean [95% CI]):")
        if _STATS_AVAILABLE:
            print(f"  T_local (local decision time): {local_dt_mean:.2f} [{local_dt_ci_low:.2f}, {local_dt_ci_high:.2f}] ms")
            print(f"  f_replan (Replanning Frequency): {replan_mean:.4f} [{replan_ci_low:.4f}, {replan_ci_high:.4f}] times/step")
            print(f"  L_rt (control latency): {latency_mean:.3f} [{latency_ci_low:.3f}, {latency_ci_high:.3f}] ms")
        else:
            std_local_dt = np.std(success_local_dt)
            print(f"  T_local (local decision time): {local_dt_mean:.2f} ± {std_local_dt:.2f} ms")
            print(f"  f_replan (Replanning Frequency): {replan_mean:.4f} times/step")
            print(f"  L_rt (control latency): {latency_mean:.3f} ms")
        
        avg_reaction = np.mean([results['avg_reaction_steps'][i] for i in successful_indices if results['avg_reaction_steps'][i] > 0])
        if not np.isnan(avg_reaction) and avg_reaction > 0:
            print(f"  H_react (dynamic obstacle reaction steps): {avg_reaction:.1f} steps")
        else:
            print(f"  H_react (dynamic obstacle reaction steps): N/A (no dynamic obstacle encountered)")
        
        # New: task completion efficiency analysis
        avg_task_time = (steps_mean * latency_mean) / 1000  # Convert to seconds
        task_efficiency = success_rate / (avg_task_time * 100) if avg_task_time > 0 else 0
        print(f"\n🎯 Task completion efficiency:")
        print(f"  Average task completion time: {avg_task_time:.3f} seconds (= {steps_mean:.1f}steps × {latency_mean:.2f}ms)")
        print(f"  Real-time system efficiency: {task_efficiency:.4f} (= Success rate{success_rate:.1f}% / 任务时间{avg_task_time:.3f}s)")
        print(f"  Success cases sample size: {success_count} (percentage {success_rate:.1f}%)")
        print(f"  Failed cases: {len(configs) - success_count} (percentage {100-success_rate:.1f}%)")
    
    # Save raw data to JSON file
    import json
    config_basename = os.path.splitext(os.path.basename(config_file))[0]
    raw_data_file = config_file.replace('.pkl', f'_{agent_name.replace(" ", "_")}_raw_data.json')
    try:
        serializable_results = {}
        for key, value in results.items():
            if key == 'local_decision_times':
                # Skip nested list, too large
                continue
            serializable_results[key] = [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in value]
        with open(raw_data_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Raw data saved to: {raw_data_file}")
    except Exception as e:
        print(f"\n⚠ Failed to save raw data: {e}")
    
    # Save detailed statistical results to TXT file
    result_txt_file = config_file.replace('.pkl', f'_{agent_name.replace(" ", "_")}_results.txt')
    try:
        with open(result_txt_file, 'w', encoding='utf-8') as f:
            f.write(f"{agent_name} - Fixed environment test results\n")
            f.write(f"{'='*80}\n")
            f.write(f"Environment configuration file: {config_file}\n")
            f.write(f"Model path: {model_path}\n")
            f.write(f"Number of test environments: {len(configs)}\n")
            f.write(f"{'='*80}\n\n")
            
            f.write(f"Number of test runs: {len(configs)}\n")
            f.write(f"Success count: {success_count}\n")
            f.write(f"Success rate: {success_rate:.2f}% ({success_count}/{len(configs)})\n")
            f.write(f"  Wilson 95% CI: [{wilson_ci_lower*100:.2f}%, {wilson_ci_upper*100:.2f}%]\n\n")
            
            if success_count > 0:
                f.write(f"Basic performance metrics (mean [95% CI]):\n")
                if _STATS_AVAILABLE:
                    f.write(f"  Average time: {time_mean:.3f} [{time_ci_low:.3f}, {time_ci_high:.3f}] seconds\n")
                    f.write(f"  Average steps: {steps_mean:.1f} [{steps_ci_low:.1f}, {steps_ci_high:.1f}]\n")
                    f.write(f"  Average path length: {path_mean:.2f} [{path_ci_low:.2f}, {path_ci_high:.2f}]\n")
                    f.write(f"  Average smoothness: {smooth_mean:.3f} [{smooth_ci_low:.3f}, {smooth_ci_high:.3f}]\n\n")
                else:
                    std_time = np.std(success_times)
                    std_steps = np.std(success_steps)
                    f.write(f"  Average time: {time_mean:.3f} ± {std_time:.3f} seconds\n")
                    f.write(f"  Average steps: {steps_mean:.1f} ± {std_steps:.1f}\n")
                    f.write(f"  Average path length: {path_mean:.2f}\n")
                    f.write(f"  Average smoothness: {smooth_mean:.3f}\n\n")
                
                f.write(f"Real-time performance metrics (mean [95% CI]):\n")
                if _STATS_AVAILABLE:
                    f.write(f"  T_local (local decision time): {local_dt_mean:.2f} [{local_dt_ci_low:.2f}, {local_dt_ci_high:.2f}] ms\n")
                    f.write(f"  f_replan (Replanning Frequency): {replan_mean:.4f} [{replan_ci_low:.4f}, {replan_ci_high:.4f}] times/step\n")
                    f.write(f"  L_rt (control latency): {latency_mean:.3f} [{latency_ci_low:.3f}, {latency_ci_high:.3f}] ms\n")
                else:
                    std_local_dt = np.std(success_local_dt)
                    f.write(f"  T_local (local decision time): {local_dt_mean:.2f} ± {std_local_dt:.2f} ms\n")
                    f.write(f"  f_replan (Replanning Frequency): {replan_mean:.4f} times/step\n")
                    f.write(f"  L_rt (control latency): {latency_mean:.3f} ms\n")
                
                f.write(f"\nTask completion efficiency:\n")
                f.write(f"  Average task completion time: {avg_task_time:.3f} seconds (= {steps_mean:.1f}steps × {latency_mean:.2f}ms)\n")
                f.write(f"  Real-time system efficiency: {task_efficiency:.4f} (= Success rate{success_rate:.1f}% / Task Time{avg_task_time:.3f}s)\n")
                f.write(f"  Success cases sample size: {success_count} (percentage {success_rate:.1f}%)\n")
                f.write(f"  Failed cases: {len(configs) - success_count} (percentage {100-success_rate:.1f}%)\n")
            else:
                f.write(f"\n⚠ All tests failed, cannot calculate performance metrics\n")
        
        print(f"✓ Detailed results saved to: {result_txt_file}")
    except Exception as e:
        print(f"⚠ Failed to save detailed results: {e}")
        import traceback
        traceback.print_exc()
    
    return results

def batch_test_rl_on_random_envs(grid_map, nodes, env, model_path, agent_class, 
                                  agent_name, num_runs=100, max_steps=1000, 
                                  agent_params=None):
    """Batch test RL algorithm on random environments (with complete performance metrics)
    
    Args:
        grid_map: Base map
        nodes: Path nodes
        env: Environment instance with obstacles configured (to get obs/dobs config)
        model_path: Model path
        agent_class: Agent class (RainbowDQN, PaperDQNAgent, Paper2DQNAgent)
        agent_name: Algorithm display name
        num_runs: Number of test runs
        max_steps: Maximum steps per test
        agent_params: Agent extra parameters (dict)
    
    Returns:
        dict: Test results containing all performance metrics
    """
    print(f"\n{'='*60}")
    print(f"Testing on random environments {agent_name}")
    print(f"{'='*60}")
    print(f"Model path: {model_path}")
    print(f"Number of test runs: {num_runs}")
    
    # 检查Model文件
    if not os.path.exists(model_path):
        print(f"✗ Error: Model file does not exist: {model_path}")
        return None
    
    # Create temporary environment to initialize Agent
    temp_env = GridWorld(
        grid_map=grid_map,
        start=nodes[0],
        goal=nodes[-1],
        goal_list=nodes,
        obs=env.num_obstacles,
        dobs=env.num_dobs
    )
    temp_env.obstacle_mode = env.obstacle_mode
    if hasattr(env, 'block_size_range'):
        temp_env.block_size_range = env.block_size_range
    
    # Create Agent and load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if agent_params is None:
        agent_params = {}
    
    agent = agent_class(
        env=temp_env,
        gamma=agent_params.get('gamma', 0.99),
        epsilon=0.0,  # No exploration during testing
        epsilon_min=agent_params.get('epsilon_min', 0.01),
        epsilon_decay=agent_params.get('epsilon_decay', 0.995),
        learning_rate=agent_params.get('learning_rate', 0.005),
        batch_size=agent_params.get('batch_size', 64),
        memory_size=agent_params.get('memory_size', 200000),
        **{k: v for k, v in agent_params.items() 
           if k not in ['gamma', 'epsilon_min', 'epsilon_decay', 'learning_rate', 'batch_size', 'memory_size']}
    )
    
    agent.load_model(model_path)
    print(f"✓ Model loaded successfully")
    
    # Test results (including real-time performance metrics)
    results = {
        'success': [],
        'time': [],
        'path_length': [],
        'smoothness': [],
        'steps': [],
        # New real-time performance metrics
        'avg_local_decision_time': [],  # T_local: Average local decision time (ms)
        'replan_count': [],  # Number of replans
        'replan_frequency': [],  # f_replan: Replan frequency (times/step)
        'control_latency': [],  # L_rt: End-to-end control latency (ms)
        'avg_reaction_steps': [],  # H_react: Dynamic obstacle reaction steps
        'dynamic_obstacle_encounters': []  # Dynamic obstacle encounter count
    }
    
    print(f"\nStarting test {num_runs} times...")
    progress_bar = tqdm(range(num_runs), desc=f"{agent_name}Test progress", unit="run")
    
    for run_idx in progress_bar:
        # Create new environment instance for each test
        test_env = GridWorld(
            grid_map=grid_map,
            start=nodes[0],
            goal=nodes[-1],
            goal_list=nodes,
            obs=env.num_obstacles,
            dobs=env.num_dobs
        )
        test_env.obstacle_mode = env.obstacle_mode
        if hasattr(env, 'block_size_range'):
            test_env.block_size_range = env.block_size_range
        
        # Run test (random env needs obstacle resampling)
        start_time = time.time()
        success, path, steps, metrics = run_rl_algorithm_without_viz(
            test_env, agent, max_steps=max_steps, resample_obstacles=True
        )
        run_time = time.time() - start_time
        
        results['success'].append(success)
        results['time'].append(run_time)
        results['steps'].append(steps)
        
        if success and len(path) > 1:
            path_length, smoothness = calculate_path_metrics(path)
            results['path_length'].append(path_length)
            results['smoothness'].append(smoothness)
        else:
            results['path_length'].append(0)
            results['smoothness'].append(0)
        
        # Process real-time performance metrics
        if metrics['local_decision_times']:
            avg_decision_time = np.mean(metrics['local_decision_times'])
            results['avg_local_decision_time'].append(avg_decision_time)
        else:
            results['avg_local_decision_time'].append(0.0)
        
        results['replan_count'].append(metrics['replan_count'])
        
        # Calculate replan frequency (only for success cases)
        if success and steps > 0:
            replan_freq = metrics['replan_count'] / steps
            results['replan_frequency'].append(replan_freq)
        else:
            results['replan_frequency'].append(0.0)
        
        # Control latency = end-to-end time / steps
        if steps > 0:
            control_latency_ms = (run_time * 1000) / steps
            results['control_latency'].append(control_latency_ms)
        else:
            results['control_latency'].append(0.0)
        
        # Average reaction steps
        if metrics['reaction_steps']:
            avg_reaction = np.mean(metrics['reaction_steps'])
            results['avg_reaction_steps'].append(avg_reaction)
        else:
            results['avg_reaction_steps'].append(0.0)
        
        results['dynamic_obstacle_encounters'].append(metrics['dynamic_obstacle_encounters'])
        
        # Update progress bar showing current success rate
        current_success_rate = sum(results['success']) / len(results['success']) * 100
        progress_bar.set_postfix({'Success rate': f'{current_success_rate:.1f}%'})
    
    progress_bar.close()
    
    # Calculate statistics
    success_count = sum(results['success'])
    success_rate = success_count / num_runs * 100
    
    # Calculate Wilson 95% confidence interval (for success rate)
    n = num_runs
    p_hat = success_count / n
    z = 1.96
    wilson_center = (p_hat + z**2/(2*n)) / (1 + z**2/n)
    wilson_width = z * np.sqrt((p_hat*(1-p_hat) + z**2/(4*n)) / n) / (1 + z**2/n)
    wilson_ci_lower = wilson_center - wilson_width
    wilson_ci_upper = wilson_center + wilson_width
    
    print(f"\n{'='*60}")
    print(f"{agent_name}Test Results")
    print(f"{'='*60}")
    print(f"Success rate: {success_rate:.2f}% ({success_count}/{num_runs})")
    print(f"  Wilson 95% CI: [{wilson_ci_lower*100:.2f}%, {wilson_ci_upper*100:.2f}%]")
    
    if success_count > 0:
        successful_indices = [i for i, s in enumerate(results['success']) if s]
        
        # Extract success cases data for confidence interval calculation
        success_times = np.array([results['time'][i] for i in successful_indices])
        success_steps = np.array([results['steps'][i] for i in successful_indices])
        # Only count valid path metrics in success cases (exclude 0 padding after failure)
        success_path_lengths = np.array([results['path_length'][i] for i in successful_indices if results['path_length'][i] > 0])
        success_smoothness = np.array([results['smoothness'][i] for i in successful_indices if results['smoothness'][i] > 0])
        success_local_dt = np.array([results['avg_local_decision_time'][i] for i in successful_indices])
        success_replan_freq = np.array([results['replan_frequency'][i] for i in successful_indices])
        success_control_latency = np.array([results['control_latency'][i] for i in successful_indices])
        
        # Calculate mean and 95% confidence interval
        if _STATS_AVAILABLE:
            from statistical_tests import calculate_confidence_interval
            
            time_mean, time_ci_low, time_ci_high = calculate_confidence_interval(success_times)
            steps_mean, steps_ci_low, steps_ci_high = calculate_confidence_interval(success_steps)
            # Path length and smoothness: only calculate CI for valid values
            if len(success_path_lengths) > 0:
                path_mean, path_ci_low, path_ci_high = calculate_confidence_interval(success_path_lengths)
            else:
                path_mean = path_ci_low = path_ci_high = 0.0
            if len(success_smoothness) > 0:
                smooth_mean, smooth_ci_low, smooth_ci_high = calculate_confidence_interval(success_smoothness)
            else:
                smooth_mean = smooth_ci_low = smooth_ci_high = 0.0
            local_dt_mean, local_dt_ci_low, local_dt_ci_high = calculate_confidence_interval(success_local_dt)
            replan_mean, replan_ci_low, replan_ci_high = calculate_confidence_interval(success_replan_freq)
            latency_mean, latency_ci_low, latency_ci_high = calculate_confidence_interval(success_control_latency)
        else:
            time_mean = np.mean(success_times)
            steps_mean = np.mean(success_steps)
            path_mean = np.mean(success_path_lengths) if len(success_path_lengths) > 0 else 0.0
            smooth_mean = np.mean(success_smoothness) if len(success_smoothness) > 0 else 0.0
            local_dt_mean = np.mean(success_local_dt)
            replan_mean = np.mean(success_replan_freq)
            latency_mean = np.mean(success_control_latency)
        
        avg_reaction_steps = np.mean([results['avg_reaction_steps'][i] for i in successful_indices 
                                     if results['avg_reaction_steps'][i] > 0])  # Only cases with reaction
        total_obstacle_encounters = sum([results['dynamic_obstacle_encounters'][i] for i in successful_indices])
        
        print(f"\nBasic performance metrics (mean [95% CI]):")
        if _STATS_AVAILABLE:
            print(f"  Average time: {time_mean:.3f} [{time_ci_low:.3f}, {time_ci_high:.3f}] seconds")
            print(f"  Average steps: {steps_mean:.1f} [{steps_ci_low:.1f}, {steps_ci_high:.1f}]")
            print(f"  Average path length: {path_mean:.2f} [{path_ci_low:.2f}, {path_ci_high:.2f}]")
            print(f"  Average smoothness: {smooth_mean:.3f} [{smooth_ci_low:.3f}, {smooth_ci_high:.3f}]")
        else:
            print(f"  Average time: {time_mean:.3f} seconds")
            print(f"  Average steps: {steps_mean:.1f}")
            print(f"  Average path length: {path_mean:.2f}")
            print(f"  Average smoothness: {smooth_mean:.3f}")
        
        print(f"\nReal-time performance metrics (mean [95% CI]):")
        if _STATS_AVAILABLE:
            print(f"  T_local (local decision time): {local_dt_mean:.3f} [{local_dt_ci_low:.3f}, {local_dt_ci_high:.3f}] ms")
            print(f"  f_replan (Replanning Frequency): {replan_mean:.4f} [{replan_ci_low:.4f}, {replan_ci_high:.4f}] times/step")
            print(f"  L_rt (End-to-End Control Latency): {latency_mean:.3f} [{latency_ci_low:.3f}, {latency_ci_high:.3f}] ms")
        else:
            print(f"  T_local (local decision time): {local_dt_mean:.3f} ms")
            print(f"  f_replan (Replanning Frequency): {replan_mean:.4f} times/step")
            print(f"  L_rt (End-to-End Control Latency): {latency_mean:.3f} ms")
        if avg_reaction_steps > 0:
            print(f"  H_react (Dynamic Obstacle Reaction Steps): {avg_reaction_steps:.2f} steps")
        else:
            print(f"  H_react (Dynamic Obstacle Reaction Steps): N/A (No Encounters)")
        print(f"  Total Dynamic Obstacle Encounters: {total_obstacle_encounters}")
    
    return results


def main():
    files = "grid_map2.txt"
    try:
        grid_map = np.loadtxt(files, dtype=int)
        print(f"Successfully loaded map file: {files}")
    except Exception as e:
        print(f"Failed to load map file: {e}")
        return
    
    nodes = [(80, 110), (71, 98), (35, 94), (42, 78), (82, 69), (75, 25), (67, 14), (31, 19), (24, 27), (13, 35), (13, 94)]
    #nodes = [(63, 161), (83, 98), (101, 79), (98, 45), (135, 34), (135, 22)]
    #nodes = [(7, 27), (7, 36), (28, 36), (34, 21), (43, 26), (53, 31), (62, 54), (61, 68)]
    #nodes = [(77, 138), (59, 124), (44, 122), (54, 104), (74, 99), (68, 58), (57, 48), (47, 18), (35, 23), (27, 38), (13, 51), (14, 147)]
    #nodes = [(94, 33), (91, 146), (14, 142), (17, 15), (78, 16), (74, 124), (34, 117), (37, 45), (60, 52), (85, 63)]
    print("1: Batch testing the performance of this algorithm (random environment)")
    print("2: Batch testing this algorithm in a fixed environment")
    print("3: Visualization of fixed environment configuration (randomly selected samples)")

    evaluation_type = input("Please select evaluation method (1/2/3): ")
        
    if evaluation_type == "1":
        print("\nPlease select map configuration:")
        print("  1) map1 (grid_map.txt)")
        print("  2) map2 (grid_map2.txt)")
        print("  3) map3 (grid_map3.txt)")
        print("  4) map4 (grid_map4.txt)")
        
        map_choice = input("Please select map (1/2/3/4) [default: current map]: ").strip()

        MAP_CONFIGS = {
                '1': {
                    'map_file': 'grid_map.txt',
                    'nodes': [(7, 27), (7, 36), (28, 36), (34, 21), (43, 26), (53, 31), (62, 54), (61, 68)],
                    'model': 'models/dqn_model_ALL_newtrain_map1.pth'
                },
                '2': {
                    'map_file': 'grid_map2.txt',
                    'nodes': [(80, 110), (71, 98), (35, 94), (42, 78), (82, 69), (75, 25), (67, 14), (31, 19), (24, 27), (13, 35), (13, 94)],
                    'model': 'models/dqn_model_ALL_newtrain_map2.pth'
                },
                '3': {
                    'map_file': 'grid_map3.txt',
                    'nodes': [(63, 161), (83, 98), (101, 79), (98, 45), (135, 34), (135, 22)],
                    'model': 'models/dqn_model_ALL_newtrain_map3.pth'
                },
                '4':{
                    'map_file': 'grid_map4.txt',
                    'nodes': [(77, 138), (59, 124), (44, 122), (54, 104), (74, 99), (68, 58), (57, 48), (47, 18), (35, 23), (27, 38), (13, 51), (14, 147)],
                    'model': 'models/dqn_model_ALL_newtrain_map2.pth'
                },
            }
        
        if map_choice in MAP_CONFIGS:
            config = MAP_CONFIGS[map_choice]
            grid_map = np.loadtxt(config['map_file'], dtype=int)
            nodes = config['nodes']
            model_path = config['model']
            print(f"\nUsing map configuration {map_choice}:")
            print(f"  - Map file: {config['map_file']}")
            print(f"  - Number of nodes: {len(nodes)}")
            print(f"  - Model path: {model_path}")
        
        # Define 5 test scenarios
        test_scenarios = [
            {
                'name': 'Level1',
                'mode': 'point',
                'obs': 50,
                'dobs': 0,
                'block_range': None
            },
            {
                'name': 'Level2',
                'mode': 'point',
                'obs': 50,
                'dobs': 30,
                'block_range': None
            },
            {
                'name': 'Level3',
                'mode': 'block',
                'obs': 50,
                'dobs': 30,
                'block_range': (2, 4)
            },
            {
                'name': 'Level4',
                'mode': 'block',
                'obs': 50,
                'dobs': 50,
                'block_range': (3, 5)
            }
        ]
        
        num_tests = int(input("\nEnter the number of tests per scenario: "))
        os.makedirs("./test", exist_ok=True)
        
        print(f"\nTesting will be conducted in the following 4 scenarios, each with {num_tests} tests:")
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"  {i}. {scenario['name']}")
        print()
        
        if not os.path.exists(model_path):
            print(f"Warning: Model file does not exist: {model_path}")
            model_path = input("Please enter the correct model path: ").strip()
        
        # Test each scenario
        all_results = {}
        map_id = map_choice if map_choice in MAP_CONFIGS else 'current'
        
        for scenario in test_scenarios:
            print(f"\n{'='*60}")
            print(f"Starting test: {scenario['name']}")
            print(f"{'='*60}")
            test_env = GridWorld(
                grid_map=grid_map,
                start=nodes[0],
                goal=nodes[-1],
                goal_list=nodes,
                obs=scenario['obs'],
                dobs=scenario['dobs']
            )
            if scenario['mode'] == 'block':
                test_env.obstacle_mode = 'block'
                test_env.block_size_range = scenario['block_range']
                print(f"  Obstacle mode: Block (size range {scenario['block_range']})")
            else:
                test_env.obstacle_mode = 'point'
                print(f"  Obstacle mode: Point")
            
            print(f"  Static obstacles: {scenario['obs']}, Dynamic obstacles: {scenario['dobs']}")
            
            results = batch_test_rl_on_random_envs(
                grid_map=grid_map,
                nodes=nodes,
                env=test_env,
                model_path=model_path,
                agent_class=RainbowDQN,
                agent_name=f"Our Method - {scenario['name']}",
                num_runs=num_tests,
                max_steps=1000
            )
            
            if results:
                result_file = f"./test/ourmethod_map{map_id}_{scenario['name']}.txt"
                with open(result_file, 'w', encoding='utf-8') as f:
                    f.write(f"Our Method - {scenario['name']} Test Results\n")
                    f.write(f"{'='*80}\n")
                    f.write(f"Map: {MAP_CONFIGS.get(map_choice, {}).get('map_file', 'Current Map')}\n")
                    f.write(f"Model: {model_path}\n")
                    f.write(f"Number of Tests: {num_tests}\n")
                    f.write(f"Obstacle Mode: {scenario['mode']}\n")
                    f.write(f"Static Obstacles: {scenario['obs']}, Dynamic Obstacles: {scenario['dobs']}\n")
                    if scenario['block_range']:
                        f.write(f"Block Size Range: {scenario['block_range']}\n")
                    f.write(f"{'='*80}\n\n")
                    
                    success_count = sum(results['success'])
                    success_rate = success_count / num_tests * 100
                    f.write(f"Success Rate: {success_rate:.2f}% ({success_count}/{num_tests})\n\n")
                    
                    if success_count > 0:
                        successful_indices = [i for i, s in enumerate(results['success']) if s]
                        avg_time = np.mean([results['time'][i] for i in successful_indices])
                        avg_steps = np.mean([results['steps'][i] for i in successful_indices])
                        avg_path_length = np.mean([results['path_length'][i] for i in successful_indices])
                        avg_smoothness = np.mean([results['smoothness'][i] for i in successful_indices])
                        avg_local_dt = np.mean([results['avg_local_decision_time'][i] for i in successful_indices])
                        avg_replan_freq = np.mean([results['replan_frequency'][i] for i in successful_indices])
                        avg_control_latency = np.mean([results['control_latency'][i] for i in successful_indices])
                        reaction_values = [results['avg_reaction_steps'][i] for i in successful_indices 
                                         if results['avg_reaction_steps'][i] > 0]
                        avg_reaction = np.mean(reaction_values) if reaction_values else 0.0
                        total_encounters = sum([results['dynamic_obstacle_encounters'][i] for i in successful_indices])
                        
                        f.write(f"Basic Performance Metrics:\n")
                        f.write(f"  Average Time: {avg_time:.3f} seconds\n")
                        f.write(f"  Average Steps: {avg_steps:.1f}\n")
                        f.write(f"  Average Path Length: {avg_path_length:.2f}\n")
                        f.write(f"  Average Smoothness: {avg_smoothness:.3f}\n\n")
                        
                        f.write(f"Real-time Performance Metrics:\n")
                        f.write(f"  T_local (Local Decision Time): {avg_local_dt:.3f} ms\n")
                        f.write(f"  f_replan (Replanning Frequency): {avg_replan_freq:.4f} times/step\n")
                        f.write(f"  L_rt (End-to-End Control Latency): {avg_control_latency:.3f} ms\n")
                
                all_results[scenario['name']] = results
                print(f"✓ Results saved to: {result_file}")
        
        # Generate summary report
        summary_file = f"./test/ourmethod_map{map_id}_multi_scenario_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"Our Method - Multi-Scenario Test Summary Report\n")
            f.write(f"{'='*80}\n")
            f.write(f"Map: {MAP_CONFIGS.get(map_choice, {}).get('map_file', 'Current Map')}\n")
            f.write(f"Model: {model_path}\n")
            f.write(f"Number of Tests per Scenario: {num_tests}\n")
            f.write(f"Number of Test Scenarios: {len(test_scenarios)}\n")
            f.write(f"{'='*80}\n\n")
            
            for scenario_name, results in all_results.items():
                f.write(f"\n{'='*80}\n")
                f.write(f"{scenario_name}\n")
                f.write(f"{'='*80}\n")
                
                success_count = sum(results['success'])
                success_rate = success_count / num_tests * 100
                f.write(f"Success Rate: {success_rate:.2f}% ({success_count}/{num_tests})\n\n")
                
                if success_count > 0:
                    successful_indices = [i for i, s in enumerate(results['success']) if s]
                    avg_time = np.mean([results['time'][i] for i in successful_indices])
                    avg_steps = np.mean([results['steps'][i] for i in successful_indices])
                    avg_path_length = np.mean([results['path_length'][i] for i in successful_indices])
                    avg_smoothness = np.mean([results['smoothness'][i] for i in successful_indices])
                    avg_local_dt = np.mean([results['avg_local_decision_time'][i] for i in successful_indices])
                    avg_replan_freq = np.mean([results['replan_frequency'][i] for i in successful_indices])
                    avg_control_latency = np.mean([results['control_latency'][i] for i in successful_indices])
                    reaction_values = [results['avg_reaction_steps'][i] for i in successful_indices 
                                     if results['avg_reaction_steps'][i] > 0]
                    avg_reaction = np.mean(reaction_values) if reaction_values else 0.0
                    total_encounters = sum([results['dynamic_obstacle_encounters'][i] for i in successful_indices])
                    
                    f.write(f"Basic Performance Metrics:\n")
                    f.write(f"  Average Time: {avg_time:.3f} seconds\n")
                    f.write(f"  Average Steps: {avg_steps:.1f}\n")
                    f.write(f"  Average Path Length: {avg_path_length:.2f}\n")
                    f.write(f"  Average Smoothness: {avg_smoothness:.3f}\n\n")
                    
                    f.write(f"Real-time Performance Metrics:\n")
                    f.write(f"  T_local (Local Decision Time): {avg_local_dt:.3f} ms\n")
                    f.write(f"  f_replan (Replanning Frequency): {avg_replan_freq:.4f} times/step\n")
                    f.write(f"  L_rt (End-to-End Control Latency): {avg_control_latency:.3f} ms\n")
                    if avg_reaction > 0:
                        f.write(f"  H_react (Dynamic Obstacle Reaction Steps): {avg_reaction:.2f} steps\n")
                    else:
                        f.write(f"  H_react (Dynamic Obstacle Reaction Steps): N/A (No Encounters)\n")
                    f.write(f"  Total Dynamic Obstacle Encounters: {total_encounters}\n")
                f.write("\n")
        
        print(f"\n{'='*60}")
        print(f"All scenario tests completed!")
        print(f"Summary report saved to: {summary_file}")
        print(f"{'='*60}")

    elif evaluation_type == "2":
        print("\n[Fixed Environment] Batch testing the algorithm described in this article...")
        
        # Select map configuration
        print("\nPlease select map configuration:")
        print("  1) map1 (grid_map.txt)")
        print("  2) map2 (grid_map2.txt)")
        print("  3) map3 (grid_map3.txt)")
        print("  4) map4 (grid_map4.txt)")
        
        map_choice = input("Please select map (1/2/3/4) [default: 2]: ").strip()
        if not map_choice:
            map_choice = '2'
        
        MAP_CONFIGS = {
            '1': {
                'map_file': 'grid_map.txt',
                'nodes': [(7, 27), (7, 36), (28, 36), (34, 21), (43, 26), (53, 31), (62, 54), (61, 68)],
                'model': 'models/dqn_model_ALL_newtrain_map1.pth'
            },
            '2': {
                'map_file': 'grid_map2.txt',
                'nodes': [(80, 110), (71, 98), (35, 94), (42, 78), (82, 69), (75, 25), (67, 14), (31, 19), (24, 27), (13, 35), (13, 94)],
                'model': 'models/dqn_model_ALL_newtrain_map2.pth'
            },
            '3': {
                'map_file': 'grid_map3.txt',
                'nodes': [(63, 161), (83, 98), (101, 79), (98, 45), (135, 34), (135, 22)],
                'model': 'models/dqn_model_ALL_newtrain_map3.pth'
            },
            '4': {
                'map_file': 'grid_map4.txt',
                'nodes': [(77, 138), (59, 124), (44, 122), (54, 104), (74, 99), (68, 58), (57, 48), (47, 18), (35, 23), (27, 38), (13, 51), (14, 147)],
                'model': 'models/dqn_model_ALL_newtrain_map2.pth'
            },
        }
        
        if map_choice in MAP_CONFIGS:
            config = MAP_CONFIGS[map_choice]
            grid_map = np.loadtxt(config['map_file'], dtype=int)
            nodes = config['nodes']
            default_model = config['model']
            print(f"\nUsing map configuration {map_choice}:")
            print(f"  - Map file: {config['map_file']}")
            print(f"  - Number of nodes: {len(nodes)}")
            print(f"  - Default model: {default_model}")
        else:
            print("Invalid map choice, using current grid_map and nodes")
            default_model = "models/dqn_model.pth"
        
        # Get fixed environment file path
        env_file = input("\nFixed environment file path [default: test_environments/test_envs_1000_obs50_dobs30.pkl]: ").strip()
        if not env_file:
            env_file = "test_environments/test_envs_1000_obs50_dobs30.pkl"
        
        if not os.path.exists(env_file):
            print(f"Error: Fixed environment file does not exist: {env_file}")
            print("Please run the following command to generate the fixed environment first:")
            print(f"  python generate_test_environments.py --num_envs 1000 --obs 50 --dobs 30")
            return
        
        # Get model path
        our_model = input(f"Model path [default: {default_model}]: ").strip()
        if not our_model:
            our_model = default_model
        
        if not os.path.exists(our_model):
            print(f"Error: Model file does not exist: {our_model}")
            return
        
        # Run tests
        results = batch_test_rl_on_fixed_envs(
            grid_map=grid_map,
            nodes=nodes,
            config_file=env_file,
            model_path=our_model,
            agent_class=RainbowDQN,
            agent_name="TSP-OR",
            agent_params={
                'gamma': 0.8,
                'epsilon_min': 0.01,
                'epsilon_decay': 0.9,
                'learning_rate': 0.0004,
                'batch_size': 64,
                'memory_size': 10000,
                'alpha': 0.6,
                'beta': 0.4,
                'n_step': 4
            }
        )
    
    elif evaluation_type == "3":
        print("\n[Visualization] Viewing fixed environment configurations...")
        
        # List available environment files
        env_dir = "test_environments"
        if os.path.exists(env_dir):
            env_files = [f for f in os.listdir(env_dir) if f.endswith('.pkl') and f.startswith('test_envs_')]
            if env_files:
                print("\nAvailable environment files:")
                for i, f in enumerate(env_files, 1):
                    # Parse filename to extract configuration info
                    parts = f.replace('.pkl', '').split('_')
                    info = f
                    for j, part in enumerate(parts):
                        if part.startswith('obs'):
                            obs_count = part[3:]
                        elif part.startswith('dobs'):
                            dobs_count = part[4:]
                        elif part.isdigit() and j == 2:
                            env_count = part
                    if 'point' in f:
                        mode_info = " [Point Obstacles]"
                    elif 'block' in f:
                        mode_info = " [Block Obstacles]"
                    else:
                        mode_info = ""
                    
                    print(f"  {i}. {f}{mode_info}")
        
        # Get fixed environment file path
        env_file = input("\nFixed environment file path [default: test_environments/test_envs_1000_obs50_dobs30.pkl]: ").strip()
        if not env_file:
            env_file = "test_environments/test_envs_1000_obs50_dobs30.pkl"
        
        # If input is a number, select from the list
        if env_file.isdigit() and os.path.exists(env_dir):
            idx = int(env_file) - 1
            if 0 <= idx < len(env_files):
                env_file = os.path.join(env_dir, env_files[idx])
                print(f"Selected: {env_file}")
        elif not env_file.startswith('test_environments/'):
            env_file = os.path.join(env_dir, env_file)
        
        if not os.path.exists(env_file):
            print(f"Error: Fixed environment file does not exist: {env_file}")
            print("Please run the following command to generate the fixed environment first:")
            print(f"  python generate_test_environments.py --num_envs 1000 --obs 50 --dobs 30")
            print(f"  Or use --obstacle_mode block to generate block obstacle environments")
            return

        num_samples_str = input("Number of samples to visualize [default: 10]: ").strip()
        if num_samples_str:
            num_samples = int(num_samples_str)
        else:
            num_samples = 10
        save_dir = input("Save directory [default: visualizations/fixed_envs]: ").strip()
        if not save_dir:
            save_dir = "visualizations/fixed_envs"
        visualize_sample_fixed_envs(
            config_file=env_file,
            num_samples=num_samples,
            save_dir=save_dir
        )
        
    else:
        print("Invalid selection. Please choose 1, 2, or 3.")

if __name__ == "__main__":
    main()