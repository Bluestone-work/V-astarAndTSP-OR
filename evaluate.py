from pathlib import Path
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

def run_rl_algorithm_without_viz(env, agent, max_steps=2000, resample_obstacles=False,
                                 navigation_mode='event', on_step=None):
    """Shared navigation protocol; exceptions propagate and collisions beat success.

    Durations separate policy decisions, complete control cycles (including
    event replans), and whole episodes. Grid steps are not physical seconds.
    """
    original_epsilon = agent.epsilon
    agent.epsilon = 0.
    if hasattr(agent, 'reset_episode'):
        agent.reset_episode()
    agent.env = env
    was_training = agent.q_network.training if hasattr(agent, 'q_network') else False
    if hasattr(agent, 'q_network'):
        agent.q_network.eval()
    env.navigation_mode = navigation_mode
    for key, value in getattr(agent, 'loaded_environment_specification', {}).items():
        if key in ('prediction_mode', 'prediction_enabled', 'avoidance_reward_form', 'reward_scales', 'subgoal_tolerance'):
            setattr(env, key, value)
    env.smoothing_factor = getattr(agent, 'smoothing_factor', .5)
    env.max_steps = max_steps
    episode_start = time.perf_counter()
    metrics = {'local_decision_times': [], 'cycle_times': [], 'action_reversal_count': 0,
               'dynamic_obstacle_encounters': 0, 'reaction_steps': []}
    path = []
    try:
        state = env.reset(resample_obstacles=resample_obstacles)
        path = [tuple(env.agent_pos)]
        if on_step is not None:
            on_step(env, path, 0, 0.)
        last_action = None
        done = bool(env.planning_events and not env.planning_events[0]['accepted'])
        if done:
            env.last_info = {'termination': 'initial_planning_failure', 'success': False}
        for _ in range(max_steps):
            if done:
                break
            cycle_start = time.perf_counter()
            start = time.perf_counter()
            action = int(agent.select_action(state) if hasattr(agent, 'select_action') else agent.act(state))
            if str(getattr(agent, 'device', 'cpu')).startswith('cuda'):
                torch.cuda.synchronize(agent.device)
            metrics['local_decision_times'].append(1000 * (time.perf_counter() - start))
            if last_action is not None:
                from navigation_protocol import ACTIONS
                if np.dot(ACTIONS[last_action], ACTIONS[action]) < 0:
                    metrics['action_reversal_count'] += 1
            last_action = action
            from random import seed as seed_obstacle_motion
            if hasattr(env, '_fixed_config_seed'):
                seed_obstacle_motion(int(env._fixed_config_seed)*1000003 + env.steps)
            state, reward, done = env.step(action)
            path.append(tuple(env.agent_pos))
            metrics['cycle_times'].append(1000 * (time.perf_counter()-cycle_start))
            if on_step is not None:
                on_step(env, path, env.steps, reward)
        metrics.update(termination=env.last_info.get('termination', 'timeout'),
                       episode_wall_time_s=time.perf_counter()-episode_start,
                       global_replans=env.global_replans,
                       planning_seconds=sum(e['seconds'] for e in env.planning_events),
                       planning_events=env.planning_events)
        return bool(env.last_info.get('success')), path, env.steps, metrics
    finally:
        agent.epsilon = original_epsilon
        if hasattr(agent, 'q_network'):
            agent.q_network.train(was_training)


def run_rl_algorithm_with_gif(env, agent, max_steps=2000, gif_path='test.gif', title_prefix='Test'):
    """Render the same transition runner; GIF encoding is outside episode time."""
    import imageio
    from visualization import _render_env_frame
    frames = []
    def capture(current_env, path, steps, reward):
        frames.append(_render_env_frame(current_env, path, title=f'{title_prefix} | Step {steps}'))
    success, path, steps, _ = run_rl_algorithm_without_viz(
        env, agent, max_steps=max_steps, resample_obstacles=False, on_step=capture)
    Path(gif_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(gif_path, frames, duration=.1)
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
        'learning_rate': .001,
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
        'action_reversal_count': [],
        'action_reversal_frequency': [],
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
        env._fixed_config_seed = config.get('seed', i)
        import random
        random.seed(env._fixed_config_seed); np.random.seed(env._fixed_config_seed)
        
        # Determine if frame recording needed (every 100 times)
        record_frames = False  # Render separately; frame encoding must not enter timing comparisons.
        
        # Run test
        start_time = time.time()
        if record_frames:
            # Test with GIF recording (not collecting detailed metrics)
            success, path, steps = run_rl_algorithm_with_gif(
                env, agent, max_steps=2000,
                gif_path=os.path.join(gif_dir, f"env_{i+1:04d}.gif"),
                title_prefix=f"Fixed Env #{i+1}"
            )
            tqdm.write(f"  Saved GIF: {gif_dir}/env_{i+1:04d}.gif")
            # Fill default metrics for GIF test
            metrics = {
                'local_decision_times': [],
                'action_reversal_count': 0,
                'dynamic_obstacle_encounters': 0,
                'reaction_steps': []
            }
        else:
            # Normal test (collecting detailed metrics)
            success, path, steps, metrics = run_rl_algorithm_without_viz(env, agent, max_steps=2000)
        
        run_time = metrics.get('episode_wall_time_s', time.time() - start_time)
        
        results['success'].append(success)
        results['time'].append(run_time)
        results['steps'].append(steps)
        
        # Collect new metrics
        if metrics['local_decision_times']:
            results['local_decision_times'].append(metrics['local_decision_times'])
            results['avg_local_decision_time'].append(np.mean(metrics['local_decision_times']))
            # Control latency = decision time + env step time
            avg_decision_time = np.mean(metrics['local_decision_times'])
            step_time = float(np.mean(metrics['cycle_times'])) if metrics.get('cycle_times') else 0.
            results['control_latency'].append(step_time)  # End-to-end latency
        else:
            results['local_decision_times'].append([])
            results['avg_local_decision_time'].append(0)
            results['control_latency'].append(0)
        
        results['action_reversal_count'].append(metrics['action_reversal_count'])
        results['action_reversal_frequency'].append(
            metrics['action_reversal_count'] / steps if steps > 0 else 0
        )
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
        success_reversal_freq = np.array([
            results['action_reversal_frequency'][i] for i in successful_indices
        ])
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
            reversal_mean, reversal_ci_low, reversal_ci_high = calculate_confidence_interval(
                success_reversal_freq
            )
            latency_mean, latency_ci_low, latency_ci_high = calculate_confidence_interval(success_control_latency)
        else:
            # If statistical_tests unavailable, use simple mean ± std
            time_mean = np.mean(success_times)
            steps_mean = np.mean(success_steps)
            path_mean = np.mean(success_path_lengths) if len(success_path_lengths) > 0 else 0.0
            smooth_mean = np.mean(success_smoothness) if len(success_smoothness) > 0 else 0.0
            local_dt_mean = np.mean(success_local_dt)
            reversal_mean = np.mean(success_reversal_freq)
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
            print(
                f"  f_reverse (sharp action-reversal frequency): "
                f"{reversal_mean:.4f} [{reversal_ci_low:.4f}, {reversal_ci_high:.4f}] times/step"
            )
            print(f"  L_rt (control latency): {latency_mean:.3f} [{latency_ci_low:.3f}, {latency_ci_high:.3f}] ms")
        else:
            std_local_dt = np.std(success_local_dt)
            print(f"  T_local (local decision time): {local_dt_mean:.2f} ± {std_local_dt:.2f} ms")
            print(f"  f_reverse (sharp action-reversal frequency): {reversal_mean:.4f} times/step")
            print(f"  L_rt (control latency): {latency_mean:.3f} ms")
        
        avg_reaction = np.mean([results['avg_reaction_steps'][i] for i in successful_indices if results['avg_reaction_steps'][i] > 0])
        if not np.isnan(avg_reaction) and avg_reaction > 0:
            print(f"  H_react (dynamic obstacle reaction steps): {avg_reaction:.1f} steps")
        else:
            print(f"  H_react (dynamic obstacle reaction steps): N/A (no dynamic obstacle encountered)")
        
        # Task throughput is measured from completed tasks and total wall-clock
        # episode time.  Decision frequency remains a separate metric.
        avg_task_time = float(np.mean(success_times)) if len(success_times) else 0.0
        total_wall_time = float(np.sum(results['time']))
        task_throughput = success_count / total_wall_time if total_wall_time > 0 else 0.0
        print(f"\nTask throughput:")
        print(f"  Mean successful-task time: {avg_task_time:.3f} seconds")
        print(f"  Completed tasks / total episode time: {task_throughput:.4f} tasks/s")
        print(f"  Decision latency and control frequency are reported separately above.")
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
                    f.write(
                        f"  f_reverse (sharp action-reversal frequency): "
                        f"{reversal_mean:.4f} [{reversal_ci_low:.4f}, {reversal_ci_high:.4f}] times/step\n"
                    )
                    f.write(f"  L_rt (control latency): {latency_mean:.3f} [{latency_ci_low:.3f}, {latency_ci_high:.3f}] ms\n")
                else:
                    std_local_dt = np.std(success_local_dt)
                    f.write(f"  T_local (local decision time): {local_dt_mean:.2f} ± {std_local_dt:.2f} ms\n")
                    f.write(f"  f_reverse (sharp action-reversal frequency): {reversal_mean:.4f} times/step\n")
                    f.write(f"  L_rt (control latency): {latency_mean:.3f} ms\n")
                
                f.write(f"\nTask completion efficiency:\n")
                f.write(f"  Average task completion time: {avg_task_time:.3f} seconds (measured complete episode wall time)\n")
                f.write(f"  Task throughput: {task_throughput:.4f} successful tasks/s (= {success_count} / total episode wall time)\n")
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
                                  agent_name, num_runs=100, max_steps=2000,
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
        learning_rate=agent_params.get('learning_rate', 0.001),
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
        'action_reversal_count': [],
        'action_reversal_frequency': [],
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
        run_time = metrics.get('episode_wall_time_s', time.time() - start_time)
        
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
        
        results['action_reversal_count'].append(metrics['action_reversal_count'])
        
        # Calculate replan frequency (only for success cases)
        if success and steps > 0:
            reversal_freq = metrics['action_reversal_count'] / steps
            results['action_reversal_frequency'].append(reversal_freq)
        else:
            results['action_reversal_frequency'].append(0.0)
        
        # Control latency = end-to-end time / steps
        if steps > 0:
            control_latency_ms = float(np.mean(metrics['cycle_times'])) if metrics.get('cycle_times') else 0.
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
        success_reversal_freq = np.array([
            results['action_reversal_frequency'][i] for i in successful_indices
        ])
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
            reversal_mean, reversal_ci_low, reversal_ci_high = calculate_confidence_interval(
                success_reversal_freq
            )
            latency_mean, latency_ci_low, latency_ci_high = calculate_confidence_interval(success_control_latency)
        else:
            time_mean = np.mean(success_times)
            steps_mean = np.mean(success_steps)
            path_mean = np.mean(success_path_lengths) if len(success_path_lengths) > 0 else 0.0
            smooth_mean = np.mean(success_smoothness) if len(success_smoothness) > 0 else 0.0
            local_dt_mean = np.mean(success_local_dt)
            reversal_mean = np.mean(success_reversal_freq)
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
            print(
                f"  f_reverse (sharp action-reversal frequency): "
                f"{reversal_mean:.4f} [{reversal_ci_low:.4f}, {reversal_ci_high:.4f}] times/step"
            )
            print(f"  L_rt (End-to-End Control Latency): {latency_mean:.3f} [{latency_ci_low:.3f}, {latency_ci_high:.3f}] ms")
        else:
            print(f"  T_local (local decision time): {local_dt_mean:.3f} ms")
            print(f"  f_reverse (sharp action-reversal frequency): {reversal_mean:.4f} times/step")
            print(f"  L_rt (End-to-End Control Latency): {latency_mean:.3f} ms")
        if avg_reaction_steps > 0:
            print(f"  H_react (Dynamic Obstacle Reaction Steps): {avg_reaction_steps:.2f} steps")
        else:
            print(f"  H_react (Dynamic Obstacle Reaction Steps): N/A (No Encounters)")
        print(f"  Total Dynamic Obstacle Encounters: {total_obstacle_encounters}")
        total_wall_time = float(np.sum(results['time']))
        task_throughput = success_count / total_wall_time if total_wall_time > 0 else 0.0
        print(f"  Task throughput (completed tasks / total episode time): {task_throughput:.4f} tasks/s")
    
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
                max_steps=2000
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
                        avg_reversal_freq = np.mean([
                            results['action_reversal_frequency'][i] for i in successful_indices
                        ])
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
                        f.write(f"  f_reverse (sharp action-reversal frequency): {avg_reversal_freq:.4f} times/step\n")
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
                    avg_reversal_freq = np.mean([
                        results['action_reversal_frequency'][i] for i in successful_indices
                    ])
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
                    f.write(f"  f_reverse (sharp action-reversal frequency): {avg_reversal_freq:.4f} times/step\n")
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
                'n_step': 1
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
