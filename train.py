import numpy as np
import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from environment import GridWorld
from agents import RainbowDQN
from models import DuelingDQN_with_LSTM

def train_agent(env, agent, episodes=1000, max_steps=1000, save_path="dqn_model.pth", log_dir="runs",
                curriculum_learning=False, curriculum_config=None): 
    """
        Training

        Args:
        curriculum_learning: Whether to use curriculum learning
        curriculum_config: Curriculum learning configuration, format:
        {
            'epsilon_reset_strategy': 'partial', # epsilon reset strategy
            'stages': [
                {'episodes': 300, 'mode': 'point', 'obs': 30,
                    'epsilon_boost': 0.3, 'description': 'Stage 1'},
                {'episodes': 400, 'mode': 'block', 'block_range': (2, 3), 'obs': 40,
                    'epsilon_boost': 0.4, 'description': 'Stage 2'},
                ...
            ]
        }

    """
    run_id = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(log_dir, f"run_{run_id}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    dummy_input = torch.rand(1, 10, 51).to(agent.device)
    # writer.add_graph(agent.q_network, dummy_input)  

    best_reward = -float('inf')
    step_counter = 0 
    all_rewards = []
    all_losses = []
    all_epsilons = []
    success_history = []

    if curriculum_learning and curriculum_config:
        print("=" * 80)
        print("Course Study Mode Activated")
        print("=" * 80)
        for i, stage in enumerate(curriculum_config['stages'], 1):
            print(f"Stage {i}: {stage['description']}")
            print(f"  - Episodes: {stage['episodes']}")
            print(f"  - Obstacle Mode: {stage['mode']}")
            print(f"  - Number of Obstacles: {stage.get('obs', env.num_obstacles)}")
            if stage['mode'] == 'block':
                print(f"  - Block Size: {stage.get('block_range', (2, 5))}")
        print("=" * 80)
    else:
        print("Training started (each episode will generate a random environment to enhance diversity)...")
    
    current_stage_info = None
    current_stage_idx = -1
    if curriculum_learning and curriculum_config:
        episode_counter = 0
        for stage in curriculum_config['stages']:
            stage['start_episode'] = episode_counter
            stage['end_episode'] = episode_counter + stage['episodes']
            episode_counter += stage['episodes']
        
        # Obtain epsilon reset strategy
        epsilon_reset_strategy = curriculum_config.get('epsilon_reset_strategy', 'partial')
    
    with tqdm(total=episodes, desc="Training Progress", unit="episode") as pbar:
        for episode in range(episodes):
            if curriculum_learning and curriculum_config:
                for stage_idx, stage in enumerate(curriculum_config['stages']):
                    if stage['start_episode'] <= episode < stage['end_episode']:
                        if current_stage_idx != stage_idx:
                            old_epsilon = agent.epsilon
                            current_stage_idx = stage_idx
                            current_stage_info = stage
                            env.obstacle_mode = stage['mode']
                            env.num_obstacles = stage.get('obs', env.num_obstacles)
                            if stage['mode'] == 'block':
                                env.block_size_range = stage.get('block_range', (2, 5))
                            
                            epsilon_boost = stage.get('epsilon_boost', 0.0)
                            
                            if epsilon_reset_strategy == 'full':
                                agent.epsilon = 1.0
                                tqdm.write(f"  Epsilon reset: {old_epsilon:.4f} → 1.0 (Full reset)")
                            elif epsilon_reset_strategy == 'partial':
                                agent.epsilon = min(1.0, agent.epsilon + epsilon_boost)
                                tqdm.write(f"  Epsilon boost: {old_epsilon:.4f} → {agent.epsilon:.4f} (+{epsilon_boost})")
                            elif epsilon_reset_strategy == 'adaptive':
                                # Adaptive: adjust based on difficulty changes
                                if stage_idx > 0:
                                    prev_mode = curriculum_config['stages'][stage_idx-1]['mode']
                                    if prev_mode == 'point' and stage['mode'] == 'block':
                                        # From point to block: significant boost
                                        agent.epsilon = min(1.0, agent.epsilon + 0.4)
                                        tqdm.write(f"  Epsilon adaptive boost: {old_epsilon:.4f} → {agent.epsilon:.4f} (Point→Block)")
                                    else:
                                        # Same type difficulty increase: small boost
                                        agent.epsilon = min(1.0, agent.epsilon + epsilon_boost)
                                        tqdm.write(f"  Epsilon adaptive boost: {old_epsilon:.4f} → {agent.epsilon:.4f}")
                            # else: 'none' - no epsilon adjustment
                            
                            tqdm.write(f"\n{'='*60}")
                            tqdm.write(f"Entering {stage['description']} (Episode {episode})")
                            tqdm.write(f"  Obstacle mode: {env.obstacle_mode}")
                            tqdm.write(f"  Number of obstacles: {env.num_obstacles}")
                            if env.obstacle_mode == 'block':
                                tqdm.write(f"  Block size: {env.block_size_range}")
                            tqdm.write(f"{'='*60}\n")
                        break
            
            env.reset_dynamic(seed=None) 
            state = env.reset()
            done = False
            total_reward = 0
            episode_steps = 0
            losses = []
            q_values = []
            reached_goal = False
            # Train paths between multiple goal points within each episode
            goal_pairs = zip(env.goal_list, env.goal_list[1:])
            for start_goal, next_goal in goal_pairs:
                env.goal = start_goal
                while not done:
                    state_tensor = torch.tensor(state, dtype=torch.float32).to(agent.device)
                    action = agent.select_action(state_tensor)
                    next_state, reward, done = env.step(action)
                    agent.store_transition(state, action, reward, next_state, done)
                    loss = agent.train()
                    state = next_state
                    total_reward += reward
                    episode_steps += 1
                    step_counter += 1
                    if loss is not None:
                        losses.append(loss.item())
                    with torch.no_grad():
                        q_values.append(agent.q_network(state_tensor.unsqueeze(0)).cpu().numpy().flatten())
                    if episode_steps >= max_steps:
                        done = True
                    if env.current_goal_index >= len(env.goal_list) - 1:
                        goal_distance = np.linalg.norm(np.array(env.agent_pos) - np.array(env.goal))
                        if goal_distance < 0.5:
                            reached_goal = True
            success_history.append(1 if reached_goal else 0)
            recent_window = min(100, len(success_history))
            success_rate = (sum(success_history[-recent_window:]) / recent_window) * 100
            agent.decay_epsilon()

            writer.add_scalar("Reward/Episode", total_reward, episode)
            writer.add_scalar("Epsilon", agent.epsilon, episode)
            writer.add_scalar("Success_Rate", success_rate, episode)

            if curriculum_learning and current_stage_info:
                stage_name = current_stage_info['description'].replace(' ', '_')
                writer.add_scalar(f"Stage_{stage_name}/Reward", total_reward, episode)
                writer.add_scalar(f"Stage_{stage_name}/Success_Rate", success_rate, episode)
            
            if losses:
                avg_loss = np.mean(losses)
                writer.add_scalar("Loss/Episode", avg_loss, episode)
                all_losses.append(avg_loss)
            all_rewards.append(total_reward)
            all_epsilons.append(agent.epsilon)

            pbar.set_postfix({
                "Total Reward": f"{total_reward:.1f}",
                "Success Rate": f"{success_rate:.1f}%",
                "Epsilon": f"{agent.epsilon:.3f}"
            })
            pbar.update(1)
            if total_reward > best_reward:
                best_reward = total_reward
                agent.save_model(save_path)
            if episode % 10 == 0:
                writer.flush()
        agent.save_model(save_path)
        writer.close()

    print(f"Training complete, the model has been saved to {save_path}")

    np.save(os.path.join(log_dir, "rewards.npy"), all_rewards)
    np.save(os.path.join(log_dir, "losses.npy"), all_losses)
    np.save(os.path.join(log_dir, "epsilons.npy"), all_epsilons)

if __name__ == "__main__":
    files = "grid_map2.txt"
    grid_map = np.loadtxt(files, dtype=int)
    obst = [[50, 30]]  
    nodes = [(80, 110), (71, 98), (35, 94), (42, 78), (82, 69), (75, 25), (67, 14), (31, 19), (24, 27), (13, 35), (13, 94)]
    # nodes = [(133, 45), (68, 55), (60, 163), (17, 158)]
    # nodes = [(7, 27), (7, 36), (28, 36), (34, 21), (43, 26), (53, 31), (62, 54), (61, 68)]
    
    # Set to True to enable curriculum learning, False to use fixed configuration
    USE_CURRICULUM = True
    
    if USE_CURRICULUM:
        # Curriculum learning: progressive difficulty increase
        curriculum_config = {
            # ===== Epsilon adjustment strategies =====
            # 'none': Do not adjust epsilon (may lead to insufficient exploration later)
            # 'partial': Increase epsilon at each new stage (recommended)
            # 'full': Fully reset epsilon to 1.0 at each new stage (sufficient exploration but may be unstable)
            # 'adaptive': Adaptive adjustment (automatically determine the increase based on difficulty changes)
            'epsilon_reset_strategy': 'adaptive',
            
            'stages': [
                {
                    'episodes': 300,
                    'mode': 'point',
                    'obs': 30,
                    'epsilon_boost': 0.0,
                    'description': 'Stage 1 - Simple Point Obstacles'
                },
                {
                    'episodes': 250,
                    'mode': 'point',
                    'obs': 40,
                    'epsilon_boost': 0.15,
                    'description': 'Stage 2 - Medium Point Obstacles'
                },
                {
                    'episodes': 200,
                    'mode': 'block',
                    'block_range': (2, 3),
                    'obs': 30,
                    'epsilon_boost': 0.4,
                    'description': 'Stage 3 - Small Block Obstacles'
                },
                {
                    'episodes': 150,
                    'mode': 'block',
                    'block_range': (2, 4),
                    'obs': 35,
                    'epsilon_boost': 0.2,
                    'description': 'Stage 4 - Medium Block Obstacles'
                },
                {
                    'episodes': 100,
                    'mode': 'block',
                    'block_range': (3, 5),
                    'obs': 40,
                    'epsilon_boost': 0.25,
                    'description': 'Stage 5 - Large Block Obstacles'
                }
            ]
        }
        
        env = GridWorld(
            grid_map, 
            start=nodes[0], 
            goal=nodes[-1], 
            goal_list=nodes, 
            obs=30,
            dobs=obst[0][1],
            obstacle_mode='point',
            block_size_range=(2, 5)
        )
        
        total_episodes = sum(stage['episodes'] for stage in curriculum_config['stages'])
        save_path = "models/dqn_model_all.pth"
        
    else:
        env = GridWorld(
            grid_map, 
            start=nodes[0], 
            goal=nodes[-1], 
            goal_list=nodes, 
            obs=obst[0][0],
            dobs=obst[0][1],
            obstacle_mode='block',
            block_size_range=(2, 3)
        )
        curriculum_config = None
        total_episodes = 1000
        save_path = "dqn_model_block.pth"
    agent = RainbowDQN(
        env=env,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        learning_rate=0.001,
        batch_size=64,
        memory_size=200000
    )

    try:
        print("Using device:", agent.device)
        print("torch.cuda.is_available():", torch.cuda.is_available())
        try:
            print("CUDA device count:", torch.cuda.device_count())
            if torch.cuda.is_available():
                print("CUDA device name(0):", torch.cuda.get_device_name(0))
        except Exception as _e:
            print("CUDA device query failed:", _e)
    except Exception as _e:
        print("Warning: failed to print device info:", _e)
    
    if not USE_CURRICULUM:
        print("=" * 80)
        print("Fixed Configuration Training Mode")
        print("=" * 80)
        print(f"Obstacle Mode: {env.obstacle_mode}")
        print(f"Number of Static Obstacles: {env.num_obstacles}")
        if env.obstacle_mode == 'block':
            print(f"Block Size Range: {env.block_size_range}")
        print("=" * 80)
    
    train_agent(
        env, 
        agent, 
        episodes=total_episodes, 
        max_steps=5000, 
        save_path=save_path, 
        log_dir="runs",
        curriculum_learning=USE_CURRICULUM,
        curriculum_config=curriculum_config
    )