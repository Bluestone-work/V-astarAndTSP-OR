import pygame
import numpy as np
import torch
from tqdm import tqdm
import time
import os
import imageio
import matplotlib
import math
import psutil
from threading import Thread, Event
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from environment import GridWorld

def visualize_training(env, agent, episodes=1000, max_steps=1000, save_path="dqn_model.pth"):
    pygame.init()
    screen_info = pygame.display.Info()
    screen_width = screen_info.current_w
    screen_height = screen_info.current_h
    grid_size = min(screen_width // env.cols, screen_height // env.rows)
    window_width = env.cols * grid_size
    window_height = env.rows * grid_size
    os.environ['SDL_VIDEO_CENTERED'] = '1'
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Training Visualization")
    clock = pygame.time.Clock()

    pygame.font.init()
    font = pygame.font.SysFont("Arial", 24)

    best_reward = -float('inf')

    for episode in range(episodes):
        env.reset_dynamic(num_obstacles=min(5 + episode // 100, 20))
        state = env.reset()
        done = False
        total_reward = 0
        path = [env.agent_pos]
        steps = 0

        while not done and steps < max_steps:
            screen.fill((255, 255, 255))
            for i in range(env.rows):
                for j in range(env.cols):
                    color = (255, 255, 255)
                    if env.grid_map[i, j] == 1:
                        color = (0, 0, 0)
                    pygame.draw.rect(screen, color, (j * grid_size, i * grid_size, grid_size, grid_size))
            pygame.draw.rect(
                screen,
                (0, 0, 255),
                (env.goal[1] * grid_size, env.goal[0] * grid_size, grid_size, grid_size)
            )
            for pos in path:
                pygame.draw.circle(
                    screen,
                    (200, 200, 200), 
                    (int(pos[1] * grid_size + grid_size // 2), int(pos[0] * grid_size + grid_size // 2)),
                    5
                )
            car_x = int(env.agent_pos[1] * grid_size + grid_size // 2)
            car_y = int(env.agent_pos[0] * grid_size + grid_size // 2)
            car_radius = grid_size // 4
            pygame.draw.circle(screen, (255, 0, 0), (car_x, car_y), car_radius)
            state_tensor = torch.tensor(state, dtype=torch.float32).to(agent.device)
            action = agent.select_action(state_tensor)
            next_state, reward, done = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.train()
            state = next_state
            total_reward += reward
            path.append(env.agent_pos)
            steps += 1
            reward_text = font.render(f"Episode: {episode}", True, (0, 0, 0))
            screen.blit(reward_text, (10, 10))
            total_reward_text = font.render(f"Total Reward: {total_reward:.2f}", True, (0, 0, 0))
            screen.blit(total_reward_text, (10, 40))

            pygame.display.flip() 

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            clock.tick(60)
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save_model(save_path)
        if episode % 10 == 0:
            print(f"Episode {episode}: Total Reward = {total_reward:.2f}")
    pygame.quit()
    print("Training Completed!")

def check_trained_model(env, agent, model_path, max_steps=1000): 

    agent.load_model(model_path)
    start_time = time.time()
    print(f"Loaded model from {model_path}")

    agent.epsilon = 0
    pygame.init()
    screen_info = pygame.display.Info()
    screen_width = screen_info.current_w
    screen_height = screen_info.current_h
    grid_size = min(screen_width // env.cols, screen_height // env.rows)
    window_width = env.cols * grid_size
    window_height = env.rows * grid_size
    os.environ['SDL_VIDEO_CENTERED'] = '1'
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Trained Model Test")
    clock = pygame.time.Clock()
    pygame.font.init()
    font = pygame.font.SysFont("Arial", 24)

    state = env.reset()
    done = False
    total_reward = 0
    path = [env.agent_pos]
    steps = 0
    
    clicked_cells = set()

    for obstacle in env.dynamic_obstacles:
        obstacle['path'] = [tuple(obstacle['position'])] 

    while not done and steps < max_steps:

        screen.fill((255, 255, 255)) 
        dynamic_positions = set([tuple(obstacle['position']) for obstacle in env.dynamic_obstacles])

        for i in range(env.rows):
            for j in range(env.cols):
                color = (255, 255, 255)
                if env.grid_map[i, j] == 1:
                    if (i, j) in clicked_cells:
                        color = (255, 0, 0)
                    elif (i, j) in dynamic_positions:
                        color = (210, 180, 140) 
                    else:
                        color = (0, 0, 0)
                pygame.draw.rect(screen, color, (j * grid_size, i * grid_size, grid_size, grid_size))


        for obstacle in env.dynamic_obstacles:
            for prev_pos in obstacle['path']:
                prev_x, prev_y = prev_pos
                pygame.draw.circle(screen, (210, 180, 140), (prev_y * grid_size + grid_size // 2, prev_x * grid_size + grid_size // 2), 2)

        pygame.draw.rect(
            screen,
            (0, 0, 255),
            (env.goal[1] * grid_size, env.goal[0] * grid_size, grid_size, grid_size)
        )

        for pos in path:
            pygame.draw.circle(screen, (200, 200, 200), (int(pos[1] * grid_size + grid_size // 2), int(pos[0] * grid_size + grid_size // 2)), 2)

        car_x = int(env.agent_pos[1] * grid_size + grid_size // 2)
        car_y = int(env.agent_pos[0] * grid_size + grid_size // 2)
        car_radius = grid_size // 4

        pygame.draw.circle(screen, (255, 0, 0), (car_x, car_y), car_radius)

        action = agent.select_action(state)
        next_state, reward, done = env.step(action)

        for obstacle in env.dynamic_obstacles:
            obstacle['path'].append(tuple(obstacle['position']))

        pygame.display.flip() 

        state = next_state
        total_reward += reward
        path.append(env.agent_pos) 
        steps += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                grid_x = mouse_x // grid_size
                grid_y = mouse_y // grid_size
                if env.grid_map[grid_y, grid_x] == 0: 
                    env.grid_map[grid_y, grid_x] = 1
                    clicked_cells.add((grid_y, grid_x))
                    print(f"Obstacle added at ({grid_y}, {grid_x})")
        
        clock.tick(30)

    end_time = time.time()
    pygame.quit() 

    # 输出测试结果
    print(f"Test Completed. Total Reward: {total_reward}, Steps Taken: {steps}")
    print(f"仿真完成，耗时 {end_time - start_time:.2f} 秒")


def _render_env_frame(env, path, title=None, obstacle_paths=None):
    from matplotlib.patches import Rectangle
    
    fig = Figure(figsize=(16, 7), dpi=100)
    canvas = FigureCanvas(fig)

    ax_global = fig.add_subplot(121)
    ax_local = fig.add_subplot(122)

    grid_image = np.ones((env.rows, env.cols, 3))  # RGB format
    
    dynamic_positions = set([tuple(obs.get("position")) for obs in getattr(env, "dynamic_obstacles", []) 
                            if obs.get("position") is not None])
    
    for i in range(env.rows):
        for j in range(env.cols):
            if env.grid_map[i, j] == 1:
                if (i, j) in dynamic_positions:
                    grid_image[i, j] = [210/255, 180/255, 140/255]
                else:
                    grid_image[i, j] = [0, 0, 0]
            else:
                grid_image[i, j] = [1, 1, 1]
    
    ax_global.imshow(grid_image, origin="upper", extent=[-0.5, env.cols-0.5, env.rows-0.5, -0.5])

    goal = getattr(env, "goal", None)
    if goal is not None:
        goal_rect = Rectangle((goal[1]-0.5, goal[0]-0.5), 1, 1, 
                             linewidth=0, edgecolor='none', facecolor='blue', zorder=3)
        ax_global.add_patch(goal_rect)

    if path and len(path) > 1:
        path_rows = [pos[0] for pos in path[:-1]]
        path_cols = [pos[1] for pos in path[:-1]]
        ax_global.scatter(path_cols, path_rows, c='lightgray', s=15, alpha=0.7, zorder=4)
    if path:
        agent_pos = path[-1]
        ax_global.scatter(agent_pos[1], agent_pos[0], c='red', s=150, marker='o', zorder=5, edgecolors='darkred', linewidths=2)

        view_rect = Rectangle((agent_pos[1]-3.5, agent_pos[0]-3.5), 7, 7,
                             linewidth=3, edgecolor='lime', facecolor='none', 
                             linestyle='--', zorder=6)
        ax_global.add_patch(view_rect)

        ax_global.text(agent_pos[1]-3.5, agent_pos[0]-4.2, 'Local View', 
                      fontsize=8, color='lime', fontweight='bold',
                      ha='left', va='bottom', zorder=7,
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

    ax_global.set_xlim(-0.5, env.cols - 0.5)
    ax_global.set_ylim(env.rows - 0.5, -0.5)
    ax_global.set_xticks([])
    ax_global.set_yticks([])
    ax_global.set_aspect('equal')
    ax_global.set_title("Global View", fontsize=11, fontweight='bold', pad=10)

    ax_global.set_xlim(-0.5, env.cols - 0.5)
    ax_global.set_ylim(env.rows - 0.5, -0.5)
    ax_global.set_xticks([])
    ax_global.set_yticks([])
    ax_global.set_aspect('equal')
    ax_global.set_title("Global View", fontsize=11, fontweight='bold', pad=10)

    if path:
        agent_pos = path[-1]
        
        local_grid = np.ones((7, 7, 3)) 
        x_min, x_max = max(0, agent_pos[0] - 3), min(env.rows, agent_pos[0] + 4)
        y_min, y_max = max(0, agent_pos[1] - 3), min(env.cols, agent_pos[1] + 4)
        r_min, r_max = 3 - (agent_pos[0] - x_min), 3 + (x_max - agent_pos[0])
        c_min, c_max = 3 - (agent_pos[1] - y_min), 3 + (y_max - agent_pos[1])

        for i in range(x_min, x_max):
            for j in range(y_min, y_max):
                local_i = i - x_min + r_min
                local_j = j - y_min + c_min
                if env.grid_map[i, j] == 1:
                    if (i, j) in dynamic_positions:
                        local_grid[local_i, local_j] = [210/255, 180/255, 140/255] 
                    else:
                        local_grid[local_i, local_j] = [0, 0, 0] 
        
        predicted_positions = []
        if hasattr(env, 'dynamic_obstacles') and hasattr(env, 'obstacle_history'):
            for idx, obstacle in enumerate(env.dynamic_obstacles):
                if hasattr(env, '_predict_obstacle_trajectory'):
                    prediction_steps = max(3, obstacle.get('speed', 1) * 2)
                    predicted_pos_list = env._predict_obstacle_trajectory(obstacle, idx, prediction_steps)
                    for pred_pos in predicted_pos_list:
                        dx = pred_pos[0] - agent_pos[0]
                        dy = pred_pos[1] - agent_pos[1]
                        if abs(dx) <= 3 and abs(dy) <= 3:
                            local_x = dx + 3
                            local_y = dy + 3
                            local_grid[local_x, local_y] = [0.8, 0.3, 0.8]
                            predicted_positions.append((local_y, local_x))

        ax_local.imshow(local_grid, origin="upper", extent=[-0.5, 6.5, 6.5, -0.5])

        for i in range(8):
            ax_local.axhline(i - 0.5, color='gray', linewidth=0.3, alpha=0.4)
            ax_local.axvline(i - 0.5, color='gray', linewidth=0.3, alpha=0.4)

        ax_local.scatter(3, 3, c='red', s=400, marker='o', zorder=5, edgecolors='darkred', linewidths=3)

        if goal is not None:
            dx_goal = goal[0] - agent_pos[0]
            dy_goal = goal[1] - agent_pos[1]
            goal_distance = np.sqrt(dx_goal**2 + dy_goal**2)
            if goal_distance > 0:

                dx_norm = dx_goal / goal_distance
                dy_norm = dy_goal / goal_distance

                ax_local.arrow(3, 3, dy_norm * 1.8, dx_norm * 1.8, 
                             head_width=0.4, head_length=0.4, fc='blue', ec='blue', 
                             linewidth=3, zorder=6, alpha=0.8)
        if predicted_positions:
            pred_cols = [pos[0] for pos in predicted_positions]
            pred_rows = [pos[1] for pos in predicted_positions]
            ax_local.scatter(pred_cols, pred_rows, c='purple', s=120, marker='x', 
                           linewidths=3, zorder=4, label='Predicted')
        
        ax_local.set_xlim(-0.5, 6.5)
        ax_local.set_ylim(6.5, -0.5)
        ax_local.set_xticks(range(7))
        ax_local.set_yticks(range(7))
        ax_local.set_xticklabels([])
        ax_local.set_yticklabels([])
        ax_local.set_aspect('equal')
        ax_local.set_title("Local View (7×7) + Prediction", fontsize=11, fontweight='bold', pad=10)

        if predicted_positions:
            ax_local.legend(loc='upper right', fontsize=7, framealpha=0.9)

    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    canvas.draw()
    width, height = canvas.get_width_height()
    buffer = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
    buffer = buffer.reshape(height, width, 4)
    image = buffer[..., 1:].copy()
    fig.clear()
    return image

try:
    import pynvml
    _GPU_AVAILABLE = True
    pynvml.nvmlInit()
    _GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0) 
except Exception as e:
    _GPU_AVAILABLE = False
    _GPU_HANDLE = None

def _path_length(path):
    if len(path) < 2:
        return 0.0
    return sum(math.dist(path[i], path[i+1]) for i in range(len(path) - 1))

def _path_smoothness(path):
    if len(path) < 3:
        return 0.0
    angles = []
    for i in range(1, len(path) - 1):
        v1 = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
        v2 = (path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        mag1 = math.hypot(*v1)
        mag2 = math.hypot(*v2)
        if mag1 > 0 and mag2 > 0:
            cos_theta = max(-1, min(1, dot / (mag1 * mag2)))
            angle = math.acos(cos_theta)
            angles.append(angle)
    return sum(angles) / len(angles) if angles else 0.0