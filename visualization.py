try:
    import pygame
except ImportError:
    pygame = None  # Headless frame/GIF rendering does not require pygame.
import numpy as np
import math
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

class _NavigationWindow:
    """Display shared-runner transitions; rendering is outside paper timings."""
    def __init__(self, env, title, fps=30, editable=False):
        if pygame is None:
            raise RuntimeError('Interactive windows require pygame; install it or use headless evaluation.')
        pygame.init()
        info = pygame.display.Info()
        self.size = max(1, min(info.current_w // env.cols, info.current_h // env.rows))
        self.screen = pygame.display.set_mode((env.cols*self.size, env.rows*self.size))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.editable = editable
        self.edited = False
        self.total_reward = 0.
        self.path = []

    def draw(self, env, path, steps, reward):
        if steps <= 1:
            self.total_reward = 0.
        self.total_reward += reward
        self.path = list(path)
        dynamic = {tuple(o['position']) for o in env.dynamic_obstacles}
        for r in range(env.rows):
            for c in range(env.cols):
                color = (210, 180, 140) if (r, c) in dynamic else ((0, 0, 0) if env.grid_map[r, c] else (255, 255, 255))
                pygame.draw.rect(self.screen, color, (c*self.size, r*self.size, self.size, self.size))
        for r, c in path:
            pygame.draw.circle(self.screen, (200, 200, 200),
                               (int((c+.5)*self.size), int((r+.5)*self.size)), max(1, self.size//5))
        r, c = env.goal
        pygame.draw.rect(self.screen, (0, 0, 255), (c*self.size, r*self.size, self.size, self.size))
        r, c = env.agent_pos
        pygame.draw.circle(self.screen, (255, 0, 0),
                           (int((c+.5)*self.size), int((r+.5)*self.size)), max(1, self.size//3))
        pygame.display.set_caption(f"Step {steps} | Reward {self.total_reward:.2f} | replans {env.global_replans}")
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt('Navigation window closed')
            if self.editable and event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                r, c = y//self.size, x//self.size
                if (0 <= r < env.rows and 0 <= c < env.cols and
                        (r, c) != tuple(env.agent_pos) and not env.grid_map[r, c]):
                    env.base_grid_map[r, c] = 1
                    env.rebuild_grid_map()
                    self.edited = True
        self.clock.tick(self.fps)


def visualize_training(env, agent, episodes=1000, max_steps=2000, save_path="dqn_model.pth",
                       *, seed=None, log_dir="runs"):
    """Render the canonical training loop, including replay and event replanning."""
    from train import train_agent
    window = _NavigationWindow(env, 'Training visualization', fps=60)
    original_step = env.step
    def rendered_step(action):
        result = original_step(action)
        path = [tuple(env.start)] if env.steps == 1 else window.path.copy()
        path.append(tuple(env.agent_pos))
        window.draw(env, path, env.steps, result[1])
        return result
    env.step = rendered_step
    try:
        return train_agent(env, agent, episodes=episodes, max_steps=max_steps,
                           save_path=save_path, seed=seed, log_dir=log_dir)
    finally:
        env.step = original_step
        pygame.quit()


def check_trained_model(env, agent, model_path, max_steps=2000):
    """Interactive display of the shared evaluator; UI time is not a benchmark."""
    from evaluate import run_rl_algorithm_without_viz
    agent.load_model(model_path)
    for key in ('robot_radius', 'cell_size', 'extra_margin'):
        if key in agent.loaded_environment_specification:
            setattr(env, key, agent.loaded_environment_specification[key])
    window = _NavigationWindow(env, 'Trained model test', editable=True)
    original_select = agent.select_action
    def select_after_edits(state):
        if window.edited:
            # Replace the current observation, retaining three past control frames.
            env.history.pop()
            env._state_step = None
            state = env.get_state()
            window.edited = False
        return original_select(state)
    agent.select_action = select_after_edits
    try:
        success, path, steps, metrics = run_rl_algorithm_without_viz(
            env, agent, max_steps=max_steps, on_step=window.draw)
        print(f"Success: {success}; termination: {metrics['termination']}; steps: {steps}")
        print(f"Interactive display wall time (includes rendering): {metrics['episode_wall_time_s']:.2f} s")
        return success, path, steps
    finally:
        agent.select_action = original_select
        pygame.quit()


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
        
        # Draw the exact causal landing candidates used by the policy.
        predicted_positions = []
        for pred_pos in getattr(env, 'predicted_landing_points', []):
            dr, dc = pred_pos[0] - agent_pos[0], pred_pos[1] - agent_pos[1]
            if abs(dr) <= 3 and abs(dc) <= 3 and not env.grid_map[tuple(pred_pos)]:
                local_x, local_y = dr + 3, dc + 3
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
