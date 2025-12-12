# gym_gridworld_env.py
"""
Encapsulate environment.GridWorld into a Gym environment:
- Obstacle generation: call the internal logic of GridWorld.reset_dynamic / reset
- Obstacle movement: call the internal logic of GridWorld.step (dynamic_obstacles are already updated inside)
"""

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYMNASIUM_API = True
except ImportError:
    import gym
    from gym import spaces
    GYMNASIUM_API = False

from environment import GridWorld


class GridWorldGymEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        grid_map: np.ndarray,
        nodes,
        obs: int = 50,
        dobs: int = 30,
        obstacle_mode: str = "point",
        block_size_range=(2, 5),
        max_steps: int = 1000,
        use_dynamic_reset: bool = True,
        seed: int | None = None,
    ):
        super().__init__()

        self.grid_map = grid_map
        self.nodes = list(nodes)
        self.start = self.nodes[0]
        self.goal = self.nodes[1] if len(self.nodes) > 1 else self.nodes[0]
        self.goal_list = self.nodes[1:]

        self.use_dynamic_reset = use_dynamic_reset
        self._seed_value = seed

        self.env = GridWorld(
            grid_map=self.grid_map,
            start=self.start,
            goal=self.goal,
            goal_list=self.goal_list,
            max_steps=max_steps,
            obs=obs,
            dobs=dobs,
            obstacle_mode=obstacle_mode,
            block_size_range=block_size_range,
        )

        if self.use_dynamic_reset:
            self.env.reset_dynamic(seed=self._seed_value)
        else:
            # resample_obstacles=True
            self.env.reset(resample_obstacles=True, seed=self._seed_value)

        state_hist = self.env.get_state()  # (n_frames, state_dim)
        state_hist = np.asarray(state_hist, dtype=np.float32)

        if state_hist.ndim == 1:
            state_hist = state_hist[None, :]

        self.n_frames, self.state_dim = state_hist.shape

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(9)

    def _state_hist_to_obs(self, state_hist) -> np.ndarray:

        arr = np.asarray(state_hist, dtype=np.float32)
        if arr.ndim == 2:
            last_frame = arr[-1]
        elif arr.ndim == 1:
            last_frame = arr
        else:
            raise ValueError(f"Unexpected state shape from GridWorld.get_state(): {arr.shape}")

        last_frame = np.nan_to_num(last_frame, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if last_frame.shape[0] > 0 and hasattr(self.env, 'rows') and hasattr(self.env, 'cols'):
            max_distance = np.sqrt(self.env.rows**2 + self.env.cols**2)
            if max_distance > 0:
                last_frame[0] = last_frame[0] / max_distance
        
        return last_frame

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self._seed_value = seed

        if self.use_dynamic_reset:
            self.env.reset_dynamic(seed=self._seed_value)
        else:
            self.env.reset(resample_obstacles=True, seed=self._seed_value)

        self.env.start = self.start 
        self.env.agent_pos = self.start
        self.env.current_goal_index = 0 
        
        if self.env.goal_list and len(self.env.goal_list) > 0:
            self.env.goal = self.env.goal_list[0]
            self.env.distance = np.linalg.norm(
                np.array(self.env.agent_pos) - np.array(self.env.goal)
            )
        else:
            raise ValueError("The goal_list is empty, unable to set a target point!")

        state_hist = self.env.get_state()
        obs = self._state_hist_to_obs(state_hist)

        info = {
            "agent_pos": getattr(self.env, "agent_pos", None),
            "goal": getattr(self.env, "goal", None),
        }

        if GYMNASIUM_API:
            return obs, info
        else:
            return obs

    def step(self, action):
        action = int(action)

        state_hist, reward, done = self.env.step(action)

        obs = self._state_hist_to_obs(state_hist)
        reward = float(reward)

        terminated = bool(done)
        truncated = False

        success = False
        if done and reward > 0:
            if (
                hasattr(self.env, "goal_list")
                and hasattr(self.env, "current_goal_index")
                and len(self.env.goal_list) > 0
            ):
                success = self.env.current_goal_index >= len(self.env.goal_list) - 1

        info = {
            "agent_pos": getattr(self.env, "agent_pos", None),
            "goal": getattr(self.env, "goal", None),
            "distance": getattr(self.env, "distance", None),
            "current_goal_index": getattr(self.env, "current_goal_index", 0),
            "success": success,
        }

        if GYMNASIUM_API:
            return obs, reward, terminated, truncated, info
        else:
            done_flag = terminated or truncated
            return obs, reward, done_flag, info

    def render(self):
        print(f"Agent: {self.env.agent_pos}, Goal: {self.env.goal}")

    def close(self):
        pass