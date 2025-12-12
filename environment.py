import numpy as np
import random
from collections import deque
import torch
import os
import seaborn as sns
import pickle

class GridWorld:
    def __init__(self, grid_map, start, goal, max_steps=1000, goal_list=None, obs=30, dobs=10,
                 obstacle_mode='point', block_size_range=(2, 5)):
        """
            Initialize GridWorld Environment

            Args:
            grid_map: Base map
            start: Starting point
            goal: Goal point
            max_steps: Maximum number of steps
            goal_list: List of path nodes
            obs: Number of static obstacles (count, not pixels)
            dobs: Number of dynamic obstacles
            obstacle_mode: Obstacle mode
            - 'point': Single-point distribution (default)
            - 'block': Block distribution (similar to Tetris)
            block_size_range: Size range of block obstacles (min, max), only effective in block mode
        """
        self.original_grid_map = grid_map.copy()
        self.base_grid_map = grid_map.copy()
        self.grid_map = grid_map.copy()
        
        self.rows, self.cols = grid_map.shape
        self.max_steps = max_steps
        self.goal_list = goal_list if goal_list else [] 
        self.history = deque(maxlen=4) 
        self.num_obstacles = obs
        self.num_dobs = dobs
        
        # Obstacle mode configuration
        self.obstacle_mode = obstacle_mode
        self.block_size_range = block_size_range
        
        # Add dynamic reward decay
        self.reward_decay = 0.99

        self.direction_vectors = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1),
            'up_left': (-1, -1),
            'up_right': (-1, 1),
            'down_left': (1, -1),
            'down_right': (1, 1)
        }
        self.turn_probability = 0.25
        self.behavior_options = ['random_walk', 'patrol', 'wander']
        
        # Define block obstacle shape templates
        self.block_shapes = {
            'I': [(0, 0), (0, 1), (0, 2), (0, 3)],           # I-shape
            'O': [(0, 0), (0, 1), (1, 0), (1, 1)],           # O-shape
            'T': [(0, 0), (0, 1), (0, 2), (1, 1)],           # T-shape
            'L': [(0, 0), (1, 0), (2, 0), (2, 1)],           # L-shape
            'J': [(0, 1), (1, 1), (2, 1), (2, 0)],           # J-shape
            'Z': [(0, 0), (0, 1), (1, 1), (1, 2)],           # Z-shape
            'S': [(0, 1), (0, 2), (1, 0), (1, 1)],           # S-shape
            'cross': [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)], # Cross-shape
            'small_rect': [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],  # Small rectangle
            'big_rect': [(i, j) for i in range(3) for j in range(3)]  # Large rectangle 3x3
        }
        
        if start is not None and goal is not None:
            self.start = start
            self.goal = goal
            self.agent_pos = self.start
            self.distance = np.linalg.norm(np.array(self.start) - np.array(self.goal))
        else:
            self.reset_dynamic()

    def _generate_block_obstacle(self, base_pos, safe_zone, occupied_positions):
        """
        Generate a block obstacle
        
        Args:
            base_pos: Base position of the block (x, y)
            safe_zone: Area where obstacles are not allowed
            occupied_positions: Set of already occupied positions
        
        Returns:
            list: List of all positions occupied by the block obstacle, empty if placement is not possible
        """
        shape_name = random.choice(list(self.block_shapes.keys()))
        shape_template = self.block_shapes[shape_name]
        rotation = random.choice([0, 90, 180, 270])
        rotated_shape = []
        for dx, dy in shape_template:
            if rotation == 0:
                rotated_shape.append((dx, dy))
            elif rotation == 90:
                rotated_shape.append((-dy, dx))
            elif rotation == 180:
                rotated_shape.append((-dx, -dy))
            else:  # 270
                rotated_shape.append((dy, -dx))
        block_positions = []
        base_x, base_y = base_pos
        
        for dx, dy in rotated_shape:
            new_x, new_y = base_x + dx, base_y + dy
            if new_x < 1 or new_x >= self.rows - 1 or new_y < 1 or new_y >= self.cols - 1:
                return []  
            if self.original_grid_map[new_x, new_y] == 1:
                return []  
            if (new_x, new_y) in safe_zone:
                return []  
            if (new_x, new_y) in occupied_positions:
                return []  
            block_positions.append((new_x, new_y))
        
        return block_positions

    def _generate_static_obstacles_point_mode(self, valid_positions, count):
        """Point mode: generate count single-point obstacles"""
        if count > len(valid_positions):
            count = len(valid_positions)
        
        if count > 0:
            return random.sample(valid_positions, count)
        return []

    def _generate_static_obstacles_block_mode(self, valid_positions, count, safe_zone):
        """
        Block mode: generate count block obstacles
        
        Args:
            valid_positions: List of available positions
            count: Number of block obstacles to generate
            safe_zone: Safe zone
        
        Returns:
            list: List of all positions occupied by the block obstacles
        """
        all_obstacle_positions = []
        occupied = set()
        attempts_per_block = 50  
        
        for block_idx in range(count):
            placed = False
            for attempt in range(attempts_per_block):
                if not valid_positions:
                    break
                base_pos = random.choice(valid_positions)
                block_positions = self._generate_block_obstacle(base_pos, safe_zone, occupied)
                if block_positions:
                    all_obstacle_positions.extend(block_positions)
                    occupied.update(block_positions)
                    for pos in block_positions:
                        if pos in valid_positions:
                            valid_positions.remove(pos)
                    
                    placed = True
                    break
            if not placed:
                if valid_positions:
                    fallback_pos = random.choice(valid_positions)
                    all_obstacle_positions.append(fallback_pos)
                    occupied.add(fallback_pos)
                    valid_positions.remove(fallback_pos)
        return all_obstacle_positions
    
    def reset_dynamic(self, seed=None):
        """
        Args:
            seed: Random seed to control the randomness of environment generation
                  - None (default): Completely random generation, suitable for training to enhance environment diversity
                  - Integer: Use a fixed seed for generation, suitable for testing/evaluation to ensure reproducibility
        
        Usage recommendations:
            Training: reset_dynamic(seed=None)  # Generate different environments each time
            Testing: reset_dynamic(seed=42)    # Fixed environment for fair comparison
        """
        # If a seed is provided, set the random seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.base_grid_map = self.original_grid_map.copy()
        self.dynamic_obstacles = []
        def ensure_tuple(pos):
            if isinstance(pos, (list, np.ndarray)):
                pos = tuple(pos)
            if isinstance(pos, tuple) and len(pos) == 2:
                return pos
            else:
                raise ValueError(f"Invalid position format: {pos}, type: {type(pos)}")
        forbidden_positions = set()
        forbidden_positions.add(ensure_tuple(self.start))
        forbidden_positions.add(ensure_tuple(self.goal))
        if self.goal_list:
            for goal_point in self.goal_list:
                forbidden_positions.add(ensure_tuple(goal_point))
        safe_zone = set()
        for fx, fy in forbidden_positions:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    safe_zone.add((fx + dx, fy + dy))
        valid_positions_for_static = [
            (i, j) for i in range(1, self.rows - 1) for j in range(1, self.cols - 1)
            if self.original_grid_map[i, j] == 0 and (i, j) not in safe_zone
        ]
        if self.obstacle_mode == 'point':
            random_obstacles = self._generate_static_obstacles_point_mode(
                valid_positions_for_static.copy(), 
                self.num_obstacles
            )
        elif self.obstacle_mode == 'block':
            random_obstacles = self._generate_static_obstacles_block_mode(
                valid_positions_for_static.copy(), 
                self.num_obstacles,
                safe_zone
            )
        else:
            raise ValueError(f"Unknown obstacle mode: {self.obstacle_mode}, please use 'point' or 'block'")
        occupied_by_static = set(random_obstacles)
        for x, y in random_obstacles:
            self.base_grid_map[x, y] = 1
        # ========== Generate dynamic obstacles ==========
        valid_positions_for_dynamic = [
            (i, j) for i in range(1, self.rows - 1) for j in range(1, self.cols - 1)
            if self.base_grid_map[i, j] == 0 and (i, j) not in safe_zone
        ]

        for obs_index in range(self.num_dobs):
            if not valid_positions_for_dynamic:
                break

            x, y = random.choice(valid_positions_for_dynamic)
            valid_positions_for_dynamic.remove((x, y))

            behavior = random.choice(self.behavior_options)
            direction = random.choice(list(self.direction_vectors.keys()))
            speed = random.randint(1, 2)

            obstacle = {
                'position': (x, y),
                'direction': direction,
                'speed': speed,
                'history': deque([None] * 4, maxlen=4),
                'behavior': behavior,
                'steps_since_turn': 0,
                'turn_interval': random.randint(4, 10)
            }
            # For patrol behavior, select patrol points (from currently available valid positions)
            if behavior == 'patrol' and len(valid_positions_for_dynamic) >= 2:
                patrol_points = random.sample(valid_positions_for_dynamic, min(2, len(valid_positions_for_dynamic)))
                obstacle['waypoints'] = patrol_points
                obstacle['waypoint_index'] = 0

            self.dynamic_obstacles.append(obstacle)

        self.obstacle_history = {i: deque(maxlen=4) for i in range(len(self.dynamic_obstacles))}

        self.kalman_filters = {}
        for i in range(len(self.dynamic_obstacles)):
            self.kalman_filters[i] = {
                'state': np.zeros(4), 
                'covariance': np.eye(4) * 0.1, 
                'initialized': False
            }
        self.valid_positions = [
            (i, j) for i in range(1, self.rows - 1) for j in range(1, self.cols - 1)
            if self.base_grid_map[i, j] == 0
        ]
        self.agent_pos = self.start
        self.steps = 0
        self.distance = np.linalg.norm(np.array(self.start) - np.array(self.goal))
        # rebuild the working grid_map that combines base (static) and dynamic obstacles
        self.rebuild_grid_map()

    def reset(self, resample_obstacles=None, seed=None):
        if resample_obstacles is None:
            if not hasattr(self, 'dynamic_obstacles'):
                self.reset_dynamic(seed=seed)
        elif resample_obstacles:
            self.reset_dynamic(seed=seed)
        self.agent_pos = self.start
        self.steps = 0
        self.distance = np.linalg.norm(np.array(self.start) - np.array(self.goal))
        self.goal_list_copy = self.goal_list.copy() if self.goal_list else []
        self.current_goal_index = 0
        if self.goal_list and len(self.goal_list) > 0:
            self.goal = self.goal_list[self.current_goal_index]
        elif self.goal is None:
            raise ValueError("Target point not set, please provide the goal parameter or goal_list during initialization")
        if hasattr(self, 'dynamic_obstacles') and len(self.dynamic_obstacles) > 0:
            def ensure_tuple(pos):
                if isinstance(pos, (list, np.ndarray)):
                    return tuple(pos)
                elif isinstance(pos, tuple):
                    return pos
                else:
                    return tuple(pos)
            
            forbidden_positions = set()
            forbidden_positions.add(ensure_tuple(self.start))
            forbidden_positions.add(ensure_tuple(self.goal))
            if self.goal_list:
                for goal_point in self.goal_list:
                    forbidden_positions.add(ensure_tuple(goal_point))
            safe_zone = set()
            for fx, fy in forbidden_positions:
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        safe_zone.add((fx + dx, fy + dy))
            conflicts = []
            occupied_positions = set()
            for obs in self.dynamic_obstacles:
                pos = tuple(obs['position'])
                occupied_positions.add(pos)
                if pos in safe_zone:
                    conflicts.append(obs)
            if conflicts:
                valid_positions = [
                    (i, j) for i in range(1, self.rows - 1) for j in range(1, self.cols - 1)
                    if self.base_grid_map[i, j] == 0 
                    and (i, j) not in safe_zone
                    and (i, j) not in occupied_positions
                ]
                for obs in conflicts:
                    if valid_positions:
                        old_pos = tuple(obs['position'])
                        occupied_positions.discard(old_pos)
                        new_pos = random.choice(valid_positions)
                        obs['position'] = new_pos
                        occupied_positions.add(new_pos)
                        valid_positions.remove(new_pos)
        self.history.clear()
        self.rebuild_grid_map()
        return self.get_state()

    def get_local_observation(self, position, size=7):
        """
            Get local observation around a specified position

            Args:
            position: (x, y) position tuple
            size: size of the local observation window (default 7x7)

            Returns:
            local_grid: local grid of size x size, where 1 indicates an obstacle and 0 indicates traversable
        """
        # ensure grid_map reflects current dynamic obstacle positions
        self.rebuild_grid_map()
        local_grid = np.zeros((size, size), dtype=int)
        half_size = size // 2

        x_min = max(0, position[0] - half_size)
        x_max = min(self.rows, position[0] + half_size + 1)
        y_min = max(0, position[1] - half_size)
        y_max = min(self.cols, position[1] + half_size + 1)

        r_min = half_size - (position[0] - x_min)
        r_max = half_size + (x_max - position[0])
        c_min = half_size - (position[1] - y_min)
        c_max = half_size + (y_max - position[1])

        local_grid[r_min:r_max, c_min:c_max] = self.grid_map[x_min:x_max, y_min:y_max]

        return local_grid

    def rebuild_grid_map(self):
        # start from base static map
        self.grid_map = self.base_grid_map.copy()
        # overlay dynamic obstacles
        if hasattr(self, 'dynamic_obstacles'):
            for obs in self.dynamic_obstacles:
                x, y = obs['position']
                # only place dynamic obstacle if within bounds
                if 0 <= x < self.rows and 0 <= y < self.cols:
                    self.grid_map[x, y] = 1

    def get_state(self, n_frames=4):  
        # ensure grid_map is up-to-date before constructing local observations
        self.rebuild_grid_map()
        nearby_grid = np.ones((7, 7), dtype=int)  # Initially 1 indicates empty area
        x_min, x_max = max(0, self.agent_pos[0] - 3), min(self.rows, self.agent_pos[0] + 4)
        y_min, y_max = max(0, self.agent_pos[1] - 3), min(self.cols, self.agent_pos[1] + 4)
        r_min, r_max = 3 - (self.agent_pos[0] - x_min), 3 + (x_max - self.agent_pos[0])
        c_min, c_max = 3 - (self.agent_pos[1] - y_min), 3 + (y_max - self.agent_pos[1])
        nearby_grid[r_min:r_max, c_min:c_max] = self.grid_map[x_min:x_max, y_min:y_max]

        dx = self.goal[0] - self.agent_pos[0]
        dy = self.goal[1] - self.agent_pos[1]
        distance_to_goal = np.sqrt(dx**2 + dy**2)
        angle_to_goal = np.arctan2(dy, dx)
        # Add predicted positions of dynamic obstacles to the state
        self._add_predicted_obstacles_to_grid(nearby_grid)
        nearby_flat_with_dynamic = nearby_grid.flatten()
        current_state = np.concatenate(([distance_to_goal, angle_to_goal], nearby_flat_with_dynamic))

        self.history.append(current_state)
        while len(self.history) < n_frames:
            self.history.appendleft(current_state)
        if len(self.history) >= 2:
            last_state = self.history[-1]
            velocity = current_state - last_state
        else:
            velocity = np.zeros_like(current_state)
        obstacle_info = []
        for obstacle in self.dynamic_obstacles:
            rel_pos = np.array(obstacle['position']) - np.array(self.agent_pos)
            rel_speed = np.array([
                1 if obstacle['direction'] in ['down', 'right'] else -1,
                1 if obstacle['direction'] in ['up', 'down'] else 0
            ]) * obstacle['speed']
            obstacle_info.extend([*rel_pos, *rel_speed])
        distance_normalized = distance_to_goal / np.sqrt(self.rows**2 + self.cols**2)
        angle_normalized = angle_to_goal / (2 * np.pi)
        
        current_state = np.concatenate([
            [distance_normalized, angle_normalized],
            nearby_flat_with_dynamic,
            velocity,
            obstacle_info
        ])

        return np.array(self.history) 
    
    def step(self, action):
        state = self.get_state()
        nearby_flat = state[-1][2:]

        actions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1), (0 , 0)]
        delta = actions[action]
        next_pos = (self.agent_pos[0] + delta[0], self.agent_pos[1] + delta[1])
        for obstacle in self.dynamic_obstacles:
            if self.agent_pos == obstacle['position']:
                reward = -5.0
                done = True
                return self.get_state(), reward, done

        if not (0 <= next_pos[0] < self.rows and 0 <= next_pos[1] < self.cols) or self.grid_map[next_pos] == 1:
            reward = -5.0
            done = True
            return self.get_state(), reward, done

        self.agent_pos = next_pos
        reward = -0.5
        done = False

        next_distance = np.sqrt((self.goal[0] - self.agent_pos[0])**2 + (self.goal[1] - self.agent_pos[1])**2)
        if(self.distance > next_distance):
            reward += 0.6
        else:
            reward -= 0.4
        self.distance = next_distance

        action_vector = np.array([delta[0], delta[1]])
        goal_vector = np.array([self.goal[0] - self.agent_pos[0], self.goal[1] - self.agent_pos[1]])
        goal_vector_norm = goal_vector / (np.linalg.norm(goal_vector) + 1e-5)
        
        if np.linalg.norm(goal_vector) > 0.1:
            alignment_reward = np.dot(action_vector, goal_vector_norm)
        else:
            alignment_reward = 0
        
        reward += alignment_reward * 0.4

        if self.distance < 0.5:
            reward += 10
            if self.current_goal_index + 1 < len(self.goal_list):
                self.current_goal_index += 1
                self.goal = self.goal_list[self.current_goal_index]  
            else:
                done = True
                
        for obstacle in self.dynamic_obstacles:
            self._update_dynamic_obstacle(obstacle)

        # Calculate minimum obstacle distance
        min_distance_to_obstacle = float('inf')
        for i in range(self.rows):
            for j in range(self.cols):
                if self.grid_map[i, j] == 1:  # Obstacle
                    distance_to_obstacle = np.linalg.norm(np.array(self.agent_pos) - np.array((i, j)))
                    min_distance_to_obstacle = min(min_distance_to_obstacle, distance_to_obstacle)

        # Reward: The closer to an obstacle, the larger the penalty
        if min_distance_to_obstacle == 1.0:
            reward -= 2.0
        elif min_distance_to_obstacle <= 2.0:
            reward -= 1.5
        elif min_distance_to_obstacle <= 3.0:
            reward -= 0.2
        else:
            reward -= 0.01
        
        # Check distance between agent and predicted landing points
        for idx, cell in enumerate(nearby_flat):
            if cell == 3:
                grid_x, grid_y = divmod(idx, 7)
                global_x = self.agent_pos[0] + (grid_x - 3)
                global_y = self.agent_pos[1] + (grid_y - 3)
                distance_to_prediction = np.linalg.norm(np.array(self.agent_pos) - np.array((global_x, global_y)))
                if distance_to_prediction < 2:  # Too close to predicted landing point
                    reward -= 2.75  # Penalty

        # Check if agent collides with dynamic obstacles
        for obstacle in self.dynamic_obstacles:
            if self.agent_pos == obstacle['position']:
                reward -= 5.0
                done = True
                return self.get_state(), reward, done

        self.steps += 1
        if self.steps >= self.max_steps:
            done = True
            reward -= 10

        return self.get_state(), reward, done

    def _add_predicted_obstacles_to_grid(self, nearby_grid):
        """
        Add predicted positions of dynamic obstacles to the grid
        Multi-hypothesis prediction strategy for discrete grid environments
        
        Args:
            nearby_grid: 7x7 local grid representing the environment around the agent
        
        Prediction strategy (optimized for discrete environments):
            L=1: First observation, mark 8-neighborhood of current position (most likely positions)
            L≥2: Calculate probability distribution of recent movement directions:
                 - Mark the most probable predicted position (based on historical direction voting)
                 - Also mark the nearest neighbor cell to the obstacle's current position (conservative strategy)
                 
        Design rationale:
            In discrete environments, velocity can only be integer vectors (8-neighborhood), no continuous noise
            When frequent turning occurs, single prediction easily fails, multi-hypothesis strategy covers main possibilities
            Marking "most likely position" ensures prediction capability, marking "nearest neighbor" ensures safety
        """
        for idx, obstacle in enumerate(self.dynamic_obstacles):
            self.obstacle_history[idx].append(obstacle['position'])
            obs_pos = np.array(obstacle['position'], dtype=int)  # 保持整数
            history_len = len(self.obstacle_history[idx])
            
            # ========== Case 1: First observation (L=1) ==========
            if history_len < 2:
                # No historical information: mark all possible positions in 8-neighborhood (equal probability assumption)
                # This is the most conservative strategy, suitable when obstacle just enters view
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue  # Skip current position
                        neighbor_pos = obs_pos + np.array([dx, dy])
                        self._mark_prediction_in_grid(neighbor_pos, nearby_grid)
            
            # ========== Case 2: Multi-hypothesis prediction (L≥2) ==========
            else:
                # Count movement directions in recent N steps (discrete velocity vectors)
                direction_votes = {}
                look_back = min(3, history_len - 1)  # Look back at most 3 steps
                
                for i in range(look_back):
                    velocity = (np.array(self.obstacle_history[idx][-1-i]) - 
                               np.array(self.obstacle_history[idx][-2-i]))
                    # Normalize velocity vector to unit direction (one of 8-neighborhood)
                    vel_tuple = tuple(np.clip(velocity, -1, 1).astype(int))
                    
                    # Special handling for zero vector (stationary)
                    if vel_tuple == (0, 0):
                        vel_tuple = (0, 0)  # Stay stationary
                    
                    # Voting: more recent history has higher weight
                    weight = 1.0 / (i + 1)  # Most recent weight=1.0, next=0.5, next=0.33
                    direction_votes[vel_tuple] = direction_votes.get(vel_tuple, 0) + weight
                
                # Find the direction with highest probability
                if direction_votes:
                    max_prob_direction = max(direction_votes.items(), key=lambda x: x[1])[0]
                    max_prob_position = obs_pos + np.array(max_prob_direction)
                    
                    # Strategy 1: Mark the most probable predicted position
                    self._mark_prediction_in_grid(max_prob_position, nearby_grid)
                    
                    # Strategy 2: Also mark the nearest neighbor cell to current position (if different from most likely position)
                    # Calculate current velocity (most recent step)
                    current_velocity = (np.array(self.obstacle_history[idx][-1]) - 
                                       np.array(self.obstacle_history[idx][-2]))
                    current_vel_tuple = tuple(np.clip(current_velocity, -1, 1).astype(int))
                    nearest_position = obs_pos + np.array(current_vel_tuple)
                    
                    # If nearest neighbor is different from most likely position, also mark it
                    if not np.array_equal(nearest_position, max_prob_position):
                        self._mark_prediction_in_grid(nearest_position, nearby_grid)
                    
                    # Optional: If second highest probability direction is also significant (>50% of highest), also mark it
                    sorted_directions = sorted(direction_votes.items(), key=lambda x: x[1], reverse=True)
                    if len(sorted_directions) > 1:
                        second_prob_direction, second_prob_value = sorted_directions[1]
                        max_prob_value = sorted_directions[0][1]
                        if second_prob_value > max_prob_value * 0.5:
                            # Second highest probability is also significant, mark as uncertain area
                            second_prob_position = obs_pos + np.array(second_prob_direction)
                            self._mark_prediction_in_grid(second_prob_position, nearby_grid)
                else:
                    # No valid direction information (all velocities are zero), mark current position
                    self._mark_prediction_in_grid(obs_pos, nearby_grid)
    
    # def _update_kalman_filter_lightweight(self, idx, position, velocity):
    #     kf = self.kalman_filters[idx]
        
    #     if not kf['initialized']:
    #         kf['state'] = np.array([position[0], position[1], velocity[0], velocity[1]])
    #         kf['initialized'] = True
    #         return
        
    #     # alpha = 0.3 表示给予新观测30%的权重，历史70%的权重
    #     alpha = 0.3
        
    #     kf['state'][0] = position[0]
    #     kf['state'][1] = position[1]
        
    #     kf['state'][2] = alpha * velocity[0] + (1 - alpha) * kf['state'][2]
    #     kf['state'][3] = alpha * velocity[1] + (1 - alpha) * kf['state'][3]
    
    def _mark_prediction_in_grid(self, predicted_pos, nearby_grid):
        """
        Mark predicted position in 7x7 grid
        
        Args:
            predicted_pos: Predicted position (float or integer array)
            nearby_grid: 7x7 local grid
        """
        pred_pos_int = np.round(predicted_pos).astype(int)
        
        # Boundary and static obstacle check: only mark positions within map and without static obstacles
        if not (0 <= pred_pos_int[0] < self.rows and 0 <= pred_pos_int[1] < self.cols):
            return  # Out of bounds, do not mark
        if self.base_grid_map[pred_pos_int[0], pred_pos_int[1]] == 1:
            return  # Static obstacle position, do not mark
        
        dx_obstacle = pred_pos_int[0] - self.agent_pos[0]
        dy_obstacle = pred_pos_int[1] - self.agent_pos[1]
        
        # If within 7x7 range, mark as predicted landing point
        if abs(dx_obstacle) <= 3 and abs(dy_obstacle) <= 3:
            grid_x = int(dx_obstacle + 3)
            grid_y = int(dy_obstacle + 3)
            # Ensure not out of bounds
            if 0 <= grid_x < 7 and 0 <= grid_y < 7:
                nearby_grid[grid_x, grid_y] = 3

    def _update_dynamic_obstacle(self, obstacle):
        current_pos = obstacle['position']
        behavior = obstacle.get('behavior', 'random_walk')

        if behavior == 'patrol' and obstacle.get('waypoints'):
            target = obstacle['waypoints'][obstacle['waypoint_index']]
            if current_pos == target:
                obstacle['waypoint_index'] = (obstacle['waypoint_index'] + 1) % len(obstacle['waypoints'])
                target = obstacle['waypoints'][obstacle['waypoint_index']]
            direction_vec = np.array(target) - np.array(current_pos)
            direction_vec = np.clip(direction_vec, -1, 1)
            patrol_direction = self._vector_to_direction(tuple(direction_vec))
            if patrol_direction:
                obstacle['direction'] = patrol_direction
        elif behavior == 'wander':
            obstacle['steps_since_turn'] = obstacle.get('steps_since_turn', 0) + 1
            if obstacle['steps_since_turn'] >= obstacle.get('turn_interval', 6):
                obstacle['direction'] = self._choose_new_direction(obstacle, allow_backtrack=False)
                obstacle['steps_since_turn'] = 0
        else:  # random walk
            if random.random() < self.turn_probability:
                obstacle['direction'] = self._choose_new_direction(obstacle)

        direction_vec = self.direction_vectors.get(obstacle['direction'], (0, 0))
        if direction_vec == (0, 0):
            obstacle['direction'] = self._choose_new_direction(obstacle, allow_backtrack=False)
            direction_vec = self.direction_vectors.get(obstacle['direction'], (0, 0))

        speed = max(1, obstacle.get('speed', 1))
        candidate_pos = current_pos
        moved = False

        for _ in range(speed):
            potential = (candidate_pos[0] + direction_vec[0], candidate_pos[1] + direction_vec[1])
            if self._is_passable(potential, ignore=obstacle):
                candidate_pos = potential
                moved = True
            else:
                obstacle['direction'] = self._choose_new_direction(obstacle, allow_backtrack=False)
                direction_vec = self.direction_vectors.get(obstacle['direction'], (0, 0))
                if direction_vec == (0, 0):
                    break

        if moved and candidate_pos != current_pos:
            obstacle['position'] = candidate_pos
        # do NOT directly write into self.grid_map here; rebuild_grid_map() will overlay dynamic obstacles

    def _choose_new_direction(self, obstacle, allow_backtrack=True):
        current_direction = obstacle.get('direction')
        directions = list(self.direction_vectors.keys())
        random.shuffle(directions)

        if not allow_backtrack and current_direction:
            current_vec = self.direction_vectors.get(current_direction, (0, 0))
            opposite_vec = (-current_vec[0], -current_vec[1])
            opposite_direction = self._vector_to_direction(opposite_vec)
            if opposite_direction:
                directions = [d for d in directions if d != opposite_direction]

        for direction in directions:
            vector = self.direction_vectors[direction]
            candidate = (obstacle['position'][0] + vector[0], obstacle['position'][1] + vector[1])
            if self._is_passable(candidate, ignore=obstacle):
                return direction

        return current_direction or random.choice(list(self.direction_vectors.keys()))

    def _is_passable(self, position, ignore=None):
        x, y = position
        if not (0 <= x < self.rows and 0 <= y < self.cols):
            return False
        if self.base_grid_map[x, y] == 1:
            return False
        # Dynamic obstacles should not actively avoid the agent, removed agent position check
        # if tuple(self.agent_pos) == (x, y):
        #     return False
        for other in self.dynamic_obstacles:
            if other is ignore:
                continue
            if tuple(other['position']) == (x, y):
                return False
        return True
    
    def _direction_to_vector(self, direction):
        return self.direction_vectors.get(direction, (0, 0))

    def _vector_to_direction(self, vector):
        for name, vec in self.direction_vectors.items():
            if vec == vector:
                return name
        return None
    
    def save_obstacle_config(self):
        """
        Save current obstacle configuration (static + dynamic)
        
        Returns:
            dict: Environment configuration dictionary containing all necessary information
        """
        # Deep copy dynamic obstacle list to avoid reference issues
        dynamic_obstacles_copy = []
        for obs in self.dynamic_obstacles:
            obs_copy = {
                'position': tuple(obs['position']),
                'direction': obs['direction'],
                'speed': obs['speed'],
                'behavior': obs['behavior'],
                'steps_since_turn': obs['steps_since_turn'],
                'turn_interval': obs['turn_interval']
            }
            # If there are patrol points, also save them
            if 'waypoints' in obs:
                obs_copy['waypoints'] = [tuple(wp) for wp in obs['waypoints']]
                obs_copy['waypoint_index'] = obs['waypoint_index']
            dynamic_obstacles_copy.append(obs_copy)
        
        config = {
            'base_grid_map': self.base_grid_map.copy(),
            'dynamic_obstacles': dynamic_obstacles_copy,
            'num_obstacles': self.num_obstacles,
            'num_dobs': self.num_dobs,
            'start': tuple(self.start),
            'goal': tuple(self.goal),
            'goal_list': [tuple(g) for g in self.goal_list] if self.goal_list else []
        }
        
        return config
    
    def load_obstacle_config(self, config):
        """
        Load previously saved obstacle configuration
        
        Args:
            config: Configuration dictionary returned from save_obstacle_config()
        """
        # Restore start and goal points
        if 'start' in config:
            self.start = tuple(config['start'])
        if 'goal' in config:
            self.goal = tuple(config['goal'])
        if 'goal_list' in config:
            self.goal_list = [tuple(g) for g in config['goal_list']]
        
        # Restore static obstacles
        self.base_grid_map = config['base_grid_map'].copy()
        # self.base_grid_map = self.original_grid_map.copy()
        
        # Restore dynamic obstacles
        self.dynamic_obstacles = []
        for obs_data in config['dynamic_obstacles']:
            obstacle = {
                'position': tuple(obs_data['position']),
                'direction': obs_data.get('direction', 'up'),
                'speed': obs_data.get('speed', 1),
                'history': deque([None] * 4, maxlen=4),
                'behavior': obs_data.get('behavior', 'random_walk'),
                'steps_since_turn': obs_data.get('steps_since_turn', 0),
                'turn_interval': obs_data.get('turn_interval', random.randint(4, 10))
            }
            
            # Restore patrol points (only when waypoints is not None)
            waypoints = obs_data.get('waypoints')
            if waypoints is not None and len(waypoints) > 0:
                obstacle['waypoints'] = [tuple(wp) for wp in waypoints]
                obstacle['waypoint_index'] = obs_data.get('waypoint_index', 0)
            
            self.dynamic_obstacles.append(obstacle)
        
        # Reinitialize obstacle history and Kalman filters
        self.obstacle_history = {i: deque(maxlen=4) for i in range(len(self.dynamic_obstacles))}
        self.kalman_filters = {}
        for i in range(len(self.dynamic_obstacles)):
            self.kalman_filters[i] = {
                'state': np.zeros(4),
                'covariance': np.eye(4) * 0.1,
                'initialized': False
            }
        
        # Rebuild grid
        self.rebuild_grid_map()
    
    @staticmethod
    def save_configs_to_file(configs, filepath):
        """
        Save multiple environment configurations to file
        
        Args:
            configs: List of configuration dictionaries
            filepath: Save path
        """
        with open(filepath, 'wb') as f:
            pickle.dump(configs, f)
        print(f"Saved {len(configs)} environment configurations to {filepath}")
    
    @staticmethod
    def load_configs_from_file(filepath):
        """
        Load environment configurations from file (enhanced version, supports multiple encodings)
        
        Args:
            filepath: File path
            
        Returns:
            list: List of configuration dictionaries
        """
        try:
            # First try standard loading
            with open(filepath, 'rb') as f:
                configs = pickle.load(f)
            print(f"Loaded {len(configs)} environment configurations from {filepath}")
            return configs
        except (UnicodeDecodeError, pickle.UnpicklingError) as e:
            print(f"Warning: Standard loading failed ({e}), trying encoding='latin1'...")
            try:
                with open(filepath, 'rb') as f:
                    configs = pickle.load(f, encoding='latin1')
                print(f"✓ Successfully loaded {len(configs)} environment configurations using encoding='latin1'")
                return configs
            except Exception as e2:
                print(f"✗ Loading failed: {e2}")
                raise RuntimeError(f"Unable to load configuration file {filepath}, file may be corrupted") from e2
        return configs