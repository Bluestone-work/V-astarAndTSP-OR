import inspect
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from models import DuelingDQN_with_LSTM

class RainbowDQN:
    def __init__(self, env, gamma=0.8, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.9, 
                learning_rate=4e-4, batch_size=64, memory_size=10000, device=None, alpha=0.6, beta=0.4,
                n_step=4, shared_replay_buffer=None, smoothing_factor=0.5, action_consistency=0.8):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.update_target_freq = 3
        self.alpha = alpha  # The Importance of Prioritization
        self.beta = beta    # Bias correction for prioritized sampling
        self.n_step = n_step  # Multi-step parameter
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.training_steps = 0
        self.log_offset = 1e-5
        self.w_reward = 0.25
        self.w_TD_error = 0.75
        self.smoothing_factor = smoothing_factor  # Path smoothing weight
        self.action_consistency = action_consistency  # Action consistency weight
        self.last_actions = deque(maxlen=3)  # Recent action history
        
        # Deadlock detection and escape mechanism
        self.stuck_threshold = 8  # Number of consecutive steps with Q-value change below threshold considered stuck
        self.stuck_counter = 0
        self.last_max_q = None
        self.escape_mode = False
        self.escape_steps = 0
        
        self.q_network = DuelingDQN_with_LSTM(51, 9).to(self.device)
        self.target_network = DuelingDQN_with_LSTM(51, 9).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Use shared buffer if provided, otherwise instantiate private buffer
        self.memory = shared_replay_buffer or deque(maxlen=memory_size)
        self.priority_sum = 0  
        self.update_target_network()
        
        # Add prioritized experience buffer - only store high-quality/successful experiences
        self.priority_buffer = deque(maxlen=min(memory_size // 10, 5000))
        
        # Action smoothness penalty matrix - represents the penalty for transitioning from one action to another
        # Reduce penalty between straight actions, increase penalty for sharp turns
        self.action_smoothness_penalty = np.ones((9, 9), dtype=np.float32) * 0.1
        
        # Configure specific action transition penalties
        # Assume 0-3 are up, down, left, right; 4-7 are diagonal moves; 8 is stop
        # Lower penalty for transitions between straight actions
        for i in range(4):
            for j in range(4):
                self.action_smoothness_penalty[i, j] = 0.05
                
        # Moderate penalty for transitions between diagonal actions
        for i in range(4, 8):
            for j in range(4, 8):
                self.action_smoothness_penalty[i, j] = 0.1
        
        # Higher penalty for sharp direction changes
        # For example, a 180-degree turn from up (0) to down (1)
        self.action_smoothness_penalty[0, 1] = 0.3
        self.action_smoothness_penalty[1, 0] = 0.3
        self.action_smoothness_penalty[2, 3] = 0.3
        self.action_smoothness_penalty[3, 2] = 0.3
        
        for i in range(8):
            self.action_smoothness_penalty[i, 8] = 0.15
            self.action_smoothness_penalty[8, i] = 0.15
    
    def update_target_network(self, tau=0.01):
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_((1.0 - tau) * target_param.data + tau * local_param.data)
    
    def select_action(self, state):
        if self.escape_mode:
            self.escape_steps += 1
            if self.escape_steps > 5:  # Exit escape mode after 5 steps
                self.escape_mode = False
                self.escape_steps = 0
                self.stuck_counter = 0
            # Escape mode: completely random action selection, but avoid stopping
            action = random.randint(0, 7)  # Do not choose stop action (8)
            self.last_actions.append(action)
            return action
        
        if random.random() < self.epsilon:
            if len(self.last_actions) > 0 and random.random() < self.action_consistency:
                last_action = self.last_actions[-1]
                similar_actions = [last_action]
                
                if last_action < 4:
                    if last_action == 0:
                        similar_actions.extend([4, 5])
                    elif last_action == 1:
                        similar_actions.extend([6, 7])
                    elif last_action == 2:
                        similar_actions.extend([4, 6]) 
                    elif last_action == 3: 
                        similar_actions.extend([5, 7])
                
                elif 4 <= last_action < 8:
                    if last_action == 4:
                        similar_actions.extend([0, 2])
                    elif last_action == 5:
                        similar_actions.extend([0, 3])
                    elif last_action == 6:
                        similar_actions.extend([1, 2])
                    elif last_action == 7:
                        similar_actions.extend([1, 3])
                
                action = random.choice(similar_actions)
            else:
                action = random.randint(0, 8)
            
            self.last_actions.append(action)
            return action
        
        if isinstance(state, torch.Tensor):
            state_tensor = state.clone().detach().float().to(self.device)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        
        state_tensor = state_tensor.unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor).cpu().numpy().flatten()
            
            #Apply a smoothing regularization to the Q-value
            if len(self.last_actions) > 0:
                last_action = self.last_actions[-1]
                for a in range(9):
                    # Apply smoothness penalty to each action (based on the transition from the last action)
                    q_values[a] -= self.smoothing_factor * self.action_smoothness_penalty[last_action, a]
            
            action = int(np.argmax(q_values))
            
        self.last_actions.append(action)
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        # Calculate multi-step TD error
        td_error = 0.0

        if len(self.memory) > 0:
            if isinstance(state, torch.Tensor):
                state_tensor = state.clone().detach().float().to(self.device).unsqueeze(0)
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)

            if isinstance(next_state, torch.Tensor):
                next_state_tensor = next_state.clone().detach().float().to(self.device).unsqueeze(0)
            else:
                next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(self.device).unsqueeze(0)
            
            # 计算 Q 值和 TD Error
            q_values = self.q_network(state_tensor)
            action_tensor = torch.tensor([action], dtype=torch.long).to(self.device)
            current_q_values = q_values.gather(1, action_tensor.view(-1, 1)).squeeze()
            next_q_values = self.target_network(next_state_tensor).max()
            td_error = abs(reward + self.gamma * next_q_values.item() - current_q_values.item())
        
        reward_shifted = reward + abs(min(reward, 0)) + 1e-5
        log_normalized_reward = np.log(reward_shifted)
        
        td_error_shifted = td_error + 1  # Avoid zero value
        log_normalized_td_error = np.log(td_error_shifted)

        # Calculate action smoothness reward/penalty
        smoothness_reward = 0.0
        if len(self.last_actions) >= 2:
            prev_action = self.last_actions[-2]
            curr_action = action
            smoothness_reward = -self.smoothing_factor * self.action_smoothness_penalty[prev_action, curr_action]
        
        # Consider action smoothness in the reward
        adjusted_reward = reward + smoothness_reward
        # Do not consider action smoothness in the reward
        #adjusted_reward = reward
        
        # Calculate priority (including smoothness consideration)
        priority = (self.w_reward * log_normalized_reward + 
                   self.w_TD_error * log_normalized_td_error)
        
        # Store experience
        experience = (state, action, adjusted_reward, next_state, done, priority)
        self.memory.append(experience)
        self.priority_sum += priority

        # If it is a high-reward experience, also add it to the priority buffer
        if adjusted_reward > 5.0:
            self.priority_buffer.append(experience)

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        # 主要批次采样
        batch_size_main = int(self.batch_size * 0.8)
        #batch_size_main = int(self.batch_size * 0.8)
        batch_size_priority = self.batch_size - batch_size_main
        
        probabilities = np.array([transition[5] for transition in self.memory])
        probabilities = np.abs(probabilities)
        probabilities += 1e-5
        probabilities /= probabilities.sum()
        
        main_indices = np.random.choice(len(self.memory), batch_size_main, p=probabilities)
        main_batch = [self.memory[idx] for idx in main_indices]
        
        # Sample from priority buffer (if enough data)
        priority_batch = []
        if self.priority_buffer and batch_size_priority > 0:
            priority_indices = np.random.choice(
                min(len(self.priority_buffer), batch_size_priority), 
                min(len(self.priority_buffer), batch_size_priority), 
                replace=False
            )
            priority_batch = [list(self.priority_buffer)[idx] for idx in priority_indices]
        
        batch = main_batch + priority_batch
        if not batch:
            return
            
        states, actions, rewards, next_states, dones, priorities = zip(*batch)
        
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)

        if states.dim() == 2:
            states = states.unsqueeze(0)  # (batch_size, n_frames, state_dim)
        if next_states.dim() == 2:
            next_states = next_states.unsqueeze(0)  # (batch_size, n_frames, state_dim)

        if isinstance(actions, torch.Tensor):
            actions = actions.clone().detach().long().to(self.device)
        else:
            actions = torch.tensor(actions, dtype=torch.long).to(self.device)

        if isinstance(rewards, torch.Tensor):
            rewards = rewards.clone().detach().float().to(self.device)
        else:
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)

        if isinstance(dones, torch.Tensor):
            dones = dones.clone().detach().float().to(self.device)
        else:
            dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        target_q_values = rewards
        for i in range(1, self.n_step):
            target_q_values += (self.gamma ** i) * rewards

        with torch.no_grad():
            next_q_values_current = self.q_network(next_states)
            next_actions = next_q_values_current.max(1)[1].unsqueeze(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze()
            
            next_q_values_target = self.target_network(next_states).max(1)[0]
            next_q_values = torch.min(next_q_values, next_q_values_target)
            
            target_q_values += (self.gamma ** self.n_step) * next_q_values * (1 - dones)

        current_q_values = self.q_network(states).gather(1, actions.view(-1, 1)).squeeze()
        loss = nn.SmoothL1Loss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        
        self.optimizer.step()

        self.training_steps += 1
        if self.training_steps % self.update_target_freq == 0:
            self.update_target_network()
            
        return loss

    def update(self):
        self.update_target_network()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.action_consistency = min(0.9, self.action_consistency + 0.001)

    def save_model(self, file_path="dqn_model.pth"):
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'action_consistency': self.action_consistency,
            'stuck_threshold': self.stuck_threshold,
        }, file_path)
        print(f"Model saved to {file_path}")
    
    def load_model(self, file_path="dqn_model.pth"):
        load_kwargs = {"map_location": self.device}
        # torch.load will flip the default of weights_only in future versions; set explicitly when supported.
        if "weights_only" in inspect.signature(torch.load).parameters:
            load_kwargs["weights_only"] = False

        checkpoint = torch.load(file_path, **load_kwargs)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        if 'action_consistency' in checkpoint:
            self.action_consistency = checkpoint['action_consistency']
        if 'stuck_threshold' in checkpoint:
            self.stuck_threshold = checkpoint['stuck_threshold']
        print(f"Model loaded from {file_path}")