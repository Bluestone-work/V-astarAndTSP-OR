import torch
import torch.nn as nn
import torch.nn.functional as F
class DuelingDQN_with_LSTM(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, lstm_layers=1):
        super(DuelingDQN_with_LSTM, self).__init__()
        self.distance_angle_fc = nn.Linear(2, 64)

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # 7x7x16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # 4x4x32
        
        self.lstm_input_dim = 32 * 4 * 4 + 64

        self.lstm = nn.LSTM(input_size=self.lstm_input_dim, hidden_size=hidden_dim, num_layers=lstm_layers, batch_first=True, bidirectional=True)

        lstm_output_dim = hidden_dim * 2

        self.value_fc = nn.Linear(lstm_output_dim, 128)
        self.value_output = nn.Linear(128, 1)

        self.advantage_fc = nn.Linear(lstm_output_dim, 128)
        self.advantage_output = nn.Linear(128, output_dim)

    def forward(self, x):
        if len(x.size()) == 2:
            x = x.unsqueeze(1)
            
        batch_size, time_steps, state_dim = x.size()

        grid_states = x[:, :, 2:].reshape(batch_size * time_steps, 1, 7, 7)

        grid_features = F.relu(self.conv1(grid_states))
        grid_features = F.relu(self.conv2(grid_features))

        grid_features = grid_features.view(batch_size * time_steps, -1)

        distance_angle = x[:, :, :2].reshape(batch_size * time_steps, 2)
        distance_angle = F.relu(self.distance_angle_fc(distance_angle))

        combined_features = torch.cat([grid_features, distance_angle], dim=1)

        combined_features = combined_features.view(batch_size, time_steps, -1)
        lstm_out, (h_n, c_n) = self.lstm(combined_features)

        lstm_out_last = lstm_out[:, -1, :]

        value = F.relu(self.value_fc(lstm_out_last))
        value = self.value_output(value)

        advantage = F.relu(self.advantage_fc(lstm_out_last))
        advantage = self.advantage_output(advantage)

        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class DuelingDQN_with_LSTM(nn.Module):
#     def __init__(self, input_dim, output_dim, hidden_dim=256, lstm_layers=1):
#         super(DuelingDQN_with_LSTM, self).__init__()

#         self.distance_angle_fc = nn.Linear(2, 64)

#         self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1) 
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1) 

#         self.lstm_input_dim = 32 * 4 * 4 + 64

#         self.lstm = nn.LSTM(
#             input_size=self.lstm_input_dim,
#             hidden_size=hidden_dim,
#             num_layers=lstm_layers,
#             batch_first=True,
#             bidirectional=False
#         )

#         self.fc1 = nn.Linear(hidden_dim, 128)
#         self.q_output = nn.Linear(128, output_dim)

#     def forward(self, x):
#         if len(x.size()) == 2:
#             x = x.unsqueeze(1) 

#         batch_size, time_steps, state_dim = x.size()

#         grid_states = x[:, :, 2:].reshape(batch_size * time_steps, 1, 7, 7)
#         grid_features = F.relu(self.conv1(grid_states))
#         grid_features = F.relu(self.conv2(grid_features))
#         grid_features = grid_features.view(batch_size * time_steps, -1)

#         distance_angle = x[:, :, :2].reshape(batch_size * time_steps, 2)
#         distance_angle = F.relu(self.distance_angle_fc(distance_angle))

#         combined_features = torch.cat([grid_features, distance_angle], dim=1)
#         combined_features = combined_features.view(batch_size, time_steps, -1)

#         lstm_out, _ = self.lstm(combined_features)
#         lstm_out_last = lstm_out[:, -1, :]

#         q = F.relu(self.fc1(lstm_out_last))
#         q = self.q_output(q)

#         return q
