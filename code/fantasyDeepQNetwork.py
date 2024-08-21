import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class FantasyDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, stats_dims, draftboard_dims, roster_dims):
        super(FantasyDeepQNetwork, self).__init__()

        # Separate input layers for stats, draftboard, and roster data
        self.stats_fc1 = nn.Linear(stats_dims[1], 128)
        self.stats_fc2 = nn.Linear(128, 64)
        self.stats_fc3 = nn.Linear(64, 32)

        self.draftboard_fc1 = nn.Linear(draftboard_dims[1], 128)
        self.draftboard_fc2 = nn.Linear(128, 64)
        self.draftboard_fc3 = nn.Linear(64, 32)

        self.roster_fc1 = nn.Linear(roster_dims[1], 128)
        self.roster_fc2 = nn.Linear(128, 64)
        self.roster_fc3 = nn.Linear(64, 32)

        # Merging the outputs
        self.fc3 = nn.Linear((stats_dims[0] + draftboard_dims[0] + roster_dims[0]) * 32, 128)
        self.fc4 = nn.Linear(128, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observations):
        stats = observations['stats']
        draftboard = observations['draftboard']
        roster = observations['roster']
        # print (f'stats in shape {stats.shape}')
        # print (f'board in shape {draftboard.shape}')
        # print (f'roster in shape {roster.shape}')

        # Process stats input
        stats_out = F.relu(self.stats_fc1(stats))
        stats_out = F.relu(self.stats_fc2(stats_out))
        stats_out = F.relu(self.stats_fc3(stats_out))

        # Process draftboard input
        draftboard_out = F.relu(self.draftboard_fc1(draftboard))
        draftboard_out = F.relu(self.draftboard_fc2(draftboard_out))
        draftboard_out = F.relu(self.draftboard_fc3(draftboard_out))

        # Process roster input
        roster_out = F.relu(self.roster_fc1(roster))
        roster_out = F.relu(self.roster_fc2(roster_out))
        roster_out = F.relu(self.roster_fc3(roster_out))

        # print(f"stats_out shape: {stats_out.shape}")
        # print(f"draftboard_out shape: {draftboard_out.shape}")
        # print(f"roster_out shape: {roster_out.shape}")

        # Concatenate all processed inputs
        combined = torch.cat([stats_out, draftboard_out, roster_out], dim=1)

        combined_flat = combined.view(combined.size(0), -1)

        # print(f'combined shape {combined.shape}')
        # print(f'combined_flat shape {combined_flat.shape}')

        # Final processing layers
        out = F.relu(self.fc3(combined_flat))
        actions = self.fc4(out)

        return actions


class Agent():
    def __init__(self, stats_dims, draftboard_dims, roster_dims, n_actions, lr, gamma=0.99, epsilon=1.0, eps_dec=1e-5, eps_min=0.01):
        self.lr = lr
        self.stats_dims = stats_dims
        self.draftboard_dims = draftboard_dims
        self.roster_dims = roster_dims
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min

        self.Q = FantasyDeepQNetwork(self.lr, self.n_actions, self.stats_dims, self.draftboard_dims, self.roster_dims)

    def choose_action(self, observations):
        if np.random.random() > self.epsilon:
            stats = observations['stats'].to_numpy()
            draftboard = observations['draftboard'].to_numpy()
            roster = observations['roster'].to_numpy()
            stats_obs = torch.tensor(stats, dtype=torch.float32).to(self.Q.device)
            draftboard_obs = torch.tensor(draftboard, dtype=torch.float32).to(self.Q.device)
            roster_obs = torch.tensor(roster, dtype=torch.float32).to(self.Q.device)
            obs = {
                'stats': stats_obs,
                'draftbord': draftboard_obs,
                'roster': roster_obs
            }
            actions = self.Q.forward(obs)
            action = torch.argmax(actions).item()
        else:
            board = observations['draftboard']
            action = np.random.choice(board['Name'])

        return action
    
    def decrement_epsilon(self):
        self.epsilon = (self.epsilon - self.eps_dec) if self.epsilon > self.eps_min else self.eps_min

    # working on learn

    def learn(self, batch):
        observations, actions, rewards, next_observations, dones = zip(*batch)

        # Perform gradient descent
        self.Q.optimizer.zero_grad()

        def totuple(a):
            try:
                return tuple(totuple(i) for i in a)
            except TypeError:
                return a

        arr = [obs['stats'].to_numpy() for obs in observations]

        test = np.stack(arr).astype(np.float32)
        # print(f'test is \n {test.shape}')

        # linearize obs space
        stats_tensor = torch.from_numpy(np.stack([obs['stats'].to_numpy() for obs in observations]).astype(np.float32))
        draftboard_tensor = torch.from_numpy(np.stack([obs['draftboard'].to_numpy() for obs in observations]).astype(np.float32))
        roster_tensor = torch.from_numpy(np.stack([obs['roster'].to_numpy() for obs in observations]).astype(np.float32))

        next_stats_tensor = torch.from_numpy(np.stack([obs['stats'].to_numpy() for obs in next_observations]).astype(np.float32))
        next_draftboard_tensor = torch.from_numpy(np.stack([obs['draftboard'].to_numpy() for obs in next_observations]).astype(np.float32))
        next_roster_tensor = torch.from_numpy(np.stack([obs['roster'].to_numpy() for obs in next_observations]).astype(np.float32))

        processed_obs = {
            'stats': stats_tensor,
            'draftboard': draftboard_tensor,
            'roster': roster_tensor
        }

        processed_next_obs = {
            'stats': next_stats_tensor,
            'draftboard': next_draftboard_tensor,
            'roster': next_roster_tensor
        }
        # print(f'action shape: {np.array(actions)}')
        actions = torch.tensor(np.array(actions), dtype=torch.long).to(self.Q.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.Q.device)
        dones = torch.tensor(np.array(dones), dtype=torch.bool).to(self.Q.device)

        # Compute predicted Q-values
        # interim =  self.Q.forward(processed_obs)
        # print(f'shape of interim is {interim.shape}')
        q_pred = self.Q.forward(processed_obs).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # Compute target Q-values
        q_next = self.Q.forward(processed_next_obs).max(dim=1)[0]
        q_target = rewards + self.gamma * q_next * (~dones)

        # Compute the loss
        loss = self.Q.loss(q_pred, q_target).to(self.Q.device)

        loss.backward()
        self.Q.optimizer.step()

        self.decrement_epsilon()
