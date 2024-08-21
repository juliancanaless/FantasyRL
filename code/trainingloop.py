from fantasyenv import FantasyFootballEnv
from fantasyDeepQNetwork import Agent
from collections import deque
import random
import os
import numpy as np

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

def train_agent(env: FantasyFootballEnv, agent: Agent, num_episodes, batch_size, update_frequency):
    replay_buffer = ReplayBuffer(max_size=10000)
    scores = []

    for episode in range(num_episodes):
        print(f'episode {episode}')
        if episode > 0:
            env.reset()
        done = False
        score = 0
        env._run_draft() # state is updated in run draft

        while not done:
            observations = env.get_observation() # observations depends on current state
            action = agent.choose_action(observations)
            next_observation, reward, done = env.step(action)
            replay_buffer.add((observations, action, reward, next_observation, done))
            score += reward

            if env.current_step % update_frequency == 0 and replay_buffer.size() >= batch_size:
                print('We are learning')
                batch = replay_buffer.sample(batch_size)
                agent.learn(batch)

        scores.append(score)
        print(f'Episode {episode}, Score: {score}, Epsilon: {agent.epsilon}')
    
    return scores

np.random.seed(42)
# Define hyperparameters
num_episodes = 10
batch_size = 16
learning_rate = 0.001
gamma = 0.99
epsilon = 1.0
eps_min = 0.01
eps_dec = 1e-5
update_frequency = 4

script_dir = os.path.dirname(os.path.abspath(''))
proj_dir = os.path.join(script_dir, 'fantasy')
data_dir = os.path.join(proj_dir, 'data')
board_path = os.path.join(data_dir, 'ppr-adp-2023-updated.csv')
weekly_stats_path = os.path.join(data_dir, 'weekly-stats-2022.csv')
weekly_info_path = os.path.join(data_dir, 'simulator-weekly-info-2023.csv')

# Initialize environment and agent
env = FantasyFootballEnv(board_path, weekly_stats_path, weekly_info_path)
stats_dims = env.observation_space['stats'].shape
board_dims = env.observation_space['draftboard'].shape
roster_dims = env.observation_space['roster'].shape
n_actions = env.action_space.n
# print(f'stats dims are: {stats_dims}')
# print(f'board dims are: {board_dims}')
# print(f'roster dims are: {roster_dims}')
# print(f'number of actions {n_actions}')

agent = Agent(stats_dims, board_dims, roster_dims, n_actions, learning_rate, gamma, epsilon, eps_dec, eps_min)

# Train the agent
scores = train_agent(env, agent, num_episodes, batch_size, update_frequency)

# Plot scores or perform further analysis
