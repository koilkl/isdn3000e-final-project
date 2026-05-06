import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pettingzoo.classic import tictactoe_v3

# ---------------------------------------------------------
# 1. Define the Neural Network Architecture
# ---------------------------------------------------------
class TTTModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Board has 9 cells. In PettingZoo, observation is usually a dict, 
        # but we'll convert the board state to a simple 9-dim vector.
        # 0 = empty, 1 = player 1, 2 = player 2
        self.fc1 = nn.Linear(9, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 9) # Outputs Q-values for 9 possible actions

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        return self.fc3(x)

# ---------------------------------------------------------
# 2. DQN Agent and Training Loop
# ---------------------------------------------------------
class DQNAgent:
    def __init__(self, model, lr=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=10000)

    def get_action(self, state, action_mask):
        if random.random() < self.epsilon:
            # Explore: choose a random legal action
            legal_actions = [i for i, mask in enumerate(action_mask) if mask == 1]
            return random.choice(legal_actions)
        else:
            # Exploit: choose best legal action according to Q-network
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state_tensor).squeeze(0).numpy()
            
            # Mask out illegal actions by setting their Q-values to -infinity
            q_values[action_mask == 0] = -float('inf')
            return int(np.argmax(q_values))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        # Current Q-values
        current_q = self.model(states).gather(1, actions).squeeze(1)
        
        # Target Q-values
        with torch.no_grad():
            max_next_q = self.model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

def convert_observation_to_state(observation_dict, agent_name):
    # PettingZoo tictactoe_v3 observation:
    # observation["observation"] is a 3D array (3, 3, 2).
    # Plane 0: my pieces, Plane 1: opponent pieces
    # We will convert this back to a flat 9-dim array (0=empty, 1=player1, 2=player2)
    # matching what the C++ node sends (which is 0, 1, 2).
    
    obs = observation_dict["observation"]
    my_pieces = obs[:, :, 0].flatten()
    opp_pieces = obs[:, :, 1].flatten()
    
    state = np.zeros(9, dtype=np.float32)
    
    # We need to map it to absolute player IDs to match C++ (1 for player 1, 2 for player 2)
    my_val = 1.0 if agent_name == "player_1" else 2.0
    opp_val = 2.0 if agent_name == "player_1" else 1.0
    
    state[my_pieces == 1] = my_val
    state[opp_pieces == 1] = opp_val
    
    return state

def train_model(episodes=1000):
    env = tictactoe_v3.env()
    model = TTTModel()
    agent = DQNAgent(model)
    
    print(f"Training DQN for {episodes} episodes...")
    for episode in range(episodes):
        env.reset()
        
        # Keep track of previous state/action for reward assignment
        prev_state = {}
        prev_action = {}
        
        for agent_name in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            done = termination or truncation
            
            # If the game just ended, the reward applies to the PREVIOUS action of this agent
            if done:
                if agent_name in prev_state:
                    agent.remember(prev_state[agent_name], prev_action[agent_name], reward, np.zeros(9), done)
                env.step(None) # Required by PettingZoo API for terminal states
                continue
            
            # Current state
            state = convert_observation_to_state(observation, agent_name)
            action_mask = observation["action_mask"]
            
            # Assign reward to previous action if the game hasn't ended but we got a step reward
            if agent_name in prev_state:
                agent.remember(prev_state[agent_name], prev_action[agent_name], reward, state, done)

            # Choose and take action
            action = agent.get_action(state, action_mask)
            env.step(action)
            
            # Store for next iteration
            prev_state[agent_name] = state
            prev_action[agent_name] = action
            
        agent.replay()
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes} completed. Epsilon: {agent.epsilon:.3f}")
            
    return model

if __name__ == "__main__":
    # Train the model
    trained_model = train_model(episodes=2000)
    
    # Export to TorchScript
    trained_model.eval()
    dummy_input = torch.zeros(1, 9)
    traced_model = torch.jit.trace(trained_model, dummy_input)
    
    model_path = os.path.join(os.path.dirname(__file__), "ttt_rl_model.pt")
    traced_model.save(model_path)
    print(f"\nTrained RL model exported to TorchScript and saved to: {model_path}")
