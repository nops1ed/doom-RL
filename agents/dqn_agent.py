import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.dqn_network import DQNNetwork, DuelingDQN
from agents.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
import os


class DQNAgent:
    def __init__(self, state_shape, n_actions, config):
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.config = config
        
        if 'device' in config:
            self.device = config['device']
        else:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        
        if config.get('dueling', True):
            self.q_network = DuelingDQN(state_shape, n_actions).to(self.device)
            self.target_network = DuelingDQN(state_shape, n_actions).to(self.device)
        else:
            self.q_network = DQNNetwork(state_shape, n_actions).to(self.device)
            self.target_network = DQNNetwork(state_shape, n_actions).to(self.device)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=config.get('learning_rate', 1e-4)
        )
        
        if config.get('prioritized_replay', True):
            self.memory = PrioritizedReplayBuffer(
                config.get('buffer_size', 100000),
                alpha=config.get('alpha', 0.6),
                beta=config.get('beta', 0.4)
            )
        else:
            self.memory = ReplayBuffer(config.get('buffer_size', 100000))
        
        self.batch_size = config.get('batch_size', 32)
        self.gamma = config.get('gamma', 0.99)
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.target_update = config.get('target_update', 1000)
        self.learning_starts = config.get('learning_starts', 1000)
        self.steps = 0
        
    def act(self, state, training=True):
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def update(self):
        if len(self.memory) < self.batch_size or self.steps < self.learning_starts:
            return
        
        self.steps += 1
        
        if isinstance(self.memory, PrioritizedReplayBuffer):
            states, actions, rewards, next_states, dones, indices, weights = \
                self.memory.sample(self.batch_size)
            weights = weights.unsqueeze(1).to(self.device)
        else:
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            weights = torch.ones(self.batch_size, 1).to(self.device)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones.float()) * self.gamma * next_q_values
            target_q_values = target_q_values.unsqueeze(1)
        
        loss = (weights * (current_q_values - target_q_values) ** 2).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
        self.optimizer.step()
        
        if isinstance(self.memory, PrioritizedReplayBuffer):
            priorities = (current_q_values - target_q_values).abs().detach().numpy().flatten()
            self.memory.update_priorities(indices, priorities)
        
        if self.steps % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, path):
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']