import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class Model(nn.Module):
    def __init__(self, include_action, ob_dim, ac_dim, batch_size=64, num_layers=2, 
                 embedding_dims=256, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(Model, self).__init__()
        self.include_action = include_action
        self.device = device
        in_dims = ob_dim + ac_dim if include_action else ob_dim
        
        # Build network
        layers = []
        last_dim = in_dims
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(last_dim, embedding_dims),
                nn.ReLU()
            ])
            last_dim = embedding_dims
        layers.append(nn.Linear(last_dim, 1))
        
        self.network = nn.Sequential(*layers)
        self.to(device)

    def forward(self, x):
        return self.network(x)

    def compute_reward(self, obs, acs=None, batch_size=1024):
        with torch.no_grad():
            if self.include_action:
                inp = torch.cat((obs, acs), dim=1)
            else:
                inp = obs
            
            rewards = []
            for i in range(0, len(obs), batch_size):
                batch = inp[i:i+batch_size].to(self.device)
                reward = self.network(batch)
                rewards.append(reward.cpu())
            
            return torch.cat(rewards, dim=0).numpy()

    def train_step(self, x, y, labels, l2_reg):
        self.train()
        
        # Compute sum of rewards for each trajectory
        x_rewards = self.network(x).sum(dim=1)  # Sum over time dimension
        y_rewards = self.network(y).sum(dim=1)  # Sum over time dimension
        
        # Compute loss using logits [x_rewards, y_rewards]
        logits = torch.stack([x_rewards, y_rewards], dim=1)
        loss = nn.CrossEntropyLoss()(logits.squeeze(-1), labels)
        
        # L2 regularization
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.norm(param)
        loss += l2_reg * l2_loss
        
        return loss

    def train_model(self, dataset, batch_size, num_epochs=10000, l2_reg=0.01, 
                   noise_level=0.1, debug=False):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        traj_trained_on = 0
        
        for epoch in tqdm(range(num_epochs)):
            # Get batch of trajectory pairs from dataset
            D = dataset.sample(batch_size, include_action=self.include_action)
            traj_trained_on += len(D)
            
            # Convert samples to tensors
            x = torch.FloatTensor(np.stack([x for x, _, _ in D])).to(self.device)
            y = torch.FloatTensor(np.stack([y for _, y, _ in D])).to(self.device)
            labels = torch.LongTensor([l for _, _, l in D]).to(self.device)
            
            # Add noise to labels
            if noise_level > 0:
                noise = torch.bernoulli(torch.full_like(labels.float(), noise_level))
                labels = (labels + noise.long()) % 2
            
            # Training step
            optimizer.zero_grad()
            loss = self.train_step(x, y, labels, l2_reg)
            loss.backward()
            optimizer.step()
            
            if debug and (epoch % 100 == 0 or epoch < 10):
                tqdm.write(f'Step {epoch}, Loss: {loss.item():.4f}')

        print(f'Trained on {traj_trained_on} trajectories')
