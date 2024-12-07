import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb

from gt_traj_dataset import GTTrajLevelDataset

class Model(nn.Module):
    def __init__(self, model_num: int, include_action, ob_dim, ac_dim, batch_size=64, num_layers=2, 
                 embedding_dims=256, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(Model, self).__init__()
        self.model_num = model_num
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
        acc = (logits.argmax(dim=1).squeeze() == labels).float().mean().item()
        return loss, acc

    def do_step(self, dataset: GTTrajLevelDataset, batch_size: int, l2_reg: float, noise_level: float) -> tuple[torch.Tensor, float]:
        # Get batch of trajectory pairs from dataset
        D = dataset.sample(batch_size, include_action=self.include_action)
        
        # Convert samples to tensors
        x = torch.FloatTensor(np.stack([x for x, _, _ in D])).to(self.device)
        y = torch.FloatTensor(np.stack([y for _, y, _ in D])).to(self.device)
        labels = torch.LongTensor([l for _, _, l in D]).to(self.device)
        
        # Add noise to labels
        if noise_level > 0:
            noise = torch.bernoulli(torch.full_like(labels.float(), noise_level))
            labels = (labels + noise.long()) % 2
        
        return self.train_step(x, y, labels, l2_reg)

    def train_model(self, dataset: GTTrajLevelDataset, val_dataset: GTTrajLevelDataset, batch_size: int, num_epochs: int = 10000, l2_reg: float = 0.01, 
                   noise_level: float = 0.1, debug: bool = False, run = None):
        optimizer = optim.Adam(self.parameters(), lr=5e-5)
        
        for epoch in tqdm(range(num_epochs)):
            # Training step
            optimizer.zero_grad()
            loss, acc = self.do_step(dataset, batch_size, l2_reg, noise_level)
            loss.backward()
            optimizer.step()
            step_info = {f"model_{self.model_num}/loss": loss.item(), f"model_{self.model_num}/step": epoch, f"model_{self.model_num}/acc": acc}

            if epoch % 100 == 0:
                # Validation step
                val_loss, val_acc = self.do_step(val_dataset, batch_size, l2_reg, noise_level)
                val_info = {f"model_{self.model_num}/val_loss": val_loss.item(), f"model_{self.model_num}/val_acc": val_acc}
                step_info.update(val_info)
                if debug:
                    tqdm.write(f"Epoch {epoch} - Loss: {loss.item()} - Val Loss: {val_loss.item()} - Acc: {acc} - Val Acc: {val_acc}")
            run.log(step_info)
