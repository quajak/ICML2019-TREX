import random
from typing import TYPE_CHECKING
import numpy as np
import torch
import torch.nn as nn
import wandb

if TYPE_CHECKING:
    from gt_traj_dataset import GTTrajDataset

class Classifier(nn.Module):
    def __init__(self, env_name, robomimic=False, include_action=False, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(Classifier, self).__init__()
        self.num_timesteps = 1
        self.include_action = include_action
        feature_map = {
            'can': 23,
            'lift': 19
        }
        self.num_features = feature_map[env_name]
        self.num_actions = 7 if robomimic else 3
        self.device = device
        self.robomimic = robomimic
        
        self.model = nn.Sequential(
            nn.Linear(self.num_timesteps * self.num_features + (self.num_actions if self.include_action else 0), 128),
            nn.ReLU(),
            # nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Linear(128, 2)
        ).to(device)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)  # Flatten the timesteps and features
        return self.model(x)

    def predict(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)

    def train_classifier(self, dataset: 'GTTrajDataset', val_dataset: 'GTTrajDataset'):
        config = {
            "batch_size": 64,
            "epochs": 900 if self.robomimic else 20,
            "weight_decay": 1e-4
        }
        train_trajectories = dataset.trajs
        val_trajectories = val_dataset.trajs
        
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4, weight_decay=config['weight_decay'])
        criterion = nn.CrossEntropyLoss()

        steps_per_epoch = 2 * len(train_trajectories) // config['batch_size']

        for epoch in range(config['epochs']):
            total_loss = 0
            for step in range(steps_per_epoch):
                selected_trajectories = random.choices(train_trajectories, k=config['batch_size'])
                x = []
                y = []
                for traj in selected_trajectories:
                    full_states = traj[1]  # States from trajectory
                    pos = np.random.randint(0, len(full_states) - self.num_timesteps)
                    if self.include_action:
                        x.append(np.concatenate([full_states[pos:pos+self.num_timesteps], traj[2].reshape(-1, self.num_actions)[pos+self.num_timesteps - 1][None, :]], axis=1))
                    else:
                        x.append(full_states[pos:pos+self.num_timesteps])
                    if self.robomimic:
                        y.append(traj[4])
                    else:
                        y.append(1 if traj[4] > 20 else 0)  # Success if max_x > 20
                
                x = torch.FloatTensor(np.array(x)).to(self.device)
                labels = torch.LongTensor(y).to(self.device)

                optimizer.zero_grad()
                logits = self.forward(x)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()


            if epoch % 30 == 0:
                # Validation step
                val_x = []
                val_y = []
                for traj in val_trajectories:
                    full_states = traj[1]
                    pos = np.random.randint(0, len(full_states) - self.num_timesteps)
                    if self.include_action:
                        val_x.append(np.concatenate([full_states[pos:pos+self.num_timesteps], traj[2].reshape(-1, self.num_actions)[pos+self.num_timesteps - 1][None, :]], axis=1))
                    else:
                        val_x.append(full_states[pos:pos+self.num_timesteps])
                    if self.robomimic:
                        val_y.append(traj[4])
                    else:
                        val_y.append(1 if traj[4] > 20 else 0)
            
                val_x = torch.FloatTensor(np.array(val_x)).to(self.device)
                val_labels = torch.LongTensor(val_y).to(self.device)
                
                with torch.no_grad():
                    val_logits = self.forward(val_x)
                    val_loss = criterion(val_logits, val_labels)
                    val_preds = torch.argmax(val_logits, dim=1)
                    val_acc = (val_preds == val_labels).float().mean().item()

                print(f"Step {epoch * steps_per_epoch} - Train Loss: {total_loss / steps_per_epoch}, Val Loss: {val_loss.item()}, Val Acc: {val_acc}")
                wandb.log({
                    "classifier/train_loss": total_loss / steps_per_epoch,
                    "classifier/val_loss": val_loss.item(),
                    "classifier/val_acc": val_acc
                })