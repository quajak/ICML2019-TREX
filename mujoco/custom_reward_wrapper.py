import torch
import torch.nn as nn
import numpy as np
from gymnasium.core import Wrapper
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.running_mean_std import RunningMeanStd
from model import Model

class VecTorchRandomReward(VecEnvWrapper):
    def __init__(self, venv):
        super().__init__(venv)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        rews = torch.randn(obs.shape[0], device=self.device).mean(dim=0).cpu().numpy()
        return obs, rews, dones, infos
    
    def reset(self, **kwargs):
        return self.venv.reset(**kwargs)

class VecTorchPreferenceReward(VecEnvWrapper):
    def __init__(self, venv, num_models: int, model_dir: str, include_action: bool, num_layers: int, 
                 embedding_dims: int, ctrl_coeff=0., alive_bonus=0.):
        super().__init__(venv)
        self.ctrl_coeff = ctrl_coeff
        self.alive_bonus = alive_bonus
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.models = []
        for i in range(num_models):
            model = Model(include_action, self.observation_space.shape[-1], 
                         self.action_space.shape[-1], num_layers=num_layers,
                         embedding_dims=embedding_dims, device=self.device)
            model.load_state_dict(torch.load(f"{model_dir}/model_{i}.pt", weights_only=True))
            model.eval()
            self.models.append(model)

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        acs = self.venv.last_actions
        
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        acs_tensor = torch.FloatTensor(acs).to(self.device)
        
        r_hat = np.zeros_like(rews)
        for model in self.models:
            r_hat += model.compute_reward(obs_tensor, acs_tensor)
            
        rews = r_hat / len(self.models) - self.ctrl_coeff * np.sum(acs**2, axis=1)
        rews += self.alive_bonus
        
        return obs, rews, dones, infos
    
    def reset(self, **kwargs):
        return self.venv.reset(**kwargs)

class VecTorchPreferenceRewardNormalized(VecTorchPreferenceReward):
    def __init__(self, venv, num_models, model_dir, include_action, num_layers, 
                 embedding_dims, ctrl_coeff=0., alive_bonus=0.):
        super().__init__(venv, num_models, model_dir, include_action, num_layers, 
                        embedding_dims, ctrl_coeff, alive_bonus)
        
        self.rew_rms = [RunningMeanStd(shape=()) for _ in range(num_models)]
        self.cliprew = 10.
        self.epsilon = 1e-8
        
    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        acs = self.venv.last_actions
        
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        acs_tensor = torch.FloatTensor(acs).to(self.device)
        
        r_hats = np.zeros_like(rews)
        for model, rms in zip(self.models, self.rew_rms):
            r_hat = model.compute_reward(obs_tensor, acs_tensor)
            rms.update(r_hat)
            r_hat = np.clip(r_hat / np.sqrt(rms.var + self.epsilon), 
                           -self.cliprew, self.cliprew)
            r_hats += r_hat
            
        rews = r_hats / len(self.models) - self.ctrl_coeff * np.sum(acs**2, axis=1)
        rews += self.alive_bonus
        
        return obs, rews, dones, infos

class TorchRandomReward(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # Generate random reward using PyTorch
        reward = torch.randn(1, device=self.device).item()
        return obs, reward, done, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class TorchPreferenceReward(Wrapper):
    def __init__(self, env, num_models, model_dir, include_action, num_layers, 
                 embedding_dims, ctrl_coeff=0., alive_bonus=0.):
        super().__init__(env)
        self.ctrl_coeff = ctrl_coeff
        self.alive_bonus = alive_bonus
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.last_action = None
        
        # Load models
        self.models = []
        for i in range(num_models):
            model = Model(i, include_action, self.observation_space.shape[0], 
                         self.action_space.shape[0], num_layers=num_layers,
                         embedding_dims=embedding_dims, device=self.device)
            model.load_state_dict(torch.load(f"{model_dir}/model_{i}.pt"))
            model.eval()
            self.models.append(model)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.last_action = action  # Store action for reward computation
        
        # Convert to tensors and add batch dimension
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        
        # Compute reward from all models
        r_hat = 0
        for model in self.models:
            r_hat += model.compute_reward(obs_tensor, action_tensor)[0]  # Remove batch dim
            
        # Average rewards and apply modifications
        reward = (r_hat / len(self.models) - 
                 self.ctrl_coeff * np.sum(action**2) + 
                 self.alive_bonus)
        
        return obs, reward, done, info
    
    def reset(self, **kwargs):
        self.last_action = None
        return self.env.reset(**kwargs)

class TorchPreferenceRewardNormalized(TorchPreferenceReward):
    def __init__(self, env, num_models, model_dir, include_action, num_layers, 
                 embedding_dims, ctrl_coeff=0., alive_bonus=0.):
        super().__init__(env, num_models, model_dir, include_action, num_layers, 
                        embedding_dims, ctrl_coeff, alive_bonus)
        
        self.rew_rms = [RunningMeanStd(shape=()) for _ in range(num_models)]
        self.cliprew = 10.
        self.epsilon = 1e-8
        
    def step(self, action):
        obs, reward, done, terminated, info = self.env.step(action)
        self.last_action = action
        
        # Convert to tensors and add batch dimension
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        
        # Compute normalized rewards from all models
        r_hats = 0
        for model, rms in zip(self.models, self.rew_rms):
            r_hat = model.compute_reward(obs_tensor, action_tensor)[0]  # Remove batch dim
            rms.update(np.array([r_hat]))
            r_hat = np.clip(r_hat / np.sqrt(rms.var + self.epsilon), 
                           -self.cliprew, self.cliprew)
            r_hats += r_hat
            
        # Average rewards and apply modifications
        reward = (r_hats / len(self.models) - 
                 self.ctrl_coeff * np.sum(action**2) + 
                 self.alive_bonus)
        
        return obs, reward, done, terminated, info