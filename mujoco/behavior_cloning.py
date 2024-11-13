import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from tqdm import tqdm
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from imgcat import imgcat

# Assume PPO2Agent is imported from appropriate location
from preference_learning import PPO2Agent

class Policy(nn.Module):
    def __init__(self, ob_dim, ac_dim, embedding_dims=512, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(Policy, self).__init__()
        self.device = device
        
        # Build network
        self.network = nn.Sequential(
            nn.Linear(ob_dim, embedding_dims),
            nn.ReLU(),
            nn.Linear(embedding_dims, embedding_dims),
            nn.ReLU(),
            nn.Linear(embedding_dims, embedding_dims),
            nn.ReLU(),
            nn.Linear(embedding_dims, embedding_dims),
            nn.ReLU(),
            nn.Linear(embedding_dims, ac_dim)
        )
        
        self.to(device)
        
    def forward(self, x):
        return self.network(x)
    
    def train_model(self, D, batch_size=64, num_epochs=20000, l2_reg=0.001, debug=False):
        obs, acs, _ = D
        
        # Split into train and validation sets
        idxes = np.random.permutation(len(obs))
        train_idxes = idxes[:int(len(obs)*0.8)]
        valid_idxes = idxes[int(len(obs)*0.8):]
        
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        
        def _batch(idx_list):
            if len(idx_list) > batch_size:
                idxes = np.random.choice(idx_list, batch_size, replace=False)
            else:
                idxes = idx_list
                
            b_ob = torch.FloatTensor(obs[idxes]).to(self.device)
            b_ac = torch.FloatTensor(acs[idxes]).to(self.device)
            
            return b_ob, b_ac
        
        for it in tqdm(range(num_epochs), dynamic_ncols=True):
            # Training batch
            b_ob, b_ac = _batch(train_idxes)
            
            # Forward pass
            pred_ac = self(b_ob)
            
            # Compute losses
            main_loss = torch.mean(torch.sum((pred_ac - b_ac)**2, dim=1))
            
            # L2 regularization
            l2_loss = 0
            for param in self.parameters():
                l2_loss += torch.norm(param)
            l2_loss *= l2_reg
            
            loss = main_loss + l2_loss
            
            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Validation
            with torch.no_grad():
                b_ob, b_ac = _batch(valid_idxes)
                pred_ac = self(b_ob)
                valid_loss = torch.mean(torch.sum((pred_ac - b_ac)**2, dim=1))
            
            if debug:
                if it % 100 == 0 or it < 10:
                    tqdm.write(f'loss: {main_loss.item():.4f} (l2_loss: {l2_loss.item():.4f}), valid_loss: {valid_loss.item():.4f}')
    
    def act(self, observation, reward, done):
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            action = self(obs_tensor)
            return action.cpu().numpy()[0]

class Dataset:
    def __init__(self, env):
        self.env = env
    
    def gen_traj(self, agent, min_length):
        obs, actions, rewards = [self.env.reset()], [], []
        
        # For debug purpose
        last_episode_idx = 0
        acc_rewards = []
        
        while True:
            action = agent.act(obs[-1], None, None)
            ob, reward, done, _ = self.env.step(action)
            
            obs.append(ob)
            actions.append(action)
            rewards.append(reward)
            
            if done:
                if len(obs) < min_length:
                    obs.pop()
                    obs.append(self.env.reset())
                    
                    acc_rewards.append(np.sum(rewards[last_episode_idx:]))
                    last_episode_idx = len(rewards)
                else:
                    obs.pop()
                    
                    acc_rewards.append(np.sum(rewards[last_episode_idx:]))
                    last_episode_idx = len(rewards)
                    break
        
        return (np.stack(obs), np.stack(actions), np.array(rewards), np.mean(acc_rewards))
    
    def prebuilt(self, agents, min_length):
        assert len(agents) > 0, 'no agent given'
        trajs = []
        for agent in tqdm(agents):
            *traj, avg_acc_reward = self.gen_traj(agent, min_length)
            
            trajs.append(traj)
            tqdm.write(f'model: {agent.model_path} avg reward: {avg_acc_reward:.4f}')
            
        obs, actions, rewards = zip(*trajs)
        self.trajs = (np.concatenate(obs), np.concatenate(actions), np.concatenate(rewards))
        
        print(self.trajs[0].shape, self.trajs[1].shape, self.trajs[2].shape)

def train(args):
    logdir = Path(args.logbase_path) / args.env_id
    if logdir.exists():
        c = input('log is already exist. continue [Y/etc]? ')
        if c in ['YES', 'yes', 'Y']:
            import shutil
            shutil.rmtree(str(logdir))
        else:
            print('good bye')
            exit()
            
    logdir.mkdir(parents=True)
    logdir = str(logdir)
    
    env = gym.make(args.env_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize agent and dataset
    agent = PPO2Agent(env, args.env_type, str(args.learner_path))
    dataset = Dataset(env)
    dataset.prebuilt([agent], args.min_length)
    
    # Initialize policy
    policy = Policy(
        ob_dim=env.observation_space.shape[0],
        ac_dim=env.action_space.shape[0],
        device=device
    )
    
    # Train policy
    D = dataset.trajs
    policy.train_model(D, l2_reg=args.l2_reg, debug=True)
    
    # Save model
    torch.save({
        'state_dict': policy.state_dict(),
        'ob_dim': env.observation_space.shape[0],
        'ac_dim': env.action_space.shape[0]
    }, f"{logdir}/model.pt")

def eval(args):
    env = gym.make(args.env_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    logdir = str(Path(args.logbase_path) / args.env_id)
    checkpoint = torch.load(f"{logdir}/model.pt")
    
    policy = Policy(
        ob_dim=checkpoint['ob_dim'],
        ac_dim=checkpoint['ac_dim'],
        device=device
    )
    policy.load_state_dict(checkpoint['state_dict'])
    policy.eval()
    
    from performance_checker import gen_traj
    from gymnasium.wrappers import Monitor
    
    perfs = []
    for j in range(args.num_trajs):
        if j == 0 and args.video_record:
            wrapped = Monitor(env, './video/', force=True)
        else:
            wrapped = env
            
        perfs.append(gen_traj(wrapped, policy, args.render, args.max_len))
    
    print(f"{logdir}, {np.mean(perfs):.4f}, {np.std(perfs):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_id', default='', help='Select the environment to run')
    parser.add_argument('--env_type', default='', help='mujoco or atari')
    parser.add_argument('--min_length', default=1000, type=int, help='minimum length of trajectory generated by each agent')
    parser.add_argument('--learner_path', default='', help='path of learning agents')
    parser.add_argument('--l2_reg', default=0.01, type=float, help='l2 regularization size')
    parser.add_argument('--logbase_path', default='./learner/models_bc/', help='path to log base (env_id will be concatenated at the end)')
    parser.add_argument('--eval', action='store_true', help='path to log base (env_id will be concatenated at the end)')
    parser.add_argument('--max_len', default=1000, type=int)
    parser.add_argument('--num_trajs', default=10, type=int, help='path of learning agents')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--video_record', action='store_true')
    args = parser.parse_args()
    
    if not args.eval:
        train(args)
    else:
        eval(args)
