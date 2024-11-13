import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from tqdm import tqdm
import matplotlib
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from model import Model
from custom_reward_wrapper import TorchPreferenceRewardNormalized, VecTorchPreferenceRewardNormalized
matplotlib.use('agg')
from matplotlib import pyplot as plt
from imgcat import imgcat
from wandb.integration.sb3 import WandbCallback


class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space
        self.model_path = 'random_agent'

    def act(self, observation, reward, done):
        return self.action_space.sample()[None]

class GTDataset(object):
    def __init__(self,env):
        self.env = env
        self.unwrapped = env
        while hasattr(self.unwrapped,'env'):
            self.unwrapped = self.unwrapped.env

    def gen_traj(self, agent: PPO, min_length: int):
        max_x_pos = -99999

        first_obs, _ = self.env.reset()
        obs, actions, rewards = [first_obs], [], []
        while len(obs) < 1000:  # we collect 1000 steps of trajectory for each agent
            action, _ = agent.predict(obs[-1], None, None)
            ob, reward, done, _, info = self.env.step(action)
            if len(action.shape) == 0:
                action = action[None]
            if info['x_position'] > max_x_pos:
                max_x_pos = info['x_position']

            obs.append(ob)
            actions.append(action)
            rewards.append(reward)

            if done:
                if len(obs) < min_length:
                    obs.pop()
                    ob, _= self.env.reset()
                    obs.append(ob)
                else:
                    obs.pop()
                    break

        return (np.stack(obs, axis=0), np.concatenate(actions, axis=0), np.array(rewards)), max_x_pos

    def prebuilt(self,agents,min_length):
        assert len(agents)>0, 'no agent given'
        trajs = []
        for agent in tqdm(agents):
            traj, max_x_pos = self.gen_traj(agent,min_length)

            trajs.append(traj)
            tqdm.write('model: %s avg reward: %f max_x_pos: %f'%(agent.model_path, np.sum(traj[2]), max_x_pos))
        obs,actions,rewards = zip(*trajs)
        self.trajs = (np.concatenate(obs,axis=0),np.concatenate(actions,axis=0),np.concatenate(rewards,axis=0))

        print(self.trajs[0].shape,self.trajs[1].shape,self.trajs[2].shape)

    def sample(self,num_samples,steps=40, include_action=False):
        obs, actions, rewards = self.trajs

        D = []
        for _ in range(num_samples):
            x_ptr = np.random.randint(len(obs)-steps)
            y_ptr = np.random.randint(len(obs)-steps)

            if include_action:
                D.append((np.concatenate((obs[x_ptr:x_ptr+steps],actions[x_ptr:x_ptr+steps]),axis=1),
                         np.concatenate((obs[y_ptr:y_ptr+steps],actions[y_ptr:y_ptr+steps]),axis=1),
                         0 if np.sum(rewards[x_ptr:x_ptr+steps]) > np.sum(rewards[y_ptr:y_ptr+steps]) else 1)
                        )
            else:
                D.append((obs[x_ptr:x_ptr+steps],
                          obs[y_ptr:y_ptr+steps],
                          0 if np.sum(rewards[x_ptr:x_ptr+steps]) > np.sum(rewards[y_ptr:y_ptr+steps]) else 1)
                        )

        return D

class GTTrajLevelDataset(GTDataset):
    def __init__(self,env):
        super().__init__(env)

    def prebuilt(self,agents,min_length):
        assert len(agents)>0, 'no agent is given'

        tqdm.write("Generating Trajectories")
        trajs = []
        max_max_x = -99999
        for agent_idx,agent in enumerate(tqdm(agents)):
            (obs,actions,rewards), max_x = self.gen_traj(agent,min_length)
            trajs.append((agent_idx,obs,actions,rewards))
            tqdm.write(f'Generated trajectory for agent {agent_idx} with reward {np.sum(rewards)} and max_x {max_x}')
            if max_x > max_max_x:
                max_max_x = max_x

        print('best max_x:', max_max_x)
        self.trajs = trajs

        _idxes = np.argsort([np.sum(rewards) for _,_,_,rewards in self.trajs]) # rank 0 is the most bad demo.
        self.trajs_rank = np.empty_like(_idxes)
        self.trajs_rank[_idxes] = np.arange(len(_idxes))

    def sample(self, num_samples, steps=50,include_action=False):
        D = []
        GT_preference = []
        for _ in range(num_samples):
            x_idx,y_idx = np.random.choice(len(self.trajs),2,replace=False)

            x_traj = self.trajs[x_idx]
            y_traj = self.trajs[y_idx]

            x_ptr = np.random.randint(len(x_traj[1])-steps)
            y_ptr = np.random.randint(len(y_traj[1])-steps)

            if include_action:
                D.append((np.concatenate((x_traj[1][x_ptr:x_ptr+steps],x_traj[2][x_ptr:x_ptr+steps]),axis=1),
                          np.concatenate((y_traj[1][y_ptr:y_ptr+steps],y_traj[2][y_ptr:y_ptr+steps]),axis=1),
                          0 if self.trajs_rank[x_idx] > self.trajs_rank[y_idx]  else 1)
                        )
            else:
                D.append((x_traj[1][x_ptr:x_ptr+steps],
                          y_traj[1][y_ptr:y_ptr+steps],
                          0 if self.trajs_rank[x_idx] > self.trajs_rank[y_idx]  else 1)
                        )

            GT_preference.append(0 if np.sum(x_traj[3][x_ptr:x_ptr+steps]) > np.sum(y_traj[3][y_ptr:y_ptr+steps]) else 1)

        # print('------------------')
        # _,_,preference = zip(*D)
        # preference = np.array(preference).astype(bool)
        # GT_preference = np.array(GT_preference).astype(bool)
        # print('Quality of time-indexed preference (0-1):', np.count_nonzero(preference == GT_preference) / len(preference))
        # print('------------------')

        return D
    
class GTTrajLevelNoStepsDataset(GTTrajLevelDataset):
    def __init__(self,env,max_steps):
        super().__init__(env)
        self.max_steps = max_steps

    def prebuilt(self,agents,min_length):
        assert len(agents)>0, 'no agent is given'

        trajs = []
        for agent_idx,agent in enumerate(tqdm(agents)):
            agent_trajs = []
            while np.sum([len(obs) for obs,_,_ in agent_trajs])  < min_length:
                (obs,actions,rewards),_ = self.gen_traj(agent,-1)
                agent_trajs.append((obs,actions,rewards))
            trajs.append(agent_trajs)

        agent_rewards = [np.mean([np.sum(rewards) for _,_,rewards in agent_trajs]) for agent_trajs in trajs]

        self.trajs = trajs

        _idxes = np.argsort(agent_rewards) # rank 0 is the most bad demo.
        self.trajs_rank = np.empty_like(_idxes)
        self.trajs_rank[_idxes] = np.arange(len(_idxes))

    def sample(self,num_samples,steps=None,include_action=False):
        assert steps == None

        D = []
        GT_preference = []
        for _ in tqdm(range(num_samples)):
            x_idx,y_idx = np.random.choice(len(self.trajs),2,replace=False)

            x_traj = self.trajs[x_idx][np.random.choice(len(self.trajs[x_idx]))]
            y_traj = self.trajs[y_idx][np.random.choice(len(self.trajs[y_idx]))]

            if len(x_traj[0]) > self.max_steps:
                ptr = np.random.randint(len(x_traj[0])-self.max_steps)
                x_slice = slice(ptr,ptr+self.max_steps)
            else:
                x_slice = slice(len(x_traj[1]))

            if len(y_traj[0]) > self.max_steps:
                ptr = np.random.randint(len(y_traj[0])-self.max_steps)
                y_slice = slice(ptr,ptr+self.max_steps)
            else:
                y_slice = slice(len(y_traj[0]))

            if include_action:
                D.append((np.concatenate((x_traj[0][x_slice],x_traj[1][x_slice]),axis=1),
                          np.concatenate((y_traj[0][y_slice],y_traj[1][y_slice]),axis=1),
                          0 if self.trajs_rank[x_idx] > self.trajs_rank[y_idx]  else 1)
                        )
            else:
                D.append((x_traj[0][x_slice],
                          y_traj[0][y_slice],
                          0 if self.trajs_rank[x_idx] > self.trajs_rank[y_idx]  else 1)
                        )

            GT_preference.append(0 if np.sum(x_traj[2][x_slice]) > np.sum(y_traj[2][y_slice]) else 1)

        print('------------------')
        _,_,preference = zip(*D)
        preference = np.array(preference).astype(np.bool)
        GT_preference = np.array(GT_preference).astype(np.bool)
        print('Quality of time-indexed preference (0-1):', np.count_nonzero(preference == GT_preference) / len(preference))
        print('------------------')

        return D

class GTTrajLevelNoSteps_Noise_Dataset(GTTrajLevelNoStepsDataset):
    def __init__(self,env,max_steps,ranking_noise=0):
        super().__init__(env,max_steps)
        self.ranking_noise = ranking_noise

    def prebuilt(self,agents,min_length):
        super().prebuilt(agents,min_length)

        original_trajs_rank = self.trajs_rank.copy()
        for _ in range(self.ranking_noise):
            x = np.random.randint(len(self.trajs)-1)

            x_ptr = np.where(self.trajs_rank==x)
            y_ptr = np.where(self.trajs_rank==x+1)
            self.trajs_rank[x_ptr], self.trajs_rank[y_ptr] = x+1, x

        from itertools import combinations
        order_correctness = [
            (self.trajs_rank[x] < self.trajs_rank[y]) == (original_trajs_rank[x] < original_trajs_rank[y])
            for x,y in combinations(range(len(self.trajs)),2)]
        print('Total Order Correctness: %f'%(np.count_nonzero(order_correctness)/len(order_correctness)))

class GTTrajLevelNoSteps_N_Mix_Dataset(GTTrajLevelNoStepsDataset):
    def __init__(self,env,N,max_steps):
        super().__init__(env,max_steps)

        self.N = N
        self.max_steps = max_steps

    def sample(self,*kargs,**kwargs):
        return None

    def batch(self,batch_size,include_action):
        #self.trajs = trajs
        #self.trajs_rank = np.argsort([np.sum(rewards) for _,_,_,rewards in self.trajs]) # rank 0 is the most bad demo.
        xs = []
        ys = []

        for _ in range(batch_size):
            idxes = np.random.choice(len(self.trajs),2*self.N)

            ranks = self.trajs_rank[idxes]
            bad_idxes = [idxes[i] for i in np.argsort(ranks)[:self.N]]
            good_idxes = [idxes[i] for i in np.argsort(ranks)[self.N:]]

            def _pick_and_merge(idxes):
                inp = []
                for idx in idxes:
                    obs, acs, rewards = self.trajs[idx][np.random.choice(len(self.trajs[idx]))]

                    if len(obs) > self.max_steps:
                        ptr = np.random.randint(len(obs)-self.max_steps)
                        slc = slice(ptr,ptr+self.max_steps)
                    else:
                        slc = slice(len(obs))

                    if include_action:
                        inp.append(np.concatenate([obs[slc],acs[slc]],axis=1))
                    else:
                        inp.append(obs[slc])
                return np.concatenate(inp,axis=0)

            x = _pick_and_merge(bad_idxes)
            y = _pick_and_merge(good_idxes)

            xs.append(x)
            ys.append(y)

        x_split = np.array([len(x) for x in xs])
        y_split = np.array([len(y) for y in ys])
        xs = np.concatenate(xs,axis=0)
        ys = np.concatenate(ys,axis=0)

        return xs, ys, x_split, y_split, np.ones((batch_size,)).astype(np.int32)

class LearnerDataset(GTTrajLevelDataset):
    def __init__(self,env,min_margin):
        super().__init__(env)
        self.min_margin = min_margin

    def sample(self,num_samples,steps=40,include_action=False):
        D = []
        GT_preference = []
        for _ in tqdm(range(num_samples)):
            x_idx,y_idx = np.random.choice(len(self.trajs),2,replace=False)
            while abs(self.trajs[x_idx][0] - self.trajs[y_idx][0]) < self.min_margin:
                x_idx,y_idx = np.random.choice(len(self.trajs),2,replace=False)

            x_traj = self.trajs[x_idx]
            y_traj = self.trajs[y_idx]

            x_ptr = np.random.randint(len(x_traj[1])-steps)
            y_ptr = np.random.randint(len(y_traj[1])-steps)

            if include_action:
                D.append((np.concatenate((x_traj[1][x_ptr:x_ptr+steps],x_traj[2][x_ptr:x_ptr+steps]),axis=1),
                         np.concatenate((y_traj[1][y_ptr:y_ptr+steps],y_traj[2][y_ptr:y_ptr+steps]),axis=1),
                         0 if x_traj[0] > y_traj[0] else 1)
                        )
            else:
                D.append((x_traj[1][x_ptr:x_ptr+steps],
                          y_traj[1][y_ptr:y_ptr+steps],
                         0 if x_traj[0] > y_traj[0] else 1)
                        )

            GT_preference.append(0 if np.sum(x_traj[3][x_ptr:x_ptr+steps]) > np.sum(y_traj[3][y_ptr:y_ptr+steps]) else 1)

        print('------------------')
        _,_,preference = zip(*D)
        preference = np.array(preference).astype(np.bool)
        GT_preference = np.array(GT_preference).astype(np.bool)
        print('Quality of time-indexed preference (0-1):', np.count_nonzero(preference == GT_preference) / len(preference))
        print('------------------')

        return D


def train(args):
    logdir = Path(args.log_dir)

    if logdir.exists():
        c = 'y' # input('log dir already exists. continue to train a preference model? [Y/etc]? ')
        if len(c) == 0 or 'y' in c.lower():
            import shutil
            shutil.rmtree(str(logdir))
        else:
            print('good bye')
            return

    logdir.mkdir(parents=True)
    with open(str(logdir/'args.txt'), 'w') as f:
        f.write(str(args))

    logdir = str(logdir)
    env = gym.make(args.env_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize agents
    train_agents = [RandomAgent(env.action_space)] if args.random_agent else []

    models = sorted([p for p in (Path(args.learners_path) / args.env_id).glob('*') if int(p.name.strip(".zip")) <= args.max_chkpt])
    for path in models:
        agent = PPO.load(path, device='cpu')
        setattr(agent,'model_path',str(path))
        print(f'Loaded agent from {path}')
        train_agents.append(agent)

    # Initialize dataset based on preference type
    if args.preference_type == 'gt':
        dataset = GTDataset(env)
    elif args.preference_type == 'gt_traj':
        dataset = GTTrajLevelDataset(env)
    elif args.preference_type == 'gt_traj_no_steps':
        dataset = GTTrajLevelNoStepsDataset(env, args.max_steps)
    elif args.preference_type == 'gt_traj_no_steps_noise':
        dataset = GTTrajLevelNoSteps_Noise_Dataset(env, args.max_steps, args.traj_noise)
    elif args.preference_type == 'gt_traj_no_steps_n_mix':
        dataset = GTTrajLevelNoSteps_N_Mix_Dataset(env, args.N, args.max_steps)
    elif args.preference_type == 'time':
        dataset = LearnerDataset(env, args.min_margin)
    else:
        raise ValueError('Invalid preference type')

    dataset.prebuilt(train_agents, args.min_length)

    # Initialize models
    models = []
    for i in range(args.num_models):
        model = Model(
            include_action=args.include_action,
            ob_dim=env.observation_space.shape[0],
            ac_dim=env.action_space.shape[0],
            num_layers=args.num_layers,
            embedding_dims=args.embedding_dims,
            device=device
        )
        models.append(model)

    # Train each model
    for i, model in enumerate(iterable=models):
        print(f"Training model {i+1}/{args.num_models}")
        
        D = dataset.sample(args.D, args.steps, include_action=args.include_action)

        if D is None:
            model.train_model(
                dataset,
                batch_size=64,
                num_epochs=args.iter,
                l2_reg=args.l2_reg,
                noise_level=args.noise,
                debug=True
            )
        else:
            # Convert dataset to PyTorch format
            x_data = [torch.FloatTensor(x) for x, _, _ in D]
            y_data = [torch.FloatTensor(y) for _, y, _ in D]
            labels = torch.LongTensor([l for _, _, l in D])
            
            dataset_torch = {
                'x': x_data,
                'y': y_data,
                'labels': labels
            }
            
            model.train_model(
                # dataset_torch,
                dataset,
                batch_size=args.batch_size,
                num_epochs=args.iter,
                l2_reg=args.l2_reg,
                noise_level=args.noise,
                debug=True
            )

        # Save model
        torch.save(model.state_dict(), f"{logdir}/model_{i}.pt")
        # Save model configuration separately for loading
        model_config = {
            'include_action': args.include_action,
            'ob_dim': env.observation_space.shape[0],
            'ac_dim': env.action_space.shape[0],
            'num_layers': args.num_layers,
            'embedding_dims': args.embedding_dims
        }
        torch.save(model_config, f"{logdir}/model_{i}_config.pt")

def eval(args):
    logdir = str(Path(args.logbase_path) / args.env_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(args.env_id)

    # Initialize validation agents
    valid_agents = []
    models = sorted(Path(args.learners_path).glob('?????'))
    for path in models:
        if path.name > args.max_chkpt:
            continue
        agent = PPO2Agent(env, str(path), stochastic=args.stochastic, device=device)
        valid_agents.append(agent)

    # Initialize test agents
    test_agents = []
    for i, path in enumerate(models):
        if i % 10 == 0:
            agent = PPO2Agent(env, str(path), stochastic=args.stochastic, device=device)
            test_agents.append(agent)

    # Prepare datasets
    gt_dataset = GTDataset(env)
    gt_dataset.prebuilt(valid_agents, -1)

    gt_dataset_test = GTDataset(env)
    gt_dataset_test.prebuilt(test_agents, -1)

    # Load and evaluate models
    for i in range(args.num_models):
        # Load model configuration
        model_config = torch.load(f"{logdir}/model_{i}_config.pt")
        model = Model(**model_config, device=device)
        model.load_state_dict(torch.load(f"{logdir}/model_{i}.pt"))
        model.eval()

        print(f'Evaluating model {i}')
        
        # Convert numpy arrays to torch tensors
        obs, acs, r = gt_dataset.trajs
        obs_tensor = torch.FloatTensor(obs).to(device)
        acs_tensor = torch.FloatTensor(acs).to(device) if args.include_action else None
        
        # Get predicted rewards
        with torch.no_grad():
            r_hat = model.compute_reward(obs_tensor, acs_tensor).squeeze()

        # Do the same for test data
        obs_test, acs_test, r_test = gt_dataset_test.trajs
        obs_test_tensor = torch.FloatTensor(obs_test).to(device)
        acs_test_tensor = torch.FloatTensor(acs_test).to(device) if args.include_action else None
        
        with torch.no_grad():
            r_hat_test = model.compute_reward(obs_test_tensor, acs_test_tensor).squeeze()

        # Plot results
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(r, r_hat, 'o', alpha=0.5)
        axes[0].set_title('Training Data')
        axes[0].set_xlabel('True Rewards')
        axes[0].set_ylabel('Predicted Rewards')
        
        axes[1].plot(r_test, r_hat_test, 'o', alpha=0.5)
        axes[1].set_title('Test Data')
        axes[1].set_xlabel('True Rewards')
        axes[1].set_ylabel('Predicted Rewards')
        
        plt.tight_layout()
        fig.savefig(f'model_{i}_evaluation.png')
        imgcat(fig)
        plt.close(fig)

        # Save evaluation results
        np.savez(
            f'model_{i}_evaluation.npz',
            r=r,
            r_hat=r_hat.cpu().numpy(),
            r_test=r_test,
            r_hat_test=r_hat_test.cpu().numpy()
        )

if __name__ == "__main__":
    # Required Args (target envs & learners)
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_id', default='', help='Select the environment to run')
    parser.add_argument('--env_type', default='', help='mujoco or atari')
    parser.add_argument('--learners_path', default='', help='path of learning agents')
    parser.add_argument('--max_chkpt', default=64000, type=int, help='decide upto what learner stage you want to give')
    parser.add_argument('--steps', default=40, type=int, help='length of snippets')
    parser.add_argument('--max_steps', default=100_000, type=int, help='length of max snippets (gt_traj_no_steps only)')  # idk
    parser.add_argument('--traj_noise', default=None, type=int, help='number of adjacent swaps (gt_traj_no_steps_noise only)')
    parser.add_argument('--min_length', default=1000,type=int, help='minimum length of trajectory generated by each agent')
    parser.add_argument('--num_layers', default=3, type=int, help='number layers of the reward network')
    parser.add_argument('--embedding_dims', default=256, type=int, help='embedding dims')
    parser.add_argument('--num_models', default=5, type=int, help='number of models to ensemble')
    parser.add_argument('--l2_reg', default=0.01, type=float, help='l2 regularization size')
    parser.add_argument('--noise', default=0.1, type=float, help='noise level to add on training label')
    parser.add_argument('--D', default=1000, type=int, help='|D| in the preference paper')
    parser.add_argument('--N', default=10, type=int, help='number of trajactory mix (gt_traj_no_steps_n_mix only)')
    parser.add_argument('--log_dir', required=True)
    parser.add_argument('--preference_type', default='gt_traj', help='gt or gt_traj or time or gt_traj_no_steps, gt_traj_no_steps_n_mix; if gt then preference will be given as a GT reward, otherwise, it is given as a time index')
    parser.add_argument('--min_margin', default=1, type=int, help='when prefernce type is "time", the minimum margin that we can assure there exist a margin')
    parser.add_argument('--include_action', action='store_true', help='whether to include action for the model or not')
    parser.add_argument('--stochastic', action='store_true', help='whether want to use stochastic agent or not')
    parser.add_argument('--random_agent', action='store_true', help='whether to use default random agent')
    parser.add_argument('--eval', action='store_true', help='path to log base (env_id will be concatenated at the end)')
    # Args for PPO
    parser.add_argument('--rl_runs', default=1, type=int)
    parser.add_argument('--ppo_log_path', default='ppo2')
    parser.add_argument('--custom_reward', default="preference_normalized", help='preference or preference_normalized')
    parser.add_argument('--ctrl_coeff', default=0.0, type=float)
    parser.add_argument('--alive_bonus', default=0.0, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--ppo_policy', default="MlpPolicy", type=str)
    parser.add_argument('--batch_size', default=64, type=int)  # from paper section 5.1.2
    parser.add_argument('--iter', default=10000, type=int)
    parser.add_argument('--num_envs', default=4, type=int)
    args = parser.parse_args()

    if not args.eval :
        # Train a Preference Model
        train(args)

        # Train an agent
        import time
        group = str(time.time())
        config = {
            "policy_type": "MlpPolicy",
            "total_timesteps": 100_000,
            "env_id": args.env_id,
        }

        run = wandb.init(
            project="trex-training",
            group=group,
            config=config,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
        )
        wandb_callback = WandbCallback(
                model_save_path=f"models/{run.id}",
                verbose=2,
            )

        for i in range(args.rl_runs):
            def make_indiv_env():
                env = gym.make(config["env_id"])
                env = TorchPreferenceRewardNormalized(env, args.num_models, args.log_dir, args.include_action, args.num_layers, args.embedding_dims)
                return env
            env = DummyVecEnv([make_indiv_env for _ in range(args.num_envs)])
            model = PPO(config['policy_type'], env, n_steps=128//args.num_envs, device="cpu", tensorboard_log=f"runs/{run.id}")

            model.learn(
                total_timesteps=config["total_timesteps"],
                callback=[wandb_callback],
            )
            env.close()

            # Evaluate the trained model
            eval_env = gym.make(config["env_id"])
            max_xs = []
            for k in range(30):
                print(f'Evaluation Run {k}')
                obs, _ = eval_env.reset()
                total_reward = 0
                num_steps = 0
                done = False
                terminated = False
                max_x_pos = -np.inf
                
                while not done and not terminated and num_steps < 1000:
                    action, _ = model.predict(obs)
                    obs, reward, done, terminated, info = eval_env.step(action)
                    total_reward += reward
                    num_steps += 1
                    max_x_pos = max(max_x_pos, info["x_position"])
                        
                run.log({
                    "total_reward": total_reward,
                    "total_steps": num_steps,
                    "max_x_pos": max_x_pos
                })
                max_xs.append(max_x_pos)
                print(f'Run {k} - Total Reward: {total_reward}, Total Steps: {num_steps}, Max X Pos: {max_x_pos}')
            print(f'Average Max X Pos: {np.mean(max_xs)} STD Max X Pos: {np.std(max_xs)}')
            eval_env.close()

            run.finish()
    else:
        # eval(args)

        import os
        from performance_checker import gen_traj_dist as get_perf
        #from performance_checker import gen_traj_return as get_perf

        env = gym.make(args.env_id)

        agents_dir = Path(os.path.abspath(os.path.join(args.log_dir,args.ppo_log_path)))
        trained_steps = sorted(list(set([path.name for path in agents_dir.glob('run_*/checkpoints/?????')])))
        print(trained_steps)
        print(str(agents_dir))
        for step in trained_steps[::-1]:
            perfs = []
            for i in range(args.rl_runs):
                path = agents_dir/('run_%d'%i)/'checkpoints'/step

                if path.exists() == False:
                    continue

                agent = PPO2Agent(env,args.env_type,str(path),stochastic=args.stochastic)
                perfs += [
                    get_perf(env,agent) for _ in range(5)
                ]
                print('[%s-%d] %f %f'%(step,i,np.mean(perfs[-5:]),np.std(perfs[-5:])))

            print('[%s] %f %f %f %f'%(step,np.mean(perfs),np.std(perfs),np.max(perfs),np.min(perfs)))

            #break
