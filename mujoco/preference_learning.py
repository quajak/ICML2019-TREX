import argparse
from pathlib import Path
from random import randint, random
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from tqdm import tqdm
import matplotlib
import wandb
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
import time
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from PIL import Image, ImageDraw
import imageio
import cv2
import os
import robomimic.utils.file_utils as FileUtils

from model import Model
from custom_reward_wrapper import TorchClassifierReward, TorchPreferenceRewardMinMaxNormalized, TorchPreferenceRewardNormalized, TorchSigmoidReward
from gt_traj_dataset import GTTrajLevelDataset
from classifier import Classifier
from classifier_traj_dataset import ClassifierTrajDataset

def get_agent(path, algorithm, device='cpu'):
    if algorithm == 'PPO':
        agent = PPO.load(path, device='cpu')
    else:  # TD3
        agent = TD3.load(path, device=device)
    setattr(agent, 'model_path', str(path))
    return agent

def train(args: dict, run):
    logdir = Path(args.log_dir)

    if logdir.exists():
        c = 'n' # input('log dir already exists. continue to train a preference model? [Y/etc]? ')
        if len(c) == 0 or 'y' in c.lower():
            import shutil
            shutil.rmtree(str(logdir))
    else:
        logdir.mkdir(parents=True)

    with open(str(logdir/'args.txt'), 'w') as f:
        f.write(str(args))

    logdir = str(logdir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize agents
    train_agents = []

    models = sorted([p for p in (Path(args.learners_path) / args.env_id).glob('*') if int(p.name.strip(".zip").strip(".pth")) <= args.max_chkpt])
    ckpt_dict = None
    for path in models:
        if args.robomimic:
            agent, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=path, device=device, verbose=False)
        else:
            agent = get_agent(path, args.algorithm, device)
        train_agents.append(agent)

    if args.robomimic:
        train_agents = train_agents * 100
        env, _ = FileUtils.env_from_checkpoint(
            ckpt_dict=ckpt_dict, 
            render=False, 
            render_offscreen=False, 
            verbose=False,
        )
    else:
        env = gym.make(args.env_id)

    # Initialize dataset based on preference type
    if args.preference_type == 'gt_traj':
        dataset = GTTrajLevelDataset(env, args.log_dir, robomimic=args.robomimic)
        val_dataset = GTTrajLevelDataset(env, args.log_dir, val=True, robomimic=args.robomimic)
        dataset.prebuilt(train_agents, args.min_length, run)
        val_dataset.prebuilt(train_agents, args.min_length, run)
    if args.preference_type == 'classifier':
        print("Initializing classifier-based training...")
        
        # First create and train the classifier on all data
        dataset = GTTrajLevelDataset(env=env, log_dir=args.log_dir, robomimic=args.robomimic)
        dataset.prebuilt(train_agents, args.min_length, run)
        val_dataset = GTTrajLevelDataset(env, args.log_dir, val=True, robomimic=args.robomimic)
        val_dataset.prebuilt(train_agents, args.min_length, run)
    
    # Training loop iterations
    for iteration in range(args.num_iterations):
        print(f"\nStarting iteration {iteration + 1}/{args.num_iterations}")
        
        if args.preference_type == 'classifier':
            print("Training classifier...")
            classifier = Classifier(args.env_id, robomimic=args.robomimic, include_action=args.include_action, device=device, noise_level=0.1)
            classifier.train_classifier(dataset, val_dataset)
            torch.save(classifier.state_dict(), f"{args.log_dir}/classifier_iter_{iteration}.pt")
            
            # Update datasets with classifier
            classifier_dataset = ClassifierTrajDataset(dataset, classifier)
            classifier_val_dataset = ClassifierTrajDataset(val_dataset, classifier)
            
            # Use classifier datasets for training reward models
            training_dataset = classifier_dataset
            validation_dataset = classifier_val_dataset
        else:
            # Use original datasets for non-classifier training
            training_dataset = dataset
            validation_dataset = val_dataset

        # Train reward models
        print("Training reward models...")
        models = []
        for i in range(args.num_models):
            model = Model(
                model_num=i,
                include_action=args.include_action,
                ob_dim=env.observation_space.shape[0],
                ac_dim=env.action_space.shape[0],
                num_layers=args.num_layers,
                embedding_dims=args.embedding_dims,
                device=device
            )
            model.train_model(training_dataset, validation_dataset, batch_size=args.batch_size,
                            num_epochs=args.iter, l2_reg=args.l2_reg,
                            noise_level=args.noise, debug=True, run=run)
            models.append(model)
            torch.save(model.state_dict(), f"{args.log_dir}/model_{i}_iter_{iteration}.pt")

        # Log both classifier and reward models on the validation dataset
        # Log validation metrics for first 10 trajectories
        print("\nValidation metrics for first 10 trajectories:")
        print("Traj | Success | Avg Model Rewards | Classifier Score")
        print("-" * 50)
        
        for i, traj in enumerate(validation_dataset.trajs[:10]):
            # Get success status
            success = "Success" if traj[4] == 1 else "Failure"
            
            # Calculate average reward from each model
            model_rewards = []
            obs = torch.FloatTensor(traj[1]).to(device)
            if args.include_action:
                actions = torch.FloatTensor(traj[2]).to(device)
            
            # Track per-step rewards for plotting
            step_rewards = []
            for model in models:
                with torch.no_grad():
                    if args.include_action:
                        rewards = model.compute_reward(obs[:len(actions)], actions)
                    else:
                        rewards = model.compute_reward(obs)
                    model_rewards.append(np.mean(rewards))
                    step_rewards.append(rewards)
            avg_model_reward = np.mean(model_rewards)
            
            # Average rewards across models for each timestep
            step_rewards = np.mean(step_rewards, axis=0)
            
            # Get classifier predictions if using classifier
            if args.preference_type == 'classifier':
                # Get classifier scores for each timestep
                classifier_scores = []
                for t in range(len(obs)):
                    if t >= len(actions):
                        break
                    if args.include_action:
                        obs_t = torch.cat([obs[t:t+1], actions[t:t+1]], dim=1)
                    else:
                        obs_t = obs[t:t+1]
                    with torch.no_grad():
                        pred = classifier(obs_t)
                        score = torch.softmax(pred, dim=1)[0,1].item()
                        classifier_scores.append(score)
                
                avg_classifier_score = np.mean(classifier_scores)
                print(f"{i:4d} | {success:7s} | {avg_model_reward:16.3f} | {avg_classifier_score:.3f}")
                
                # Plot rewards and classifier scores
                # plt.figure(figsize=(10,5))
                # plt.subplot(1,2,1)
                # plt.plot(step_rewards)
                # plt.title(f'Trajectory {i} Rewards')
                # plt.xlabel('Step')
                # plt.ylabel('Reward')
                
                # plt.subplot(1,2,2)
                # plt.plot(classifier_scores)
                # plt.title(f'Trajectory {i} Classifier Scores')
                # plt.xlabel('Step')
                # plt.ylabel('Score')
                
                # os.makedirs(f"{args.log_dir}/trajectory_plots", exist_ok=True)
                # plt.savefig(f"{args.log_dir}/trajectory_plots/traj_{i}_iter_{iteration}.png")
                # plt.close()
                # print(f"Saved plot to {args.log_dir}/trajectory_plots/traj_{i}_iter_{iteration}.png")
            else:
                print(f"{i:4d} | {success:7s} | {avg_model_reward:16.3f} | N/A")
                
                # Plot just rewards
                # plt.figure(figsize=(6,4))
                # plt.plot(step_rewards)
                # plt.title(f'Trajectory {i} Rewards')
                # plt.xlabel('Step')
                # plt.ylabel('Reward')
                
                # os.makedirs(f"{args.log_dir}/trajectory_plots", exist_ok=True)
                # plt.savefig(f"{args.log_dir}/trajectory_plots/traj_{i}_iter_{iteration}.png")
                # plt.close()

        # Train TD3 agent
        print("Training agent...")
        all_trajectories = train_rl_agent(args, models, env, run, iteration, ckpt_dict, classifier)
        # Add all trajectories to dataset
        for traj in all_trajectories:
            dataset.add_trajectory(traj[0], traj[1], traj[2])

            
        print(f"Dataset size after iteration {iteration + 1}: {len(dataset.trajs)}")
        
    print("Training completed!")

def train_rl_agent(args, models, env, run, iteration, ckpt_dict, classifier):
    print(f"Training {args.algorithm} agent with learned reward models...")
    wandb_callback = WandbCallback(
        model_save_path=f"models/{run.id}",
        verbose=2,
    )

    agent_config = config['final_agent']
    for i in range(args.rl_runs):
        def make_indiv_env():
            if args.robomimic:
                env, _ = FileUtils.env_from_checkpoint(
                    ckpt_dict=ckpt_dict,
                    render=False,
                    render_offscreen=False,
                    verbose=False,
                )
            else:
                env = gym.make(agent_config["env_id"])
            
            env = TorchPreferenceRewardNormalized(env, args.num_models, args.log_dir, args.include_action, args.num_layers, args.embedding_dims, robomimic=args.robomimic, cliprew=args.cliprew)
            # env = TorchPreferenceRewardMinMaxNormalized(env, args.num_models, args.log_dir, args.include_action, args.num_layers, args.embedding_dims, robomimic=args.robomimic)
            # env = TorchSigmoidReward(env, args.num_models, args.log_dir, args.include_action, args.num_layers, args.embedding_dims, robomimic=args.robomimic)
            # env = TorchClassifierReward(env, classifier, robomimic=args.robomimic, convert_obs=args.include_action)
            return env

        env = DummyVecEnv([make_indiv_env for _ in range(args.num_envs)], args.robomimic)
        
        # Common parameters for both algorithms
        common_params = {
            'policy': agent_config['policy_type'],
            'env': env,
            'learning_rate': agent_config['learning_rate'],
            'device': agent_config['device'],
            'tensorboard_log': f"runs/{run.id}",
            'verbose': 0
        }
        
        model = build_rl_model(args, agent_config, common_params)

        best_model = None
        best_model_length = 10000

        class SaveShortestModelCallback(BaseCallback):
            def __init__(self, verbose=0):
                super(SaveShortestModelCallback, self).__init__(verbose)
                self.best_model = None
                
            def _on_step(self) -> bool:
                nonlocal best_model, best_model_length
                if np.mean(self.parent.evaluations_length) <= best_model_length:
                    best_model_length = np.mean(self.parent.evaluations_length)
                    self.model.save(f"{args.log_dir}/{run.id}/best_model/model_{iteration}.zip")
                    best_model = build_rl_model(args, agent_config, common_params)
                    best_model.load(f"{args.log_dir}/{run.id}/best_model/model_{iteration}.zip")
                return True
            
        model.learn(
            total_timesteps=agent_config["total_timesteps"],
            callback=[
                wandb_callback,
                EvalCallback(
                    eval_env=make_indiv_env(),
                    eval_freq=5000,
                    n_eval_episodes=10,
                    best_model_save_path=f"{args.log_dir}/{run.id}/best_model",
                    log_path=f"{args.log_dir}/{run.id}",
                    deterministic=False,
                    callback_after_eval=SaveShortestModelCallback()
                )
            ],
        )
        env.close()

    # Evaluate the the model
    print("Evaluating model with best model length: ", best_model_length)
    model = best_model

    if args.robomimic:
        eval_env, _ = FileUtils.env_from_checkpoint(
            ckpt_dict=ckpt_dict,
            render=True,
            render_offscreen=True,
            verbose=False,
        )
        # eval_env = TorchClassifierReward(eval_env, classifier, robomimic=args.robomimic, convert_obs=args.include_action)
        # eval_env = TorchSigmoidReward(eval_env, args.num_models, args.log_dir, args.include_action, args.num_layers, args.embedding_dims, robomimic=args.robomimic)
        # eval_env = TorchPreferenceRewardMinMaxNormalized(eval_env, args.num_models, args.log_dir, args.include_action, args.num_layers, args.embedding_dims, robomimic=args.robomimic)
        eval_env = TorchPreferenceRewardNormalized(eval_env, args.num_models, args.log_dir, args.include_action, args.num_layers, args.embedding_dims, robomimic=args.robomimic, cliprew=args.cliprew)
    else:
        eval_env = gym.make(agent_config["env_id"])

    # Track all trajectories during evaluation
    all_trajectories = []
    
    # During evaluation, store trajectories that exceed the reward threshold
    for eval_ep in range(30):  # TODO: Used to be 30
        obs = eval_env.reset()
        trajectory_obs = []
        trajectory_actions = []
        trajectory_rewards = []
        total_reward = 0
        frames = []
        done = False
        terminated = False
        while not done and not terminated:
            action, _ = model.predict(obs)
            next_obs, reward, done, terminated, info = eval_env.step(action)
            
            trajectory_obs.append(obs)
            trajectory_actions.append(action)
            trajectory_rewards.append(reward)
            total_reward += reward

            # Get classifier prediction if available
            pred_text = ""
            if args.preference_type == 'classifier':
                with torch.no_grad():
                    if args.include_action:
                        obs_tensor = torch.FloatTensor(np.concatenate([next_obs, action])).unsqueeze(0).to(classifier.device)
                    else:
                        obs_tensor = torch.FloatTensor(np.concatenate([next_obs])).unsqueeze(0).to(classifier.device)
                    pred = classifier(obs_tensor)
                    pred = torch.softmax(pred, dim=1)[0,1].item()
                    pred_text = f"Classifier Score: {pred:.3f}"
            
            # Render frame and add text
            frame = eval_env.render(mode='rgb_array', width=256, height=256)
            if pred_text:
                # Convert frame to PIL Image
                frame_pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(frame_pil)
                draw.text((10, 10), pred_text, fill=(255, 255, 255))
                draw.text((10, 30), f"Reward: {reward.item():.3f}", fill=(255, 255, 255))
                frame = np.array(frame_pil)
            
                frames.append(frame)
            
            obs = next_obs
            
        # Save video using imageio
        os.makedirs(f"{args.log_dir}/{run.id}/videos", exist_ok=True)
        video_path = f"{args.log_dir}/{run.id}/videos/run_{eval_ep}.mp4"
        imageio.mimsave(video_path, frames, fps=10)
        print(f"Saved video to {video_path}")
        print("length of trajectory: ", len(trajectory_obs))

        all_trajectories.append((
            trajectory_obs,
            trajectory_actions,
            trajectory_rewards
            ))
    
    return all_trajectories

def build_rl_model(args, agent_config, common_params):
    if args.algorithm == 'PPO':
        common_params['device'] = 'cpu'
        model = PPO(
                **common_params,
                n_steps=128 // args.num_envs,
                batch_size=128,
                clip_range=0.2,
            )
    else:  # TD3
        model = TD3(
                **common_params,
                buffer_size=agent_config['buffer_size'],
                learning_starts=500,
                batch_size=agent_config['batch_size'],
                tau=agent_config['tau'],
                policy_delay=agent_config['policy_delay'],
            )
        
    return model

if __name__ == "__main__":
    # Required Args (target envs & learners)
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_id', default='', help='Select the environment to run')
    parser.add_argument('--robomimic', action='store_true', help='whether to use robomimic')
    parser.add_argument('--learners_path', default='./robomimic/policies', help='path of learning agents')
    parser.add_argument('--max_chkpt', default=64000, type=int, help='decide upto what learner stage you want to give')
    parser.add_argument('--steps', default=40, type=int, help='length of snippets')
    parser.add_argument('--max_steps', default=200_000, type=int, help='length of max snippets (gt_traj_no_steps only)')  # idk
    parser.add_argument('--traj_noise', default=None, type=int, help='number of adjacent swaps (gt_traj_no_steps_noise only)')
    parser.add_argument('--min_length', default=0,type=int, help='minimum length of trajectory generated by each agent')  # 1000 for mujoco
    parser.add_argument('--num_layers', default=2, type=int, help='number layers of the reward network')
    parser.add_argument('--embedding_dims', default=256, type=int, help='embedding dims')
    parser.add_argument('--num_models', default=3, type=int, help='number of models to ensemble')  # 5 for mujoco
    parser.add_argument('--l2_reg', default=0, type=float, help='l2 regularization size')
    parser.add_argument('--noise', default=0, type=float, help='noise level to add on training label')
    parser.add_argument('--D', default=1000, type=int, help='|D| in the preference paper')
    parser.add_argument('--N', default=10, type=int, help='number of trajactory mix (gt_traj_no_steps_n_mix only)')
    parser.add_argument('--log_dir', default='./robomimic/lift/log')
    parser.add_argument('--preference_type', default='gt_traj', help='gt or gt_traj or time or classifier; if gt then preference will be given as a GT reward, otherwise, it is given as a time index or classifier prediction')
    parser.add_argument('--min_margin', default=1, type=int, help='when prefernce type is "time", the minimum margin that we can assure there exist a margin')
    parser.add_argument('--include_action', action='store_true', help='whether to include action for the model or not')
    parser.add_argument('--stochastic', action='store_true', help='whether want to use stochastic agent or not')
    # Args for PPO
    parser.add_argument('--rl_runs', default=1, type=int)
    parser.add_argument('--ppo_log_path', default='ppo2')
    parser.add_argument('--ctrl_coeff', default=0.0, type=float)
    parser.add_argument('--alive_bonus', default=0.0, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--ppo_policy', default="MlpPolicy", type=str)
    parser.add_argument('--batch_size', default=512, type=int)  # 256 for mujoco from paper section 5.1.2
    parser.add_argument('--iter', default=2000, type=int)  # how long to train the reward model - 5000 for mujoco and best performance on robomimic 
    parser.add_argument('--num_envs', default=4, type=int)
    parser.add_argument('--cliprew', default=10.0, type=float)
    parser.add_argument('--num_iterations', default=1, type=int, help='number of training iterations')
    parser.add_argument('--reward_threshold', default=1, type=float, help='reward threshold for adding trajectories to dataset')
    parser.add_argument('--algorithm', default='PPO', choices=['PPO', 'TD3'], 
                   help='RL algorithm to use for training')
    args = parser.parse_args()

    config = {
        "final_agent": {
            "policy_type": "MlpPolicy",
            "total_timesteps": 10_000_000,
            "env_id": args.env_id,
            "device": "cuda" if args.algorithm == 'TD3' else "cpu",
            "learning_rate": 1e-4,
            # TD3-specific parameters (only used when algorithm is TD3)
            "buffer_size": 1000000,
            "batch_size": 512,
            "tau": 0.005,
            "policy_delay": 2,
            "policy_noise": 0.2,  # Standard deviation of Gaussian noise added to actions for exploration
            "noise_clip": 0.5,    # Maximum value of noise to add to actions
            "target_policy_noise": 0.2,  # Noise added to target policy during critic update
            "target_noise_clip": 0.5,    # Maximum noise added to target policy
        }
    }
    run = wandb.init(
        project="trex-training-robomimic",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )
    # Train a Preference Model
    train(args, run)
