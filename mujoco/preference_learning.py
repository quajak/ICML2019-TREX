import argparse
from pathlib import Path
from random import randint, random
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
import time
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import EvalCallback
from PIL import Image, ImageDraw
import imageio
import cv2
import os
import robomimic.utils.file_utils as FileUtils

from model import Model
from custom_reward_wrapper import TorchPreferenceRewardNormalized
from gt_traj_dataset import GTTrajLevelDataset
from classifier import Classifier
from classifier_traj_dataset import ClassifierTrajDataset

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
            agent = PPO.load(path, device='cpu')
            setattr(agent, 'model_path', str(path))
        train_agents.append(agent)

    if args.robomimic:
        train_agents = train_agents * 300
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
    elif args.preference_type == 'classifier':
        print("Initializing classifier-based training...")
        
        # First create and train the classifier on all data
        base_dataset = GTTrajLevelDataset(env=env, log_dir=args.log_dir, robomimic=args.robomimic)
        base_dataset.prebuilt(train_agents, args.min_length, run)
        base_val_dataset = GTTrajLevelDataset(env, args.log_dir, val=True, robomimic=args.robomimic)
        base_val_dataset.prebuilt(train_agents, args.min_length, run)
        
        print("Training classifier...")
        classifier = Classifier(robomimic=args.robomimic, device=device)
        classifier.train_classifier(base_dataset, base_val_dataset)
        
        # Save the trained classifier
        torch.save(classifier.state_dict(), f"{args.log_dir}/classifier.pt")
        
        # Now create datasets using the trained classifier
        print("Building classifier-based training dataset...")
        dataset = ClassifierTrajDataset(env, args.log_dir, classifier)
        start_time = time.time()
        dataset.prebuilt(train_agents, args.min_length, run)
        print(f"Training dataset built in {time.time() - start_time:.2f} seconds")

        print("Building classifier-based validation dataset...")
        val_dataset = ClassifierTrajDataset(env, args.log_dir, classifier, val=True)
        start_time = time.time()
        val_dataset.prebuilt(train_agents, args.min_length, run)
        print(f"Validation dataset built in {time.time() - start_time:.2f} seconds")
    else:
        raise ValueError('Invalid preference type')

    # Initialize models
    print("Initializing reward models...")
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
        models.append(model)

    # Train each model
    for i, model in enumerate(models):
        print(f"Training reward model {i+1}/{args.num_models}")
        
        model.train_model(
            dataset,
            val_dataset,
            batch_size=args.batch_size,
            num_epochs=args.iter,
            l2_reg=args.l2_reg,
            noise_level=args.noise,
            debug=True,
            run=run
        )

        # Save model
        torch.save(model.state_dict(), f"{args.log_dir}/model_{i}.pt")
        # Save model configuration separately for loading
        model_config = {
            'include_action': args.include_action,
            'ob_dim': env.observation_space.shape[0],
            'ac_dim': env.action_space.shape[0],
            'num_layers': args.num_layers,
            'embedding_dims': args.embedding_dims
        }
        torch.save(model_config, f"{args.log_dir}/model_{i}_config.pt")

        
    # Visualize the classifier and learned reward models
    print("Visualizing classifier and reward models...")
    
    # Create environment for visualization
    if args.robomimic:
        vis_env, _ = FileUtils.env_from_checkpoint(
            ckpt_dict=ckpt_dict, 
            render=True,
            render_offscreen=True,
            verbose=False
        )
    else:
        vis_env = gym.make(config['final_agent']['env_id'], render_mode='rgb_array')
    
    # Wrap environment to get reward predictions
    vis_env = TorchPreferenceRewardNormalized(
        vis_env,
        args.num_models,
        args.log_dir,
        args.include_action,
        args.num_layers,
        args.embedding_dims,
        robomimic=args.robomimic,
        convert_obs=False
    )

    # Do a rollout and collect frames
    for i in range(5):
        frames = []
        for _ in range(randint(1, 5)):
            obs = vis_env.reset()
        last_obs = np.concatenate([obs[k] for k in ['object', 'robot0_eef_pos', 'robot0_gripper_qpos', 'robot0_eef_quat']])
        done = False
        
        while not done:
            action = train_agents[0](obs)
            obs, reward, done, terminated, info = vis_env.step(action)
            frame = vis_env.render(mode="rgb_array")
            img = Image.fromarray(frame)
            draw = ImageDraw.Draw(img)
            y = 10
            if args.preference_type == 'classifier':
                if args.robomimic:
                    obs_pt = np.concatenate([obs[k] for k in ['object', 'robot0_eef_pos', 'robot0_gripper_qpos', 'robot0_eef_quat']])
                else:
                    obs_pt = obs
                classifier_pred = classifier.forward(torch.from_numpy(np.concatenate([obs_pt])[None, :]).cuda().float())
                classifier_pred = torch.softmax(classifier_pred, dim=1)[:, 1]
                draw.text((10, y), f"Classifier: {classifier_pred.item():.3f}", fill=(255,255,255))
            y += 20
            draw.text((10, y), f"Reward: {reward.item():.3f}", fill=(255,255,255))
            y += 20

            last_obs = obs_pt
            frames.append(np.array(img))
    
        # Save video
        imageio.mimsave(f"{args.log_dir}/rollout_vis_{i}.mp4", frames, fps=30)
        print(f"Saved visualization video to {args.log_dir}/rollout_vis_{i}.mp4")
        
    vis_env.close()

    print("Training PPO agent with learned reward models...")
    # Train an agent
    group = str(time.time())

    wandb_callback = WandbCallback(
            model_save_path=f"models/{run.id}",
            verbose=2,
        )

    agent_config = config['final_agent']
    for i in range(args.rl_runs):
        def make_indiv_env():
            if args.robomimic:
                # Load robomimic env using same structure as earlier
                env, _ = FileUtils.env_from_checkpoint(
                    ckpt_dict=ckpt_dict,
                    render=False,
                    render_offscreen=False,
                    verbose=False,
                )
            else:
                # Regular gym/mujoco env
                env = gym.make(agent_config["env_id"])
            
            env = TorchPreferenceRewardNormalized(env, args.num_models, args.log_dir, args.include_action, args.num_layers, args.embedding_dims, robomimic=args.robomimic, cliprew=args.cliprew)
            return env

        env = DummyVecEnv([make_indiv_env for _ in range(args.num_envs)], args.robomimic)
        model = PPO(
            agent_config['policy_type'], 
            env, 
            learning_rate=3e-4,
            n_steps=128,  # this was 128//4 for mujoco
            batch_size=128,  # added for robomimic
            clip_range=0.2,
            device="cpu", 
            tensorboard_log=f"runs/{run.id}",
            verbose=0
        )

        model.learn(
            total_timesteps=agent_config["total_timesteps"],
            callback=[
                wandb_callback,
                # Add evaluation callback to monitor performance during training
                EvalCallback(
                    eval_env=make_indiv_env(),
                    eval_freq=5000,  # Evaluate every 5000 steps
                    n_eval_episodes=10,
                    best_model_save_path=f"{args.log_dir}/{run.id}/best_model",
                    log_path=f"{args.log_dir}/{run.id}",
                    deterministic=True
                )
            ],
        )
        env.close()

        # Evaluate the trained model
        if args.robomimic:
            eval_env, _ = FileUtils.env_from_checkpoint(
                ckpt_dict=ckpt_dict,
                render=True,
                render_offscreen=True,
                verbose=False,
            )

            eval_env = TorchPreferenceRewardNormalized(eval_env, args.num_models, args.log_dir, args.include_action, args.num_layers, args.embedding_dims, robomimic=args.robomimic, cliprew=args.cliprew)
        else:
            eval_env = gym.make(agent_config["env_id"])

        max_xs = []
        for k in range(30):
            print(f'Evaluation Run {k}')
            if args.robomimic:
                for i in range(randint(2, 5)):
                    obs = eval_env.reset()  # to ensure random start seed
                    # obs = np.concatenate([obs[k] for k in ['object', 'robot0_eef_pos', 'robot0_gripper_qpos', 'robot0_eef_quat']])
            else:
                obs, _ = eval_env.reset()
            total_reward = 0
            num_steps = 0
            done = False
            terminated = False
            max_x_pos = -np.inf
            last_obs = obs
            frames = []
            
            while not done and not terminated and num_steps < 500:  # 1000 for mujoco
                action, _ = model.predict(obs)
                next_obs, reward, done, terminated, info = eval_env.step(action)
                
                # Get classifier prediction if available
                pred_text = ""
                if args.preference_type == 'classifier':
                    with torch.no_grad():
                        obs_tensor = torch.FloatTensor(np.concatenate([next_obs])).unsqueeze(0).to(classifier.device)
                        pred = classifier(obs_tensor)
                        pred = torch.softmax(pred, dim=1)[0,1].item()
                        pred_text = f"Classifier Score: {pred:.3f}"
                
                last_obs = next_obs
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
                total_reward += reward if not isinstance(reward, np.ndarray) else reward.item()
                num_steps += 1
                if not args.robomimic:  # Only track x_position for mujoco envs
                    max_x_pos = max(max_x_pos, info["x_position"])
            
            # Save video using imageio
            os.makedirs(f"{args.log_dir}/{run.id}/videos", exist_ok=True)
            video_path = f"{args.log_dir}/{run.id}/videos/run_{k}.mp4"
            imageio.mimsave(video_path, frames, fps=10)
            print(f"Saved video to {video_path}")

            run.log({
                "eval/total_reward": total_reward,
                "eval/total_steps": num_steps,
                "eval/max_x_pos": max_x_pos if not args.robomimic else 0,
                "eval/run": k
            })
            max_xs.append(max_x_pos if not args.robomimic else 0)
            print(f'Run {k} - Total Reward: {total_reward}, Total Steps: {num_steps}, Max X Pos: {max_x_pos if not args.robomimic else "N/A"}')
        
        if not args.robomimic:
            print(f'Average Max X Pos: {np.mean(max_xs)} STD Max X Pos: {np.std(max_xs)}')
        eval_env.close()

        run.finish()

if __name__ == "__main__":
    # Required Args (target envs & learners)
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_id', default='', help='Select the environment to run')
    parser.add_argument('--robomimic', action='store_true', help='whether to use robomimic')
    parser.add_argument('--learners_path', default='./robomimic/policies', help='path of learning agents')
    parser.add_argument('--max_chkpt', default=64000, type=int, help='decide upto what learner stage you want to give')
    parser.add_argument('--steps', default=40, type=int, help='length of snippets')
    parser.add_argument('--max_steps', default=100_000, type=int, help='length of max snippets (gt_traj_no_steps only)')  # idk
    parser.add_argument('--traj_noise', default=None, type=int, help='number of adjacent swaps (gt_traj_no_steps_noise only)')
    parser.add_argument('--min_length', default=40,type=int, help='minimum length of trajectory generated by each agent')  # 1000 for mujoco
    parser.add_argument('--num_layers', default=2, type=int, help='number layers of the reward network')
    parser.add_argument('--embedding_dims', default=256, type=int, help='embedding dims')
    parser.add_argument('--num_models', default=5, type=int, help='number of models to ensemble')  # 5 for mujoco
    parser.add_argument('--l2_reg', default=0, type=float, help='l2 regularization size')
    parser.add_argument('--noise', default=0, type=float, help='noise level to add on training label')
    parser.add_argument('--D', default=1000, type=int, help='|D| in the preference paper')
    parser.add_argument('--N', default=10, type=int, help='number of trajactory mix (gt_traj_no_steps_n_mix only)')
    parser.add_argument('--log_dir', default='./robomimic/log')
    parser.add_argument('--preference_type', default='gt_traj', help='gt or gt_traj or time or classifier; if gt then preference will be given as a GT reward, otherwise, it is given as a time index or classifier prediction')
    parser.add_argument('--min_margin', default=1, type=int, help='when prefernce type is "time", the minimum margin that we can assure there exist a margin')
    parser.add_argument('--include_action', action='store_true', help='whether to include action for the model or not')
    parser.add_argument('--stochastic', action='store_true', help='whether want to use stochastic agent or not')
    # Args for PPO
    parser.add_argument('--rl_runs', default=1, type=int)
    parser.add_argument('--ppo_log_path', default='ppo2')
    parser.add_argument('--custom_reward', default="preference_normalized", help='preference or preference_normalized')
    parser.add_argument('--ctrl_coeff', default=0.0, type=float)
    parser.add_argument('--alive_bonus', default=0.0, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--ppo_policy', default="MlpPolicy", type=str)
    parser.add_argument('--batch_size', default=256, type=int)  # from paper section 5.1.2
    parser.add_argument('--iter', default=1000, type=int)  # how long to train the reward model - 5000 for mujoco
    parser.add_argument('--num_envs', default=4, type=int)
    parser.add_argument('--cliprew', default=10.0, type=float)
    args = parser.parse_args()

    config = {
        "final_agent": {
            "policy_type": "MlpPolicy",
            "total_timesteps": 100_000,
            "env_id": args.env_id,
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
