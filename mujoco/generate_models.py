import gymnasium as gym
import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 64000,
    "env_id": "HalfCheetah-v4",
}

run = wandb.init(
    project="trex-data-collection",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    # monitor_gym=True,  # auto-upload the videos of agents playing the game
    # save_code=True,  # optional
)

class ExtraLogWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.best_x = -float("inf")
        
    def step(self, action):
        obs, reward, done, term, info = self.env.step(action)
        self.best_x = max(self.best_x, info["x_position"])
        return obs, reward, done, term, info

    def reset(self, **kwargs):
        run.log({"best_x": self.best_x}, commit=False)
        self.best_x = -float("inf")
        return self.env.reset(**kwargs)

env = gym.make(config["env_id"])
env = ExtraLogWrapper(env)

class SaveCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(SaveCallback, self).__init__(verbose)
        self.counter = 0
    
    def _on_step(self):
        if self.counter % (5 * 128) == 0:
            self.model.save(f"trex_models/{config['env_id']}/{self.counter}")
        self.counter += 1
        return True
    
    def init_callback(self, model):
        self.model = model
    

model = PPO(config['policy_type'], env, n_steps=128, device="cpu", tensorboard_log=f"runs/{run.id}")
wandb_callback = WandbCallback(
        model_save_path=f"models/{run.id}",
        verbose=2,
    )

model.learn(
    total_timesteps=config["total_timesteps"],
    callback=[wandb_callback, SaveCallback()],
)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

env.close()
run.finish()