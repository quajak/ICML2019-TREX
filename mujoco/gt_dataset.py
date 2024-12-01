import numpy as np
from tqdm import tqdm
from stable_baselines3 import PPO

class GTDataset:
    def __init__(self,env, robomimic=False):
        self.env = env
        self.robomimic = robomimic
        self.unwrapped = env
        while hasattr(self.unwrapped,'env'):
            self.unwrapped = self.unwrapped.env

    def gen_traj(self, agent: PPO, min_length: int):
        max_x_pos = -99999

        if self.robomimic:
            first_obs = self.env.reset()
        else:
            first_obs, _ = self.env.reset()
        obs, actions, rewards = [first_obs], [], []
        max_length = 1000 if not self.robomimic else 200
        while len(obs) < max_length:  # we collect 1000 steps of trajectory for each agent
            if self.robomimic:
                action = agent(obs[-1])
            else:
                action, _ = agent.predict(obs[-1], None, None)
            if self.robomimic:
                ob, reward, done, info = self.env.step(action)
            else:
                ob, reward, done, _, info = self.env.step(action)
            if len(action.shape) == 0:
                action = action[None]
            if self.robomimic:
                max_x_pos = max(max_x_pos, reward)
            else:   
                if info['x_position'] > max_x_pos:
                    max_x_pos = info['x_position']

            if self.robomimic:
                done |= reward == 1

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

        obs = [np.concatenate([ob[k] for k in ['object', 'robot0_eef_pos', 'robot0_gripper_qpos', 'robot0_eef_quat']]) for ob in obs]
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
