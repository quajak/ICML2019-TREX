import os
import numpy as np
from tqdm import tqdm
from stable_baselines3 import PPO
from gt_dataset import GTDataset
import wandb
import pickle

class GTTrajLevelDataset(GTDataset):
    def __init__(self,env, log_dir: str, val: bool = False):
        super().__init__(env)
        self.log_dir = log_dir
        self.val = val

    def prebuilt(self, agents: list[PPO], min_length: int, run):
        assert len(agents)>0, 'no agent is given'
        save_file = self.log_dir + f'/trajs{"_val" if self.val else ""}.npy'
        if os.path.exists(save_file):
            with open(save_file, 'rb') as f:
                self.trajs = pickle.load(f)
            self.sort_trajs()
            return

        tqdm.write("Generating Trajectories")
        trajs = []
        max_max_x = -99999
        for agent_idx,agent in enumerate(tqdm(agents)):
            (obs,actions,rewards), max_x = self.gen_traj(agent,min_length)
            trajs.append((agent_idx,obs,actions,rewards, max_x))
            run.log({"dataset/agent_reward": np.sum(rewards), "dataset/agent_max_x": max_x})
            if max_x > max_max_x:
                max_max_x = max_x

        run.log({"dataset/max_max_x": max_max_x})
        print('best max_x:', max_max_x)
        self.trajs = trajs
        self.sort_trajs()
        
        # save using pickle 
        with open(save_file, 'wb') as f:
            pickle.dump(self.trajs, f)
        
    def sort_trajs(self):
        _idxes = np.argsort([np.sum(rewards) for _,_,_,rewards,_ in self.trajs]) # rank 0 is the most bad demo.
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
                          0 if self.trajs_rank[x_idx] > self.trajs_rank[y_idx] else 1)
                        )
            else:
                D.append((x_traj[1][x_ptr:x_ptr+steps],
                          y_traj[1][y_ptr:y_ptr+steps],
                          0 if self.trajs_rank[x_idx] > self.trajs_rank[y_idx] else 1)
                        )

            GT_preference.append(0 if np.sum(x_traj[3][x_ptr:x_ptr+steps]) > np.sum(y_traj[3][y_ptr:y_ptr+steps]) else 1)

        # print('------------------')
        # _,_,preference = zip(*D)
        # preference = np.array(preference).astype(bool)
        # GT_preference = np.array(GT_preference).astype(bool)
        # print('Quality of time-indexed preference (0-1):', np.count_nonzero(preference == GT_preference) / len(preference))
        # print('------------------')

        return D
