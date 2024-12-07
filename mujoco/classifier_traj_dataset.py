import torch
import numpy as np
from gt_traj_dataset import GTTrajLevelDataset
from classifier import Classifier

class ClassifierTrajDataset(GTTrajLevelDataset):
    def __init__(self, env, log_dir: str, classifier: Classifier, robomimic: bool = False, val: bool = False):
        super().__init__(env, log_dir, robomimic, val)
        self.classifier = classifier
        self.steps = self.classifier.num_timesteps
        self.traj_scores = None

    def prebuilt(self, agents, min_length, run):
        super().prebuilt(agents, min_length, run)
        # Pre-calculate classifier scores for all possible segments in each trajectory
        self.traj_scores = []
        
        for traj in self.trajs:
            obs = traj[1]  # Get observations from trajectory
            scores = []
            
            # Calculate scores for all segments at once
            segments = []
            actions = traj[2].reshape(-1, self.classifier.num_actions)
            episode_length = min(len(obs), len(actions))
            for i in range(episode_length - self.steps + 1):
                if self.classifier.include_action:
                    segments.append(np.concatenate([obs[i:i+self.steps], actions[i+self.steps - 1][None, :]], axis=1))
                else:
                    segments.append(obs[i:i+self.steps])
            segments = torch.FloatTensor(np.array(segments)).to(self.classifier.device)
            
            with torch.no_grad():
                scores_batch = self.classifier(segments)
                scores_batch = torch.softmax(scores_batch, dim=1)[:, 1]
                scores.extend(scores_batch.cpu().tolist())
                
            self.traj_scores.append(scores)

    def sample(self, num_samples, include_action=False):
        D = []
        for _ in range(num_samples):
            x_idx, y_idx = np.random.choice(len(self.trajs), 2, replace=False)

            x_traj = self.trajs[x_idx]
            y_traj = self.trajs[y_idx]

            x_ptr = np.random.randint(len(x_traj[1])-self.steps)
            y_ptr = np.random.randint(len(y_traj[1])-self.steps)

            x_segment = x_traj[1][x_ptr:x_ptr+self.steps]
            y_segment = y_traj[1][y_ptr:y_ptr+self.steps]

            # Get pre-calculated classifier scores
            x_score = self.traj_scores[x_idx][x_ptr]
            y_score = self.traj_scores[y_idx][y_ptr]

            if include_action:
                D.append((
                    np.concatenate((x_segment, x_traj[2].reshape(-1, self.classifier.num_actions)[x_ptr:x_ptr+self.steps]), axis=1),
                    np.concatenate((y_segment, y_traj[2].reshape(-1, self.classifier.num_actions)[y_ptr:y_ptr+self.steps]), axis=1),
                    0 if x_score > y_score else 1
                ))
            else:
                D.append((
                    x_segment,
                    y_segment, 
                    0 if x_score > y_score else 1
                ))

        return D