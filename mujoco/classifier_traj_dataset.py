import torch
import numpy as np
from gt_traj_dataset import GTTrajLevelDataset
from classifier import Classifier

class ClassifierTrajDataset(GTTrajLevelDataset):
    def __init__(self, base_dataset: GTTrajLevelDataset, classifier: Classifier):
        """
        Initialize ClassifierTrajDataset from an existing GTTrajLevelDataset
        
        Args:
            base_dataset: Existing GTTrajLevelDataset to build from
            classifier: Trained classifier to use for scoring trajectories
        """
        # Initialize with same parameters as base dataset
        super().__init__(
            base_dataset.env,
            base_dataset.log_dir,
            base_dataset.robomimic,
            base_dataset.val
        )
        
        # Copy existing trajectories from base dataset
        self.trajs = base_dataset.trajs.copy()
        
        self.classifier = classifier
        self.steps = self.classifier.num_timesteps
        self.traj_scores = None
        
        # Calculate scores for existing trajectories
        self._calculate_traj_scores()

    def _calculate_traj_scores(self):
        """Pre-calculate classifier scores for all possible segments in each trajectory"""
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
                    segment = np.concatenate([
                        obs[i:i+self.steps],
                        actions[i:i+self.steps]
                    ], axis=1)
                else:
                    segment = obs[i:i+self.steps]
                segments.append(segment)
                
            if segments:  # Check if we have any valid segments
                segments = torch.FloatTensor(np.array(segments)).to(self.classifier.device)
                
                with torch.no_grad():
                    scores_batch = self.classifier(segments)
                    scores_batch = torch.softmax(scores_batch, dim=1)[:, 1]
                    scores.extend(scores_batch.cpu().tolist())
            
            self.traj_scores.append(scores)

    def add_trajectory(self, obs, actions, rewards, max_x=0):
        """Override add_trajectory to update classifier scores when adding new trajectory"""
        super().add_trajectory(obs, actions, rewards, max_x)
        
        # Calculate scores for the newly added trajectory
        new_scores = []
        new_obs = np.stack(obs, axis=0)
        new_actions = np.array(actions).reshape(-1, self.classifier.num_actions)
        episode_length = min(len(new_obs), len(new_actions))
        
        segments = []
        for i in range(episode_length - self.steps + 1):
            if self.classifier.include_action:
                segment = np.concatenate([
                    new_obs[i:i+self.steps],
                    new_actions[i:i+self.steps]
                ], axis=1)
            else:
                segment = new_obs[i:i+self.steps]
            segments.append(segment)
            
        if segments:
            segments = torch.FloatTensor(np.array(segments)).to(self.classifier.device)
            with torch.no_grad():
                scores_batch = self.classifier(segments)
                scores_batch = torch.softmax(scores_batch, dim=1)[:, 1]
                new_scores.extend(scores_batch.cpu().tolist())
        
        self.traj_scores.append(new_scores)

    def sample(self, num_samples, include_action=False):
        """Sample trajectory pairs and use classifier scores to determine preferences"""
        D = []
        for _ in range(num_samples):
            x_idx, y_idx = np.random.choice(len(self.trajs), 2, replace=False)

            x_traj = self.trajs[x_idx]
            y_traj = self.trajs[y_idx]

            # Make sure we don't exceed the valid range for segments
            x_max = len(x_traj[1]) - self.steps
            y_max = len(y_traj[1]) - self.steps
            
            if x_max <= 0 or y_max <= 0:
                continue  # Skip if trajectory is too short
                
            x_ptr = np.random.randint(x_max)
            y_ptr = np.random.randint(y_max)

            x_segment = x_traj[1][x_ptr:x_ptr+self.steps]
            y_segment = y_traj[1][y_ptr:y_ptr+self.steps]

            # Get pre-calculated classifier scores
            x_score = self.traj_scores[x_idx][x_ptr]
            y_score = self.traj_scores[y_idx][y_ptr]

            if include_action:
                x_actions = x_traj[2].reshape(-1, self.classifier.num_actions)[x_ptr:x_ptr+self.steps]
                y_actions = y_traj[2].reshape(-1, self.classifier.num_actions)[y_ptr:y_ptr+self.steps]
                
                D.append((
                    np.concatenate((x_segment, x_actions), axis=1),
                    np.concatenate((y_segment, y_actions), axis=1),
                    0 if x_score > y_score else 1
                ))
            else:
                D.append((
                    x_segment,
                    y_segment, 
                    0 if x_score > y_score else 1
                ))

        return D