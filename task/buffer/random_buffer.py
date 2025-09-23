import collections
import random
import torch
import numpy as np


class RandomBuffer:
    def __init__(self, buffer_size):
        self.buffer = collections.deque(maxlen=buffer_size)

    def __len__(self):
        return len(self.buffer)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        keys = batch[0].keys()
        collated_batch = {key: [d[key] for d in batch] for key in keys}

        return {
            'obs': torch.from_numpy(np.vstack(collated_batch['obs'])).float(),
            'state': torch.from_numpy(np.vstack(collated_batch['state'])).float(),
            'actions': torch.from_numpy(np.vstack(collated_batch['actions'])).float(),
            'rewards': torch.from_numpy(np.vstack(collated_batch['rewards'])).float(),
            'next_obs': torch.from_numpy(np.vstack(collated_batch['next_obs'])).float(),
            'next_state': torch.from_numpy(np.vstack(collated_batch['next_state'])).float(),
            'terminated': torch.from_numpy(np.vstack(collated_batch['terminated'])).bool(),
            'truncated': torch.from_numpy(np.vstack(collated_batch['truncated'])).bool(),
        }
