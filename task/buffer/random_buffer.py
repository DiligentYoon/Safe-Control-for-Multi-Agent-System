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

        final_batch = {}
        for key, value_list in collated_batch.items():
            if key in ['obs', 'state', 'next_obs', 'next_state']:
                final_batch[key] = {
                        k: torch.from_numpy(np.vstack([d[k] for d in value_list])).float()
                        for k in value_list[0].keys()}
            elif key == "info":
                final_batch[key] = {
                    k: torch.from_numpy(np.vstack([d[k] for d in value_list])).float()
                    for k in value_list[0].keys()}
            else:
                tensor = torch.from_numpy(np.vstack(value_list))
                if key in ['terminated', 'truncated']:
                    final_batch[key] = tensor.bool()
                else:
                    final_batch[key] = tensor.float()

        return final_batch
