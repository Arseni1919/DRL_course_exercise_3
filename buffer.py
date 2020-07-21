import random
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'not_done'))


class ReplayBuffer(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        # TODO
        self.memory.append(Transition(*args))
        if len(self) > self.capacity:
            self.memory.pop(0)
        # raise NotImplementedError

    def sample(self, batch_size):
        # TODO
        return random.sample(self.memory, batch_size)
        # raise NotImplementedError

    def __len__(self):
        return len(self.memory)
