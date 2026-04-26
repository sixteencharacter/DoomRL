import collections
import random
import math
from typing import List, Optional
import torch
from datamodel import Transition, SampledBatch, training_info
from config import ALPHA, BETA_START, BETA_END, PER_EPSILON, MAX_STEPS


class _BaseReplayMemory:
    """Concrete base — owns deque storage. Subclasses override sample()."""

    def __init__(self, capacity, device):
        self.memory = collections.deque(maxlen=capacity)
        self.device = device
        self.capacity = capacity

    def push(self, *args):
        self.memory.append(Transition(*args))

    def __len__(self):
        return len(self.memory)

    def sample(self, batch_size, preprocessor) -> SampledBatch:
        raise NotImplementedError

    def update_priorities(self, indices, td_errors):
        # default no-op; PER overrides
        pass


class UniformReplayMemory(_BaseReplayMemory):
    """Random uniform sampling — preserves original ReplayMemory behavior."""

    def sample(self, batch_size, preprocessor=None) -> SampledBatch:
        samples: List[Transition] = random.sample(self.memory, batch_size)

        if preprocessor is not None:
            state_processed = preprocessor([s.state for s in samples], device=self.device)
            next_state_processed = preprocessor([s.next_state for s in samples], device=self.device)
            actions = [s.action for s in samples]
            rewards = [s.reward for s in samples]
            transitions = [
                Transition(s, a, n, r)
                for s, a, n, r in zip(state_processed, actions, next_state_processed, rewards)
            ]
        else:
            transitions = samples

        weights = torch.ones(batch_size, device=self.device)
        return SampledBatch(transitions=transitions, indices=None, weights=weights)


class SumTree:
    """Array-backed sum-tree for proportional PER.

    Capacity N => 2N-1 nodes. Leaves [N-1, 2N-1) hold priorities. Internal nodes
    hold partial sums. Sampling and updates are O(log N).
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = [0.0] * (2 * capacity - 1)
        self.size = 0  # number of valid leaves populated

    @property
    def total(self) -> float:
        return self.tree[0]

    def update(self, data_idx: int, priority: float) -> None:
        tree_idx = data_idx + self.capacity - 1
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        # propagate delta up to root
        parent = (tree_idx - 1) // 2
        while parent >= 0:
            self.tree[parent] += delta
            if parent == 0:
                break
            parent = (parent - 1) // 2

    def add(self, data_idx: int, priority: float) -> None:
        self.update(data_idx, priority)
        if self.size < self.capacity:
            self.size += 1

    def get(self, s: float) -> int:
        """Find leaf data_idx whose cumulative priority covers s."""
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            right = left + 1
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        return idx - (self.capacity - 1)

    def leaves(self):
        """Iterate priorities of valid leaves (for stats)."""
        return self.tree[self.capacity - 1: self.capacity - 1 + self.size]


class PrioritizedReplayMemory(_BaseReplayMemory):
    """Proportional PER (Schaul et al. 2016) backed by a sum-tree."""

    def __init__(self, capacity, device):
        super().__init__(capacity, device)
        # ring buffer of transitions (indexed 0..capacity-1)
        self.buffer: List[Optional[Transition]] = [None] * capacity
        self.write_idx = 0
        self.entries = 0
        self.tree = SumTree(capacity)
        self.max_priority = 1.0

    def push(self, *args):
        transition = Transition(*args)
        self.buffer[self.write_idx] = transition
        # new transitions assigned current max_priority to ensure they're sampled
        priority = self.max_priority ** ALPHA
        self.tree.add(self.write_idx, priority)
        self.write_idx = (self.write_idx + 1) % self.capacity
        if self.entries < self.capacity:
            self.entries += 1

    def __len__(self):
        return self.entries

    def _beta(self) -> float:
        # linear schedule based on training_info.learning_step / MAX_STEPS
        if MAX_STEPS is None:
            return BETA_END
        progress = min(1.0, training_info.learning_step / MAX_STEPS)
        return BETA_START + (BETA_END - BETA_START) * progress

    def sample(self, batch_size, preprocessor=None) -> SampledBatch:
        assert self.entries >= batch_size, \
            f"PER sample requested {batch_size} but only {self.entries} entries"

        total = self.tree.total
        segment = total / batch_size
        indices_list = []
        priorities = []
        samples: List[Transition] = []

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            s = random.uniform(lo, hi)
            data_idx = self.tree.get(s)
            tree_priority = self.tree.tree[data_idx + self.capacity - 1]
            indices_list.append(data_idx)
            priorities.append(tree_priority)
            samples.append(self.buffer[data_idx])

        # importance-sampling weights
        beta = self._beta()
        priorities_t = torch.tensor(priorities, dtype=torch.float32, device=self.device)
        probs = priorities_t / max(total, 1e-12)
        weights = (self.entries * probs).clamp(min=1e-12).pow(-beta)
        weights = weights / weights.max()  # normalize for stability

        indices = torch.tensor(indices_list, dtype=torch.long, device=self.device)

        if preprocessor is not None:
            state_processed = preprocessor([s.state for s in samples], device=self.device)
            next_state_processed = preprocessor([s.next_state for s in samples], device=self.device)
            actions = [s.action for s in samples]
            rewards = [s.reward for s in samples]
            transitions = [
                Transition(s, a, n, r)
                for s, a, n, r in zip(state_processed, actions, next_state_processed, rewards)
            ]
        else:
            transitions = samples

        return SampledBatch(transitions=transitions, indices=indices, weights=weights)

    def update_priorities(self, indices, td_errors):
        if indices is None:
            return
        idx_list = indices.detach().cpu().tolist()
        err_list = td_errors.detach().abs().cpu().tolist()
        for i, e in zip(idx_list, err_list):
            priority = (float(e) + PER_EPSILON) ** ALPHA
            self.tree.update(i, priority)
            if (float(e) + PER_EPSILON) > self.max_priority:
                self.max_priority = float(e) + PER_EPSILON

    # observability helpers for wandb
    def priority_mean(self) -> float:
        leaves = self.tree.leaves()
        return sum(leaves) / len(leaves) if leaves else 0.0

    def priority_max(self) -> float:
        leaves = self.tree.leaves()
        return max(leaves) if leaves else 0.0

    def priority_std(self) -> float:
        leaves = self.tree.leaves()
        if not leaves:
            return 0.0
        m = sum(leaves) / len(leaves)
        var = sum((p - m) ** 2 for p in leaves) / len(leaves)
        return math.sqrt(var)


def create_replay_memory(method: str, capacity: int, device) -> _BaseReplayMemory:
    if method == "Uniform":
        return UniformReplayMemory(capacity, device)
    if method == "PER":
        return PrioritizedReplayMemory(capacity, device)
    raise ValueError(f"Unsupported SAMPLING_METHOD: {method}")


# backwards-compat alias so any import of `ReplayMemory` still works
ReplayMemory = UniformReplayMemory
