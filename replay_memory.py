import collections
import logging
import random
import math
from typing import Dict, List, Optional
import torch
from datamodel import Transition, SampledBatch, WindowSampledBatch, training_info
from config import (
    ALPHA, BETA_START, BETA_END, PER_EPSILON, MAX_STEPS,
    STARFORMER_K, STARFORMER_RTG_TARGET, MIN_VALID_ANCHORS,
)

logger = logging.getLogger(__name__)


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


class WindowPrioritizedReplayMemory(PrioritizedReplayMemory):
    """Per-transition ring + SumTree where each leaf is interpreted as the *anchor*
    of a K-length window ``[anchor-K+1 ... anchor]``.

    Anchors whose window crosses an episode boundary, references a yet-unwritten slot,
    or wraps onto stale data carry priority 0 — invisible to proportional sampling.
    """

    def __init__(self, capacity, device, K: int = STARFORMER_K):
        super().__init__(capacity, device)
        self.K = K
        # validity flag per slot, 1 byte each
        self._valid: List[bool] = [False] * capacity
        self.valid_anchor_count = 0
        # episode index map for backfill
        self._episode_indices: Dict[int, List[int]] = {}
        # observability counters
        self.rtg_backfill_overwrites = 0
        self.rtg_none_ratio_last = 0.0

    # ---- helpers ----
    def _is_valid_anchor(self, i: int) -> bool:
        """Anchor i is valid iff buffer[i-K+1 ... i] all populated and same episode_id,
        and the underlying ring slots have not been overwritten by a fresher write that
        would create an episode-id discontinuity (handled implicitly by the same-episode_id check)."""
        if self.entries < self.K:
            return False
        anchor = self.buffer[i]
        if anchor is None:
            return False
        anchor_ep = anchor.episode_id
        for k in range(1, self.K):
            j = (i - k) % self.capacity
            t = self.buffer[j]
            if t is None or t.episode_id != anchor_ep:
                return False
        return True

    def _set_valid(self, i: int, valid: bool, default_priority: Optional[float] = None) -> None:
        was = self._valid[i]
        if was == valid:
            return
        self._valid[i] = valid
        if valid:
            self.valid_anchor_count += 1
            prio = (self.max_priority ** ALPHA) if default_priority is None else default_priority
            self.tree.update(i, prio)
        else:
            self.valid_anchor_count -= 1
            self.tree.update(i, 0.0)

    def push(self, *args):
        # Construct mutable transition; required args = (state, action, next_state, reward)
        # optional args = (rtg, episode_id)
        transition = Transition(*args)
        write_i = self.write_idx

        # Track episode-index mapping
        ep_id = transition.episode_id
        # If we're about to overwrite a slot, evict its index from any episode list it belongs to.
        existing = self.buffer[write_i]
        if existing is not None:
            old_ep = existing.episode_id
            lst = self._episode_indices.get(old_ep)
            if lst:
                try:
                    lst.remove(write_i)
                    if not lst:
                        del self._episode_indices[old_ep]
                except ValueError:
                    pass

        self.buffer[write_i] = transition
        # Ring slot is brand new: clear any prior validity, set leaf priority 0 until validated
        if self._valid[write_i]:
            self.valid_anchor_count -= 1
            self._valid[write_i] = False
        self.tree.update(write_i, 0.0)
        if self.entries < self.capacity:
            # mirror parent SumTree.add bookkeeping (size grows with entries)
            self.tree.size = min(self.tree.size + 1, self.capacity)
            self.entries += 1

        # Append to episode list
        self._episode_indices.setdefault(ep_id, []).append(write_i)

        # Recompute validity for the K leaves [write_i, write_i+1, ..., write_i+K-1]
        # because adding a new transition can newly satisfy K-1 predecessors of *future*
        # anchors. We additionally re-check write_i itself.
        for offset in range(self.K):
            j = (write_i + offset) % self.capacity
            if self.buffer[j] is None:
                # cannot be a valid anchor yet
                self._set_valid(j, False)
                continue
            ok = self._is_valid_anchor(j)
            self._set_valid(j, ok)

        self.write_idx = (self.write_idx + 1) % self.capacity

    def backfill_rtg(self, episode_id: int, rtgs: List[float]) -> None:
        """Write rtg values into buffer slots that still belong to ``episode_id``.

        Slots that have been overwritten by later episodes are skipped silently and
        counted in ``rtg_backfill_overwrites``.
        """
        idx_list = self._episode_indices.get(episode_id, [])
        # idx_list is in push-order. Apply rtgs[k] to the k-th slot in chronological order.
        for k, idx in enumerate(idx_list):
            t = self.buffer[idx]
            if t is None or t.episode_id != episode_id:
                self.rtg_backfill_overwrites += 1
                continue
            if k >= len(rtgs):
                self.rtg_backfill_overwrites += 1
                continue
            t.rtg = float(rtgs[k])
        # episode complete; drop the bookkeeping
        if episode_id in self._episode_indices:
            del self._episode_indices[episode_id]

    def update_priorities(self, indices, td_errors):
        if indices is None:
            return
        idx_list = indices.detach().cpu().tolist()
        err_list = td_errors.detach().abs().cpu().tolist()
        for i, e in zip(idx_list, err_list):
            if not self._valid[i]:
                # Defensive: refuse to lift an invalid leaf back into the sampling pool.
                continue
            priority = (float(e) + PER_EPSILON) ** ALPHA
            self.tree.update(i, priority)
            if (float(e) + PER_EPSILON) > self.max_priority:
                self.max_priority = float(e) + PER_EPSILON

    def valid_anchor_ratio(self) -> float:
        if self.entries == 0:
            return 0.0
        return self.valid_anchor_count / self.entries

    def sample(self, batch_size, preprocessor=None) -> Optional[WindowSampledBatch]:
        if self.valid_anchor_count < MIN_VALID_ANCHORS:
            logger.info(
                "WindowPER skip: valid_anchor_count=%d < MIN_VALID_ANCHORS=%d",
                self.valid_anchor_count, MIN_VALID_ANCHORS,
            )
            return None

        total = self.tree.total
        if total <= 0.0:
            return None
        segment = total / batch_size

        anchors: List[int] = []
        priorities: List[float] = []
        # Stratified sampling. Invalid leaves carry priority 0 so cannot be picked.
        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            for _attempt in range(8):
                s = random.uniform(lo, hi)
                data_idx = self.tree.get(s)
                if self._valid[data_idx]:
                    break
            else:
                # fallback: scan for a valid leaf
                data_idx = next(i for i, v in enumerate(self._valid) if v)
            tree_priority = self.tree.tree[data_idx + self.capacity - 1]
            anchors.append(data_idx)
            priorities.append(tree_priority)

        # IS weights
        beta = self._beta()
        priorities_t = torch.tensor(priorities, dtype=torch.float32, device=self.device)
        probs = priorities_t / max(total, 1e-12)
        weights = (self.entries * probs).clamp(min=1e-12).pow(-beta)
        weights = weights / weights.max()
        indices = torch.tensor(anchors, dtype=torch.long, device=self.device)

        # Build window tensors
        states_seq_list: List[torch.Tensor] = []
        actions_seq_list: List[List[int]] = []
        rewards_seq_list: List[List[float]] = []
        rtgs_seq_list: List[List[float]] = []
        next_state_last_list: List[Optional[torch.Tensor]] = []
        terminal_flags: List[bool] = []
        last_actions_list: List[int] = []
        last_rewards_list: List[float] = []

        rtg_none_count = 0
        rtg_total = 0

        for anchor in anchors:
            window_states_raw = []
            window_actions: List[int] = []
            window_rewards: List[float] = []
            window_rtgs: List[float] = []
            for k in range(self.K):
                idx = (anchor - (self.K - 1) + k) % self.capacity
                t: Transition = self.buffer[idx]
                window_states_raw.append(t.state)
                # action stored as (1,1) tensor in training; reduce to int
                if isinstance(t.action, torch.Tensor):
                    window_actions.append(int(t.action.item()))
                else:
                    window_actions.append(int(t.action))
                if isinstance(t.reward, torch.Tensor):
                    window_rewards.append(float(t.reward.item()))
                else:
                    window_rewards.append(float(t.reward))
                rtg_total += 1
                if t.rtg is None:
                    rtg_none_count += 1
                    window_rtgs.append(STARFORMER_RTG_TARGET)
                else:
                    window_rtgs.append(float(t.rtg))

            # Preprocess states
            if preprocessor is not None:
                processed = preprocessor(window_states_raw, device=self.device)
                # processed is a list of (1, 3, H, W) tensors
                states_tensor = torch.cat(processed, dim=0)  # (K, 3, H, W)
            else:
                states_tensor = torch.cat(window_states_raw, dim=0)

            states_seq_list.append(states_tensor)
            actions_seq_list.append(window_actions)
            rewards_seq_list.append(window_rewards)
            rtgs_seq_list.append(window_rtgs)

            anchor_t: Transition = self.buffer[anchor]
            terminal = anchor_t.next_state is None
            terminal_flags.append(terminal)
            if terminal:
                # zero placeholder; mask will skip it
                next_state_last_list.append(None)
            else:
                if preprocessor is not None:
                    nxt = preprocessor([anchor_t.next_state], device=self.device)[0]
                else:
                    nxt = anchor_t.next_state
                next_state_last_list.append(nxt)

            last_actions_list.append(window_actions[-1])
            last_rewards_list.append(window_rewards[-1])

        # Stack
        states_seq = torch.stack(states_seq_list, dim=0)  # (B, K, 3, H, W)
        # template for zero next-state on terminal
        zero_next = torch.zeros_like(states_seq[:, -1])
        next_state_last = zero_next.clone()
        for i, ns in enumerate(next_state_last_list):
            if ns is not None:
                next_state_last[i] = ns.squeeze(0) if ns.dim() == 4 else ns

        actions_seq = torch.tensor(actions_seq_list, dtype=torch.long, device=self.device)
        rewards_seq = torch.tensor(rewards_seq_list, dtype=torch.float32, device=self.device)
        rtgs_seq = torch.tensor(rtgs_seq_list, dtype=torch.float32, device=self.device)
        terminal_mask = torch.tensor(terminal_flags, dtype=torch.bool, device=self.device)
        last_actions = torch.tensor(last_actions_list, dtype=torch.long, device=self.device).unsqueeze(1)
        last_rewards = torch.tensor(last_rewards_list, dtype=torch.float32, device=self.device)

        self.rtg_none_ratio_last = (rtg_none_count / rtg_total) if rtg_total > 0 else 0.0

        return WindowSampledBatch(
            states_seq=states_seq,
            actions_seq=actions_seq,
            rewards_seq=rewards_seq,
            rtgs_seq=rtgs_seq,
            next_state_last=next_state_last,
            terminal_mask=terminal_mask,
            last_actions=last_actions,
            last_rewards=last_rewards,
            indices=indices,
            weights=weights,
        )


def create_replay_memory(method: str, capacity: int, device) -> _BaseReplayMemory:
    if method == "Uniform":
        return UniformReplayMemory(capacity, device)
    if method == "PER":
        return PrioritizedReplayMemory(capacity, device)
    if method == "WindowPER":
        return WindowPrioritizedReplayMemory(capacity, device)
    raise ValueError(f"Unsupported SAMPLING_METHOD: {method}")


# backwards-compat alias so any import of `ReplayMemory` still works
ReplayMemory = UniformReplayMemory
