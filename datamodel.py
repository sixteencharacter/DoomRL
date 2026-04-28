import collections
import dataclasses
import pandas as pd
import torch
from typing import Tuple, List, Optional, Any, Deque

ActionRes = collections.namedtuple('ActionRes',('step','logits', 'td'), defaults=(None,))

@dataclasses.dataclass
class Transition:
    """Mutable transition record. Mutability allows in-place RTG backfill at episode close.

    Field order matters: round-trip via ``Transition(*zip(*list_of_transitions))`` relies on
    positional construction matching declared field order.
    """

    state: Any
    action: Any
    next_state: Any
    reward: Any
    rtg: Optional[float] = None
    episode_id: int = 0

    def __iter__(self):
        # Yields fields in declaration order so `zip(*transitions)` works as with namedtuple.
        yield self.state
        yield self.action
        yield self.next_state
        yield self.reward
        yield self.rtg
        yield self.episode_id


ActionRes = collections.namedtuple('ActionRes', ('step', 'logits'))


@dataclasses.dataclass
class TrainingInfo:
    learning_step = 0
    eval_mean_rewards: List[Tuple[int, float]] = dataclasses.field(default_factory=list)

    def to_csv(self, path):
        df = pd.DataFrame(self.eval_mean_rewards, columns=['t', 'mean_reward'])
        df.to_csv(path)


training_info = TrainingInfo()


@dataclasses.dataclass
class SampledBatch:
    transitions: List[Transition]
    indices: Optional[torch.Tensor]  # None for Uniform; tensor of buffer indices for PER
    weights: torch.Tensor             # ones for Uniform; IS weights for PER


@dataclasses.dataclass
class WindowSampledBatch:
    """Window-sampled batch for STARFORMER.

    Tensor shapes:
      states_seq:        (B, K, C, H, W)
      actions_seq:       (B, K) long
      rewards_seq:       (B, K) float
      rtgs_seq:          (B, K) float
      next_state_last:   (B, C, H, W) — None-mask via terminal_mask
      terminal_mask:     (B,) bool — True where the anchor's next_state is None
      last_actions:      (B, 1) long — anchor action for gather
      last_rewards:      (B,) float — anchor reward for TD target
      indices:           (B,) long — anchor indices in ring buffer
      weights:           (B,) float — IS weights
    """

    states_seq: torch.Tensor
    actions_seq: torch.Tensor
    rewards_seq: torch.Tensor
    rtgs_seq: torch.Tensor
    next_state_last: torch.Tensor
    terminal_mask: torch.Tensor
    last_actions: torch.Tensor
    last_rewards: torch.Tensor
    indices: torch.Tensor
    weights: torch.Tensor


@dataclasses.dataclass
class StarformerContext:
    """Per-episode rolling window of (state, action, rtg) used by ``select_action``.

    Created at the top of each episode (after env.reset). Pushed after each chosen action.
    K and rtg_target read once at construction from cfg.
    """

    K: int
    rtg_target: float
    states_window: Deque = dataclasses.field(default_factory=collections.deque)
    actions_window: Deque = dataclasses.field(default_factory=collections.deque)
    rtgs_window: Deque = dataclasses.field(default_factory=collections.deque)

    def __post_init__(self):
        # Bound deques to K
        self.states_window = collections.deque(self.states_window, maxlen=self.K)
        self.actions_window = collections.deque(self.actions_window, maxlen=self.K)
        self.rtgs_window = collections.deque(self.rtgs_window, maxlen=self.K)

    def push(self, state: torch.Tensor, action: int, rtg: Optional[float] = None) -> None:
        self.states_window.append(state)
        self.actions_window.append(int(action))
        self.rtgs_window.append(self.rtg_target if rtg is None else float(rtg))

    def __len__(self):
        return len(self.states_window)
