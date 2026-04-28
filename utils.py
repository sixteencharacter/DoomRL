import torch
import math
import random
from typing import Optional
from config import *
from datamodel import ActionRes, StarformerContext
from model import policy_net
from env import env as GameEnv , device
from datamodel import ActionRes
from env import env as GameEnv, device
import logging
from copy import deepcopy
import os
from collections import deque
from datetime import datetime
from icecream import ic
from tensordict import TensorDict

logger = logging.getLogger(__name__)

weights_tracker = deque(maxlen=5)


def _starformer_predict(state: torch.Tensor, ctx: StarformerContext) -> torch.Tensor:
    """Build (1, K, ...) inputs from the rolling context (left-padded with zeros)
    and return argmax logits at the last sequence position."""
    K = ctx.K
    state_dim = state.shape[1:]  # (C, H, W)

    # Build sequence: take up to K-1 from history then append current state.
    history_states = list(ctx.states_window)
    history_actions = list(ctx.actions_window)
    history_rtgs = list(ctx.rtgs_window)

    history_states.append(state)
    history_actions.append(0)  # placeholder action at current step (Decision-Transformer convention)
    history_rtgs.append(ctx.rtg_target if not history_rtgs else history_rtgs[-1])

    if len(history_states) > K:
        history_states = history_states[-K:]
        history_actions = history_actions[-K:]
        history_rtgs = history_rtgs[-K:]

    seq_len = len(history_states)
    pad_len = K - seq_len

    # Left-pad with zero state and zero action, RTG=rtg_target.
    if pad_len > 0:
        zero_state = torch.zeros((1,) + tuple(state_dim), device=state.device, dtype=state.dtype)
        history_states = [zero_state] * pad_len + history_states
        history_actions = [0] * pad_len + history_actions
        history_rtgs = [ctx.rtg_target] * pad_len + history_rtgs

    states_seq = torch.cat([s.unsqueeze(0) if s.dim() == 3 else s for s in history_states], dim=0)
    if states_seq.dim() == 4:
        states_seq = states_seq.unsqueeze(0)  # (1, K, C, H, W)
    actions_seq = torch.tensor([history_actions], dtype=torch.long, device=state.device)
    rtgs_seq = torch.tensor([history_rtgs], dtype=torch.float32, device=state.device)

    # Validity mask: True at real positions
    valid = [False] * pad_len + [True] * seq_len
    attn_mask = torch.tensor([valid], dtype=torch.bool, device=state.device)

    with torch.no_grad():
        q_seq = policy_net(states_seq, actions_seq, rtgs_seq, attn_mask=attn_mask)
    return q_seq[:, -1, :].max(1).indices.view(1, 1)


def select_action(state, steps, ctx: Optional[StarformerContext] = None, inference: bool = False):

    logits = None
    is_starformer = (METHOD == 'STARFORMER') and (ctx is not None)

    if inference:
        if is_starformer:
            logits = _starformer_predict(state, ctx)
            ctx.push(state, int(logits.item()))
        else:
            with torch.no_grad():
                q_value = policy_net(state)
                logits = q_value.max(1).indices.view(1, 1)
    else:
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-steps / EPS_DECAY)
        # STARFORMER may opt out of epsilon if STARFORMER_USE_EPSILON=False.
        use_eps = True
        if is_starformer and not STARFORMER_USE_EPSILON:
            use_eps = False

        if (not use_eps) or sample > eps_threshold:
            if is_starformer:
                logits = _starformer_predict(state, ctx)
            else:
                with torch.no_grad():
                    logits = policy_net(state).max(1).indices.view(1, 1)
        else:
            logits = torch.tensor([[GameEnv.action_space.sample()]], device=device, dtype=torch.long)

        if is_starformer:
            ctx.push(state, int(logits.item()))

def select_action(state, steps, inference=False):
    # Import locally to avoid circular dependency
    from model import policy_net
    if METHOD == "PPO":
        from model import actor

    if METHOD == "PPO":
        if inference:
            with torch.no_grad():
                logits, _ = policy_net(state)
                action_idx = logits.argmax(1).view(1, 1)
                return ActionRes(steps, action_idx)
        else:
            # For PPO training, we use the actor module which returns action and log_prob
            with torch.no_grad():
                td = TensorDict({"observation": state}, batch_size=[state.shape[0]], device=device)
                td = actor(td)
                # td["action"] is one-hot because of OneHotCategorical
                action_idx = td["action"].argmax(-1).view(1, 1)
                # We return the whole TensorDict for PPO as it contains log_prob etc.
                return ActionRes(steps, action_idx, td)

    if inference:
        with torch.no_grad():
            q_value = policy_net(state)
            logits = q_value.max(1).indices.view(1, 1)
    else:
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-steps / EPS_DECAY)
        if sample > eps_threshold:
            with torch.no_grad():
                logits = policy_net(state).max(1).indices.view(1, 1)
        else:
            logits = torch.tensor([[GameEnv.action_space.sample()]], device=device, dtype=torch.long)

    return ActionRes(steps, logits)

def save_state_dict(model, optimizer, steps=None, persisted=False):
    if not os.path.isdir("weights"):
        os.mkdir("weights")
    save_path = f"weights/{ARCH}-{VERSION}-{VARIANT}-{SAMPLING_METHOD}{'-{}steps'.format(steps + CHKPOINT_NUM) if steps is not None else ''}.pth"
    logger.info("Save state dict to {}".format(save_path))

    localModel = deepcopy(model).to("cpu")
    localOptimizer = deepcopy(optimizer)

    if not persisted:
        if len(weights_tracker) == weights_tracker.maxlen:
            old_path = weights_tracker[0]
            if os.path.exists(old_path):
                os.remove(old_path)
        
        weights_tracker.append(save_path)

    torch.save({
        'model': localModel.state_dict(),
        'optimizer': localOptimizer.state_dict()
    }, save_path)
