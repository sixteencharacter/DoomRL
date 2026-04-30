import logging
from datetime import datetime
import torch
import torch.nn as nn
from torch.distributions import Categorical
from config import *
from datamodel import Transition, training_info, StarformerContext, WindowSampledBatch
from env import device, env as GameEnv
from model import policy_net, target_net, optimizer, scheduler, memory, loss_module, advantage_module
from itertools import count
from utils import select_action, save_state_dict
from tqdm import tqdm
from preprocessor import base_preprocessor
from icecream import ic
from inference import infer
import wandb
import os

if USE_PPO and METHOD != 'STARFORMER':
    from tensordict import TensorDict

if USE_PPO:
    _PPO_USE_AMP = bool(PPO_USE_AMP) and (device.type == "cuda")
    _ppo_scaler = torch.cuda.amp.GradScaler(enabled=_PPO_USE_AMP) if _PPO_USE_AMP else None
else:
    _PPO_USE_AMP = False
    _ppo_scaler = None


if not os.path.isdir("logs"):
    os.mkdir("logs")

logging.basicConfig(
    filename='logs/run-{}.log'.format(datetime.now().isoformat()),
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s',
    filemode="a"
)

isDatapointEnough = False
ppo_buffer = []
# STARFORMER+PPO: each entry is dict with windowed sequences + transition info
sf_ppo_buffer = []


def _build_window_tensors(ctx, current_state, device):
    """Return (states_seq[1,K,C,H,W], actions_seq[1,K], rtgs_seq[1,K], attn_mask[1,K]) for current step."""
    K = ctx.K
    state_dim = current_state.shape[1:]  # (C,H,W)
    history_states = list(ctx.states_window)
    history_actions = list(ctx.actions_window)
    history_rtgs = list(ctx.rtgs_window)

    history_states.append(current_state)
    history_actions.append(0)
    history_rtgs.append(ctx.rtg_target if not history_rtgs else history_rtgs[-1])

    if len(history_states) > K:
        history_states = history_states[-K:]
        history_actions = history_actions[-K:]
        history_rtgs = history_rtgs[-K:]

    seq_len = len(history_states)
    pad_len = K - seq_len
    if pad_len > 0:
        zero_state = torch.zeros((1,) + tuple(state_dim), device=device, dtype=current_state.dtype)
        history_states = [zero_state] * pad_len + history_states
        history_actions = [0] * pad_len + history_actions
        history_rtgs = [ctx.rtg_target] * pad_len + history_rtgs

    states_seq = torch.cat([s.unsqueeze(0) if s.dim() == 3 else s for s in history_states], dim=0)
    if states_seq.dim() == 4:
        states_seq = states_seq.unsqueeze(0)
    actions_seq = torch.tensor([history_actions], dtype=torch.long, device=device)
    rtgs_seq = torch.tensor([history_rtgs], dtype=torch.float32, device=device)
    valid = [False] * pad_len + [True] * seq_len
    attn_mask = torch.tensor([valid], dtype=torch.bool, device=device)
    return states_seq, actions_seq, rtgs_seq, attn_mask


def _starformer_ppo_select_action(ctx, processed_state):
    """Query StarformerActorCritic on K-window, sample Categorical(logits)."""
    states_seq, actions_seq, rtgs_seq, attn_mask = _build_window_tensors(ctx, processed_state, device)
    if _PPO_USE_AMP:
        with torch.no_grad(), torch.cuda.amp.autocast():
            logits, value = policy_net(states_seq, actions_seq, rtgs_seq, attn_mask=attn_mask)
        logits = logits.float()
        value = value.float()
    else:
        with torch.no_grad():
            logits, value = policy_net(states_seq, actions_seq, rtgs_seq, attn_mask=attn_mask)
    dist = Categorical(logits=logits)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action, logits, value, log_prob, states_seq.squeeze(0), actions_seq.squeeze(0), rtgs_seq.squeeze(0), attn_mask.squeeze(0)


def _starformer_ppo_collect(states_seq, actions_seq, rtgs_seq, attn_mask, action, log_prob, value, reward, done, terminated, episode_id):
    sf_ppo_buffer.append({
        "states_seq": states_seq.detach(),
        "actions_seq": actions_seq.detach(),
        "rtgs_seq": rtgs_seq.detach(),
        "attn_mask": attn_mask.detach(),
        "action": int(action.item()),
        "log_prob": log_prob.detach().view(-1)[0],
        "value": value.detach().view(-1)[0],
        "reward": float(reward.item() if torch.is_tensor(reward) else reward),
        "done": bool(done),
        "terminated": bool(terminated),
        "episode_id": int(episode_id),
        "buffer_idx": len(sf_ppo_buffer),
    })


def _starformer_ppo_backfill_rtg(episode_id, rtgs):
    """Write per-step RTG into rtgs_seq[-1] for transitions belonging to this episode."""
    j = 0
    for entry in sf_ppo_buffer:
        if entry["episode_id"] != episode_id:
            continue
        if j < len(rtgs):
            entry["rtgs_seq"] = entry["rtgs_seq"].clone()
            entry["rtgs_seq"][-1] = float(rtgs[j])
        j += 1


def _compute_gae(rewards, values, dones, gamma, lam):
    """Standard GAE-Lambda. Bootstrap with 0 at trajectory tail (handled by done)."""
    T = len(rewards)
    advantages = [0.0] * T
    last_gae = 0.0
    next_value = 0.0
    for t in reversed(range(T)):
        non_terminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        last_gae = delta + gamma * lam * non_terminal * last_gae
        advantages[t] = last_gae
        next_value = values[t]
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


def _optimize_starformer_ppo():
    if len(sf_ppo_buffer) < PPO_BUFFER_SIZE:
        return False
    training_info.learning_step += 1

    rewards = [e["reward"] for e in sf_ppo_buffer]
    values = [float(e["value"].item()) for e in sf_ppo_buffer]
    dones = [e["done"] or e["terminated"] for e in sf_ppo_buffer]
    advantages, returns = _compute_gae(rewards, values, dones, GAMMA, GAE_LAMBDA)

    states_seq = torch.stack([e["states_seq"] for e in sf_ppo_buffer], dim=0).to(device)
    actions_seq = torch.stack([e["actions_seq"] for e in sf_ppo_buffer], dim=0).to(device)
    rtgs_seq = torch.stack([e["rtgs_seq"] for e in sf_ppo_buffer], dim=0).to(device)
    attn_mask = torch.stack([e["attn_mask"] for e in sf_ppo_buffer], dim=0).to(device)
    actions = torch.tensor([e["action"] for e in sf_ppo_buffer], dtype=torch.long, device=device)
    old_log_probs = torch.stack([e["log_prob"] for e in sf_ppo_buffer]).to(device).detach()
    adv_t = torch.tensor(advantages, dtype=torch.float32, device=device)
    ret_t = torch.tensor(returns, dtype=torch.float32, device=device)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    N = states_seq.shape[0]
    last_total = last_obj = last_critic = last_ent = 0.0
    for _epoch in range(PPO_EPOCHS):
        perm = torch.randperm(N, device=device)
        for i in range(0, N, PPO_BATCH_SIZE):
            idx = perm[i:i + PPO_BATCH_SIZE]
            mb_states = states_seq[idx]
            mb_actions_seq = actions_seq[idx]
            mb_rtgs = rtgs_seq[idx]
            mb_attn = attn_mask[idx]
            mb_act = actions[idx]
            mb_old_logp = old_log_probs[idx]
            mb_adv = adv_t[idx]
            mb_ret = ret_t[idx]

            optimizer.zero_grad(set_to_none=True)
            if _PPO_USE_AMP:
                with torch.cuda.amp.autocast():
                    logits, value = policy_net(mb_states, mb_actions_seq, mb_rtgs, attn_mask=mb_attn)
                    dist = Categorical(logits=logits)
                    new_logp = dist.log_prob(mb_act)
                    entropy = dist.entropy().mean()
                    ratio = torch.exp(new_logp - mb_old_logp)
                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * mb_adv
                    loss_obj = -torch.min(surr1, surr2).mean()
                    value = value.squeeze(-1)
                    loss_critic = nn.functional.smooth_l1_loss(value, mb_ret)
                    loss_ent = -ENTROPY_COEF * entropy
                    loss = loss_obj + CRITIC_COEF * loss_critic + loss_ent
                _ppo_scaler.scale(loss).backward()
                _ppo_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                _ppo_scaler.step(optimizer)
                _ppo_scaler.update()
            else:
                logits, value = policy_net(mb_states, mb_actions_seq, mb_rtgs, attn_mask=mb_attn)
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(mb_act)
                entropy = dist.entropy().mean()
                ratio = torch.exp(new_logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * mb_adv
                loss_obj = -torch.min(surr1, surr2).mean()
                value = value.squeeze(-1)
                loss_critic = nn.functional.smooth_l1_loss(value, mb_ret)
                loss_ent = -ENTROPY_COEF * entropy
                loss = loss_obj + CRITIC_COEF * loss_critic + loss_ent
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            last_total = float(loss.item())
            last_obj = float(loss_obj.item())
            last_critic = float(loss_critic.item())
            last_ent = float(entropy.item())

    sf_ppo_buffer.clear()
    wandb.log({
        "loss/total": last_total,
        "loss/objective": last_obj,
        "loss/critic": last_critic,
        "loss/entropy": last_ent,
        "advantage_mean": float(adv_t.mean().item()),
        "lr": optimizer.param_groups[0]['lr'],
    }, step=training_info.learning_step)
    return True


def _optimize_starformer():
    """STARFORMER branch: sample K-windows, transformer forward, TD loss at last position.
    Returns True if step taken, False if skipped."""
    sampled = memory.sample(BATCH_SIZE, base_preprocessor)
    if sampled is None:
        wandb.log({"optimize_skipped": 1}, step=training_info.learning_step)
        return False
    assert isinstance(sampled, WindowSampledBatch)
    training_info.learning_step += 1

    states_seq = sampled.states_seq
    actions_seq = sampled.actions_seq
    rtgs_seq = sampled.rtgs_seq
    last_actions = sampled.last_actions
    last_rewards = sampled.last_rewards
    next_state_last = sampled.next_state_last
    terminal_mask = sampled.terminal_mask
    weights = sampled.weights
    indices = sampled.indices

    B, K = actions_seq.shape

    q_seq = policy_net(states_seq, actions_seq, rtgs_seq)
    state_action_values = q_seq[:, -1, :].gather(1, last_actions).squeeze(1)

    next_state_values = torch.zeros(B, device=device)
    with torch.no_grad():
        non_term = ~terminal_mask
        if non_term.any():
            target_states = torch.cat([states_seq[:, 1:], next_state_last.unsqueeze(1)], dim=1)
            target_actions = torch.zeros_like(actions_seq)
            target_actions[:, :-1] = actions_seq[:, 1:]
            target_rtgs = torch.zeros_like(rtgs_seq)
            target_rtgs[:, :-1] = rtgs_seq[:, 1:]
            target_rtgs[:, -1] = rtgs_seq[:, -1]
            policy_q_next = policy_net(target_states, target_actions, target_rtgs)[:, -1, :]
            next_actions = policy_q_next.argmax(1, keepdim=True)
            target_q_next = target_net(target_states, target_actions, target_rtgs)[:, -1, :]
            chosen = target_q_next.gather(1, next_actions).squeeze(1)
            next_state_values = chosen * non_term.float()

    expected = (next_state_values * GAMMA) + last_rewards

    criterion = nn.SmoothL1Loss(reduction='none')
    elementwise_loss = criterion(state_action_values, expected)
    loss = (weights * elementwise_loss).mean()

    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    td_errors = elementwise_loss.detach()
    memory.update_priorities(indices, td_errors)

    wandb.log({
        "loss": loss.item(),
        "seq_len": K,
        "rtg_mean": rtgs_seq.mean().item(),
        "transformer_grad_norm": float(grad_norm),
        "valid_anchor_ratio": memory.valid_anchor_ratio(),
        "rtg_none_ratio": memory.rtg_none_ratio_last,
        "rtg_backfill_overwrites": memory.rtg_backfill_overwrites,
        "priority/mean": memory.priority_mean(),
        "priority/std":  memory.priority_std(),
        "priority/max":  memory.priority_max(),
        "is_weight/mean": weights.mean().item(),
        "is_weight/max":  weights.max().item(),
        "lr": optimizer.param_groups[0]['lr'],
    }, step=training_info.learning_step)
    return True


def _optimize_ppo():
    """PPO branch: when ppo_buffer is full, run K epochs of minibatch updates.
    Returns True if a step was taken."""
    if len(ppo_buffer) < PPO_BUFFER_SIZE:
        return False
    training_info.learning_step += 1

    data = torch.stack(ppo_buffer, dim=0).to(device)
    ppo_buffer.clear()

    policy_net.eval()
    with torch.no_grad():
        advantage_module(data)
    policy_net.train()

    adv = data["advantage"]
    data["advantage"] = (adv - adv.mean()) / (adv.std() + 1e-8)

    last_loss_total = 0.0
    last_loss_obj = 0.0
    last_loss_critic = 0.0
    last_loss_entropy = 0.0
    for _epoch in range(PPO_EPOCHS):
        perm = torch.randperm(data.shape[0])
        for i in range(0, data.shape[0], PPO_BATCH_SIZE):
            batch_indices = perm[i:i + PPO_BATCH_SIZE]
            batch_data = data[batch_indices]
            optimizer.zero_grad(set_to_none=True)
            if _PPO_USE_AMP:
                with torch.cuda.amp.autocast():
                    loss_vals = loss_module(batch_data)
                    loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                    )
                _ppo_scaler.scale(loss_value).backward()
                _ppo_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                _ppo_scaler.step(optimizer)
                _ppo_scaler.update()
            else:
                loss_vals = loss_module(batch_data)
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()
            last_loss_total = loss_value.item()
            last_loss_obj = loss_vals["loss_objective"].item()
            last_loss_critic = loss_vals["loss_critic"].item()
            last_loss_entropy = loss_vals["loss_entropy"].item()

    # NOTE: torch.cuda.empty_cache() removed — was a per-update perf killer.

    wandb.log({
        "loss/total": last_loss_total,
        "loss/objective": last_loss_obj,
        "loss/critic": last_loss_critic,
        "loss/entropy": last_loss_entropy,
    }, step=training_info.learning_step)
    return True


def optimize_model(preprocessor):
    """Returns True iff a gradient step occurred this call."""
    global isDatapointEnough

    if USE_PPO and METHOD != 'STARFORMER':
        stepped = _optimize_ppo()
        if stepped:
            isDatapointEnough = True
        return stepped

    if METHOD == "STARFORMER" and USE_PPO:
        stepped = _optimize_starformer_ppo()
        if stepped:
            isDatapointEnough = True
        return stepped

    if memory is None or len(memory) < BATCH_SIZE:
        logging.info("Skipped Optimization due to insufficient data points")
        return False

    if METHOD == 'STARFORMER':
        stepped = _optimize_starformer()
        if stepped:
            isDatapointEnough = True
        return stepped

    isDatapointEnough = True
    training_info.learning_step += 1
    sampled = memory.sample(BATCH_SIZE, preprocessor)
    batch = Transition(*zip(*sampled.transitions))
    weights = sampled.weights

    non_final_mask = torch.tensor(
        tuple(s is not None for s in batch.next_state),
        device=device, dtype=torch.bool,
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        if non_final_next_states.size(0) > 0:
            if METHOD == "DDQN":
                next_actions = policy_net(non_final_next_states).argmax(1, keepdim=True)
                next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, next_actions).squeeze(1)
            else:
                next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    expected = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss(reduction='none')
    elementwise_loss = criterion(state_action_values, expected.unsqueeze(1)).squeeze(1)
    assert elementwise_loss.shape == weights.shape
    loss = (weights * elementwise_loss).mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    td_errors = elementwise_loss.detach()
    memory.update_priorities(sampled.indices, td_errors)

    log_payload = {"loss": loss.item()}
    if sampled.indices is not None:
        log_payload.update({
            "priority/mean": memory.priority_mean(),
            "priority/std":  memory.priority_std(),
            "priority/max":  memory.priority_max(),
            "is_weight/mean": weights.mean().item(),
            "is_weight/max":  weights.max().item(),
        })
    wandb.log(log_payload, step=training_info.learning_step)
    return True


def _compute_episode_rtg(rewards):
    """Discounted return-to-go: g_t = sum_{u>=t} gamma^{u-t} r_u."""
    rtgs = [0.0] * len(rewards)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = float(rewards[t]) + GAMMA * running
        rtgs[t] = running
    return rtgs


def _ppo_select_action(processed_state):
    """For PPO: sample action via Categorical(logits) on policy_net actor head.
    Returns (action_int, logits, value, log_prob)."""
    if _PPO_USE_AMP:
        with torch.no_grad(), torch.cuda.amp.autocast():
            logits, value = policy_net(processed_state)
        # cast back to float32 for distribution stability
        logits = logits.float()
        value = value.float()
    else:
        with torch.no_grad():
            logits, value = policy_net(processed_state)
    dist = Categorical(logits=logits)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action, logits, value, log_prob


def _ppo_collect(processed_state, next_processed_state, action, logits, value, log_prob, reward, terminated, done):
    """Build a single-step TensorDict and push into ppo_buffer."""
    n_act = logits.shape[-1]
    action_one_hot = torch.zeros((1, n_act), device=device)
    action_one_hot[0, int(action.item())] = 1.0
    td = TensorDict({
        "observation": processed_state.detach(),
        "logits": logits.detach(),
        "action": action_one_hot,
        "sample_log_prob": log_prob.detach().view(1),
        "next": TensorDict({
            "observation": next_processed_state.detach(),
            "reward": reward.view(1, 1),
            "done": torch.tensor([[done]], device=device, dtype=torch.bool),
            "terminated": torch.tensor([[terminated]], device=device, dtype=torch.bool),
        }, batch_size=[1], device=device),
    }, batch_size=[1], device=device)
    ppo_buffer.append(td.squeeze(0))


def train(num_episodes, preprocessor):
    try:
        iteration_mode = "episode" if NUM_EPISODE is not None else "steps"
        logging.info(f"Using iteration mode of {iteration_mode}")

        total_iteration = NUM_EPISODE if NUM_EPISODE is not None else MAX_STEPS
        iteration = tqdm(range(total_iteration), total=total_iteration)

        training_end = False
        mx_cum_reward = -1e9
        episode_id_counter = 0

        for i_episodes in count():
            cum_reward = 0
            game_state, _info = GameEnv.reset()
            state = torch.tensor(game_state['screen'].copy(), dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)
            processed_state = preprocessor(state, device=device)

            current_episode_id = episode_id_counter
            episode_id_counter += 1

            ctx = None
            episode_rewards = []
            if METHOD == 'STARFORMER':
                ctx = StarformerContext(K=STARFORMER_K, rtg_target=STARFORMER_RTG_TARGET)

            for _t in count():
                # Action selection
                if USE_PPO and METHOD != 'STARFORMER':
                    action_t, logits_t, value_t, log_prob_t = _ppo_select_action(processed_state)
                    action_int = int(action_t.item())
                elif METHOD == "STARFORMER" and USE_PPO:
                    (action_t, logits_t, value_t, log_prob_t,
                     sf_states_seq, sf_actions_seq, sf_rtgs_seq, sf_attn) = _starformer_ppo_select_action(ctx, processed_state)
                    action_int = int(action_t.item())
                else:
                    action = select_action(processed_state, training_info.learning_step, ctx=ctx)
                    action_int = int(action.logits.item())

                observation, reward, terminated, truncated, _ = GameEnv.step(action_int)
                if iteration_mode == "steps":
                    iteration.n = training_info.learning_step
                    iteration.refresh()
                    iteration.set_description(f"Step Reward {reward}")

                episode_rewards.append(float(reward))
                reward_t = torch.tensor([reward], device=device, dtype=torch.float32)
                done = terminated or truncated

                if terminated:
                    next_state = None
                    next_processed_state = torch.zeros_like(processed_state)
                else:
                    next_state = torch.tensor(observation['screen'].copy(), dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)
                    next_processed_state = preprocessor(next_state, device=device)

                # Buffer push
                if USE_PPO and METHOD != 'STARFORMER':
                    _ppo_collect(
                        processed_state, next_processed_state,
                        action_t, logits_t, value_t, log_prob_t,
                        reward_t, terminated, done,
                    )
                elif METHOD == "STARFORMER" and USE_PPO:
                    _starformer_ppo_collect(
                        sf_states_seq, sf_actions_seq, sf_rtgs_seq, sf_attn,
                        action_t, log_prob_t, value_t, reward_t,
                        done, terminated, current_episode_id,
                    )
                    # Push action into rolling ctx so next window includes the chosen action.
                    ctx.push(processed_state, action_int)
                else:
                    memory.push(state, action.logits, next_state, reward_t, None, current_episode_id)

                cum_reward += float(reward)
                if iteration_mode == "episode":
                    iteration.set_description(f"Episode reward: {cum_reward}")

                state = next_state
                processed_state = next_processed_state if next_state is not None else processed_state

                did_optimize = optimize_model(preprocessor)

                # Soft target update — DQN/DDQN/STARFORMER only
                if not USE_PPO and target_net is not None:
                    target_state = target_net.state_dict()
                    policy_state = policy_net.state_dict()
                    for key in policy_state:
                        target_state[key] = policy_state[key] * TAU + target_state[key] * (1 - TAU)
                    target_net.load_state_dict(target_state)

                # Validation
                if did_optimize and iteration_mode == "steps":
                    if training_info.learning_step % VALIDATION_INTERVAL == 0 and training_info.learning_step != 0:
                        print(f"Validation at step {training_info.learning_step}")
                        policy_net.eval()
                        rewards_list = infer(VALIDATION_EPISODES)
                        val_mean = sum(rewards_list) / len(rewards_list)
                        training_info.eval_mean_rewards.append((training_info.learning_step, val_mean))
                        training_info.to_csv("evaluation.csv")
                        wandb.log({"val_mean_reward": val_mean}, step=training_info.learning_step)
                        policy_net.train()
                        game_state, _info = GameEnv.reset()
                        state = torch.tensor(game_state['screen'], dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)
                        processed_state = preprocessor(state, device=device)
                        if METHOD == 'STARFORMER':
                            ctx = StarformerContext(K=STARFORMER_K, rtg_target=STARFORMER_RTG_TARGET)
                            current_episode_id = episode_id_counter
                            episode_id_counter += 1
                            episode_rewards = []
                            cum_reward = 0

                # Saving
                if did_optimize and training_info.learning_step % SAVING_INTERVAL == 0 and training_info.learning_step != 0:
                    if isDatapointEnough:
                        should_persist = (cum_reward >= mx_cum_reward)
                        mx_cum_reward = max(cum_reward, mx_cum_reward)
                        if should_persist:
                            logging.info(f"Saving weight with persisting = {should_persist}")
                            save_state_dict(policy_net, optimizer, steps=training_info.learning_step, persisted=should_persist)
                        else:
                            logging.info(f"Saving weight with persisting = {should_persist} (interval saving)")
                            save_state_dict(policy_net, optimizer, steps=training_info.learning_step, persisted=False)

                if (iteration_mode == "steps" and training_info.learning_step >= MAX_STEPS - 1) or \
                   (iteration_mode == "episode" and i_episodes >= NUM_EPISODE - 1):
                    training_end = True
                    break

                if done:
                    if METHOD == 'STARFORMER' and USE_PPO:
                        rtgs = _compute_episode_rtg(episode_rewards)
                        _starformer_ppo_backfill_rtg(current_episode_id, rtgs)
                    elif METHOD == 'STARFORMER' and memory is not None and hasattr(memory, 'backfill_rtg'):
                        rtgs = _compute_episode_rtg(episode_rewards)
                        memory.backfill_rtg(current_episode_id, rtgs)
                    wandb.log({"episode_reward": cum_reward}, step=training_info.learning_step)
                    break

            if training_end:
                break
    finally:
        logging.info("closing preprocessor thread")
        base_preprocessor.close()
        wandb.finish()


if __name__ == "__main__":
    api_key = input("Enter wandb API key: ").strip()
    wandb.login(key=api_key)
    wandb.init(
        project="vizdoom-dqn",
        name=f"{ARCH}-{VERSION}-{VARIANT}-{METHOD}-{SAMPLING_METHOD}",
        config={
            "gamma": GAMMA, "lr": LR, "batch_size": BATCH_SIZE,
            "eps_start": EPS_START, "eps_end": EPS_END, "eps_decay": EPS_DECAY,
            "tau": TAU, "memory_cap": MEMORY_CAP, "max_steps": MAX_STEPS,
            "frame_skip": FRAME_SKIP, "resolution": RESOLUTION,
            "method": METHOD,
            "sampling_method": SAMPLING_METHOD,
            "alpha": ALPHA if SAMPLING_METHOD in ("PER", "WindowPER") else None,
            "beta_start": BETA_START if SAMPLING_METHOD in ("PER", "WindowPER") else None,
            "beta_end": BETA_END if SAMPLING_METHOD in ("PER", "WindowPER") else None,
            "starformer_k": STARFORMER_K if METHOD == 'STARFORMER' else None,
            "starformer_layers": STARFORMER_LAYERS if METHOD == 'STARFORMER' else None,
            "starformer_heads": STARFORMER_HEADS if METHOD == 'STARFORMER' else None,
            "starformer_dim": STARFORMER_DIM if METHOD == 'STARFORMER' else None,
            "starformer_lr": STARFORMER_LR if METHOD == 'STARFORMER' else None,
            "use_rtg": USE_RTG if METHOD == 'STARFORMER' else None,
            "ppo_epochs": PPO_EPOCHS if METHOD == 'PPO' else None,
            "ppo_clip": PPO_CLIP if METHOD == 'PPO' else None,
            "ppo_buffer_size": PPO_BUFFER_SIZE if METHOD == 'PPO' else None,
        }
    )
    wandb.watch(policy_net, log="all", log_freq=100)
    logging.info(f"Running with batch_size={BATCH_SIZE} method={METHOD}")
    train(NUM_EPISODE, base_preprocessor)
