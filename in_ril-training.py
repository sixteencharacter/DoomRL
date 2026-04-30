from datamodel import training_info
import torch
import torch.nn.functional as F
import numpy as np
import random
from env import device
from model import policy_net, advantage_module, loss_module, optimizer
from itertools import count
from env import device, env as GameEnv
from utils import select_action, save_state_dict
from tqdm import tqdm
from tensordict import TensorDict
from preprocessor import base_preprocessor
import fire
import wandb
import logging
import os
from datetime import datetime
from typing import List, TypedDict
from collections import namedtuple
from config import *
from inference import infer

TrainingConfig = namedtuple('TrainingConfig',(
    'ppo_buffer_size','ppo_epochs','ppo_batch_size',
    'validation_interval','max_steps','saving_interval','m','use_wandb'
))

if not os.path.isdir("logs"):
    os.mkdir("logs")

logging.basicConfig(
    filename='logs/run-{}.log'.format(datetime.now().isoformat()),
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s',
    filemode="a"
)


def sample_il_batch(batch_size: int = 32):

    # Use mmap_mode to avoid loading the full dataset into RAM
    obs = np.load("dataset/maze/human_states.npy", mmap_mode="r")      # shape: (N, H, W, C)
    actions = np.load("dataset/maze/human_actions.npy", mmap_mode="r")  # shape: (N,)

    indices = np.random.choice(len(obs), size=batch_size, replace=False)

    # Copy only the sampled rows off the memory-mapped file
    obs_t = torch.tensor(obs[indices].copy(), dtype=torch.float32).permute(0, 3, 1, 2).to(device)  # (B, C, H, W)
    actions_t = torch.tensor(actions[indices].copy(), dtype=torch.long).to(device)                  # (B,)

    return TensorDict(
        {"observation": obs_t, "action": actions_t},
        batch_size=[batch_size],
        device=device
    )


def do_gradient_surgery(base_parameters, il_gradient, rl_gradients):
    # PCGrad algorithm (Yu et al., 2020).
    #
    # rl_gradients : List[List[Tensor]]  -- one gradient list per RL mini-batch step (m total)
    # il_gradient  : List[Tensor]        -- one gradient list from the single IL update
    #
    # All m+1 gradient vectors are treated as independent tasks.
    # For each task i, iterate over every other task j in a random order;
    # if g_i conflicts with g_j (dot < 0), project g_i onto the plane
    # orthogonal to g_j:  g_i <- g_i - (g_i.g_j / ||g_j||^2) * g_j
    # param.grad is set to the sum of all m+1 processed gradients.

    # Build the flat list of all tasks: m RL gradients + 1 IL gradient
    all_gradients = rl_gradients + [il_gradient]  # List[List[Tensor]], length m+1
    n_tasks = len(all_gradients)

    processed = []
    for i in range(n_tasks):
        other_indices = list(range(n_tasks))
        other_indices.remove(i)
        random.shuffle(other_indices)  # random projection order per PCGrad

        grads_i = [g.clone() if g is not None else None for g in all_gradients[i]]

        for j in other_indices:
            grads_j = all_gradients[j]
            for k, (g_i, g_j) in enumerate(zip(grads_i, grads_j)):
                if g_i is None or g_j is None:
                    continue
                # dot product and projection are per-parameter (not global)
                g_i_flat, g_j_flat = g_i.flatten(), g_j.flatten()
                dot = torch.dot(g_i_flat, g_j_flat)
                if dot < 0:
                    norm_sq = torch.dot(g_j_flat, g_j_flat).clamp(min=1e-12)
                    grads_i[k] = g_i - (dot / norm_sq) * g_j

        processed.append(grads_i)

    # Sum all processed task gradients and write into param.grad
    params = list(base_parameters)
    for k, param in enumerate(params):
        grads_k = [processed[i][k] for i in range(n_tasks) if processed[i][k] is not None]
        param.grad = torch.stack(grads_k).sum(dim=0).reshape(param.shape) if grads_k else None


def optimize_model(
        preprocessor,
        ppo_buffer: List[TensorDict],
        cfg: TrainingConfig,
    ):

    data = torch.stack(ppo_buffer, dim=0).to(device)
    data["observation"] = preprocessor(data["observation"])
    ppo_buffer.clear()

    policy_net.eval()
    with torch.no_grad():
        advantage_module(data)
    policy_net.train()

    adv = data["advantage"]
    data["advantage"] = (adv - adv.mean()) / (adv.std() + 1e-8)

    il_batch = sample_il_batch(batch_size=cfg.ppo_batch_size)
    il_obs = preprocessor(il_batch["observation"], device=device)
    il_actions = il_batch["action"]

    for _ in range(cfg.ppo_epochs):
        indices = torch.randperm(data.shape[0])
        rl_grads_accumulator = []  # collects one gradient list per RL mini-batch (up to m)

        for i in range(0, data.shape[0], cfg.ppo_batch_size):
            batch_indices = indices[i: i + cfg.ppo_batch_size]
            batch_data = data[batch_indices]

            # --- RL (PPO) gradient for this mini-batch ---
            loss_vals = loss_module(batch_data)
            rl_loss = loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"]

            rl_grads_accumulator.append(
                list(torch.autograd.grad(rl_loss, policy_net.parameters(), allow_unused=True))
            )

            # Every m RL steps, perform one IL update via gradient surgery
            if len(rl_grads_accumulator) >= cfg.m:
                # --- IL (behavioural cloning) gradient ---
                logits, _ = policy_net(il_obs)
                il_loss = F.cross_entropy(logits, il_actions)
                il_grads = list(torch.autograd.grad(il_loss, policy_net.parameters(), allow_unused=True))

                # --- Gradient surgery: m RL gradients + 1 IL gradient ---
                optimizer.zero_grad()
                do_gradient_surgery(policy_net.parameters(), il_grads, rl_grads_accumulator)
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()
                rl_grads_accumulator = []  # reset for the next m-step window

    training_info.learning_step += 1
    torch.cuda.empty_cache()

    if cfg.use_wandb:
        wandb.log({
            "loss/rl_total": rl_loss.item(),
            "loss/objective": loss_vals["loss_objective"].item(),
            "loss/critic": loss_vals["loss_critic"].item(),
            "loss/entropy": loss_vals["loss_entropy"].item(),
            "loss/il_bc": il_loss.item(),
        }, step=training_info.learning_step)


def train(
        ppo_buffer_size=16,
        ppo_epochs=1,
        ppo_batch_size=4,
        validation_interval=5000,
        max_steps=600000,
        saving_interval=5000,
        m=2,
        use_wandb=False
    ):

    cfg = TrainingConfig(
        ppo_buffer_size=ppo_buffer_size,
        ppo_epochs=ppo_epochs,
        ppo_batch_size=ppo_batch_size,
        validation_interval=validation_interval,
        max_steps=max_steps,
        saving_interval=saving_interval,
        m=m,
        use_wandb=use_wandb
    )

    n_minibatches = cfg.ppo_buffer_size // cfg.ppo_batch_size
    assert n_minibatches >= cfg.m, (
        f"Insufficient mini-batches per optimize_model call to trigger IL: "
        f"ppo_buffer_size ({cfg.ppo_buffer_size}) // ppo_batch_size ({cfg.ppo_batch_size}) "
        f"= {n_minibatches} mini-batches, but m={cfg.m} RL steps are required before each IL update. "
        f"Increase ppo_buffer_size or decrease ppo_batch_size / m."
    )

    ppo_buffer = []
    mx_cum_reward = -1e9

    if cfg.use_wandb:
        api_key = input("Enter wandb API key: ").strip()
        wandb.login(key=api_key)
        wandb.init(
            project="vizdoom-inril",
            name=f"{ARCH}-{VERSION}-{VARIANT}-inril",
            config={
                "gamma": GAMMA, "lr": LR,
                "ppo_clip": PPO_CLIP, "gae_lambda": GAE_LAMBDA,
                "entropy_coef": ENTROPY_COEF, "critic_coef": CRITIC_COEF,
            }
        )
        wandb.watch(policy_net, log="all", log_freq=100)

    try:
        progress = tqdm(total=cfg.max_steps)

        for i_episode in count():
            cum_reward = 0
            game_state, _ = GameEnv.reset()
            state = torch.tensor(
                game_state['screen'].copy(), dtype=torch.float32
            ).unsqueeze(0).permute(0, 3, 1, 2)
            processed_state = base_preprocessor(state, device=device)

            for t in count():
                action = select_action(processed_state, training_info.learning_step)
                observation, reward, terminated, truncated, _ = GameEnv.step(action.logits.item())
                done = terminated or truncated
                reward_t = torch.tensor([reward], device=device)

                # Build next observation
                if terminated:
                    next_obs = torch.zeros_like(processed_state)
                else:
                    next_state_raw = torch.tensor(
                        observation['screen'].copy(), dtype=torch.float32
                    ).unsqueeze(0).permute(0, 3, 1, 2)
                    next_obs = base_preprocessor(next_state_raw, device=device)

                # Store transition in PPO buffer as TensorDict
                # MUST DETACH to avoid keeping the rollout computation graph
                current_td = action.td.detach().clone()
                current_td.set("next", TensorDict({
                    "observation": next_obs,
                    "reward": reward_t.unsqueeze(0),
                    "done": torch.tensor([done], device=device, dtype=torch.bool).unsqueeze(0),
                    "terminated": torch.tensor([terminated], device=device, dtype=torch.bool).unsqueeze(0),
                }, batch_size=[1], device=device))
                ppo_buffer.append(current_td.squeeze(0))

                processed_state = next_obs
                cum_reward += reward

                # Run PPO + IL (gradient surgery) optimisation when buffer is full
                if len(ppo_buffer) >= cfg.ppo_buffer_size:
                    optimize_model(base_preprocessor, ppo_buffer, cfg)
                    progress.n = training_info.learning_step
                    progress.set_description(f"step={training_info.learning_step} reward={cum_reward:.1f}")
                    progress.refresh()

                    # Validation
                    if training_info.learning_step % cfg.validation_interval == 0 and training_info.learning_step > 0:
                        policy_net.eval()
                        rewards_list = infer(VALIDATION_EPISODES)
                        val_mean = sum(rewards_list) / len(rewards_list)
                        training_info.eval_mean_rewards.append((training_info.learning_step, val_mean))
                        training_info.to_csv("evaluation.csv")
                        if cfg.use_wandb:
                            wandb.log({"val_mean_reward": val_mean}, step=training_info.learning_step)
                        policy_net.train()

                    # Checkpoint saving
                    if training_info.learning_step % cfg.saving_interval == 0 and training_info.learning_step > 0:
                        should_persist = cum_reward >= mx_cum_reward
                        mx_cum_reward = max(cum_reward, mx_cum_reward)
                        save_state_dict(
                            policy_net, optimizer,
                            steps=training_info.learning_step,
                            persisted=should_persist
                        )

                    if training_info.learning_step >= cfg.max_steps:
                        return

                if done:
                    if cfg.use_wandb:
                        wandb.log({"episode_reward": cum_reward}, step=training_info.learning_step)
                    break

    except Exception as e:
        raise e
    finally:
        base_preprocessor.close()
        if cfg.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    fire.Fire(train)
