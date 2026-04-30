from datamodel import training_info
import torch
import torch.nn.functional as F 
import numpy as np
from env import device
from model import policy_net, advantage_module, loss_module, optimizer
from itertools import count
from env import device, env as GameEnv
from utils import select_action, save_state_dict
from tqdm import tqdm
from tensordict import TensorDict
from preprocessor import base_preprocessor

# ── Hyper-parameters ──────────────────────────────────────────────────────────
PPO_BUFFER_SIZE      = 32
PPO_EPOCHS           = 4
PPO_BATCH_SIZE       = 8
VALIDATION_INTERVAL  = 5000
MAX_STEPS            = 600_000
SAVING_INTERVAL      = 5000

# IN-RIL ratio controls ────────────────────────────────────────────────────────
#   IL_BATCH_RATIO  – how many IL samples per RL sample  (e.g. 0.5 → half as many)
#   IL_WEIGHT       – gradient blending weight for the IL gradient
#   RL_WEIGHT       – gradient blending weight for the RL gradient
IL_BATCH_RATIO  = 0.5          # replaces RL2IL_RATIO
IL_WEIGHT       = 0.5
RL_WEIGHT       = 1.0 - IL_WEIGHT

ppo_buffer: list = []

# ── Demo-buffer utilities ─────────────────────────────────────────────────────

def load_demo_npy(scene_path: str, action_path: str,preprocessor) -> list[TensorDict]:
    """
    Load ground-truth demonstrations from two .npy files.

    scene_path  – shape (N, H, W, C)  uint8 / float32 frames
    action_path – shape (N,)           int64 / int32  discrete actions

    Returns a list of TensorDicts ready to stack into a demo buffer.
    """
    scenes  = np.load(scene_path)                           # (N, H, W, C)
    actions = np.load(action_path)                          # (N,)

    assert len(scenes) == len(actions), \
        f"Scene/action length mismatch: {len(scenes)} vs {len(actions)}"

    demos = []
    for obs, act in zip(scenes, actions):
        obs_t = (
            torch.tensor(obs, dtype=torch.float32)
            .unsqueeze(0)           # (1, H, W, C)
            .permute(0, 3, 1, 2)   # (1, C, H, W)
        )
        act_t = torch.tensor([act], dtype=torch.long)       # (1,)

        td = TensorDict(
            {
                "observation":    preprocessor(obs_t,device=device),
                "expert_action":  act_t,
            },
            batch_size=[1],
        )
        demos.append(td.squeeze(0))

    return demos

def _sample_buffer(buffer: list[TensorDict], batch_size: int):
    """Return a stacked TensorDict batch, or None if buffer is empty."""
    if batch_size <= 0 or len(buffer) == 0:
        return None
    batch_size = min(batch_size, len(buffer))
    idx = torch.randperm(len(buffer))[:batch_size].tolist()
    return torch.stack([buffer[i] for i in idx], dim=0).to(device)


# ── Gradient surgery (fixed) ──────────────────────────────────────────────────

def _gradient_surgery(
    grads_rl: tuple,
    grads_il: tuple,
    params:   list,
) -> tuple[list, list]:
    """
    Project conflicting gradients so they do not point in opposing directions.
    """
    proj_rl, proj_il = [], []

    for p, g_r, g_i in zip(params, grads_rl, grads_il):
        g_r = torch.zeros_like(p) if g_r is None else g_r.detach()
        g_i = torch.zeros_like(p) if g_i is None else g_i.detach()

        dot = torch.dot(g_r.flatten(), g_i.flatten())

        if dot < 0:
            g_r_proj = g_r - (dot / (g_i.flatten().pow(2).sum() + 1e-12)) * g_i
            g_i_proj = g_i - (dot / (g_r.flatten().pow(2).sum() + 1e-12)) * g_r
        else:
            g_r_proj, g_i_proj = g_r, g_i

        proj_rl.append(g_r_proj)
        proj_il.append(g_i_proj)

    return proj_rl, proj_il


# ── Core optimisation step ────────────────────────────────────────────────────

def optimize_model(preprocessor, scene_path: str, action_path: str) -> bool:
    """
    Run one IN-RIL optimisation step.

    IL interleaving is controlled by IL_BATCH_RATIO:
      il_batch_size = max(1, round(PPO_BATCH_SIZE * IL_BATCH_RATIO))
    Set IL_BATCH_RATIO = 0 to disable IL entirely.
    """
    if len(ppo_buffer) < PPO_BUFFER_SIZE:
        return False

    training_info.learning_step += 1
    data = torch.stack(ppo_buffer, dim=0).to(device)
    ppo_buffer.clear()

    # ── Advantage estimation ──────────────────────────────────────────────────
    policy_net.eval()
    with torch.no_grad():
        advantage_module(data)
    policy_net.train()

    adv = data["advantage"]
    data["advantage"] = (adv - adv.mean()) / (adv.std() + 1e-8)

    # ── Compute IL batch size once (fix #4: must be int) ─────────────────────
    il_batch_size = max(1, round(PPO_BATCH_SIZE * IL_BATCH_RATIO)) if IL_BATCH_RATIO > 0 else 0

    # ── PPO epochs ───────────────────────────────────────────────────────────
    for _epoch in range(PPO_EPOCHS):
        indices = torch.randperm(data.shape[0], device=device)

        for i in range(0, data.shape[0], PPO_BATCH_SIZE):
            batch_indices = indices[i : i + PPO_BATCH_SIZE]
            rl_batch = data[batch_indices]

            # ── RL loss ───────────────────────────────────────────────────────
            loss_vals = loss_module(rl_batch)
            loss_rl = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            # ── IL loss ───────────────────────────────────────────────────────
            loss_il    = torch.tensor(0.0, device=device)
            il_present = False

            if il_batch_size > 0:
                demo_buffer = load_demo_npy(
                    scene_path, action_path, preprocessor
                )
                demo_batch = _sample_buffer(demo_buffer, il_batch_size)

                if demo_batch is not None and "expert_action" in demo_batch.keys():
                    il_present  = True
                    demo_obs    = demo_batch["observation"]            # already preprocessed
                    targets     = demo_batch["expert_action"].view(-1).long()

                    # fix #5: stay in train() mode; never flip to eval mid-loop
                    logits = policy_net(demo_obs)
                    n_actions = logits.shape[-1]
                    loss_il = F.cross_entropy(logits.view(-1, n_actions), targets)

            # ── Gradient update ───────────────────────────────────────────────
            params = [p for p in policy_net.parameters() if p.requires_grad]

            if not il_present:
                optimizer.zero_grad()
                loss_rl.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()
                continue

            grads_rl = torch.autograd.grad(
                loss_rl, params, retain_graph=True, allow_unused=True
            )
            grads_il = torch.autograd.grad(
                loss_il, params, allow_unused=True   # no need to retain after this
            )

            proj_rl, proj_il = _gradient_surgery(grads_rl, grads_il, params)

            optimizer.zero_grad()
            for p, g_r, g_i in zip(params, proj_rl, proj_il):
                p.grad = RL_WEIGHT * g_r + IL_WEIGHT * g_i

            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
            optimizer.step()

    torch.cuda.empty_cache()
    return True


# ── Training loop ─────────────────────────────────────────────────────────────

def train(preprocessor, total_iteration, scene_path: str, action_path: str):

    mx_cum_reward  = float("-inf")  
    training_end   = False         

    iteration = tqdm(range(total_iteration), total=total_iteration)

    for _episode in count():
        cum_reward = 0
        game_state, _info = GameEnv.reset()
        state = (
            torch.tensor(game_state["screen"].copy(), dtype=torch.float32)
            .unsqueeze(0)
            .permute(0, 3, 1, 2)
        )
        processed_state = preprocessor(state, device=device)

        for _t in count():
            action = select_action(processed_state, training_info.learning_step)

            observation, reward, terminated, truncated, _ = GameEnv.step(
                action.logits.item()
            )

            iteration.n = training_info.learning_step
            iteration.refresh()
            iteration.set_description(f"Step Reward {reward}")

            reward = torch.tensor([reward], device=device)
            done   = terminated or truncated

            next_state = (
                None
                if terminated
                else torch.tensor(
                    observation["screen"].copy(), dtype=torch.float32
                )
                .unsqueeze(0)
                .permute(0, 3, 1, 2)
            )

            current_td = action.td.detach().clone()

            with torch.no_grad():
                next_obs = (
                    preprocessor(next_state, device=device)
                    if next_state is not None
                    else torch.zeros_like(processed_state)
                )

            current_td.set(
                "next",
                TensorDict(
                    {
                        "observation": next_obs,
                        "reward":      reward.unsqueeze(0),
                        "done":        torch.tensor([done],       device=device, dtype=torch.bool).unsqueeze(0),
                        "terminated":  torch.tensor([terminated], device=device, dtype=torch.bool).unsqueeze(0),
                    },
                    batch_size=[1],
                    device=device,
                ),
            )
            ppo_buffer.append(current_td.squeeze(0))

            processed_state = next_obs
            cum_reward      += reward.item()

            did_optimize = optimize_model(preprocessor, scene_path, action_path)

            if did_optimize:
                step = training_info.learning_step

                if step % VALIDATION_INTERVAL == 0 and step != 0:
                    print(f"Validation at step {step}")
                    policy_net.eval()
                    training_info.to_csv("evaluation.csv")
                    policy_net.train()
                    game_state, _ = GameEnv.reset()
                    state = (
                        torch.tensor(game_state["screen"], dtype=torch.float32)
                        .unsqueeze(0)
                        .permute(0, 3, 1, 2)
                    )
                    processed_state = preprocessor(state, device=device)

                if step % SAVING_INTERVAL == 0 and step != 0:
                    should_persist  = cum_reward >= mx_cum_reward
                    mx_cum_reward   = max(cum_reward, mx_cum_reward)
                    save_state_dict(
                        policy_net,
                        optimizer,
                        steps=step,
                        persisted=should_persist,
                    )

                if step >= MAX_STEPS - 1:
                    training_end = True

            if training_end or done:
                break

        if training_end:
            break

if __name__ == "__main__" :
    train(
        base_preprocessor,
        MAX_STEPS,
        "dataset/maze/human_states.npy",
        "dataset/maze/human_actions.npy"
    )