import logging
from datetime import datetime
import torch
import torch.nn as nn
from model import memory
from config import *
from datamodel import Transition, training_info, StarformerContext, WindowSampledBatch
from env import device , env as GameEnv
from model import policy_net , target_net , optimizer, scheduler
from model import policy_net , target_net , optimizer
if METHOD == "PPO":
    from model import loss_module, advantage_module
from itertools import count
from utils import select_action , save_state_dict
from tqdm import tqdm
from preprocessor import base_preprocessor
from icecream import ic
from inference import infer
import wandb
import os
from tensordict import TensorDict


if(not os.path.isdir("logs")) :
    os.mkdir("logs")

logging.basicConfig(
    filename='logs/run-{}.log'.format(datetime.now().isoformat()),
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s',
    filemode="a"
)

isDatapointEnough = False
ppo_buffer = []


def _optimize_starformer():
    """STARFORMER branch: sample K-windows from WindowPrioritizedReplayMemory,
    forward through transformer, compute TD loss at last sequence position.
    Returns (loss_item, did_step) or (None, False) if skipped."""
    sampled = memory.sample(BATCH_SIZE, base_preprocessor)
    if sampled is None:
        # gate: not enough valid anchors yet
        wandb.log({"optimize_skipped": 1}, step=training_info.learning_step)
        return None, False
    assert isinstance(sampled, WindowSampledBatch)
    training_info.learning_step += 1

    states_seq = sampled.states_seq                    # (B, K, 3, H, W)
    actions_seq = sampled.actions_seq                  # (B, K)
    rtgs_seq = sampled.rtgs_seq                        # (B, K)
    last_actions = sampled.last_actions                # (B, 1)
    last_rewards = sampled.last_rewards                # (B,)
    next_state_last = sampled.next_state_last          # (B, 3, H, W)
    terminal_mask = sampled.terminal_mask              # (B,)
    weights = sampled.weights                          # (B,)
    indices = sampled.indices                          # (B,)

    B, K = actions_seq.shape

    # Q at anchor (last position)
    q_seq = policy_net(states_seq, actions_seq, rtgs_seq)              # (B,K,n_actions)
    state_action_values = q_seq[:, -1, :].gather(1, last_actions).squeeze(1)  # (B,)

    # Target: roll window forward by one step, replace last with next_state_last.
    next_state_values = torch.zeros(B, device=device)
    with torch.no_grad():
        non_term = ~terminal_mask
        if non_term.any():
            target_states = torch.cat(
                [states_seq[:, 1:], next_state_last.unsqueeze(1)], dim=1
            )                                                            # (B,K,3,H,W)
            # actions/rtgs for target window: shift forward, last position uses zero placeholder
            target_actions = torch.zeros_like(actions_seq)
            target_actions[:, :-1] = actions_seq[:, 1:]
            target_rtgs = torch.zeros_like(rtgs_seq)
            target_rtgs[:, :-1] = rtgs_seq[:, 1:]
            target_rtgs[:, -1] = rtgs_seq[:, -1]  # carry last RTG forward
            # DDQN-style: action from policy_net, value from target_net
            policy_q_next = policy_net(target_states, target_actions, target_rtgs)[:, -1, :]
            next_actions = policy_q_next.argmax(1, keepdim=True)
            target_q_next = target_net(target_states, target_actions, target_rtgs)[:, -1, :]
            chosen = target_q_next.gather(1, next_actions).squeeze(1)
            next_state_values = chosen * non_term.float()

    expected_state_action_values = (next_state_values * GAMMA) + last_rewards

    criterion = nn.SmoothL1Loss(reduction='none')
    elementwise_loss = criterion(state_action_values, expected_state_action_values)
    loss = (weights * elementwise_loss).mean()

    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    td_errors = elementwise_loss.detach()
    memory.update_priorities(indices, td_errors)

    log_payload = {
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
    }
    wandb.log(log_payload, step=training_info.learning_step)
    return loss.item(), True


def optimize_model(preprocessor) :
    global isDatapointEnough
    if METHOD == "PPO":
        if len(ppo_buffer) < PPO_BUFFER_SIZE:
            return False
        isDatapointEnough = True
        training_info.learning_step += 1
        
        # Stack all TensorDicts in the buffer
        data = torch.stack(ppo_buffer, dim=0).to(device)
        ppo_buffer.clear()
        
        # MUST set to eval() because GAE uses vmap, which fails with BatchNorm in train mode
        policy_net.eval()
        with torch.no_grad():
            advantage_module(data)
        policy_net.train()
        
        # Advantage Normalization (Crucial for Actor stability)
        adv = data["advantage"]
        data["advantage"] = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        # PPO Optimization loop
        for epoch in range(PPO_EPOCHS):
            # Shuffle and batch
            indices = torch.randperm(data.shape[0])
            for i in range(0, data.shape[0], PPO_BATCH_SIZE):
                batch_indices = indices[i : i + PPO_BATCH_SIZE]
                batch_data = data[batch_indices]
                
                loss_vals = loss_module(batch_data)
                loss_value = loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"]
                
                optimizer.zero_grad()
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()
        
        # Clear cache after optimization
        torch.cuda.empty_cache()
        
        wandb.log({
            "loss/total": loss_value.item(),
            "loss/objective": loss_vals["loss_objective"].item(),
            "loss/critic": loss_vals["loss_critic"].item(),
            "loss/entropy": loss_vals["loss_entropy"].item(),
        }, step=training_info.learning_step)
        return True

    if len(memory) < BATCH_SIZE :
        logging.info("Skipped Optimization due to insufficient data points")
        return
    global isDatapointEnough

    if METHOD == 'STARFORMER':
        loss_item, did_step = _optimize_starformer()
        if did_step:
            isDatapointEnough = True
        return False
    isDatapointEnough = True
    training_info.learning_step += 1
    sampled = memory.sample(BATCH_SIZE,preprocessor)
    batch = Transition(*zip(*sampled.transitions))
    weights = sampled.weights

    non_final_mask = torch.tensor(tuple(map(lambda s : s is not None,batch.next_state)),device=device,dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1,action_batch)

    next_state_values = torch.zeros(BATCH_SIZE , device=device)

    with torch.no_grad() :
        if non_final_next_states.size(0) > 0:
            if METHOD == "DDQN":
                next_actions = policy_net(non_final_next_states).argmax(1, keepdim=True)
                next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, next_actions).squeeze(1)
            else:
                next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    expected_state_action_values = ( next_state_values * GAMMA ) + reward_batch

    criterion = nn.SmoothL1Loss(reduction='none')
    elementwise_loss = criterion(state_action_values , expected_state_action_values.unsqueeze(1)).squeeze(1)
    assert elementwise_loss.shape == weights.shape, \
        f"shape mismatch: loss {elementwise_loss.shape} vs weights {weights.shape}"
    loss = (weights * elementwise_loss).mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(),100)
    optimizer.step()

    # priority update for PER (no-op for Uniform)
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
    """Discounted return-to-go for one episode: g_t = sum_{u>=t} gamma^{u-t} r_u."""
    rtgs = [0.0] * len(rewards)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = float(rewards[t]) + GAMMA * running
        rtgs[t] = running
    return rtgs


def train(num_episodes,preprocessor) :

    try :

        iteration_mode = "episode" if NUM_EPISODE is not None else "steps"

        logging.info(f"Using iteration mode of {iteration_mode}")

        total_iteration = NUM_EPISODE if NUM_EPISODE is not None else MAX_STEPS

        iteration = tqdm(range(total_iteration),total=total_iteration)

        training_end = False

        mx_cum_reward = -1e9

        episode_id_counter = 0

        for i_episodes in count() :
            cum_reward = 0
            game_state , info = GameEnv.reset()
            state = torch.tensor(game_state['screen'].copy() , dtype = torch.float32).unsqueeze(0).permute(0,3,1,2)
            processed_state = preprocessor(state,device=device)

            current_episode_id = episode_id_counter
            episode_id_counter += 1

            ctx = None
            episode_rewards = []
            if METHOD == 'STARFORMER':
                ctx = StarformerContext(K=STARFORMER_K, rtg_target=STARFORMER_RTG_TARGET)

            for t in count() :
                action = select_action(processed_state,training_info.learning_step,ctx=ctx)
                observation , reward , terminated , truncated , _ = GameEnv.step(action.logits.item())
                if iteration_mode == "steps" :
                    iteration.n = training_info.learning_step
                    iteration.refresh()
                    iteration.set_description(f"Step Reward {reward}")
                reward_scalar = float(reward)
                episode_rewards.append(reward_scalar)
                reward = torch.tensor([reward],device=device)
                done = terminated or truncated

                if terminated :
                    next_state = None
                else :
                    next_state = torch.tensor(observation['screen'].copy() , dtype = torch.float32).unsqueeze(0).permute(0,3,1,2)

                # push transition (with episode_id for STARFORMER; harmless for others)
                memory.push(state, action.logits, next_state, reward, None, current_episode_id)
                if METHOD == "PPO":
                    # For PPO, we need to store the transition in a TensorDict
                    # action.td already contains observation, logits, action, log_prob
                    # MUST DETACH to avoid memory leak (keeping the computation graph of the rollout)
                    current_td = action.td.detach().clone()
                    
                    with torch.no_grad():
                        next_obs = preprocessor(next_state, device=device) if next_state is not None else torch.zeros_like(processed_state)
                    
                    current_td.set("next", TensorDict({
                        "observation": next_obs,
                        "reward": reward.unsqueeze(0),
                        "done": torch.tensor([done], device=device, dtype=torch.bool).unsqueeze(0),
                        "terminated": torch.tensor([terminated], device=device, dtype=torch.bool).unsqueeze(0),
                    }, batch_size=[1], device=device))
                    ppo_buffer.append(current_td.squeeze(0))
                    
                    # Optimization: Reuse next_obs for the next step's processed_state
                    state = next_state
                    processed_state = next_obs
                else:
                    memory.push(state , action.logits , next_state , reward)
                    state = next_state
                    processed_state = preprocessor(state, device=device)

                cum_reward += reward.item()

                if iteration_mode == "episode" :
                    iteration.set_description(f"Episode reward: {cum_reward}")

                did_optimize = optimize_model(preprocessor)

                if METHOD != "PPO":
                    target_net_state_dict = target_net.state_dict()
                    policy_net_state_dict = policy_net.state_dict()
                    
                    for key in policy_net.state_dict() : 
                        target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1-TAU)
                    
                    target_net.load_state_dict(target_net_state_dict)

                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()

                for key in policy_net.state_dict() :
                    target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1-TAU)

                target_net.load_state_dict(target_net_state_dict)

                if iteration_mode == "steps" :
                if did_optimize and iteration_mode == "steps" :
                    # validation
                    if training_info.learning_step % VALIDATION_INTERVAL == 0 and training_info.learning_step != 0:
                        print("Validation at step {}".format(training_info.learning_step))
                        policy_net.eval()
                        rewards_list = infer(VALIDATION_EPISODES) # testing with 1000 episodes
                        val_mean = sum(rewards_list)/len(rewards_list)
                        training_info.eval_mean_rewards.append((training_info.learning_step, val_mean))
                        training_info.to_csv("evaluation.csv")
                        wandb.log({"val_mean_reward": val_mean}, step=training_info.learning_step)
                        policy_net.train()
                        game_state , info = GameEnv.reset()
                        state = torch.tensor(game_state['screen'] , dtype = torch.float32).unsqueeze(0).permute(0,3,1,2)
                        processed_state = preprocessor(state,device=device)
                        # validation re-entered the env — start a fresh logical episode
                        if METHOD == 'STARFORMER':
                            ctx = StarformerContext(K=STARFORMER_K, rtg_target=STARFORMER_RTG_TARGET)
                            current_episode_id = episode_id_counter
                            episode_id_counter += 1
                            episode_rewards = []
                            cum_reward = 0

                if did_optimize and training_info.learning_step % SAVING_INTERVAL == 0 and training_info.learning_step != 0:
                    if isDatapointEnough :
                        shouldPersist = (cum_reward >= mx_cum_reward)
                        mx_cum_reward = max(cum_reward,mx_cum_reward)
                        if(shouldPersist) :
                            logging.info("Saving weight with persisting = {}".format(shouldPersist))
                            save_state_dict(policy_net,optimizer,steps=training_info.learning_step,persisted=shouldPersist)
                        elif training_info.learning_step % SAVING_INTERVAL == 0 :
                            logging.info("Saving weight with persisting = {} (interval saving)".format(shouldPersist))
                            save_state_dict(policy_net,optimizer,steps=training_info.learning_step,persisted=False)

                if (iteration_mode == "steps" and (training_info.learning_step >= MAX_STEPS - 1)) or (iteration_mode == "episode" and (i_episodes >= NUM_EPISODE - 1)) :
                    training_end = True
                    break

                if done :
                    # backfill RTG for STARFORMER — must be done before next sample sees None
                    if METHOD == 'STARFORMER' and hasattr(memory, 'backfill_rtg'):
                        rtgs = _compute_episode_rtg(episode_rewards)
                        memory.backfill_rtg(current_episode_id, rtgs)
                    wandb.log({"episode_reward": cum_reward}, step=training_info.learning_step)
                    break

            if training_end :
                break
    except Exception as e:
        raise e
    finally :
        logging.info("closing preprocessor thread")
        base_preprocessor.close()
        wandb.finish()


if __name__ == "__main__" :
    api_key = input("Enter wandb API key: ").strip()
    wandb.login(key=api_key)
    wandb.init(
        project="vizdoom-dqn",
        name=f"{ARCH}-{VERSION}-{VARIANT}-{SAMPLING_METHOD}",
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
        }
    )
    wandb.watch(policy_net, log="all", log_freq=100)
    logging.info("Running with {}".format(BATCH_SIZE))
    train(NUM_EPISODE,base_preprocessor)
