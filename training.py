import logging
from datetime import datetime
import torch
import torch.nn as nn
from model import memory
from config import *
from datamodel import Transition , training_info
from env import device , env as GameEnv
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


def train(num_episodes,preprocessor) :

    try :

        iteration_mode = "episode" if NUM_EPISODE is not None else "steps"

        logging.info(f"Using iteration mode of {iteration_mode}")

        total_iteration = NUM_EPISODE if NUM_EPISODE is not None else MAX_STEPS

        iteration = tqdm(range(total_iteration),total=total_iteration)
        
        training_end = False

        mx_cum_reward = -1e9

        for i_episodes in count() :
            cum_reward = 0
            game_state , info = GameEnv.reset()
            state = torch.tensor(game_state['screen'].copy() , dtype = torch.float32).unsqueeze(0).permute(0,3,1,2)
            processed_state = preprocessor(state,device=device)
            for t in count() :
                action = select_action(processed_state,training_info.learning_step)
                observation , reward , terminated , truncated , _ = GameEnv.step(action.logits.item())
                if iteration_mode == "steps" :
                    iteration.n = training_info.learning_step
                    iteration.refresh()
                    iteration.set_description(f"Step Reward {reward}")
                reward = torch.tensor([reward],device=device)
                done = terminated or truncated

                if terminated : 
                    next_state = None
                else :
                    next_state = torch.tensor(observation['screen'].copy() , dtype = torch.float32).unsqueeze(0).permute(0,3,1,2)

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
            "sampling_method": SAMPLING_METHOD,
            "alpha": ALPHA if SAMPLING_METHOD == "PER" else None,
            "beta_start": BETA_START if SAMPLING_METHOD == "PER" else None,
            "beta_end": BETA_END if SAMPLING_METHOD == "PER" else None,
        }
    )
    wandb.watch(policy_net, log="all", log_freq=100)
    logging.info("Running with {}".format(BATCH_SIZE))
    train(NUM_EPISODE,base_preprocessor)