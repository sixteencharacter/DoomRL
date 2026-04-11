import torch
import torch.nn as nn
from model import memory
from config import *
from datamodel import Transition
from env import device , env as GameEnv
from model import policy_net , target_net , optimizer
from itertools import count
from utils import select_action , save_state_dict
from tqdm import tqdm
import logging
import os
from datetime import datetime

isDatapointEnough = False

if(not os.path.isdir("logs")) :
    os.mkdir("logs")

logging.basicConfig(
    filename='logs/run-{}.log'.format(datetime.now().isoformat()),
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s',
    filemode="a"
)

def optimize_model() :
    if len(memory) < BATCH_SIZE :
        logging.info("Skipped Optimization due to insufficient data points")
        return
    global isDatapointEnough
    isDatapointEnough = True
    logging.info("Optimizing model...")
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s : s is not None,batch.next_state)),device=device,dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1,action_batch)

    next_state_values = torch.zeros(BATCH_SIZE , device=device)
    
    with torch.no_grad() :
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    expected_state_action_values = ( next_state_values * GAMMA ) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values , expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(),100)
    optimizer.step()


def train(num_episodes) :

    iteration = tqdm(range(num_episodes),total=num_episodes)
    mx_cum_reward = -1e9

    for i_episodes in iteration :
        cum_reward = 0
        game_state , info = GameEnv.reset()
        state = torch.tensor(game_state['screen'] , dtype = torch.float32 , device = device).unsqueeze(0).permute(0,3,1,2)
        for t in count() :
            action = select_action(state,t)
            observation , reward , terminated , truncated , _ = GameEnv.step(action.item())
            reward = torch.tensor([reward],device=device)
            done = terminated or truncated

            if terminated : 
                next_state = None
            else :
                next_state = torch.tensor(observation['screen'] , dtype = torch.float32 , device = device ).unsqueeze(0).permute(0,3,1,2)
            
            memory.push(state , action , next_state , reward)

            cum_reward += reward.item()

            iteration.set_description(f"Episode reward: {cum_reward}")

            state = next_state

            optimize_model()

            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            
            for key in policy_net.state_dict() : 
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1-TAU)
                target_net.load_state_dict(target_net_state_dict)

            if done :
                logging.info("Episode {} reward: {}".format(i_episodes,cum_reward))
                if isDatapointEnough and cum_reward >= mx_cum_reward :
                    mx_cum_reward = max(cum_reward,mx_cum_reward)
                    save_state_dict(policy_net,optimizer,episode=i_episodes)
                break


if __name__ == "__main__" : 
    logging.info("Running with BATCH_SIZE",BATCH_SIZE)
    train(NUM_EPISODE)