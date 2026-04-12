from datetime import datetime
import logging
logging.basicConfig(
    filename='logs/infer-run-{}.log'.format(datetime.now().isoformat()),
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s',
    filemode="a"
)

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
import os
from preprocessor import base_preprocessor

isDatapointEnough = False

if(not os.path.isdir("logs")) :
    os.mkdir("logs")



def infer() :

    cum_reward = 0
    game_state , info = GameEnv.reset()
    state = base_preprocessor(torch.tensor(game_state['screen'] , dtype = torch.float32 , device = device).unsqueeze(0).permute(0,3,1,2))
    for t in count() :
        action = select_action(state,t,inference=True)
        observation , reward , terminated , truncated , _ = GameEnv.step(action.logits.item())
        reward = torch.tensor([reward],device=device)
        done = terminated or truncated

        next_state = base_preprocessor(torch.tensor(observation['screen'] , dtype = torch.float32 , device = device ).unsqueeze(0).permute(0,3,1,2))

        state = next_state

        cum_reward += reward.item()

        if done :
            logging.info("reward: {}".format(cum_reward))
            logging.info("closing preprocessor thread")
            base_preprocessor.close()
            break


if __name__ == "__main__" : 
    infer()