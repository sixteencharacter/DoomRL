from datetime import datetime
import logging
import os
import torch
import gymnasium as gym
import vizdoom.gymnasium_wrapper
from config import *
from env import device
from itertools import count
from utils import select_action
from model import policy_net
from tqdm import tqdm
from preprocessor import base_preprocessor

if(not os.path.isdir("logs")) :
    os.mkdir("logs")

logging.basicConfig(
    filename='logs/infer-run-{}.log'.format(datetime.now().strftime("%Y%m%d-%H%M%S")),
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s',
    filemode="a"
)



def infer(n_episodes) :

    policy_net.eval()
    rewards_list = []
    eval_env = gym.make(SCENARIO_NAME,screen_resolution=RESOLUTION,render_mode=RENDER_MODE,frame_skip=FRAME_SKIP)
    action_hist = [0 for _ in range(eval_env.action_space.n)]

    try:
        with torch.inference_mode():
            for i in tqdm(range(n_episodes)) :

                cum_reward = 0
                game_state , info = eval_env.reset()
                state = base_preprocessor(torch.tensor(game_state['screen'].copy() , dtype = torch.float32).unsqueeze(0).permute(0,3,1,2),device=device)
                for t in count() :
                    action = select_action(state,t,inference=True)
                    action_id = action.logits.item()
                    action_hist[action_id] += 1
                    observation , reward , terminated , truncated , _ = eval_env.step(action_id)
                    reward = torch.tensor([reward],device=device)
                    done = terminated or truncated

                    raw = torch.tensor(observation['screen'].copy() , dtype = torch.float32)  # CPU first, avoid MPS lazy eval
                    next_state = base_preprocessor(raw.unsqueeze(0).permute(0,3,1,2), device=device)

                    state = next_state

                    cum_reward += reward.item()

                    if done :
                        logging.info("reward: {}".format(cum_reward))
                        rewards_list.append(cum_reward)
                        break
    finally:
        eval_env.close()

    logging.info("action histogram: {}".format(action_hist))

    return rewards_list


if __name__ == "__main__" : 
    rewards_list = infer(1000)
    logging.info("closing preprocessor thread")
    print("1000 eps averaged reward:",sum(rewards_list) / len(rewards_list))
    base_preprocessor.close()