from datetime import datetime
import logging
import os
import torch
from config import *
from env import device , env as GameEnv
from itertools import count
from utils import select_action
from model import policy_net
from datamodel import StarformerContext
from tqdm import tqdm
import os
from preprocessor import base_preprocessor
from icecream import ic

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
    action_hist = [0 for _ in range(GameEnv.action_space.n)]

    with torch.inference_mode():
        for i in tqdm(range(n_episodes)) :

            cum_reward = 0
            game_state , info = GameEnv.reset()
            state = base_preprocessor(torch.tensor(game_state['screen'].copy() , dtype = torch.float32).unsqueeze(0).permute(0,3,1,2),device=device)

            ctx = None
            if METHOD == 'STARFORMER':
                ctx = StarformerContext(K=STARFORMER_K, rtg_target=STARFORMER_RTG_TARGET)

            for t in count() :
                action = select_action(state.to(device),t,ctx=ctx,inference=True)
                action_id = action.logits.item()
                action_hist[action_id] += 1
                observation , reward , terminated , truncated , _ = GameEnv.step(action_id)
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

    logging.info("action histogram: {}".format(action_hist))

    return rewards_list


if __name__ == "__main__" : 
    rewards_list = infer(1000)
    logging.info("closing preprocessor thread")
    print("1000 eps averaged reward:",sum(rewards_list) / len(rewards_list))
    base_preprocessor.close()