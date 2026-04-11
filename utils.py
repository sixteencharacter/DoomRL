import torch
import math
import random
from config import *
from datamodel import ActionRes
from model import policy_net
from env import env as GameEnv , device
import logging
from icecream import ic
from copy import deepcopy
import os


logger = logging.getLogger(__name__)

def select_action(state,steps) :
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * steps / EPS_DECAY)
    steps += 1

    if sample > eps_threshold :
        with torch.no_grad() :
            return policy_net(state).max(1).indices.view(1,1)
    else :
        return torch.tensor([[GameEnv.action_space.sample()]],device=device , dtype=torch.long)

    return ActionRes(steps,)

def save_state_dict(model,optimizer,episode = None) :

    if(not os.path.isdir("weights")) :
        os.mkdir("weights")

    logger.info("Save state dict to {}".format(f"weights/{ARCH}-{VERSION}.pth"))

    localModel = deepcopy(model).to("cpu")
    localOptimizer = deepcopy(optimizer)

    torch.save({
        'model' : localModel.state_dict(),
        'optimizer' : localOptimizer.state_dict()
    },f"weights/{ARCH}-{VERSION}{"-{}eps".format(episode) if episode is not None else ""}.pth")