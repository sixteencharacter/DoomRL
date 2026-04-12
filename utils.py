import torch
import math
import random
from config import *
from datamodel import ActionRes
from model import policy_net
from env import env as GameEnv , device
import logging
from copy import deepcopy
import os
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)

weights_tracker = deque(maxlen=5)

def select_action(state,steps,inference=False) :

    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * steps / EPS_DECAY)
    logits = None
    if sample > eps_threshold and not inference :
        with torch.no_grad() :
            logits = policy_net(state).max(1).indices.view(1,1)
    else :
        logits = torch.tensor([[GameEnv.action_space.sample()]],device=device , dtype=torch.long)

    return ActionRes(steps,logits)

def save_state_dict(model,optimizer,episode = None,persisted = False) :

    if(not os.path.isdir("weights")) :
        os.mkdir("weights")

    logger.info("Save state dict to {}".format(f"weights/{ARCH}-{VERSION}{"-{}eps".format(episode + CHKPOINT_NUM) if episode is not None else ""}.pth"))

    localModel = deepcopy(model).to("cpu")
    localOptimizer = deepcopy(optimizer)

    if not persisted :
        if len(weights_tracker) == weights_tracker.maxlen :
            old_path = weights_tracker[0]
            if os.path.exists(old_path) :
                os.remove(old_path)
        
        weights_tracker.append(f"weights/{ARCH}-{VERSION}{"-{}eps".format(episode + CHKPOINT_NUM) if episode is not None else ""}.pth")

    torch.save({
        'model' : localModel.state_dict(),
        'optimizer' : localOptimizer.state_dict()
    },f"weights/{ARCH}-{VERSION}{"-{}eps".format(episode + CHKPOINT_NUM) if episode is not None else ""}.pth")