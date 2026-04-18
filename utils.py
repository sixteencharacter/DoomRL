import torch
import math
import random
from config import *
from datamodel import ActionRes
from model import policy_net
from env import env as GameEnv , device
import logging
import os
from collections import deque

logger = logging.getLogger(__name__)

weights_tracker = deque(maxlen=5)

def select_action(state,steps,inference=False) :

    logits = None

    if inference :
        with torch.no_grad() :
            q_value = policy_net(state)
            # ic(q_value)
            logits = q_value.max(1).indices.view(1,1)
    else :
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-steps / EPS_DECAY)
        if sample > eps_threshold :
            # ic("exploit")
            with torch.no_grad() :
                logits = policy_net(state).max(1).indices.view(1,1)
        else :
            # ic("explore")
            logits = torch.tensor([[GameEnv.action_space.sample()]],device=device , dtype=torch.long)

    # ic(logits)
    return ActionRes(steps,logits)

def save_state_dict(model,optimizer,steps = None,persisted = False) :

    if(not os.path.isdir("weights")) :
        os.mkdir("weights")

    logger.info("Save state dict to {}".format(f"weights/{ARCH}-{VERSION}{"-{}eps".format(steps + CHKPOINT_NUM) if steps is not None else ""}.pth"))

    model_state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    optimizer_state_dict = optimizer.state_dict()

    if not persisted :
        if len(weights_tracker) == weights_tracker.maxlen :
            old_path = weights_tracker[0]
            if os.path.exists(old_path) :
                os.remove(old_path)
        
        weights_tracker.append(save_path)

    torch.save({
        'model' : model_state_dict,
        'optimizer' : optimizer_state_dict
    },save_path)