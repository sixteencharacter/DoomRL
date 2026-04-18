from network import create_q_network
from env import n_actions , device
from torch import optim as O
from config import *
from replay_memory import ReplayMemory
import logging
import torch
import re
import os

logger = logging.getLogger(__name__)

def _infer_resume_step_from_path(path: str) -> int:
    match = re.search(r"-(\d+)(?:steps|eps)\.pth$", os.path.basename(path))
    return int(match.group(1)) if match else 0

loaded_checkpoint = None
resume_step = 0

policy_net = create_q_network(
    arch=ARCH,
    n_actions=n_actions
)

if PRELOAD_WEIGHT is not None and os.path.isfile(PRELOAD_WEIGHT):
    loaded_checkpoint = torch.load(PRELOAD_WEIGHT,map_location=torch.device("cpu"))

    if isinstance(loaded_checkpoint, dict) and 'model' in loaded_checkpoint:
        policy_net.load_state_dict(loaded_checkpoint['model'])
    else:
        policy_net.load_state_dict(loaded_checkpoint)

    if resume_step == 0 and CHKPOINT_NUM is not None:
        resume_step = int(CHKPOINT_NUM)
    elif resume_step == 0 and PRELOAD_WEIGHT is not None:
        resume_step = _infer_resume_step_from_path(PRELOAD_WEIGHT)

    print("model weight loaded from {}".format(PRELOAD_WEIGHT))
    logger.info("model weight loaded from {}".format(PRELOAD_WEIGHT))
    logger.info("resume_step inferred as {}".format(resume_step))
else : 
    logger.info("model weight loaded from {}".format(PRELOAD_WEIGHT))
    print("No initital weight provided, fall back to random weight")

policy_net = policy_net.to(device)
target_net = create_q_network(
    arch=ARCH,
    n_actions=n_actions
).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = O.AdamW(policy_net.parameters(),lr=LR,amsgrad=True)

if isinstance(loaded_checkpoint, dict) and loaded_checkpoint.get("optimizer") is not None:
    try:
        optimizer.load_state_dict(loaded_checkpoint["optimizer"])
        logger.info("optimizer state restored from checkpoint")
    except Exception as e:
        logger.warning("failed to restore optimizer state: {}".format(e))
        
memory = ReplayMemory(MEMORY_CAP,device)