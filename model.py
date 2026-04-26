from network import create_q_network
from env import n_actions , device
from torch import optim as O
from config import *
from replay_memory import create_replay_memory
import logging
import torch

logger = logging.getLogger(__name__)

policy_net = create_q_network(
    arch=ARCH,
    n_actions=n_actions
)

if PRELOAD_WEIGHT is not None :
    policy_net.load_state_dict(torch.load(PRELOAD_WEIGHT,map_location=torch.device("cpu"))['model'])
    print("model weight loaded from {}".format(PRELOAD_WEIGHT))
    logger.info("model weight loaded from {}".format(PRELOAD_WEIGHT))
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
memory = create_replay_memory(SAMPLING_METHOD, MEMORY_CAP, device)