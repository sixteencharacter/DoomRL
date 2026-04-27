from network import create_q_network
from env import n_actions , device
from torch import optim as O
from config import *
from replay_memory import create_replay_memory
import logging
import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, ValueOperator, OneHotCategorical
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

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

class ActorWrapper(nn.Module):
    def __init__(self, policy_net):
        super().__init__()
        self.policy_net = policy_net
    def forward(self, x):
        logits, _ = self.policy_net(x)
        return logits

class ValueWrapper(nn.Module):
    def __init__(self, policy_net):
        super().__init__()
        self.policy_net = policy_net
    def forward(self, x):
        _, value = self.policy_net(x)
        return value

if METHOD == "PPO":
    # For PPO, we use the same policy_net but wrapped in torchrl modules
    actor_module = TensorDictModule(
        module=ActorWrapper(policy_net),
        in_keys=["observation"],
        out_keys=["logits"],
    )
    actor = ProbabilisticActor(
        module=actor_module,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=OneHotCategorical,
        return_log_prob=True,
    )

    value_module = ValueOperator(
        module=ValueWrapper(policy_net),
        in_keys=["observation"],
    )

    loss_module = ClipPPOLoss(
        actor=actor,
        critic=value_module,
        clip_epsilon=PPO_CLIP,
        entropy_bonus=True,
        entropy_coeff=ENTROPY_COEF,
        critic_coeff=CRITIC_COEF,
        loss_critic_type="smooth_l1",
    )

    advantage_module = GAE(
        gamma=GAMMA, lmbda=GAE_LAMBDA, value_network=value_module, average_gae=True
    )

    target_net = None # PPO doesn't use target_net in the same way as DQN
    optimizer = O.AdamW(policy_net.parameters(), lr=LR)
    memory = None # We'll handle PPO buffer in training.py
else:
    target_net = create_q_network(
        arch=ARCH,
        n_actions=n_actions
    ).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = O.AdamW(policy_net.parameters(),lr=LR,amsgrad=True)
    memory = create_replay_memory(SAMPLING_METHOD, MEMORY_CAP, device)