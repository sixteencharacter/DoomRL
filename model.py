from network import create_q_network
from env import n_actions, device
from torch import optim as O
from torch.optim.lr_scheduler import LambdaLR
import math
from config import *
from replay_memory import create_replay_memory
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

policy_net = create_q_network(arch=ARCH, n_actions=n_actions)

if PRELOAD_WEIGHT is not None:
    policy_net.load_state_dict(torch.load(PRELOAD_WEIGHT, map_location=torch.device("cpu"))['model'])
    print("model weight loaded from {}".format(PRELOAD_WEIGHT))
    logger.info("model weight loaded from {}".format(PRELOAD_WEIGHT))
else:
    logger.info("model weight not provided, using random weight")
    print("No initial weight provided, fall back to random weight")

policy_net = policy_net.to(device)

# Defaults — overridden per branch below
target_net = None
memory = None
scheduler = None
loss_module = None
advantage_module = None

if USE_PPO and METHOD != 'STARFORMER':
    # Standard PPO: ResNet or Baseline ActorCritic + TorchRL
    from tensordict.nn import TensorDictModule
    from torchrl.modules import ProbabilisticActor, ValueOperator, OneHotCategorical
    from torchrl.objectives import ClipPPOLoss
    from torchrl.objectives.value import GAE

    class _ActorWrapper(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net
        def forward(self, x):
            logits, _ = self.net(x)
            return logits

    class _ValueWrapper(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net
        def forward(self, x):
            _, value = self.net(x)
            return value

    actor_module = TensorDictModule(
        module=_ActorWrapper(policy_net),
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
        module=_ValueWrapper(policy_net),
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
    optimizer = O.AdamW(policy_net.parameters(), lr=LR)

    if PPO_COMPILE:
        try:
            policy_net = torch.compile(policy_net)
            logger.info("PPO: torch.compile(policy_net) enabled")
        except Exception as _e:
            logger.warning("PPO: torch.compile failed (%s); running uncompiled", _e)

elif METHOD == 'STARFORMER':
    # STARFORMER backbone with AdamW + cosine LR (shared by both USE_PPO=true/false)
    optimizer = O.AdamW(
        policy_net.parameters(),
        lr=STARFORMER_LR,
        weight_decay=0.1,
        betas=(0.9, 0.95),
    )

    def _starformer_lr_lambda(step: int) -> float:
        warmup = max(1, STARFORMER_WARMUP_STEPS)
        if step < warmup:
            return float(step) / float(warmup)
        if MAX_STEPS is None or MAX_STEPS <= warmup:
            return 1.0
        progress = (step - warmup) / float(MAX_STEPS - warmup)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda=_starformer_lr_lambda)

    if USE_PPO:
        # On-policy: no target net, no replay memory
        target_net = None
        memory = None
    else:
        target_net = create_q_network(arch=ARCH, n_actions=n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        memory = create_replay_memory(SAMPLING_METHOD, MEMORY_CAP, device)

else:
    # DQN / DDQN
    optimizer = O.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    target_net = create_q_network(arch=ARCH, n_actions=n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    memory = create_replay_memory(SAMPLING_METHOD, MEMORY_CAP, device)
