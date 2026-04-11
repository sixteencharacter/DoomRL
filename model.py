from network import CNN_DQN
from env import n_actions , device
from torch import optim as O
from config import *
from replay_memory import ReplayMemory

policy_net = CNN_DQN(n_actions=(n_actions)).to(device)
target_net = CNN_DQN(n_actions=(n_actions)).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = O.AdamW(policy_net.parameters(),lr=LR,amsgrad=True)
memory = ReplayMemory(MEMORY_CAP)