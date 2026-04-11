import torch
import gymnasium as gym
from config import *
import vizdoom.gymnasium_wrapper

env = gym.make("VizdoomBasic-v1",screen_resolution='RES_160x120',render_mode='human',frame_skip=FRAME_SKIP)

n_actions = env.action_space.n

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

