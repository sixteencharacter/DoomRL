import yaml

with open('cfg.yml','r') as fp :
    data = yaml.safe_load(fp)

# Training setting
GAMMA = data['GAMMA']
LR = data['LR']
BATCH_SIZE = data['BATCH_SIZE']
EPS_START = data['EPS_START']
EPS_END = data['EPS_END']
EPS_DECAY = data['EPS_DECAY']
TAU = data['TAU']
NUM_EPISODE = None if 'NUM_EPISODE' not in data else data['NUM_EPISODE']
MAX_STEPS = None if 'MAX_STEPS' not in data else data['MAX_STEPS']

# naming
ARCH = data['ARCH']
VERSION = data['VERSION']
VARIANT = data["VARIANT"]

# gaming
FRAME_SKIP = data['FRAME_SKIP']
SCENARIO_NAME = data['SCENARIO_NAME']

# memory limit
MEMORY_CAP = data['MEMORY_CAP']

# weights loading / saving
PRELOAD_WEIGHT = None if 'PRELOAD_WEIGHT' not in data else data['PRELOAD_WEIGHT']
CHKPOINT_NUM = 0 if 'CHKPOINT_NUM' not in data else data['CHKPOINT_NUM']
SAVING_INTERVAL = data['SAVING_INTERVAL']
VALIDATION_INTERVAL = data['VALIDATION_INTERVAL']
VALIDATION_EPISODES = data['VALIDATION_EPISODES']

# inferencing
RENDER_MODE = None if 'RENDER_MODE' not in data else data['RENDER_MODE']
RESOLUTION = data['RESOLUTION']

METHOD = 'DQN' if 'METHOD' not in data else data['METHOD']

# PPO parameters
PPO_EPOCHS = data.get('PPO_EPOCHS', 10)
PPO_CLIP = data.get('PPO_CLIP', 0.2)
ENTROPY_COEF = data.get('ENTROPY_COEF', 0.01)
CRITIC_COEF = data.get('CRITIC_COEF', 0.5)
GAE_LAMBDA = data.get('GAE_LAMBDA', 0.95)
PPO_BATCH_SIZE = data.get('PPO_BATCH_SIZE', 128)
PPO_BUFFER_SIZE = data.get('PPO_BUFFER_SIZE', 1024)

# Replay buffer sampling
SAMPLING_METHOD = data.get('SAMPLING_METHOD', 'Uniform')
ALPHA = data.get('ALPHA', 0.6)
BETA_START = data.get('BETA_START', 0.4)
BETA_END = data.get('BETA_END', 1.0)
PER_EPSILON = data.get('PER_EPSILON', 1e-6)

if SAMPLING_METHOD == "PER" and MAX_STEPS is None:
    raise ValueError(
        "SAMPLING_METHOD='PER' requires MAX_STEPS to be set in cfg.yml "
        "(beta annealing schedule depends on it). Switch to 'Uniform' "
        "or set MAX_STEPS for step-mode training."
    )