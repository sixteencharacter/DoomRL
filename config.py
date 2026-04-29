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

if SAMPLING_METHOD in ("PER", "WindowPER") and MAX_STEPS is None:
    raise ValueError(
        f"SAMPLING_METHOD='{SAMPLING_METHOD}' requires MAX_STEPS to be set in cfg.yml "
        "(beta annealing schedule depends on it). Switch to 'Uniform' "
        "or set MAX_STEPS for step-mode training."
    )

# STARFORMER hyperparameters
STARFORMER_K = data.get('STARFORMER_K', 30)
STARFORMER_LAYERS = data.get('STARFORMER_LAYERS', 6)
STARFORMER_HEADS = data.get('STARFORMER_HEADS', 8)
STARFORMER_DIM = data.get('STARFORMER_DIM', 192)
STARFORMER_LR = data.get('STARFORMER_LR', 6e-4)
STARFORMER_WARMUP_STEPS = data.get('STARFORMER_WARMUP_STEPS', 4000)
STARFORMER_RTG_SCALE = data.get('STARFORMER_RTG_SCALE', 1.0)
STARFORMER_RTG_TARGET = data.get('STARFORMER_RTG_TARGET', 80.0)
USE_RTG = data.get('USE_RTG', True)
STARFORMER_USE_EPSILON = data.get('STARFORMER_USE_EPSILON', True)
_min_valid_anchors_raw = data.get('MIN_VALID_ANCHORS', None)
MIN_VALID_ANCHORS = (4 * BATCH_SIZE) if _min_valid_anchors_raw is None else int(_min_valid_anchors_raw)

# METHOD validation
if METHOD not in ('DQN', 'DDQN', 'STARFORMER', 'PPO'):
    raise ValueError(f"Unsupported METHOD={METHOD}; expected DQN | DDQN | STARFORMER | PPO")

if METHOD == 'PPO':
    if MAX_STEPS is None and NUM_EPISODE is None:
        raise ValueError(
            "METHOD=PPO requires either MAX_STEPS or NUM_EPISODE in cfg.yml."
        )
    if SAMPLING_METHOD != 'Uniform':
        import warnings as _warnings
        _warnings.warn(
            f"METHOD=PPO with SAMPLING_METHOD={SAMPLING_METHOD!r} is contradictory "
            "(PPO is on-policy; replay buffer is unused). Forcing SAMPLING_METHOD='Uniform'."
        )
        SAMPLING_METHOD = 'Uniform'

if METHOD == 'STARFORMER':
    if MAX_STEPS is None:
        raise ValueError(
            "METHOD=STARFORMER requires MAX_STEPS in cfg.yml (cosine LR schedule depends on it)."
        )
    if ARCH != 'STARFORMER':
        raise ValueError(
            "METHOD=STARFORMER requires ARCH=STARFORMER (config.py reads ARCH for create_q_network)."
        )
    if SAMPLING_METHOD == 'Uniform':
        import warnings as _warnings
        _warnings.warn(
            "METHOD=STARFORMER with SAMPLING_METHOD=Uniform overridden to 'WindowPER'."
        )
        SAMPLING_METHOD = 'WindowPER'
    elif SAMPLING_METHOD == 'PER':
        import warnings as _warnings
        _warnings.warn(
            "METHOD=STARFORMER auto-upgrading SAMPLING_METHOD from 'PER' to 'WindowPER'."
        )
        SAMPLING_METHOD = 'WindowPER'