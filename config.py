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