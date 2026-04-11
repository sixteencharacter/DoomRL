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
NUM_EPISODE = data['NUM_EPISODE']

ARCH = data['ARCH']
VERSION = data['VERSION']
FRAME_SKIP = data['FRAME_SKIP']