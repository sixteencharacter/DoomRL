import collections
from datamodel import Transition
import random
import os
from typing import List
import torch

class ReplayMemory:

    def __init__(self , capacity,device) :
        self.memory = collections.deque(maxlen=capacity)
        self.device = device

    def push(self,*args) :
        self.memory.append(Transition(*args))

    def sample(self,batch_size,preprocessor=None) :
        if preprocessor == None :
            return random.sample(self.memory,batch_size)
        else :
            samples : List[Transition]= random.sample(self.memory,batch_size)
            state_processed = preprocessor([s.state for s in samples],device=self.device)
            next_state_processed = preprocessor([s.next_state for s in samples],device=self.device)
            action = [s.action for s in samples]
            reward = [s.reward for s in samples]
            return [Transition(state,action,next_state,reward) for state,action,reward,next_state in zip(state_processed,action,reward,next_state_processed)]
    
    def __len__(self) :
        return len(self.memory)