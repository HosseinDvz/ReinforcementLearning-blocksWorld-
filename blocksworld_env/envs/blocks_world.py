import gymnasium as gym
from gymnasium import spaces
import pygame
from screen import Display
from swiplserver import PrologMQI, PrologThread
import numpy as np



class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).

        self.mqi = PrologMQI()
        self.prolog_thread = self.mqi.create_thread()
        self.prolog_thread.query('[blocks_world]')

     
        #print(result)

env = GridWorldEnv()

states_dict = {}
prolog_states =  env.prolog_thread.query('state(S)') 
for i, state in enumerate(prolog_states):
    states_dict[state['S']] = i
print(states_dict)


#env.prolog_thread.query('[blocks_world]')
result = env.prolog_thread.query('action(A)')
actions_dict = {}
#print(result)
for i,A in enumerate(result):
     action_string = A['A']['functor']
     first=True
     for arg in A['A']['args']:
         if first:
             first = False
             action_string += '('
         else:
             action_string += ','
         action_string += str(arg)
     action_string += ')'
     actions_dict[i] = action_string

print(actions_dict)

observation_space = spaces.Discrete(len(states_dict))
action_space = spaces.Discrete(len(actions_dict))




)

#list(states_dict.keys())[list(states_dict.values()).index(state)]