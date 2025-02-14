
import gymnasium as gym
from gymnasium import spaces
import pygame
from screen import Display
from swiplserver import PrologMQI, PrologThread
import numpy as np



class BlocksWorldEnv(gym.Env):
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


        prolog_states = self.prolog_thread.query('state(S)') 

        self.states_dict = {}
        self.int_to_state = {}
        for i, state in enumerate(prolog_states):
            self.states_dict[state['S']] = i
            self.int_to_state[i] = state['S'] #i is unique can be used as key. most efficient way
        #print(self.states_dict)

        
        actions = self.prolog_thread.query('action(A)')
        self.actions_dict = {}
        #print(result)
        for i,A in enumerate(actions):
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
            self.actions_dict[i] = action_string

        print(self.actions_dict)


        self.acttion_to_int = {val: key for key, val in self.actions_dict.items()}
        #print(self.acttion_to_int)

        self.observation_space = spaces.Discrete(len(self.states_dict))
        self.action_space = spaces.Discrete(len(self.actions_dict))

        self.init_state = list(self.states_dict.keys())[0]
        print(f'initial state is: {self.init_state}')

        

    def reset(self):

        #generating a random target for target
        target_num = np.random.randint(1,120)

        # transforming number to state
        self.target_state = self.int_to_state[target_num] #target state
        print(f'target is: {self.target_state}')

        #issuing the prolog reset query
        self.prolog_thread.query('reset')

        # retrieving current state - agent state
        result = self.prolog_thread.query('current_state(State)')
        self.state_str = result[0]('State')

        return self.state_str, self.target_state
        

    def step(self, action):
        
        current_state = self.prolog_thread.query('current_state(State)')
        print(f'state before action: {current_state}')
        act = self.actions_dict[action]
        result = self.prolog_thread.query(f'step({act})')
        #print(result)
        current_state = self.prolog_thread.query('current_state(State)')
        '''
        if result:
            current_state = self.prolog_thread.query('current_state(State)')
            done = (current_state[0]['State'] == self.target_state)
            if done:
                reward = 100
            else:
                reward = -1

        else:
            reward = -100
        
        '''
        print(current_state)
        return ''


        


env = BlocksWorldEnv()
#states = env.states_dict
#print(states.get('1a4', 00))

for i in [64, 88, 87,75]:

    print(env.step(i))
