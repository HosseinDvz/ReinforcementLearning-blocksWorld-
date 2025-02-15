import gymnasium as gym
from gymnasium import spaces
import pygame
from screen import Display
from swiplserver import PrologMQI, PrologThread
import numpy as np



class BlocksWorldEnv_v0(gym.Env):
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

        print(f'hello from version 0')
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

        #print(self.actions_dict)


        self.acttion_to_int = {val: key for key, val in self.actions_dict.items()}
        #print(self.acttion_to_int)

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Discrete(len(self.states_dict)),
                "target": spaces.Discrete(len(self.states_dict))
            }
        )

        #self.observation_space = spaces.Discrete(len(self.states_dict))
        #print(self.observation_space.n)
        self.action_space = spaces.Discrete(len(self.actions_dict))

        #self.init_state = list(self.states_dict.keys())[0]
        #print(f'initial state is: {self.init_state}')

        self.display = Display()

    def _get_obs(self):
        return {"agent": self.state_num, "target": self.target_num}
    
    def _get_info(self):
        return {
            "target":
                self.target_num
            
        }

    def reset(self, seed=None, options=None):

        #generating a random target for target
        #self.target_num = np.random.randint(1,120)
        self.target_num = 10

        # transforming number to state
        self.target_str = self.int_to_state[self.target_num] #target state
        #print(f'target is: {self.target_str}')
        self.display.target = self.target_str

        #issuing the prolog reset query
        self.prolog_thread.query('reset')

        # retrieving current state - agent state
        result = self.prolog_thread.query('current_state(State)')
        #print(result)
        self.state_str = result[0]['State']
        print(f'initial state is: { self.state_str}')
        self.state_num = self.states_dict[self.state_str]


        #observation = self.state_num
        #observation = self._get_obs()['agent']
        observation = self._get_obs()
        info = self._get_info()
        

        return observation,info
        

    def step(self, action):
        
        #current_state = self.prolog_thread.query('current_state(State)')
        #print(f'state before action: {self.state_str}')

        # getting move str from the action number
        act = self.actions_dict[action]

        # result is true of action is possible
        result = self.prolog_thread.query(f'step({act})')
        #print(result)
        done = False
        if result:

            self.state_str = self.prolog_thread.query('current_state(State)')[0]['State']
            done = (self.state_str == self.target_str)

            if done:
                reward = 100
            else:
                reward = -1

        else:
            reward = -100
        
        self.display.step(self.state_str)

        self.state_num = self.states_dict[self.state_str]
        observation = self._get_obs()
        #observation = self.state_num
        #observation = self._get_obs()['agent']
        info = self._get_info()

        #if self.render_mode == "human":
        #    self._render_frame()
        
        #print(f'final current status: {self.state_str}', f'reward is: {reward}')
        return observation, reward, done, False, info


        

if __name__ == '__main__':

    env = BlocksWorldEnv_v0()
    #states = env.states_dict
    #print(states.get('1a4', 00))

    state, target = env.reset()

    for i in [64, 88, 87,75]:

        print(env.step(i))