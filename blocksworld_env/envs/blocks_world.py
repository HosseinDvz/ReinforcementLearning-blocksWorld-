
import gymnasium as gym
from gymnasium import spaces
import pygame
from screen import Display
from swiplserver import PrologMQI, PrologThread
import numpy as np



class BlocksWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        #self.size = size  # The size of the square grid
        #self.window_size = 512  # The size of the PyGame window

        self.mqi = PrologMQI()
        self.prolog_thread = self.mqi.create_thread()
        self.prolog_thread.query('[blocks_world_v2]')

        print('hello from target version')

        prolog_states = self.prolog_thread.query('state(S)') 

        self.states_dict = {}
        self.int_to_state = {}
        for i, state in enumerate(prolog_states):
            self.states_dict[state['S']] = i
            self.int_to_state[i] = state['S'] #i is unique can be used as key. most efficient way
        #print(list(self.states_dict.items())[:120])

        #creating a three char target list for efficient training- each target will be trained 30 times
        self.target_list = []
        for state, number in list(self.states_dict.items())[:120]:  # Taking first 120 states
            target = state[3:]  # Extract the last three characters
            self.target_list.extend([target] * 30)  # Append 30 separate occurrences

        self.reset_index = 0 # To track position in the list
        #print(len(self.target_list))


        #print('*'*15)
        #print(self.int_to_state)

        
        actions = self.prolog_thread.query('action(A)')
        self.actions_dict = {}
        #print(actions)
        # explanation for this code is in oblocks_world_v0
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

        self.acttion_to_int = {val: key for key, val in self.actions_dict.items()}
        #print(self.acttion_to_int)

        #self.observation_space = spaces.Discrete(len(self.states_dict))
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Discrete(len(self.states_dict)),
                "target": spaces.Discrete(len(self.states_dict))
            }
        )

        self.action_space = spaces.Discrete(len(self.actions_dict))

        self.init_state = list(self.states_dict.keys())[0]
        #print(f'initial state is: {self.init_state}')

        self.display = Display()

    def _get_obs(self):
        return {"agent": self.state_num, "target": self.target_num}
    
    def _get_info(self):
        return {
            "target":
                self.target_num,
            'target string' : self.int_to_state[self.target_num]
            
        }


    def reset(self, seed=None, options=None):

        self.target_str_3char = self.target_list[self.reset_index]
        self.reset_index = (self.reset_index + 1) % len(self.target_list)  # Loop back when reaching the end

        # generating a random number between 1, 14400
        #rand_sate = np.random.randint(1,self.observation_space['agent'].n)


        #finding the target state by selecting the last three characters
        #self.target_str_3char = self.int_to_state[rand_sate][3:]

        #creating full target character - Example = full target state char must be '12a12a'
        self.full_target_str = self.target_str_3char + self.target_str_3char
    
        self.target_num = self.states_dict[self.full_target_str ]
        '''
        print(f'target stat number is {self.target_num}')
        print(f'target is: {self.target_str_3char}')
        print(f'full target char is {self.full_target_str}')
        print(f'target from dic is {self.int_to_state[self.target_num]}')
        '''


        # Displaying target
        self.display.target = self.target_str_3char

        #issuing the prolog reset query
        self.prolog_thread.query('reset')

        # retrieving current state - agent state
        result = self.prolog_thread.query('current_state(State)')
        #print(result)
        self.state_str = result[0]['State'] + self.target_str_3char
        #print(f'state string is :{self.state_str}')
        self.state_num = self.states_dict[self.state_str]


        #observation = self.state_num
        observation = self._get_obs()
        info = self._get_info()

        return observation,info
        

    def step(self, action):

        # getting move str from the action number
        act = self.actions_dict[action]

        # result is true of action is possible
        result = self.prolog_thread.query(f'step({act})')
        #print(result)
        done = False
        if result:

            three_char_state = self.prolog_thread.query('current_state(State)')[0]['State']
            done = (three_char_state == self.target_str_3char)
            self.state_str = three_char_state + self.target_str_3char

            if done:
                reward = 100
            else:
                reward = -1

        else:
            reward = -100
        
        #commenting for fast training
        self.display.step(self.state_str) # works with 6 chars state
        

        self.state_num = self.states_dict[self.state_str]
        #observation = self.state_num
        observation = self._get_obs()
        info = self._get_info()
        
        #print(f'final current status: {self.state_str}', f'reward is: {reward}')
        return observation, reward, done, False, info


        

if __name__ == '__main__':

    env = BlocksWorldEnv()
    #states = env.states_dict
    #print(states.get('1a4', 00))

    obs, info = env.reset()

    for i in [64, 88, 87,75]:

        print(env.step(i))
