import gymnasium as gym
import blocksworld_env
import numpy as np
import time


env = gym.make("blocksworld_env/BlocksWorld-v0", render_mode="human")
observation, info = env.reset() 
target_num = info['target']

numstates= env.observation_space.n
numactions = env.action_space.n
# QTable : contains the Q-Values for every (state,action) pair
qtable = np.random.rand(numstates, numactions).tolist()

# setting qtable final state (Q(terminal,.)) to zero
for action_num in range(numactions):
    qtable[target_num][action_num] = 0

# hyperparameters
episodes = 50  
alpha = 1 #learning rate
gamma = 0.4 #increased gamma to assign more weight to value of next state
epsilon = 0.01 #decrease epsilon to decrease the chance of taking random actions
decay = 0.01 
success_counter = 0 #optional variable to count the number of success in all epidsodes. helps to set number of episodes

# training loop
all_rewards = [] #list to store sum of collected rewards in each episode
for i in range(episodes):

    observation, info = env.reset()
    state = observation
    episode_reward = [] # list to store rewards in each step
    steps = 0

    done = False 

    while (not done): 
        #os.system('clear')
        #print("episode #", i+1, "/", episodes)
        #env.render()
        time.sleep(0.01)

        # act randomly sometimes to allow exploration
        if np.random.uniform() < epsilon:
            action = env.action_space.sample()
        else:
            action = qtable[state].index(max(qtable[state]))

        # take action
        next_state, reward, done, truncated, info = env.step(action)
        
        episode_reward.append(reward) #appending the collected reward for each action

        # update qtable value with Bellman equation
        qtable[state][action] = qtable[state][action] + alpha * (reward + gamma * max(qtable[next_state])-qtable[state][action]) 
    
        # update state
        state = next_state

    print(f'sum of rewards: {sum(episode_reward)}')
    all_rewards.append(sum(episode_reward))

    # The more we learn, the less we take random actions
    epsilon -= decay*epsilon
env.close()
