import gymnasium as gym
import blocksworld_env
import numpy as np
import time
import matplotlib.pyplot as plt
import pickle


env = gym.make("blocksworld_env/BlocksWorld-v1", render_mode="human")
#observation, info = env.reset() 
#target_num = info['target']

numstates= env.observation_space['agent'].n
numactions = env.action_space.n
# QTable : contains the Q-Values for every (state,action) pair


    
#qtable = np.random.rand(numstates, numactions).tolist()

def load_qtable(filename="qtable.pkl"):
    try:
        with open(filename, "rb") as f:
            qtable = pickle.load(f)
        print(f"Q-table loaded from {filename}")
        return qtable
    except FileNotFoundError:
        print("No saved Q-table found, starting fresh.")
        return None

qtable = load_qtable('/Users/hosseindavarzanisani/GitRepos/ReinforcementLearning/blocksWorld/qtable_6_3600.pkl')

# hyperparameters
episodes = 3600 
alpha = 1 #learning rate
gamma = 0.4 #increased gamma to assign more weight to value of next state
epsilon = 0.01 #decrease epsilon to decrease the chance of taking random actions
decay = 0.01 
success_counter = 0 #optional variable to count the number of success in all epidsodes. helps to set number of episodes

# training loop
all_rewards = [] #list to store sum of collected rewards in each episode
for i in range(episodes):

    print("episode #", i+1, "/", episodes)

    observation, info = env.reset()

    print(f"target is: {info['target string']}")

    # Q(terminal,.) = 0
    target_num = info['target']
    for action_num in range(numactions):
        qtable[target_num][action_num] = 0

    state = observation['agent']
    episode_reward = [] # list to store rewards in each step
    steps = 0
    done = False 

    while (not done): 
    
        time.sleep(0.9)

        # act randomly sometimes to allow exploration
        if np.random.uniform() < epsilon:
            action = env.action_space.sample()
        else:
            action = qtable[state].index(max(qtable[state]))

        # take action
        next_state_dic, reward, done, truncated, info = env.step(action)
        next_state = next_state_dic['agent']
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



def save_qtable(qtable, filename="qtable_6_6000.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(qtable, f)
    print(f"Q-table saved to {filename}")

save_qtable(qtable=qtable)

plt.plot(all_rewards, linestyle='-')
plt.xlabel("Episodes")
plt.ylabel("Total Rewards")
plt.title("Rewards per Episode")
plt.savefig("qlearning_plot_6.png")
plt.close()