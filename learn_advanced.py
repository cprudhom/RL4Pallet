import time

import gym
import numpy as np
import math


def choose_action(state, q_table, action_space, epsilon):
    if np.random.random_sample() < epsilon:
        print("rnd")
        return action_space.sample()
    else:
        print("q_table")
        return np.argmax(q_table[state])


def flatten_grid(array):
    return int(array.sum())

def get_action(action):
    return action

def get_state(observation):
    state = [0] * len(observation)
    state[0] = flatten_grid(observation['fill'])
    state[1] = observation['left']
    state[2] = observation['type']
    return tuple(state)


get_epsilon = lambda j: max(0.01, min(1, 1.0 - math.log10((j + 1) / 25)))
get_lr = lambda j: max(0.01, min(0.5, 1.0 - math.log10((j + 1) / 25)))
gamma = 0.999  # reward discount factor

env = gym.make('gym_pallet:pallet-v0')
n_pos, n_ori = env.action_space.spaces['pos'].n, env.action_space.spaces['ori'].n

# states
print(env.observation_space)
n_buckets = (49 * 4, 25, 2)

episodes = 2000

q_table = np.zeros(n_buckets + (n_pos, n_ori,))

for i_episode in range(episodes):
    env.render()
    time.sleep(.1)
    epsilon = get_epsilon(i_episode)
    lr = get_lr(i_episode)
    print(lr)
    observation = env.reset()
    totalReward = 0
    state = get_state(observation)
    items = observation['left']
    for t in range(items):
        action = choose_action(state, q_table, env.action_space, epsilon)
        print("action %s" % action)
        observation, reward, done, _ = env.step(action)

        totalReward += reward
        next_state = get_state(observation)
        q_next_max = np.amax(q_table[next_state])
        tup = state + (action['pos'], action['ori'],)
        print(tup)
        q_table[tup] += lr * (reward + gamma * q_next_max - q_table[tup])
        state = next_state
        if done:
            print('Episode {} finished after {} timesteps, total rewards {}'.format(i_episode + 1, t, totalReward))
            break

print(q_table)

observation = env.reset()
state = get_state(observation)
done = False
step = 0
while not done:
    step += 1
    print(step)
    env.render()
    action = np.argmax(q_table[state])
    observation, _, done, _ = env.step(action)
    if not done:
        state = get_state(observation)
