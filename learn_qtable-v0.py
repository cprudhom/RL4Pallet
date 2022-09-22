import time

import math
import gym
import numpy as np
import matplotlib.pyplot as plt


def numberToAction(number):
    if number < env.X * env.Y:
        o = 0
        p = number
    else:  # o = 1:
        o = 1
        p = number - env.X * env.Y
    px = p // env.Y
    py = p % env.Y
    return dict({
        "pos_x": px,
        "pos_y": py,
        "ori": o
    })


def actionToNumber(action):
    return action['pos_x'] * env.Y + action['pos_y'] + env.X * env.Y * action['ori']


def choose_action(state, q_table, action_space, epsilon):
    if np.random.random_sample() < epsilon:
        return action_space.sample()
    else:
        if state not in q_table:
            return action_space.sample()
        return numberToAction(np.argmax(q_table.get(state)))


def get_state(observation):
    return tuple(observation['fill'].flatten())


get_epsilon = lambda j: max(0.01, min(1, 1.0 - math.log10((j + 1) / 800)))
get_alpha = lambda j: max(0.01, min(1, 0.8 - math.log10((j + 1) / 800)))
gamma = 0.99  # reward discount factor

env = gym.make('gym_pallet:pallet-v0')
n_pos_x, n_pos_y, n_ori = env.action_space.spaces['pos_x'].n, \
                          env.action_space.spaces['pos_y'].n, \
                          env.action_space.spaces['ori'].n

# states
print(env.observation_space)
n_buckets = tuple([env.Z + 1 for _ in range(env.X * env.Y)])

episodes = 19000

q_table = {}#np.zeros(n_buckets + tuple([(n_pos_x * n_pos_y) * n_ori]))

rList = []
eList = []
lList = []

for i_episode in range(episodes):
    epsilon = get_epsilon(i_episode)
    alpha = get_alpha(i_episode)
    eList.append(epsilon)
    lList.append(alpha)
    # env.render()
    # time.sleep(.1)
    observation = env.reset()
    totalReward = 0
    state = get_state(observation)
    done = False
    t = 0
    while not done:
        t += 1
        action = choose_action(state, q_table, env.action_space, epsilon)
        observation, reward, done, _ = env.step(action)
        totalReward += reward
        next_state = get_state(observation)
        an = actionToNumber(action)
        q_next_max = 0.
        if next_state in q_table:
            q_next_max = np.amax(q_table.get(next_state))
        if state not in q_table:
            q_table[state] = [0. for i in range(n_pos_x * n_pos_y * n_ori)]
            q_table[state][an] += alpha * (reward + gamma * q_next_max)
        q_table[state][an] += alpha * (reward + gamma * q_next_max - q_table[state][an])
        state = next_state
        if done:
            print('Episode {} finished after {} timesteps, total rewards {}'.format(i_episode + 1, t, totalReward))
            break
    rList.append(totalReward)

plt.plot(rList)
plt.legend("Reward")
plt.show()
plt.plot(eList)
plt.legend("Epsilon")
plt.show()
plt.plot(lList)
plt.legend("Alpha")
plt.show()

observation = env.reset()
state = get_state(observation)
done = False
step = 0
while not done:
    step += 1
    time.sleep(.5)
    print(step)
    env.render()
    action = numberToAction(np.argmax(q_table[state]))
    print(action)
    observation, _, done, _ = env.step(action)
    if not done:
        state = get_state(observation)
    else:
        time.sleep(.5)
        env.render()
input()
