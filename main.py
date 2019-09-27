import gym
import numpy as np
import math

from palette import PaletteEnv


def choose_action(state, q_table, action_space, epsilon):
    if np.random.random_sample() < epsilon:
        return action_space.sample()
    else:
        return np.argmax(q_table[state])

def get_state(observation, n_buckets, state_bounds):
    state = [0] * len(observation)
    for i, s in enumerate(observation):
        l, u = state_bounds[i][0], state_bounds[i][1]
        if s <= l:
            state[i] = 0
        elif s >= u:
            state[i] = n_buckets[i] - 1
        else:
            state[i] = int(((s - l) / (u - l)) * n_buckets[i])
    return tuple(state)

get_epsilon = lambda j: max(0.01, min(1, 1.0 - math.log10((j + 1) / 25)))
get_lr = lambda j: max(0.01, min(0.5, 1.0 - math.log10((j + 1) / 25)))
gamma = 0.999 # reward discount factor

env = PaletteEnv()#gym.make('CartPole-v0')
n_buckets = (1, 1, 6, 3)
n_actions = env.action_space.n
# states
print(env.observation_space)
state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
state_bounds[1] = [-1, 1]
state_bounds[3] = [-math.radians(50), math.radians(50)]

episodes = 200
max_steps = 200

q_table = np.zeros(n_buckets + (n_actions,))

for i_episode in range(episodes):
    env.render()
    epsilon = get_epsilon(i_episode)
    lr = get_lr(i_episode)
    print(lr)
    observation = env.reset()
    totalReward = 0
    state = get_state(observation, n_buckets, state_bounds)
    for i in range(max_steps):
        action = choose_action(state, q_table, env.action_space, epsilon)
        observation, reward, done, _ = env.step(action)

        totalReward += reward
        next_state = get_state(observation, n_buckets, state_bounds)
        q_next_max = np.amax(q_table[next_state])
        q_table[state + (action,)] += lr * (reward + gamma * q_next_max - q_table[state + (action,)])
        state = next_state
        if done:
            print('Episode finished after {} timesteps, total rewards {}'.format(i + 1, totalReward))
            break

print(q_table)

observation = env.reset()
state = get_state(observation, n_buckets, state_bounds)
done = False
step = 0
while not done:
    step+=1
    print(step)
    env.render()
    action = np.argmax(q_table[state])
    observation, _, done, _ = env.step(action)
    if not done:
        state = get_state(observation, n_buckets, state_bounds)
