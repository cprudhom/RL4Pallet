import time

import gym

env = gym.make('gym_pallet:pallet-v0')
for i_episode in range(200):
    observation = env.reset()
    items = observation['left']
    for t in range(items):
        env.render()
        time.sleep(.1)
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            env.render()
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()