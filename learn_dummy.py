import time

import gym

env = gym.make('gym_pallet:pallet-v0')
for i_episode in range(200):
    observation = env.reset()
    done = False
    t = 0
    while not done:
        t += 1
        env.render()
        time.sleep(.2)
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            env.render()
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()