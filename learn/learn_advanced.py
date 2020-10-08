import time

import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math


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


env = gym.make('gym_pallet:pallet-v0')
n_pos_x, n_pos_y, n_ori = env.action_space.spaces['pos_x'].n, \
                          env.action_space.spaces['pos_y'].n, \
                          env.action_space.spaces['ori'].n


# Parameters
get_epsilon = lambda j: max(0.01, min(1, 1.0 - math.log10((j + 1) / 500)))
get_alpha = lambda j: max(0.01, min(1, 0.8 - math.log10((j + 1) / 500)))
gamma = 0.99  # reward discount factor

learning_rate = 0.001
training_epochs = 4000

# Network Parameters
n_input = env.X * env.Y
n_hidden_1 = 256  # 1st layer number of neurons
n_hidden_2 = 128  # 2nd layer number of neurons
n_classes = (n_pos_x * n_pos_y) * n_ori

X = tf.placeholder("float", [None, n_input])
# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
# Create model
# Hidden fully connected layer with 256 neurons
layer_1 = tf.add(tf.matmul(X, weights['h1']), biases['b1'])
# Hidden fully connected layer with 256 neurons
layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
# Output fully connected layer with a neuron for each class
Qout = tf.matmul(layer_2, weights['out']) + biases['out']
predict = tf.argmax(Qout, 1)

# Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[1, n_classes], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

# Initializing the variables
init = tf.global_variables_initializer()

# create lists to contain total rewards and steps per episode
rList = []
eList = []
lList = []
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        print("Episode #%d" % epoch)
        epsilon = get_epsilon(epoch)
        alpha = get_alpha(epoch)
        eList.append(epsilon)
        lList.append(alpha)
        s = env.reset()
        rAll = 0
        done = False
        j = 0
        while not done:
            j += 1
            # Choose an action by greedily (with e chance of random action) from the Q-network
            a, allQ = sess.run([predict, Qout], feed_dict={X: s["fill"].reshape([1, n_input])})
            if np.random.random_sample() < epsilon:
                act = env.action_space.sample()
            else:
                act = numberToAction(a[0])
            # Get new state and reward from environment
            s1, r, done, _ = env.step(act)
            # Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(Qout, feed_dict={X: s1["fill"].reshape([1, n_input])})
            # Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0, a[0]] += (1 - alpha) * targetQ[0, a[0]] + alpha * (r + gamma * maxQ1)
            # Train our network using target and predicted Q values
            _, _ = sess.run([updateModel, weights['out']], feed_dict={X: s["fill"].reshape([1, n_input]), nextQ: targetQ})
            rAll += r
            s = s1
            if done:
                break
        print("Reward: %d" % rAll)
        rList.append(rAll)
print("Percent of succesful episodes: " + str(sum(rList) / training_epochs) + "%")

plt.plot(rList)
plt.legend("Reward")
plt.show()
plt.plot(eList)
plt.legend("Epsilon")
plt.show()
plt.plot(lList)
plt.legend("Alpha")
plt.show()

with tf.Session() as sess:
    sess.run(init)
    s = env.reset()
    done = False
    step = 0
    while not done:
        step += 1
        time.sleep(.5)
        print(step)
        env.render()
        a, allQ = sess.run([predict, Qout], feed_dict={X: s["fill"].reshape([1, n_input])})
        act = numberToAction(a[0])
        print(act)
        s1, _, done, _ = env.step(act)
        if not done:
            s = s1
