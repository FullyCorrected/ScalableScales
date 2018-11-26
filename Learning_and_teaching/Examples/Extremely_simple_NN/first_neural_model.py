import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def squared_loss(y_hat, y):
    dif = tf.subtract(y_hat, y)
    loss = tf.square(dif)
    return loss

X1 = tf.placeholder(dtype=tf.float32, shape=[None])
y = tf.placeholder(dtype=tf.float32, shape=[None])

W = tf.Variable(tf.random_normal(shape=[], stddev=0.1))
X2 = tf.multiply(X1, W)

b = tf.Variable(tf.zeros(shape=[]))
X3 = tf.add(X2, b)

y_hat = tf.nn.sigmoid(X3)

L = tf.reduce_mean(squared_loss(y_hat, y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(L)

nbr_of_examples = 10000
X_train = np.linspace(0,10, nbr_of_examples).astype(np.float32)
y_train = (X_train > 5).astype(np.float32)
indices = np.arange(nbr_of_examples)
np.random.shuffle(indices)
X_train = X_train[indices]
y_train = y_train[indices]

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

losses = []

epochs = 10
batch_size = 512
for epoch in range(epochs):
    for batch in range(0,nbr_of_examples,batch_size):
        _, loss_val, W_val, b_val = sess.run([optimizer, L, W, b], feed_dict={X1: X_train[batch:batch+batch_size], y: y_train[batch:batch+batch_size]})
        losses.append(loss_val)
        print('hej')

plt.plot(np.arange(len(losses)), losses)
plt.xlabel('step')
plt.ylabel('loss')

nbr_of_test_examples=1000
X_test = np.random.uniform(-100,100,nbr_of_test_examples).astype(np.float32)
y_test = (X_test > 5).astype(np.float32)

y_hat_test = sess.run(y_hat, feed_dict={X1: X_test, y: y_test})
predictions = (y_hat_test > 0.5).astype(np.float32)
print('Accuracy: {:.2f}'.format(np.count_nonzero(y_test == predictions)/nbr_of_test_examples))

tf.reset_default_graph()
sess.close()