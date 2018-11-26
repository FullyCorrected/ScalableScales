import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.datasets import mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

x = tf.placeholder(tf.float32, shape=[None, 28, 28])
y = tf.placeholder(tf.float32, shape=[None, 10])

inputs = 28*28
hidden_units = 512
outputs = 10

FC_1_parameters = {'weights': tf.Variable(tf.random_normal(shape=[inputs, hidden_units], stddev=0.01)),
                   'biases': tf.Variable(tf.random_normal(shape=[hidden_units], stddev=0.01))}
FC_2_parameters = {'weights': tf.Variable(tf.random_normal(shape=[hidden_units, outputs], stddev=0.01)),
                   'biases': tf.Variable(tf.random_normal(shape=[outputs], stddev=0.01))}

input_layer = tf.reshape(x, [-1, 784])

FC_1_after_weights = tf.matmul(input_layer, FC_1_parameters['weights']) # in: shape=[None, 784] out: shape=[None, hidden_units]
FC_1_after_bias = tf.add(FC_1_after_weights, FC_1_parameters['biases'])

after_activation = tf.nn.relu(FC_1_after_bias)

FC_2_after_weights = tf.matmul(after_activation, FC_2_parameters['weights'])# in: shape=[None, hidden_units] out: shape=[None, outputs]
FC_2_after_bias = tf.add(FC_2_after_weights, FC_2_parameters['biases'])

y_hat = tf.nn.softmax(FC_2_after_bias)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=FC_2_after_bias, labels=y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

epochs = 5
batch_size = 256
nbr_of_examples = x_train.shape[0]
nbr_of_batches_per_epoch = nbr_of_examples // batch_size

losses = []
accuracies = []
for epoch in range(epochs):
    print('Beginning epoch {} out of {}'.format(epoch+1, epochs))
    print('__________________________________________')
    total_epoch_loss = 0
    for batch in range(nbr_of_batches_per_epoch):
        x_batch = x_train[batch_size*batch:batch_size*(batch+1),:,:]
        y_batch = to_categorical(y_train[batch_size*batch:batch_size*(batch+1)], num_classes=10)
        
        _, batch_loss, batch_predictions= sess.run(fetches=[optimizer, loss, y_hat], feed_dict={x: x_batch, y: y_batch})
        losses.append(batch_loss)
        total_epoch_loss+=batch_loss
        
        batch_predictions = np.argmax(batch_predictions, axis=1)
        batch_accuracy = np.count_nonzero(np.equal(batch_predictions, y_train[batch_size*batch:batch_size*(batch+1)])) / y_batch.shape[0]
        accuracies.append(batch_accuracy)
        if (batch) % 100 == 0:
            print('Beginning batch {} out of {}'.format(batch+1,nbr_of_batches_per_epoch))
            print('Batch loss: {:.2f}'.format(batch_loss))
            print('Batch accuracy: {:.2f}'.format(batch_accuracy))
            print('------------------------------------------')
    print('==========================================')

plt.figure(1)
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Batch loss')
plt.show()

plt.figure(2)
plt.plot(accuracies)
plt.xlabel('Iteration')
plt.ylabel('Batch accuracy')
plt.show()

test_set_predictions = sess.run(y_hat, feed_dict={x: x_test})
test_set_predictions = np.argmax(test_set_predictions, axis=1)
accuracy = np.count_nonzero(np.equal(test_set_predictions, y_test)) / y_test.shape[0]
print('test accuracy: {:.2f}'.format(accuracy))