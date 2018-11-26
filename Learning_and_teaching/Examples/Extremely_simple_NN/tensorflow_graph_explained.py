import tensorflow as tf

print('-------------------------------------')

a = tf.constant(3, dtype=tf.int32)
b = tf.constant(2, dtype=tf.int32)

c = tf.add(a,b)
print(c)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
c_value = sess.run(c, feed_dict={})
print(c_value)
tf.reset_default_graph()
sess.close()

print('-------------------------------------')

a = tf.placeholder(dtype=tf.int32, shape=[])
b = tf.placeholder(dtype=tf.int32, shape=[])

c = tf.add(a,b)
print(c)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
c_value = sess.run(c, feed_dict={a: 11, b: 5})
print(c_value)
tf.reset_default_graph()
sess.close()

print('-------------------------------------')