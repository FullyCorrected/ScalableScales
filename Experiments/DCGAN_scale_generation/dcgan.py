import tensorflow as tf
import numpy as np

class DCGAN(object):
    @staticmethod
    def __leaky_relu(x):
        y = tf.maximum(x, tf.multiply(x, 0.2))
        return y
    
    @staticmethod
    def __binary_cross_entropy(y_hat, y):
        eps = 1e-12
        L = (-(y_hat * tf.log(y + eps) + (1. - y_hat) * tf.log(1. - y + eps)))
        return L
    
    def __init__(self, path_to_save=None):
        self.noise_dimensions = 100
        self.image_shape = [128, 128, 3]
        
        self.X_in = tf.placeholder(dtype=tf.float32, shape=[None]+self.image_shape, name='X')
        self.noise = tf.placeholder(dtype=tf.float32, shape=[None, self.noise_dimensions])
        
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
        self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')
        self.learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')
        
        self.X_generated = self.__build_generator_graph(self.noise)
        d_prob_real = self.__build_discriminator_graph(self.X_in, reuse = None)
        d_prob_fake = self.__build_discriminator_graph(self.X_generated, reuse=True)
        
        self.avg_prob_of_reals = tf.reduce_mean(d_prob_real)
        self.avg_prob_of_fakes = tf.reduce_mean(d_prob_fake)
        
        generator_parameters = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
        discriminator_parameters = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        
        generator_regularization = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), generator_parameters)
        discriminator_regularization = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), discriminator_parameters)
        
        self.generator_loss = tf.reduce_mean(self.__binary_cross_entropy(tf.ones_like(d_prob_fake), d_prob_fake))
        discriminator_loss_real = self.__binary_cross_entropy(tf.ones_like(d_prob_real), d_prob_real)
        discriminator_loss_fake = self.__binary_cross_entropy(tf.zeros_like(d_prob_fake), d_prob_fake)
        self.discriminator_loss = tf.reduce_mean(0.5 * (discriminator_loss_real + discriminator_loss_fake))
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.generator_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5).minimize(self.generator_loss + generator_regularization, var_list=generator_parameters)
            self.discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5).minimize(self.discriminator_loss + discriminator_regularization, var_list=discriminator_parameters)
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=config)
        new_network = path_to_save is None
        if new_network:
            init = tf.global_variables_initializer()
            self.sess.run(init)
        else:
            saver = tf.train.Saver()
            saver.restore(self.sess, path_to_save)
            
        
    def __build_generator_graph(self, z):
        activation = tf.nn.relu
        momentum = 0.99
        with tf.variable_scope("generator", reuse=None):
            d1 = 4
            d2 = 1024
            x = tf.layers.dense(z, units=d1 * d1 * d2, activation=activation)
            x = tf.layers.dropout(x, self.keep_prob)      
            x = tf.contrib.layers.batch_norm(x, is_training=self.is_training, decay=momentum)  
            x = tf.reshape(x, shape=[-1, d1, d1, d2])
            x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=512, strides=2, padding='same', activation=activation)
            x = tf.layers.dropout(x, self.keep_prob)
            x = tf.contrib.layers.batch_norm(x, is_training=self.is_training, decay=momentum)
            x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=256, strides=2, padding='same', activation=activation)
            x = tf.layers.dropout(x, self.keep_prob)
            x = tf.contrib.layers.batch_norm(x, is_training=self.is_training, decay=momentum)
            x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=128, strides=2, padding='same', activation=activation)
            x = tf.layers.dropout(x, self.keep_prob)
            x = tf.contrib.layers.batch_norm(x, is_training=self.is_training, decay=momentum)
            x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=64, strides=2, padding='same', activation=activation)
            x = tf.layers.dropout(x, self.keep_prob)
            x = tf.contrib.layers.batch_norm(x, is_training=self.is_training, decay=momentum)
            x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=3, strides=2, padding='same', activation=tf.nn.sigmoid)
            return x
    
    def __build_discriminator_graph(self, img_in, reuse):
        activation = self.__leaky_relu
#        momentum = 0.99
        with tf.variable_scope("discriminator", reuse=reuse):
            x = tf.reshape(img_in, shape=[-1]+self.image_shape)
            x = tf.layers.conv2d(x, kernel_size=5, filters=64, strides=2, padding='same', activation=activation)
#            x = tf.contrib.layers.batch_norm(x, is_training=self.is_training, decay=momentum)
            x = tf.layers.dropout(x, self.keep_prob)
            x = tf.layers.conv2d(x, kernel_size=5, filters=128, strides=2, padding='same', activation=activation)
#            x = tf.contrib.layers.batch_norm(x, is_training=self.is_training, decay=momentum)
            x = tf.layers.dropout(x, self.keep_prob)
            x = tf.layers.conv2d(x, kernel_size=5, filters=256, strides=2, padding='same', activation=activation)
#            x = tf.contrib.layers.batch_norm(x, is_training=self.is_training, decay=momentum)
            x = tf.layers.dropout(x, self.keep_prob)
            x = tf.layers.conv2d(x, kernel_size=5, filters=512, strides=2, padding='same', activation=activation)
#            x = tf.contrib.layers.batch_norm(x, is_training=self.is_training, decay=momentum)
            x = tf.layers.dropout(x, self.keep_prob)
            x = tf.layers.conv2d(x, kernel_size=5, filters=1024, strides=2, padding='same', activation=activation)
#            x = tf.contrib.layers.batch_norm(x, is_training=self.is_training, decay=momentum)
#            x = tf.layers.dropout(x, self.keep_prob)
            x = tf.contrib.layers.flatten(x)
#            x = tf.layers.dense(x, units=128, activation=activation)
            x = tf.layers.dense(x, units=1, activation=tf.nn.sigmoid)
            return x
    
    def train_on_batch(self, batch,learning_rate=0.00015):
        keep_prob_train = 0.5
        train_generator = True
        train_discriminator = True
        
        noise = np.random.uniform(0.0, 1.0, [batch.shape[0], self.noise_dimensions]).astype(np.float32)
        
        g_ls, d_ls, p_reals, p_fakes, imgs = self.sess.run([self.generator_loss, self.discriminator_loss, self.avg_prob_of_reals, self.avg_prob_of_fakes, self.X_generated], feed_dict={self.X_in: batch, self.noise: noise, self.keep_prob: keep_prob_train, self.is_training:True})
        
#        print('During training:')
#        print('min: {}'.format(np.min(imgs)))
#        print('mean: {}'.format(np.mean(imgs)))
#        print('max: {}'.format(np.max(imgs)))
        print('Average estimated probability of reals being real: {}'.format(p_reals))
        print('Average estimated probability of fakes being real: {}'.format(p_fakes))
#        
#        print('Generator loss: {}'.format(g_ls))
#        print('Discriminator loss: {}'.format(d_ls))
        
        if g_ls * 1.5 < d_ls:
            train_generator = False
            print('Not training generator')
        elif d_ls * 1.5 < g_ls:
            train_discriminator = False
            print('Not training discriminator')
            
        if train_discriminator:
            self.sess.run([self.discriminator_optimizer], feed_dict={self.noise: noise, self.X_in: batch, self.keep_prob: keep_prob_train, self.is_training: True, self.learning_rate: learning_rate})
        if train_generator:
            self.sess.run([self.generator_optimizer], feed_dict={self.noise: noise, self.X_in: batch, self.keep_prob: keep_prob_train, self.is_training: True, self.learning_rate: learning_rate})
    
    def train(self, examples, epochs=1, batch_size=128, learning_rate=0.00015):
        keep_prob_train = 0.5
        for e in range(epochs):
            for b in range(0,examples.shape[0],batch_size):
                train_generator = True
                train_discriminator = True
                
                batch = examples[b:b+batch_size, :, :, :]
                noise = np.random.uniform(0.0, 1.0, [batch_size, self.noise_dimensions]).astype(np.float32)
                
                g_ls, d_ls = self.sess.run([self.generator_loss, self.discriminator_loss], feed_dict={self.X_in: batch, self.noise: noise, self.keep_prob: keep_prob_train, self.is_training:True})
                
                if g_ls * 1.5 < d_ls:
                    train_generator = False
                if d_ls * 2 < g_ls:
                    train_discriminator = False
                    
                if train_discriminator:
                    self.sess.run([self.discriminator_optimizer], feed_dict={self.noise: noise, self.X_in: batch, self.keep_prob: keep_prob_train, self.is_training:True, self.learning_rate: learning_rate})
                if train_generator:
                    self.sess.run([self.generator_optimizer], feed_dict={self.noise: noise, self.keep_prob: keep_prob_train, self.is_training: True, self.learning_rate: learning_rate})
    
    def generate(self, nbr_of_imgs=64, noise=None):
        if noise is None:
            noise = np.random.uniform(0.0, 1.0, [nbr_of_imgs, self.noise_dimensions]).astype(np.float32)
        
#        generated_imgs = self.sess.run(self.X_generated, feed_dict = {self.noise: noise, self.keep_prob: 1.0, self.is_training:False})
        [generated_imgs] = self.sess.run([self.X_generated], feed_dict={self.noise: noise, self.keep_prob: 1.0, self.is_training: True})
#        print('During generation:')
#        print('min: {}'.format(np.min(generated_imgs)))
#        print('mean: {}'.format(np.mean(generated_imgs)))
#        print('max: {}'.format(np.max(generated_imgs)))
        return generated_imgs