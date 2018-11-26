from tensoroflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Activation, Dropout, Flatten, BatchNormalization, Reshape, UpSampling2D, Conv2DTranspose
from tensorflow.keras.activations import LeakyReLU
from tensorflow.keras.optimizers import RMSprop

# Discriminator
Discriminator = Sequential()
depth = 64
dropout = 0.4
# In: 28 x 28 x 1, depth = 1
# Out: 14 x 14 x 1, depth=64
input_shape = (28, 28, 1)
Discriminator.add(Conv2D(depth*1, 5, strides=2, input_shape=input_shape, padding='same', activation=LeakyReLU(alpha=0.2)))
Discriminator.add(Dropout(dropout))
Discriminator.add(Conv2D(depth*2, 5, strides=2, padding='same',\
activation=LeakyReLU(alpha=0.2)))
Discriminator.add(Dropout(dropout))
Discriminator.add(Conv2D(depth*4, 5, strides=2, padding='same',\
activation=LeakyReLU(alpha=0.2)))
Discriminator.add(Dropout(dropout))
Discriminator.add(Conv2D(depth*8, 5, strides=1, padding='same',\
activation=LeakyReLU(alpha=0.2)))
Discriminator.add(Dropout(dropout))
# Out: 1-dim probability
Discriminator.add(Flatten())
Discriminator.add(Dense(1))
Discriminator.add(Activation('sigmoid'))

# Generator
Generator = Sequential()
dropout = 0.4
depth = 64+64+64+64
dim = 7
# In: 100
# Out: dim x dim x depth
Generator.add(Dense(dim*dim*depth, input_dim=100))
Generator.add(BatchNormalization(momentum=0.9))
Generator.add(Activation('relu'))
Generator.add(Reshape((dim, dim, depth)))
Generator.add(Dropout(dropout))
# In: dim x dim x depth
# Out: 2*dim x 2*dim x depth/2
Generator.add(UpSampling2D())
Generator.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
Generator.add(BatchNormalization(momentum=0.9))
Generator.add(Activation('relu'))
Generator.add(UpSampling2D())
Generator.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
Generator.add(BatchNormalization(momentum=0.9))
Generator.add(Activation('relu'))
Generator.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
Generator.add(BatchNormalization(momentum=0.9))
Generator.add(Activation('relu'))
# Out: 28 x 28 x 1 grayscale image [0.0,1.0] per pix
Generator.add(Conv2DTranspose(1, 5, padding='same'))
Generator.add(Activation('sigmoid'))

# Discriminator Model
optimizer = RMSprop(lr=0.0008, clipvalue=1.0, decay=6e-8)
Discriminator_model = Sequential()
Discriminator_model.add(Discriminator)
Discriminator_model.compile(loss='binary_crossentropy', optimizer=optimizer,\
metrics=['accuracy'])

# Adversarial Model
optimizer = RMSprop(lr=0.0004, clipvalue=1.0, decay=3e-8)
Adversarial_model = Sequential()
Adversarial_model.add(Generator())
Adversarial_model.add(Discriminator())
Adversarial_model.compile(loss='binary_crossentropy', optimizer=optimizer,\
metrics=['accuracy'])

# Training
images_train = self.x_train[np.random.randint(0,
self.x_train.shape[0], size=batch_size), :, :, :]
noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
images_fake = self.generator.predict(noise)
x = np.concatenate((images_train, images_fake))
y = np.ones([2*batch_size, 1])
y[batch_size:, :] = 0
d_loss = self.discriminator.train_on_batch(x, y)
y = np.ones([batch_size, 1])
noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
a_loss = self.adversarial.train_on_batch(noise, y)