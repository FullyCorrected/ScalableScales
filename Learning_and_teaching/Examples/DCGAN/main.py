import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tensorflow.examples.tutorials.mnist import input_data

from dcgan import DCGAN

def plot(samples, idn=1):
    fig = plt.figure(idn,figsize=(3, 3))
    gs = gridspec.GridSpec(3, 3)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate([cv2.resize(sample, (512, 512)) for sample in samples]):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample, cmap='Greys_r')

    return fig

print('Loading dataset')
mnist = input_data.read_data_sets('../MNIST', one_hot=True)
X_flat, _ = mnist.train.next_batch(110000)
X = []
for i in range (X_flat.shape[0]):
    img = np.reshape(X_flat[i,:], [28, 28, 1])
    reshaped_img = cv2.resize(img, (32, 32)) 
    X.append(reshaped_img)

X = np.expand_dims(np.array(X), axis=-1)

print('Building GAN model')
nn = DCGAN()

print('Training GAN')
epochs = 1000000
batch_size = 128
iteration = 0
for e in range(epochs):
    print('Epoch {}/{}'.format(e+1,epochs))
    print('-------------------------------------------')
    np.random.shuffle(X)
    for b in range(0, X.shape[0], batch_size):
        iteration += 1
        print('Batch {}/{}'.format(b//batch_size + 1, X.shape[0] // batch_size + 1))
        X_batch = X[b:b+batch_size,:,:]
        nn.train_on_batch(X_batch)
        if iteration % 10 == 0:
            image_examples = nn.generate(nbr_of_imgs=9)
            fig = plot(image_examples, 1)
            plt.show()
            plt.pause(0.05)
#            fig = plot(X_batch[0:9,:,:,:], 2)
#            plt.show()
#            plt.pause(0.05)
    print('-------------------------------------------')
