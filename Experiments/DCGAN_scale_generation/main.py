import cv2
import os
import glob
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
        if sample.shape[-1] == 1:
            plt.imshow(sample[:,:,0], cmap='Greys_r')
        else:
            plt.imshow(sample)

    return fig

print('Loading dataset')
#img_dir = "./Images_of_scales"
#data_path = os.path.join(img_dir,'*g')
#files = glob.glob(data_path)
#data = []
#for i, file in enumerate(files):
#    raw_image = cv2.imread(file)
#    if raw_image is None:
#        print(str(file))
#        continue
#    
#    normalized_image = raw_image / 255
#    smallest_dim_size = np.min(normalized_image.shape[0:-1])
#    cropped_image = normalized_image[0:smallest_dim_size,0:smallest_dim_size,:]
#    resized_image = cv2.resize(cropped_image, (128, 128)) 
#    data.append(resized_image)
#    
#X = np.array(data)
#data_to_save = {'X': X}
#output = open('X.pkl', 'wb')
#pickle.dump(data_to_save, output)
#output.close()

pkl_file = open('X.pkl', 'rb')
data_to_load = pickle.load(pkl_file)
X = data_to_load['X']
pkl_file.close()

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
        if iteration % 1 == 0:
            image_examples = nn.generate(nbr_of_imgs=9)
            fig = plot(image_examples, 1)
            plt.show()
            plt.pause(0.05)
#            fig = plot(X_batch[0:9,:,:,:], 2)
#            plt.show()
#            plt.pause(0.05)
    print('-------------------------------------------')
