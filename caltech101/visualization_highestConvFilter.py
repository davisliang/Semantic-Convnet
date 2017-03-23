'''
Reference: https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
Modified by group, Feb 18th, 2017

Visualization of the filters of VGG16, via gradient ascent in input space.

This script can run on CPU in a few minutes (with the TensorFlow backend).

'''
from __future__ import print_function
#from scipy.misc import imsave
import numpy as np
import stdio
import time
from keras.applications import vgg16
from keras import backend as K
import scipy 
from keras.layers import Convolution2D, ZeroPadding2D
#import tensorflow as  tf
import matplotlib.pyplot as plt
from scipy import signal
import  pickle
from scipy import spatial
from PIL import Image 
#cirfar10_vec= pickle.load(open('caltech256.pkl', 'rb'),encoding='bytes')
Word2Vec=np.load('caltech95TrainWordvec.npy')
X_train=np.load('caltech95TrainData.npy')
vectors= Word2Vec[0:15]
images= X_train[0:15]
fp = fopen('vectors.txt', 'w');
fwrite(fp, vectors)
fclose(fp)
 


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

val1=199
val2=2000

img1.show()
gray_img1=rgb2gray(img1)
gray_img2=rgb2gray(img2)   
im1_prep=(gray_img1-np.mean(gray_img1))/np.std(gray_img1)
im2_prep=(gray_img2-np.mean(gray_img2))/np.std(gray_img2)
convolved= scipy.signal.correlate2d(im1_prep,im2_prep, mode='full', boundary='fill', fillvalue=0)
correlation_max=np.max(convolved)/corr_self



## cosine similarity
w2v_fetch=cifar10_vec[1]
vec_1=w2v_fetch[val1]
vec_2=w2v_fetch[val2]
cosin_sim= -(scipy.spatial.distance.cosine(vec_1, vec_2) - 1 )
