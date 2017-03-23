import os
import PIL
from PIL import Image
import numpy as np
from scipy import misc
import cPickle as pickle

numSamples = 5999 

imageDir = 'resizedCaltech250Val'
categories = os.listdir(imageDir)
train = np.zeros((numSamples,224,224,3))
trainLabel = np.zeros((numSamples,300))
trainClass = ['']*numSamples

caltechDict = pickle.load(open('caltech256Dict.pkl'))

i = 0
for category in categories:
    imgs = os.listdir(imageDir + '/' + category)
    imgs.sort()
    print(category)
    for image in imgs:
        sample  = misc.imread(imageDir + '/' + category + '/' + image)
        if sample.ndim  == 2:
	    train[i,:,:,0] = sample
	    train[i,:,:,1] = sample
	    train[i,:,:,2] = sample
	else:
	    train[i,:,:,:] = sample
	trainLabel[i,:] = caltechDict[category]
	trainClass[i] = category
	i += 1

np.save('caltech250ValData.npy',train)
np.save('caltech250ValWordvec.npy',trainLabel)
np.save('caltech250ValClass.npy',trainClass) 
