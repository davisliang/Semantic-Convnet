import os
import PIL
from PIL import Image
import numpy as np
from scipy import misc
import cPickle as pickle

imageDir = 'resizedCaltechVal'
categories = os.listdir(imageDir)
train = np.zeros((1615,224,224,3))
trainLabel = np.zeros((1615,300))
trainClass = ['']*1615

caltechDict = pickle.load(open('caltech95Dict.pkl'))

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

np.save('caltechVal.npy',train)
np.save('caltechValLabel.npy',trainLabel)
np.save('caltechValClass.npy',trainClass) 
