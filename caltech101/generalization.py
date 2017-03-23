import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
import cPickle as pickle
from scipy import spatial
from scipy import misc
import os 
import PIL
from PIL import Image

k = 5

newDict = pickle.load(open('/home/ubuntu/caltech256/caltech256Dict.pkl'))
tl_model = load_model('nadamModel.hd5')

newImgDir = '/home/ubuntu/caltech256/resizedCaltech250Train/'
newClasses = os.listdir(newImgDir)
#newClasses = ['teddy_bear','gun']
caltech101Classes = os.listdir('/home/ubuntu/caltech101/resizedCaltechTrain/')

for newClass in newClasses:
    if newClass not in caltech101Classes:
	newClassDir = newImgDir + newClass
	newClassImgs = os.listdir(newClassDir)

	newData = np.zeros((len(newClassImgs),224,224,3))
	for i,image in enumerate(newClassImgs):
	    newImg = misc.imread(newImgDir + newClass + '/' + image)
	    if newImg.ndim == 2:
		newData[i,:,:,0] = newImg
		newData[i,:,:,1] = newImg
		newData[i,:,:,2] = newImg
	    else:
		newData[i,:,:,:] = newImg

	newLabel = newDict[newClass] # word vector for snake

	# Load {label:word embedding vector} dictionary
	caltechDict = pickle.load(open('caltech95Dict.pkl'))

	# Get validation set labels from cosine similarity
	classTargets = [caltechDict[key] for key in caltechDict]
	classLabels = [key for key in caltechDict]
	classTargets.append(newLabel)
	classLabels.append(newClass)

	predictions = tl_model.predict(newData, batch_size = 100, verbose = 1)
	dist = [[spatial.distance.cosine(x,y) for y in classTargets] for x in predictions]
	ind = np.argsort(dist,1)
	ind = ind[:,0:k]
	#ind = np.argmin(dist,1)
	predictedLabels = [[classLabels[top] for top in x] for x in ind]
	#print(predictedLabels)
	#print(predictedLabel[0].shape)

	# Calculate error rate
	errorCounts = {key:0 for key in caltechDict}
	errorDict = []
	correct = 0
	for i in range(len(newClassImgs)):
	    print(predictedLabels[i])
	    if newClass in predictedLabels[i]:	
		correct += 1

	print(newClass + ' Accuracy: ' + str(float(correct)/len(newClassImgs)))
	'''
	    else:
		errorCounts[valClass[i]] += 1
		#errorDict.append((valData[i,:,:,:],predictedLabel[i]))
		misc.imsave('semanticErrorPics/' + predictedLabel[i] + '_' + str(i) + '.png', valData[i,:,:,:])
	    print('Accuracy: ' + str(float(correct)/len(newImages))
	#print('Misclassifications per class: ')
	#print(errorCounts)
	f = open('semanticConvnetErrors.pkl','wb')
	pickle.dump(errorDict,f)
	f.close()
	'''
