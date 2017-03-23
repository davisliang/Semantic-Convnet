import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
import cPickle as pickle
from scipy import spatial
from scipy import misc
import numpy as np
import functools

# Load features
valData = np.load('caltech250ValData.npy')
# Load class labels
valClass = np.load('caltech250ValClass.npy')
# Load {label:word embedding vector} dictionary
caltechDict = pickle.load(open('caltech256Dict.pkl'))
# Load trained model
tl_model = load_model('vanillaModel.hd5')

classTargets = [caltechDict[key] for key in caltechDict]
classLabels = [key for key in caltechDict]
ind = np.argsort(classLabels)
classLabels.sort()
classTargets = [classTargets[i] for i in ind]
#print(classLabels)

predictions = tl_model.predict(valData, batch_size = 100, verbose=1)

orderedInd = np.argsort(predictions,1)
#print(ind.shape)
#print(predictions.shape)
for k in range(1,6):
    topK = orderedInd[:,-k:]

    predictedLabel = [[classLabels[top] for top in x] for x in topK]

    correct = 0

    for i,label in enumerate(valClass):
	if valClass[i] in predictedLabel[i]:
	    correct += 1 

    print('Top ' + str(k) + ' Validation Accuracy: ' + str(float(correct)/valClass.shape[0]))
