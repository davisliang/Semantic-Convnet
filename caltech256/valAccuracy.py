import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
import cPickle as pickle
from scipy import spatial
from scipy import misc

# Load features
valData = np.load('caltech250ValData.npy')
# Load class labels
valClass = np.load('caltech250ValClass.npy')
# Load {label:word embedding vector} dictionary
caltechDict = pickle.load(open('caltech256Dict.pkl'))
# Load trained model
tl_model = load_model('test.hd5')

# Get validation set labels from cosine similarity
classTargets = [caltechDict[key] for key in caltechDict]
classLabels = [key for key in caltechDict]
predictions = tl_model.predict(valData, batch_size = 100, verbose = 1)
dist = [[spatial.distance.cosine(x,y) for y in classTargets] for x in predictions]
orderedInd = np.argsort(dist,1)

for k in range(1,6):
    ind = orderedInd[:,0:k]
    #ind = np.argmin(dist,1)
    predictedLabel = [[classLabels[top] for top in x] for x in ind]
    #print(len(predictedLabel))
    #print(predictedLabel[0].shape)

    # Calculate error rate
    errorCounts = {key:0 for key in caltechDict}
    errorDict = []
    correct = 0
    for i,label in enumerate(valClass):
	if valClass[i] in predictedLabel[i]:	
	    correct += 1
	'''
	else:
	    errorCounts[valClass[i]] += 1
	    #errorDict.append((valData[i,:,:,:],predictedLabel[i]))
	    misc.imsave('semanticErrorPics/' + predictedLabel[i] + '_' + str(i) + '.png', valData[i,:,:,:])
	'''
    print('Top ' + str(k) + ' Accuracy: ' + str(float(correct)/valClass.shape[0]))
    #print('Misclassifications per class: ')
    #print(errorCounts)
    '''
    f = open('semanticConvnetErrors.pkl','wb')
    pickle.dump(errorDict,f)
    f.close()
    '''
