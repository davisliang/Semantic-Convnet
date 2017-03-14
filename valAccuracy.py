import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
import cPickle as pickle
from scipy import spatial

# Load features
valData = np.load('caltechVal.npy')
# Load class labels
valClass = np.load('caltechValClass.npy')
# Load {label:word embedding vector} dictionary
caltechDict = pickle.load(open('caltech95Dict.pkl'))
# Load trained model
tl_model = load_model('semanticModel.hd5')

# Get validation set labels from cosine similarity
classTargets = [caltechDict[key] for key in caltechDict]
classLabels = [key for key in caltechDict]
predictions = tl_model.predict(valData, batch_size = 100, verbose = 1)
dist = [[spatial.distance.cosine(x,y) for y in classTargets] for x in predictions]
ind = np.argmin(dist,1)
predictedLabel = [classLabels[x] for x in ind]

# Calculate error rate
correct = 0
for i,label in enumerate(valClass):
    if valClass[i] == predictedLabel[i]:	
	correct += 1
print('Accuracy: ' + str(float(correct)/valClass.shape[0]))

