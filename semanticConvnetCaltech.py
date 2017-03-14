import keras
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
import matplotlib.pyplot as plt
from keras.callbacks import History, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import os
import cPickle as pickle
import scipy

def getModel( output_dim ):
    ''' 
        * output_dim: the number of classes (int)
        
        * return: compiled model (keras.engine.training.Model)
    '''
    vgg_model = VGG16( weights='imagenet', include_top=True )
    vgg_out = vgg_model.layers[-2].output #Last FC layer's output  
    
    #Create softmax layer taking input as vgg_out
    regression_layer =  keras.layers.core.Dense(output_dim,
                          init='lecun_uniform')(vgg_out)
                          
    #Create new transfer learning model
    tl_model = Model( input=vgg_model.input, output=regression_layer)
    
    #Freeze all layers of VGG16 and Compile the model
    for layers in vgg_model.layers:
        layers.trainable = False;
        
    tl_model.compile(optimizer='rmsprop',
              loss='cosine_proximity')
              
    #Confirm the model is appropriate
    tl_model.summary()

    return tl_model

if __name__ == '__main__':
    #Output dim for your dataset
    output_dim = 300 #For word2vec output  
       
    # Training parameters
    batchSize = 100 
    
    tl_model = getModel( output_dim ) 
    
    trainData = np.load('caltechTrain.npy')
    trainLabel  = np.load('caltechTrainLabel.npy')
    valData = np.load('caltechVal.npy')
    valLabel  = np.load('caltechValLabel.npy')
	
    # Input data generator

    train_datagen = ImageDataGenerator(
        featurewise_center = True)
 
    train_generator = train_datagen.flow(
        trainData,
	trainLabel,
        batch_size = batchSize)
    
    val_datagen = ImageDataGenerator(
        featurewise_center = True)
 
    val_generator = val_datagen.flow(
        valData,
	valLabel,
        batch_size = batchSize)
    
    train_datagen.fit(trainData)
    val_datagen.fit(trainData)    
    
    caltechDict = pickle.load(open('caltech95Dict.pkl'))
    
    saveModel = ModelCheckpoint('currModel.hd5',
	verbose=1,
	save_best_only=True)
    
    reduceLR = ReduceLROnPlateau() 
    history = tl_model.fit_generator(train_generator,
	samples_per_epoch = trainData.shape[0],
   	nb_epoch = 25,
	verbose = 1,
	validation_data = val_generator,
	nb_val_samples = valData.shape[0],
	callbacks = [saveModel, reduceLR])

    #Test the model
    plt.figure()
    plt.plot(history.history['loss'])
    plt.show()
    
    plt.figure()
    plt.plot(history.history['val_loss'])
    plt.show()
    
   
