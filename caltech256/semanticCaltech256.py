import keras
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
import matplotlib.pyplot as plt
from keras.callbacks import History
import numpy as np
import os
import cPickle as pickle
import scipy
from scipy import spatial

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
        
    tl_model.compile(optimizer='Nadam',
              loss='cosine_proximity')
              
    #Confirm the model is appropriate
    tl_model.summary()

    return tl_model

if __name__ == '__main__':
    #Output dim for your dataset
    output_dim = 300 #For word2vec output  
       
    # Training parameters
    batchSize = 100 
    numEpochs = 15
    
    tl_model = getModel( output_dim ) 
    trainClass = np.load('caltech250TrainClass.npy')
    trainLabel = np.load('caltech250TrainWordvec.npy') 
    valData = np.load('caltech250ValData.npy')
    valClass = np.load('caltech250ValClass.npy')
    valLabel = np.load('caltech250ValWordvec.npy')

    # Input data generator
    train_datagen = ImageDataGenerator(
	featurewise_center = True)
        
    train_generator = train_datagen.flow_from_directory(
        'resizedCaltech250Train',
        target_size= (224,224),
        class_mode = 'sparse',
        batch_size = batchSize)
    
    test_datagen = ImageDataGenerator(
	featurewise_center = True)
        
    test_generator = test_datagen.flow_from_directory(
        'resizedCaltech250Val',
        target_size=(224,224),
        class_mode = 'sparse',
        batch_size = batchSize)

    train_datagen.fit(valData)
    test_datagen.fit(valData)
    
    caltechDict = pickle.load(open('caltech256Dict.pkl'))

    epoch = 0
    numImg = 0
    numImgsPerEpoch = 23954
    classLabels = os.listdir('./resizedCaltech250Train/')
    classTargets = [caltechDict[key] for key in caltechDict]
    classLabels = [key for key in caltechDict] 

    ind = np.argsort(classLabels)
    classLabels.sort()
    classTargets = [classTargets[i] for i in ind]

    print("Epoch 0")
    batchCount = 0	 
    for batch in train_generator:
        imgs = batch[0]
        labels = batch[1]

        wordEmbed = [np.asarray(caltechDict[classLabels[classInd]]) for classInd in labels]
        tl_model.train_on_batch(imgs,np.asarray(wordEmbed))
	#print(batchCount)
        numImg += batchSize
        batchCount += 1
        # Epoch checkpoint
        if numImg > numImgsPerEpoch:
	    tl_model.save('test.hd5')
            epoch += 1
            numImg = 0
            print('Epoch: ' + str(epoch))
        
            # Calculate validation loss after each epoch
            loss = 0
            imgCount = 0
            numbatch = 1
	    print("Calculating validation loss")


	    # Get validation set labels from cosine similarity
	    predictions = tl_model.predict(valData, batch_size = 100, verbose = 1)
	    #print('Validation Loss: ' + str(spatial.distance.cosine(predictions,valLabel)))
	    valLoss = [spatial.distance.cosine(predictions[j,:],valLabel[j,:]) for j in range(valClass.shape[0])]
	    print('Validation Loss: ' + str(sum(valLoss)/valClass.shape[0]))
	    dist = [[spatial.distance.cosine(x,y) for y in classTargets] for x in predictions]
	    ind = np.argmin(dist,1)
	    predictedLabel = [classLabels[x] for x in ind]

	    # Calculate error rate
	    correct = 0
	    for i,label in enumerate(valClass):
		if valClass[i] == predictedLabel[i]:
		    correct += 1

	    print('Validation Accuracy: ' + str(float(correct)/valClass.shape[0]))

	    print("Calculating Training Accuracy and Loss")
            t_preds = np.array([])
            t_targets = np.array([])
            t_numImg = 0
            t_batch_count = 1
            for t_batch in train_generator:

                #print('    train batch ' + str(t_batch_count))
                t_numImg += batchSize
                t_imgs = t_batch[0]

                t_batch_preds = tl_model.predict_on_batch(t_imgs)
                t_batch_targets = t_batch[1]

                #print(classLabels[t_batch_targets[0]])

                t_preds = np.vstack([t_preds, t_batch_preds]) if t_preds.size else t_batch_preds
                t_targets = np.hstack([t_targets, t_batch_targets]) if t_targets.size else t_batch_targets
                #print('    size ' + str(np.shape(t_targets)))

                t_batch_count += 1
                if t_numImg >= numImgsPerEpoch:
                        break

            dist = [[spatial.distance.cosine(x,y) for y in classTargets] for x in t_preds]
            ind = np.argmin(dist,1)
            predictedLabel = [classLabels[x] for x in ind]

            # Calculate Training Accuracy
            correct = 0
            #for i,label in enumerate(trainClass):
            for i in range(numImgsPerEpoch):
                #if trainClass[i] == predictedLabel[i]:
                if classLabels[t_targets[i]] == predictedLabel[i]:
                    correct += 1
            print('Training Accuracy: ' + str(float(correct)/trainClass.shape[0]))
	    
	    trainLoss = [spatial.distance.cosine(t_preds[sample,:],trainLabel[sample,:]) for sample in range(trainClass.shape[0])]
	    print('Training Loss: ' + str(sum(trainLoss)/trainClass.shape[0]))
	    '''
            for valBatch in test_generator:
                imgs = valBatch[0]
                labels = valBatch[1]
                wordEmbed = [np.asarray(caltechDict[classLabels[classInd]]) for classInd in labels]
                loss += np.sum(tl_model.test_on_batch(imgs,np.asarray(wordEmbed)))
                imgCount += batchSize
                if imgCount > 1:
                    print("Validation: " + str(loss/imgCount))
		    loss = 0
		    imgCount = 0
                    break
            '''
        if epoch >= numEpochs:
            break
            
        
    #Test the model
    '''
    plt.plot(history.history['acc'])
    plt.show()
    
    plt.figure()
    plt.plot(history.history['loss'])
    plt.show()
    
    plt.figure()
    plt.plot(history.history['val_acc'])
    plt.show()
    
    plt.figure()
    plt.plot(history.history['val_loss'])
    plt.show()
    
    epoch = 0
    numImg = 0
    classLabels = os.listdir('C:/Users/xsaardo/Desktop/Caltech97Train/')
    
    for batch in train_generator:
        imgs = batch[0]
        labels = batch[1]
        
        
    
        print(batch[0].shape)
        print(classLabels[np.argmax(batch[1][0,:])])
        img = np.reshape(batch[0][0,:,:,:],(224,224,3)).astype('uint8')
        print(img.shape)
        plt.imshow(img)
        plt.show()
        break;
        
        numImg += batchSize
        if numImg > numImgsPerEpoch:
            epoch += 1
        if epoch > numEpochs:
            break
        
        
    '''
