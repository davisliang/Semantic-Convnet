import keras
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
import matplotlib.pyplot as plt
from keras.callbacks import History,ModelCheckpoint

def getModel( output_dim ):
    ''' 
        * output_dim: the number of classes (int)
        
        * return: compiled model (keras.engine.training.Model)
    '''
    vgg_model = VGG16( weights='imagenet', include_top=True )
    vgg_out = vgg_model.layers[-2].output #Last FC layer's output  
    
    #Create softmax layer taking input as vgg_out
    softmax_layer = keras.layers.core.Dense(output_dim,
                          init='lecun_uniform',
                          activation='softmax')(vgg_out)
                          
    #Create new transfer learning model
    tl_model = Model( input=vgg_model.input, output=softmax_layer )
    
    #Freeze all layers of VGG16 and Compile the model
    for layers in vgg_model.layers:
        layers.trainable = False;
        
    tl_model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
              
    #Confirm the model is appropriate
    tl_model.summary()

    return tl_model

if __name__ == '__main__':
    #Output dim for your dataset
    output_dim = 95 #For Caltech97
    
    tl_model = getModel( output_dim ) 
    
    # Input data generator
    train_datagen = ImageDataGenerator(
        samplewise_center = True)
        
    train_generator = train_datagen.flow_from_directory(
        'caltech95Train/',
        target_size=(224, 224),
        batch_size=100,
        class_mode='categorical')
    
    test_datagen = ImageDataGenerator(
        samplewise_center = True)
        
    test_generator = test_datagen.flow_from_directory(
        'caltech95Val/',
        target_size=(224, 224),
        batch_size=100,
        class_mode='categorical')
    
    #Train the model
    saveModel = ModelCheckpoint('vanillaModel.hd5',
	verbose=1)

    history = tl_model.fit_generator(train_generator,
        samples_per_epoch = 6000,
        nb_epoch = 25,
        verbose = 1,
        validation_data = test_generator,
        nb_val_samples = 1600,
	callbacks = [saveModel])
    
    #Test the model
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
    
    
    
