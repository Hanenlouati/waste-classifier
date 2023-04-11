from keras.layers import Input, Dense, Flatten, Dropout, Activation 
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from keras.preprocessing import image
from keras.models import load_model
from keras.models import Model
import matplotlib.pyplot as plt
import glob, os, cv2, random,time
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D,Flatten,MaxPooling2D,Dense 
from keras.optimizers import SGD
from keras.applications.vgg16 import VGG16

def processing_data(data_path):
   
    train_data = ImageDataGenerator(
            
            rescale=1. / 225,  
            
            shear_range=0.1,  
            
            zoom_range=0.1,
           
            width_shift_range=0.1,
           
            height_shift_range=0.1,
            
            horizontal_flip=True,
            
            vertical_flip=True,
            
            validation_split=0.1  
    )

    
    validation_data = ImageDataGenerator(
            rescale=1. / 255,
            validation_split=0.1)

    train_generator = train_data.flow_from_directory(
           
            data_path, 
            
            target_size=(150, 150),
            
            batch_size=16,
            
            class_mode='categorical',
            
            subset='training', 
            seed=0)
    validation_generator = validation_data.flow_from_directory(
            data_path,
            target_size=(150, 150),
            batch_size=16,
            class_mode='categorical',
            subset='validation',
            seed=0)

    return train_generator, validation_generator

def model(train_generator, validation_generator, save_model_path):
    
    vgg16_model = VGG16(weights='imagenet',include_top=False, input_shape=(150,150,3))
    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))
    top_model.add(Dense(256,activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(6,activation='softmax'))

    model = Sequential()
    model.add(vgg16_model)
    model.add(top_model)
    # 编
    model.compile(
             
            optimizer=SGD(lr=1e-3,momentum=0.9),
            
            loss='categorical_crossentropy',
            
            metrics=['accuracy'])

    model.fit_generator(
           
            generator=train_generator,
            
            epochs=200,
           
            steps_per_epoch=2259 // 16,
           
            validation_data=validation_generator,
           
            validation_steps=248 // 16,
            )
    model.save(save_model_path)

    return model

def evaluate_mode(validation_generator, save_model_path):
    
    model = load_model('results/knn.h5')
    
    loss, accuracy = model.evaluate_generator(validation_generator)
    print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy * 100))

def predict(img):
   
    img = img.resize((150, 150))
    img = image.img_to_array(img)
   
    model_path = 'results/knn.h5'
    try:
      
        model_path = os.path.realpath(__file__).replace('main.py', model_path)
    except NameError:
        model_path = './' + model_path
    
   
    model = load_model(model_path)
    

    x = np.expand_dims(img, axis=0)

    
    y = model.predict(x)

   
    labels = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}

    # -------------------------------------------------------------------------
    predict = labels[np.argmax(y)]

    
    return predict

def main():
   
    data_path = "./datasets/la1ji1fe1nle4ishu4ju4ji22-momodel/dataset-resized"  # 数据集路径
    save_model_path = 'results/knn.h5'  
    
    train_generator, validation_generator = processing_data(data_path)
   
    model(train_generator, validation_generator, save_model_path)
   
    evaluate_mode(validation_generator, save_model_path)


if __name__ == '__main__':
    main()
