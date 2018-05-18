import os
import keras
import numpy as np
from keras.models import load_model
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

#%%
def predictClass():
#%%    
#    os.chdir(r"C:\Users\sasaa\Desktop\FinalProjectAI\code")
    
    train_path = r"../Data/train"
    train_batches = ImageDataGenerator().flow_from_directory(train_path,target_size=(28,28),batch_size=10)
    label_map = (train_batches.class_indices)
    
    model = load_model(r"../MODEL.h5")
#%%

#    imgTest = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True).flow_from_directory(r'../TESTING',target_size=(28,28),batch_size=10)
    
    imgTest = ImageDataGenerator().flow_from_directory(r'../TESTING',target_size=(28,28),batch_size=10)
    
#    print(os.getcwd())

    pred_class = model.predict_classes(imgTest[0][0],batch_size=10,verbose=2)
    pred_prop = model.predict(imgTest[0][0],batch_size=10,verbose=2)

    #dd =[ [ [ (label_map[key],key,max(pred_prop[i]),imgTest.filenames[j].split('\\')[1]) for key in label_map if pred_class[i] == label_map[key] ] for i in range(len(pred_class)) ] for j in range(len(imgTest.filenames)) ] 

    predictions = [ (label_map[key],key,max(pred_prop[0])) for key in label_map if pred_class[0] == label_map[key]]
    #predSomeImgs = [ [ (label_map[key],key,max(pred_prop[i])) for key in label_map if pred_class[i] == label_map[key] ] for i in range(len(pred_class)) ]
    #predSomeImgs
    print(predictions)

    #pred_with_filenames = [(predictions[i],imgTest.filenames[i].split('\\')[1]) for i in range(len(predictions))]
    #pred_with_filenames
#%%
    return predictions[0] #Class Number