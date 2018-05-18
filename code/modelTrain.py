# Larger CNN for the MNIST Dataset
import numpy as np
import keras
from keras.models import load_model
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from IPython.display import clear_output

#%%
seed = 7
np.random.seed(seed)
#%%

# load data

train_path = r"../Data/train"
valid_path = r"../Data/valid"
test_path  = r"../Data/test"

train_batches = ImageDataGenerator().flow_from_directory(train_path,target_size=(28,28),batch_size=10)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path,target_size=(28,28),batch_size=10)
test_batches = ImageDataGenerator().flow_from_directory(test_path,target_size=(28,28),batch_size=10)

label_map = (train_batches.class_indices)

#%%
""" Functions Area """

def saveModel(model,arcPath,weightPath):
    from keras.models import model_from_json

    model_json = model.to_json()
    with open(arcPath, "w") as json_file:
        json_file.write(model_json)
    
    #model.save_weights(weightPath)
    print("Saved model to ",weightPath)
    print("Saved model layout to ",arcPath)
    model.save(weightPath)
	#model.save(weightPath)

def createModel(input_shape,nClasses):
    
    model = Sequential()
    
    model.add(Conv2D(28, (3, 3), padding='same', activation='relu', input_shape=input_shape))
#    model.add(Conv2D(28, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(56, (3, 3), padding='same', activation='relu'))
#    model.add(Conv2D(112, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(112, (3, 3), padding='same', activation='relu'))
#    model.add(Conv2D(224, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Flatten())
    model.add(Dense(224, activation='relu'))
#    model.add(Dropout(0.5))
    model.add(Dense(nClasses, activation='softmax'))
    
    model.compile(Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])     
    return model
    
def MODEL(input_shape,nClasses):
    
    model = Sequential()
    
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
 
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
 
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(nClasses, activation='softmax'))
    
    model.compile(Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])     
    return model
    
    
class PlotLosses(keras.callbacks.Callback):
    
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []            
        self.fig = plt.figure()            
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show();

class PlotLearning(keras.callbacks.Callback):
    
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
        
        clear_output(wait=True)
        
        ax1.set_yscale('log')
        ax1.plot(self.x, self.losses, label="loss")
        ax1.plot(self.x, self.val_losses, label="val_loss")
        ax1.legend()
        
        ax2.plot(self.x, self.acc, label="accuracy")
        ax2.plot(self.x, self.val_acc, label="validation accuracy")
        ax2.legend()
        
        plt.show();
    
    
#%%
# build the model
plot_losses = PlotLosses()
plot = PlotLearning()

model = createModel((28,28,3),len(label_map))
model2 = MODEL((28,28,3),len(label_map))

# Fit the model
model.fit_generator(train_batches,validation_data=valid_batches,verbose=2,epochs=30,callbacks=[plot])
#,steps_per_epoch=30,validation_steps=10, #,use_multiprocessing=True
model2.fit_generator(train_batches,validation_data=valid_batches,verbose=2,epochs=30,callbacks=[plot])

# Final evaluation of the model
scores = model.evaluate_generator(test_batches)
print("Error: %.7f%%" % (100-scores[1]*100))  

saveModel(model,r'../modelArc.json',r'../MODEL.h5')
