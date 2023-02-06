import os
import math
import time
import pickle
import numpy as np
import configparser
import random
import matplotlib.pyplot as plt


from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator

from helpers import load_hdf5
from allofthenet import myalexnet,vgg,InceptionResNetV2,ResNet,ResNet_LRN,MyNetList

#-----------------------------------------------------------------------------
# Exponential Decay of learning rate


def exp_decay(epoch):
   initial_lrate = 0.1
   k = 0.1
   lrate = initial_lrate * math.exp(-k*epoch)
   return lrate


#-----------------------------------------------------------------------------
#read config file
time1 = time.time()

config = configparser.RawConfigParser()
config.read('../configuration.txt')

net_type = config.get('CNN types','net_type')
num_epochs = int(config.get('training settings', 'num_epochs'))
batch_size = int(config.get('training settings', 'batch_size'))


num_classes = int(config.get('data attributes','num_classes'))
patch_size = int(config.get('data attributes','patch_size'))

train_patches = config.get('data paths','train_patches')
train_labels = config.get('data paths','train_labels')
best_weights = config.get('data paths','best_weights')

history_saved = config.get('data paths','history_saved')

augment = config.get('training settings', 'augment')

#-----------------------------------------------------------------------------

#load and transform data
os.environ["CUDA_VISIBLE_DEVICES"]="1"

X_train = load_hdf5(train_patches)
Y_train = load_hdf5(train_labels)
Y_train = to_categorical(Y_train)

s = np.arange(X_train.shape[0])
np.random.shuffle(s)
X_train = X_train[s]
Y_train = Y_train[s]

#get model and train it
cfg = {'image_size' : patch_size,'channel_no' : 3,'class_no': num_classes}

model = MyNetList[net_type](cfg)

#model = alexnet(cfg)

model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
    
checkpointer = ModelCheckpoint(best_weights, verbose=1, monitor='val_loss', mode='auto', save_best_only=True) #save at each epoch if the validation decreased
patienceCallBack = EarlyStopping(monitor='val_loss',patience=50)
learningRateCallBack = LearningRateScheduler(exp_decay ,verbose = 1)
tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True, profile_batch = 100000000)

batch_size = int(batch_size)


if augment == 'True':
    
    print('Train with augmentation')
    
    datagen = ImageDataGenerator(
                            rotation_range=10,
                            horizontal_flip=True,
                            vertical_flip=True,
                            fill_mode='nearest')
    
    validation_split = 0.2
    indx = list(range(0,len(X_train)))
    random.shuffle(indx)
    random.shuffle(indx)

    k = int(0.2*len(X_train))

    X_val = X_train[indx[0:k]]
    Y_val = Y_train[indx[0:k]]

    X_train = X_train[indx[k:len(X_train)]]
    Y_train = Y_train[indx[k:len(Y_train)]]


    history = model.fit(datagen.flow(x = X_train, y = Y_train, batch_size = batch_size),
                                  #validation_data = datagen.flow(x = X_val, y = Y_val, batch_size = batch_size),
                                  validation_data = (X_val, Y_val),
                                  steps_per_epoch = len(X_train)/batch_size,
                                  epochs = num_epochs,
                                  #callbacks = [checkpointer,tbCallBack,patienceCallBack])                  
                                  callbacks = [checkpointer,patienceCallBack])        
else:
    
    print('No augmentation')
    
    history = model.fit(x=X_train, 
                    y=Y_train, 
                    validation_split=0.2, 
                    epochs = num_epochs, 
                    batch_size=batch_size, 
                    shuffle=True, 
                    callbacks = [checkpointer,tbCallBack,patienceCallBack])

#-----------------------------------------------------------------------------
#plot train history
print("Times take: %.3fs"%(time.time()-time1))
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc',color = "r")
plt.plot(epochs, val_acc, 'b', label='Validation acc',color = "magenta")
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss',color = "green")
plt.plot(epochs, val_loss, 'b', label='Validation loss',color = "cyan")
plt.title('Training and validation loss')
plt.legend()
plt.show()

pickle.dump(history, open(history_saved, "wb"))