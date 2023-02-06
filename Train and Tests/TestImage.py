import os
import pickle
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import configparser
from skimage import io

from keras.models import model_from_json
from helpers import plot_confusion_matrix, imagePreprocessing, BarChart

from allofthenet import myalexnet,vgg,InceptionResNetV2,ResNet,ResNet_LRN,MyNetList

#-----------------------------------------------------------------------------
#read config file

config = configparser.RawConfigParser()
config.read(r'../configuration.txt')


net_type = config.get('CNN types','net_type')
num_classes = int(config.get('data attributes','num_classes'))
patch_size = int(config.get('data attributes','patch_size'))

test_patches = config.get('data paths','test_patches')
test_labels = config.get('data paths','test_labels')
best_weights = config.get('data paths','best_weights')


#pred_dir = config.get('data paths', 'preds_dir')
test_dir = config.get('data paths', 'test_dir')

conf_mat_images_saved = config.get('data paths','conf_mat_images_saved')
#-----------------------------------------------------------------------------    
#load CNN model

cfg = {'image_size' : patch_size,'channel_no' : 3,'class_no': num_classes}

model = MyNetList[net_type](cfg,verbose = 0)
model.load_weights(best_weights)
Top1ChartFlag =1
Nottop1Chartflag = 1

#-----------------------------------------------------------------------------   
# predict image patch by patch
    
def predictImage(img, patchSize,numClasses):
    global model
    stride = 120 #nên nhỏ hơn một chút.
    
    
    [rows, cols, nch] = img.shape
    
    num = (1 + (rows - patchSize)  / stride) * (1 + (cols - patchSize)  / stride)  
    num = int(num+1)
    
    X_test = np.zeros((num, patchSize, patchSize, nch))
    probs = np.zeros((numClasses,1))
    time_1 = time.time()
    y = 0
    patchId = 0;
    
    while y+patchSize < rows:
       
        x = 0
    
        while x+patchSize < cols:
                        
            patch = img[y:y+patchSize,x:x+patchSize]
            X_test[patchId] = patch
            patchId += 1
            x += stride

        y += stride

    
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis = 1) 

    y = 0
    patchId = 0;
     
    while y+patchSize < rows:
       
        x = 0
    
        while x+patchSize < cols:
            probs[y_pred[patchId]] +=1
            patchId += 1
            x += stride

        y += stride
    
    probs = probs/patchId
    fin = np.argmax(probs)
    print("Number of patches in this image: %d,Size = (%d,%d)"%(patchId,rows,cols))
    tempprobs = []
    for i in probs:
        tempprobs.append(i[0])
    print(tempprobs)
    print("Predict time: %.3fs"%(time.time() - time_1))

    return fin , tempprobs

#-----------------------------------------------------------------------------  
# predict files in test directory
def get_key(my_dict,val):
    for key, value in my_dict.items():
         if val == value:
             return key
 
    return "key doesn't exist"

dict_labels = pickle.load(open(r'../Weights and Misc/dictLabels.p', "rb"))
dict_names = pickle.load(open(r'../Weights and Misc/dictNames.p', "rb"))

confMat = np.zeros((num_classes,num_classes))
confMat_top3 = np.zeros((num_classes,num_classes))

wrong_guess_list = []
names = list()
for i in range(0, num_classes):
    names.append(dict_labels[i])
#print(names)

for dirname, dirnames, files in os.walk(test_dir):
    
    for subdir in dirnames:
        
        print(subdir)
            
        classId = dict_names[subdir]
            
        f = os.listdir(os.path.join(dirname, subdir))
        
        for file in f:
            probs = []
            imgpath = os.path.join(dirname,subdir, file)
            img = io.imread(imgpath)
            img = img[:,:,0:3]
            print(imgpath)
            
            img = imagePreprocessing(img)
            

            label , probs = predictImage(img, patch_size, num_classes)

            confMat[classId,label] +=1
            
            checktop3 = sorted(probs,reverse = True)[:3]
            label_top3 = label
            if (probs[label_top3] in checktop3) & (probs[classId] in checktop3):
                label_top3 = classId 
            confMat_top3[classId,label_top3] += 1

            if (Nottop1Chartflag | Top1ChartFlag == 0):
                plt.show()

            if classId != label:

                #print("Check for top 3:",checktop3 , probs[label_top3] ,probs[classId])
                print('!!!!!!!!!!!!!')
                print("True = %s != Guess = %s"% (get_key(dict_names,classId),get_key(dict_names,label)))
                print('!!!!!!!!!!!!!')
                wrong_guess_list.append(imgpath)
                if Nottop1Chartflag:
                    valcount = 0
                    for i in range(num_classes):
                        if probs[i] > 0:
                            valcount+=1
                    if valcount > 5:
                        if random.randint(1,6) == 1:
                            BarChart(img,classId,probs,names)
                            Nottop1Chartflag = 0
            else:
                if Top1ChartFlag:
                    valcount = 0
                    for i in range(num_classes):
                        if probs[i] > 0:
                            valcount+=1
                    if valcount > 5:
                        if random.randint(1,11) == 10:
                            BarChart(img,classId,probs,names)
                            Top1ChartFlag = 0

for i in wrong_guess_list:
    print(i)
    


plot_confusion_matrix(confMat_top3, normalize=True, target_names = names,cmap= 'YlOrRd',title = 'AlexNet - Top 3 Accuracy')
plot_confusion_matrix(confMat, normalize=True, target_names = names,cmap= 'GnBu',title = 'AlexNet - Top 1 Accuracy')
plt.show()
pickle.dump(confMat, open(conf_mat_images_saved, "wb"))
