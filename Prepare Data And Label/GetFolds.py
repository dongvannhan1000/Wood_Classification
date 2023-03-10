import os
import pickle
import configparser
from PIL import Image


from skimage import io
from skimage.transform import rescale, resize, downscale_local_mean

#----------------------------------------------------------------------------
#read config file
config = configparser.RawConfigParser()
config.read(r'../configuration.txt')

#----------------------------------------------------------------------------
#data location info
main_dir = config.get('data paths', 'data_dir')
org_data_dir = main_dir + 'org'

#test fold info
num_folds = int(config.get('general settings','total_folds_no'))

#----------------------------------------------------------------------------
#create fold_folders with subfolders

for i in range(1, num_folds + 1):
    
    fold_dir_path = main_dir + 'fold_' + str(i)
    #print(fold_dir_path)
    
    if not os.path.isdir(fold_dir_path):    
        os.mkdir(fold_dir_path)
    else:
        print('Direcory: ' + fold_dir_path + ' already exists')        
    
    for dirpath, dirnames, filenames in os.walk(org_data_dir):
        structure = fold_dir_path + '/' + dirpath[len(org_data_dir)+1:]
        print(structure)
        
        if not os.path.isdir(structure):
            os.mkdir(structure)
        else:
            print('Folder: ' + structure + ' does already exits!')

#----------------------------------------------------------------------------
#divide data into folds + generate dictionary label:spiecies_name

dict_labels = dict()
dict_names = dict()

l = 0;

for dirname, dirnames, files in os.walk(org_data_dir):
        
        for subdir in dirnames:
            
            num = 0
            
            print('Processing subdirectory: ' + subdir)
            
            dict_labels.update({l:subdir})
            dict_names.update({subdir:l})
            
            l = l + 1
            
            f = os.listdir(os.path.join(dirname, subdir))
            
            for files in f:
                
                fold_id = 1 + num % num_folds    
                print (files + ' fold: ' + str(fold_id)) 
                
                path_src = os.path.join(os.path.join(dirname, subdir), files)
                img = io.imread(path_src)
                path_dest = os.path.join(os.path.join(os.path.join(main_dir, 'fold_' + str(fold_id)), subdir), files)

                #Important note: That's how I reshape image to fit the features of logs, grains, wood, etc,... to the dataset.
                #If you guys don't want to do that, probably your net will have a worse result.
                img = resize(img, (img.shape[0] // 4, img.shape[1] // 4),anti_aliasing=True)
                

                io.imsave(path_dest,img)
                
                print('src: ' + path_src)
                print('dest: ' + path_dest)
                
                num = num + 1       
                
pickle.dump(dict_labels, open(r'../Weights and Misc/dictLabels.p', "wb"))
pickle.dump(dict_names, open(r'../Weights and Misc/dictNames.p', "wb"))