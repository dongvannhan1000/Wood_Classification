
import h5py
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def write_hdf5(arr,outfile):
  with h5py.File(outfile,"w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)

##
def load_hdf5(infile):
  with h5py.File(infile,"r") as f:  
    return f["image"][()]

##  
def mirrorImage(inputs):
    rows, cols, nch = inputs.shape
    outputs = np.zeros((rows,2*cols,nch), dtype='uint8')
    outputs[:,0:cols,:] = inputs
    outputs[:,cols:2*cols,:] = np.flip(inputs,1)
    
    return outputs

##
def imagePreprocessing(img):
    
    img = img/255
    # m = img.mean(axis=0)
    # s = img.std(axis=0)
    # img = (img-m)/s
    
    return img            

##





#màu đẹp : YlGn, YlOrRd, Greens

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('summer')
    else:
        cmap = plt.get_cmap(cmap)
    csfont = {'fontname':'Tahoma'}
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title+" (Unit:%)",**csfont)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=60,**csfont)
        plt.yticks(tick_marks, target_names,**csfont)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.1f}".format(cm[i, j]*100),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",**csfont)
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",**csfont)


    plt.tight_layout()
    plt.ylabel('True label',**csfont)
    plt.xlabel('Predicted label\nAccuracy = {:0.4f}%; Misclass = {:0.4f}%'.format(accuracy*100, misclass*100),**csfont)
    

def testmatrix(cmap):
    confMat = np.zeros((13,13))
    confMat = np.random.randint(200,size=(13,13))
    for i in range(13):
        confMat[i][i]*=4
    names = [ 'temp' + str(i) for i in range(13)]

    plot_confusion_matrix(confMat, normalize=False, target_names = names,cmap= cmap)

def BarChart(img,truth,performance,names):
    y_pos = np.arange(len(names))
    color = []
    for i in range(len(performance)):
        if i == truth:
            color.append('green')
        elif performance[i] < max(performance):
            color.append('red')
        else:
            color.append('blue')
    performance = [round(i*100) for i in performance]
    plt.figure(figsize=(10,6))
    plt.subplot( 2,2, 1)
    """
    path = r'''‪D:\DendroDataset\org\Gõ đỏ\1.jpg'''[1:]
    img = mpimg.imread(path)
    """
    plt.imshow(img)
    csfont = {'fontname':'Tahoma'}
    plt.suptitle('Wood image',**csfont)
    plt.subplot( 2,2, 2) 
    plt.barh(y_pos, performance, align='center', alpha=0.5, color = color,edgecolor='black')
    plt.yticks(y_pos, names,**csfont)
    plt.xlabel('Probability (%)',**csfont)
    plt.suptitle('Determine which type of log',**csfont)




#BarChart('liễu',performance,names)
        
'''
csfont = {'fontname':'Comic Sans MS'}
hfont = {'fontname':'Helvetica'}

plt.title('title',**csfont)
plt.xlabel('xlabel', **hfont)
plt.show()
'''