import numpy as np
import h5py
import scipy.io as spio

# Need to give a file extension.
# Labels: are open with scipy and then converted to a numpy array.

def getLabels(given_file,num): #1 brings training labels, rest test    
    f = spio.loadmat(given_file)
    if(num==1):
        print('Retrieving labels from the file', given_file, 'from train data')
        labels = f["actionLabelsTrain"]        
    else:
        print('Retrieving labels from the file', given_file, 'from test data')
        labels = f["actionLabelsTest"]        
    x,y = labels.shape
    size = (max(labels))[0]
    data_labels = np.zeros((x,size))    
    i=0
    while(i<x):
        data_labels[i][labels[i]-1] = 1
        i+=1
    return data_labels
    
#Training data: are open with h5py and converted to a numpy.ndarray

def getData(given_file,num): #1 brings training data, rest test    
    data = list()
    with h5py.File(given_file ,'r') as f:
        if(num ==1):
            print('Retrieving data from the file', given_file, 'from train data')
            x,y = np.asarray(f.get("actionSequencesTrain")).shape
        else:
            print('Retrieving data from the file', given_file, 'from test data')
            x,y = np.asarray(f.get("actionSequencesTest")).shape
        try:
            x = 0
            mat_data=[]
            if(num==1):
                while(x<y):
                    data.append([f[element[x]][:] for element in f["actionSequencesTrain"]])
                    x = x+1
            else:
                while(x<y):
                    data.append([f[element[x]][:] for element in f["actionSequencesTest"]])
                    x = x+1
            x=0                           
        except IndexError:
            pass
    return data
    
def maxRowSize(trainArray,testArray):
    print('calculating max frame sequence in train and test data')
    #Finds maximum frame size of trainArray
    
    i=0
    maxFrames=0
    while(i<len(trainArray)):
        x,y = np.asarray(trainArray[i][0]).shape
        i+=1
        if(x>maxFrames):
            maxFrames = x   
    i=0    
    while(i<len(testArray)):
        x,y = np.asarray(testArray[i][0]).shape
        i+=1
        if(x>maxFrames):
            maxFrames = x

    print('Maximum number of rows found and it is:', maxFrames)
    return maxFrames

def padAndReshape(array, size):
    # Values needed to create a big matrix
    batchsize,_ = np.array(array).shape
    _,_,shape = np.asarray(array[0]).shape
    # Creation of the matrix of Zeros of size [batchsize,shape * size]    
    reshaped_matrix = np.zeros([batchsize, size * shape])
    #Loop: Reshaping each component of the incoming array and putting it into the matrix
    i=0
    while(i<batchsize):
        _,x,y = np.asarray(array[i]).shape
        aux = np.reshape(array[i],[x*y]) 
        reshaped_matrix[i,0:len(aux)] = aux
        i+=1
    return reshaped_matrix

def getAllLabels(fileLabels):
#Returns labels in a dictionary
#Data is in '/media/data/garbade/action_data_Ntrajp/bn_mocap/split_1/actionLabels.mat'
    print('Getting a dictionary with the labels of train and test')
    labelsTrain = getLabels(fileLabels,1)
    labelsTest = getLabels(fileLabels,2)    
    return {'train': labelsTrain, 'test': labelsTest}

def getAllData(fileData):    
#Returns data which is padded with zeros based on the frame containing the maximum lenght size
# Data is in '/media/data/garbade/action_data_Ntrajp/bn_mocap/split_1/allVidDescs_feats.mat'data
    print('Getting a dictionary with the data of train and test')
    #1. Gets the data from function getData
    dataTrain = getData(fileData,1)
    dataTest = getData(fileData,2)
    #2. Using the maxRowSize, calculates the biggest row of both data
    frameSize = maxRowSize(dataTrain,dataTest)    
    #3. Pads with zeros the data of both matrices using the function padWithZeros
    PDataTrain = padAndReshape(dataTrain,frameSize)
    print('from train data')
    PDataTest = padAndReshape(dataTest,frameSize)
    print('from test data')
    return {'train': PDataTrain, 'test': PDataTest}
