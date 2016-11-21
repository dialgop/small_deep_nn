import scipy.io as sp
import numpy as np
import math

#Load data and labels from the file
def loadData(file_name):
    data = sp.loadmat(file_name)    
    train_data = data['window_train_data']    
    avg_test = data['avgActionSeqTest']
    max_test = data['maxActionSeqTest']    
    return train_data, avg_test, max_test
    
def loadLabels(file_name):
    data = sp.loadmat(file_name)
    train_labels = data['actionLabelsTrain']
    avg_max_labels = data['actionLabelsTest']
    return train_labels, avg_max_labels
    
'''This could also change the way of the results... Ask later to Martin'''
def shuffleWindows(data,label):
    permutation = np.random.permutation(data.shape[0])
    newData = data[permutation,:]
    newLabel = label[permutation,:]
    return newData, newLabel

def extractValidationData(shuf_data,shuf_label):
    train_size = math.ceil(shuf_data.shape[0]*0.7)
    train_data = shuf_data[0:train_size]
    validation_data = shuf_data[train_size:]
    train_label = shuf_label[0:train_size]
    validation_label = shuf_label[train_size:]
    return train_data,validation_data,train_label,validation_label

def loadAll(dataFileName,labelsFileName,validation=0):
    #If validation = 0 --> returns just 5 data, otherwise it will return 7 data (5+2 of validation tests)
    trainData,testAvgData,testMaxData = loadData(dataFileName)
    trainLabels,testLabels = loadLabels(labelsFileName)
    shufData,shufLabels = shuffleWindows(trainData,trainLabels)
    if validation == 0:
        data = {'trainD': shufData, 'testMax': testMaxData, 'testAvg': testAvgData}
        labels = {'trainL': shufLabels, 'testL': testLabels}                
    else:
        trainData,validationData,trainLabel,validationLabel = extractValidationData(shufData,shufLabels)
        data = {'trainD': trainData, 'vldtD': validationData,'testMax': testMaxData, 'testAvg': testAvgData}
        labels = {'trainL': trainLabel, 'vldtL': validationLabel, 'testL': testLabels}                        
    return data, labels

def next_batch(array,initial,batch_size):
    if(batch_size>array.shape[0]):
        print('The batch amount required is bigger than the size of the batch itself, returning NONE')
        return None
    else:
        fixed_initial = initial%array.shape[0]
        fixed_end = (initial + batch_size)%array.shape[0]
        if(fixed_initial > fixed_end):
            return np.concatenate([array[fixed_initial:],array[0:fixed_end]])
        else:
            return array[fixed_initial:fixed_end]