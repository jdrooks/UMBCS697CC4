import pandas as pd
import numpy as np
import math
import sklearn.linear_model as lm


#This just reads in a file as a pandas dataframe
def LoadDataFile(filename):
    return pd.read_csv(filename,header=None)


#This looks for a tumorname.csv in a folder called datasetname.
#the idea here is that we could clean the data in multiple ways,
#all saved as seperate subfolders containing COAD.csv, KIRC.csv, etc;
#this function allows us to switch between them by giving the
#folder name as the datasetname. Right now, we just read in 
#one csv, 'all'.  This function was written more for the unsupervised
#portion, so we're kinda sidestepping it right now.
def LoadDataSet(datasetname,tumors):
    dataset = {}    
    
    for tumor in tumors:
        dataset[tumor] = LoadDataFile('./' + datasetname + '/' + tumor + '.csv')

    return dataset
    
#Here we're building a list of class labels for the data we have.
#We're using COAD=0, KIRC=1,PRAD=2,SKCM=3,UCS=4.
def BuildTargetVector(partitions):
    target = np.zeros(partitions[len(partitions)-1]['end_row']+1)
    cls = 0

    for partition in partitions:
        target[partition['start_row'] : partition['end_row']] = cls
        cls += 1
    
    return target
    
def BuildTargetVectorFromShapes(shapes):
    vector_length = 0
    
    for shape in shapes:
        vector_length += shape[0]
        
    target = np.zeros(vector_length)
    cls = 0
    
    for shape in shapes:
        for i in range(0,shape[0]):
            target[i]=cls
        cls+=1    
            
    return target
    
def RunLogisticRegression(X,y):
    
    #Initialize model object
    model = lm.LogisticRegression()
    
    #This is the one-liner that runs the model! It says take data matrix X,
    #and set of class labels y, and run a regression on them.
    model.fit(X,y)
    
    #This line means 'use the regression we just ran (the 'model' object) to
    #make a prediction for every row in matrix X. The 'ground truth' (actual classifications)
    #are in target vector y. Compare the predictions with the ground truth and return the %
    #that were classified correctly.
    print('Percent classified correctly: ' + str(model.score(X,y) * 100.0))
    
    #This line returns the coeffients, the thetas that define the linear model. 
    #We're probably not going to need them for this part of the assignment.
    #print(model.coef_)
    
def RunLogisticRegressionWCV(X_train,y_train,X_test,y_test):
    model = lm.LogisticRegression()
    model.fit(X_train,y_train)
    
    print('Percent classified correctly: ' + str(model.score(X_test,y_test) * 100.0))
    
#Our partition script only allows us to split consecutive groups of rows.
#so we can't have the non-tumor class be both before and after the tumor
#class.So we pull the tumor rows into a new dataframe, then append the rest of the rows afterwards.
def MoveTumorToFrontOfData(data,tumorstartrow,tumorendrow):
    #tumor data is already at the start of the data, nothing needs to be done    
    if (tumorstartrow == 0):
        return data
    
    #tumordata is the chunk of the data from the tumor start to the tumor end
    #pre tumor data is from the beginning of the dataset to the start of the tumor data
    #post tumor data is after the tumor class rows
    tumordata = data[tumorstartrow:(tumorendrow+1)]
    pretumordata = data[0:tumorstartrow]
    posttumordata = data[tumorendrow:]
    
    #these are the rows to concatenate
    to_concat = [tumordata,pretumordata,posttumordata]
    
    #put the data together in the new order, and return.
    return pd.concat(to_concat)

def GetTrainAndTestSet(data, partitions, pcttrain):
    
    #initialize the train and test sets to have the same schema as the data
    #but with no rows.
    train = data[0:0]
    test = data[0:0]
    train_shapes = []
    test_shapes = []
    previous_end_row = 0
    
    for partition in partitions:
        split_idx = int(math.floor((partition['end_row']-previous_end_row) * pcttrain) + partition['start_row'])
        
        prt_train = data[partition['start_row']:split_idx]
        prt_test = data[split_idx:partition['end_row']]
        
        train_shapes.append(prt_train.shape)
        test_shapes.append(prt_test.shape)        
        
        #Add the first 80% of each tumor's rows to the train set
        #Add the last 20% of each tumor's rows to the test set
        to_concat_train = [train, prt_train]
        to_concat_test = [test, prt_test]
        
        train = pd.concat(to_concat_train)
        test = pd.concat(to_concat_test)
        
        previous_end_row = int(partition['end_row'])
        
    return train,test,train_shapes,test_shapes

"""
Work in progress...
    
def GetPartitionsTrain(partitions, pcttrain):
    partitions_train = partitions
    next_start_row = 0
    new_end_row = 0
    
    for partition in partitions_train:        
        
        new_end_row = math.floor(partition['end_row'] * pcttrain)
        partition['end_row'] = new_end_row
        
    return partitions_train
    
def GetPartitionsTest(partitions)

"""

#---- Main Program -----


#This is the subfolder of the working directory where the ALL.csv data file is stored.
datasetname = 'AllDataCleaned'

#This is where we point the data reading functions to the data file.
#by putting 'ALL' in as a tumor, we point the script to ALL.csv
#instead of COAD.csv, KIRC.csv, etc.
tumors = ['ALL']

#Load in data
dataset = LoadDataSet(datasetname,tumors)

data = dataset[tumors[0]]

"""
Here we define how the data is to be partitioned into tumors.

Row 1-270:   COAD, colon adenocarcinoma (270 total rows, 80% = 216, 20% = 54)
Row 271-690: KIRC, kidney renal clear cell carcinoma (420 total rows, 80% = 336,20% = 84)
Row 691-942: PRAD, prostate adenocarcinoma (252 total rows, 80% = 201, 20% = 51)
Row 943-1285: SKCM, skin cutaneous melanoma (343 total rows, 80% = 274, 20% = 69)
Row 1286-1342:  UCS, uterine carcinosarcoma (56 total rows, 80% = 44, 20% = 12)

"""

#We're going to move the class data we want to the fron tof the data, so its indices will be 
#different, based on the number of rows in the tumor data.

partitions = ({'start_row':0,   'end_row':269, 'name':"COAD"},
              {'start_row':270, 'end_row':690, 'name':"KIRC"},
              {'start_row':690, 'end_row':942, 'name':"PRAD"},
              {'start_row':943, 'end_row':1285,'name':"SKCM"},
              {'start_row':1286,'end_row':1341,'name':"UCS"})

#Create the array that contains the class labels.
target = BuildTargetVector(partitions)

train, test, train_shapes, test_shapes = GetTrainAndTestSet(dataset[tumors[0]],partitions,.2)

target_train = BuildTargetVectorFromShapes(train_shapes)
target_test = BuildTargetVectorFromShapes(test_shapes)

RunLogisticRegressionWCV(train,target_train,test,target_test)
#Run the regression against the data we read in and the target vector 
#we created.
"""
for tumor in dataset:
    MoveTumorToFrontOfData(dataset[tumor],943,1285)
    RunLogisticRegression(dataset[tumor],target)
"""