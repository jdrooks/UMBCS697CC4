import pandas as pd
import os
import errno
import operator
import nimfa
import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list, cophenet
import sklearn.decomposition as skld
import matplotlib.pylab as mplpl
import matplotlib.pyplot as plt
from matplotlib import gridspec

#Dump a list, array or matrix into a csv
def DumpToCSV(data):

    df = pd.DataFrame(data)
    df.to_csv('./dump.csv',header=False,index=False)

def LoadDataFile(filename):
    return pd.read_csv(filename,header=None)
    
def LoadDataSet(datasetname,tumors):
    dataset = {}    
    
    for tumor in tumors:
        dataset[tumor] = LoadDataFile('./' + datasetname + '/' + tumor + '.csv')

    return dataset

#runs a basic NMF decomp, returns W and H.
def RunNMF(data,n_components,init):
    model = skld.NMF(n_components=n_components,init=init,beta=0.3,eta=0.5)
    W = model.fit_transform(data);
    H = model.components_;
    
    return W,H
	
#With help from Joe Dollard's group - originally we were not transposing
#the matrix and running against the rows of W instead of the columns of H.
#This is the code they used to help get the predictions vector.
def GetClustersFromH(H):
    clusters = []
    model_width = len(H[0])

    for col_idx in range(model_width):
        # convert column into an array
        col_vals = H[:,col_idx]

        # determine biggest row index and it's value from the array
        (row_idx,max_val) = max( enumerate(col_vals), key=lambda x: x[1] )

        clusters.append(row_idx)
        
    return clusters

#Generates the 1-hot clustering matrix from matrix W
def BuildClusteringMatrixProb(predictions, n_clusters):
    cm = np.zeros(shape = (len(predictions),n_clusters))
    counter = 0
    
    for p in predictions:
        cm[counter,p]=1
        counter += 1
        
    return cm

#Creates a consensus matrix from the clustering matrix
def BuildConsensusMatrix(cm):
    mat = np.matrix(cm)
    consensus = np.dot(mat, np.matrix.transpose(mat))
      
    return consensus

#This is where the meat of the work is done. Runs NMF for k = n_components
#num_iterations times. Keeps cumulative sum of consensus matrices and then
#divides by the number of runs to get the averaged consensus matrix.
def ComputeAverageConsensusMatrix(data, n_components, num_iterations,init):
    i = 0.0
    
    base = np.zeros(shape = (data.shape[1],data.shape[1]))
    
    #for j in xrange(0, num_iterations):
    for j in range(0, num_iterations):
        W, H = RunNMF(data,n_components,init)
        
        predictions = GetClustersFromH(H)
        cm = BuildClusteringMatrixProb(predictions,n_components)
        consensus = BuildConsensusMatrix(cm)
        base = base + consensus
        i += 1.0
        
    return base / i

#Both of these functions have been incorporated from the NIMFA
#NMF code at https://github.com/marinkaz/nimfa/. Functions are used for 
#reordering consensus matrix rows and for calculating cophenetic
#coeffients.

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def reorder(C):
    Y = 1 - C
    Z = linkage(Y, method='average')
    ivl = leaves_list(Z)
    ivl = ivl[::-1]
    return C[:, ivl][ivl, :]
    
def coph_cor(A, idx=None):
    avec = np.array([A[i, j] for i in range(A.shape[0] - 1)
                    for j in range(i + 1, A.shape[1])])
    Y = 1 - avec
    Z = linkage(Y, method='average')
    return cophenet(Z, Y)[0]

def savematrixplot(datasetname,tumorname, A, k):
    """    
    plt.figure("%s_consensus_rank_%d.png" % (tumorname,k))
    plt.subplot(211)
    plt.matshow(A)
    plt.savefig("./" + datasetname + "_results/%s_consensus_rank_%d.png" % (tumorname,k))
    """
    mplpl.matshow(A)
    mplpl.savefig("./" + datasetname + "_results/%s_consensus_rank_%d.png" % (tumorname,k))

def savecophcorplot(datasetname,tumorname,k_to_c):
    plt.figure("%s_cophenetic_v_rank.png" % tumorname)
    plt.subplot(211)
    plt.plot(k_to_c.keys(), k_to_c.values())
    plt.ylabel('cophenetic correlation')
    plt.xlabel('Rank k')
    plt.savefig("./" + datasetname + "_results/%s_cophenetic_v_rank.png" % tumorname)

def generateHeatPlot(datasetname, tumor, header_list, matrix, W, H):
    plot_name = tumor + '_heat_plot'
    plt.figure(plot_name)
    gs = gridspec.GridSpec(2, 6) #, width_ratios=[1, 1, 2]) 
    ax4 = plt.subplot(gs[1, 3:])#(2,3,6)
    ax4.plot(np.array(H)[0], color="blue", linewidth=1.0, linestyle="-")
    ax4.plot(np.array(H)[1], color="green", linewidth=1.0, linestyle="-")
    ax4.set_xticklabels(['' for i in range(20)], minor=False)
    ax4.set_yticklabels(['' for i in range(20)], minor=False)
    #ax4.ylabel('metagene expression profile')
    #ax4.xlabel('samples')
    ax1 = plt.subplot(gs[0, :]) #(2,1,1)
    row_labels = ['' for i in range(len(matrix))]
    column_labels = ['' for i in range(len(matrix))]
    heatmap = ax1.pcolor(np.array(matrix), cmap=plt.cm.Blues)
    plt.gcf()
    ax1.set_xticklabels(column_labels, minor=False)
    ax1.set_yticklabels(row_labels, minor=False)
    ax2 = plt.subplot(gs[1, 0:1])#(2,3,4)
    ax2.pcolor(np.array(W), cmap=plt.cm.Blues)
    ax2.set_xticklabels(column_labels, minor=False)
    ax2.set_yticklabels(row_labels, minor=False) 
    ax3 = plt.subplot(gs[1, 1:3])#(2,3,5)
    ax3.pcolor(np.array(H), cmap=plt.cm.Blues)
    ax3.set_xticklabels(column_labels, minor=False)
    ax3.set_yticklabels(row_labels, minor=False)
    plt.savefig("./" + datasetname + plot_name)

def RunTumor(datasetname,tumorname,data,mink,maxk,num_iterations,init):
    
    k_cophs = {}    
    
    data = data.as_matrix()
    data = np.matrix.transpose(data)
    
    #for k in xrange(mink,maxk):
    for k in range(mink,maxk):
        mat = ComputeAverageConsensusMatrix(data,k,num_iterations,init)
        mat = reorder(mat)
        A = np.asarray(mat)
        
        k_cophs[k]=coph_cor(A)
        savematrixplot(datasetname,tumorname,A,k)        
        
    print(tumorname,"cophs:",k_cophs)
    savecophcorplot(datasetname,tumorname,k_cophs)
    rank = evaluateStability(k_cophs)
    nmf = nimfa.Nmf(data, rank=rank, seed="random_vcol", max_iter=200, update='euclidean',
                    objective='conn', conn_change=40)
    nmf_fit = nmf() 
    generateHeatPlot(datasetname, tumorname, '', data, nmf_fit.basis(), nmf_fit.coef())


def evaluateStability(rank_to_cophenetic):
    rank = max(rank_to_cophenetic.iteritems(), key=operator.itemgetter(1))[0]
    print 'the k with the highest cophenetic: ', rank   
    return rank

#-----Main Program-----

#initialization type for NMF runs
init='random'

#subfolder name where cleaned data is stored
datasetname='data_split_pre_processed'

#each of these should be a csv in the datasetname directory. Crop this list to restrict 
#which tumors are run.

tumors = ['COAD']

dataset = LoadDataSet(datasetname,tumors)

make_sure_path_exists("./" + datasetname + '_results')

for tumor in dataset:
    RunTumor(datasetname,tumor,dataset[tumor],2,11,100,init)
    
