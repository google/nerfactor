import numpy as np
import scipy as sp
import time
import sys
from coordinateFunctions import *


#Pre-compute maskmap (boolean map of where data is valid). 
# If horizonCheck is True, all directions are bruteforce tested,
# if they are below horizon (gives a better maskMap, but VERY SLOW)
def ComputeMaskMap(dataMatrix, horizonCheck=True):
    #Generate 1st version of maskMap
    maskMap = (dataMatrix[:,0] != -1)
    
    if(horizonCheck):
        #Run through all valid regions and investigate if they have a view below horizon
        for vId in np.where(maskMap)[0]:
            rusCoord = MERLToRusink(IDToMERL(vId))[0,:]
            (v,i) = np.squeeze(RusinkToDirections(rusCoord[0],rusCoord[1],rusCoord[2]))
            
            #Is either view or illumination significantly negative
            if(v[2]<-0.01 or i[2]<-0.01):
                maskMap[vId] = False    
    return maskMap
    

#Pre-compute cosine-map (VERY SLOW!)
def ComputeCosMap(maskMap):
    cosMap = np.ones((np.sum(maskMap),1))
    minVal = 0.01
    #Run through all valid regions and compute cosines
    j = 0
    N = (0,0,1)
    for vId in np.where(maskMap)[0]:
        rusCoord = MERLToRusink(IDToMERL(vId))[0,:]  #Get rusink coordinate
        (v,i) = np.squeeze(RusinkToDirections(rusCoord[0],rusCoord[1],rusCoord[2])) #get view/light vectrors
        cosMap[j] = np.dot(v,N)*np.dot(i,N) #calculate cos products
        j += 1
    cosMap = np.clip(cosMap,minVal,1)    
    return cosMap
    
    
#Perform PCA on dataMatrix
#In     dataMatrix, data to perform PCA on
#       maskMap, precomputed maskmap indicating valid regions in datamatrix
#       cosMap, precomputed cosine weights to data
#       explVar, only return PCs corresponding to [explVar]% explained variance
#       mapCosine, if cosMap should be used or not
#Out    tuple of:
#       scaledPCs, principal components scaled by their eigenvalues (variance explained)
#       relativeOffset, offset subtracted from data to mean-center it
#       median, median of data, used as reference in mapping
def LearnMapping(dataMatrix, maskMap, cosMap, explVar = 100, mapCosine=True):
    dataMatrix = np.array(dataMatrix)   #Make a copy!
    n = np.shape(dataMatrix)[1]

    #Compute variation statistics and make sampling list from groupA
    print("Learning statistics on %d elements..."%n)
    sys.stdout.flush()
    t1 = time.time()
    #Do cosine mapping of data
    if(mapCosine):
        dataMatrix[maskMap] = dataMatrix[maskMap]*cosMap 

    validObs = dataMatrix[maskMap,:]   #Valid observations
    median = np.median(validObs,1)[:,np.newaxis]
    mapped = MapBRDF(dataMatrix, maskMap,median,)
    relativeOffset = np.mean(mapped,1)
    #Extract principal components (V)
    x = np.transpose(mapped)-relativeOffset     #Subtract mean
    U, s, Vt = sp.linalg.svd(x, full_matrices=False, check_finite=False, overwrite_a = True) #Use scipy instead of numpy for SVD
    
    #Select only the components corresponding to explVar % of variation    
    cumVar = np.cumsum(s/np.sum(s)*100)
    enoughVar = np.where(cumVar<explVar)[0][-1]+2
    scaledPCs = np.transpose(Vt)[:,0:enoughVar]*s[0:enoughVar]
    t2 = time.time()
    print("Took %2.2f seconds"%(t2-t1))
    sys.stdout.flush()
    return (scaledPCs,relativeOffset[:,np.newaxis],median)
