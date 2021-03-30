import numpy as np
import scipy as sp


def MapBRDF(brdfData,maskMap,median):
    pTotal = np.shape(maskMap)[0]
    brdfData = np.reshape(brdfData,(pTotal,-1)) #Reshape to always 2-dim
    brdfValid = brdfData[maskMap,:]
    return np.log((brdfValid+0.001)/(median+0.001)) #Our mapping
#    return np.log(brdfValid)    #Only log mapping
#    return brdfValid #Raw values

def UnmapBRDF(mappedData,maskMap,median):
    pTotal = np.shape(maskMap)[0]
    mappedData = np.reshape(mappedData,(np.shape(median)[0],-1)) #Reshape to always 2-dim
    n = np.shape(mappedData)[1]
    unmapped = np.ones((pTotal,n))*-1
    unmapped[maskMap,:] = np.exp(mappedData)*(median+0.001)-0.001 #Our mapping
#    unmapped[maskMap,:] = np.exp(mappedData) #Only log mapping
#    unmapped[maskMap,:] = mappedData #Raw values
    return unmapped

    
def ReconstructMappedBRDF(knownMappedData, knownSelector, scaledPCs, relativeOffset, eta=0):
    proj = ProjectToPCSpace(knownMappedData,scaledPCs[knownSelector,:],relativeOffset[knownSelector], eta)
    nPCs = np.shape(proj)[0]
    return np.dot(scaledPCs[:,0:nPCs],proj)+relativeOffset

    
def ReconstructBRDF(knownData, knownSelector, maskMap, scaledPCs, median, relativeOffset, cosineMap=None, eta=0):
    if(knownSelector.dtype != bool):
        sortKeys = np.argsort(knownSelector)    #Sort data so it matches logical indices
        knownData = knownData[sortKeys]
        knownSelector = knownSelector[sortKeys]
    mappedKnownSelector = np.zeros(np.shape(maskMap))   #Create a selector for the mapped version
    mappedKnownSelector[knownSelector] = 1
    mappedKnownSelector = mappedKnownSelector[maskMap].astype(bool)
    if(cosineMap is not None):
        knownData *= cosineMap[mappedKnownSelector]
    mapped = MapBRDF(knownData, maskMap[knownSelector], median[mappedKnownSelector])
    recon = ReconstructMappedBRDF(mapped, mappedKnownSelector, scaledPCs, relativeOffset, eta)  
    if(cosineMap is None):
        return UnmapBRDF(recon, maskMap, median)
    else:
        um = UnmapBRDF(recon, maskMap, median)
        um[maskMap] /= cosineMap
        return um

def ProjectToPCSpace(data,PCs,relativeOffset, eta=0):
    nPCs = np.clip(np.shape(data)[0],1,np.shape(PCs)[1])    #Number of PCs to use (if only 5 values are known, only the first 5 PCs are used)
    b = data-relativeOffset
    A = PCs[:,0:nPCs]

    U, s, Vt = sp.linalg.svd(A, full_matrices=False, check_finite=False) #Use scipy instead of numpy for SVD
    Ut = np.transpose(U)
    V = np.transpose(Vt)
    Sinv = np.diag(s/(s*s+eta))
    x = V.dot(Sinv).dot(Ut).dot(b)   
    return x

    
