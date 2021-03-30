import numpy as np
from optimizationFunctions import *

#Load precomputed data
dataDir = "data/"
maskMap = np.load('%s/MaskMap.npy'%dataDir)   #Indicating valid regions in MERL BRDFs
Q = np.load('%s/ScaledEigenvectors.npy'%dataDir) #Scaled eigenvectors, learned from trainingdata

#Find up to 3 optimal sampling directions of Vs:
(pointHierachi, C) = FewToMany(Q, maskMap, 3)

#Print points: (in MERL coordinates)
for (i,pointSet) in enumerate(pointHierachi):
    print("#Optimum directions, n=%d:\n%s\n"%(i+1,pointSet))
