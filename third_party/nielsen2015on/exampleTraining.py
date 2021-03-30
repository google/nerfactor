import numpy as np
import os.path as path
from os import listdir
from trainingFunctions import *
from merlFunctions import *
from coordinateFunctions import *

MERLDir = "MERLDir/"
OutputDir = "data/"

#Parse filenames
materials = [brdfFile for i,brdfFile in enumerate(listdir(MERLDir)) \
                if (path.isfile(MERLDir+brdfFile)
                and path.splitext(brdfFile)[1] == ".binary")]         

#Fill observation array
obs = np.zeros((90*90*180, 3*len(materials)),'float32')
#Add each color channel as a single observation
for i in range(0,len(materials)):
    mat = readMERLBRDF("%s/%s"%(MERLDir,materials[i]))
    obs[:,3*i] = np.reshape(mat[:,:,:,0],(-1))
    obs[:,3*i+1] = np.reshape(mat[:,:,:,1],(-1))
    obs[:,3*i+2] = np.reshape(mat[:,:,:,2],(-1))

#Pre-compute maskMap (VERY SLOW if horizonCheck=True)
maskMap = ComputeMaskMap(obs, horizonCheck=False)

#Pre-compute cosine-map (VERY SLOW! - do once and store, or download!)
cosMap = ComputeCosMap(maskMap)

#Perform PCA on data
(scaledPCs,relativeOffset,median) = LearnMapping(obs,maskMap,cosMap)

#Save data
np.save('%s/MaskMap'%OutputDir,maskMap)
np.save('%s/CosineMap'%OutputDir,cosMap)
np.save('%s/ScaledEigenvectors'%OutputDir,scaledPCs)
np.save('%s/Median'%OutputDir,median)
np.save('%s/RelativeOffset'%OutputDir, relativeOffset)