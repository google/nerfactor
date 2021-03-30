import numpy as np
from merlFunctions import *
from coordinateFunctions import *
from reconstructionFunctions import *

#BRDF observations (5 RGB values)
obs = np.array([[0.09394814, 0.01236500, 0.00221087],
                [0.09005638, 0.00315711, 0.00270478],
                [1.38033974, 1.21132099, 1.19253075],
                [0.97795460, 0.85147798, 0.84648135],
                [0.10845871, 0.05911538, 0.05381590]])
                
#Coordinates for observations (phi_d, theta_h, theta_d)              
coords = np.array([[36.2,  1.4,  4.0 ],
                   [86.5,  76.7, 13.1],
                   [85.5,  7.6,  78.9],
                   [144.8, 2.5,  73.8],
                   [80.4,  12.9, 51.6]])        

#Convert to BRDF coordinates
MERLCoords = RusinkToMERL(np.deg2rad(coords))
#Convert to IDs (i.e. rows-ids in the PC matrix)
MERLIds = MERLToID(MERLCoords)

#Load precomputed data
dataDir = "data/"
maskMap = np.load('%s/MaskMap.npy'%dataDir)   #Indicating valid regions in MERL BRDFs
median = np.load('%s/Median.npy'%dataDir)     #Median, learned from trainingdata
cosMap = np.load('%s/CosineMap.npy'%dataDir)  #Precomputed cosine-term for all BRDF locations (ids)
relativeOffset = np.load('%s/RelativeOffset.npy'%dataDir) #Offset, learned from trainingdata
Q = np.load('%s/ScaledEigenvectors.npy'%dataDir) #Scaled eigenvectors, learned from trainingdata

#Reconstruct BRDF
recon = ReconstructBRDF(obs, MERLIds, maskMap, Q, median, relativeOffset, cosMap, eta=40)
#Save reconstruction as MERL .binary
saveMERLBRDF("reconstruction.binary",recon)