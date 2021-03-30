import sys
import time
import numpy as np
from coordinateFunctions import *


#Get selector arrays for different modes
def PointSelector(BRDFCoordinates,maskMap):
    ids = MERLToValidID(BRDFCoordinates,maskMap)
    return np.delete(ids,np.where(ids==-1))  #Remove invalid ones 
          

#Main optimization function
#Returns optimal sampling directions for up to [maxPoints] samples.
#In:    Z, the matrix of basisvectors hvor cond-num is optimized (PC-matrix)
#       maskMap, boolean map indicating what values in the MERL data are valid
#       maxPoints, maximum number of sampling-directions to optimize for.
#       addEvals, number of random initial guesses every time a point is added to the sampling
#       optStepSize, stepsize during gradient descent
#       optMaxIter, maximum number of iterations during gradient descent
def FewToMany(Z, maskMap, maxPoints, addEvals = 500, optStepSize = 3, optMaxIter=100):
    
    print("\n ##################\nFewToMany Optimization, adding %d points, using %d evaluations. \n"%(maxPoints,addEvals))
    points = np.array(())
    pointHierachy = []
    while(np.shape(points)[0] < maxPoints):
        t1 = time.time()
        points = AddMostSignificant(Z, maskMap, points, 1, numEvals=addEvals)
        t2 = time.time()
        print("Took: %f"%(t2-t1))
        if(points.shape[0]>1):
            (points, C) = OptimizeLocations(Z, maskMap, points, maxIter=optMaxIter, convPerc=0.001, stepSize = optStepSize)
        pointHierachy.append(points)
    return (pointHierachy, C)


#Calculate condition number of Z
#If rows is non-zero, condition number is only based on the "rows" largest eigenvalues
#If usePrimaryColumnsOnly is True and Z has less rows than columns, only the
#first columns of Z are used.
def CondNum(Z,rows=0,usePrimaryColumnsOnly=True):
    (p,n) = np.shape(Z)
    if(p<n and usePrimaryColumnsOnly):
        Z = Z[:,0:p]
    #print("Shape of Z for condition number: (%dx%d)"%np.shape(Z))
    if(len(np.shape(np.squeeze(Z)))>1):
        if(rows>0):
            eigVs = sp.linalg.svd(Z,0)[1]
            if(rows<=len(eigVs)):
                return eigVs[0]/eigVs[rows-1]    #Calculate condition number on limited number of rows
            else:
                #print("Warning: Too few eigenvalues to extract #%d for condition number. Using #%d instead."%(rows,len(eigVs)))
                return eigVs[0]/eigVs[-1]
        else:
            return np.linalg.cond(Z)    #Calculate condition number on all
    else:
        return 1/np.linalg.svd(np.reshape(Z,(1,-1)),0)[1] #If only 1 value, use eigenvalue
        
#Add the nadd most significant [nadd] points to points, based on change in cond. number
#of Z. (Evaluating a random subset of Z based on numEvals)
#In:    Z, the matrix of basisvectors hvor cond-num is optimized (PC-matrix)
#       maskMap, boolean map indicating what values in the MERL data are valid
#       points, existing MERL coordinates where a point should be added to
#       nadd, number of points to add
#       numEvals, number of random initial guesses every time a point is added to the sampling
def AddMostSignificant(Z, maskMap, points, nadd, numEvals = 1000):    
    points = np.reshape(points,(-1,3))
    n = np.shape(points)[0] #Number of points
    sampPoints = np.array(points) #Make a copy of the points   
    print("Adding %d points to existing %d points..."%(nadd,n))
    sys.stdout.flush()
    
    if(numEvals == 0):
        numEvals = np.shape(Z)[0]
 
    madd = 0
    #If it is the first element we simply find the largest leverage value
    if(n==0):
        knownIds = [np.argmax(np.linalg.norm(Z,axis=1))]
        madd += 1
        print("First point added, based on leverage of Z.")
        sys.stdout.flush()
    else:
        knownIds = MERLToValidID(sampPoints,maskMap)
        
    while madd<nadd:
        #Do a random permutation of numbers
        idList = np.random.permutation(np.shape(Z)[0])
        
        #Remove known points
        for j in range(0,len(knownIds)):
            idList = np.delete(idList,np.where(idList==knownIds[j]))
        
        #Test all
        Cs = []
        print("Evaluating condition number for %d locations"%numEvals)
        t0 = time.time()
        for i,indx in enumerate(idList[0:numEvals]):               
            newList = np.append(knownIds,indx)
            coords = ValidIDToMERL(newList,maskMap)
            selector = PointSelector(coords, maskMap)
          
            Cs.append(CondNum(Z[selector,:]))
            if(i==10):
                t1 = time.time()
                print("Estimated time: %2.2f seconds"%((t1-t0)/10*numEvals))
                sys.stdout.flush()
      
        #Find the index of the best condition number for lowestRows number of rows
        bestI = idList[np.argmin(Cs)]   
                
        knownIds = np.append(knownIds,bestI)
        madd += 1

    return ValidIDToMERL(knownIds,maskMap)

                
#Optimize positions of points based on condition number of Z
#In:    Z, the matrix of basisvectors hvor cond-num is optimized (PC-matrix)
#       maskMap, boolean map indicating what values in the MERL data are valid
#       points, existing MERL coordinates where a point should be added to
#       maxIter, maximum number of iterations during gradient descent
#       convPerc, threshold for "percent change" that triggers convergence/termination                
#       stepSize, stepsize during gradient descent
def OptimizeLocations(Z, maskMap, points, maxIter=100, convPerc=0.1, stepSize = 1):
    shp = BRDFSHAPE
    sampPoints = np.reshape(points,(-1,3))   #Make a copy of input positions (we dont wanna change original data)
    n = np.shape(sampPoints)[0] #Number of points
    
    gradDirs = np.array([[1,0,0],    #Gradient directions
                        [-1,0,0],
                        [0,1,0],
                        [0,-1,0],
                        [0,0,1],
                        [0,0,-1]])
                                        
        
    print("Optimizing positions of %d sampling directions..."%n)
    sys.stdout.flush()
        
    nIter = 0
    belowThresholdTimes = 0
    converged = False
    lastC = np.inf
    while(not converged and nIter<maxIter):
        t1 = time.time()

        currentSelector = PointSelector(sampPoints,maskMap)
        bestC = iterC = CondNum(Z[currentSelector,:])    #Current best Condition number
        selOrder = np.random.permutation(n)  #Permutated selection order
        for i in selOrder:
            newSampPoints = np.array(sampPoints)    #Copy of positions
            
            #Create list of selectors to evaluate condition number on
            newSampSelector = []
            for j,d in enumerate(gradDirs):  #Look in all directions and find the lowest number of rows selected
                newSampPoints[i,:] = np.clip(sampPoints[i,:] + d * stepSize, (0,0,0), np.subtract(shp,1)) #Upper limit is bounds-1
                                
                #Verify that we are not moving outside valid region and not merging 
                #(constant number of unique sampling directions)!
                samplingIds = MERLToValidID(newSampPoints,maskMap)
                if(len(np.unique(np.delete(samplingIds,np.where(samplingIds==-1)))) == n):
                    
                    #Store tuple of selector array and points for this direction for evaluation
                    selector = PointSelector(newSampPoints,maskMap) #Get selector array
                    newSampSelector.append((selector,np.array(newSampPoints)))  #Tuple 
                        
            for (selector,points) in newSampSelector:    #From all directions find the lowest condition number
                c = CondNum(Z[selector,:])    #Calculate condition number of this change, using the lowest number of rows available                
                if c<bestC:
                    bestC = c
                    sampPoints = np.array(points)
        t2 = time.time()
        
        print("New iteration-----------------")
        print("Stepsize: %2.2f"%stepSize)
        print("Current condition number: %2.2f"%bestC)
        print("Last iteration's cond. reduction: %2.2f (%2.2f%%)"%(iterC-bestC,(iterC-bestC)/iterC*100))
        print("Iteration time: %2.2f seconds"%(t2-t1))
        print("Number of unique ids: %d (should be constant)"%len(np.unique(MERLToID(sampPoints))))
        if((lastC-bestC)/lastC*100.0 <= convPerc):    #converge on less than [convPerc] percent change
            stepSize = 1.0    #Force fine stepsize for final steps
            belowThresholdTimes += 1
            if(belowThresholdTimes > 2):
                converged = True
                print("Converged!")
        else:
            belowThresholdTimes = 0      
        nIter += 1
        sys.stdout.flush()
        
        lastC = bestC

    if(not converged):
        print("Warning! Did not converge!")
        
        
    return (sampPoints, bestC)
