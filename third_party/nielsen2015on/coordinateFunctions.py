import numpy as np

#BRDF-Structure: phi_d, theta_h, theta_d
#(theta_h has a non-linear mapping!)
BRDFSHAPE = (180,90,90)

#Convert from rus-coords (angles) to coords used in the BRDF structure (ids)
#In:    Rusinkiewicz coordinates (phi_d, theta_h, theta_d) [rad]
#Out:   BRDF  coordinates
def RusinkToMERL(rusinkCoords):
    shp = BRDFSHAPE
    coords = np.array(np.reshape(rusinkCoords,(-1,3)))
    coords[:,0] = np.clip(np.floor(coords[:,0]/(np.pi)*shp[0]),0,shp[0]-1)
    coords[:,1] = np.clip(np.floor(np.sqrt(coords[:,1]/(np.pi/2))*shp[1]),0,shp[1]-1)
    coords[:,2] = np.clip(np.floor(coords[:,2]/(np.pi/2)*shp[2]),0,shp[2]-1)
    return coords

#Convert from brdf-coords (ids) to rus-coords (angles)
#In:    BRDF coordinates
#Out;   Rusinkiewicz coordinates (phi_d, theta_h, theta_d)  [rad]
def MERLToRusink(merlCoords):
    shp = BRDFSHAPE
    coords = np.array(np.reshape(merlCoords,(-1,3)),'float')
    coords[:,0] = (coords[:,0])/(shp[0]-1)*(np.pi)
    coords[:,1] = np.square((coords[:,1]+0.105)/shp[1]) * (np.pi/2)
    coords[:,2] = (coords[:,2])/(shp[2]-1)*(np.pi/2)
    return coords
    
#Convert two spherical directions to BRDF coordinates (ids)
#In:    array of two spherical directions: (thetaA,phiA,thetaB,phiB) [rad]
#       theta is the azimuthal angle and phi is the polar angle.
#Out:   BRDF coordinates
def SphericalToMERL(coords):
    BRDFCoords = []
    for prad in coords:
        print(prad)
        ax = np.cos(prad[0])*np.sin(prad[1])
        ay = np.sin(prad[0])*np.sin(prad[1])
        az = np.cos(prad[1])
        bx = np.cos(prad[2])*np.sin(prad[3])
        by = np.sin(prad[2])*np.sin(prad[3])
        bz = np.cos(prad[3])
        BRDFCoords.append(np.squeeze(RusinkToMERL(DirectionsToRusink((ax,ay,az),(bx,by,bz)))).astype(int))
    return np.array(BRDFCoords)



#ID/ValidID distinction: 
# IDs span the full length of vectorized MERL data, i.e. 0-1458000 (from 180*90*90)
# ValidIDs span only the length of the valid samples in the MERL data, i.e. all elements != -1.
# The length is defined by the Boolean maskMap that indicates where valid samples are located.
# There are 1105588 elements != -1.

#Convert a valid id to an id
def ValidIDToID(validId, maskMap):
    validId = np.array(validId)
    idList = np.cumsum(maskMap)-1
    ValIds = np.array(np.searchsorted(idList, validId))
    if(np.shape(ValIds) != ()):
        ValIds[validId<0] = -1
    elif(validId<0):
        ValIds = -1
    return ValIds
#Convert an id to a valid id
def IDtoValidID(Id, maskMap):
    idList = np.cumsum(maskMap)-1
    ValIds = idList[Id]
    ValIds[~maskMap[Id]] = -1   #Mark elements not contained in maskMap
    return ValIds
#Convert an ID to BRDF coordinates
def IDToMERL(elementId):
    elementId = np.reshape(elementId,-1)
    try:
        return np.transpose(np.unravel_index(elementId,BRDFSHAPE))
    except:
        print("WARNING: Couldnt convert:")
        print(elementId)
        raise
#Convert a valid id to BRDF coordinates
def ValidIDToMERL(validId, maskMap):
    return IDToMERL(ValidIDToID(validId, maskMap))
#Convert BRDF coordinates to an id
def MERLToID(coord):
    coord = np.reshape(coord,(-1,3)).astype(int)
    return np.ravel_multi_index(np.transpose(coord),BRDFSHAPE)
#Convert BRDF coordinates to a valid id
def MERLToValidID(coord,maskMap):
    return IDtoValidID(MERLToID(coord),maskMap)

#Convert rus-coords to two direction vectors
#In:    Rusinkiewicz coordinates
#Out:   Tuple of direction vectors (omega_o,omega_i) 
def RusinkToDirections(phi_d,theta_h,theta_d):
    #Initially put Halfvector along Z axis    
    H = [0,0,1]
    omega_o = [np.sin(theta_d),0,np.cos(theta_d)]
    omega_i = [-np.sin(theta_d),0,np.cos(theta_d)]
    #Rotate phiD-pi/2 around the z-axis
    omega_o = np.dot(RzMatrix(phi_d-np.pi/2),omega_o)
    omega_i = np.dot(RzMatrix(phi_d-np.pi/2),omega_i)
    H = np.dot(RzMatrix(phi_d+np.pi/2),H)
    #Rotate thetaH around x-axis    
    omega_o = np.dot(RxMatrix(-theta_h),omega_o)
    omega_i = np.dot(RxMatrix(-theta_h),omega_i)
    H = np.dot(RxMatrix(-theta_h),H)
    #Rotate around z-axis so omega_o aligns with x-axis
    angl = -np.arccos(np.dot((1,0,0),normalize((omega_o[0],omega_o[1],0))))*np.sign(omega_o[1])##-omega_o[1]
    omega_o = np.dot(RzMatrix(angl),omega_o)
    omega_i = np.dot(RzMatrix(angl),omega_i)
    H = np.dot(RzMatrix(angl),H)

    return (omega_o,omega_i)

#Convert two direction vectors to rus-coords
#In:    Direction vectors to view/lightsource
#Out:   Rusinkiewicz coordinates (phi_d, theta_h, theta_d)  [rad]
def DirectionsToRusink(a,b):
    a = np.reshape(normalize(a),(-1,3))
    b = np.reshape(normalize(b),(-1,3))
    H = normalize((a+b)/2)   
    theta_h = np.arccos(H[:,2])   
    phi_h = np.arctan2(H[:,1],H[:,0])
    biNormal = np.array((0,1,0))
    normal = np.array((0,0,1))
    tmp = rotateVector(b,normal,-phi_h)
    diff = rotateVector(tmp, biNormal, -theta_h)
    theta_d = np.arccos(diff[:,2])
    phi_d = np.mod(np.arctan2(diff[:,1],diff[:,0]),np.pi)
    return np.column_stack((phi_d,theta_h,theta_d))
    

# HELPER FUNCTIONS -----------------------------------------------------------    
    
#Create rotation matrices for rotations around x,y, and z axes.
def RxMatrix(theta):
    return np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])    
def RyMatrix(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
def RzMatrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],[np.sin(theta), np.cos(theta), 0],[0,0,1]])
    
#Rotate vector around arbitrary axis
def rotateVector(vector, axis, angle):
    cos_ang = np.reshape(np.cos(angle),(-1));
    sin_ang = np.reshape(np.sin(angle),(-1));
    vector = np.reshape(vector,(-1,3))
    axis = np.reshape(np.array(axis),(-1,3))
    return vector * cos_ang[:,np.newaxis] + axis*np.dot(vector,np.transpose(axis))*(1-cos_ang)[:,np.newaxis] + np.cross(axis,vector) * sin_ang[:,np.newaxis]

#Normalize vector(s)
def normalize(x):
    if(len(np.shape(x)) == 1):
        return x/np.linalg.norm(x)
    else:
        return x/np.linalg.norm(x,axis=1)[:,np.newaxis]
    
    
