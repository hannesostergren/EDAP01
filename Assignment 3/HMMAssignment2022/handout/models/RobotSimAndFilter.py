
import random
import numpy as np

from models import TransitionModel,ObservationModel,StateModel

#
# Add your Robot Simulator here
#Perfrom simulation here ?




class RobotSim:
    def __init__(self, pose:(int, int, int)):
        self.__x = pose[0]
        self.__y = pose[1]
        self.__head = pose[2]

        
    def update_pos(self, direction):
        directionVector = [(-1,0),(0,1),(1,0),(0,-1)]
        newDirection = directionVector[direction]
        self.__x += newDirection[0]
        self.__y += newDirection[1]
        self.__head = direction
        return self.__x,self.__y,self.__head
    

        
            
    
        
#
# Add your Filtering approach here (or within the Localiser, that is your choice!)
# FIlter here ? 
class HMMFilter:
    def __init__(self, om):
        print('Hello World')
        self.__om = om


    # Calculate the new probability distribution
    
    def forward_filter(self, sensor_position, T_transp, f):
        r = None
        if sensor_position != None:
            r = sensor_position


        O = self.__om.get_o_reading(r)
        f_new = np.dot(O, np.dot(T_transp, f))
        best_state = None
        best_proba = -1
        for i in range(len(f_new)):
            if f_new[i] > best_proba:
                best_proba = f_new[i]
                best_state = i
                
    
        return f_new/np.sum(f_new), best_state
    

        
        