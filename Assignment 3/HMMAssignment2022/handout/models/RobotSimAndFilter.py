
import random
import numpy as np
from typing import *

from models import TransitionModel,ObservationModel,StateModel

#
# Add your Robot Simulator here
#Perfrom simulation here ?




class RobotSim:
    def __init__(self, pose:Tuple[int, int, int]):
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
class HMMFilter:
    def __init__(self, om):
        self.__om = om

    # Calculate the new probability distribution
    
    def forward_filter(self, sensor_reading, T_transp, f):
        reading = None
        if sensor_reading != None:
            reading = sensor_reading

        O = self.__om.get_o_reading(reading)
        newF = np.dot(O, np.dot(T_transp, f))
        best_state = None
        best_proba = -1
        for t in range(len(newF)):
            if newF[t] > best_proba:
                best_proba = newF[t]
                best_state = t
                
        return newF/np.sum(newF), best_state
    

        
        