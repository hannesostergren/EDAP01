
#
# The Localizer binds the models together and controls the update cycle in its "update" method.
#

import numpy as np
import matplotlib.pyplot as plt
import random

from models import StateModel,TransitionModel,ObservationModel,RobotSimAndFilter

class Localizer:
    def __init__(self, sm):

        self.__sm = sm

        self.__tm = TransitionModel(self.__sm)
        self.__om = ObservationModel(self.__sm)

        # change in initialise in case you want to start out with something else
        # initialise can also be called again, if the filtering is to be reinitialised without a change in size
        
        self.initialise()

    # retrieve the transition model that we are currently working with
    def get_transition_model(self) -> np.array:
        return self.__tm

    # retrieve the observation model that we are currently working with
    def get_observation_model(self) -> np.array:
        return self.__om

    # the current true pose (x, h, h) that should be kept in the local variable __trueState
    def get_current_true_pose(self) -> (int, int, int):
        x, y, h = self.__sm.state_to_pose(self.__trueState)
        return x, y, h

    # the current probability distribution over all states
    def get_current_f_vector(self) -> np.array(float):
        return self.__probs

    # the current sensor reading (as position in the grid). "Nothing" is expressed as None
    def get_current_reading(self) -> (int, int):
        ret = None
        if self.__sense != None:
            ret = self.__sm.reading_to_position(self.__sense)
        return ret;

    # get the currently most likely position, based on single most probable pose
    def most_likely_position(self) -> (int, int):
        return self.__estimate

    ################################### Here you need to really fill in stuff! ##################################
    # if you want to start with something else, change the initialisation here!
    #
    # (re-)initialise for a new run without change of size
    
    
    def initialise(self):
        self.__trueState = random.randint(0, self.__sm.get_num_of_states() - 1)
        self.__sense = None
        
        
        self.__probs = np.ones(self.__sm.get_num_of_states()) / (self.__sm.get_num_of_states()) # replace with sensor vector
        self.__estimate = self.__sm.state_to_position(np.argmax(self.__probs))
        
    # add your simulator and filter here, for example
        
        self.__rs = RobotSimAndFilter.RobotSim(self.__sm.state_to_pose(self.__trueState))
        self.__HMM = RobotSimAndFilter.HMMFilter(self.__om)
        self.__totalError = 0.0
        self.__correctEstimates = 0
        self.__dist = []
        
        
        
        
    #
    #  Implement the update cycle:
    #  - robot moves one step, generates new state / pose
    #  - sensor produces one reading based on the true state / pose
    #  - filtering approach produces new probability distribution based on
    #  sensor reading, transition and sensor models
    #
    #  Add an evaluation in terms of Manhattan distance (average over time) and "hit rate"
    #  you can do that here or in the simulation method of the visualisation, using also the
    #  options of the dashboard to show errors...
    #
    #  Report back to the caller (viewer):
    #  Return
    #  - true if sensor reading was not "nothing", else false,
    #  - AND the three values for the (new) true pose (x, y, h),
    #  - AND the two values for the (current) sensor reading (if not "nothing")
    #  - AND the error made in this step
    #  - AND the new probability distribution
    #

    
    def move_robot_and_update(self):
        x, y, h = self.__sm.state_to_pose(self.__trueState)
        
        dims = self.__sm.get_grid_dimensions()
        headings = [0, 1, 2, 3]
        if x-1 < 0:
            headings.remove(0)
        if y+1 >= dims[0]:
            headings.remove(1)
        if x+1 >= dims[1]:
            headings.remove(2)
        if y-1 < 0:
            headings.remove(3)
            
        # Find the probability of each new heading
        problist = [0, 0, 0, 0]
        for H in headings:
            if H == 0:
                newState = self.__sm.pose_to_state(x-1, y, H)
            elif H == 1:
                newState = self.__sm.pose_to_state(x, y+1, H)
            elif H == 2:
                newState = self.__sm.pose_to_state(x+1, y, H)
            else:
                newState = self.__sm.pose_to_state(x, y-1, H)
            p = self.__tm.get_T_ij(self.__trueState,newState)
            problist[H] = p
               
        while len(headings) != len(problist):
            for item in problist:
                if item == 0:
                    problist.remove(0)
        
        # Choose a new heading from the possible ones
        new_head = random.choices(headings, weights = problist)[0]
        x, y, h = self.__rs.update_pos(new_head)
        self.__trueState = self.__sm.pose_to_state(x,y,h)
        return x,y,h
    
    
    
    def get_sensor_reading(self):
        posibilityDict = {}
        possibleNewPos1 = [(-1,0),(-1,-1),(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1)]
        possibleNewPos2 = [(-2,0),(-2,-1),(-2,-2),(-1,-2),(0,-2),(1,-2),(2,-2),(2,-1),(2,0),(2,1),(2,2),(1,2),(0,2),(-1,2),(-2,2),(-2,1)]
        n_Ls = 1
        n_Ls_2 = 1
        readingTrueState = self.__sm.state_to_reading(self.__trueState)
        posibilityDict[readingTrueState] = self.__om.get_o_reading_state(readingTrueState,self.__trueState)
        
        
        for move in possibleNewPos1:
            currPos = self.__sm.state_to_position(self.__trueState)
            newPosX = currPos[0] + move[0]
            newPosY = currPos[1] + move[1]

            readingState = self.__sm.position_to_reading(newPosX,newPosY)
            if readingState < self.__om.get_nr_of_readings()-1 and readingState>0:
                p = self.__om.get_o_reading_state(readingState, self.__trueState)
                posibilityDict[readingState] = p
                if p != 0:
                    n_Ls += 1

        for move in possibleNewPos2:
            currPos = self.__sm.state_to_position(self.__trueState)
            newPosX = currPos[0] + move[0]
            newPosY = currPos[1] + move[1]
            readingState = self.__sm.position_to_reading(newPosX,newPosY)
            if readingState < self.__om.get_nr_of_readings()-1 and readingState>0:
                p = self.__om.get_o_reading_state(readingState, self.__trueState)
                posibilityDict[readingState] = p
                if p != 0:
                    n_Ls_2 += 1                
        
        nothing = 1-0.1-n_Ls*0.05-n_Ls_2*0.025
        posibilityDict[None] = nothing
        return posibilityDict
    
    def update(self) -> (bool, int, int, int, int, int, int, int, int, np.array(1)) :
        # update all the values to something sensible instead of just reading the old values...
        # 
        # this block can be kept as is
        
        newX,newY,newH = self.move_robot_and_update()
        sensor_reading_dict = self.get_sensor_reading()
        tsX, tsY, tsH = newX,newY,newH
        
        allsquares = np.zeros(len(self.__probs))
        allsquares[0] = sensor_reading_dict[None]
        for i in range(len(allsquares)):
            if i in list(sensor_reading_dict.keys()):
                allsquares[i] = sensor_reading_dict[i]
        
        
        sens_choice = random.choices(np.arange(len(allsquares)), weights = allsquares)[0]
        if sens_choice == 0:
            self.__sense = None
        else:
            self.__sense = sens_choice
        print(self.__sense)
        T_transp = self.__tm.get_T_transp()
        self.__probs, best_state = self.__HMM.forward_filter(self.__sense, T_transp, self.__probs)
        self.__estimate = self.__sm.state_to_position(best_state)
        
        
        eX, eY = self.__estimate
        
        if self.__sense != None:
            srX, srY = self.__sm.reading_to_position(self.__sense)
            ret = True
            
            if tsX == eX and tsY == eY :
                self.__correctEstimates += 1
                self.__dist.append(0)
            else:
                x_diff = abs(eX - tsX)
                y_diff = abs(eY - tsY)
                manhattan = x_diff + y_diff
                self.__totalError += 1
                self.__dist.append(manhattan)
        else:
            srX = -1
            srY = -1
            ret = False
            
        
        
        # this should be updated to spit out the actual error for this step
        #error = self.__dist[-1]      
        
        #hit_rate = self.__correctEstimates/(len(self.__dist))
        error = abs(eY - tsY) + abs(eX - tsX)

        return ret, tsX, tsY, tsH, srX, srY, eX, eY, error, self.__probs