from models import *
from viewer import *

ROWS = 4
COLS = 4

# Testing the models, e.g., for an 4x8 grid

states = StateModel( 4, 8)
loc = Localizer(states)
tMat = loc.get_transition_model()
sVecs = loc.get_observation_model()
tMat.plot_T()
sVecs.plot_o_diags()
print(sVecs.get_o_reading(0))
print(sVecs.get_o_reading(None))

print(loc.update())

#Visualization


# the dashboard creates a state model of the dimensions given by ROWS and COLS, sets up the respective 
# Transition and Observation models, as well as an instance of class Localizer. All methods already 
# given in Localizer should thus keep their behaviour - otherwise the calls from Dashboard might result in 
# wrong output
dash = Dashboard.Dashboard(ROWS, COLS)
