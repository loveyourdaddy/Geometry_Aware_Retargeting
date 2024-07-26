from pymovis.vis.appmanager import AppManager
from pymovis.vis.app import MyApp
from etc.etc import render_motions
import option_parser
from run_auramesh import get_character, get_motion
from copy import deepcopy
import numpy as np
from xalglib import xalglib

app_manager = AppManager()
args = option_parser.get_args()

# load character
src_names = ["SMPLx", "SMPLx"]
src_chars = get_character(args, src_names)

# load motion data (motion_0, motion_1)
motion_0 = get_motion(src_names[0], "greeting002_S1")
motion_1 = deepcopy(motion_0)
for pose in motion_1.poses:
    pose.local_R = np.identity(3)[None, :].repeat(22, axis=0)
    pose.update()

epsx = 1e-6  # Desired precision for variables
maxits = 100 # Maximum number of iterations
stpmax = 0.1 # Maximum step length
epsg = 0.0
epsf = 0.0
# optimize 
for i, pose in enumerate(motion_0.poses):
    target = pose.local_R.reshape(-1)

    # target = [1.0] * 198
    def function1_grad(x, grad, param):
        func = 0
        for i in range(198):
            func += abs(x[i] - target[i])
            grad[i] = x[i] - target[i]
        return func

    x0 = [0.0]*198
    state = xalglib.minlbfgscreate(2, x0)
    xalglib.minlbfgssetcond(state, epsg, epsf, epsx, maxits)
    xalglib.minlbfgsoptimize_g(state, function1_grad)
    x_optimized, rep = xalglib.minlbfgsresults(state)
    
    motion_1.poses[i].local_R = np.array(x_optimized).reshape(22, 3, 3)
    motion_1.poses[i].update()
    # print("i {}: {}".format(i, pose.local_R - motion_1.poses[i].local_R))

characters = src_chars
motions = [motion_0, motion_1]
render_motions(args, characters, motions)

app = MyApp(characters, motions)
app_manager.run(app)
