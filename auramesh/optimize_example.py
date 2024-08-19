""" 
Goal of optimize
- motion1 follow motion0
"""

import sys, os
sys.path.append('.') # import: python을 실행하는 path 
sys.path.append('../') # Retargeting_workspace

from pymovis.vis.appmanager import AppManager
from pymovis.vis.app import MyApp
from etc.etc import render_motions
from copy import deepcopy
from xalglib import xalglib
from datasets.character_functions import get_a_character
from datasets.motion_functions import get_interaction_motions_from_list
from Geometry.geometry import Geometry
import option_parser
import numpy as np


# target = [1.0] * 198
def function1_grad(x, grad, param):
    func = 0
    for i in range(198):
        func += abs(x[i] - target[i])
        grad[i] = x[i] - target[i]
    return func

if __name__=="__main__":
    app_manager = AppManager()
    args = option_parser.get_args()

    # load character
    src_names = ["SMPLx", "SMPLx"]
    src_chars = []
    for name in src_names:
        char, _, _ = get_a_character(args, name)
        src_chars.append(char)

    # load motion data (motion_0, motion1)
    motion_0 = get_interaction_motions_from_list(src_names[0], ["HandShaking_JIS"])[0] # greeting002_S1

    # Target: zero rot
    motion_1 = deepcopy(motion_0)
    for pose in motion_1.poses:
        pose.local_R = np.identity(3)[None, :].repeat(22, axis=0)
        pose.update()
        # print(pose.local_R)


    """ optimize """
    # update motion1 
    epsx = 1e-6  # Desired precision for variables
    maxits = 100 # Maximum number of iterations
    stpmax = 0.1 # Maximum step length
    epsg = 0.0
    epsf = 0.0
    for i, pose in enumerate(motion_0.poses):
        # target의 이름은 같아야 공유됨
        target = pose.local_R.reshape(-1)

        # optimize 
        x0 = [0.0]*198
        state = xalglib.minlbfgscreate(2, x0)
        xalglib.minlbfgssetcond(state, epsg, epsf, epsx, maxits)
        xalglib.minlbfgsoptimize_g(state, function1_grad)
        x_optimized, rep = xalglib.minlbfgsresults(state)
        x_optimized = np.array(x_optimized) # list to np array
        
        # update 
        updated_target = x_optimized.reshape(22, 3, 3) 
        motion_1.poses[i].local_R = updated_target
        motion_1.poses[i].update()

    characters = src_chars
    motions = [motion_0, motion_1]
    render_motions(args, characters, motions)

    app = MyApp(characters, motions, args)
    app_manager.run(app)
