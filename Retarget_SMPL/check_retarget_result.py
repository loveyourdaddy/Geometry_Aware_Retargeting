import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append('../')

import option_parser
from detect_foot_contact import *
from option_motion import example_bvh
from Retarget_SMPL.retarget_smpl import load_edited_npy_motion
from etc.etc import *
from datasets.motion_functions import *
from datasets.character_functions import *
from pymovis.vis.app import MyApp
from pymovis.vis.appmanager import AppManager


app_manager = AppManager()
args = option_parser.get_args()
args.path = args.proj_name + '/'


# character_dfm
args.test_type = "SMPLx" # SMPLx Mixamo
args.target_characters[1] = "SMPLx" # SMPLx_fat
deformed_name = args.target_characters[1]

# scale index (0~9 3 7)
index = 0 # small 
# index = 7 # fat
# role
role_change = False


""" load characters """
source_name = "SMPLx"
character_normal, Tpose_normal, _ = get_a_smpl_character(args, source_name)
character_ptn = character_normal
character_dfm, Tpose_deformed = get_a_smpl_character_wo_geo(args, deformed_name, scale=0.7)

""" load edited motion """
# original motion
motion_keys   = example_bvh.keys()
motion_values = example_bvh.values()
idx_in_example = 0
motion0 = get_interaction_motions_from_list("SMPLx", motion_keys)[idx_in_example]
motion1 = get_interaction_motions_from_list("SMPLx", motion_values)[idx_in_example]
motion_name = list(motion_keys)[idx_in_example] # original motion of example BVH list

# scale
scales = get_scale()
scale = scales[index]
# scale = [1.0, 1.0, 1.0]

leg_scale  = scale[0]
body_scale = scale[1]
hand_scale = scale[2]
sid = index

if role_change==False:
    rid = 0
else:
    rid = 1

# load saved edited motion
# copy and paste motion
motionA, motionB = \
    load_edited_npy_motion(args, motion0, motion1, deformed_name, motion_name,
                            leg_scale, body_scale, hand_scale, rid, sid)

characters, motions = \
    render_result(args,
                  character_normal, character_normal, character_ptn, character_dfm,
                  motion0, motion1, motionA, motionB)

app = MyApp(characters, motions, args)
app_manager.run(app)
