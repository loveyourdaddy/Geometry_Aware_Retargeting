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


""" load characters """
source_name = "SMPLx"
character_normal, Tpose_normal, _ = get_a_smpl_character(args, source_name)
character_ptn = character_normal

# character_dfm
deformed_name = args.target_characters[1] # 0 1
index = -1 # 0~9 3 7 
character_dfm, Tpose_deformed, = get_a_smpl_character_wo_geo(args, deformed_name, scale=0.7)

# role
role_change = False
if role_change==False:
    rid = 0
else:
    rid = 1


""" load edited motion """
# original motion
motion_keys   = example_bvh.keys()
motion_values = example_bvh.values()
idx_in_example = 0
motion0 = get_interaction_motions_from_list("SMPLx", motion_keys)[idx_in_example]
motion1 = get_interaction_motions_from_list("SMPLx", motion_values)[idx_in_example]
motion_name = list(motion_keys)[idx_in_example] # original motion of example BVH list

# load saved edited motion
# index 
scales = get_scale()
scale = scales[index]
# scale
leg_scale  = scale[0]
body_scale = scale[1]
hand_scale = scale[2]
sid = index

# copy and paste motion
motionA, motionB = \
    load_edited_npy_motion(args, motion0, motion1, deformed_name, motion_name,
                            leg_scale, body_scale, hand_scale, rid, sid)

characters, motions = \
    render_result(args,
                  character_normal, character_normal, character_ptn, character_dfm,
                  motion0, motion1, motionA, motionB)


# foot contact 
# by motion
# foot_contact0 = detect_foot_contact(args, motion0)
# foot_contact_index = np.where(foot_contact0)
# len_frame = len(motion0.poses) # position0.shape[0]
# for i in range(len(foot_contact_index[0])):
#     f = foot_contact_index[0][i]
#     j = foot_contact_index[1][i]
#     print(f"{f} {j}")
#     if f > len_frame/2 and 4==4:
#         # args.debug_points0.append(motion0.poses[f].global_p[j])
#         args.debug_points0.append(motionB.poses[f].global_p[j])

# by position
position0 = [] 
for pose in motion0.poses:
    position0.append(pose.global_p)
position0 = np.array(position0)

# print()
len_frame = len(motion0.poses)
foot_contact0 = detect_foot_contact_from_position(args, position0)
foot_contact_index = np.where(foot_contact0)
for i in range(len(foot_contact_index[0])):
    # ee 
    f = foot_contact_index[0][i]
    j = foot_contact_index[1][i]
    if j < len_frame/2:
        args.debug_points0.append(motion0.poses[f].global_p[j])
    
    # # before ee 
    # f = foot_heel_contact_index[0][i]
    # j = foot_heel_contact_index[1][i]
    # if j < len_frame/2:
    #     args.debug_points1.append(motion0.poses[f].global_p[j])

app = MyApp(characters, motions, args)
app_manager.run(app)
