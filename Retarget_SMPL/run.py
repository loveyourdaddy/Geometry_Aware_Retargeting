"""
python Retarget_SMPL/run.py
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

current_file = os.path.abspath(__file__)
retarget_smpl_dir = os.path.dirname(current_file)  # Retarget_SMPL
geometry_aware_dir = os.path.dirname(retarget_smpl_dir)  # geometry_aware_retargeting
workspace_dir = os.path.dirname(geometry_aware_dir)  # Retargeting_workspace
# sys.path.append(geometry_aware_dir)
sys.path.append(workspace_dir)

import time
import datetime
from datasets.motion_functions import *
from datasets.character_functions import *
import option_parser
from option_motion import RD_bvh, start_frame_dict, end_frame_dict
from retarget_smpl import *
from pymovis.vis.appmanager import AppManager

def retarget_with_scale(
    motion0, geometry_nor,
    character_dfm, motion1, geometry_dfm, scales, dfm_offsets,  
    ptn_root_joints, ptn_spine_joints, ptn_limb_joints,
    dfm_root_joints, dfm_spine_joints, dfm_limb_joints,
    root_p_ptn, local_R_ptn, root_p_dfm, local_R_dfm): 
    
    # motion1 is changed 
    for scale in scales:
        # set character and motion 
        motion_ptn = deepcopy(motion0)
        motion_dfm = deepcopy(motion1)
        character_dfm.set_source_skeleton(motion_dfm.skeleton, "")
        geometry_dfm.source_skeleton = motion_dfm.skeleton
        
        # fit smplx motion to new skeleton (fat)
        # offset of skeleton 
        offset_scale = deepcopy(dfm_offsets)
        for i in range(len(motion_dfm.skeleton.joints)):
            motion_dfm.skeleton.joints[i].offset = offset_scale[i]
        # root scale (normal에서는 효과 없음. fat에서는 효과있지만)
        root_scale = offset_scale[0][1]
        for i in range(len(motion_dfm.poses)):
            motion_dfm.poses[i].root_p *= root_scale
        
        # fit (scaling) to deformed character
        leg_scale = scale[0]
        body_scale = scale[1]
        arm_scale = scale[2]
        root_scale = leg_scale
        scale_offset_and_root(args, motion_dfm, root_scale, leg_scale, body_scale, arm_scale)
        
        # retarget 
        root_p0, local_R0, root_p1, local_R1, = \
            retarget_smpl(args, 
                        geometry_nor, geometry_nor, geometry_nor, geometry_dfm,
                        motion0, motion1, motion_ptn, motion_dfm,
                        render=False, 
                        ptn_root_joints=ptn_root_joints, ptn_spine_joints=ptn_spine_joints, ptn_limb_joints=ptn_limb_joints, 
                        dfm_root_joints=dfm_root_joints, dfm_spine_joints=dfm_spine_joints, dfm_limb_joints=dfm_limb_joints,)
        root_p_ptn .append(root_p0.numpy())
        local_R_ptn.append(local_R0.numpy())
        root_p_dfm .append(root_p1.numpy())
        local_R_dfm.append(local_R1.numpy())

app_manager = AppManager()
args = option_parser.get_args()
args.device = "cpu"

""" source character, moiton """
# characater0
character_nor, _, geometry_nor = get_a_smpl_character(args, "SMPLx")
deformed_names = args.target_characters # TODO [1:2]
print("deformed_names:", deformed_names)

# motion
motion_name0s = RD_bvh.keys()
motion_name1s = RD_bvh.values()

scales = np.load("scale_values/scales_sampled.npy")
# print("{} listOfNumbers: {}".format(deformed_name, scales))

time0 = time.time()
time_start = time.time()
for deformed_name in deformed_names:

    """ target character, moiton """
    # characater dfm 
    character_dfm, Tpose_dfm, geometry_dfm = get_a_smpl_character(args, deformed_name)
    dfm_offsets = []
    for i in range(22): 
        dfm_offsets.append(Tpose_dfm.skeleton.joints[i].offset)
    
    # retarget 
    for motion_name0, motion_name1 in zip(motion_name0s, motion_name1s):
        # mkdir
        file_path = "./edited_RD/{}/".format(deformed_name)
        os.makedirs(file_path, exist_ok=True)
        # clipping
        if motion_name0 in start_frame_dict.keys() or motion_name0[7:] in start_frame_dict.keys(): 
            clip_motion_name = motion_name0
            # mirror case 
            if clip_motion_name not in start_frame_dict.keys():
                clip_motion_name = motion_name0[7:]
            args.interaction_start_frame = start_frame_dict[clip_motion_name]
            args.interaction_end_frame = end_frame_dict[clip_motion_name]
            args.update_by_clampping_range = True
        else:
            args.update_by_clampping_range = False
        
        # motion 
        motion0 = get_interaction_motions_from_list("SMPLx", [motion_name0])[0]
        motion1 = get_interaction_motions_from_list("SMPLx", [motion_name1])[0]

        # limb target by relational rescritor
        ptn_root_joints, ptn_spine_joints, ptn_limb_joints, \
        dfm_root_joints, dfm_spine_joints, dfm_limb_joints = \
            update_target_joints(args, motion_name0, motion_name1)
        # only for fat 
        ptn_spine_joints = [] 
        if motion_name0 in ptn_spine_for_fat and deformed_name=="SMPLx_fat":
            ptn_spine_joints += args.RD_spine_joints

        # RD 
        # char 1 changes
        root_p0s, local_R0s, root_p1s, local_R1s = [], [], [], []
        for rid in range(2):
            if rid==0:
                motion_ptn = motion0
                motion_dfm = motion1
            else: #  rid==1: role_change 
                motion_dfm = motion0
                motion_ptn = motion1
                
            root_p_ptn, local_R_ptn, root_p_dfm, local_R_dfm = [], [], [], []
            retarget_with_scale(
                motion_ptn, geometry_nor,
                character_dfm, motion_dfm, geometry_dfm, scales, dfm_offsets, 
                ptn_root_joints, ptn_spine_joints, ptn_limb_joints,
                dfm_root_joints, dfm_spine_joints, dfm_limb_joints,
                root_p_ptn, local_R_ptn, root_p_dfm, local_R_dfm)
            if rid==0:
                root_p0s.append(root_p_ptn)
                local_R0s.append(local_R_ptn)
                root_p1s.append(root_p_dfm)
                local_R1s.append(local_R_dfm)
            else:
                root_p0s.append(root_p_dfm)
                local_R0s.append(local_R_dfm)
                root_p1s.append(root_p_ptn)
                local_R1s.append(local_R_ptn)
                
        np.save(file_path + '{}_root_p0.npy'.format(motion_name0), np.array(root_p0s))
        np.save(file_path + '{}_root_p1.npy'.format(motion_name0), np.array(root_p1s))
        np.save(file_path + '{}_local_R0.npy'.format(motion_name0), np.array(local_R0s))
        np.save(file_path + '{}_local_R1.npy'.format(motion_name0), np.array(local_R1s))
        
        time_now = time.time()
        print("{} : {} saved in ".format(deformed_name, motion_name0), datetime.timedelta(seconds=time_now - time0))
        time0 = time_now
    time_end = time.time()
    print("Done: ", datetime.timedelta(seconds=(time_end - time_start)))
