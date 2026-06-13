"""
Apply adapation and save for all motion
    python Retarget_SMPL/run.py
"""

import os
import sys

current_file = os.path.abspath(__file__)
retarget_smpl_dir = os.path.dirname(current_file)
geometry_aware_dir = os.path.dirname(retarget_smpl_dir)
workspace_dir = os.path.dirname(geometry_aware_dir)
sys.path.append(os.path.dirname(geometry_aware_dir))
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

    for scale in scales:
        motion_ptn = deepcopy(motion0)
        motion_dfm = deepcopy(motion1)
        character_dfm.set_source_skeleton(motion_dfm.skeleton, "")
        geometry_dfm.source_skeleton = motion_dfm.skeleton

        offset_scale = deepcopy(dfm_offsets)
        for i in range(len(motion_dfm.skeleton.joints)):
            motion_dfm.skeleton.joints[i].offset = offset_scale[i]
        root_scale = offset_scale[0][1]
        for i in range(len(motion_dfm.poses)):
            motion_dfm.poses[i].root_p *= root_scale

        leg_scale, body_scale, arm_scale = scale[0], scale[1], scale[2]
        scale_offset_and_root(args, motion_dfm, leg_scale, leg_scale, body_scale, arm_scale)

        root_p0, local_R0, root_p1, local_R1 = retarget_smpl(
            args,
            geometry_nor, geometry_nor, geometry_nor, geometry_dfm,
            motion0, motion1, motion_ptn, motion_dfm,
            render=False,
            ptn_root_joints=ptn_root_joints, ptn_spine_joints=ptn_spine_joints, ptn_limb_joints=ptn_limb_joints,
            dfm_root_joints=dfm_root_joints, dfm_spine_joints=dfm_spine_joints, dfm_limb_joints=dfm_limb_joints)
        root_p_ptn.append(root_p0.numpy())
        local_R_ptn.append(local_R0.numpy())
        root_p_dfm.append(root_p1.numpy())
        local_R_dfm.append(local_R1.numpy())


app_manager = AppManager()
args = option_parser.get_args()
args.device = "cpu"

character_nor, _, geometry_nor = get_a_smpl_character(args, "SMPLx")
deformed_names = ["SMPLx_fat"]
print("deformed_names:", deformed_names)

motion_name0s = RD_bvh.keys()
motion_name1s = RD_bvh.values()
scales = [[1.0, 1.0, 1.0]]

time0 = time.time()
time_start = time.time()
for deformed_name in deformed_names:
    character_dfm, Tpose_dfm, geometry_dfm = get_a_smpl_character(args, deformed_name)
    dfm_offsets = [Tpose_dfm.skeleton.joints[i].offset for i in range(22)]

    for motion_name0, motion_name1 in zip(motion_name0s, motion_name1s):
        file_path = "./edited_RD/{}/".format(deformed_name)
        os.makedirs(file_path, exist_ok=True)

        clip_motion_name = motion_name0 if motion_name0 in start_frame_dict else motion_name0[7:]
        if clip_motion_name in start_frame_dict:
            args.interaction_start_frame = start_frame_dict[clip_motion_name]
            args.interaction_end_frame = end_frame_dict[clip_motion_name]
            args.update_by_clampping_range = True
        else:
            args.update_by_clampping_range = False

        motion0 = get_interaction_motions_from_list("SMPLx", [motion_name0])[0]
        motion1 = get_interaction_motions_from_list("SMPLx", [motion_name1])[0]

        ptn_root_joints, ptn_spine_joints, ptn_limb_joints, \
            dfm_root_joints, dfm_spine_joints, dfm_limb_joints = \
            update_target_joints(args, motion_name0, motion_name1)

        ptn_spine_joints = []
        if motion_name0 in ptn_spine_for_fat and deformed_name == "SMPLx_fat":
            ptn_spine_joints += args.RD_spine_joints

        root_p0s, local_R0s, root_p1s, local_R1s = [], [], [], []
        for rid in range(2):
            motion_ptn = motion0 if rid == 0 else motion1
            motion_dfm = motion1 if rid == 0 else motion0

            root_p_ptn, local_R_ptn, root_p_dfm, local_R_dfm = [], [], [], []
            retarget_with_scale(
                motion_ptn, geometry_nor,
                character_dfm, motion_dfm, geometry_dfm, scales, dfm_offsets,
                ptn_root_joints, ptn_spine_joints, ptn_limb_joints,
                dfm_root_joints, dfm_spine_joints, dfm_limb_joints,
                root_p_ptn, local_R_ptn, root_p_dfm, local_R_dfm)  
    
            if rid == 0:
                root_p0s.append(root_p_ptn);  local_R0s.append(local_R_ptn)
                root_p1s.append(root_p_dfm);  local_R1s.append(local_R_dfm)
            else:
                root_p0s.append(root_p_dfm);  local_R0s.append(local_R_dfm)
                root_p1s.append(root_p_ptn);  local_R1s.append(local_R_ptn)
        
        # np.save(file_path + '{}_root_p0.npy'.format(motion_name0), np.array(root_p0s))
        # np.save(file_path + '{}_root_p1.npy'.format(motion_name0), np.array(root_p1s))
        # np.save(file_path + '{}_local_R0.npy'.format(motion_name0), np.array(local_R0s))
        # np.save(file_path + '{}_local_R1.npy'.format(motion_name0), np.array(local_R1s))

        time_now = time.time()
        print("[{}] {} : {} ({}f) saved in {}".format(
            datetime.datetime.now().strftime("%H:%M:%S"),
            deformed_name, motion_name0, len(motion0.poses),
            datetime.timedelta(seconds=time_now - time0)))
        time0 = time_now

    time_end = time.time()
    print("[{}] Done: {}".format(
        datetime.datetime.now().strftime("%H:%M:%S"),
        datetime.timedelta(seconds=time_end - time_start)))
