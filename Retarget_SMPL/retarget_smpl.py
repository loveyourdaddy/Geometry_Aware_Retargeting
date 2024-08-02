import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append('..')

from pymovis.motion.ops.npmotion import R_to_R6
import option_parser
from option_motion import example_bvh, start_frame_dict, end_frame_dict,\
    ptn_root_motion, ptn_spine_motion, ptn_leg_motion, ptn_spine_for_fat, ptn_not_arm_motion,\
    dfm_root_motion, dfm_spine_motion, dfm_leg_motion
from Retarget_SMPL.relationship_descriptor import *
from etc.etc import render_result, render_compare, deepcopy
from pymovis.vis.app import MyApp
from pymovis.vis.appmanager import AppManager
# from Geometry.compare_geometry import collision_check_and_resolve

def retarget_smpl(args,
                  geo_source0, geo_source1, geo_target0, geo_target1,
                  motion0, motion1, edited_motion0, edited_motion1,
                  render=True, pene=False,
                  ptn_root_joints=None, ptn_spine_joints=None, ptn_limb_joints=None, 
                  dfm_root_joints=None, dfm_spine_joints=None, dfm_limb_joints=None,):

    """ edit motion """
    # Partner (charA, motion0) : joint A <-> anchor B
    edited_motion0 = retarget_one_motion(args,
                                   geo_source1, geo_target1,
                                   motion0, motion1,
                                   edited_motion0, edited_motion1,
                                   root_joints=ptn_root_joints, spine_joints=ptn_spine_joints, limb_joints=ptn_limb_joints)

    # Deformed (charB, motion1): anchor A <-> Joint B
    edited_motion1 = retarget_one_motion(args,
                                   geo_source0, geo_target0,
                                   motion1, motion0,
                                   edited_motion1, edited_motion0,
                                   root_joints=dfm_root_joints, spine_joints=dfm_spine_joints, limb_joints=dfm_limb_joints)

    # if pene:
    #     joints = [15,16,17, 19,20,21]
    #     ptn_joints = list(range(22))
    #     collision_check_and_resolve(args, geo_target0, geo_target1, edited_motion0, motion1, joints, ptn_joints)
    
    # output
    root_p0, local_R0, _ = get_rootP_localR_globalP_from_motion(args, edited_motion0.poses)
    root_p1, local_R1, _ = get_rootP_localR_globalP_from_motion(args, edited_motion1.poses)

    if render:
        return root_p0, local_R0, root_p1, local_R1, \
            motion0, motion1, edited_motion0, edited_motion1

    return root_p0, local_R0, root_p1, local_R1


""" etc """
# load a edited motion
def load_full_edited_npy_motion(args, character, motion_name):
    # A, 0: ptn.  B, 1: deform
    deform_name = character.meshes[0].mesh_gl.name

    # info of edited motion
    root_p0 = np.load('./edited_RD/{}/{}_root_p0.npy'.format(deform_name, motion_name))
    local_R0 = np.load('./edited_RD/{}/{}_local_R0.npy'.format(deform_name, motion_name))
    root_p1 = np.load('./edited_RD/{}/{}_root_p1.npy'.format(deform_name, motion_name))
    local_R1 = np.load('./edited_RD/{}/{}_local_R1.npy'.format(deform_name, motion_name))

    num_role, num_scale, len_frame, _ = root_p0.shape

    # args.rotation_rep == 'R6'
    motion0 = np.concatenate((R_to_R6(local_R0).reshape(num_role, num_scale, len_frame, -1), root_p0), axis=-1)
    motion1 = np.concatenate((R_to_R6(local_R1).reshape(num_role, num_scale, len_frame, -1), root_p1), axis=-1)

    return motion0, motion1

def load_edited_npy_motion(args, motionA, motionB, char_name, motion_name,
                           leg_scale, body_scale, hand_scale, rid, sid):
    # skeleton
    motion0, motion1 = deepcopy(motionA), deepcopy(motionB)
    if rid==0:
        scale_offset(args, motion1, leg_scale, body_scale, hand_scale)
    else:
        scale_offset(args, motion0, leg_scale, body_scale, hand_scale)
    
    root_p0  = np.load('./edited_RD/{}/{}_root_p0.npy' .format(char_name, motion_name))[rid, sid]
    local_R0 = np.load('./edited_RD/{}/{}_local_R0.npy'.format(char_name, motion_name))[rid, sid]
    root_p1  = np.load('./edited_RD/{}/{}_root_p1.npy' .format(char_name, motion_name))[rid, sid]
    local_R1 = np.load('./edited_RD/{}/{}_local_R1.npy'.format(char_name, motion_name))[rid, sid]

    # motion to edit
    for i in range(len(motion0.poses)):
        motion0.poses[i].root_p = root_p0[i]
        motion0.poses[i].local_R = local_R0[i]
        motion0.poses[i].update()
        motion1.poses[i].root_p = root_p1[i]
        motion1.poses[i].local_R = local_R1[i]
        motion1.poses[i].update()

    return motion0, motion1

def motion_add_virtual_joints(args, motion):
    offset_r = args.offset_r

    # motion
    skeleton = motion.skeleton
    left_hand_idx = 17
    left_hand = skeleton.joints[left_hand_idx]
    right_hand_idx = 21
    right_hand = skeleton.joints[right_hand_idx]
    skeleton.add_joint(left_hand.name+"_ee",
                       parent_idx=left_hand_idx,  offset=offset_r*left_hand.offset)
    skeleton.add_joint(right_hand.name+"_ee",
                       parent_idx=right_hand_idx, offset=offset_r*right_hand.offset)

    iden = np.expand_dims(np.eye(3, dtype=np.float32), axis=0)
    for pose in motion.poses:
        pose.local_R = np.append(pose.local_R, iden, axis=0)
        pose.local_R = np.append(pose.local_R, iden, axis=0)
        pose.update()

    return motion

def scale_character(args, character, leg_scale, body_scale, hand_scale):
    # edit character skeleton
    for joint in args.leg_joints:
        character.meshes[0].source_skeleton.joints[joint].offset *= leg_scale
    for joint in args.body_joints:
        character.meshes[0].source_skeleton.joints[joint].offset *= body_scale
    for joint in args.hand_joints:
        character.meshes[0].source_skeleton.joints[joint].offset *= hand_scale

def scale_offset_and_root(args, motion, root_scale, leg_scale, body_scale, hand_scale):
    # edit character skeleton
    scale_offset(args, motion, leg_scale, body_scale, hand_scale)
    
    # edit root
    for pose in motion.poses:
        pose.root_p[1] *= root_scale
        pose.update()

def scale_offset(args, motion, leg_scale, body_scale, hand_scale):
    for joint in args.leg_joints:
        motion.skeleton.joints[joint].offset *= leg_scale
    for joint in args.body_joints:
        motion.skeleton.joints[joint].offset *= body_scale
    for joint in args.hand_joints:
        motion.skeleton.joints[joint].offset *= hand_scale

def update_target_joints(args, motion_name0, motion_name1):
    """ ptn """
    # root 
    ptn_root_joints = []
    if motion_name0 in ptn_root_motion or motion_name1 in ptn_root_motion:
        ptn_root_joints += [0]
    # spine 
    ptn_spine_joints = [] 
    if motion_name0 in ptn_spine_motion or motion_name1 in ptn_spine_motion:
        ptn_spine_joints += args.RD_spine_joints
    # limb
    ptn_limb_joints = args.RD_hand_joints
    if motion_name0 in ptn_not_arm_motion or motion_name1 in ptn_not_arm_motion:
        ptn_limb_joints = []
    if motion_name0 in ptn_leg_motion or motion_name1 in ptn_leg_motion:
        ptn_limb_joints += args.RD_leg_joints

    
    """ dfm """
    # Root
    dfm_root_joints = []
    if motion_name0 in dfm_root_motion or motion_name1 in dfm_root_motion:
        dfm_root_joints += [0]
    # spine
    dfm_spine_joints = [] 
    if motion_name0 in dfm_spine_motion or motion_name1 in dfm_spine_motion:
        dfm_spine_joints += args.RD_spine_joints
    # limb
    dfm_limb_joints = args.RD_hand_joints
    if motion_name0 in dfm_leg_motion or motion_name1 in dfm_leg_motion:
        dfm_limb_joints += args.RD_leg_joints
        
    return ptn_root_joints, ptn_spine_joints, ptn_limb_joints,\
        dfm_root_joints, dfm_spine_joints, dfm_limb_joints

if __name__ == '__main__':
    app_manager = AppManager()
    args = option_parser.get_args()
    args.device = "cpu" # "cuda"

    """ character, moiton """
    from datasets.character_functions import get_a_smpl_character
    # ptn
    ptn_name = "SMPLx"
    character_ptn, _, geometry_nor = get_a_smpl_character(args, ptn_name)
    # dfm
    deformed_name = args.target_characters[1] # 0 1
    index = -1 # 0 # 
    role_change = False # True False
    character_dfm, Tpose_dfm, geometry_dfm = get_a_smpl_character(args, deformed_name)

    # motion name
    from datasets.motion_functions import get_interaction_motions_from_list
    motion_name0 = list(example_bvh.keys())[0]
    motion_name1 = list(example_bvh.values())[0]
    motion0 = get_interaction_motions_from_list("SMPLx", [motion_name0])[0]
    motion1 = get_interaction_motions_from_list("SMPLx", [motion_name1])[0]

    # limb target
    ptn_root_joints, ptn_spine_joints, ptn_limb_joints, \
    dfm_root_joints, dfm_spine_joints, dfm_limb_joints = \
        update_target_joints(args, motion_name0, motion_name1)
    # only for fat 
    ptn_spine_joints = [] 
    if motion_name0 in ptn_spine_for_fat and deformed_name=="SMPLx_fat":
        ptn_spine_joints += args.RD_spine_joints
    
    # swap role 
    if role_change:
        tmp = deepcopy(motion1)
        motion1 = deepcopy(motion0)
        motion0 = deepcopy(tmp)
    # target motion
    motion_ptn = deepcopy(motion0)
    motion_dfm = deepcopy(motion1)
    character_dfm.set_source_skeleton(motion_dfm.skeleton, "")
    geometry_dfm.source_skeleton = motion_dfm.skeleton
    
    # fit smplx motion to new skeleton
    # offset 
    for i in range(len(motion_dfm.skeleton.joints)):
        motion_dfm.skeleton.joints[i].offset = Tpose_dfm.skeleton.joints[i].offset
    # root scale
    root_scale = Tpose_dfm.skeleton.joints[0].offset[1]
    for i in range(len(motion_dfm.poses)):
        motion_dfm.poses[i].root_p *= root_scale
    
    # scale
    if deformed_name == "SMPLx":
        scales = np.load("./scale_values/scales.npy")
    else:
        scales = np.load("./scale_values/scales_fat.npy")
    
    leg_idx  = index
    body_idx = index
    hand_idx = index
    leg_scale  = scales[leg_idx]
    body_scale = scales[body_idx]
    hand_scale = scales[hand_idx]
    print("leg_scale {}, body_scale {}, hand_scale {} ".format(leg_scale, body_scale, hand_scale))
    
    # scale by joint
    root_scale = leg_scale
    scale_offset_and_root(args, motion_dfm, root_scale, leg_scale, body_scale, hand_scale)
    
    
    """ setting """
    # clipping
    args.update_by_clampping_range = True # False
    if args.update_by_clampping_range:
        if motion_name0 in start_frame_dict.keys():
            args.interaction_start_frame = start_frame_dict[motion_name0]
            args.interaction_end_frame = end_frame_dict[motion_name0]
            print("{}'s interaction {} ~ {}".format(motion_name0, args.interaction_start_frame, args.interaction_end_frame))

    """ retarget """
    root_p0, local_R0, root_p1, local_R1, motion0, motion1, motion_ptn, motion_dfm = \
        retarget_smpl(args, 
                      geometry_nor, geometry_nor, geometry_nor, geometry_dfm,
                      motion0, motion1, motion_ptn, motion_dfm,
                      render=True, pene=False,
                      ptn_root_joints=ptn_root_joints, ptn_spine_joints=ptn_spine_joints, ptn_limb_joints=ptn_limb_joints, 
                      dfm_root_joints=dfm_root_joints, dfm_spine_joints=dfm_spine_joints, dfm_limb_joints=dfm_limb_joints,)

    # render
    characters, motions = \
        render_result(args, character_ptn, character_ptn, character_ptn, character_dfm,
                        motion0, motion1, motion_ptn, motion_dfm)
    app = MyApp(characters, motions, args)
    app_manager.run(app)
