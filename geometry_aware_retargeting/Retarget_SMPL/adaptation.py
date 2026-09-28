"""SMPL adaptation: edit the motions of two characters with respect to each other's surface."""
from etc.etc import deepcopy
from option_motion import start_frame_dict, end_frame_dict, \
    ptn_root_motion, ptn_spine_motion, ptn_leg_motion, ptn_spine_for_fat, ptn_not_arm_motion, \
    dfm_root_motion, dfm_spine_motion, dfm_leg_motion
from Retarget_SMPL.relationship_descriptor import retarget_one_motion
from Retarget_SMPL.motion_utils import get_rootP_localR_globalP_from_motion
from Retarget_SMPL.skeleton_scale import fit_motion_to_offsets, scale_offset_and_root


def retarget_smpl(args,
                  geo_source0, geo_source1, geo_target0, geo_target1,
                  motion0, motion1, edited_motion0, edited_motion1,
                  ptn_root_joints=None, ptn_spine_joints=None, ptn_limb_joints=None,
                  dfm_root_joints=None, dfm_spine_joints=None, dfm_limb_joints=None):
    """Edit edited_motion0 (partner) and edited_motion1 (deformed) in place and return (root_p0, local_R0, root_p1, local_R1)."""
    # partner (charA, motion0): joint A <-> anchor B
    edited_motion0 = retarget_one_motion(args,
                                         geo_source1, geo_target1,
                                         motion0, motion1,
                                         edited_motion0, edited_motion1,
                                         root_joints=ptn_root_joints, spine_joints=ptn_spine_joints, limb_joints=ptn_limb_joints)

    # deformed (charB, motion1): anchor A <-> joint B
    edited_motion1 = retarget_one_motion(args,
                                         geo_source0, geo_target0,
                                         motion1, motion0,
                                         edited_motion1, edited_motion0,
                                         root_joints=dfm_root_joints, spine_joints=dfm_spine_joints, limb_joints=dfm_limb_joints)

    root_p0, local_R0, _ = get_rootP_localR_globalP_from_motion(args, edited_motion0.poses)
    root_p1, local_R1, _ = get_rootP_localR_globalP_from_motion(args, edited_motion1.poses)

    return root_p0, local_R0, root_p1, local_R1

def update_target_joints(args, motion_name0, motion_name1, deformed_name=None):
    """Joints to update for each motion: root, spine, and limb for the partner (ptn) and the deformed (dfm)."""
    names = (motion_name0, motion_name1)
    def listed(motion_list):
        return any(name in motion_list for name in names)

    """ ptn """
    ptn_root_joints = []
    if listed(ptn_root_motion):
        ptn_root_joints += [0]
    ptn_spine_joints = []
    if listed(ptn_spine_motion) or (deformed_name == "SMPLx_fat" and motion_name0 in ptn_spine_for_fat):
        ptn_spine_joints += args.RD_spine_joints
    # TODO: += mutates args.RD_hand_joints itself, so leg joints remain for the motions after a leg motion
    ptn_limb_joints = args.RD_hand_joints
    if listed(ptn_not_arm_motion):
        ptn_limb_joints = []
    if listed(ptn_leg_motion):
        ptn_limb_joints += args.RD_leg_joints

    """ dfm """
    dfm_root_joints = []
    if listed(dfm_root_motion):
        dfm_root_joints += [0]
    dfm_spine_joints = []
    if listed(dfm_spine_motion):
        dfm_spine_joints += args.RD_spine_joints
    dfm_limb_joints = args.RD_hand_joints
    if listed(dfm_leg_motion):
        dfm_limb_joints += args.RD_leg_joints

    return ptn_root_joints, ptn_spine_joints, ptn_limb_joints,\
        dfm_root_joints, dfm_spine_joints, dfm_limb_joints

def set_interaction_range(args, motion_name):
    """Clamp to the interaction range only for motions with registered start/end frames. Mirrored motions use the original range."""
    clip_name = motion_name if motion_name in start_frame_dict else motion_name[len("mirror_"):]
    args.update_by_clampping_range = clip_name in start_frame_dict
    if args.update_by_clampping_range:
        args.interaction_start_frame = start_frame_dict[clip_name]
        args.interaction_end_frame = end_frame_dict[clip_name]

def make_target_motion(args, motion, character, geometry, offsets, scale, rel_dict=""):
    """Copy of the source motion fitted to the target character's skeleton (offsets) and scale (leg, body, hand)."""
    target_motion = deepcopy(motion)
    character.set_source_skeleton(target_motion.skeleton, rel_dict)
    geometry.source_skeleton = target_motion.skeleton

    fit_motion_to_offsets(target_motion, offsets)
    leg_scale, body_scale, hand_scale = scale
    scale_offset_and_root(args, target_motion, leg_scale, leg_scale, body_scale, hand_scale)

    return target_motion
