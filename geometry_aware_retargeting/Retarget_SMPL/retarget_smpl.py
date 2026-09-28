'''
conda activate py_env37_2
python Retarget_SMPL/retarget_smpl.py

Parameters
--save: adapt all of RD_bvh and save to adapted_motion/ without rendering. True/False(default)
--adapt_char: SMPLx(default), Mixamo
--target_characters: [partner, deformed]. The viewer uses the deformed character as the target

Viewer (first motion of example_bvh)
    python Retarget_SMPL/retarget_smpl.py --target_characters "['SMPLx','SMPLx_fat']"

Save (all of RD_bvh, both roles)
    python Retarget_SMPL/retarget_smpl.py --save True
'''

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('../')

import time
import datetime

from pymovis.vis.appmanager import AppManager
from pymovis.vis.app import MyApp
from pymovis.vis.const import MIXAMO_BVH_TO_FBX
from pymovis.motion.data import bvh
from etc.etc import render_result, deepcopy
from datasets.character_functions import get_a_smpl_character, get_a_character
from datasets.motion_functions import get_interaction_motions_from_list
import option_parser
from option_motion import example_bvh, RD_bvh
from Retarget_SMPL.adaptation import retarget_smpl, update_target_joints, set_interaction_range, make_target_motion
from Retarget_SMPL.edited_motion import save_edited_npy_motion
# names that other scripts import from this module
from Retarget_SMPL.skeleton_scale import scale_character, scale_offset, scale_offset_and_root
from Retarget_SMPL.edited_motion import load_full_edited_npy_motion, load_edited_npy_motion


def set_adapt_args(args):
    args.device         = "cpu"
    args.deformed_names = ["SMPLx_fat"]      # target characters to build with --save True
    args.scales         = [[1.0, 1.0, 1.0]]  # (leg, body, hand)  # np.load("scale_values/scales_sampled.npy")


def load_characters(args):
    """(source, partner, deformed) names, skeleton mapping, and (character, Tpose, geometry) of each character."""
    if args.adapt_char == "SMPLx":
        names = ("SMPLx", "SMPLx", args.target_characters[1])
        rel_dict = ""
        characters = [get_a_smpl_character(args, name) for name in names]
    else:
        names = ("Ybot", "Leonard", "Amy") # Ortiz Amy
        rel_dict = MIXAMO_BVH_TO_FBX
        template_Tpose = bvh.load(
            "../Resource/Tpose_template.bvh", v_forward=[0, 0, 1], v_up=[0, 1, 0]
        )
        characters = [get_a_character(args, name, template_Tpose) for name in names]

    return names, rel_dict, characters


def view_adaptation(args, app_manager):
    """Adapt the first motion of example_bvh and render it."""
    (src_name, _, deformed_name), rel_dict, characters = load_characters(args)
    (character_src, _, geometry_src), \
        (character_ptn, _, geometry_ptn), \
        (character_dfm, Tpose_dfm, geometry_dfm) = characters

    motion_name0, motion_name1 = list(example_bvh.items())[0]
    motion0 = get_interaction_motions_from_list(src_name, [motion_name0])[0]
    motion1 = get_interaction_motions_from_list(src_name, [motion_name1])[0]
    print(f"> motion:    {motion_name0}")
    print(f"> character: {deformed_name}")

    ptn_root_joints, ptn_spine_joints, ptn_limb_joints, \
        dfm_root_joints, dfm_spine_joints, dfm_limb_joints = \
        update_target_joints(args, motion_name0, motion_name1, deformed_name)
    set_interaction_range(args, motion_name0)

    offsets = [joint.offset for joint in Tpose_dfm.skeleton.joints]
    motion_ptn = deepcopy(motion0)
    motion_dfm = make_target_motion(args, motion1, character_dfm, geometry_dfm, offsets, [1, 1, 1], rel_dict)

    retarget_smpl(args,
                  geometry_src, geometry_src, geometry_ptn, geometry_dfm,
                  motion0, motion1, motion_ptn, motion_dfm,
                  ptn_root_joints=ptn_root_joints, ptn_spine_joints=ptn_spine_joints, ptn_limb_joints=ptn_limb_joints,
                  dfm_root_joints=dfm_root_joints, dfm_spine_joints=dfm_spine_joints, dfm_limb_joints=dfm_limb_joints)

    characters, motions = \
        render_result(args,
                      character_src, character_src, character_ptn, character_dfm,
                      motion0, motion1, motion_ptn, motion_dfm)
    app = MyApp(characters, motions, args)
    app_manager.run(app)


def adapt_all_scales(args, motion_ptn, motion_dfm, geometry_nor, character_dfm, geometry_dfm, offsets, joints):
    """Adapt one pair of partner and deformed motions for every scale. Results are lists in scale order."""
    ptn_root_joints, ptn_spine_joints, ptn_limb_joints, \
        dfm_root_joints, dfm_spine_joints, dfm_limb_joints = joints

    root_p_ptn, local_R_ptn, root_p_dfm, local_R_dfm = [], [], [], []
    for scale in args.scales:
        edited_ptn = deepcopy(motion_ptn)
        edited_dfm = make_target_motion(args, motion_dfm, character_dfm, geometry_dfm, offsets, scale)

        root_p0, local_R0, root_p1, local_R1 = retarget_smpl(
            args,
            geometry_nor, geometry_nor, geometry_nor, geometry_dfm,
            motion_ptn, motion_dfm, edited_ptn, edited_dfm,
            ptn_root_joints=ptn_root_joints, ptn_spine_joints=ptn_spine_joints, ptn_limb_joints=ptn_limb_joints,
            dfm_root_joints=dfm_root_joints, dfm_spine_joints=dfm_spine_joints, dfm_limb_joints=dfm_limb_joints)
        root_p_ptn.append(root_p0.numpy())
        local_R_ptn.append(local_R0.numpy())
        root_p_dfm.append(root_p1.numpy())
        local_R_dfm.append(local_R1.numpy())

    return root_p_ptn, local_R_ptn, root_p_dfm, local_R_dfm


def save_adaptations(args):
    """Adapt all of RD_bvh for both roles and every scale, and save to adapted_motion/{character}/."""
    _, _, geometry_nor = get_a_smpl_character(args, "SMPLx")
    print("deformed_names:", args.deformed_names)

    time_start = time_prev = time.time()
    for deformed_name in args.deformed_names:
        character_dfm, Tpose_dfm, geometry_dfm = get_a_smpl_character(args, deformed_name)
        offsets = [joint.offset for joint in Tpose_dfm.skeleton.joints]

        for motion_name0, motion_name1 in RD_bvh.items():
            motion0 = get_interaction_motions_from_list("SMPLx", [motion_name0])[0]
            motion1 = get_interaction_motions_from_list("SMPLx", [motion_name1])[0]

            joints = update_target_joints(args, motion_name0, motion_name1, deformed_name)
            set_interaction_range(args, motion_name0)

            # role 0: motion1 is deformed, role 1: motion0 is deformed
            root_p_ptn, local_R_ptn, root_p_dfm, local_R_dfm = adapt_all_scales(
                args, motion0, motion1, geometry_nor, character_dfm, geometry_dfm, offsets, joints)
            root_p0s, local_R0s = [root_p_ptn], [local_R_ptn]
            root_p1s, local_R1s = [root_p_dfm], [local_R_dfm]

            root_p_ptn, local_R_ptn, root_p_dfm, local_R_dfm = adapt_all_scales(
                args, motion1, motion0, geometry_nor, character_dfm, geometry_dfm, offsets, joints)
            root_p0s.append(root_p_dfm);  local_R0s.append(local_R_dfm)
            root_p1s.append(root_p_ptn);  local_R1s.append(local_R_ptn)

            saved = save_edited_npy_motion(deformed_name, motion_name0, root_p0s, local_R0s, root_p1s, local_R1s)

            time_now = time.time()
            print("[{}] {} : {} ({}f) {} in {}".format(
                datetime.datetime.now().strftime("%H:%M:%S"),
                deformed_name, motion_name0, len(motion0.poses),
                "saved" if saved else "not saved",
                datetime.timedelta(seconds=time_now - time_prev)))
            time_prev = time_now

    print("[{}] Done: {}".format(
        datetime.datetime.now().strftime("%H:%M:%S"),
        datetime.timedelta(seconds=time.time() - time_start)))


def main(args):
    app_manager = AppManager()
    set_adapt_args(args)

    if args.save:
        save_adaptations(args)
        return

    view_adaptation(args, app_manager)


if __name__ == '__main__':
    args = option_parser.get_args()
    main(args)
