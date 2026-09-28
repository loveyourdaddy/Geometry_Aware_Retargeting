'''
Render the adaptation results saved in adapted_motion/
    python Retarget_SMPL/check_retarget_result.py

Parameters
--role_change: show the result with the roles of the two characters swapped. True/False(default)
'''

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('../')

from pymovis.vis.appmanager import AppManager
from pymovis.vis.app import MyApp
from etc.etc import render_result
from datasets.character_functions import get_a_smpl_character, get_a_smpl_character_wo_geo, get_scale
from datasets.motion_functions import get_interaction_motions_from_list
import option_parser
from option_motion import example_bvh
from Retarget_SMPL.edited_motion import load_edited_npy_motion


def set_check_args(args):
    args.test_type     = "SMPLx"
    args.deformed_name = "SMPLx"  # SMPLx_fat
    args.scale_index   = 0        # index into scale_values/scales_sampled.npy (0~9)


def main(args):
    app_manager = AppManager()
    set_check_args(args)

    character_normal, _, _ = get_a_smpl_character(args, "SMPLx")
    character_dfm, _ = get_a_smpl_character_wo_geo(args, args.deformed_name, scale=0.7)

    motion_name0, motion_name1 = list(example_bvh.items())[0]
    motion0 = get_interaction_motions_from_list("SMPLx", [motion_name0])[0]
    motion1 = get_interaction_motions_from_list("SMPLx", [motion_name1])[0]

    leg_scale, body_scale, hand_scale = get_scale()[args.scale_index]
    rid = 1 if args.role_change else 0
    motionA, motionB = \
        load_edited_npy_motion(args, motion0, motion1, args.deformed_name, motion_name0,
                               leg_scale, body_scale, hand_scale, rid, args.scale_index)

    characters, motions = \
        render_result(args,
                      character_normal, character_normal, character_normal, character_dfm,
                      motion0, motion1, motionA, motionB)
    app = MyApp(characters, motions, args)
    app_manager.run(app)


if __name__ == '__main__':
    args = option_parser.get_args()
    main(args)
