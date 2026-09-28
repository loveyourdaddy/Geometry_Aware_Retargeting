'''
                          
  cd geometry_aware_retargeting
  python ../datasets/save_motion_data.py

'''
import os, sys
_workspace_root = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(_workspace_root)
sys.path.append(os.path.join(_workspace_root, 'geometry_aware_retargeting'))

from datasets.character_functions import *
from datasets.motion_functions import *
from datasets.motion_dataset import *
import option_parser
from pymovis.vis.appmanager import AppManager

def main(args): 
    app_manager = AppManager()
    args.is_train = True
    set_rot_dim(args)

    train_mesh_char_list = args.target_characters
    print("train mesh character list: ", train_mesh_char_list)
    
    # character
    source0_character, source1_character, target0_characters, target1_characters, _, _ = \
        get_smpl_characters_wo_geo(args, train_mesh_char_list) # SMPLx_small = False

    # motion (same input motion)
    train_motions0, train_motions1 = get_train_bvh("SMPLx", "SMPLx")

    # fixed reference character always used to compute the anchor input feature (train_mesh_char_list[0]).
    # same convention as the anchor loss using anchor_vids_src = anchor_vids[0].
    from datasets.character_functions import get_a_smpl_character
    _, _, ref_geo_obj = get_a_smpl_character(args, train_mesh_char_list[0])
    ref_geo = {
        "anchor_vids": ref_geo_obj.anchor_vids,
        "anchor_vpos": ref_geo_obj.anchor_vpos,
        "name_to_idx": ref_geo_obj.name_to_idx,
        "bind_trf_inv": ref_geo_obj.bind_trf_inv,
        "skinning_indices1": ref_geo_obj.skinning_indices1[ref_geo_obj.anchor_vids],
        "skinning_weights1": ref_geo_obj.skinning_weights1[ref_geo_obj.anchor_vids],
        "skinning_indices2": ref_geo_obj.skinning_indices2[ref_geo_obj.anchor_vids],
        "skinning_weights2": ref_geo_obj.skinning_weights2[ref_geo_obj.anchor_vids],
        "perjoint_vids": ref_geo_obj.perjoint_vids,  # actual vertex ids per joint (used to group the anchors by joint)
        "parents": ref_geo_obj.parents,
        "names": ref_geo_obj.names,
    }

    # dataset (the target character changes)
    for i, (target0_character, target1_character) in enumerate(zip(target0_characters, target1_characters)):
        dataset = Dataset(args)

        # load character info
        dataset.get_char_data(source0_character, source1_character, target0_character, target1_character)
        dataset.load_geo_data(train_mesh_char_list[i])

        # load motion
        dataset.get_input_motion(train_motions0, train_motions1)
        dataset.get_input_anchor_pos(ref_geo)
        dataset.get_input_relative_pos()
        dataset.get_gt_motion()

        # save
        dataset.save_char_and_motion_data(train_mesh_char_list[i])

if __name__ == "__main__":
    args = option_parser.get_args()
    main(args)
