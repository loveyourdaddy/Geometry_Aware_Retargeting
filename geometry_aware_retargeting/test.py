'''
conda activate py_env37_2
python test.py --checkpoint saved/best.pt --resolve_ground_pene True

Parameters
--checkpoint: path to a checkpoint file
--test_proj, --test_epoch: used when --checkpoint is not given (saved/{test_proj}/spatio_temp_net_{test_epoch}.pt)
--save: save the motions without rendering. True/False(default)
--network_type: mlp, no_cross_attn, cross_attn(default)
--use_anchor_input: True/False(default)
--use_relative_pos_input: True/False(default)
--resolve_ground_pene: post-process so that the feet do not penetrate the ground. True/False(default)

Network ablation 
    python test.py --network_type mlp           --test_proj ablation_mlp      --test_epoch 500 --save True
    python test.py --network_type no_cross_attn --test_proj ablation_no_cross --test_epoch 500 --save True

Input ablation
    python test.py --use_anchor_input True --use_relative_pos_input False --test_proj 260703_anchor_only --test_epoch 4000 --save True
    python test.py --use_anchor_input False --use_relative_pos_input True --test_proj 260703_relpos_only --test_epoch 4000 --save True

Viewer keys
    ←/→: previous/next motion,  ↑/↓: ±10 frames while paused,  Space: play/pause
'''

import os
import sys
sys.path.append('../')

from pymovis.vis.appmanager import AppManager
from datasets.motion_functions import set_rot_dim
import option_parser
from option_motion import example_bvh
from test_utils.loader import load_test_setup
from test_utils.saver import save_all_motions
from test_utils.viewer import PlaylistApp


def set_test_args(args):
    set_rot_dim(args)
    args.device         = "cpu"
    args.is_train       = False
    args.save_norm_info = False
    args.save           = False # True: save all motions without rendering and exit
    args.test_type      = "Mixamo" # SMPLx
    args.test_char      = "fat"    # small fat
    # args.SMPLx_mesh_scale = 0.7  # char1 mesh scale (uses preprocess/SMPLx/scale_mesh0.7_*)

    if args.checkpoint:
        args.test_proj = os.path.splitext(os.path.basename(args.checkpoint))[0] # folder name for saved results

    print(f"> checkpoint: {args.checkpoint or f'saved/{args.test_proj}/spatio_temp_net_{args.test_epoch}.pt'}")
    print(f"> character:  {args.test_type} {args.test_char}")


def main(args):
    app_manager = AppManager()
    set_test_args(args)

    setup = load_test_setup(args)
    motion_pairs = list(example_bvh.items())

    # save all motions without rendering and exit
    if args.save:
        save_all_motions(args, setup, motion_pairs)
        return

    app = PlaylistApp(args, setup, motion_pairs)
    app_manager.run(app)


if __name__ == "__main__":
    args = option_parser.get_args()
    main(args)
