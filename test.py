'''
python test.py
'''

import sys
sys.path.append('../')

import os
import numpy as np

from pymovis.vis.appmanager import AppManager
from pymovis.vis.app import MyApp
from Network.network import Network
from datasets.character_functions import load_char
from datasets.motion_functions import get_interaction_motions_from_list, make_new_motions, set_rot_dim
from datasets.motion_dataset import Dataset
import option_parser
from option_motion import example_bvh
from Retarget_SMPL.relationship_descriptor import resolve_ground_pene
from etc.etc import render_result


def main(args):
    app_manager = AppManager()
    set_rot_dim(args)
    args.device         = "cpu"
    args.is_train       = False
    args.save_norm_info = False
    args.test_type      = "SMPLx"
    args.test_char      = "small"
    # args.SMPLx_mesh_scale = 0.7  # char1 mesh scale (preprocess/SMPLx/scale_mesh0.7_* 사용)

    motion_name0 = list(example_bvh.keys())[0]
    motion_name1 = list(example_bvh.values())[0]

    print(f"> proj:      {args.test_proj}")
    print(f"> character: {args.test_type} {args.test_char}")
    print(f"> motion:    {motion_name0}")

    # ── Characters ──
    src_char0, src_char1, tgt_char0, tgt_char1, \
        _, _, src_name0, src_name1 = load_char(args)

    # ── Motions ──
    src_motion0 = get_interaction_motions_from_list(src_name0, [motion_name0])[0]
    src_motion1 = get_interaction_motions_from_list(src_name1, [motion_name1])[0]

    # ── Dataset ──
    dataset = Dataset(args)
    dataset.get_char_data(src_char0, src_char1, tgt_char0, tgt_char1)
    dataset.get_input_motion(src_motion0, src_motion1)
    if args.data_normalized:
        dataset.load_norm_info()
        dataset.normalize()

    # ── Network forward ──
    net = Network(args)
    net.load(args.test_proj + '/', args.test_epoch, device=args.device)
    net.eval()

    out_p0, out_R0, out_p1, out_R1 = net.forward(dataset)

    out_motion0, out_motion1 = make_new_motions(
        args, out_p0, out_R0, out_p1, out_R1,
        tgt_char0, tgt_char1, src_motion0, src_motion1,
    )
    # out_motion0, out_motion1 = resolve_ground_pene(args, out_motion0, out_motion1)

    # ── Save (interaction_mesh와 동일한 형식) ──
    # ./auramesh/saved_result/net_{motion_name}_s{idx}.npz
    
    SAVE =True
    if SAVE:
        motion_name = motion_name0.replace("_S1", "")
        save_dir = f"./auramesh/saved_result/{motion_name}"
        os.makedirs(save_dir, exist_ok=True)
        name0 = os.path.splitext(os.path.basename(motion_name0))[0]
        name1 = os.path.splitext(os.path.basename(motion_name1))[0]

        for motion, name, idx in [(out_motion0, name0, 0), (out_motion1, name1, 1)]:
            root_p  = np.stack([pose.root_p  for pose in motion.poses])  # (T, 3)
            local_R = np.stack([pose.local_R for pose in motion.poses])  # (T, J, 3, 3)
            path = os.path.join(save_dir, f"net_{name}_s{idx}.npz")
            np.savez(path, root_p=root_p, local_R=local_R)
            print(f"Saved: {path}  root_p={root_p.shape}  local_R={local_R.shape}")

    # ── Render ──
    characters, motions = render_result(
        args,
        src_char0, src_char1, tgt_char0, tgt_char1,
        src_motion0, src_motion1, out_motion0, out_motion1,
    )
    app = MyApp(characters, motions, args, net)
    app_manager.run(app)


if __name__ == "__main__":
    args = option_parser.get_args()
    main(args)
