"""Save inference results as npz."""
import os
import numpy as np

from test_utils.inference import run_network


def save_motion(motion, path):
    root_p  = np.stack([pose.root_p  for pose in motion.poses])  # (T, 3)
    local_R = np.stack([pose.local_R for pose in motion.poses])  # (T, J, 3, 3)
    np.savez(path, root_p=root_p, local_R=local_R)
    print(f"Saved: {path}  root_p={root_p.shape}  local_R={local_R.shape}")


def save_all_motions(args, setup, motion_pairs):
    """Save every motion pair to ./saved_result/{test_proj}/{motion_name}/net_{name}_s{idx}.npz."""
    save_root = os.path.join('./saved_result', args.test_proj)
    os.makedirs(save_root, exist_ok=True)

    for motion_name0, motion_name1 in motion_pairs:
        print(f"\n{'='*60}")
        print(f"motion: {motion_name0} / {motion_name1}")
        print(f"{'='*60}")

        out_motion0, out_motion1, _, _ = run_network(args, setup, motion_name0, motion_name1)

        motion_name = motion_name0.replace("_S1", "")
        motion_dir  = os.path.join(save_root, motion_name)
        os.makedirs(motion_dir, exist_ok=True)
        name0 = os.path.splitext(os.path.basename(motion_name0))[0]
        name1 = os.path.splitext(os.path.basename(motion_name1))[0]

        save_motion(out_motion0, os.path.join(motion_dir, f"net_{name0}_s0.npz"))
        save_motion(out_motion1, os.path.join(motion_dir, f"net_{name1}_s1.npz"))

    print(f"\n> Saved {len(motion_pairs)} motions: {save_root}")
