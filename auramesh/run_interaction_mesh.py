"""
Interaction Mesh based motion retargeting.
Ho et al., "Spatial Relationship Preserving Character Motion Adaptation", ACM TOG 2010.

Run from geometry_aware_retargeting directory:
    python auramesh/run_interaction_mesh.py
"""

import sys
sys.path.append('./')
sys.path.append('../')

import copy
import datetime
import numpy as np
import torch

from pymovis.vis.appmanager import AppManager
from pymovis.vis.app import MotionApp
from pymovis.motion.ops.npmotion import *

from datasets.character_functions import get_a_smpl_character as get_a_character
from datasets.motion_functions import get_interaction_motions_from_list
from Retarget_SMPL.relationship_descriptor import (
    get_rootP_localR_globalP_from_motion,
    update_motion_by_global_p,
)
from Retarget_SMPL.retarget_smpl import scale_character

import option_parser
from option_motion import example_bvh
from auramesh.interaction_mesh import InteractionMesh
from xalglib import xalglib


# ─── Energy terms ────────────────────────────────────────────────────────────

def _get_bone_lengths(skeleton):
    """Target bone lengths from skeleton joint offsets."""
    return np.array([np.linalg.norm(j.offset) for j in skeleton.joints])


def _bone_energy_grad(pos1, parent_idx, target_lengths, lam=10.0):
    """
    Soft bone-length constraint.
    E = lam * Σ_e ( ||p_j - p_parent||² - l_e² )²
    """
    grad = np.zeros_like(pos1)
    energy = 0.0
    for j in range(1, len(pos1)):
        p = parent_idx[j]
        if p < 0:
            continue
        diff = pos1[j] - pos1[p]
        dist_sq = float(np.dot(diff, diff))
        res = dist_sq - target_lengths[j] ** 2
        energy += lam * res ** 2
        g = lam * 4.0 * res * diff
        grad[j] += g
        grad[p] -= g
    return energy, grad


def _source_follow_energy_grad(pos1, src_pos1, lam=1.0):
    """
    Source-pose following term (keeps optimization near source).
    E = lam * ||pos1 - src_pos1||²
    """
    diff = pos1 - src_pos1
    energy = lam * float(np.sum(diff ** 2))
    grad = lam * 2.0 * diff
    return energy, grad


# ─── Per-frame optimization ───────────────────────────────────────────────────

def _optimize_frame(f, pos0_t, src_pos1_t, delta_t, im, parent_idx, target_lengths,
                    lam_bone=10.0, lam_follow=0.5):
    """Optimize char1 global positions for one frame using L-BFGS."""

    def func_grad(x, grad, param=None):
        x_np = np.array(x, dtype=np.float64)
        pos1 = x_np.reshape(im.n1, 3)

        e_lap, g_lap = im.compute_energy_and_grad(x_np, pos0_t, delta_t)
        e_bone, g_bone = _bone_energy_grad(pos1, parent_idx, target_lengths, lam_bone)
        e_fol, g_fol = _source_follow_energy_grad(pos1, src_pos1_t, lam_follow)

        total_e = e_lap + e_bone + e_fol
        total_g = g_lap + g_bone.flatten() + g_fol.flatten()

        for i in range(len(grad)):
            grad[i] = float(total_g[i])
        return float(total_e)

    x0 = src_pos1_t.flatten().tolist()
    state = xalglib.minlbfgscreate(5, x0)
    xalglib.minlbfgssetcond(state, 0.0, 0.0, 1e-6, 100)
    xalglib.minlbfgsoptimize_g(state, func_grad)
    x_opt, _ = xalglib.minlbfgsresults(state)

    return np.array(x_opt, dtype=np.float64).reshape(im.n1, 3)


# ─── Main retargeting function ────────────────────────────────────────────────

def retarget_with_interaction_mesh(args, src_motion_0, src_motion_1,
                                   tgt_motion_1, im, lam_bone=10.0, lam_follow=0.5):
    """
    Optimize tgt_motion_1 to preserve interaction mesh Laplacian from source.
    src_motion_0 acts as the fixed pattern character (char0).
    char1 positions are optimized per frame.

    Args:
        src_motion_0: source pattern motion (char0, fixed)
        src_motion_1: source deformed motion (char1, reference)
        tgt_motion_1: target motion to optimize (initialized, e.g. T-pose)
        im:           InteractionMesh built from source motions
        lam_bone:     weight for bone-length soft constraint
        lam_follow:   weight for source-pose following term
    Returns:
        Optimized tgt_motion_1
    """
    skeleton_1 = tgt_motion_1.skeleton
    parent_idx = skeleton_1.parent_idx
    target_lengths = _get_bone_lengths(skeleton_1)

    # Source char0 positions (fixed anchors): (T, n0, 3)
    _, _, src_gp0 = get_rootP_localR_globalP_from_motion(args, src_motion_0.poses)
    src_gp0 = src_gp0.numpy().astype(np.float64)

    # Source char1 positions (for initialization / follow term): (T, n1, 3)
    _, _, src_gp1 = get_rootP_localR_globalP_from_motion(args, src_motion_1.poses)
    src_gp1 = src_gp1.numpy().astype(np.float64)

    T = len(src_motion_0.poses)
    optimized_p1 = np.zeros((T, im.n1, 3), dtype=np.float64)

    print(f"Optimizing {T} frames...")
    t0 = datetime.datetime.now()

    for f in range(T):
        optimized_p1[f] = _optimize_frame(
            f,
            src_gp0[f],
            src_gp1[f],
            im.src_laplacian[f],
            im,
            parent_idx,
            target_lengths,
            lam_bone=lam_bone,
            lam_follow=lam_follow,
        )
        if (f + 1) % 50 == 0 or f == T - 1:
            elapsed = (datetime.datetime.now() - t0).total_seconds()
            print(f"  [{datetime.datetime.now().strftime('%H:%M:%S')}] "
                  f"Frame {f+1}/{T}  ({elapsed:.1f}s elapsed)")

    # Convert optimized global positions back to local rotations
    opt_tensor = torch.tensor(optimized_p1, dtype=torch.float32)
    update_motion_by_global_p(tgt_motion_1, opt_tensor)

    return tgt_motion_1


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app_manager = AppManager()
    args = option_parser.get_args()
    args.path = args.proj_name + '/'
    args.device = 'cpu'
    scale = 0.7  # target char1 body scale

    # ── Source ──
    src_names = ["SMPLx", "SMPLx"]
    src_chars = []
    for name in src_names:
        char, _, _ = get_a_character(args, name)
        src_chars.append(char)

    motion_name0 = list(example_bvh.keys())[0]
    motion_name1 = list(example_bvh.values())[0]
    motion_0 = get_interaction_motions_from_list(src_names[0], [motion_name0])[0]
    motion_1 = get_interaction_motions_from_list(src_names[0], [motion_name1])[0]

    src_chars[0].set_source_skeleton(motion_0.skeleton, "")
    src_chars[1].set_source_skeleton(motion_1.skeleton, "")

    print(f"Source motions: {motion_name0} ({len(motion_0.poses)}f), "
          f"{motion_name1} ({len(motion_1.poses)}f)")

    # ── Build Interaction Mesh ──
    print("\nBuilding Interaction Mesh from source motions...")
    _, _, src_gp0 = get_rootP_localR_globalP_from_motion(args, motion_0.poses)
    _, _, src_gp1 = get_rootP_localR_globalP_from_motion(args, motion_1.poses)
    im = InteractionMesh(src_gp0.numpy(), src_gp1.numpy())

    # ── Target ──
    tgt_names = ["SMPLx", "SMPLx"]
    tgt_chars = []
    for i, name in enumerate(tgt_names):
        if i == 0:
            char, _, _ = get_a_character(args, name)
        else:
            char, _, _ = get_a_character(args, name, mesh_scale=scale)
        tgt_chars.append(char)

    tgt_motion_0 = copy.deepcopy(motion_0)
    tgt_motion_1 = copy.deepcopy(motion_1)

    # Initialize char1 target motion: T-pose with scaled root height
    for pose in tgt_motion_1.poses:
        pose.local_R = np.identity(3)[None, :].repeat(22, axis=0).astype(np.float32)
        pose.root_p[1] *= scale
        pose.update()

    tgt_chars[0].set_source_skeleton(tgt_motion_0.skeleton, "")
    tgt_chars[1].set_source_skeleton(tgt_motion_1.skeleton, "")

    # Scale char1 skeleton offsets
    scale_character(args, tgt_chars[1], scale, scale, scale)
    for pose in tgt_motion_1.poses:
        pose.update()

    # ── Optimize ──
    print("\nRetargeting with Interaction Mesh...")
    t_start = datetime.datetime.now()
    tgt_motion_1 = retarget_with_interaction_mesh(
        args,
        motion_0, motion_1,
        tgt_motion_1, im,
        lam_bone=10.0,
        lam_follow=0.5,
    )
    t_total = datetime.datetime.now() - t_start
    print(f"Total: {t_total}")

    # ── Render ──
    for f in range(len(motion_0.poses)):
        motion_0.poses[f].translate_root_p([args.source_pos, 0, 0])
        motion_1.poses[f].translate_root_p([args.source_pos, 0, 0])
        tgt_motion_0.poses[f].translate_root_p([args.joint_pos, 0, 0])
        tgt_motion_1.poses[f].translate_root_p([args.joint_pos, 0, 0])

    chars = src_chars + tgt_chars
    motions = [motion_0, motion_1, tgt_motion_0, tgt_motion_1]

    app = MotionApp(chars, motions, args)
    app_manager.run(app)
