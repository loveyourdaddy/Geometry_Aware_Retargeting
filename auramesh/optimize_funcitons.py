import sys, os
sys.path.append('.')
sys.path.append('../') # Retargeting_workspace

from pymovis.motion.ops import torchmotion
from xalglib import xalglib
from pymovis.motion.ops.npmotion import *
from Geometry.geometry import Geometry
from pymovis.motion.core import Pose, Motion
import copy
import time
import numpy as np
import torch

body_joints = [0, 9, 10, 11, 12, 13]
left_leg_joints = [1, 2, 3, 4]
right_leg_joints = [5, 6, 7, 8]
left_hand_joints = [14, 15, 16, 17]
right_hand_joints = [18, 19, 20, 21]
joint_chains = [body_joints, left_leg_joints, right_leg_joints, left_hand_joints, right_hand_joints]


def optimize_motion(args,
                    src_motion, tgt_motion,
                    tgt_geo, ptn_auramesh,
                    all_col_vids, ptn_all_col_vids,
                    col_frame, tgt_jids, ptn_jids):
    lamda_1 = 1.0
    lamda_2 = 10.0
    collision_preserving = True
    len_rot_x = 198
    num_joint = 22

    # ── Pre-compute lookup tables (avoid repeated work inside the hot callback) ──

    # Per-frame collision indices
    col_ids_per_frame = {}
    for f_key in np.unique(col_frame):
        col_ids_per_frame[int(f_key)] = np.where(col_frame == f_key)[0]

    # Pattern auramesh vertex positions are FIXED during optimization.
    # Pre-compute them once for all frames so the callback never calls ptn_auramesh again.
    print("Pre-computing pattern vertex positions...")
    _t_pre = time.time()
    ptn_vpos_cache = {}   # {frame: {cid: (V, 3) ndarray}}
    for f_key, cids in col_ids_per_frame.items():
        ptn_vpos_cache[f_key] = {}
        for cid in cids:
            vids  = ptn_all_col_vids[cid][None, None, :]
            n     = vids.shape[-1]
            batch = torch.zeros(1, 1, n, dtype=torch.long)
            frame = torch.full((1, 1, n), f_key, dtype=torch.long)
            ptn_vpos_cache[f_key][cid] = \
                ptn_auramesh.get_positions_from_vids(vids, batch, frame)[0, 0].numpy()
    print(f"Pre-computation done. ({time.time() - _t_pre:.2f}s)")

    # Pre-convert target root_p to tensors (shape 1×1×3) once per frame.
    T = len(src_motion.poses)
    tgt_root_p_t = [
        torch.tensor(tgt_motion.poses[f].root_p[None, None, :], dtype=torch.float32)
        for f in range(T)
    ]

    # Fixed tensor for tgt_geo queries (always frame index 0 after set_pose)
    def _make_vids_batch_frame(vids_raw):
        n    = len(vids_raw)
        vids = vids_raw[None, None, :]
        bat  = torch.zeros(1, 1, n, dtype=torch.long)
        frm  = torch.zeros(1, 1, n, dtype=torch.long)
        return vids, bat, frm

    # ── Accumulated timing counters (reset per frame) ──
    _tm = {
        'goal0': 0.0, 'goal1_svd': 0.0,
        'lbs': 0.0, 'get_vpos': 0.0, 'jacobian': 0.0,
        'n_calls': 0,
    }

    def _reset_tm():
        for k in _tm:
            _tm[k] = 0

    # compute loss func + grad
    def function1_grad(x, grad, param=None):
        nonlocal f, pose, target
        func = 0.0
        _tm['n_calls'] += 1

        # goal0: follow source rotation (L1, vectorized)
        _t0 = time.perf_counter()
        x_np = np.array(x, dtype=np.float32)
        diff = x_np - target
        func += float(np.sum(np.abs(diff)))
        grad_np = diff.astype(np.float64)
        _tm['goal0'] += time.perf_counter() - _t0

        # goal1: rotation matrix orthogonality (SVD projection)
        _t0 = time.perf_counter()
        target_local_R = x_np.reshape(22, 3, 3)
        normalized_R   = normalize_rotation_matrix(target_local_R)
        R_diff = target_local_R - normalized_R
        func      += lamda_1 * float(np.sum(np.linalg.norm(R_diff, axis=(1, 2))))
        grad_np   += (lamda_1 * 2.0 * R_diff).flatten()
        _tm['goal1_svd'] += time.perf_counter() - _t0

        # goal2: preserve skin-level contact points
        if not collision_preserving:
            grad[:] = grad_np.tolist()
            return func

        col_ids = col_ids_per_frame.get(f, np.array([], dtype=int))
        if len(col_ids) == 0:
            grad[:] = grad_np.tolist()
            return func

        # LBS once per gradient call (was: once per cid → N× redundant)
        _t0 = time.perf_counter()
        tgt_geo.set_pose_by_source_batch_frame(
            torch.tensor(target_local_R, dtype=torch.float32)[None, None, :],
            tgt_root_p_t[f])
        _tm['lbs'] += time.perf_counter() - _t0

        for cid in col_ids:
            _t0 = time.perf_counter()
            vids, bat, frm = _make_vids_batch_frame(all_col_vids[cid])
            tgt_vpos = tgt_geo.get_positions_from_vids(vids, bat, frm)[0, 0].numpy()
            _tm['get_vpos'] += time.perf_counter() - _t0

            # pattern positions from cache (no auramesh call)
            ptn_vpos = ptn_vpos_cache[f][cid]

            func += lamda_2 * float(np.mean(np.abs(ptn_vpos - tgt_vpos)))

            # Gradient via kinematic chain Jacobian
            _t0 = time.perf_counter()
            ee_joint = tgt_jids[cid]
            eeT = np.array(pose.global_p[ee_joint])

            joint_chain = None
            for chain in joint_chains:
                if ee_joint in chain:
                    joint_chain = chain[:chain.index(ee_joint) + 1]
                    break

            dLdP = (lamda_2 * 2.0 * np.mean(ptn_vpos - tgt_vpos, axis=0))[None, :]  # (1, 3)

            dPdX = np.zeros((3, len_rot_x))
            for j in reversed(joint_chain):
                x_start  = j * 9
                curT     = np.array(pose.global_p[j])
                dist     = eeT - curT
                quat     = R_to_Q(target_local_R[j])
                axis, angle = Q_to_A(quat)
                aaxis    = axis * angle
                quat_inv = np.array([quat[0], -quat[1], -quat[2], -quat[3]])
                world_R  = np.array(pose.global_R[j])
                for eid in range(3):
                    upd = aaxis.copy()
                    upd[eid] += 1.0
                    norm_upd = np.linalg.norm(upd)
                    upd_quat = A_to_Q(norm_upd, upd / norm_upd)
                    d_quat   = quaternion_multiply(quat_inv, upd_quat)
                    d_axis, d_angle = Q_to_A(d_quat)
                    dT = d_angle * np.cross(world_R @ d_axis, dist)
                    dPdX[eid, x_start:x_start + 9] = dT[eid]

            grad_np += (dLdP @ dPdX)[0]
            _tm['jacobian'] += time.perf_counter() - _t0

        grad[:] = grad_np.tolist()
        return func

    # ── Per-frame L-BFGS ──
    # Start from identity; after each frame, warm-start the next from the previous result.
    x0 = num_joint * [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

    _t_opt_total = 0.0
    _PRINT_EVERY = 10  # print timing summary every N frames

    for f, pose in enumerate(src_motion.poses):
        target = pose.local_R.reshape(-1).astype(np.float32)
        _reset_tm()

        _t_frame = time.perf_counter()
        state = xalglib.minlbfgscreate(2, x0)
        xalglib.minlbfgssetcond(state, 0.0, 0.0, 1e-6, 100)
        xalglib.minlbfgsoptimize_g(state, function1_grad)
        x_opt, _ = xalglib.minlbfgsresults(state)
        x0 = list(x_opt)  # warm-start next frame from current result
        _dt_frame = time.perf_counter() - _t_frame
        _t_opt_total += _dt_frame

        updated_R = normalize_rotation_matrix(
            np.array(x_opt, dtype=np.float32).reshape(22, 3, 3))
        tgt_motion.poses[f].local_R = updated_R
        tgt_motion.poses[f].update()

        if (f + 1) % _PRINT_EVERY == 0 or f == T - 1:
            nc = max(_tm['n_calls'], 1)
            print(
                f"[frame {f+1:4d}/{T}] "
                f"frame={_dt_frame*1e3:.1f}ms  calls={_tm['n_calls']}  "
                f"| goal0={_tm['goal0']*1e3/nc:.2f}ms  "
                f"svd={_tm['goal1_svd']*1e3/nc:.2f}ms  "
                f"lbs={_tm['lbs']*1e3/nc:.2f}ms  "
                f"get_vpos={_tm['get_vpos']*1e3/nc:.2f}ms  "
                f"jac={_tm['jacobian']*1e3/nc:.2f}ms  (per call)"
            )

    print(f"\n=== optimize_motion total: {_t_opt_total:.1f}s  ({_t_opt_total/T*1e3:.1f}ms/frame) ===")
    return tgt_motion


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])

def get_character_geometry(args, names, fbx_models):
    geometry = []
    for i, name in enumerate(names):
        geometry.append(Geometry(args, character=fbx_models[i], name=name))
    return geometry

def quaternion_slerp(q1, q2, t):
    if not np.isscalar(t):
        raise ValueError("t should be a scalar in the range [0, 1]")

    dot = np.sum(q1 * q2, axis=-1)
    dot = np.clip(dot, -1.0, 1.0)

    flip = dot < 0
    q1_flipped = q1 * np.where(flip[:, None], -1, 1)

    near_one = dot > 0.9999
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    theta = theta_0 * t
    sin_theta = np.sin(theta)

    s1 = np.cos(theta) - dot * sin_theta / np.where(sin_theta_0 == 0, 1, sin_theta_0)
    s2 = sin_theta / np.where(sin_theta_0 == 0, 1, sin_theta_0)

    result = s1[..., None] * q1_flipped + s2[..., None] * q2
    result[near_one] = (1 - t) * q1[near_one] + t * q2[near_one]

    norms = np.linalg.norm(result, axis=-1, keepdims=True)
    result /= norms

    return result

def normalize_rotation_matrix(matrix):
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 3, 3)

    normalized = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        U, _, Vt = np.linalg.svd(matrix[i])
        normalized[i] = U @ Vt
        if np.linalg.det(normalized[i]) < 0:
            normalized[i][:, 2] *= -1

    return normalized
