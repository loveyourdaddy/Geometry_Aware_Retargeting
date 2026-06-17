import sys, os
sys.path.append('.')
sys.path.append('../') # Retargeting_workspace

from xalglib import xalglib
from pymovis.motion.ops.npmotion import *
from Geometry.geometry import Geometry
import time
import numpy as np
import torch
# from pymovis.motion.ops import torchmotion
# from pymovis.motion.core import Pose, Motion
# import copy

def _smooth_gaussian(delta_R, sigma=3):
    """
    Gaussian low-pass filter를 delta(변화량)에 적용.
    delta_R: (T, J, 3, 3)  =  tgt_R - src_R
    반환:    (T, J, 3, 3)  smoothed delta
    """
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(delta_R.astype(np.float64), sigma=sigma, axis=0)


def _smooth_oneeuro(delta_R, fps=30, min_cutoff=5.0, beta=0.5, d_cutoff=1.0):
    """
    One-Euro Filter를 delta(변화량)에 적용.
    delta_R:    (T, J, 3, 3)  =  tgt_R - src_R
    min_cutoff: 낮을수록 slow 구간 smoothing 강화 (기본 5.0 → 변화 보존 우선)
    beta:       높을수록 fast 구간 smoothing 감소 (기본 0.5)
    반환:       (T, J, 3, 3)  smoothed delta
    """
    T, J, _, _ = delta_R.shape
    flat = delta_R.reshape(T, -1).astype(np.float64)  # (T, J*9)
    dt = 1.0 / fps

    def _alpha(cutoff):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    alpha_d = _alpha(d_cutoff)
    out = np.zeros_like(flat)
    out[0] = flat[0]
    x_prev  = flat[0].copy()
    dx_prev = np.zeros(flat.shape[1])

    for i in range(1, T):
        dx      = (flat[i] - x_prev) / dt
        dx_hat  = dx_prev + alpha_d * (dx - dx_prev)
        cutoff  = min_cutoff + beta * np.abs(dx_hat)
        alpha   = _alpha(cutoff)
        x_hat   = x_prev + alpha * (flat[i] - x_prev)
        out[i]  = x_hat
        x_prev  = x_hat
        dx_prev = dx_hat

    return out.reshape(T, J, 3, 3)


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
                    col_frame, tgt_jids, ptn_jids,
                    ): 
    lamda_0 = 0.1   # goal0: source rotation regularization (줄일수록 goal2 우선)
    lamda_1 = 1.0   # goal1: SO(3) orthogonality
    lamda_2 = 100.0 # goal2: contact preservation (키울수록 접촉 보존 강화)
    # collision_preserving = True
    len_rot_x = 198
    num_joint = 22

    # ── Pre-compute lookup tables (avoid repeated work inside the hot callback) ──

    # Per-frame collision indices (71:0, 72:1, ..., 76:5,6,7)
    col_ids_per_frame = {}
    for f_key in np.unique(col_frame):
        col_ids_per_frame[int(f_key)] = np.where(col_frame == f_key)[0]

    # Pattern auramesh vertex positions are FIXED during optimization.
    # Pre-compute them once for all frames so the callback never calls ptn_auramesh again.
    print("Pre-computing pattern vertex positions...")
    _t_pre = time.time()
    ptn_vpos_cache = {}   # {frame: {cid: (V, 3) ndarray}} (71: {0: (V, 3)}, 72: {1: ...}, ...) # ptn auramesh의 위치 (target)
    for f_key, cids in col_ids_per_frame.items():
        ptn_vpos_cache[f_key] = {}
        for cid in cids:
            vids  = ptn_all_col_vids[cid][None, None, :] # collision vid
            n     = vids.shape[-1]
            batch = torch.zeros(1, 1, n, dtype=torch.long)
            frame = torch.full((1, 1, n), f_key, dtype=torch.long)
            ptn_vpos_cache[f_key][cid] = ptn_auramesh.get_positions_from_vids(vids, batch, frame)[0, 0].numpy() # ptn auramesh의 위치 (target)
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

        # goal0: follow source rotation (L1)
        _t0 = time.perf_counter()
        x_np = np.array(x, dtype=np.float32)
        diff = x_np - target
        func += lamda_0 * float(np.sum(np.abs(diff)))
        grad_np = (lamda_0 * np.sign(diff)).astype(np.float64)
        _tm['goal0'] += time.perf_counter() - _t0
        # print(f"frame {f}, iter {_tm['n_calls']}")
        # print(f"goal0. diff norm: {np.linalg.norm(diff):.4f}")

        # goal1: rotation matrix orthogonality (SVD projection)
        _t0 = time.perf_counter()
        target_local_R = x_np.reshape(22, 3, 3)
        normalized_R   = normalize_rotation_matrix(target_local_R)
        R_diff = target_local_R - normalized_R
        func      += lamda_1 * float(np.sum(np.linalg.norm(R_diff, axis=(1, 2))))
        grad_np   += (lamda_1 * 2.0 * R_diff).flatten()
        _tm['goal1_svd'] += time.perf_counter() - _t0
        # print(f"goal1. R_diff norm: {np.linalg.norm(R_diff):.4f}")

        # goal2: preserve skin-level contact points
        col_ids = col_ids_per_frame.get(f, np.array([], dtype=int))
        if len(col_ids) == 0:
            grad_np[:9] = 0.0  # root rotation excluded from optimization
            grad[:] = grad_np.tolist()
            return func

        # LBS once per gradient call
        _t0 = time.perf_counter()
        tgt_geo.set_pose_by_source_batch_frame(
            torch.tensor(target_local_R, dtype=torch.float32)[None, None, :],
            tgt_root_p_t[f]) # small character의 pose 설정 (tgt main)
        _tm['lbs'] += time.perf_counter() - _t0

        # target current joint positions/rotations (set by LBS above)
        cur_global_p = tgt_geo.global_p[0, 0].detach().cpu().numpy()  # (J, 3)
        cur_global_R = tgt_geo.global_R[0, 0].detach().cpu().numpy()  # (J, 3, 3)
        parent_idx   = tgt_geo.parents.cpu().numpy()                   # (J,)

        for cid in col_ids:
            _t0 = time.perf_counter()
            vids, bat, frm = _make_vids_batch_frame(all_col_vids[cid])
            tgt_vpos = tgt_geo.get_positions_from_vids(vids, bat, frm)[0, 0].numpy() # optimize에 의해 변경된 tgt_geo의 vertex position
            _tm['get_vpos'] += time.perf_counter() - _t0

            ptn_vpos = ptn_vpos_cache[f][cid] # partner auramesh의 vertex position (고정)
            diff_vpos = ptn_vpos - tgt_vpos  # (N_v, 3)
            # print("goal2. diff_vpos norm: {:.4f}".format(float(np.mean(np.abs(diff_vpos))))) # np.abs(diff_vpos)

            func += lamda_2 * float(np.mean(np.abs(diff_vpos)))

            # Gradient: rotation matrix Jacobian
            # d(pos_ee)/d(local_R[j]_{kl}) = dist_local[l] * R_parent[:, k]
            # where dist = pos_ee - pos_j (world),  dist_local = R_parent^T @ dist
            _t0 = time.perf_counter()
            ee_joint = tgt_jids[cid]
            eeT = cur_global_p[ee_joint]

            joint_chain = None
            for chain in joint_chains:
                if ee_joint in chain:
                    joint_chain = chain[:chain.index(ee_joint) + 1]
                    break

            dLdP = -(lamda_2 * np.sign(np.mean(diff_vpos, axis=0)))[None, :]  # (1, 3)  d(loss)/d(ee_pos) = -λ₂*sign(ptn-tgt)

            dPdX = np.zeros((3, len_rot_x))
            for j in reversed(joint_chain):
                x_start = j * 9
                curT    = cur_global_p[j]
                dist    = eeT - curT                    # (3,) world frame

                pj       = parent_idx[j]
                R_parent = cur_global_R[pj] if pj >= 0 else np.eye(3, dtype=np.float64)
                dist_loc = R_parent.T @ dist            # (3,) parent-local frame

                for k in range(3):
                    for l in range(3):
                        dPdX[:, x_start + k * 3 + l] += dist_loc[l] * R_parent[:, k]

            grad_np += (dLdP @ dPdX)[0]
            _tm['jacobian'] += time.perf_counter() - _t0

        grad_np[:9] = 0.0  # root rotation excluded from optimization
        grad[:] = grad_np.tolist()
        return func

    # ── Per-frame L-BFGS ──
    _t_opt_total = 0.0
    _PRINT_EVERY = 10  # print timing summary every N frames

    for f, pose in enumerate(src_motion.poses):
        target = pose.local_R.reshape(-1).astype(np.float32)
        x0 = target.tolist()  # start from source rotation each frame
        _reset_tm()

        _t_frame = time.perf_counter()
        state = xalglib.minlbfgscreate(2, x0) # memory, x0
        xalglib.minlbfgssetcond(state, 0.0, 0.0, 1e-6, 100) # 종료 조건: state, epsg, epsf, epsx, maxits
        xalglib.minlbfgsoptimize_g(state, function1_grad) # 최적화 상태 객체, 매 iteration마다 호출되는 콜백 (x->목적 함수 및 기울기 계산)
        x_opt, rep = xalglib.minlbfgsresults(state)
        _dt_frame = time.perf_counter() - _t_frame
        _t_opt_total += _dt_frame

        x_opt_np = np.array(x_opt, dtype=np.float32)
        diff_norm = float(np.linalg.norm(x_opt_np - target))  # |x_opt - src_rotation|

        updated_R = normalize_rotation_matrix(x_opt_np.reshape(22, 3, 3))
        tgt_motion.poses[f].local_R = updated_R
        tgt_motion.poses[f].update()

        if (f + 1) % _PRINT_EVERY == 0 or f == T - 1:
            nc = max(_tm['n_calls'], 1)
            # rep.iterationscount: 실제 L-BFGS iteration 횟수
            # rep.terminationtype: 종료 이유 (4=epsx 충족, 5=maxits 도달, 음수=에러)
            print(
                f"[frame {f+1:4d}/{T}] "
                f"frame={_dt_frame*1e3:.1f}ms  "
                f"iters={rep.iterationscount}  calls={_tm['n_calls']}  "
                f"term={rep.terminationtype}  diff={diff_norm:.4f}  "
                f"| goal0={_tm['goal0']*1e3/nc:.2f}ms  "
                f"svd={_tm['goal1_svd']*1e3/nc:.2f}ms  "
                f"lbs={_tm['lbs']*1e3/nc:.2f}ms  "
                f"get_vpos={_tm['get_vpos']*1e3/nc:.2f}ms  "
                f"jac={_tm['jacobian']*1e3/nc:.2f}ms  (per call)"
            )

    print(f"\n=== optimize_motion total: {_t_opt_total:.1f}s  ({_t_opt_total/T*1e3:.1f}ms/frame) ===")

    return tgt_motion


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


def smooth_motion(src_motion, tgt_motion,
                  mode,                  # 'gaussian' | 'oneeuro'
                  sigma=3,               # gaussian: sigma (frames)
                  min_cutoff=5.0,        # oneeuro: min cutoff Hz (높을수록 변화 보존)
                  beta=0.5,              # oneeuro: speed coeff (높을수록 빠른 변화 보존)
                  fps=30):
    """
    src_motion 대비 delta에 temporal smoothing을 적용한 뒤 tgt_motion을 in-place 업데이트.

    delta = tgt_R - src_R 에 필터를 적용하므로:
      - 변화 없는 프레임은 소스 그대로 유지
      - 변화 있는 프레임만 부드럽게 전파
    """
    T = len(src_motion.poses)
    src_R = np.stack([p.local_R for p in src_motion.poses])  # (T, J, 3, 3)
    tgt_R = np.stack([p.local_R for p in tgt_motion.poses])  # (T, J, 3, 3)
    delta_R = tgt_R - src_R                                   # (T, J, 3, 3)

    if mode == 'gaussian':
        smooth_delta = _smooth_gaussian(delta_R, sigma=sigma)
        print(f"smooth_motion: gaussian  sigma={sigma}")
    elif mode == 'oneeuro':
        smooth_delta = _smooth_oneeuro(delta_R, fps=fps,
                                       min_cutoff=min_cutoff, beta=beta)
        print(f"smooth_motion: oneeuro  min_cutoff={min_cutoff}  beta={beta}")
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'gaussian' or 'oneeuro'.")

    smoothed_R = normalize_rotation_matrix(
        (src_R + smooth_delta).reshape(-1, 3, 3)
    ).reshape(T, -1, 3, 3)

    for f in range(T):
        tgt_motion.poses[f].local_R = smoothed_R[f]
        tgt_motion.poses[f].update()

    return tgt_motion


def get_character_geometry(args, names, fbx_models):
    geometry = []
    for i, name in enumerate(names):
        geometry.append(Geometry(args, character=fbx_models[i], name=name))
    return geometry
