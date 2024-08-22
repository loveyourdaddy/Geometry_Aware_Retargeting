import sys, os
sys.path.append('.')
sys.path.append('../') # Retargeting_workspace

from pymovis.motion.ops import torchmotion
from xalglib import xalglib
from pymovis.motion.ops.npmotion import *
from Geometry.geometry import Geometry
from pymovis.motion.core import Pose, Motion
import copy
import numpy as np
import torch

body_joints = [0, 9, 10, 11, 12, 13]
left_leg_joints = [1, 2, 3, 4]
right_leg_joints = [5, 6, 7, 8]
left_hand_joints = [14, 15, 16, 17]
right_hand_joints = [18, 19, 20, 21]
joint_chains = [body_joints, left_leg_joints, right_leg_joints, left_hand_joints, right_hand_joints]

"""
1. rec
2. col_cids의 position - col position
"""

def optimize_motion(args, 
                    src_motion, tgt_motion,
                    tgt_geo, ptn_auramesh,
                    all_col_vids, ptn_all_col_vids,
                    col_frame, tgt_jids, ptn_jids):
    lamda_1 = 1.0 # 0.1 # 0.01 # 1.0
    lamda_2 = 3.0 # 3.0 # 3.0 # 10.0
    collision_preserving = True
    
    # compute loss func, grad
    def function1_grad(x, grad, param=None):
        nonlocal f, pose, target
        func = 0
        len_rot_x = 198
        
        # goal0. follow source motion
        for i in range(len_rot_x):
            val0 = abs(x[i] - target[i])
            func += val0
            grad[i] = x[i] - target[i]
        
        # goal1. rotation matrix
        target_local_R = np.array(x).reshape(22, 3, 3)
        target_local_R = np.asarray(target_local_R, dtype=np.float32)
        normalized_R = normalize_rotation_matrix(target_local_R)

        # Compute the difference between original and normalized matrices
        R_diff = np.linalg.norm(target_local_R - normalized_R, axis=(1, 2))
        func += lamda_1 * np.sum(R_diff)
        R_grad = lamda_1 * 2 * (target_local_R - normalized_R)
        grad[:] += R_grad.flatten()
        
        
        # target1. collision cpos
        if collision_preserving:
            target_local_R = torch.tensor(target_local_R)
            col_ids = np.where(col_frame == f)[0]
            tgt_root_p = tgt_motion.poses[f].root_p
            for cid in col_ids:
                # updated own pose 
                col_vids = all_col_vids[cid]
                len_vids = len(col_vids)
                col_vids = col_vids[None, None, :]
                batch = torch.tensor([0])[None, None, :].repeat(1, 1, len_vids)
                
                # update auramesh from updated pose
                tgt_geo.set_pose_by_source_batch_frame(
                    target_local_R[None, None, :], 
                    torch.tensor(tgt_root_p[None, None, :]))
                frame = torch.tensor([0])[None, None, :].repeat(1, 1, len_vids)
                
                # tgt cpos
                tgt_vpos = tgt_geo.get_positions_from_vids(col_vids, batch, frame)[0,0]
                tgt_vpos = np.array(tgt_vpos)

                
                # ptn_vpos
                # updated own pose 
                col_vids = ptn_all_col_vids[cid]
                len_vids = len(col_vids)
                col_vids = col_vids[None, None, :]
                batch = torch.tensor([0])[None, None, :].repeat(1, 1, len_vids)
                frame = torch.tensor([f])[None, None, :].repeat(1, 1, len_vids)
                
                ptn_vpos = ptn_auramesh.get_positions_from_vids(col_vids, batch, frame)[0,0]
                ptn_vpos = np.array(ptn_vpos)
                
                if f==246:
                    args.debug_points0 = ptn_vpos[::30]
                    args.debug_points1 = tgt_vpos[::30]
                
                # energy term
                val1 = lamda_2 * np.mean(np.abs(ptn_vpos - tgt_vpos)) # TODO
                func += val1
                
                # grad
                # effector
                ee_joint = tgt_jids[cid] # TODO: joint chain
                eeT = np.array(pose.global_p[ee_joint])
                
                # select kinematic chain
                for chain in joint_chains:
                    if ee_joint in chain:
                        joint_chain = chain
                        break
                inchain_idx = joint_chain.index(ee_joint)
                joint_chain = joint_chain[:inchain_idx+1]
                
                # dLdP: (1, num_joint*3)
                dLdP = lamda_2 * 2 * np.mean((ptn_vpos - tgt_vpos), axis=0)[None, :]  # 
                
                # compute dPdX (num_joint*3, num_joint_chain*3)
                dPdX = np.zeros((3, len_rot_x))
                for j in reversed(joint_chain):
                    x_start = j*9
                    x_end = (j+1)*9
                    
                    # dist
                    curT = np.array(pose.global_p[j])
                    dist = eeT - curT # dim: (3,)
                    # dist = np.abs(curT - eeT) # dim: (3,)
                    
                    # origin
                    quat = R_to_Q(np.array(target_local_R[j])) 
                    axis, angle = Q_to_A(quat)
                    aaxis = axis * angle
                    quat_inv = np.array([quat[0], -quat[1], -quat[2], -quat[3]])
                    for eid in range(3):
                        # updated
                        updated_aaxis = copy.deepcopy(aaxis)
                        updated_aaxis[eid] += 1.0
                        updated_angle, updated_axis = np.linalg.norm(updated_aaxis), updated_aaxis / np.linalg.norm(updated_aaxis)
                        updated_quat = A_to_Q(updated_angle, updated_axis)
                        
                        # delta
                        delta_quat = quaternion_multiply(quat_inv, updated_quat) # 뒤집어 보기
                        delta_axis, delta_angle = Q_to_A(delta_quat)
                        
                        # dT
                        world_R = np.array(pose.global_R[j])
                        world_axis = world_R @ delta_axis
                        dT = delta_angle * np.cross(world_axis, dist) 
                        
                        # update joint rotation (x,y,z)
                        for xid in range(x_start, x_end):
                            dPdX[eid, xid] = dT[eid]

                    # update 
                    dLdX = dLdP @ dPdX # (1, num_joint_chain*3)
                    for xid in range(len_rot_x):
                        grad[xid] += dLdX[0, xid] # lamda_2 * 
        
        return func

    # paramterters
    epsx = 1e-6 # 1e-10 # Desired precision for variables
    maxits = 100 # Maximum number of iterations
    epsf = 0.0 #1.0e-10 # 0.0 # Epsilon for Function
    epsg = 0.0 #1.0e-10 # Epsilon for Gradient
    
    # Input 
    len_x = 198 # 66
    num_joint = 22
    for f, pose in enumerate(src_motion.poses):
        # target
        target = pose.local_R.reshape(-1)
        
        # x: angle axis로 바꾸기. aaxis(3) -> rot matrix(9)
        x0 = num_joint * \
            [1.0, 0.0, 0.0, 
             0.0, 1.0, 0.0, 
             0.0, 0.0, 1.0,]
        state = xalglib.minlbfgscreate(2, x0)
        xalglib.minlbfgssetcond(state, epsg, epsf, epsx, maxits)
        xalglib.minlbfgsoptimize_g(state, function1_grad)
        x_optimized, rep = xalglib.minlbfgsresults(state)
        x_optimized = np.array(x_optimized)
        
        # if f >= 0: # 246
        #     print(f"Frame {f} Optimization Report:")
        #     print(f"  Iterations: {rep.iterationscount}")
        #     print(f"  Termination type: {rep.terminationtype}")
        #     print(f"  Function evaluations: {rep.nfev}")
        
        # update local
        updated_target = x_optimized.reshape(22, 3, 3)
        # normalize 
        updated_target = normalize_rotation_matrix(updated_target)
        # update 
        tgt_motion.poses[f].local_R = updated_target
        tgt_motion.poses[f].update()
        
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
    q1_flipped = q1 * np.where(flip[:, None], -1, 1)  # Flip sign without modifying original q1

    near_one = dot > 0.9999
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    theta = theta_0 * t
    sin_theta = np.sin(theta)

    s1 = np.cos(theta) - dot * sin_theta / np.where(sin_theta_0 == 0, 1, sin_theta_0)
    s2 = sin_theta / np.where(sin_theta_0 == 0, 1, sin_theta_0)

    result = s1[..., None] * q1_flipped + s2[..., None] * q2
    result[near_one] = (1 - t) * q1[near_one] + t * q2[near_one]  # Linear interpolation when close

    norms = np.linalg.norm(result, axis=-1, keepdims=True)
    result /= norms

    return result

def normalize_rotation_matrix(matrix):
    # Reshape the input if it's not already in the correct shape
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 3, 3)
    
    # Initialize the output array
    normalized = np.zeros_like(matrix)
    
    for i in range(matrix.shape[0]):
        # Perform SVD
        U, _, Vt = np.linalg.svd(matrix[i])
        
        # Reconstruct the rotation matrix
        normalized[i] = U @ Vt
        
        # Ensure right-handed coordinate system
        if np.linalg.det(normalized[i]) < 0:
            normalized[i][:, 2] *= -1
    
    return normalized