from pymovis.motion.ops import torchmotion
from xalglib import xalglib
import copy
import numpy as np
import torch
from pymovis.motion.ops.npmotion import *

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])


body_joints = [0, 9, 10, 11, 12, 13]
left_leg_joints = [1, 2, 3, 4]
right_leg_joints = [5, 6, 7, 8]
left_hand_joints = [14, 15, 16, 17]
right_hand_joints = [18, 19, 20, 21]
joint_chains = [body_joints, left_leg_joints, right_leg_joints, left_hand_joints, right_hand_joints]

col_frame  = None
src_colliding_vpos = None
src_colliding_vids = None
tgt_auramesh = None
jids1 = None
pose = None
root_p = None
f = None
check_col = None
def optimize(src_motion, tgt_motion, tgt_auramesh_, root_scale,
             col_frame_, jids1_, src_colliding_vpos_, src_colliding_vids_):
    """ 
    1. rec 
    2. colliding_cids의 position - colliding position
    """
    global col_frame 
    global src_colliding_vpos
    global src_colliding_vids
    global tgt_auramesh
    global jids1
    global pose
    global f
    global root_p
    global count_iter
    global check_col
    col_frame = col_frame_
    src_colliding_vpos = src_colliding_vpos_
    src_colliding_vids = src_colliding_vids_
    tgt_auramesh = tgt_auramesh_
    jids1 = jids1_

    for f_, pose_ in enumerate(src_motion.poses):
        # init setting for frame f
        pose = pose_
        f = f_
        count_iter = 0
        check_col = False 
        
        # Input 
        len_x = 66
        # root 
        root_p = torch.tensor(src_motion.poses[f].root_p).to('cpu')
        root_p[1] = root_p[1]*root_scale # 0.7
        
        
        # x: angle axis로 바꾸기. aaxis(3) -> rot matrix(9)
        x0 = [0.0] * len_x
        epsx = 1e-10 # Desired precision for variables
        maxits = 100 # Maximum number of iterations
        epsg = 1.0e-10 # Epsilon for Gradient
        epsf = 1.0e-10 # Epsilon for Function
        state = xalglib.minlbfgscreate(2, x0)
        xalglib.minlbfgssetcond(state, epsg, epsf, epsx, maxits)
        xalglib.minlbfgsoptimize_g(state, function1_grad)
        x_optimized, _ = xalglib.minlbfgsresults(state)
        
        # update 
        delta_aaxis = np.array(x_optimized).reshape(22, 3) # [:-3] # delta_angleAxis  # aaxis -> R
        delta_angle = np.linalg.norm(delta_aaxis, axis=-1)
        delta_angle[np.where(delta_angle < 1e-6)] = 1e-6
        delta_axis = delta_aaxis / delta_angle[:, None]
        delta_local_R = A_to_R(delta_angle, delta_axis)

        # update delta
        out_local_R = delta_local_R @ pose.local_R # TODO
        
        # set pose 
        tgt_motion.poses[f].local_R = out_local_R.astype('float32')
        tgt_motion.poses[f].root_p = np.array(root_p) # src_motion.poses[f].root_p # np.array(x_optimized[-3:])
        tgt_motion.poses[f].update()
        
        if check_col:
            # print("f {}, count_iter {}: localR {} \n".format(f, count_iter, x_optimized[54:]))
            check_col=False
        
    return tgt_motion


# compute loss func, grad
def function1_grad(x, grad, param=None):
    global count_iter
    global check_col
    func = 0
    len_rot_x = 66
    lamda_1 = 3.0 # 5000.0 delta값으로 바꾼다면 lambda값을 다시 확인해봐야겠다. 
    # lamda_2 = 1.0
    # lamda_3 = 0.1
    
    # check collision frame
    col_ids = np.where(col_frame == f)[0]
    if col_ids.shape[0] > 0:
        check_col = True
        for cid in col_ids:
            # src 
            src_vpos = np.array(src_colliding_vpos[cid].to('cpu'))
            src_vids = src_colliding_vids[cid]
            src_vids = src_vids[None, None, :].to('cpu')
            # tgt 
            len_vids = len(src_vids)
            batch = torch.tensor([0])[None, None, :].repeat(1, 1, len_vids)
            frame = torch.tensor([0])[None, None, :].repeat(1, 1, len_vids)

            
            # 1. spatial relationship for col frame
            count_iter += 1
            # aaxis -> R 
            """ this part is using (torch) """
            delta_aaxis = torch.tensor(x).reshape(22, 3) 
            delta_angle = torch.norm(delta_aaxis, dim=-1)
            delta_angle[torch.where(delta_angle < 1e-6)] = 1e-6
            delta_axis = delta_aaxis / delta_angle[:, None]
            delta_local_R = torchmotion.A_to_R(delta_angle, delta_axis)
            
            # update
            local_R = delta_local_R @ pose.local_R
            
            # set pose
            tgt_auramesh.set_pose_by_source_batch_frame(local_R[None, None, :], root_p[None, None, :])
            
            # tgt cpos
            tgt_cpos = tgt_auramesh.get_positions_from_vids(src_vids, batch, frame)[0,0]
            tgt_cpos = np.array(tgt_cpos.to('cpu'))
            
            
            """ update energy term"""
            # value 
            # num_col = src_vpos.shape[0]
            func += lamda_1*np.sum(np.square(src_vpos - tgt_cpos)) # *1/num_col
            
            # grad 
            # ee joint 
            ee_joint = jids1[cid]
            eeT = np.array(tgt_auramesh.global_p[0,0,ee_joint].to('cpu'))
            # select kinematic chain
            for chain in joint_chains:
                if ee_joint in chain:
                    joint_chain = chain
                    break
            inchain_idx = joint_chain.index(ee_joint)
            joint_chain = joint_chain[:inchain_idx+1]
            # joint chain다시 만들기, reverse order for loop
            
            # dLdP: (1, num_joint*3)
            dLdP = lamda_1*2*np.mean((src_vpos - tgt_cpos), axis=0)[None, :] # 1/num_col*
            
            # compute dPdX (num_joint*3, num_joint_chain*3)
            dPdX = np.zeros((3, len_rot_x))
            for j in reversed(joint_chain):
                x_start = j*3
                x_end = (j+1)*3
                # dist 
                curT = np.array(tgt_auramesh.global_p[0,0,j].to('cpu'))
                dist = eeT - curT # dim: (3,)
                # origin
                quat = R_to_Q(np.array(local_R[j].cpu())) 
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
                    world_rot = np.array(tgt_auramesh.global_R[0,0,j].to('cpu'))
                    world_axis = world_rot @ delta_axis
                    dT = delta_angle * np.cross(world_axis, dist) # np.deg2rad
                    
                    # update joint rotation (x,y,z)
                    for xid in range(x_start, x_end):
                        dPdX[eid, xid] = dT[eid]

                # update 
                dLdX = dLdP @ dPdX # (1, num_joint_chain*3)
                for xid in range(len_rot_x):
                    grad[xid] += lamda_1*(dLdX[0, xid])
    else:
        check_col = False        
    
    return func