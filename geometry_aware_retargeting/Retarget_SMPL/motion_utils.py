"""Motion ↔ tensor conversion and pose update from global positions."""
import numpy as np
import torch
from pymovis.motion.ops import torchmotion


def get_rootP_localR_globalP_from_motion(args, poses):
    global_p = []
    local_R = []
    root_p = []
    for p in poses:
        root_p.append(p.root_p)
        local_R.append(p.local_R)
        global_p.append(p.global_p)

    return torch.tensor(np.array(root_p)).to(args.device), torch.tensor(np.array(local_R)).to(args.device), torch.tensor(np.array(global_p)).to(args.device)

def get_rootP_localR_globalP_from_numpy_motion(args, poses): 
    global_p = []
    local_R = []
    root_p = []
    for p in poses:
        root_p.append(p.root_p)
        local_R.append(p.local_R)
        global_p.append(p.global_p)
    
    root_p = torch.from_numpy(np.stack(root_p, axis=0)).to(args.device)
    local_R = torch.from_numpy(np.stack(local_R, axis=0)).to(args.device)
    global_p = torch.from_numpy(np.stack(global_p, axis=0)).to(args.device)
    
    return root_p, local_R, global_p

def rotation_matrix_from_vectors(vec1, vec2):
    dev = vec1.device
    a, b = (vec1 / torch.norm(vec1)).reshape(3), (vec2 / torch.norm(vec2)).reshape(3)
    v = torch.cross(a, b)
    if torch.any(v): # if not all zeros (v1 != v2)
        c = torch.dot(a, b)
        s = torch.norm(v)
        kmat = torch.tensor([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]).to(dev) # skew symmetric cross product matrix
        return torch.eye(3).to(dev) + kmat + torch.matmul(kmat, kmat) * ((1 - c) / (s ** 2))
    else:
        return torch.eye(3).to(dev) # cross of all zeros only occurs on identical directions

def update_pose_by_global_p(local_R, global_p, global_R, 
                            skeleton, update_global_p):
    parent_idx = skeleton.parent_idx
    root_p = update_global_p[0]
    for i in range(1, len(global_p)):
        parent_i = parent_idx[i]
        if parent_i == -1:
            continue
        original_p = global_p[i] - global_p[parent_i] 
        delta_p = update_global_p[i] - update_global_p[parent_i]
        delta_global_R = rotation_matrix_from_vectors(original_p, delta_p)

        # update parent rotation 
        grandparent_i = parent_idx[parent_i]
        if grandparent_i == -1:
            continue
        parent_global_R_inv = torch.inverse(global_R[grandparent_i])
        parent_global_R = global_R[parent_i]
        local_R[parent_i] = torch.matmul(torch.matmul(parent_global_R_inv, delta_global_R), parent_global_R)
        
        global_R, global_p = torchmotion.R_fk(
            local_R, root_p, skeleton
        )
        
    return local_R, root_p

def update_motion_by_global_p(motion, update_global_p):
    device = update_global_p.device
    skeleton = motion.skeleton
    for f, pose in enumerate(motion.poses):
        local_R, root_p = update_pose_by_global_p(torch.tensor(pose.local_R) .float().to(device), 
                                                  torch.tensor(pose.global_p).float().to(device), 
                                                  torch.tensor(pose.global_R).float().to(device), 
                                                  skeleton, update_global_p[f])
        pose.local_R = local_R.cpu().numpy()
        pose.root_p = root_p.cpu().numpy()
        pose.update()

    return motion
