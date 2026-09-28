import pickle
import torch
import numpy as np
import os 

class BVH_Tree:
    def __init__(self, args, geometry):
        self.args = args
        self.args.device = args.device
        self.bvh_preprocessed = self.args.bvh_preprocess 
        
        self.geometry = geometry
        self.c_position = self.geometry.v_position
        self.Tpose = self.geometry.Tpose

        # depth level: root 0, joint 1, 2, leaf 3
        self.leaf_depth = 3

        # fids & subsampling 
        perjoint_fids = [] 
        self.max_len_fids = 0
        for i, fids in enumerate(geometry.perjoint_fids):
            # sub-sample when there are too many fids (TODO: uniform subsampling )
            # print("i {}, fid {}".format(i, len(fids)))
            if len(fids) > 10000:
                fids = fids[::10]
            perjoint_fids.append(fids)
            if len(fids) > self.max_len_fids:
                self.max_len_fids = len(fids)

        # Set bvh tree
        if self.bvh_preprocessed==False:
            # saved geometry data 
            path = "./preprocess"
            os.makedirs(path, exist_ok=True)
            path = path+'/{}/{}.pkl'.format(geometry.name, "bvh_tree")

            # if there is no pkl file, make new one
            if os.path.isfile(path) is False:
                print(f"make new bvh tree for {geometry.name}")
                self.save_BVH_tree() # TODO: saving does not work.
            
            with open(path, 'rb') as f:
                self.root = pickle.load(f)
        else:
            self.save_BVH_tree()
    
    def save_BVH_tree(self):
        geometry = self.geometry
        
        # depth 0: whole body 
        self.root = Node(self.args, torch.tensor([0, 0, 0]), torch.tensor([0, 0, 0, 0, 0, 0]), [], [], depth=0)
        self.root.boundary_diff = torch.zeros(geometry.num_joints, 8, 3)
        
        # depth 1 (joint level)
        global_p = torch.tensor(self.Tpose.global_p)
        for j in range(geometry.num_joints): 
            # perjoint cid, vids 
            joint_cids = geometry.perjoint_cids[j]
            joint_fids = geometry.perjoint_fids[j]
            
            # geo info: aabb (6dim) = [min_x, min_y, min_z, max_x, max_y, max_z]
            perjoint_min = geometry.perjoint_min[j] 
            perjoint_max = geometry.perjoint_max[j] 
            aabb = \
                torch.tensor([perjoint_min[0], perjoint_min[1], perjoint_min[2],
                              perjoint_max[0], perjoint_max[1], perjoint_max[2]])
            
            # Node: joint pos, center pos, geo, cids, fids, depth
            node = Node(self.args, global_p[j], aabb, joint_cids, joint_fids, depth=1)
            self.root.add_node(node)
            self.root.boundary_diff[j] = node.boundary_diff
            
        with open('./preprocess/{}/{}.pkl'.format(geometry.name, "bvh_tree"), 'wb') as f:
            pickle.dump(self.root, f)
        print("BVH Tree is saved")
        
        # cuda setting 
        self.geometry.v_position = self.geometry.v_position.to(self.args.device)
        
    """ Update """
    def update_joint_aabb(self):
        """ 
        d1: [b, f, 22 joints (d1),8 boundary, 3 pos]
        d2: [b, f, 22 joints (d1), 8 d2, 8 boundary, 3 pos]
        """ 
        mesh_global_R = self.geometry.global_R
        num_batch, num_frame = mesh_global_R.shape[0], mesh_global_R.shape[1]

        # Expand Tpose.global_R
        Tpose_global_R = torch.tensor(self.Tpose.global_R).to(self.args.device)
        expanded_Tpose_global_R = Tpose_global_R.unsqueeze(0).unsqueeze(0).expand(
            num_batch, num_frame, -1, -1, -1
        ).to(self.args.device)
        Tpose_global_R_inverse = torch.inverse(expanded_Tpose_global_R)

        # Calculate global_R for all joints 
        global_R = torch.matmul(mesh_global_R, Tpose_global_R_inverse)
        global_R = global_R.unsqueeze(3).repeat(1,1,1,8,1,1)

        # Expand global_p 
        expanded_global_p = self.geometry.global_p.unsqueeze(3).unsqueeze(-1).repeat(1,1,1,8,1,1)

        # boundary positions in batch
        boundary_diff = self.root.boundary_diff.to(self.args.device)
        boundary_diff = boundary_diff.unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(num_batch, num_frame, 1, 1, 1, 1)

        boundary_pos = expanded_global_p + torch.matmul(global_R, boundary_diff)
        boundary_pos = boundary_pos.squeeze(-1)

        self.boundary_pos = boundary_pos
    

class Node:
    # Node: joint pos, geo, cids, depth
    def __init__(self, args, Tpose_joint_pos, aabb, cids, fids, depth):
        self.args = args
        self.args.device = args.device
        self.Tpose_joint_pos = Tpose_joint_pos
        self.cids = cids
        self.fids = fids
        self.depth = depth
        self.child = []

        # boundary position and diff (8 dim) <- aabb (min/max 6 dim)
        self.set_boundary_position(aabb)
        self.set_boundary_diff()

    def add_node(self, node):
        self.child.append(node)

    """ Set """
    def set_boundary_position(self, aabb: torch.Tensor):
        min_x, min_y, min_z, max_x, max_y, max_z = aabb[0], aabb[1], aabb[2], aabb[3], aabb[4], aabb[5]

        # boundary position
        self.boundary_pos = torch.zeros(8, 3)
        self.boundary_pos[0] = torch.tensor((max_x, max_y, max_z))
        self.boundary_pos[1] = torch.tensor((min_x, max_y, max_z))
        self.boundary_pos[2] = torch.tensor((max_x, min_y, max_z))
        self.boundary_pos[3] = torch.tensor((min_x, min_y, max_z))
        self.boundary_pos[4] = torch.tensor((max_x, max_y, min_z))
        self.boundary_pos[5] = torch.tensor((min_x, max_y, min_z))
        self.boundary_pos[6] = torch.tensor((max_x, min_y, min_z))
        self.boundary_pos[7] = torch.tensor((min_x, min_y, min_z))
    
    # diff vec = joint pos -> boundary pos
    def set_boundary_diff(self):
        self.boundary_diff = torch.zeros(8, 3).to(self.args.device)
        for i in range(8):
            self.boundary_diff[i] = self.boundary_pos[i] - self.Tpose_joint_pos
    
# Common functions 
""" Get """
def get_aabbs_from_pos(pos, valid_idx):
    min_values = torch.min(pos[valid_idx], dim=-2).values
    max_values = torch.max(pos[valid_idx], dim=-2).values
    valid_aabbs = torch.cat((min_values, max_values), dim=-1)

    # padded (invalid) slots are filled with a min>max sentinel so that they never overlap.
    aabbs = torch.empty(pos.shape[0], pos.shape[1], 6, device=pos.device)
    aabbs[..., :3] = float('inf')
    aabbs[..., 3:] = float('-inf')
    aabbs[valid_idx] = valid_aabbs

    return aabbs

""" overlap test """
# pos: [col_frame, subaabb 8, boundary 8, pos 3]
def joint_level_overlap_test(bot_boundary_pos, source_boundary_pos):
    device = bot_boundary_pos.device
    num_batch, num_frame, num_joints, _, _ = bot_boundary_pos.shape

    # boundary position ->  min max (aabb)
    # (col 8, subpart 8, boundary pos 8, pose 3) -> (col 8, subpart 8, minmax 6)
    # b,64,22,6 
    aabb1 = get_joint_level_min_max(bot_boundary_pos)
    aabb2 = get_joint_level_min_max(source_boundary_pos)
    # b,64,22,22,6 
    ext_aabb1 = aabb1[:, :, :, None, :].repeat(1, 1, 1, num_joints, 1)
    ext_aabb2 = aabb2[:, :, None, :, :].repeat(1, 1, num_joints, 1, 1)

    # (col 8, subpart 8, boundary pos 8, pose 3) 
    # 64,22,8,3
    points1 = bot_boundary_pos    
    points2 = source_boundary_pos 
    # b,64,22,22,8,3
    ext_points1 = points1[:, :, :, None, :, :].repeat(1, 1, 1, num_joints, 1, 1)
    ext_points2 = points2[:, :, None, :, :, :].repeat(1, 1, num_joints, 1, 1, 1)

    # (col_frame, col_joint, subpart 8) # 64,22,8 # last dim 8 -> sum
    mask1 = torch.zeros(num_batch, num_frame, num_joints, num_joints, 8, dtype=torch.bool).to(device) 
    mask2 = torch.zeros(num_batch, num_frame, num_joints, num_joints, 8, dtype=torch.bool).to(device) 

    # for each sub part(8) of joint
    for i in range(8):
        # point: 64, 22, (8[i],) 3, aabb: 64,22,6 -> mask: 64,22,22
        mask1[..., i] = point_and_aabb_overlap_test(ext_points1[..., i, :], ext_aabb2[..., :])
        mask2[..., i] = point_and_aabb_overlap_test(ext_points2[..., i, :], ext_aabb1[..., :])

    # col 8, subpart 8 # 64,22
    mask1_ = mask1.sum(dim=-1).to(device) 
    mask2_ = mask2.sum(dim=-1).to(device) 

    # Combine the masks along all axes
    # mask1_/mask2_ are integers made by bool.sum() (number of corners, 0~8), so
    # a plain & (bitwise AND) must not be used (e.g. 2 & 1 == 0, False even though both overlap).
    # "whether both have at least one overlapping corner" must be decided with a logical AND.
    intersection_mask = (mask1_ > 0) & (mask2_ > 0)

    return intersection_mask 

def point_and_aabb_overlap_test(point, aabb):
    """
    # points : 64,22,3
    # aabb: 64,22,6
    # return 64,22
    """
    return torch.gt(point[..., 0], aabb[..., 0]) & torch.lt(point[..., 0], aabb[..., 3]) &\
           torch.gt(point[..., 1], aabb[..., 1]) & torch.lt(point[..., 1], aabb[..., 4]) &\
           torch.gt(point[..., 2], aabb[..., 2]) & torch.lt(point[..., 2], aabb[..., 5])

# aabb: [col_frame, num_case n, pos 6] -> overlap: [frame, n, m]
def minmax_overlap_test(aabb1, aabb2, margin=0.0):
    """
    margin(m): two AABBs count as "touching" when the gap between them is at most margin (relaxes the previous
    strict overlap condition, which was True only on a full overlap). margin=0 is the same pure overlap test as before.
    """
    aabb1 = aabb1[..., :, None, :]
    aabb2 = aabb2[..., None, :, :]

    cond1 = (aabb1[..., 0] < aabb2[..., 3] + margin) & (aabb1[..., 3] + margin > aabb2[..., 0]) & \
            (aabb1[..., 1] < aabb2[..., 4] + margin) & (aabb1[..., 4] + margin > aabb2[..., 1]) & \
            (aabb1[..., 2] < aabb2[..., 5] + margin) & (aabb1[..., 5] + margin > aabb2[..., 2])

    cond2 = (aabb2[..., 0] < aabb1[..., 3] + margin) & (aabb2[..., 3] + margin > aabb1[..., 0]) & \
            (aabb2[..., 1] < aabb1[..., 4] + margin) & (aabb2[..., 4] + margin > aabb1[..., 1]) & \
            (aabb2[..., 2] < aabb1[..., 5] + margin) & (aabb2[..., 5] + margin > aabb1[..., 2])

    overlap = cond1 & cond2

    return overlap

""" min max """
def get_joint_level_min_max(boundary_pos: torch.Tensor):
    # B, T, J, 8, 3 = boundary_pos.shape
    min_vals = boundary_pos.min(dim=-2).values # [B, T, 22, 3]
    max_vals = boundary_pos.max(dim=-2).values # [B, T, 22, 3]
    aabb = torch.cat((min_vals, max_vals), dim=-1) # [B, T, 22, 6]
    return aabb