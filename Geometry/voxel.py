import torch 
import time 
import os 
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
class Voxel:
    def __init__(self, args, frames, \
            bot_cids, bot_cids_positions, \
            source_cids, source_cids_positions, device):
        self.args = args
        self.device = device
        self.interval = args.interval
        self.num_voxel = 1 / self.interval
        self.init_val = args.init_val
        self.source_cids_positions = source_cids_positions

        # remove duplicates from frames and build the index
        self.colliding_frames = frames
        self.voxel_frames = torch.unique(frames)
        self.len_voxel_frames = len(self.voxel_frames)
        
        self.colliding_frame_idx_to_voxel_frame_idx = {}
        for i, frame in enumerate(frames):
            for j, voxel_frame in enumerate(self.voxel_frames):
                if frame == voxel_frame:
                    self.colliding_frame_idx_to_voxel_frame_idx[i] = j
                    break

        # min max over the valid cids of the joint
        bot_index = torch.where(bot_cids != -1)
        bot_min_max = get_min_max_from_positions(bot_cids_positions[bot_index]) # min max 6
        source_index = torch.where(source_cids != -1)
        source_min_max = get_min_max_from_positions(source_cids_positions[source_index]) # min max 6
        min_pos = get_min_pos(bot_min_max, source_min_max) # 3
        max_pos = get_max_pos(bot_min_max, source_min_max)

        self.min_int_pos = self.get_int_pos(min_pos)
        self.max_int_pos = self.get_int_pos(max_pos)

        self.len_x = int(self.max_int_pos[0] - self.min_int_pos[0] + 1)
        self.len_y = int(self.max_int_pos[1] - self.min_int_pos[1] + 1)
        self.len_z = int(self.max_int_pos[2] - self.min_int_pos[2] + 1)
        
        # uni frame, x,y,z
        # min max range -> voxel space : long tensor
        self.voxels_dist = torch.full((self.len_voxel_frames, self.len_x, self.len_y, self.len_z), self.init_val).to(self.device) # there must be no init val
        self.voxels_ref  = torch.full((self.len_voxel_frames, self.len_x, self.len_y, self.len_z), -1).to(self.device)           # there must be no -1

    # Set voxel (zero for body surface)
    def set_init_val(self, all_cids, all_positions):
        # Get valid cids idx 
        index = torch.nonzero(all_cids != -1, as_tuple=True)

        # cid positions from colliding cids & colliding frame
        cids = all_cids[index]  # 1 dim 
        positions = all_positions[index]
        voxel_frames = torch.zeros_like(index[0])
        for idx in self.colliding_frame_idx_to_voxel_frame_idx.keys():
            voxel_frame_indices = torch.where(index[0]==idx)[0]
            voxel_frames[voxel_frame_indices] = self.colliding_frame_idx_to_voxel_frame_idx[idx]

        # Set init of voxels
        voxel_idxs = self.get_voxel_idxs_by_position(positions) 
        self.voxels_dist[voxel_frames, voxel_idxs[:, 0], voxel_idxs[:, 1], voxel_idxs[:, 2]] = 0
        self.voxels_ref [voxel_frames, voxel_idxs[:, 0], voxel_idxs[:, 1], voxel_idxs[:, 2]] = cids
 
        # Update voxel
        for _ in range(100):
            count = self.update_distance() # x,y,z
            if count == 0:
                break

    def get_int_pos(self, pos):
        return self.get_int_val(pos).long() 
    
    def get_int_val(self, value):
        return torch.round(self.num_voxel * value)
    
    # get voxel indexes for all frames
    def get_voxel_idxs_by_position(self, position):
        return self.get_int_pos(position) - self.min_int_pos.to(self.device)
    
    def update_distance(self):
        # 3d voxel -> 1d array
        update_count = 0
        voxel_ids = torch.tensor([(x, y, z) for x in range(self.len_x) for y in range(self.len_y) for z in range(self.len_z)]) # 5120,3 for 1 frame
        onedim_voxel_dist = self.voxels_dist.reshape(len(self.voxels_dist), -1) # long tensor 
        onedim_voxel_ref  = self.voxels_ref.reshape(len(self.voxels_ref), -1) # long tensor 
        
        # Get neighbor distance and ref cid (for all frames)
        voxel_neighbor_dists, voxel_neighbor_refs = self.get_neighor_dist(voxel_ids) # f, ids, 6 

        # Condition 1: if any neighbor is initialized, propagate from this 
        check_all_neigh_init_val = torch.any(voxel_neighbor_dists.ne(self.init_val), dim=-1) # indices where at least one is not init val
        cond1_frame_ids, cond1_voxel_ids = torch.where(check_all_neigh_init_val == True) # index within 62,5120
        
        if len(cond1_frame_ids) > 0:
            # Condition 2: current voxel is initialized
            dist_values = onedim_voxel_dist[cond1_frame_ids, cond1_voxel_ids] # cond
            init_value_cond_index = torch.where(dist_values == self.init_val)[0]
            if init_value_cond_index.numel() > 0:
                cond2_frame_ids = cond1_frame_ids[init_value_cond_index]
                cond2_voxel_ids = cond1_voxel_ids[init_value_cond_index]

                # Get min_dist/min_ref_cids from neighbor
                min_dist, min_index = torch.min(voxel_neighbor_dists[cond2_frame_ids, cond2_voxel_ids], dim=1)
                min_ref_cids = voxel_neighbor_refs[cond2_frame_ids, cond2_voxel_ids, min_index]

                # Update
                onedim_voxel_dist[cond2_frame_ids, cond2_voxel_ids] = min_dist + 1
                onedim_voxel_ref [cond2_frame_ids, cond2_voxel_ids] = min_ref_cids
                self.voxels_dist = onedim_voxel_dist.reshape(self.voxels_dist.shape)
                self.voxels_ref  = onedim_voxel_ref.reshape(self.voxels_ref.shape)
                update_count += init_value_cond_index.numel()
 
        return update_count
    
    def get_neighor_dist(self, indices):
        # neighbor values [right left up down front back]
        len_indices = indices.shape[0]
        # torch.zeros(self.len_voxel_frames, len_indices, 6).cuda()
        neighbor_dis = torch.full((self.len_voxel_frames, len_indices, 6), 0).to(self.device)
        neighbor_ref = torch.full((self.len_voxel_frames, len_indices, 6), -1).to(self.device)
        
        # Clamp indices for neighboring 
        x_plus_indices = torch.clamp(indices[:, 0] + 1, max=self.len_x - 1)
        x_minu_indices = torch.clamp(indices[:, 0] - 1, min=0)
        y_plus_indices = torch.clamp(indices[:, 1] + 1, max=self.len_y - 1)
        y_minu_indices = torch.clamp(indices[:, 1] - 1, min=0)
        z_plus_indices = torch.clamp(indices[:, 2] + 1, max=self.len_z - 1)
        z_minu_indices = torch.clamp(indices[:, 2] - 1, min=0)

        # Assign neighbor distances and references using array indexing
        neighbor_dis[:, :, 0] = self.voxels_dist[:, x_plus_indices, indices[:, 1], indices[:, 2]] # f,c,6
        neighbor_dis[:, :, 1] = self.voxels_dist[:, x_minu_indices, indices[:, 1], indices[:, 2]]
        neighbor_dis[:, :, 2] = self.voxels_dist[:, indices[:, 0], y_plus_indices, indices[:, 2]]
        neighbor_dis[:, :, 3] = self.voxels_dist[:, indices[:, 0], y_minu_indices, indices[:, 2]]
        neighbor_dis[:, :, 4] = self.voxels_dist[:, indices[:, 0], indices[:, 1], z_plus_indices]
        neighbor_dis[:, :, 5] = self.voxels_dist[:, indices[:, 0], indices[:, 1], z_minu_indices]

        neighbor_ref[:, :, 0] = self.voxels_ref [:, x_plus_indices, indices[:, 1], indices[:, 2]]
        neighbor_ref[:, :, 1] = self.voxels_ref [:, x_minu_indices, indices[:, 1], indices[:, 2]]
        neighbor_ref[:, :, 2] = self.voxels_ref [:, indices[:, 0], y_plus_indices, indices[:, 2]]
        neighbor_ref[:, :, 3] = self.voxels_ref [:, indices[:, 0], y_minu_indices, indices[:, 2]]
        neighbor_ref[:, :, 4] = self.voxels_ref [:, indices[:, 0], indices[:, 1], z_plus_indices]
        neighbor_ref[:, :, 5] = self.voxels_ref [:, indices[:, 0], indices[:, 1], z_minu_indices]

        return neighbor_dis, neighbor_ref

    def get_voxel_pair(self, cids, all_positions):
        # Get valid cids idx 
        index = torch.nonzero(cids != -1, as_tuple=True)

        # Get colliding positions and frames 
        positions = all_positions[index]
        voxel_frames = torch.zeros_like(index[0])
        for idx in self.colliding_frame_idx_to_voxel_frame_idx.keys():
            voxel_frame_indices = torch.where(index[0]==idx)[0]
            voxel_frames[voxel_frame_indices] = self.colliding_frame_idx_to_voxel_frame_idx[idx]

        # position -> voxel idx 
        cid_int_positions = self.get_int_pos(positions) 
        voxel_idxs = cid_int_positions - self.min_int_pos.to(self.device)
        
        # voxel idx -> ref cid 
        ref_cids = self.voxels_ref[voxel_frames, voxel_idxs[:, 0], voxel_idxs[:, 1], voxel_idxs[:, 2]] # must be made 2-dim.

        return ref_cids


def get_min_max_from_positions(positions):
    aabb_of_motion = torch.zeros(6)
    aabb_of_motion[0] = torch.min(positions[..., 0])
    aabb_of_motion[1] = torch.min(positions[..., 1])
    aabb_of_motion[2] = torch.min(positions[..., 2])
    aabb_of_motion[3] = torch.max(positions[..., 0])
    aabb_of_motion[4] = torch.max(positions[..., 1])
    aabb_of_motion[5] = torch.max(positions[..., 2])

    return aabb_of_motion

def get_min_pos(positions0, positions1):
    min_pos = torch.zeros(3)
    min_pos[0] = torch.min(positions0[..., 0], positions1[..., 0])
    min_pos[1] = torch.min(positions0[..., 1], positions1[..., 1])
    min_pos[2] = torch.min(positions0[..., 2], positions1[..., 2])

    return min_pos

def get_max_pos(positions0, positions1):
    max_pos = torch.zeros(3)
    max_pos[0] = torch.max(positions0[..., 3], positions1[..., 3])
    max_pos[1] = torch.max(positions0[..., 4], positions1[..., 4])
    max_pos[2] = torch.max(positions0[..., 5], positions1[..., 5])

    return max_pos
