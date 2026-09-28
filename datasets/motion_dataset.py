'''
python datasets/save_motion_data.py
'''
import sys
sys.path.append('../0_geometry_aware_retargeting')
# sys.path.append('..')

from pymovis.motion.ops.torchmotion import Q_to_R, R_fk_from_given_info
from datasets.motion_functions import *
from option_motion import trainset_bvh
from etc.etc import deepcopy
import torch
import numpy as np

""" Prepare motion """


class Dataset():
    def __init__(self, args):
        self.args = args

    """ Character """
    # 1 characters, 2 times 
    def get_char_data(self, source_character0=None, source_character1=None, target_character0=None, target_character1=None,
                      source_name=None, target_name0=None, target_name1=None):
        # set character
        self.source_character0, self.source_character1 = source_character0, source_character1
        self.target_character0, self.target_character1 = target_character0, target_character1
        self.set_skeleton() # source skeleton & parent index 
        
        # Source info (offset & anchor position)
        # ptn: 0, main (dfm): 1
        source_name = self.source_character0.meshes[0].mesh_gl.name
        if "_sub" in source_name:
            source_name = source_name.replace("_sub", "")
        # offset
        self.source_offsets0 = torch.tensor(np.load("./preprocess/{}/offset.npy".format(source_name))).to(self.args.device)
        self.source_offsets1 = torch.tensor(np.load("./preprocess/{}/offset.npy".format(source_name))).to(self.args.device)
        # aabb max min
        self.source_aabb_max_min0 = torch.tensor(np.load("./preprocess/{}/aabb_max_min.npy".format(source_name))).to(self.args.device)
        self.source_aabb_max_min1 = torch.tensor(np.load("./preprocess/{}/aabb_max_min.npy".format(source_name))).to(self.args.device)
        
        # load target info
        if self.args.is_train:
            self.load_target_train_character_info()
        else:
            target_name0 = self.target_character0.meshes[0].mesh_gl.name
            target_name1 = self.target_character1.meshes[0].mesh_gl.name
            target_name0 = target_name0.replace("_sub", "")
            target_name1 = target_name1.replace("_sub", "")
            self.load_test_target_character_info(target_name0, target_name1)
        
    def set_skeleton(self):
        self.source_skeleton0 = self.source_character0.meshes[0].source_skeleton
        self.source_skeleton1 = self.source_character1.meshes[0].source_skeleton
        self.target_skeleton0 = self.target_character0.meshes[0].source_skeleton
        self.target_skeleton1 = self.target_character1.meshes[0].source_skeleton
        self.parent_idx0 = self.target_character0.meshes[0].source_skeleton.parent_idx
        self.parent_idx1 = self.target_character1.meshes[0].source_skeleton.parent_idx
    
    def load_geo_data(self, name):
        from datasets.character_functions import get_a_smpl_character
        _, _, geo = get_a_smpl_character(self.args, name)
        
        self.name_to_idx = geo.name_to_idx
        self.bind_trf_inv = geo.bind_trf_inv
        self.anchor_vids = geo.anchor_vids
        self.anchor_vpos = geo.anchor_vpos
        self.names = geo.names
        self.parents = geo.parents
        
        # select anchor vids 
        self.skinning_indices1 = geo.skinning_indices1[self.anchor_vids]
        self.skinning_weights1 = geo.skinning_weights1[self.anchor_vids]
        self.skinning_indices2 = geo.skinning_indices2[self.anchor_vids]
        self.skinning_weights2 = geo.skinning_weights2[self.anchor_vids]

    # 1 characters, 2 times 
    def load_target_train_character_info(self):
        self.load_scale()
        
        # target 
        self.target_offsets0 = []
        self.target_offsets1 = []
        self.target_aabb_max_min0 = []
        self.target_aabb_max_min1 = []
        
        name0 = self.target_character0.meshes[0].mesh_gl.name
        name1 = self.target_character1.meshes[0].mesh_gl.name
        # print("character: ", name1)
        
        # relative values
        # scaled offset (num_char, num_scale, 22, 3)
        target_offsets0 = torch.tensor(np.load("./preprocess/{}/offset.npy".format(name0))).to(self.args.device)
        target_offsets1 = torch.tensor(np.load("./preprocess/{}/scaled_offset.npy".format(name1))).to(self.args.device)
        target_offsets0 = target_offsets0.reshape(1, 22, 3).repeat(self.num_scale, 1, 1)
        # append: rid 
        self.target_offsets0.append(target_offsets0)
        self.target_offsets0.append(target_offsets1)
        self.target_offsets1.append(target_offsets1)
        self.target_offsets1.append(target_offsets0)
        
        # absolute values
        # aabb length (num_char, num_scale, mesh_dim, 3)
        target_aabb_max_min0 = torch.tensor(np.load("./preprocess/{}/aabb_max_min.npy".format(name0))).to(self.args.device)
        target_aabb_max_min1 = torch.tensor(np.load("./preprocess/{}/scaled_aabb_max_min.npy".format(name1))).to(self.args.device)
        target_aabb_max_min0 = target_aabb_max_min0.repeat(self.num_scale, 1, 1)
        self.target_aabb_max_min0.append(target_aabb_max_min0)
        self.target_aabb_max_min0.append(target_aabb_max_min1)
        self.target_aabb_max_min1.append(target_aabb_max_min1)
        self.target_aabb_max_min1.append(target_aabb_max_min0)
        # stack
        self.target_offsets0      = torch.stack(self.target_offsets0, dim=0)
        self.target_offsets1      = torch.stack(self.target_offsets1, dim=0)
        self.target_aabb_max_min0 = torch.stack(self.target_aabb_max_min0, dim=0)
        self.target_aabb_max_min1 = torch.stack(self.target_aabb_max_min1, dim=0)

        # valid 
        valid_source_name0 = "Ybot"
        valid_source_name1 = "Ybot"
        valid_target_name0 = "Ybot"
        valid_target_name1 = "Amy"
        
        self.valid_source_offsets0 = torch.tensor(np.load("./preprocess/{}/offset.npy".format(valid_source_name0))).to(self.args.device)
        self.valid_source_offsets1 = torch.tensor(np.load("./preprocess/{}/offset.npy".format(valid_source_name1))).to(self.args.device)
        self.valid_source_aabb_max_min0 = torch.tensor(np.load("./preprocess/{}/aabb_max_min.npy".format(valid_source_name0))).to(self.args.device)
        self.valid_source_aabb_max_min1 = torch.tensor(np.load("./preprocess/{}/aabb_max_min.npy".format(valid_source_name1))).to(self.args.device)
        
        self.valid_target_offsets0 = torch.tensor(np.load("./preprocess/{}/offset.npy".format(valid_target_name0))).to(self.args.device)
        self.valid_target_offsets1 = torch.tensor(np.load("./preprocess/{}/offset.npy".format(valid_target_name1))).to(self.args.device)
        self.valid_target_aabb_max_min0 = torch.tensor(np.load("./preprocess/{}/aabb_max_min.npy".format(valid_target_name0))).to(self.args.device)
        self.valid_target_aabb_max_min1 = torch.tensor(np.load("./preprocess/{}/aabb_max_min.npy".format(valid_target_name1))).to(self.args.device)

    def load_test_target_character_info(self, target_name0, target_name1):
        # target0
        # print("Character0 input info load from {}".format("./preprocess/{}/".format(target_name0)))
        self.target_offsets0 = torch.tensor(np.load("./preprocess/{}/offset.npy".format(target_name0))).to(self.args.device)
        self.target_aabb_max_min0 = torch.tensor(np.load("./preprocess/{}/aabb_max_min.npy".format(target_name0))).to(self.args.device)

        # target1
        # only small
        if self.args.test_type=="SMPLx":
            check_SMPLx_scale = True # True # needs an additional condition
        else:
            check_SMPLx_scale = False

        if check_SMPLx_scale:
            print("load from saved scaled info")
            # index
            total_index = 0 # self.args.SMPLx_scale_index
            
            # offset and anchor
            target_offsets1 = torch.tensor(np.load("./preprocess/{}/scaled_offset.npy".format(self.args.test_smpl_mesh))).to(self.args.device)
            target_aabb_max_min1 = torch.tensor(np.load("./preprocess/{}/scaled_aabb_max_min.npy".format(self.args.test_smpl_mesh))).to(self.args.device)
            self.target_offsets1 = target_offsets1[total_index]
            self.target_aabb_max_min1 = target_aabb_max_min1[total_index]
        # SMPLx & Mixamo
        else:
            if self.args.test_type == "SMPLx":
                # prefix : only one of scale and mesh scale should change
                if self.args.SMPLx_scale != 1: 
                    prefix = f"scale{self.args.SMPLx_scale}_"
                elif self.args.SMPLx_mesh_scale != 1: 
                    prefix = f"scale_mesh{self.args.SMPLx_mesh_scale}_"
                else:
                    prefix = ""
            else:
                prefix = ""
            # print("Character1 input info load from {}".format("./preprocess/{}/{}".format(target_name1, prefix)))
            
            # scaled_offset scaled_aabb_max_min
            # self.target_offsets1 = torch.tensor(np.load("./preprocess/{}/{}offset.npy".format(target_name1, prefix))).to(self.args.device)
            # self.target_aabb_max_min1 = torch.tensor(np.load("./preprocess/{}/{}aabb_max_min.npy".format(target_name1, prefix))).to(self.args.device)
            self.target_offsets1 = torch.tensor(np.load("./preprocess/{}/{}offset.npy".format(target_name1, prefix))).to(self.args.device)
            self.target_aabb_max_min1 = torch.tensor(np.load("./preprocess/{}/{}aabb_max_min.npy".format(target_name1, prefix))).to(self.args.device)


    """ Motion """
    def get_input_motion(self, motions0, motions1):
        """ input motion """
        if self.args.is_train:
            # train index
            if self.args.sepearte_train_data:
                f = open("./datasets/index/train_index", "r")
                lines = f.readlines()
                train_idx = []
                for line in lines:
                    train_idx.append([int(i) for i in line.split()][0])
                f.close()

            # frames
            num_frames = 0
            for i in range(len(motions0)):
                num_frames += len(motions0[i])
            self.fps = 30 # 24
            fps = self.fps
            print("Motion: {} frames ({}s, {} Motions)".format(num_frames, round(num_frames/fps, 3), len(motions0)))

            # motion
            if self.args.windowed_motion:
                input0, input1 = get_window_motion(self.args, motions0, motions1)
                
                num_window = input0.shape[0]
                window_size = input0.shape[1]
                
                frame = input0.shape[0] * input0.shape[1]
                sec = round(frame/fps, 3)
                minutes = round(sec/60, 3)
                
                print("Source: {} frames ({} windows, {} window_size, {}s, {}m)".format(frame, num_window, window_size, sec, minutes))
                self.length_list = 0
            else:
                input0, input1, length_list = get_full_motion_with_pad(self.args, motions0, motions1)
                self.length_list = length_list

                frame = input0.shape[0] * input0.shape[1]
                print("Padded: {} frames ({}s, {} windows, {} window_size)".format(frame, round(frame/fps, 3), input0.shape[0], input0.shape[1]))
        else:
            input0, input1 = get_vectorized_motion(self.args, motions0, motions1)
            input0, input1 = input0[None, ...], input1[None, ...]
            self.length_list = [input0.shape[1]]
        self.input_motion0, self.input_motion1 = deepcopy(input0), deepcopy(input1)
        
        # Root Factor out
        self.num_window, self.window_size, _ = input0.shape
        # factor out
        factorout_input0, factorout_input1 = factor_out(input0, input1)
        factorout_input_pos0, factorout_input_pos1 = self.get_factor_outed_pos(input0, input1)
        # self
        self.input0, self.input1 = factorout_input0, factorout_input1 
        self.input_pos0, self.input_pos1 = factorout_input_pos0, factorout_input_pos1 
        
    def get_factor_outed_pos(self, input0, input1):
        # 1. put the root at 0 and the root orientation forward.
        # 2. compute the positions.
        num_window, window_size, _ = input0.shape
        
        from pymovis.motion.ops.torchmotion import R6_to_R
        input_R0 = R6_to_R(input0[..., :-3].reshape(num_window, window_size, 22, 6))
        input_R1 = R6_to_R(input1[..., :-3].reshape(num_window, window_size, 22, 6))
        # root as zero 
        zero_root_p = torch.zeros_like(input1[..., -3:])
        forward_root_R = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).to(self.args.device)
        input_R0[:, :, 0] = forward_root_R
        input_R1[:, :, 0] = forward_root_R
        # root orientation as forward
        _, input_pos0 = R_fk_from_given_info(input_R0, zero_root_p, self.source_offsets0.to(self.args.device), self.parent_idx0)
        _, input_pos1 = R_fk_from_given_info(input_R1, zero_root_p, self.source_offsets1.to(self.args.device), self.parent_idx1)
        
        return input_pos0, input_pos1

    def get_input_anchor_pos(self, ref_geo):
        """
        Precompute the per-frame anchor positions and store them in self.input_anchor0/1 (num_window, window_size, 22, 15).
        (computed once in preprocessing -> saved to the pkl -> only loaded in training. Not recomputed at every step)

        Design:
          - the anchor definition (anchor_vids/anchor_vpos) and mesh binding (name_to_idx/bind_trf_inv/skinning)
            always use the fixed reference character (ref_geo, train_mesh_char_list[0]).
            (same convention as the anchor loss using anchor_vids_src = anchor_vids[0])
          - the number of anchors differs per joint (looked up directly with perjoint_vids), so they are
            padded to 5 slots into a (22, 5, 3) tensor. Missing slots are filled with 0 and their
            validity is recorded in self.anchor_mask (22, 5).
          - same scheme as root_a: "the partner's anchor positions expressed in my root-local frame".
            That is, the input of A holds the anchor positions of B, and the input of B those of A.
        """
        from Network.network import set_pose_by_source_batch_frame, get_positions_from_vids
        from pymovis.motion.ops.torchmotion import R6_to_R

        device = self.args.device
        num_window, window_size, _ = self.input_motion0.shape

        anchor_vids  = ref_geo['anchor_vids'].to(device)   # (V,)
        anchor_vpos  = ref_geo['anchor_vpos'].to(device)   # (V,3)
        len_vids     = anchor_vids.shape[0]

        batch = torch.arange(num_window).reshape(num_window, 1, 1).repeat(1, window_size, len_vids).to(device)
        frame = torch.arange(window_size).reshape(1, window_size, 1).repeat(num_window, 1, len_vids).to(device)
        anchor_vids_b = anchor_vids.reshape(1, 1, len_vids).repeat(num_window, window_size, 1)
        anchor_vpos_b = anchor_vpos.reshape(1, 1, len_vids, 3).repeat(num_window, window_size, 1, 1)

        # rotations/root of the actual recorded source motion (absolute values before factor_out)
        R0 = R6_to_R(self.input_motion0[..., :-3].reshape(num_window, window_size, 22, 6))
        R1 = R6_to_R(self.input_motion1[..., :-3].reshape(num_window, window_size, 22, 6))
        root_p0 = self.input_motion0[..., -3:]
        root_p1 = self.input_motion1[..., -3:]

        mesh_global_R0 = set_pose_by_source_batch_frame(
            ref_geo['name_to_idx'], ref_geo['bind_trf_inv'].to(device),
            self.source_offsets0, ref_geo['parents'], ref_geo['names'], R0, root_p0)
        mesh_global_R1 = set_pose_by_source_batch_frame(
            ref_geo['name_to_idx'], ref_geo['bind_trf_inv'].to(device),
            self.source_offsets1, ref_geo['parents'], ref_geo['names'], R1, root_p1)

        anchor_pos0 = get_positions_from_vids(
            ref_geo['skinning_indices1'].to(device), ref_geo['skinning_weights1'].to(device),
            ref_geo['skinning_indices2'].to(device), ref_geo['skinning_weights2'].to(device),
            mesh_global_R0, anchor_vpos_b, anchor_vids_b, batch, frame)   # (num_window, window_size, V, 3)
        anchor_pos1 = get_positions_from_vids(
            ref_geo['skinning_indices1'].to(device), ref_geo['skinning_weights1'].to(device),
            ref_geo['skinning_indices2'].to(device), ref_geo['skinning_weights2'].to(device),
            mesh_global_R1, anchor_vpos_b, anchor_vids_b, batch, frame)

        # the number of anchors per joint differs by character/joint (not exactly 4 or 5),
        # so perjoint_vids of the Geometry (the actual vertex ids per joint) is used to look up
        # which joint each of anchor_vids belongs to, and they are grouped accordingly.
        perjoint_vids = ref_geo['perjoint_vids']  # list[num_joints] of vertex id list
        vid_to_jid = {}
        for j, vids in enumerate(perjoint_vids):
            for vid in vids:
                vid_to_jid[int(vid)] = j

        anchor_vids_list = anchor_vids.cpu().tolist()
        joint_to_slots = [[] for _ in range(22)]  # per joint, the indices of its anchors within anchor_vids
        for idx, vid in enumerate(anchor_vids_list):
            j = vid_to_jid.get(int(vid))
            assert j is not None, f"anchor vertex {vid} does not belong to any joint (check perjoint_vids)"
            joint_to_slots[j].append(idx)

        # always padded to 5 slots to match the fixed anchor_dim=5*3 of the network (Network.__init__).
        num_slots = 5
        max_per_joint = max(len(slots) for slots in joint_to_slots)
        assert max_per_joint <= num_slots, \
            f"the number of anchors per joint exceeds {num_slots} ({max_per_joint}) - " \
            f"increase anchor_dim in Network.__init__ and num_slots in this function together"

        padded0 = torch.zeros(num_window, window_size, 22, num_slots, 3, device=device)
        padded1 = torch.zeros(num_window, window_size, 22, num_slots, 3, device=device)
        mask = torch.zeros(22, num_slots, device=device)
        for j, slots in enumerate(joint_to_slots):
            c = len(slots)
            if c == 0:
                continue
            idx_t = torch.tensor(slots, device=device)
            padded0[:, :, j, :c] = anchor_pos0[:, :, idx_t]
            padded1[:, :, j, :c] = anchor_pos1[:, :, idx_t]
            mask[j, :c] = 1

        # root_a scheme: the partner's anchors expressed in my root-local frame
        inv_root_R0 = torch.linalg.inv(R0[:, :, 0]).reshape(num_window, window_size, 1, 1, 3, 3)
        inv_root_R1 = torch.linalg.inv(R1[:, :, 0]).reshape(num_window, window_size, 1, 1, 3, 3)
        root_p0_ = root_p0.reshape(num_window, window_size, 1, 1, 3)
        root_p1_ = root_p1.reshape(num_window, window_size, 1, 1, 3)

        rel_anchor0 = torch.matmul(inv_root_R0, (padded1 - root_p0_).unsqueeze(-1)).squeeze(-1)  # input of A: anchors of B
        rel_anchor1 = torch.matmul(inv_root_R1, (padded0 - root_p1_).unsqueeze(-1)).squeeze(-1)  # input of B: anchors of A

        mask_ = mask.reshape(1, 1, 22, num_slots, 1)
        rel_anchor0 = (rel_anchor0 * mask_).reshape(num_window, window_size, 22, num_slots * 3)
        rel_anchor1 = (rel_anchor1 * mask_).reshape(num_window, window_size, 22, num_slots * 3)

        self.input_anchor0 = rel_anchor0
        self.input_anchor1 = rel_anchor1
        self.anchor_mask = mask

    def get_input_relative_pos(self):
        """
        Express the partner's global joint positions in my root-local frame and store them in
        self.input_relpos0/1 (num_window, window_size, 22, 3).

        pos_a_t (= self.input_pos0/1) is in "its own" canonical frame, with the root at 0 and the direction
        fixed forward, so it carries no partner information at all. With the same scheme as root_a, this function
        computes "where the partner's actual joint positions are in my root-local frame".
        Unlike the anchors, no skinning is needed, so it is cheap.
        """
        from pymovis.motion.ops.torchmotion import R6_to_R

        num_window, window_size, _ = self.input_motion0.shape
        device = self.args.device

        R0 = R6_to_R(self.input_motion0[..., :-3].reshape(num_window, window_size, 22, 6))
        R1 = R6_to_R(self.input_motion1[..., :-3].reshape(num_window, window_size, 22, 6))
        root_p0 = self.input_motion0[..., -3:]
        root_p1 = self.input_motion1[..., -3:]

        _, global_p0 = R_fk_from_given_info(R0, root_p0, self.source_offsets0.to(device), self.parent_idx0)
        _, global_p1 = R_fk_from_given_info(R1, root_p1, self.source_offsets1.to(device), self.parent_idx1)

        inv_root_R0 = torch.linalg.inv(R0[:, :, 0]).reshape(num_window, window_size, 1, 3, 3)
        inv_root_R1 = torch.linalg.inv(R1[:, :, 0]).reshape(num_window, window_size, 1, 3, 3)
        root_p0_ = root_p0.reshape(num_window, window_size, 1, 3)
        root_p1_ = root_p1.reshape(num_window, window_size, 1, 3)

        # input of A: joint positions of B in the root-local frame of A
        rel_pos0 = torch.matmul(inv_root_R0, (global_p1 - root_p0_).unsqueeze(-1)).squeeze(-1)
        # input of B: joint positions of A in the root-local frame of B
        rel_pos1 = torch.matmul(inv_root_R1, (global_p0 - root_p1_).unsqueeze(-1)).squeeze(-1)

        self.input_relpos0 = rel_pos0
        self.input_relpos1 = rel_pos1

    def save_norm_info(self):
        input0, input1, input_pos0, input_pos1 = \
            self.input0, self.input1, self.input_pos0, self.input_pos1
            
        # get mean and std
        self.input0_mean = torch.mean(input0, dim=(0,1))[None, None, :]  # 1 J 4
        self.input1_mean = torch.mean(input1, dim=(0,1))[None, None, :]
        self.input0_std  = torch.std(input0,  dim=(0,1))[None, None, :]
        self.input1_std  = torch.std(input1,  dim=(0,1))[None, None, :]
        self.input_pos0_mean = torch.mean(input_pos0, dim=(0,1))[None, None, :]
        self.input_pos1_mean = torch.mean(input_pos1, dim=(0,1))[None, None, :]
        self.input_pos0_std  = torch.std(input_pos0,  dim=(0,1))[None, None, :]
        self.input_pos1_std  = torch.std(input_pos1,  dim=(0,1))[None, None, :]
        
        # remove (std==0)
        zero_ths = 1e-6
        self.input0_std[self.input0_std == 0] = zero_ths
        self.input1_std[self.input1_std == 0] = zero_ths
        self.input_pos0_std[self.input_pos0_std == 0] = zero_ths
        self.input_pos1_std[self.input_pos1_std == 0] = zero_ths
        
        # save 
        if self.args.rotation_rep == 'quat':
            np.save("./normalization_info/input0_mean.npy",self.input0_mean.cpu().numpy())
            np.save("./normalization_info/input0_std.npy", self.input0_std.cpu().numpy())
            np.save("./normalization_info/input1_mean.npy",self.input1_mean.cpu().numpy())
            np.save("./normalization_info/input1_std.npy", self.input1_std.cpu().numpy())
        elif self.args.rotation_rep == 'R6':
            np.save("./normalization_info/R6_input0_mean.npy",self.input0_mean.cpu().numpy())
            np.save("./normalization_info/R6_input0_std.npy", self.input0_std.cpu().numpy())
            np.save("./normalization_info/R6_input1_mean.npy",self.input1_mean.cpu().numpy())
            np.save("./normalization_info/R6_input1_std.npy", self.input1_std.cpu().numpy())
        else:
            raise NotImplementedError
        np.save("./normalization_info/input_pos0_mean.npy",self.input_pos0_mean.cpu().numpy())
        np.save("./normalization_info/input_pos0_std.npy", self.input_pos0_std.cpu().numpy())
        np.save("./normalization_info/input_pos1_mean.npy",self.input_pos1_mean.cpu().numpy())
        np.save("./normalization_info/input_pos1_std.npy", self.input_pos1_std.cpu().numpy())
        
    def load_norm_info(self):
        # rot 
        if self.args.rotation_rep == 'quat':
            self.input0_mean = torch.tensor(np.load("./normalization_info/input0_mean.npy")).to(self.args.device)
            self.input1_mean = torch.tensor(np.load("./normalization_info/input1_mean.npy")).to(self.args.device)
            self.input0_std = torch.tensor(np.load("./normalization_info/input0_std.npy")).to(self.args.device)
            self.input1_std = torch.tensor(np.load("./normalization_info/input1_std.npy")).to(self.args.device)
        elif self.args.rotation_rep == 'R6':
            self.input0_mean = torch.tensor(np.load("./normalization_info/R6_input0_mean.npy")).to(self.args.device)
            self.input1_mean = torch.tensor(np.load("./normalization_info/R6_input1_mean.npy")).to(self.args.device)
            self.input0_std  = torch.tensor(np.load("./normalization_info/R6_input0_std.npy")).to(self.args.device)
            self.input1_std  = torch.tensor(np.load("./normalization_info/R6_input1_std.npy")).to(self.args.device)
        # pos
        self.input_pos0_mean = torch.tensor(np.load("./normalization_info/input_pos0_mean.npy")).to(self.args.device)
        self.input_pos1_mean = torch.tensor(np.load("./normalization_info/input_pos1_mean.npy")).to(self.args.device)
        self.input_pos0_std = torch.tensor(np.load("./normalization_info/input_pos0_std.npy")).to(self.args.device)
        self.input_pos1_std = torch.tensor(np.load("./normalization_info/input_pos1_std.npy")).to(self.args.device)
        
    def normalize(self):
        self.input0 = (self.input0 - self.input0_mean) / self.input0_std
        self.input1 = (self.input1 - self.input1_mean) / self.input1_std
        self.input_pos0 = (self.input_pos0 - self.input_pos0_mean) / self.input_pos0_std
        self.input_pos1 = (self.input_pos1 - self.input_pos1_mean) / self.input_pos1_std
        
    def get_gt_motion(self):
        from Retarget_SMPL.retarget_smpl import load_full_edited_npy_motion
        from pymovis.motion.ops.torchmotion import R6_to_R
        trainset_bvh_keys = list(trainset_bvh.keys())
        
        # motion
        target_character = self.target_character1
        gt_c0 = []
        gt_c1 = []
        # window (role, character, scale, num_motion, window_size, pose)
        for motion_name in trainset_bvh_keys:
            motion0, motion1 = load_full_edited_npy_motion(self.args, target_character, motion_name)
            gt_c0_ = []
            gt_c1_ = []
            for sid in range(2):
                motion0_ = cut_by_window(self.args, motion0[sid])
                motion1_ = cut_by_window(self.args, motion1[sid])
                if motion0_ is None or motion1_ is None:
                    continue
                gt_c0_.append(motion0_)
                gt_c1_.append(motion1_)
            gt_c0.append(torch.stack(gt_c0_, dim=0))
            gt_c1.append(torch.stack(gt_c1_, dim=0))
        self.retargeted_gt0 = torch.cat(gt_c0, dim=2)
        self.retargeted_gt1 = torch.cat(gt_c1, dim=2)

        # check error 
        num_window = self.num_window
        window_size = self.window_size
        num_role, num_scale, num_window_, window_size_, _ = self.retargeted_gt0.shape
        if num_window != num_window_ or window_size != window_size_:
            raise ValueError("num_window or window_size is not same")
        
        # calculate pos
        target0_skeleton = self.target_skeleton0
        target1_skeleton = self.target_skeleton1
        parents0 = target0_skeleton.parent_idx
        parents1 = target1_skeleton.parent_idx
        gt_pos0 = []
        gt_pos1 = []
        for rid in range(num_role):
            gt_pos0_ = []
            gt_pos1_ = []
            for sid in range(num_scale):
                # data 
                gt0 = self.retargeted_gt0[rid, sid]
                gt1 = self.retargeted_gt1[rid, sid] 
                # rot 
                gt_R0 = gt0[..., :-3].reshape(num_window, window_size, 22, self.args.rot_dim) 
                gt_R1 = gt1[..., :-3].reshape(num_window, window_size, 22, self.args.rot_dim)
                if self.args.rotation_rep == 'quat':
                    gt_R0 = Q_to_R(gt_R0)
                    gt_R1 = Q_to_R(gt_R1)
                elif self.args.rotation_rep == 'R6':
                    gt_R0 = R6_to_R(gt_R0)
                    gt_R1 = R6_to_R(gt_R1)
                # root 
                gt_root_p0 = gt0[..., -3:] 
                gt_root_p1 = gt1[..., -3:] 
                # position
                _, gt_pos0_value = R_fk_from_given_info(gt_R0, gt_root_p0, self.target_offsets0[rid, sid], parents0)
                _, gt_pos1_value = R_fk_from_given_info(gt_R1, gt_root_p1, self.target_offsets1[rid, sid], parents1)
                gt_pos0_.append(gt_pos0_value)
                gt_pos1_.append(gt_pos1_value)
            gt_pos0.append(torch.stack(gt_pos0_, dim=0))
            gt_pos1.append(torch.stack(gt_pos1_, dim=0))
        self.gt_pos0 = torch.stack(gt_pos0, dim=0)
        self.gt_pos1 = torch.stack(gt_pos1, dim=0)
        
        # print 
        num_role, num_scale, num_window, window_size, _ = self.retargeted_gt0.shape
        num_frames = num_scale*num_window*window_size
        sec = round(num_frames/self.fps, 3)
        minutes = round(sec/60, 3)
        print("Augmeneted: {} frames ({}s, {}m)".format(num_frames, sec, minutes))
    
    """ load and save """
    def save_char_and_motion_data(self, name1):
        import pickle
        data = {
            # same for input char
            # geo data 
            "name_to_idx": self.name_to_idx, 
            "bind_trf_inv":np.array(self.bind_trf_inv), 
            "anchor_vids": np.array(self.anchor_vids), 
            "anchor_vpos": np.array(self.anchor_vpos), 
            "skinning_indices1": np.array(self.skinning_indices1.to('cpu')), 
            "skinning_weights1": np.array(self.skinning_weights1.to('cpu')), 
            "skinning_indices2": np.array(self.skinning_indices2.to('cpu')), 
            "skinning_weights2": np.array(self.skinning_weights2.to('cpu')), 
            "names": self.names, 
            "parents": np.array(self.parents), 
            
            # input motion 
            "input0": np.array(self.input0.to('cpu')),
            "input1": np.array(self.input1.to('cpu')),
            "input_motion0": np.array(self.input_motion0.to('cpu')),
            "input_motion1": np.array(self.input_motion1.to('cpu')),
            "input_pos0": np.array(self.input_pos0.to('cpu')),
            "input_pos1": np.array(self.input_pos1.to('cpu')),
            "input_anchor0": np.array(self.input_anchor0.to('cpu')),
            "input_anchor1": np.array(self.input_anchor1.to('cpu')),
            "input_relpos0": np.array(self.input_relpos0.to('cpu')),
            "input_relpos1": np.array(self.input_relpos1.to('cpu')),

            "parent_idx0": np.array(self.parent_idx0),
            "parent_idx1": np.array(self.parent_idx1),
            
            # different for input char
            # skel data 
            "source_offsets0": np.array(self.source_offsets0.to('cpu')),
            "source_offsets1": np.array(self.source_offsets1.to('cpu')),
            "target_offsets0": np.array(self.target_offsets0.to('cpu')),
            "target_offsets1": np.array(self.target_offsets1.to('cpu')),
            
            # absolute values
            "source_aabb_max_min0": np.array(self.source_aabb_max_min0.to('cpu')),
            "source_aabb_max_min1": np.array(self.source_aabb_max_min1.to('cpu')),
            "target_aabb_max_min0": np.array(self.target_aabb_max_min0.to('cpu')),
            "target_aabb_max_min1": np.array(self.target_aabb_max_min1.to('cpu')),
            
            # valid 
            "valid_source_offsets0":np.array(self.valid_source_offsets0.to('cpu')),
            "valid_source_offsets1":np.array(self.valid_source_offsets1.to('cpu')),
            "valid_source_aabb_max_min0":np.array(self.valid_source_aabb_max_min0.to('cpu')),
            "valid_source_aabb_max_min1":np.array(self.valid_source_aabb_max_min1.to('cpu')),
            "valid_target_offsets0":np.array(self.valid_target_offsets0.to('cpu')),
            "valid_target_offsets1":np.array(self.valid_target_offsets1.to('cpu')),
            "valid_target_aabb_max_min0":np.array(self.valid_target_aabb_max_min0.to('cpu')),
            "valid_target_aabb_max_min1":np.array(self.valid_target_aabb_max_min1.to('cpu')),
            
            # gt motion data 
            "retargeted_gt0": np.array(self.retargeted_gt0.to('cpu')),
            "retargeted_gt1": np.array(self.retargeted_gt1.to('cpu')),
            "gt_pos0": np.array(self.gt_pos0.to('cpu')),
            "gt_pos1": np.array(self.gt_pos1.to('cpu')),
        }

        # Save motion data
        # if self.args.windowed_motion:
        repre = "motion_windowed_randomScaled" 
        # TODO: motion_windowed_uniformScaled
        pkl_path='./preprocess/{}/{}.pkl'.format(name1, repre)
        with open(pkl_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"{name1} data saved at {pkl_path}")
    
    def load_motion_and_geo_data(self, names1):
        # data shape (characters, roles, scales, motions, frames, joints, 3)
        import pickle
        
        # geo info 
        self.name_to_idx = []
        self.bind_trf_inv = []
        self.anchor_vids = []
        self.anchor_vpos = []
        self.skinning_indices1 = []
        self.skinning_weights1 = []
        self.skinning_indices2 = []
        self.skinning_weights2 = []
        # relative values
        self.source_offsets0 = [] 
        self.source_offsets1 = [] 
        self.target_offsets0 = [] 
        self.target_offsets1 = [] 
        # absolute values
        self.source_aabb_max_min0 = []
        self.source_aabb_max_min1 = []
        self.target_aabb_max_min0 = []
        self.target_aabb_max_min1 = []
        # valid
        self.valid_source_offsets0 = []
        self.valid_source_offsets1 = []
        self.valid_source_aabb_max_min0 = []
        self.valid_source_aabb_max_min1 = []
        
        self.valid_target_offsets0 = []
        self.valid_target_offsets1 = []
        self.valid_target_aabb_max_min0 = []
        self.valid_target_aabb_max_min1 = []
        
        # Gt 
        self.retargeted_gt0 = []
        self.retargeted_gt1 = []
        self.gt_pos0 = []
        self.gt_pos1 = []
        
        repre = "motion_windowed_randomScaled" 
        # TODO: motion_windowed_uniformScaled
        for name1 in names1:
            with open('./preprocess/{}/{}.pkl'.format(name1, repre), 'rb') as f:
                pkl = pickle.load(f)
                print("> Data load: {}, {}".format(name1, repre))
            device = self.args.device

            # geo data 
            self.name_to_idx.append(pkl["name_to_idx"])
            self.bind_trf_inv     .append(torch.from_numpy(pkl["bind_trf_inv"]).to(device))
            self.anchor_vids      .append(torch.from_numpy(pkl["anchor_vids"]).to(device))
            self.anchor_vpos      .append(torch.from_numpy(pkl["anchor_vpos"]).to(device))
            self.skinning_indices1.append(torch.from_numpy(pkl["skinning_indices1"]).to(device))
            self.skinning_weights1.append(torch.from_numpy(pkl["skinning_weights1"]).to(device))
            self.skinning_indices2.append(torch.from_numpy(pkl["skinning_indices2"]).to(device))
            self.skinning_weights2.append(torch.from_numpy(pkl["skinning_weights2"]).to(device))

            # offset 
            self.source_offsets0.append(torch.from_numpy(pkl["source_offsets0"]).to(device))
            self.source_offsets1.append(torch.from_numpy(pkl["source_offsets1"]).to(device))
            self.target_offsets0.append(torch.from_numpy(pkl["target_offsets0"]).to(device))
            self.target_offsets1.append(torch.from_numpy(pkl["target_offsets1"]).to(device))
            # absolute values
            self.source_aabb_max_min0.append(torch.from_numpy(pkl["source_aabb_max_min0"]).to(device))
            self.source_aabb_max_min1.append(torch.from_numpy(pkl["source_aabb_max_min1"]).to(device))
            self.target_aabb_max_min0.append(torch.from_numpy(pkl["target_aabb_max_min0"]).to(device))
            self.target_aabb_max_min1.append(torch.from_numpy(pkl["target_aabb_max_min1"]).to(device))
            
            # valid 
            self.valid_source_offsets0.append(torch.from_numpy(pkl["valid_source_offsets0"]).to(device))
            self.valid_source_offsets1.append(torch.from_numpy(pkl["valid_source_offsets1"]).to(device))
            self.valid_source_aabb_max_min0.append(torch.from_numpy(pkl["valid_source_aabb_max_min0"]).to(device))
            self.valid_source_aabb_max_min1.append(torch.from_numpy(pkl["valid_source_aabb_max_min1"]).to(device))
            
            self.valid_target_offsets0.append(torch.from_numpy(pkl["valid_target_offsets0"]).to(device))
            self.valid_target_offsets1.append(torch.from_numpy(pkl["valid_target_offsets1"]).to(device))
            self.valid_target_aabb_max_min0.append(torch.from_numpy(pkl["valid_target_aabb_max_min0"]).to(device))
            self.valid_target_aabb_max_min1.append(torch.from_numpy(pkl["valid_target_aabb_max_min1"]).to(device))
            
            self.retargeted_gt0.append(torch.from_numpy(pkl["retargeted_gt0"]).to(device))
            self.retargeted_gt1.append(torch.from_numpy(pkl["retargeted_gt1"]).to(device))
            self.gt_pos0.append(torch.from_numpy(pkl["gt_pos0"]).to(device))
            self.gt_pos1.append(torch.from_numpy(pkl["gt_pos1"]).to(device))
            
            # char common info 
            self.names       = pkl["names"]
            self.parents     = pkl["parents"]
            self.parent_idx0 = pkl["parent_idx0"]
            self.parent_idx1 = pkl["parent_idx1"]
            
            # input motion: same for every character
            self.input0 = torch.from_numpy(pkl["input0"]).to(device)
            self.input1 = torch.from_numpy(pkl["input1"]).to(device)
            self.input_motion0 = torch.from_numpy(pkl["input_motion0"]).to(device)
            self.input_motion1 = torch.from_numpy(pkl["input_motion1"]).to(device)
            self.input_pos0 = torch.from_numpy(pkl["input_pos0"]).to(device)
            self.input_pos1 = torch.from_numpy(pkl["input_pos1"]).to(device)
            if self.args.use_anchor_input:
                self.input_anchor0 = torch.from_numpy(pkl["input_anchor0"]).to(device)
                self.input_anchor1 = torch.from_numpy(pkl["input_anchor1"]).to(device)
            if self.args.use_relative_pos_input:
                self.input_relpos0 = torch.from_numpy(pkl["input_relpos0"]).to(device)
                self.input_relpos1 = torch.from_numpy(pkl["input_relpos1"]).to(device)

        if self.args.data_normalized:
            self.normalize()
        
        # geo 
        self.anchor_vids = torch.stack(self.anchor_vids, dim=0)
        self.anchor_vpos = torch.stack(self.anchor_vpos, dim=0)
        self.skinning_indices1 = torch.stack(self.skinning_indices1, dim=0)
        self.skinning_weights1 = torch.stack(self.skinning_weights1, dim=0)
        self.skinning_indices2 = torch.stack(self.skinning_indices2, dim=0)
        self.skinning_weights2 = torch.stack(self.skinning_weights2, dim=0)
        # relative values
        self.source_offsets0 = torch.stack(self.source_offsets0, dim=0)
        self.source_offsets1 = torch.stack(self.source_offsets1, dim=0)
        self.target_offsets0 = torch.stack(self.target_offsets0, dim=0)
        self.target_offsets1 = torch.stack(self.target_offsets1, dim=0)
        
        # absolute values
        self.source_aabb_max_min0 = torch.stack(self.source_aabb_max_min0, dim=0)
        self.source_aabb_max_min1 = torch.stack(self.source_aabb_max_min1, dim=0)
        self.target_aabb_max_min0 = torch.stack(self.target_aabb_max_min0, dim=0)
        self.target_aabb_max_min1 = torch.stack(self.target_aabb_max_min1, dim=0)
        
        # valid 
        self.valid_source_offsets0 = torch.stack(self.valid_source_offsets0, dim=0)
        self.valid_source_offsets1 = torch.stack(self.valid_source_offsets1, dim=0)
        self.valid_target_offsets0 = torch.stack(self.valid_target_offsets0, dim=0)
        self.valid_target_offsets1 = torch.stack(self.valid_target_offsets1, dim=0)
        self.valid_source_aabb_max_min0 = torch.stack(self.valid_source_aabb_max_min0, dim=0)
        self.valid_source_aabb_max_min1 = torch.stack(self.valid_source_aabb_max_min1, dim=0)
        self.valid_target_aabb_max_min0 = torch.stack(self.valid_target_aabb_max_min0, dim=0)
        self.valid_target_aabb_max_min1 = torch.stack(self.valid_target_aabb_max_min1, dim=0)
        
        self.retargeted_gt0 = torch.stack(self.retargeted_gt0, dim=0)
        self.retargeted_gt1 = torch.stack(self.retargeted_gt1, dim=0)
        self.gt_pos0 = torch.stack(self.gt_pos0, dim=0) # cat dim=1
        self.gt_pos1 = torch.stack(self.gt_pos1, dim=0)
        
    def load_scale(self):
        self.scales = np.load("./scale_values/scales_sampled.npy")
        self.num_scale = len(self.scales) # pow(len(self.scales), 3) 

        # scale 
        # name1 = self.target_character1.meshes[0].mesh_gl.name
        # if name1 == "SMPLx":
        #     self.scales = np.load("scales.npy")
        # elif name1 == "SMPLx_fat": 
        #     self.scales = np.load("scales_fat.npy")
        # else: 
        #     raise ValueError("no scale name1: ", name1)