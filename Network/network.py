from pymovis.motion.ops.torchmotion import R6_to_R, Q_to_R   
from pymovis.motion.ops.torchmotion import R_fk_from_given_info
from torch import optim
from option_parser import *
from Network.define_network import SpatioTemporalTransformer, FeedForward
import torch
import torch.nn as nn
import os

class Network():
    def __init__(self, args):
        self.args = args
        self.num_joints = 22
        
        """ hyper para """
        # pose
        position_dim = 3 
        self.args.motion_dim = self.args.rot_dim + position_dim # 9
        # feature dim 
        self.char_dim = 6 

        """ models """
        # Transformer
        self.spatio_temp_net = SpatioTemporalTransformer(self.args, 
            spat_num_token=self.num_joints + 1, 
            max_temp_num_token=2000,
            char_dim = self.char_dim,
        ).to(args.device)

        # optimizer
        self.nets = [self.spatio_temp_net]
        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(self.spatio_temp_net_para(), args.learning_rate, weight_decay=0.001)
    
    @torch.no_grad()
    def eval(self):
        for net in self.nets:
            net.eval()

    def forward(self, dataset):
        # input
        if True:
            input0 = dataset.input0
            input1 = dataset.input1
            input_motion0 = dataset.input_motion0
            input_motion1 = dataset.input_motion1
            input_pos0 = dataset.input_pos0
            input_pos1 = dataset.input_pos1
            # 상대값 
            source_offsets0 = dataset.source_offsets0
            source_offsets1 = dataset.source_offsets1
            target_offsets0 = dataset.target_offsets0
            target_offsets1 = dataset.target_offsets1
            # 절대값 
            source_aabb_max_min0 = dataset.source_aabb_max_min0
            source_aabb_max_min1 = dataset.source_aabb_max_min1
            target_aabb_max_min0 = dataset.target_aabb_max_min0
            target_aabb_max_min1 = dataset.target_aabb_max_min1

        len_frame = input0.shape[1]
        
        # norm info
        if True:
            # source 
            sour_info0, sour_info1 = \
                self.get_char_length_info(source_offsets0, source_offsets1,
                                source_aabb_max_min0, source_aabb_max_min1)
            # target
            targ_info0, targ_info1 = \
                self.get_char_length_info(target_offsets0, target_offsets1,
                                target_aabb_max_min0, target_aabb_max_min1)
            
        # forward
        # trf
        sour_rest0 = sour_info0.reshape(1, 22, self.char_dim).repeat(len_frame, 1, 1)
        sour_rest1 = sour_info1.reshape(1, 22, self.char_dim).repeat(len_frame, 1, 1)
        targ_rest0 = targ_info0.reshape(1, 22, self.char_dim).repeat(len_frame, 1, 1)
        targ_rest1 = targ_info1.reshape(1, 22, self.char_dim).repeat(len_frame, 1, 1)
        delta0, delta1 = \
            self.spatio_temp_net(input0[..., :-3], input1[..., :-3],
                                 input0[..., -3:], input1[..., -3:],
                                 input_pos0, input_pos1,
                                 sour_rest0, sour_rest1,
                                 targ_rest0, targ_rest1)
        output_motion0 = input_motion0 + delta0
        output_motion1 = input_motion1 + delta1
        
        # separate root and rotation
        out_qs0 = output_motion0[0, :, :-3].reshape(-1, 22, self.args.rot_dim)
        out_qs1 = output_motion1[0, :, :-3].reshape(-1, 22, self.args.rot_dim)
        out_root_ps0 = output_motion0[0, :, -3:]
        out_root_ps1 = output_motion1[0, :, -3:]
        
        if self.args.rotation_rep == 'quat':
            out_R0 = Q_to_R(out_qs0)
            out_R1 = Q_to_R(out_qs1)
        elif self.args.rotation_rep == 'R6':
            out_R0 = R6_to_R(out_qs0)
            out_R1 = R6_to_R(out_qs1)

        return out_root_ps0, out_R0, \
            out_root_ps1, out_R1

    def train_(self, dataset):
        import wandb
        from datasets.motion_functions import get_displacement_map, get_distance_map
        print("train start")

        if True:
            # input motion
            input0 = dataset.input0
            input1 = dataset.input1
            input_motion0 = dataset.input_motion0
            input_motion1 = dataset.input_motion1
            input_pos0 = dataset.input_pos0
            input_pos1 = dataset.input_pos1
            # gt 
            gt0 = dataset.retargeted_gt0
            gt1 = dataset.retargeted_gt1
            gt_pos0 = dataset.gt_pos0
            gt_pos1 = dataset.gt_pos1
            # 상대값 
            source_offsets0 = dataset.source_offsets0
            source_offsets1 = dataset.source_offsets1
            target_offsets0 = dataset.target_offsets0
            target_offsets1 = dataset.target_offsets1
            # 절대값 
            source_aabb_max_min0 = dataset.source_aabb_max_min0
            source_aabb_max_min1 = dataset.source_aabb_max_min1
            target_aabb_max_min0 = dataset.target_aabb_max_min0
            target_aabb_max_min1 = dataset.target_aabb_max_min1
            # valid 
            valid_source_offsets0 = dataset.valid_source_offsets0 
            valid_source_offsets1 = dataset.valid_source_offsets1
            valid_target_offsets0 = dataset.valid_target_offsets0
            valid_target_offsets1 = dataset.valid_target_offsets1
            valid_source_aabb_max_min0 = dataset.valid_source_aabb_max_min0
            valid_source_aabb_max_min1 = dataset.valid_source_aabb_max_min1
            valid_target_aabb_max_min0 = dataset.valid_target_aabb_max_min0
            valid_target_aabb_max_min1 = dataset.valid_target_aabb_max_min1

            # char, mesh
            parent_idx0 = dataset.parent_idx0
            parent_idx1 = dataset.parent_idx1
            self.name_to_idx = dataset.name_to_idx
            self.bind_trf_inv = dataset.bind_trf_inv
            anchor_vids = dataset.anchor_vids
            anchor_vpos = dataset.anchor_vpos
            self.skinning_indices1 = dataset.skinning_indices1
            self.skinning_weights1 = dataset.skinning_weights1
            self.skinning_indices2 = dataset.skinning_indices2
            self.skinning_weights2 = dataset.skinning_weights2
            self.names = dataset.names
            self.parents = dataset.parents

        if self.args.train_one_chararacter_only:
            gt0 = gt0[:, :1]
            gt1 = gt1[:, :1]
            gt_pos0 = gt_pos0[:, :1]
            gt_pos1 = gt_pos1[:, :1]

        # info
        num_motion, len_frame, _ = input0.shape
        num_joint = self.args.num_joint
        
        num_char, num_role, num_scale, _, _, _ = gt0.shape
        num_batch = num_motion // self.args.batch_size + 1
        len_vids = anchor_vids.shape[1]
        
        # rest info
        if True:
            # source
            sour_info0, sour_info1 = \
                self.get_char_length_info(source_offsets0, source_offsets1,
                                            source_aabb_max_min0, source_aabb_max_min1)
            sour_info0 = sour_info0.reshape(num_char, 1, num_joint, self.char_dim).repeat(1, num_scale, 1, 1)
            sour_info1 = sour_info1.reshape(num_char, 1, num_joint, self.char_dim).repeat(1, num_scale, 1, 1)
            # target
            targ_info0, targ_info1 = \
                self.get_char_length_info(target_offsets0, target_offsets1,
                                          target_aabb_max_min0, target_aabb_max_min1)
            # valid 
            # source 
            valid_sour_info0, valid_sour_info1 = \
                self.get_char_length_info(valid_source_offsets0, valid_source_offsets1,
                                          valid_source_aabb_max_min0, valid_source_aabb_max_min1)
            # target
            valid_targ_info0, valid_targ_info1 = \
                self.get_char_length_info(valid_target_offsets0, valid_target_offsets1,
                                          valid_target_aabb_max_min0, valid_target_aabb_max_min1)
            # validation index 
            valid_rid = 0
            valid_cid = 0
            valid_sid0 = 0 # "Ybot"
            valid_sid1 = 0 # "Amy"
            valid_gt0 = gt0[valid_rid, valid_cid, valid_sid0]
            valid_gt1 = gt1[valid_rid, valid_cid, valid_sid1]
            valid_sour_rest0 = valid_sour_info0[:valid_cid+1].reshape(1,1,-1).repeat(1,len_frame,1)
            valid_sour_rest1 = valid_sour_info1[:valid_cid+1].reshape(1,1,-1).repeat(1,len_frame,1)
            valid_targ_rest0 = valid_targ_info0[:valid_cid+1].reshape(1,1,-1).repeat(1,len_frame,1)
            valid_targ_rest1 = valid_targ_info1[:valid_cid+1].reshape(1,1,-1).repeat(1,len_frame,1)
        
        # source anchor in Tpose (normal SMPLx)
        anchor_vids_src = anchor_vids[0]
        anchor_vpos_src = anchor_vpos[0]
        
        # foot contact detection
        from detect_foot_contact import detect_foot_contact_from_batched_position
        foot_contact_label0 = detect_foot_contact_from_batched_position(self.args, input_pos0)
        foot_contact_label1 = detect_foot_contact_from_batched_position(self.args, input_pos1)
        
        # source TODO 
        # anchor position
        # source_anchor_positions0 = \
        #     self.get_anchor_position(cid,
        #                                 source_offsets0[0], source_R0, source_root_p0, 
        #                                 anchor_vpos_src, anchor_vids_src, batch, frame)
        # source_anchor_positions1 = \
        #     self.get_anchor_position(cid,
        #                                 source_offset1, source_R1, source_root_p1,
        #                                 anchor_vpos_src_b, anchor_vids_src_b, batch, frame)
        # distance map
        # source_anchor_map0 = get_distance_map(source_anchor_positions0, source_anchor_positions1)
        # source_anchor_map1 = get_distance_map(source_anchor_positions1, source_anchor_positions0)
        
        # load 
        if self.args.begin_epoch != 0:
            self.load(self.args.path, self.args.begin_epoch)

        # train
        for epoch in range(self.args.begin_epoch, self.args.end_epoch):
            # loss record 
            sum_rec_loss0 = 0
            sum_rec_loss1 = 0
            sum_root_loss0 = 0
            sum_root_loss1 = 0
            sum_fk_loss0 = 0
            sum_fk_loss1 = 0
            sum_foot_contact_loss0 = 0
            sum_foot_contact_loss1 = 0
            
            sum_smooth_loss0 = 0
            sum_smooth_loss1 = 0
            sum_anchor_disp_loss0 = 0
            sum_anchor_disp_loss1 = 0
            sum_skel_disp_loss0 = 0
            sum_skel_disp_loss1 = 0
            sum_reg_loss = 0

            # train
            for cid in range(num_char):
                for rid in range(num_role):
                    # parter id 
                    pid = 1 - rid
                    if self.args.loss_anchor:
                        # rid==0: dfm char1, rid==1: dfm char0
                        anchor_vids0 = anchor_vids[rid]
                        anchor_vids1 = anchor_vids[pid]
                        anchor_vpos0_Tpose = anchor_vpos[rid]
                        anchor_vpos1_Tpose = anchor_vpos[pid]
                    
                    for sid in range(num_scale):
                        for bid in range(num_batch):
                            start = bid * self.args.batch_size
                            end = min((bid + 1) * self.args.batch_size, num_motion)
                            b_size = end - start
                            
                            batch = torch.arange(b_size).reshape(b_size, 1, 1).repeat(1, len_frame, len_vids).to(self.args.device)
                            frame = torch.arange(len_frame).reshape(1, len_frame, 1).repeat(b_size, 1, len_vids).to(self.args.device)

                            """ input motion """
                            # gt
                            gt_b0 = gt0[cid, rid, sid, start:end]
                            gt_b1 = gt1[cid, rid, sid, start:end]
                            gt_pos_b0 = gt_pos0[cid, rid, sid, start:end]
                            gt_pos_b1 = gt_pos1[cid, rid, sid, start:end]
                            
                            # input 
                            input_motion_b0 = input_motion0[start:end]
                            input_motion_b1 = input_motion1[start:end]
                            input_b0 = input0[start:end]
                            input_b1 = input1[start:end]
                            input_pos_b0 = input_pos0[start:end]
                            input_pos_b1 = input_pos1[start:end]
                            
                            # foot contact 
                            foot_contact_label_b0 = foot_contact_label0[start:end]
                            foot_contact_label_b1 = foot_contact_label1[start:end]
                            
                            
                            """ feed """
                            # trf
                            sour_rest0 = sour_info0[cid, sid].repeat(b_size*len_frame, 1, 1)
                            sour_rest1 = sour_info1[cid, sid].repeat(b_size*len_frame, 1, 1)
                            targ_rest0 = targ_info0[cid, rid, sid].repeat(b_size*len_frame, 1, 1)
                            targ_rest1 = targ_info1[cid, rid, sid].repeat(b_size*len_frame, 1, 1)
                            sour_rest0 = sour_rest0.reshape(b_size, len_frame, -1)
                            sour_rest1 = sour_rest1.reshape(b_size, len_frame, -1)
                            targ_rest0 = targ_rest0.reshape(b_size, len_frame, -1)
                            targ_rest1 = targ_rest1.reshape(b_size, len_frame, -1)
                            
                            # forward 
                            delta0, delta1 = \
                                self.spatio_temp_net(input_b0[..., :-3], input_b1[..., :-3],
                                                    input_b0[..., -3:], input_b1[..., -3:],
                                                    input_pos_b0, input_pos_b1,
                                                    sour_rest0, sour_rest1,
                                                    targ_rest0, targ_rest1)
                            output_motion0 = input_motion_b0 + delta0
                            output_motion1 = input_motion_b1 + delta1


                            """ output """
                            # rec 
                            out_R0 = output_motion0[..., :-3]
                            out_R1 = output_motion1[..., :-3]
                            gt_R0 = gt_b0[..., :-3]
                            gt_R1 = gt_b1[..., :-3]
                            # root
                            root_p0 = output_motion0[..., -3:]
                            root_p1 = output_motion1[..., -3:]
                            gt_root_p0 = gt_b0[..., -3:]
                            gt_root_p1 = gt_b1[..., -3:]
                            # offset 
                            tar_offset0 = target_offsets0[cid, rid, sid].reshape(22, 3)
                            tar_offset1 = target_offsets1[cid, rid, sid].reshape(22, 3)


                            """ loss"""
                            # base loss
                            rec_loss0 = self.loss_func(out_R0, gt_R0)
                            rec_loss1 = self.loss_func(out_R1, gt_R1)
                            root_loss0 = self.loss_func(root_p0, gt_root_p0)
                            root_loss1 = self.loss_func(root_p1, gt_root_p1)
                            loss0 = self.args.lambda_rec*rec_loss0 + self.args.lambda_root*root_loss0
                            loss1 = self.args.lambda_rec*rec_loss1 + self.args.lambda_root*root_loss1
                            sum_rec_loss0 += rec_loss0.item()
                            sum_rec_loss1 += rec_loss1.item()
                            sum_root_loss0 += root_loss0.item()
                            sum_root_loss1 += root_loss1.item()
                            
                            # rot reshape
                            out_R0 = out_R0.reshape(b_size, len_frame, 22, 6)
                            out_R1 = out_R1.reshape(b_size, len_frame, 22, 6)
                            if self.args.rotation_rep == 'quat':
                                out_R0 = Q_to_R(out_R0)
                                out_R1 = Q_to_R(out_R1)
                            elif self.args.rotation_rep == 'R6':
                                out_R0 = R6_to_R(out_R0)
                                out_R1 = R6_to_R(out_R1)
                            else:
                                raise ValueError('Invalid rotation representation')
                            
                            # fk loss
                            _, out_pos0 = R_fk_from_given_info(out_R0, output_motion0[..., -3:], tar_offset0, parent_idx0)
                            _, out_pos1 = R_fk_from_given_info(out_R1, output_motion1[..., -3:], tar_offset1, parent_idx1)
                            if self.args.loss_fk:
                                fk_loss0 = self.loss_func(out_pos0, gt_pos_b0)
                                fk_loss1 = self.loss_func(out_pos1, gt_pos_b1)
                                loss0 += self.args.lambda_fk * fk_loss0
                                loss1 += self.args.lambda_fk * fk_loss1
                                sum_fk_loss0 += fk_loss0.item()
                                sum_fk_loss1 += fk_loss1.item()
                            
                            # foot contact loss
                            if self.args.loss_foot_contact:
                                foot_contact_loss0 = self.loss_func(out_pos0[foot_contact_label_b0][:, 1], gt_pos_b0[foot_contact_label_b0][:, 1])                             
                                foot_contact_loss1 = self.loss_func(out_pos1[foot_contact_label_b1][:, 1], gt_pos_b1[foot_contact_label_b1][:, 1])
                                loss0 += self.args.lambda_foot_contact * foot_contact_loss0
                                loss1 += self.args.lambda_foot_contact * foot_contact_loss1
                                sum_foot_contact_loss0 += foot_contact_loss0.item()
                                sum_foot_contact_loss1 += foot_contact_loss1.item()
                            
                            # anchor loss 
                            if self.args.loss_anchor:
                                # vid 
                                anchor_vids_src_b = anchor_vids_src.reshape(1, 1, len_vids).repeat(b_size, len_frame, 1).to(self.args.device)
                                anchor_vids0_b = anchor_vids0.reshape(1, 1, len_vids).repeat(b_size, len_frame, 1).to(self.args.device)
                                anchor_vids1_b = anchor_vids1.reshape(1, 1, len_vids).repeat(b_size, len_frame, 1).to(self.args.device)
                                # vpos
                                anchor_vpos_src_b = anchor_vpos_src.reshape(1, 1, len_vids, 3).repeat(b_size, len_frame, 1, 1).to(self.args.device)
                                anchor_vpos0_Tpose_b = anchor_vpos0_Tpose.reshape(1, 1, len_vids, 3).repeat(b_size, len_frame, 1, 1).to(self.args.device)
                                anchor_vpos1_Tpose_b = anchor_vpos1_Tpose.reshape(1, 1, len_vids, 3).repeat(b_size, len_frame, 1, 1).to(self.args.device)
                                
                                # source 
                                # offset 
                                source_offset0 = source_offsets0[cid]
                                source_offset1 = source_offsets1[cid]
                                # root 
                                source_root_p0 = gt_root_p0
                                source_root_p1 = gt_root_p1
                                # rot reshape 
                                gt_R0 = gt_R0.reshape(b_size, len_frame, 22, 6)
                                gt_R1 = gt_R1.reshape(b_size, len_frame, 22, 6)
                                if self.args.rotation_rep == 'quat':
                                    source_R0 = Q_to_R(gt_R0)
                                    source_R1 = Q_to_R(gt_R1)
                                    # out_R0
                                elif self.args.rotation_rep == 'R6':
                                    source_R0 = R6_to_R(gt_R0)
                                    source_R1 = R6_to_R(gt_R1)
                                
                                # anchor position
                                # source 
                                # source_anchor_positions0 = \
                                #     self.get_anchor_position(cid,
                                #                              source_offset0, source_R0, source_root_p0, 
                                #                              anchor_vpos_src_b, anchor_vids_src_b, batch, frame)
                                # source_anchor_positions1 = \
                                #     self.get_anchor_position(cid,
                                #                              source_offset1, source_R1, source_root_p1,
                                #                              anchor_vpos_src_b, anchor_vids_src_b, batch, frame)

                                # output
                                out_anchor_positions0 = \
                                    self.get_anchor_position(cid,
                                                             tar_offset0, out_R0, root_p0, 
                                                             anchor_vpos0_Tpose_b, anchor_vids0_b, batch, frame)
                                out_anchor_positions1 = \
                                    self.get_anchor_position(cid,
                                                             tar_offset1, out_R1, root_p1,
                                                             anchor_vpos1_Tpose_b, anchor_vids1_b, batch, frame)
                                
                                # distance map
                                # source_anchor_map0 = get_distance_map(source_anchor_positions0, source_anchor_positions1)
                                # source_anchor_map1 = get_distance_map(source_anchor_positions1, source_anchor_positions0)
                                out_anchor_map0 = get_distance_map(out_anchor_positions0, out_anchor_positions1.detach())
                                out_anchor_map1 = get_distance_map(out_anchor_positions1, out_anchor_positions0.detach())
                                # loss
                                anchor_loss0 = self.distance_map_loss(source_anchor_map0, out_anchor_map0)
                                anchor_loss1 = self.distance_map_loss(source_anchor_map1, out_anchor_map1)
                                # add loss 
                                loss0 += self.args.lambda_anchor*anchor_loss0
                                loss1 += self.args.lambda_anchor*anchor_loss1
                                sum_anchor_disp_loss0 += anchor_loss0.item()
                                sum_anchor_disp_loss1 += anchor_loss1.item()
                            
                            
                            """ backward """
                            loss = loss0 + loss1
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()
                            # batch end 
                        # scale end 
                    # role end 
                # character end 
            
            
            """ validation loss """
            delta0, delta1 = \
                self.spatio_temp_net(input0[:1, ..., :-3], input1[:1, ..., :-3],
                                     input0[:1, ..., -3:], input1[:1, ..., -3:],
                                     input_pos0[:1], input_pos1[:1],
                                     valid_sour_rest0, valid_sour_rest1,
                                     valid_targ_rest0, valid_targ_rest1)
            output_motion0 = input_motion0[:1] + delta0
            output_motion1 = input_motion1[:1] + delta1
            valid_loss0 = self.loss_func(output_motion0, valid_gt0[:1])
            valid_loss1 = self.loss_func(output_motion1, valid_gt1[:1])
            
            
            """ log """
            wandb.log({
                "epoch": epoch, 
                "rec_loss0": sum_rec_loss0, 
                "rec_loss1": sum_rec_loss1, 
                "root_loss0": sum_root_loss0, 
                "root_loss1": sum_root_loss1, 
                "fk_loss0": sum_fk_loss0, 
                "fk_loss1": sum_fk_loss1, 
                "foot_contact_loss0": sum_foot_contact_loss0,
                "foot_contact_loss1": sum_foot_contact_loss1,
                
                "disp_loss0": sum_anchor_disp_loss0,
                "disp_loss1": sum_anchor_disp_loss1,
                "valid_loss0": valid_loss0,
                "valid_loss1": valid_loss1,
                }
            )
            # end epoch
            if epoch != self.args.begin_epoch and (epoch == self.args.end_epoch - 1 or epoch % self.args.save_iter_epoch == 0):
                self.save(self.args.path, epoch)
                print("save model at epoch {}".format(epoch))
        wandb.finish()

    def get_anchor_position(self, cid, offset, gt_R, gt_root_p, 
                            anchor_vpos_b, anchor_vids_b, batch, frame):
        gt_global_R = set_pose_by_source_batch_frame(self.name_to_idx[cid], self.bind_trf_inv[cid], offset, self.parents, self.names, gt_R, gt_root_p)
        gt_anchor_positions = get_positions_from_vids(self.skinning_indices1[cid], self.skinning_weights1[cid], self.skinning_indices2[cid], self.skinning_weights2[cid],
                gt_global_R, anchor_vpos_b, anchor_vids_b, batch, frame)
    
        return gt_anchor_positions
    
    def displacement_map_loss(self, input_displacement_map, output_displacement_map):  # input: gt
        # exp weight (input_dist가 작아질수록 exp_weight는 커짐)
        input_dist_map = torch.norm(input_displacement_map, dim=-1)
        exp_weight = self.exp_weight_of_distance(input_dist_map) / input_dist_map
        
        # 예외처리
        zero_idx = torch.where(input_dist_map == 0)
        if zero_idx[0].shape[0] != 0:
            exp_weight[zero_idx] = 0
            
        # displacement_map (가까울수록 큰 loss (input_dist가 작아질수록 diff 값은 커짐))
        displacement_diff_map = torch.abs(input_displacement_map - output_displacement_map)
        displacement_diff_map = torch.norm(displacement_diff_map, dim=-1) 

        # loss
        disp_error = exp_weight * displacement_diff_map
        disp_error = self.loss_func(disp_error, torch.zeros_like(disp_error))

        return disp_error

    def distance_map_loss(self, input_dist_map, output_dist_map):  # input: gt
        # exp weight (input_dist가 작아질수록 exp_weight는 커짐)
        exp_weight = self.exp_weight_of_distance(input_dist_map) / input_dist_map
        
        # 예외처리
        zero_idx = torch.where(input_dist_map == 0)
        if zero_idx[0].shape[0] != 0:
            exp_weight[zero_idx] = 0
            
        # displacement_map (가까울수록 큰 loss  (input_dist가 작아질수록 diff 값은 커짐))
        displacement_diff_map = torch.abs(input_dist_map - output_dist_map)

        # loss
        disp_error = exp_weight * displacement_diff_map
        disp_error = self.loss_func(disp_error, torch.zeros_like(disp_error))

        return disp_error
    
    def exp_weight_of_distance(self, input_dist_map):
        k = self.args.anchor_exp_k
        exp_dist = torch.exp(-k * input_dist_map)
        return exp_dist
    
    """ functions in training """
    def save(self, path, epoch):
        path = "./saved/" + path
        os.makedirs(path, exist_ok=True)
        torch.save(self.spatio_temp_net.state_dict(), path +'spatio_temp_net_{}.pt'.format(epoch))

    def load(self, path, epoch, device='cuda'):
        path = "./saved/" + path
        self.spatio_temp_net.load_state_dict(torch.load(path + 'spatio_temp_net_{}.pt'.format(epoch), map_location=torch.device(device)))
        
    def spatio_temp_net_para(self):
        return list(self.spatio_temp_net.parameters())

    def get_char_length_info(self, 
                      source_offsets0, source_offsets1, 
                      source_aabb_max_min0, source_aabb_max_min1):
        sour_offset_length0 = source_offsets0 # torch.norm(source_offsets0, dim=-1).unsqueeze(-1)
        sour_offset_length1 = source_offsets1 # torch.norm(source_offsets1, dim=-1).unsqueeze(-1)
        sour_aabb_length0 = (source_aabb_max_min0[..., :3] - source_aabb_max_min0[..., 3:])
        sour_aabb_length1 = (source_aabb_max_min1[..., :3] - source_aabb_max_min1[..., 3:])
        sour_info0 = torch.cat((sour_offset_length0, sour_aabb_length0), dim=-1) # sour_offset_length0 #
        sour_info1 = torch.cat((sour_offset_length1, sour_aabb_length1), dim=-1) # sour_offset_length1 #
        return sour_info0, sour_info1
    
    def normalize_char_info(self, 
                            sour_info0, sour_info1, 
                            char_info_mean, char_info_var):
        sour_info0 -= char_info_mean
        sour_info1 -= char_info_mean
        sour_info0 /= char_info_var
        sour_info1 /= char_info_var
        
        return sour_info0, sour_info1

""" geometry class """
def set_pose_by_source_batch_frame(name_to_idx, mesh_bind_trf_inv, 
                                   bone_offset, parents, names, 
                                   local_R, root_p):
    # target
    num_batch, num_frame, _, _, _ = local_R.shape
    mesh_global_R = torch.zeros(num_batch, num_frame, 22, 4, 4).to(local_R.device)

    global_R, global_p = R_fk_from_given(local_R, root_p, bone_offset, parents)
    for i in range(22):
        # world trf 
        world_trf = torch.cat([global_R[..., i, :, :], global_p[..., i, :, None]], axis=-1)
        world_trf = torch.cat([world_trf, torch.tensor([[[0, 0, 0, 1]]]).repeat(num_batch, num_frame, 1, 1).to(local_R.device)], axis=-2)

        # (i -> mesh jid) for binding
        joint_name = names[i]
        if joint_name not in name_to_idx.keys():
            joint_name_ = 'mixamorig:'+joint_name 
            if joint_name_ not in name_to_idx.keys():
                continue
        else:
            joint_name_ = joint_name
        mesh_jid = name_to_idx[joint_name_]

        # bind_trf_inv: Joint의 global trf (Joint마다 고유한 값이기 때문에 1번만 적용)
        bind_trf_inv = mesh_bind_trf_inv[mesh_jid]
        bind_trf_inv = bind_trf_inv.repeat(num_batch, num_frame, 1, 1).to(local_R.device)

        # update
        mesh_global_R[..., i, :, :] += torch.matmul(world_trf, bind_trf_inv.transpose(-2, -1))

    return mesh_global_R

# skinning_joint, skinning_weight
def get_onedim_lbsModel(mesh_global_R, joint, weight, batches, frames):
    from etc.etc import update_index
    lbs_shape = (1,) * (len(joint.shape)) + (4, 4)
    
    # Set -1 weight to 0
    joint = joint.long()
    nega_idx = torch.where(joint == -1)
    if len(nega_idx[0]) == torch.numel(joint): 
        return torch.zeros_like(joint, dtype=torch.float).unsqueeze(-1).unsqueeze(-1).repeat(*lbs_shape).to(joint.device) # lbsModel
    if len(nega_idx[0]) != 0:
        joint = update_index(joint, 0, nega_idx)

    # Rotation matrix of joint
    global_R = mesh_global_R[batches, frames, joint]

    # Skinning weight of cid
    if len(nega_idx[0]) != 0:
        weight = update_index(weight, 0, nega_idx)

    # lbs
    lbsModel = global_R * weight.unsqueeze(-1).unsqueeze(-1)

    return lbsModel

def get_positions_from_vids(joint_ids1, weights1, joint_ids2, weights2, 
                            mesh_global_R, v_positions, vids, batches, frames): # input [shape] -> output [shape, 4]  # (shape should be same)
    # vids = vids.long()
    lbs_shape = (1,) * (len(vids.shape)) + (4, 4)
    lbsModel = torch.zeros_like(vids, dtype=torch.float).unsqueeze(-1).unsqueeze(-1).repeat(*lbs_shape).to(vids.device)
    one_shape = (1,) * (len(vids.shape)) + (1,)
    one_tensor = torch.ones_like(vids, dtype=torch.float).unsqueeze(-1).repeat(*one_shape).to(vids.device)
    
    # lbsModel [shape,4,4]
    lbsModel1 = torch.zeros_like(lbsModel, device=lbsModel.device)
    lbsModel2 = torch.zeros_like(lbsModel, device=lbsModel.device)
    for i in range(4):
        lbsModel1 += get_onedim_lbsModel(mesh_global_R, joint_ids1[..., i], weights1[..., i], batches, frames)
        lbsModel2 += get_onedim_lbsModel(mesh_global_R, joint_ids2[..., i], weights2[..., i], batches, frames)
    lbsModel = lbsModel1 + lbsModel2

    # v pos at Tpose 
    pos = torch.cat((v_positions, one_tensor), dim=-1).unsqueeze(-1)

    # get moved position
    fPosition = torch.matmul(lbsModel, pos).squeeze(-1).to(vids.device)
    
    return fPosition[..., :3]


""" FK """
def R_fk_from_given(local_R, root_p, bone_offsets, parents):
        """
        Args:
            local_R: (..., N, 3, 3)
            root_p: (..., 3)
            bone_offset: (N, 3)
            parents: (N,)
        Returns:
            Global rotation matrix and position of each joint.
        """

        global_R, global_p = [local_R[..., 0, :, :]], [root_p]
        for i in range(1, len(parents)):
            global_R.append(torch.matmul(global_R[parents[i]], local_R[..., i, :, :]))
            global_p.append(torch.matmul(global_R[parents[i]], bone_offsets[i]) + global_p[parents[i]])
        
        global_R = torch.stack(global_R, dim=-3) # (..., N, 3, 3)
        global_p = torch.stack(global_p, dim=-2) # (..., N, 3)
        return global_R, global_p
