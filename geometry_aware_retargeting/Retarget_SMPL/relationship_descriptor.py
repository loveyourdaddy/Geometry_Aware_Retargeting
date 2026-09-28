"""Relationship descriptor (RD): adapt a motion by editing joint positions with respect to the partner's surface."""
import torch

from Retarget_SMPL.motion_utils import (
    get_rootP_localR_globalP_from_motion, get_rootP_localR_globalP_from_numpy_motion,
    rotation_matrix_from_vectors, update_motion_by_global_p, update_pose_by_global_p,
)
from Retarget_SMPL.ground_pene import lift_by_pene_val, parent_recursive_update, resolve_ground_pene

child_of_division = [1, 5, 9, 12, 14, 18]


def retarget_one_motion(args,
                  geo_source_ptn, geo_target_ptn,
                  source_motion0, source_motion1,
                  updated_motion0, updated_motion1,
                  root_joints=None, spine_joints=None, limb_joints=None):
    """Edit updated_motion0 so that the relation between the partner's (motion1) surface descriptors and my joints (motion0) is kept on the target."""
    if args.adapt_char=="SMPLx":
        descriptor_vids = torch.tensor(geo_source_ptn.descriptor_vids).to(args.device)
        tgt_descriptor_vids = descriptor_vids
    else:
        descriptor_vids = geo_source_ptn.anchor_vids
        tgt_descriptor_vids = geo_target_ptn.anchor_vids
        
    len_frame, len_anchor = len(source_motion1.poses), len(descriptor_vids)
    batch = torch.tensor([0]).repeat(len_frame, len_anchor).to(args.device)
    frame = torch.arange(len_frame).unsqueeze(-1).repeat(1, len_anchor).to(args.device)

    """ source descriptor """
    # own (charA, motion0)
    _, _, charA_global_p0 = get_rootP_localR_globalP_from_motion(args, source_motion0.poses)

    # partner (charB, motion1)
    source_root_p1, local_R1, _ = get_rootP_localR_globalP_from_motion(args, source_motion1.poses)
    geo_source_ptn.set_pose_by_source_batch_frame(local_R1.unsqueeze(0), source_root_p1.unsqueeze(0))
    charB_vpos1 = geo_source_ptn.get_positions_from_vids(descriptor_vids.repeat(len_frame, 1), batch, frame)

    desc1_to_joint0 = charA_global_p0[:, :, None, :] - charB_vpos1[:, None, :, :]  # pj - di
    dist_desc1_to_joint0 = torch.norm(desc1_to_joint0, dim=-1)  # ||pj - di||

    """ target descriptor """
    root_p1, local_R1, _ = get_rootP_localR_globalP_from_motion(args, updated_motion1.poses)
    geo_target_ptn.set_pose_by_source_batch_frame(local_R1.unsqueeze(0), root_p1.unsqueeze(0))
    updated_charB_vpos1 = geo_target_ptn.get_positions_from_vids(tgt_descriptor_vids.repeat(len_frame, 1), batch, frame)
    
    """ update motion A """
    _, _, source_global_p0 = get_rootP_localR_globalP_from_motion(args, updated_motion0.poses)

    # root
    if root_joints != []:
        ret_global_p0 = \
            update_by_part(args, tgt_descriptor_vids,
                        desc1_to_joint0, dist_desc1_to_joint0,
                        updated_motion0, source_global_p0,
                        geo_target_ptn, updated_charB_vpos1,
                        update_part="root", update_joints=root_joints,
                        root_r2_dist=1/10, root_pow_lambda=1/10)
    else:
        ret_global_p0 = source_global_p0

    # spine
    if spine_joints != []:
        ret_global_p0 = \
            update_by_part(args, tgt_descriptor_vids,
                            desc1_to_joint0, dist_desc1_to_joint0,
                            updated_motion0, ret_global_p0,
                            geo_target_ptn, updated_charB_vpos1,
                            update_part="spine", update_joints=spine_joints,
                            limb_r2_dist=1/10, limb_pow_lambda=1/10)

    # limb
    ret_global_p0 = \
        update_by_part(args, tgt_descriptor_vids,
                        desc1_to_joint0, dist_desc1_to_joint0,
                        updated_motion0, ret_global_p0,
                        geo_target_ptn, updated_charB_vpos1,
                        update_part="limb", update_joints=limb_joints,
                        limb_r2_dist=1/10, limb_pow_lambda=1/10)

    # ground pene
    # TODO: the thresholds are swapped: heel_joints gets toe_pene_ths and toe_joints gets heel_pene_ths
    ret_global_p0 = lift_by_pene_val(args, ret_global_p0, updated_motion0, args.heel_joints, args.toe_pene_ths)
    ret_global_p0 = lift_by_pene_val(args, ret_global_p0, updated_motion0, args.toe_joints,  args.heel_pene_ths)

    # keep the original motion outside the interaction range
    if args.update_by_clampping_range:
        _, _, original_global_p0 = get_rootP_localR_globalP_from_motion(
            args, updated_motion0.poses)
        ret_global_p0 = clampping_by_interaction_range(
            args, original_global_p0, ret_global_p0)
    
    update_motion_by_global_p(updated_motion0, ret_global_p0)
    
    return updated_motion0

def update_by_part(args, anchor_vids,
                   desc_to_joint, dist_desc_to_joint, # relationship
                   updated_motion0, source_global_p0, # own
                   geo_ptn, updated_charB_vpos1, # ptn
                   update_part="", update_joints=None,
                   root_r2_dist=None, root_pow_lambda=None,
                   limb_r2_dist=None, limb_pow_lambda=None):
    """If update_part is root, translate the whole body; otherwise move only update_joints and keep the bone lengths."""
    if update_part=="root":
        output_global_p0 = \
            relation_descriptor(args,
                                desc_to_joint, dist_desc_to_joint, anchor_vids, updated_charB_vpos1, geo_ptn,
                                r2_dist=root_r2_dist, pow_lambda=root_pow_lambda)
        diff = (output_global_p0 - source_global_p0) # f, 22
        diff = scale_diff_by_dist(diff, torch.min(dist_desc_to_joint, dim=-1)[0])
        diff = diff[:, 0].unsqueeze(1).repeat(1, args.num_joint, 1)
        ret_global_p0 = source_global_p0 + diff[:, 0].unsqueeze(1).repeat(1, args.num_joint, 1)
    else: 
        output_global_p0 = \
            relation_descriptor(args,
                                desc_to_joint, dist_desc_to_joint, anchor_vids, updated_charB_vpos1, geo_ptn,
                                r2_dist=limb_r2_dist, pow_lambda=limb_pow_lambda)

        # scale by distance
        diff = torch.zeros_like(output_global_p0)
        diff[:, update_joints] = (output_global_p0 - source_global_p0)[:, update_joints]
        diff = scale_diff_by_dist(diff, torch.min(dist_desc_to_joint, dim=-1)[0])
        output_global_p0 = source_global_p0 + diff
        
        output_global_p0 = update_absolute_p_by_index(source_global_p0, output_global_p0, update_joints)
        ret_global_p0 = target_position_scaled_by_offset(args, source_global_p0, output_global_p0, updated_motion0.skeleton)

    return ret_global_p0

def relation_descriptor(args,
                        desc_to_joint, dist_desc_to_joint, anchor_vids, updated_charB_vpos1, geo_ptn,
                        r2_dist, pow_lambda):
    """Compute joint positions by blending (target surface position + original descriptor) with distance-based weights."""
    len_frame, len_anchor = len(desc_to_joint), len(anchor_vids)
    batch = torch.tensor([0]).repeat(len_frame, len_anchor).to(args.device)
    frame = torch.arange(len_frame).unsqueeze(-1).repeat(1,len_anchor).to(args.device)

    """ weight """
    # 1. weight_prime
    normal_charB = geo_ptn.get_normal_from_vid(anchor_vids.repeat(len_frame, 1), batch, frame)
    normal_charB = normal_charB / torch.norm(normal_charB, dim=-1).unsqueeze(-1)
    normal_weight_prime = torch.sum(normal_charB.unsqueeze(1).repeat(1, args.num_joint, 1, 1) * desc_to_joint, dim=-1) / dist_desc_to_joint
    normal_weight_prime = torch.abs(normal_weight_prime)

    # 2. weight_twoprime
    # fade func
    fade_func = torch.full_like(dist_desc_to_joint, -1).to(args.device)
    r1 = torch.min(dist_desc_to_joint, dim=-1)[0]
    r2 = r1 + r2_dist * torch.tensor(geo_ptn.height).to(args.device) # own, not ptn

    # close
    cond = (dist_desc_to_joint < r2.unsqueeze(-1))
    fade_func[cond] = 1 - torch.pow(((dist_desc_to_joint[cond] - r1.unsqueeze(-1).repeat(1, 1, len_anchor)[cond])
                                    / (r2.unsqueeze(-1).repeat(1, 1, len_anchor)[cond] - r1.unsqueeze(-1).repeat(1, 1, len_anchor)[cond])), pow_lambda)
    # far
    fade_func[dist_desc_to_joint > r2.unsqueeze(-1)] = 0

    weight_twoprime = fade_func  # normal_weight_prime *

    # 3. final weight
    weight = weight_twoprime / torch.sum(weight_twoprime, dim=-1).unsqueeze(-1)

    """ charB_vpos + desc_to_joint -> charA_joint_p """
    updated_charA_global_p0_ = weight.unsqueeze(-1) * (updated_charB_vpos1[:, None, :, :] + desc_to_joint)
    ret_global_p0 = torch.sum(updated_charA_global_p0_, dim=-2)

    return ret_global_p0

def clampping_by_interaction_range(args, backup_global_p0, ret_global_p0):
    """Update only within the interaction range, blending with the original motion over the margin frames around it."""
    if args.interaction_start_frame != -1 and args.interaction_end_frame != -1:
        update_global_p0 = ret_global_p0.clone()
        global_p0 = backup_global_p0.clone()  # output

        start = args.interaction_start_frame
        end = args.interaction_end_frame
        margin = 20
        if start - margin < 0 :
            print("interaction range start error")
            start = margin 
        if end + margin > len(ret_global_p0):
            print("interaction range end error")
            end = len(ret_global_p0) - margin 

        # weight
        smoothing_range = torch.arange(0, 2*margin, 1).to(args.device)
        weight = smoothing_range / len(smoothing_range)
        weight = weight.unsqueeze(-1).unsqueeze(-1).repeat(1,args.num_joint, 3)

        # interpolation
        global_p0[start-margin:start+margin] = (1-weight)*backup_global_p0[start-margin:start+margin] + weight*update_global_p0[start-margin:start+margin]
        global_p0[start+margin:end-margin] = update_global_p0[start+margin:end-margin]
        global_p0[end-margin:end+margin] = (weight)*backup_global_p0[end-margin:end+margin] + (1-weight)*update_global_p0[end-margin:end+margin]

        return global_p0
    else:
        return ret_global_p0

def scale_diff_by_dist(diff, dist):
    """The weight grows as the distance gets smaller."""
    k = 1
    weight = torch.exp(-k * dist)
    weight = weight.unsqueeze(-1).repeat(1, 1, 3)

    return weight * diff


""" keep bone lengths """
def dot_prod(a, b):
    return torch.sum(a * b, dim=-1)

def position_scaled_by_offset(args, global_p, skeleton):
    parent_idx = skeleton.parent_idx
    for j in range(skeleton.num_joints):
        # Not update
        if j in child_of_division:
            continue

        # Not update root
        parent_j = parent_idx[j]
        if parent_j == -1:
            continue

        diff_p = global_p[:, j] - global_p[:, parent_j]

        offset = torch.tensor(skeleton.joints[j].offset).to(args.device)
        exp_offset = offset.repeat(diff_p.shape[0], 1)
        offset_ratio = torch.norm(exp_offset, dim=-1) / \
            torch.norm(diff_p, dim=-1)
        offset_ratio = offset_ratio.unsqueeze(-1).repeat(1, 3)

        scaled_diff_p = offset_ratio * diff_p
        update_value = global_p[:, parent_j] + scaled_diff_p
        child_recursive_update(skeleton, j, global_p,
                               update_value - global_p[:, j])
        global_p[:, j] = update_value

    return global_p

def target_position_scaled_by_offset(args, origin_p, target_p, skeleton):
    """ 
    origin_p: global_p before the update, shape: [frame, joint, 3]
    target_p: global_p after the update
    """
    # not update globap_p of root, child of division
    global_p = origin_p.clone()

    for j in range(global_p.shape[1]):
        parent_j = skeleton.parent_idx[j]
        # Not update
        if j in child_of_division:
            continue

        # Not update root
        if parent_j == -1:
            continue
        diff_p = target_p[:, j] - global_p[:, parent_j]

        # offset
        offset = torch.tensor(skeleton.joints[j].offset).to(args.device)
        exp_offset = offset.repeat(diff_p.shape[0], 1)
        offset_ratio = torch.norm(exp_offset, dim=-1) / \
            torch.norm(diff_p, dim=-1)
        offset_ratio = offset_ratio.unsqueeze(-1).repeat(1, 3)

        scaled_diff_p = offset_ratio * diff_p
        update_value = global_p[:, parent_j] + scaled_diff_p
        child_recursive_update(skeleton, j, global_p,
                               update_value - global_p[:, j])
        global_p[:, j] = update_value

    return global_p

def child_recursive_update(skeleton, j, global_p, diff_p):
    for child_j in skeleton.children_idx[j]:
        global_p[:, child_j] += diff_p
        child_recursive_update(skeleton, child_j, global_p, diff_p)

def update_absolute_p_by_index(origin_motion, updated_motion, index):
    ret_motion = origin_motion.clone()
    ret_motion[:, index] = updated_motion[:, index]
    
    return ret_motion
