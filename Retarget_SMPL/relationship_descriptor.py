import torch
import numpy as np
from option_motion import root_motion_by_root_vids

division = [0, 11]
child_of_division = [1, 5, 9, 12, 14, 18]
division_and_child_of_division = [0, 1, 5, 9, 11, 12, 14, 18]

""" RD """


def retarget_one_motion(args,
                  geo_source_ptn, geo_target_ptn,
                  source_motion0, source_motion1,
                  updated_motion0, updated_motion1,
                  root_joints=None, spine_joints=None, limb_joints=None):
    if args.adapt_char=="SMPLx":
        # src 
        descriptor_vids = geo_source_ptn.descriptor_vids
        descriptor_vids = torch.tensor(descriptor_vids).to(args.device)
        # tgt
        tgt_descriptor_vids = descriptor_vids
    else:
        descriptor_vids = geo_source_ptn.anchor_vids
        # tgt
        tgt_descriptor_vids = geo_target_ptn.anchor_vids
        
    len_frame, len_anchor = len(source_motion1.poses), len(descriptor_vids)
    batch = torch.tensor([0]).repeat(len_frame, len_anchor).to(args.device)
    frame = torch.arange(len_frame).unsqueeze(-1).repeat(1, len_anchor).to(args.device)

    """ Source descriptor B """
    # own (charA, motion0)
    _, _, charA_global_p0 = get_rootP_localR_globalP_from_motion(args, source_motion0.poses)

    # Partner (charB, motion1)
    source_root_p1, local_R1, _ = get_rootP_localR_globalP_from_motion(args, source_motion1.poses)
    geo_source_ptn.set_pose_by_source_batch_frame(local_R1.unsqueeze(0), source_root_p1.unsqueeze(0))
    charB_vpos1 = geo_source_ptn.get_positions_from_vids(descriptor_vids.repeat(len_frame, 1), batch, frame)

    # desc
    desc1_to_joint0 = charA_global_p0[:, :, None, :] - charB_vpos1[:, None, :, :]  # pj - di
    dist_desc1_to_joint0 = torch.norm(desc1_to_joint0, dim=-1)  # ||pj - di||

    
    """ target descriptor B """
    # motion of target B
    root_p1, local_R1, _ = get_rootP_localR_globalP_from_motion(args, updated_motion1.poses)
    geo_target_ptn.set_pose_by_source_batch_frame(local_R1.unsqueeze(0), root_p1.unsqueeze(0))
    updated_charB_vpos1 = geo_target_ptn.get_positions_from_vids(tgt_descriptor_vids.repeat(len_frame, 1), batch, frame)
    
    # 예외처리: descriptor을 특정 joint로 제한하고 싶을때 # TODO check
    # if source_motion0.name in root_motion_by_root_vids: 
    #     len_root_anchor = len(geo_source_ptn.root_descriptor_vids)
    #     root_batch = torch.tensor([0]).repeat(len_frame, len_root_anchor).to(args.device)
    #     root_frame = torch.arange(len_frame).unsqueeze(-1).repeat(1, len_root_anchor).to(args.device)
    
    #     root_descriptor_vids = torch.tensor(geo_source_ptn.root_descriptor_vids).to(args.device)
    #     root_charB_vpos1 = geo_source_ptn.get_positions_from_vids(root_descriptor_vids.repeat(len_frame, 1), root_batch, root_frame)
    #     root_desc1_to_joint0 = charA_global_p0[:, :, None, :] - root_charB_vpos1[:, None, :, :]  # pj - di
    #     dist_root_desc1_to_joint0 = torch.norm(root_desc1_to_joint0, dim=-1)  # ||pj - di||
    
    #     updated_root_charB_vpos1 = geo_target_ptn.get_positions_from_vids(root_descriptor_vids.repeat(len_frame, 1), root_batch, root_frame)
    
    """ update motion A """
    # charA, motion0
    _, _, source_global_p0 = get_rootP_localR_globalP_from_motion(args, updated_motion0.poses)

    # update root
    if root_joints != []:
        ret_global_p0 = \
            update_by_part(args, tgt_descriptor_vids,
                        desc1_to_joint0, dist_desc1_to_joint0,
                        updated_motion0, source_global_p0, # ptn
                        geo_target_ptn, updated_charB_vpos1, # dfm
                        update_part="root", update_joints=root_joints,
                        root_r2_dist=1/10, root_pow_lambda=1/10)
        
        # 예외처리: descriptor을 특정 joint로 제한하고 싶을때
        # if source_motion0.name in root_motion_by_root_vids:
        #     ret_global_p0 = \
        #         update_by_part(args, root_descriptor_vids,
        #                     root_desc1_to_joint0, dist_root_desc1_to_joint0,
        #                     updated_motion0, source_global_p0, # ptn
        #                     geo_target_ptn, updated_root_charB_vpos1, # dfm
        #                     update_part="root", update_joints=root_joints,
        #                     root_r2_dist=1/10, root_pow_lambda=1/10)
        
        # pene
        # heel
        # ret_global_p0 = lift_by_pene_val(
        #     args, ret_global_p0, updated_motion0, args.foot_heel_index, args.pene_ths_heel)
        # # ee joint
        # ret_global_p0 = lift_by_pene_val(
        #     args, ret_global_p0, updated_motion0, args.foot_ee_index, args.pene_ths)
        # TODO: check other joint: 손...
        
    else:
        ret_global_p0 = source_global_p0

    # update spine
    if spine_joints != []:
        ret_global_p0 = \
            update_by_part(args, tgt_descriptor_vids,
                            desc1_to_joint0, dist_desc1_to_joint0,
                            updated_motion0, ret_global_p0,  # source_global_p0
                            geo_target_ptn, updated_charB_vpos1,
                            update_part="spine", update_joints=spine_joints,
                            limb_r2_dist=1/10, limb_pow_lambda=1/10)

    # update limb
    ret_global_p0 = \
        update_by_part(args, tgt_descriptor_vids,
                        desc1_to_joint0, dist_desc1_to_joint0,
                        updated_motion0, ret_global_p0,  # source_global_p0
                        geo_target_ptn, updated_charB_vpos1,
                        update_part="limb", update_joints=limb_joints,
                        limb_r2_dist=1/10, limb_pow_lambda=1/10)
    # pene 
    # heel 
    ret_global_p0 = lift_by_pene_val(args, ret_global_p0, updated_motion0, args.heel_joints, args.toe_pene_ths)
    # ee joint
    ret_global_p0 = lift_by_pene_val(args, ret_global_p0, updated_motion0, args.toe_joints,  args.heel_pene_ths)

    # edit by range 
    if args.update_by_clampping_range:
        _, _, original_global_p0 = get_rootP_localR_globalP_from_motion(
            args, updated_motion0.poses)
        ret_global_p0 = clampping_by_interaction_range(
            args, original_global_p0, ret_global_p0)
    
    update_motion_by_global_p(updated_motion0, ret_global_p0)
    
    return updated_motion0
    
def update_motion_by_global_p(motion, update_global_p):
    device = update_global_p.device
    skeleton = motion.skeleton #.parent_idx
    for f, pose in enumerate(motion.poses):
        local_R, root_p = update_pose_by_global_p(torch.tensor(pose.local_R) .float().to(device), 
                                                  torch.tensor(pose.global_p).float().to(device), 
                                                  torch.tensor(pose.global_R).float().to(device), 
                                                  skeleton, update_global_p[f])
        pose.local_R = local_R.cpu().numpy()
        pose.root_p = root_p.cpu().numpy()
        pose.update()

    return motion
    
from pymovis.motion.ops import torchmotion
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

def update_by_part(args, anchor_vids,
                   desc_to_joint, dist_desc_to_joint, # relationship
                   updated_motion0, source_global_p0, # own
                   geo_ptn, updated_charB_vpos1, # ptn
                   update_part="", update_joints=None,
                   root_r2_dist=None, root_pow_lambda=None,
                   limb_r2_dist=None, limb_pow_lambda=None):
    """
    updated_motion0: to update
    updated_motion1: updated partner
    """

    # update root
    if update_part=="root":
        output_global_p0 = \
            relation_descriptor(args,
                                desc_to_joint, dist_desc_to_joint, anchor_vids, updated_charB_vpos1, geo_ptn,
                                r2_dist=root_r2_dist, pow_lambda=root_pow_lambda)
        diff = (output_global_p0 - source_global_p0) # f, 22
        diff = scale_diff_by_dist(diff, torch.min(dist_desc_to_joint, dim=-1)[0]) # .unsqueeze(-1).repeat(1,1,3)
        diff = diff[:, 0].unsqueeze(1).repeat(1, args.num_joint, 1)
        ret_global_p0 = source_global_p0 + diff[:, 0].unsqueeze(1).repeat(1, args.num_joint, 1)
    
    # update spine, limb
    else: 
        output_global_p0 = \
            relation_descriptor(args,
                                desc_to_joint, dist_desc_to_joint, anchor_vids, updated_charB_vpos1, geo_ptn,
                                r2_dist=limb_r2_dist, pow_lambda=limb_pow_lambda)

        # 거리에 따라 정도 조절
        diff = torch.zeros_like(output_global_p0)
        diff[:, update_joints] = (output_global_p0 - source_global_p0)[:, update_joints]
        diff = scale_diff_by_dist(diff, torch.min(dist_desc_to_joint, dim=-1)[0])  # .unsqueeze(-1).repeat(1,1,3)
        output_global_p0 = source_global_p0 + diff
        
        # update position by index
        output_global_p0 = update_absolute_p_by_index(source_global_p0, output_global_p0, update_joints)
        
        # scale by offset
        ret_global_p0 = target_position_scaled_by_offset(args, source_global_p0, output_global_p0, updated_motion0.skeleton)

    return ret_global_p0

def lift_by_pene_val(args, ret_global_p0, motion, end_effectors, pene_ths):
    import copy 
    ret_global_p0_backup = copy.deepcopy(ret_global_p0)
    parent_idxs = motion.skeleton.parent_idx
    end_effectors = torch.tensor(end_effectors)
    
    # parent index of moving joint
    len_end_effectors = len(end_effectors)
    
    # pene tensor 
    pene_ths_tensor = torch.tensor(pene_ths).repeat(len_end_effectors).to(args.device)
    
    # update pene value
    len_frame = ret_global_p0.shape[0]
    updated_joints = torch.zeros(len_frame, len_end_effectors)
    for f in range(len_frame):
        pene_index = torch.where(ret_global_p0[f, end_effectors, 1] < pene_ths_tensor)
        pene_joints = end_effectors[pene_index]
        if pene_joints.shape[0] == 0: # or pene_joints[0].size()[0] ==0
            continue
        
        # pene val -> parent로 전파해주기
        penetrated_disp = torch.zeros(22).to(args.device)
        pene_values = (ret_global_p0[f][pene_joints][:, 1] - pene_ths) # 음수
        for i, pene_idx in enumerate(pene_joints):
            diff_val = pene_values[i]
            
            # update to joint and parent (until root)
            penetrated_disp[pene_idx] = diff_val
            parent_recursive_update(parent_idxs, pene_idx, penetrated_disp, diff_val) # get reculsive displacement
        ret_global_p0[f, :, 1] -= penetrated_disp # 양수
        # print("frame {} pene_val {} \npenetrated_disp{}".format(f, pene_values, penetrated_disp))
        
        # check update
        for j, _ in enumerate(pene_joints):
            updated_joints[f, j] = 1

        # import pdb; pdb.set_trace()
        # for f in range(len_frame):
        # position updated by offset
        pose = motion.poses[f]
        local_R, root_p = update_pose_by_global_p(torch.tensor(pose.local_R), 
                                                  torch.tensor(pose.global_p), 
                                                  torch.tensor(pose.global_R), 
                                                  pose.skeleton, ret_global_p0[f])
        pose.local_R = local_R.cpu().numpy()
        pose.root_p = root_p.cpu().numpy()
        pose.update()
        
        # if not reachable, apply IK
        for i, pene_idx in enumerate(pene_joints):
            grand_parent_idx = parent_idxs[parent_idxs[pene_idx]]
            if pene_idx in end_effectors:
                pose.two_bone_ik(grand_parent_idx, pene_idx, ret_global_p0[f, pene_idx]) # TODO pene_idx: recursive한 parent에 대해서 다 확인?
            else:
                pose.two_bone_ik(grand_parent_idx, pene_idx, ret_global_p0[f, pene_idx], use_forward=True)
    
    _, _, ret_global_p0 = get_rootP_localR_globalP_from_motion(args, motion.poses)
    
    return ret_global_p0

# parent에 대해서 전달, disp가 이미 값이 크다면 그냥 두기. 
def parent_recursive_update(parent_idxs, j, penetrated_disp, diff_val):
    parent_idx = parent_idxs[j]
    if parent_idx != 0:
        if penetrated_disp[parent_idx] > diff_val:
            penetrated_disp[parent_idx] = diff_val
        parent_recursive_update(parent_idxs, parent_idx, penetrated_disp, diff_val)
    else: # if parent_idx == -1:
        return penetrated_disp
    
def scale_diff_by_dist(diff, dist):
    # 가까울수록 weight 가 커져. k가 커지면 거리에 따라 값의 차이는 커지지만 가까울때 영향력 줄어듦 
    k = 1
    exp_dist = torch.exp(-k * dist)

    weight = exp_dist
    weight = weight.unsqueeze(-1).repeat(1, 1, 3)

    return weight * diff

def propagate_diff_to_child(diff, skeleton, index):
    for j in index:
        child_j = skeleton.children_idx[j]
        if len(child_j) == 0:
            continue
        child_j = child_j[0]
        tmp = torch.norm(diff[:, j], dim=-1) - \
            torch.norm(diff[:, child_j], dim=-1)
        index = torch.where(tmp > 0)
        diff[index[0], child_j] = diff[index[0], j]

    return diff

def clampping_by_interaction_range(args, backup_global_p0, ret_global_p0):
    if args.interaction_start_frame != -1 and args.interaction_end_frame != -1:
        update_global_p0 = ret_global_p0.clone()
        global_p0 = backup_global_p0.clone()  # output

        start = args.interaction_start_frame
        end = args.interaction_end_frame
        range = 20
        if start - range < 0 :
            print("interaction range start error")
            start = range 
        if end + range > len(ret_global_p0):
            print("interaction range end error")
            end = len(ret_global_p0) - range 

        # weight
        smoothing_range = torch.arange(0, 2*range, 1).to(args.device)  # (-10~10) 1씩 증가: 20개
        weight = smoothing_range / len(smoothing_range) # torch.exp(-1*(smoothing_range))
        weight = weight.unsqueeze(-1).unsqueeze(-1).repeat(1,args.num_joint, 3)

        # interpolation
        global_p0[start-range:start+range] = (1-weight)*backup_global_p0[start-range:start+range] + weight*update_global_p0[start-range:start+range]
        global_p0[start+range:end-range] = update_global_p0[start+range:end-range]
        global_p0[end-range:end+range] = (weight)*backup_global_p0[end-range:end+range] + (1-weight)*update_global_p0[end-range:end+range]

        return global_p0
    else:
        return ret_global_p0


def relation_descriptor(args,
                        desc_to_joint, dist_desc_to_joint, anchor_vids, updated_charB_vpos1, geo_ptn,
                        r2_dist, pow_lambda):
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
    r2 = r1 + r2_dist * torch.tensor(geo_ptn.height).to(args.device) # ptn이 아니라 own

    # close
    cond = (dist_desc_to_joint < r2.unsqueeze(-1))
    fade_func[cond] = 1 - torch.pow(((dist_desc_to_joint[cond] - r1.unsqueeze(-1).repeat(1, 1, len_anchor)[cond])
                                    / (r2.unsqueeze(-1).repeat(1, 1, len_anchor)[cond] - r1.unsqueeze(-1).repeat(1, 1, len_anchor)[cond])), pow_lambda)
    # far
    fade_func[dist_desc_to_joint > r2.unsqueeze(-1)] = 0

    # update by clipping
    weight_twoprime = fade_func  # normal_weight_prime *

    # 3. final weight
    weight = weight_twoprime / torch.sum(weight_twoprime, dim=-1).unsqueeze(-1)

    """ charB_vpos + desc_to_joint -> charA_joint_p """
    updated_charA_global_p0_ = weight.unsqueeze(-1) * (updated_charB_vpos1[:, None, :, :] + desc_to_joint)
    ret_global_p0 = torch.sum(updated_charA_global_p0_, dim=-2)

    return ret_global_p0


""" useful functions """
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

def dot_prod(a, b):
    return torch.sum(a * b, dim=-1)

# Scaling by offset
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
    origin_p: update 전 global_p, shape: [frame, joint, 3]
    target_p: update 후 global_p
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

# Update global_p
def child_recursive_update(skeleton, j, global_p, diff_p):
    for child_j in skeleton.children_idx[j]:
        global_p[:, child_j] += diff_p
        child_recursive_update(skeleton, child_j, global_p, diff_p)

# Not update: child of division
def update_relative_p(origin_motion, updated_motion, index, skeleton):
    origin_relative_motion = origin_motion - origin_motion[:, 0:1]
    updated_relative_motion = updated_motion - updated_motion[:, 0:1]
    
    for j in index:
        if j not in child_of_division: 
            diff = (updated_relative_motion[:,j] - origin_relative_motion[:, j])
            origin_relative_motion[:, j] = origin_relative_motion[:, j] + diff
            child_recursive_update(skeleton, j, origin_relative_motion, diff)

    # update
    ret_motion = origin_relative_motion + origin_motion[:, 0:1]

    return ret_motion

def update_absolute_p_by_index(origin_motion, updated_motion, index):
    ret_motion = origin_motion.clone()
    ret_motion[:, index] = updated_motion[:, index]
    
    return ret_motion

""" pene"""
def resolve_ground_pene(args, output_motion0, output_motion1):
    """ Post-proces: joint별 ground pene ths에 대한 ik """
    
    # joints
    end_effector    = args.toe_joints # [4, 8,] # 16, 17, 20, 21
    joint_before_ee = args.heel_joints # [3, 7,] # 15, 16, 19, 20
    
    _, _, output_global_p0 = get_rootP_localR_globalP_from_motion(args, output_motion0.poses)
    _, _, output_global_p1 = get_rootP_localR_globalP_from_motion(args, output_motion1.poses)
    
    # heel
    output_global_p0 = lift_by_pene_val(args, output_global_p0, output_motion0, joint_before_ee, args.heel_pene_ths)
    output_global_p1 = lift_by_pene_val(args, output_global_p1, output_motion1, joint_before_ee, args.heel_pene_ths)
    
    # toe
    output_global_p0 = lift_by_pene_val(args, output_global_p0, output_motion0, end_effector, args.toe_pene_ths)
    output_global_p1 = lift_by_pene_val(args, output_global_p1, output_motion1, end_effector, args.toe_pene_ths)
    
    # update 
    update_motion_by_global_p(output_motion0, output_global_p0)
    update_motion_by_global_p(output_motion1, output_global_p1)
    
    return output_motion0, output_motion1
