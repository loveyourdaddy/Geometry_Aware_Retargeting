"""Resolve foot-ground penetration."""
import torch

from Retarget_SMPL.motion_utils import (
    get_rootP_localR_globalP_from_motion, update_motion_by_global_p, update_pose_by_global_p,
)


def parent_recursive_update(parent_idxs, j, penetrated_disp, diff_val):
    """Propagate the same displacement to the parents up to the root. A parent that already has a larger displacement is kept."""
    parent_idx = parent_idxs[j]
    if parent_idx != 0:
        if penetrated_disp[parent_idx] > diff_val:
            penetrated_disp[parent_idx] = diff_val
        parent_recursive_update(parent_idxs, parent_idx, penetrated_disp, diff_val)
    else:
        return penetrated_disp

def lift_by_pene_val(args, ret_global_p0, motion, end_effectors, pene_ths):
    """Lift the legs in frames where end_effectors are lower than pene_ths."""
    parent_idxs = motion.skeleton.parent_idx
    end_effectors = torch.tensor(end_effectors)
    pene_ths_tensor = torch.tensor(pene_ths).repeat(len(end_effectors)).to(args.device)

    len_frame = ret_global_p0.shape[0]
    for f in range(len_frame):
        pene_index = torch.where(ret_global_p0[f, end_effectors, 1] < pene_ths_tensor)
        pene_joints = end_effectors[pene_index]
        if pene_joints.shape[0] == 0:
            continue
        
        # pene val -> propagate to the parents
        penetrated_disp = torch.zeros(22).to(args.device)
        pene_values = (ret_global_p0[f][pene_joints][:, 1] - pene_ths) # negative
        for i, pene_idx in enumerate(pene_joints):
            diff_val = pene_values[i]
            penetrated_disp[pene_idx] = diff_val
            parent_recursive_update(parent_idxs, pene_idx, penetrated_disp, diff_val)
        ret_global_p0[f, :, 1] -= penetrated_disp # positive

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
            pose.two_bone_ik(grand_parent_idx, pene_idx, ret_global_p0[f, pene_idx])

    _, _, ret_global_p0 = get_rootP_localR_globalP_from_motion(args, motion.poses)
    
    return ret_global_p0

def resolve_ground_pene(args, output_motion0, output_motion1):
    """Post-process: lift the heels and toes of both motions above the threshold heights."""
    _, _, output_global_p0 = get_rootP_localR_globalP_from_motion(args, output_motion0.poses)
    _, _, output_global_p1 = get_rootP_localR_globalP_from_motion(args, output_motion1.poses)
    
    # heel
    output_global_p0 = lift_by_pene_val(args, output_global_p0, output_motion0, args.heel_joints, args.heel_pene_ths)
    output_global_p1 = lift_by_pene_val(args, output_global_p1, output_motion1, args.heel_joints, args.heel_pene_ths)
    
    # toe
    output_global_p0 = lift_by_pene_val(args, output_global_p0, output_motion0, args.toe_joints, args.toe_pene_ths)
    output_global_p1 = lift_by_pene_val(args, output_global_p1, output_motion1, args.toe_joints, args.toe_pene_ths)
    
    update_motion_by_global_p(output_motion0, output_global_p0)
    update_motion_by_global_p(output_motion1, output_global_p1)
    
    return output_motion0, output_motion1
