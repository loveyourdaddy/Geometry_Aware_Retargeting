import numpy as np

def detect_foot_contact(args, motion):
    len_frame = len(motion.poses)
    num_joints = motion.skeleton.num_joints
    
    # motion.poses to position vector
    position = np.zeros((len_frame, num_joints, 3))
    for i, pose in enumerate(motion.poses):
        position[i] = pose.global_p
    
    # bool [frame, joint]
    foot_contact_label = np.zeros((len_frame, num_joints), dtype=bool)
    # ee
    foot_idx = args.toe_joints
    joint_positions = position[:, foot_idx, 1]
    contact_mask = np.logical_and(joint_positions < args.toe_pene_ths, joint_positions > 0)
    foot_contact_label[:, foot_idx] = contact_mask
    
    # heel
    foot_idx = args.heel_joints
    joint_positions = position[:, foot_idx, 1]
    contact_mask = np.logical_and(joint_positions < args.heel_pene_ths, joint_positions > 0)
    foot_contact_label[:, foot_idx] = contact_mask

    return foot_contact_label

def detect_foot_contact_from_position(args, position):
    # bool [frame, joint]
    len_frame, num_joints, _ = position.shape 
    foot_contact_label = np.zeros((len_frame, num_joints), dtype=bool)
    
    # Create a boolean mask for foot contact
    # ee 
    foot_idx = args.toe_joints
    joint_positions = position[:, foot_idx, 1]
    contact_mask = joint_positions < args.pene_ths
    foot_contact_label[:, foot_idx] = contact_mask
    
    # before ee
    foot_idx = args.heel_joints
    joint_positions = position[:, foot_idx, 1]
    contact_mask = joint_positions < args.pene_ths_heel
    foot_contact_label[:, foot_idx] = contact_mask
    
    return foot_contact_label

# torch version
def detect_foot_contact_from_batched_position(args, position):
    import torch 
    batch_size, len_frame, num_joints, _ = position.shape 
    
    # bool [batch, frame, joint]
    foot_contact_label = torch.zeros((batch_size, len_frame, num_joints), dtype=torch.bool).to(position.device)
    
    # Extract the y-coordinates of the relevant joints
    # ee 
    foot_idx = args.toe_joints
    joint_positions = position[:, :, foot_idx, 1]
    contact_mask = joint_positions < args.toe_pene_ths
    foot_contact_label[:, :, foot_idx] = contact_mask
    
    # Extract the y-coordinates of the relevant joints
    # heel
    foot_idx = args.heel_joints
    joint_positions = position[:, :, foot_idx, 1]
    contact_mask = joint_positions < args.heel_pene_ths
    foot_contact_label[:, :, foot_idx] = contact_mask
    
    return foot_contact_label
