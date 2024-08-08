import numpy as np


def detect_foot_contact(args, motion):
    # bool [frame, joint]
    foot_idx = args.foot_ee_index
    ground_height = args.pene_ths
    
    len_frame = len(motion.poses)
    num_joints = motion.skeleton.num_joints
    foot_contact_label = np.zeros((len_frame, num_joints), dtype=bool)
    for f, pose in enumerate(motion.poses):
        for joint_idx in foot_idx:
            if pose.global_p[joint_idx][1] < ground_height:
                foot_contact_label[f, joint_idx] = True
            
    return foot_contact_label

def detect_foot_contact_from_position(args, position):
    # bool [frame, joint]
    len_frame, num_joints, _ = position.shape 
    foot_contact_label = np.zeros((len_frame, num_joints), dtype=bool)
    
    # Create a boolean mask for foot contact
    # ee 
    foot_idx = args.foot_ee_index
    joint_positions = position[:, foot_idx, 1]
    contact_mask = joint_positions < args.pene_ths
    foot_contact_label[:, foot_idx] = contact_mask
    
    # before ee
    foot_idx = args.foot_heel_index
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
    foot_idx = args.foot_ee_index
    joint_positions = position[:, :, foot_idx, 1]
    contact_mask = joint_positions < args.pene_ths
    foot_contact_label[:, :, foot_idx] = contact_mask
    
    # Extract the y-coordinates of the relevant joints
    foot_idx = args.foot_heel_index
    joint_positions = position[:, :, foot_idx, 1]
    contact_mask = joint_positions < args.pene_ths_heel
    foot_contact_label[:, :, foot_idx] = contact_mask
    
    return foot_contact_label
