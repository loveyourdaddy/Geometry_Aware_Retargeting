import numpy as np


def detect_foot_contact(args, motion):
    # bool [frame, joint]
    foot_idx = args.foot_index
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
    foot_idx = args.foot_index
    ground_height = args.pene_ths
    
    len_frame, num_joints, _ = position.shape 
    foot_contact_label = np.zeros((len_frame, num_joints), dtype=bool)
    
    # Extract the y-coordinates of the relevant joints
    joint_positions = position[:, foot_idx, 1]
    
    # Create a boolean mask for foot contact
    contact_mask = joint_positions < ground_height
    
    # Place the contact mask into the correct positions
    foot_contact_label[:, foot_idx] = contact_mask
            
    return foot_contact_label

# torch version
def detect_foot_contact_from_batched_position(args, position):
    import torch 
    # bool [frame, joint]
    foot_idx = args.foot_index
    ground_height = args.pene_ths
    
    batch_size, len_frame, num_joints, _ = position.shape 
    foot_contact_label = torch.zeros((batch_size, len_frame, num_joints), dtype=torch.bool).to(position.device)
    
    # Extract the y-coordinates of the relevant joints
    joint_positions = position[:, :, foot_idx, 1]
    
    # Create a boolean mask for foot contact
    contact_mask = joint_positions < ground_height
    
    # Place the contact mask into the correct positions
    foot_contact_label[:, :, foot_idx] = contact_mask
    
    return foot_contact_label
