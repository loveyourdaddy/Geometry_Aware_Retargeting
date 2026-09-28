import sys
sys.path.append('./0_geometry_aware_retargeting/') # Retargeting workspace
sys.path.append('../0_geometry_aware_retargeting/') # 0_

from pymovis.motion.data import bvh
from pymovis.motion.core import Skeleton, Pose, Motion
from pymovis.vis.const import skeleton_dic, finger_dic
from option_motion import trainset_bvh
from etc.etc import deepcopy
import torch
import numpy as np

""" Get motion """

def get_train_bvh(name0, name1):
    # name
    source0_motion_names = trainset_bvh.keys()
    source1_motion_names = trainset_bvh.values()

    # motions
    source0_motions = get_interaction_motions_from_list(name0, source0_motion_names)
    source1_motions = get_interaction_motions_from_list(
        name1, source1_motion_names)

    return source0_motions, source1_motions

def get_single_motion_from_list(character, motion_list):
    # load list
    input_motion_lists = []
    for name in motion_list:
        input_motion_lists.append(
            "../Resource/single_motion/{}/{}.bvh".format(character, name))

    # motion
    input_motions = []
    for motion_name in input_motion_lists:
        bvh_motion = bvh.load(motion_name, v_forward=[0, 0, 1], v_up=[0, 1, 0])
        input_motions.append(bvh_motion)

    return input_motions

def get_interaction_motions_from_list(character, motion_list):
    # load list
    input_motion_lists = []
    for name in motion_list:
        input_motion_lists.append(
            "../Resource/motions/interaction_motion/{}/{}.bvh".format(character, name))

    # motion
    input_motions = []
    for motion_name in input_motion_lists:
        bvh_motion = bvh.load(motion_name, v_forward=[0, 0, 1], v_up=[0, 1, 0])
        input_motions.append(bvh_motion)

    return input_motions


""" Full motion with padded """
# full length motion 
def get_full_motion_with_pad(args, source0_motions, source1_motions):
    # source0 motion
    motions0 = []
    for motion in source0_motions:
        motions0.append(full_motion(args, motion))

    # source1 motions
    motions1 = []
    for motion in source1_motions:
        motions1.append(full_motion(args, motion))
        
    # add padding
    max_length = max([len(motion) for motion in motions0])
    length_list = []
    for motion in motions0:
        length_list.append(len(motion))
        pose = motion[0]
        zero_pose = np.zeros_like(pose)
        motion += [zero_pose] * (max_length - len(motion))
        
    for motion in motions1:
        pose = motion[0]
        zero_pose = np.zeros_like(pose)
        motion += [zero_pose] * (max_length - len(motion))

    # [n_window, window_size, dim] dim: (frames x (joint 22*6 + root 3 = 135)]
    motions0 = np.array(motions0)
    motions0 = torch.tensor(motions0).to(args.device)
    motions1 = np.array(motions1)
    motions1 = torch.tensor(motions1).to(args.device)

    return motions0, motions1, length_list

def full_motion(args, motion):
    from pymovis.motion.ops.npmotion import R_to_R6, mat2quat
    
    source_p = []
    for p in motion.poses:
        if args.rotation_rep == 'quat':
            vectorized_pose = np.concatenate((mat2quat(p.local_R).reshape(-1), p.root_p))
        elif args.rotation_rep == 'R6':
            vectorized_pose = np.concatenate((R_to_R6(p.local_R).reshape(-1), p.root_p))
        else:
            raise ValueError('Unknown rotation representation')
        source_p.append(vectorized_pose)
        
    return source_p

# runtime
def get_vectorized_motion(args, source0_motions, source1_motions):
    # test time, use whole motion
    source0_window_motion = get_vectorized_pose(args, source0_motions)
    source1_window_motion = get_vectorized_pose(args, source1_motions)

    # [n_window, window_size, frames x (joint + root)] (frames x (22*6 + 3 = 135)]
    source0_window_motion = np.array(source0_window_motion)
    source0_window_motion = torch.tensor(source0_window_motion).to(args.device)
    source1_window_motion = np.array(source1_window_motion)
    source1_window_motion = torch.tensor(source1_window_motion).to(args.device)

    return source0_window_motion, source1_window_motion

def get_vectorized_pose(args, motion):
    from pymovis.motion.ops.npmotion import R_to_R6 # , mat2quat
    
    source = []
    for p in motion.poses:
        # if args.rotation_rep == 'quat':
        #     vectorized_pose = np.concatenate(
        #         (mat2quat(p.local_R).reshape(-1), p.root_p))
        # elif args.rotation_rep == 'R6':
        vectorized_pose = np.concatenate(
            (R_to_R6(p.local_R).reshape(-1), p.root_p))
        source.append(vectorized_pose)
    return source


""" Get windowed motion """
# windowed motion 
def get_window_motion(args, source0_motions, source1_motions):
    # source0 motion
    source0_window_motion = []
    f = 0 
    for motion in source0_motions:
        window_motions = windowed_motion(args, motion)
        for window_motion in window_motions:
            source0_window_motion.append(window_motion)
            
    # check frames 
    # for motion in source0_motions:
    #     f+=motion.num_frames

    # source1 motions
    source1_window_motion = []
    for motion in source1_motions:
        window_motion = windowed_motion(args, motion)
        for window in window_motion:
            source1_window_motion.append(window)

    # [n_window, window_size, dim] dim: (frames x (joint 22*6 + root 3 = 135)]
    source0_window_motion = np.array(source0_window_motion)
    source0_window_motion = torch.tensor(source0_window_motion).to(args.device)
    source1_window_motion = np.array(source1_window_motion)
    source1_window_motion = torch.tensor(source1_window_motion).to(args.device)

    return source0_window_motion, source1_window_motion

def windowed_motion(args, motion):
    from pymovis.motion.ops.npmotion import R_to_R6 # , mat2quat

    step_size = args.window_size # // 2
    window_size = step_size #* 2
    len_pose = len(motion.poses)
    num_window = int(len_pose/step_size) #- 1  # len_pose//step_size - 1

    source = []
    for i in range(num_window):
        start = i*step_size
        end = start + window_size  # (i+1)*step_size
        poses = motion.poses[start:end]

        window_source = []
        for p in poses:
            if args.rotation_rep == 'quat':
                vectorized_pose = np.concatenate(
                    (mat2quat(p.local_R).reshape(-1), p.root_p))
            elif args.rotation_rep == 'R6':
                vectorized_pose = np.concatenate(
                    (R_to_R6(p.local_R).reshape(-1), p.root_p))
            else:
                raise ValueError('Unknown rotation representation')
            window_source.append(vectorized_pose)
        source.append(window_source)

    return source

def cut_by_window(args, motion):
    step_size = args.window_size #// 2
    window_size = step_size #* 2
    len_pose = motion.shape[1]

    motion = np.expand_dims(motion, axis=1)

    windows = [motion[:, :, i:i+window_size, :]
               for i in range(0, len_pose-window_size+1, step_size)]
    if len(windows) == 0:
        return None
    window_motion = np.concatenate(windows, axis=1)

    return torch.tensor(window_motion).to(args.device)


""" padding """
# add padding to set as same length
def add_padding(motion, max_length):
    _, len_motion, _ = motion.shape
    pad_length = max(0, max_length - len_motion)

    # Pad only the second dimension
    padding = ((0, 0), (0, pad_length), (0, 0))
    padded_motion = np.pad(motion, padding, mode='constant', constant_values=0)

    return padded_motion


""" Refine motion """
# Select 22 joints in motion (remove finger joint) 
def refine_motion(motion, template):
    # index order
    joints = []
    joint_index = []
    for j, joint in enumerate(motion.skeleton.joints):
        if joint.name in template.skeleton.idx_by_name.keys():
            joints.append(joint)
            joint_index.append(j)

    # joints, index_by_name, name_by_index
    skeleton = Skeleton(joints=joints, v_up=template.skeleton.v_up,
                        v_forward=template.skeleton.v_forward)
    skeleton.children_idx = template.skeleton.children_idx
    skeleton.idx_by_name = template.skeleton.idx_by_name
    skeleton.name_by_idx = template.skeleton.name_by_idx
    skeleton.parent_idx = template.skeleton.parent_idx
    skeleton.v_forward = template.skeleton.v_forward
    skeleton.v_up = template.skeleton.v_up

    poses = []
    for pose in motion.poses:
        pose_ = Pose(skeleton, pose.local_R[joint_index], pose.root_p)
        poses.append(pose_)

    return Motion(skeleton, poses, motion.fps, motion.name, motion.type)

def refine_motion_by_template(motion, template):
    # index order
    joints = []
    joint_index = []
    for j, template_joint in enumerate(template.skeleton.joints):
        if template_joint.name in motion.skeleton.idx_by_name.keys():
            joint = motion.skeleton.joints[motion.skeleton.idx_by_name[template_joint.name]]
            joint_id = motion.skeleton.idx_by_name[template_joint.name]
            joints.append(joint)
            joint_index.append(joint_id)

    # joints, index_by_name, name_by_index
    skeleton = Skeleton(joints=joints, v_up=template.skeleton.v_up,
                        v_forward=template.skeleton.v_forward)
    skeleton.children_idx = template.skeleton.children_idx
    skeleton.idx_by_name = template.skeleton.idx_by_name
    skeleton.name_by_idx = template.skeleton.name_by_idx
    skeleton.parent_idx = template.skeleton.parent_idx
    skeleton.v_forward = template.skeleton.v_forward
    skeleton.v_up = template.skeleton.v_up

    poses = []
    for pose in motion.poses:
        pose_ = Pose(skeleton, pose.local_R[joint_index], pose.root_p)
        poses.append(pose_)

    return Motion(skeleton, poses, motion.fps, motion.name, motion.type)

# arrange joint 
def refine_different_skeleton_motion(motion, template):
    # index order
    joints = []
    joint_index = []
    new_joint_index = []
    for j, joint in enumerate(template.skeleton.joints):
        if joint.name in motion.skeleton.idx_by_name.keys():
            # print("{} {}".format(j, joint.name))
            motion_j = motion.skeleton.idx_by_name[joint.name]
            motion_joint = motion.skeleton.joints[motion_j]
            
            joints.append(motion_joint)
            joint_index.append(j)
            new_joint_index.append(motion_j)
        else:
            raise ValueError('Unknown joint name')
    
    # joints, index_by_name, name_by_index
    skeleton = Skeleton(joints=joints, v_up=template.skeleton.v_up,
                        v_forward=template.skeleton.v_forward)
    skeleton.children_idx = template.skeleton.children_idx
    skeleton.idx_by_name = template.skeleton.idx_by_name
    skeleton.name_by_idx = template.skeleton.name_by_idx
    skeleton.parent_idx = template.skeleton.parent_idx
    skeleton.v_forward = template.skeleton.v_forward
    skeleton.v_up = template.skeleton.v_up

    poses = []
    for pose in motion.poses:
        pose_ = Pose(skeleton, pose.local_R[new_joint_index], pose.root_p)
        poses.append(pose_)

    return Motion(skeleton, poses, motion.fps, motion.name, motion.type)
    
def select_skeleton_and_finger_idx(input):
    skeleton_joints = []
    finger_joints = []
    for j, joint in enumerate(input.skeleton.joints):
        # skeleton
        for _, key_name in enumerate(skeleton_dic.keys()):
            if joint.name in skeleton_dic[key_name]:
                skeleton_joints.append(j)
                break

        # finger
        for _, key_name in enumerate(finger_dic.keys()):
            if joint.name in finger_dic[key_name]:
                finger_joints.append(j)
                break

    return skeleton_joints, finger_joints


""" make motion """
# wo finger

def make_new_motions(args, jit_output_p0, jit_output_R0, jit_output_p1, jit_output_R1, target0_character, target1_character, source_motion0, source_motion1):
    output_motion0, output_motion1 = deepcopy(source_motion0), deepcopy(source_motion1)
    output_motion0 = make_new_motion(jit_output_p0, jit_output_R0, target0_character, output_motion0)
    output_motion1 = make_new_motion(jit_output_p1, jit_output_R1, target1_character, output_motion1)
    return output_motion0, output_motion1

def make_new_motion(root_p, local_R, target_a_character, info):
    # original_motion: update finger motion
    output_skeleton = target_a_character.meshes[0].source_skeleton
    root_p = root_p.cpu().detach().numpy()
    local_R = local_R.cpu().detach().numpy()
    
    # refine local_R
    local_R = refine_local_R(local_R,
                             target_a_character)

    poses = []
    length = len(root_p)
    for f in range(length):
        pose = Pose(output_skeleton, local_R[f], root_p[f])
        poses.append(pose)

    return Motion(output_skeleton, poses, info.fps, info.name, info.type)

# local_R 22 -> target_a_character 52 (bot_finger_motion 65)
def refine_local_R(local_R, target_a_character):
    target_skeleton = target_a_character.meshes[0].source_skeleton
    frame = local_R.shape[0]
    local_R_ = np.zeros_like(local_R)

    for f in range(frame):
        for j in range(target_skeleton.num_joints):
            local_R_[f, j] = local_R[f, j]

    return local_R_

# with finger
def make_new_motion_with_finger(root_p, local_R, target_a_character, info, skeleton_joints, finger_joints):
    # original_motion: update finger motion
    output_skeleton = target_a_character.meshes[0].source_skeleton
    root_p = root_p.cpu().detach().numpy()
    local_R = local_R.cpu().detach().numpy()

    # refine local_R
    local_R = refine_local_R_with_finger(local_R,
                                         target_a_character,
                                         skeleton_joints, finger_joints)
    poses = []
    length = len(root_p)
    for f in range(length):
        pose = Pose(output_skeleton, local_R[f], root_p[f])
        poses.append(pose)

    return Motion(output_skeleton, poses, info.fps, info.name, info.type)

# local_R 22 -> target_a_character 52 (bot_finger_motion 65)
def refine_local_R_with_finger(local_R, target_a_character, skeleton_joints, finger_joints):
    target_skeleton = target_a_character.meshes[0].source_skeleton
    frame, _, d0, d1 = local_R.shape
    local_R_ = np.zeros((frame, target_skeleton.num_joints, d0, d1))

    # fill local_R_
    for f in range(frame):
        # skeleton
        for j, source_jid in enumerate(skeleton_joints):
            local_R_[f, source_jid] = local_R[f, j]

        # finger
        for j, source_jid in enumerate(finger_joints):
            local_R_[f, source_jid] = np.array(
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    return local_R_


""" Displacement map """""
def get_displacement_map(pos1, pos2):
    pos1_expanded = pos1.unsqueeze(-2)
    pos2_expanded = pos2.unsqueeze(-3)

    # Calculate the displacement map using broadcasting
    displacement_map = pos1_expanded - pos2_expanded

    return displacement_map

def get_distance_map(pos1, pos2):
    pos1_expanded = pos1.unsqueeze(-2)
    pos2_expanded = pos2.unsqueeze(-3)

    # Calculate the displacement map using broadcasting
    displacement_map = pos1_expanded - pos2_expanded
    distance_map = torch.norm(displacement_map, dim=-1)

    return distance_map


""" rotation rep """

def set_rot_dim(args):
    if args.rotation_rep == 'quat':
        args.rot_dim = 4
    elif args.rotation_rep == 'R6':
        args.rot_dim = 6

def factor_out(input0, input1):
    from pymovis.motion.ops.torchmotion import R_to_R6, R6_to_R
    
    # root 
    root_pos0 = input0[..., -3:].clone()
    root_pos1 = input1[..., -3:].clone()
    root_R0 = R6_to_R(input0[..., 0:6]).clone()
    root_R1 = R6_to_R(input1[..., 0:6]).clone()
    inv_root_R0 = torch.linalg.inv(root_R0)
    inv_root_R1 = torch.linalg.inv(root_R1)
    
    # own root position seen from the partner
    input0[..., -3:] = torch.matmul(inv_root_R0, (root_pos1 - root_pos0).unsqueeze(-1)).squeeze(-1)
    input1[..., -3:] = torch.matmul(inv_root_R1, (root_pos0 - root_pos1).unsqueeze(-1)).squeeze(-1)
    return input0, input1


""" etc """
def swap_skeleton_of_motion(args, motionA, motionB, source0_name, source1_name):
    from datasets.character_functions import get_a_character_wo_geo_with_finger
    template_Tpose = bvh.load(
        "../Resource/Tpose_template.bvh", v_forward=[0, 0, 1], v_up=[0, 1, 0]
    )
    
    character0, Tpose0, _, skeleton_idx, finger_idx = \
        get_a_character_wo_geo_with_finger(args, source0_name, template_Tpose) # replace with wo finger
    motionA.skeleton = character0.meshes[0].source_skeleton
    
    character1, Tpose1, _, skeleton_idx, finger_idx = \
        get_a_character_wo_geo_with_finger(args, source1_name, template_Tpose) # replace with wo finger
    motionB.skeleton = character1.meshes[0].source_skeleton

    for i in range(len(motionB.poses)):
        motionA.poses[i].skeleton = motionA.skeleton
        motionB.poses[i].skeleton = motionB.skeleton
        motionA.poses[i].update()
        motionB.poses[i].update()
        
    return character0, character1, motionA, motionB
