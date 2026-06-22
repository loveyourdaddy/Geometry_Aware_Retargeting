import ast
import argparse

def get_args():
    parser = get_parser()
    return parser.parse_args()

def get_parser():
    parser = argparse.ArgumentParser()

    # train
    parser.add_argument('--proj_name', type=str, default='full')

    # test
    parser.add_argument('--test_proj',  type=str, default='240804_Gt1Root10Fk100Anchor10_footContact')
    parser.add_argument('--test_epoch', type=int, default=2000)
    
    # character
    parser.add_argument('--test_type', type=str, default="Mixamo") # SMPLx Mixamo
    parser.add_argument('--test_char', type=str, default="small") # small
    parser.add_argument('--role_change', type=str_to_bool, default=False)
    parser.add_argument('--subsampled', type=str_to_bool, default=False) # True
    
    """ setting """
    parser.add_argument('--save', type=str_to_bool, default=False)
    parser.add_argument('--debug', type=str_to_bool, default=False)
    parser.add_argument('--load_from_saved', type=str, default='')
    parser.add_argument('--path', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda') # cpu cuda
    parser.add_argument('--save_iter_epoch', type=int, default=500)

    """ network structure """
    # network_type: ablation 선택
    #   full          - SharingTransformer (self-attn + cross-attn, default)
    #   no_cross_attn - TwinTransformer    (self-attn only, cross-attn 제거)
    #   mlp           - TwinMLPEncoder     (attention 없음)
    parser.add_argument('--network_type', type=str, default='full',
                        choices=['full', 'no_cross_attn', 'mlp'])
    parser.add_argument('--weight_sharing', type=str_to_bool, default=True)
    parser.add_argument('--temporal_attn',  type=str_to_bool, default=False)
    parser.add_argument('--char_info', type=str, default="length")
    parser.add_argument('--normalize_char_info', type=str_to_bool, default=False)
    parser.add_argument('--data_normalized', type=str_to_bool, default=False)
    parser.add_argument('--target_characters', type=arg_as_list, default=["SMPLx", "SMPLx"]) # SMPLx_fat
    parser.add_argument('--motion0', type=str, default="")
    parser.add_argument('--motion1', type=str, default="")
    parser.add_argument('--SMPLx_scale', type=float, default=1.0)
    parser.add_argument('--SMPLx_mesh_scale', type=float, default=1.0)
    parser.add_argument('--train_one_chararacter_only', type=str_to_bool, default=False)

    """ lambda """
    # base loss
    parser.add_argument('--lambda_rec', type=float, default=1.0) # lambda1
    parser.add_argument('--lambda_anchor', type=float, default=10.0) # lambda2
    parser.add_argument('--lambda_root', type=float, default=10.0) # lambda3
    # lambda4(Rotation loss) 1
    parser.add_argument('--lambda_fk', type=float, default=100.0) # lambda 5
    parser.add_argument('--lambda_foot_contact', type=float, default=10.0) # lambda6 (additional)
    # fk
    parser.add_argument('--loss_fk', type=str_to_bool, default=True)
    # anchor loss
    parser.add_argument('--loss_anchor', type=str_to_bool, default=True)
    # foot contact loss
    parser.add_argument('--loss_foot_contact', type=str_to_bool, default=True)

    # settings
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--kp', type=float, default=0.9)
    parser.add_argument('--exp_k', type=float, default=3.0)
    parser.add_argument('--anchor_exp_k', type=float, default=3.0)
    parser.add_argument('--num_heads', type=int, default=4)

    """ data preprocess """
    parser.add_argument('--geo_preprocess', type=str_to_bool, default=False)
    parser.add_argument('--bvh_preprocess', type=str_to_bool, default=False)

    """ dataset representation """
    parser.add_argument('--rotation_rep', type=str, default='R6')
    parser.add_argument('--windowed_motion', type=str_to_bool, default=True)
    parser.add_argument('--seed_value', type=int, default=0)

    parser.add_argument('--window_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--sepearte_train_data', type=str_to_bool, default=False)
    parser.add_argument('--train_data_ratio', type=float, default=0.9)

    """ RD """
    parser.add_argument('--update_by_clampping_range', type=str_to_bool, default=True)
    parser.add_argument('--interaction_start_frame', type=int, default=-1)
    parser.add_argument('--interaction_end_frame', type=int, default=-1)
    parser.add_argument('--num_joint', type=int, default=22)
    parser.add_argument('--num_anchor_perjoint', type=int, default=4)

    """ joint setting """
    # part
    parser.add_argument('--root_joint', type=list, default=[0])
    parser.add_argument('--RD_spine_joints', type=list, default=[10, 11])
    parser.add_argument('--RD_hand_joints', type=list, default=[16, 17, 20, 21])
    parser.add_argument('--RD_leg_joints', type=list, default=[2, 3, 4, 6, 7, 8])

    # joints for augment scaling
    parser.add_argument('--leg_joints', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8]) # 12,13
    parser.add_argument('--body_joints', type=list, default=[0, 9, 10, 11, 12, 13])
    parser.add_argument('--hand_joints', type=list, default=[14, 15, 16, 17, 18, 19, 20, 21])
    parser.add_argument('--toe_joints', type=list, default=[4, 8])   # foot
    parser.add_argument('--heel_joints', type=list, default=[3, 7]) # heel

    parser.add_argument('--left_leg_joints', type=list,  default=[1, 2, 3, 4])
    parser.add_argument('--right_leg_joints', type=list, default=[5, 6, 7, 8])
    parser.add_argument('--left_hand_joints', type=list, default=[14, 15, 16, 17])
    parser.add_argument('--right_hand_joints', type=list, default=[18, 19, 20, 21])
    
    # ground pene # detect 
    parser.add_argument('--toe_pene_ths',  type=float, default=0.02)
    parser.add_argument('--heel_pene_ths', type=float, default=0.06) # 작을수록 강한 규제 # 클수록 변화하는게 보임

    ''' debug & render '''
    parser.add_argument('--source_pos', type=float, default=-4) # -2
    parser.add_argument('--joint_pos',  type=float, default=0) # 2
    parser.add_argument('--geo_pos',    type=float, default=6)
    parser.add_argument('--debug_points0', type=list, default=[])
    parser.add_argument('--debug_points1', type=list, default=[])
    parser.add_argument('--debug_points2', type=list, default=[])
    parser.add_argument('--debug_points3', type=list, default=[])
    parser.add_argument('--debug_points4', type=list, default=[])
    parser.add_argument('--align_motion_in_z_axis', type=str_to_bool, default=False)
    parser.add_argument('--debug_lines0', type=list, default=[])
    parser.add_argument('--debug_lines1', type=list, default=[])
    parser.add_argument('--debug_weight0', type=list, default=[])

    ''' adaptation '''
    parser.add_argument('--adapt_char', type=str, default='SMPLx')  # SMPLx, Mixamo

    return parser

def str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v


"""
12 anchors * 17 joints = 204 
13 anchors * 5 joints  = 65
269 = 204 + 65

SMPL
- Hips 0 0
    - LeftUpLeg 1 1
        - LeftLeg 2 2
            - LeftFoot 3 3
                - LeftToeBase 4 9 0
    # - RightUpLeg 5 4
        - RightLeg 6 5
            - RightFoot 7 6
                - RightToeBase 8 18 1
    - Spine 9 7
        - Spine1 10 8
            - Spine2 11 9
                - Neck 12 10
                    - Head 13 19 2
                - LeftShoulder 14 11
                    - LeftArm 15 12
                        - LeftForeArm 16 13
                            - LeftHand 17 20 3
                - RightShoulder 18 14
                    - RightArm 19 15
                        - RightForeArm 20 16
                            - RightHand 21 21 4
tipbone 22 23 
"""

bones = [
    "Hips",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
    "Spine",
    "Spine1",
    "Spine2",
    "Neck",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
]
