import sys
sys.path.append('..')

from pymovis.vis.appmanager import AppManager
from pymovis.vis.app import MyApp
from Network.network import Network
from datasets.character_functions import *
from datasets.motion_functions import *
from datasets.motion_dataset import *
import option_parser
from option_motion import example_bvh
from Retarget_SMPL.relationship_descriptor import resolve_ground_pene, get_rootP_localR_globalP_from_motion
from detect_foot_contact import *


def main(args):
    app_manager = AppManager()
    set_rot_dim(args)
    args.device = "cpu"
    args.is_train = False
    args.save_norm_info = False

    # motion 
    if args.motion0 == '':
        source0_motion_names = list(example_bvh.keys())
        source1_motion_names = list(example_bvh.values())
    else:
        source0_motion_names = [args.motion0]
        source1_motion_names = [args.motion1]

    print("> proj_name: ", args.test_proj)
    print("> character: ", args.test_type, args.test_char)
    print("> motion: ", source0_motion_names[0])


    """ load data """
    # load character 
    source0_character, source1_character, tgt0_character, tgt1_character, \
        targ0_Tpose, targ1_Tpose, source0_name, source1_name = \
        load_char(args)
    
    if args.test_type=="Mixamo":
        target0_skeleton_idx, target0_finger_idx, target1_skeleton_idx, target1_finger_idx = \
            get_skeleton_finger_idx(targ0_Tpose, targ1_Tpose)
    
    # load motion
    source_motion0 = get_interaction_motions_from_list(source0_name, source0_motion_names)[0]
    source_motion1 = get_interaction_motions_from_list(source1_name, source1_motion_names)[0]
    
    # translate input motion
    source_motion0, source_motion1 = input_motion_translate(args, source_motion0, source_motion1)
    
    # dataset
    dataset = Dataset(args)
    dataset.get_char_data(source0_character, source1_character, tgt0_character, tgt1_character)
    
    # motion
    dataset.get_input_motion(source_motion0, source_motion1)
    if args.data_normalized:
        dataset.load_norm_info()
        dataset.normalize()

    # swap role
    if args.role_change:
        target1_character, target0_character = tgt0_character, tgt1_character
        # offset 
        tmp = dataset.target_offsets1.clone()
        dataset.target_offsets1 = dataset.target_offsets0.clone()
        dataset.target_offsets0 = tmp.clone()
        
        tmp = dataset.target_aabb_max_min1.clone()
        dataset.target_aabb_max_min1 = dataset.target_aabb_max_min0.clone()
        dataset.target_aabb_max_min0 = tmp.clone()
        
        if args.test_type=="Mixamo":
            tmp0, tmp1 = target0_skeleton_idx, target0_finger_idx 
            target0_skeleton_idx, target0_finger_idx = target1_skeleton_idx, target1_finger_idx
            target1_skeleton_idx, target1_finger_idx = tmp0, tmp1
    else: 
        target0_character, target1_character = tgt0_character, tgt1_character
    
    
    # Network
    net = Network(args)
    net.load(args.test_proj + '/' , args.test_epoch, device=args.device)
    net.eval()

    # forward
    jit_output_p0, jit_output_R0, \
    jit_output_p1, jit_output_R1 = \
        net.forward(dataset)
    
    output_motion0, output_motion1 = \
        make_new_motions(args, jit_output_p0, jit_output_R0, 
                        jit_output_p1, jit_output_R1, 
                        target0_character, target1_character, 
                        source_motion0, source_motion1)
    
    # post processing
    # output_motion0, output_motion1 = \
    #     resolve_ground_pene(args, output_motion0, output_motion1)
    
    # both two characters retargeted 
    swap_data = True # False
    if swap_data:
        # dfm(1) fat, ptn(0) small
        args.test_char = "small"
        _, _, _, tgt0_character, \
            targ0_Tpose, targ1_Tpose, source0_name, source1_name = \
            load_char(args)
        
        if args.test_type=="Mixamo":
            target0_skeleton_idx, target0_finger_idx, target1_skeleton_idx, target1_finger_idx = \
                get_skeleton_finger_idx(targ0_Tpose, targ1_Tpose)
        
        
        # dataset
        dataset = Dataset(args)
        # 여기를 source character을 바꾸면, unseen source character가 될 것 같은데. 잘 되려나?
        dataset.get_char_data(source0_character, source1_character, tgt0_character, tgt1_character) # target0_character, target1_character,
        
        # motion
        dataset.get_input_motion(output_motion0, output_motion1) # source_motion0, source_motion1
        if args.data_normalized:
            dataset.load_norm_info()
            dataset.normalize() 
        
        # swap
        # import pdb; pdb.set_trace()
        import copy 
        target1_character_swap = copy.deepcopy(target1_character)
        target0_character_swap = copy.deepcopy(tgt0_character) # updated
        # offset
        tmp = dataset.target_offsets1.clone()
        dataset.target_offsets1 = dataset.target_offsets0.clone()
        dataset.target_offsets0 = tmp.clone()
        
        tmp = dataset.target_aabb_max_min1.clone()
        dataset.target_aabb_max_min1 = dataset.target_aabb_max_min0.clone()
        dataset.target_aabb_max_min0 = tmp.clone()
        
        if args.test_type=="Mixamo":
            tmp0, tmp1 = target0_skeleton_idx, target0_finger_idx 
            target0_skeleton_idx, target0_finger_idx = target1_skeleton_idx, target1_finger_idx
            target1_skeleton_idx, target1_finger_idx = tmp0, tmp1
        
        # forward
        jit_output_p0, jit_output_R0, \
        jit_output_p1, jit_output_R1 = \
            net.forward(dataset)
        
        output_motion0_swap, output_motion1_swap = \
            make_new_motions(args, jit_output_p0, jit_output_R0, 
                            jit_output_p1, jit_output_R1, 
                            target0_character_swap, target1_character_swap, 
                            source_motion0, source_motion1)
        
        # # post processing
        # output_motion0, output_motion1 = \
        #     resolve_ground_pene(args, output_motion0, output_motion1)
        
        # output_motion0_swap, output_motion1_swap = \
        #     resolve_ground_pene(args, output_motion0_swap, output_motion1_swap)
        
    
    # test with aura mesh
    if False:
        motion_name0 = source0_motion_names[0]
        jit_output_R0 = np.load('auramesh/{}_local_R0.npy'.format(motion_name0))
        jit_output_R1 = np.load('auramesh/{}_local_R1.npy'.format(motion_name0))
        
        jit_output_R0 = torch.tensor(jit_output_R0)
        jit_output_R1 = torch.tensor(jit_output_R1)
        
        source_output_p0 = []
        source_output_p1 = []
        source_output_R0 = []
        source_output_R1 = []
        for pose in source_motion0.poses:
            source_output_p0.append(pose.root_p)
            source_output_R0.append(pose.local_R)
        for pose in source_motion1.poses:
            source_output_p1.append(pose.root_p)
            source_output_R1.append(pose.local_R)
        source_output_p0 = torch.tensor(np.array(source_output_p0))
        source_output_p1 = torch.tensor(np.array(source_output_p1))
        source_output_p1[:, 1] *= source_output_p1[:, 1] * 0.7
        source_output_R0 = torch.tensor(np.array(source_output_R0))
        source_output_R1 = torch.tensor(np.array(source_output_R1))
        
        jit_output_R0[:, 0] = source_output_R0[:, 0]
        jit_output_R1[:, 0] = source_output_R1[:, 0]
        output_motion0, output_motion1 = \
            make_new_motions(args, 
                            source_output_p0, jit_output_R0, 
                            source_output_p1, jit_output_R1, 
                            target0_character, target1_character, 
                            source_motion0, source_motion1)

    # make motion 
    if args.test_type=="Mixamo":
        # finger가 없는데 필요한가? 
        # finger motion from source 
        jit_output_p0, jit_output_R0, _ = get_rootP_localR_globalP_from_motion(args, output_motion0.poses)
        jit_output_p1, jit_output_R1, _ = get_rootP_localR_globalP_from_motion(args, output_motion1.poses)
        output_motion0 = make_new_motion(jit_output_p0, jit_output_R0, target0_character, source_motion0)
        output_motion1 = make_new_motion(jit_output_p1, jit_output_R1, target1_character, source_motion1)
        
    # check foot contact preserving
    if False:
        src_foot_contact0 = detect_foot_contact(args, source_motion0)
        src_foot_contact1 = detect_foot_contact(args, source_motion1)
        tgt_foot_contact0 = detect_foot_contact(args, output_motion0)
        tgt_foot_contact1 = detect_foot_contact(args, output_motion1)
        
        foot_joints = args.toe_joints + args.heel_joints
        src_foot_contact0 = src_foot_contact0[:, foot_joints]
        src_foot_contact1 = src_foot_contact1[:, foot_joints]
        tgt_foot_contact0 = tgt_foot_contact0[:, foot_joints]
        tgt_foot_contact1 = tgt_foot_contact1[:, foot_joints]
        
        def get_contact_preserving(src_foot_contact, tgt_foot_contact):
            num_src_contact = np.sum(src_foot_contact)
            contact_preserving = np.logical_and(src_foot_contact, tgt_foot_contact)
            num_tgt_contact = np.sum(contact_preserving)
            return num_tgt_contact / num_src_contact
        ratio_contact_preseving0 = get_contact_preserving(src_foot_contact0, tgt_foot_contact0)
        ratio_contact_preseving1 = get_contact_preserving(src_foot_contact1, tgt_foot_contact1)
        print("ratio_contact_preseving0: ", ratio_contact_preseving0)
        print("ratio_contact_preseving1: ", ratio_contact_preseving1)
    

    """ option """
    # save
    if args.save:
        save_path = './result_saved/' + args.test_proj + '/'
        os.makedirs(save_path, exist_ok=True)
        
        target_name0 = target0_character.meshes[0].mesh_gl.name
        target_name1 = target1_character.meshes[0].mesh_gl.name
        name = source0_motion_names[0]+'_'+target_name0+'_'+target_name1+'/'
        os.makedirs(save_path+name, exist_ok=True)
        
        np.save(save_path+name+'jit_output_p0', jit_output_p0.detach().numpy())
        np.save(save_path+name+'jit_output_p1', jit_output_p1.detach().numpy())
        np.save(save_path+name+'jit_output_R0', jit_output_R0.detach().numpy())
        np.save(save_path+name+'jit_output_R1', jit_output_R1.detach().numpy())
    # render
    else:
        from etc.etc import render_result, render_compare
        if swap_data==False:
            characters, motions = \
                render_result(args, 
                            source0_character, source1_character, target0_character, target1_character, 
                            source_motion0, source_motion1, output_motion0, output_motion1) 
        else:
            characters, motions = \
                render_compare(args, 
                            source0_character, source1_character, target0_character, target1_character, target0_character_swap, target1_character_swap, 
                            source_motion0, source_motion1, output_motion0, output_motion1, output_motion0_swap, output_motion1_swap) 
            
        app = MyApp(characters, motions, args, net)
        app_manager.run(app)

def input_motion_translate(args, motion0, motion1):
    translate0 = np.array([0.0, 0, 0.0])
    translate1 = np.array([0.0, 0, 0.0])
    
    if motion0.name == "move_03_03_male_30fps":
        translate1 = np.array([+0.1, 0, -0.1])
    elif motion0.name == "one_leg_back_stretch_S1":
        translate1 = np.array([0, 0, +0.1])
    else:
        return motion0, motion1
    print("input motion1 translated in {}".format(translate1))
    
    # update motion
    for pose in motion0.poses:
        pose.root_p += translate0
        pose.update()
    for pose in motion1.poses:
        pose.root_p += translate1
        pose.update()
        
    return motion0, motion1

if __name__ == "__main__":
    args = option_parser.get_args()
    main(args)
