import sys
sys.path.append('..')

from pymovis.vis.appmanager import AppManager
from pymovis.vis.app import MyApp
from Network.network import Network
from datasets.character_dataset import *
from datasets.motion_dataset import *
import option_parser
from option_motion import example_bvh
from Retarget_SMPL.relationship_descriptor import resolve_ground_pene, get_rootP_localR_globalP_from_motion


def motion_translate(motion0, motion1):
    translate0 = np.array([0.0, 0, 0.0])
    translate1 = np.array([0.0, 0, 0.0])
    if motion0.name == "move_03_03_male_30fps":
        translate1 = np.array([+0.1, 0, -0.1])
        
    elif motion0.name == "one_leg_back_stretch_S1":
        translate1 = np.array([0, 0, +0.1])
    
    # elif motion0.name=="greeting002_S1":
    #     translate1 = np.array([0, 0.05, 0])
    #     for pose in motion0.poses:
    #         pose.root_p[1] += 0.05
    #         pose.update()
    #     for pose in motion1.poses:
    #         pose.root_p[1] += 0.05
    #         pose.update()
    else:
        return motion0, motion1
        
    for pose in motion0.poses:
        pose.root_p += translate0
        pose.update()
    for pose in motion1.poses:
        pose.root_p += translate1
        pose.update()
        
    return motion0, motion1

def main(args):
    app_manager = AppManager()
    set_rot_dim(args)
    args.device = "cpu"
    args.is_train = False
    args.save_norm_info = False

    # SMPLx Mixamo
    # args.test_type = "Mixamo"  # Mixamo
    # normal small fat
    # args.test_char = "fat"
    
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
    source0_character, source1_character, targ0_character, targ1_character, \
    targ0_Tpose, targ1_Tpose, source0_name, source1_name = \
        load_char(args)
    
    if args.test_type=="Mixamo":
        target0_skeleton_idx, target0_finger_idx, target1_skeleton_idx, target1_finger_idx = \
            get_skeleton_finger_idx(targ0_Tpose, targ1_Tpose)
    
    # load motion
    source_motion0 = get_motions_from_list(source0_name, source0_motion_names)[0]
    source_motion1 = get_motions_from_list(source1_name, source1_motion_names)[0]
    
    # translate
    source_motion0, source_motion1 = motion_translate(source_motion0, source_motion1)
    
    # dataset
    dataset = Dataset(args)
    dataset.get_char_data(source0_character, source1_character, targ0_character, targ1_character)
    # motion
    dataset.get_input_motion(source_motion0, source_motion1)
    if args.data_normalized:
        dataset.load_norm_info()
        dataset.normalize()

    # swap rolearget character의 set skeleton이 되어야함
    if args.role_change:
        target1_character, target0_character = targ0_character, targ1_character
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
        target0_character, target1_character = targ0_character, targ1_character
    
    """ make motion """
    net = Network(args)
    net.load(args.test_proj + '/' , args.test_epoch, device=args.device)
    
    # test with aura mesh
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
    
    # # post process
    # for pose in output_motion0.poses:
    #     pose.root_p[1] += 0.01
    #     pose.update()
    # for pose in output_motion1.poses:
    #     pose.root_p[1] += 0.01
    #     pose.update()
    # output_motion0, output_motion1 = \
    #     resolve_ground_pene(args, output_motion0, output_motion1)

    # auramesh
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
        # finger motion from source 
        jit_output_p0, jit_output_R0, _ = get_rootP_localR_globalP_from_motion(args, output_motion0.poses)
        jit_output_p1, jit_output_R1, _ = get_rootP_localR_globalP_from_motion(args, output_motion1.poses)
        output_motion0 = make_new_motion(jit_output_p0, jit_output_R0, target0_character, source_motion0)
        output_motion1 = make_new_motion(jit_output_p1, jit_output_R1, target1_character, source_motion1)
    
    # draw lines for anchors
    if False:
        f = 90
        def get_anchor_position(geo, motion):
            root_p, local_R, _ = get_rootP_localR_globalP_from_motion(args, motion.poses)
            geo.set_pose_by_source_batch_frame(local_R[None, ...], root_p[None, ...])
            vpos = geo.get_positions_from_vids(torch.tensor(np.array(geo.anchor_vids)), torch.tensor([0]), torch.tensor([f]))
            return vpos
        
        # source_anchor_vpos0 = get_anchor_position(source_geo0, source_motion0)
        # source_anchor_vpos1 = get_anchor_position(source_geo1, source_motion1)
        # global_ps0 =[]
        # global_ps1 =[]
        # global_ps0 = (source_motion0.poses[f].global_p)
        # global_ps1 = (source_motion1.poses[f].global_p)
        # source_anchor_vpos0 = torch.tensor(np.array(global_ps0))
        # source_anchor_vpos1 = torch.tensor(np.array(global_ps1))
        # source_anchor_vpos0 = get_anchor_position(source_geo0, source_motion0)
        # source_anchor_vpos1 = get_anchor_position(source_geo1, source_motion1)
        target_anchor_vpos0 = get_anchor_position(target_geo0, output_motion0)
        target_anchor_vpos1 = get_anchor_position(target_geo1, output_motion1)
        args.debug_points0 = target_anchor_vpos0
        args.debug_points1 = target_anchor_vpos1
        
        source_anchor_vpos0 = target_anchor_vpos0
        source_anchor_vpos1 = target_anchor_vpos1
        
        args.debug_lines0 = []
        args.debug_lines1 = []
        # 88, 88, 3
        for i in range(len(source_anchor_vpos0)):
            debug_lines0 = [] 
            debug_lines1 = [] 
            for j in range(len(source_anchor_vpos1)):
                debug_lines0.append(source_anchor_vpos0[i])
                debug_lines1.append(source_anchor_vpos1[j])
            args.debug_lines0.append(torch.stack(debug_lines0))
            args.debug_lines1.append(torch.stack(debug_lines1))
        args.debug_lines0 = torch.stack(args.debug_lines0)
        args.debug_lines1 = torch.stack(args.debug_lines1)
        
        from compare_quantity import exp_weight_of_distance
        weight = exp_weight_of_distance(args, torch.abs(args.debug_lines0 - args.debug_lines1))
        weight = np.linalg.norm(weight, axis=-1)
        eps = 1e-6
        weight = np.where(weight < eps, eps, weight)
        min_ele = np.min(weight)
        weight -= min_ele
        weight /= np.max(weight)
        
        
        # masked_weights = np.where(weight > 0.5, weight, 0)
        index = np.where(weight < 0.5)
        # 0.5 초과 값들을 0~1로 스케일링
        min_val = 0.5
        max_val = 1.0
        scaled_weights = (weight - min_val) / (max_val - min_val)
        scaled_weights[index] = 0
        
        # filtered_weights = weight[weight > 0.5]
        # # 0.5 초과 값들을 0~1로 스케일링
        # min_val = 0.5
        # max_val = 1.0
        # scaled_weights = (filtered_weights - min_val) / (max_val - min_val)
        args.debug_weight0 = scaled_weights
    
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
        # print('saved: ' + save_path+name)
    else: # render
        from etc.etc import render_result
        # from option_motion import align_motion_in_z_axis
        
        # if source0_motion_names[0] in align_motion_in_z_axis:
        #     characters, motions = \
        #         render_result(args, 
        #                     source0_character, source1_character, target0_character, target1_character, 
        #                     source_motion0, source_motion1, output_motion0, output_motion1, align_z_direction=True)
        # else:
        characters, motions = \
            render_result(args, 
                        source0_character, source1_character, target0_character, target1_character, 
                        source_motion0, source_motion1, output_motion0, output_motion1) 
        # characters = [target0_character, target1_character]
        # motions    = [output_motion0, output_motion1]
        # characters = [source0_character, source1_character]
        # motions    = [source_motion0, source_motion1]
        app = MyApp(characters, motions, args, net)
        app_manager.run(app)

if __name__ == "__main__":
    args = option_parser.get_args()
    main(args)
