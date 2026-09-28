# save_offset.py
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from pymovis.vis.appmanager import AppManager
from pymovis.vis.app import MyApp
from datasets.motion_functions import *
from datasets.character_functions import *
import option_parser
from pymovis.motion.ops.npmotion import *
from etc.etc import *
from pymovis.motion.data.fbx import FBX
from Geometry.geometry import Geometry


def get_aabb_length(geometry, motion):
    local_R = torch.tensor(motion.poses[0].local_R).reshape(1,1,22,3,3).to(args.device)
    root_p = torch.tensor(motion.poses[0].root_p).reshape(1,1,3).to(args.device)
    geometry.set_pose_by_source_batch_frame(local_R, root_p)

    len_vids = len(geometry.vid_to_cid)
    batch = torch.tensor([0]).reshape(1,1,1).repeat(1,1,len_vids).to(args.device)
    frame = torch.tensor([0]).reshape(1,1,1).repeat(1,1,len_vids).to(args.device)

    # get position
    vids = torch.arange(0, len(geometry.vid_to_cid))
    vids = vids.reshape(1,1,-1).to(args.device)
    v_position = geometry.get_positions_from_vids(vids, batch, frame)[0,0] # slow 
    cid_to_vids, vid_to_cid, cid_to_first_vid = geometry._get_cid_vid()
    skinning_jids, skinning_weights = geometry._get_skinning(cid_to_vids)
    vid_to_jid, cid_to_jid, jid_to_vids, jid_to_cids, jid_to_vpositions = geometry._get_joint_data(skinning_jids, skinning_weights, v_position, vid_to_cid, cid_to_vids)

    """ 6. anchor vid for each joint"""
    jid_to_vposition_max = np.stack([np.max(j2v, axis=0) for j2v in jid_to_vpositions], axis=0)
    jid_to_vposition_min = np.stack([np.min(j2v, axis=0) for j2v in jid_to_vpositions], axis=0)
    aabb_length = jid_to_vposition_max - jid_to_vposition_min
    
    return jid_to_vposition_max, jid_to_vposition_min, aabb_length


# this code is for network input
app_manager = AppManager()
args = option_parser.get_args()
args.path = args.proj_name + '/'
args.geo_preprocess = False 
args.bvh_preprocess = False 
args.device = "cpu"

parent_path = "../0_geometry_aware_retargeting/"
template_Tpose = bvh.load("../Resource/Tpose_template.bvh", v_forward=[0, 0, 1],v_up=[0, 1, 0])
characters = []
motions = [] 

if False:
    # target_character = args.target_characters[0]
    for character_name in args.target_characters:
        # random sampling
        import random
        num_scale = 10
        start = 0.7
        end = 1.2
        num_part = 3
        scales = []
        for i in range(num_scale):
            scale = []
            for i in range(num_part):
                scale.append(round(random.uniform(start, end), 2))
            scales.append(scale)
        np.save("scales_sampled.npy", scales)
        # scales.append([start,start,start])
        # old_scale = np.load("scales_sampled.npy")
        # for scale in old_scale:
        #     scales.append(scale)
        # scales.append([end,end,end])

""" scaling model for trainig """
if True:
    # scale character
    scale = 1
    scales = [] # 1.2 0.6, 0.5, 0.4 
    if len(scales)!=0:
        args.target_characters = ["SMPLx",] * len(scales) # SMPLx_fat
    
    # scale mesh only 
    mesh_scale = 1
    mesh_scales = [0.8] # 1.0, 0.6, 0.5, 0.4 
    if len(mesh_scales)!=0:
        args.target_characters = ["SMPLx_fat",] * len(mesh_scales) # SMPLx_fat
    
    # save offset / aabb 
    for i, character_name in enumerate(args.target_characters):
        scaled_offsets = []
        scaled_aabb_max_min = []

        # motion_name0 = ["Tpose"]
        # motion = get_single_motion_from_list(character_name, motion_name0)[0]
        # motion = refine_motion(motion, template_Tpose)
        
        if len(scales)!=0:
            scale = scales[i]
        if len(mesh_scales)!=0:
            mesh_scale = mesh_scales[i]
        character, motion, geometry = \
            get_a_smpl_character(args, args.target_characters[i], scale=scale, mesh_scale=mesh_scale)
        # if i==0:
        #     character, _, geometry = \
        #         get_a_smpl_character(args, args.target_characters[0])
        # elif i==1:
        #     # args.SMPLx_mesh_scale = SMPLx_mesh_scale
        #     character, _, geometry = \
        #         get_a_smpl_character(args, args.target_characters[0], mesh_scale=args.SMPLx_mesh_scale)
        
        character.set_source_skeleton(motion.skeleton, "") # MIXAMO_BVH_TO_FBX
        characters.append(character)
        motions.append(motion)
        
        
        """ scale """
        leg_scale  = scale
        body_scale = scale
        arm_scale  = scale

        # skeleton
        for joint in args.leg_joints:
            motion.skeleton.joints[joint].offset *= leg_scale
        for joint in args.body_joints:
            motion.skeleton.joints[joint].offset *= body_scale
        for joint in args.hand_joints:
            motion.skeleton.joints[joint].offset *= arm_scale
        
        # root and update
        pose = motion.poses[0]
        pose.root_p[1] *= leg_scale
        for j in range(22):
            motion.poses[0].local_R[j] = torch.tensor(np.eye(3))
        motion.poses[0].update()
        motion.poses[0] = pose
        
        
        """ save """
        # save offset 22,3
        offset = []
        for j in range(22):
            offset.append(motion.skeleton.joints[j].offset)
        offset = np.array(offset)
        scaled_offsets.append(offset)
        
        # save aabb: length 22,3
        # geometry = Geometry(args, character, motion)
        local_R = torch.tensor(motion.poses[0].local_R).reshape(1,1,22,3,3)
        root_p = torch.tensor(motion.poses[0].root_p).reshape(1,1,3)
        geometry.set_pose_by_source_batch_frame(local_R, root_p)
        
        jid_to_vposition_max, jid_to_vposition_min, aabb_length = get_aabb_length(geometry, motion)
        scaled_aabb_max_min.append(np.concatenate((jid_to_vposition_max, jid_to_vposition_min), axis=-1))
        
        if i==0:
            args.debug_points0 = np.concatenate((jid_to_vposition_max, jid_to_vposition_min), axis=0)
        elif i==1:
            args.debug_points1 = np.concatenate((jid_to_vposition_max, jid_to_vposition_min), axis=0)
        
        # save
        # np.save("{}/preprocess/{}/scaled_offset.npy".format(parent_path, character_name), np.array(scaled_offsets))
        # np.save("{}/preprocess/{}/scaled_aabb_max_min.npy".format(parent_path, character_name), np.array(scaled_aabb_max_min))
        # np.save("./preprocess/{}/scaled_global_p.npy".format(character_name), np.array(scaled_global_p))
        
        # skeleton scalec
        # _trace()
        if scale != 1:
            np.save("{}/preprocess/{}/scale{}_offset.npy".format(parent_path, character_name, scale), np.array(scaled_offsets))
            np.save("{}/preprocess/{}/scale{}_aabb_max_min.npy".format(parent_path, character_name, scale), np.array(scaled_aabb_max_min))
            print(f"saved {scale} scale")

        # mesh scale
        if mesh_scale != 1:
            np.save("{}/preprocess/{}/scale_mesh{}_offset.npy".format(parent_path, character_name, mesh_scale), np.array(scaled_offsets))
            np.save("{}/preprocess/{}/scale_mesh{}_aabb_max_min.npy".format(parent_path, character_name, mesh_scale), np.array(scaled_aabb_max_min))
            print(f"saved {mesh_scale} mesh scale")

# regular models
if False: 
    offsets = []
    # global_ps = []
    aabb_max_mins = []
    character_names = ["Remy"] # , "Ybot", "Amy", "Ortiz","Leonard" , "SMPLx_fat"
    for character_name in character_names: # args.target_characters
        # load character, motion
        if character_name =="SMPLx" or character_name =="SMPLx_sub" or character_name =="SMPLx_fat" or character_name =="SMPLx_fat_sub":
            dict = ""
        else:
            dict = MIXAMO_BVH_TO_FBX

        # char 
        path = "Resource/models/{}.fbx".format(character_name)
        fbx = FBX(path, character_name)
        character = fbx.model()

        # motion 
        motion_name = ["Tpose"]
        motion = get_interaction_motions_from_list(character_name, motion_name)[0]
        motion = refine_motion(motion, template_Tpose)
        character.set_source_skeleton(motion.skeleton, dict)
        motion.poses = motion.poses[:1]
        
        """ save offset """
        # offset 
        # pose = motion.poses[0]
        # global_p = []
        # for i in range(22):
        #     global_p.append(pose.global_p[i])
        # global_p = np.array(global_p)
        
        offset = []
        for i in range(22):
            offset.append(character.meshes[0].source_skeleton.joints[i].offset)
        offsets.append(offset)

        """ aabb """
        geometry = Geometry(args, character, motion)
        jid_to_vposition_max, jid_to_vposition_min, aabb_length = get_aabb_length(geometry, motion)
        aabb_max_min = np.concatenate((jid_to_vposition_max, jid_to_vposition_min), axis=-1)

        # save
        np.save("./preprocess/{}/offset.npy".format(character_name), np.array(offset))
        # np.save("./preprocess/{}/global_p.npy".format(character_name), np.array(global_p))
        np.save("./preprocess/{}/aabb_max_min.npy".format(character_name), np.array(aabb_max_min))

# save normalization info
if False:
    offsets = []
    global_ps = []
    aabb_max_mins = []
    
    character_names = ["SMPLx", "SMPLx_fat", "Ybot", "Amy", "Ortiz", "Leonard"] 
    if True:
        for name in character_names:
            global_p = np.load("./preprocess/{}/global_p.npy".format(name))
            aabb_max_min = np.load("./preprocess/{}/aabb_max_min.npy".format(name))
            offset = np.load("./preprocess/{}/offset.npy".format(name))
            global_ps.append(global_p)
            aabb_max_mins.append(aabb_max_min)
            offsets.append(offset)
        
        # # scaled 
        # character_names = ["SMPLx", "SMPLx_fat"] # 
        # for name in character_names:
        #     global_p = np.load("./preprocess/{}/scaled_global_p.npy".format(name)) # offset 
        #     aabb_max_min = np.load("./preprocess/{}/scaled_aabb_max_min.npy".format(name)) # aabb  
        #     scaled_offset = np.load("./preprocess/{}/scaled_offset.npy".format(name)) # offset 
        #     for i in range(len(aabb_max_min)): # CHECK 27
        #         global_ps.append(global_p[i])
        #         offsets.append(scaled_offset[i])
        #         aabb_max_mins.append(aabb_max_min[i])

    import numpy as np
    if True:
        # offset 
        offsets = torch.tensor(np.array(offsets))
        char_info = offsets
        mean = torch.mean(char_info, dim=(0,1))
        var  = torch.var( char_info, dim=(0,1))
        np.save('./preprocess/offset_char_info_mean.npy', mean.cpu().numpy())
        np.save('./preprocess/offset_char_info_var.npy', var.cpu().numpy())
        
        # offset length 
        offsets = torch.tensor(np.array(offsets))
        char_info = torch.norm(offsets, dim=-1) 
        mean = torch.mean(char_info, dim=(0,1))
        var  = torch.var( char_info, dim=(0,1))
        np.save('./preprocess/offset_length_char_info_mean.npy', mean.cpu().numpy())
        np.save('./preprocess/offset_length_char_info_var.npy', var.cpu().numpy())
        
        # aabb length
        aabb_max_mins = torch.tensor(np.array(aabb_max_mins))
        char_info = aabb_max_mins[..., :3] - aabb_max_mins[..., 3:]
        mean = torch.mean(char_info, dim=(0,1))
        var  = torch.var( char_info, dim=(0,1))
        np.save('./preprocess/aabb_length_char_info_mean.npy', mean.cpu().numpy()) # aabb_maxmin_char_info_mean
        np.save('./preprocess/aabb_length_char_info_var.npy', var.cpu().numpy())
    
        # position
        # global_p
        global_ps = torch.tensor(np.array(global_ps))
        char_info = global_ps
        mean = torch.mean(char_info, dim=(0,1))
        var  = torch.var( char_info, dim=(0,1))
        np.save('./preprocess/global_p_char_info_mean.npy', mean.cpu().numpy())
        np.save('./preprocess/global_p_char_info_var.npy', var.cpu().numpy())
        
        # aabb max min position
        char_info = aabb_max_mins
        mean = torch.mean(char_info, dim=(0,1))
        var  = torch.var( char_info, dim=(0,1))
        np.save('./preprocess/aabb_maxmin_char_info_mean.npy', mean.cpu().numpy())
        np.save('./preprocess/aabb_maxmin_char_info_var.npy', var.cpu().numpy())

# # Arrange
# distance = 2.0
# # for i in range(len(global_ps)):

render_motions(args, characters, motions)
app = MyApp(characters, motions, args)
app_manager.run(app)
