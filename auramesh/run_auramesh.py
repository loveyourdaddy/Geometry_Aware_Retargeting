import sys
# python을 실행시키는 path 기준: 
# python auramesh/run_auramesh.py 
sys.path.append('./')
sys.path.append('../') # Retargeting_workspace

from pymovis.vis.appmanager import AppManager
from pymovis.vis.app import MotionApp
from pymovis.vis.model import Model
from pymovis.vis.render import Render
from pymovis.motion.data import bvh
from pymovis.motion.data.fbx import FBX
from pymovis.motion.core import Pose, Motion
from pymovis.motion.ops.npmotion import *

# from datasets import *
from datasets.character_functions import get_a_character
from datasets.motion_functions import get_interaction_motions_from_list

# from Geometry.geometry import Geometry
from Geometry.auramesh import AuraMesh
from Geometry.compare_geometry import collision_detection, update_boundary_position
from Retarget_SMPL.relationship_descriptor import get_rootP_localR_globalP_from_motion, get_rootP_localR_globalP_from_numpy_motion

from optimize_funcitons import *
from typing import List, Optional

import option_parser
from option_motion import example_bvh
import copy 
import numpy as np
import torch
import copy


class MyApp(MotionApp):
    def __init__(self, model: List[Geometry], auramesh: List[AuraMesh], motion: Motion, args, collision=None, net=None):
        # self.model = [m.renderable_model() for m in model]
        self.model = model
        # self.auramesh = [m.renderable_model() for m in auramesh]
        if net is None:
            super().__init__(self.model, motion, args)
        else:
            super().__init__(self.model, motion, args, net)
            self.net = net
        print("Start render")
        self.auramesh_char_index = [1, 3]

    def render(self):
        super().render()
        
        # for idx, mesh in enumerate(self.auramesh):
        #     char_idx = self.auramesh_char_index[idx]
        #     mesh.set_pose_by_source(self.motions[char_idx].poses[self.frame])
        #     Render.model(mesh).set_albedo([1, 0.5, 0.5]).set_all_alphas(0.5).draw()

""" this code run in CPU """
if __name__ == "__main__":
    app_manager = AppManager()
    args = option_parser.get_args()
    args.path = args.proj_name + '/'
    args.device = 'cpu'
    small = True
    if small:
        scale = 0.7
    else:
        scale = 1
    path = "./auramesh/"


    """ source """
    # load character
    src_names = ["SMPLx", "SMPLx"]
    src_chars = []
    for name in src_names:
        char, _, _ = get_a_character(args, name)
        src_chars.append(char)

    # load motion data (motion_0, motion_1)
    motion_name0 = list(example_bvh.keys())[0]
    motion_name1 = list(example_bvh.values())[0]
    motion_0 = get_interaction_motions_from_list(src_names[0], [motion_name0])[0]
    motion_1 = get_interaction_motions_from_list(src_names[0], [motion_name1])[0]
    
    # motion clap
    clap = 380 # 330
    motion_0.poses = motion_0.poses[:clap]
    motion_1.poses = motion_1.poses[:clap]
    
    # set source skeleton
    src_chars[0].set_source_skeleton(motion_0.skeleton, "")
    src_chars[1].set_source_skeleton(motion_1.skeleton, "")
    
    # load geo
    src_geoms = get_character_geometry(args, src_names, src_chars)
    src_geoms[0].source_skeleton = motion_0.skeleton
    src_geoms[1].source_skeleton = motion_1.skeleton
    
    # AuraMesh
    src_auramesh = []
    for geom, name in zip(src_geoms, src_names):
        src_auramesh.append(AuraMesh(args, geom, name))

    # set geometry of auramesh
    root_p0, local_R0, _ = get_rootP_localR_globalP_from_motion(args, motion_0.poses)
    root_p1, local_R1, _ = get_rootP_localR_globalP_from_motion(args, motion_1.poses)
    src_geoms[0].set_pose_by_source_batch_frame(local_R0.unsqueeze(0), root_p0.unsqueeze(0))
    src_geoms[1].set_pose_by_source_batch_frame(local_R1.unsqueeze(0), root_p1.unsqueeze(0))
    src_auramesh[0].set_pose_by_source_batch_frame(local_R0.unsqueeze(0), root_p0.unsqueeze(0))
    src_auramesh[1].set_pose_by_source_batch_frame(local_R1.unsqueeze(0), root_p1.unsqueeze(0))


    """ Collision detection """
    # Collision detection in src
    col_det = False
    if col_det:
        print("Start Collision detection in src")
        import time
        time0 = time.time()
        
        """ col det """
        geo_cids0, auramesh_cids1, jids0, jids1, col_frame = collision_detection(args, src_geoms[0], src_auramesh[1], motion_0, motion_1)
        torch.save(cids0, path+'pt/cids0.pt')
        torch.save(cids1, path+'pt/cids1.pt')
        torch.save(jids0, path+'pt/jids0.pt')
        torch.save(jids1, path+'pt/jids1.pt')
        torch.save(col_frame, path_+'pt/col_frame.pt')
        time1 = time.time()
        print("time: ", time1-time0)
    else:
        tgt_geo_cids0 = torch.load(path+'pt/cids0.pt')
        tgt_auramesh_cids1 = torch.load(path+'pt/cids1.pt')
        jids0 = torch.load(path+'pt/jids0.pt')
        jids1 = torch.load(path+'pt/jids1.pt')
        col_frame = torch.load(path+'pt/col_frame.pt')
        col_frame = np.array(col_frame)
        print("load collision detection results")

    """ target """
    # target character
    tgt_names = ["SMPLx", "SMPLx"]
    tgt_chars = []
    for i, name in enumerate(tgt_names):
        if i ==0:
            char, _, _ = get_a_character(args, name)
        else:
            char, _, _ = get_a_character(args, name, mesh_scale=scale)
        tgt_chars.append(char)
    
    # target motion: zero rot
    tgt_motion_0 = copy.deepcopy(motion_0)
    tgt_motion_1 = copy.deepcopy(motion_1)
    for pose in tgt_motion_0.poses:
        pose.local_R = np.identity(3)[None, :].repeat(22, axis=0).astype(np.float32)
        pose.update()
    for pose in tgt_motion_1.poses:
        # pose.local_R = np.identity(3)[None, :].repeat(22, axis=0).astype(np.float32)
        pose.root_p[1] *= scale
        pose.update()

    # set skeleton
    tgt_chars[0].set_source_skeleton(tgt_motion_0.skeleton, "")
    tgt_chars[1].set_source_skeleton(tgt_motion_1.skeleton, "")

    # scale
    if small:
        leg_scale, body_scale, hand_scale = scale, scale, scale
        root_scale = leg_scale
        from Retarget_SMPL.retarget_smpl import scale_character 
        scale_character(args, tgt_chars[1], leg_scale, body_scale, hand_scale)

    # load geo
    tgt_geoms = get_character_geometry(args, tgt_names, tgt_chars)
    tgt_auramesh = []
    for i in range(len(tgt_geoms)):
        auramesh = AuraMesh(args, tgt_geoms[i], tgt_names[i])
        if i==0:
            motion = tgt_motion_0
        else:
            motion = tgt_motion_1
            
        root_p, local_R, _ = get_rootP_localR_globalP_from_numpy_motion(args, motion.poses)
        auramesh.set_pose_by_source_batch_frame(local_R.unsqueeze(0), root_p.unsqueeze(0))
        tgt_auramesh.append(auramesh)

    """ optimize """
    optimize = True
    if optimize:
        # Get vpositions for cid1
        tgt_geo_vids0, tgt_auramesh_vids1 = [], []
        num_col = len(tgt_geo_cids0)
        for i in range(num_col):
            # geo of 0
            vids0 = tgt_geoms[0].cid_to_first_vid[tgt_geo_cids0[i]]
            tgt_geo_vids0.append(vids0)

            # auramesh of 1
            vids1 = src_auramesh[1].cid_to_first_vid[tgt_auramesh_cids1[i]]
            tgt_auramesh_vids1.append(vids1)
            
        
        """ optimize tgt motion0 from col vpos of tgt motion1 """
        
        # # update motion1 (main)
        # tgt_motion_0 = optimize_motion(args,
        #     motion_0, 
        #     tgt_motion_0, tgt_motion_1, tgt_auramesh[0], tgt_auramesh[1],
        #     src_col_vids0, auramesh_col_vids1,
        #     col_frame, jids0, jids1)
        
        # to remove
        for f, pose in enumerate(tgt_motion_1.poses):
            pose.local_R = motion_1.poses[f].local_R
            pose.update()
        
        import time
        time0 = time.time()
        
        # update motion0 (ptn)
        tgt_motion_0 = optimize_motion(args,
            motion_0, tgt_motion_0, 
            tgt_geoms[0], tgt_auramesh[1],
            tgt_geo_vids0, tgt_auramesh_vids1,
            col_frame, jids0, jids1)
        
        time1 = time.time()
        print("time: ", time1-time0)
        
        save = True
        if save:
            local_Rs = []
            for i, pose in enumerate(tgt_motion_0.poses):
                local_Rs.append(pose.local_R)
            np.save(path+"{}_local_Rs0.npy".format(motion_name0), np.array(local_Rs))
            
            local_Rs = []
            for i, pose in enumerate(tgt_motion_1.poses):
                local_Rs.append(pose.local_R)
            np.save(path+"{}_local_Rs1.npy".format(motion_name1), np.array(local_Rs))
    else:
        # load 
        local_Rs0 = np.load(path+"{}_local_Rs0.npy".format(motion_name0))
        for i, pose in enumerate(tgt_motion_0.poses):
            pose.local_R = local_Rs0[i]
            pose.root_p[1] = pose.root_p[1]*0.7
            pose.update()
            
        local_Rs1 = np.load(path+"{}_local_Rs1.npy".format(motion_name0))
        for i, pose in enumerate(tgt_motion_1.poses):
            pose.local_R = local_Rs1[i]
            pose.root_p[1] = pose.root_p[1]*0.7
            pose.update()
    
    """ interpolate """
    # i=1 부터 시작해서, i-1과 quaternion interpolation
    # 일단 slerp 하지 않고 뒤에 값을 그대로 써보기. 
    # 79 이전, 264 이후는 변화 없을꺼야.

    smooth = False
    if smooth:
        numbers = col_frame
        # numbers = [79, 93, 94, 105, 117, 119, 125, 130, 140, 152, 153, 154, 155, 156, 157, 159, 160, 161, 170, 180, 189, 206, 210, 220, 235, 245, 256, 259, 260, 261, 262, 263, 264]
        
        # updated frame in optimization
        # motion0
        motion = copy.deepcopy(tgt_motion_0) # 튐이 있는 모션 
        for i in range(1, len(numbers)):
            curr_f = numbers[i]
            prev_f = numbers[i-1]
            
            prev_quat = R_to_Q(motion.poses[prev_f].local_R) # q2 
            curr_quat = R_to_Q(motion.poses[curr_f].local_R) # q2 
            prev_range = curr_f - prev_f
            # prev 
            for j in range(1, prev_range+1):
                curr_val = 1 - j/prev_range # t2
                interp_quat = quaternion_slerp(prev_quat, curr_quat, curr_val) 
                smooth_local_R = Q_to_R(interp_quat)
                tgt_motion_0.poses[curr_f-j].local_R = smooth_local_R
                tgt_motion_0.poses[curr_f-j].update() # smoothing 된 모션 
        
        # motion1
        motion = copy.deepcopy(tgt_motion_1) # 튐이 있는 모션 
        for i in range(1, len(numbers)):
            curr_f = numbers[i]
            prev_f = numbers[i-1]
            
            prev_quat = R_to_Q(motion.poses[prev_f].local_R) # q2 
            curr_quat = R_to_Q(motion.poses[curr_f].local_R) # q2 
            prev_range = curr_f - prev_f
            # prev 
            for j in range(1, prev_range+1):
                curr_val = 1 - j/prev_range # t2
                interp_quat = quaternion_slerp(prev_quat, curr_quat, curr_val) 
                smooth_local_R = Q_to_R(interp_quat)
                tgt_motion_1.poses[curr_f-j].local_R = smooth_local_R
                tgt_motion_1.poses[curr_f-j].update() # smoothing 된 모션 


    """ save """
    # local_Rs0 = []
    # root_ps0 = []
    # for pose in tgt_motion_0.poses:
    #     local_Rs0.append(pose.local_R)
    #     root_ps0.append(pose.root_p)
    # local_Rs1 = []
    # root_ps1 = []
    # for pose in tgt_motion_1.poses:
    #     local_Rs1.append(pose.local_R)
    #     root_ps1.append(pose.root_p)
    # np.save('auramesh/{}_local_R0.npy'.format(motion_name0), np.array(local_Rs0))
    # np.save('auramesh/{}_root_p0.npy' .format(motion_name0), np.array(root_ps0))
    # np.save('auramesh/{}_local_R1.npy'.format(motion_name0), np.array(local_Rs1))
    # np.save('auramesh/{}_root_p1.npy' .format(motion_name0), np.array(root_ps1))


    """ render """
    # translate 
    for f in range(len(motion_0.poses)):
        motion_0.poses[f].translate_root_p([args.source_pos, 0, 0])
        motion_1.poses[f].translate_root_p([args.source_pos, 0, 0])
        tgt_motion_0.poses[f].translate_root_p([args.joint_pos, 0, 0])
        tgt_motion_1.poses[f].translate_root_p([args.joint_pos, 0, 0])
    
    # render 
    chars =[]
    aurameshes = []
    motions = []
    
    # src 
    for char in src_chars:
        chars.append(char)
    aurameshes.append(src_auramesh[1])
    motions.append(motion_0)
    motions.append(motion_1)
    
    # target 
    for char in tgt_chars:
        chars.append(char)
    aurameshes.append(tgt_auramesh[1])
    motions.append(tgt_motion_0)
    motions.append(tgt_motion_1)

    # render 
    app = MyApp(chars, aurameshes, motions, args)
    app_manager.run(app)
    