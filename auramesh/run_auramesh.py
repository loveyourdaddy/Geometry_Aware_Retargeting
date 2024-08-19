import sys
# python을 실행시키는 path 기준: 
# python auramesh/run_auramesh.py 
sys.path.append('.')
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
from Retarget_SMPL.relationship_descriptor import get_rootP_localR_globalP_from_motion

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
        self.auramesh = [m.renderable_model() for m in auramesh]
        if net is None:
            super().__init__(self.model, motion, args)
        else:
            super().__init__(self.model, motion, args, net)
            self.net = net
        print("Start render")
        self.auramesh_char_index = [1, 3]

    def render(self):
        super().render()
        
        for idx, mesh in enumerate(self.auramesh):
            char_idx = self.auramesh_char_index[idx]
            mesh.set_pose_by_source(self.motions[char_idx].poses[self.frame])
            Render.model(mesh).set_albedo([1, 0.5, 0.5]).set_all_alphas(0.5).draw()

""" this code run in CPU """
if __name__ == "__main__":
    app_manager = AppManager()
    args = option_parser.get_args()
    args.path = args.proj_name + '/'
    args.device = 'cpu'
    small = True
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


    """ target """
    # target character
    tgt_names = ["SMPLx", "SMPLx"]
    tgt_chars = []
    for name in tgt_names:
        char, _, _ = get_a_character(args, name)
        tgt_chars.append(char)

    # target motion: zero rot
    tgt_motion_0 = copy.deepcopy(motion_0)
    tgt_motion_1 = copy.deepcopy(motion_0)
    for pose in tgt_motion_0.poses:
        pose.local_R = np.identity(3)[None, :].repeat(22, axis=0).astype(np.float32)
        pose.update()
    for pose in tgt_motion_1.poses:
        pose.local_R = np.identity(3)[None, :].repeat(22, axis=0).astype(np.float32)
        pose.update()

    # set skeleton
    tgt_chars[0].set_source_skeleton(tgt_motion_0.skeleton, "")
    tgt_chars[1].set_source_skeleton(tgt_motion_1.skeleton, "")
    
    # scale
    if small:
        from datasets.character_functions import get_scale
        scales = get_scale()
        index = 0 # args.SMPLx_scale_index 
        scale = scales[index]
        leg_scale, body_scale, hand_scale = scale[0], scale[1], scale[2]
        root_scale = leg_scale
        from Retarget_SMPL.retarget_smpl import scale_character 
        scale_character(args, tgt_chars[1], leg_scale, body_scale, hand_scale)
    else:
        root_scale = 1

    # load geo
    tgt_geoms = get_character_geometry(args, tgt_names, tgt_chars)
    tgt_auramesh = []
    for i, (geom, name) in enumerate(zip(src_geoms, src_names)):
        auramesh = AuraMesh(args, tgt_geoms[i], tgt_names[i])
        if i==0:
            motion = tgt_motion_0
        else:
            motion = tgt_motion_1
        update_boundary_position(args, auramesh, motion)
        tgt_auramesh.append(auramesh)

    # # set pose identity 
    # for pose in tgt_motion_1.poses:
    #     pose.local_R = np.identity(3)[None, :].repeat(22, axis=0)  
    #     pose.update()
    
    """ Collision detection """
    # Collision detection in src
    col_det = False
    if col_det:
        print("Start Collision detection in src")
        import time
        time0 = time.time()
        
        """ col det """
        cids0, cids1, jids0, jids1, col_frame = collision_detection(args, src_geoms[0], src_auramesh[1], motion_0, motion_1)
        torch.save(cids0, path+'pt/cids0.pt')
        torch.save(cids1, path+'pt/cids1.pt')
        torch.save(jids0, path+'pt/jids0.pt')
        torch.save(jids1, path+'pt/jids1.pt')
        torch.save(col_frame, path_+'pt/col_frame.pt')
        time1 = time.time()
        print("time: ", time1-time0)
    else:
        cids0 = torch.load(path+'pt/cids0.pt')
        cids1 = torch.load(path+'pt/cids1.pt')
        jids0 = torch.load(path+'pt/jids0.pt')
        jids1 = torch.load(path+'pt/jids1.pt')
        col_frame = torch.load(path+'pt/col_frame.pt')
        col_frame = np.array(col_frame)
        print("load collision detection results")

    optimize = True
    if optimize:
        import time
        time0 = time.time()
        
        """ optimize """
        # Get vpositions
        src_colliding_vpos = []
        src_colliding_vids = []
        # for i, cids in enumerate(cids1):
        #     f = col_frame[i]
        
        #     # unique vid index 
        #     vids1 = src_auramesh[1].cid_to_first_vid[cids] # src_auramesh 
        #     vids1 = torch.unique(vids1).to(args.device)
        #     len_vids = len(vids1)
        #     src_colliding_vids.append(vids1)
            
        #     # src colliding vpos
        #     batch = torch.tensor([0])[None, None, :].repeat(1, 1, len_vids)
        #     frame = torch.tensor([f])[None, None, :].repeat(1, 1, len_vids)
        #     vids1_ = vids1[None, None, :]
        #     vpos1 = src_auramesh[1].get_positions_from_vids(vids1_, batch, frame)[0,0]
        #     src_colliding_vpos.append(vpos1)

        time1 = time.time()
        # update character 1
        tgt_motion_1 = optimize_motion(\
            motion_1, tgt_motion_1, tgt_auramesh[1], root_scale,
            col_frame, jids1, src_colliding_vpos, src_colliding_vids)
        
        time2 = time.time()
        tgt_motion_0 = optimize_motion(\
            motion_0, tgt_motion_0, tgt_auramesh[0], 1.0,
            col_frame, jids0, src_colliding_vpos, src_colliding_vids)
        time3 = time.time()
        print("time: ", time1-time0, time2-time1, time3-time2)
        # tgt_motion_0 = motion_0
        
        save = False
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
    