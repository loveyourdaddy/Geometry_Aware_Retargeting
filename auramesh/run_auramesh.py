# python을 실행시키는 path 기준: 
'''
python auramesh/run_auramesh.py 
'''

import sys
sys.path.append('./')
sys.path.append('../') # Retargeting_workspace

from pymovis.vis.appmanager import AppManager
from pymovis.vis.app import MotionApp
from pymovis.motion.core import Motion # Pose, 
from pymovis.motion.ops.npmotion import *

from datasets.character_functions import get_a_smpl_character as get_a_character
from datasets.motion_functions import get_interaction_motions_from_list
from Geometry.geometry import Geometry
from Geometry.compare_geometry import collision_detection # , update_boundary_position


class AuraMesh(Geometry):
    def __init__(self, args, character, name=None, dist=0.1, scale=1):
        super().__init__(args, character, name)
        self.v_position *= scale
        self.v_position = self.v_position + self.c_normal[self.vid_to_cid] * dist

from Retarget_SMPL.relationship_descriptor import get_rootP_localR_globalP_from_motion, get_rootP_localR_globalP_from_numpy_motion
from optimize_funcitons import *
from typing import List # , Optional

import option_parser
from option_motion import example_bvh
import copy 
import numpy as np
import torch
import copy
# from pymovis.vis.model import Model
# from pymovis.vis.render import Render
# from pymovis.motion.data import bvh
# from pymovis.motion.data.fbx import FBX
# from datasets import *
# from Geometry.geometry import Geometry


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
        
        # 아우라메쉬에 할당된 motions에서 몇번째 것인지.
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
    # clap = 340 # 200 330
    # motion_0.poses = motion_0.poses[:clap]
    # motion_1.poses = motion_1.poses[:clap]
    
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
        time0_ = time.time()
        
        """ col det """
        tgt_geo_cids0, tgt_auramesh_cids1, tgt_geo_jids0, tgt_auramesh_jids1, col_frame0 = \
            collision_detection(args, 
                                src_geoms[0], src_auramesh[1], 
                                motion_0, motion_1)
        torch.save(tgt_geo_cids0,      path+'pt/tgt_geo_cids0.pt')
        torch.save(tgt_auramesh_cids1, path+'pt/tgt_auramesh_cids1.pt')
        torch.save(tgt_geo_jids0,      path+'pt/tgt_geo_jids0.pt')
        torch.save(tgt_auramesh_jids1, path+'pt/tgt_auramesh_jids1.pt')
        torch.save(col_frame0,         path+'pt/col_frame0.pt')
        
        tgt_geo_cids1, tgt_auramesh_cids0, tgt_geo_jids1, tgt_auramesh_jids0, col_frame1 = \
            collision_detection(args, 
                                src_geoms[1], src_auramesh[0],
                                motion_1, motion_0)
        torch.save(tgt_geo_cids1,      path+'pt/tgt_geo_cids1.pt')
        torch.save(tgt_auramesh_cids0, path+'pt/tgt_auramesh_cids0.pt')
        torch.save(tgt_geo_jids1,      path+'pt/tgt_geo_jids1.pt')
        torch.save(tgt_auramesh_jids0, path+'pt/tgt_auramesh_jids0.pt')
        torch.save(col_frame1,         path+'pt/col_frame1.pt')
    else:
        tgt_geo_cids0      = torch.load(path+'pt/tgt_geo_cids0.pt')
        tgt_auramesh_cids1 = torch.load(path+'pt/tgt_auramesh_cids1.pt')
        tgt_geo_jids0      = torch.load(path+'pt/tgt_geo_jids0.pt')
        tgt_auramesh_jids1 = torch.load(path+'pt/tgt_auramesh_jids1.pt')
        col_frame0         = torch.load(path+'pt/col_frame0.pt')
        col_frame0         = np.array(col_frame0)
        
        tgt_geo_cids1      = torch.load(path+'pt/tgt_geo_cids1.pt')
        tgt_auramesh_cids0 = torch.load(path+'pt/tgt_auramesh_cids0.pt')
        tgt_geo_jids1      = torch.load(path+'pt/tgt_geo_jids1.pt')
        tgt_auramesh_jids0 = torch.load(path+'pt/tgt_auramesh_jids0.pt')
        col_frame1         = torch.load(path+'pt/col_frame1.pt')
        col_frame1         = np.array(col_frame1)
        print("load collision detection results")

    """ Set target character  """
    # target character
    tgt_names = ["SMPLx", "SMPLx"]
    tgt_chars = []
    for i, name in enumerate(tgt_names):
        if i==0:
            char, _, _ = get_a_character(args, name)
        else:
            char, _, _ = get_a_character(args, name, mesh_scale=scale) # mesh_scale=scale
        tgt_chars.append(char)

    # target motion: zero rot
    tgt_motion_0 = copy.deepcopy(motion_0)
    tgt_motion_1 = copy.deepcopy(motion_1)
    
    # set char 0 Tpose (update 0)
    # for pose in tgt_motion_0.poses:
        # pose.local_R = np.identity(3)[None, :].repeat(22, axis=0).astype(np.float32)
        # pose.update()
    # set char 1 Tpose (update 1)
    for pose in tgt_motion_1.poses:
        pose.local_R = np.identity(3)[None, :].repeat(22, axis=0).astype(np.float32)
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
        for pose in tgt_motion_1.poses:
            pose.update()


    # load geo
    tgt_geoms = get_character_geometry(args, tgt_names, tgt_chars)
    tgt_auramesh = []
    for i in range(len(tgt_geoms)):
        if i==0:
            auramesh = AuraMesh(args, tgt_geoms[i], tgt_names[i])
            motion = tgt_motion_0
        else:
            auramesh = AuraMesh(args, tgt_geoms[i], tgt_names[i], scale=scale)
            motion = tgt_motion_1
            
        root_p, local_R, _ = get_rootP_localR_globalP_from_numpy_motion(args, motion.poses)
        auramesh.set_pose_by_source_batch_frame(local_R.unsqueeze(0), root_p.unsqueeze(0))
        tgt_auramesh.append(auramesh)

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
    
    tgt_geo_vids1, tgt_auramesh_vids0 = [], []
    num_col = len(tgt_geo_cids1)
    for i in range(num_col):
        # geo of 0
        vids1 = tgt_geoms[1].cid_to_first_vid[tgt_geo_cids1[i]]
        tgt_geo_vids1.append(vids1)

        # auramesh of 1
        vids0 = src_auramesh[0].cid_to_first_vid[tgt_auramesh_cids0[i]]
        tgt_auramesh_vids0.append(vids0)
    
    
    """ optimize """
    # optimize: 0 ptn
    # auramesh : 1 main (dfm)
    
    # update motion0 (ptn)
    # tgt_motion_0 = optimize_motion(args,
    #     motion_0, tgt_motion_0, 
    #     tgt_geoms[0], tgt_auramesh[1],
    #     tgt_geo_vids0, tgt_auramesh_vids1,
    #     col_frame0, tgt_geo_jids0, tgt_auramesh_jids1)

    # update motion1 (main)
    tgt_motion_1 = optimize_motion(args,
        motion_1, tgt_motion_1, 
        tgt_geoms[1], tgt_auramesh[0],
        tgt_geo_vids1, tgt_auramesh_vids0,
        col_frame1, tgt_geo_jids1, tgt_auramesh_jids0)
    
    
    """ interpolate """
    smooth = False
    if smooth:
        # i=1 부터 시작해서, i-1과 quaternion interpolation
        # 일단 slerp 하지 않고 뒤에 값을 그대로 써보기. 
        # 79 이전, 264 이후는 변화 없을꺼야.

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

    """ render """
    # translate 
    for f in range(len(motion_0.poses)):
        motion_0.poses[f].translate_root_p([args.source_pos, 0, 0])
        motion_1.poses[f].translate_root_p([args.source_pos, 0, 0])
        tgt_motion_0.poses[f].translate_root_p([args.joint_pos, 0, 0])
        tgt_motion_1.poses[f].translate_root_p([args.joint_pos, 0, 0])
    
    # 
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
    