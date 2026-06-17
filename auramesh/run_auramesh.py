'''
python auramesh/run_auramesh.py
'''

import sys
sys.path.append('./')
sys.path.append('../')  # Retargeting_workspace

import copy
import os
import numpy as np
import torch
from typing import List

import glfw
from pymovis.vis.appmanager import AppManager
from pymovis.vis.app import MotionApp
from pymovis.vis.render import Render
from pymovis.motion.core import Motion
from pymovis.motion.ops.npmotion import *
from datasets.character_functions import get_a_smpl_character as get_a_character
from datasets.motion_functions import get_interaction_motions_from_list
from Retarget_SMPL.relationship_descriptor import (
    get_rootP_localR_globalP_from_motion,
    get_rootP_localR_globalP_from_numpy_motion,
)
from Retarget_SMPL.retarget_smpl import scale_character
from optimizer import get_character_geometry, optimize_motion, smooth_motion

import option_parser
from option_motion import example_bvh
from optimizer import _smooth_gaussian, _smooth_oneeuro, normalize_rotation_matrix


from Geometry.compare_geometry import collision_detection
from Geometry.geometry import Geometry
class AuraMesh(Geometry):
    def __init__(self, args, character, name=None, dist=0.1, scale=1):
        super().__init__(args, character, name)
        self.v_position *= scale
        self.v_position = self.v_position + self.c_normal[self.vid_to_cid] * dist


class MyApp(MotionApp):
    def __init__(self, model: List[Geometry], motion: List[Motion], args,
                 aurameshes=None, am_motions=None):
        super().__init__(model, motion, args)
        self.aurameshes = aurameshes or []
        self.am_motions = am_motions or []
        self.draw_auramesh = False
        print("Start render")

    def key_callback(self, window, key, scancode, action, mods):
        super().key_callback(window, key, scancode, action, mods)
        if key == glfw.KEY_N and action == glfw.PRESS:
            self.draw_auramesh = not self.draw_auramesh

    def render(self):
        super().render()
        if not self.draw_auramesh:
            return
        for am, motion in zip(self.aurameshes, self.am_motions):
            model = am.renderable_model_by_pose(motion.poses[self.frame])
            Render.model(model).draw()


def _run_collision_detection(args, src_geoms, src_auramesh, motion_0, motion_1, pt_dir):
    os.makedirs(pt_dir, exist_ok=True)
    print("Running collision detection...")

    cids0, am_cids1, jids0, am_jids1, frames0 = collision_detection(
        args, src_geoms[0], src_auramesh[1], motion_0, motion_1)
    cids1, am_cids0, jids1, am_jids0, frames1 = collision_detection(
        args, src_geoms[1], src_auramesh[0], motion_1, motion_0)

    torch.save(cids0,     pt_dir + 'tgt_geo_cids0.pt')
    torch.save(am_cids1,  pt_dir + 'tgt_auramesh_cids1.pt')
    torch.save(jids0,     pt_dir + 'tgt_geo_jids0.pt')
    torch.save(am_jids1,  pt_dir + 'tgt_auramesh_jids1.pt')
    torch.save(frames0,   pt_dir + 'col_frame0.pt')

    torch.save(cids1,     pt_dir + 'tgt_geo_cids1.pt')
    torch.save(am_cids0,  pt_dir + 'tgt_auramesh_cids0.pt')
    torch.save(jids1,     pt_dir + 'tgt_geo_jids1.pt')
    torch.save(am_jids0,  pt_dir + 'tgt_auramesh_jids0.pt')
    torch.save(frames1,   pt_dir + 'col_frame1.pt')

    return (cids0, am_cids1, jids0, am_jids1, np.array(frames0),
            cids1, am_cids0, jids1, am_jids0, np.array(frames1))


def _load_collision_detection(pt_dir):
    print("Loading collision detection results...")
    cids0    = torch.load(pt_dir + 'tgt_geo_cids0.pt')
    am_cids1 = torch.load(pt_dir + 'tgt_auramesh_cids1.pt')
    jids0    = torch.load(pt_dir + 'tgt_geo_jids0.pt')
    am_jids1 = torch.load(pt_dir + 'tgt_auramesh_jids1.pt')
    frames0  = np.array(torch.load(pt_dir + 'col_frame0.pt'))

    cids1    = torch.load(pt_dir + 'tgt_geo_cids1.pt')
    am_cids0 = torch.load(pt_dir + 'tgt_auramesh_cids0.pt')
    jids1    = torch.load(pt_dir + 'tgt_geo_jids1.pt')
    am_jids0 = torch.load(pt_dir + 'tgt_auramesh_jids0.pt')
    frames1  = np.array(torch.load(pt_dir + 'col_frame1.pt'))

    return (cids0, am_cids1, jids0, am_jids1, frames0,
            cids1, am_cids0, jids1, am_jids0, frames1)


if __name__ == "__main__":
    app_manager = AppManager()
    args = option_parser.get_args()
    args.path = args.proj_name + '/'
    args.device = 'cpu'
    scale = 0.7
    pt_dir = "./auramesh/pt/"

    # ── Source characters & motions ──
    src_names = ["SMPLx", "SMPLx"]
    src_chars = [get_a_character(args, n)[0] for n in src_names]

    motion_name0 = list(example_bvh.keys())[0]
    motion_name1 = list(example_bvh.values())[0]
    motion_0 = get_interaction_motions_from_list(src_names[0], [motion_name0])[0]
    motion_1 = get_interaction_motions_from_list(src_names[0], [motion_name1])[0]
    # motion_0.poses = motion_0.poses[:90]
    # motion_1.poses = motion_1.poses[:90]

    src_chars[0].set_source_skeleton(motion_0.skeleton, "")
    src_chars[1].set_source_skeleton(motion_1.skeleton, "")

    src_geoms = get_character_geometry(args, src_names, src_chars)
    src_geoms[0].source_skeleton = motion_0.skeleton
    src_geoms[1].source_skeleton = motion_1.skeleton

    src_auramesh = [AuraMesh(args, g, n) for g, n in zip(src_geoms, src_names)]

    root_p0, local_R0, _ = get_rootP_localR_globalP_from_motion(args, motion_0.poses)
    root_p1, local_R1, _ = get_rootP_localR_globalP_from_motion(args, motion_1.poses)
    src_geoms[0].set_pose_by_source_batch_frame(local_R0.unsqueeze(0), root_p0.unsqueeze(0))
    src_geoms[1].set_pose_by_source_batch_frame(local_R1.unsqueeze(0), root_p1.unsqueeze(0))
    src_auramesh[0].set_pose_by_source_batch_frame(local_R0.unsqueeze(0), root_p0.unsqueeze(0))
    src_auramesh[1].set_pose_by_source_batch_frame(local_R1.unsqueeze(0), root_p1.unsqueeze(0))

    # ── Collision detection ──
    col_det = False
    if col_det:
        (geo_cids0, am_cids1, geo_jids0, am_jids1, col_frames0,
         geo_cids1, am_cids0, geo_jids1, am_jids0, col_frames1) = \
            _run_collision_detection(args, src_geoms, src_auramesh, motion_0, motion_1, pt_dir)
    else:
        (geo_cids0, am_cids1, geo_jids0, am_jids1, col_frames0,
         geo_cids1, am_cids0, geo_jids1, am_jids0, col_frames1) = \
            _load_collision_detection(pt_dir)

    # 모션을 슬라이싱한 경우, 캐시된 충돌 데이터의 프레임 범위도 맞춰서 필터링
    T = len(motion_0.poses)
    def _filter_by_frames(cids, am_cids, jids, am_jids, frames):
        mask = frames < T
        idx  = np.where(mask)[0]
        return ([cids[i]    for i in idx],
                [am_cids[i] for i in idx],
                [jids[i]    for i in idx],
                [am_jids[i] for i in idx],
                frames[idx])
    geo_cids0, am_cids1, geo_jids0, am_jids1, col_frames0 = \
        _filter_by_frames(geo_cids0, am_cids1, geo_jids0, am_jids1, col_frames0)
    geo_cids1, am_cids0, geo_jids1, am_jids0, col_frames1 = \
        _filter_by_frames(geo_cids1, am_cids0, geo_jids1, am_jids0, col_frames1)

    # ── Target characters & motions ──
    tgt_names = ["SMPLx", "SMPLx"]
    tgt_chars = []
    for i, name in enumerate(tgt_names):
        char, _, _ = get_a_character(args, name, **({"mesh_scale": scale} if i == 1 else {}))
        tgt_chars.append(char)

    tgt_motion_0 = copy.deepcopy(motion_0)
    tgt_motion_1 = copy.deepcopy(motion_1)

    # char1: scale root height
    for pose in tgt_motion_1.poses:
        # pose.local_R = np.identity(3)[None, :].repeat(22, axis=0).astype(np.float32)
        pose.root_p[1] *= scale
        pose.update()

    tgt_chars[0].set_source_skeleton(tgt_motion_0.skeleton, "")
    tgt_chars[1].set_source_skeleton(tgt_motion_1.skeleton, "")

    scale_character(args, tgt_chars[1], scale, scale, scale)
    for pose in tgt_motion_1.poses:
        pose.update()

    tgt_geoms = get_character_geometry(args, tgt_names, tgt_chars)
    tgt_auramesh = []
    for i, (geom, name) in enumerate(zip(tgt_geoms, tgt_names)):
        motion = tgt_motion_0 if i == 0 else tgt_motion_1
        am_scale = 1 if i == 0 else scale
        auramesh = AuraMesh(args, geom, name, scale=am_scale)
        root_p, local_R, _ = get_rootP_localR_globalP_from_numpy_motion(args, motion.poses)
        auramesh.set_pose_by_source_batch_frame(local_R.unsqueeze(0), root_p.unsqueeze(0))
        tgt_auramesh.append(auramesh)

    # vertex ID lists for collision pairs
    tgt_geo_vids0  = [tgt_geoms[0].cid_to_first_vid[geo_cids0[i]]  for i in range(len(geo_cids0))]
    tgt_am_vids1   = [src_auramesh[1].cid_to_first_vid[am_cids1[i]] for i in range(len(am_cids1))]

    tgt_geo_vids1  = [tgt_geoms[1].cid_to_first_vid[geo_cids1[i]]  for i in range(len(geo_cids1))]
    tgt_am_vids0   = [src_auramesh[0].cid_to_first_vid[am_cids0[i]] for i in range(len(am_cids0))]

    # ════════════════════════════════════════════════════════
    #  옵션 설정
    # ════════════════════════════════════════════════════════
    USE_SAVED   = True          # True: 저장된 .npz 로드 (최적화 skip)
    motion_name = motion_name0.replace("_S1", "")
    save_dir = f"./auramesh/saved_result/{motion_name}"
    name0 = os.path.splitext(os.path.basename(motion_name0))[0]
    name1 = os.path.splitext(os.path.basename(motion_name1))[0]

    if USE_SAVED:
        # ── 저장된 최적화 결과 로드 (최적화 skip) ──
        print(f"Loading saved motion from {save_dir} ...")
        for motion, name, idx in [(tgt_motion_0, name0, 0), (tgt_motion_1, name1, 1)]:
            path = os.path.join(save_dir, f"am_{name}_s{idx}.npz")
            breakpoint()
            data = np.load(path)
            for f, pose in enumerate(motion.poses):
                pose.root_p  = data['root_p'][f]
                pose.local_R = data['local_R'][f]
                pose.update()
            print(f"  Loaded: {path}")
    else:
        # ── Optimize char1 motion ──
        tgt_motion_1 = optimize_motion(
            args,
            motion_1, tgt_motion_1,
            tgt_geoms[1], tgt_auramesh[0],
            tgt_geo_vids1, tgt_am_vids0,
            col_frames1, geo_jids1, am_jids0,
        )

        # 최적화 원본 저장 (smooth 전)
        os.makedirs(save_dir, exist_ok=True)
        for motion, name, idx in [(tgt_motion_0, name0, 0), (tgt_motion_1, name1, 1)]:
            root_p  = np.stack([p.root_p  for p in motion.poses])
            local_R = np.stack([p.local_R for p in motion.poses])
            path = os.path.join(save_dir, f"am_{name}_s{idx}.npz")
            np.savez(path, root_p=root_p, local_R=local_R)
            print(f"Saved raw: {path}")

    # ════════════════════════════════════════════════════════
    SMOOTH_MODE = 'oneeuro' # None | 'gaussian' | 'oneeuro'

    # gaussian 파라미터
    SMOOTH_SIGMA = 3
    # oneeuro 파라미터 (높을수록 변화 보존, 낮을수록 smoothing)
    SMOOTH_MIN_CUTOFF = 5.0
    SMOOTH_BETA       = 0.5
    SMOOTH_FPS        = 30

    # ── Smoothing (선택) ──
    if SMOOTH_MODE is not None:
        tgt_motion_1 = smooth_motion(
            motion_1, tgt_motion_1,
            mode=SMOOTH_MODE,
            sigma=SMOOTH_SIGMA,
            min_cutoff=SMOOTH_MIN_CUTOFF,
            beta=SMOOTH_BETA,
            fps=SMOOTH_FPS,
        )

    # ── Save smoothed result ──
    if SMOOTH_MODE is not None:
        out_dir = os.path.join(save_dir, f"am_{SMOOTH_MODE}")
        os.makedirs(out_dir, exist_ok=True)
        for motion, name, idx in [(tgt_motion_0, name0, 0), (tgt_motion_1, name1, 1)]:
            root_p  = np.stack([p.root_p  for p in motion.poses])
            local_R = np.stack([p.local_R for p in motion.poses])
            path = os.path.join(out_dir, f"am_{name}_s{idx}.npz")
            np.savez(path, root_p=root_p, local_R=local_R)
            print(f"Saved: {path}")

    # ── Render ──
    T = len(motion_0.poses)
    for f in range(T):
        motion_0.poses[f].translate_root_p([args.source_pos, 0, 0])
        motion_1.poses[f].translate_root_p([args.source_pos, 0, 0])
        tgt_motion_0.poses[f].translate_root_p([args.joint_pos, 0, 0])
        tgt_motion_1.poses[f].translate_root_p([args.joint_pos, 0, 0])

    chars   = src_chars + tgt_chars
    motions = [motion_0, motion_1, tgt_motion_0, tgt_motion_1]

    # aurameshes = src_auramesh + tgt_auramesh
    # am_motions = [motion_0, motion_1, tgt_motion_0, tgt_motion_1]
    # app = MyApp(chars, motions, args, aurameshes=aurameshes, am_motions=am_motions)
    
    app = MyApp(chars, motions, args)
    app_manager.run(app)
