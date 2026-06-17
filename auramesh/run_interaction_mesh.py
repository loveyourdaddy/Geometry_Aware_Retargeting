"""
Interaction Mesh based motion retargeting.
Ho et al., "Spatial Relationship Preserving Character Motion Adaptation", ACM TOG 2010.

레퍼런스: Physics-based-retargeting/interaction_mesh.py
주요 변경:
  - L-BFGS 대신 closed-form LS (M_lap @ rhs), 프레임당 행렬곱 1회
  - update_motion_by_global_p → per-joint swing 계산 (multi-child 버그 수정)
  - Bone-length rescaling으로 스케일 스켈레톤 적합성 보장
"""

import sys
sys.path.append('./')
sys.path.append('../')

import copy
import time
import numpy as np
from OpenGL.GL import glDisable, glEnable, GL_DEPTH_TEST

from pymovis.vis.appmanager import AppManager
from pymovis.vis.app import MotionApp
from pymovis.vis.render import Render
from pymovis.motion.ops.npmotion import *

from datasets.character_functions import get_a_smpl_character as get_a_character
from datasets.motion_functions import get_interaction_motions_from_list
import torch
from Retarget_SMPL.relationship_descriptor import (
    get_rootP_localR_globalP_from_motion,
    get_rootP_localR_globalP_from_numpy_motion,
    update_motion_by_global_p,
)
from Retarget_SMPL.retarget_smpl import scale_character

import option_parser
from option_motion import example_bvh
from auramesh.interaction_mesh import InteractionMesh


# ── Bone-length utilities ────────────────────────────────────────────────────

def _get_bone_lengths(skeleton):
    """타겟 스켈레톤의 bone 길이 배열 (joint offset 기준)."""
    return np.array([np.linalg.norm(j.offset) for j in skeleton.joints])


def _rescale_bone_lengths(pos, parent_idx, target_lengths):
    """
    위상 순서(SMPL: parent < child)로 각 bone을 target_lengths[j] 길이로 정규화.
    root(joint 0) 위치는 고정.
    """
    pos_out = pos.copy()
    for j in range(1, len(pos)):
        p = parent_idx[j]
        if p < 0:
            continue
        bone = pos_out[j] - pos_out[p]
        length = np.linalg.norm(bone)
        if length > 1e-7 and target_lengths[j] > 1e-7:
            pos_out[j] = pos_out[p] + bone / length * target_lengths[j]
    return pos_out


# ── Main retargeting ─────────────────────────────────────────────────────────

class DebugApp(MotionApp):
    """MotionApp 확장: B 키로 args.debug_points (최적화 관절 위치) 표시."""

    def render(self):
        super().render()

        if not self.draw_debug:
            return
        debug_pts = getattr(self.args, 'debug_points', None)
        if debug_pts is None:
            return

        pts = debug_pts[self.frame]  # (J, 3) numpy

        if not hasattr(self, '_debug_sphere'):
            self._debug_sphere = Render.sphere(0.03)

        glDisable(GL_DEPTH_TEST)
        for p in pts:
            (self._debug_sphere
             .set_position(float(p[0]), float(p[1]), float(p[2]))
             .set_albedo([1.0, 1.0, 0.0])   # 노란색
             .set_color_mode(True)
             .draw())
        glEnable(GL_DEPTH_TEST)


def retarget_with_interaction_mesh(args, src_motion_0, tgt_motion_1, im,
                                   alpha=0.8):
    """
    Closed-form Interaction Mesh 기반 리타겟팅 (레퍼런스 방식).

    알고리즘:
      1. 각 프레임: pos1_lap = M_lap @ (delta_src - LA @ pos0)   [행렬곱 1회]
         → Laplacian 보존 최적 위치 (≈ 소스 char1 위치)
      2. tgt_gp1_init (스케일 초기값)와 alpha 블렌드
         alpha=1: 완전 Laplacian(상호작용 보존), alpha=0: 초기값 유지
      3. Root는 스케일된 초기 위치로 고정
      4. Bone-length rescaling (위상 순서)
      5. Per-joint local_R 계산 (swing+twist, multi-child 안전)

    Args:
        src_motion_0: char0 소스 모션 (고정 앵커)
        tgt_motion_1: char1 타겟 모션 (초기화: 소스 회전 + 스케일 skeleton)
        im:           InteractionMesh (소스에서 빌드)
        alpha:        Laplacian vs 초기값 블렌드 가중치 (0~1)
    """
    skeleton_1     = tgt_motion_1.skeleton
    parent_idx     = skeleton_1.parent_idx
    target_lengths = _get_bone_lengths(skeleton_1)

    # char0 소스 위치 (T, 22, 3) — 고정 앵커
    _, _, src_gp0 = get_rootP_localR_globalP_from_motion(args, src_motion_0.poses)
    src_gp0 = src_gp0.numpy().astype(np.float64)

    # char1 초기 글로벌 위치 (스케일 적용 후)
    _, _, tgt_gp1_init = get_rootP_localR_globalP_from_numpy_motion(args, tgt_motion_1.poses)
    tgt_gp1_init = tgt_gp1_init.numpy().astype(np.float64)   # (T, 22, 3)

    T = len(tgt_motion_1.poses)
    print(f"Retargeting {T} frames (closed-form LS, alpha={alpha:.2f})...")
    t_start = time.perf_counter()

    all_pos1_final = np.zeros((T, im.n1, 3), dtype=np.float32)

    for f in range(T):
        # 1. Laplacian closed-form: pos1_lap ≈ src_gp1 (소스 위치)
        pos1_lap = im.compute_target_positions(
            src_gp0[f], im.src_laplacian[f]
        )  # (22, 3)

        # 2. alpha 블렌드 (root 제외)
        pos1_init_f  = tgt_gp1_init[f]
        pos1_blended = (1.0 - alpha) * pos1_init_f + alpha * pos1_lap
        pos1_blended[0] = pos1_init_f[0]  # root는 스케일된 초기 위치 고정

        # 3. Bone-length rescaling
        pos1_final = _rescale_bone_lengths(pos1_blended, parent_idx, target_lengths)

        all_pos1_final[f] = pos1_final.astype(np.float32)

    t_lap = time.perf_counter()

    # spine2(11), leftshoulder(14), rightshoulder(18) rotation 고정
    FIXED_JOINTS = [11, 14, 18]
    saved_lr = [
        tgt_motion_1.poses[f].local_R[FIXED_JOINTS].copy()
        for f in range(T)
    ]

    # 전체 모션 pose 업데이트 (local_R, root_p)
    update_motion_by_global_p(
        tgt_motion_1,
        torch.tensor(all_pos1_final, dtype=torch.float32)
    )

    # 고정 관절 rotation 복원
    for f in range(T):
        tgt_motion_1.poses[f].local_R[FIXED_JOINTS] = saved_lr[f]
        tgt_motion_1.poses[f].update()

    t_end = time.perf_counter()

    print(f"Done.  total={t_end - t_start:.2f}s  "
          f"(laplacian={t_lap - t_start:.2f}s, "
          f"pose_update={t_end - t_lap:.2f}s, "
          f"{(t_end - t_start) / T * 1000:.1f}ms/frame)")

    # 디버그용: 최적화된 관절 위치 저장 (DebugApp이 렌더링에 사용)
    args.debug_points = all_pos1_final  # (T, 22, 3)

    return tgt_motion_1


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app_manager = AppManager()
    args = option_parser.get_args()
    args.path = args.proj_name + '/'
    args.device = 'cpu'
    scale = 0.7

    # ── 소스 캐릭터 & 모션 ──
    src_names = ["SMPLx", "SMPLx"]
    src_chars = []
    for name in src_names:
        char, _, _ = get_a_character(args, name)
        src_chars.append(char)

    motion_name0 = list(example_bvh.keys())[0]
    motion_name1 = list(example_bvh.values())[0]
    motion_0 = get_interaction_motions_from_list(src_names[0], [motion_name0])[0]
    motion_1 = get_interaction_motions_from_list(src_names[0], [motion_name1])[0]
    # motion_0.poses = motion_0.poses[:10]
    # motion_1.poses = motion_1.poses[:10]

    src_chars[0].set_source_skeleton(motion_0.skeleton, "")
    src_chars[1].set_source_skeleton(motion_1.skeleton, "")

    print(f"Source motions: {motion_name0} ({len(motion_0.poses)}f), "
          f"{motion_name1} ({len(motion_1.poses)}f)")

    # ── InteractionMesh 빌드 (소스 기준) ──
    print("\nBuilding Interaction Mesh from source motions...")
    _, _, src_gp0 = get_rootP_localR_globalP_from_motion(args, motion_0.poses)
    _, _, src_gp1 = get_rootP_localR_globalP_from_motion(args, motion_1.poses)
    im = InteractionMesh(src_gp0.numpy(), src_gp1.numpy())

    # ── 타겟 캐릭터 & 모션 ──
    tgt_names = ["SMPLx", "SMPLx"]
    tgt_chars = []
    for i, name in enumerate(tgt_names):
        if i == 0:
            char, _, _ = get_a_character(args, name)
        else:
            char, _, _ = get_a_character(args, name, mesh_scale=scale)
        tgt_chars.append(char)

    tgt_motion_0 = copy.deepcopy(motion_0)
    tgt_motion_1 = copy.deepcopy(motion_1)

    # char1 초기화: 소스 회전 유지 + root Y 스케일
    for pose in tgt_motion_1.poses:
        pose.root_p[1] *= scale
        pose.update()

    tgt_chars[0].set_source_skeleton(tgt_motion_0.skeleton, "")
    tgt_chars[1].set_source_skeleton(tgt_motion_1.skeleton, "")

    # char1 skeleton 스케일 적용
    scale_character(args, tgt_chars[1], scale, scale, scale)
    for pose in tgt_motion_1.poses:
        pose.update()

    # ── Retarget ──
    print("\nRetargeting with Interaction Mesh...")
    tgt_motion_1 = retarget_with_interaction_mesh(
        args, motion_0, tgt_motion_1, im, alpha=0.8
    )

    # ── 렌더 ──
    for f in range(len(motion_0.poses)):
        motion_0.poses[f].translate_root_p([args.source_pos, 0, 0])
        motion_1.poses[f].translate_root_p([args.source_pos, 0, 0])
        tgt_motion_0.poses[f].translate_root_p([args.joint_pos, 0, 0])
        tgt_motion_1.poses[f].translate_root_p([args.joint_pos, 0, 0])

    chars   = src_chars + tgt_chars
    motions = [motion_0, motion_1, tgt_motion_0, tgt_motion_1]

    app = DebugApp(chars, motions, args)
    app_manager.run(app)
