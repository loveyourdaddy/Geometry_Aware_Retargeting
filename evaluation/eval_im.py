"""
Quantitative evaluation for Interaction Mesh / AuraMesh retargeting results.

Run from project root:
    python evaluation/eval_im.py

Saved .npz files (auramesh/saved_result/) must contain:
    root_p : (T, 3)
    local_R: (T, J, 3, 3)

Metrics:
    joint_dist_diff   — skeletal spatial relationship error (weighted L2)
    anchor_dist_diff  — surface anchor spatial relationship error
    contact_preserving / contact_missing / wrong_contact / non_contact_preserving
"""

import sys, os
sys.path.append('./')
sys.path.append('../')

import copy
import numpy as np
import torch

from pymovis.vis.appmanager import AppManager
import option_parser
from option_motion import example_bvh
from datasets.character_functions import get_a_smpl_character as get_a_character
from datasets.motion_functions import get_interaction_motions_from_list
from Retarget_SMPL.relationship_descriptor import get_rootP_localR_globalP_from_motion
from Retarget_SMPL.retarget_smpl import scale_character

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from evaluation.eval_metric import check_semantic, get_contact_tensor


# ── Helpers ──────────────────────────────────────────────────────────────────

def exp_weight_of_distance(args, dist_map):
    return torch.exp(-args.exp_k * dist_map)


def load_npz_to_motion(npz_path, src_motion, tgt_skeleton):
    """root_p + local_R를 .npz에서 읽어 src_motion 복사본에 설정."""
    data    = np.load(npz_path)
    root_p  = data['root_p'].astype(np.float32)   # (T, 3)
    local_R = data['local_R'].astype(np.float32)  # (T, J, 3, 3)
    motion  = copy.deepcopy(src_motion)
    motion.skeleton = tgt_skeleton
    for f, pose in enumerate(motion.poses):
        pose.skeleton = tgt_skeleton
        pose.root_p   = root_p[f]
        pose.local_R  = local_R[f]
        pose.update()
    return motion


def _get_offsets(char):
    """(J, 3) joint offset tensor from character skeleton."""
    return torch.tensor(
        np.array([char.meshes[0].source_skeleton.joints[j].offset for j in range(22)]),
        dtype=torch.float32
    )


def _set_geo_pose(args, geo, motion):
    """Geometry에 motion 전체 프레임 pose를 일괄 설정."""
    root_p, local_R, _ = get_rootP_localR_globalP_from_motion(args, motion.poses)
    geo.set_pose_by_source_batch_frame(local_R.unsqueeze(0), root_p.unsqueeze(0))
    geo.bvh_tree.update_joint_aabb()


# ── Main ─────────────────────────────────────────────────────────────────────

def evaluate(args, method_name, npz_path0, npz_path1,
             motion_name0, motion_name1, scale=0.7):
    """
    단일 method에 대해 evaluation 수치를 계산하여 dict로 반환.

    Args:
        npz_path0: char0 저장 파일 (.npz) 또는 None (소스 모션 그대로 사용)
        npz_path1: char1 저장 파일 (.npz)  char1은 scale된 타겟
    """
    print(f"\n=== Evaluating [{method_name}] ===")
    print(f"  motion0: {motion_name0}  motion1: {motion_name1}")

    # ── Source characters & geometry ──
    src_char0, _, src_geo0 = get_a_character(args, 'SMPLx')
    src_char1, _, src_geo1 = get_a_character(args, 'SMPLx')

    src_motion0 = get_interaction_motions_from_list('SMPLx', [motion_name0])[0]
    src_motion1 = get_interaction_motions_from_list('SMPLx', [motion_name1])[0]

    src_char0.set_source_skeleton(src_motion0.skeleton, "")
    src_char1.set_source_skeleton(src_motion1.skeleton, "")
    _set_geo_pose(args, src_geo0, src_motion0)
    _set_geo_pose(args, src_geo1, src_motion1)

    # ── Target characters & geometry ──
    tgt_char0, _, tgt_geo0 = get_a_character(args, 'SMPLx')
    tgt_char1, _, tgt_geo1 = get_a_character(args, 'SMPLx', mesh_scale=scale)

    tgt_char0.set_source_skeleton(src_motion0.skeleton, "")
    tgt_char1.set_source_skeleton(src_motion1.skeleton, "")
    scale_character(args, tgt_char1, scale, scale, scale)

    # ── Joint offsets ──
    src_off0 = _get_offsets(src_char0)
    src_off1 = _get_offsets(src_char1)
    tgt_off0 = _get_offsets(tgt_char0)
    tgt_off1 = _get_offsets(tgt_char1)
    parent_idx = src_char0.meshes[0].source_skeleton.parent_idx

    # ── Target motions ──
    # char0: interaction mesh는 char0를 수정하지 않음 → 소스 복사
    if npz_path0 is not None and os.path.isfile(npz_path0):
        tgt_motion0 = load_npz_to_motion(npz_path0, src_motion0,
                                          tgt_char0.meshes[0].source_skeleton)
        print(f"  char0 loaded from {npz_path0}")
    else:
        tgt_motion0 = copy.deepcopy(src_motion0)
        tgt_motion0.skeleton = tgt_char0.meshes[0].source_skeleton
        for pose in tgt_motion0.poses:
            pose.skeleton = tgt_char0.meshes[0].source_skeleton
            pose.update()
        print("  char0: using source motion (not retargeted)")

    # char1: .npz에서 로드
    tgt_motion1 = load_npz_to_motion(npz_path1, src_motion1,
                                      tgt_char1.meshes[0].source_skeleton)
    print(f"  char1 loaded from {npz_path1}")

    _set_geo_pose(args, tgt_geo0, tgt_motion0)
    _set_geo_pose(args, tgt_geo1, tgt_motion1)

    # ── Semantic preserving ──
    print("  Computing semantic metrics...")
    src_dist_map, src_anchor_dist = check_semantic(
        args,
        src_motion0, src_motion1,
        src_off0, src_off1,
        src_geo0, src_geo1,
        parent_idx, parent_idx,
    )
    tgt_dist_map, tgt_anchor_dist = check_semantic(
        args,
        tgt_motion0, tgt_motion1,
        tgt_off0, tgt_off1,
        tgt_geo0, tgt_geo1,
        parent_idx, parent_idx,
    )

    jw = exp_weight_of_distance(args, src_dist_map)
    joint_dist_diff = torch.norm(jw * (src_dist_map - tgt_dist_map), dim=1).mean()

    aw = exp_weight_of_distance(args, src_anchor_dist)
    anchor_dist_diff = torch.norm(aw * (src_anchor_dist - tgt_anchor_dist), dim=1).mean()

    # ── Contact ──
    print("  Computing contact metrics...")
    body_joints       = args.body_joints
    left_leg_joints   = args.left_leg_joints
    right_leg_joints  = args.right_leg_joints
    left_hand_joints  = args.left_hand_joints
    right_hand_joints = args.right_hand_joints
    parts = [body_joints, left_leg_joints, right_leg_joints,
             left_hand_joints, right_hand_joints]

    T = len(src_motion0.poses)

    src_ct = get_contact_tensor(args, src_geo0, src_geo1, src_motion0, src_motion1)
    src_cp = torch.full((T, 5, 5), False)
    for i, part in enumerate(parts):
        src_cp[:, i, i] += src_ct[:, part, part].sum().bool()
    src_contact    = src_cp.sum()
    src_no_contact = (~src_cp).sum()

    tgt_ct = get_contact_tensor(args, tgt_geo0, tgt_geo1, tgt_motion0, tgt_motion1)
    tgt_cp = torch.full((T, 5, 5), False)
    for i, part in enumerate(parts):
        tgt_cp[:, i, i] += tgt_ct[:, part, part].sum().bool()

    cp  = torch.logical_and( src_cp,  tgt_cp).sum()
    cm  = torch.logical_and( src_cp, ~tgt_cp).sum()
    wc  = torch.logical_and(~src_cp,  tgt_cp).sum()
    ncp = torch.logical_and(~src_cp, ~tgt_cp).sum()

    total = cp + cm + wc + ncp
    results = {
        'joint_dist_diff':        round(joint_dist_diff.item(), 4),
        'anchor_dist_diff':       round(anchor_dist_diff.item(), 4),
        'contact_preserving':     round((cp  / (src_contact    + 1e-8)).item(), 4),
        'contact_missing':        round((cm  / (src_contact    + 1e-8)).item(), 4),
        'wrong_contact':          round((wc  / (src_no_contact + 1e-8)).item(), 4),
        'non_contact_preserving': round((ncp / (src_no_contact + 1e-8)).item(), 4),
        # TP=cp, FN=cm, FP=wc, TN=ncp  (contact = positive class)
        'contact_precision':      round((cp / (cp + wc + 1e-8)).item(), 4),
        'contact_recall':         round((cp / (cp + cm + 1e-8)).item(), 4),
        'contact_accuracy':       round(((cp + ncp) / (total  + 1e-8)).item(), 4),
    }

    print(f"\n  Results [{method_name}] — {motion_name0}")
    for k, v in results.items():
        print(f"    {k:<26}: {v}")

    return results


def main():
    # AppManager가 GLFW window + OpenGL context를 초기화함.
    # get_a_character가 fbx.model()에서 glGenVertexArrays를 호출하므로 반드시 먼저 실행해야 함.
    import glfw
    app_manager = AppManager()
    glfw.hide_window(app_manager.window)  # 평가 중 창이 화면을 가리지 않도록 숨김

    args = option_parser.get_args()
    args.device      = 'cpu'
    args.is_train    = False
    args.save_norm_info = False
    args.test_type   = 'SMPLx'
    args.test_char   = 'small'
    scale = 0.7

    motion_name0 = list(example_bvh.keys())[0]   # "greeting002_S1"
    motion_name1 = list(example_bvh.values())[0]  # "greeting002_S2"

    motion_name = motion_name0.replace("_S1", "")
    save_dir = f'./auramesh/saved_result/{motion_name}/'

    # ── 평가할 method별 파일 경로 ──
    # npz_path0=None → char0는 소스 모션 그대로 사용
    # raw_dir = './auramesh/saved_result/'
    methods = {
        'network': {
            'npz_path0': os.path.join(save_dir, f'net_{motion_name0}_s0.npz'),
            'npz_path1': os.path.join(save_dir, f'net_{motion_name1}_s1.npz'),
        },
        'interaction_mesh': {
            'npz_path0': os.path.join(save_dir, f'im_{motion_name0}_s0.npz'),
            'npz_path1': os.path.join(save_dir, f'im_{motion_name1}_s1.npz'),
        },
        'auramesh_wo_smooth': {
            'npz_path0': os.path.join(save_dir, f'am_{motion_name0}_s0.npz'),
            'npz_path1': os.path.join(save_dir, f'am_{motion_name1}_s1.npz'),
        },
    }

    all_results = {}
    for method_name, paths in methods.items():
        npz1 = paths['npz_path1']
        if not os.path.isfile(npz1):
            print(f"[SKIP] {method_name}: {npz1} not found")
            continue
        results = evaluate(
            args, method_name,
            npz_path0=paths['npz_path0'],
            npz_path1=npz1,
            motion_name0=motion_name0,
            motion_name1=motion_name1,
            scale=scale,
        )
        all_results[method_name] = results

    # ── 요약 출력 ──
    if all_results:
        header = ['method'] + list(next(iter(all_results.values())).keys())
        print('\n' + '=' * 80)
        print('  '.join(f'{h:<26}' for h in header))
        print('-' * 80)
        for method, res in all_results.items():
            row = [f'{method:<26}'] + [f'{v:<26}' for v in res.values()]
            print('  '.join(row))
        print('=' * 80)

        # 파일 저장
        os.makedirs('./result_saved', exist_ok=True)
        out_path = f'./result_saved/eval_im_{motion_name0}.txt'
        with open(out_path, 'w') as f:
            f.write('  '.join(header) + '\n')
            for method, res in all_results.items():
                f.write(method + '  ' + '  '.join(map(str, res.values())) + '\n')
        print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
