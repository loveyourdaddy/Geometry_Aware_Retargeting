"""Save and load adaptation results (adapted_motion/{character}/{motion}_*.npy)."""
import os
import numpy as np

from pymovis.motion.ops.npmotion import R_to_R6
from etc.etc import deepcopy
from Retarget_SMPL.skeleton_scale import scale_offset

ADAPTED_DIR = './adapted_motion'
KEYS = ['root_p0', 'local_R0', 'root_p1', 'local_R1']


def edited_path(char_name, motion_name, key):
    return os.path.join(ADAPTED_DIR, char_name, '{}_{}.npy'.format(motion_name, key))

def save_edited_npy_motion(char_name, motion_name, root_p0, local_R0, root_p1, local_R1):
    """Save the (role, scale, frame, ...) arrays. Does not overwrite when the existing file has a different number of scales."""
    arrays = dict(zip(KEYS, [np.array(root_p0), np.array(local_R0), np.array(root_p1), np.array(local_R1)]))

    path = edited_path(char_name, motion_name, KEYS[0])
    if os.path.exists(path):
        num_scale = np.load(path, mmap_mode='r').shape[1]
        if num_scale != arrays[KEYS[0]].shape[1]:
            print("skip save: {} has {} scales, new data has {}".format(path, num_scale, arrays[KEYS[0]].shape[1]))
            return False

    os.makedirs(os.path.dirname(path), exist_ok=True)
    for key, array in arrays.items():
        np.save(edited_path(char_name, motion_name, key), array)
    return True

def load_full_edited_npy_motion(args, character, motion_name):
    """(R6, root_p) motions for every role and scale. 0: ptn, 1: deform."""
    deform_name = character.meshes[0].mesh_gl.name

    root_p0, local_R0, root_p1, local_R1 = [
        np.load(edited_path(deform_name, motion_name, key)) for key in KEYS]

    num_role, num_scale, len_frame, _ = root_p0.shape
    motion0 = np.concatenate((R_to_R6(local_R0).reshape(num_role, num_scale, len_frame, -1), root_p0), axis=-1)
    motion1 = np.concatenate((R_to_R6(local_R1).reshape(num_role, num_scale, len_frame, -1), root_p1), axis=-1)

    return motion0, motion1

def load_edited_npy_motion(args, motionA, motionB, char_name, motion_name,
                           leg_scale, body_scale, hand_scale, rid, sid):
    """Return the result of role rid and scale sid as Motions."""
    motion0, motion1 = deepcopy(motionA), deepcopy(motionB)
    if rid==0:
        scale_offset(args, motion1, leg_scale, body_scale, hand_scale)
    else:
        scale_offset(args, motion0, leg_scale, body_scale, hand_scale)

    root_p0, local_R0, root_p1, local_R1 = [
        np.load(edited_path(char_name, motion_name, key))[rid, sid] for key in KEYS]

    for i in range(len(motion0.poses)):
        motion0.poses[i].root_p = root_p0[i]
        motion0.poses[i].local_R = local_R0[i]
        motion0.poses[i].update()
        motion1.poses[i].root_p = root_p1[i]
        motion1.poses[i].local_R = local_R1[i]
        motion1.poses[i].update()

    return motion0, motion1
