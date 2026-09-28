"""Skeleton offset scaling."""


def scale_character(args, character, leg_scale, body_scale, hand_scale):
    for joint in args.leg_joints:
        character.meshes[0].source_skeleton.joints[joint].offset *= leg_scale
    for joint in args.body_joints:
        character.meshes[0].source_skeleton.joints[joint].offset *= body_scale
    for joint in args.hand_joints:
        character.meshes[0].source_skeleton.joints[joint].offset *= hand_scale

def scale_offset(args, motion, leg_scale, body_scale, hand_scale):
    for joint in args.leg_joints:
        motion.skeleton.joints[joint].offset *= leg_scale
    for joint in args.body_joints:
        motion.skeleton.joints[joint].offset *= body_scale
    for joint in args.hand_joints:
        motion.skeleton.joints[joint].offset *= hand_scale

def scale_offset_and_root(args, motion, root_scale, leg_scale, body_scale, hand_scale):
    scale_offset(args, motion, leg_scale, body_scale, hand_scale)
    for pose in motion.poses:
        pose.root_p[1] *= root_scale
        pose.update()

def fit_motion_to_offsets(motion, offsets):
    """Replace the skeleton offsets of the motion with the target character's and match the root height."""
    for joint, offset in zip(motion.skeleton.joints, offsets):
        joint.offset = offset.copy()
    root_scale = offsets[0][1]
    for pose in motion.poses:
        pose.root_p *= root_scale
        pose.update()
