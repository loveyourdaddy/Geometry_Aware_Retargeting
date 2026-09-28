"""Post-processing of the network output motions."""
from Retarget_SMPL.relationship_descriptor import resolve_ground_pene


def postprocess(args, out_motion0, out_motion1):
    """Apply only the post-processing enabled by the options."""
    if args.resolve_ground_pene:
        out_motion0, out_motion1 = resolve_ground_pene(args, out_motion0, out_motion1)
    return out_motion0, out_motion1
