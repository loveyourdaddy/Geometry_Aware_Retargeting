"""Inference for a single motion pair."""
from datasets.motion_dataset import Dataset
from datasets.motion_functions import get_interaction_motions_from_list, make_new_motions
from test_utils.postprocess import postprocess


def run_network(args, setup, motion_name0, motion_name1):
    """Return (out_motion0, out_motion1, src_motion0, src_motion1) for a single motion pair."""
    src_motion0 = get_interaction_motions_from_list(setup.src_name0, [motion_name0])[0]
    src_motion1 = get_interaction_motions_from_list(setup.src_name1, [motion_name1])[0]

    dataset = Dataset(args)
    dataset.get_char_data(setup.src_char0, setup.src_char1, setup.tgt_char0, setup.tgt_char1)
    dataset.get_input_motion(src_motion0, src_motion1)
    if args.use_anchor_input:
        dataset.get_input_anchor_pos(setup.ref_geo)
    if args.use_relative_pos_input:
        dataset.get_input_relative_pos()
    if args.data_normalized:
        dataset.load_norm_info()
        dataset.normalize()

    out_p0, out_R0, out_p1, out_R1 = setup.net.forward(dataset)
    out_motion0, out_motion1 = make_new_motions(
        args, out_p0, out_R0, out_p1, out_R1,
        setup.tgt_char0, setup.tgt_char1, src_motion0, src_motion1,
    )
    out_motion0, out_motion1 = postprocess(args, out_motion0, out_motion1)
    return out_motion0, out_motion1, src_motion0, src_motion1
