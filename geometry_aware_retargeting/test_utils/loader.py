"""Load the characters, network, and reference geometry for testing."""
from dataclasses import dataclass

from Network.network import Network
from datasets.character_functions import load_char


@dataclass
class TestSetup:
    """Characters and network shared by all motions."""
    src_char0: object
    src_char1: object
    tgt_char0: object
    tgt_char1: object
    src_name0: str
    src_name1: str
    net: Network
    ref_geo: dict = None


def get_ref_geo(args):
    """Fixed reference character geometry for computing the anchor input."""
    from datasets.character_functions import get_a_smpl_character
    _, _, ref_geo_obj = get_a_smpl_character(args, "SMPLx")
    return {
        "anchor_vids": ref_geo_obj.anchor_vids,
        "anchor_vpos": ref_geo_obj.anchor_vpos,
        "name_to_idx": ref_geo_obj.name_to_idx,
        "bind_trf_inv": ref_geo_obj.bind_trf_inv,
        "skinning_indices1": ref_geo_obj.skinning_indices1[ref_geo_obj.anchor_vids],
        "skinning_weights1": ref_geo_obj.skinning_weights1[ref_geo_obj.anchor_vids],
        "skinning_indices2": ref_geo_obj.skinning_indices2[ref_geo_obj.anchor_vids],
        "skinning_weights2": ref_geo_obj.skinning_weights2[ref_geo_obj.anchor_vids],
        "perjoint_vids": ref_geo_obj.perjoint_vids,
        "parents": ref_geo_obj.parents,
        "names": ref_geo_obj.names,
    }


def load_network(args):
    """Load --checkpoint if given, otherwise saved/{test_proj}/spatio_temp_net_{test_epoch}.pt, and return the network in eval mode."""
    net = Network(args)
    if args.checkpoint:
        net.load_file(args.checkpoint, device=args.device)
    else:
        net.load(args.test_proj + '/', args.test_epoch, device=args.device)
    net.eval()
    return net


def load_test_setup(args):
    """Load the characters and network once. Must be called after the GL context (AppManager) is created."""
    src_char0, src_char1, tgt_char0, tgt_char1, \
        _, _, src_name0, src_name1 = load_char(args)
    net = load_network(args)
    ref_geo = get_ref_geo(args) if args.use_anchor_input else None
    return TestSetup(src_char0, src_char1, tgt_char0, tgt_char1,
                     src_name0, src_name1, net, ref_geo)
