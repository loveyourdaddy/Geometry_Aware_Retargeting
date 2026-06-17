from Geometry.compare_geometry import *
from pymovis.motion.ops.torchmotion import R6_to_R
# from datasets.character_dataset import *
from datasets.motion_dataset import *
from datasets.motion_dataset import get_distance_map

def check_semantic(args,
                   source_motion0, source_motion1,
                   source_offset0, source_offset1,
                   source_geo0, source_geo1,
                   parent_idx0, parent_idx1):
    # source
    source_p0, source_R0, _ = get_rootP_localR_globalP_from_motion(args, source_motion0.poses)
    source_p1, source_R1, _ = get_rootP_localR_globalP_from_motion(args, source_motion1.poses)
    _, source_pos0 = R_fk_from_given_info(source_R0, source_p0, source_offset0, parent_idx0)
    _, source_pos1 = R_fk_from_given_info(source_R1, source_p1, source_offset1, parent_idx1)

    """ skel distance """
    input_dist_map = get_distance_map(source_pos0, source_pos1)
    
    """ anchor distance """
    b_size, len_frame = 1, source_R0.shape[0]
    len_vids = source_geo0.anchor_vpos.shape[0]
    batch = torch.arange(b_size).reshape(b_size, 1, 1).repeat(1, len_frame, len_vids).to(args.device)
    frame = torch.arange(len_frame).reshape(1, len_frame, 1).repeat(b_size, 1, len_vids).to(args.device)

    # SMPLx: mesh skinning weight잘못설정되어있음 : 수정해주기. TODO
    # source
    geo, offset, local_R, root_p = source_geo0, source_offset0, source_R0, source_p0
    source_anchor_positions0 = get_anchor_position(
        offset, local_R.unsqueeze(0), root_p.unsqueeze(0),
        geo.anchor_vpos.reshape(1, 1, len_vids, 3).repeat(b_size, len_frame, 1, 1).to(args.device), 
        geo.anchor_vids.reshape(1, 1, len_vids).repeat(b_size, len_frame, 1).to(args.device), 
        batch, frame,
        geo.name_to_idx, geo.bind_trf_inv, geo.parents, geo.names,
        geo.skinning_indices1[geo.anchor_vids], geo.skinning_weights1[geo.anchor_vids], geo.skinning_indices2[geo.anchor_vids], geo.skinning_weights2[geo.anchor_vids])
    
    geo, offset, local_R, root_p = source_geo1, source_offset1, source_R1, source_p1
    source_anchor_positions1 = get_anchor_position(
        offset, local_R.unsqueeze(0), root_p.unsqueeze(0),
        geo.anchor_vpos.reshape(1, 1, len_vids, 3).repeat(b_size, len_frame, 1, 1).to(args.device), 
        geo.anchor_vids.reshape(1, 1, len_vids).repeat(b_size, len_frame, 1).to(args.device), 
        batch, frame,
        geo.name_to_idx, geo.bind_trf_inv, geo.parents, geo.names,
        geo.skinning_indices1[geo.anchor_vids], geo.skinning_weights1[geo.anchor_vids], geo.skinning_indices2[geo.anchor_vids], geo.skinning_weights2[geo.anchor_vids])
    
    # distance  
    input_anchor_dist = get_distance_map(source_anchor_positions0, source_anchor_positions1)

    return input_dist_map, input_anchor_dist
    
def get_contact_tensor(args, 
                   geo0, geo1,
                   motion0, motion1,
                   ):
    
    """ count interpenetration """
    # set as CPU 
    args.device='cpu'
    set_geo_device(geo0, args.device)
    set_geo_device(geo1, args.device)
    len_frame = len(motion0.poses)
    
    d2_cids0, d2_cids1, d2_jids0, d2_jids1, d2_frame = \
        collision_detection(args, geo0, geo1, motion0, motion1) 
    if d2_frame == None:
        return torch.full((len_frame, 22, 22), False)
    # triangle level collision detection
    # d2_num_col = 0
    # for d2_cid in d2_cids0:
    #     d2_num_col += len(d2_cid)
    # print("d2_num_col: ", d2_num_col)
    
    
    """ contact missing / preserving """
    # # unique 
    contact_tensor = torch.full((len_frame, 22, 22), False)
    contact_tensor[d2_frame, d2_jids0, d2_jids1] = True
    
    return contact_tensor # [frames, j0, j1]
    

""" geo & anchor """
def set_geo_device(geo, device):
    geo.args.device='cpu'
    geo.bvh_tree.args.device='cpu'
    
    geo.device = device
    geo.cid_to_first_vid = geo.cid_to_first_vid.to(device)
    geo.v_position = geo.v_position.to(device)
    geo.bvh_tree.boundary_pos = geo.bvh_tree.boundary_pos.to(device)
    geo.skinning_indices1 = geo.skinning_indices1.to(device)
    geo.skinning_indices2 = geo.skinning_indices2.to(device)
    geo.skinning_weights1 = geo.skinning_weights1.to(device)
    geo.skinning_weights2 = geo.skinning_weights2.to(device)
    geo.mesh_global_R = geo.mesh_global_R.to(device)
    
from Network.network import set_pose_by_source_batch_frame, get_positions_from_vids

def get_anchor_position(offset, gt_R, gt_root_p, 
                        anchor_vpos_b, anchor_vids_b, batch, frame,
                        name_to_idx, bind_trf_inv, parents, names,
                        skinning_indices1, skinning_weights1, skinning_indices2, skinning_weights2):
    gt_global_R = set_pose_by_source_batch_frame(name_to_idx, bind_trf_inv, offset, parents, names, gt_R, gt_root_p)
    gt_anchor_positions = get_positions_from_vids(skinning_indices1, skinning_weights1, skinning_indices2, skinning_weights2,
            gt_global_R, anchor_vpos_b, anchor_vids_b, batch, frame)

    return gt_anchor_positions
