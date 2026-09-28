import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
from etc.etc import deepcopy
from datasets.motion_functions import refine_motion, select_skeleton_and_finger_idx
from pymovis.vis.const import MIXAMO_BVH_TO_FBX
from pymovis.motion.data import bvh

def load_char(args):
    test_char = args.test_char
    if args.test_type=="SMPLx":
        # name 
        source0_name = "SMPLx"
        source1_name = "SMPLx"
        
        # set
        if test_char == "normal":
            args.test_smpl_mesh = "SMPLx"
        elif test_char == "small":
            args.test_smpl_mesh = "SMPLx"
            args.SMPLx_scale = 0.7
            # args.SMPLx_mesh_scale = 0.7
        elif test_char == "fat":
            args.test_smpl_mesh = "SMPLx_fat"
        else:
            raise ValueError("Invalid test_char")
        
        # character
        source0_character, source1_character,\
            target0_characters, target1_characters,\
            target0_Tpose, target1_Tpose =\
                get_smpl_characters_wo_geo(args, [args.test_smpl_mesh], subsampled=args.subsampled)
        
        target0_character, target1_character = target0_characters[0], target1_characters[0]
        
        # scale character
        from Retarget_SMPL.retarget_smpl import scale_character 
        scale = args.SMPLx_scale
        leg_scale, body_scale, hand_scale = scale, scale, scale
        scale_character(args, target1_character, leg_scale, body_scale, hand_scale)
        
        print(f"scale: {leg_scale} {body_scale} {hand_scale}")
    elif args.test_type=="Mixamo":
        # name 
        source0_name = "Ybot" # Leonard
        source1_name = "Ybot" # Leonard
        test_mixamo_char0 = "Leonard" # Remy
        
        # set
        if test_char == "normal":
            test_mixamo_char1 = "Ybot" # Ybot Sporty_Granny CastleGuard
        elif test_char == "small":
            test_mixamo_char1 = "Amy"
        elif test_char == "fat":
            test_mixamo_char1 = "Ortiz"
        else:
            raise ValueError("Invalid test_char")

        # character, T pose
        source0_character, source1_character, \
        target0_characters, target1_characters, \
        target0_Tpose, target1_Tpose, = \
            get_characters_wo_geo(args, source0_name, source1_name, test_mixamo_char0, test_mixamo_char1)
        target0_character, target1_character = target0_characters[0], target1_characters[0]
    else:
        raise ValueError("Invalid test_type: ", args.test_type)
    
    return source0_character, source1_character, target0_character, target1_character, \
        target0_Tpose, target1_Tpose, source0_name, source1_name 

def get_skeleton_finger_idx(targ0_Tpose, targ1_Tpose):
    target0_skeleton_idx, target0_finger_idx = select_skeleton_and_finger_idx(targ0_Tpose)
    target1_skeleton_idx, target1_finger_idx = select_skeleton_and_finger_idx(targ1_Tpose)
    return target0_skeleton_idx, target0_finger_idx, target1_skeleton_idx, target1_finger_idx

""" smpl """
# a character
def get_a_smpl_character(args, name_normal, scale=1, mesh_scale=1):
    # print(mesh_scale)
    from pymovis.motion.data.fbx import FBX
    from Geometry.geometry import Geometry 
    template_Tpose = bvh.load(
        "../Resource/Tpose_template.bvh", v_forward=[0, 0, 1], v_up=[0, 1, 0]
    )

    # model
    path = "../Resource/models/{}.fbx".format(name_normal)
    fbx = FBX(path, name_normal, scale=scale, mesh_scale=mesh_scale)
    character = fbx.model()
    Tpose = bvh.load(
        "../Resource/motions/single_motion/{}/Tpose.bvh".format(name_normal),
        v_forward=[0, 0, 1],
        v_up=[0, 1, 0]  
    )
    Tpose.poses = Tpose.poses[0:1]
    Tpose = refine_motion(Tpose, template_Tpose)
    character.set_source_skeleton(Tpose.skeleton, "")
    
    # geometry
    if scale!=1:
        geo = Geometry(args, character, Tpose, scale=scale)# m3sh_Scale
    elif mesh_scale!=1:
        geo = Geometry(args, character, Tpose, scale=mesh_scale)
    else:
        geo = Geometry(args, character, Tpose)

    return character, Tpose, geo

def get_a_smpl_character_wo_geo(args, target_name, scale=1, mesh_scale=1):
    from pymovis.motion.data.fbx import FBX
    template_Tpose = bvh.load(
        "../Resource/Tpose_template.bvh", v_forward=[0, 0, 1], v_up=[0, 1, 0]
    )

    path = "../Resource/models/{}.fbx".format(target_name)
    fbx = FBX(path, target_name, scale=scale, mesh_scale=mesh_scale)
    character = fbx.model()
    
    # if subsampled, remove "_sub"
    if args.subsampled and "_sub" in target_name:
        target_name = target_name.replace("_sub", "")
        
    Tpose = bvh.load(
        "../Resource/motions/single_motion/{}/Tpose.bvh".format(target_name),
        v_forward=[0, 0, 1],
        v_up=[0, 1, 0]
    )
    Tpose.poses = Tpose.poses[0:1]
    Tpose = refine_motion(Tpose, template_Tpose)
    character.set_source_skeleton(Tpose.skeleton, "")
    
    return character, Tpose

# scale 
def get_uniform_scale(target_name):
    if target_name == "SMPLx":
        scales = np.load("scales.npy")
    elif target_name == "SMPLx_fat": 
        scales = np.load("scales_fat.npy")
    else: 
        raise ValueError("no scale name1: ", target_name)
    return scales

def get_scale():
    scales = np.load("./scale_values/scales_sampled.npy")
    return scales

def get_a_mesh_scaled_smpl_character_with_geo(args, target_name, scale=1.0):
    from pymovis.motion.data.fbx import FBX
    from Geometry.geometry import Geometry 
    template_Tpose = bvh.load(
        "../Resource/Tpose_template.bvh", v_forward=[0, 0, 1], v_up=[0, 1, 0]
    )
    scales = get_scale()
    scale = scales[args.SMPLx_scale_index]
    mean_scale = scale[0]

    path = "../Resource/models/{}.fbx".format(target_name)
    fbx = FBX(path, target_name, scale=mean_scale)
    character_normal = fbx.model()
    Tpose_normal = bvh.load(
        "../Resource/single_motion/{}/Tpose.bvh".format(target_name),
        v_forward=[0, 0, 1],
        v_up=[0, 1, 0]
    )
    Tpose_normal.poses = Tpose_normal.poses[0:1]
    Tpose_normal = refine_motion(Tpose_normal, template_Tpose)
    character_normal.set_source_skeleton(Tpose_normal.skeleton, "")
    geometry_normal = Geometry(args, character_normal, Tpose_normal)


    character_target = character_normal
    character_Tpose = Tpose_normal
    geometry_target = geometry_normal

    return character_target, character_Tpose, geometry_target

# characters 
def get_smpl_characters(args, deform_names, SMPLx_small=False):
    # load sample character
    character_normal, Tpose_normal, geometry_normal = get_a_smpl_character(args, "SMPLx")

    # source0
    source0_character, source0_geometry = \
        character_normal, deepcopy(geometry_normal)
    source0_geometrys = source0_geometry

    # source1
    source1_character, source1_geometry = \
        character_normal, deepcopy(geometry_normal)
    target0_Tpose = deepcopy(Tpose_normal)
    source1_geometrys = source1_geometry

    # target0
    target0_characters = []
    for deform_name in deform_names:
        target0_character, target0_geometry = \
            deepcopy(character_normal), deepcopy(geometry_normal)
        target0_characters.append(target0_character)
        target0_geometrys = target0_geometry

    # target1
    target1_characters = []
    for deform_name in deform_names:
        if SMPLx_small:
            target1_character, target1_Tpose, target1_geometry = get_a_mesh_scaled_smpl_character_with_geo(args, deform_name)
        else:
            target1_character, target1_Tpose, target1_geometry = get_a_smpl_character(args, deform_name)
        target1_characters.append(target1_character)
        target1_geometrys = target1_geometry

    return source0_character, source1_character, target0_characters, target1_characters,\
        source0_geometrys, source1_geometrys, target0_geometrys, target1_geometrys,\
        target0_Tpose, target1_Tpose

def get_smpl_characters_wo_geo(args, deform_names, subsampled=False):
    source_name = "SMPLx"
    if subsampled:
        source_name += "_sub"
        
    # source0, source1
    source0_character, Tpose_normal = get_a_smpl_character_wo_geo(args, source_name)
    source1_character = deepcopy(source0_character)

    # target0
    target0_characters = []
    for deform_name in deform_names: # does not have to be a list
        target0_character = source0_character 
        target0_Tpose = deepcopy(Tpose_normal)
        target0_characters.append(target0_character)

    # target1
    target1_characters = []
    for deform_name in deform_names: # does not have to be a list
        if subsampled:
            deform_name += "_sub"
            
        # the scale option is the size of the whole character
        target1_character, target1_Tpose = get_a_smpl_character_wo_geo(args, deform_name, scale=args.SMPLx_scale, mesh_scale=args.SMPLx_mesh_scale)
        target1_characters.append(target1_character)

    return source0_character, source1_character, target0_characters, target1_characters, \
        target0_Tpose, target1_Tpose


""" Load Mixamo """
# a character
def get_a_character(args, name, template=None, scale=1, mesh_scale=1):
    from pymovis.motion.data.fbx import FBX
    path = "../Resource/models/{}.fbx".format(name)
    fbx = FBX(path, name, scale, mesh_scale)
    character = fbx.model()
    Tpose = bvh.load(
        "./Resource/motions/{}/Tpose.bvh".format(name),
        v_forward=[0, 0, 1],
        v_up=[0, 1, 0],
    )
    if template is None:
        template = bvh.load(
            "../Resource/Tpose_template.bvh", v_forward=[0, 0, 1], v_up=[0, 1, 0]
        )
    Tpose = refine_motion(Tpose, template)
    character.set_source_skeleton(Tpose.skeleton, MIXAMO_BVH_TO_FBX)

    return character, Tpose

def get_a_smt_character(args, name, template=None, scale=1, mesh_scale=1):
    from pymovis.motion.data.fbx import FBX
    path = "../Resource/smt_models/{}.fbx".format(name)
    fbx = FBX(path, name, scale, mesh_scale)
    character = fbx.model()
    Tpose = bvh.load(
        "../Resource/smt_motions/{}/Tpose.bvh".format(name),
        v_forward=[0, 0, 1],
        v_up=[0, 1, 0],
    )
    if template is None:
        template = bvh.load(
            "../Resource/Tpose_template.bvh", v_forward=[0, 0, 1], v_up=[0, 1, 0]
        )
    Tpose = refine_motion(Tpose, template)
    character.set_source_skeleton(Tpose.skeleton, MIXAMO_BVH_TO_FBX)

    return character, Tpose

def get_a_character_wo_geo(args, name, template=None):
    from pymovis.motion.data.fbx import FBX
    path = "../Resource/models/{}.fbx".format(name)
    print(f"get_a_character_wo_geo: {path}, {name}")
    fbx0 = FBX(path, name)
    character = fbx0.model()
    
    # if subsampled, remove "_sub"
    if args.subsampled and "_sub" in name:
        name = name.replace("_sub", "")
    
    Tpose = bvh.load(
        "../Resource/motions/single_motion/{}/Tpose.bvh".format(name),
        v_forward=[0, 0, 1],
        v_up=[0, 1, 0],
    )
    Tpose_full = deepcopy(Tpose)
    if template is None:
        template = bvh.load(
            "../Resource/motions/Tpose_template.bvh", v_forward=[0, 0, 1], v_up=[0, 1, 0]
        )
    Tpose = refine_motion(Tpose, template)

    if name == "SMPLx" or name == "SMPLx_fat":
        character.set_source_skeleton(Tpose.skeleton, "")
    else:
        character.set_source_skeleton(Tpose.skeleton, MIXAMO_BVH_TO_FBX)

    return character, Tpose_full, Tpose

def get_a_character_wo_geo_with_finger(name, template):
    from pymovis.motion.data.fbx import FBX
    path = "../Resource/models/{}.fbx".format(name)
    fbx0 = FBX(path, name)
    character = fbx0.model()
    
    # if subsampled, remove "_sub"
    if args.subsampled and "_sub" in name:
        name = name.replace("_sub", "")
    
    Tpose = bvh.load(
        "../Resource/single_motion/{}/Tpose.bvh".format(name),
        v_forward=[0, 0, 1],
        v_up=[0, 1, 0],
    )
    Tpose_full = deepcopy(Tpose)
    Tpose = refine_motion(Tpose, template)

    character.set_source_skeleton(Tpose.skeleton, MIXAMO_BVH_TO_FBX)
    skeleton_idx, finger_idx = select_skeleton_and_finger_idx(Tpose_full)

    return character, Tpose_full, Tpose, skeleton_idx, finger_idx

# characters
def get_characters_wo_geo(args, source0_name, source1_name, target0_name, target1_name):
    template_Tpose = bvh.load(
        "../Resource/Tpose_template.bvh", v_forward=[0, 0, 1], v_up=[0, 1, 0]
    )
    
    # sub
    if args.subsampled:
        source0_name = source0_name + "_sub"
        source1_name = source1_name + "_sub"
        target0_name = target0_name + "_sub"
        target1_name = target1_name + "_sub"

    # load sample_character
    character, Tpose, _ = \
        get_a_character_wo_geo(args, source0_name, template_Tpose) # replace with wo finger

    # source0: ybot
    source0_character = deepcopy(character)

    # source0: ybot
    source1_character = deepcopy(character)

    # target0
    target0_character, target0_Tpose, _ = \
        get_a_character_wo_geo(args, target0_name, template_Tpose)
    target0_characters = []
    target0_characters.append(target0_character)

    # target1
    target1_character, target1_Tpose, _ = \
        get_a_character_wo_geo(args, target1_name, template_Tpose)
    target1_characters = []
    target1_characters.append(target1_character)

    return source0_character, source1_character, target0_characters, target1_characters,\
        target0_Tpose, target1_Tpose

def get_characters(args, source0_name, source1_name, target0_name, target1_name):
    template_Tpose = bvh.load(
        "../Resource/Tpose_template.bvh", v_forward=[0, 0, 1], v_up=[0, 1, 0]
    )

    # load sample_character
    character, Tpose, geo = \
        get_a_character(args, source0_name, template_Tpose)

    # source0: ybot
    source0_character = deepcopy(character)
    source_geo0 = deepcopy(geo)

    # source0: ybot
    source1_character = deepcopy(character)
    source_geo1 = deepcopy(geo)

    # target0
    target0_character, target0_Tpose, target_geo0 = \
        get_a_character(args, target0_name, template_Tpose)
    target0_characters = []
    target0_characters.append(target0_character)

    # target1
    target1_character, target1_Tpose, target_geo1 = \
        get_a_character(args, target1_name, template_Tpose)
    target1_characters = []
    target1_characters.append(target1_character)

    return source0_character, source1_character, target0_characters, target1_characters,\
        source_geo0, source_geo1, target_geo0, target_geo1, \
        target0_Tpose, target1_Tpose
