import sys
sys.path.append('..')

from pymovis.vis.appmanager import AppManager
from pymovis.vis.app import MyApp
from datasets.character_functions import *
from datasets.motion_functions import *
import option_parser
from etc.etc import *
from pymovis.motion.ops.torchmotion import *
from option_motion import example_bvh
from Retarget_SMPL.relationship_descriptor import get_rootP_localR_globalP_from_motion

app_manager = AppManager()
args = option_parser.get_args()
args.path = args.proj_name + '/'
args.device = "cpu" # cuda

template = bvh.load(
    "../Resource/Tpose_template.bvh", v_forward=[0, 0, 1], v_up=[0, 1, 0]
)

# char
# args.geo_preprocess = True
# args.bvh_preprocess = True
names = ["SMPLx_fat", "SMPLx_fat"] # Ybot Amy SMPLx Remy Leonard Amy Ortiz
characters, motions = [], []
geos = []
vids = []
vpositions = []
after_vpositions = []
motion_name = "Tpose"

for i, character_name0 in enumerate(names):
    if i==0:
        mesh_scale = 1
    else:
        mesh_scale = 1.1

    source0_character, motion0, _ = get_a_character(args, character_name0, template, mesh_scale) # 
    if i==0:
        for mesh in source0_character.meshes:
            mesh.materials[0].alpha=0.5
    # motion0 = get_single_motion_from_list("SMPLx", [motion_name])[0] # character_name0 interaction
    source0_character.set_source_skeleton(motion0.skeleton, "") # MIXAMO_BVH_TO_FBX
    characters.append(source0_character)
    motions.append(motion0)

distance = 3
for i, motion in enumerate(motions):
    for pose in motion.poses:
        # pose.translate_root_p([i*distance, 0, 0])
        pose.translate_root_p([0, 0, -i*distance])

app = MyApp(characters, motions, args)
app_manager.run(app)
