import sys
sys.path.append('..')

from pymovis.vis.appmanager import AppManager
from pymovis.vis.app import MyApp
from datasets.character_functions import *
from datasets.motion_functions import *
import option_parser
from etc.etc import *
from pymovis.motion.ops.torchmotion import *
# from option_motion import example_bvh
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
names = ["SMPLx", "SMPLx"] # Remy Leonard Amy Ortiz
characters, motions = [], []
geos = []
vids = []
vpositions = []
after_vpositions = []
motion_name = "back_lift001_S1"

# TODO: interaction이름을 각각 가져오기.
for character_name0 in names:
    source0_character, motion0 = get_a_character(args, character_name0, template)
    # source0_character, motion0, geo0 = get_a_character(args, character_name0, template)
    motion0 = get_interaction_motions_from_list(character_name0, [motion_name])[0]
    # motion0 = refine_motion(motion0, template)
    
    # source0_character.set_source_skeleton(motion0.skeleton, MIXAMO_BVH_TO_FBX) # character도 같이 입력으로 받아, skeleton을 return해주기
    characters.append(source0_character)
    motions.append(motion0)
    # print(character_name0, (geo0.f_length))
    
    # vid = geo0.anchor_vids
    # root_p, local_R, _ = get_rootP_localR_globalP_from_motion(args, motion0.poses)
    # geo0.set_pose_by_source_batch_frame(local_R[None, ...], root_p[None, ...])
    # f = 517
    # vpos = geo0.get_positions_from_vids(torch.tensor(np.array(geo0.anchor_vids)), torch.tensor([0]), torch.tensor([f]))
    # # print(vpos)
    # # vpos = vpos[30]
    # # vpositions.append(vpos)
    # args.debug_points0 = vpos

for i, motion in enumerate(motions):
    for pose in motion.poses:
        pose.translate_root_p([i*2, 0, 0])
app = MyApp(characters, motions, args)
app_manager.run(app)
