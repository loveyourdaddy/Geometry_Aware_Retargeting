import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from pymovis.vis.appmanager import AppManager
from pymovis.vis.app import MyApp
# from dataset import *
from datasets.motion_functions import *
from datasets.character_functions import *
import option_parser
from pymovis.motion.ops.npmotion import *
from etc.etc import *
from pymovis.motion.data.fbx import FBX
from Geometry.geometry import Geometry 


app_manager = AppManager()
args = option_parser.get_args()
args.path = args.proj_name + '/'
args.geo_preprocess = False  
args.bvh_preprocess = False
args.device = "cpu"

template_Tpose = bvh.load("../Resource/Tpose_template.bvh", v_forward=[0, 0, 1],v_up=[0, 1, 0])
characters = []
motions = [] 
offsets = []
anchor_positions = []
render_pos = []

character_names = ["SMPLx_sub", "SMPLx_fat_sub",] # "SMPLx", "SMPLx_fat", "Ybot", "Amy", "Ortiz","Leonard",
for character_name in character_names:
    # load character, motion
    if character_name =="SMPLx" or character_name =="SMPLx_sub" or character_name =="SMPLx_fat" or character_name =="SMPLx_fat_sub":
        dict = ""
    else:
        dict = MIXAMO_BVH_TO_FBX

    # char 
    path = "Resource/models/{}.fbx".format(character_name)
    fbx = FBX(path, character_name)
    character_normal = fbx.model()

    # motion 
    motion_name = ["Tpose"]  # SMPLx_1_C0
    motion = get_interaction_motions_from_list(character_name, motion_name)[0]
    motion = refine_motion(motion, template_Tpose)
    character_normal.set_source_skeleton(motion.skeleton, dict)
    motion.poses = motion.poses[:1]
    
    """ aabb """
    # set pose 
    geometry = Geometry(args, character_normal, motion)
    geometry.set_pose_by_source_batch_frame(
        torch.tensor(motion.poses[0].local_R).reshape(1,1,22,3,3), 
        torch.tensor(motion.poses[0].root_p).reshape(1,1,3))

    vids = geometry.descriptor_vids
    len_vids = vids.shape[0]
    vids = torch.tensor(vids).reshape(1,1,len_vids)
    batches = torch.tensor([0]).reshape(1,1,1).repeat(1,1,len_vids)
    frames = torch.tensor([0]).reshape(1,1,1).repeat(1,1,len_vids)
    anchor_pos = geometry.get_positions_from_vids(vids, batches, frames)[0,0].cpu().numpy()
    
    # render 
    characters.append(character_normal)
    motions.append(motion)
    anchor_positions.append(anchor_pos) # .reshape(-1,3)

# Arrange
distance = 2.0
for i in range(len(characters)):
    # aabb
    anchor_pos = anchor_positions[i].reshape(-1, 3)
    anchor_pos[:, 0] += i*distance
    if i == 0:
        args.debug_points1.append(anchor_pos) # 176, 3
    else:
        args.debug_points2.append(anchor_pos) # 176, 3

render_motions(args, characters, motions)

app = MyApp(characters, motions, args)
app_manager.run(app)
