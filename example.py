import sys
sys.path.append('..')

from pymovis.vis.appmanager import AppManager
from pymovis.vis.app import MyApp
from datasets.character_functions import *
from datasets.motion_functions import *
import option_parser
from etc.etc import *
from pymovis.motion.ops.torchmotion import *
from pymovis.motion.data.fbx import FBX

app_manager = AppManager()
args = option_parser.get_args()
args.path = args.proj_name + '/'
args.device = "cpu" # cuda

template = bvh.load(
    "../Resource/Tpose_template.bvh", v_forward=[0, 0, 1], v_up=[0, 1, 0]
)

# char
names = ["Leonard"] # "SMPLx",  "YBot" "Leonard" "Remy" "Amy" "Ortiz" "SMPLx"
characters, motions = [], []
for i, name in enumerate(names):
    path = "../Resource/models/{}.fbx".format(name)
    fbx = FBX(path, name)
    character = fbx.model()
    Tpose = bvh.load(
        "../Resource/motions/single_motion/{}/Tpose.bvh".format(name),
        v_forward=[0, 0, 1],
        v_up=[0, 1, 0],
    )
    
    if template is None:
        template = bvh.load(
            "../Resource/Tpose_template.bvh", v_forward=[0, 0, 1], v_up=[0, 1, 0]
        )
    # Tpose = refine_motion(Tpose, template)
    
    # character.set_source_skeleton(Tpose.skeleton, "") # MIXAMO_BVH_TO_FBX
    character.set_source_skeleton(Tpose.skeleton, MIXAMO_BVH_TO_FBX)
    characters.append(character)
    motions.append(Tpose)

# distance = 3
# for i, motion in enumerate(motions):
#     for pose in motion.poses:
#         # pose.translate_root_p([i*distance, 0, 0])
#         pose.translate_root_p([0, 0, -i*distance])

app = MyApp(characters, motions, args)
app_manager.run(app)
