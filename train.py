'''
python train.py

python train.py --network_type no_cross_attn --proj_name ablation_no_cross
python train.py --network_type mlp --proj_name ablation_mlp

conda env create -f environment.yml
conda activate py_env37_2
'''
import sys
sys.path.append('..')

import random
import torch.backends.cudnn as cudnn
# import wandb
import option_parser
from datasets.motion_dataset import *
from datasets.motion_functions import *
from datasets.character_functions import *
from Network.network import Network


def main(args):
    args.path = args.proj_name + '/'

    seed_value = args.seed_value
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(0)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(0)

    set_rot_dim(args)
    args.is_train = True
    args.begin_epoch = 0
    args.end_epoch = 100000  # 100,000
    # wandb.init(
    #     project="GeometryAwareRetargeting",
    #     name=args.proj_name,
    #     mode="online"
    # )

    # set train character
    train_mesh_char_list = args.target_characters
    dataset = Dataset(args)
    dataset.load_motion_and_geo_data(train_mesh_char_list)

    # print
    if args.loss_fk:
        print("> lambda_fk:", args.lambda_fk)
    if args.loss_anchor:
        print("> lambda_anchor:", args.lambda_anchor)
    print("> proj_name: ", args.proj_name)
    print("> train character: ", train_mesh_char_list)

    # train
    net = Network(args)
    net.spatio_temp_net.train()
    net.train_(dataset)
    print("train done")


if __name__ == "__main__":
    args = option_parser.get_args()
    main(args)
