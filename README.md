# Geometry-aware Retargeting

Training and testing code for a network that transfers the interaction motion of two characters to a pair of characters with different body shapes (SMPLx, Mixamo).

## Folder structure

Run every command inside `geometry_aware_retargeting/`. The code refers to `../datasets`, `../pymovis`, and `../Resource` by relative paths, so the layout below is required.

```
Retargeting_workspace/
├── Resource/                     # characters (FBX), motions (BVH)
│   ├── models/{character}.fbx
│   ├── motions/single_motion/{character}/Tpose.bvh
│   ├── motions/interaction_motion/{character}/{motion}.bvh
│   └── Tpose_template.bvh
├── datasets/                     # data loading, preprocessing scripts
├── Geometry/  pymovis/  etc/     # mesh geometry, renderer, utilities
└── geometry_aware_retargeting/   # this repository
    ├── train.py                  # training
    ├── test.py                   # testing (viewer / saving results)
    ├── test_utils/               # loader, inference, postprocess, saver, viewer, foot contact detection
    ├── prepare/                  # download scripts for the data and checkpoint
    ├── Network/                  # network, loss
    ├── option_parser.py          # run options
    ├── option_motion.py          # motion lists for training and testing
    ├── Retarget_SMPL/            # data acquisition through SMPL
    ├── adapted_motion/           # SMPL retargeting results (not in git)
    └── saved/                    # checkpoints (not in git)
```

## Environment

```bash
conda env create -f environment.yml
conda activate gar
```

- Python 3.7, PyTorch 1.12.1 (CUDA 11.3)
- The Autodesk FBX Python SDK is not installed by `environment.yml`. Install it separately so that `import fbx` works.
- The viewer requires OpenGL 4.3 or later and a display.

## Data download

```bash
pip install gdown
bash prepare/download_data.sh
```

Downloads the motion BVH files from Google Drive into `../Resource/`.

```
Resource/
├── Tpose_template.bvh
└── motions/
    ├── interaction_motion/{SMPLx,Ybot}/{motion}.bvh   # input motions
    └── single_motion/{character}/Tpose.bvh            # character T-poses
```

### Files required for testing

The motion BVH files and the checkpoint alone are not enough to run `test.py`. All four items below are required.

| File | Location | How to prepare |
|---|---|---|
| Motion BVH | `../Resource/motions/`, `../Resource/Tpose_template.bvh` | `bash prepare/download_data.sh` |
| Checkpoint | `saved/best.pt` | `bash prepare/download_checkpoints.sh` |
| Character models | `../Resource/models/{character}.fbx` | prepare yourself |
| Preprocessed files | `preprocess/{character}/` | prepare yourself |

The characters read by each test setting are as follows.

| `test_type` | `test_char` | Characters |
|---|---|---|
| `Mixamo` | `normal` | Ybot, Leonard |
| `Mixamo` | `small` | Amy |
| `Mixamo` | `fat` | Ortiz |
| `SMPLx` | `normal`, `small` | SMPLx |
| `SMPLx` | `fat` | SMPLx, SMPLx_fat |

### Character models

The models (FBX) are not included in the released files. Obtain them yourself and place them at `../Resource/models/{character}.fbx`.

| Character | Source |
|---|---|
| SMPLx, SMPLx_fat | based on [SMPL-X](https://smpl-x.is.tue.mpg.de) |
| Ybot, Leonard, Amy, Ortiz | [Mixamo](https://www.mixamo.com) |

The SMPL-X license prohibits distributing the model to third parties, so it is not uploaded here.

### Preprocessed files

The skeleton offsets and per-joint bounding boxes of each character.

| File | Shape | Required for |
|---|---|---|
| `offset.npy` | (22, 3) | every character |
| `aabb_max_min.npy` | (22, 6) | every character |
| `scaled_offset.npy` | (10, 22, 3) | SMPLx, SMPLx_fat used as the target |
| `scaled_aabb_max_min.npy` | (10, 22, 6) | SMPLx, SMPLx_fat used as the target |

They are created with `../datasets/save_offset.py`. Set the block to run, the characters, and the save paths inside the file before running it.

## Data preprocessing
### SMPL retargeting

Creates the retargeting data used for training through SMPL adaptation.

```bash
python Retarget_SMPL/retarget_smpl.py               # adapt the first motion of example_bvh and render it
python Retarget_SMPL/retarget_smpl.py --save True   # adapt all of RD_bvh and save
python Retarget_SMPL/check_retarget_result.py       # render the saved results
```

- Results are saved to `adapted_motion/{character}/{motion}_{root_p0,local_R0,root_p1,local_R1}.npy`, as arrays of shape (role, scale, frame, ...).
- The target characters and scales to save are set in `set_adapt_args` of `retarget_smpl.py`.
- If an existing file has a different number of scales, it is skipped instead of overwritten. To rebuild with different scales, move the existing folder away first.

| File | Role |
|---|---|
| `retarget_smpl.py` | entry point (viewer / save) |
| `check_retarget_result.py` | view the saved results |
| `adaptation.py` | adaptation of the two characters, per-motion joints to update and interaction range |
| `relationship_descriptor.py` | relationship descriptor algorithm |
| `motion_utils.py` | motion and tensor conversion, pose update from global positions |
| `ground_pene.py` | resolves foot-ground penetration |
| `skeleton_scale.py` | skeleton offset scaling |
| `edited_motion.py` | saving and loading `adapted_motion/` |

### Saving offset and aabb 
```bash
python ../datasets/save_motion_data.py   # preprocess/{character}/motion_windowed_randomScaled.pkl
```

`offset.npy` and `aabb_max_min.npy` are read in both training and testing (see "Preprocessed files" above for how to create them). `motion_windowed_randomScaled.pkl` is read in training.

## Training

```bash
python train.py --proj_name full
```

| Output | Location |
|---|---|
| Checkpoints | `saved/{yymmdd}_{proj_name}/spatio_temp_net_{epoch}.pt` (every `--save_iter_epoch`, default 500) |
| TensorBoard logs | `runs/{yymmdd}_{proj_name}/` |

The training motions are set by `trainset_bvh` in `option_motion.py`, and the training characters by `--target_characters`.

Ablation settings:

```bash
# network structure
python train.py --network_type no_cross_attn --proj_name ablation_no_cross
python train.py --network_type mlp           --proj_name ablation_mlp

# input composition
python train.py --use_anchor_input True  --use_relative_pos_input False --proj_name anchor_only
python train.py --use_anchor_input False --use_relative_pos_input True  --proj_name relpos_only
```

## Checkpoint download

```bash
pip install gdown
bash prepare/download_checkpoints.sh
```

Downloads the checkpoint from Google Drive to `saved/best.pt`.

## Testing

```bash
python test.py --checkpoint saved/best.pt                                                   # downloaded checkpoint
```

- Checkpoint: pass the file path to `--checkpoint`. Without it, `saved/{test_proj}/spatio_temp_net_{test_epoch}.pt` is read. The checkpoint must be read with the same network options as in training.

- Playlist: `example_bvh` in `option_motion.py`. The uncommented motions are played in order, and the next motion starts when one ends.
- Test characters: set by `test_type` and `test_char` in `set_test_args` of `test.py`.

| `test_type` | `test_char` | Target characters |
|---|---|---|
| `SMPLx` | `normal` / `small` / `fat` | SMPLx / SMPLx scaled by 0.7 / SMPLx_fat |
| `Mixamo` | `normal` / `small` / `fat` | Leonard with Ybot / Amy / Ortiz |

The retargeted result is drawn at the origin and the input motion at x = -4. With Mixamo characters, the first frame takes about a minute because the textures are loaded.

### Viewer controls

| Key | Action |
|---|---|
| `←` / `→` | previous / next motion |
| `Space` | play / pause |
| `0`–`9` | jump to 0–90% of the motion |
| `[` / `]` | ±1 frame while paused |
| `↓` / `↑` | ±10 frames while paused |
| `Q` `W` `E` `R` | toggle the input 0 / input 1 / result 0 / result 1 character |
| `M` / `X` | toggle the mesh / skeleton |
| `G` / `A` / `T` | toggle the floor / axes / frame number |
| `L` | change the light direction |
| `V` | switch between perspective / orthographic projection |
| `F1` / `F2` | filled / wireframe |
| `F5` | screenshot (`capture/{date}/images/`) |
| `F6` | start / stop recording (`capture/{date}/videos/`) |
| `Alt` + left drag / middle drag | rotate / move the camera |
| scroll / `Alt` + scroll | zoom / dolly |
| `Esc` | quit |

### Post-processing

```bash
python test.py --checkpoint saved/best.pt --test_epoch 8000 --resolve_ground_pene True
```

`--resolve_ground_pene True` edits the network output so that the feet do not penetrate the ground (default False). It lifts the legs in frames where the heel or toe joints are lower than the threshold heights, and applies to both the viewer and `--save True`.

| Option | Default | Meaning |
|---|---|---|
| `--heel_pene_ths` | 0.06 | minimum height of the heel joints (m) |
| `--toe_pene_ths` | 0.02 | minimum height of the toe joints (m) |

### Saving results

```bash
python test.py --checkpoint saved/best.pt --save True
```

`--save True` saves every motion in `example_bvh` to `saved_result/best/{motion}/net_{name}_s{0,1}.npz` without a window. The folder is named after the checkpoint file. Each file contains `root_p` (T, 3) and `local_R` (T, J, 3, 3).

## Mesh preprocessing notes

Making a low-resolution mesh in Blender:

1. Select the mesh in Object mode.
2. Apply Edge → Un-Subdivide in Edit mode.
3. Go back to Object mode and apply the same to the remaining meshes.
