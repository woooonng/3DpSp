# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import os
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# tf.get_logger().setLevel('ERROR') MTCNN 때문에 log 숨겨야하나 고민

parser = argparse.ArgumentParser()
parser.add_argument('--indir', type=str, required=True)
parser.add_argument('--mirror', action='store_true')
parser.add_argument('--conf_map', action='store_true')
parser.add_argument('--train_flag', action='store_true')

args = parser.parse_args()
args.indir = os.path.join(args.indir, "")[:-1]
args.indir = os.path.abspath(args.indir)

flag = 'train' if args.train_flag else "inference"

# notice
print("="*20 + f" [Start] Preprocessing the {flag} data starts ! " + "="*20)

# run mtcnn for Deep3DFaceRecon
command = "python batch_mtcnn.py --in_root " + args.indir
print("[COMMAND] Keypoints detection on the faces" )
os.system(command)
out_folder = args.indir.split("/")[-2] if args.indir.endswith("/") else args.indir.split("/")[-1]

# run Deep3DFaceRecon
os.chdir('Deep3DFaceRecon_pytorch')
command = "python test.py --img_folder=" + args.indir + " --gpu_ids=0 --name=pretrained --epoch=20 --use_opengl False"
print("[COMMAND] The data is are being reconstructed in 3D through Deep3DFaceRecon")
os.system(command)
os.chdir('..')

# crop out the input image
command = "python crop_images_in_the_wild.py --indir=" + args.indir
print("[COMMAND] The data are being cropped")
os.system(command)

# convert the pose to our format
command = f"python 3dface2idr_mat.py --in_root Deep3DFaceRecon_pytorch/checkpoints/pretrained/results/{out_folder}/epoch_20_000000 --out_path {os.path.join(args.indir, 'crop', 'cameras.json')}"
print("[COMMAND] Camera parameters are being modified to our format")
os.system(command)

# additional correction to match the submission version
out_folder = os.path.join(args.indir, "preprocessed")
if args.mirror:
    command = f"python preprocess_face_cameras.py --source {os.path.join(args.indir, 'crop')} --dest {out_folder} --mode orig --mirror"
else:
    command = f"python preprocess_face_cameras.py --source {os.path.join(args.indir, 'crop')} --dest {out_folder} --mode orig"
print("[COMMAND] Camera parameters are being preprocessed(normalized)")
os.system(command)
print("="*20 + f" [END] Preprocessed images with camera labels have been saved to {out_folder} " + "="*20)

if args.conf_map:
    out_folder = os.path.join(args.indir, "preprocessed", "conf_map")
    command = f"python conf_map.py --source {os.path.join(args.indir, 'preprocessed')} --dest {out_folder}"
    print("[COMMAND] Confidence map of the data are being calculated")
    os.system(command)
    print(f"[Confidence maps have been saved to {out_folder}")

