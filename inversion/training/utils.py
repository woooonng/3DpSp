import numpy as np
import torch
import os
import random
import shutil
import json
from PIL import Image
from matplotlib import pyplot as plt

from training.camera_utils import FOV_to_intrinsics, LookAtPoseSampler


def tensor2im(var):
    var = var.detach().cpu().transpose(0, 2).transpose(0, 1).numpy()
    var = (var + 1) / 2
    var[var < 0] = 0
    var[var > 255] = 1
    var = var * 255
    return Image.fromarray(var.astype('uint8'))


def vis_faces(log_hooks):
    display_count = len(log_hooks)
    fig = plt.figure(figsize=(8, 4 * display_count))
    gs = fig.add_gridspec(display_count, 2)
    for i in range(display_count):
        hooks_dict = log_hooks[i]
        fig.add_subplot(gs[i, 0])
        plt.imshow(hooks_dict['input_face'], cmap="gray")
        plt.title('Input')
        fig.add_subplot(gs[i, 1])
        plt.imshow(hooks_dict['y_hat'])
        plt.title('Output')
    plt.tight_layout()
    return fig


def synthesis_avg_image(device, network, latent_avg):
    intrinsic = FOV_to_intrinsics(18.837, device=device).reshape(-1, 9)
    cam_pivot = torch.tensor(network.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
    cam_radius = network.rendering_kwargs.get('avg_camera_radius', 2.7)
    cam2world_pose = LookAtPoseSampler.sample(np.pi / 2, np.pi / 2, cam_pivot, radius=cam_radius,
                                              device=device).reshape(-1, 16)
    camera_param = torch.cat((cam2world_pose, intrinsic), dim=1)
    img_mean = network.synthesis(latent_avg, camera_param)['image'][0]
    img_mean = torch.nn.functional.interpolate(img_mean.unsqueeze(0), (256, 256), mode='bilinear', align_corners=False)
    return img_mean.squeeze(0)


def aggregate_loss_dict(agg_loss_dict):
    mean_vals = {}
    for output in agg_loss_dict:
        for key in output:
            mean_vals[key] = mean_vals.setdefault(key, []) + [output[key]]
    for key in mean_vals:
        if len(mean_vals[key]) > 0:
            mean_vals[key] = sum(mean_vals[key]) / len(mean_vals[key])
        else:
            print('{} has no value'.format(key))
            mean_vals[key] = 0
    return mean_vals

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU 

    # CUDA randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    np.random.seed(seed)
    random.seed(seed)

def train_val_split(preprocessed_path, train_path, val_path, ratio):
    os.makedirs(os.path.join(train_path, "conf_map"), exist_ok=True)
    os.makedirs(os.path.join(val_path, "conf_map"), exist_ok=True)

    conf_map_path = os.path.join(preprocessed_path, "conf_map")

    all_img_files = sorted([f for f in os.listdir(preprocessed_path) if f.endswith(".png")])
    base_img_files = [f for f in all_img_files if "_mirror" not in f]

    val_samples = int(len(base_img_files) * ratio)

    val_img_files = random.sample(base_img_files, val_samples)
    train_img_files = list(set(base_img_files) - set(val_img_files))

    for val_file in val_img_files:
        # move original images and mirrored images to validation directory
        mirror_val_file = val_file.replace(".png", "_mirror.png")
        shutil.move(os.path.join(preprocessed_path, val_file), os.path.join(val_path, val_file))
        shutil.move(os.path.join(preprocessed_path, mirror_val_file), os.path.join(val_path, mirror_val_file))
        
         # move confidence map files corresponding to the validation images to validation directory 
        conf_val_file = val_file.replace(".png", ".npy")
        mirror_conf_val_file = conf_val_file.replace(".npy", "_mirror.npy")
        shutil.move(os.path.join(conf_map_path, conf_val_file), os.path.join(val_path, "conf_map", conf_val_file))
        shutil.move(os.path.join(conf_map_path, mirror_conf_val_file), os.path.join(val_path, "conf_map", mirror_conf_val_file))



    for train_file in train_img_files:
        # move original images and mirrored images to train directory
        mirror_train_file = train_file.replace(".png", "_mirror.png")
        shutil.move(os.path.join(preprocessed_path, train_file), os.path.join(train_path, train_file))
        shutil.move(os.path.join(preprocessed_path, mirror_train_file), os.path.join(train_path, mirror_train_file))

        # move confidence map files corresponding to the train images to train directory 
        conf_train_file = train_file.replace(".png", ".npy")
        mirror_conf_train_file = conf_train_file.replace(".npy", "_mirror.npy")
        shutil.move(os.path.join(conf_map_path, conf_train_file), os.path.join(train_path, "conf_map", conf_train_file))
        shutil.move(os.path.join(conf_map_path, mirror_conf_train_file), os.path.join(train_path, "conf_map", mirror_conf_train_file))


    # move json files
    json_path = os.path.join(preprocessed_path, "dataset.json")
    with open (json_path, 'r') as f:
        camera_params = json.load(f)["labels"]

    val_cam_params = [item for item in camera_params if item[0] in val_img_files or item[0].replace("_mirror", "") in val_img_files]
    val_cam_params = {"labels": val_cam_params}
    val_cam_path = os.path.join(val_path, "dataset.json")
    if val_cam_params["labels"]:
        with open(val_cam_path, 'w') as f:
            json.dump(val_cam_params, f)

    train_cam_params = [item for item in camera_params if item[0] in train_img_files or item[0].replace("_mirror", "") in train_img_files]
    train_cam_params = {"labels": train_cam_params}
    train_cam_path = os.path.join(train_path, "dataset.json")
    if train_cam_params["labels"]:
        with open(train_cam_path, 'w') as f:
            json.dump(train_cam_params, f)

    if os.path.exists(conf_map_path):
        os.rmdir(conf_map_path)
    print("All processed data have been divided into the train and the validation set")
        



