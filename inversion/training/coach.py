import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import warnings
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from lpips import LPIPS
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from inversion.criteria.id_loss import IDLoss
from inversion.datasets.dataset import ImageFolderDataset
from inversion.models.psp import pSp
from inversion.training.ranger import Ranger
from inversion.training.utils import tensor2im, vis_faces, synthesis_avg_image, aggregate_loss_dict, train_val_split
from training.camera_utils import LookAtPoseSampler, FOV_to_intrinsics
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

matplotlib.use('Agg')

'''
This code is adapted from ReStyle (https://github.com/yuval-alaluf/restyle-encoder) and TriPlaneNet (https://github.com/anantarb/triplanenet)
'''

class Coach:
    def __init__(self, opts):
        self.opts = opts
        self.device = torch.device('cuda:1')
        self.opts.device = self.device
        self.global_step = 1
        self.epoch = 1

        # Initialize network
        self.net = pSp(opts).to(self.device)

        # loss function
        self.mse_loss = nn.MSELoss().to(self.device).eval()
        self.lpips_loss = LPIPS(net='alex').to(self.device).eval()
        for param in self.lpips_loss.parameters():
            param.requires_grad = False
        self.id_loss = IDLoss().to(self.device).eval()
        for param in self.id_loss.parameters():
            param.requires_grad = False
            
        # Initialize optimizer
        self.optimizer = Ranger(self.net.encoder.parameters(), lr=self.opts.learning_rate)
        
        # train, validation split
        self.opts.train_dataset_path = os.path.join(self.opts.preprocessed_dataset_path, "train")
        self.opts.val_dataset_path = os.path.join(self.opts.preprocessed_dataset_path, "val")
        os.makedirs(self.opts.train_dataset_path, exist_ok=True)
        os.makedirs(self.opts.val_dataset_path, exist_ok=True)
        train_val_split(self.opts.preprocessed_dataset_path, self.opts.train_dataset_path, self.opts.val_dataset_path, opts.train_val_ratio)

        # Initialize datasets
        self.train_dataset, self.val_dataset = self.configure_dataset()
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.opts.batch_size,
                                           num_workers=self.opts.num_workers, shuffle=True, drop_last=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.opts.batch_size,
                                          num_workers=self.opts.num_workers, shuffle=False, drop_last=True)

        log_dir = os.path.join(opts.exp_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.logger = SummaryWriter(log_dir=log_dir)

        # Initialize checkpoint dir
        self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_loss = None

        if self.opts.save_interval is None:
            self.opts.save_interval = self.opts.max_steps

        if self.opts.checkpoint_path is not None:
            ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')

            if "state_dict" in ckpt:
                self.net.encoder.load_state_dict(ckpt["state_dict"])
                print("Load pSp encoder from checkpoint")

            if "latent_avg" in ckpt:
                self.net.latent_avg = ckpt["latent_avg"]
                print("Load latent average from checkpoint")

        if self.net.latent_avg is None:
            self.net.latent_avg = self.mean_latent(100000)
        self.avg_image = synthesis_avg_image(self.device, self.net.decoder, self.net.latent_avg).float().detach()

    # y_0 = avg_image, (x_resized, y_0, cameras) : y_1 -> (x, y_1, cameras) : y_2 -> ... -> (x, y_4, cameras) -> y_5 (final)
    def perform_train_iteration_on_batch(self, x_resized, x, x_mirror, camera_params, camera_params_mirror,
                                         conf_map_mirror):
        loss_dict, latent, y_hat = None, None, None
        for iter in range(self.opts.n_iters_per_batch):
            if iter == 0:
                avg_image_for_batch = self.avg_image.unsqueeze(0).repeat(x_resized.shape[0], 1, 1, 1)   # [B x 3 x 256 x 256]
                x_input = torch.cat([x_resized, avg_image_for_batch], dim=1)    # [B x 6 x 256 x 256]
                # latent: [B x 14 x 512], y_hat: [B x 3 x 512 x 512], y_hat_mirror: [B x 3 x 512 x 512]
                latent, y_hat, y_hat_mirror = self.net.forward(x_input, camera_params.clone().detach(),
                                                               camera_params_mirror.clone().detach(), latent=None)
            else:
                # y_hat_clone: [B x 3 x 256 x 256]
                y_hat_clone = F.adaptive_avg_pool2d(y_hat, (256, 256)).detach().requires_grad_(True)
                latent_clone = latent.clone().detach().requires_grad_(True)
                # x_input: [B x 6 x 256 x 256]
                x_input = torch.cat([x_resized, y_hat_clone], dim=1)
                # latent: [B x 14 x 512], y_hat: [B x 3 x 512 x 512], y_hat_mirror: [B x 3 x 512 x 512]
                latent, y_hat, y_hat_mirror = self.net.forward(x_input, camera_params.clone().detach(),
                                                               camera_params_mirror.clone().detach(), latent_clone)

            loss, loss_dict = self.calc_loss(x, x_mirror, y_hat, y_hat_mirror, conf_map_mirror)

            loss.backward()

        return loss_dict, y_hat

    def train(self):
        self.net.encoder.train()
        torch.cuda.empty_cache()
        train_flag = True
        while train_flag:
            for batch_idx, batch in enumerate(tqdm(self.train_dataloader, desc=f"Training epoch: {self.epoch}")):
                self.optimizer.zero_grad(set_to_none=True)
                
                x_resized, x, camera_params, x_mirror, camera_params_mirror, conf_maps_mirror, _ = batch
                x_resized, x, camera_params, x_mirror, camera_params_mirror, conf_maps_mirror = x_resized.to(
                    self.device).float(), x.to(self.device).float(), camera_params.to(self.device).float(), x_mirror.to(
                    self.device).float(), camera_params_mirror.to(self.device).float(), conf_maps_mirror.to(
                    self.device).float()

                loss_dict, y_hat = self.perform_train_iteration_on_batch(x_resized, x, x_mirror, camera_params,
                                                                         camera_params_mirror, conf_maps_mirror)

                self.optimizer.step()

                # log predicted images (batch_idx_steps.jpg)
                if self.global_step % self.opts.image_interval == 0:
                    self.parse_and_log_images(x, y_hat, title='images/train', subscript=f"{batch_idx:04d}", val_flag=False)

                # log metrics
                if self.global_step % self.opts.board_interval == 0:
                    self.print_metrics(loss_dict, prefix='train')
                    self.log_metrics(loss_dict, prefix='train')

                # validation
                val_loss_dict = None
                if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
                    val_loss_dict = self.validate()
                    if self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss:
                        self.best_val_loss = val_loss_dict['loss']
                        self.checkpoint_me(val_loss_dict, is_best=True)

                # save the latest model
                if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
                    if val_loss_dict is not None:
                        self.checkpoint_me(val_loss_dict, is_best=False)
                    else:
                        self.checkpoint_me(loss_dict, is_best=False)
                
                if self.global_step > self.opts.max_steps:
                    train_flag = False
                    break
                self.global_step += 1 
            self.epoch += 1
        print("[END] Training has been done")

    def perform_val_iteration_on_batch(self, x_resized, x, x_mirror, camera_params, camera_params_mirror,
                                       conf_map_mirror):
        loss_dict, latent, y_hat = None, None, None
        for iter in range(self.opts.n_iters_per_batch):
            if iter == 0:
                avg_image_for_batch = self.avg_image.unsqueeze(0).repeat(x_resized.shape[0], 1, 1, 1)
                x_input = torch.cat([x_resized, avg_image_for_batch], dim=1)
            else:
                x_input = torch.cat([x_resized, y_hat], dim=1)

            latent, y_hat, y_hat_mirror = self.net.forward(x_input, camera_params, camera_params_mirror, latent=latent)

            _, loss_dict = self.calc_loss(x, x_mirror, y_hat, y_hat_mirror, conf_map_mirror)

            y_hat = F.adaptive_avg_pool2d(y_hat, (256, 256))

        return loss_dict, y_hat

    def validate(self):
        self.net.encoder.eval()

        agg_loss_dict = []
        for batch_idx, batch in enumerate(tqdm(self.val_dataloader)):
            x_resized, x, camera_params, x_mirror, camera_params_mirror, conf_maps_mirror, _ = batch
            x_resized, x, camera_params, x_mirror, camera_params_mirror, conf_maps_mirror = x_resized.to(
                self.device).float(), x.to(self.device).float(), camera_params.to(self.device).float(), x_mirror.to(
                self.device).float(), camera_params_mirror.to(self.device).float(), conf_maps_mirror.to(
                self.device).float()
            with torch.no_grad():
                cur_loss_dict, y_hat = self.perform_val_iteration_on_batch(x_resized, x, x_mirror, camera_params,
                                                                           camera_params_mirror, conf_maps_mirror)
            agg_loss_dict.append(cur_loss_dict)

            self.parse_and_log_images(x, y_hat, title='images/val', subscript=f"{batch_idx:04d}", val_flag=True)

        loss_dict = aggregate_loss_dict(agg_loss_dict)
        self.log_metrics(loss_dict, prefix='test')
        self.print_metrics(loss_dict, prefix='test')

        self.net.encoder.train()
        return loss_dict

    def calc_loss(self, x, x_mirror, y_hat, y_hat_mirror, conf_map_mirror):
        loss_dict = {}

        loss_l2 = self.mse_loss(y_hat, x)
        loss_dict['loss_l2'] = float(loss_l2)
        loss = loss_l2 * self.opts.l2_lambda

        loss_id = self.id_loss(y_hat, x)
        loss_dict['loss_id'] = float(loss_id)
        loss += loss_id * self.opts.id_lambda

        loss_lpips = self.lpips_loss(y_hat, x).mean()
        loss_dict['loss_lpips'] = float(loss_lpips)
        loss += loss_lpips * self.opts.lpips_lambda

        loss_l2_mirror = torch.square(x_mirror - y_hat_mirror)
        loss_l2_mirror = loss_l2_mirror.mean(dim=1)
        loss_l2_mirror = loss_l2_mirror / (conf_map_mirror + 1)
        loss_l2_mirror = loss_l2_mirror.mean()
        loss_dict['loss_l2_mirror'] = float(loss_l2_mirror)
        loss += loss_l2_mirror * self.opts.l2_lambda_mirror

        loss_id_mirror = self.id_loss(y_hat_mirror, x_mirror)
        loss_dict['loss_id_mirror'] = float(loss_id_mirror)
        loss += loss_id_mirror * self.opts.id_lambda_mirror

        loss_lpips_mirror = self.lpips_loss(y_hat, x_mirror).mean()
        loss_dict['loss_lpips_mirror'] = float(loss_lpips_mirror)
        loss += loss_lpips_mirror * self.opts.lpips_lambda_mirror

        loss_dict['loss'] = float(loss)
        return loss, loss_dict

    def log_metrics(self, metrics_dict, prefix):
        for key, value in metrics_dict.items():
            self.logger.add_scalar(f'{prefix}/{key}', value, self.global_step)

    def print_metrics(self, metrics_dict, prefix):
        print(f'Metrics for {prefix}, step {self.global_step}')
        for key, value in metrics_dict.items():
            print(f'\t{key} = ', value)

    def parse_and_log_images(self, x, y_hat, title, subscript=None, display_count=4, val_flag=False):
        display_count = min(display_count, x.size(0))   # for batch=1
        im_data = []
        for i in range(display_count):
            cur_im_data = {
                'input_face': tensor2im(x[i]),
                'y_hat': tensor2im(y_hat[i])
            }
            im_data.append(cur_im_data)
        self.log_images(title, im_data=im_data, subscript=subscript, val_flag=val_flag)

    def log_images(self, name, im_data, subscript=None, val_flag=False):
        fig = vis_faces(im_data)
        step = self.global_step
        log_dir_path = os.path.join(self.logger.log_dir, name)
        os.makedirs(log_dir_path, exist_ok=True)

        val_dir_path_per_interval = os.path.join(self.logger.log_dir, name, str(step))
        if val_flag:
            os.makedirs(val_dir_path_per_interval , exist_ok=True)

        if subscript:
            if val_flag:
                path = os.path.join(val_dir_path_per_interval, f'{subscript}_{step:04d}.jpg')
            else:
                path = os.path.join(log_dir_path, f'{subscript}_{step:04d}.jpg')

        fig.savefig(path)
        plt.close(fig)

    def checkpoint_me(self, loss_dict, is_best):
        save_name = 'best_model.pt' if is_best else f'iteration_{self.global_step}.pt'
        save_dict = self.__get_save_dict()
        checkpoint_path = os.path.join(self.checkpoint_dir, save_name)

        if not is_best:
            old_file = glob.glob(os.path.join(self.checkpoint_dir, 'iteration_*.pt'))
            if old_file:
                os.remove(old_file[0])

        torch.save(save_dict, checkpoint_path)
        with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
            if is_best:
                f.write(f'**Best**: Step - {self.global_step}, Loss - {self.best_val_loss} \n{loss_dict}\n')
            else:
                f.write(f'Step - {self.global_step}, \n{loss_dict}\n')

    def __get_save_dict(self):
        save_dict = {
            'state_dict': self.net.encoder.state_dict(),
            'opts': vars(self.opts),
            'best_val_loss': self.best_val_loss,
            'step': self.global_step,
            'optimizer': self.optimizer.state_dict(),
            'latent_avg': self.net.latent_avg
        }
        return save_dict

    def configure_dataset(self):
        train_dataset = ImageFolderDataset(path=self.opts.train_dataset_path, resolution=None, load_conf_map=True,
                                           use_labels=True)
        val_dataset = ImageFolderDataset(path=self.opts.val_dataset_path, resolution=None, load_conf_map=True,
                                          use_labels=True)
        print(f'Number of training samples: {len(train_dataset)}')
        print(f'Number of validation samples: {len(val_dataset)}')
        return train_dataset, val_dataset

    def mean_latent(self, n_latent):
        z_in = torch.from_numpy(np.random.randn(n_latent, self.net.decoder.z_dim)).float().to(self.device)
        cam_pivot = torch.tensor(self.net.decoder.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]),
                                 device=self.device)        # rotation axis of the camera, [0.0, 0.0, 0.2]
        cam_radius = self.net.decoder.rendering_kwargs.get('avg_camera_radius', 2.7)        # the position of the camera from the world origin
        intrinsic = FOV_to_intrinsics(fov_degrees=18.837, device=self.device).reshape(-1, 9).repeat(n_latent, 1)
        cam2world_pose = LookAtPoseSampler.sample(
            np.pi / 2, np.pi / 2, cam_pivot, radius=cam_radius, batch_size=n_latent, device=self.device)
        cam2world_pose = cam2world_pose.reshape(-1, 16)
        camera_param = torch.cat((cam2world_pose, intrinsic), dim=1)
        w_mean = self.net.decoder.mapping(z_in, camera_param).mean(0, keepdim=True)
        return w_mean
