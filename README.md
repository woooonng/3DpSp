# 3DpSp: 3D-Aware pSp-based EG3D Encoder
<img width="927" alt="background" src="https://github.com/user-attachments/assets/bb289ead-e4c2-4b21-b598-fa4f78c8eb7f" />

**3DpSp** inverts images into the latent space of EG3D network which is a 3D-aware GAN network. Based on pixel2style2pixel (pSp), an encoder-based inversion of StyleGAN, we modified it for EG3D. Specifically, We used camera parameters for the 3D priors in the map2style network of the pSp encoder and applied ReStyle's iterative refinement methods to achieve high-quality results. Inspired by TriPlaneNet, we also included mirrored images and confidence maps to enhance the learning 3D priors more robustly.

<div style="display: flex; justify-content: space-between; align-items: center;">
    <img src="https://github.com/user-attachments/assets/bce1a440-59dc-4ae8-9592-b78ca409af6a" width="48%">
    <img src="https://github.com/user-attachments/assets/d8456008-12e5-4dde-badf-4f5a292ba44a" width="48%">
</div>

The above images illustrates the training scheme(left) and our network's architecture(right). Starting with an input image and a synthesized image generated from the mean latent vector in the W+ space of EG3D, the two images are concatenated and passed to the 3D-pSp encoder with the camera parameters. Since EG3D's map2style network utilizes camera parameters to project the inputs into the W+ space, we leverage these parameters in the encoder network to robustly learn 3D prior information. Specifically, the camera parameters are added to the map2style networks, as seen in the right figure. Additionally, we introduce a 1x1 convolutional layer in the map2style network to enhance its expressiveness. Finally, the 3D-pSp encoder computes the residual between the two input images and adds it to the original latent vector to refine the inverted latent vector. Repeating this process iteratively enables our network to achieve finer details compared to the original pSp network.


**Notice: Revised version with some adjustments**
- This repository is the same as [KyungWonCho](https://github.com/KyungWonCho/3DpSp) because we worked together on the term project. This is a revised version with some adjustments. The overall content, including the model architecture and training scheme, remains the same as mentioned above.
- The faceswap with hair transfer was conducted as part of the project; however, this repository only includes the codes for 3DpSp. Nonetheless, metrics related to the faceswap are presented here.

## Requirements
* Operating System: Linux (Tested on Ubuntu 18.04.6 LTS)
* GPU: High-end Nvidia GPU (Tested on a single Nvidia A100 GPU)
  * We also conducted tests using the NVIDIA GeForce GTX 1080 Ti with a batch size of 1.
* Environment:
  *  Dockerfile: nvcr.io/nvidia/pytorch:22.12-py3
 * Dependencies
   * See '''environment.sh''' for checking the dependencies
---

## Getting Started (Docker Environment Setup)
After building a image using Dockerfile in the "docker" directory and make a container, you can set up the environment using ```environment.sh```
```
bash envirionment.sh
```
---
## Pretrained models
The following pretrained models are required:

1. **EG3D Networks**: ```./pretrained_models/ffhqrebalanced512-128.pkl``` <br> Download from: [EG3D on NVIDIA NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/eg3d)
2. **ArcFace Networks**: ```./pretrained_models/model_ir_se50.pth``` <br>
Download from: [InsightFace PyTorch](https://github.com/TreB1eN/InsightFace_Pytorch/blob/master/README.md)

3. **Basel Face Model (BFM09)**: ```./dataset_preprocessing/ffhq/Deep3DFaceRecon_pytorch/BFM/01_MorphableModel.mat```,&nbsp;&nbsp;&nbsp;``` ./dataset_preprocessing/ffhq/Deep3DFaceRecon_pytorch/BFM/Exp_Pca.bin``` <br>
Download from: [Basel Face Model](https://github.com/jadewu/3D-Human-Face-Reconstruction-with-3DMM-face-model-from-RGB-image/tree/main/BFM )
5. **Deep3DFace Pytorch**: ```./dataset_preprocessing/ffhq/Deep3DFaceRecon_pytorch/checkpoints/pretrained/epoch_20.pth```<br>
Download from: [Deep3DFaceRecon PyTorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch)

6. **Unsup3d**: ```./dataset_preprocessing/ffhq/unsup3d/pretrained/pretrained_celeba/checkpoint030.pth```<br>
Download from: [Unsup3d](https://github.com/elliottwu/unsup3d)

7. **3DpSp (Ours)**: ```./pretrained_models/3DpSp.pt```<br>
Download from: [3DpSp Pretrained Model](https://github.com/KyungWonCho/3DpSp?tab=readme-ov-file)
---
## Train
To train the model, download the FFHQ dataset with a resolution of 1024 and save them in your desired directory. For more details, refer to [NVlabs](https://github.com/NVlabs/ffhq-dataset).
### Preprocessing
We need to crop the images, confidence maps, camera parameters and mirrored images.
```
bash preprocessing_train.sh
```

### Training
```
bash train.sh
```
**Explanation of the train options**

Key options can be set in ```inversion/options/train_options.py```:
* Train and validation split
  * --train_val_ratio: ffhq dataset is divided into the train and validation set with a ratio of 0.1

* Loss Weights:
  * --id_lambda, --lpips_lambda, --l2_lambda: weights for ID, LPIPS, and L2 losses.
  * --id_lambda_mirror, --lpips_lambda_mirror, --l2_lambda_mirror: weights for mirrored loss components.
* ReStyle Iterative Refinements
  * --n_iters_per_batch: Number of refinement steps per batch(Default=5)
* Learning Parameters:
  * --batch_size: Batch size. (Default is 4 in A100 and 1 in GTX 1080 Ti)
  * --learning_rate: Learning rate. Default is 0.0001.
* Checkpoints:
  * Use --checkpoint_path to resume training from a checkpoint.
---
## Inference
### Preprocessing
We need to crop the images, confidence maps, and camera parameters. Unlike the training process, mirror images are not required for testing.
```
bash preprocessing_infer.sh
```
### Test (inference)
```
bash inference.sh
```
---
## Acknowledgement
This project was highly inspired by and builds upon the following outstanding open-source projects:
* [EG3D](https://github.com/NVlabs/eg3d)
* [pSp](https://github.com/eladrich/pixel2style2pixel)
* [ReStyle](https://github.com/yuval-alaluf/restyle-encoder)
* [TriPlaneNet](https://github.com/anantarb/triplanenet)


