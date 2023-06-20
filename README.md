# [WACV'23] AT-DDPM: Restoring Faces degraded by Atmospheric Turbulence using Denoising Diffusion Probabilistic Models


[Paper link](https://arxiv.org/pdf/2208.11284.pdf)

Although many long-range imaging systems are designed to support extended vision applications, a natural
obstacle to their operation is degradation due to atmospheric turbulence. Atmospheric turbulence causes significant degradation to image quality by introducing blur
and geometric distortion. In recent years, various deep
learning-based single image atmospheric turbulence mitigation methods, including CNN-based and GAN inversionbased, have been proposed in the literature which attempt
to remove the distortion in the image. However, some of
these methods are difficult to train and often fail to reconstruct facial features and produce unrealistic results especially in the case of high turbulence. Denoising Diffusion Probabilistic Models (DDPMs) have recently gained
some traction because of their stable training process and
their ability to generate high quality images. In this paper,
we propose the first DDPM-based solution for the problem of atmospheric turbulence mitigation. We also propose a fast sampling technique for reducing the inference
times for conditional DDPMs. Extensive experiments are
conducted on synthetic and real-world data to show the
significance of our model. 

## Prerequisites:
1. Create a conda environment and activate using 
```
conda env create -f environment.yml
conda activate AT-diff
```
## Data Preparation
2. Prepare Data in the following format
```
    ├── data 
    |   ├── train # Training  images
    |   └── test  # Testing
    |       ├── at             # turbulence images 
```
## Training and Testing
3 Run following commands to train and test 
```
For training:
export PYTHONPATH=$PYTHONPATH:$(pwd)
CUDA_VISIBLE_DEVICES="0" NCCL_P2P_DISABLE=1  torchrun --nproc_per_node=1 --master_port=4326 scripts/AT_train.py 

For testing:
export PYTHONPATH=$PYTHONPATH:$(pwd)
CUDA_VISIBLE_DEVICES="0" NCCL_P2P_DISABLE=1  torchrun --nproc_per_node=1 --master_port=4326 scripts/AT_test.py --weights /pathtoweights/ --data_dir /pathtodata/
```
## Pretrained models
4 Please download pretrained models from the following link
```
https://www.dropbox.com/sh/1cvq055t7umbmkb/AAA4hR0Ah06SJoj2wvn_tS1wa?dl=0
```
## Citation
5 If you use our work, please use the following citation
```
@inproceedings{nair2023ddpm,
  title={At-ddpm: Restoring faces degraded by atmospheric turbulence using denoising diffusion probabilistic models},
  author={Nair, Nithin Gopalakrishnan and Mei, Kangfu and Patel, Vishal M},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={3434--3443},
  year={2023}
}
```

## Acknowledgements
Thanks to authors of Diffusion Models Beat GANs on Image Synthesis sharing their code. Most of the code is borrowed from the guided diffusion
```
https://github.com/openai/guided-diffusion
```
