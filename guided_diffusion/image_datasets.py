import math
import random
import torch as th
from PIL import Image
import blobfile as bf
# from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import cv2
import torch.distributed as dist
import os
import imgaug.augmenters as iaa
from basicsr.data import degradations as degradations

def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    deterministic=False,
    random_flip=True,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    img_files =os.listdir(data_dir)

    dataset = ImageDataset(
        image_size,
        img_files,
        data_dir,
        shard=dist.get_rank(),
        num_shards=dist.get_world_size(),
        random_flip=random_flip,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class RandomCrop(object):

    def __init__(self, crop_size=[256,256]):
        """Set the height and weight before and after cropping"""
        self.crop_size_h  = crop_size[0]
        self.crop_size_w  = crop_size[1]

    def __call__(self, inputs, target):
        input_size_h, input_size_w, _ = inputs.shape
        try:
            x_start = random.randint(0, input_size_w - self.crop_size_w)
            y_start = random.randint(0, input_size_h - self.crop_size_h)
            inputs = inputs[y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w] 
            target = target[y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w] 
        except:
            inputs=cv2.resize(inputs,(256,256))
            target=cv2.resize(target,(256,256))

        return inputs,target

class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        data_dir,
        shard=0,
        num_shards=1,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.random_flip = random_flip
        self.data_dir=data_dir

        self.deformation = iaa.ElasticTransformation(alpha=100, sigma=[10., 20.])

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir,self.local_images[idx])
        gt_path = os.path.join(self.data_dir,self.local_images[idx])

        with bf.BlobFile(path, "rb") as f:
            img_lq = self.process_and_load_images(path)
        with bf.BlobFile(path, "rb") as f:
            img_hlq = self.process_and_load_images(path)
        with bf.BlobFile(gt_path, "rb") as f1:
            img_hq = self.process_and_load_images(gt_path)

        if(np.random.uniform()<0.9):
            img_lq = self.get_degraded_image(img_lq)

        img_lq = np.clip(img_lq,0,1.0)*2-1.0
        img_hq = np.clip(img_hq,0,1.0)*2-1.0
        img_hlq =cv2.resize(img_hlq,(64,64),interpolation=cv2.INTER_LINEAR)
        img_hlq =cv2.resize(img_hlq,(256,256),interpolation=cv2.INTER_LINEAR)
        img_hlq = np.clip(img_hlq,0,1.0)*2-1.0
        img_hlq=  np.transpose(img_hlq, [2, 0, 1])
        img_lq=  np.transpose(img_lq, [2, 0, 1])
        img_hq= np.transpose(img_hq, [2, 0, 1])

        out_dict = {}
        out_dict["high_turb"]=img_lq
        out_dict["low_turb"]=img_hlq
        out_dict["clean"]=img_hq

        return img_hq, out_dict
        
    def process_and_load_images(self,path):

        pil_image = Image.open(path)
        pil_image.load()
        pil_image=pil_image.resize((self.resolution,self.resolution))
        arr=np.array(pil_image).astype(np.float32)
        arr=arr/255.0

        return arr

    def get_degraded_image(self,lq):

            img_lq = self.deformation(image=lq)
            kernel = degradations.random_mixed_kernels(
                            ['iso', 'aniso'],
                            [0.5,0.5],
                            int(np.random.uniform(25, 45)) * 2 + 1,
                            [5, 25],
                            [5, 25], [-math.pi, math.pi],
                            noise_range=None)
            img_lq = cv2.filter2D(img_lq, -1, kernel)
            # downsample
            scale = np.random.uniform(1, 4)
            img_lq = cv2.resize(img_lq, (int(256 // scale), int(256 // scale)), interpolation=cv2.INTER_LINEAR)
            # noise
            img_lq = degradations.random_add_gaussian_noise(img_lq, [0,20])
            # resize to original size
            img_lq = cv2.resize(img_lq, (256,256), interpolation=cv2.INTER_LINEAR)

            return img_lq

