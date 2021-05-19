from __future__ import print_function, division, absolute_import
import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from munch import munchify

class ToSpaceBGR(object):

    def __init__(self, is_bgr):
        self.is_bgr = is_bgr

    def __call__(self, tensor):
        if self.is_bgr:
            new_tensor = tensor.clone()
            new_tensor[0] = tensor[2]
            new_tensor[2] = tensor[0]
            tensor = new_tensor
        return tensor


class ToRange255(object):

    def __init__(self, is_255):
        self.is_255 = is_255

    def __call__(self, tensor):
        if self.is_255:
            tensor.mul_(255)
        return tensor


class TransformImage(object):
    def __init__(self, opts, scale=1.0, random_crop=False,
                 random_hflip=False, rotate=True, random_resized_crop=True):
        if type(opts) == dict:
            opts = munchify(opts)
        self.input_size = opts.input_size
        self.input_space = opts.input_space
        self.input_range = opts.input_range
        self.mean = opts.mean
        self.std = opts.std
        self.scale = scale
        self.random_crop = random_crop
        self.random_hflip = random_hflip
        self.rotate= rotate
        self.random_resized_crop = random_resized_crop

        tfs = []
        
        if random_resized_crop:
            tfs.append(transforms.RandomResizedCrop(299))

        if random_crop:
          tfs.append(transforms.RandomCrop(299))

        if random_hflip:
          tfs.append(transforms.RandomHorizontalFlip())
        
        if rotate:
          tfs.append(transforms.RandomRotation(8))
        
        tfs.append(transforms.ToTensor())
        tfs.append(ToSpaceBGR(self.input_space=='BGR'))
        tfs.append(ToRange255(max(self.input_range)==255))
        tfs.append(transforms.Normalize(mean=self.mean, std=self.std))
        # tfs.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        self.tf = transforms.Compose(tfs)

    def __call__(self, img):
        tensor = self.tf(img)
        return tensor


class LoadImage(object):

    def __init__(self, space='RGB'):
        self.space = space

    def __call__(self, path_img):
        with open(path_img, 'rb') as f:
            with Image.open(f) as img:
                img = img.convert(self.space)
        return img


class LoadTransformImage(object):

    def __init__(self, model, scale=0.8):
        self.load = LoadImage()
        self.tf = TransformImage(model, scale=scale)

    def __call__(self, path_img):
        img = self.load(path_img)
        tensor = self.tf(img)
        return tensor


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x