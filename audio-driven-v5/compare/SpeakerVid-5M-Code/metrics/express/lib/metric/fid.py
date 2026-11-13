"""Calculates the Frechet Inception Distance (FID) to evalulate GANs
The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.
When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).
The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.
See --help to see further details.
Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow
Copyright 2018 Institute of Bioinformatics, JKU Linz
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
#################################################
# Copyright (c) 2021-present, xiaobing.ai, Inc. #
# All rights reserved.                          #
#################################################
# CV Research, DEV(USA) xiaobing.               #
# written by wangduomin@xiaobing.ai             #
#################################################

import os
import pathlib
# from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import time
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

from lib.metric.inception import InceptionV3

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}

# class ImagePathDataset(torch.utils.data.Dataset):
#     def __init__(self, files, transforms=None):
#         self.files = files
#         self.transforms = transforms

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, i):
#         path = self.files[i]
#         img = Image.open(path).convert('RGB')
#         if self.transforms is not None:
#             img = self.transforms(img)
#         return img

class FidCompute(object):
    def __init__(self, dims=2048):
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.inception_model = InceptionV3([block_idx])
        self.inception_model.cuda()
        self.inception_model.eval()

    def get_activations(self, images, batch_size=50, dims=2048, num_workers=1):
        """Calculates the activations of the pool_3 layer for all images.
        Params:
        -- files       : List of image files paths
        -- model       : Instance of inception model
        -- batch_size  : Batch size of images for the model to process at once.
                        Make sure that the number of samples is a multiple of
                        the batch size, otherwise some samples are ignored. This
                        behavior is retained to match the original FID score
                        implementation.
        -- dims        : Dimensionality of features returned by Inception
        -- device      : Device to run calculations
        -- num_workers : Number of parallel dataloader workers
        Returns:
        -- A numpy array of dimension (num images, dims) that contains the
        activations of the given tensor when feeding inception with the
        query tensor.
        """
        # model.eval()
        images_len = images.shape[0]
        if batch_size > images_len:
            print(('Warning: batch size is bigger than the data size. '
                'Setting batch size to data size'))
            batch_size = images_len

        pred_arr = np.empty((images_len, dims))

        start_idx = 0

        for idx in range(int(round(images_len / batch_size + 0.4999))):
            batch = images[idx*batch_size:(idx+1)*batch_size]
            batch = batch.cuda()

            with torch.no_grad():
                pred = self.inception_model(batch)[0]

            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred = pred.squeeze(3).squeeze(2).cpu().numpy()

            pred_arr[start_idx:start_idx + pred.shape[0]] = pred

            start_idx = start_idx + pred.shape[0]

        # print(pred_arr.shape)

        return pred_arr
        
    def compute_fid(self, mu1, sigma1, mu2, sigma2):
        # mu1 = np.atleast_1d(mu1)
        # mu2 = np.atleast_1d(mu2)
        # sigma1 = np.atleast_2d(sigma1)
        # sigma2 = np.atleast_2d(sigma2)
        m = np.square(mu1 - mu2).sum()
        s, _ = linalg.sqrtm(np.dot(sigma1, sigma2), disp=False) # pylint: disable=no-member
        fid = np.real(m + np.trace(sigma1 + sigma2 - s * 2))
        return float(fid)
    
    def calculate_statistics(self, acts):
        mu = np.mean(acts, axis=0)
        sigma = np.cov(acts, rowvar=False)
        return mu, sigma



    
    