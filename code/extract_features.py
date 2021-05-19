import shutil
import subprocess
import glob
from tqdm import tqdm
import numpy as np
import os
import argparse

import torch
from torch import nn
import torch.nn.functional as F
import pretrainedmodels
import feature_util as utils

C, H, W = 3, 299, 299


def extract_feats(params, model, load_image_fn):
    global C, H, W
    model.eval()

    feat_dir = '/content/drive/MyDrive/cs5242_project/extracted_feats/11_try/'
    if not os.path.isdir(feat_dir):
        os.mkdir(feat_dir)
    print("save video feats to %s" % (feat_dir))
    
    train_dir = '/content/drive/MyDrive/cs5242_project/dataset/train/train/'
    video_list = os.listdir(train_dir) 

    for video in tqdm(video_list):
        print(video)
        imgs_path = os.path.join(train_dir,video)
        image_list = sorted(glob.glob(os.path.join(imgs_path, '*.jpg')))
        samples = np.round(np.linspace(
            0, len(image_list) - 1, params['n_frame_steps']))
        image_list = [image_list[int(sample)] for sample in samples]
        images = torch.zeros((len(image_list), C, H, W))
        for iImg in range(len(image_list)):
            img = load_image_fn(image_list[iImg])
            images[iImg] = img
        print('images',images.shape)
        with torch.no_grad():
            fc_feats = model(images.cuda()).squeeze()
        print('feats', fc_feats.shape)
        img_feats = fc_feats.cpu().numpy()
        # Save the features
        outfile = os.path.join(feat_dir, video + '.npy')
        np.save(outfile, img_feats)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu", dest='gpu', type=str, default='0',
                        help='Set CUDA_VISIBLE_DEVICES environment variable, optional')

    parser.add_argument("--n_frame_steps", dest='n_frame_steps', type=int, default=30,
                        help='how many frames to sampler per video')

    parser.add_argument("--model", dest="model", type=str, default='efficientNet',
                        help='the CNN model you want to use to extract_feats')
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    params = vars(args)

    if params['model'] == 'resnet152':
        C, H, W = 3, 224, 224
        model = pretrainedmodels.resnet152(pretrained='imagenet')
        load_image_fn = utils.LoadTransformImage(model)

    elif params['model'] == 'inception_v4':
        C, H, W = 3, 299, 299
        model = pretrainedmodels.inceptionv4(
            num_classes=1000, pretrained='imagenet')
        load_image_fn = utils.LoadTransformImage(model)
    
    
    elif params['model'] == 'efficientNet':
        C, H, W = 3, 600, 600
        model = EfficientNet.from_pretrained('efficientnet-b7')
        load_image_fn = utils.LoadTransformImage(model)

    else:
        print("doesn't support %s" % (params['model']))

    model.last_linear = utils.Identity()
    model = nn.DataParallel(model)
    model = model.cuda()
    extract_feats(params, model, load_image_fn)
