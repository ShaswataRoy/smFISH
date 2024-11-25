#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Create the Mask for a Specific Experiment

Author: Shaswata Roy
Email: roy134@purdue.edu
Date: 2024/03/28

This code converts all the _seg.npy files (used for cellpose annotation) in a specific experiment folder into a single tiff file.
The code presumes that the path of the experiment is given as:
    `root_folder/section_folder/experiment_folder`
where section_folder is either Cytoplasm or Nucleus and experiment_folder refers to the experiment instance.

For example the the directory structure is `./Cytoplasm/221218-Hela-IFNG-16h-2_1` we will run the following code:
    `python create_mask.py --root ./ --section Cytoplasm --exp 221218-Hela-IFNG-16h-2_1 `
The mask will be saved in the root directory as `masks/221218-Hela-IFNG-16h-2_1_Cytoplasm_mask.tif`
"""

from glob import glob
import numpy as np
from tqdm import tqdm
import tifffile as tiff
import cv2
import argparse
import os
from natsort import natsorted

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-r","--root", help="root directory",type=str, required =True)
    parser.add_argument("-e","--exp", help="exp name to save the mask",type=str, required =True)
    parser.add_argument("-s","--section", help="cell section chosen (Cytoplasm or Nucleus)",
                        type=str, required =True)
    parser.add_argument("-i","--image_size", help="Image Size (default 1844)", default=1844, type=int)
    args = parser.parse_args()

    if not os.path.exists(args.root):
        print("Root directory not found.")
        exit()
    
    root_folder = args.root

    try:
        section_folder = os.path.join(args.root,args.section)
    except:
        print("Cell Section not found. Choose either Cytoplasm or Nucleus.")
        exit()

    try:
        image_folder = os.path.join(section_folder,args.exp)
    except:
        print("Chosen experiment not found.")
        exit()
    seg_files = natsorted(glob(image_folder+'/*.npy'))
    n_files = len(seg_files)
    mask = np.zeros((n_files,args.image_size,args.image_size)).astype(np.uint8)
    for i in tqdm(range(n_files)):
        mask[i] = np.load(seg_files[i],allow_pickle=True).item()['masks'].astype(np.uint8)
        mask[i] = cv2.normalize(mask[i], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    mask_folder = os.path.join(root_folder,'masks')
    if not os.path.isdir(mask_folder):
        os.mkdir(mask_folder)
    tiff.imwrite(mask_folder+"/"+args.exp+"_"+args.section+"_mask.tif",mask)