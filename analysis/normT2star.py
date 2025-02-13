#!/usr/bin/env python

import os
import argparse, sys
import nibabel as nib
import numpy as np
import pandas as pd

from glob import glob
from nilearn.image import concat_imgs, mean_img, math_img, clean_img, new_img_like
from nilearn import plotting

parser = argparse.ArgumentParser(
                description='Compute normative T2star maps',
                epilog='Example: python normT2star.py --sub SSP002'
        )

parser.add_argument("--sub", help="participant id", type=str)

args = parser.parse_args()

if len(sys.argv) < 1:
    parser.print_help()
    print(' ')
    sys.exit(1)

sub_id = args.sub
task_label = 'alice'

bidsroot = os.path.join('/bgfs/bchandrasekaran/',
                        'krs228/data/',
                        'SSP/data_bids/')
out_dir = os.path.join(bidsroot, 
                       'derivatives', 
                       'nilearn', 
                       'normT2star',
                       f'task-{task_label}')

''' Get functional files and concatenate '''
func_dir = os.path.join(bidsroot, 'derivatives/fmriprep-23.2.1/',
                        f'sub-{sub_id}', 'func')    
func_files = sorted(glob(func_dir+f'/*task-{task_label}*space-MNI152NLin2009cAsym*bold.nii.gz'))
print(func_files)

func_example_fpath = func_files[0]

func_imgs = [nib.load(x) for x in func_files]

''' Get brain masks '''
print('Loading brain masks')
bmask_files = sorted(glob(func_dir+'/*mask.nii.gz'))
bmask_example_fpath = bmask_files[0]

bmask_imgs = [nib.load(x) for x in bmask_files]

''' Get region of interest masks '''
print('Loading region of interest masks')
atlas_label = 'tian-S3'
mask_dir = os.path.join(bidsroot, 
                        'derivatives/nilearn/masks/',
                        f'sub-{sub_id}/space-MNI152NLin2009cAsym/',
                        f'masks-{atlas_label}')

mask_list = sorted(glob(mask_dir + '/*.gz'))

print('Creating volume-normalized images')
def create_norm_img(func_img):
    func_data = func_img.get_fdata()
    volume_means = func_data.mean(axis=2).mean(axis=1).mean(axis=0)
    norm_data = func_data / volume_means

    norm_img = new_img_like(func_img, norm_data, copy_header=True)
    
    return norm_img

norm_imgs = []
for fx, func_img in enumerate(func_imgs):
    norm_img = create_norm_img(func_img)
    norm_imgs.append(norm_img)

print('Concatenating normalized functional images')
norm_concat = concat_imgs(norm_imgs)

print('Compute normative T2star')
mean_concat = mean_img(norm_concat)

out_fpath = os.path.join(out_dir, f'sub-{sub_id}_task-{task_label}_bold_normT2star.nii.gz')
nib.save(mean_concat, out_fpath)

print('Invert to normative R2star')
mean_r2s = math_img('1 / img1',
                    img1 = mean_concat)

out_fpath = os.path.join(out_dir, f'sub-{sub_id}_task-{task_label}_bold_normR2star.nii.gz')
nib.save(mean_r2s, out_fpath)

print('Mask by region of interest')
from nilearn.maskers import NiftiMasker
def mask_fmri(fmri_niimgs, mask_filename, fwhm):
    masker = NiftiMasker(mask_img=mask_filename,
                         smoothing_fwhm=fwhm, standardize=True,
                         memory="nilearn_cache", memory_level=1)
    fmri_masked = masker.fit_transform(fmri_niimgs)
    return fmri_masked, masker

mask_dict = {}
mean_dict = {}
mask_name_list = []
mean_val_list = []
for mx, mask_fpath in enumerate(mask_list):
    mask_name = mask_fpath.split('_mask-')[1].split('.')[0]
    print(mask_name)
    
    # get masked data and masker object
    mri_masked, masker = mask_fmri(mean_concat, mask_fpath, 0)
    
    # turn matrix back into image
    masked_img = masker.inverse_transform(mri_masked)
    
    # add to dict
    mask_dict[mask_name] = masked_img
    
    # replace all ROI values with the mean
    mean_masked = np.full(mri_masked.shape, mri_masked.mean())
    mean_masked_img = masker.inverse_transform(mean_masked)
    
    mean_dict[mask_name] = mean_masked_img
    
    #single value for saving to tsv file
    mask_name_list.append(mask_name)
    mean_val_list.append(mri_masked.mean())

# save single values per ROI to csv
mean_val_dict = pd.DataFrame({'region': mask_name_list, 
                              'normT2star': mean_val_list})

mean_val_dict.to_csv(os.path.join(out_dir, 
                                  f'{sub_id}_mean_normT2star.csv'),
                     index=False)
    
roi_stat_data_list = []
for rx, mask_name in enumerate(mean_dict):
    roi_stat_data = mean_dict[mask_name].get_fdata()
    roi_stat_data_list.append(roi_stat_data)

# add maps together and make nifti img
roi_stat_map_data = sum(roi_stat_data_list)
roi_stat_map_img = new_img_like(mean_masked_img, roi_stat_map_data)

out_fpath = os.path.join(out_dir, 
                         f'sub-{sub_id}_task-{task_label}_atlas-{atlas_label}_bold_normT2star.nii.gz')
nib.save(roi_stat_map_img, out_fpath)
