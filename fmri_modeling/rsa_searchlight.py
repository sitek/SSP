#!/usr/bin/env python
# coding: utf-8

# Based on the rsatoolbox tutorial: https://rsatoolbox.readthedocs.io/en/stable/demo_searchlight.html
import os
import argparse
import sys
import itertools

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nibabel as nib
import seaborn as sns

from nilearn import plotting
from nilearn.image import new_img_like

from rsatoolbox.inference import eval_fixed
from rsatoolbox.model import ModelFixed, Model
from rsatoolbox.rdm import RDMs

from rsatoolbox.util.searchlight import get_volume_searchlight, get_searchlight_RDMs, evaluate_models_searchlight
from glob import glob

parser = argparse.ArgumentParser(
                description='Create subject-specific searchlight RSA',
                epilog=('Example: python rsa_searchlight.py --sub=FLT02 '
                        ' --space=MNI152NLin2009cAsym '
                        ' --analysis_window=run '
                        ' --fwhm=1.5 --searchrad=3'
                        ' --contrast=sound '
                        ' --mask_dir=/PATH/TO/MASK/DIR/ '                        
                        ' --bidsroot=/PATH/TO/BIDS/DIR/ ' 
                        ' --fmriprep_dir=/PATH/TO/FMRIPREP/DIR/'))

parser.add_argument("--sub", help="participant id", 
                    type=str)
parser.add_argument("--space", help="space label", 
                    type=str)
parser.add_argument("--analysis_window", 
                    help="analysis window (options: session, run}", 
                    type=str)
parser.add_argument("--fwhm", help="spatial smoothing full-width half-max", 
                    type=str)
parser.add_argument("--searchrad", help="radius of searchlight (in voxels)", 
                    type=int)
parser.add_argument("--contrast", help="contrast to analyze (options: sound, resp)", 
                    type=str)
parser.add_argument("--mask_dir", 
                    help="directory containing subdirectories with masks for each subject", 
                    type=str)
parser.add_argument("--bidsroot", 
                    help="top-level directory of the BIDS dataset", 
                    type=str)
parser.add_argument("--fmriprep_dir", 
                    help="directory of the fMRIprep preprocessed dataset", 
                    type=str)
args = parser.parse_args()

if len(sys.argv) < 2:
    parser.print_help()
    print(' ')
    sys.exit(1)
    
sub_id          = args.sub
space_label     = args.space
analysis_window = args.analysis_window
fwhm           = args.fwhm
searchrad      = args.searchrad
contrast_label = args.contrast
mask_dir     = args.mask_dir
bidsroot     = args.bidsroot
fmriprep_dir = args.fmriprep_dir

# other directory definitions
deriv_dir = os.path.join(bidsroot, 'derivatives')

print('participant ID: ', sub_id, 
      '\nfirst-level FWHM: ', fwhm, 
      '\ndesired searchlight radius (in voxels): ', searchrad)

''' Helper functions '''
def upper_tri(RDM):
    """upper_tri returns the upper triangular index of an RDM

    Args:
        RDM 2Darray: squareform RDM

    Returns:
        1D array: upper triangular vector of the RDM
    """
    # returns the upper triangle
    m = RDM.shape[0]
    r, c = np.triu_indices(m, 1)
    return RDM[r, c]

def get_searchlight_rdm(mask_data, image_paths, centers, neighbors):
    # loop over all images
    x, y, z = mask_data.shape
    data = np.zeros((len(image_paths), x, y, z))
    for ix, im in enumerate(image_paths):
        #print(im)
        data[ix] = nib.load(im).get_fdata()

    # only one pattern per image
    image_value = np.arange(len(image_paths))

    # reshape data so we have n_observastions x n_voxels
    data_2d = data.reshape([data.shape[0], -1])
    data_2d = np.nan_to_num(data_2d)

    # Get RDMs – takes approx. 5 min
    # per https://github.com/rsagroup/rsatoolbox/issues/248#issuecomment-1437358066: 
    # only works with method='euclidean' if mask includes some 0s
    print('getting searchlight RDMs')
    SL_RDM = get_searchlight_RDMs(data_2d, centers, neighbors, 
                                  image_value, method='euclidean')
    
    return SL_RDM, data

def create_RDM_img(test_model, SL_RDM, data, mask_img):
    # takes a couple minutes to start running – don't give up too early!
    # in total, takes about 15 minutes to run with 2 cores
    print('Comparing searchlight RDMs')
    eval_results = evaluate_models_searchlight(SL_RDM, 
                                               test_model, 
                                               eval_fixed, 
                                               method='spearman', 
                                               n_jobs=-1)

    # get the evaulation score for each voxel
    # We only have one model, but evaluations returns a list. 
    # By using float we just grab the value within that list
    eval_score = [float(e.evaluations) for e in eval_results]

    # Create an 3D array, with the size of mask, and
    x, y, z = data.shape[1:]
    RDM_brain = np.zeros([x*y*z])
    RDM_brain[list(SL_RDM.rdm_descriptors['voxel_index'])] = eval_score
    RDM_brain = RDM_brain.reshape([x, y, z])

    plot_img = new_img_like(mask_img, RDM_brain)
    
    return plot_img

''' Make models '''

# Define categorical variables
snrs = ["Q", "8", "0", "N2", "N6"]  # 5 SNRs
syllables = ["ba", "da", "ga", "ma"]  # 4 Syllables
talkers = ["F1", "F2", "M1", "M2"]  # 4 Talkers

# Generate all possible stimuli (5 SNRs × 4 Syllables × 4 Talkers = 80 stimuli)
stimuli = list(itertools.product(snrs, syllables, talkers))  # 80 stimuli

# Create the pattern_descriptors dictionary
pattern_descriptors = {
    idx: {"SNR": stim[0], "Syllable": stim[1], "Talker": stim[2]} 
    for idx, stim in enumerate(stimuli)
}

# Initialize RDMs (96x96 matrices)
rdm_snr = np.zeros((80, 80), dtype=int)
rdm_syllable = np.zeros((80, 80), dtype=int)
rdm_talker = np.zeros((80, 80), dtype=int)

# Fill RDMs
for i in range(80):
    for j in range(80):
        # SNR RDM: 1 if different SNRs, 0 if same
        rdm_snr[i, j] = 1 if stimuli[i][0] != stimuli[j][0] else 0

        # Syllable RDM: 1 if different syllables, 0 if same
        rdm_syllable[i, j] = 1 if stimuli[i][1] != stimuli[j][1] else 0

        # Talker RDM: 1 if different talkers, 0 if same
        rdm_talker[i, j] = 1 if stimuli[i][2] != stimuli[j][2] else 0

# Print matrix dimensions to confirm correctness
print("Syllable RDM shape:", rdm_syllable.shape)
print("SNR RDM shape:", rdm_snr.shape)
print("Talker RDM shape:", rdm_talker.shape)

# convert to models
rdms_array = np.array([rdm_snr, rdm_syllable, rdm_talker])

model_rdms = RDMs(rdms_array,
                  rdm_descriptors={'categorical_model':['snr', 'syllable', 'talker'],},
                  #pattern_descriptors=pattern_descriptors,
                  dissimilarity_measure='Euclidean'
                 )

snr_rdms = model_rdms.subset('categorical_model','snr')
syllable_rdms = model_rdms.subset('categorical_model','syllable')
syllable_rdms = model_rdms.subset('categorical_model','talker')

# #### Convert from RDM to Model
model_snr      = ModelFixed( 'snr RDM', model_rdms.subset('categorical_model', 
                                                       'snr'))
model_syllable = ModelFixed( 'syllable RDM', model_rdms.subset('categorical_model', 
                                                           'syllable'))
model_talker   = ModelFixed( 'syllable RDM', model_rdms.subset('categorical_model', 
                                                           'talker'))
all_models = [model_snr, model_syllable, model_talker]

''' Get searchlight and RDMs '''
mask_fpath = os.path.join(mask_dir, 
                          'sub-{}'.format(sub_id),
                          'space-{}'.format(space_label), 
                          'masks-gm',
                          'sub-{}_space-{}_mask-gm.nii.gz'.format(sub_id, space_label))

mask_img = nib.load(mask_fpath)
mask_data = mask_img.get_fdata()
x, y, z = mask_data.shape

# takes about 10 minutes with 2,540,000 voxels; 
# grey matter-masked (449,000), about 3 min
print('getting searchlight voxels')
centers, neighbors = get_volume_searchlight(mask_data, 
                                            radius=searchrad, 
                                            threshold=0.5)


if analysis_window == 'session':
    model_desc = 'run-all'
    # set this path to wherever you saved the folder containing the img-files
    data_folder = os.path.join(model_dir, 
                               'sub-{}_space-{}'.format(sub_id, space_label),
                               model_desc)

    print(data_folder)
    image_paths = sorted(glob(f'{data_folder}/*contrast-{contrast_label}*map-tstat.nii.gz'))
    assert len(image_paths)

    SL_RDM, data = get_searchlight_rdm(mask_data, image_paths, centers, neighbors)

    # define output path
    out_dir = os.path.join(model_dir, 
                           'sub-{}_space-{}'.format(sub_id, space_label),
                           'rsa-searchlight_fwhm-{}_searchvox-{}_{}'.format(fwhm, searchrad, model_desc))
    if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    # ## Compare RDMs
    for mi, test_model in enumerate(all_models): # cat_models

        plot_img = create_RDM_img(test_model, SL_RDM, data, mask_img)

        model_id = test_model.name.split(' ')[0]

        # #### Save correlation image
        sub_outname = (f'sub-{sub_id}_fwhm-{fwhm}_'+
                       f'searchvox-{searchrad}_rsa-searchlight_'+
                       f'contrast-{contrast_label}_model-{model_id}.nii.gz')
        out_fpath = os.path.join(out_dir, sub_outname)
        nib.save(plot_img, out_fpath)
        print('saved image to ', out_fpath)


elif analysis_window == 'run':
    model_desc = 'run-specific_eventtype-stimulus'
    
    # set this path to wherever you saved the folder containing the img-files
    sub_model_folder = os.path.join(deriv_dir, 'nilearn', model_desc, f'sub-{sub_id}')
    print('sub_model_folder:', sub_model_folder)
    
    print('img directory:', sub_model_folder)
    
    run_design_fpaths = glob(sub_model_folder+'/*run*design.tsv')
    print('run design files:', run_design_fpaths)
    
    n_runs = len(run_design_fpaths)
    print('number of runs:', n_runs)
    
    for rx in range(n_runs):
        run_label = rx + 1
        print('creating searchlight RDMs for run', run_label)
        
        # get image files per stimulus
        #image_paths = sorted(glob(f'{data_folder}/*contrast-{contrast_label}*map-tstat.nii.gz'))
        # order the same as the RDMs
        image_paths = []
        for talk in talkers:
            for syll in syllables:
                for snr in snrs:
                    stim_img_fpath = glob(f'{sub_model_folder}/*run-{run_label}_contrast-{syll}{talk}{snr}*stat-t_statmap.nii.gz')[0]
                    #print('stim_img_fpath:', stim_img_fpath)
                    image_paths.append(stim_img_fpath)
        
        assert len(image_paths)
        #print('image files:', image_paths)

        SL_RDM, data = get_searchlight_rdm(mask_data, image_paths, centers, neighbors)

        # define output path
        out_dir = os.path.join(deriv_dir, 'rsa-searchlight_run-specific', 
                               f'sub-{sub_id}',
                               f'run-{run_label}', )
        if not os.path.exists(out_dir):
                os.makedirs(out_dir)

        # ## Compare RDMs
        for mi, test_model in enumerate(all_models): # cat_models
            plot_img = create_RDM_img(test_model, SL_RDM, data, mask_img)

            model_id = test_model.name.split(' ')[0]

            # #### Save correlation image
            sub_outname = (f'sub-{sub_id}_{run_label}_fwhm-{fwhm}_'+
                           f'searchvox-{searchrad}_rsa-searchlight_'+
                           f'contrast-{contrast_label}_model-{model_id}.nii.gz')
            out_fpath = os.path.join(out_dir, sub_outname)
            nib.save(plot_img, out_fpath)
            print('saved image to ', out_fpath)