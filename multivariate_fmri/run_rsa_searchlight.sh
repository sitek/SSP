#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH -c 2
#SBATCH --mem-per-cpu=32GB

bidsroot=/ix1/bchandrasekaran/krs228/data/SSP/data_bids/

python rsa_searchlight.py --sub=$1 \
    --space=MNI152NLin2009cAsym \
    --analysis_window=session \
    --fwhm=0.00 \
    --searchrad=5 \
    --contrast=sound \
    --mask_dir=$bidsroot/derivatives/nilearn/masks/ \
    --bidsroot=$bidsroot \
    --fmriprep_dir=$bidsroot/derivatives/denoised_fmriprep-23.2.1/


