#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH -c 2

# atlas options: carpet_dseg, subcort_aud, tian_S3, carpet_motor
# SSP002 SSP005 SSP011 SSP012 SSP013 SSP014 SSP018 SSP020 SSP032 SSP033 SSP034 SSP039 SSP041 SSP045 SSP051
for sub in SSP011 SSP039; do
  python make_atlas_region_masks.py --sub=$sub \
    --space=MNI152NLin2009cAsym \
    --fwhm=0.00 \
    --atlas_label=tian_S3 \
    --bidsroot=/bgfs/bchandrasekaran/krs228/data/SSP/data_bids/ \
    --fmriprep_dir=/bgfs/bchandrasekaran/krs228/data/SSP/data_bids/derivatives/denoised_fmriprep-23.1.1/
done
