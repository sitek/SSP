#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH -c 2

# atlas options: carpet_dseg, subcort_aud, tian_S2, tian_S3, carpet_motor

for subpath in /bgfs/bchandrasekaran/krs228/data/SSP/data_bids/sub*/; do 
  subid=$(basename $subpath)
  echo $subid
  sub_label=${subid#"sub-"}
#for sub_label in 'SSP013' 'SSP018' 'SSP032'; do
#  echo $sub_label
  python make_atlas_region_masks.py --sub=$sub_label \
    --space=MNI152NLin2009cAsym \
    --fwhm=0.00 \
    --atlas_label=carpet_dseg \
    --bidsroot=/bgfs/bchandrasekaran/krs228/data/SSP/data_bids/ \
    --fmriprep_dir=/bgfs/bchandrasekaran/krs228/data/SSP/data_bids/derivatives/fmriprep-23.2.1/
done
