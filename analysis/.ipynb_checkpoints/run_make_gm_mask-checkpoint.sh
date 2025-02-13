#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH -c 2

for sub in SSP002 SSP005; do
  echo $sub
  python make_gm_mask.py --sub=$sub \
    --space=MNI152NLin2009cAsym \
    --fwhm=0 \
    --bidsroot=/bgfs/bchandrasekaran/krs228/data/SSP/data_bids/ \
    --fmriprep_dir=/bgfs/bchandrasekaran/krs228/data/SSP/data_bids/derivatives/fmriprep-23.2.1/
done

