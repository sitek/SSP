#!/bin/bash

#SBATCH --time=4:00:00
#SBATCH --mem=16G

bidsroot=/bgfs/bchandrasekaran/krs228/data/SSP/data_bids/

python GLMsingle_mask-betas.py --sub=$1 \
                        --space=MNI152NLin2009cAsym \
                        --fwhm=0 \
                        --bidsroot=$bidsroot \
                        --mask_dir=$bidsroot/derivatives/nilearn/masks/
