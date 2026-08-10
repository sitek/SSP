#!/bin/bash

#SBATCH --time=2:00:00
#SBATCH --mem=32G

bidsroot=/ix1/bchandrasekaran/krs228/data/SSP/data_bids/

python univariate_first-level.py --sub=$1 \
                        --task=badaga \
                        --space=MNI152NLin2009cAsym \
                        --fwhm=6 \
                        --event_type=snr \
                        --t_acq=2 --t_r=2 \
                        --bidsroot=$bidsroot \
                        --fmriprep_dir=$bidsroot/derivatives/fmriprep-23.2.1/
