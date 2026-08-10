#!/bin/bash

#SBATCH --time=1-00:00:00
#SBATCH --mem=64G

bidsroot=/ix1/bchandrasekaran/krs228/data/SSP/data_bids/

python GLMsingle_first-level.py --sub=$1 \
                        --task=badaga \
                        --space=MNI152NLin2009cAsym \
                        --fwhm=0 \
                        --stimdur=0.3 \
                        --t_acq=2 --t_r=2 \
                        --bidsroot=$bidsroot \
                        --fmriprep_dir=$bidsroot/derivatives/fmriprep-23.2.1/
