#!/bin/bash

#SBATCH --time=1:00:00
#SBATCH --mem=16G

bidsroot=/bgfs/bchandrasekaran/krs228/data/SSP/data_bids/

python GLMsingle_split-betas.py --sub=$1 \
                        --task=badaga \
                        --space=MNI152NLin2009cAsym \
                        --bidsroot=$bidsroot \
                        --fmriprep_dir=$bidsroot/derivatives/fmriprep-23.2.1/
