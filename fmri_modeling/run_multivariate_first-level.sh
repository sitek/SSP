#!/bin/bash

#SBATCH --time=2-00:00:00

bidsroot=/bgfs/bchandrasekaran/krs228/data/SSP/data_bids/

python multivariate_first-level.py --sub=$1 \
                        --task=badaga \
                        --space=MNI152NLin2009cAsym \
                        --analysis_window=session \
                        --fwhm=3 \
                        --event_type=stimulus \
                        --model_type=LSA \
                        --t_acq=2 --t_r=2 \
                        --bidsroot=$bidsroot \
                        --fmriprep_dir=$bidsroot/derivatives/fmriprep-23.2.1/
