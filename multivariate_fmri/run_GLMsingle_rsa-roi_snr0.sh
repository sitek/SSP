#!/bin/bash

#SBATCH --time=1:00:00
#SBATCH --mem=16G

bidsroot=/bgfs/bchandrasekaran/krs228/data/SSP/data_bids/

# Restricted to the noise_level=='0' condition -- syllable/speaker models only, since the SNR
# model is automatically skipped as degenerate (GLMsingle_rsa-group.ipynb's
# build_categorical_model_rdms) when every trial shares the same noise level.
python GLMsingle_rsa-roi.py --sub=$1 \
                        --method=crossnobis \
                        --noise_level=0 \
                        --bidsroot=$bidsroot
