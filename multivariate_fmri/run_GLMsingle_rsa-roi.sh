#!/bin/bash

#SBATCH --time=1:00:00
#SBATCH --mem=16G

# Usage: sbatch run_GLMsingle_rsa-roi.sh <sub_label> <noise_level>
# e.g.:  sbatch run_GLMsingle_rsa-roi.sh SSP009 Q
#
# noise_level restricts the RSA to trials at that noise level only (syllable/speaker/acoustic
# models; the SNR model is automatically skipped as degenerate -- GLMsingle_rsa-group.ipynb's
# build_categorical_model_rdms -- since every trial then shares the same noise level). Acoustic
# models are only meaningful for noise_level=Q (the only level speech metrics were measured for).
# Valid values: Q, 8, 0, n2, n6.

bidsroot=/bgfs/bchandrasekaran/krs228/data/SSP/data_bids/

python GLMsingle_rsa-roi.py --sub=$1 \
                        --noise_level=$2 \
                        --method=crossnobis \
                        --bidsroot=$bidsroot \
                        "${@:3}"
