#!/bin/bash

#SBATCH --time=1:00:00
#SBATCH --mem=16G

bidsroot=/bgfs/bchandrasekaran/krs228/data/SSP/data_bids/

# Default: crossnobis RDM across all 80 conditions. For a noise-level-restricted run (e.g. just
# 'Q'), pass --noise_level=Q as an extra positional arg when invoking this script manually, e.g.:
#   sbatch run_GLMsingle_rsa-roi.sh SSP009 --noise_level=Q
python GLMsingle_rsa-roi.py --sub=$1 \
                        --noise_level=Q \
                        --method=crossnobis \
                        --bidsroot=$bidsroot \
                        "${@:2}"
