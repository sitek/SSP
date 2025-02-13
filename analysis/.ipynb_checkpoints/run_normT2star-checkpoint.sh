#!/bin/bash
#SBATCH --mem=64G
#SBATCH --time=1:00:00

sub=$1

python normT2star.py --sub $sub
