#!/bin/bash

#SBATCH --time=1:00:00

# Runs the crossnobis RSA for every subject, at every noise level (Q, 8, 0, n2, n6) -- one sbatch
# submission per (subject, noise_level) pair.
for subpath in /bgfs/bchandrasekaran/krs228/data/SSP/data_bids/sub*/; do
  subid=$(basename $subpath)
  echo $subid
  sub_label=${subid#"sub-"}
  echo $sub_label
  for noise_level in Q 8 0 n2 n6; do
    sbatch run_GLMsingle_rsa-roi.sh $sub_label $noise_level
  done
done
