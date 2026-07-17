#!/bin/bash

#SBATCH --time=1:00:00

# Runs the default (all-conditions, crossnobis) RSA for every subject. For a noise-level-
# restricted batch (e.g. just 'Q'), duplicate this loop and append --noise_level=Q to the
# sbatch call below.
for subpath in /bgfs/bchandrasekaran/krs228/data/SSP/data_bids/sub*/; do
  subid=$(basename $subpath)
  echo $subid
  sub_label=${subid#"sub-"}
  echo $sub_label
  sbatch run_GLMsingle_rsa-roi.sh $sub_label
done
