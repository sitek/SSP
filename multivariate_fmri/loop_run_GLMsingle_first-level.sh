#!/bin/bash

#SBATCH --time=1-00:00:00

for subpath in /bgfs/bchandrasekaran/krs228/data/SSP/data_bids/sub*/; do
  subid=$(basename $subpath)
  echo $subid
  sub_label=${subid#"sub-"}
  echo $sub_label
  sbatch run_GLMsingle_first-level.sh $sub_label
done
