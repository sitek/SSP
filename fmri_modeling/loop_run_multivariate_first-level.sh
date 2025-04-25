#!/bin/bash

#SBATCH --time=23:00:00

for subpath in /bgfs/bchandrasekaran/krs228/data/SSP/data_bids/sub*/; do 
  subid=$(basename $subpath)
  echo $subid
  sub_label=${subid#"sub-"}
  sbatch run_multivariate_first-level.sh $sub_label
done
