#!/bin/bash

#SBATCH --time=2:00:00

for subpath in /ix1/bchandrasekaran/krs228/data/SSP/data_bids/sub*/; do
  subid=$(basename $subpath)
  echo $subid
  sub_label=${subid#"sub-"}
  echo $sub_label
  sbatch run_univariate_first-level.sh $sub_label
done
