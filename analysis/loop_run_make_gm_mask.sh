#!/bin/bash

for subpath in /bgfs/bchandrasekaran/krs228/data/SSP/data_bids/sub*/; do 
  subid=$(basename $subpath)
  echo $subid
  sub_label=${subid#"sub-"}
  sbatch run_make_gm_mask.sh $sub_label
done
