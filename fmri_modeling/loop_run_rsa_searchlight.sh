#!/bin/bash

#SBATCH --time=23:00:00

firstlevel_dir=/bgfs/bchandrasekaran/krs228/data/SSP/data_bids/derivatives/nilearn/run-specific_eventtype-stimulus/

for subpath in $firstlevel_dir/sub*/; do 
  subid=$(basename $subpath)
  echo $subid
  sub_label=${subid#"sub-"}
  sbatch run_rsa_searchlight.sh $sub_label
done
