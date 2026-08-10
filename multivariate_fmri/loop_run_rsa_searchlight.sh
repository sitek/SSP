#!/bin/bash

firstlevel_dir=/ix1/bchandrasekaran/krs228/data/SSP/data_bids/derivatives/nilearn/run-specific_eventtype-stimulus/

for subpath in $firstlevel_dir/sub*/; do 
  subid=$(basename $subpath)
  echo $subid
  sub_label=${subid#"sub-"}
#for sub_label in SSP008 SSP015 SSP021 SSP056 SSP062 SSP064 SSP069 SSP072 SSP074 SSP077 SSP078 SSP081 SSP083 SSP084 #SSP085 SSP086 SSP087 SSP090 SSP092 SSP094; do
  echo $sub_label
  sbatch run_rsa_searchlight.sh $sub_label
done
