#!/bin/bash

#SBATCH --time=2-00:00:00

for subpath in /ix1/bchandrasekaran/krs228/data/SSP/data_bids/sub*/; do 
  subid=$(basename $subpath)
  echo $subid
  sub_label=${subid#"sub-"}
#for sub_label in SSP008 SSP015 SSP021 SSP054 SSP056 SSP060 SSP062 SSP064 SSP069 SSP072 SSP074 SSP077 SSP078 SSP081 #SSP083 SSP084 SSP085 SSP086 SSP087 SSP090 SSP092 SSP094; do
  echo $sub_label
  sbatch run_multivariate_first-level.sh $sub_label
done
