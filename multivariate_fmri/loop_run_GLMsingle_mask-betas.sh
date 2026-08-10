#!/bin/bash

#SBATCH --time=1:00:00

for subpath in /ix1/bchandrasekaran/krs228/data/SSP/data_bids/sub*/; do
  subid=$(basename $subpath)
  echo $subid
  sub_label=${subid#"sub-"}
  echo $sub_label
  sbatch run_GLMsingle_mask-betas.sh $sub_label
done
