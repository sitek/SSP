import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import nibabel as nib

from nilearn.glm.first_level import first_level_from_bids


''' Set up and interpret command line arguments '''
parser = argparse.ArgumentParser(
                description=('Split a subject\'s GLMsingle TYPED_FITHRF_GLMDENOISE_RR output '
                            'into one labeled NIfTI per trial, using the trial manifest saved '
                            'by GLMsingle_first-level.py. Does not re-run GLMsingle -- reloads '
                            'the already-saved .npy output, so it is cheap to re-run.'),
                epilog=('Example: python GLMsingle_split-betas.py --sub=SSP009 '
                        '--task=badaga --space=MNI152NLin2009cAsym '
                        '--bidsroot=/PATH/TO/BIDS/DIR/ '
                        '--fmriprep_dir=/PATH/TO/FMRIPREP/DIR/')
                )

parser.add_argument("--sub",
                    help="participant id", type=str)
parser.add_argument("--task",
                    help="task id", type=str)
parser.add_argument("--space",
                    help="space label", type=str)
parser.add_argument("--bidsroot",
                    help="top-level directory of the BIDS dataset", type=str)
parser.add_argument("--fmriprep_dir",
                    help="directory of the fMRIprep preprocessed dataset", type=str)

args = parser.parse_args()

if len(sys.argv) < 2:
    parser.print_help()
    print(' ')
    sys.exit(1)

subject_id = args.sub
task_label = args.task
space_label = args.space
bidsroot = args.bidsroot
fmriprep_dir = args.fmriprep_dir

print('participant ID:', subject_id)

bidsderiv_dir = os.path.join(bidsroot, 'derivatives', 'glmsingle')
subject_out_dir = os.path.join(bidsderiv_dir, f'sub-{subject_id}')

info_fpath = os.path.join(subject_out_dir, f'sub-{subject_id}_glmsingle_info.json')
manifest_fpath = os.path.join(subject_out_dir,
                              f'sub-{subject_id}_task-{task_label}_desc-trialbetas_manifest.csv')

if not os.path.exists(info_fpath):
    raise FileNotFoundError(
        f'No GLMsingle info file found for sub-{subject_id} at {info_fpath} -- '
        'run GLMsingle_first-level.py for this subject first.'
    )
if not os.path.exists(manifest_fpath):
    raise FileNotFoundError(
        f'No trial manifest found for sub-{subject_id} at {manifest_fpath} -- '
        'run GLMsingle_first-level.py for this subject first.'
    )

with open(info_fpath) as f:
    info = json.load(f)
# GLMsingle_first-level.py records which output type is this subject's final single-trial
# estimate: TYPED_FITHRF_GLMDENOISE_RR (cross-validated denoising + ridge) when the subject has
# at least one condition repeating across runs, or TYPEB_FITHRF (no cross-validated denoising/
# ridge -- GLMsingle can't do that with only within-run repeats) otherwise. Load whichever this
# subject actually has, instead of hardcoding TYPED.
final_type = info['final_type']

final_type_fpath = os.path.join(subject_out_dir, f'{final_type}.npy')
if not os.path.exists(final_type_fpath):
    raise FileNotFoundError(
        f'{info_fpath} says sub-{subject_id}\'s final type is {final_type}, but '
        f'{final_type_fpath} does not exist -- re-run GLMsingle_first-level.py for this subject.'
    )

manifest = pd.read_csv(manifest_fpath)

print(f'Loading GLMsingle {final_type} output from {final_type_fpath}')
results = np.load(final_type_fpath, allow_pickle=True).item()
# 'betasmd' is documented as a common field across GLMsingle's output types (TYPEA/B/C/D all
# expose beta estimates under this key) -- not independently confirmed for TYPEB specifically
# against an actual installed GLMsingle, since none is available locally. If this KeyErrors on
# a real TYPEB-only subject, check `results.keys()` and update the key name here.
try:
    betasmd = results['betasmd']
except KeyError:
    raise KeyError(
        f"'{final_type}' output for sub-{subject_id} has no 'betasmd' key (has: "
        f"{list(results.keys())}) -- GLMsingle's non-TYPED output types may use a different key "
        "name for beta estimates than assumed here; update this script once confirmed."
    )

# Validate the trial count matches the manifest before trusting any labeling -- this is a
# direct, data-driven check (rather than assuming a fixed run/condition/repeat structure, as
# the FLT2 reference this pipeline is adapted from does) since real repeats-per-condition-per-run
# in this dataset look irregular rather than fixed.
n_trials_betas = betasmd.shape[-1]
n_trials_manifest = len(manifest)
if n_trials_betas != n_trials_manifest:
    raise ValueError(
        f'betasmd has {n_trials_betas} trial volumes but the manifest has {n_trials_manifest} '
        'rows -- these must match before trial labels can be trusted. This likely means the '
        'manifest is stale relative to the GLMsingle run that produced this .npy (e.g. '
        'GLMsingle_first-level.py was re-run with different motion-based run-dropping since the '
        'manifest was last written) -- re-run GLMsingle_first-level.py to regenerate both '
        'together, don\'t just re-run this script.'
    )

# IMPORTANT: this labels betasmd's trials using the manifest's row order, which was built in
# chronological (run, then onset-within-run) order when the design matrix was constructed in
# GLMsingle_first-level.py. This assumes GLMsingle preserves that same chronological trial order
# in its output. This has NOT been independently confirmed against GLMsingle's internals (the
# FLT2 reference this pipeline is adapted from instead assumes a fixed run/condition/repeat
# nested-loop order, which doesn't fit this dataset's irregular repeat counts) -- spot-check a
# handful of labeled trials' onset times against the raw events.tsv the first time this runs on
# real data, before trusting the full output.
print(f'Labeling {n_trials_betas} trials from manifest (chronological order assumption -- see '
     'comment above; spot-check before trusting on first real run).')

# global (not per-run) running repeat count per condition, so e.g. "rep-02" of a condition means
# its second presentation anywhere in the session, not its second presentation within one run
manifest = manifest.sort_values('trial_index').reset_index(drop=True)
manifest['rep'] = manifest.groupby('condition').cumcount() + 1

# re-derive the affine from one of this subject's functional images (GLMsingle's betasmd array
# has no header/affine of its own -- it's a bare numpy array)
_models, models_run_imgs, _raw_models_events, _models_confounds = first_level_from_bids(
    bidsroot,
    task_label,
    space_label=space_label,
    sub_labels=[subject_id],
    derivatives_folder=fmriprep_dir,
    slice_time_ref=None,
    minimize_memory=True,
)
imgs = models_run_imgs[0]
if not isinstance(imgs, list):
    imgs = [imgs]
affine = nib.load(imgs[0]).affine

# Keep TYPEB (degraded, no cross-run denoising/ridge) and TYPED (cross-validated) output
# physically separate, not just tagged in metadata -- the user explicitly does not want RSA
# work to mix TypeB subjects in with TypeD ones, and a JSON marker alone is too easy for a
# downstream script's glob to miss. beta_images/ is the default, RSA-ready location and only
# ever contains TYPED output; a naive `glob('sub-*/beta_images/*.nii.gz')` (e.g. exactly what
# FLT2/rsa_roi.py already does) structurally cannot pick up a TYPEB subject by accident --
# including one requires deliberately pointing at beta_images_degraded-typeb/ instead.
if final_type == 'TYPED_FITHRF_GLMDENOISE_RR':
    out_dir = os.path.join(subject_out_dir, 'beta_images')
    desc_tag = 'typed'
else:
    out_dir = os.path.join(subject_out_dir, 'beta_images_degraded-typeb')
    desc_tag = 'typeb'
os.makedirs(out_dir, exist_ok=True)

for _, row in manifest.iterrows():
    vol = betasmd[..., int(row['trial_index'])]
    fname = (f"sub-{subject_id}_run-{int(row['run']):02d}_stim-{row['condition']}"
            f"_rep-{int(row['rep']):02d}_desc-{desc_tag}.nii.gz")
    img = nib.Nifti1Image(vol.astype(np.float32), affine)
    nib.save(img, os.path.join(out_dir, fname))

print(f'Saved {len(manifest)} per-trial beta images ({final_type}) to {out_dir}')
