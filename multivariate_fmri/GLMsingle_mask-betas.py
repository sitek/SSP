import os
import sys
import argparse

import numpy as np

from glob import glob
from nilearn.maskers import NiftiMasker


''' Set up and interpret command line arguments '''
parser = argparse.ArgumentParser(
                description=('Mask a subject\'s per-trial GLMsingle beta images into per-ROI '
                            'CSV vectors, for the cortical ROIs used in RSA. Only ever masks '
                            'the TYPED (cross-validated) beta_images/ output, never the '
                            'degraded TYPEB fallback -- see GLMsingle_first-level.py.'),
                epilog=('Example: python GLMsingle_mask-betas.py --sub=SSP009 '
                        '--space=MNI152NLin2009cAsym --fwhm=0 '
                        '--bidsroot=/PATH/TO/BIDS/DIR/ '
                        '--mask_dir=/PATH/TO/BIDS/DIR/derivatives/nilearn/masks/')
                )

parser.add_argument("--sub",
                    help="participant id", type=str)
parser.add_argument("--space",
                    help="space label", type=str)
parser.add_argument("--fwhm",
                    help="spatial smoothing full-width half-max applied during masking",
                    type=float)
parser.add_argument("--bidsroot",
                    help="top-level directory of the BIDS dataset", type=str)
parser.add_argument("--mask_dir",
                    help="directory containing subdirectories with masks for each subject",
                    type=str)

args = parser.parse_args()

if len(sys.argv) < 2:
    parser.print_help()
    print(' ')
    sys.exit(1)

subject_id = args.sub
space_label = args.space
fwhm = args.fwhm
bidsroot = args.bidsroot
mask_dir = args.mask_dir

print('participant ID:', subject_id)

# Cortical ROIs for RSA: the 14-ROI "auditory" network already used in
# univariate_fmri/group_level_all_ROI.ipynb, the 6-ROI "extended language network"
# (supramarginal gyrus anterior/posterior + angular gyrus, bilaterally), and 6 sensorimotor ROIs
# (precentral/postcentral/SMA, bilaterally). The first two sets come from
# masking/make_atlas_region_masks.py's roi_dict_MNI_dseg (--atlas_label=carpet_dseg) and already
# have masks generated under masks-dseg/ from the earlier univariate ROI analysis -- no new mask
# generation needed for those. The sensorimotor set comes from that same script's
# roi_dict_MNI_motor (--atlas_label=carpet_motor, same underlying carpet_dseg atlas file, just a
# different label subset) -- this DOES need a fresh mask-generation run
# (masking/run_make_atlas_region_masks.sh with --atlas_label=carpet_motor) before these ROIs have
# any masks to reuse; both atlas_label runs now write into the same masks-dseg/ directory.
# Ordered all-L-then-all-R (not per-network L/R blocks) so the two halves mirror each other
# region-for-region -- makes hemisphere comparisons in downstream plots easier to read.
CORTICAL_ROI_LIST = [
    'L-HG', 'L-PT', 'L-PP', 'L-STGp', 'L-STGa', 'L-ParsOp', 'L-ParsTri', 'L-SMGa', 'L-SMGp', 'L-Ang',
    'L-Precentral', 'L-Postcentral', 'L-SMA',
    'R-HG', 'R-PT', 'R-PP', 'R-STGp', 'R-STGa', 'R-ParsOp', 'R-ParsTri', 'R-SMGa', 'R-SMGp', 'R-Ang',
    'R-Precentral', 'R-Postcentral', 'R-SMA',
]

glmsingle_dir = os.path.join(bidsroot, 'derivatives', 'glmsingle')
subject_glmsingle_dir = os.path.join(glmsingle_dir, f'sub-{subject_id}')
beta_images_dir = os.path.join(subject_glmsingle_dir, 'beta_images')

# Only ever mask TYPED (cross-validated) betas, never the degraded TYPEB fallback -- RSA should
# only use subjects with multiple runs / cross-run condition repeats. If a subject only has the
# degraded directory, that's not an error in this script -- it's expected for some subjects
# (see GLMsingle_first-level.py) -- but they don't qualify for RSA, so fail loudly and clearly
# rather than silently masking lower-quality data.
if not os.path.isdir(beta_images_dir):
    degraded_dir = os.path.join(subject_glmsingle_dir, 'beta_images_degraded-typeb')
    if os.path.isdir(degraded_dir):
        raise RuntimeError(
            f'sub-{subject_id} only has degraded (TYPEB) GLMsingle output at {degraded_dir}, '
            'not the cross-validated TYPED estimate -- this subject does not qualify for RSA '
            '(which should only use subjects with cross-run condition repeats). Not an error '
            'in this script; this subject genuinely lacks the required data.'
        )
    raise FileNotFoundError(
        f'No beta_images/ found for sub-{subject_id} at {beta_images_dir} -- run '
        'GLMsingle_first-level.py and GLMsingle_split-betas.py for this subject first.'
    )

beta_fpaths = sorted(glob(os.path.join(beta_images_dir, '*.nii.gz')))
if len(beta_fpaths) == 0:
    raise RuntimeError(f'beta_images/ exists but contains no .nii.gz files for sub-{subject_id} '
                       f'at {beta_images_dir}.')
print(f'{len(beta_fpaths)} per-trial beta images found for sub-{subject_id}')


def mask_fmri(fmri_niimg, mask_filename, fwhm):
    # NOTE: deliberately standardize=False, unlike FLT2's mask_glmsingle_betas.py (which uses
    # standardize=True). That function masks ONE 3D volume at a time (n_samples=1 in the
    # time/sample axis) -- standardizing (z-scoring) across a single sample isn't well-defined
    # and would likely produce degenerate all-zero/NaN output rather than a meaningful value.
    # Worth a first-run sanity check that these masked CSVs contain real, non-degenerate beta
    # values before trusting downstream RDMs.
    masker = NiftiMasker(mask_img=mask_filename, smoothing_fwhm=fwhm, standardize=False)
    fmri_masked = masker.fit_transform(fmri_niimg)
    return fmri_masked


# Mask every trial once, regardless of any condition subsetting (e.g. restricting RSA to just
# the 'Q' noise level) -- that filtering happens cheaply later in GLMsingle_rsa-roi.py, so this
# expensive masking step doesn't need to be repeated per condition subset.
out_dir_base = os.path.join(glmsingle_dir, 'masked_statmaps', f'sub-{subject_id}', 'statmaps_masked')

for roi in CORTICAL_ROI_LIST:
    print(roi)
    mask_fpath = os.path.join(mask_dir, f'sub-{subject_id}', f'space-{space_label}', 'masks-dseg',
                              f'sub-{subject_id}_space-{space_label}_mask-{roi}.nii.gz')
    if not os.path.exists(mask_fpath):
        print(f'No mask found for {roi} at {mask_fpath} -- skipping this ROI.')
        continue

    out_dir = os.path.join(out_dir_base, f'mask-{roi}')
    os.makedirs(out_dir, exist_ok=True)

    for beta_fpath in beta_fpaths:
        cond_label = os.path.basename(beta_fpath).replace('.nii.gz', '')
        try:
            fmri_masked = mask_fmri(beta_fpath, mask_fpath, fwhm)
        except ValueError:
            print(f'Cannot mask {cond_label} for {roi} -- empty mask or image/mask mismatch?')
            continue

        out_fpath = os.path.join(out_dir, f'{cond_label}_mask-{roi}.csv')
        np.savetxt(out_fpath, fmri_masked)

print(f'Done masking betas for sub-{subject_id}')
