import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import nibabel as nib

from nilearn.glm.first_level import first_level_from_bids
from nilearn.image import smooth_img

from glmsingle import GLM_single


''' Set up and interpret command line arguments '''
parser = argparse.ArgumentParser(
                description='Subject-level GLMsingle single-trial modeling of fmriprep-preprocessed data',
                epilog=('Example: python GLMsingle_first-level.py --sub=SSP009 '
                        '--task=badaga --space=MNI152NLin2009cAsym '
                        '--fwhm=0 --stimdur=0.3 '
                        '--t_acq=2 --t_r=2 '
                        '--bidsroot=/PATH/TO/BIDS/DIR/ '
                        '--fmriprep_dir=/PATH/TO/FMRIPREP/DIR/')
                )

parser.add_argument("--sub",
                    help="participant id", type=str)
parser.add_argument("--task",
                    help="task id", type=str)
parser.add_argument("--space",
                    help="space label", type=str)
parser.add_argument("--fwhm",
                    help=("spatial smoothing full-width half-max. GLMsingle is typically used "
                          "for single-trial decoding/RSA, where smoothing blurs exactly the "
                          "fine-grained spatial patterns of interest -- 0 (no smoothing) is the "
                          "conventional choice unless there's a specific reason to smooth."),
                    type=float)
parser.add_argument("--stimdur",
                    help="stimulus/event duration in seconds, passed to GLMsingle",
                    type=float)
parser.add_argument("--t_acq",
                    help=("BOLD acquisition time (if different from repetition time [TR]). "
                          "Not currently used in any computation here -- slice_time_ref is left "
                          "to nilearn's BIDS-metadata inference instead of being hand-derived "
                          "from t_acq/t_r -- kept for CLI consistency with the other first-level "
                          "scripts in this repo."),
                    type=float)
parser.add_argument("--t_r",
                    help="BOLD repetition time (seconds)",
                    type=float)
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
fwhm = args.fwhm
stimdur = args.stimdur
t_r = args.t_r
bidsroot = args.bidsroot
fmriprep_dir = args.fmriprep_dir

print('participant ID:', subject_id)
print('bidsroot:', bidsroot)
print('fmriprep dir:', fmriprep_dir)

# create per-subject output directory (exist_ok=True: this same shared derivatives root is
# targeted by every subject's parallel SLURM job, so check-then-create would race -- see the
# same fix already applied in univariate_fmri/univariate_first-level.py)
bidsderiv_dir = os.path.join(bidsroot, 'derivatives', 'glmsingle')
subject_out_dir = os.path.join(bidsderiv_dir, f'sub-{subject_id}')
os.makedirs(subject_out_dir, exist_ok=True)
print('output directory:', subject_out_dir)


''' Pipeline functions '''

def build_condition_labels(events):
    """Filter to 'sound' events (badaga's trial_type is only ever 'sound' or 'response' --
    'response' rows are behavioral button-presses with no stimulus identity and aren't
    modeled here) and build a condition label directly from the syllable/speaker/noise_level
    columns. This is deliberately NOT built from `stim_file` (as multivariate_first-level.py's
    `update_events()` 'stimulus' branch does): at least one real stimulus filename has a
    double-period typo (e.g. 'GA_F1_n6..wav'), which naive '.wav' stripping would turn into a
    condition label with a stray trailing '.', silently splitting one real condition into two
    mismatched labels. The syllable/speaker/noise_level columns are already clean.
    """
    sound_events = events[events['trial_type'] == 'sound'].copy()
    sound_events = sound_events.dropna(subset=['onset', 'syllable', 'speaker', 'noise_level'])
    sound_events['condition'] = (sound_events['syllable'].astype(str) + '_' +
                                 sound_events['speaker'].astype(str) + '_' +
                                 sound_events['noise_level'].astype(str))
    return sound_events


def build_glmsingle_design(sound_events_by_run, cond_list, t_r, n_trs_per_run):
    """Build one (n_TRs, n_conditions) indicator matrix per run for GLMsingle -- one column per
    unique stimulus condition (repeats of the same condition share a column; this is what
    GLMsingle's cross-validated denoising/HRF-fitting needs, NOT a unique column per trial
    occurrence, even though the goal is single-trial output betas -- GLMsingle emits one beta
    per trial *occurrence* natively regardless of column grouping; see the comment in this
    file's header for why).

    Also builds a trial manifest in chronological order (run, then onset within run) recording
    which condition each design-matrix '1' belongs to. GLMsingle_split-betas.py uses this
    manifest's row order to label each of GLMsingle's output beta slices -- deliberately NOT
    using a flat run/condition/repeat index formula (as in the FLT2 reference this pipeline is
    adapted from), since real repeats-per-condition-per-run in this dataset look irregular
    rather than fixed, so that formula's regular-structure assumption doesn't reliably hold here.
    """
    design = []
    manifest_rows = []
    for run_idx, run_events in enumerate(sound_events_by_run):
        n_trs = n_trs_per_run[run_idx]
        run_design = np.zeros((n_trs, len(cond_list)), dtype=int)

        for _, row in run_events.sort_values('onset').iterrows():
            tr_idx = int(round(row['onset'] / t_r))
            if tr_idx >= n_trs:
                print(f'WARNING: run {run_idx + 1} onset {row["onset"]:.3f}s maps to TR {tr_idx}, '
                     f'at/beyond run length ({n_trs} TRs) -- skipping this trial.')
                continue

            cond_idx = cond_list.index(row['condition'])
            if run_design[tr_idx, cond_idx] == 1:
                # Two same-condition onsets rounding to the same TR can only occupy one design-
                # matrix cell -- skip logging this trial in the manifest too, so the manifest's
                # row count always matches the design matrix's actual 1-count (and therefore
                # GLMsingle's output trial count). Confirmed to matter, not just theoretical: a
                # synthetic dry run of this exact scenario showed the manifest and design 1-count
                # silently diverging by one when this trial was still logged.
                print(f'WARNING: run {run_idx + 1} has two "{row["condition"]}" onsets landing '
                     f'on the same TR ({tr_idx}) -- skipping the second one (only one can be '
                     'modeled at that TR).')
                continue
            run_design[tr_idx, cond_idx] = 1

            manifest_rows.append({
                'trial_index': len(manifest_rows),
                'run': run_idx + 1,
                'onset': row['onset'],
                'tr_index': tr_idx,
                'condition': row['condition'],
                'syllable': row['syllable'],
                'speaker': row['speaker'],
                'noise_level': row['noise_level'],
            })

        design.append(run_design)

    manifest = pd.DataFrame(manifest_rows)
    return design, manifest


''' Run the modeling pipeline '''

# Use first_level_from_bids purely to enumerate run images/events/confounds (matches this
# repo's other first-level scripts); the nilearn FirstLevelModel objects it also returns are
# never fit -- GLMsingle replaces nilearn's own GLM fitting entirely.
# slice_time_ref=None: let nilearn infer it from BIDS metadata rather than hand-computing it
# (0.5*t_acq/t_r was already found to be wrong for this dataset's real slice-timing protocol
# in univariate_fmri/univariate_first-level.py -- don't reintroduce that bug here).
_models, models_run_imgs, raw_models_events, models_confounds = first_level_from_bids(
    bidsroot,
    task_label,
    space_label=space_label,
    sub_labels=[subject_id],
    derivatives_folder=fmriprep_dir,
    slice_time_ref=None,
    minimize_memory=False,
)

midx = 0  # only 1 subject per analysis
imgs = models_run_imgs[midx]
events_list = raw_models_events[midx]
confounds_list = models_confounds[midx]

# first_level_from_bids returns bare (non-list) values instead of length-1 lists for single-run
# subjects -- same normalization already needed in univariate_first-level.py.
if not isinstance(imgs, list):
    imgs = [imgs]
    events_list = [events_list]
    confounds_list = [confounds_list]

# Per-run mean framewise displacement, for documentation/QC and an optional run-drop guard.
# GLMsingle has no sample_mask/censoring input (unlike nilearn's FirstLevelModel.fit()) -- it
# does its own data-driven denoising internally -- so this is QC logging only, not something
# fed into the model.
fd_by_run = []
for rc in confounds_list:
    rc_df = rc if isinstance(rc, pd.DataFrame) else pd.read_csv(rc, sep='\t')
    fd_by_run.append(rc_df['framewise_displacement'].mean())

# Drop a run only for egregious motion. This threshold is a QC convention, not derived from this
# data -- revisit with the PI if it's excluding more runs/subjects than expected.
MOTION_FD_DROP_THRESHOLD_MM = 2.0
keep_runs = [rx for rx, fd in enumerate(fd_by_run) if fd <= MOTION_FD_DROP_THRESHOLD_MM]
if len(keep_runs) == 0:
    raise RuntimeError(
        f'All {len(imgs)} run(s) for sub-{subject_id} exceed the motion QC threshold '
        f'(mean FD > {MOTION_FD_DROP_THRESHOLD_MM}mm) -- no usable data for this subject.'
    )
if len(keep_runs) < len(imgs):
    dropped = [rx + 1 for rx in range(len(imgs)) if rx not in keep_runs]
    print(f'Dropping run(s) {dropped} for sub-{subject_id}: mean FD exceeds '
         f'{MOTION_FD_DROP_THRESHOLD_MM}mm.')
    imgs = [imgs[rx] for rx in keep_runs]
    events_list = [events_list[rx] for rx in keep_runs]
    fd_by_run = [fd_by_run[rx] for rx in keep_runs]

motion_qc_row = pd.DataFrame([{
    'subject_id': f'sub-{subject_id}',
    'mean_fd': pd.Series(fd_by_run).mean(),
    **{f'mean_fd_run-{rx + 1}': fd for rx, fd in enumerate(fd_by_run)},
}])
motion_qc_fpath = os.path.join(subject_out_dir, f'sub-{subject_id}_motion_qc.csv')
motion_qc_row.to_csv(motion_qc_fpath, index=False)

# build per-run sound-event tables + the full condition list (built dynamically from what's
# actually observed for this subject, not a hardcoded stimulus count -- the real stimulus set
# size/composition has been observed to vary from what's hardcoded elsewhere in this repo)
sound_events_by_run = [build_condition_labels(ev) for ev in events_list]
cond_list = sorted(set().union(*(set(ev['condition']) for ev in sound_events_by_run)))
print(f'{len(cond_list)} unique conditions found for sub-{subject_id}')

# load (and optionally smooth) each run's functional data
data = []
n_trs_per_run = []
for img_fpath in imgs:
    img = nib.load(img_fpath)
    if fwhm and fwhm > 0:
        img = smooth_img(img, fwhm)
    arr = img.get_fdata()
    data.append(arr)
    n_trs_per_run.append(arr.shape[3])

design, manifest = build_glmsingle_design(sound_events_by_run, cond_list, t_r, n_trs_per_run)

if len(manifest) == 0:
    raise RuntimeError(f'No sound-event trials found for sub-{subject_id} -- check events.tsv.')

manifest_fpath = os.path.join(subject_out_dir,
                              f'sub-{subject_id}_task-{task_label}_desc-trialbetas_manifest.csv')
manifest.to_csv(manifest_fpath, index=False)
print(f'{len(manifest)} total trials across {len(design)} run(s); manifest saved to {manifest_fpath}')

# GLMsingle's cross-validated denoising (wantglmdenoise/TYPEC) and ridge regularization
# (wantfracridge/TYPED) both require at least one condition to repeat ACROSS runs, not just
# within a single run -- GLMsingle silently disables both and falls back to TYPEB (FITHRF, no
# cross-validated denoising/ridge) otherwise. We check for this ourselves and disable both
# explicitly instead, so the decision is traceable here rather than only visible as a GLMsingle
# UserWarning, and so downstream (GLMsingle_split-betas.py) knows which output type is this
# subject's final estimate instead of discovering TYPED is simply missing. This was a real case,
# not hypothetical: sub-SSP106 only has 1 badaga run on disk (likely the same undocumented reason
# it's already excluded in univariate_group-level.ipynb's ignore_subs).
runs_per_condition = manifest.groupby('condition')['run'].nunique()
has_cross_run_repeats = bool((runs_per_condition > 1).any())

if not has_cross_run_repeats:
    print(f'WARNING: no condition repeats across runs for sub-{subject_id} ({len(design)} '
         'run(s) found) -- GLMsingle\'s cross-validated denoising/ridge regularization needs '
         'repeats across runs, not just within one. Disabling wantglmdenoise/wantfracridge '
         'explicitly and using TYPEB_FITHRF as the final estimate for this subject instead of '
         'TYPED_FITHRF_GLMDENOISE_RR.')
    final_type = 'TYPEB_FITHRF'
else:
    final_type = 'TYPED_FITHRF_GLMDENOISE_RR'

# Record which output type is this subject's final estimate. Downstream RSA work should only
# ever use TYPED subjects (per the user: "I don't want to mix in some TypeB data with TypeD") --
# GLMsingle_split-betas.py reads this to route TYPEB-derived betas into a physically separate
# output directory, not just a metadata flag that could be missed.
info_fpath = os.path.join(subject_out_dir, f'sub-{subject_id}_glmsingle_info.json')
with open(info_fpath, 'w') as f:
    json.dump({
        'final_type': final_type,
        'has_cross_run_repeats': has_cross_run_repeats,
        'n_runs_used': len(design),
    }, f, indent=2)

# running python GLMsingle involves creating a GLM_single object and then running the
# procedure using the .fit() routine
opt = dict()
opt['wantlibrary'] = 1
opt['wantglmdenoise'] = 1 if has_cross_run_repeats else 0
opt['wantfracridge'] = 1 if has_cross_run_repeats else 0
opt['wantfileoutputs'] = [1, 1, 1, 1]
opt['wantmemoryoutputs'] = [0, 0, 0, 0]  # never used in-memory here -- GLMsingle_split-betas.py
                                          # always reloads from disk, and requesting a type that
                                          # may not even be computed just produces a confusing
                                          # "nothing selected to return" message for no benefit

glmsingle_obj = GLM_single(opt)
print(glmsingle_obj.params)

# Idempotency check on this subject's actual final-type output, not just the output directory's
# existence: subject_out_dir already exists by this point (created above to hold
# motion_qc.csv/the manifest), so checking directory existence alone would always skip re-running
# GLMsingle even if a prior run crashed before finishing the fit.
final_type_fpath = os.path.join(subject_out_dir, f'{final_type}.npy')
if os.path.exists(final_type_fpath):
    print(f'GLMsingle {final_type} output already exists for sub-{subject_id}:\n\t{final_type_fpath}')
else:
    print(f'Running GLMsingle for sub-{subject_id}...')
    results_glmsingle = glmsingle_obj.fit(
        design,
        data,
        stimdur,
        t_r,
        outputdir=subject_out_dir,
    )
    print(f'GLMsingle finished for sub-{subject_id}; outputs saved to {subject_out_dir}')
