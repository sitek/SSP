import os
import sys
import argparse

import pandas as pd

from nilearn.glm.first_level import first_level_from_bids


''' Set up and interpret command line arguments '''
parser = argparse.ArgumentParser(
                description='Subject-level univariate modeling of fmriprep-preprocessed data',
                epilog=('Example: python univariate_first-level.py --sub=SSP009 '
                        '--task=badaga --space=MNI152NLin2009cAsym '
                        '--fwhm=6 --event_type=snr '
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
                    help="spatial smoothing full-width half-max",
                    type=float)
parser.add_argument("--event_type",
                    help="what to model (options: `snr`, `sound`)",
                    type=str, choices=['snr', 'sound'])
parser.add_argument("--t_acq",
                    help=("BOLD acquisition time (if different from "
                          "repetition time [TR], as in sparse designs)"),
                    type=float)
parser.add_argument("--t_r",
                    help="BOLD repetition time",
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
event_type = args.event_type
t_acq = args.t_acq
t_r = args.t_r
bidsroot = args.bidsroot
fmriprep_dir = args.fmriprep_dir

# correct the fmriprep-given slice reference (middle slice, or 0.5)
slice_time_ref = 0.5 * t_acq / t_r

print('bidsroot:', bidsroot)
print('fmriprep dir:', fmriprep_dir)

# create output directory (across-runs / session-level GLM only -- unlike
# multivariate_first-level.py, there is no per-run analysis_window option here)
bidsderiv_dir = os.path.join(bidsroot,
                             'derivatives',
                             'nilearn',
                             f'run-all_contrast-{event_type}' if event_type == 'snr' else 'run-all')
# exist_ok=True instead of check-then-create: run_univariate_first-level.sh is launched once per
# subject via loop_run_univariate_first-level.sh, and every subject's job targets this same
# shared directory -- a plain `if not exists: makedirs` races when jobs start close together and
# more than one passes the exists check before either creates the directory.
os.makedirs(bidsderiv_dir, exist_ok=True)
print('output directory:', bidsderiv_dir)


''' Pipeline functions (kept in sync with univariate_first-level.ipynb) '''

# Rename events based on desired analysis
def update_events(models_events, event_type='sound'):
    # SNR-level events (badaga task's speech-in-noise conditions)
    if event_type == 'snr':
        for sx, sub_events in enumerate(models_events):
            for mx, run_events in enumerate(sub_events):
                run_events['trial_type'] = run_events['noise_level']

        # create stimulus list from updated events.tsv file
        stim_list = sorted([str(s) for s in run_events['trial_type'].unique() if str(s) not in ['nan', 'None']])

    # all sound events, collapsed across trial-level suffixes
    elif event_type == 'sound':
        for sx, sub_events in enumerate(models_events):
            for mx, run_events in enumerate(sub_events):
                run_events['trial_type'] = run_events.trial_type.str.split('_', expand=True)[0]

        # create stimulus list from updated events.tsv file
        stim_list = sorted([str(s) for s in run_events['trial_type'].unique() if str(s) not in ['nan', 'None']])

    else:
        raise ValueError(f"No trial_type construction defined for event_type={event_type!r} "
                          "(only 'snr' and 'sound' are supported)")

    return stim_list, models_events


# Across-runs GLM
def nilearn_glm_across_runs(stim_list, task_label,
                            models, models_run_imgs,
                            models_events,
                            models_confounds,
                            event_type,
                            out_dir):
    from nilearn.interfaces.bids import save_glm_to_bids
    from nilearn.interfaces.fmriprep import load_confounds_strategy

    midx = 0  # only 1 subject per analysis

    if event_type == 'sound':
        contrast_list = ['sound', 'response']
    elif event_type == 'snr':
        contrast_list = ['Q', '8', '0', 'n2', 'n6', 'Q - 0', 'Q - n6']
    else:
        raise ValueError(f"No contrast_list defined for event_type={event_type!r} "
                          "(only 'snr' and 'sound' are supported)")

    model = models[midx]
    imgs = models_run_imgs[midx]
    events = models_events[midx]

    # Select confounds + a motion-based censoring mask ONCE per subject. 'scrubbing' adds
    # FD/DVARS-based volume censoring on top of compcor + motion regressors; sample_mask is
    # passed into model.fit() below so scrubbing actually takes effect.
    # FD/DVARS thresholds here are nilearn's 'scrubbing' preset defaults (fd_threshold=0.5mm,
    # std_dvars_threshold=1.5) -- confirm with the PI whether these are appropriate for this
    # pediatric/stuttering population, where task-related orofacial/vocal motion during speech
    # trials may differ from adult norms.
    print('selecting confounds and motion-scrubbing mask')
    confounds_ltd, sample_mask = load_confounds_strategy(img_files=imgs,
                                                         denoise_strategy='scrubbing')

    # Log per-subject mean framewise displacement from the raw per-run fMRIPrep confounds
    # tables first_level_from_bids already loaded into models_confounds. This manifest feeds
    # the group-level motion covariate.
    raw_confounds = models_confounds[midx]
    fd_by_run = []
    for rc in raw_confounds:
        rc_df = rc if isinstance(rc, pd.DataFrame) else pd.read_csv(rc, sep='\t')
        fd_by_run.append(rc_df['framewise_displacement'].mean())

    motion_qc_row = pd.DataFrame([{
        'subject_id': f'sub-{model.subject_label}',
        'mean_fd': pd.Series(fd_by_run).mean(),
        **{f'mean_fd_run-{rx + 1}': fd for rx, fd in enumerate(fd_by_run)},
    }])
    # One file per subject, not a shared appended motion_qc.csv: like the output directory
    # above, every subject's job would otherwise be appending to the same file from parallel
    # SLURM jobs, which can duplicate/interleave rows with no locking in place. Each job only
    # ever writes its own subject's file, so there's nothing to race. The subject's output
    # subdirectory already exists at this point (save_glm_to_bids creates it), but create it
    # here too in case contrast saving hasn't run yet.
    subject_out_dir = os.path.join(out_dir, f'sub-{model.subject_label}')
    os.makedirs(subject_out_dir, exist_ok=True)
    motion_qc_fpath = os.path.join(subject_out_dir, f'sub-{model.subject_label}_motion_qc.csv')
    motion_qc_row.to_csv(motion_qc_fpath, index=False)

    # fit the GLM once
    print('fitting GLM')
    # NOTE: the keyword for the censoring mask changed between nilearn versions
    # ('sample_mask' singular pre-0.10, 'sample_masks' plural for multi-run fits in newer
    # releases). Confirm this matches the nilearn version installed on the cluster.
    model.fit(imgs, events, confounds_ltd, sample_masks=sample_mask)
    print(model)

    for contrast_label in contrast_list:
        print('Running for contrast', contrast_label)

        # compute the contrast of interest
        print('computing contrast of interest')
        summary_statistics = model.compute_contrast(contrast_label, output_type='all')

        # save model outputs
        # NOTE: `--fwhm` is parsed as float (e.g. 6 -> 6.0), but univariate_group-level.ipynb's
        # file glob expects "fwhm-6" (as written by the notebook version of this pipeline, where
        # fwhm is a bare int). Use :g formatting so a whole-number fwhm renders as "6", not "6.0",
        # and the group-level glob keeps matching.
        out_prefix = f"sub-{model.subject_label}_task-{task_label}_fwhm-{model.smoothing_fwhm:g}"
        save_glm_to_bids(model,
                         contrast_label,
                         out_dir=out_dir,
                         prefix=out_prefix,
                        )
        print(f'Saved model outputs to {out_dir}')

    return summary_statistics


''' Run the modeling pipeline '''
models, models_run_imgs, \
        raw_models_events, \
        models_confounds = first_level_from_bids(bidsroot,
                                                 task_label,
                                                 space_label=space_label,
                                                 sub_labels=[subject_id],
                                                 smoothing_fwhm=fwhm,
                                                 derivatives_folder=fmriprep_dir,
                                                 slice_time_ref=slice_time_ref,
                                                 minimize_memory=False)

stim_list, models_events = update_events(raw_models_events,
                                         event_type=event_type)

summary_statistics = nilearn_glm_across_runs(stim_list, task_label,
                                             models, models_run_imgs,
                                             models_events,
                                             models_confounds,
                                             event_type,
                                             out_dir=bidsderiv_dir)
