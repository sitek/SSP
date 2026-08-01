import os
import sys
import argparse

import numpy as np
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

# NOTE: slice_time_ref is intentionally NOT computed/passed here. It used to be hand-derived as
# 0.5 * t_acq / t_r (a "middle slice" guess for a naive continuous acquisition), but that guess
# doesn't match this dataset's actual slice-timing protocol -- nilearn warned that the provided
# value (0.5) differed from what it read from the BIDS bold.json metadata (0.174). Leaving
# slice_time_ref=None below lets first_level_from_bids infer the correct value directly from
# that metadata instead of silently overriding it with a wrong hand-computed one.

print('participant ID:', subject_id)
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
                if 'noise_level' not in run_events.columns:
                    raise KeyError(
                        f"Run {mx + 1}'s events.tsv has no 'noise_level' column (columns present: "
                        f"{list(run_events.columns)}). event_type='snr' can't build SNR-level "
                        "trial types without it. This is usually NOT a transient/fixable failure --"
                        " it means this subject's events.tsv predates the current badaga "
                        "SNR-condition design (e.g. an early pilot subject run under a different "
                        "protocol), not that data is merely missing. Check this run's events.tsv "
                        "columns directly before assuming a rerun will help."
                    )
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


def two_term_diff_contrast(model, positive_label, negative_label):
    """Explicit 'positive_label minus negative_label' contrast vector, built per run directly
    from model.design_matrices_' column positions -- bypasses nilearn's string-formula parser
    (pandas.DataFrame.eval), which can never reference a design matrix column literally named
    with a bare numeral (e.g. '0') as an identifier: a bare '0' always parses as the numeric
    literal zero, so e.g. the string 'Q - 0' would silently evaluate to just 'Q' (subtracting
    zero is a no-op) instead of "Q minus the 0 dB SNR condition". Needed for any two-term
    contrast where either condition label could be misread as a number (currently 'Q - 0'; add
    a call here for any future contrast with the same shape, e.g. '8 - 0').
    """
    contrast_def = []
    for dm in model.design_matrices_:
        cols = list(dm.columns)
        vec = np.zeros(len(cols))
        vec[cols.index(positive_label)] = 1
        vec[cols.index(negative_label)] = -1
        contrast_def.append(vec)
    return contrast_def


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
    # fd_threshold/std_dvars_threshold are set explicitly to fMRIPrep's own conventional values
    # (0.5mm / 1.5).
    # Confirm with the PI whether 0.5mm/1.5 is still too strict for this population -- some
    # developmental/clinical studies use a more lenient FD threshold (e.g. 0.9mm).
    print('selecting confounds and motion-scrubbing mask')
    confounds_ltd, sample_mask = load_confounds_strategy(img_files=imgs,
                                                         denoise_strategy='scrubbing',
                                                         fd_threshold=0.9,
                                                         std_dvars_threshold=1.5)

    # first_level_from_bids (and load_confounds_strategy, mirroring its input) return bare,
    # non-list values for imgs/events/confounds_ltd/sample_mask when a subject has exactly one
    # run, instead of length-1 lists -- confirmed by a KeyError here for a single-run subject
    # (confounds_ltd[0] was being treated as DataFrame column indexing, not list indexing,
    # because confounds_ltd was a bare DataFrame). Normalize to lists so the per-run loop below
    # and model.fit(..., sample_masks=...) can treat single-run and multi-run subjects the same way.
    if not isinstance(confounds_ltd, list):
        imgs = [imgs]
        events = [events]
        confounds_ltd = [confounds_ltd]
        sample_mask = [sample_mask]

    # Drop any run where scrubbing censored more than half its volumes: fitting a run with an
    # empty (or near-empty) sample_mask crashes deep inside nilearn with an unhelpful numpy error
    # instead of a clear message, and a run that's mostly-censored contributes little useful
    # signal anyway. MIN_RETAINED_FRACTION is a QC convention, not derived from this data -- revisit
    # with the PI if it's excluding more runs/subjects than expected.
    MIN_RETAINED_FRACTION = 0.5
    keep_runs = []
    for rx, mask in enumerate(sample_mask):
        n_total = len(confounds_ltd[rx])
        n_retained = n_total if mask is None else len(mask)
        frac_retained = n_retained / n_total
        if frac_retained < MIN_RETAINED_FRACTION:
            print(f'WARNING: run {rx + 1} had only {frac_retained:.0%} of volumes retained after '
                 f'motion scrubbing (< {MIN_RETAINED_FRACTION:.0%} threshold) -- dropping this run.')
        else:
            keep_runs.append(rx)

    if len(keep_runs) == 0:
        raise RuntimeError(
            f'All {len(imgs)} run(s) for sub-{model.subject_label} had excessive motion after '
            f'scrubbing (< {MIN_RETAINED_FRACTION:.0%} volumes retained in every run) -- no '
            'usable data for this subject.'
        )
    if len(keep_runs) < len(imgs):
        imgs = [imgs[rx] for rx in keep_runs]
        events = [events[rx] for rx in keep_runs]
        confounds_ltd = [confounds_ltd[rx] for rx in keep_runs]
        sample_mask = [sample_mask[rx] for rx in keep_runs]

    # Log per-subject mean framewise displacement from the raw per-run fMRIPrep confounds
    # tables first_level_from_bids already loaded into models_confounds. This manifest feeds
    # the group-level motion covariate.
    # Same single-run degeneration as above: models_confounds[midx] is a bare DataFrame (not a
    # length-1 list) for single-run subjects, and `for rc in raw_confounds` over a bare DataFrame
    # silently iterates its column names instead of the confounds table itself.
    raw_confounds = models_confounds[midx]
    if not isinstance(raw_confounds, list):
        raw_confounds = [raw_confounds]
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
        if contrast_label == 'Q - 0':
            contrast_def = two_term_diff_contrast(model, 'Q', '0')
        else:
            contrast_def = contrast_label
        summary_statistics = model.compute_contrast(contrast_def, output_type='all')

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


def drop_empty_event_runs(imgs, events, confounds, subject_label):
    """Drop any run with no 'sound' trials in its events.tsv (e.g. a run started then aborted
    almost immediately) BEFORE update_events() tries to build trial types from it -- otherwise
    one empty/malformed run crashes the whole subject (a raw KeyError deep in pandas) instead of
    just being excluded, the same way nilearn_glm_across_runs() below already drops a run for
    excessive motion. Operates on imgs/events/confounds together so the three per-run parallel
    lists never drift out of sync with each other.

    Real case that motivated this: sub-SSP001 has 3 badaga runs on disk: run-02's events.tsv has
    no usable trials (empty/aborted run), while run-01 and run-03 are fine -- this should drop
    just run-02 and proceed with the other two, not fail the whole subject.
    """
    # first_level_from_bids returns bare, non-list values for a subject with exactly one run,
    # instead of a length-1 list -- same normalization already needed in nilearn_glm_across_runs().
    if not isinstance(imgs, list):
        imgs, events, confounds = [imgs], [events], [confounds]

    keep_runs = []
    for rx, run_events in enumerate(events):
        has_sound_trials = ('trial_type' in run_events.columns
                            and (run_events['trial_type'] == 'sound').any())
        if not has_sound_trials:
            print(f'WARNING: run {rx + 1} for sub-{subject_label} has no "sound" trials in its '
                 'events.tsv (empty/aborted run?) -- dropping this run.')
        else:
            keep_runs.append(rx)

    if len(keep_runs) == 0:
        raise RuntimeError(
            f'All {len(imgs)} run(s) for sub-{subject_label} have no usable badaga events -- '
            'no usable data for this subject.'
        )
    if len(keep_runs) < len(imgs):
        imgs = [imgs[rx] for rx in keep_runs]
        events = [events[rx] for rx in keep_runs]
        confounds = [confounds[rx] for rx in keep_runs]
    return imgs, events, confounds


''' Run the modeling pipeline '''
models, models_run_imgs, \
        raw_models_events, \
        models_confounds = first_level_from_bids(bidsroot,
                                                 task_label,
                                                 space_label=space_label,
                                                 sub_labels=[subject_id],
                                                 smoothing_fwhm=fwhm,
                                                 derivatives_folder=fmriprep_dir,
                                                 slice_time_ref=None,  # infer from BIDS metadata, see note above
                                                 minimize_memory=False)

midx = 0  # only 1 subject per analysis
models_run_imgs[midx], raw_models_events[midx], models_confounds[midx] = drop_empty_event_runs(
    models_run_imgs[midx], raw_models_events[midx], models_confounds[midx], subject_id)

stim_list, models_events = update_events(raw_models_events,
                                         event_type=event_type)

summary_statistics = nilearn_glm_across_runs(stim_list, task_label,
                                             models, models_run_imgs,
                                             models_events,
                                             models_confounds,
                                             event_type,
                                             out_dir=bidsderiv_dir)
