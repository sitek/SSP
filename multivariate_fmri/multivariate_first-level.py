import os
import sys
import argparse

from glob import glob
from nilearn.glm.first_level import first_level_from_bids
from nilearn.interfaces.fmriprep import load_confounds


''' Set up and interpret command line arguments '''
parser = argparse.ArgumentParser(
                description='Subject-level modeling of fmriprep-preprocessed data',
                epilog=('Example: python multivariate_first-level.py --sub=SSP009 '
                        '--task=badaga --space=MNI152NLin2009cAsym '
                        ' --analysis_window=run '
                        '--fwhm=6 --event_type=stimulus --model_type=LSA '
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
parser.add_argument("--analysis_window", 
                    help="analysis window (options: session, run}", 
                    type=str)
parser.add_argument("--fwhm", 
                    help="spatial smoothing full-width half-max", 
                    type=float)
parser.add_argument("--event_type", 
                    help="what to model (options: `trial`, `sound`, `stimulus`, `feedback`, or `motor`)", 
                    type=str)
parser.add_argument("--model_type", 
                    help="trial model scheme (options: `LSA` or `LSS`)", 
                    type=str)
parser.add_argument("--t_acq", 
                    help=("BOLD acquisition time (if different from "
                          "repetition time [TR], as in sparse designs)"), 
                    type=float)
parser.add_argument("--t_r", 
                    help="BOLD repetition time", 
                    type=float)
parser.add_argument("--bidsroot", 
                    help="top-level directory of the BIDS dataset", 
                    type=str)
parser.add_argument("--fmriprep_dir", 
                    help="directory of the fMRIprep preprocessed dataset", 
                    type=str)

args = parser.parse_args()

if len(sys.argv) < 2:
    parser.print_help()
    print(' ')
    sys.exit(1)
    
subject_id = args.sub
task_label = args.task
space_label=args.space
analysis_window = args.analysis_window
fwhm = args.fwhm
event_type=args.event_type
model_type=args.model_type
t_acq = args.t_acq
t_r = args.t_r
bidsroot = args.bidsroot
fmriprep_dir = args.fmriprep_dir

# correct the fmriprep-given slice reference (middle slice, or 0.5)
slice_time_ref = 0.5 * t_acq / t_r
                          
print('bidsroot:', bidsroot)
print('fmriprep dir:', fmriprep_dir)
                    
if analysis_window == 'run':
    glm_label = 'run-specific'
elif analysis_window == 'session':
    glm_label = 'run-all'

# create output directory
bidsderiv_dir = os.path.join(bidsroot, 
                             'derivatives', 
                             'nilearn', 
                             f'{glm_label}_eventtype-{event_type}')
if not os.path.exists(bidsderiv_dir):
    os.makedirs(bidsderiv_dir)
print('output directory:', bidsderiv_dir)

''' Pipeline functions '''
# Rename events based on desired analysis
def update_events(models_events, event_type='sound'):
    # stimulus events
    if event_type == 'stimulus':
        for sx, sub_events in enumerate(models_events):
            for mx, run_events in enumerate(sub_events):
                run_events['trial_type'] = run_events['stim_file'].str.replace('.wav', '')
                run_events['trial_type'] = run_events['trial_type'].str.replace('-','_')

        # create stimulus list from updated events.tsv file
        stim_list = sorted([s for s in run_events['trial_type'].unique() if str(s) != 'nan'])
    
    # trial-specific events
    if event_type == 'trial':
        for sx, sub_events in enumerate(models_events):
            for mx, run_events in enumerate(sub_events):
                name_groups = run_events.groupby('stim_file')['stim_file']
                suffix = name_groups.cumcount() + 1
                #repeats = name_groups.transform('size')
                print(suffix)

                run_events['trial_type'] = run_events['stim_file'].str.replace('.wav', '') + \
                                           '_trial' + suffix.map(str)[:-2]
                                           
                run_events['trial_type'] = run_events['trial_type'].str.replace('-','_')
                run_events['trial_type'] = run_events['trial_type'].str.replace('.0','')

        # create stimulus list from updated events.tsv file
        stim_list = sorted([s for s in run_events['trial_type'].unique() if str(s) != 'nan'])

    # all sound events
    elif event_type == 'sound':
        for sx, sub_events in enumerate(models_events):
            for mx, run_events in enumerate(sub_events):
                orig_stim_list = sorted([str(s) for s in run_events['trial_type'].unique() if str(s) not in ['nan', 'None']])

                run_events['trial_type'] = run_events.trial_type.str.split('_', expand=True)[0]

        # create stimulus list from updated events.tsv file
        stim_list = sorted([str(s) for s in run_events['trial_type'].unique() if str(s) not in ['nan', 'None']])
    
    #print('stim list: ', stim_list)
    return stim_list, models_events

# transform full event design matrix (LSA) into single-event only (LSS)
def lss_transformer(event_df, event_name):
    other_idx = np.array(event_df.loc[:,'trial_type'] != event_name)
    lss_event_df = event_df.copy()
    lss_event_df.loc[other_idx, 'trial_type'] = 'other_events' 
    return lss_event_df

# Per-run GLM
def nilearn_glm_per_run(stim_list, task_label, 
                        models, models_run_imgs, 
                        models_events, 
                        models_confounds,
                        out_dir,
                        model_type='LSA'):
    import nibabel as nib
    from nilearn.interfaces.bids import save_glm_to_bids
    from nilearn.interfaces.fmriprep import load_confounds_strategy
    
    midx = 0 # only 1 subject per analysis
    model = models[midx]

    # set limited confounds
    print('selecting confounds')
    imgs = models_run_imgs[midx]
    confounds_ltd, sample_mask = load_confounds_strategy(img_files=imgs, 
                                                         denoise_strategy='compcor')    
    #contrast_list = stim_list
    for contrast_label in stim_list:

        for rx in range(len(confounds_ltd)):
            img = models_run_imgs[midx][rx]
            confound = confounds_ltd[rx]

            if model_type == 'LSA':
                event = models_events[midx][rx]
            elif model_type == 'LSS':
                event = lss_transformer(models_events[midx][rx], stim)
            print(event)

            print('fitting GLM')
            model.fit(img, event, confound); 
            print(model)

            # compute the contrast of interest
            print('computing contrast of interest')
            summary_statistics = model.compute_contrast(contrast_label, output_type='all')

            # save model outputs
            out_prefix = f"sub-{model.subject_label}_task-{task_label}_" + \
                         f"fwhm-{model.smoothing_fwhm}_run-{rx+1}"
            '''
            save_glm_to_bids(model, 
                             contrast_label,
                             out_dir=out_dir,
                             prefix=out_prefix,
                            )
            '''
            # currently a JSONDecodeError ^^
            t_stat = model.compute_contrast(contrast_label, output_type='stat')
            os.makedirs(os.path.join(out_dir, f'sub-{model.subject_label}'), exist_ok=True)
            nib.save(t_stat, os.path.join(out_dir, f'sub-{model.subject_label}',
                                          out_prefix+f'_contrast-{contrast_label}_stat-t_statmap.nii.gz'))
            print(f'Saved model outputs to {out_dir}')

    return summary_statistics


# All-runs GLM
def nilearn_glm_across_runs(stim_list, task_label, 
                            models, models_run_imgs, 
                            models_events, 
                            models_confounds,
                            out_dir,
                            model_type='LSA'):
    from nilearn.interfaces.bids import save_glm_to_bids
    from nilearn.interfaces.fmriprep import load_confounds_strategy
    import nibabel as nib
    
    midx = 0 # only 1 subject per analysis
    model = models[midx]
    
    #contrast_list = stim_list
    for contrast_label in stim_list:

        # set limited confounds
        print('selecting confounds')
        imgs = models_run_imgs[midx]
        confounds_ltd, sample_mask = load_confounds_strategy(img_files=imgs, 
                                                             denoise_strategy='compcor')

        if model_type == 'LSA':
            events = models_events[midx]
        elif model_type == 'LSS':
            events = [lss_transformer(ev, stim) for ev in len(models_events[midx])]

        #try:
        # fit the GLM
        print('fitting GLM')
        model.fit(imgs, events, confounds_ltd); 
        print(model)

        # compute the contrast of interest
        print('computing contrast of interest')
        summary_statistics = model.compute_contrast(contrast_label, output_type='all')

        # save model outputs
        out_prefix = f"sub-{model.subject_label}_task-{task_label}_" + \
                     f"fwhm-{model.smoothing_fwhm}_run-all"
        '''
        save_glm_to_bids(model, 
                         contrast_label,
                         out_dir=out_dir,
                         prefix=out_prefix,
                        )
        '''
        # currently a JSONDecodeError ^^
        t_stat = model.compute_contrast(contrast_label, output_type='stat')
        os.makedirs(os.path.join(out_dir, f'sub-{model.subject_label}'), exist_ok=True)
        nib.save(t_stat, os.path.join(out_dir, f'sub-{model.subject_label}',
                                      out_prefix+f'_contrast-{contrast_label}_stat-t_statmap.nii.gz'))
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

if analysis_window == 'run':
    # Per-run GLM
    summary_statistics = nilearn_glm_per_run(stim_list, task_label, 
                                             models, models_run_imgs, 
                                             models_events, 
                                             models_confounds,
                                             out_dir=bidsderiv_dir)
elif analysis_window == 'session':
    # Per-run GLM
    summary_statistics = nilearn_glm_across_runs(stim_list, task_label, 
                                             models, models_run_imgs, 
                                             models_events, 
                                             models_confounds,
                                             out_dir=bidsderiv_dir)