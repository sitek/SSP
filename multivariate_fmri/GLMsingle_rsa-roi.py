# Based on the rsatoolbox tutorial: https://rsatoolbox.readthedocs.io/en/stable/demo_searchlight.html
# and adapted from FLT2/rsa_roi.py
import os
import sys
import re
import argparse

import numpy as np

import rsatoolbox

from glob import glob


parser = argparse.ArgumentParser(
                description='Compute per-ROI RDMs from a subject\'s masked GLMsingle single-trial betas.',
                epilog=('Example: python GLMsingle_rsa-roi.py --sub=SSP009 '
                        '--method=crossnobis '
                        '--noise_level=Q '
                        '--bidsroot=/PATH/TO/BIDS/DIR/')
                )

parser.add_argument("--sub", help="participant id",
                    type=str)
parser.add_argument("--method",
                    help="RDM calculation method (options: crossnobis, euclidean, correlation)",
                    type=str, default='crossnobis', choices=['crossnobis', 'euclidean', 'correlation'])
parser.add_argument("--noise_level",
                    help=("restrict RSA to a single noise_level (options: Q, 8, 0, n2, n6). "
                          "If omitted, use all noise levels / all 80 conditions."),
                    type=str, default=None)
parser.add_argument("--bidsroot",
                    help="top-level directory of the BIDS dataset", type=str)
args = parser.parse_args()

if len(sys.argv) < 2:
    parser.print_help()
    print(' ')
    sys.exit(1)

subject_id = args.sub
method_label = args.method
noise_level_filter = args.noise_level
bidsroot = args.bidsroot

print('participant ID:', subject_id)
print('RDM calculation method:', method_label)
print('noise_level filter:', noise_level_filter if noise_level_filter else 'none (all conditions)')

# other directory definitions
deriv_dir = os.path.join(bidsroot, 'derivatives')
model_dir = os.path.join(deriv_dir, 'glmsingle')

# Same 20-ROI cortical list as GLMsingle_mask-betas.py (14 auditory + 6 extended language
# network) -- kept in sync with that script. Ordered all-L-then-all-R (mirrored) rather than
# per-network L/R blocks, for easier hemisphere comparisons in downstream plots.
network_name = 'cortical'
roi_list = [
    'L-HG', 'L-PT', 'L-PP', 'L-STGp', 'L-STGa', 'L-ParsOp', 'L-ParsTri', 'L-SMGa', 'L-SMGp', 'L-Ang',
    'R-HG', 'R-PT', 'R-PP', 'R-STGp', 'R-STGa', 'R-ParsOp', 'R-ParsTri', 'R-SMGa', 'R-SMGp', 'R-Ang',
]

model_folder = os.path.join(model_dir, 'masked_statmaps', f'sub-{subject_id}', 'statmaps_masked')
print('model_folder:', model_folder)

# ---- regex patterns ----
# GLMsingle_split-betas.py writes filenames like:
#   sub-<id>_run-<n>_stim-<syllable>_<speaker>_<noise_level>_rep-<n>_desc-typed_mask-<ROI>.csv
# The 'stim-' capture is non-greedy up to '_rep-' since the condition string itself contains
# underscores (e.g. 'BA_F1_Q').
run_re = re.compile(r'run-(\d+)')
stim_re = re.compile(r'stim-(.+?)_rep-')
rep_re = re.compile(r'rep-(\d+)')

roi_rdm_list = []

for roi in roi_list:
    roi_folder = os.path.join(model_folder, f'mask-{roi}')
    csv_files = sorted(glob(os.path.join(roi_folder, '*.csv')))

    if len(csv_files) == 0:
        print(f'No files found for ROI {roi}')
        continue

    # group files by run (needed for the non-crossnobis, per-run RDM branch below)
    run_files = {}
    for f in csv_files:
        fname = os.path.basename(f)
        m_run = run_re.search(fname)
        m_stim = stim_re.search(fname)
        m_rep = rep_re.search(fname)

        if m_run is None or m_stim is None or m_rep is None:
            continue

        stim_label = m_stim.group(1)

        # --noise_level filter: syllable/speaker/noise_level are always exactly 3 clean
        # underscore-joined tokens (e.g. 'BA_F1_Q') -- splitting on '_' safely recovers each
        # component since none of the three values ever contain a literal underscore
        # themselves. This is what implements "RSA for just the Q condition": restricting to
        # 16 conditions (4 syllable x 4 speaker) instead of all 80, without needing separate
        # masked files per condition subset.
        if noise_level_filter is not None:
            stim_parts = stim_label.split('_')
            if len(stim_parts) != 3 or stim_parts[2] != noise_level_filter:
                continue

        run_label = f'run-{m_run.group(1)}'
        run_files.setdefault(run_label, []).append((f, stim_label, f'rep-{m_rep.group(1)}'))

    if method_label == 'crossnobis':
        data_list = []
        obs_desc = {'run': [], 'stimulus': [], 'rep': []}

        for run_label, file_entries in sorted(run_files.items()):
            for f, stim_label, rep_label in file_entries:
                try:
                    vec = np.atleast_1d(np.genfromtxt(f))
                except Exception:
                    continue
                data_list.append(vec)
                obs_desc['run'].append(run_label)
                obs_desc['stimulus'].append(stim_label)
                obs_desc['rep'].append(rep_label)

        if len(data_list) < 2:
            print(f'Skipping ROI {roi} (not enough trials)')
            continue

        dataset = rsatoolbox.data.Dataset(
            np.vstack(data_list),
            descriptors={'participant': subject_id, 'ROI': roi},
            obs_descriptors=obs_desc
        )
        rdm = rsatoolbox.rdm.calc_rdm(
            dataset, method=method_label, descriptor='stimulus', cv_descriptor='run'
        )
        roi_rdm_list.append(rdm)

    else:
        # euclidean/correlation have no built-in cross-validation across runs (unlike
        # crossnobis) -- compute one RDM per run instead, to avoid a circular/non-independent
        # estimate.
        for run_label, file_entries in sorted(run_files.items()):
            data_list = []
            obs_desc = {'run': [], 'stimulus': [], 'rep': []}

            for f, stim_label, rep_label in file_entries:
                try:
                    vec = np.atleast_1d(np.genfromtxt(f))
                except Exception:
                    continue
                data_list.append(vec)
                obs_desc['run'].append(run_label)
                obs_desc['stimulus'].append(stim_label)
                obs_desc['rep'].append(rep_label)

            if len(data_list) < 2:
                print(f'Skipping ROI {roi}, {run_label} (not enough trials)')
                continue

            dataset = rsatoolbox.data.Dataset(
                np.vstack(data_list),
                descriptors={'participant': subject_id, 'ROI': roi, 'run': run_label},
                obs_descriptors=obs_desc
            )
            rdm = rsatoolbox.rdm.calc_rdm(
                dataset, method=method_label, descriptor='stimulus'
            )
            roi_rdm_list.append(rdm)

if len(roi_rdm_list) == 0:
    raise RuntimeError(
        f'No RDMs computed for sub-{subject_id} -- check that GLMsingle_mask-betas.py has been '
        'run for this subject, and that --noise_level (if set) matches real data.'
    )

concat_rdms = rsatoolbox.rdm.rdms.concat(roi_rdm_list)
concat_rdms.descriptors['participant'] = subject_id

# save subject-level RDMs -- output filename encodes the noise-level filter so filtered and
# unfiltered results don't collide
noise_level_tag = noise_level_filter if noise_level_filter else 'all'
out_dir = os.path.join(model_dir, f'rsa-roi_glmsingle_rdmcalc-{method_label}')
os.makedirs(out_dir, exist_ok=True)
basename = f'sub-{subject_id}_glmsingle_{network_name}_{method_label}_noiselevel-{noise_level_tag}_rdms'
out_fpath = os.path.join(out_dir, f'{basename}.hdf5')
concat_rdms.save(out_fpath, file_type='hdf5', overwrite=True)
print('saved RDMs to', out_fpath)
