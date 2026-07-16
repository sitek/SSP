# SSP

fMRI analysis pipeline studying subcortical/cortical auditory speech processing, comparing
children who stutter (**CWS**) and children who do not stutter (**CWNS**). Developed for the
Chandrasekaran Lab (University of Pittsburgh). Task names in the data: `rest`, `badaga` (a
speech-in-noise / phoneme discrimination task), and `alice` (narrative listening).

Data lives on the lab's HPC cluster under `/bgfs/bchandrasekaran/krs228/data/SSP/`, organized as
a BIDS dataset (`data_bids/`) with fMRIPrep derivatives under `data_bids/derivatives/`. This repo
holds the processing/analysis code only, not the data itself.

## Pipeline stages

Run roughly in this order. Each stage is its own top-level directory.

1. **`dicom_conversion/`** — Convert raw DICOMs to BIDS using [HeuDiConv](https://heudiconv.readthedocs.io/).
   - `initialize_dicoms_heudiconv.sh` → `convert_dicoms_heudiconv.sh` (heuristic in `heuristic.py`)
   - `dcm_2_bids.ipynb` — interactive/exploratory version of the same conversion

2. **`fmriprep/`** — Preprocess the BIDS dataset with [fMRIPrep](https://fmriprep.org/).
   - `run_fmriprep.sh`

3. **`hmri-toolbox/`** — Quantitative T1/T2* mapping via SPM's [hMRI toolbox](https://hmri-group.github.io/hMRI-toolbox/).
   - SPM batch files (`.mat`) for auto-reorientation, B1 mapping, denoising, and quantitative map creation

4. **`masking/`** and **`analysis/`** — Brain masks, atlas-based ROI definitions, and normalized
   T2* extraction per ROI.
   - `make_gm_mask.py`, `make_atlas_region_masks.py` (+ `run_*.sh` / `loop_run_*.sh` cluster wrappers)
   - `normT2star.py` / `normT2star.ipynb` — per-subject, per-ROI normalized T2* values (e.g.
     `analysis/SSP011_mean_normT2star.csv`)
   - `mask_trial_betas.py` — mask single-trial beta maps for downstream multivariate analysis

5. **`fmri_modeling/`** — First- and group-level GLM and multivariate analysis. See below.

6. **`behavior/`** — Behavioral data analysis and participant demographics/covariates
   (`behavior_analysis.ipynb`, `participant_covariates.ipynb`), which feed the group-level
   covariates used in `fmri_modeling/`.

## `fmri_modeling/`: univariate pipeline

The univariate GLM pipeline (nilearn-based) has two stages, run in order:

### 1. First level (per subject)

Fits one subject's GLM across all runs of the `badaga` task and saves per-contrast effect/z-stat
maps. Two equivalent ways to run it:

- **Interactively**: `univariate_first-level.ipynb` — set `subject_list` in the notebook and run.
- **On the cluster (recommended for the full cohort)**:
  ```bash
  # one subject
  sbatch run_univariate_first-level.sh SSP009

  # all subjects found under the BIDS root
  bash loop_run_univariate_first-level.sh
  ```
  `univariate_first-level.py` is the underlying CLI script (see `--help` for all arguments).

Each subject's run:
- Selects confounds and a motion-based censoring mask (nilearn's `'scrubbing'` denoise strategy)
  and fits the GLM **once**, computing all contrasts from that single fit.
- Writes per-contrast effect/z-stat maps to
  `data_bids/derivatives/nilearn/run-all_contrast-snr/<sub-id>/` (or `run-all/` for the `sound`
  event type).
- Appends to two manifests in that same output directory:
  - `motion_qc.csv` — per-subject mean framewise displacement (feeds the group-level motion
    covariate below).
  - `first_level_run_manifest.csv` — per-subject success/failure log, so a failed subject is
    recorded with its error instead of silently skipped.

Contrasts computed: `Q, 8, 0, n2, n6, Q - 0, Q - n6` (SNR levels, `event_type='snr'`) or
`sound, response` (`event_type='sound'`).

### 2. Group level

`univariate_group-level.ipynb` — run after first level has completed for the subjects you want
in the group analysis. Steps, in notebook order:

1. Reads `participants.tsv`, drops `ignore_subs`, and builds `subjects_label` (all included
   subjects) plus per-group subject lists (`sub_list_cws`, `sub_list_cwns`), using a
   case-/whitespace-normalized comparison against the `group` column.
2. Builds `cov_design_df`: mean-centers continuous covariates (age, TONI, CTOPP, CELF subscales,
   PTA, WIN scores), one-hot encodes `CWS`/`CWNS`, and reports (rather than silently drops) any
   missing covariate values.
3. Merges in `motion_qc.csv` as a `mean_fd` covariate, if the first-level run that produced it has
   already completed.
4. For **each of the 9 contrasts** (`qMinusN6, q, 8, 0, n2, n6, qMinus0, sound, response`): fits
   one `SecondLevelModel` on the combined CWS+CWNS sample, then computes three FDR-corrected
   z-maps from that single fit — the CWNS main effect, the CWS main effect, and the CWS-vs-CWNS
   group difference — via the shared `prepare_group_inputs` / `compute_group_contrast` /
   `plot_mosaic_with_contours` helpers defined near the top of the notebook.

Mosaic figures are saved to `data_bids/derivatives/nilearn/group_run-all/`.

### Multivariate / RSA

`multivariate_first-level.py` + `run_multivariate_first-level.sh` / `loop_run_multivariate_first-level.sh`
(per-run or per-trial LSA/LSS modeling), `GLMsingle.ipynb`, and `rsa_searchlight.py` +
`run_rsa_searchlight.sh` / `loop_run_rsa_searchlight.sh` (searchlight RSA) — a separate track from
the univariate pipeline above, feeding `group_level_all_ROI.ipynb` and
`group_level_rsa_searchlight_WIP.ipynb`.

## Open items before publication

A few things in the univariate pipeline are flagged inline in the code as needing PI/team
confirmation before relying on the results:
- Whether `participants.tsv` actually stores the `group` column as lowercase `'cws'` (the code
  now normalizes case either way, but the *source* casing should still be confirmed).
- The FD/DVARS motion-scrubbing thresholds (`'scrubbing'` denoise strategy defaults), given this
  is a pediatric/stuttering population where task-related motion may differ from adult norms.
- The `sample_mask`/`sample_masks` keyword name in `nilearn_glm_across_runs()`, which changed
  between nilearn versions — confirm against the version installed on the cluster.
- The undocumented `sub-SSP106`/`sub-SSP107` exclusion in `univariate_group-level.ipynb`'s
  `ignore_subs` (marked `TODO` in the notebook).
