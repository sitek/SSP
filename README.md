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

5. **`univariate_fmri/`** and **`multivariate_fmri/`** — First- and group-level GLM analysis,
   split into two independent tracks. See below.

6. **`behavior/`** — Behavioral data analysis and participant demographics/covariates
   (`behavior_analysis.ipynb`, `participant_covariates.ipynb`), which feed the group-level
   covariates used in `univariate_fmri/` and `multivariate_fmri/`.

7. **`report/`** — Generates a self-contained HTML status report from the cached outputs of
   stages 5-6. See below.

## `univariate_fmri/`: univariate pipeline

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

1. Reads `participants.tsv` and builds `subjects_label` (all included subjects) plus per-group
   subject lists (`sub_list_cws`, `sub_list_cwns`), using a case-/whitespace-normalized comparison
   against the `group` column. No hardcoded `ignore_subs` list anymore — a subject is included
   here and dropped later, gracefully and per-file, wherever their derivatives are actually
   missing (see `prepare_group_inputs`).
2. Builds `cov_design_df`: mean-centers continuous covariates (age, TONI, CTOPP, CELF subscales,
   PTA, WIN scores), one-hot encodes `CWS`/`CWNS`, and reports (rather than silently drops) any
   missing covariate values.
3. Merges in `motion_qc.csv` as a `mean_fd` covariate, if the first-level run that produced it has
   already completed.
4. For **each of the 10 contrasts** (`qMinusN6, q, 8, 0, n2, n6, qMinus0, sound, response`, plus a
   derived `snrTrend` — a parametric linear contrast across all 5 SNR levels, cleanest-to-noisiest,
   built via `math_img` from the already-fit per-condition statmaps rather than a new first-level
   run): fits one `SecondLevelModel` on the combined CWS+CWNS sample, then computes three
   FDR-corrected z-maps from that single fit — the CWNS main effect, the CWS main effect, and the
   CWS-vs-CWNS group difference — via the shared `prepare_group_inputs` / `compute_group_contrast`
   / `plot_mosaic_with_contours` helpers defined near the top of the notebook.

Mosaic figures are saved to `data_bids/derivatives/nilearn/group_run-all/`.

### Other notebooks in `univariate_fmri/`

- `group_level_all_ROI.ipynb` — ROI-based (rather than whole-brain) comparison of the SNR
  contrasts across a cortical-network atlas (20 ROIs: 14 auditory + 6 extended language network),
  with FDR correction throughout. Includes an omnibus ANOVA (hemisphere &times; SNR &times;
  region) with FDR-corrected post-hoc pairwise comparisons, a parametric linear SNR-trend test per
  ROI, and a laterality index ((R&minus;L)/(|L|+|R|)) per ROI/group/SNR-level, adapted from the
  `acfMRI` repo's `make_li_plot`.
- `plot_first-level_outputs.ipynb` — quick visualization of individual first-level `sound`
  contrast maps, ahead of/alongside the full group-level notebook.

## `multivariate_fmri/`: multivariate / RSA pipeline

A separate track from the univariate pipeline above, modeling single-trial (rather than
condition-averaged) responses. The active pipeline is
[GLMsingle](https://glmsingle.readthedocs.io/)-based, run in this order:

### 1. Single-trial beta estimation and ROI masking (per subject)

- `GLMsingle_first-level.py` + `run_GLMsingle_first-level.sh` / `loop_run_GLMsingle_first-level.sh`
  — GLMsingle single-trial beta estimation across all `badaga` runs (`sbatch
  run_GLMsingle_first-level.sh SSP009`, or `bash loop_run_GLMsingle_first-level.sh` for the full
  cohort). `GLMsingle.ipynb` is the interactive/exploratory counterpart.
- `GLMsingle_mask-betas.py` + `run_GLMsingle_mask-betas.sh` / `loop_run_GLMsingle_mask-betas.sh` —
  masks every trial's beta map into the same 20-ROI cortical atlas the univariate ROI pipeline
  uses (14 auditory + 6 extended language-network ROIs), once per subject regardless of any
  downstream condition subsetting.
- `GLMsingle_split-betas.py` + `run_GLMsingle_split-betas.sh` / `loop_run_GLMsingle_split-betas.sh`
  — splits each subject's masked betas into per-condition, per-ROI CSVs
  (`stim-<syllable>_<speaker>_<noise_level>_rep-<n>_mask-<ROI>.csv`) for RSA.

### 2. RSA (per subject, then group level)

- `GLMsingle_rsa-roi.py` + `run_GLMsingle_rsa-roi.sh` / `loop_run_GLMsingle_rsa-roi.sh` — computes
  a crossnobis (or euclidean/correlation) RDM per ROI from one subject's masked single-trial
  betas, restricted to one noise level at a time (`sbatch run_GLMsingle_rsa-roi.sh SSP009 Q`; the
  loop script submits one job per subject &times; noise level for all 5 levels: `Q, 8, 0, n2, n6`).
- `GLMsingle_rsa-group.ipynb` — group-level RSA (CWS vs. CWNS), run separately per noise level:
  correlates each subject's empirical RDM against categorical model RDMs (syllable identity,
  speaker identity) and acoustic model RDMs (Praat-derived speech-metric features, reused across
  noise levels since noise doesn't change the underlying recording's acoustics), computes noise
  ceiling (Nili et al. 2014) and subject-level bootstrap statistics with FDR correction, then
  combines results across noise levels for a linear noise-level trend test.

### Superseded

- `multivariate_first-level.py`/`.ipynb` + `run_multivariate_first-level.sh` /
  `loop_run_multivariate_first-level.sh` — an earlier per-run/per-trial (LSA/LSS) first-level
  approach, superseded by the GLMsingle pipeline above.
- `rsa_searchlight.py` + `run_rsa_searchlight.sh` / `loop_run_rsa_searchlight.sh` and
  `group_level_rsa_searchlight_WIP.ipynb` — an earlier searchlight RSA approach (over the LSA/LSS
  betas above), superseded by the ROI-based `GLMsingle_rsa-roi.py`/`GLMsingle_rsa-group.ipynb`
  pipeline. Left in place but not actively maintained; worth deprecating explicitly.

`roi_surface_plotting.py` is a shared helper module (used by both `group_level_all_ROI.ipynb` and
`GLMsingle_rsa-group.ipynb`) that projects per-ROI group statistics onto the fsaverage cortical
surface for lateral-view figures.

## `report/`: status report generator

`generate_report.py` builds a self-contained HTML status report
(`report/SSP_status_report.html`) from the cached CSVs/JSON/PNGs the notebooks above already
write to disk — no nilearn/rsatoolbox/GLMsingle required to run it, so it can run anywhere those
cached outputs have been copied to. Covers: a per-subject attrition ledger (why is a given
subject missing from univariate vs. RSA, built from `motion_qc.csv` /
`glmsingle_info.json` / RDM presence, not a hardcoded exclusion list), descriptive stats,
whole-brain and ROI results (including the linear SNR-trend and laterality-index analyses) and
RSA results (model RDMs, model-fit boxplots, noise ceiling, cross-noise-level trend), split out
by CWNS/CWS/between-group. Run with `python report/generate_report.py` after a fresh cluster run
to refresh it.

## Open items before publication

A few things flagged inline in the code as needing PI/team confirmation before relying on the
results:
- Whether `participants.tsv` actually stores the `group` column as lowercase `'cws'` (the code
  now normalizes case either way, but the *source* casing should still be confirmed).
- The FD/DVARS motion-scrubbing thresholds (`'scrubbing'` denoise strategy defaults), given this
  is a pediatric/stuttering population where task-related motion may differ from adult norms.
- The `sample_mask`/`sample_masks` keyword name in `nilearn_glm_across_runs()`, which changed
  between nilearn versions — confirm against the version installed on the cluster.
- See `report/SSP_status_report.html`'s own "Open items before publication" section (regenerate
  it for the current list) for analysis-level caveats — e.g. RSA sample size for CWS, and which
  findings are well-powered enough to build on.
