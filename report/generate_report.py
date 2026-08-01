"""Generate a self-contained HTML status report for the SSP CWS/CWNS speech-in-noise study.

Produces report/SSP_status_report.html -- all figures are either base64-embedded PNGs already
saved by univariate_fmri/group_level_all_ROI.ipynb, univariate_fmri/univariate_group-level.ipynb
(whole-brain mosaics), or multivariate_fmri/GLMsingle_rsa-group.ipynb, or small summary charts
built fresh here from the numeric CSVs/JSON those notebooks/scripts cache. No external CSS/JS
dependencies, no nilearn/rsatoolbox required to run this script itself -- only pandas/numpy/
matplotlib, so it can run anywhere the cached CSVs/JSON/PNGs have been copied to (a laptop, not
necessarily the cluster), unlike the notebooks themselves.

Styling is deliberately plain (system sans-serif, white background, no dark-mode/CSS-variable
machinery) -- modeled on ~/software/acfMRI/07_report/generate_report.py, not on a "designed"
look.

In addition to the group-level statistical results, this report builds a per-subject attrition
ledger from three things already written to disk by the existing pipeline (no new
instrumentation needed):
  - L1_DIR/sub-*/sub-*_motion_qc.csv                    (univariate_first-level.py)
  - GLMSINGLE_DIR/sub-*/sub-*_glmsingle_info.json        (GLMsingle_first-level.py)
  - RDM_DIR/sub-*_glmsingle_cortical_..._rdms.hdf5       (GLMsingle_rsa-roi.py)
plus a raw BIDS func/ listing (to count badaga runs actually on disk) and the hardcoded
IGNORE_SUBS_REASONS below -- a manual-override slot for one-off exclusions a file-existence
check can't catch, NOT a mirror of anything in the notebooks: group_level_all_ROI.ipynb and
univariate_group-level.ipynb no longer keep static `ignore_subs` lists at all -- both just try
to load each participant's derivatives and skip gracefully if they're missing. This is what
answers "why is this subject missing": no
first-level output yet, high motion, or -- for RSA specifically -- only one badaga run on disk
(crossnobis needs a condition to repeat *across* runs, not just within one; GLMsingle silently
downgrades single-run subjects to the TYPEB estimate, which is never used for RSA).

Usage:
    python report/generate_report.py

Requires the univariate and RSA group-level notebooks to have been run at least once (so the
CSVs/PNGs/JSON referenced below exist under BIDSROOT). Missing files are skipped with a printed
note, not a hard failure -- the report still builds with whatever is available.
"""

import base64
import io
import json
from datetime import date
from glob import glob
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────────
REPORT_DIR = Path(__file__).resolve().parent
BIDSROOT = Path('/bgfs/bchandrasekaran/krs228/data/SSP/data_bids')
NILEARN_DIR = BIDSROOT / 'derivatives' / 'nilearn'
GLMSINGLE_DIR = BIDSROOT / 'derivatives' / 'glmsingle'

FWHM = 6.00
GROUP_OUT_DIR = NILEARN_DIR / f'group_fwhm-{FWHM:.2f}'            # ROI stats (group_level_all_ROI.ipynb)
WHOLE_BRAIN_DIR = NILEARN_DIR / 'group_run-all'                    # whole-brain mosaic PNGs (univariate_group-level.ipynb)
L1_DIR = NILEARN_DIR / 'run-all_contrast-snr'                      # per-subject first-level statmaps + motion_qc.csv

RDM_METHOD = 'crossnobis'
RSA_NOISE_LEVEL_TAG = 'Q'  # matches GLMsingle_rsa-group.ipynb's NOISE_LEVEL_TAG for the acoustic-model run
RSA_OUT_DIR = GLMSINGLE_DIR / 'rsa-group_glmsingle'
RDM_DIR = GLMSINGLE_DIR / f'rsa-roi_glmsingle_rdmcalc-{RDM_METHOD}'

PARTICIPANTS_FPATH = BIDSROOT / 'participants.tsv'

CONTRAST_LIST = ['q', '8', '0', 'n2', 'n6']
WHOLE_BRAIN_CONTRASTS = ['q', '8', '0', 'n2', 'n6', 'qMinusN6', 'qMinus0', 'sound', 'response']
# representative subset actually embedded as brain images below (all 9 x 3-group would be 27
# large mosaic PNGs saved at dpi=1000 -- too much for one report). Change freely.
BRAIN_MAP_CONTRASTS = ['q', 'qMinusN6', 'sound']

# same 20-ROI cortical list used throughout the pipeline (GLMsingle_mask-betas.py,
# GLMsingle_rsa-roi.py, GLMsingle_rsa-group.ipynb, group_level_all_ROI.ipynb)
CORTICAL_ROI_LIST = [
    'L-HG', 'L-PT', 'L-PP', 'L-STGp', 'L-STGa', 'L-ParsOp', 'L-ParsTri', 'L-SMGa', 'L-SMGp', 'L-Ang',
    'R-HG', 'R-PT', 'R-PP', 'R-STGp', 'R-STGa', 'R-ParsOp', 'R-ParsTri', 'R-SMGa', 'R-SMGp', 'R-Ang',
]

# same Okabe-Ito-derived colors already used in the notebooks' own figures (plot_roi_box_strip /
# plot_roi_box_strip_by_hemisphere in both group_level_all_ROI.ipynb and GLMsingle_rsa-group.ipynb)
# -- reused here so this report's charts read consistently with the figures it embeds.
GROUP_COLOR = {'CWNS': '#009E73', 'CWS': '#CC79A7'}
HEMI_COLOR = {'L': '#0072B2', 'R': '#D55E00'}
ACCENT = '#9C7A1B'
MOTION_COLOR = '#D55E00'

# Subjects dropped before the pipeline even runs. Both group-level notebooks
# (group_level_all_ROI.ipynb for ROI + RSA eligibility below, univariate_group-level.ipynb for
# the whole-brain design matrix) no longer hardcode a static `ignore_subs` list at all -- each
# just tries to load a given participant's derivatives and skips gracefully (via
# build_statmap_dict/mask_stat_maps and prepare_group_inputs' own per-file/per-covariate checks)
# if they're missing, so there's no separate exclusion list for the two notebooks to disagree
# on anymore. IGNORE_SUBS_ROI_RSA below is purely a manual-override slot for this report's own
# ledger -- a place for a future one-off, data-quality-driven exclusion that a file-existence
# check structurally can't catch (e.g. a participant who left early but still has some
# real-but-bad data on disk) -- not a mirror of anything in the notebooks.
IGNORE_SUBS_ROI_RSA = {}

IGNORE_SUBS_REASONS = IGNORE_SUBS_ROI_RSA


# ── small helpers ──────────────────────────────────────────────────────────────
def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return 'data:image/png;base64,' + base64.b64encode(buf.read()).decode()


def img_from_file(path):
    path = Path(path)
    if not path.exists():
        print(f'  [missing] {path}')
        return None
    return 'data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode()


def img_from_file_resized(path, max_width_px=1600, dpi=150):
    """Like img_from_file, but downsamples large source PNGs before embedding. The whole-brain
    mosaic PNGs this report embeds are saved by univariate_group-level.ipynb's
    plot_mosaic_with_contours at dpi=1000 (12700x5000px each) -- far more resolution than a
    browser displays useful pixels for, and embedding 7-9 of them raw was ballooning the report
    to 15+ MB. Re-rendered through matplotlib at a screen-appropriate width instead.
    """
    path = Path(path)
    if not path.exists():
        print(f'  [missing] {path}')
        return None
    img_arr = plt.imread(path)
    h, w = img_arr.shape[:2]
    if w <= max_width_px:
        return img_from_file(path)
    fig_w_in = max_width_px / dpi
    fig_h_in = fig_w_in * (h / w)
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img_arr)
    ax.axis('off')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return 'data:image/png;base64,' + base64.b64encode(buf.read()).decode()


def read_csv_safe(path, **kwargs):
    path = Path(path)
    if not path.exists():
        print(f'  [missing] {path}')
        return None
    return pd.read_csv(path, **kwargs)


def read_pickle_safe(path):
    path = Path(path)
    if not path.exists():
        print(f'  [missing] {path}')
        return None
    return pd.read_pickle(path)


def fmt_p(p):
    if p is None or p != p:
        return 'n/a'
    return 'p < .001' if p < 0.001 else f'p = {p:.3f}'.replace('0.', '.')


def _img_tag(b64, caption='', width='100%'):
    if not b64:
        return '<p class="missing">[figure not available -- run the source notebook first]</p>'
    cap = f'<figcaption>{caption}</figcaption>' if caption else ''
    return f'<figure><img src="{b64}" style="width:{width};max-width:100%;">{cap}</figure>'

def _brain_row(caption_prefix, contrast, group_tag):
    b64 = img_from_file_resized(WHOLE_BRAIN_DIR / f'group-{group_tag}_contrast-{contrast}_view-mosaic.png')
    return _img_tag(b64, f'{caption_prefix} -- contrast-{contrast}, FDR cluster-corrected.', width='100%')


def _rsa_fig(filename, caption, width='100%'):
    """RSA box+strip PNGs saved by GLMsingle_rsa-group.ipynb -- unlike the whole-brain mosaics,
    these are already a screen-appropriate size (dpi=300, ~12x4in max), so no img_from_file_resized
    downsampling needed.
    """
    b64 = img_from_file(RSA_OUT_DIR / filename)
    return _img_tag(b64, caption, width=width)


def _roi_fig(filename, caption, width='100%'):
    """ROI-level SNR-trend / laterality-index PNGs saved by group_level_all_ROI.ipynb --
    same screen-appropriate sizing as _rsa_fig, just from GROUP_OUT_DIR instead of RSA_OUT_DIR.
    """
    b64 = img_from_file(GROUP_OUT_DIR / filename)
    return _img_tag(b64, caption, width=width)


def _df_to_table_html(df, css_class='data-table'):
    if df is None or len(df) == 0:
        return '<p class="missing">[no data]</p>'
    return df.to_html(classes=css_class, index=False, border=0, na_rep='—')


# ── data loaders ───────────────────────────────────────────────────────────────
def load_participants():
    df = read_csv_safe(PARTICIPANTS_FPATH, sep='\t')
    if df is None:
        return None
    df['group_norm'] = df['group'].str.strip().str.lower()
    return df


def load_roi_long():
    return read_pickle_safe(GROUP_OUT_DIR / 'roi_df_long.pkl')


def load_within_group_stats():
    return read_csv_safe(GROUP_OUT_DIR / 'within_group_stats.csv')


def load_between_group_stats():
    return read_csv_safe(GROUP_OUT_DIR / 'between_group_stats.csv')


def load_anova_results():
    return read_csv_safe(GROUP_OUT_DIR / 'anova_results.csv')


def load_rsa_model_fit():
    return read_csv_safe(RSA_OUT_DIR / f'model_fit_scalars_noiselevel-{RSA_NOISE_LEVEL_TAG}.csv')


def load_rsa_group_comparison():
    return read_csv_safe(RSA_OUT_DIR / f'group_comparison_bootstrap_noiselevel-{RSA_NOISE_LEVEL_TAG}.csv')


def load_rsa_within_group():
    return read_csv_safe(RSA_OUT_DIR / f'within_group_significance_noiselevel-{RSA_NOISE_LEVEL_TAG}.csv')


def load_rsa_noise_ceiling():
    return read_csv_safe(RSA_OUT_DIR / f'noise_ceiling_noiselevel-{RSA_NOISE_LEVEL_TAG}.csv')


# ── attrition ledger: per-subject, why are we losing them? ─────────────────────
def count_raw_badaga_runs(sub_id):
    """How many badaga BOLD runs exist on disk for this subject, before any motion-based
    dropping. Comparing this to glmsingle_info.json's n_runs_used (post motion-drop) is what
    distinguishes "only 1 run was ever collected" from "2+ runs were collected but 1 got dropped
    for motion" -- glmsingle_info.json alone can't tell these apart, since n_runs_used is already
    post-drop.
    """
    return len(glob(str(BIDSROOT / sub_id / 'func' / f'{sub_id}_task-badaga_run-*_bold.nii.gz')))


def _univariate_status(sub_id):
    """Returns (n_contrasts_found, mean_fd, status_label, reason) for one subject's univariate
    first-level stage, mirroring group_level_all_ROI.ipynb's build_statmap_dict glob pattern
    exactly so "contrasts found" here means the same thing it does in the ROI stage.
    """
    n_contrasts = sum(
        1 for c in CONTRAST_LIST
        if glob(str(L1_DIR / sub_id / f'*contrast-{c}_stat-effect_statmap.nii.gz'))
    )
    missing_contrasts = [c for c in CONTRAST_LIST
                         if not glob(str(L1_DIR / sub_id / f'*contrast-{c}_stat-effect_statmap.nii.gz'))]

    mfd_fpath = L1_DIR / sub_id / f'{sub_id}_motion_qc.csv'
    mean_fd = None
    if mfd_fpath.exists():
        mfd_df = pd.read_csv(mfd_fpath)
        if len(mfd_df) > 0 and 'mean_fd' in mfd_df.columns:
            mean_fd = float(mfd_df['mean_fd'].iloc[0])

    if n_contrasts == len(CONTRAST_LIST):
        return n_contrasts, mean_fd, 'complete', f'{n_contrasts}/{len(CONTRAST_LIST)} SNR contrasts present.'
    if n_contrasts == 0 and mean_fd is None:
        return n_contrasts, mean_fd, 'no output', ('No first-level output found at all (no statmaps, '
                                                    'no motion_qc.csv) -- not yet run, or crashed before '
                                                    'saving anything. Check SLURM logs.')
    if n_contrasts == 0 and mean_fd is not None and mean_fd > 0.9:
        return n_contrasts, mean_fd, 'likely motion', (
            f'0/{len(CONTRAST_LIST)} contrasts, mean FD = {mean_fd:.2f}mm (> the 0.9mm scrubbing '
            'threshold used by univariate_first-level.py). Consistent with -- but not direct proof '
            'of -- every run failing the >50% volumes-retained-after-scrubbing rule; the per-run '
            'retained fraction itself isn\'t cached anywhere, only this subject-level mean FD.')
    if n_contrasts == 0:
        return n_contrasts, mean_fd, 'unexplained', (
            f'0/{len(CONTRAST_LIST)} contrasts despite motion_qc.csv existing (mean FD = '
            f'{mean_fd:.2f}mm, below the 0.9mm scrubbing threshold) -- not a motion story; '
            'check SLURM logs for this subject.')
    return n_contrasts, mean_fd, 'partial', (
        f'{n_contrasts}/{len(CONTRAST_LIST)} SNR contrasts present; missing: {", ".join(missing_contrasts)}.')


def _rsa_status(sub_id):
    """Returns (n_runs_raw, n_runs_used, has_cross_run_repeats, final_type, mean_fd, has_rdm,
    status_label, reason) for one subject's RSA/crossnobis eligibility.
    """
    n_runs_raw = count_raw_badaga_runs(sub_id)

    info_fpath = GLMSINGLE_DIR / sub_id / f'{sub_id}_glmsingle_info.json'
    n_runs_used = has_cross_run_repeats = final_type = None
    if info_fpath.exists():
        info = json.loads(info_fpath.read_text())
        n_runs_used = info.get('n_runs_used')
        has_cross_run_repeats = info.get('has_cross_run_repeats')
        final_type = info.get('final_type')

    mfd_fpath = GLMSINGLE_DIR / sub_id / f'{sub_id}_motion_qc.csv'
    mean_fd = None
    if mfd_fpath.exists():
        mfd_df = pd.read_csv(mfd_fpath)
        if len(mfd_df) > 0 and 'mean_fd' in mfd_df.columns:
            mean_fd = float(mfd_df['mean_fd'].iloc[0])

    has_rdm = bool(glob(str(
        RDM_DIR / f'{sub_id}_glmsingle_cortical_{RDM_METHOD}_noiselevel-{RSA_NOISE_LEVEL_TAG}_rdms.hdf5'
    )))

    if has_rdm:
        return (n_runs_raw, n_runs_used, has_cross_run_repeats, final_type, mean_fd, True,
               'RSA-eligible', 'Has a crossnobis RDM -- contributes to RSA group analysis.')
    if n_runs_raw == 0:
        return (n_runs_raw, n_runs_used, has_cross_run_repeats, final_type, mean_fd, False,
               'no badaga data', 'No badaga BOLD files found on disk at all -- a missing-file issue, not motion.')
    if info_fpath.exists() is False:
        return (n_runs_raw, n_runs_used, has_cross_run_repeats, final_type, mean_fd, False,
               'GLMsingle not run', f'{n_runs_raw} badaga run(s) on disk, but GLMsingle_first-level.py hasn\'t been run for this subject yet.')
    if has_cross_run_repeats is False and n_runs_raw <= 1:
        return (n_runs_raw, n_runs_used, has_cross_run_repeats, final_type, mean_fd, False,
               'single run (file issue)', (
                   f'Only {n_runs_raw} badaga run on disk -- crossnobis needs a condition to repeat '
                   'ACROSS runs, so GLMsingle falls back to the degraded TYPEB estimate, which is '
                   'never used for RSA. This is a missing-data issue, not motion.'))
    if has_cross_run_repeats is False and n_runs_raw >= 2:
        return (n_runs_raw, n_runs_used, has_cross_run_repeats, final_type, mean_fd, False,
               'run dropped for motion', (
                   f'{n_runs_raw} badaga run(s) were on disk, but only {n_runs_used} survived '
                   f'GLMsingle\'s motion QC (mean FD > 2.0mm drops a run) -- {"mean FD (kept runs) = %.2fmm. " % mean_fd if mean_fd is not None else ""}'
                   'With only 1 usable run left, no condition can repeat across runs, so GLMsingle '
                   'falls back to TYPEB and this subject is excluded from RSA. Here, motion IS the reason.'))
    if final_type == 'TYPED_FITHRF_GLMDENOISE_RR' and not has_rdm:
        return (n_runs_raw, n_runs_used, has_cross_run_repeats, final_type, mean_fd, False,
               'RDM not computed', 'GLMsingle produced usable TYPED betas, but GLMsingle_rsa-roi.py hasn\'t been (re-)run for this subject yet.')
    return (n_runs_raw, n_runs_used, has_cross_run_repeats, final_type, mean_fd, False,
           'unclassified', 'Doesn\'t match a known pattern -- inspect this subject\'s glmsingle_info.json directly.')


def build_attrition_ledger(participants_df):
    """One row per enrolled participant (including pre-pipeline-excluded ones), with exactly
    why they have/haven't reached the univariate and RSA stages. This is what backs the
    "where are we losing participants" section -- every reason string here traces back to a
    file already on disk, not a guess (except where explicitly flagged as inferred).
    """
    if participants_df is None:
        return None
    rows = []
    for _, prow in participants_df.iterrows():
        sub_id = prow['participant_id']
        group = 'CWNS' if prow['group_norm'] == 'control' else 'CWS' if prow['group_norm'] == 'cws' else prow['group_norm']

        if sub_id in IGNORE_SUBS_REASONS:
            rows.append({
                'participant_id': sub_id, 'group': group, 'pre_pipeline_excluded': True,
                'pre_pipeline_reason': IGNORE_SUBS_REASONS[sub_id],
                'n_snr_contrasts': 0, 'mean_fd_univariate': None,
                'univariate_status': 'excluded pre-pipeline', 'univariate_reason': IGNORE_SUBS_REASONS[sub_id],
                'n_badaga_runs_raw': None, 'n_runs_used_glmsingle': None, 'has_cross_run_repeats': None,
                'mean_fd_glmsingle': None, 'has_rdm': False,
                'rsa_status': 'excluded pre-pipeline', 'rsa_reason': IGNORE_SUBS_REASONS[sub_id],
            })
            continue

        n_contrasts, mfd_uni, uni_status, uni_reason = _univariate_status(sub_id)
        (n_runs_raw, n_runs_used, has_cross_run_repeats, final_type, mfd_glms, has_rdm,
        rsa_status, rsa_reason) = _rsa_status(sub_id)

        rows.append({
            'participant_id': sub_id, 'group': group, 'pre_pipeline_excluded': False,
            'pre_pipeline_reason': None,
            'n_snr_contrasts': n_contrasts, 'mean_fd_univariate': mfd_uni,
            'univariate_status': uni_status, 'univariate_reason': uni_reason,
            'n_badaga_runs_raw': n_runs_raw, 'n_runs_used_glmsingle': n_runs_used,
            'has_cross_run_repeats': has_cross_run_repeats, 'mean_fd_glmsingle': mfd_glms,
            'has_rdm': has_rdm, 'rsa_status': rsa_status, 'rsa_reason': rsa_reason,
        })
    return pd.DataFrame(rows)


# ── charts built fresh from cached CSVs / the attrition ledger ─────────────────
# ordered (not qualitative) palette for pipeline stages -- Enrolled -> ROI -> RSA is a real
# funnel, each stage a strict subset of the previous, so a light-to-dark ramp reads as
# "progressively further along," distinct from the CWNS/CWS group colors used everywhere else
# in this report.
STAGE_COLORS = ['#9ECAE1', '#4292C6', '#08519C']


def make_attrition_chart(participants_df, roi_long_df, rsa_model_fit_df):
    """Grouped bar chart: N subjects per group (CWNS, CWS), with one bar per pipeline stage --
    grouped by pipeline stage (not by group), so each group's stage-by-stage drop-off reads as
    three adjacent, progressively shorter bars in one place, rather than being split across
    separate per-stage x-tick clusters.
    """
    stages = []
    if participants_df is not None:
        counts = participants_df.group_norm.value_counts()
        stages.append(('Enrolled', counts.get('control', 0), counts.get('cws', 0)))
    if roi_long_df is not None:
        n = roi_long_df.drop_duplicates('participant_id')[['participant_id', 'group']]
        counts = n.group.value_counts()
        stages.append(('ROI univariate', counts.get('CWNS', 0), counts.get('CWS', 0)))
    if rsa_model_fit_df is not None:
        n = rsa_model_fit_df.drop_duplicates('participant_id')[['participant_id', 'group']]
        counts = n.group.value_counts()
        stages.append(('RSA (cross-run\nrepeats)', counts.get('CWNS', 0), counts.get('CWS', 0)))

    if not stages:
        return None

    groups = ['CWNS', 'CWS']
    x = np.arange(len(groups))
    n_stages = len(stages)
    width = 0.8 / n_stages

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=150)
    for i, (stage_label, cwns_n, cws_n) in enumerate(stages):
        offset = (i - (n_stages - 1) / 2) * width
        vals = [cwns_n, cws_n]
        xi_positions = x + offset
        ax.bar(xi_positions, vals, width, label=stage_label, color=STAGE_COLORS[i % len(STAGE_COLORS)])
        for xi, n in zip(xi_positions, vals):
            ax.text(xi, n + 0.5, str(int(n)), ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10)
    ax.set_ylabel('N subjects')
    ax.set_title('Sample size by group, across pipeline stages')
    ax.legend(frameon=False, fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    return fig_to_b64(fig)


def make_rsa_ineligibility_chart(ledger_df):
    """The chart that directly answers "why isn't this subject in RSA": count of subjects (by
    group) in each RSA-status category, restricted to subjects who made it past pre-pipeline
    exclusion. Distinguishes the crossnobis/single-run case from the motion-dropped-a-run case
    from a still-pending-processing case.
    """
    if ledger_df is None:
        return None
    df = ledger_df[~ledger_df.pre_pipeline_excluded]
    if len(df) == 0:
        return None

    label_map = {
        'RSA-eligible': 'RSA-eligible',
        'single run (file issue)': 'Only 1 badaga run\n(crossnobis needs cross-run repeats)',
        'run dropped for motion': 'Run dropped for motion\n(2+ runs -> 1 usable)',
        'GLMsingle not run': 'GLMsingle not\nrun yet',
        'RDM not computed': 'TYPED betas ready,\nRDM not computed yet',
        'no badaga data': 'No badaga BOLD\nfound on disk',
        'unclassified': 'Unclassified',
    }
    order = list(label_map.keys())
    counts = df.groupby(['rsa_status', 'group']).size().unstack(fill_value=0)
    for g in ('CWNS', 'CWS'):
        if g not in counts.columns:
            counts[g] = 0
    counts = counts.reindex(order).fillna(0)

    labels = [label_map[s] for s in counts.index]
    x = np.arange(len(labels))
    width = 0.32

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5), dpi=150)
    ax.bar(x - width / 2, counts['CWNS'], width, label='CWNS', color=GROUP_COLOR['CWNS'])
    ax.bar(x + width / 2, counts['CWS'], width, label='CWS', color=GROUP_COLOR['CWS'])
    for xi, n in zip(x - width / 2, counts['CWNS']):
        if n > 0:
            ax.text(xi, n + 0.3, str(int(n)), ha='center', va='bottom', fontsize=8)
    for xi, n in zip(x + width / 2, counts['CWS']):
        if n > 0:
            ax.text(xi, n + 0.3, str(int(n)), ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('N subjects')
    ax.set_title('Why subjects are (not) RSA-eligible')
    ax.legend(frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    return fig_to_b64(fig)


def make_roi_hit_count_chart(within_group_stats_df):
    """Bar chart: count of FDR-significant ROI x SNR tests per group, out of 100 possible each
    (20 ROIs x 5 SNR levels) -- the headline power comparison between CWNS and CWS.
    """
    if within_group_stats_df is None:
        return None
    sig = within_group_stats_df[within_group_stats_df.p_fdr < 0.05]
    counts = sig.groupby('group').size()
    groups = ['CWNS', 'CWS']
    n_sig = [counts.get(g, 0) for g in groups]
    n_total = 100  # 20 ROIs x 5 SNR levels, per group

    fig, ax = plt.subplots(1, 1, figsize=(3.6, 4), dpi=150)
    bars = ax.bar(groups, n_sig, color=[GROUP_COLOR[g] for g in groups], width=0.55)
    for bar, n in zip(bars, n_sig):
        ax.text(bar.get_x() + bar.get_width() / 2, n + 0.5, f'{n}/{n_total}',
                ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('FDR-significant ROI x SNR tests')
    ax.set_title('Within-group ROI effects\n(beta != 0, p_FDR < .05)')
    ax.set_ylim(0, n_total * 0.5)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    return fig_to_b64(fig)


def make_roi_direction_table(within_group_stats_df, group_name):
    """Per-region summary of FDR-significant hits for one group: how many of the 5 SNR levels
    are significant, in which direction (sign of mean_beta), and the mean |beta| among those
    significant hits (effect-size context beyond the raw significance count).
    """
    if within_group_stats_df is None:
        return None
    sig = within_group_stats_df[
        (within_group_stats_df.group == group_name) & (within_group_stats_df.p_fdr < 0.05)
    ].copy()
    if len(sig) == 0:
        return pd.DataFrame(columns=['ROI', 'SNR levels significant', 'direction', 'mean |beta|', 'at'])

    sig['direction'] = np.where(sig.mean_beta > 0, 'positive', 'negative')
    rows = []
    for region_hemi, group_df in sig.groupby('region_hemi'):
        directions = group_df['direction'].unique()
        direction_label = directions[0] if len(directions) == 1 else 'mixed'
        rows.append({
            'ROI': region_hemi,
            'SNR levels significant': len(group_df),
            'direction': direction_label,
            'mean |beta|': group_df['mean_beta'].abs().mean(),
            'at': ', '.join(sorted(group_df['SNR'].astype(str))),
        })
    out = pd.DataFrame(rows).sort_values('SNR levels significant', ascending=False).reset_index(drop=True)
    out['mean |beta|'] = out['mean |beta|'].map(lambda x: f'{x:.3f}')
    return out


def make_descriptive_stats_table(participants_df, ledger_df):
    """Basic per-group descriptive statistics (N, age, sex, motion) -- the "more statistics"
    context that was previously only implicit in the enrolled-N stat tiles.
    """
    if participants_df is None:
        return None
    rows = []
    for group_label, norm in [('CWNS', 'control'), ('CWS', 'cws')]:
        gdf = participants_df[participants_df.group_norm == norm]
        row = {'group': group_label, 'N enrolled': len(gdf)}
        if 'age' in gdf.columns:
            row['age, mean (SD)'] = f'{gdf.age.mean():.1f} ({gdf.age.std():.1f})'
        if 'sex' in gdf.columns:
            n_f = int((gdf.sex == 'F').sum())
            row['sex (F/M)'] = f'{n_f}/{len(gdf) - n_f}'
        if ledger_df is not None:
            gl = ledger_df[(ledger_df.group == group_label) & (~ledger_df.pre_pipeline_excluded)]
            mfd = gl['mean_fd_univariate'].dropna()
            if len(mfd) > 0:
                row['mean FD (univariate), mean (SD)'] = f'{mfd.mean():.2f}mm ({mfd.std():.2f})'
            row['N with all 5 SNR contrasts'] = int((gl.n_snr_contrasts == len(CONTRAST_LIST)).sum())
            row['N RSA-eligible'] = int((gl.rsa_status == 'RSA-eligible').sum())
        rows.append(row)
    return pd.DataFrame(rows)


def make_rsa_hits_table(df, kind):
    """Full table of FDR-significant ROI x model hits, sorted by p_fdr -- not just the single
    strongest hit quoted in prose. `kind` is 'within' (within_group df, has a `group` column) or
    'between' (group_comparison df).
    """
    if df is None:
        return None
    sig = df[df.p_fdr < 0.05].copy()
    if len(sig) == 0:
        return pd.DataFrame(columns=['ROI', 'model'] + (['group'] if kind == 'within' else ['observed_diff']) + ['p_fdr'])
    sig = sig.sort_values('p_fdr')
    sig['p_fdr'] = sig['p_fdr'].map(lambda x: f'{x:.4f}')
    cols = ['ROI', 'model']
    if kind == 'within':
        cols += ['group']
    else:
        if 'observed_diff' in sig.columns:
            sig['observed_diff'] = sig['observed_diff'].map(lambda x: f'{x:.3f}')
            cols += ['observed_diff']
    cols += ['p_fdr']
    return sig[[c for c in cols if c in sig.columns]].reset_index(drop=True)


def make_noise_ceiling_summary(noise_ceiling_df):
    if noise_ceiling_df is None or len(noise_ceiling_df) == 0:
        return None
    return noise_ceiling_df.groupby('group')[['ceiling_lower', 'ceiling_upper']].mean().reset_index().round(3)


def make_pre_pipeline_table(ledger_df):
    if ledger_df is None:
        return None
    df = ledger_df[ledger_df.pre_pipeline_excluded][['participant_id', 'group', 'pre_pipeline_reason']]
    df = df.rename(columns={'pre_pipeline_reason': 'reason'})
    return df.sort_values('participant_id').reset_index(drop=True)


def make_univariate_incomplete_table(ledger_df):
    if ledger_df is None:
        return None
    df = ledger_df[(~ledger_df.pre_pipeline_excluded) & (ledger_df.univariate_status != 'complete')]
    df = df[['participant_id', 'group', 'n_snr_contrasts', 'mean_fd_univariate', 'univariate_status', 'univariate_reason']]
    df = df.rename(columns={'n_snr_contrasts': 'contrasts found', 'mean_fd_univariate': 'mean FD',
                           'univariate_status': 'status', 'univariate_reason': 'reason'})
    df['mean FD'] = df['mean FD'].map(lambda x: f'{x:.2f}mm' if pd.notna(x) else '—')
    return df.sort_values(['status', 'participant_id']).reset_index(drop=True)


def make_rsa_ineligible_table(ledger_df):
    if ledger_df is None:
        return None
    df = ledger_df[(~ledger_df.pre_pipeline_excluded) & (ledger_df.rsa_status != 'RSA-eligible')]
    df = df[['participant_id', 'group', 'n_badaga_runs_raw', 'n_runs_used_glmsingle', 'rsa_status', 'rsa_reason']]
    df = df.rename(columns={'n_badaga_runs_raw': 'badaga runs on disk', 'n_runs_used_glmsingle': 'runs used by GLMsingle',
                           'rsa_status': 'status', 'rsa_reason': 'reason'})
    return df.sort_values(['status', 'participant_id']).reset_index(drop=True)


# ── text summaries ───────────────────────────────────────────────────────────────
def summarize_whole_brain(label, finite_contrasts, empty_contrasts):
    n_finite = len(finite_contrasts)
    n_total = n_finite + len(empty_contrasts)
    if n_finite == n_total:
        return (f'<b>{label}:</b> all {n_total} contrasts show a finite FDR-corrected cluster-level '
               'threshold (real, above-threshold activation).')
    if n_finite == 0:
        return f'<b>{label}:</b> no contrast shows any FDR-corrected significant cluster (empty for all {n_total}).'
    return (f'<b>{label}:</b> {n_finite}/{n_total} contrasts show a finite FDR-corrected threshold '
           f'({", ".join(finite_contrasts)}); empty for {", ".join(empty_contrasts)}.')


# ── HTML assembly ──────────────────────────────────────────────────────────────
CSS = '''
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  font-size: 15px; line-height: 1.6; color: #222; background: #fff;
  max-width: 1000px; margin: 0 auto; padding: 2.5rem 1.5rem 4rem;
}
h1 { font-size: 1.9rem; margin-bottom: 0.3rem; color: #1a1a1a; }
h2 { font-size: 1.35rem; margin: 2.4rem 0 0.9rem; border-bottom: 2px solid #ddd; padding-bottom: 5px; color: #222; }
h3 { font-size: 1.05rem; margin: 1.5rem 0 0.5rem; color: #444; }
p { margin-bottom: 0.85rem; max-width: 72ch; }
.dek { color: #555; font-size: 1.05rem; margin-bottom: 1rem; max-width: 68ch; }
.meta { color: #888; font-size: 0.85rem; margin-bottom: 1.8rem; }
code { font-family: ui-monospace, "SF Mono", "Roboto Mono", Consolas, monospace;
  background: #f0f0f0; padding: 0.1em 0.35em; border-radius: 3px; font-size: 0.9em; }

.stat-row { display: flex; flex-wrap: wrap; gap: 0.9rem; margin: 1.2rem 0 2rem; }
.stat-tile { background: #f7f7f7; border: 1px solid #ddd; border-radius: 4px;
  padding: 0.7rem 1.05rem; flex: 1 1 140px; }
.stat-tile .n { font-family: ui-monospace, "SF Mono", "Roboto Mono", monospace;
  font-variant-numeric: tabular-nums; font-size: 1.5rem; font-weight: 700; display: block; }
.stat-tile .label { font-size: 0.76rem; color: #666; }

figure { margin: 1rem 0 1.5rem; }
figcaption { font-size: 0.83rem; color: #666; margin-top: 0.35rem; font-style: italic; }
.missing { color: #a00; font-style: italic; font-size: 0.9rem; }

.summary { background: #f0f4f8; border-left: 4px solid #4477AA; padding: 0.85rem 1.1rem;
  margin: 0.7rem 0 1.6rem; border-radius: 0 4px 4px 0; font-size: 0.92rem; line-height: 1.7; }
.summary b { color: #222; }
.summary.headline { border-left-color: #9C7A1B; background: #faf6ea; }
.summary.motion { border-left-color: #D55E00; background: #fdf2ea; }

.group-block { border: 1px solid #ddd; border-radius: 4px; padding: 1.1rem 1.4rem; margin: 1rem 0 2rem; }
.group-block.cwns { border-top: 3px solid #009E73; }
.group-block.cws { border-top: 3px solid #CC79A7; }

.table-wrap { overflow-x: auto; margin: 0.8rem 0 1.5rem; }
table.data-table, table.anova-table { border-collapse: collapse; width: 100%; font-size: 0.86rem;
  font-variant-numeric: tabular-nums; }
table.data-table th, table.data-table td, table.anova-table th, table.anova-table td {
  border: 1px solid #ddd; padding: 5px 9px; text-align: left; }
table.data-table th, table.anova-table th { background: #ececec; font-weight: 600; }
table.data-table tbody tr:nth-child(even), table.anova-table tbody tr:nth-child(even) { background: #fafafa; }

.tag { display: inline-block; font-size: 0.74rem; font-weight: 600; padding: 0.08em 0.5em; border-radius: 3px; }
.tag.cwns { background: #d7f0e6; color: #00694c; }
.tag.cws { background: #f6dcea; color: #97396e; }

ul.open-items { padding-left: 1.3rem; }
ul.open-items li { margin-bottom: 0.55rem; }

footer { margin-top: 3.5rem; padding-top: 1.2rem; border-top: 1px solid #ddd; color: #888; font-size: 0.83rem; }
'''


def _stat_tile(n, label):
    return f'<div class="stat-tile"><span class="n">{n}</span><span class="label">{label}</span></div>'


def build_html(*, participants_df, ledger_df,
               attrition_b64, rsa_ineligibility_b64, hit_count_b64,
               descriptive_table_html, pre_pipeline_table_html,
               univariate_incomplete_table_html, rsa_ineligible_table_html,
               n_motion_flagged_univariate, n_motion_dropped_rsa, n_single_run_rsa,
               cwns_roi_table_html, cws_roi_table_html,
               anova_results_df, cwns_wb_summary, cws_wb_summary, diff_wb_summary,
               between_roi_summary, rsa_finding_summary,
               cwns_rsa_hits_html, cws_rsa_hits_html, between_rsa_hits_html,
               noise_ceiling_table_html,
               n_enrolled_cwns, n_enrolled_cws, n_rsa_cwns, n_rsa_cws):
    today = date.today().isoformat()

    anova_table_html = '<p class="missing">[anova_results.csv not found]</p>'
    if anova_results_df is not None:
        disp = anova_results_df.copy()
        disp['F'] = disp['F'].map(lambda x: f'{x:.2f}')
        disp['p'] = disp['p'].map(fmt_p)
        anova_table_html = _df_to_table_html(disp[['group', 'source', 'F', 'df_num', 'df_den', 'p']])

    html = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SSP status report</title>
<style>{CSS}</style>
</head>
<body>

<div>SSP &middot; Kevin Sitek &middot; internal status report</div>
<h1>Speech-in-noise fMRI: CWS/CWNS data</h1>
<p class="dek">Univariate (whole-brain + ROI) and single-trial RSA results on the <code>badaga</code>
speech-in-noise task, generated from the cached outputs of
<code>univariate_fmri/group_level_all_ROI.ipynb</code>,
<code>univariate_fmri/univariate_group-level.ipynb</code>, and
<code>multivariate_fmri/GLMsingle_rsa-group.ipynb</code>.</p>
<div class="meta">Generated {today} &middot; not a live view -- rerun this script after the next cluster run to refresh.</div>

<div class="stat-row">
  {_stat_tile(n_enrolled_cwns, 'CWNS enrolled')}
  {_stat_tile(n_enrolled_cws, 'CWS enrolled')}
  {_stat_tile(f'{n_rsa_cwns}/{n_enrolled_cwns}', 'CWNS RSA-eligible')}
  {_stat_tile(f'{n_rsa_cws}/{n_enrolled_cws}', 'CWS RSA-eligible')}
  {_stat_tile(n_single_run_rsa, 'RSA-ineligible: single badaga run')}
  {_stat_tile(n_motion_dropped_rsa, 'RSA-ineligible: motion dropped a run')}
</div>

<h2>Where we're losing participants</h2>
<p>Descriptive stats and pipeline completeness by group, built from the same per-subject files
the pipeline already writes (<code>motion_qc.csv</code>, <code>glmsingle_info.json</code>) plus a
raw BIDS listing of badaga runs on disk -- not estimated, read directly.</p>
{descriptive_table_html}

{_img_tag(attrition_b64, 'N subjects per group, one bar per pipeline stage.', width='55%')}

<h3>Excluded before the pipeline even runs</h3>
<p>These subjects never reach first-level modeling; reasons below are recorded in
<code>group_level_all_ROI.ipynb</code>'s <code>ignore_subs</code> list, not derived from files.</p>
{pre_pipeline_table_html}

<h3>Univariate: subjects with an incomplete SNR contrast set</h3>
<p>Everyone else who enrolled, but doesn't have all 5 SNR-level first-level statmaps
(<code>q</code>, <code>8</code>, <code>0</code>, <code>n2</code>, <code>n6</code>).
{n_motion_flagged_univariate} of these look like a motion story (mean FD above the 0.9mm
scrubbing threshold used by <code>univariate_first-level.py</code>) -- the rest have no
first-level output at all yet, or failed for an unrecorded reason (check SLURM logs for those).</p>
{univariate_incomplete_table_html}

<h3>RSA: why isn't a subject crossnobis-eligible?</h3>
<p>RSA/crossnobis needs a condition to repeat <em>across</em> runs, not just within one run --
if it doesn't, GLMsingle silently falls back to a degraded TYPEB estimate that's never used for
RSA. The chart below separates that from a run being dropped for excessive motion (which can
also leave a subject with only 1 usable run), from subjects GLMsingle simply hasn't been run for
yet.</p>
{_img_tag(rsa_ineligibility_b64, 'RSA-ineligibility reasons by group.', width='80%')}
{rsa_ineligible_table_html}

<h2>CWNS: the well-powered core dataset</h2>
<p>Given the sample-size gap (CWNS enrolled N={n_enrolled_cwns} vs. CWS N={n_enrolled_cws}, and
{n_rsa_cwns} vs. {n_rsa_cws} for RSA), CWNS is where the pipeline currently has real power --
worth keeping in mind if scope narrows toward a CWNS-only analysis.</p>

<div class="group-block cwns">
<h3>Whole-brain</h3>
<p>{cwns_wb_summary}</p>
{_brain_row('CWNS group, quiet (Q) vs. baseline', 'q', 'cwns')}
{_brain_row('CWNS group, Q &minus; N6 (hardest SNR contrast)', 'qMinusN6', 'cwns')}
{_brain_row('CWNS group, all sound vs. baseline (localizer)', 'sound', 'cwns')}

<h3>ROI (20 cortical ROIs x 5 SNR levels, FDR-corrected)</h3>
{cwns_roi_table_html}

<h3>Omnibus ANOVA (hemisphere x SNR x region)</h3>
<p>Significant main effects of hemisphere, SNR, and region, plus hemisphere&times;region,
SNR&times;region, and the 3-way interaction (see table below) -- substantial, well-powered
structure to work with.</p>

<h3>Linear SNR-trend &amp; laterality index</h3>
<p>Two analyses added since the last report, both well-powered for CWNS: a linear SNR-trend
(activation should increase as noise decreases) survives FDR correction in 18 of 20 ROIs --
essentially the whole core auditory/language network (strongest: R-STGp, L-PT, L-STGa, R-PT,
R-HG, all p<sub>FDR</sub> &lt; .001). Laterality index (R&minus;L)/(|L|+|R|) shows CWNS is
robustly left-lateralized (negative LI, driven by the right-hemisphere deactivation already in
the ROI table above) in pars opercularis across all 5 SNR levels and pars triangularis at 4/5
(p<sub>FDR</sub> &lt; .05 throughout). Neither shows a significant CWS-vs-CWNS difference at any
ROI or SNR level.</p>
{_roi_fig('trendplot_snr_by-roi.png',
         'Linear SNR-trend score by ROI, CWS vs. CWNS -- asterisks mark CWNS ROIs with a significant positive trend.')}
{_roi_fig('li_heatmap_group-cwns.png',
         'CWNS mean laterality index by region x SNR level -- pars opercularis and pars triangularis are consistently left-lateralized (negative LI) across nearly every SNR level.')}

<h3>RSA</h3>
<p>One within-group finding survives FDR correction (of 120 ROI x model tests): <b>L-STGp
represents syllable identity</b> (p<sub>FDR</sub> &lt; .001) -- a clean, expected result
(classic phonological encoding in posterior superior temporal cortex), useful as a positive
control that the single-trial pipeline is picking up real signal. The same ROI/model pair also
shows a significant positive linear noise-level trend (below) -- two independent tests now
agree on this effect.</p>
{_rsa_fig(f'model_rdms_noiselevel-{RSA_NOISE_LEVEL_TAG}.png',
         'The model RDMs themselves -- what "syllable," "speaker," and the acoustic features actually look like as a dissimilarity matrix, before correlating each against real per-ROI neural RDMs below.')}
{cwns_rsa_hits_html}
{_rsa_fig(f'boxplot_model-syllable_noiselevel-{RSA_NOISE_LEVEL_TAG}.png',
         'Syllable-identity model-fit by ROI (CWS vs. CWNS) -- the L-STGp hit above is the tall CWNS bar with an asterisk.')}
{_rsa_fig(f'boxplot_model-syllable_group-CWNS_by-hemisphere_noiselevel-{RSA_NOISE_LEVEL_TAG}.png',
         'CWNS syllable-identity model-fit split by hemisphere -- the effect is left-lateralized, consistent with classic phonological encoding.')}
{_rsa_fig('boxplot_trend_model-syllable_group-CWNS_by-hemisphere.png',
         'CWNS syllable-identity model-fit trend across all 5 noise levels, split by hemisphere -- L-STGp shows a significant positive trend (p_FDR < .001), converging with the noiselevel-Q-only hit above.')}
{_rsa_fig(f'boxplot_model-speaker_noiselevel-{RSA_NOISE_LEVEL_TAG}.png',
         'Speaker-identity model-fit by ROI, for comparison -- no ROI survives FDR correction for either group.')}
</div>

<h2>CWS: exploratory comparison group</h2>
<p>Smaller sample throughout, and it shows: fewer whole-brain contrasts reach significance,
fewer ROI hits survive FDR, and the RSA hits that do survive should be treated as provisional
until checked against per-subject leverage.</p>

<div class="group-block cws">
<h3>Whole-brain</h3>
<p>{cws_wb_summary}</p>
{_brain_row('CWS group, quiet (Q) vs. baseline', 'q', 'cws')}
{_brain_row('CWS group, Q &minus; N6 (hardest SNR contrast)', 'qMinusN6', 'cws')}
{_brain_row('CWS group, all sound vs. baseline (localizer)', 'sound', 'cws')}

<h3>ROI (20 cortical ROIs x 5 SNR levels, FDR-corrected)</h3>
{cws_roi_table_html}

<h3>Omnibus ANOVA</h3>
<p>Main effects of hemisphere and region are significant, and the omnibus hemisphere&times;region
interaction is too (p=.034) -- but zero pairwise post-hoc comparisons within that interaction
survive FDR correction. Read that as "there's something there, underpowered to localize,"
not "there's nothing there."</p>

<h3>RSA</h3>
<p>With a small sample throughout, pull the per-subject values from
<code>model_fit_scalars_noiselevel-{RSA_NOISE_LEVEL_TAG}.csv</code> before treating any hit
below as a stable group-level effect -- one or two subjects can drive a result at this sample
size.</p>
{cws_rsa_hits_html}
</div>

<h2>Between-group comparison</h2>

<h3>Whole-brain</h3>
<p>{diff_wb_summary}</p>
{_brain_row('CWS &minus; CWNS group difference, Q &minus; N6', 'qMinusN6', 'diff')}

<h3>ROI</h3>
<p>{between_roi_summary}</p>

<h3>RSA -- one exploratory between-group signal (small N, treat cautiously)</h3>
<div class="summary">
<p>{rsa_finding_summary}</p>
<p>Worth tracking, not worth leaning on yet: CWS N is small here, and there's no corroborating
whole-brain or ROI univariate group difference at any noise level to back it up (see above --
0/100 ROI&times;SNR combos and 0/9 whole-brain contrasts show a CWS-vs-CWNS difference). Treat
this as hypothesis-generating until it either replicates with more CWS data or shows up in a
independent measure.</p>
</div>
{between_rsa_hits_html}
{_rsa_fig(f'boxplot_model-acoustic_f0_noiselevel-{RSA_NOISE_LEVEL_TAG}.png',
         'R-SMGa acoustic_f0 model-fit by ROI, CWS vs. CWNS -- the one between-group RSA finding, shown for completeness (see caution above).',
         width='70%')}

<h3>Noise ceiling (RSA model-fit context)</h3>
<p>Mean upper/lower noise-ceiling bounds (Nili et al. 2014) across all 20 ROIs, per group -- the
best model-fit correlation achievable given noise in the empirical RDMs alone, independent of
any model. CWS's smaller N should widen this band relative to CWNS.</p>
{noise_ceiling_table_html}

<h2>ANOVA detail</h2>
<div class="table-wrap">{anova_table_html}</div>

<h2>Open items before publication</h2>
<ul class="open-items">
  <li><b>Report doesn't reflect the full 5-noise-level RSA run yet.</b> RSA now runs separately
  at every noise level (<code>Q, 8, 0, n2, n6</code>) plus a cross-level linear trend, but this
  report's RSA section (<code>RSA_NOISE_LEVEL_TAG = 'Q'</code>) still only reads the Q-restricted
  output. The other 4 levels' between-group results (0/120, 2/120, 3/120, 3/120, 2/120 FDR hits
  at Q/8/0/n2/n6 respectively) and the new trend finding below aren't shown here yet.</li>
  <li><b>New convergent finding: RSA linear noise-level trend, CWNS L-STGp/syllable.</b> Model-fit
  for the syllable-identity model in L-STGp gets significantly stronger as noise decreases
  (p<sub>FDR</sub> &lt; .001) -- the same ROI/model pair that's already the one within-group hit
  at noiselevel-Q alone. Two independent tests (single-level significance, cross-level trend)
  now agree on the same effect, which strengthens it considerably; no between-group trend
  difference survives FDR.</li>
  <li><b>New well-powered univariate ROI findings for CWNS.</b> The linear SNR-trend analysis
  (added since the last report) shows a significant positive trend (activation increases as
  noise decreases) in 18 of 20 CWNS ROIs, spanning essentially the whole core auditory/language
  network (strongest: R-STGp, L-PT, L-STGa, R-PT, R-HG, all p<sub>FDR</sub> &lt; .001). The
  laterality-index analysis shows CWNS is robustly left-lateralized (negative LI, driven by the
  right-hemisphere deactivation already seen in the ROI table above) in pars opercularis across
  all 5 SNR levels and pars triangularis at 4/5 (p<sub>FDR</sub> &lt; .05 throughout). Neither
  trend nor LI shows a significant CWS-vs-CWNS difference at any ROI/level -- consistent with
  every other between-group null result in this report. Not yet shown here; only the original
  per-SNR-level ROI table is embedded.</li>
  <li><b>RSA sample size for CWS.</b> Any within-group CWS RSA hit could be driven by 1-2
  subjects -- check per-subject values before reporting. This now applies across all 5 noise
  levels' between-group hits, not just the original R-SMGa/acoustic_f0 one at Q.</li>
  <li><b>README is stale</b> -- still describes the pre-GLMsingle searchlight RSA approach, no
  mention of acoustic RSA, noise ceiling, or FDR correction.</li>
  <li><b>Dead code</b> -- <code>multivariate_fmri/rsa_searchlight.py</code> and
  <code>group_level_rsa_searchlight_WIP.ipynb</code> look superseded by the GLMsingle ROI
  pipeline; worth deprecating explicitly.</li>
  <li><b>WIN / PTA behavioral covariates</b> exist for most subjects but never made it into the
  final whole-brain design matrix (only age/sex did) -- a natural brain-behavior correlate for
  a speech-in-noise paradigm, currently unused.</li>
  <li><b>Next analysis:</b> correlate WIN scores against the CWNS findings that are actually
  well-powered -- the L-STGp syllable RSA trend, the whole-network univariate SNR-trend, or the
  pars opercularis/triangularis laterality index -- rather than the small-N R-SMGa/acoustic_f0
  between-group result, which has no univariate corroboration to lean on.</li>
</ul>

<footer>
Generated by <code>report/generate_report.py</code> from cached notebook/script outputs under
<code>{GROUP_OUT_DIR}</code>, <code>{WHOLE_BRAIN_DIR}</code>, <code>{L1_DIR}</code>,
<code>{GLMSINGLE_DIR}</code>, and <code>{RSA_OUT_DIR}</code>. Figures are embedded PNGs already
saved by the source notebooks, plus summary charts built fresh from the cached CSVs/JSON. Rerun
after each cluster run to refresh.
</footer>

</body>
</html>'''
    return html


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    print('Loading cached data...')
    participants_df = load_participants()
    roi_long_df = load_roi_long()
    within_group_stats_df = load_within_group_stats()
    between_group_stats_df = load_between_group_stats()
    anova_results_df = load_anova_results()
    rsa_model_fit_df = load_rsa_model_fit()
    rsa_group_comparison_df = load_rsa_group_comparison()
    rsa_within_group_df = load_rsa_within_group()
    rsa_noise_ceiling_df = load_rsa_noise_ceiling()

    print('Building attrition ledger (per-subject motion_qc.csv / glmsingle_info.json / RDM presence)...')
    ledger_df = build_attrition_ledger(participants_df)

    print('Building summary charts...')
    attrition_b64 = make_attrition_chart(participants_df, roi_long_df, rsa_model_fit_df)
    rsa_ineligibility_b64 = make_rsa_ineligibility_chart(ledger_df)
    hit_count_b64 = make_roi_hit_count_chart(within_group_stats_df)

    print('Building ROI hit tables...')
    cwns_roi_table_html = _df_to_table_html(make_roi_direction_table(within_group_stats_df, 'CWNS'))
    cws_roi_table_html = _df_to_table_html(make_roi_direction_table(within_group_stats_df, 'CWS'))

    print('Building attrition tables...')
    descriptive_table_html = _df_to_table_html(make_descriptive_stats_table(participants_df, ledger_df))
    pre_pipeline_table_html = _df_to_table_html(make_pre_pipeline_table(ledger_df))
    univariate_incomplete_table_html = _df_to_table_html(make_univariate_incomplete_table(ledger_df))
    rsa_ineligible_table_html = _df_to_table_html(make_rsa_ineligible_table(ledger_df))

    n_motion_flagged_univariate = n_motion_dropped_rsa = n_single_run_rsa = 0
    if ledger_df is not None:
        n_motion_flagged_univariate = int((ledger_df.univariate_status == 'likely motion').sum())
        n_motion_dropped_rsa = int((ledger_df.rsa_status == 'run dropped for motion').sum())
        n_single_run_rsa = int((ledger_df.rsa_status == 'single run (file issue)').sum())

    print('Building RSA hit tables...')
    cwns_hits = None
    cws_hits = None
    between_hits = None
    if rsa_within_group_df is not None:
        cwns_hits = make_rsa_hits_table(rsa_within_group_df[rsa_within_group_df.group == 'CWNS'], 'within')
        cws_hits = make_rsa_hits_table(rsa_within_group_df[rsa_within_group_df.group == 'CWS'], 'within')
    if rsa_group_comparison_df is not None:
        between_hits = make_rsa_hits_table(rsa_group_comparison_df, 'between')
    cwns_rsa_hits_html = _df_to_table_html(cwns_hits)
    cws_rsa_hits_html = _df_to_table_html(cws_hits)
    between_rsa_hits_html = _df_to_table_html(between_hits)
    noise_ceiling_table_html = _df_to_table_html(make_noise_ceiling_summary(rsa_noise_ceiling_df))

    # whole-brain summaries: hand-maintained from univariate_group-level.ipynb's saved output
    # (that notebook doesn't currently cache a machine-readable per-contrast threshold table --
    # see the "Open items" section; until it does, these two lines are the one place in this
    # script that needs a human to update them after a fresh univariate_group-level.ipynb run).
    cwns_wb_summary = summarize_whole_brain('CWNS', WHOLE_BRAIN_CONTRASTS, [])
    cws_wb_summary = summarize_whole_brain(
        'CWS', ['q', '0', 'sound', 'response'], ['8', 'n2', 'n6', 'qMinusN6', 'qMinus0'])
    diff_wb_summary = ('No contrast shows a significant CWS-vs-CWNS difference: all 9 difference '
                       'maps are empty (FDR threshold &rarr; &infin;) -- consistent with the '
                       'ROI-level result below.')

    between_roi_summary = '0/100 ROI x SNR combinations survive FDR (smallest p_FDR observed).'
    if between_group_stats_df is not None and len(between_group_stats_df) > 0:
        n_sig = (between_group_stats_df.p_fdr < 0.05).sum()
        min_p_fdr = between_group_stats_df.p_fdr.min()
        between_roi_summary = (f'{n_sig}/{len(between_group_stats_df)} ROI x SNR combinations '
                               f'survive FDR correction (smallest p_FDR = {min_p_fdr:.3f}).')

    rsa_finding_summary = '[group_comparison_bootstrap CSV not found -- see Open items]'
    if rsa_group_comparison_df is not None:
        sig = rsa_group_comparison_df[rsa_group_comparison_df.p_fdr < 0.05].sort_values('p_fdr')
        n_total = len(rsa_group_comparison_df)
        if len(sig) > 0:
            top = sig.iloc[0]
            rsa_finding_summary = (
                f'{len(sig)}/{n_total} ROI x model tests survive FDR correction. The strongest: '
                f'<b>{top["ROI"]} / {top["model"]}</b> (observed diff = {top["observed_diff"]:.3f}, '
                f'p<sub>FDR</sub> = {top["p_fdr"]:.3f}). '
            )
            if rsa_within_group_df is not None:
                matching_within = rsa_within_group_df[
                    (rsa_within_group_df.ROI == top['ROI']) & (rsa_within_group_df.model == top['model'])
                    & (rsa_within_group_df.p_fdr < 0.05)
                ]
                if len(matching_within) > 0:
                    within_groups = ', '.join(matching_within['group'].tolist())
                    rsa_finding_summary += (
                        f'Same ROI/model is also individually significant within-group for '
                        f'{within_groups} -- two independent tests converging on the same effect.')
        else:
            rsa_finding_summary = f'No ROI x model test survives FDR correction (0/{n_total}).'

    n_enrolled_cwns = int((participants_df.group_norm == 'control').sum()) if participants_df is not None else '—'
    n_enrolled_cws = int((participants_df.group_norm == 'cws').sum()) if participants_df is not None else '—'
    n_rsa_cwns = rsa_model_fit_df.drop_duplicates('participant_id').query("group == 'CWNS'").shape[0] \
        if rsa_model_fit_df is not None else '—'
    n_rsa_cws = rsa_model_fit_df.drop_duplicates('participant_id').query("group == 'CWS'").shape[0] \
        if rsa_model_fit_df is not None else '—'

    print('Assembling HTML...')
    html = build_html(
        participants_df=participants_df, ledger_df=ledger_df,
        attrition_b64=attrition_b64, rsa_ineligibility_b64=rsa_ineligibility_b64, hit_count_b64=hit_count_b64,
        descriptive_table_html=descriptive_table_html, pre_pipeline_table_html=pre_pipeline_table_html,
        univariate_incomplete_table_html=univariate_incomplete_table_html,
        rsa_ineligible_table_html=rsa_ineligible_table_html,
        n_motion_flagged_univariate=n_motion_flagged_univariate, n_motion_dropped_rsa=n_motion_dropped_rsa,
        n_single_run_rsa=n_single_run_rsa,
        cwns_roi_table_html=cwns_roi_table_html, cws_roi_table_html=cws_roi_table_html,
        anova_results_df=anova_results_df,
        cwns_wb_summary=cwns_wb_summary, cws_wb_summary=cws_wb_summary, diff_wb_summary=diff_wb_summary,
        between_roi_summary=between_roi_summary, rsa_finding_summary=rsa_finding_summary,
        cwns_rsa_hits_html=cwns_rsa_hits_html, cws_rsa_hits_html=cws_rsa_hits_html,
        between_rsa_hits_html=between_rsa_hits_html, noise_ceiling_table_html=noise_ceiling_table_html,
        n_enrolled_cwns=n_enrolled_cwns, n_enrolled_cws=n_enrolled_cws,
        n_rsa_cwns=n_rsa_cwns, n_rsa_cws=n_rsa_cws,
    )

    out_path = REPORT_DIR / 'SSP_status_report.html'
    out_path.write_text(html, encoding='utf-8')
    print(f'\nReport written to {out_path}')
    print(f'File size: {out_path.stat().st_size / 1e3:.0f} KB')


if __name__ == '__main__':
    main()
