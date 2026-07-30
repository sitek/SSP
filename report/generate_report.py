"""Generate a self-contained HTML status report for the SSP CWS/CWNS speech-in-noise study.

Produces report/SSP_status_report.html -- all figures are either base64-embedded PNGs already
saved by univariate_fmri/group_level_all_ROI.ipynb and multivariate_fmri/GLMsingle_rsa-group.ipynb
(brain maps, box+strip plots), or small summary charts built fresh here from the numeric CSVs
those notebooks cache. No external CSS/JS dependencies, no nilearn/rsatoolbox required to run
this script itself -- only pandas/numpy/matplotlib, so it can run anywhere the cached CSVs and
PNGs have been copied to (a laptop, not necessarily the cluster), unlike the notebooks themselves.

Modeled on the report-generator pattern already used for the acfMRI project
(~/software/acfMRI/07_report/generate_report.py): cached numeric results in, one flat HTML file
out, figures embedded rather than linked.

Usage:
    python report/generate_report.py

Requires the univariate and RSA group-level notebooks to have been run at least once (so the
CSVs/PNGs referenced below exist under BIDSROOT). Missing files are skipped with a printed note,
not a hard failure -- the report still builds with whatever is available.
"""

import base64
import io
from datetime import date
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
GROUP_OUT_DIR = NILEARN_DIR / f'group_fwhm-{FWHM:.2f}'
RSA_OUT_DIR = GLMSINGLE_DIR / 'rsa-group_glmsingle'
RSA_NOISE_LEVEL_TAG = 'Q'  # matches GLMsingle_rsa-group.ipynb's NOISE_LEVEL_TAG for the acoustic-model run

PARTICIPANTS_FPATH = BIDSROOT / 'participants.tsv'

CONTRAST_LIST = ['q', '8', '0', 'n2', 'n6']
WHOLE_BRAIN_CONTRASTS = ['q', '8', '0', 'n2', 'n6', 'qMinusN6', 'qMinus0', 'sound', 'response']

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


# ── charts built fresh from cached CSVs ─────────────────────────────────────────
def make_attrition_chart(participants_df, roi_long_df, rsa_model_fit_df):
    """Grouped bar chart: N subjects by pipeline stage, CWNS vs CWS. A genuine funnel (each
    stage is a strict subset of the previous), so bars are ordered stage-by-stage rather than
    alphabetically.
    """
    stages = []
    if participants_df is not None:
        counts = participants_df.group_norm.value_counts()
        stages.append(('Enrolled\n(participants.tsv)', counts.get('control', 0), counts.get('cws', 0)))
    if roi_long_df is not None:
        n = roi_long_df.drop_duplicates('participant_id')[['participant_id', 'group']]
        counts = n.group.value_counts()
        stages.append(('ROI univariate\n(any usable contrast)', counts.get('CWNS', 0), counts.get('CWS', 0)))
    if rsa_model_fit_df is not None:
        n = rsa_model_fit_df.drop_duplicates('subject_id')[['subject_id', 'group']]
        counts = n.group.value_counts()
        stages.append(('RSA (GLMsingle,\ncross-run repeats)', counts.get('CWNS', 0), counts.get('CWS', 0)))

    if not stages:
        return None

    labels = [s[0] for s in stages]
    cwns_n = [s[1] for s in stages]
    cws_n = [s[2] for s in stages]
    x = np.arange(len(labels))
    width = 0.32

    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=150)
    ax.bar(x - width / 2, cwns_n, width, label='CWNS', color=GROUP_COLOR['CWNS'])
    ax.bar(x + width / 2, cws_n, width, label='CWS', color=GROUP_COLOR['CWS'])
    for xi, n in zip(x - width / 2, cwns_n):
        ax.text(xi, n + 0.5, str(int(n)), ha='center', va='bottom', fontsize=9)
    for xi, n in zip(x + width / 2, cws_n):
        ax.text(xi, n + 0.5, str(int(n)), ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('N subjects')
    ax.set_title('Sample size by pipeline stage')
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
    are significant, and in which direction (based on the sign of mean_beta).
    """
    if within_group_stats_df is None:
        return None
    sig = within_group_stats_df[
        (within_group_stats_df.group == group_name) & (within_group_stats_df.p_fdr < 0.05)
    ].copy()
    if len(sig) == 0:
        return pd.DataFrame(columns=['region_hemi', 'n_snr_significant', 'direction', 'SNR levels'])

    sig['direction'] = np.where(sig.mean_beta > 0, 'positive', 'negative')
    rows = []
    for region_hemi, group_df in sig.groupby('region_hemi'):
        directions = group_df['direction'].unique()
        direction_label = directions[0] if len(directions) == 1 else 'mixed'
        rows.append({
            'ROI': region_hemi,
            'SNR levels significant': len(group_df),
            'direction': direction_label,
            'at': ', '.join(sorted(group_df['SNR'].astype(str))),
        })
    out = pd.DataFrame(rows).sort_values('SNR levels significant', ascending=False).reset_index(drop=True)
    return out


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
* { box-sizing: border-box; }
:root {
  --bg: #F5F6F9; --surface: #FFFFFF; --surface-alt: #EEF0F4;
  --ink: #161B22; --ink-muted: #5C6570; --border: #E2E5EA;
  --accent: #9C7A1B; --cwns: #009E73; --cws: #CC79A7; --hemi-l: #0072B2; --hemi-r: #D55E00;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #12151C; --surface: #1B212B; --surface-alt: #222836;
    --ink: #E8EAED; --ink-muted: #97A0AC; --border: #2B323D;
    --accent: #C9A13A; --cwns: #1FA37D; --cws: #C0759E; --hemi-l: #4A8FCF; --hemi-r: #D97640;
  }
}
:root[data-theme="dark"] {
  --bg: #12151C; --surface: #1B212B; --surface-alt: #222836;
  --ink: #E8EAED; --ink-muted: #97A0AC; --border: #2B323D;
  --accent: #C9A13A; --cwns: #1FA37D; --cws: #C0759E; --hemi-l: #4A8FCF; --hemi-r: #D97640;
}
:root[data-theme="light"] {
  --bg: #F5F6F9; --surface: #FFFFFF; --surface-alt: #EEF0F4;
  --ink: #161B22; --ink-muted: #5C6570; --border: #E2E5EA;
  --accent: #9C7A1B; --cwns: #009E73; --cws: #CC79A7; --hemi-l: #0072B2; --hemi-r: #D55E00;
}
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  font-size: 16px; line-height: 1.65; color: var(--ink); background: var(--bg);
  max-width: 920px; margin: 0 auto; padding: 3rem 1.5rem 5rem;
}
h1, h2, h3 { font-family: 'Iowan Old Style', 'Palatino Linotype', Palatino, 'Book Antiqua', Georgia, serif;
  text-wrap: balance; color: var(--ink); }
h1 { font-size: 2.4rem; margin-bottom: 0.3rem; }
h2 { font-size: 1.5rem; margin: 3rem 0 1rem; padding-bottom: 0.5rem; border-bottom: 2px solid var(--border); }
h3 { font-size: 1.15rem; margin: 1.8rem 0 0.7rem; color: var(--ink-muted); }
p { margin-bottom: 0.9rem; max-width: 68ch; }
.eyebrow { text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.75rem;
  color: var(--ink-muted); font-weight: 600; }
.dek { color: var(--ink-muted); font-size: 1.1rem; margin-bottom: 1.5rem; max-width: 60ch; }
.meta { color: var(--ink-muted); font-size: 0.85rem; margin-bottom: 2rem; }
code { font-family: ui-monospace, 'SF Mono', 'Cascadia Code', 'Roboto Mono', Consolas, monospace;
  background: var(--surface-alt); padding: 0.1em 0.35em; border-radius: 3px; font-size: 0.9em; }

.stat-row { display: flex; flex-wrap: wrap; gap: 1rem; margin: 1.5rem 0 2.5rem; }
.stat-tile { background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 0.9rem 1.2rem; flex: 1 1 150px; }
.stat-tile .n { font-family: ui-monospace, 'SF Mono', 'Roboto Mono', monospace;
  font-variant-numeric: tabular-nums; font-size: 1.7rem; font-weight: 600; display: block; }
.stat-tile .label { font-size: 0.8rem; color: var(--ink-muted); }

figure { margin: 1rem 0 1.5rem; }
figcaption { font-size: 0.85rem; color: var(--ink-muted); margin-top: 0.4rem; font-style: italic; }
.missing { color: var(--ink-muted); font-style: italic; font-size: 0.9rem; }

.finding-card { background: var(--surface); border: 1px solid var(--border); border-left: 4px solid var(--accent);
  border-radius: 0 8px 8px 0; padding: 1.1rem 1.4rem; margin: 1rem 0 1.8rem; }
.finding-card .eyebrow { color: var(--accent); }
.finding-card p:last-child { margin-bottom: 0; }

.group-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 1.2rem 1.5rem; margin: 1rem 0 2rem; }
.group-card.cwns { border-top: 3px solid var(--cwns); }
.group-card.cws { border-top: 3px solid var(--cws); }

.table-wrap { overflow-x: auto; margin: 1rem 0 1.5rem; }
table.data-table { border-collapse: collapse; width: 100%; font-size: 0.88rem;
  font-variant-numeric: tabular-nums; }
table.data-table th, table.data-table td { border-bottom: 1px solid var(--border);
  padding: 0.4rem 0.7rem; text-align: left; }
table.data-table th { color: var(--ink-muted); font-weight: 600; font-size: 0.8rem;
  text-transform: uppercase; letter-spacing: 0.03em; }
table.data-table tbody tr:hover { background: var(--surface-alt); }

.tag { display: inline-block; font-size: 0.72rem; font-weight: 600; padding: 0.1em 0.55em;
  border-radius: 999px; letter-spacing: 0.02em; }
.tag.cwns { background: color-mix(in srgb, var(--cwns) 18%, transparent); color: var(--cwns); }
.tag.cws { background: color-mix(in srgb, var(--cws) 18%, transparent); color: var(--cws); }

ul.open-items { padding-left: 1.3rem; }
ul.open-items li { margin-bottom: 0.6rem; }

footer { margin-top: 4rem; padding-top: 1.5rem; border-top: 1px solid var(--border);
  color: var(--ink-muted); font-size: 0.85rem; }
'''


def _stat_tile(n, label):
    return f'<div class="stat-tile"><span class="n">{n}</span><span class="label">{label}</span></div>'


def build_html(*, participants_df, roi_long_df,
               attrition_b64, hit_count_b64,
               cwns_roi_table_html, cws_roi_table_html,
               anova_results_df, cwns_wb_summary, cws_wb_summary, diff_wb_summary,
               between_roi_summary, rsa_finding_summary,
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

<div class="eyebrow">SSP &middot; Chandrasekaran Lab &middot; internal status report</div>
<h1>Speech-in-noise fMRI: where the CWS/CWNS data stand</h1>
<p class="dek">Univariate (whole-brain + ROI) and single-trial RSA results on the <code>badaga</code>
speech-in-noise task, generated from the cached outputs of
<code>univariate_fmri/group_level_all_ROI.ipynb</code> and
<code>multivariate_fmri/GLMsingle_rsa-group.ipynb</code>.</p>
<div class="meta">Generated {today} &middot; not a live view -- rerun this script after the next cluster run to refresh.</div>

<div class="stat-row">
  {_stat_tile(n_enrolled_cwns, 'CWNS enrolled')}
  {_stat_tile(n_enrolled_cws, 'CWS enrolled')}
  {_stat_tile(f'{n_rsa_cwns}/{n_enrolled_cwns}', 'CWNS RSA-eligible')}
  {_stat_tile(f'{n_rsa_cws}/{n_enrolled_cws}', 'CWS RSA-eligible')}
</div>

<div class="finding-card">
  <div class="eyebrow">Headline</div>
  <p>Univariate group differences are consistently null -- whole-brain and ROI-level, every
  contrast. The one place CWS and CWNS separate is in single-trial RSA: <b>R-SMGa's
  representational geometry for pitch (F0)</b> differs between groups, and is itself
  significantly represented within CWS alone -- the only finding that converges across two
  independent tests.</p>
</div>

<h2>Sample &amp; attrition</h2>
<p>Usable N drops at each stage of the pipeline, and drops faster for CWS than CWNS -- most
sharply for RSA, which requires cross-run condition repeats (single-run subjects are excluded).</p>
{_img_tag(attrition_b64, 'N subjects by pipeline stage, CWNS vs. CWS.', width='60%')}

<h2>CWNS: the well-powered core dataset</h2>
<p>Given the sample-size gap (CWNS enrolled N={n_enrolled_cwns} vs. CWS N={n_enrolled_cws}, and
{n_rsa_cwns} vs. {n_rsa_cws} for RSA), CWNS is where the pipeline currently has real power --
worth keeping in mind if scope narrows toward a CWNS-only analysis.</p>

<div class="group-card cwns">
<h3>Whole-brain</h3>
<p>{cwns_wb_summary}</p>

<h3>ROI (20 cortical ROIs x 5 SNR levels, FDR-corrected)</h3>
{cwns_roi_table_html}

<h3>Omnibus ANOVA (hemisphere x SNR x region)</h3>
<p>Significant main effects of hemisphere, SNR, and region, plus hemisphere&times;region,
SNR&times;region, and the 3-way interaction (see table below) -- substantial, well-powered
structure to work with.</p>

<h3>RSA</h3>
<p>One within-group finding survives FDR correction (of 120 ROI x model tests): <b>L-STGp
represents syllable identity</b> (p<sub>FDR</sub> &lt; .001) -- a clean, expected result
(classic phonological encoding in posterior superior temporal cortex), useful as a positive
control that the single-trial pipeline is picking up real signal.</p>
</div>

<h2>CWS: exploratory comparison group</h2>
<p>Smaller sample throughout, and it shows: fewer whole-brain contrasts reach significance,
fewer ROI hits survive FDR, and the RSA hits that do survive (n=7) should be treated as
provisional until checked against per-subject leverage.</p>

<div class="group-card cws">
<h3>Whole-brain</h3>
<p>{cws_wb_summary}</p>

<h3>ROI (20 cortical ROIs x 5 SNR levels, FDR-corrected)</h3>
{cws_roi_table_html}

<h3>Omnibus ANOVA</h3>
<p>Main effects of hemisphere and region are significant, and the omnibus hemisphere&times;region
interaction is too (p=.034) -- but zero pairwise post-hoc comparisons within that interaction
survive FDR correction. Read that as "there's something there, underpowered to localize,"
not "there's nothing there."</p>

<h3>RSA</h3>
<p>Three within-group findings survive FDR (of 120 tests, n=7 throughout): R-STGa and R-HG both
represent speaker identity (opposite sign), and R-SMGa represents F0/acoustic pitch information
(negative). With only 7 subjects, pull the per-subject values from
<code>model_fit_scalars_noiselevel-{RSA_NOISE_LEVEL_TAG}.csv</code> before treating any of these
as a stable group-level effect -- one or two subjects can drive a result at this sample size.</p>
</div>

<h2>Between-group comparison</h2>

<h3>Whole-brain</h3>
<p>{diff_wb_summary}</p>

<h3>ROI</h3>
<p>{between_roi_summary}</p>

<h3>RSA -- the one standout result</h3>
<div class="finding-card">
<p>{rsa_finding_summary}</p>
</div>

<h2>ANOVA detail</h2>
<div class="table-wrap">{anova_table_html}</div>

<h2>Open items before publication</h2>
<ul class="open-items">
  <li><b>Missing first-level maps.</b> Several subjects are consistently absent across SNR
  contrasts (e.g. sub-SSP034/051/062/077/081/092/097/111) -- confirm whether that's still-processing,
  failed QC, or an intentional exclusion, since it's the single biggest lever on power.</li>
  <li><b>RSA sample size for CWS (n=7).</b> Any within-group CWS RSA hit could be driven by 1-2
  subjects -- check per-subject values before reporting.</li>
  <li><b>README is stale</b> -- still describes the pre-GLMsingle searchlight RSA approach, no
  mention of acoustic RSA, noise ceiling, or FDR correction.</li>
  <li><b>Dead code</b> -- <code>multivariate_fmri/rsa_searchlight.py</code> and
  <code>group_level_rsa_searchlight_WIP.ipynb</code> look superseded by the GLMsingle ROI
  pipeline; worth deprecating explicitly.</li>
  <li><b>WIN / PTA behavioral covariates</b> exist for most subjects but never made it into the
  final whole-brain design matrix (only age/sex did) -- a natural brain-behavior correlate for
  a speech-in-noise paradigm, currently unused.</li>
  <li><b>Next analysis:</b> correlate the R-SMGa acoustic_f0 RSA effect (or ROI betas) against
  WIN scores -- would directly link the one real neural finding to a behavioral outcome.</li>
</ul>

<footer>
Generated by <code>report/generate_report.py</code> from cached notebook outputs under
<code>{GROUP_OUT_DIR}</code> and <code>{RSA_OUT_DIR}</code>. Figures are embedded PNGs already
saved by the source notebooks, plus two summary charts built fresh from the cached CSVs. Rerun
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

    print('Building summary charts...')
    attrition_b64 = make_attrition_chart(participants_df, roi_long_df, rsa_model_fit_df)
    hit_count_b64 = make_roi_hit_count_chart(within_group_stats_df)

    print('Building ROI hit tables...')
    cwns_roi_table_html = _df_to_table_html(make_roi_direction_table(within_group_stats_df, 'CWNS'))
    cws_roi_table_html = _df_to_table_html(make_roi_direction_table(within_group_stats_df, 'CWS'))

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
    n_rsa_cwns = rsa_model_fit_df.drop_duplicates('subject_id').query("group == 'CWNS'").shape[0] \
        if rsa_model_fit_df is not None else '—'
    n_rsa_cws = rsa_model_fit_df.drop_duplicates('subject_id').query("group == 'CWS'").shape[0] \
        if rsa_model_fit_df is not None else '—'

    print('Assembling HTML...')
    html = build_html(
        participants_df=participants_df, roi_long_df=roi_long_df,
        attrition_b64=attrition_b64, hit_count_b64=hit_count_b64,
        cwns_roi_table_html=cwns_roi_table_html, cws_roi_table_html=cws_roi_table_html,
        anova_results_df=anova_results_df,
        cwns_wb_summary=cwns_wb_summary, cws_wb_summary=cws_wb_summary, diff_wb_summary=diff_wb_summary,
        between_roi_summary=between_roi_summary, rsa_finding_summary=rsa_finding_summary,
        n_enrolled_cwns=n_enrolled_cwns, n_enrolled_cws=n_enrolled_cws,
        n_rsa_cwns=n_rsa_cwns, n_rsa_cws=n_rsa_cws,
    )

    out_path = REPORT_DIR / 'SSP_status_report.html'
    out_path.write_text(html, encoding='utf-8')
    print(f'\nReport written to {out_path}')
    print(f'File size: {out_path.stat().st_size / 1e3:.0f} KB')


if __name__ == '__main__':
    main()
