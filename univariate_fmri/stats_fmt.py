import os

"""
Formatting helpers for consistent APA-style statistical reporting.

Usage:
    from stats_fmt import fmt_t, fmt_F, fmt_p, fmt_r, stat_str

    print(fmt_t(11, 2.345))           # "t(11) = 2.35"
    print(fmt_F(2, 22, 4.123))        # "F(2, 22) = 4.12"
    print(fmt_p(0.0003))              # "p < .001"
    print(fmt_r(0.7823))              # "r = .78"
    print(stat_str('t', 11, 2.345, 0.038))  # "t(11) = 2.35, p = .038"
"""


def fmt_p(p):
    """Format a p-value to 3 decimal places, APA style (no leading zero).
    Values below .001 reported as 'p < .001'. Returns 'n/a' for None/NaN."""
    if p is None or p != p:  # NaN check: NaN != NaN
        return "n/a"
    p = float(p)
    if p < 0.001:
        return "p < .001"
    return f"p = {p:.3f}".replace("0.", ".")


#def fmt_t(df: int | float, t: float) -> str:
def fmt_t(df, t): # running Python 3.9 without doc typing
    """Format a t-statistic with degrees of freedom."""
    return f"t({df}) = {t:.2f}"


#def fmt_F(df1: int | float, df2: int | float, F: float) -> str:
def fmt_F(df1, df2, F):
    """Format an F-statistic with numerator and denominator degrees of freedom."""
    return f"F({df1}, {df2}) = {F:.2f}"


def fmt_r(r: float) -> str:
    """Format a correlation coefficient to 2 decimal places, no leading zero."""
    return f"r = {r:.2f}".replace("r = 0.", "r = .").replace("r = -0.", "r = -.")


def fmt_z(z: float) -> str:
    """Format a z-statistic to 2 decimal places."""
    return f"z = {z:.2f}"


def stat_str(stat_type: str, *args) -> str:
    """
    Convenience wrapper that returns a full 'stat, p' string.

    Examples
    --------
    stat_str('t', df, t_val, p_val)
    stat_str('F', df1, df2, F_val, p_val)
    stat_str('r', r_val, p_val)
    """
    if stat_type == 't':
        df, t_val, p_val = args
        return f"{fmt_t(df, t_val)}, {fmt_p(p_val)}"
    elif stat_type == 'F':
        df1, df2, F_val, p_val = args
        return f"{fmt_F(df1, df2, F_val)}, {fmt_p(p_val)}"
    elif stat_type == 'r':
        r_val, p_val = args
        return f"{fmt_r(r_val)}, {fmt_p(p_val)}"
    elif stat_type == 'z':
        z_val, p_val = args
        return f"{fmt_z(z_val)}, {fmt_p(p_val)}"
    else:
        raise ValueError(f"Unknown stat_type '{stat_type}'. Use 't', 'F', 'r', or 'z'.")


def stat_str_fdr(stat_type, *args):
    """
    Like stat_str but appends a FDR-corrected p-value.
    Last argument is always p_fdr.

    Examples
    --------
    stat_str_fdr('t', 11, 2.345, 0.038, 0.045)   # "t(11) = 2.35, p = .038, p_FDR = .045"
    stat_str_fdr('F', 2, 22, 4.12, 0.031, 0.048) # "F(2, 22) = 4.12, p = .031, p_FDR = .048"
    """
    *stat_args, p_fdr = args
    # strip the p-value from stat_str, replace with FDR q
    stat_part = stat_str(stat_type, *stat_args).rsplit(',', 1)[0]
    if p_fdr < 0.001:
        q_str = "q < .001"
    else:
        q_str = f"q = {p_fdr:.3f}".replace("0.", ".")
    return f"{stat_part}, {q_str}"


def export_anova(aov, label, out_dir, filename=None):
    """
    Convert an AnovaRM result to a clean DataFrame and save as TSV.

    Parameters
    ----------
    aov : AnovaRM fitted result (has .anova_table attribute)
    label : str, used in filename if filename not provided (e.g. 'sound_striatum')
    out_dir : str, directory to save TSV
    filename : str, optional override for output filename
    """
    table = aov.anova_table.copy()
    table.index.name = 'source'
    table = table.reset_index()
    table.columns = ['source', 'F', 'df_num', 'df_den', 'p']
    table['stat_str'] = table.apply(
        lambda r: stat_str('F', int(r['df_num']), int(r['df_den']), r['F'], r['p']), axis=1)
    table = table[['source', 'F', 'df_num', 'df_den', 'p', 'stat_str']]
    table['F'] = table['F'].map('{:.2f}'.format)
    table['p'] = table['p'].map(lambda x: fmt_p(float(x)) if x is not None else x)

    fname = filename or f'anova_{label}.tsv'
    table.to_csv(os.path.join(out_dir, fname), sep='\t', index=False)
    return table


def export_posthoc(pg_df, label, out_dir, filename=None):
    """
    Convert a pg.pairwise_tests result to a clean DataFrame and save as TSV.
    Expects p-corr column (FDR-corrected) already present from padjust='fdr'.

    Parameters
    ----------
    pg_df : pd.DataFrame, output of pg.pairwise_tests()
    label : str, used in filename if filename not provided
    out_dir : str, directory to save TSV
    filename : str, optional override for output filename
    """
    from numpy import nan
    cols = [c for c in ['Contrast', 'region', 'hemisphere', 'learning_stage', 'A', 'B']
            if c in pg_df.columns]
    try:
        table = pg_df[cols + ['T', 'dof', 'p-unc', 'p-corr']].copy()
        table = table.rename(columns={'T': 't', 'dof': 'df', 'p-unc': 'p', 'p-corr': 'p_fdr'})
        table['stat_str'] = table.apply(
            lambda r: stat_str('t', int(r['df']), r['t'], r['p']), axis=1)
        table['stat_str_fdr'] = table.apply(
            lambda r: stat_str_fdr('t', int(r['df']), r['t'], r['p'], r['p_fdr']), axis=1)
    except KeyError:
        table = pg_df[cols + ['T', 'dof', 'p-unc']].copy()
        table = table.rename(columns={'T': 't', 'dof': 'df', 'p-unc': 'p'})
        table['stat_str'] = table.apply(
            lambda r: stat_str('t', int(r['df']), r['t'], r['p']), axis=1)
        table['stat_str_fdr'] = nan        

    table['t'] = table['t'].map('{:.2f}'.format)
    table['p'] = table['p'].map(fmt_p)
    if 'p_fdr' in table.columns:
        table['p_fdr'] = table['p_fdr'].map(fmt_p)

    fname = filename or f'posthoc_{label}.tsv'
    table.to_csv(os.path.join(out_dir, fname), sep='\t', index=False)
    return table


def export_ttests(records, label, out_dir, filename=None):
    """
    Convert a list of ttest_1samp or ttest_rel results to a clean DataFrame and save as TSV.
    Applies FDR correction across all tests in the table.

    Each record should have:
        label (str), t (float), df (int), p (float)
    plus any grouping columns you want to keep (e.g. region, contrast)

    Example
    -------
    records = []
    for region, data in roi_df.groupby('region'):
        t, p = ttest_1samp(data['beta'], 0)
        records.append({'region': region, 't': t, 'df': len(data) - 1, 'p': p})
    export_ttests(records, 'sound_baseline', stats_out_dir)
    """
    from statsmodels.stats.multitest import multipletests
    from pandas import DataFrame
    table = DataFrame(records)
    _, p_fdr = multipletests(table['p'], method='fdr_bh')[:2]
    table['p_fdr'] = p_fdr
    table['stat_str'] = table.apply(
        lambda r: stat_str('t', int(r['df']), r['t'], r['p']), axis=1)
    table['stat_str_fdr'] = table.apply(
        lambda r: stat_str_fdr('t', int(r['df']), r['t'], r['p'], r['p_fdr']), axis=1)

    table['t'] = table['t'].map('{:.2f}'.format)
    # Format every float column whose name starts with 'p' (catches p, p_fdr, p_val_fdr, etc.)
    for col in list(table.columns):
        if col.startswith('p') and hasattr(table[col], 'dtype') and str(table[col].dtype).startswith('float'):
            table[col] = table[col].map(fmt_p)

    fname = filename or f'ttests_{label}.tsv'
    table.to_csv(os.path.join(out_dir, fname), sep='\t', index=False)
    return table


def export_pg_anova(aov_pg, label, out_dir, filename=None):
    """
    Convert a pingouin rm_anova DataFrame to a clean TSV.
    When GG correction was applied (eps < 1), reports GG-adjusted dfs and epsilon.
    Falls back to uncorrected dfs and p-unc when correction is not needed.

    Parameters
    ----------
    aov_pg : pd.DataFrame, output of pg.rm_anova(detailed=True)
    label : str, used in filename if filename not provided
    out_dir : str, directory to save TSV
    filename : str, optional override for output filename
    """
    import math
    eta_col = 'np2' if 'np2' in aov_pg.columns else 'ng2'
    cols = ['Source', 'ddof1', 'ddof2', 'F', 'p-unc', 'p-GG-corr', 'eps', eta_col]
    table = aov_pg[[c for c in cols if c in aov_pg.columns]].copy()
    table = table.rename(columns={'Source': 'source', 'ddof1': 'df_num',
                                  'ddof2': 'df_den', 'p-unc': 'p_unc',
                                  'p-GG-corr': 'p_gg', 'eps': 'epsilon',
                                  eta_col: 'eta_p2'})
    p_report = table['p_gg'].where(table['p_gg'].notna(), table['p_unc'])

    def _stat_str_row(r):
        p = p_report[r.name]
        eps = r.get('epsilon', float('nan'))
        gg_applied = ('p_gg' in r and r['p_gg'] == r['p_gg']  # not NaN
                      and 'epsilon' in r and eps == eps and eps < 0.999)
        if gg_applied:
            df1 = r['df_num'] * eps
            df2 = r['df_den'] * eps
            return (f"F({df1:.2f}, {df2:.2f}) = {r['F']:.2f}, "
                    f"{fmt_p(float(p))}, ε = {eps:.2f}")
        return stat_str('F', int(r['df_num']), int(r['df_den']), r['F'], float(p))

    table['stat_str'] = table.apply(_stat_str_row, axis=1)
    table['F'] = table['F'].map('{:.2f}'.format)
    table['eta_p2'] = table['eta_p2'].map(
        lambda x: f'{x:.2f}' if (x == x and x is not None) else x)
    table['p_unc'] = table['p_unc'].map(
        lambda x: fmt_p(float(x)) if (x == x and x is not None) else x)
    if 'p_gg' in table.columns:
        table['p_gg'] = table['p_gg'].map(
            lambda x: fmt_p(float(x)) if (x == x and x is not None) else x)
    if 'epsilon' in table.columns:
        table['epsilon'] = table['epsilon'].map(
            lambda x: f'{x:.3f}' if (x == x and x is not None) else x)
    fname = filename or f'anova_pg_{label}.tsv'
    table.to_csv(os.path.join(out_dir, fname), sep='\t', index=False)
    return table


def fmt_pingouin_anova(aov_df, term_col: str = 'Source') -> dict[str, str]:
    """
    Convert a pingouin ANOVA DataFrame to a dict of formatted strings keyed by term name.
    Each value is ready to paste into manuscript text.

    Parameters
    ----------
    aov_df : pd.DataFrame
        Output of pg.anova(), pg.rm_anova(), or pg.mixed_anova()
    term_col : str
        Column name that contains the effect labels (default 'Source')

    Returns
    -------
    dict mapping effect name -> formatted string, e.g.
        {'learning_stage': 'F(2, 22) = 4.12, p = .038'}
    """
    out = {}
    for _, row in aov_df.iterrows():
        name = row[term_col]
        # pingouin uses 'ddof1'/'ddof2' or 'DF' depending on the test
        df1 = int(row.get('ddof1', row.get('DF', '?')))
        df2_key = 'ddof2' if 'ddof2' in row else ('DF2' if 'DF2' in row else None)
        df2 = int(row[df2_key]) if df2_key else '?'
        F = row.get('F', float('nan'))
        p = row.get('p-unc', row.get('p-GG-corr', float('nan')))
        out[name] = stat_str('F', df1, df2, F, p)
    return out
