"""
Analyze and compare dimensionality reduction baselines vs CoRe-DR.

Usage:
    python scripts/analyze_results.py
    python scripts/analyze_results.py --latex    # Also output LaTeX table
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Metrics to report (display name -> column name)
# Selected for relevance to DR quality evaluation
METRICS = {
    'Trust.': 'trustworthiness',
    'Cont.': 'continuity',
    'NH': 'neighborhood_hit',
    'Jaccard': 'jaccard',
    'Shep.': 'shepard_goodness',
    'Dist. Cons.': 'distance_consistency',
    'Norm. Stress': 'normalized_stress',
    'SNS': 'scale_normalized_stress',
}

# Higher is better for these metrics; lower is better for the rest
HIGHER_IS_BETTER = {
    'trustworthiness', 'continuity', 'neighborhood_hit', 'jaccard',
    'shepard_goodness', 'distance_consistency',
    'class_aware_trustworthiness', 'class_aware_continuity',
    'pearson_correlation',
}

DATASET_SHORT = {
    'mnist_clip': 'MNIST',
    'cifar10_clip': 'CIFAR-10',
    'fashion_mnist_clip': 'F-MNIST',
    'kmnist_clip': 'KMNIST',
    'flowers102_clip': 'Flowers',
    'fgvc_aircraft_clip': 'Aircraft',
    'oxford_pets_clip': 'Pets',
    'food101_clip': 'Food-101',
}


def load_and_merge(baselines_path: str, gnn_path: str,
                   pumap_path: str = None,
                   gnn2_path: str = None, gnn2_label: str = 'CoRe-DR-v2') -> pd.DataFrame:
    """Load CSVs and merge into a single DataFrame."""
    df_base = pd.read_csv(baselines_path)
    df_gnn = pd.read_csv(gnn_path)

    frames = [df_base, df_gnn]
    if pumap_path:
        df_pumap = pd.read_csv(pumap_path)
        # Drop old P-UMAP entries from baselines to avoid duplicates
        df_base = df_base[df_base['method'] != 'Parametric_UMAP']
        frames = [df_base, df_gnn, df_pumap]

    if gnn2_path:
        df_gnn2 = pd.read_csv(gnn2_path)
        df_gnn2['method'] = gnn2_label
        frames.append(df_gnn2)

    # Keep only common columns
    common_cols = list(set.intersection(*(set(f.columns) for f in frames)))
    df = pd.concat([f[common_cols] for f in frames], ignore_index=True)
    return df


def print_summary(df: pd.DataFrame):
    """Print a high-level summary of the results."""
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Methods:  {sorted(df['method'].unique())}")
    print(f"Datasets: {sorted(df['dataset'].unique())}")
    print(f"Metric columns available: {[c for c in METRICS.values() if c in df.columns]}")
    print()


def print_per_dataset_tables(df: pd.DataFrame):
    """Print metric comparison tables per dataset."""
    metric_cols = {name: col for name, col in METRICS.items() if col in df.columns}

    for dataset in sorted(df['dataset'].unique()):
        ds_short = DATASET_SHORT.get(dataset, dataset)
        subset = df[df['dataset'] == dataset].copy()
        subset = subset.set_index('method')

        print(f"\n--- {ds_short} (n={int(subset['n_samples'].iloc[0])}) ---")

        # Build display table
        table = pd.DataFrame(index=subset.index)
        for display_name, col_name in metric_cols.items():
            if col_name in subset.columns:
                table[display_name] = subset[col_name]

        # Format and mark best
        formatted = table.copy().astype(str)
        for display_name, col_name in metric_cols.items():
            if display_name not in table.columns:
                continue
            col = table[display_name]
            valid = col.dropna()
            if valid.empty:
                continue
            best_idx = valid.idxmax() if col_name in HIGHER_IS_BETTER else valid.idxmin()
            for idx in table.index:
                val = table.loc[idx, display_name]
                if pd.isna(val):
                    formatted.loc[idx, display_name] = '  ---  '
                else:
                    marker = ' *' if idx == best_idx else '  '
                    formatted.loc[idx, display_name] = f'{val:.4f}{marker}'

        print(formatted.to_string())
    print("\n(* = best per metric)")


def print_aggregate_ranking(df: pd.DataFrame):
    """Print aggregate ranking across datasets."""
    metric_cols = {name: col for name, col in METRICS.items() if col in df.columns}
    methods = sorted(df['method'].unique())

    # Count wins per method
    wins = {m: 0 for m in methods}
    total = 0

    for dataset in df['dataset'].unique():
        subset = df[df['dataset'] == dataset]
        for _, col_name in metric_cols.items():
            if col_name not in subset.columns:
                continue
            valid = subset[['method', col_name]].dropna(subset=[col_name])
            if valid.empty:
                continue
            if col_name in HIGHER_IS_BETTER:
                best_method = valid.loc[valid[col_name].idxmax(), 'method']
            else:
                best_method = valid.loc[valid[col_name].idxmin(), 'method']
            wins[best_method] += 1
            total += 1

    print("\n" + "=" * 70)
    print("AGGREGATE RANKING (wins across all dataset-metric combinations)")
    print("=" * 70)
    for method, count in sorted(wins.items(), key=lambda x: -x[1]):
        pct = 100 * count / total if total > 0 else 0
        print(f"  {method:20s}: {count:3d} / {total}  ({pct:.1f}%)")


def print_mean_metrics(df: pd.DataFrame):
    """Print mean metrics across datasets per method."""
    metric_cols = {name: col for name, col in METRICS.items() if col in df.columns}

    # Only use datasets present for all methods
    datasets_per_method = df.groupby('method')['dataset'].apply(set)
    common_datasets = set.intersection(*datasets_per_method)

    df_common = df[df['dataset'].isin(common_datasets)]

    print("\n" + "=" * 70)
    print(f"MEAN METRICS (across {len(common_datasets)} common datasets)")
    print(f"Datasets: {sorted(common_datasets)}")
    print("=" * 70)

    means = df_common.groupby('method')[[col for col in metric_cols.values()]].mean()
    means.columns = [name for name, col in metric_cols.items() if col in means.columns or True][:len(means.columns)]

    # Re-map column names properly
    col_remap = {}
    for display_name, col_name in metric_cols.items():
        if col_name in means.columns:
            col_remap[col_name] = display_name
    means = means.rename(columns=col_remap)

    # Format with best markers
    formatted = means.copy().astype(str)
    for display_name, col_name in metric_cols.items():
        if display_name not in means.columns:
            continue
        col = means[display_name]
        valid = col.dropna()
        if valid.empty:
            continue
        best_idx = valid.idxmax() if col_name in HIGHER_IS_BETTER else valid.idxmin()
        for idx in means.index:
            val = means.loc[idx, display_name]
            if pd.isna(val):
                formatted.loc[idx, display_name] = '  ---  '
            else:
                marker = ' *' if idx == best_idx else '  '
                formatted.loc[idx, display_name] = f'{val:.4f}{marker}'

    print(formatted.to_string())
    print("\n(* = best)")


def print_timing(df: pd.DataFrame):
    """Print timing comparison."""
    print("\n" + "=" * 70)
    print("TIMING (seconds)")
    print("=" * 70)

    time_col = 'time_seconds'
    if time_col not in df.columns:
        print("  No timing data available.")
        return

    pivot = df.pivot_table(index='dataset', columns='method', values=time_col)
    pivot.index = [DATASET_SHORT.get(d, d) for d in pivot.index]

    print(pivot.round(2).to_string())

    print("\nMean time per method:")
    for method in sorted(df['method'].unique()):
        mean_t = df[df['method'] == method][time_col].mean()
        print(f"  {method:20s}: {mean_t:.2f}s")


def generate_latex_table(df: pd.DataFrame) -> str:
    """Generate a single-column LaTeX table with grouped metric headers."""
    methods_order = ['PCA', 't-SNE', 'UMAP', 'Parametric_UMAP', 'CoRe-DR']
    methods_display = {
        'PCA': 'PCA', 't-SNE': 't-SNE', 'UMAP': 'UMAP',
        'Parametric_UMAP': 'P-UMAP',
        'CoRe-DR': r'\textbf{Ours}',
    }

    # Metrics grouped by category
    local_metrics = {
        'Tr.': 'trustworthiness',
        'Co.': 'continuity',
        'NH': 'neighborhood_hit',
    }
    global_metrics = {
        'Sh.': 'shepard_goodness',
        'SNS': 'scale_normalized_stress',
    }
    local_metrics = {k: v for k, v in local_metrics.items() if v in df.columns}
    global_metrics = {k: v for k, v in global_metrics.items() if v in df.columns}
    all_metrics = {**local_metrics, **global_metrics}

    has_timing = 'time_seconds' in df.columns

    # Training datasets first, then held-out
    datasets_order = [
        'mnist_clip', 'fashion_mnist_clip', 'cifar10_clip',
        'kmnist_clip', 'fgvc_aircraft_clip', 'oxford_pets_clip', 'food101_clip',
    ]
    datasets_order = [d for d in datasets_order if d in df['dataset'].unique()]
    n_training = 3  # first 3 are training datasets

    n_local = len(local_metrics)
    n_global = len(global_metrics)
    n_cols = 2 + n_local + n_global + (1 if has_timing else 0)
    col_spec = '@{}ll' + 'c' * n_local + 'c' * n_global
    if has_timing:
        col_spec += 'r'
    col_spec += '@{}'

    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering')
    lines.append(r'\caption{Quantitative comparison. '
                 r'\textbf{Bold}: best, \underline{underline}: second best per metric per dataset. '
                 r'Time includes $k$-NN graph construction (CPU: AMD EPYC 7742; Ours additionally uses an NVIDIA A6000 GPU).}')
    lines.append(r'\label{tab:results}')
    lines.append(r'\footnotesize')
    lines.append(r'\setlength{\tabcolsep}{3pt}')
    lines.append(r'\begin{tabular}{' + col_spec + '}')
    lines.append(r'\toprule')

    # Grouped header row
    group_parts = ['', '']
    group_parts.append(r'\multicolumn{' + str(n_local) +
                       r'}{c}{\textit{Local}}')
    group_parts.append(r'\multicolumn{' + str(n_global) +
                       r'}{c}{\textit{Global}}')
    if has_timing:
        group_parts.append('')
    lines.append(' & '.join(group_parts) + r' \\')

    # cmidrule under group headers
    local_start = 3  # 1-indexed column
    global_start = local_start + n_local
    cmidrule = (r'\cmidrule(lr){' + str(local_start) + '-' +
                str(local_start + n_local - 1) + '}')
    cmidrule += (r' \cmidrule(lr){' + str(global_start) + '-' +
                 str(global_start + n_global - 1) + '}')
    lines.append(cmidrule)

    # Metric names header row — each metric gets its own arrow
    header_parts = ['', 'Method']
    for name, col in local_metrics.items():
        arrow = r'$\uparrow$' if col in HIGHER_IS_BETTER else r'$\downarrow$'
        header_parts.append(f'{name}\\,{arrow}')
    for name, col in global_metrics.items():
        arrow = r'$\uparrow$' if col in HIGHER_IS_BETTER else r'$\downarrow$'
        header_parts.append(f'{name}\\,{arrow}')
    if has_timing:
        header_parts.append(r'\!\!Time\,(s)')
    lines.append(' & '.join(header_parts) + r' \\')
    lines.append(r'\midrule')

    for ds_idx, ds in enumerate(datasets_order):
        # Insert held-out separator
        if ds_idx == n_training:
            lines.append(r'\midrule')
            lines.append(r'\multicolumn{' + str(n_cols) +
                         r'}{@{}l}{\textit{Held-out (unseen during training):}} \\')
            lines.append(r'\midrule')

        ds_short = DATASET_SHORT.get(ds, ds)
        subset = df[(df['dataset'] == ds) & (df['method'].isin(methods_order))]
        if subset.empty:
            continue

        # Get dataset size for label
        n_samples = int(subset['n_samples'].iloc[0]) if 'n_samples' in subset.columns else None
        if n_samples is not None:
            if n_samples >= 10000:
                size_str = f'{n_samples // 1000}k'
            elif n_samples >= 1000:
                size_str = f'{n_samples / 1000:.1f}k'
            else:
                size_str = str(n_samples)
            ds_label = ds_short + r' {\scriptsize(' + size_str + r')}'
        else:
            ds_label = ds_short

        # Find best and second-best per metric
        bests = {}
        second_bests = {}
        for name, col in all_metrics.items():
            if col not in subset.columns:
                continue
            valid = subset[['method', col]].dropna(subset=[col])
            if valid.empty:
                continue
            if col in HIGHER_IS_BETTER:
                sorted_valid = valid.sort_values(col, ascending=False)
            else:
                sorted_valid = valid.sort_values(col, ascending=True)
            bests[name] = sorted_valid.iloc[0]['method']
            if len(sorted_valid) > 1:
                second_bests[name] = sorted_valid.iloc[1]['method']

        # Best/second-best time (exclude PCA — trivial linear projection)
        if has_timing:
            nonlinear = subset[subset['method'] != 'PCA']
            valid_time = nonlinear[['method', 'time_seconds']].dropna(
                subset=['time_seconds'])
            if not valid_time.empty:
                sorted_time = valid_time.sort_values('time_seconds')
                bests['time'] = sorted_time.iloc[0]['method']
                if len(sorted_time) > 1:
                    second_bests['time'] = sorted_time.iloc[1]['method']

        n_methods = sum(
            1 for m in methods_order
            if not subset[subset['method'] == m].empty)

        for i, method in enumerate(methods_order):
            row_data = subset[subset['method'] == method]
            if row_data.empty:
                continue

            row_parts = []
            if i == 0:
                row_parts.append(
                    r'\multirow{' + str(n_methods) + '}{*}{' +
                    ds_label + '}')
            else:
                row_parts.append('')

            row_parts.append(methods_display.get(method, method))

            for name, col in all_metrics.items():
                val = row_data[col].values[0]
                if pd.isna(val):
                    row_parts.append('--')
                else:
                    s = f'{val:.3f}'
                    if bests.get(name) == method:
                        s = r'\textbf{' + s + '}'
                    elif second_bests.get(name) == method:
                        s = r'\underline{' + s + '}'
                    row_parts.append(s)

            if has_timing:
                time_val = row_data['time_seconds'].values[0]
                if pd.isna(time_val):
                    row_parts.append('--')
                else:
                    if time_val < 1.0:
                        s = f'{max(time_val, 0.1):.1f}'
                    else:
                        s = f'{time_val:.0f}'
                    if bests.get('time') == method:
                        s = r'\textbf{' + s + '}'
                    elif second_bests.get('time') == method:
                        s = r'\underline{' + s + '}'
                    row_parts.append(s)

            lines.append(' & '.join(row_parts) + r' \\')

        # Separator between datasets within the same group
        if ds_idx < len(datasets_order) - 1 and ds_idx != n_training - 1:
            lines.append(r'\midrule')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    return '\n'.join(lines)


def generate_appendix_table(df: pd.DataFrame) -> str:
    """Generate a full-metric appendix table (one sub-table per dataset, landscape)."""
    # Include CoRe-DR-v2 only if present in the data
    has_v2 = 'CoRe-DR-v2' in df['method'].unique()
    methods_order = ['PCA', 't-SNE', 'UMAP', 'Parametric_UMAP', 'CoRe-DR']
    methods_display = {
        'PCA': 'PCA', 't-SNE': 't-SNE', 'UMAP': 'UMAP',
        'Parametric_UMAP': 'P-UMAP',
        'CoRe-DR': r'\textbf{Ours}',
    }
    if has_v2:
        methods_order.append('CoRe-DR-v2')
        methods_display['CoRe-DR-v2'] = r'\textbf{Ours-v2}'

    # All metrics grouped
    all_metrics_groups = [
        ('Local', {
            'Tr.': 'trustworthiness',
            'Co.': 'continuity',
            'NH': 'neighborhood_hit',
            'Jacc.': 'jaccard',
            'CA-Tr.': 'class_aware_trustworthiness',
            'CA-Co.': 'class_aware_continuity',
        }),
        ('Global', {
            'Sh.': 'shepard_goodness',
            'DC': 'distance_consistency',
            r'N.\,Str.': 'normalized_stress',
            'SNS': 'scale_normalized_stress',
            'Pears.': 'pearson_correlation',
        }),
        ('Error', {
            'ALE': 'average_local_error',
            r'MRRE$_d$': 'mrre_data',
            r'MRRE$_p$': 'mrre_proj',
        }),
    ]

    # Filter to columns that exist
    filtered_groups = []
    for group_name, metrics in all_metrics_groups:
        filtered = {k: v for k, v in metrics.items() if v in df.columns}
        if filtered:
            filtered_groups.append((group_name, filtered))

    # Flatten for iteration
    all_metrics = {}
    for _, metrics in filtered_groups:
        all_metrics.update(metrics)

    # Datasets: training first, then held-out
    datasets_order = [
        'mnist_clip', 'fashion_mnist_clip', 'cifar10_clip',
        'kmnist_clip', 'fgvc_aircraft_clip', 'oxford_pets_clip', 'food101_clip',
    ]
    datasets_order = [d for d in datasets_order if d in df['dataset'].unique()]
    n_training = 3

    # Build column spec
    n_metric_cols = sum(len(m) for _, m in filtered_groups)
    col_spec = '@{}ll'
    for _, metrics in filtered_groups:
        col_spec += 'c' * len(metrics)
    col_spec += '@{}'

    lines = []
    lines.append(r'\begin{table*}[t]')
    lines.append(r'\centering')
    lines.append(r'\caption{Full quantitative comparison across all metrics. '
                 r'\textbf{Bold}: best, \underline{underline}: second best per metric per dataset. '
                 r'Arrows indicate direction ($\uparrow$\,=\,higher is better, $\downarrow$\,=\,lower is better). '
                 r'Metrics: Trustworthiness (Tr.), Continuity (Co.), Neighborhood Hit (NH), '
                 r'Jaccard index (Jacc.), Class-Aware Trustworthiness (CA-Tr.), Class-Aware Continuity (CA-Co.), '
                 r'Shepard goodness (Sh.), Distance Consistency (DC), Normalized Stress (N.\,Str.), '
                 r'Scale-Normalized Stress (SNS), Pearson correlation (Pears.), '
                 r'Average Local Error (ALE), MRRE in data space (MRRE$_d$), MRRE in projection space (MRRE$_p$).}')
    lines.append(r'\label{tab:full_results}')
    lines.append(r'\scriptsize')
    lines.append(r'\setlength{\tabcolsep}{2.5pt}')
    lines.append(r'\begin{tabular}{' + col_spec + '}')
    lines.append(r'\toprule')

    # Group header row
    group_parts = ['', '']
    for group_name, metrics in filtered_groups:
        group_parts.append(r'\multicolumn{' + str(len(metrics)) +
                           r'}{c}{\textit{' + group_name + '}}')
    lines.append(' & '.join(group_parts) + r' \\')

    # cmidrules
    col_idx = 3  # 1-indexed, after dataset + method
    cmidrule_parts = []
    for _, metrics in filtered_groups:
        end = col_idx + len(metrics) - 1
        cmidrule_parts.append(r'\cmidrule(lr){' + str(col_idx) + '-' + str(end) + '}')
        col_idx = end + 1
    lines.append(' '.join(cmidrule_parts))

    # Metric names header
    header_parts = ['', 'Method']
    for _, metrics in filtered_groups:
        for name, col in metrics.items():
            arrow = r'$\uparrow$' if col in HIGHER_IS_BETTER else r'$\downarrow$'
            header_parts.append(f'{name}\\,{arrow}')
    lines.append(' & '.join(header_parts) + r' \\')
    lines.append(r'\midrule')

    n_total_cols = 2 + n_metric_cols

    for ds_idx, ds in enumerate(datasets_order):
        # Insert held-out separator
        if ds_idx == n_training:
            lines.append(r'\midrule')
            lines.append(r'\multicolumn{' + str(n_total_cols) +
                         r'}{@{}l}{\textit{Held-out (unseen during training):}} \\')
            lines.append(r'\midrule')

        ds_short = DATASET_SHORT.get(ds, ds)
        subset = df[(df['dataset'] == ds) & (df['method'].isin(methods_order))]
        if subset.empty:
            continue

        # Dataset size label
        n_samples = int(subset['n_samples'].iloc[0]) if 'n_samples' in subset.columns else None
        if n_samples is not None:
            if n_samples >= 10000:
                size_str = f'{n_samples // 1000}k'
            elif n_samples >= 1000:
                size_str = f'{n_samples / 1000:.1f}k'
            else:
                size_str = str(n_samples)
            ds_label = ds_short + r' {\tiny(' + size_str + r')}'
        else:
            ds_label = ds_short

        # Find best and second-best per metric
        bests = {}
        second_bests = {}
        for name, col in all_metrics.items():
            if col not in subset.columns:
                continue
            valid = subset[['method', col]].dropna(subset=[col])
            if valid.empty:
                continue
            if col in HIGHER_IS_BETTER:
                sorted_valid = valid.sort_values(col, ascending=False)
            else:
                sorted_valid = valid.sort_values(col, ascending=True)
            bests[name] = sorted_valid.iloc[0]['method']
            if len(sorted_valid) > 1:
                second_bests[name] = sorted_valid.iloc[1]['method']

        n_methods = sum(1 for m in methods_order
                        if not subset[subset['method'] == m].empty)

        for i, method in enumerate(methods_order):
            row_data = subset[subset['method'] == method]
            if row_data.empty:
                continue

            row_parts = []
            if i == 0:
                row_parts.append(
                    r'\multirow{' + str(n_methods) + '}{*}{' +
                    ds_label + '}')
            else:
                row_parts.append('')

            row_parts.append(methods_display.get(method, method))

            for name, col in all_metrics.items():
                val = row_data[col].values[0]
                if pd.isna(val):
                    row_parts.append('--')
                else:
                    # Use appropriate formatting based on value magnitude
                    if abs(val) >= 100:
                        s = f'{val:.0f}'
                    elif abs(val) >= 10:
                        s = f'{val:.1f}'
                    else:
                        s = f'{val:.3f}'
                    if bests.get(name) == method:
                        s = r'\textbf{' + s + '}'
                    elif second_bests.get(name) == method:
                        s = r'\underline{' + s + '}'
                    row_parts.append(s)

            lines.append(' & '.join(row_parts) + r' \\')

        # Separator between datasets
        if ds_idx < len(datasets_order) - 1 and ds_idx != n_training - 1:
            lines.append(r'\midrule')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table*}')

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Analyze DR evaluation results')
    parser.add_argument('--baselines', default='results/baselines.csv',
                        help='Path to baselines CSV')
    parser.add_argument('--gnn', default='results/gnn_results.csv',
                        help='Path to GNN results CSV')
    parser.add_argument('--pumap', default=None,
                        help='Path to Parametric UMAP results CSV')
    parser.add_argument('--gnn2', default=None,
                        help='Path to second GNN checkpoint results CSV')
    parser.add_argument('--gnn2-label', default='CoRe-DR-v2',
                        help='Display label for second GNN (default: CoRe-DR-v2)')
    parser.add_argument('--latex', action='store_true',
                        help='Generate LaTeX table')
    parser.add_argument('--latex-out', default='results/results_table.tex',
                        help='Output path for LaTeX table')
    parser.add_argument('--appendix', action='store_true',
                        help='Generate full-metric appendix table')
    parser.add_argument('--appendix-out', default='results/appendix_table.tex',
                        help='Output path for appendix LaTeX table')
    args = parser.parse_args()

    df = load_and_merge(args.baselines, args.gnn, args.pumap,
                        gnn2_path=args.gnn2, gnn2_label=args.gnn2_label)

    print_summary(df)
    print_per_dataset_tables(df)
    print_aggregate_ranking(df)
    print_mean_metrics(df)
    print_timing(df)

    if args.latex:
        latex = generate_latex_table(df)
        Path(args.latex_out).write_text(latex)
        print(f"\nLaTeX table written to: {args.latex_out}")
        print("\n" + latex)

    if args.appendix:
        appendix = generate_appendix_table(df)
        Path(args.appendix_out).write_text(appendix)
        print(f"\nAppendix table written to: {args.appendix_out}")
        print("\n" + appendix)


if __name__ == '__main__':
    main()
