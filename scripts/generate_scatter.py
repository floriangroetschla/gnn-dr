"""
Generate UMAP scatter plot as a standalone image for TikZ inclusion.

Reads pre-exported data from export_data_for_tikz.py.

Usage:
    python scripts/generate_scatter.py
    python scripts/generate_scatter.py --input-dir tikz_data --output tikz_data/umap_scatter.pdf
"""
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Professional color palette
COLORS = {
    0: '#3182bd', 1: '#e6550d', 2: '#31a354', 3: '#de2d26', 4: '#756bb1',
    5: '#8c6d31', 6: '#d53e4f', 7: '#636363', 8: '#969696', 9: '#6baed6',
}

# Set publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.linewidth': 0,
    'xtick.major.size': 0,
    'ytick.major.size': 0,
})


def main():
    parser = argparse.ArgumentParser(description='Generate UMAP scatter plot from exported data')
    parser.add_argument('--input-dir', default='tikz_data',
                        help='Directory with umap_digit_*.dat files')
    parser.add_argument('--output', default=None,
                        help='Output path (default: <input-dir>/umap_scatter.pdf)')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output = args.output or str(input_dir / 'umap_scatter.pdf')

    fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5), dpi=300)

    # Load and plot each digit class
    for digit in range(10):
        data_file = input_dir / f'umap_digit_{digit}.dat'
        if not data_file.exists():
            print(f"Warning: {data_file} not found, skipping digit {digit}")
            continue
        data = np.loadtxt(data_file, comments='#')
        if len(data) > 0:
            ax.scatter(data[:, 0], data[:, 1], c=COLORS[digit], s=2, alpha=0.7,
                      edgecolors='none', rasterized=True)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout(pad=0)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, bbox_inches='tight', pad_inches=0,
               facecolor='white', edgecolor='none', transparent=True)

    # Also save PNG
    png_output = str(Path(output).with_suffix('.png'))
    plt.savefig(png_output, bbox_inches='tight', pad_inches=0, dpi=300,
               facecolor='white', edgecolor='none', transparent=True)
    plt.close()

    print(f"Generated {output} and {png_output}")


if __name__ == '__main__':
    main()
