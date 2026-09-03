import sys
import os
import re
import csv
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from scipy.signal import find_peaks

WAVELENGTHS = np.arange(900, 1700 + 0.5, 0.5)
N_PEAKS = 5
CHUNK_SIZE = 10000

PEAK_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
CLASS_LABELS = {0: 'normal', 1: 'ring', 2: 'trophozoite', 3: 'schizont'}


def darken_color(color, factor=0.35):
    """Return a darker shade of the given color by scaling RGB toward black."""
    r, g, b = to_rgb(color)
    return (r * factor, g * factor, b * factor)


def parse_perturbation(filename):
    """Extract perturbation percentage string from dataset filename.

    'cancer_dataset_1M_7_5.h5' -> '7.5'
    'cancer_dataset_1M_1.h5'   -> '1'
    'malaria_stages.h5' -> '0.5'
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    # Check for cancer dataset pattern
    match = re.search(r'_1M_(.+)$', base)
    if match:
        return match.group(1).replace('_', '.')
    # Check for malaria dataset
    if 'malaria' in base.lower():
        return '0.5'
    return 'unknown'


def parse_cell_type_from_filename(h5_path):
    """Extract cell type from dataset filename for folder naming.

    'cancer_dataset_breast_MCF7_1M_0_5.h5' -> 'breast_MCF7'
    'malaria_stages.h5' -> 'malaria_stages_blood'
    """
    basename = os.path.basename(h5_path)
    # Check for malaria dataset
    if 'malaria' in basename.lower():
        return 'malaria_stages_blood'
    # Check for cancer dataset pattern
    match = re.match(r'cancer_dataset_(.+?)_\d+[MKk]_', basename)
    if match:
        return match.group(1)
    return 'unknown'




def find_dips(spectrum, n_peaks=N_PEAKS, min_distance=50, prominence=0.005):
    """Find the n_peaks deepest dips (local minima) in a reflectance spectrum.

    Returns sorted wavelength indices of the dips.
    """
    peaks, props = find_peaks(-spectrum, distance=min_distance, prominence=prominence)
    if len(peaks) == 0:
        return np.array([], dtype=int)
    if len(peaks) > n_peaks:
        top_idx = np.argsort(props['prominences'])[-n_peaks:]
        peaks = np.sort(peaks[top_idx])
    return peaks


def process_dataset(h5_path, plot_cfg=None):
    """Load dataset in chunks and extract peak positions."""
    if plot_cfg is None:
        plot_cfg = {}
    n_peaks = plot_cfg.get('n_peaks', N_PEAKS)
    chunk_size = plot_cfg.get('chunk_size', CHUNK_SIZE)
    min_distance = plot_cfg.get('min_peak_distance', 50)
    prominence = plot_cfg.get('peak_prominence', 0.005)

    with h5py.File(h5_path, 'r') as f:
        total = f['Reflectance'].shape[0]
        class_idx_all = f['labels'][:].flatten().astype(int)
        
        # Load wavelengths from file or use provided ones
        if '_wavelengths' in plot_cfg:
            wavelengths = plot_cfg['_wavelengths']
        else:
            wavelengths = f['wavelengths'][:]

        peak_positions = np.full((total, n_peaks), np.nan)

        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            R_chunk = f['Reflectance'][start:end]

            for i in range(R_chunk.shape[0]):
                dips = find_dips(R_chunk[i], n_peaks=n_peaks,
                                 min_distance=min_distance, prominence=prominence)
                n_found = len(dips)
                if n_found > 0:
                    peak_positions[start + i, :n_found] = wavelengths[dips]

            print(f'  Processed {end}/{total} samples', flush=True)

    return peak_positions, class_idx_all


def compute_peak_stats(peak_data, class_idx, plot_cfg=None):
    """Compute mean and std for each peak per class.

    Returns a list of dicts with keys: peak, class_id, class_name, count, mean, std.
    """
    if plot_cfg is None:
        plot_cfg = {}
    n_peaks = plot_cfg.get('n_peaks', N_PEAKS)
    class_labels = plot_cfg.get('class_labels', CLASS_LABELS)
    unique_classes = np.unique(class_idx)

    stats = []
    for p in range(n_peaks):
        for cls in unique_classes:
            data = peak_data[class_idx == cls, p]
            data = data[~np.isnan(data)]
            stats.append({
                'peak': p + 1,
                'class_id': int(cls),
                'class_name': class_labels.get(cls, f'Class {cls}'),
                'count': len(data),
                'mean': float(np.mean(data)) if len(data) > 0 else np.nan,
                'std': float(np.std(data)) if len(data) > 0 else np.nan,
            })
    return stats


def write_peak_stats_csv(stats, csv_path, cell_type, perturbation_str):
    """Write peak statistics to a CSV file."""
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['cell_type', 'perturbation_pct', 'peak',
                         'class_id', 'class_name', 'count', 'mean_nm', 'std_nm'])
        for s in stats:
            writer.writerow([cell_type, perturbation_str, s['peak'],
                             s['class_id'], s['class_name'], s['count'],
                             f"{s['mean']:.2f}" if not np.isnan(s['mean']) else '',
                             f"{s['std']:.2f}" if not np.isnan(s['std']) else ''])


def plot_overlaid_histograms(peak_data, class_idx, perturbation_str, cell_type,
                             xlabel, title_prefix, save_path, plot_cfg=None,
                             stats=None):
    """Plot overlaid histograms of normal vs cancer for all peaks."""
    if plot_cfg is None:
        plot_cfg = {}
    n_peaks = plot_cfg.get('n_peaks', N_PEAKS)
    hist_bins = plot_cfg.get('hist_bins', 80)
    class_labels = plot_cfg.get('class_labels', CLASS_LABELS)

    # Index stats by (peak, class_id) for quick lookup
    stats_map = {}
    if stats:
        for s in stats:
            stats_map[(s['peak'], s['class_id'])] = s

    fig, ax = plt.subplots(figsize=(9, 5))

    unique_classes = np.unique(class_idx)

    for p in range(n_peaks):
        color = PEAK_COLORS[p % len(PEAK_COLORS)]
        for cls in unique_classes:
            mask = class_idx == cls
            data = peak_data[mask, p]
            data = data[~np.isnan(data)]
            if len(data) == 0:
                continue

            s = stats_map.get((p + 1, int(cls)))
            base_label = f'{class_labels.get(cls, f"Class {cls}")} Peak {p + 1}'
            if s and not np.isnan(s['mean']):
                label = f'{base_label} (\u03bc={s["mean"]:.1f}, \u03c3={s["std"]:.1f})'
            else:
                label = base_label

            if cls == 0:  # Normal: lighter fill
                ax.hist(data, bins=hist_bins, alpha=0.4, color=color,
                        edgecolor=color, linewidth=1.5, label=label)
                if s and not np.isnan(s['mean']):
                    ax.axvline(s['mean'], color=color, linestyle='--',
                               linewidth=1.0, alpha=0.7)
            else:  # Cancer: darker fill
                dark = darken_color(color)
                ax.hist(data, bins=hist_bins, alpha=0.5, color=dark,
                        edgecolor=dark, linewidth=1.5, label=label)
                if s and not np.isnan(s['mean']):
                    ax.axvline(s['mean'], color=dark, linestyle='--',
                               linewidth=1.0, alpha=0.7)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(
        f'{title_prefix} \u2014 {perturbation_str}% perturbation, {cell_type}',
        fontsize=13, fontweight='bold',
    )
    # Only create legend if there are labeled artists
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 0:
        ax.legend(frameon=True, fontsize=8, ncol=2)
    ax.grid(True, color='lightgray', linewidth=0.5)
    ax.tick_params(direction='in')
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_distributions(h5_path, plot_cfg=None, wl_cfg=None):
    """Main entry point: process dataset and create distribution plots."""
    if plot_cfg is None:
        plot_cfg = {}

    # Build wavelengths array from config
    if wl_cfg is None:
        wl_cfg = {}
    wl_range = wl_cfg.get('range', [900, 1700])
    wl_step = wl_cfg.get('step', 0.5)
    wavelengths = np.arange(wl_range[0], wl_range[1] + wl_step, wl_step)
    plot_cfg['_wavelengths'] = wavelengths

    perturbation_str = parse_perturbation(h5_path)
    cell_type = cell_type_folder = parse_cell_type_from_filename(h5_path)
    print(f'Dataset: {h5_path}')
    print(f'Perturbation: {perturbation_str}%')
    print(f'Cell type: {cell_type}')
    print('Finding peaks...', flush=True)

    peak_positions, class_idx = process_dataset(h5_path, plot_cfg)

    base_dir = plot_cfg.get('save_dir', os.path.join(os.path.dirname(__file__), 'plots'))
    save_dir = os.path.join(base_dir, cell_type_folder)
    os.makedirs(save_dir, exist_ok=True)

    suffix = f'{cell_type_folder}_{perturbation_str}pct'

    stats = compute_peak_stats(peak_positions, class_idx, plot_cfg)

    plot_overlaid_histograms(
        peak_positions, class_idx, perturbation_str, cell_type,
        xlabel='Wavelength (nm)',
        title_prefix='Peak Position Distribution',
        save_path=os.path.join(save_dir, f'peak_positions_{suffix}.png'),
        plot_cfg=plot_cfg,
        stats=stats,
    )
    print('  Peak positions plot: done', flush=True)

    csv_path = os.path.join(save_dir, f'peak_stats_{suffix}.csv')
    write_peak_stats_csv(stats, csv_path, cell_type, perturbation_str)
    print(f'  Peak stats CSV: {csv_path}', flush=True)

    print(f'All plots saved to {save_dir}')


if __name__ == '__main__':
    h5_path = sys.argv[1] if len(sys.argv) > 1 else '/home/user2/BloodSensor/DataSet/malaria_stages.h5'
    plot_distributions(h5_path)
