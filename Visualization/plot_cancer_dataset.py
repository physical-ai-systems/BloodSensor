import sys
import os
import re
import h5py
import numpy as np
import matplotlib.pyplot as plt


def parse_dataset_filename(h5_path):
    basename = os.path.basename(h5_path)
    # Check for malaria dataset
    if 'malaria' in basename.lower():
        return 'malaria_stages_blood', '0.5'
    # Check for cancer dataset pattern
    match = re.match(r'cancer_dataset_(.+?)_\d+[MKk]_(\d+(?:_\d+)?)\.h5', basename)
    if match:
        cell_type = match.group(1)
        perturbation = match.group(2).replace('_', '.')
    else:
        cell_type = "unknown"
        perturbation = "unknown"
    return cell_type, perturbation


def plot_random_samples(h5_path, num_samples=3, plot_cfg=None, wl_cfg=None):
    cell_type, perturbation = parse_dataset_filename(h5_path)

    with h5py.File(h5_path, 'r') as f:
        # Handle both cancer dataset ('class_idx', 'R') and malaria dataset ('labels', 'Reflectance')
        if 'labels' in f:
            class_idx = f['labels'][:]
            reflectance_key = 'Reflectance'
        else:
            class_idx = f['class_idx'][:]
            reflectance_key = 'R'
        
        # Get unique classes
        unique_classes = np.unique(class_idx)
        
        # For malaria: normal(0), ring(1), trophozoite(2), schizont(3)
        # For cancer: normal(0), cancer(1)
        class_names = {
            0: 'normal',
            1: 'ring' if len(unique_classes) > 2 else 'cancer',
            2: 'trophozoite',
            3: 'schizont'
        }
        
        # Collect samples for each class
        samples_by_class = {}
        for cls in unique_classes:
            indices = np.where(class_idx == cls)[0]
            pick = np.sort(np.random.choice(indices, num_samples, replace=False))
            samples_by_class[cls] = f[reflectance_key][pick]

    if wl_cfg is None:
        wl_cfg = {}
    wl_range = wl_cfg.get('range', [900, 1700])
    wl_step = wl_cfg.get('step', 0.5)
    
    # Load wavelengths from file if available
    with h5py.File(h5_path, 'r') as f:
        if 'wavelengths' in f:
            wavelengths = f['wavelengths'][:]
        else:
            wavelengths = np.arange(wl_range[0], wl_range[1] + wl_step, wl_step)

    if plot_cfg is None:
        plot_cfg = {}
    save_dir = plot_cfg.get('save_dir', os.path.join(os.path.dirname(__file__), 'plots'))
    save_dir = os.path.join(save_dir, cell_type)

    dataset_type = "malaria stages" if "malaria" in h5_path.lower() else "cancer"
    title = f"Reflectance of {cell_type} — {perturbation}% perturbation ({dataset_type})"

    # Define colors for each class
    color_maps = {
        0: plt.cm.Blues(np.linspace(0.4, 0.8, num_samples)),      # Normal: blues
        1: plt.cm.Reds(np.linspace(0.4, 0.8, num_samples)),       # Ring/Cancer: reds
        2: plt.cm.Greens(np.linspace(0.4, 0.8, num_samples)),     # Trophozoite: greens
        3: plt.cm.Purples(np.linspace(0.4, 0.8, num_samples)),    # Schizont: purples
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    for cls in sorted(unique_classes):
        colors = color_maps.get(cls, plt.cm.gray(np.linspace(0.4, 0.8, num_samples)))
        for i in range(num_samples):
            ax.plot(wavelengths, samples_by_class[cls][i], linewidth=1.5,
                    color=colors[i], label=f"{class_names[cls].capitalize()} Sample {i + 1}")

    ax.set_xlabel("Wavelength (nm)", fontsize=12)
    ax.set_ylabel("Reflectance", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, color='lightgray', linewidth=0.5)
    ax.tick_params(direction='in')
    fig.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    filename = f"samples_{cell_type}_{perturbation}_perturbation.png"
    fig.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.close(fig)
    print(f"Plot saved to {os.path.join(save_dir, filename)}")


if __name__ == "__main__":
    h5_path = sys.argv[1] if len(sys.argv) > 1 else "/home/user2/BloodSensor/DataSet/malaria_stages.h5"
    plot_random_samples(h5_path)
