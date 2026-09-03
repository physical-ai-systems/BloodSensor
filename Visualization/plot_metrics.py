import os
import re
import csv
import numpy as np
import matplotlib.pyplot as plt


def plot_metrics_bar(csv_path, plot_cfg=None):
    """Plot bar chart with metrics table from test results CSV."""
    if not os.path.exists(csv_path):
        print(f"Results CSV not found: {csv_path}")
        return

    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("No results found in CSV.")
        return

    if plot_cfg is None:
        plot_cfg = {}
    save_dir = plot_cfg.get('save_dir', 'Examples/CancerCellsPC/plots')

    for row in rows:
        experiment = row['experiment']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1']
        metric_keys = ['accuracy', 'precision', 'recall', 'f1']
        values = [float(row[k]) for k in metric_keys]

        # Parse cell type and perturbation from experiment name
        # e.g. "refractive_index_skin_basal_1M_0_5" -> cell_type="skin_basal", perturbation="0.5"
        exp_match = re.match(r'refractive_index_(.+?)_\d+[MKk]_(.+)$', experiment)
        if exp_match:
            cell_type = exp_match.group(1)
            perturbation = exp_match.group(2).replace('_', '.')
        else:
            cell_type = experiment
            perturbation = "unknown"

        colors = ['#4C72B0', '#55A868', '#DD8452', '#C44E52']
        x = np.arange(len(metric_names))

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(x, values, width=0.5, color=colors, edgecolor='white', linewidth=0.8)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, fontsize=11)
        ax.set_ylabel('Score', fontsize=12)
        title = f'Test Metrics — {cell_type} — {perturbation}% perturbation'
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_ylim(0, max(values) * 1.2)
        ax.grid(True, axis='y', color='lightgray', linewidth=0.5)
        ax.tick_params(direction='in')

        # Add table below the chart
        table_data = [[f'{v:.4f}' for v in values]]
        table = ax.table(
            cellText=table_data,
            colLabels=metric_names,
            loc='bottom',
            cellLoc='center',
            bbox=[0.0, -0.25, 1.0, 0.1],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        for key, cell in table.get_celld().items():
            cell.set_edgecolor('lightgray')
            if key[0] == 0:  # header row
                cell.set_text_props(fontweight='bold')
                cell.set_facecolor('#f0f0f0')

        fig.subplots_adjust(bottom=0.2)
        fig.tight_layout(rect=[0, 0.1, 1, 1])

        save_dir_out = os.path.join(save_dir, cell_type)
        os.makedirs(save_dir_out, exist_ok=True)
        filename = f'test_metrics_{cell_type}_{perturbation}_perturbation.png'
        fig.savefig(os.path.join(save_dir_out, filename), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Metrics plot saved to {os.path.join(save_dir_out, filename)}")
