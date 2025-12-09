#!/usr/bin/env python3
"""
Visualize the 1/(1+|x|) transformation and explain why it
preferentially amplifies hypokinesia detection.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Output path - handle both script and notebook execution
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_DIR = SCRIPT_DIR.parent
except NameError:
    # Running in Jupyter notebook
    PROJECT_DIR = Path.cwd()
    if PROJECT_DIR.name == 'scripts':
        PROJECT_DIR = PROJECT_DIR.parent

OUTPUT_PATH = PROJECT_DIR / "transformation_analysis.png"


def main():
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("1/(1+|x|) Transformation Analysis\nAcross Different Feature Scales",
                 fontsize=14, fontweight='bold')

    # =========================================================
    # Panel 1: The transformation function with ALL feature scales
    # =========================================================
    ax1 = axes[0, 0]

    # Show transformation over wide range to cover all features
    x = np.linspace(0, 200, 1000)
    f_x = 1 / (1 + np.abs(x))

    ax1.plot(x, f_x, 'b-', linewidth=2.5, label='f(x) = 1/(1+|x|)')

    # Mark the different feature scales with their typical means
    feature_scales = {
        'Linear vel\n(0.1-0.3)': (0.2, 'green'),
        'Linear acc\n(1.0-1.1)': (1.1, 'orange'),
        'Angular vel\n(~38)': (38, 'purple'),
        'Angular acc\n(~170)': (170, 'red'),
    }

    for label, (x_val, color) in feature_scales.items():
        f_val = 1 / (1 + x_val)
        ax1.plot(x_val, f_val, 'o', markersize=10, color=color, zorder=5)
        ax1.annotate(f'{label}\nf={f_val:.3f}', xy=(x_val, f_val),
                    xytext=(x_val + 10, f_val + 0.1),
                    fontsize=8, color=color,
                    arrowprops=dict(arrowstyle='->', color=color, alpha=0.7))

    ax1.set_xlabel('Original Value (x)', fontsize=11)
    ax1.set_ylabel('Transformed Value f(x)', fontsize=11)
    ax1.set_title('A. Transformation Across Feature Scales', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 200)
    ax1.set_ylim(0, 1.05)
    ax1.set_xscale('linear')

    # =========================================================
    # Panel 2: The derivative (sensitivity) - LOG SCALE to show all features
    # =========================================================
    ax2 = axes[0, 1]

    x_wide = np.linspace(0.01, 200, 1000)
    df_x = 1 / (1 + np.abs(x_wide))**2

    ax2.plot(x_wide, df_x, 'r-', linewidth=2.5, label="|f'(x)| = 1/(1+|x|)²")

    # Mark sensitivity at each feature scale
    for label, (x_val, color) in feature_scales.items():
        sens = 1 / (1 + x_val)**2
        ax2.plot(x_val, sens, 'o', markersize=10, color=color, zorder=5)
        short_label = label.split('\n')[0]
        ax2.annotate(f'{short_label}\nsens={sens:.1e}', xy=(x_val, sens),
                    xytext=(x_val * 1.5, sens * 2),
                    fontsize=8, color=color,
                    arrowprops=dict(arrowstyle='->', color=color, alpha=0.7))

    ax2.set_xlabel('Original Value (x)', fontsize=11)
    ax2.set_ylabel('Sensitivity |f\'(x)|', fontsize=11)
    ax2.set_title('B. Sensitivity by Feature Scale (log-log)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlim(0.01, 300)
    ax2.set_ylim(1e-6, 2)

    # =========================================================
    # Panel 3: Transformed values by feature type
    # =========================================================
    ax3 = axes[1, 0]

    # Show what typical values transform to for each feature type
    feature_data = {
        'Linear velocity': {'mean': 0.2, 'std': 0.1, 'color': 'green'},
        'Linear accel': {'mean': 1.1, 'std': 0.5, 'color': 'orange'},
        'Angular velocity': {'mean': 38, 'std': 20, 'color': 'purple'},
        'Angular accel': {'mean': 170, 'std': 100, 'color': 'red'},
    }

    y_pos = np.arange(len(feature_data))
    bar_height = 0.6

    for i, (feat_name, data) in enumerate(feature_data.items()):
        mean_val = data['mean']
        std_val = data['std']

        # Typical range
        low = max(0, mean_val - std_val)
        high = mean_val + std_val

        # Transform
        trans_mean = 1 / (1 + mean_val)
        trans_low = 1 / (1 + high)  # Note: inverted because 1/(1+x)
        trans_high = 1 / (1 + low)

        # Plot bar showing transformed range
        ax3.barh(i, trans_high - trans_low, left=trans_low, height=bar_height,
                color=data['color'], alpha=0.6, edgecolor='black')
        ax3.plot(trans_mean, i, 'ko', markersize=8, zorder=5)

        # Annotate with original scale
        ax3.text(trans_high + 0.02, i, f'x={mean_val:.1f}±{std_val:.1f}',
                va='center', fontsize=9)

    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(feature_data.keys())
    ax3.set_xlabel('Transformed Value f(x) = 1/(1+|x|)', fontsize=11)
    ax3.set_title('C. Transformed Ranges by Feature Type', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.set_xlim(0, 1.05)
    ax3.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='f(x)=0.5')

    # =========================================================
    # Panel 4: Z-score comparison
    # =========================================================
    ax4 = axes[1, 1]

    # Data from our test (at 2.0 std deviation)
    conditions = ['Hypokinesia\n(2 std below)', 'Hyperkinesia\n(2 std above)']
    baseline_z = [-2.02, -2.02]
    transformed_z = [-3.78, -1.76]

    x_pos = np.arange(len(conditions))
    width = 0.35

    bars1 = ax4.bar(x_pos - width/2, baseline_z, width, label='Baseline', color='steelblue', edgecolor='black')
    bars2 = ax4.bar(x_pos + width/2, transformed_z, width, label='Transformed', color='coral', edgecolor='black')

    # Add threshold line
    ax4.axhline(y=-0.5, color='red', linestyle='--', linewidth=2, label='Detection threshold (z < -0.5)')

    # Add value labels
    for bar, val in zip(bars1, baseline_z):
        ax4.text(bar.get_x() + bar.get_width()/2, val - 0.15, f'{val:.2f}',
                ha='center', va='top', fontsize=10, fontweight='bold')
    for bar, val in zip(bars2, transformed_z):
        ax4.text(bar.get_x() + bar.get_width()/2, val - 0.15, f'{val:.2f}',
                ha='center', va='top', fontsize=10, fontweight='bold')

    # Add amplification factors
    ax4.annotate('1.87× amplified', xy=(0.175, -3.78), xytext=(0.5, -4.5),
                fontsize=11, color='darkgreen', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='darkgreen'))
    ax4.annotate('0.87× reduced', xy=(1.175, -1.76), xytext=(1.5, -2.5),
                fontsize=11, color='darkred', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='darkred'))

    ax4.set_ylabel('Z-Score', fontsize=11)
    ax4.set_title('D. Z-Score Comparison (at 2 std deviation)', fontsize=12, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(conditions, fontsize=11)
    ax4.legend(loc='lower left', fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim(-5, 0.5)

    # =========================================================
    # Add summary text box
    # =========================================================
    summary_text = """
Key Insights:
• The 11 transformed features span 3 orders of magnitude (0.1 to 170)
• Linear velocity features (x~0.2): f(x)~0.83, HIGH sensitivity to changes
• Angular accel features (x~170): f(x)~0.006, LOW sensitivity (compressed near 0)
• The transformation "normalizes" different scales but with ASYMMETRIC sensitivity
• Features with small typical values (velocity) have more influence on detection
"""
    fig.text(0.5, 0.02, summary_text, ha='center', va='bottom', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='gray', alpha=0.9),
             family='monospace')

    plt.tight_layout(rect=[0, 0.12, 1, 0.95])
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved visualization to: {OUTPUT_PATH}")
    plt.close()


if __name__ == "__main__":
    main()
