"""
Generate high-quality publication-ready plots from evaluation results.
"""
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Load results
with open('results/evaluation_results.json') as f:
    results = json.load(f)

# Main 4 datasets
MAIN_DATASETS = ['ham10000', 'aptos2019', 'pcam', 'chestxray14']
DATASET_NAMES = {
    'ham10000': 'HAM10000\n(Skin Lesions)',
    'aptos2019': 'APTOS2019\n(Retinopathy)',
    'pcam': 'PCam\n(Histopathology)',
    'chestxray14': 'ChestXray14\n(X-ray)'
}

# ============================================================
# Plot 1: Linear Probe Performance
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

datasets = []
accuracies = []
aucs = []

for ds in MAIN_DATASETS:
    if ds in results and 'linear_probing' in results[ds]:
        lp = results[ds]['linear_probing']
        datasets.append(DATASET_NAMES[ds])
        accuracies.append(lp.get('accuracy', 0) * 100)
        aucs.append(lp.get('auc', 0))

x = np.arange(len(datasets))
width = 0.35

bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy (%)',
               color='steelblue', edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x + width/2, [a*100 for a in aucs], width, label='AUC × 100',
               color='coral', edgecolor='black', linewidth=1.2)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xlabel('Dataset', fontweight='bold')
ax.set_ylabel('Score', fontweight='bold')
ax.set_title('Linear Probe Performance Across Medical Imaging Datasets',
             fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.legend(loc='upper right', framealpha=0.95)
ax.set_ylim(0, 110)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/linear_probe_performance.png', dpi=300, bbox_inches='tight')
print("Saved: results/linear_probe_performance.png")
plt.close()

# ============================================================
# Plot 2: N-Shot Learning Performance
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))

colors = {'5-shot': 'darkgreen', '10-shot': 'seagreen', '20-shot': 'lightseagreen'}
markers = {'5-shot': 'o', '10-shot': 's', '20-shot': '^'}

for shot_type in ['5-shot', '10-shot', '20-shot']:
    x_pos = []
    y_vals = []
    for i, ds in enumerate(MAIN_DATASETS):
        if ds in results and 'n_shot' in results[ds]:
            ns = results[ds]['n_shot']
            if shot_type in ns:
                x_pos.append(i)
                y_vals.append(ns[shot_type]['accuracy'] * 100)

    ax.plot(x_pos, y_vals, marker=markers[shot_type], markersize=10,
            linewidth=2.5, label=shot_type, color=colors[shot_type],
            markeredgecolor='black', markeredgewidth=1.2)

ax.set_xlabel('Dataset', fontweight='bold', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
ax.set_title('Few-Shot Learning Performance (5/10/20-Shot)',
             fontweight='bold', fontsize=13, pad=15)
ax.set_xticks(range(len(MAIN_DATASETS)))
ax.set_xticklabels([DATASET_NAMES[ds] for ds in MAIN_DATASETS])
ax.legend(loc='best', framealpha=0.95, edgecolor='black')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig('results/n_shot_performance.png', dpi=300, bbox_inches='tight')
print("Saved: results/n_shot_performance.png")
plt.close()

# ============================================================
# Plot 3: MedJEPA vs ImageNet Baseline
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

datasets_compared = []
medjepa_scores = []
imagenet_scores = []

for ds in MAIN_DATASETS:
    if ds in results:
        ft = results[ds].get('fine_tuning', {}).get('accuracy')
        inet = results[ds].get('imagenet_baseline', {}).get('accuracy')
        if ft is not None and inet is not None:
            datasets_compared.append(DATASET_NAMES[ds])
            medjepa_scores.append(ft * 100)
            imagenet_scores.append(inet * 100)

x = np.arange(len(datasets_compared))
width = 0.35

bars1 = ax.bar(x - width/2, medjepa_scores, width, label='MedJEPA (Fine-tuned)',
               color='darkblue', edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x + width/2, imagenet_scores, width, label='ImageNet Pretrained',
               color='darkorange', edgecolor='black', linewidth=1.2)

# Add value labels and win indicators
for i, (m, im) in enumerate(zip(medjepa_scores, imagenet_scores)):
    # MedJEPA bar
    ax.text(i - width/2, m + 1, f'{m:.1f}%',
            ha='center', va='bottom', fontsize=9, fontweight='bold')
    # ImageNet bar
    ax.text(i + width/2, im + 1, f'{im:.1f}%',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Winner indicator
    if m > im:
        ax.text(i, max(m, im) + 5, '★', ha='center', fontsize=16, color='gold')

ax.set_xlabel('Dataset', fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontweight='bold')
ax.set_title('MedJEPA vs ImageNet Pretrained Baseline',
             fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(datasets_compared)
ax.legend(loc='lower right', framealpha=0.95)
ax.set_ylim(0, 110)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/medjepa_vs_imagenet.png', dpi=300, bbox_inches='tight')
print("Saved: results/medjepa_vs_imagenet.png")
plt.close()

# ============================================================
# Plot 4: Comprehensive Results Table
# ============================================================
fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('tight')
ax.axis('off')

# Build table data
table_data = []
headers = ['Dataset', 'Linear Probe\nAcc (%)', 'AUC',
           '5-shot (%)', '10-shot (%)', '20-shot (%)',
           'Fine-tune (%)', 'ImageNet (%)']

for ds in MAIN_DATASETS:
    if ds not in results:
        continue

    row = [DATASET_NAMES[ds].replace('\n', ' ')]

    # Linear probe
    lp = results[ds].get('linear_probing', {})
    row.append(f"{lp.get('accuracy', 0)*100:.1f}")
    row.append(f"{lp.get('auc', 0):.3f}")

    # N-shot
    ns = results[ds].get('n_shot', {})
    for shot in ['5-shot', '10-shot', '20-shot']:
        if shot in ns:
            row.append(f"{ns[shot]['accuracy']*100:.1f}")
        else:
            row.append('-')

    # Fine-tune vs ImageNet
    ft = results[ds].get('fine_tuning', {}).get('accuracy')
    inet = results[ds].get('imagenet_baseline', {}).get('accuracy')
    row.append(f"{ft*100:.1f}" if ft else '-')
    row.append(f"{inet*100:.1f}" if inet else '-')

    table_data.append(row)

table = ax.table(cellText=table_data, colLabels=headers,
                cellLoc='center', loc='center',
                colWidths=[0.18, 0.11, 0.08, 0.09, 0.09, 0.09, 0.11, 0.11])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header
for i in range(len(headers)):
    cell = table[(0, i)]
    cell.set_facecolor('#4472C4')
    cell.set_text_props(weight='bold', color='white')

# Style rows
colors = ['#E7E6E6', 'white']
for i, row in enumerate(table_data):
    for j in range(len(headers)):
        cell = table[(i+1, j)]
        cell.set_facecolor(colors[i % 2])
        cell.set_edgecolor('gray')

plt.title('MedJEPA Evaluation Results - Complete Summary',
          fontweight='bold', fontsize=14, pad=20)

plt.savefig('results/results_table.png', dpi=300, bbox_inches='tight')
print("Saved: results/results_table.png")
plt.close()

# ============================================================
# Plot 5: Segmentation Dice Scores (BraTS + Decathlon)
# ============================================================
seg_datasets = []
seg_scores = []

for key, data in results.items():
    if data.get('type') == 'segmentation':
        name = data['dataset'].replace('Task0', 'T').replace('_', ' ')
        seg_datasets.append(name)
        seg_scores.append(data['mean_dice'])

if seg_datasets:
    fig, ax = plt.subplots(figsize=(10, 6))

    colors_seg = ['darkgreen' if 'BraTS' in name else 'steelblue'
                  for name in seg_datasets]

    bars = ax.barh(seg_datasets, seg_scores, color=colors_seg,
                   edgecolor='black', linewidth=1.2)

    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, seg_scores)):
        ax.text(score + 0.01, i, f'{score:.3f}',
                va='center', fontweight='bold', fontsize=9)

    ax.set_xlabel('Mean Dice Score', fontweight='bold')
    ax.set_ylabel('Dataset', fontweight='bold')
    ax.set_title('Segmentation Performance (Dice Score)',
                 fontweight='bold', pad=15)
    ax.set_xlim(0, 1.0)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/segmentation_dice.png', dpi=300, bbox_inches='tight')
    print("Saved: results/segmentation_dice.png")
    plt.close()

print("\nAll high-quality plots generated successfully!")
print("Location: f:/Projects/MedJEPA/results/")
