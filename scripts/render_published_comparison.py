"""Restyle the manuscript's published-method comparison from public scores."""
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
rows = json.loads((ROOT / 'results/published-methods.json').read_text())['results']
measured = list(csv.DictReader((ROOT / 'results/robotwin-50.csv').open()))
means = {condition: sum(int(r['restart_successes']) for r in measured if r['condition'] == condition) / 50
         for condition in ['demo_clean', 'demo_randomized']}
bg, fg, grid = '#08131f', '#e3edf6', '#263849'
ours, published = '#66dfbb', '#9cacc0'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
    'text.color': fg, 'axes.labelcolor': fg, 'xtick.color': '#b8c8d8',
    'ytick.color': '#b8c8d8', 'svg.fonttype': 'none', 'svg.hashsalt': 'restart-field',
    'savefig.facecolor': bg})
fig, ax = plt.subplots(figsize=(11, 8), facecolor=bg)
fig.subplots_adjust(left=.105, right=.96, bottom=.18, top=.77)
fig.text(.105, .94, 'Across published methods', fontsize=24, weight='bold')
fig.text(.105, .89, 'RoboTwin 2.0 · clean-to-random evaluation · 50-task means', fontsize=12, color='#b8c8d8')
fig.legend(handles=[Line2D([], [], marker='s', color=published, linestyle='None', markersize=7,
                          label='30 published fine-tuned results'),
                    Line2D([], [], marker='*', color=ours, linestyle='None', markersize=13,
                          label='ReStart · frozen UniWAM-base')],
           loc='upper left', bbox_to_anchor=(.1, .85), ncol=2, frameon=False)
ax.set_facecolor(bg)
ax.plot([40, 100], [40, 100], '--', color='#70859a', lw=1.1)
ax.text(84, 91, 'Equal clean / randomized success', color='#8fa5b8', fontsize=9, ha='center')
ax.scatter([r['clean'] for r in rows], [r['random'] for r in rows],
           marker='s', s=49, color=published, edgecolors=bg, linewidths=.7, alpha=.85, zorder=3)
ax.scatter(means['demo_clean'], means['demo_randomized'], marker='*', s=240, color=ours,
           edgecolors=bg, linewidths=.8, zorder=5)
ax.annotate('ReStart  80.74 / 80.82', (means['demo_clean'], means['demo_randomized']),
            xytext=(-12, 13), textcoords='offset points', ha='right', color=ours, fontsize=12, weight='bold')
labels = {
    'GigaBrain-0.7': (-8, 0, 'right', 'GigaBrain-0.7'),
    'UniWAM SFT': (0, -18, 'center', 'UniWAM SFT'),
    'PrismWAM*': (8, 5, 'left', 'PrismWAM*'),
    'PICA-WAM §': (8, -12, 'left', 'PICA-WAM §'),
    'OpenWAM-alpha': (9, 1, 'left', 'OpenWAM-alpha'),
    'pi0.5': (-9, 0, 'right', r'$\pi_{0.5}$'),
    'Fast-WAM': (8, 0, 'left', 'Fast-WAM'),
}
for row in rows:
    if row['method'] in labels:
        dx, dy, align, label = labels[row['method']]
        ax.annotate(label, (row['clean'], row['random']), xytext=(dx, dy),
                    textcoords='offset points', ha=align, va='center', fontsize=10, color='#c1cede')
ax.set_xlim(40, 110); ax.set_ylim(-4, 103)
ax.set_xticks(range(40, 101, 10)); ax.set_yticks(range(0, 101, 20))
ax.set_xlabel('Clean success (%)', labelpad=12)
ax.set_ylabel('Randomized success (%)', labelpad=12)
ax.grid(color=grid, linewidth=.6); ax.set_axisbelow(True); ax.tick_params(length=0)
for spine in ax.spines.values(): spine.set_visible(False)
fig.text(.105, .075, 'Other methods: author-reported scores, not local reproductions. Variant symbols follow the sources.',
         fontsize=10, color='#b8c8d8')
fig.text(.105, .035, 'Sources and exact values: results/published-methods.json · ReStart: results/robotwin-50.csv',
         fontsize=9, color='#8fa5b8')
for ext in ['png', 'svg']:
    fig.savefig(ROOT / f'results/robotwin-published-methods.{ext}', dpi=180, metadata={'Date': None} if ext == 'svg' else None)
plt.close(fig)
