"""Render the release CSV with the manuscript's paired-score layout."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader((ROOT / 'results/robotwin-50.csv').open()))
data = {}
for row in rows:
    data.setdefault(row['task'], {})[row['condition']] = row
order = sorted(data, key=lambda task: (float(data[task]['demo_clean']['authors_reported_sft_percent']), task))
bg, fg, grid = '#08131f', '#e3edf6', '#263849'
ours, sft = '#66dfbb', '#9cacc0'
decline = '#ff897d'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'text.color': fg, 'axes.labelcolor': fg,
                     'xtick.color': '#b8c8d8', 'ytick.color': '#b8c8d8',
                     'svg.fonttype': 'none', 'savefig.facecolor': bg})
fig, axes = plt.subplots(2, 1, figsize=(16, 8.3), sharex=True, facecolor=bg)
fig.subplots_adjust(left=.055, right=.985, top=.80, bottom=.27, hspace=.26)
fig.text(.055,.945,'RoboTwin 2.0',fontsize=24,weight='bold')
fig.text(.055,.902,'50 tasks  /  100 episodes per task and condition',fontsize=12,color='#b8c8d8')
fig.legend(handles=[Line2D([],[],marker='o',color=ours,linestyle='None',markersize=8,label='ReStart · frozen UniWAM-base'),
                    Line2D([],[],marker='s',color=sft,linestyle='None',markersize=7,label='Fine-tuned UniWAM · authors’ report'),
                    Line2D([],[],color=ours,linewidth=2,label='Gain'),
                    Line2D([],[],color=decline,linewidth=2,label='Decline')],
           loc='upper left',bbox_to_anchor=(.05,.873),ncol=4,frameon=False,columnspacing=2)
for ax, condition, label in zip(axes, ['demo_clean','demo_randomized'], ['CLEAN SCENES','RANDOMIZED SCENES']):
    ax.set_facecolor(bg)
    a=[int(data[t][condition]['restart_successes']) for t in order]
    b=[float(data[t][condition]['authors_reported_sft_percent']) for t in order]
    delta_colors = [ours if new > old else decline if new < old else sft for new, old in zip(a, b)]
    ax.vlines(range(50),b,a,color=delta_colors,linewidth=1.7,alpha=.85,zorder=2)
    ax.scatter(range(50),b,s=29,c=sft,marker='s',edgecolors=bg,linewidths=.6,zorder=3)
    ax.scatter(range(50),a,s=33,c=ours,edgecolors=bg,linewidths=.7,zorder=4)
    ax.set_ylim(-3,106);ax.set_xlim(-.8,49.8);ax.set_yticks([0,25,50,75,100])
    ax.set_ylabel('Success (%)',labelpad=10)
    ax.set_title(label,loc='left',fontsize=10,fontweight='bold',color='#b8c8d8',pad=8)
    ax.grid(axis='y',color=grid,linewidth=.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():spine.set_visible(False)
    ax.tick_params(axis='both',length=0)
axes[-1].set_xticks(range(50),order,rotation=55,ha='right',fontsize=8.2)
fig.text(.055,.037,'Tasks ordered by the authors’ clean-scene SFT score; the same order is used in both panels.',fontsize=10,color='#b8c8d8')
fig.text(.055,.012,'SFT is author-reported, not a local reproduction. Source: results/robotwin-50.csv.',fontsize=9,color='#8fa5b8')
for ext in ['png','svg']:
    fig.savefig(ROOT / f'results/robotwin-task-performance.{ext}',dpi=180)
plt.close(fig)
