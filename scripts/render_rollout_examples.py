"""Lay out the manuscript's unaltered demonstration and rollout frames."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
bg, fg = '#08131f', '#e3edf6'
plt.rcParams.update({'font.family':'DejaVu Sans', 'text.color':fg, 'svg.fonttype':'none', 'svg.hashsalt':'restart-rollouts'})
fig=plt.figure(figsize=(18,10.5),facecolor=bg)
fig.text(.035,.95,'From clean demonstrations to randomized scenes',fontsize=25,weight='bold')
fig.text(.035,.912,'Same frozen policy · task context built from clean demonstrations only',fontsize=13,color='#b8c8d8')
# One demo, three clean frames, four randomized frames: preserve the paper's sequences.
xs=[.035,.165,.267,.369,.505,.607,.709,.811]
w=.098; h=.126
for x,label,color in [(.035,'EXPERT DEMO','#dfb96e'),(.165,'RESTART / CLEAN','#9cacc0'),(.505,'RESTART / RANDOMIZED','#66dfbb')]:
 fig.text(x,.858,label,fontsize=12,weight='bold',color=color)
for row,task in enumerate(['handover_block','rotate_qrcode','pick_dual_bottles','place_container_plate']):
 y=.674-row*.177
 fig.text(.035,y+h+.013,task,fontsize=12,weight='bold',color=fg)
 for col,(x,name) in enumerate(zip(xs,['demo','clean-1','clean-2','clean-3','random-1','random-2','random-3','random-4'])):
  ax=fig.add_axes([x,y,w,h])
  ax.imshow(Image.open(ROOT/'docs/assets/rollout-frames'/f'{task}-{name}.png'))
  ax.set_axis_off()
  color='#dfb96e' if col==0 else '#9cacc0' if col<4 else '#66dfbb'
  ax.add_patch(Rectangle((0,0),1,1,transform=ax.transAxes,fill=False,edgecolor=color,lw=1.5))
fig.text(.035,.075,'Frames progress left to right within each rollout. Each rollout is the first success of its condition.',fontsize=11,color='#b8c8d8')
fig.text(.035,.039,'Original manuscript examples · 4 tasks · selected successful episodes, not aggregate performance',fontsize=10,color='#8fa5b8')
for ext in ['png','svg']:
 fig.savefig(ROOT/f'docs/assets/restart-rollout-examples.{ext}',dpi=160,facecolor=bg,metadata={'Date':None} if ext=='svg' else None)
plt.close(fig)
