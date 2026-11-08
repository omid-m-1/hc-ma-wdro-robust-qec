"""Plot exact X/Z pair and burst mechanisms, not learned scores."""
from pathlib import Path
import sys,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'decoder'),str(ROOT),str(ROOT/'scripts')]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
from physics import setup
from qecrobust.case_bank import get_case
p=argparse.ArgumentParser();p.add_argument('--case',default='band_family_topology');p.add_argument('--distance',type=int,default=19);p.add_argument('--out',default='correlation_profiles.png');a=p.parse_args()
fig,axes=plt.subplots(2,3,figsize=(12,8));norm=LogNorm(.001,.18)
for col,mode in enumerate(['segments','long_segments','long_aligned']):
 c,xy,ed,env,F,H,Z=setup(a.distance,get_case(a.case)['specs'],noise_geometry=mode);e=env[4];B=e['_burst_matrix'].tocsc();half=len(e['_burst_prob'])//2
 for row,key in enumerate(['pxx','pzz']):
  ax=axes[row,col];lines=[xy[list(q)] for q in ed];rates=list(e[key]);width=[.7]*len(lines)
  for j in range(row*half,(row+1)*half):
   ids=B.indices[B.indptr[j]:B.indptr[j+1]]-row*c.n;coords=xy[ids];axis=int(np.ptp(coords[:,1])>np.ptp(coords[:,0]));coords=coords[np.argsort(coords[:,axis])];lines.append(coords);rates.append(e['_burst_prob'][j]);width.append(2)
  lc=LineCollection(lines,cmap='viridis',norm=norm,linewidths=width,alpha=.85);lc.set_array(np.asarray(rates));ax.add_collection(lc);ax.scatter(*xy.T,s=3,c='#151515',zorder=3);ax.set_aspect('equal');ax.set_xlim(-.04,1.04);ax.set_ylim(-.04,1.04)
  ax.set_title(mode+'; L='+str(e['burst_length']) if row==0 else '');ax.set_ylabel('X faults' if row==0 else 'Z faults');ax.set_xticks([0,.5,1]);ax.set_yticks([0,.5,1])
fig.suptitle(f'Actual event supports and probabilities: {a.case}, deployment candidate, d={a.distance}',fontsize=12)
fig.subplots_adjust(left=.07,right=.9,bottom=.14,top=.9,hspace=.12,wspace=.24);cax=fig.add_axes([.93,.25,.015,.5]);fig.colorbar(lc,cax=cax,label='Bernoulli event probability')
fig.text(.5,.05,'Thin: adjacent pairs. Thick: collective segments. All profiles preserve expected correlated endpoints per qubit.\nHalf-length segments and logical-axis coupling change correlation structure; they are not fixed-local-noise threshold tests.',ha='center',fontsize=9)
fig.savefig(a.out,dpi=160);print(a.out)
