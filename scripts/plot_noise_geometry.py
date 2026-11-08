"""Plot actual ZZ mechanism supports/rates; no experimental outcomes used."""
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
p=argparse.ArgumentParser();p.add_argument('--case',default='hv_checker_even_center_cross');p.add_argument('--distance',type=int,default=19);p.add_argument('--out',default='noise_geometry.png');a=p.parse_args()
fig,axes=plt.subplots(1,3,figsize=(12,4.7));norm=LogNorm(.001,.18)
for ax,mode in zip(axes,['standard','ribbons','segments']):
 c,xy,ed,env,F,H,Z=setup(a.distance,get_case(a.case)['specs'],noise_geometry=mode);e=env[4]
 lines=[xy[list(q)] for q in ed];rates=list(e['pzz']);width=[.8]*len(lines)
 if mode=='segments':
  B=e['_burst_matrix'].tocsc();half=len(e['_burst_prob'])//2
  for j in range(half,len(e['_burst_prob'])):
   ids=B.indices[B.indptr[j]:B.indptr[j+1]]-c.n;coords=xy[ids];axis=int(np.ptp(coords[:,1])>np.ptp(coords[:,0]));coords=coords[np.argsort(coords[:,axis])]
   lines.append(coords);rates.append(e['_burst_prob'][j]);width.append(2)
 lc=LineCollection(lines,cmap='viridis',norm=norm,linewidths=width,alpha=.8);lc.set_array(np.asarray(rates));ax.add_collection(lc);ax.scatter(*xy.T,s=4,c='#111111',zorder=3)
 ax.set_aspect('equal');ax.set_xlim(-.05,1.05);ax.set_ylim(-.05,1.05);ax.set_title(mode+'\n'+('length '+str(e['burst_length'])+' segments + pairs' if mode=='segments' else 'adjacent pairs'));ax.set_xlabel('Normalized horizontal coordinate');ax.set_ylabel('Normalized vertical coordinate')
fig.suptitle(f'Actual ZZ supports/rates: {a.case}, d={a.distance}',fontsize=12)
fig.subplots_adjust(bottom=.25,top=.78,wspace=.35,right=.9);cax=fig.add_axes([.92,.28,.015,.44]);fig.colorbar(lc,cax=cax,label='Bernoulli event probability')
fig.text(.5,.07,'Endpoint budget is fixed. Segment length scales with distance; physical marginals can change through XOR overlap.\nThin lines: adjacent-pair events. Thick lines: collective segment events. All off-ribbon mechanisms remain nonzero.',ha='center',fontsize=9)
fig.savefig(a.out,dpi=170);print(a.out)
