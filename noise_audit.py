"""Deterministic noise-definition audit; generates no decoder test outcomes."""
from pathlib import Path
import sys,argparse,csv
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'decoder'),str(ROOT),str(ROOT/'scripts')]
import numpy as np
from physics import setup
from canonical_noise import marginals
from qecrobust.case_bank import get_case,names
p=argparse.ArgumentParser();p.add_argument('--cases',default='hv_checker_even_center_cross');p.add_argument('--distances',default='3,5,7,15,19');p.add_argument('--geometries',default='standard,ribbons,segments');p.add_argument('--out',default='noise_audit.csv');a=p.parse_args();rows=[]
for case in a.cases.split(','):
 for d in map(int,a.distances.split(',')):
  for kind in a.geometries.split(','):
   c,xy,edges,env,F,H,Z=setup(d,get_case(case)['specs'],noise_geometry=kind)
   for k,e in enumerate(env):
    mx,mz=marginals(e,edges);pair=2*(e['pxx'].sum()+e['pzz'].sum())/c.n
    burst=0 if '_burst_matrix' not in e else float(np.asarray(e['_burst_matrix'].sum(0)).ravel()@e['_burst_prob']/c.n)
    rows.append(dict(case=case,distance=d,environment=k,geometry=kind,segment_length=e.get('burst_length',2),fault_locations=F.shape[1],pair_endpoints_per_qubit=pair,burst_endpoints_per_qubit=burst,total_correlated_endpoints_per_qubit=pair+burst,mean_physical_x=float(mx.mean()),mean_physical_z=float(mz.mean()),burst_x_horizontal_share=e.get('burst_x_horizontal_share'),burst_z_horizontal_share=e.get('burst_z_horizontal_share'),logical_x_axis=e.get('logical_axis_x'),logical_z_axis=e.get('logical_axis_z'),max_event_probability=float(Z[k].max())))
path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(path)
