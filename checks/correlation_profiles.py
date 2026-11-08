from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'decoder'),str(ROOT),str(ROOT/'scripts')]
import numpy as np
from physics import setup,draw
from canonical_noise import marginals
from mechanism_bp import MechanismBP
from qecrobust.case_bank import get_case
spec=get_case('hv_checker_even_center_cross')['specs']
for d in [3,5,7,15,19]:
 _,_,_,old,F0,H0,Z0=setup(d,spec)
 for g in ['segments','long_segments','long_aligned']:
  c,xy,ed,env,F,H,Z=setup(d,spec,noise_geometry=g)
  for e,b in zip(env,old):
   B=e['_burst_matrix'];n=len(e['_burst_prob'])//2;cnt=np.asarray(B.sum(0)).ravel()
   assert np.isclose(2*e['pxx'].sum()+cnt[:n]@e['_burst_prob'][:n],2*b['pxx'].sum())
   assert np.isclose(2*e['pzz'].sum()+cnt[n:]@e['_burst_prob'][n:],2*b['pzz'].sum())
  te=draw(c,F,H,Z,np.eye(5)[4],3000,1,902+d)
  expected=np.r_[marginals(env[4],ed)];observed=te['ERR'].mean(0);se=np.sqrt(expected*(1-expected)/3000)
  assert np.all(np.abs(expected-observed)<6*se+.003)
  assert all(np.array_equal(c.measure_syndrome(e),s) for e,s in zip(te['ERR'][:10],te['S'][:10]))
  bp=MechanismBP(c,ed,env,np.eye(5)[4]);assert (bp.M!=F).nnz==0 and np.allclose(bp.p,Z[4])
  print(d,g,'L',env[4]['burst_length'],'H shares',env[4]['burst_x_horizontal_share'],env[4]['burst_z_horizontal_share'],'logical axes',env[4].get('logical_axis_x'),env[4].get('logical_axis_z'),flush=True)
print('ALL LENGTH/ORIENTATION PHYSICS CHECKS PASSED')
