from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'decoder'),str(ROOT),str(ROOT/'scripts')]
import numpy as np
from physics import setup,draw
from canonical_noise import marginals,mechanism_vector
from mechanism_bp import MechanismBP
from qecrobust.case_bank import get_case
spec=get_case('hv_checker_even_center_cross')['specs']
for d in [3,5,7,15,19]:
 c,xy,ed,e0,F0,H0,Z0=setup(d,spec)
 for kind in ['ribbons','segments']:
  c,xy,ed,env,F,H,Z=setup(d,spec,noise_geometry=kind)
  for old,new in zip(e0,env):
   for key,col in [('pxx',0),('pzz',1)]:
    extra=0 if '_burst_matrix' not in new else np.asarray(new['_burst_matrix'].sum(0)).ravel()[col*len(new['_burst_prob'])//2:(col+1)*len(new['_burst_prob'])//2]@new['_burst_prob'][col*len(new['_burst_prob'])//2:(col+1)*len(new['_burst_prob'])//2]
    assert np.isclose(2*old[key].sum(),2*new[key].sum()+extra)
  te=draw(c,F,H,Z,np.eye(5)[4],4000,1,771+d)
  for err,s in zip(te['ERR'][:16],te['S'][:16]):assert np.array_equal(c.measure_syndrome(err),s)
  expected=np.r_[marginals(env[4],ed)];observed=te['ERR'].mean(0);se=np.sqrt(expected*(1-expected)/4000)
  assert np.all(np.abs(expected-observed)<6*se+.003)
  bp=MechanismBP(c,ed,env,np.eye(5)[4]);assert bp.M.shape==F.shape and (bp.M!=F).nnz==0 and np.allclose(bp.p,Z[4])
  for s in te['S'][:4]:assert c.in_codespace(bp.decode(s)^te['ERR'][np.flatnonzero(np.all(te['S']==s,axis=1))[0]])
  print('OK',d,kind,'M',F.shape[1],'burst_length',env[4].get('burst_length',2),'max_marginal_difference',float(np.max(np.abs(expected-observed))),flush=True)
print('BUDGET, HYPEREDGE BASIS, PARITY, BP CHANNEL, MARGINAL MONTE CARLO CHECKS PASSED')
